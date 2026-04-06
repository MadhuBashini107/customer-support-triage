"""
Tests for the Customer Support Triage OpenEnv environment.
Run with: pytest tests/test_environment.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import pytest
from environment import (
    Action,
    Category,
    Priority,
    SupportTriageEnv,
    grade_action,
    TICKETS,
)


# ─────────────────────────── Grader Tests ───────────────────────────

class TestGrader:
    def test_perfect_easy_action(self):
        gt = TICKETS["easy"][0]["ground_truth"]
        action = Action(
            category=gt["category"].value,
            priority=gt["priority"].value,
            response="To reset your password, please click the 'Forgot Password' link. The reset email may be in your spam folder.",
            escalate=gt["escalate"],
            resolve=True,
        )
        reward, breakdown, msg = grade_action(action, gt, step=1, max_steps=5)
        assert breakdown["category"] == 0.25
        assert breakdown["priority"] == 0.25
        assert breakdown["escalation"] == 0.15
        assert reward > 0.5

    def test_wrong_category_gives_partial(self):
        gt = TICKETS["easy"][0]["ground_truth"]
        action = Action(category="billing", priority="low", response="x" * 50, escalate=False, resolve=True)
        reward, breakdown, msg = grade_action(action, gt, step=1, max_steps=5)
        assert breakdown["category"] == 0.05

    def test_adjacent_priority_partial_credit(self):
        # medium vs high should give partial credit
        gt = {"category": Category.BILLING, "priority": Priority.HIGH, "escalate": False,
              "key_topics": [], "response_must_include": []}
        action = Action(category="billing", priority="medium", response="ok", escalate=False, resolve=False)
        reward, breakdown, msg = grade_action(action, gt, step=1, max_steps=5)
        assert 0 < breakdown["priority"] < 0.25

    def test_missing_escalation_penalized(self):
        gt = TICKETS["hard"][0]["ground_truth"]
        action = Action(
            category=gt["category"].value,
            priority=gt["priority"].value,
            response="We are looking into your account security.",
            escalate=False,  # WRONG — should be True
            resolve=True,
        )
        reward, breakdown, msg = grade_action(action, gt, step=1, max_steps=10)
        assert breakdown["escalation"] == 0.0

    def test_prohibited_phrase_penalized(self):
        gt = TICKETS["hard"][0]["ground_truth"]
        action = Action(
            category="account", priority="urgent",
            response="Don't worry, no problem here, we will fix it",
            escalate=True, resolve=True,
        )
        reward, breakdown, msg = grade_action(action, gt, step=1, max_steps=10)
        assert "inappropriate phrase" in msg

    def test_reward_clamped_to_range(self):
        for diff in ["easy", "medium", "hard"]:
            for ticket in TICKETS[diff]:
                gt = ticket["ground_truth"]
                action = Action(
                    category=gt["category"].value,
                    priority=gt["priority"].value,
                    response=" ".join(gt["key_topics"]) + " " + " ".join(gt.get("response_must_include", [])) + " " * 100,
                    escalate=gt["escalate"],
                    resolve=True,
                )
                r, _, _ = grade_action(action, gt, 1, 10)
                assert 0.0 <= r <= 1.0, f"Reward {r} out of range for {ticket['ticket_id']}"


# ─────────────────────────── Environment Tests ───────────────────────────

class TestEnvironment:
    def test_reset_returns_observation(self):
        env = SupportTriageEnv(task_id="easy", seed=42)
        result = env.reset()
        assert result.observation.ticket_id
        assert result.observation.subject
        assert result.observation.step == 0
        assert result.observation.max_steps == 5

    def test_step_increments(self):
        env = SupportTriageEnv(task_id="easy", seed=42)
        env.reset()
        action = Action(category="account", priority="low", response="Hello", escalate=False, resolve=False)
        result = env.step(action)
        assert result.observation.step == 1
        assert 0.0 <= result.reward <= 1.0

    def test_resolve_ends_episode(self):
        env = SupportTriageEnv(task_id="easy", seed=42)
        env.reset()
        action = Action(category="account", priority="low", response="Done", escalate=False, resolve=True)
        result = env.step(action)
        assert result.done is True

    def test_state_reflects_current_state(self):
        env = SupportTriageEnv(task_id="medium", seed=1)
        env.reset()
        state = env.state()
        assert state.task_id == "medium"
        assert state.step == 0

    def test_all_tasks_work(self):
        for task_id in ["easy", "medium", "hard"]:
            env = SupportTriageEnv(task_id=task_id, seed=0)
            result = env.reset()
            assert result.observation.ticket_id
            
            action = Action(
                category="billing",
                priority="medium",
                response="Thank you for contacting us. We will look into this immediately.",
                escalate=False,
                resolve=True,
            )
            step_result = env.step(action)
            assert 0.0 <= step_result.reward <= 1.0
            assert step_result.done is True

    def test_max_steps_ends_episode(self):
        env = SupportTriageEnv(task_id="easy", seed=0)
        env.reset()
        for i in range(5):
            action = Action(response="ok", resolve=False)
            result = env.step(action)
        assert result.done is True

    def test_graders_score_in_range(self):
        """Graders must never return the same score for all inputs."""
        for task_id in ["easy", "medium", "hard"]:
            env = SupportTriageEnv(task_id=task_id, seed=42)
            env.reset()
            scores = []
            
            # Perfect action
            gt = env._ticket["ground_truth"]
            action_perfect = Action(
                category=gt["category"].value,
                priority=gt["priority"].value,
                response=" ".join(gt["key_topics"]) + " " + " ".join(gt.get("response_must_include", [])),
                escalate=gt["escalate"],
                resolve=False,
            )
            r1, _, _ = grade_action(action_perfect, gt, 1, 10)
            scores.append(r1)
            
            # Bad action
            action_bad = Action(category="general", priority="low", response="x", escalate=False, resolve=False)
            r2, _, _ = grade_action(action_bad, gt, 1, 10)
            scores.append(r2)
            
            # Scores must differ
            assert r1 != r2, f"Task {task_id}: grader returns same score for perfect and bad actions!"
            assert r1 > r2, f"Task {task_id}: perfect action should score higher than bad action"


# ─────────────────────────── Spec Compliance ───────────────────────────

class TestSpecCompliance:
    def test_observation_is_pydantic(self):
        env = SupportTriageEnv(task_id="easy", seed=0)
        result = env.reset()
        # Must be serializable
        data = result.observation.model_dump()
        assert isinstance(data, dict)
        assert "ticket_id" in data
        assert "step" in data

    def test_action_accepts_partial(self):
        # Actions should have all optional fields
        action = Action()
        assert action.category is None
        assert action.escalate is False

    def test_reward_in_range(self):
        env = SupportTriageEnv(task_id="hard", seed=99)
        env.reset()
        action = Action(category="account", priority="urgent", escalate=True,
                       response="We take this seriously and will escalate immediately.", resolve=True)
        result = env.step(action)
        assert 0.0 <= result.reward <= 1.0

    def test_done_is_bool(self):
        env = SupportTriageEnv(task_id="easy", seed=0)
        env.reset()
        result = env.step(Action(resolve=True))
        assert isinstance(result.done, bool)
