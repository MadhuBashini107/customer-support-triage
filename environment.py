"""
Customer Support Triage Environment
Real-world task: classify, prioritize, and respond to customer support tickets.
"""
from __future__ import annotations

import json
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ─────────────────────────────── Enums ───────────────────────────────

class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"
    REFUND = "refund"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

# ─────────────────────────────── OpenEnv Models ───────────────────────────────

class Observation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: str
    previous_contacts: int
    attachments: List[str]
    task_description: str
    step: int
    max_steps: int
    current_score: float
    feedback: Optional[str] = None

class Action(BaseModel):
    category: Optional[str] = Field(None, description="Ticket category: billing|technical|account|general|refund")
    priority: Optional[str] = Field(None, description="Priority: low|medium|high|urgent")
    response: Optional[str] = Field(None, description="Draft response to send to the customer")
    escalate: Optional[bool] = Field(False, description="Whether to escalate to human agent")
    resolve: Optional[bool] = Field(False, description="Mark ticket as resolved")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags to apply to the ticket")

class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = {}
    message: str = ""

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}

class ResetResult(BaseModel):
    observation: Observation

class StateResult(BaseModel):
    task_id: str
    difficulty: str
    step: int
    max_steps: int
    cumulative_reward: float
    done: bool
    ticket: Dict[str, Any]
    agent_actions: List[Dict[str, Any]]

# ─────────────────────────────── Ticket Bank ───────────────────────────────

TICKETS = {
    "easy": [
        {
            "ticket_id": "TKT-001",
            "subject": "How do I change my password?",
            "body": "Hi, I forgot my password and need to reset it. Can you help me? I've tried the forgot password link but haven't received an email.",
            "customer_tier": "free",
            "previous_contacts": 0,
            "attachments": [],
            "ground_truth": {
                "category": Category.ACCOUNT,
                "priority": Priority.LOW,
                "escalate": False,
                "key_topics": ["password", "reset", "email"],
                "response_must_include": ["reset", "email", "spam"],
            },
        },
        {
            "ticket_id": "TKT-002",
            "subject": "Request for invoice",
            "body": "Please send me an invoice for my subscription for tax purposes. Account email: john@example.com",
            "customer_tier": "pro",
            "previous_contacts": 1,
            "attachments": [],
            "ground_truth": {
                "category": Category.BILLING,
                "priority": Priority.LOW,
                "escalate": False,
                "key_topics": ["invoice", "billing", "subscription"],
                "response_must_include": ["invoice", "billing"],
            },
        },
        {
            "ticket_id": "TKT-003",
            "subject": "What are your business hours?",
            "body": "Hello, I would like to know what your support hours are and if you offer phone support.",
            "customer_tier": "free",
            "previous_contacts": 0,
            "attachments": [],
            "ground_truth": {
                "category": Category.GENERAL,
                "priority": Priority.LOW,
                "escalate": False,
                "key_topics": ["hours", "support", "phone"],
                "response_must_include": ["hours", "support"],
            },
        },
    ],
    "medium": [
        {
            "ticket_id": "TKT-101",
            "subject": "Charged twice for subscription",
            "body": "I was charged $49.99 twice on March 15th. My order numbers are ORD-4521 and ORD-4522. I need an immediate refund for the duplicate charge. This is unacceptable.",
            "customer_tier": "pro",
            "previous_contacts": 2,
            "attachments": ["bank_statement.pdf"],
            "ground_truth": {
                "category": Category.BILLING,
                "priority": Priority.HIGH,
                "escalate": False,
                "key_topics": ["duplicate", "charge", "refund", "ORD-4521", "ORD-4522"],
                "response_must_include": ["refund", "investigate", "apologize"],
            },
        },
        {
            "ticket_id": "TKT-102",
            "subject": "API returning 500 errors intermittently",
            "body": "Our integration has been getting random 500 errors from your API since yesterday. Error rate is about 5%. Request IDs: req_abc123, req_def456. This is affecting our production system.",
            "customer_tier": "pro",
            "previous_contacts": 0,
            "attachments": ["error_logs.txt"],
            "ground_truth": {
                "category": Category.TECHNICAL,
                "priority": Priority.HIGH,
                "escalate": True,
                "key_topics": ["API", "500", "error", "production", "req_abc123"],
                "response_must_include": ["investigate", "engineer", "monitor"],
            },
        },
        {
            "ticket_id": "TKT-103",
            "subject": "Need to downgrade plan before renewal",
            "body": "My renewal is in 3 days (March 20). I want to downgrade from Enterprise ($299/mo) to Pro ($49/mo) before the renewal date. I don't want to be charged the full Enterprise price again.",
            "customer_tier": "enterprise",
            "previous_contacts": 1,
            "attachments": [],
            "ground_truth": {
                "category": Category.BILLING,
                "priority": Priority.MEDIUM,
                "escalate": False,
                "key_topics": ["downgrade", "renewal", "enterprise", "pro"],
                "response_must_include": ["downgrade", "renewal", "plan"],
            },
        },
    ],
    "hard": [
        {
            "ticket_id": "TKT-201",
            "subject": "URGENT: Data breach - my account was compromised",
            "body": "I believe my account was hacked. I'm seeing login activity from IP 192.168.1.1 from Russia that wasn't me. Login time: March 17 at 3:45 AM UTC. I've already changed my password but I'm worried my payment info and private data was accessed. I need to know what data was exposed and I want my account secured immediately. If this isn't resolved in 24 hours I'll be forced to take legal action and report this to the FTC.",
            "customer_tier": "enterprise",
            "previous_contacts": 0,
            "attachments": ["screenshot_logins.png"],
            "ground_truth": {
                "category": Category.ACCOUNT,
                "priority": Priority.URGENT,
                "escalate": True,
                "key_topics": ["breach", "hacked", "security", "login", "Russia", "legal", "FTC"],
                "response_must_include": ["security", "team", "immediate", "apologize"],
                "response_must_not_include": ["don't worry", "no problem"],
            },
        },
        {
            "ticket_id": "TKT-202",
            "subject": "Platform outage affecting 500+ of our users - SLA breach",
            "body": "Your platform has been down for our organization for 2.5 hours (since 9:00 AM EST). We have an Enterprise SLA that guarantees 99.9% uptime. This outage is already a breach. We have 500+ users unable to work. We are losing approximately $50,000/hour. I need: 1) ETA for resolution, 2) Root cause analysis, 3) SLA credit calculation, 4) Emergency phone number for our CTO to call. Ticket must be escalated to VP level.",
            "customer_tier": "enterprise",
            "previous_contacts": 3,
            "attachments": ["sla_agreement.pdf", "user_impact_report.xlsx"],
            "ground_truth": {
                "category": Category.TECHNICAL,
                "priority": Priority.URGENT,
                "escalate": True,
                "key_topics": ["outage", "SLA", "enterprise", "credit", "RCA", "CTO", "VP"],
                "response_must_include": ["escalate", "engineer", "SLA", "apologize"],
                "response_must_not_include": ["unfortunately we cannot", "not our fault"],
            },
        },
        {
            "ticket_id": "TKT-203",
            "subject": "Unauthorized charges + account closure request + GDPR deletion",
            "body": "I am writing to formally request: 1) Immediate refund of $1,847 in charges I did not authorize over the past 6 months (see attached statements), 2) Permanent closure of account #ACC-98765, 3) Complete deletion of all my personal data under GDPR Article 17 (right to erasure). I am a UK resident. You have 30 days to comply with the GDPR request or I will file a complaint with the ICO. I also dispute the $1,847 and will initiate a chargeback with my bank if not refunded within 7 days.",
            "customer_tier": "pro",
            "previous_contacts": 4,
            "attachments": ["bank_statements_6mo.pdf", "account_closure_form.pdf"],
            "ground_truth": {
                "category": Category.BILLING,
                "priority": Priority.URGENT,
                "escalate": True,
                "key_topics": ["GDPR", "refund", "closure", "chargeback", "ICO", "Article 17", "erasure"],
                "response_must_include": ["GDPR", "legal", "escalate", "refund"],
                "response_must_not_include": ["we can't delete", "no refund"],
            },
        },
    ],
}

# ─────────────────────────────── Grader Logic ───────────────────────────────

def clamp(value: float) -> float:
    """Clamp value to strictly between 0.001 and 0.999."""
    return round(min(0.999, max(0.001, value)), 4)

def grade_action(action: Action, ground_truth: Dict, step: int, max_steps: int) -> Tuple[float, Dict[str, float], str]:
    breakdown: Dict[str, float] = {}
    messages: List[str] = []

    # 1. Category (0.25)
    if action.category:
        try:
            cat = Category(action.category.lower())
            if cat == ground_truth["category"]:
                breakdown["category"] = 0.25
                messages.append("Correct category")
            else:
                breakdown["category"] = 0.05
                messages.append(f"Wrong category (got {cat}, expected {ground_truth['category']})")
        except ValueError:
            breakdown["category"] = 0.001
            messages.append("Invalid category value")
    else:
        breakdown["category"] = 0.001

    # 2. Priority (0.25)
    if action.priority:
        try:
            pri = Priority(action.priority.lower())
            gt_pri = ground_truth["priority"]
            if pri == gt_pri:
                breakdown["priority"] = 0.25
                messages.append("Correct priority")
            else:
                pri_order = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.URGENT]
                diff = abs(pri_order.index(pri) - pri_order.index(gt_pri))
                partial = max(0.001, 0.25 - diff * 0.08)
                breakdown["priority"] = round(partial, 3)
                messages.append(f"Priority off by {diff} level(s)")
        except ValueError:
            breakdown["priority"] = 0.001
            messages.append("Invalid priority value")
    else:
        breakdown["priority"] = 0.001

    # 3. Escalation (0.15)
    gt_escalate = ground_truth.get("escalate", False)
    if action.escalate == gt_escalate:
        breakdown["escalation"] = 0.15
        messages.append("Correct escalation decision")
    elif action.escalate and not gt_escalate:
        breakdown["escalation"] = 0.05
        messages.append("Over-escalated")
    else:
        breakdown["escalation"] = 0.001
        messages.append("Should have escalated")

    # 4. Response quality (0.35)
    response_score = 0.001
    if action.response and len(action.response.strip()) > 20:
        response_lower = action.response.lower()
        key_topics = ground_truth.get("key_topics", [])
        must_include = ground_truth.get("response_must_include", [])
        must_not_include = ground_truth.get("response_must_not_include", [])

        topics_found = sum(1 for t in key_topics if t.lower() in response_lower)
        topic_ratio = topics_found / len(key_topics) if key_topics else 1.0

        required_found = sum(1 for r in must_include if r.lower() in response_lower)
        required_ratio = required_found / len(must_include) if must_include else 1.0

        prohibited_found = any(p.lower() in response_lower for p in must_not_include)

        response_score = topic_ratio * 0.15 + required_ratio * 0.15
        if prohibited_found:
            response_score = max(0.001, response_score - 0.1)
            messages.append("Response contains inappropriate phrase")
        if len(action.response) > 100:
            response_score = min(0.35, response_score + 0.05)
        messages.append(f"Response covers {topics_found}/{len(key_topics)} key topics")

    breakdown["response"] = round(max(0.001, response_score), 3)

    # 5. Resolve penalty
    if action.resolve and breakdown["category"] <= 0.001:
        breakdown["resolve_penalty"] = -0.05
        messages.append("Resolved without categorization")
    else:
        breakdown["resolve_penalty"] = 0.0

    total = sum(breakdown.values())
    total = clamp(total)
    feedback = " | ".join(messages)
    return total, breakdown, feedback


# ─────────────────────────────── Environment ───────────────────────────────

class SupportTriageEnv:

    TASK_CONFIGS = {
        "easy":   {"max_steps": 5,  "tickets": TICKETS["easy"]},
        "medium": {"max_steps": 8,  "tickets": TICKETS["medium"]},
        "hard":   {"max_steps": 10, "tickets": TICKETS["hard"]},
    }

    def __init__(self, task_id: str = "easy", seed: Optional[int] = None):
        if task_id not in self.TASK_CONFIGS:
            raise ValueError(f"task_id must be one of {list(self.TASK_CONFIGS.keys())}")
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._ticket: Dict = {}
        self._actions: List[Dict] = []
        self._last_feedback = ""
        self._max_steps = 5

    def reset(self) -> ResetResult:
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._actions = []
        self._last_feedback = ""
        cfg = self.TASK_CONFIGS[self.task_id]
        self._ticket = self._rng.choice(cfg["tickets"])
        self._max_steps = cfg["max_steps"]
        return ResetResult(observation=self._make_observation())

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._make_observation(),
                reward=0.001,
                done=True,
                info={"error": "Episode already done"}
            )

        self._step += 1
        gt = self._ticket["ground_truth"]
        reward, breakdown, feedback = grade_action(action, gt, self._step, self._max_steps)

        self._cumulative_reward = min(0.999, self._cumulative_reward + reward / self._max_steps)
        self._cumulative_reward = max(0.001, self._cumulative_reward)
        self._last_feedback = feedback
        self._actions.append({
            "step": self._step,
            "action": action.model_dump(),
            "reward": reward,
            "breakdown": breakdown,
        })

        done = bool(action.resolve) or self._step >= self._max_steps
        self._done = done

        return StepResult(
            observation=self._make_observation(),
            reward=reward,
            done=done,
            info={"breakdown": breakdown, "feedback": feedback, "cumulative_reward": self._cumulative_reward},
        )

    def state(self) -> StateResult:
        return StateResult(
            task_id=self.task_id,
            difficulty=self.task_id,
            step=self._step,
            max_steps=self._max_steps,
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            ticket={k: v for k, v in self._ticket.items() if k != "ground_truth"},
            agent_actions=self._actions,
        )

    def _make_observation(self) -> Observation:
        task_descs = {
            "easy": "Triage this support ticket: classify the category, set priority, and draft a helpful response.",
            "medium": "Triage this ticket carefully: correct category and priority are critical. Decide whether to escalate.",
            "hard": "This is a complex high-stakes ticket. Classify accurately, set urgent priority if warranted, decide escalation, and draft a thorough empathetic response.",
        }
        return Observation(
            ticket_id=self._ticket.get("ticket_id", ""),
            subject=self._ticket.get("subject", ""),
            body=self._ticket.get("body", ""),
            customer_tier=self._ticket.get("customer_tier", "free"),
            previous_contacts=self._ticket.get("previous_contacts", 0),
            attachments=self._ticket.get("attachments", []),
            task_description=task_descs[self.task_id],
            step=self._step,
            max_steps=self._max_steps,
            current_score=round(self._cumulative_reward, 4),
            feedback=self._last_feedback if self._step > 0 else None,
        )