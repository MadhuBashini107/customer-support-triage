#!/usr/bin/env python3
"""
inference.py — Baseline inference script for Customer Support Triage OpenEnv environment.

Follows the mandatory [START] / [STEP] / [END] log format as required by the contest spec.
Uses OpenAI client for all LLM calls.

Environment variables required:
  API_BASE_URL   The API endpoint for the LLM (e.g. https://api.openai.com/v1)
  MODEL_NAME     The model identifier (e.g. gpt-4o-mini)
  HF_TOKEN       Your Hugging Face / API key (used as OpenAI API key)

Optional:
  ENV_BASE_URL   The environment server URL (default: http://localhost:7860)
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ─────────────────────────── Config ───────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY: str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "customer-support-triage"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = {"easy": 5, "medium": 8, "hard": 10}
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

# ─────────────────────────── Logging (mandatory format) ───────────────────────────

def log_start(*, task: str, env: str, model: str) -> None:
    print(json.dumps({
        "event": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }), flush=True)


def log_step(*, step: int, action: Any, reward: float, done: bool, error: Optional[str]) -> None:
    print(json.dumps({
        "event": "STEP",
        "step": step,
        "action": action if isinstance(action, str) else json.dumps(action),
        "reward": reward,
        "done": done,
        "error": error,
        "timestamp": time.time(),
    }), flush=True)


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({
        "event": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "timestamp": time.time(),
    }), flush=True)


# ─────────────────────────── LLM Agent ───────────────────────────

SYSTEM_PROMPT = """You are an expert customer support triage agent. Given a support ticket, you must:
1. Classify the category (billing, technical, account, general, refund)
2. Set the priority (low, medium, high, urgent)
3. Decide whether to escalate to a human agent (true/false)
4. Draft a professional, empathetic response to the customer
5. Set resolve=true when you have completed your triage

Guidelines:
- urgent priority: data breaches, outages, legal threats, SLA violations
- high priority: production-affecting issues, duplicate charges, multiple previous contacts
- medium priority: billing questions, plan changes
- low priority: general questions, information requests
- escalate=true for: security incidents, SLA violations, legal threats, unresolvable technical issues
- Responses should be professional, empathetic, and address ALL customer concerns

Respond ONLY with a valid JSON object matching this schema:
{
  "category": "billing|technical|account|general|refund",
  "priority": "low|medium|high|urgent",
  "response": "Your professional response to the customer here",
  "escalate": true|false,
  "resolve": true|false,
  "tags": ["optional", "tags"]
}"""


def get_model_action(
    client: OpenAI,
    observation: Dict,
    step: int,
    last_reward: float,
    history: List[str],
) -> Dict:
    """Call the LLM to get a triage action for the current observation."""
    ticket_context = f"""
TICKET ID: {observation.get('ticket_id', 'N/A')}
CUSTOMER TIER: {observation.get('customer_tier', 'unknown')}
PREVIOUS CONTACTS: {observation.get('previous_contacts', 0)}
ATTACHMENTS: {', '.join(observation.get('attachments', [])) or 'None'}

SUBJECT: {observation.get('subject', '')}

BODY:
{observation.get('body', '')}

TASK: {observation.get('task_description', '')}
STEP: {step}
"""
    if last_reward > 0 and step > 1:
        feedback = observation.get('feedback', '')
        ticket_context += f"\nPREVIOUS FEEDBACK: {feedback}\nPREVIOUS REWARD: {last_reward:.3f}\n"

    user_msg = f"Triage this support ticket:\n{ticket_context}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1000,
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # Fallback: return a generic action
        return {
            "category": "general",
            "priority": "medium",
            "response": "Thank you for contacting support. We have received your ticket and will review it shortly.",
            "escalate": False,
            "resolve": True,
            "tags": [],
        }


# ─────────────────────────── Environment Client ───────────────────────────

class EnvClient:
    """HTTP client for the OpenEnv REST API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

    def reset(self, task_id: str, session_id: str = "default") -> Dict:
        r = self._client.post(f"{self.base_url}/reset", json={"task_id": task_id, "session_id": session_id})
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict, session_id: str = "default") -> Dict:
        r = self._client.post(f"{self.base_url}/step", json={"action": action, "session_id": session_id})
        r.raise_for_status()
        return r.json()

    def state(self, session_id: str = "default") -> Dict:
        r = self._client.get(f"{self.base_url}/state", params={"session_id": session_id})
        r.raise_for_status()
        return r.json()

    def close(self):
        self._client.close()


# ─────────────────────────── Run single task ───────────────────────────

def run_task(
    client: OpenAI,
    env: EnvClient,
    task_id: str,
    session_id: str,
) -> Dict:
    """Run a single task episode. Returns summary dict."""
    task_name = f"{BENCHMARK}/{task_id}"
    max_steps = MAX_STEPS_PER_TASK[task_id]

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = env.reset(task_id=task_id, session_id=session_id)
        obs = reset_result["observation"]
        last_reward = 0.0
        result = {"done": False, "observation": obs, "reward": 0.0}

        for step in range(1, max_steps + 1):
            if result.get("done"):
                break

            action = get_model_action(client, obs, step, last_reward, history)

            result = env.step(action, session_id=session_id)
            obs = result["observation"]
            reward = result.get("reward") or 0.0
            done = result.get("done", False)
            info = result.get("info", {})
            error = info.get("error")

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=json.dumps(action), reward=reward, done=done, error=error)

            history.append(f"Step {step}: reward={reward:.3f}")

            if done:
                break

        # Score = mean per-step reward (clamped)
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


# ─────────────────────────── Main ───────────────────────────

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or OPENAI_API_KEY not set", flush=True)
        sys.exit(1)

    print(f"[DEBUG] Connecting to LLM: {API_BASE_URL} model={MODEL_NAME}", flush=True)
    print(f"[DEBUG] Connecting to env: {ENV_BASE_URL}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(base_url=ENV_BASE_URL)

    all_results = []

    try:
        for task_id in TASKS:
            session_id = f"session_{task_id}"
            print(f"\n[DEBUG] Starting task: {task_id}", flush=True)
            result = run_task(client, env, task_id, session_id)
            all_results.append(result)
            print(f"[DEBUG] Task {task_id} done: score={result['score']:.4f} success={result['success']}", flush=True)

    finally:
        env.close()

    # Summary
    overall_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
    print("\n" + "=" * 60, flush=True)
    print("BASELINE RESULTS", flush=True)
    print("=" * 60, flush=True)
    for r in all_results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(f"  {r['task_id']:8s}  score={r['score']:.4f}  steps={r['steps']}  {status}", flush=True)
    print(f"\n  Overall score: {overall_score:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
