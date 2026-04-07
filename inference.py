import os
import sys
import json
import httpx

try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY      = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

if not API_KEY:
    print("[ERROR] HF_TOKEN not set", flush=True)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Logging helpers ────────────────────────────────────────────────────────
def log(obj: dict):
    print(json.dumps(obj), flush=True)

def log_start(task: str, model: str):
    log({"event": "START", "task": task, "model": model})

def log_step(step: int, reward: float, action: dict, observation: dict):
    log({"event": "STEP", "step": step, "reward": reward,
         "action": action, "observation": observation})

def log_end(task: str, score: float, steps: int, success: bool):
    log({"event": "END", "task": task, "score": score,
         "steps": steps, "success": success})

# ── Agent ──────────────────────────────────────────────────────────────────
def run_task(task_id: str):
    # Reset
    r = httpx.post(f"{ENV_BASE_URL}/reset",
                   json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    obs = r.json()

    log_start(f"customer-support-triage/{task_id}", MODEL_NAME)

    step_num = 0
    total_reward = 0.0
    done = False

    while not done:
        step_num += 1

        # Build prompt
        prompt = f"""You are a customer support triage agent.

Ticket: {json.dumps(obs.get('ticket', obs), indent=2)}

Respond with a JSON object containing:
- category: one of [billing, technical, account, general]
- priority: one of [low, medium, high, urgent]
- escalate: true or false
- response: a helpful reply to the customer (2-3 sentences)

Return ONLY valid JSON, nothing else."""

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = completion.choices[0].message.content.strip()

        # Parse action
        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            action = json.loads(match.group()) if match else {
                "category": "general", "priority": "low",
                "escalate": False, "response": raw
            }

        # Step
        step_r = httpx.post(f"{ENV_BASE_URL}/step",
                             json={"action": action}, timeout=30)
        step_r.raise_for_status()
        result = step_r.json()

        reward   = result.get("reward", 0.0)
        done     = result.get("done", True)
        obs      = result.get("observation", {})
        total_reward += reward

        log_step(step_num, reward, action, obs)

    log_end(f"customer-support-triage/{task_id}",
            total_reward / max(step_num, 1), step_num, total_reward > 0.5)

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        try:
            run_task(task)
        except Exception as e:
            print(json.dumps({"event": "ERROR", "task": task,
                              "error": str(e)}), flush=True)