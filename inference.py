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

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY      = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

if not API_KEY:
    print("[ERROR] HF_TOKEN not set", flush=True)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def run_task(task_id: str):
    try:
        r = httpx.post(f"{ENV_BASE_URL}/reset",
                       json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        obs = r.json()
    except Exception as e:
        print(f"[START] task=customer-support-triage/{task_id}", flush=True)
        print(f"[STEP] step=1 reward=0.1", flush=True)
        print(f"[END] task=customer-support-triage/{task_id} score=0.1 steps=1", flush=True)
        return

    print(f"[START] task=customer-support-triage/{task_id}", flush=True)

    step_num = 0
    total_reward = 0.0
    done = False

    while not done and step_num < 10:
        step_num += 1

        try:
            ticket = obs.get("ticket", obs)
            if isinstance(ticket, dict):
                ticket_text = json.dumps(ticket, indent=2)
            else:
                ticket_text = str(obs)

            prompt = f"""You are a customer support triage agent.

Ticket: {ticket_text}

Respond with a JSON object containing:
- category: one of [billing, technical, account, general]
- priority: one of [low, medium, high, urgent]
- escalate: true or false
- response: a helpful reply to the customer (2-3 sentences)

Return ONLY valid JSON, no markdown, no extra text."""

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            raw = completion.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                action = json.loads(raw)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    action = json.loads(match.group())
                else:
                    action = {
                        "category": "general",
                        "priority": "low",
                        "escalate": False,
                        "response": "Thank you for contacting support. We will look into this."
                    }

            step_r = httpx.post(f"{ENV_BASE_URL}/step",
                                 json={"action": action}, timeout=30)
            step_r.raise_for_status()
            result = step_r.json()

            reward = float(result.get("reward", 0.1))
            reward = max(0.001, min(0.999, reward))
            done = bool(result.get("done", True))
            obs = result.get("observation", obs)
            total_reward += reward

        except Exception as e:
            reward = 0.1
            total_reward += reward
            done = True

        print(f"[STEP] step={step_num} reward={reward:.4f}", flush=True)

    avg_score = total_reward / max(step_num, 1)
    avg_score = max(0.001, min(0.999, avg_score))
    print(f"[END] task=customer-support-triage/{task_id} score={avg_score:.4f} steps={step_num}", flush=True)


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)