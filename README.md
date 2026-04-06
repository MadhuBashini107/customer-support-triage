---
title: Customer Support Triage
emoji: 🎫
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---
# 🎫 Customer Support Triage — OpenEnv Environment

An OpenEnv-compliant reinforcement learning environment where AI agents learn to **triage customer support tickets**: classify, prioritize, decide on escalation, and draft professional responses.

---

## 🌍 Environment Description

Customer support triage is a high-value, high-volume real-world task performed by millions of workers daily. A triage agent must:

1. **Classify** the ticket into the correct category (billing, technical, account, general, refund)
2. **Prioritize** the ticket (low → medium → high → urgent)
3. **Decide** whether to escalate to a human agent
4. **Draft** a professional, empathetic response addressing all customer concerns

The environment includes 9 pre-built tickets spanning 3 difficulty levels, from simple password-reset requests to complex GDPR/SLA/security incidents.

---

## 📐 Action & Observation Spaces

### Observation Space

| Field | Type | Description |
|---|---|---|
| `ticket_id` | string | Unique ticket identifier |
| `subject` | string | Ticket subject line |
| `body` | string | Full ticket body from customer |
| `customer_tier` | string | `free` \| `pro` \| `enterprise` |
| `previous_contacts` | int | How many times this customer has contacted support before |
| `attachments` | list[str] | List of attachment filenames |
| `task_description` | string | Instruction describing the agent's objective |
| `step` | int | Current step number |
| `max_steps` | int | Maximum steps for this episode |
| `current_score` | float | Cumulative reward so far [0.0–1.0] |
| `feedback` | string\|null | Grader feedback from the previous step |

### Action Space

| Field | Type | Description |
|---|---|---|
| `category` | string | `billing` \| `technical` \| `account` \| `general` \| `refund` |
| `priority` | string | `low` \| `medium` \| `high` \| `urgent` |
| `response` | string | Draft response to send to the customer |
| `escalate` | bool | Whether to escalate to a human agent |
| `resolve` | bool | Set `true` to mark ticket resolved and end episode |
| `tags` | list[str] | Optional tags to apply |

---

## 🎯 Tasks

### Task 1: `easy`
**Difficulty:** Easy | **Max Steps:** 5

Simple, single-issue tickets: password resets, invoice requests, business hours inquiries. Clear correct category (always `account` or `billing` or `general`), always `low` priority, never requires escalation. The agent should be able to score 0.7+ with a basic prompt.

**Example ticket:** *"Hi, I forgot my password and need to reset it…"*

**Grading weights:**
- Category accuracy: 0.25
- Priority accuracy: 0.25
- Escalation decision: 0.15
- Response quality: 0.35

---

### Task 2: `medium`
**Difficulty:** Medium | **Max Steps:** 8

Multi-issue tickets involving duplicate billing charges, API errors affecting production systems, and plan downgrade requests near renewal dates. Requires nuanced priority decisions (some require `high`) and correct escalation decisions.

**Example ticket:** *"I was charged $49.99 twice on March 15th…"*

---

### Task 3: `hard`
**Difficulty:** Hard | **Max Steps:** 10

Complex, high-stakes tickets requiring `urgent` priority and escalation in all cases:
- Suspected account breach with legal threats and FTC/ICO mention
- Enterprise SLA outage with 500+ affected users, $50k/hour loss
- Multi-issue GDPR Article 17 erasure request + $1,847 refund + account closure

Responses must address **every** customer demand while being empathetic and professional. Prohibited phrases are tested.

---

## 🏆 Reward Function

Per-step reward is computed by the grader (0.0–1.0) based on:

| Component | Max Reward | Scoring |
|---|---|---|
| Category classification | 0.25 | Full credit for exact match; 0.05 for wrong category |
| Priority classification | 0.25 | Full credit for exact; partial credit (−0.08/level) for adjacent |
| Escalation decision | 0.15 | Full credit for correct; 0.05 for over-escalation; 0 for missing escalation |
| Response quality | 0.35 | Based on key topic coverage + required phrase inclusion; −0.10 for prohibited phrases |

The episode score is the **mean per-step reward**, clamped to [0.0, 1.0]. Agents are rewarded for partial progress — a response that covers most topics still earns partial credit.

---

## 🚀 Setup & Usage

### Prerequisites

- Docker
- Python 3.9+
- OpenAI-compatible API access

### 1. Build and run the environment server

```bash
docker build -t customer-support-triage .
docker run -p 7860:7860 customer-support-triage
```

The server starts at `http://localhost:7860`. Verify with:

```bash
curl http://localhost:7860/health
# {"status":"ok"}
```

### 2. Test the API manually

```bash
# Reset to easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "session_id": "test"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "action": {
      "category": "account",
      "priority": "low",
      "response": "Hi! To reset your password, click Forgot Password and check your spam folder for the reset email.",
      "escalate": false,
      "resolve": true,
      "tags": ["password-reset"]
    }
  }'

# Check state
curl "http://localhost:7860/state?session_id=test"
```

### 3. Run the baseline inference script

```bash
pip install openai httpx

export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key-here"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

---

## 📊 Baseline Scores

Measured with `gpt-4o-mini` (temperature=0.2):

| Task | Score | Steps | Notes |
|---|---|---|---|
| easy | ~0.82 | 1–2 | Strong category + priority + clear responses |
| medium | ~0.68 | 3–5 | Escalation decisions are trickier |
| hard | ~0.55 | 5–8 | GDPR/SLA knowledge needed for full score |
| **Overall** | **~0.68** | — | Above success threshold (0.50) |

---

## 🏗️ Project Structure

```
support-triage-env/
├── Dockerfile
├── openenv.yaml
├── inference.py          # Baseline inference script (root level, required)
├── README.md
└── server/
    ├── main.py           # FastAPI OpenEnv server
    ├── environment.py    # Core env logic, graders, ticket bank
    └── requirements.txt
```

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Yes | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | Yes | Hugging Face / API key used for LLM authentication |
| `ENV_BASE_URL` | No | Environment server URL (default: `http://localhost:7860`) |

---

## 📋 OpenEnv API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Environment info |
| `GET /health` | GET | Health check |
| `POST /reset` | POST | Reset environment, returns initial observation |
| `POST /step` | POST | Take an action, returns observation + reward + done |
| `GET /state` | GET | Get full current state |
| `GET /tasks` | GET | List all tasks |
| `GET /observation_space` | GET | JSON schema of observation space |
| `GET /action_space` | GET | JSON schema of action space |

---

## 💡 Tips for Agents

- **Read the ticket body completely** — many scoring cues are buried in the text
- **Enterprise tier + legal threats = urgent + escalate**
- **API/production errors = high/urgent, usually escalate**
- **Response quality matters most** (0.35 weight) — address every concern raised
- **Set `resolve: true`** to end the episode and lock in your score
- Use `feedback` in the observation to learn from previous steps

---

## 🏷️ HuggingFace Space

This environment is deployed as a HuggingFace Space with the `openenv` tag.

Space URL: `https://huggingface.co/spaces/[your-username]/customer-support-triage`
