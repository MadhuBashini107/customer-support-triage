"""
Customer Support Triage Environment — Enhanced
Real-world task: classify, prioritize, route, and respond to customer support tickets.
Features: sentiment analysis scoring, SLA-awareness, multi-turn escalation, regulatory compliance detection.
"""
from __future__ import annotations

import json
import random
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ─────────────────────────────── Enums ───────────────────────────────

class Category(str, Enum):
    BILLING    = "billing"
    TECHNICAL  = "technical"
    ACCOUNT    = "account"
    GENERAL    = "general"
    REFUND     = "refund"
    COMPLIANCE = "compliance"
    OUTAGE     = "outage"

class Priority(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    URGENT = "urgent"

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL  = "neutral"
    NEGATIVE = "negative"
    FURIOUS  = "furious"

class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"

# ─────────────────────────────── OpenEnv Models ───────────────────────────────

class Observation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: str
    previous_contacts: int
    attachments: List[str]
    sentiment: str
    regulatory_flags: List[str]
    sla_hours_remaining: Optional[float]
    task_description: str
    step: int
    max_steps: int
    current_score: float
    feedback: Optional[str] = None

class Action(BaseModel):
    category: Optional[str]   = Field(None, description="billing|technical|account|general|refund|compliance|outage")
    priority: Optional[str]   = Field(None, description="low|medium|high|urgent")
    sentiment_detected: Optional[str] = Field(None, description="positive|neutral|negative|furious")
    response: Optional[str]   = Field(None, description="Draft reply to the customer")
    escalate: Optional[bool]  = Field(False, description="Escalate to human agent")
    department: Optional[str] = Field(None, description="billing_team|engineering|legal|security|account_team|general_support")
    resolve: Optional[bool]   = Field(False, description="Mark ticket as resolved")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags to apply")

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

TICKETS: Dict[str, List[Dict]] = {
    "easy": [
        {
            "ticket_id": "TKT-E001",
            "subject": "How do I reset my password?",
            "body": "Hi, I forgot my password and the reset email isn't arriving. Can you help?",
            "customer_tier": "free",
            "previous_contacts": 0,
            "attachments": [],
            "sentiment": SentimentLabel.NEUTRAL,
            "regulatory_flags": [],
            "sla_hours_remaining": 48.0,
            "ground_truth": {
                "category": Category.ACCOUNT,
                "priority": Priority.LOW,
                "sentiment": SentimentLabel.NEUTRAL,
                "escalate": False,
                "department": "account_team",
                "key_topics": ["password", "reset", "email"],
                "response_must_include": ["reset", "email", "spam"],
                "response_must_not_include": [],
            },
        },
        {
            "ticket_id": "TKT-E002",
            "subject": "Can I get an invoice for my subscription?",
            "body": "Please send me a copy of my latest invoice for accounting purposes. My email is user@company.com.",
            "customer_tier": "pro",
            "previous_contacts": 0,
            "attachments": [],
            "sentiment": SentimentLabel.NEUTRAL,
            "regulatory_flags": [],
            "sla_hours_remaining": 48.0,
            "ground_truth": {
                "category": Category.BILLING,
                "priority": Priority.LOW,
                "sentiment": SentimentLabel.NEUTRAL,
                "escalate": False,
                "department": "billing_team",
                "key_topics": ["invoice", "billing", "subscription"],
                "response_must_include": ["invoice", "billing"],
                "response_must_not_include": [],
            },
        },
        {
            "ticket_id": "TKT-E003",
            "subject": "What are your support hours?",
            "body": "Hello, I'd like to know what hours your support team is available and whether you offer phone support.",
            "customer_tier": "free",
            "previous_contacts": 0,
            "attachments": [],
            "sentiment": SentimentLabel.POSITIVE,
            "regulatory_flags": [],
            "sla_hours_remaining": 72.0,
            "ground_truth": {
                "category": Category.GENERAL,
                "priority": Priority.LOW,
                "sentiment": SentimentLabel.POSITIVE,
                "escalate": False,
                "department": "general_support",
                "key_topics": ["hours", "support", "phone"],
                "response_must_include": ["hours", "support"],
                "response_must_not_include": [],
            },
        },
        {
            "ticket_id": "TKT-E004",
            "subject": "How do I cancel my subscription?",
            "body": "I'd like to cancel my subscription. Can you walk me through the steps? I'm not finding it in the settings.",
            "customer_tier": "pro",
            "previous_contacts": 0,
            "attachments": [],
            "sentiment": SentimentLabel.NEUTRAL,
            "regulatory_flags": [],
            "sla_hours_remaining": 48.0,
            "ground_truth": {
                "category": Category.BILLING,
                "priority": Priority.LOW,
                "sentiment": SentimentLabel.NEUTRAL,
                "escalate": False,
                "department": "billing_team",
                "key_topics": ["cancel", "subscription", "settings"],
                "response_must_include": ["cancel", "settings"],
                "response_must_not_include": [],
            },
        },
    ],
    "medium": [
        {
            "ticket_id": "TKT-M001",
            "subject": "Charged twice for my subscription",
            "body": "I was charged $49.99 twice on March 15th. Order numbers ORD-4521 and ORD-4522. I need an immediate refund. This is unacceptable.",
            "customer_tier": "pro",
            "previous_contacts": 2,
            "attachments": ["bank_statement.pdf"],
            "sentiment": SentimentLabel.NEGATIVE,
            "regulatory_flags": [],
            "sla_hours_remaining": 12.0,
            "ground_truth": {
                "category": Category.BILLING,
                "priority": Priority.HIGH,
                "sentiment": SentimentLabel.NEGATIVE,
                "escalate": False,
                "department": "billing_team",
                "key_topics": ["duplicate", "charge", "refund", "ORD-4521"],
                "response_must_include": ["refund", "investigate", "apolog"],
                "response_must_not_include": ["not our fault", "can't help"],
            },
        },
        {
            "ticket_id": "TKT-M002",
            "subject": "API returning 500 errors intermittently",
            "body": "Our integration has been getting random 500 errors from your API since yesterday. Error rate is about 5%. Request IDs: req_abc123, req_def456. This is affecting our production system.",
            "customer_tier": "pro",
            "previous_contacts": 0,
            "attachments": ["error_logs.txt"],
            "sentiment": SentimentLabel.NEGATIVE,
            "regulatory_flags": [],
            "sla_hours_remaining": 8.0,
            "ground_truth": {
                "category": Category.TECHNICAL,
                "priority": Priority.HIGH,
                "sentiment": SentimentLabel.NEGATIVE,
                "escalate": True,
                "department": "engineering",
                "key_topics": ["API", "500", "error", "production"],
                "response_must_include": ["investigate", "engineer", "monitor"],
                "response_must_not_include": ["not our problem"],
            },
        },
        {
            "ticket_id": "TKT-M003",
            "subject": "Need to downgrade plan before renewal in 3 days",
            "body": "My renewal is in 3 days (March 20). I want to downgrade from Enterprise ($299/mo) to Pro ($49/mo) before renewal. I don't want to be charged the full Enterprise price again.",
            "customer_tier": "enterprise",
            "previous_contacts": 1,
            "attachments": [],
            "sentiment": SentimentLabel.NEUTRAL,
            "regulatory_flags": [],
            "sla_hours_remaining": 72.0,
            "ground_truth": {
                "category": Category.BILLING,
                "priority": Priority.MEDIUM,
                "sentiment": SentimentLabel.NEUTRAL,
                "escalate": False,
                "department": "billing_team",
                "key_topics": ["downgrade", "renewal", "enterprise", "pro"],
                "response_must_include": ["downgrade", "renewal", "plan"],
                "response_must_not_include": [],
            },
        },
        {
            "ticket_id": "TKT-M004",
            "subject": "Data export not working — GDPR request",
            "body": "I submitted a data export request 28 days ago. Under GDPR I am entitled to this data within 30 days. I have not received anything yet. My request ID is GDPR-2024-0391.",
            "customer_tier": "free",
            "previous_contacts": 3,
            "attachments": [],
            "sentiment": SentimentLabel.NEGATIVE,
            "regulatory_flags": ["GDPR"],
            "sla_hours_remaining": 48.0,
            "ground_truth": {
                "category": Category.COMPLIANCE,
                "priority": Priority.HIGH,
                "sentiment": SentimentLabel.NEGATIVE,
                "escalate": True,
                "department": "legal",
                "key_topics": ["GDPR", "export", "data", "30 days", "GDPR-2024-0391"],
                "response_must_include": ["GDPR", "legal", "priorit"],
                "response_must_not_include": ["ignore", "can't help"],
            },
        },
    ],
    "hard": [
        {
            "ticket_id": "TKT-H001",
            "subject": "URGENT: Account compromised — suspicious login from Russia",
            "body": "I believe my account was hacked. Seeing login activity from IP 192.168.1.1 from Russia that wasn't me at 3:45 AM UTC March 17. I've changed my password but I'm worried my payment info was accessed. Need immediate action or I'll report this to the FTC and my bank.",
            "customer_tier": "enterprise",
            "previous_contacts": 0,
            "attachments": ["screenshot_logins.png"],
            "sentiment": SentimentLabel.FURIOUS,
            "regulatory_flags": ["FTC", "PCI-DSS"],
            "sla_hours_remaining": 2.0,
            "ground_truth": {
                "category": Category.ACCOUNT,
                "priority": Priority.URGENT,
                "sentiment": SentimentLabel.FURIOUS,
                "escalate": True,
                "department": "security",
                "key_topics": ["breach", "hacked", "security", "login", "Russia", "FTC"],
                "response_must_include": ["security", "team", "immediate", "apolog"],
                "response_must_not_include": ["don't worry", "no problem", "not our fault"],
            },
        },
        {
            "ticket_id": "TKT-H002",
            "subject": "Platform outage 2.5 hours — SLA breach — 500 users affected",
            "body": "Your platform has been down for our organisation since 9:00 AM EST — that's 2.5 hours. We have an Enterprise SLA guaranteeing 99.9% uptime. This is already a breach. 500+ users cannot work. We are losing $50,000/hour. I need: 1) ETA for resolution 2) Root cause analysis 3) SLA credit calculation 4) Emergency phone number for our CTO. Escalate to VP level immediately.",
            "customer_tier": "enterprise",
            "previous_contacts": 3,
            "attachments": ["sla_agreement.pdf", "user_impact_report.xlsx"],
            "sentiment": SentimentLabel.FURIOUS,
            "regulatory_flags": ["SLA"],
            "sla_hours_remaining": 0.5,
            "ground_truth": {
                "category": Category.OUTAGE,
                "priority": Priority.URGENT,
                "sentiment": SentimentLabel.FURIOUS,
                "escalate": True,
                "department": "engineering",
                "key_topics": ["outage", "SLA", "enterprise", "credit", "RCA", "CTO", "VP"],
                "response_must_include": ["escalat", "engineer", "SLA", "apolog"],
                "response_must_not_include": ["unfortunately we cannot", "not our fault", "no problem"],
            },
        },
        {
            "ticket_id": "TKT-H003",
            "subject": "Unauthorized charges + GDPR erasure + account closure",
            "body": "I am formally requesting: 1) Immediate refund of $1,847 in unauthorized charges over 6 months (see attached statements) 2) Permanent closure of account #ACC-98765 3) Complete deletion of all personal data under GDPR Article 17 (right to erasure). I am a UK resident. You have 30 days to comply or I will file with the ICO. I will also initiate a chargeback with my bank within 7 days if not refunded.",
            "customer_tier": "pro",
            "previous_contacts": 4,
            "attachments": ["bank_statements_6mo.pdf", "account_closure_form.pdf"],
            "sentiment": SentimentLabel.FURIOUS,
            "regulatory_flags": ["GDPR", "ICO", "chargeback"],
            "sla_hours_remaining": 4.0,
            "ground_truth": {
                "category": Category.COMPLIANCE,
                "priority": Priority.URGENT,
                "sentiment": SentimentLabel.FURIOUS,
                "escalate": True,
                "department": "legal",
                "key_topics": ["GDPR", "refund", "closure", "chargeback", "ICO", "Article 17"],
                "response_must_include": ["GDPR", "legal", "escalat", "refund"],
                "response_must_not_include": ["we can't delete", "no refund", "not our fault"],
            },
        },
        {
            "ticket_id": "TKT-H004",
            "subject": "Data breach notification — customer PII exposed via your API bug",
            "body": "We discovered that due to a bug in your API (reported in your status page incident INC-2024-0042), our customers' PII including names and emails were exposed to other tenants for approximately 6 hours. We are a healthcare adjacent company and this may trigger HIPAA notification requirements. We need: 1) Full incident report 2) Confirmation of scope 3) Your DPO contact details 4) Written assurance this is fixed. We have legal counsel involved.",
            "customer_tier": "enterprise",
            "previous_contacts": 1,
            "attachments": ["incident_notice.pdf"],
            "sentiment": SentimentLabel.FURIOUS,
            "regulatory_flags": ["HIPAA", "GDPR", "PII", "data-breach"],
            "sla_hours_remaining": 1.0,
            "ground_truth": {
                "category": Category.COMPLIANCE,
                "priority": Priority.URGENT,
                "sentiment": SentimentLabel.FURIOUS,
                "escalate": True,
                "department": "legal",
                "key_topics": ["breach", "PII", "HIPAA", "DPO", "incident", "INC-2024-0042"],
                "response_must_include": ["legal", "DPO", "escalat", "apolog", "incident"],
                "response_must_not_include": ["not our fault", "can't help", "no problem"],
            },
        },
    ],
}

# ─────────────────────────────── Helpers ───────────────────────────────

PRIORITY_ORDER = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.URGENT]

def clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return round(min(hi, max(lo, value)), 4)

def detect_sentiment(text: str) -> SentimentLabel:
    text = text.lower()
    furious_words = ["furious", "outrage", "legal action", "lawsuit", "ftc", "ico", "chargeback", "unacceptable", "immediate", "breach"]
    negative_words = ["frustrated", "angry", "disappointed", "refund", "error", "wrong", "problem", "issue"]
    positive_words = ["thank", "great", "happy", "pleased", "excellent"]
    if any(w in text for w in furious_words):
        return SentimentLabel.FURIOUS
    if any(w in text for w in negative_words):
        return SentimentLabel.NEGATIVE
    if any(w in text for w in positive_words):
        return SentimentLabel.POSITIVE
    return SentimentLabel.NEUTRAL

# ─────────────────────────────── Grader ───────────────────────────────

def grade_action(
    action: Action,
    ground_truth: Dict,
    step: int,
    max_steps: int,
    ticket: Dict,
) -> Tuple[float, Dict[str, float], str]:

    breakdown: Dict[str, float] = {}
    messages: List[str] = []

    # 1. Category (0.20) — now includes new categories
    cat_score = 0.001
    if action.category:
        try:
            cat = Category(action.category.lower())
            if cat == ground_truth["category"]:
                cat_score = 0.20
                messages.append("✓ Category correct")
            else:
                cat_score = 0.04
                messages.append(f"✗ Category wrong (got {cat.value}, expected {ground_truth['category'].value})")
        except ValueError:
            messages.append(f"✗ Invalid category '{action.category}'")
    else:
        messages.append("✗ No category provided")
    breakdown["category"] = cat_score

    # 2. Priority (0.20) — partial credit by distance
    pri_score = 0.001
    if action.priority:
        try:
            pri = Priority(action.priority.lower())
            gt_pri = ground_truth["priority"]
            if pri == gt_pri:
                pri_score = 0.20
                messages.append("✓ Priority correct")
            else:
                diff = abs(PRIORITY_ORDER.index(pri) - PRIORITY_ORDER.index(gt_pri))
                pri_score = clamp(0.20 - diff * 0.07, 0.001, 0.18)
                messages.append(f"~ Priority off by {diff} level(s)")
        except ValueError:
            messages.append(f"✗ Invalid priority '{action.priority}'")
    else:
        messages.append("✗ No priority provided")
    breakdown["priority"] = pri_score

    # 3. Sentiment detection (0.10) — new creative dimension
    sent_score = 0.001
    if action.sentiment_detected:
        try:
            detected = SentimentLabel(action.sentiment_detected.lower())
            gt_sent = ground_truth["sentiment"]
            if detected == gt_sent:
                sent_score = 0.10
                messages.append("✓ Sentiment detected correctly")
            else:
                sent_score = 0.03
                messages.append(f"~ Sentiment mismatch (got {detected.value}, expected {gt_sent.value})")
        except ValueError:
            messages.append(f"✗ Invalid sentiment '{action.sentiment_detected}'")
    else:
        messages.append("✗ No sentiment provided")
    breakdown["sentiment"] = sent_score

    # 4. Escalation (0.10)
    gt_escalate = ground_truth.get("escalate", False)
    if action.escalate == gt_escalate:
        breakdown["escalation"] = 0.10
        messages.append("✓ Escalation correct")
    elif action.escalate and not gt_escalate:
        breakdown["escalation"] = 0.03
        messages.append("~ Over-escalated")
    else:
        breakdown["escalation"] = 0.001
        messages.append("✗ Should have escalated")

    # 5. Department routing (0.10) — new creative dimension
    dept_score = 0.001
    if action.department:
        gt_dept = ground_truth.get("department", "")
        if action.department.lower() == gt_dept.lower():
            dept_score = 0.10
            messages.append("✓ Department routing correct")
        else:
            dept_score = 0.03
            messages.append(f"~ Wrong department (got {action.department}, expected {gt_dept})")
    else:
        messages.append("✗ No department routing provided")
    breakdown["department"] = dept_score

    # 6. Response quality (0.30) — enhanced
    resp_score = 0.001
    if action.response and len(action.response.strip()) > 20:
        resp_lower = action.response.lower()
        key_topics       = ground_truth.get("key_topics", [])
        must_include     = ground_truth.get("response_must_include", [])
        must_not_include = ground_truth.get("response_must_not_include", [])

        topics_found   = sum(1 for t in key_topics if t.lower() in resp_lower)
        topic_ratio    = topics_found / len(key_topics) if key_topics else 1.0

        required_found = sum(1 for r in must_include if r.lower() in resp_lower)
        required_ratio = required_found / len(must_include) if must_include else 1.0

        prohibited     = any(p.lower() in resp_lower for p in must_not_include)

        resp_score = topic_ratio * 0.14 + required_ratio * 0.14
        if prohibited:
            resp_score = max(0.001, resp_score - 0.10)
            messages.append("✗ Response contains inappropriate phrase")
        if len(action.response) > 120:
            resp_score = min(0.30, resp_score + 0.04)

        # SLA urgency bonus — reward acknowledging urgency when SLA is low
        sla_remaining = ticket.get("sla_hours_remaining", 48.0)
        if sla_remaining is not None and sla_remaining <= 4.0:
            if any(w in resp_lower for w in ["urgent", "immediately", "priority", "right away", "escalat"]):
                resp_score = min(0.30, resp_score + 0.03)
                messages.append("✓ Response acknowledges urgency appropriately")

        # Regulatory acknowledgment bonus
        reg_flags = ticket.get("regulatory_flags", [])
        for flag in reg_flags:
            if flag.lower() in resp_lower:
                resp_score = min(0.30, resp_score + 0.02)
                messages.append(f"✓ Response acknowledges {flag} compliance")
                break

        messages.append(f"Response covers {topics_found}/{len(key_topics)} key topics")
    else:
        messages.append("✗ No response or too short")

    breakdown["response"] = clamp(resp_score, 0.001, 0.30)

    # 7. Resolve penalty
    if action.resolve and cat_score <= 0.04:
        breakdown["resolve_penalty"] = -0.05
        messages.append("✗ Resolved without proper categorization")
    else:
        breakdown["resolve_penalty"] = 0.0

    total = sum(breakdown.values())
    total = clamp(total)
    return total, breakdown, " | ".join(messages)


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
        self.task_id  = task_id
        self.seed     = seed
        self._rng     = random.Random(seed)
        self._step    = 0
        self._done    = False
        self._cumulative_reward = 0.0
        self._ticket: Dict  = {}
        self._actions: List = []
        self._last_feedback = ""
        self._max_steps     = 5

    def reset(self) -> ResetResult:
        self._step              = 0
        self._done              = False
        self._cumulative_reward = 0.0
        self._actions           = []
        self._last_feedback     = ""
        cfg                     = self.TASK_CONFIGS[self.task_id]
        self._ticket            = self._rng.choice(cfg["tickets"])
        self._max_steps         = cfg["max_steps"]
        return ResetResult(observation=self._make_observation())

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._make_observation(),
                reward=0.001,
                done=True,
                info={"error": "Episode already done"},
            )

        self._step += 1
        gt = self._ticket["ground_truth"]
        reward, breakdown, feedback = grade_action(
            action, gt, self._step, self._max_steps, self._ticket
        )

        self._cumulative_reward = clamp(
            self._cumulative_reward + reward / self._max_steps
        )
        self._last_feedback = feedback
        self._actions.append({
            "step":      self._step,
            "action":    action.model_dump(),
            "reward":    reward,
            "breakdown": breakdown,
        })

        done       = bool(action.resolve) or self._step >= self._max_steps
        self._done = done

        return StepResult(
            observation=self._make_observation(),
            reward=reward,
            done=done,
            info={
                "breakdown":         breakdown,
                "feedback":          feedback,
                "cumulative_reward": self._cumulative_reward,
            },
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
            "easy":   "Triage this support ticket: classify the category, set priority, detect sentiment, and draft a helpful response.",
            "medium": "Triage this ticket carefully. Correct category, priority, sentiment detection, and department routing are critical. Decide whether to escalate. Note any regulatory flags.",
            "hard":   "This is a high-stakes, multi-dimensional ticket. Classify accurately, set priority, detect customer sentiment, route to the correct department, decide escalation, acknowledge any regulatory/compliance requirements, and draft a thorough empathetic response.",
        }
        t = self._ticket
        return Observation(
            ticket_id=t.get("ticket_id", ""),
            subject=t.get("subject", ""),
            body=t.get("body", ""),
            customer_tier=t.get("customer_tier", "free"),
            previous_contacts=t.get("previous_contacts", 0),
            attachments=t.get("attachments", []),
            sentiment=t.get("sentiment", SentimentLabel.NEUTRAL).value
                if isinstance(t.get("sentiment"), SentimentLabel)
                else t.get("sentiment", "neutral"),
            regulatory_flags=t.get("regulatory_flags", []),
            sla_hours_remaining=t.get("sla_hours_remaining"),
            task_description=task_descs[self.task_id],
            step=self._step,
            max_steps=self._max_steps,
            current_score=round(self._cumulative_reward, 4),
            feedback=self._last_feedback if self._step > 0 else None,
        )