"""
FastAPI server exposing the OpenEnv standard REST API
for the Customer Support Triage environment.
"""
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import (
    Action, SupportTriageEnv, Category, Priority, SentimentLabel
)

app = FastAPI(
    title="Customer Support Triage — OpenEnv",
    description="An OpenEnv environment where AI agents learn to triage real-world customer support tickets.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: Dict[str, SupportTriageEnv] = {}


class ResetRequest(BaseModel):
    task_id: Optional[str]    = "easy"
    seed: Optional[int]       = None
    session_id: Optional[str] = "default"


class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: Optional[str] = "default"


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/")
def root():
    return {
        "name": "customer-support-triage",
        "version": "2.0.0",
        "description": "OpenEnv environment for AI-powered customer support triage",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/observation_space", "/action_space"],
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    try:
        env = SupportTriageEnv(task_id=req.task_id or "easy", seed=req.seed)
        _envs[req.session_id or "default"] = env
        result = env.reset()
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    env = _envs.get(req.session_id or "default")
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    try:
        action = Action(**req.action)
        result = env.step(action)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(session_id: str = "default"):
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    return env.state().model_dump()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Easy Triage",
                "description": "Simple, unambiguous tickets: password resets, invoice requests, general enquiries.",
                "max_steps": 5,
                "ticket_count": 4,
            },
            {
                "id": "medium",
                "name": "Medium Triage",
                "description": "Emotionally charged tickets with billing disputes, API errors, GDPR requests.",
                "max_steps": 8,
                "ticket_count": 4,
            },
            {
                "id": "hard",
                "name": "Hard Triage",
                "description": "High-stakes tickets: security breaches, SLA violations, multi-regulation compliance.",
                "max_steps": 10,
                "ticket_count": 4,
            },
        ]
    }


@app.get("/observation_space")
def observation_space():
    return {
        "type": "dict",
        "fields": {
            "ticket_id":           {"type": "string"},
            "subject":             {"type": "string"},
            "body":                {"type": "string"},
            "customer_tier":       {"type": "string", "enum": ["free", "pro", "enterprise"]},
            "previous_contacts":   {"type": "integer"},
            "attachments":         {"type": "array", "items": {"type": "string"}},
            "sentiment":           {"type": "string", "enum": ["positive", "neutral", "negative", "furious"]},
            "regulatory_flags":    {"type": "array", "items": {"type": "string"}},
            "sla_hours_remaining": {"type": "number", "nullable": True},
            "task_description":    {"type": "string"},
            "step":                {"type": "integer"},
            "max_steps":           {"type": "integer"},
            "current_score":       {"type": "number"},
            "feedback":            {"type": "string", "nullable": True},
        },
    }


@app.get("/action_space")
def action_space():
    return {
        "type": "dict",
        "fields": {
            "category":           {"type": "string", "enum": [c.value for c in Category]},
            "priority":           {"type": "string", "enum": [p.value for p in Priority]},
            "sentiment_detected": {"type": "string", "enum": [s.value for s in SentimentLabel]},
            "escalate":           {"type": "boolean"},
            "department":         {"type": "string", "enum": ["billing_team", "engineering", "legal", "security", "account_team", "general_support"]},
            "response":           {"type": "string"},
            "resolve":            {"type": "boolean"},
            "tags":               {"type": "array", "items": {"type": "string"}},
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)