"""
FastAPI server exposing the OpenEnv standard REST API
for the Customer Support Triage environment.
"""
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import (
    Action,
    Observation,
    Reward,
    ResetResult,
    StateResult,
    StepResult,
    SupportTriageEnv,
)

app = FastAPI(
    title="Customer Support Triage – OpenEnv",
    description="An OpenEnv environment for AI agents to learn customer support triage: classify, prioritize, escalate, and respond to tickets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment registry (single-session for simplicity)
_envs: Dict[str, SupportTriageEnv] = {}


def _get_env(session_id: str) -> SupportTriageEnv:
    if session_id not in _envs:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _envs[session_id]


# ─────────────────────────── Health ───────────────────────────

@app.get("/")
def root():
    return {
        "name": "customer-support-triage",
        "version": "1.0.0",
        "status": "ok",
        "tasks": ["easy", "medium", "hard"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# ─────────────────────────── OpenEnv API ───────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    seed: Optional[int] = None
    session_id: Optional[str] = "default"


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    env = SupportTriageEnv(task_id=req.task_id, seed=req.seed)
    _envs[req.session_id] = env
    result = env.reset()
    return result


class StepRequest(BaseModel):
    action: Action
    session_id: str = "default"


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    env = _get_env(req.session_id)
    result = env.step(req.action)
    return result


@app.get("/state", response_model=StateResult)
def state(session_id: str = Query("default")):
    env = _get_env(session_id)
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "difficulty": "easy",
                "description": "Simple, single-issue tickets (password reset, invoice request). Clear category and low priority.",
                "max_steps": 5,
                "ticket_count": 3,
            },
            {
                "id": "medium",
                "difficulty": "medium",
                "description": "Multi-issue tickets (duplicate charges, API errors). Requires correct escalation decisions.",
                "max_steps": 8,
                "ticket_count": 3,
            },
            {
                "id": "hard",
                "difficulty": "hard",
                "description": "Complex, high-stakes tickets: data breaches, SLA violations, GDPR requests. Requires nuanced response.",
                "max_steps": 10,
                "ticket_count": 3,
            },
        ]
    }


@app.get("/observation_space")
def observation_space():
    return Observation.model_json_schema()


@app.get("/action_space")
def action_space():
    return Action.model_json_schema()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
