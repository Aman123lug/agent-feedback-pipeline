"""
Adaptive Feedback Pipeline — Package

Split from the original monolith (pipeline.py) into clean modules.
All logic preserved — just organized like a human would write it.

Modules:
    config      — Azure OpenAI client, thresholds, constants
    session     — Per-session state, working memory helpers
    classifier  — LLM-based feedback classification
    guardrails  — Confidence gating, contradiction, dedup
    clustering  — Token-overlap similarity, LLM synthesis
    eval        — LLM-as-judge compliance scoring
    response    — Agent response generation (orchestrator + handoffs)
    engine      — Main pipeline orchestrator (run_pipeline)
    api         — FastAPI app, endpoints, dashboard serving
"""

from pipeline.engine import run_pipeline
from pipeline.api import app
from pipeline.session import get_session, get_full_state

__all__ = ["run_pipeline", "app", "get_session", "get_full_state"]
