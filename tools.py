"""
Adaptive Feedback Pipeline — Agent Tools

Tools that agents can call:
- provide_feedback: collect user feedback (same pattern as gagent_core)
- get_feedback_summary: view feedback analytics
- get_active_skills: see what the system has learned
- search_knowledge: semantic search over knowledge base
- log_decision: explicit decision tracing
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

from pydantic import Field

from stores import (
    FeedbackRecord,
    feedback_store,
    skills_registry,
    episodic_memory,
    decision_tracer,
)

import logging
logger = logging.getLogger(__name__)


# ─── LLM-based feedback classification (same pattern as gagent_core) ─────────

CLASSIFICATION_PROMPT = """You are a feedback classification assistant.

Given user feedback, return a JSON array. Each element:
  "classification": "bug" | "reprompting" | "business_context" | "other"
  "structured_feedback": concise rephrasing (max 200 chars)
  "business_context_payload": if classification is "business_context", the actionable rule string; otherwise null

Classification guide:
  bug              — something broken or incorrect
  reprompting      — better prompt/guardrail would fix it
  business_context — domain rule the agent should follow (e.g. "Always respond in Arabic")
  other            — general praise, requests, or uncategorised

Return valid JSON array only. No markdown fences."""


async def classify_feedback(raw_feedback: str) -> list[dict]:
    """Classify feedback using Azure OpenAI (with fallback for POC)."""
    try:
        from openai import AsyncAzureOpenAI
        from gagent_core.settings import settings

        client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai.ENDPOINT or "",
            api_key=settings.azure_openai.API_KEY or "",
            api_version=settings.azure_openai.API_VERSION or "2024-06-01",
        )
        deployment = settings.azure_openai.CHAT_DEPLOYMENT_NAME or "gpt-4o"

        response = await client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": f"User feedback:\n{raw_feedback}"},
            ],
            temperature=0,
            max_tokens=512,
        )
        content = response.choices[0].message.content or "[]"
        result = json.loads(content)
        if isinstance(result, dict):
            result = [result]
        return result
    except Exception as e:
        logger.warning(f"Feedback classification failed, using fallback: {e}")
        # Fallback: simple keyword-based classification
        lower = raw_feedback.lower()
        if any(w in lower for w in ["bug", "broken", "error", "wrong", "crash"]):
            cls = "bug"
        elif any(w in lower for w in ["always", "never", "should", "must", "rule", "prefer"]):
            cls = "business_context"
        elif any(w in lower for w in ["prompt", "rephrase", "better", "instead"]):
            cls = "reprompting"
        else:
            cls = "other"

        biz_payload = raw_feedback if cls == "business_context" else None
        return [{
            "classification": cls,
            "structured_feedback": raw_feedback[:200],
            "business_context_payload": biz_payload,
        }]


async def _process_feedback_background(
    feedback_id: str,
    raw_feedback: str,
    session_id: str,
    user_id: Optional[str],
    agent_name: Optional[str],
    submitted_at: str,
) -> None:
    """Background: classify → store → auto-register skills."""
    try:
        items = await classify_feedback(raw_feedback)

        for idx, item in enumerate(items):
            record_id = feedback_id if idx == 0 else str(uuid.uuid4())
            record = FeedbackRecord(
                id=record_id,
                raw_feedback=raw_feedback,
                classification=item.get("classification", "other"),
                structured_feedback=item.get("structured_feedback", raw_feedback[:200]),
                business_context_payload=item.get("business_context_payload"),
                session_id=session_id,
                user_id=user_id,
                agent_name=agent_name,
                submitted_at=submitted_at,
            )
            feedback_store.add(record)
            logger.info(
                f"Feedback stored: id={record_id} classification={record.classification} "
                f"({idx + 1}/{len(items)})"
            )

            # Auto-register business_context as skills immediately
            if record.classification == "business_context" and record.business_context_payload:
                skill = skills_registry.register(
                    rule=record.business_context_payload,
                    source_feedback_id=record.id,
                )
                feedback_store.mark_incorporated(record.id)
                logger.info(f"Auto-registered skill from feedback: {skill.rule[:80]}")

    except Exception as e:
        logger.error(f"Feedback processing failed: {e}", exc_info=True)


# ─── Tool factory (same pattern as gagent_core create_feedback_tools) ────────

def create_adaptive_feedback_tools(session_id: str, user_id: Optional[str] = None) -> list:
    """Create feedback + introspection tools with session context pre-injected."""

    async def provide_feedback(
        feedback: Annotated[str, Field(description="The user's verbatim feedback text. Pass raw text, do not classify.")],
        agent_name: Annotated[Optional[str], Field(description="Agent this feedback is about, or null.", default=None)] = None,
    ) -> str:
        """Collect natural-language feedback from the user. Classification happens automatically in background."""
        feedback_id = str(uuid.uuid4())
        submitted_at = datetime.now(timezone.utc).isoformat()

        asyncio.create_task(
            _process_feedback_background(
                feedback_id=feedback_id,
                raw_feedback=feedback,
                session_id=session_id,
                user_id=user_id,
                agent_name=agent_name,
                submitted_at=submitted_at,
            )
        )

        return "✅ Thank you for your feedback! It has been recorded and will be used to improve the system."

    async def get_feedback_summary() -> str:
        """Get a summary of all collected feedback and learned skills."""
        summary = feedback_store.summary()
        active_skills = skills_registry.get_active_rules()
        recent_feedback = feedback_store.get_all()[:5]

        result = {
            "feedback_stats": summary,
            "active_skills_count": len(active_skills),
            "active_skills": active_skills,
            "recent_feedback": [
                {
                    "classification": r.classification,
                    "feedback": r.structured_feedback,
                    "state": r.incorporation_state,
                }
                for r in recent_feedback
            ],
        }
        return json.dumps(result, indent=2)

    async def get_decision_trace() -> str:
        """Get recent decision traces showing what context the agent had for each action."""
        traces = decision_tracer.get_recent(10)
        result = [
            {
                "timestamp": t.timestamp,
                "agent": t.agent_name,
                "action": t.action,
                "context": t.context_snapshot,
                "result": t.result_summary,
            }
            for t in traces
        ]
        return json.dumps(result, indent=2)

    return [provide_feedback, get_feedback_summary, get_decision_trace]
