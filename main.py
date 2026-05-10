"""
Adaptive Feedback Pipeline — Main Entry Point

POC demonstrating a real-time, context-aware, self-improving AI agent system.

Architecture:
- Streaming-first multi-agent system using agent-framework + gagent_core
- Layered memory: working (per-turn), episodic (session summaries), semantic (skills)
- Self-evolving skills from feedback — business_context feedback auto-registers as rules
- Decision tracing — every action logged with context snapshot

Run: python main.py
UI: http://localhost:8001 (gagent_core built-in UI)
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# Override APPLICATION_ID for this POC
os.environ.setdefault("APPLICATION_ID", "adaptive-feedback-poc")

import uvicorn
from pydantic import BaseModel
from fastapi import Request
from fastapi.responses import FileResponse
from pathlib import Path

from agent_framework import Message, Content

# gagent_core framework
from gagent_core.base_agent import BaseChatAgent
from gagent_core.base_app import create_app
from gagent_core.logs import logger
from gagent_core.websocket_manager import ConnectionManager
from gagent_core.schemas.events import EventFactory
from gagent_core.schemas.messages import TextMessage

# MAFStreamProcessor for streaming
try:
    from gagent_core.base_utils import MAFStreamProcessor
except ImportError:
    from gagent_core.base_utils import StreamProcessor as MAFStreamProcessor

# Local imports
from orchestrator import initialize_orchestrator
from stores import (
    feedback_store,
    skills_registry,
    episodic_memory,
    decision_tracer,
    chat_history,
    wm_manager,
    sc_manager,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ─── Stream Processor ────────────────────────────────────────────────────────

stream_processor = MAFStreamProcessor(
    tokens_before_delta=10,
    generate_followups=True,
    followup_instruction=(
        "Generate follow-up questions based on the conversation. "
        "If the user gave feedback, suggest testing the learned behavior. "
        "If the system learned a new skill, suggest verifying it works."
    ),
)


# ─── Custom Chat Agent ──────────────────────────────────────────────────────

class AdaptiveChatAgent(BaseChatAgent):
    """Chat agent with adaptive feedback pipeline."""

    def __init__(self, manager: "AdaptiveManager"):
        super().__init__(manager=manager)
        self.agent_type = "maf"
        self.last_message_by = "default"
        self.thought_mapping = {
            "provide_feedback": "Recording user feedback for system improvement",
            "get_feedback_summary": "Analyzing collected feedback and learned skills",
            "get_decision_trace": "Retrieving decision traces for debugging",
        }

    async def process_message(
        self,
        thread_id: str,
        query: str,
        user_name: str,
        user_context: Optional[str] = None,
        files: Optional[list] = None,
        metadata: Optional[dict] = None,
    ):
        """Process user message through the adaptive agent pipeline."""
        logger.info(f"[Adaptive] Processing message in {thread_id}: {query}")

        # Get session-scoped stores
        session_sc = sc_manager.get(thread_id)
        session_wm = wm_manager.get(thread_id)

        # ── Store user message in chat history ───────────────────────────
        chat_history.add_message(thread_id, "user", query)

        # ── SENSE: Collect signal from user query ────────────────────────
        session_sc.sense_query(query, user_id=user_name)

        # Detect implicit signals (e.g. repeated topics indicate interest)
        patterns = session_sc.estimate_patterns()
        if patterns.get("repeated_topics"):
            session_sc.sense_implicit(
                f"User repeatedly asks about: {', '.join(patterns['repeated_topics'])}"
            )

        # Build agent query with context envelope
        agent_query = (
            f"thread_id:{thread_id}\n"
            f"user: {user_name or 'anonymous'}\n"
            f"feedback_stats: {json.dumps(feedback_store.summary())}\n"
            f"active_skills: {len(skills_registry.get_active_rules())}\n"
            f"working_memory_signals: {len(session_wm.get_relevant())}\n"
            f"message: {query}"
        )

        user_message = Message(role="user", contents=[Content.from_text(text=agent_query)])

        # Get workflow agent
        workflow_session = self.manager.active_threads.get(thread_id)
        if not workflow_session:
            logger.error(f"No workflow session for {thread_id}")
            return
        workflow_agent = workflow_session.get("workflow_agent")
        if not workflow_agent:
            logger.error(f"No workflow agent for {thread_id}")
            return

        # Stream response
        async def _stream():
            run_stream = workflow_agent.run(user_message, stream=True)
            async for event in run_stream:
                yield event

        maf_stream = stream_processor.process_stream(thread_id, _stream())

        # Track turns for episodic memory
        turn_count = workflow_session.get("turn_count", 0) + 1
        workflow_session["turn_count"] = turn_count

        full_response_parts = []
        async for message in maf_stream:
            # Capture text parts for chat history
            if hasattr(message, "delta") and message.delta:
                full_response_parts.append(message.delta)
            yield message

        # ── Store assistant response in chat history ─────────────────────
        if full_response_parts:
            full_text = "".join(full_response_parts)
            chat_history.add_message(thread_id, "assistant", full_text)

        # Every 5 turns, create an episodic memory summary
        if turn_count % 5 == 0:
            episodic_memory.add(
                session_id=thread_id,
                summary=f"Conversation at turn {turn_count}. Last query: {query[:100]}",
                turn_count=turn_count,
            )
            logger.info(f"[Episodic] Added session summary at turn {turn_count}")


# ─── Custom Connection Manager ──────────────────────────────────────────────

class AdaptiveManager(ConnectionManager):
    """Connection manager with adaptive feedback pipeline state."""

    def __init__(self):
        super().__init__(enable_citations=True)
        self.active_threads: dict = {}
        self._thread_users: dict[str, str] = {}

    async def initialize_agent(self, app, thread_id: str, session_type: str = "text"):
        """Initialize the adaptive workflow agent."""
        user_id = self._thread_users.get(thread_id)

        workflow_agent = await initialize_orchestrator(
            thread_id=thread_id,
            user_id=user_id,
        )

        self.active_threads[thread_id] = {
            "thread_id": thread_id,
            "workflow_agent": workflow_agent,
            "turn_count": 0,
        }

        logger.info(f"✅ Adaptive session ready for {thread_id}")

    async def connect(
        self,
        websocket,
        thread_id: str,
        created_by: str,
        include_history: bool,
        last_timestamp: str | None,
        session_type: str = "text",
    ):
        await super().connect(websocket, thread_id, created_by, include_history, last_timestamp, session_type)
        app = websocket.scope["app"]
        self._thread_users[thread_id] = created_by

        await self.initialize_agent(app, thread_id, session_type)

        # Send initial status events
        skills_count = len(skills_registry.get_active_rules())
        feedback_count = feedback_store.count()

        event = EventFactory.create_action_tracker(
            action_name="Adaptive Pipeline",
            session_id=thread_id,
            status="completed" if skills_count > 0 else "pending",
            description=(
                f"{skills_count} active skills, {feedback_count} feedback records"
                if skills_count > 0
                else "Give feedback to teach the system new behaviors"
            ),
        )
        await self.send_event(thread_id=thread_id, event=event.model_dump())

    async def disconnect(self, thread_id: str):
        self._thread_users.pop(thread_id, None)
        self.active_threads.pop(thread_id, None)
        # Clean up session-scoped ephemeral stores (chat history stays for API access)
        wm_manager.remove(thread_id)
        sc_manager.remove(thread_id)
        await super().disconnect(thread_id)
        logger.info(f"✅ Cleaned up adaptive session {thread_id}")


# ─── App Setup ───────────────────────────────────────────────────────────────

manager = AdaptiveManager()
agent = AdaptiveChatAgent(manager)
app = create_app(agent, manager)


# ─── API Endpoints ───────────────────────────────────────────────────────────

POC_DIR = Path(__file__).parent

@app.get("/dashboard")
async def serve_dashboard():
    """Serve the pipeline visualization dashboard."""
    return FileResponse(POC_DIR / "dashboard.html", media_type="text/html")

@app.get("/api/feedback")
async def api_get_feedback():
    """Get all feedback records."""
    records = feedback_store.get_all()
    return {
        "total": len(records),
        "records": [
            {
                "id": r.id,
                "classification": r.classification,
                "feedback": r.structured_feedback,
                "raw_feedback": r.raw_feedback,
                "business_context": r.business_context_payload,
                "state": r.incorporation_state,
                "session_id": r.session_id,
                "submitted_at": r.submitted_at,
            }
            for r in records
        ],
    }


@app.get("/api/skills")
async def api_get_skills():
    """Get all active skills (business rules learned from feedback)."""
    skills = skills_registry.get_all()
    return {
        "total": len(skills),
        "active": len([s for s in skills if s.active]),
        "skills": [
            {
                "id": s.id,
                "rule": s.rule,
                "source_feedback_id": s.source_feedback_id,
                "created_at": s.created_at,
                "active": s.active,
            }
            for s in skills
        ],
    }


@app.get("/api/traces")
async def api_get_traces(session_id: Optional[str] = None, limit: int = 20):
    """Get decision traces."""
    if session_id:
        traces = decision_tracer.get_for_session(session_id)
    else:
        traces = decision_tracer.get_recent(limit)
    return {
        "total": len(traces),
        "traces": [
            {
                "id": t.id,
                "timestamp": t.timestamp,
                "agent": t.agent_name,
                "action": t.action,
                "context": t.context_snapshot,
                "result": t.result_summary,
                "session_id": t.session_id,
            }
            for t in traces
        ],
    }


@app.get("/api/system_state")
async def api_system_state():
    """Get complete system state — feedback, skills, episodes, traces."""
    return {
        "feedback": feedback_store.summary(),
        "skills": {
            "total": len(skills_registry.get_all()),
            "active_rules": skills_registry.get_active_rules(),
        },
        "chat_threads": chat_history.all_threads(),
        "episodic_memory": {
            "total_episodes": len(episodic_memory.get_recent(100)),
            "recent": [
                {"summary": e.summary, "turn": e.turn_count, "session": e.session_id}
                for e in episodic_memory.get_recent(5)
            ],
        },
        "decision_traces": {
            "total": len(decision_tracer.get_recent(500)),
            "recent_count": len(decision_tracer.get_recent(10)),
        },
    }


@app.get("/api/chat_history")
async def api_chat_history(thread_id: str, limit: int = 50):
    """Get chat history for a specific thread."""
    messages = chat_history.get_recent(thread_id, limit)
    return {
        "thread_id": thread_id,
        "total": len(messages),
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
            }
            for m in messages
        ],
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ADAPTIVE FEEDBACK PIPELINE — POC")
    print("  Real-time, context-aware, self-improving agent system")
    print("=" * 60)
    print("\n  Features:")
    print("  • Feedback collection & LLM classification")
    print("  • Self-evolving skills (business rules from feedback)")
    print("  • Layered memory (working / episodic / semantic)")
    print("  • Decision tracing (context snapshots)")
    print("  • Multi-agent handoff (orchestrator → specialists)")
    print(f"\n  🎯 Dashboard: http://localhost:8001/dashboard")
    print(f"  💬 Agent UI:  http://localhost:8001 (gagent_core)")
    print(f"  📊 API:       http://localhost:8001/api/system_state")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8001)
