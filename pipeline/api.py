"""
Pipeline API — FastAPI app and endpoints.

Serves:
  POST /api/chat   — run the full pipeline for a user message
  GET  /api/state  — get the full system state for the dashboard
  GET  /           — serve the HTML dashboard
"""

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.engine import run_pipeline
from pipeline.session import get_full_state, get_session, _sessions, delete_session

logger = logging.getLogger("pipeline")

# ── FastAPI app ───────────────────────────────────────────────

app = FastAPI(title="Adaptive Feedback Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Run the full pipeline for a user message. Returns response + all stage data."""
    result = await run_pipeline(req.message, req.session_id)
    return result


@app.get("/api/state")
async def api_state(session_id: str = "default"):
    """Get full pipeline state for the dashboard."""
    return get_full_state(session_id)


@app.get("/api/threads")
async def api_threads():
    """List all active threads/sessions."""
    threads = []
    for sid, session in _sessions.items():
        turn = session.get("turn_count", 0)
        # Get first user message as preview
        preview = ""
        for s in session.get("working_memory", []):
            if s.get("type") == "query":
                preview = s["content"][:50]
                break
        threads.append({
            "id": sid,
            "turns": turn,
            "preview": preview or "New chat",
        })
    return {"threads": threads}


@app.delete("/api/threads/{session_id}")
async def api_delete_thread(session_id: str):
    """Delete a thread/session."""
    delete_session(session_id)
    return {"ok": True}


@app.get("/")
async def serve_dashboard():
    """Serve the HTML dashboard."""
    return FileResponse("dashboard.html", media_type="text/html")
