"""
Session State — Per-session in-memory state + working memory helpers.

Ephemeral by design: sessions live in memory, durable data lives in SQLite (stores.py).
"""

from datetime import datetime, timezone

from pipeline.config import DECAY_RATE, RELEVANCE_THRESHOLD


# ── Session store ─────────────────────────────────────────────

_sessions: dict[str, dict] = {}


def get_session(session_id: str = "default") -> dict:
    """Get or create a session. Ephemeral state lives here; durable state in SQLite."""
    if session_id not in _sessions:
        _sessions[session_id] = {
            "turn_count": 0,
            "working_memory": [],           # list[dict] — {type, content, relevance, timestamp}
            "chat_history": [],             # list[dict] — {role, content} for LLM context
            "eval_history": [],             # list[dict] — {turn, score, violations, compliant, summary}
            "feedback_clusters": [],        # list[dict] — {id, category, centroid_text, members, learning, ...}
            "feedback_records": [],         # in-memory mirror for clustering (_clustered flag)
        }
    return _sessions[session_id]


def delete_session(session_id: str):
    """Delete a session."""
    _sessions.pop(session_id, None)


# ── Working memory helpers ────────────────────────────────────

def decay_working_memory(session: dict):
    """Apply exponential decay to all signals. Evict below floor."""
    for s in session["working_memory"]:
        s["relevance"] *= DECAY_RATE
    session["working_memory"] = [
        s for s in session["working_memory"] if s["relevance"] >= 0.05
    ]


def add_signal(session: dict, sig_type: str, content: str, relevance: float = 1.0):
    """Ingest a new signal into working memory (with decay on existing)."""
    decay_working_memory(session)
    session["working_memory"].append({
        "type": sig_type,
        "content": content[:300],
        "relevance": relevance,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    if len(session["working_memory"]) > 20:
        session["working_memory"] = session["working_memory"][-20:]


def get_relevant_signals(session: dict) -> list[dict]:
    """Return only signals above relevance threshold (curated context)."""
    return [s for s in session["working_memory"] if s["relevance"] >= RELEVANCE_THRESHOLD]


def detect_patterns(session: dict) -> list[str]:
    """Detect repeated topics in query history (implicit interest signals)."""
    queries = [
        s["content"].lower()
        for s in session["working_memory"]
        if s["type"] == "query"
    ]
    words: dict[str, int] = {}
    for q in queries:
        for w in set(q.split()):
            if len(w) > 4:
                words[w] = words.get(w, 0) + 1
    return [w for w, c in words.items() if c >= 2]


# ── Full state snapshot (for dashboard) ───────────────────────

def get_full_state(session_id: str = "default") -> dict:
    """Get complete system state for the dashboard."""
    from stores import skills_registry, decision_tracer, episodic_memory

    session = get_session(session_id)
    active_skills = skills_registry.get_active_rules()
    all_skills = skills_registry.get_all()
    relevant = get_relevant_signals(session)
    active_clusters = [c for c in session["feedback_clusters"] if not c.get("archived")]
    recent_traces = decision_tracer.get_recent(10)
    recent_episodes = episodic_memory.get_recent(10)

    return {
        "turn_count": session["turn_count"],
        "metrics": {
            "feedback": len(session["feedback_records"]),
            "skills": len(active_skills),
            "signals": len(relevant),
            "clusters": len(active_clusters),
            "traces": len(recent_traces),
        },
        "active_skills": active_skills,
        "all_skills": [
            {
                "id": s.id,
                "rule": s.rule,
                "active": s.active,
                "created_at": s.created_at,
            }
            for s in all_skills
        ],
        "working_memory": [
            {
                "type": s["type"],
                "content": s["content"][:60],
                "relevance": round(s["relevance"], 2),
            }
            for s in relevant[-5:]
        ],
        "feedback_records": [
            {"structured": r["structured"][:45], "classification": r["classification"]}
            for r in session["feedback_records"][-5:]
        ],
        "clusters": [
            {
                "category": c["category"],
                "members": len(c["members"]),
                "learning": c.get("learning"),
                "centroid": c["centroid_text"][:50],
            }
            for c in active_clusters[-5:]
        ],
        "eval_history": session["eval_history"][-4:],
        "eval_avg": (
            sum(e.get("score", 0) for e in session["eval_history"])
            / len(session["eval_history"])
        )
        if session["eval_history"]
        else None,
        "traces": [
            {
                "agent": t.agent_name,
                "action": t.action,
                "context": t.context_snapshot,
            }
            for t in recent_traces[-4:]
        ],
        "episodes": [
            {"turn": e.turn_count, "summary": e.summary[:80]}
            for e in recent_episodes[-3:]
        ],
    }
