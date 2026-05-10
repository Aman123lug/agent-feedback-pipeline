"""
Adaptive Feedback Pipeline — Stores & Memory Layer

DURABLE (SQLite — survives restarts):
- FeedbackStore: classified feedback records
- SkillsRegistry: self-evolving business rules extracted from feedback
- EpisodicMemory: session history summaries
- DecisionTracer: logs every agent action with context snapshot

EPHEMERAL (In-memory — fast, TTL/decay-based):
- WorkingMemory: per-session curated context window (context rot fix)
- ChatHistory: per-thread message history
- SignalCollector: sense → estimate → act pipeline
"""

import uuid
import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass
from collections import deque

import logging
logger = logging.getLogger(__name__)

# ─── SQLite Setup ─────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "adaptive_data.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Thread-local SQLite connection (SQLite doesn't allow cross-thread sharing)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")  # better concurrent reads
    return _local.conn


def _init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            raw_feedback TEXT NOT NULL,
            classification TEXT NOT NULL,
            structured_feedback TEXT NOT NULL,
            business_context_payload TEXT,
            session_id TEXT NOT NULL,
            user_id TEXT,
            agent_name TEXT,
            incorporation_state TEXT NOT NULL DEFAULT 'pending',
            remediation_target TEXT NOT NULL DEFAULT 'pending_clustering',
            submitted_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skills (
            id TEXT PRIMARY KEY,
            rule TEXT NOT NULL,
            source_feedback_id TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            turn_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS traces (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            action TEXT NOT NULL,
            context_snapshot TEXT NOT NULL,
            result_summary TEXT NOT NULL,
            session_id TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_classification ON feedback(classification);
        CREATE INDEX IF NOT EXISTS idx_skills_active ON skills(active);
        CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
        CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
    """)
    conn.commit()
    logger.info(f"✅ SQLite DB initialized at {DB_PATH}")


# Initialize on import
_init_db()


# ─── Feedback Store (SQLite) ─────────────────────────────────────────────────

@dataclass
class FeedbackRecord:
    id: str
    raw_feedback: str
    classification: str  # user_preference | response_format | business_context | bug | reprompting | other
    structured_feedback: str
    business_context_payload: Optional[str]
    session_id: str
    user_id: Optional[str]
    agent_name: Optional[str]
    submitted_at: str
    incorporation_state: str = "pending"  # pending | incorporated | dismissed
    remediation_target: str = "pending_clustering"  # user_profile | system_prompt | knowledge_base | codebase | prompt_engineering | pending_clustering


class FeedbackStore:
    """SQLite-backed feedback store — durable across restarts."""

    def add(self, record: FeedbackRecord) -> None:
        conn = _get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO feedback
               (id, raw_feedback, classification, structured_feedback,
                business_context_payload, session_id, user_id, agent_name,
                incorporation_state, remediation_target, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.id, record.raw_feedback, record.classification,
             record.structured_feedback, record.business_context_payload,
             record.session_id, record.user_id, record.agent_name,
             record.incorporation_state, record.remediation_target, record.submitted_at),
        )
        conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> FeedbackRecord:
        return FeedbackRecord(**dict(row))

    def get_all(self, session_id: Optional[str] = None) -> list[FeedbackRecord]:
        conn = _get_conn()
        if session_id:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE session_id=? ORDER BY submitted_at DESC", (session_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM feedback ORDER BY submitted_at DESC").fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_by_classification(self, classification: str) -> list[FeedbackRecord]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM feedback WHERE classification=? ORDER BY submitted_at DESC",
            (classification,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_pending_business_context(self) -> list[FeedbackRecord]:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT * FROM feedback
               WHERE classification='business_context'
               AND incorporation_state='pending'
               AND business_context_payload IS NOT NULL""",
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_pending_by_remediation(self, target: str) -> list[FeedbackRecord]:
        """Get pending feedback by remediation target (action-oriented query)."""
        conn = _get_conn()
        rows = conn.execute(
            """SELECT * FROM feedback
               WHERE remediation_target=?
               AND incorporation_state='pending'""",
            (target,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_actionable_pending(self) -> list[FeedbackRecord]:
        """Get all pending feedback that can auto-become skills (user_profile, system_prompt, knowledge_base)."""
        conn = _get_conn()
        rows = conn.execute(
            """SELECT * FROM feedback
               WHERE remediation_target IN ('user_profile', 'system_prompt', 'knowledge_base')
               AND incorporation_state='pending'""",
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def mark_incorporated(self, feedback_id: str) -> None:
        conn = _get_conn()
        conn.execute(
            "UPDATE feedback SET incorporation_state='incorporated' WHERE id=?",
            (feedback_id,),
        )
        conn.commit()

    def count(self) -> int:
        conn = _get_conn()
        return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

    def summary(self) -> dict:
        conn = _get_conn()
        total = self.count()
        rows = conn.execute(
            "SELECT classification, COUNT(*) as cnt FROM feedback GROUP BY classification"
        ).fetchall()
        by_class = {r["classification"]: r["cnt"] for r in rows}
        remed_rows = conn.execute(
            "SELECT remediation_target, COUNT(*) as cnt FROM feedback GROUP BY remediation_target"
        ).fetchall()
        by_remediation = {r["remediation_target"]: r["cnt"] for r in remed_rows}
        pending = len(self.get_pending_business_context())
        actionable = len(self.get_actionable_pending())
        return {
            "total": total,
            "by_classification": by_class,
            "by_remediation_target": by_remediation,
            "pending_business_context": pending,
            "actionable_pending": actionable,
        }


# ─── Skills Registry (SQLite) ────────────────────────────────────────────────

@dataclass
class Skill:
    id: str
    rule: str
    source_feedback_id: str
    created_at: str
    active: bool = True


class SkillsRegistry:
    """SQLite-backed skills registry — skills survive restarts.

    Self-evolving business rules extracted from feedback.
    These get hot-injected into agent context on every turn.
    """

    def register(self, rule: str, source_feedback_id: str) -> Skill:
        skill = Skill(
            id=str(uuid.uuid4()),
            rule=rule,
            source_feedback_id=source_feedback_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        conn = _get_conn()
        conn.execute(
            "INSERT INTO skills (id, rule, source_feedback_id, active, created_at) VALUES (?, ?, ?, ?, ?)",
            (skill.id, skill.rule, skill.source_feedback_id, 1, skill.created_at),
        )
        conn.commit()
        logger.info(f"🧠 Skill registered (SQLite): {skill.rule[:80]}")
        return skill

    def get_active_rules(self) -> list[str]:
        conn = _get_conn()
        rows = conn.execute("SELECT rule FROM skills WHERE active=1").fetchall()
        return [r["rule"] for r in rows]

    def deactivate(self, skill_id: str) -> None:
        conn = _get_conn()
        conn.execute("UPDATE skills SET active=0 WHERE id=?", (skill_id,))
        conn.commit()

    def get_all(self) -> list[Skill]:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM skills ORDER BY created_at DESC").fetchall()
        return [Skill(id=r["id"], rule=r["rule"], source_feedback_id=r["source_feedback_id"],
                       created_at=r["created_at"], active=bool(r["active"])) for r in rows]


# ─── Episodic Memory (SQLite) ────────────────────────────────────────────────

@dataclass
class Episode:
    id: str
    session_id: str
    summary: str
    turn_count: int
    created_at: str


class EpisodicMemory:
    """SQLite-backed episodic memory — session summaries persist across restarts."""

    def add(self, session_id: str, summary: str, turn_count: int) -> Episode:
        episode = Episode(
            id=str(uuid.uuid4()),
            session_id=session_id,
            summary=summary,
            turn_count=turn_count,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        conn = _get_conn()
        conn.execute(
            "INSERT INTO episodes (id, session_id, summary, turn_count, created_at) VALUES (?, ?, ?, ?, ?)",
            (episode.id, episode.session_id, episode.summary, episode.turn_count, episode.created_at),
        )
        conn.commit()
        return episode

    def get_recent(self, n: int = 10) -> list[Episode]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (n,)
        ).fetchall()
        return [Episode(**dict(r)) for r in reversed(rows)]

    def get_for_session(self, session_id: str) -> list[Episode]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM episodes WHERE session_id=? ORDER BY created_at ASC", (session_id,)
        ).fetchall()
        return [Episode(**dict(r)) for r in rows]


# ─── Decision Tracer (SQLite) ────────────────────────────────────────────────

@dataclass
class TraceEntry:
    id: str
    timestamp: str
    agent_name: str
    action: str
    context_snapshot: dict
    result_summary: str
    session_id: str


class DecisionTracer:
    """SQLite-backed decision tracer — full audit trail across restarts.

    Logs every agent action with a snapshot of available context.
    Makes failures detectable and debuggable — you can replay exactly
    what the agent saw and understand why it went wrong.
    """

    def log(
        self,
        agent_name: str,
        action: str,
        context_snapshot: dict,
        result_summary: str,
        session_id: str,
    ) -> TraceEntry:
        entry = TraceEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=agent_name,
            action=action,
            context_snapshot=context_snapshot,
            result_summary=result_summary,
            session_id=session_id,
        )
        conn = _get_conn()
        conn.execute(
            """INSERT INTO traces (id, timestamp, agent_name, action, context_snapshot, result_summary, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (entry.id, entry.timestamp, entry.agent_name, entry.action,
             json.dumps(entry.context_snapshot), entry.result_summary, entry.session_id),
        )
        conn.commit()
        return entry

    def get_for_session(self, session_id: str) -> list[TraceEntry]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM traces WHERE session_id=? ORDER BY timestamp ASC", (session_id,)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_recent(self, n: int = 20) -> list[TraceEntry]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM traces ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_entry(r) for r in reversed(rows)]

    def _row_to_entry(self, row: sqlite3.Row) -> TraceEntry:
        d = dict(row)
        d["context_snapshot"] = json.loads(d["context_snapshot"])
        return TraceEntry(**d)


# ═════════════════════════════════════════════════════════════════════════════
# EPHEMERAL (In-Memory) — fast, per-session, TTL/decay-based
# ═════════════════════════════════════════════════════════════════════════════


# ─── Chat History (In-Memory, per-thread) ─────────────────────────────────────

@dataclass
class ChatMessage:
    role: str          # "user" | "assistant"
    content: str
    timestamp: str
    thread_id: str


class ChatHistoryStore:
    """In-memory chat history — per-thread message storage.

    Provides conversation continuity within a session. Ephemeral by design:
    working conversations don't need to survive restarts for a POC.
    """

    def __init__(self, max_messages_per_thread: int = 100):
        self._history: dict[str, list[ChatMessage]] = {}
        self._max = max_messages_per_thread

    def add_message(self, thread_id: str, role: str, content: str) -> ChatMessage:
        msg = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            thread_id=thread_id,
        )
        if thread_id not in self._history:
            self._history[thread_id] = []
        self._history[thread_id].append(msg)
        # Evict oldest if over limit
        if len(self._history[thread_id]) > self._max:
            self._history[thread_id] = self._history[thread_id][-self._max:]
        return msg

    def get_history(self, thread_id: str) -> list[ChatMessage]:
        return self._history.get(thread_id, [])

    def get_recent(self, thread_id: str, n: int = 10) -> list[ChatMessage]:
        return self._history.get(thread_id, [])[-n:]

    def clear(self, thread_id: str) -> None:
        self._history.pop(thread_id, None)

    def all_threads(self) -> list[str]:
        return list(self._history.keys())


# ─── Working Memory (In-Memory, per-session) ─────────────────────────────────

@dataclass
class Signal:
    """A single signal captured from the user or environment."""
    type: str          # "query" | "feedback" | "tool_result" | "implicit"
    content: str
    timestamp: str
    relevance_score: float = 1.0  # 0.0–1.0, decays over time


class WorkingMemory:
    """Current-turn context window — the 'RAM' of the agent.

    Instead of dumping everything into the prompt (context rot at turn 15+),
    this curates, scores, and compresses signals so the model gets only
    the most relevant context. Stale signals decay and get evicted.

    This is the fix for: "context bloat causing the agent to silently degrade
    over long sessions" — the failure case described in the interview.
    """

    def __init__(self, max_signals: int = 20, decay_rate: float = 0.85):
        self._signals: deque[Signal] = deque(maxlen=max_signals)
        self._decay_rate = decay_rate

    def ingest(self, signal_type: str, content: str, relevance: float = 1.0) -> None:
        """Add a new signal and decay all existing signals."""
        for s in self._signals:
            s.relevance_score *= self._decay_rate

        self._signals.append(Signal(
            type=signal_type,
            content=content[:500],
            timestamp=datetime.now(timezone.utc).isoformat(),
            relevance_score=relevance,
        ))

    def get_relevant(self, threshold: float = 0.3) -> list[Signal]:
        """Return only signals above relevance threshold (curated context)."""
        return [s for s in self._signals if s.relevance_score >= threshold]

    def compress(self) -> str:
        """Compress working memory into a summary string for the context window."""
        relevant = self.get_relevant()
        if not relevant:
            return ""
        lines = []
        for s in relevant:
            score_bar = "█" * int(s.relevance_score * 5)
            lines.append(f"[{s.type}|{score_bar}] {s.content[:200]}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._signals.clear()


class SessionWorkingMemoryManager:
    """Returns a per-thread WorkingMemory instance — thread-level isolation."""

    def __init__(self):
        self._sessions: dict[str, WorkingMemory] = {}

    def get(self, session_id: str) -> WorkingMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = WorkingMemory()
        return self._sessions[session_id]

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# ─── Signal Collector (sense → estimate → act pipeline, per-session) ─────────

class SignalCollector:
    """Continuously collects signals from user interactions.

    The sense → estimate → act pipeline:
    1. SENSE: Collect raw signals (queries, feedback, tool results, timing)
    2. ESTIMATE: Score relevance, detect patterns
    3. ACT: Update working memory with only high-signal context
    """

    def __init__(self, working_memory: WorkingMemory):
        self._wm = working_memory
        self._query_history: list[str] = []
        self._interaction_count: int = 0

    def sense_query(self, query: str, user_id: Optional[str] = None) -> None:
        self._query_history.append(query)
        self._interaction_count += 1
        self._wm.ingest("query", query, relevance=1.0)

    def sense_feedback(self, feedback: str, classification: str) -> None:
        self._wm.ingest("feedback", f"[{classification}] {feedback}", relevance=1.0)

    def sense_tool_result(self, tool_name: str, result_summary: str) -> None:
        self._wm.ingest("tool_result", f"{tool_name}: {result_summary[:300]}", relevance=0.7)

    def sense_implicit(self, signal: str) -> None:
        self._wm.ingest("implicit", signal, relevance=0.5)

    def estimate_patterns(self) -> dict:
        return {
            "interaction_count": self._interaction_count,
            "repeated_topics": self._detect_repeated_topics(),
            "feedback_velocity": len([
                s for s in self._wm.get_relevant() if s.type == "feedback"
            ]),
        }

    def _detect_repeated_topics(self) -> list[str]:
        if len(self._query_history) < 3:
            return []
        recent = self._query_history[-10:]
        word_counts: dict[str, int] = {}
        for q in recent:
            for word in set(q.lower().split()):
                if len(word) > 4:
                    word_counts[word] = word_counts.get(word, 0) + 1
        return [w for w, c in word_counts.items() if c >= 3]


class SessionSignalCollectorManager:
    """Returns a per-thread SignalCollector — thread-level isolation."""

    def __init__(self, wm_manager: SessionWorkingMemoryManager):
        self._wm_manager = wm_manager
        self._sessions: dict[str, SignalCollector] = {}

    def get(self, session_id: str) -> SignalCollector:
        if session_id not in self._sessions:
            wm = self._wm_manager.get(session_id)
            self._sessions[session_id] = SignalCollector(wm)
        return self._sessions[session_id]

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# ─── Singleton instances ─────────────────────────────────────────────────────

# DURABLE (SQLite)
feedback_store = FeedbackStore()
skills_registry = SkillsRegistry()
episodic_memory = EpisodicMemory()
decision_tracer = DecisionTracer()

# EPHEMERAL (In-memory, per-session)
chat_history = ChatHistoryStore()
wm_manager = SessionWorkingMemoryManager()
sc_manager = SessionSignalCollectorManager(wm_manager)

# Legacy aliases for backward compat (global instances — will be removed)
working_memory = WorkingMemory()
signal_collector = SignalCollector(working_memory)
