"""
Adaptive Feedback Pipeline — Context Providers

ContextProviders that inject layered memory into every agent turn:
- FeedbackContextProvider: injects active business rules from feedback
- EpisodicMemoryProvider: injects recent session summaries
- DecisionTracingProvider: logs context snapshots for every turn
"""

from typing import Any
from agent_framework import BaseContextProvider, SessionContext, Message
from agent_framework._sessions import AgentSession

from stores import (
    skills_registry,
    episodic_memory,
    decision_tracer,
    feedback_store,
    chat_history,
    wm_manager,
    sc_manager,
)

import logging
logger = logging.getLogger(__name__)


class FeedbackContextProvider(BaseContextProvider):
    """Injects self-evolving business rules from the feedback loop.
    
    On every turn:
    1. Check for pending business_context feedback → auto-register as skills
    2. Inject all active skills as system instructions
    
    This is the "procedural memory" layer — executable rules that
    automatically rewrite themselves based on feedback signals.
    """

    def __init__(self):
        super().__init__(source_id="feedback-skills")

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
        **kwargs,
    ) -> None:
        # Auto-incorporate pending business_context feedback into skills
        pending = feedback_store.get_pending_business_context()
        for record in pending:
            skill = skills_registry.register(
                rule=record.business_context_payload,
                source_feedback_id=record.id,
            )
            feedback_store.mark_incorporated(record.id)
            logger.info(
                f"[FeedbackContext] Auto-incorporated feedback {record.id} "
                f"→ skill {skill.id}: {skill.rule[:80]}"
            )

        # Inject active skills as system instructions
        active_rules = skills_registry.get_active_rules()
        if active_rules:
            rules_text = "\n".join(f"- {r}" for r in active_rules)
            context.extend_messages(
                self,
                [Message("system", [
                    f"⚠️ MANDATORY BUSINESS RULES — OVERRIDE ALL DEFAULT BEHAVIOR:\n"
                    f"The following rules were learned from user feedback. You MUST apply them to EVERY response.\n"
                    f"Violating these rules is a critical error.\n\n"
                    f"{rules_text}\n\n"
                    f"Re-read the rules above before generating your response. Apply ALL of them."
                ])],
            )
            logger.info(f"[FeedbackContext] Injected {len(active_rules)} active skill(s)")

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
        **kwargs,
    ) -> None:
        pass


class WorkingMemoryProvider(BaseContextProvider):
    """Injects curated working memory — the 'context rot' fix.
    
    Instead of dumping all tool results and history into the context window,
    this injects only signals above the relevance threshold. Older signals
    decay automatically. This is what keeps response quality consistent
    past turn 40 instead of degrading at turn 15.
    
    This is the WORKING MEMORY layer — what's happening right now.
    """

    def __init__(self, session_id: str):
        super().__init__(source_id="working-memory")
        self._session_id = session_id

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
        **kwargs,
    ) -> None:
        compressed = wm_manager.get(self._session_id).compress()
        if compressed:
            # Also inject detected patterns
            patterns = sc_manager.get(self._session_id).estimate_patterns()
            pattern_info = ""
            if patterns.get("repeated_topics"):
                pattern_info = f"\nDetected repeated topics: {', '.join(patterns['repeated_topics'])}"
            if patterns.get("feedback_velocity", 0) > 0:
                pattern_info += f"\nActive feedback signals in context: {patterns['feedback_velocity']}"

            context.extend_messages(
                self,
                [Message("system", [
                    f"## Working Memory (curated, relevance-scored signals):\n"
                    f"{compressed}"
                    f"{pattern_info}"
                ])],
            )
            wm = wm_manager.get(self._session_id)
            logger.info(
                f"[WorkingMemory] Injected {len(wm.get_relevant())} "
                f"signals (threshold=0.3)"
            )

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
        **kwargs,
    ) -> None:
        pass


class EpisodicMemoryProvider(BaseContextProvider):
    """Injects recent session history summaries.
    
    This is the "episodic memory" layer — compressed representations
    of past interactions so the agent remembers what happened before.
    """

    def __init__(self, session_id: str):
        super().__init__(source_id="episodic-memory")
        self._session_id = session_id

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
        **kwargs,
    ) -> None:
        episodes = episodic_memory.get_for_session(self._session_id)
        if not episodes:
            # Also check recent cross-session episodes
            episodes = episodic_memory.get_recent(5)

        if episodes:
            summaries = "\n".join(
                f"- [Turn {e.turn_count}] {e.summary}" for e in episodes[-5:]
            )
            context.extend_messages(
                self,
                [Message("system", [
                    f"## Session History (episodic memory):\n{summaries}"
                ])],
            )
            logger.info(f"[EpisodicMemory] Injected {len(episodes)} episode(s)")

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
        **kwargs,
    ) -> None:
        pass


class DecisionTracingProvider(BaseContextProvider):
    """Logs a context snapshot on every turn for debugging/replay.
    
    After each agent turn, records what context was available and
    what the agent decided — making failures detectable and recoverable.
    """

    def __init__(self, agent_name: str, session_id: str):
        super().__init__(source_id="decision-tracing")
        self._agent_name = agent_name
        self._session_id = session_id
        self._pre_run_snapshot: dict = {}

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
        **kwargs,
    ) -> None:
        # Capture context snapshot before the agent runs
        self._pre_run_snapshot = {
            "active_skills": skills_registry.get_active_rules(),
            "feedback_count": feedback_store.count(),
            "feedback_summary": feedback_store.summary(),
            "episode_count": len(episodic_memory.get_for_session(self._session_id)),
            "state_keys": list(state.keys()),
        }

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
        **kwargs,
    ) -> None:
        # Log the decision trace
        decision_tracer.log(
            agent_name=self._agent_name,
            action="turn_complete",
            context_snapshot=self._pre_run_snapshot,
            result_summary=f"Turn completed with {len(self._pre_run_snapshot.get('active_skills', []))} active skills",
            session_id=self._session_id,
        )
        logger.info(f"[DecisionTracer] Logged trace for {self._agent_name} in {self._session_id}")


class ChatHistoryProvider(BaseContextProvider):
    """Injects recent chat history for the current thread.

    Gives the agent conversation continuity — it can see what was said before
    in this thread without relying on the framework's built-in history.
    """

    def __init__(self, thread_id: str, max_messages: int = 15):
        super().__init__(source_id="chat-history")
        self._thread_id = thread_id
        self._max_messages = max_messages

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
        **kwargs,
    ) -> None:
        recent = chat_history.get_recent(self._thread_id, self._max_messages)
        if recent:
            history_lines = []
            for msg in recent:
                prefix = "User" if msg.role == "user" else "Assistant"
                history_lines.append(f"{prefix}: {msg.content[:500]}")
            history_text = "\n".join(history_lines)
            context.extend_messages(
                self,
                [Message("system", [
                    f"## Conversation History (this thread):\n{history_text}"
                ])],
            )
            logger.info(f"[ChatHistory] Injected {len(recent)} messages for {self._thread_id}")

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
        **kwargs,
    ) -> None:
        pass
