"""
Response Generation — agent response logic.

Mirrors the multi-agent pattern: orchestrator handles normal conversation,
hands off to feedback_analyst for introspection queries.
"""

import re
import logging

from pipeline.config import call_llm
from pipeline.session import get_relevant_signals, detect_patterns

logger = logging.getLogger("pipeline")


def build_system_prompt(
    active_skills: list[str], relevant_signals: list[dict]
) -> str:
    """Build the agent's system prompt with injected skills + working memory."""
    prompt = (
        "You are a helpful, friendly AI assistant. Answer any question the user asks "
        "— be knowledgeable, concise, and natural like ChatGPT.\n\n"
    )
    if active_skills:
        prompt += "## Mandatory Business Rules (learned from user feedback — ALWAYS follow these):\n"
        for i, s in enumerate(active_skills):
            prompt += f"{i + 1}. {s}\n"
        prompt += "\n"
    if relevant_signals:
        prompt += "## Working Memory (recent context):\n"
        for s in relevant_signals[-5:]:
            prompt += f"- [{s['type']}] {s['content']}\n"
        prompt += "\n"
    return prompt


async def generate_response(
    query: str, session: dict, active_skills: list[str]
) -> dict:
    """Generate agent response — orchestrator for normal chat, feedback_analyst for introspection."""
    lower = query.lower().strip()

    # ── Handoff → feedback_analyst (introspection only) ──────
    if re.search(
        r"show.*learn|what.*learn|system state|feedback summary|active skills|introspect",
        lower,
    ):
        from stores import decision_tracer, episodic_memory

        feedback_records = session["feedback_records"]
        relevant = get_relevant_signals(session)
        clusters = [c for c in session["feedback_clusters"] if not c.get("archived")]
        resp = "Here's what the system has learned:\n\n"
        resp += f"**Active Skills ({len(active_skills)}):**\n"
        if not active_skills:
            resp += "- None yet — give me feedback to teach new behaviors\n"
        else:
            for s in active_skills:
                resp += f'- ✅ "{s}"\n'
        resp += f"\n**Feedback Records:** {len(feedback_records)}\n"
        resp += f"**Working Memory Signals:** {len(relevant)} relevant / {len(session['working_memory'])} total\n"
        resp += f"**Clusters:** {len(clusters)}\n"
        resp += f"**Eval History:** {len(session['eval_history'])} evals\n"
        patterns = detect_patterns(session)
        resp += f"\nRepeated topics: {', '.join(patterns) or 'none detected yet'}"
        return {"text": resp, "agent": "feedback_analyst", "handoff": True}

    if re.search(r"trace|debug|replay|snapshot", lower):
        from stores import decision_tracer

        traces = decision_tracer.get_recent(3)
        resp = "**Recent Decision Traces:**\n\n"
        if not traces:
            resp += "No traces recorded yet.\n"
        else:
            for t in traces:
                ctx = t.context_snapshot
                resp += (
                    f"🔍 {t.agent_name} | {t.action} | "
                    f"skills: {ctx.get('active_skills', 0)} | "
                    f"signals: {ctx.get('working_memory', 0)}\n"
                )
        return {"text": resp, "agent": "feedback_analyst", "handoff": True}

    # ── Recall from learned skills ───────────────────────────
    is_recall = bool(
        re.search(
            r"(?:which|what|do i|what'?s my|how do i|tell me what|remember)"
            r".+(?:like|love|prefer|hate|want|fav|told|said|taught|know about me)",
            lower,
        )
        or re.search(
            r"(?:which|what).+(?:i like|i love|i prefer|i enjoy|my fav)", lower
        )
    )

    if is_recall and active_skills:
        stop_words = {
            "i", "me", "my", "do", "did", "what", "which", "how", "the", "a", "an",
            "is", "are", "you", "about", "that", "this", "to", "of", "in", "for",
            "and", "or", "like", "love", "prefer", "hate", "want", "fav", "tell",
            "know", "remember", "taught", "said", "told",
        }
        query_words = [w for w in lower.split() if len(w) > 2 and w not in stop_words]
        matched = []
        for skill in active_skills:
            hits = [w for w in query_words if w in skill.lower()]
            if hits:
                matched.append({"skill": skill, "score": len(hits)})
        matched.sort(key=lambda m: m["score"], reverse=True)

        if matched:
            resp = "Based on what you've told me:\n\n"
            for m in matched:
                resp += f'• **"{m["skill"]}"**\n'
        else:
            resp = "Here's everything I've learned about you:\n\n"
            for s in active_skills:
                resp += f'• **"{s}"**\n'
        return {"text": resp, "agent": "orchestrator", "handoff": False}

    if is_recall and not active_skills:
        return {
            "text": (
                "You haven't taught me any preferences yet. Try saying things like "
                '"I like X over Y", "Always call me Z", or "Keep answers short" — I\'ll remember!'
            ),
            "agent": "orchestrator",
            "handoff": False,
        }

    # ── Orchestrator: general conversation via Azure OpenAI ──
    relevant = get_relevant_signals(session)
    system_prompt = build_system_prompt(active_skills, relevant)

    session["chat_history"].append({"role": "user", "content": query})
    if len(session["chat_history"]) > 30:
        session["chat_history"] = session["chat_history"][-30:]

    messages = [{"role": "system", "content": system_prompt}] + session["chat_history"]
    llm_response = await call_llm(messages)
    session["chat_history"].append({"role": "assistant", "content": llm_response})

    return {"text": llm_response, "agent": "orchestrator", "handoff": False}
