"""
Pipeline Engine — the main orchestrator.

Runs the full 11-stage adaptive feedback pipeline for one user message:
  sense → working memory → classify → guardrails → skills →
  cluster → episodic → chat → agent → eval → trace
"""

import json
import uuid
import logging
from datetime import datetime, timezone

from stores import (
    feedback_store, skills_registry, episodic_memory, decision_tracer,
    FeedbackRecord,
)
from pipeline.config import CLUSTER_THRESHOLD, CONFIDENCE_THRESHOLD, SIMILARITY_THRESHOLD, DECAY_RATE, RELEVANCE_THRESHOLD
from pipeline.session import (
    get_session, get_full_state,
    add_signal, get_relevant_signals, detect_patterns,
)
from pipeline.classifier import classify_with_llm
from pipeline.guardrails import run_guardrails
from pipeline.clustering import run_clustering, get_unclustered_feedback
from pipeline.eval import eval_response
from pipeline.response import generate_response

logger = logging.getLogger("pipeline")


async def run_pipeline(text: str, session_id: str = "default") -> dict:
    """Run the full adaptive feedback pipeline for one user message.

    Returns:
        {
            "response": {"text": str, "agent": str, "handoff": bool},
            "stages": [{"id": str, "content": str}, ...],
            "system_messages": [str, ...],
            "state": { ... full state snapshot ... },
        }
    """
    session = get_session(session_id)
    session["turn_count"] += 1
    turn = session["turn_count"]
    active_skills = skills_registry.get_active_rules()

    stages: list[dict] = []
    system_messages: list[str] = []

    def stage(stage_id: str, content: str):
        stages.append({"id": stage_id, "content": content})

    # ── Stage 1: Signal Collector (SENSE) ─────────────────────
    add_signal(session, "query", text, 1.0)
    patterns = detect_patterns(session)
    if patterns:
        add_signal(session, "implicit", f"Repeated topics: {', '.join(patterns)}", 0.5)

    stage(
        "sense",
        f'SENSE query: "{text[:80]}"\n'
        f"Signals collected: {len(session['working_memory'])}\n"
        f"Patterns detected: {', '.join(patterns) or 'none'}\n"
        f"⏳ LLM classifier running in parallel...\n"
        f"💾 Storage: In-Memory (per-session, ephemeral)",
    )

    # ── Stage 2: Working Memory ───────────────────────────────
    relevant = get_relevant_signals(session)
    wm_lines = "\n".join(
        f"  [{s['type']}|{round(s['relevance'] * 100)}%] {s['content'][:60]}"
        for s in relevant[-3:]
    )
    stage(
        "working",
        f"Relevant signals: {len(relevant)}/{len(session['working_memory'])} (threshold={RELEVANCE_THRESHOLD})\n"
        f"Context rot prevention: decay_rate={DECAY_RATE}\n"
        f"{wm_lines}\n"
        f"💾 Storage: In-Memory (per-thread, decays each turn)",
    )

    # ── Stage 3: Feedback Classification (LLM-based) ─────────
    llm_result = await classify_with_llm(text)
    is_feedback = llm_result.get("is_feedback", False)

    feedback_issues: list[dict] = []
    if is_feedback:
        if llm_result.get("_multi"):
            feedback_issues = llm_result["issues"]
        else:
            feedback_issues = [llm_result]

    if is_feedback:
        # Store each issue as a separate feedback record
        for issue in feedback_issues:
            record_id = str(uuid.uuid4())
            issue["id"] = record_id
            issue["is_feedback"] = True
            issue["submitted_at"] = datetime.now(timezone.utc).isoformat()
            issue["_source"] = llm_result["_source"]
            session["feedback_records"].append(issue)
            add_signal(
                session, "feedback", f"[{issue['classification']}] {issue['structured']}", 1.0
            )
            # Persist to SQLite
            feedback_store.add(
                FeedbackRecord(
                    id=record_id,
                    raw_feedback=text,
                    classification=issue["classification"],
                    structured_feedback=issue["structured"],
                    business_context_payload=issue.get("business_context"),
                    session_id=session_id,
                    user_id=None,
                    agent_name=None,
                    submitted_at=issue["submitted_at"],
                    remediation_target=issue.get("remediation_target", "pending_clustering"),
                )
            )

        issues_summary = "\n".join(
            f"  Issue {i + 1}: {iss['classification']} → {iss.get('remediation_target', '?')} "
            f'| "{iss.get("structured", "")[:60]}"'
            for i, iss in enumerate(feedback_issues)
        )
        stage(
            "feedback",
            f"⚡ FEEDBACK CLASSIFIER (Azure OpenAI LLM)\n"
            f'Input: "{text[:80]}"\n'
            f"→ is_feedback: true\n"
            f"→ issues detected: {len(feedback_issues)}"
            f"{'  (MULTI-ISSUE SPLIT)' if llm_result.get('_multi') else ''}\n"
            f"{issues_summary}\n"
            f"→ source: {llm_result['_source']}\n"
            f"💾 Stored: {len(feedback_issues)} record(s) → SQLite feedback table",
        )

        # ── Stage 3.5: Guardrails ────────────────────────────
        guarded = run_guardrails(feedback_issues, active_skills)
        blocked = [i for i in guarded if i.get("_guardrail_blocked")]
        overrides = [i for i in guarded if i.get("_guardrail_override")]
        deduped = [i for i in guarded if i.get("_guardrail_deduped")]
        passed = [
            i for i in guarded
            if not i.get("_guardrail_blocked") and not i.get("_guardrail_deduped")
        ]

        guard_lines = []
        for b in blocked:
            guard_lines.append(f"  ❌ BLOCKED: {b.get('_block_reason', '')}")
        for o in overrides:
            guard_lines.append(
                f"  ⚠️ OVERRIDE: contradicts \"{o.get('_override_skill', '')[:40]}\""
            )
        for d in deduped:
            guard_lines.append(
                f"  🔁 DEDUP: {d.get('_dedup_similarity', 0):.0%} similar to existing skill"
            )

        stage(
            "guard",
            f"🛡️ GUARDRAILS ENGINE\n"
            f"Checks: schema ✓ | confidence (≥{CONFIDENCE_THRESHOLD}) | contradiction | dedup\n"
            f"Input: {len(feedback_issues)} issue(s)\n"
            f"Passed: {len(passed)} | Blocked: {len(blocked)} | Deduped: {len(deduped)} | Override: {len(overrides)}\n"
            + "\n".join(guard_lines),
        )

        # Handle overrides — deactivate contradicted skill
        for o in overrides:
            old_skill = o.get("_override_skill")
            if old_skill:
                for sk in skills_registry.get_all():
                    if sk.rule == old_skill and sk.active:
                        skills_registry.deactivate(sk.id)
                        logger.info(f"Deactivated contradicted skill: {old_skill[:50]}")

        # ── Stage 4: Skills Registry ──────────────────────────
        skill_issues = [
            i for i in passed
            if i.get("should_become_skill") and i.get("business_context")
        ]
        non_skill_issues = [i for i in passed if not i.get("should_become_skill")]

        new_skills_text: list[str] = []
        if skill_issues:
            for issue in skill_issues:
                skill_text = issue["structured"] or issue["business_context"]
                skills_registry.register(rule=skill_text, source_feedback_id=issue["id"])
                feedback_store.mark_incorporated(issue["id"])
                new_skills_text.append(skill_text)
                system_messages.append(f'🧬 Skill learned: "{skill_text}"')

            for issue in non_skill_issues:
                system_messages.append(
                    f"📋 Logged ({issue['classification']}): "
                    f"\"{issue['structured'][:80]}\" → {issue.get('remediation_target', '')}"
                )

            active_skills = skills_registry.get_active_rules()  # refresh
            stage(
                "skills",
                f"🧬 {len(new_skills_text)} NEW SKILL(S) REGISTERED\n"
                + "\n".join(f'  Skill: "{s[:80]}"' for s in new_skills_text)
                + f"\nTotal active skills: {len(active_skills)}\n"
                f"→ Will be hot-injected into ALL agents next turn\n"
                f"💾 Stored: SQLite → skills table (persists across restarts)",
            )
        else:
            for issue in feedback_issues:
                if not issue.get("_guardrail_blocked") and not issue.get("_guardrail_deduped"):
                    system_messages.append(
                        f"📋 Logged ({issue['classification']}): "
                        f"\"{issue['structured'][:80]}\" → {issue.get('remediation_target', '')}"
                    )
            stage(
                "skills",
                "No new skills to register\n"
                + "\n".join(
                    f"  {i['classification']} → {i.get('remediation_target', '')} (should_become_skill: false)"
                    for i in feedback_issues
                )
                + f"\nActive skills: {len(active_skills)}",
            )
    else:
        # Not feedback — normal conversation path
        stage(
            "feedback",
            f"FEEDBACK CLASSIFIER (Azure OpenAI LLM)\n"
            f'Input: "{text[:80]}"\n'
            f"→ is_feedback: false (LLM determined: normal conversation)\n"
            f"→ source: {llm_result.get('_source', '')}\n"
            f"Active feedback records: {len(session['feedback_records'])}",
        )
        stage(
            "guard",
            "🛡️ GUARDRAILS ENGINE\nNo feedback to guard — pass-through",
        )
        skills_lines = (
            "\n".join(f'  ✅ "{s[:60]}"' for s in active_skills) or "  (none)"
        )
        stage(
            "skills",
            f"Injecting {len(active_skills)} active skill(s) into context\n{skills_lines}",
        )

    # ── Stage 5: Feedback Clustering ──────────────────────────
    unclustered = get_unclustered_feedback(session)
    if len(unclustered) >= CLUSTER_THRESHOLD:
        cluster_result = await run_clustering(session)
        active_clusters = [c for c in session["feedback_clusters"] if not c.get("archived")]
        with_learnings = [c for c in session["feedback_clusters"] if c.get("learning")]
        stage(
            "cluster",
            f"🔮 CLUSTERING RUN TRIGGERED ({len(unclustered)} unclustered ≥ threshold {CLUSTER_THRESHOLD})\n"
            f"New clusters: {cluster_result['newClusters']} | Updated: {cluster_result['updatedClusters']} | Reactivated: {cluster_result['reactivated']}\n"
            f"Learnings synthesized: {cluster_result['synthesized']}\n"
            f"Active clusters: {len(active_clusters)} | With learnings: {len(with_learnings)}\n"
            + "\n".join(f"  → {op}" for op in cluster_result.get("ops", [])[:5])
            + "\n💾 Stored: clusters + centroids + learnings",
        )
        for cluster in session["feedback_clusters"]:
            if cluster.get("learning") and not cluster.get("_announced"):
                system_messages.append(
                    f"🔮 Cluster learning [{cluster['category']}]: "
                    f"\"{cluster['learning']}\" ({len(cluster['members'])} items)"
                )
                cluster["_announced"] = True
    else:
        active_clusters = [c for c in session["feedback_clusters"] if not c.get("archived")]
        cluster_lines = (
            "\n".join(
                f"  [{c['category']}] {len(c['members'])} members: "
                f"\"{(c.get('learning') or c['centroid_text'])[:50]}\""
                for c in session["feedback_clusters"][-3:]
            )
            if session["feedback_clusters"]
            else "  No clusters yet"
        )
        stage(
            "cluster",
            f"Unclustered feedback: {len(unclustered)}/{CLUSTER_THRESHOLD} (threshold not reached)\n"
            f"Active clusters: {len(active_clusters)}\n"
            f"Similarity threshold: {SIMILARITY_THRESHOLD}\n"
            f"{cluster_lines}",
        )

    # ── Stage 6: Episodic Memory ──────────────────────────────
    if turn % 5 == 0:
        episodic_memory.add(
            session_id=session_id,
            summary=f'Conversation at turn {turn}. Last: "{text[:60]}"',
            turn_count=turn,
        )
        stage(
            "episodic",
            f"📚 EPISODIC SUMMARY CREATED (every 5 turns)\n"
            f"Turn: {turn}\n"
            f'Summary: "Conversation at turn {turn}..."\n'
            f"Total episodes: {len(episodic_memory.get_recent(100))}\n"
            f"💾 Stored: SQLite → episodes table (cross-session)",
        )
    else:
        recent_eps = episodic_memory.get_recent(1)
        latest = f'Latest: "{recent_eps[0].summary[:80]}"' if recent_eps else "No episodes yet"
        stage(
            "episodic",
            f"Episodes: {len(episodic_memory.get_recent(100))} | "
            f"Next summary at turn {((turn // 5) + 1) * 5}\n{latest}",
        )

    # ── Stage 7: Chat History ─────────────────────────────────
    stage(
        "chat",
        f"💾 Storing user message in thread history\n"
        f"Thread messages: {turn} turns\n"
        f"Max per thread: 30 messages (FIFO eviction)\n"
        f"💾 Storage: In-Memory (per-thread, fast access)",
    )

    # ── Stage 8: Agent Response ───────────────────────────────
    response = await generate_response(text, session, active_skills)
    agent_info = (
        f"Orchestrator → handoff → {response['agent']}"
        if response.get("handoff")
        else f"{response['agent']} (direct)"
    )
    stage(
        "agent",
        f"🤖 Agent: {agent_info}\n"
        f"Context injected:\n"
        f"  • {len(active_skills)} business rules (procedural memory) [SQLite]\n"
        f"  • {len(relevant)} working memory signals [In-Memory]\n"
        f"  • {len(episodic_memory.get_recent(100))} episodic summaries [SQLite]\n"
        f"  • Last 30 chat messages [In-Memory]\n"
        f"Response generated.",
    )
    add_signal(
        session, "tool_result", f"{response['agent']}: responded to \"{text[:40]}\"", 0.7
    )

    # ── Stage 9: Response Eval (LLM-as-Judge) ─────────────────
    if active_skills:
        eval_result = await eval_response(text, response["text"], active_skills)
        if eval_result and eval_result.get("score") is not None:
            session["eval_history"].append({"turn": turn, **eval_result})
            avg_score = sum(e.get("score", 0) for e in session["eval_history"]) / len(
                session["eval_history"]
            )
            violations = eval_result.get("violations", [])
            violation_lines = "\n".join(f'  ❌ "{v[:60]}"' for v in violations)
            stage(
                "eval",
                f"📏 RESPONSE EVAL (LLM-as-Judge)\n"
                f"Score: {eval_result['score'] * 100:.0f}% compliance\n"
                f"Compliant: {len(eval_result.get('compliant', []))}/{len(active_skills)} skills\n"
                f"Violations: {len(violations)}\n"
                f"{violation_lines}\n"
                f"Summary: {eval_result.get('summary', 'N/A')}\n"
                f"Trend: avg {avg_score * 100:.0f}% over {len(session['eval_history'])} evals\n"
                f"💾 Stored: evals table (compliance audit trail)",
            )
            if violations:
                system_messages.append(
                    f"📏 Eval: {eval_result['score'] * 100:.0f}% compliance — "
                    f"{len(violations)} violation(s)"
                )
        else:
            stage(
                "eval",
                f"Eval skipped (error or no result)\nActive skills: {len(active_skills)}",
            )
    else:
        stage(
            "eval",
            f"No skills to evaluate against\n"
            f"Evals activate after first skill is learned\n"
            f"Total evals: {len(session['eval_history'])}",
        )

    # ── Stage 10: Decision Trace ──────────────────────────────
    active_clusters = [c for c in session["feedback_clusters"] if not c.get("archived")]
    eval_avg = (
        f"{sum(e.get('score', 0) for e in session['eval_history']) / len(session['eval_history']):.2f}"
        if session["eval_history"]
        else "n/a"
    )
    trace_context = {
        "active_skills": len(active_skills),
        "working_memory": len(get_relevant_signals(session)),
        "feedback_count": len(session["feedback_records"]),
        "clusters": len(active_clusters),
        "episodes": len(episodic_memory.get_recent(100)),
        "eval_avg": eval_avg,
    }
    decision_tracer.log(
        agent_name=response["agent"],
        action="feedback_processed" if is_feedback else "query_answered",
        context_snapshot=trace_context,
        result_summary=f"Turn {turn}: {response['agent']}",
        session_id=session_id,
    )
    stage(
        "trace",
        f"📸 CONTEXT SNAPSHOT LOGGED\n"
        f"Turn: {turn} | Agent: {response['agent']}\n"
        f"Action: {'feedback_processed' if is_feedback else 'query_answered'}\n"
        + json.dumps(trace_context, indent=2)
        + f"\n💾 Stored: SQLite → traces table (full audit trail)",
    )

    return {
        "response": response,
        "stages": stages,
        "system_messages": system_messages,
        "state": get_full_state(session_id),
    }
