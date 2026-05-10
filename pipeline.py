"""
Adaptive Feedback Pipeline — Standalone Python Backend

All pipeline logic runs here:
- LLM-based feedback classification (Azure OpenAI)
- Guardrails engine (confidence gating, contradiction, schema validation, dedup)
- Skills registry (SQLite, self-evolving business rules)
- Feedback clustering (token-overlap similarity, LLM synthesis)
- Response eval (LLM-as-judge, compliance scoring)
- Working memory (decay-based, relevance-scored)
- Episodic memory (session summaries, SQLite)
- Decision tracing (full audit trail, SQLite)

Run:  python pipeline.py
UI:   http://localhost:8000
API:  POST /api/chat, GET /api/state
"""

import os
import json
import uuid
import re
import logging
from datetime import datetime, timezone
from typing import Optional

from openai import AsyncAzureOpenAI
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from stores import (
    feedback_store, skills_registry, episodic_memory, decision_tracer,
    FeedbackRecord,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")


# ══════════════════════════════════════════════════════════════════
# Azure OpenAI Config
# ══════════════════════════════════════════════════════════════════

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

llm_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)


async def call_llm(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Call Azure OpenAI chat completions API."""
    try:
        response = await llm_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"LLM error: {e}"


# ══════════════════════════════════════════════════════════════════
# Pipeline Config
# ══════════════════════════════════════════════════════════════════

CLUSTER_THRESHOLD = 2       # auto-cluster after N unclustered feedback
SIMILARITY_THRESHOLD = 0.55 # text similarity cutoff for clustering
CONFIDENCE_THRESHOLD = 0.7  # guardrail: minimum confidence to accept
DECAY_RATE = 0.85           # working memory decay per turn
RELEVANCE_THRESHOLD = 0.3   # minimum relevance to stay in working memory


# ══════════════════════════════════════════════════════════════════
# Session State (per-session, in-memory — ephemeral by design)
# ══════════════════════════════════════════════════════════════════

_sessions: dict[str, dict] = {}


def get_session(session_id: str = "default") -> dict:
    """Get or create a session. Ephemeral state lives here; durable state in SQLite."""
    if session_id not in _sessions:
        _sessions[session_id] = {
            "turn_count": 0,
            "working_memory": [],       # list[dict] — {type, content, relevance, timestamp}
            "chat_history": [],         # list[dict] — {role, content} for LLM context
            "eval_history": [],         # list[dict] — {turn, score, violations, compliant, summary}
            "feedback_clusters": [],    # list[dict] — {id, category, centroid_text, members, learning, ...}
            "feedback_records": [],     # in-memory mirror for clustering (_clustered flag)
        }
    return _sessions[session_id]


# ══════════════════════════════════════════════════════════════════
# Working Memory — decay-based relevance scoring
# Context rot prevention: stale signals decay, only relevant ones
# get injected into agent context. Keeps quality consistent past
# turn 40 instead of degrading at turn 15.
# ══════════════════════════════════════════════════════════════════

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
    # Cap at 20 signals (FIFO eviction)
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


# ══════════════════════════════════════════════════════════════════
# LLM-based Feedback Classification
# Exact same prompt and logic as the HTML dashboard.
# Pure LLM — no regex fallback.
# ══════════════════════════════════════════════════════════════════

CLASSIFICATION_PROMPT = """You are an intent classifier inside an adaptive AI assistant's feedback pipeline.

Your ONLY job: decide if the user's message is **actionable feedback** (a rule, preference, correction, or complaint the system should learn) — or just normal conversation.

## Output schema (JSON)
If the message contains MULTIPLE distinct feedback items, return an array. If single, return one object.
Single: { "is_feedback": bool, "classification": string, "structured": string, "should_become_skill": bool, "remediation_target": string, "confidence": float }
Multi:  { "is_feedback": true, "issues": [ { "classification": string, "structured": string, "should_become_skill": bool, "remediation_target": string, "confidence": float }, ... ] }

Use the multi-issue format ONLY when the message clearly contains 2+ SEPARATE actionable instructions (e.g. "call me Aman and always use bullet points" = 2 issues). Do NOT split a single idea into parts.

remediation_target values: "user_profile" | "system_prompt" | "knowledge_base" | "codebase" | "prompt_engineering" | "pending_clustering"
confidence: a float 0.0–1.0 indicating how confident you are in this classification. Use 0.9+ for clear-cut cases, 0.5–0.8 for ambiguous, below 0.5 for guesses.

## Classification categories (only used when is_feedback=true)
### Action-oriented taxonomy: ask "Where does the fix live?"

| Category | Where the fix lives | Remediation action | Examples |
|---|---|---|---|
| user_preference | **User profile / persona memory** — the system must remember WHO the user is | Store in long-term user profile; inject into every future prompt as identity context | "call me Aman", "I like cats over dogs", "speak in Arabic", "I'm a backend engineer" |
| response_format | **System prompt / output template** — HOW the assistant structures replies | Update output formatting instructions in system prompt | "use bullet points", "keep it short", "give me tables", "number your steps" |
| business_context | **Knowledge base / business rules** — domain constraints the assistant must follow | Extract as a durable business rule; inject into agent context as a hard constraint | "always cite sources", "never mention competitors", "prices must include tax", "greet users by name first" |
| bug | **Codebase / model behavior** — something is factually wrong or broken | Log for triage; flag for human review; do NOT auto-create a skill | "that answer was wrong", "you hallucinated a link", "the calculation is off" |
| reprompt | **Prompt engineering / system prompt tuning** — the assistant's APPROACH is wrong, not the facts | Adjust prompt strategy, reasoning chain, or tool selection | "rephrase that more simply", "don't overthink it", "try a different approach", "be more creative" |
| other | **Unclear remediation path** — feedback that doesn't map to a clear fix location | Store for pattern analysis; may become actionable after clustering with similar feedback | anything else that's clearly feedback but ambiguous |

## Decision rules (follow strictly, in order)
1. **Questions are NEVER feedback.** If the message ends with "?" OR starts with a question word (what, which, who, where, when, why, how, is, are, can, do, does, will, could, would, should, tell me, show me) → is_feedback=false, stop.
2. **Greetings & casual chat are NOT feedback.** "hi", "hello", "thanks", "ok", "cool" → is_feedback=false.
3. **Factual statements without instruction are NOT feedback.** "The sky is blue", "I went to Paris" → is_feedback=false. The user must be telling the assistant to DO or BE something.
4. **Vague single preferences without contrast are NOT feedback.** "I like flowers" → too vague, is_feedback=false. But "I like roses over tulips" → is_feedback=true (has a comparison/instruction).
5. **If the message instructs the assistant to change behavior, remember something, or follow a rule → is_feedback=true.**
6. **Disambiguation rule (from gagent-core):** When classifying, ask "where does the fix live?" — user profile→user_preference, output template→response_format, knowledge base→business_context, codebase→bug, prompt strategy→reprompt.
7. **should_become_skill=true** only for user_preference, response_format, business_context (these have clear remediation paths that can be auto-applied).
8. **should_become_skill=false** for bug (needs human triage), reprompt (needs prompt engineering), other (ambiguous).
9. **structured**: Rewrite the feedback as a clean imperative rule. E.g. "always call me Aman paglu" → "Address the user as Aman paglu".
10. **remediation_target**: One of "user_profile", "system_prompt", "knowledge_base", "codebase", "prompt_engineering", "pending_clustering" — tells the system WHERE to apply the fix.

User message: "{USER_MESSAGE}" """

VALID_CATEGORIES = [
    "user_preference", "response_format", "business_context",
    "bug", "reprompting", "other",
]

REMEDIATION_MAP = {
    "user_preference": "user_profile",
    "response_format": "system_prompt",
    "business_context": "knowledge_base",
    "bug": "codebase",
    "reprompting": "prompt_engineering",
    "other": "pending_clustering",
}


def normalize_issue(issue: dict, raw_text: str) -> dict:
    """Normalize a classified issue — enforce valid categories, set remediation target."""
    cls = issue.get("classification", "other")
    if cls not in VALID_CATEGORIES:
        cls = "other"
    should_become_skill = cls in ("user_preference", "response_format", "business_context")
    structured = issue.get("structured", raw_text[:200])
    return {
        "classification": cls,
        "structured": structured,
        "should_become_skill": should_become_skill,
        "remediation_target": REMEDIATION_MAP.get(cls, "pending_clustering"),
        "business_context": structured if should_become_skill else None,
        "confidence": issue.get("confidence", 0.8),
    }


async def classify_with_llm(text: str) -> dict:
    """Classify user message as feedback or normal conversation using Azure OpenAI."""
    prompt = CLASSIFICATION_PROMPT.replace("{USER_MESSAGE}", text[:300])
    try:
        raw = await call_llm(
            [
                {
                    "role": "system",
                    "content": "You are a JSON-only classifier. Respond with valid JSON only, no markdown fences, no explanation.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        # Strip markdown fences if model wraps output
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```", "", cleaned).strip()
        parsed = json.loads(cleaned)

        # Multi-issue path
        if (
            isinstance(parsed, dict)
            and parsed.get("is_feedback")
            and isinstance(parsed.get("issues"), list)
            and len(parsed["issues"]) > 1
        ):
            issues = [normalize_issue(i, text) for i in parsed["issues"]]
            return {"is_feedback": True, "_multi": True, "issues": issues, "_source": "llm"}

        # Single-issue path
        if isinstance(parsed, dict) and isinstance(parsed.get("is_feedback"), bool):
            if not parsed["is_feedback"]:
                return {"is_feedback": False, "_source": "llm"}
            normalized = normalize_issue(parsed, text)
            return {"is_feedback": True, "_multi": False, **normalized, "_source": "llm"}

        logger.warning(f"LLM classifier returned invalid schema: {parsed}")
        return {"is_feedback": False, "_source": "llm_invalid"}

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {"is_feedback": False, "_source": "llm_error"}


# ══════════════════════════════════════════════════════════════════
# Guardrails Engine
# Eugene Yan pattern #5: "Guardrails help prevent the model from
# generating undesirable output."
# 4 checks: schema validation, confidence gating, contradiction
# detection, duplicate detection.
# ══════════════════════════════════════════════════════════════════

def run_guardrails(issues: list[dict], active_skills: list[str]) -> list[dict]:
    """Run guardrail checks on classified feedback issues.

    Returns annotated issues with guardrail flags:
    - _guardrail_blocked: rejected (low confidence or invalid schema)
    - _guardrail_override: contradicts existing skill (old skill will be deactivated)
    - _guardrail_deduped: too similar to existing skill (skip)
    """
    results = []
    for issue in issues:
        # 1. Schema validation
        if not issue.get("structured") or not issue.get("classification"):
            logger.warning("Guardrail BLOCKED: invalid schema")
            continue

        # 2. Confidence gating
        confidence = issue.get("confidence", 0.8)
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(f"Guardrail BLOCKED: low confidence {confidence:.2f}")
            issue["_guardrail_blocked"] = True
            issue["_block_reason"] = f"confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD}"
            results.append(issue)
            continue

        # 3. Contradiction detection
        structured_lower = issue["structured"].lower()
        contradiction_found = False
        for skill in active_skills:
            skill_lower = skill.lower()
            if (
                "never" in structured_lower
                and any(w in skill_lower for w in structured_lower.split() if len(w) > 3)
            ) or ("always" in skill_lower and "never" in structured_lower):
                skill_words = set(skill_lower.split())
                issue_words = set(structured_lower.split())
                stop = {"always", "never", "the", "a", "in", "to", "and", "or", "is", "be"}
                overlap = (skill_words & issue_words) - stop
                if len(overlap) >= 2:
                    issue["_guardrail_override"] = True
                    issue["_override_skill"] = skill
                    contradiction_found = True
                    logger.info(f"Guardrail OVERRIDE: contradicts skill '{skill[:50]}'")
                    break

        # 4. Duplicate detection
        if not contradiction_found and issue.get("should_become_skill"):
            for skill in active_skills:
                sim = text_similarity(issue["structured"], skill)
                if sim >= 0.7:
                    issue["_guardrail_deduped"] = True
                    issue["_dedup_skill"] = skill
                    issue["_dedup_similarity"] = sim
                    logger.info(f"Guardrail DEDUP: {sim:.2f} similar to '{skill[:50]}'")
                    break

        results.append(issue)
    return results


# ══════════════════════════════════════════════════════════════════
# Feedback Clustering Engine
# Simplified version of gagent-core's embedding-based clustering.
# Production uses Azure text-embedding-3-small + cosine similarity.
# POC uses token-overlap similarity as a proxy (same algorithm,
# simpler math).
# ══════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list[str]:
    """Tokenize text for similarity comparison."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if len(w) > 2]


def text_similarity(a: str, b: str) -> float:
    """Jaccard-like token overlap — proxy for cosine similarity of embeddings."""
    tok_a = set(tokenize(a))
    tok_b = set(tokenize(b))
    if not tok_a or not tok_b:
        return 0.0
    intersection = len(tok_a & tok_b)
    return intersection / min(len(tok_a), len(tok_b))


def get_unclustered_feedback(session: dict) -> list[dict]:
    """Get feedback records not yet assigned to a cluster."""
    return [
        r for r in session["feedback_records"]
        if r.get("is_feedback") and not r.get("_clustered")
    ]


def find_best_cluster(
    feedback: dict, clusters: list[dict]
) -> tuple[Optional[dict], float]:
    """Find the best matching cluster for a feedback item.
    gagent-core pattern: archived clusters can REACTIVATE if new feedback matches.
    """
    best = None
    best_score = 0.0
    for cluster in clusters:
        if cluster.get("archived"):
            score = text_similarity(feedback["structured"], cluster["centroid_text"])
            if score >= SIMILARITY_THRESHOLD and score > best_score:
                best, best_score = cluster, score
        elif cluster["category"] == feedback["classification"]:
            score = text_similarity(feedback["structured"], cluster["centroid_text"])
            if score >= SIMILARITY_THRESHOLD and score > best_score:
                best, best_score = cluster, score
    return best, best_score


def update_centroid(cluster: dict, new_text: str):
    """Update cluster centroid. In production this would be average embedding vector."""
    if len(new_text) > len(cluster["centroid_text"]):
        cluster["centroid_text"] = new_text
    cluster["updated_at"] = datetime.now(timezone.utc).isoformat()


async def synthesize_learning(cluster: dict) -> str:
    """LLM synthesizes a 'learning' from cluster members — same as gagent-core."""
    member_texts = "\n".join(f"- {m['structured']}" for m in cluster["members"])
    category_prompts = {
        "user_preference": "These are user preferences. Synthesize a single clear user profile rule:",
        "response_format": "These are response format instructions. Synthesize a single output formatting rule:",
        "business_context": "These are business/domain constraints. Synthesize a single business rule:",
        "bug": "These are bug reports. Synthesize a triage summary for developers:",
        "reprompting": "These are prompt strategy feedback items. Synthesize a prompt improvement suggestion:",
        "other": "These are uncategorized feedback items. Identify the common pattern:",
    }
    cat_prompt = category_prompts.get(cluster["category"], category_prompts["other"])
    try:
        result = await call_llm(
            [
                {
                    "role": "system",
                    "content": "You synthesize feedback clusters into single actionable learnings. Respond with one concise sentence only.",
                },
                {
                    "role": "user",
                    "content": f"{cat_prompt}\n\nFeedback items ({len(cluster['members'])}):\n{member_texts}",
                },
            ],
            temperature=0,
        )
        return result.strip()
    except Exception:
        return f"Pattern: {len(cluster['members'])} similar {cluster['category']} items"


async def run_clustering(session: dict) -> dict:
    """Run the clustering engine on unclustered feedback.

    Groups by category → text similarity → merge or create cluster →
    LLM synthesizes learnings for clusters with 2+ members.
    """
    unclustered = get_unclustered_feedback(session)
    if len(unclustered) < CLUSTER_THRESHOLD:
        return {
            "ran": False,
            "reason": f"{len(unclustered)}/{CLUSTER_THRESHOLD} unclustered",
        }

    clusters = session["feedback_clusters"]

    # Group by category first (gagent-core does this)
    by_category: dict[str, list] = {}
    for f in unclustered:
        by_category.setdefault(f["classification"], []).append(f)

    new_clusters = 0
    updated_clusters = 0
    reactivated = 0
    ops: list[str] = []

    for category, items in by_category.items():
        for feedback in items:
            match, score = find_best_cluster(feedback, clusters)
            if match:
                match["members"].append(
                    {"structured": feedback["structured"], "id": feedback["id"]}
                )
                update_centroid(match, feedback["structured"])
                if match.get("archived"):
                    match["archived"] = False
                    reactivated += 1
                    ops.append(
                        f'REACTIVATED cluster "{match["centroid_text"][:40]}..." (score={score:.2f})'
                    )
                else:
                    updated_clusters += 1
                    ops.append(
                        f'ADDED to cluster "{match["centroid_text"][:40]}..." '
                        f'({len(match["members"])} members, score={score:.2f})'
                    )
            else:
                new_cluster = {
                    "id": str(uuid.uuid4()),
                    "category": category,
                    "centroid_text": feedback["structured"],
                    "members": [
                        {"structured": feedback["structured"], "id": feedback["id"]}
                    ],
                    "learning": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "archived": False,
                    "_announced": False,
                }
                clusters.append(new_cluster)
                new_clusters += 1
                ops.append(
                    f'NEW cluster [{category}]: "{feedback["structured"][:50]}"'
                )
            feedback["_clustered"] = True

    # Synthesize learnings for clusters with 2+ members
    synthesized = 0
    for cluster in clusters:
        if len(cluster["members"]) >= 2 and not cluster.get("archived"):
            cluster["learning"] = await synthesize_learning(cluster)
            synthesized += 1

    return {
        "ran": True,
        "newClusters": new_clusters,
        "updatedClusters": updated_clusters,
        "reactivated": reactivated,
        "synthesized": synthesized,
        "ops": ops,
    }


# ══════════════════════════════════════════════════════════════════
# Response Eval Engine (LLM-as-Judge)
# Eugene Yan pattern #1: "Evals enable us to measure how well our
# system is doing." Uses G-Eval style scoring.
# Runs after every response when skills exist.
# ══════════════════════════════════════════════════════════════════

async def eval_response(
    query: str, response_text: str, skills: list[str]
) -> Optional[dict]:
    """Score how well the response follows learned skills (LLM-as-judge)."""
    if not skills:
        return None
    skills_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(skills))
    try:
        raw = await call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an eval judge for an AI assistant. Score how well "
                        "the response follows the learned skills/rules.\n\n"
                        "Return JSON only:\n"
                        "{\n"
                        '  "score": <0.0-1.0>,\n'
                        '  "compliant": ["skill text that WAS followed"],\n'
                        '  "violations": ["skill text that was NOT followed"],\n'
                        '  "summary": "one sentence explanation"\n'
                        "}\n\n"
                        "Be strict: if a skill is clearly applicable to this query and "
                        "the response ignores it, it's a violation. If the skill isn't "
                        "relevant to this query, count it as compliant (benefit of the doubt)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## Active Skills:\n{skills_list}\n\n"
                        f"## User Query:\n{query}\n\n"
                        f"## Assistant Response:\n{response_text[:500]}"
                    ),
                },
            ],
            temperature=0,
        )
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```", "", cleaned).strip()
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Eval failed: {e}")
        return {
            "score": None,
            "compliant": [],
            "violations": [],
            "summary": f"Eval error: {e}",
        }


# ══════════════════════════════════════════════════════════════════
# Response Generation
# Mirrors the multi-agent pattern: orchestrator handles normal
# conversation, hands off to feedback_analyst for introspection.
# ══════════════════════════════════════════════════════════════════

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

    # ── Handoff → feedback_analyst (introspection only) ──
    if re.search(
        r"show.*learn|what.*learn|system state|feedback summary|active skills|introspect",
        lower,
    ):
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

    # ── Recall from learned skills ──
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


# ══════════════════════════════════════════════════════════════════
# Pipeline Orchestrator — the main flow
# Exact same sequence as HTML dashboard's sendMessage():
#   sense → working memory → classify → guardrails → skills →
#   cluster → episodic → chat → agent → eval → trace
# ══════════════════════════════════════════════════════════════════

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
            i
            for i in guarded
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
            i for i in passed if i.get("should_become_skill") and i.get("business_context")
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
                if not issue.get("_guardrail_blocked") and not issue.get(
                    "_guardrail_deduped"
                ):
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


# ══════════════════════════════════════════════════════════════════
# State API — returns full pipeline state for dashboard
# ══════════════════════════════════════════════════════════════════

def get_full_state(session_id: str = "default") -> dict:
    """Get complete system state for the dashboard."""
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


# ══════════════════════════════════════════════════════════════════
# FastAPI App + API Endpoints
# ══════════════════════════════════════════════════════════════════

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


@app.get("/")
async def serve_dashboard():
    """Serve the HTML dashboard."""
    return FileResponse("dashboard.html", media_type="text/html")


# ══════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  ADAPTIVE FEEDBACK PIPELINE — Python Backend")
    print("  Real-time, context-aware, self-improving agent system")
    print("=" * 60)
    print()
    print("  Pipeline stages:")
    print("    1. 📡 Signal Collector (SENSE)")
    print("    2. 🧠 Working Memory (decay-based)")
    print("    3. 💬 Feedback Classifier (LLM)")
    print("    4. 🛡️  Guardrails (confidence/contradiction/dedup)")
    print("    5. 🧬 Skills Registry (SQLite)")
    print("    6. 🔮 Feedback Clustering (similarity)")
    print("    7. 📚 Episodic Memory (SQLite)")
    print("    8. 💾 Chat History")
    print("    9. 🤖 Multi-Agent Response")
    print("   10. 📏 Response Eval (LLM-as-judge)")
    print("   11. 🔍 Decision Tracer (SQLite)")
    print()
    print("  🎯 Dashboard: http://localhost:8000")
    print("  📊 State API: http://localhost:8000/api/state")
    print("  💬 Chat API:  POST http://localhost:8000/api/chat")
    print()
    print("=" * 60)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
