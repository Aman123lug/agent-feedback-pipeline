"""
Feedback Classifier — LLM-based intent classification.

Determines whether a user message is actionable feedback or normal conversation.
Supports multi-issue splitting (e.g. "call me Aman and use bullet points" → 2 issues).
"""

import re
import json
import logging

from pipeline.config import call_llm

logger = logging.getLogger("pipeline")

# ── Valid categories and remediation mapping ──────────────────

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

# ── Classification prompt ─────────────────────────────────────

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
| user_preference | **User profile / persona memory** | Store in long-term user profile; inject into every future prompt as identity context | "call me Aman", "I like cats over dogs", "speak in Arabic" |
| response_format | **System prompt / output template** | Update output formatting instructions in system prompt | "use bullet points", "keep it short", "give me tables" |
| business_context | **Knowledge base / business rules** | Extract as a durable business rule; inject into agent context | "always cite sources", "never mention competitors" |
| bug | **Codebase / model behavior** | Log for triage; flag for human review; do NOT auto-create a skill | "that answer was wrong", "you hallucinated a link" |
| reprompt | **Prompt engineering** | Adjust prompt strategy, reasoning chain, or tool selection | "rephrase that more simply", "be more creative" |
| other | **Unclear remediation path** | Store for pattern analysis; may become actionable after clustering | anything else that's clearly feedback but ambiguous |

## Decision rules (follow strictly, in order)
1. **Questions are NEVER feedback.** If the message ends with "?" OR starts with a question word → is_feedback=false, stop.
2. **Greetings & casual chat are NOT feedback.** "hi", "hello", "thanks", "ok", "cool" → is_feedback=false.
3. **Factual statements without instruction are NOT feedback.** "The sky is blue" → is_feedback=false.
4. **Vague single preferences without contrast are NOT feedback.** "I like flowers" → too vague. But "I like roses over tulips" → is_feedback=true.
5. **If the message instructs the assistant to change behavior → is_feedback=true.**
6. **Disambiguation rule:** ask "where does the fix live?" to pick the category.
7. **should_become_skill=true** only for user_preference, response_format, business_context.
8. **should_become_skill=false** for bug, reprompt, other.
9. **structured**: Rewrite the feedback as a clean imperative rule.
10. **remediation_target**: Tells the system WHERE to apply the fix.

User message: "{USER_MESSAGE}" """


# ── Helpers ───────────────────────────────────────────────────

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


# ── Main classifier ──────────────────────────────────────────

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
