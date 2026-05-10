"""
Response Eval — LLM-as-judge compliance scoring.

Eugene Yan pattern #1: "Evals enable us to measure how well our system is doing."
Uses G-Eval style scoring to check whether responses follow learned skills.
Runs after every response when skills exist.
"""

import re
import json
import logging
from typing import Optional

from pipeline.config import call_llm

logger = logging.getLogger("pipeline")


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
