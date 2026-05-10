"""
Guardrails Engine — validates classified feedback before skill creation.

4 checks:
1. Schema validation  — structured text must be present
2. Confidence gating  — reject low-confidence classifications
3. Contradiction detection — new skill contradicts existing skill (override)
4. Duplicate detection — new skill too similar to existing skill (skip)

Eugene Yan pattern #5: "Guardrails help prevent the model from
generating undesirable output."
"""

import logging

from pipeline.config import CONFIDENCE_THRESHOLD
from pipeline.clustering import text_similarity

logger = logging.getLogger("pipeline")


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
