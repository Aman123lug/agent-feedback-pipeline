"""
Feedback Clustering Engine — groups similar feedback into clusters.

Simplified version of gagent-core's embedding-based clustering.
Production uses Azure text-embedding-3-small + cosine similarity.
POC uses token-overlap similarity as a proxy (same algorithm, simpler math).
"""

import re
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from pipeline.config import call_llm, CLUSTER_THRESHOLD, SIMILARITY_THRESHOLD

logger = logging.getLogger("pipeline")


# ── Text similarity (token-overlap proxy for cosine) ──────────

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


# ── Cluster helpers ───────────────────────────────────────────

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


# ── Main clustering engine ────────────────────────────────────

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

    # Group by category first (gagent-core pattern)
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
