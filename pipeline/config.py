"""
Pipeline Config — Azure OpenAI client and pipeline constants.
"""

import os
import logging
from openai import AsyncAzureOpenAI

logger = logging.getLogger("pipeline")

# ── Azure OpenAI ──────────────────────────────────────────────

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

if not AZURE_ENDPOINT or not AZURE_API_KEY:
    raise EnvironmentError(
        "Missing required environment variables: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY. "
        "Copy .env.example to .env and fill in your values."
    )
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

llm_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

# ── Pipeline thresholds ───────────────────────────────────────

CLUSTER_THRESHOLD = 2           # auto-cluster after N unclustered feedback
SIMILARITY_THRESHOLD = 0.55     # text similarity cutoff for clustering
CONFIDENCE_THRESHOLD = 0.7      # guardrail: minimum confidence to accept
DECAY_RATE = 0.85               # working memory decay per turn
RELEVANCE_THRESHOLD = 0.3       # minimum relevance to stay in working memory


# ── LLM helper ────────────────────────────────────────────────

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
