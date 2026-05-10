"""
Adaptive Feedback Pipeline — Orchestrator

Creates a multi-agent swarm using HandoffBuilder:
  - Orchestrator: routes queries, has feedback tools
  - Knowledge Agent: answers questions using context
  - Feedback Agent: dedicated feedback analysis specialist

All agents get FeedbackContextProvider (self-evolving skills injection),
EpisodicMemoryProvider, and DecisionTracingProvider.
"""

from typing import Optional, Any

from agent_framework import WorkflowAgent
from agent_framework.orchestrations import HandoffBuilder

from context_providers import (
    FeedbackContextProvider,
    EpisodicMemoryProvider,
    DecisionTracingProvider,
    WorkingMemoryProvider,
    ChatHistoryProvider,
)
from tools import create_adaptive_feedback_tools
from stores import episodic_memory

import logging
logger = logging.getLogger(__name__)


def _create_client(model_name: Optional[str] = None):
    """Create Azure OpenAI client using env vars (API key auth)."""
    import os
    from agent_framework.azure import AzureOpenAIChatClient
    from gagent_core.settings import settings as core_settings

    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")

    return AzureOpenAIChatClient(
        api_key=api_key,
        azure_endpoint=core_settings.azure_openai.ENDPOINT,
        api_version=core_settings.azure_openai.API_VERSION,
        deployment_name=model_name or core_settings.azure_openai.CHAT_DEPLOYMENT_NAME,
    )


def _create_agent(name, instructions, tools=None, model_name=None, description=None):
    """Create an agent with standard config (same pattern as core_agents/base.py)."""
    client = _create_client(model_name)
    return client.as_agent(
        name=name,
        instructions=instructions,
        tools=tools or [],
        description=description or f"{name} agent",
        default_options={"temperature": 0},
    )


# ─── System Prompts ──────────────────────────────────────────────────────────

ORCHESTRATOR_PROMPT = """You are an intelligent orchestrator for an adaptive AI system.

## ⚠️ CRITICAL — MANDATORY BUSINESS RULES
Before generating ANY response, check your system messages for "MANDATORY BUSINESS RULES".
If any rules exist, you MUST apply ALL of them to EVERY response — no exceptions.
These rules were learned from user feedback and OVERRIDE your default behavior.
For example, if a rule says "always call me aman pagalu", then EVERY response must
address the user as "aman pagalu". Not sometimes — EVERY time.

## Your Capabilities:
1. **Answer questions** directly using your knowledge and any injected context
2. **Collect feedback** when users express opinions, complaints, or suggestions
3. **Show system state** — what the system has learned from feedback, active skills, decision traces
4. **Route to specialists** when the query needs deep analysis

## Feedback Handling:
- When a user gives feedback (complaints, suggestions, preferences like "always respond in Arabic"),
  use the provide_feedback tool IMMEDIATELY
- Business rules from feedback are automatically learned and injected into your context on the next turn
- You can show what the system has learned using get_feedback_summary

## Decision Tracing:
- Every turn is traced with a snapshot of your context
- Users can ask to see decision traces via get_decision_trace

Always be helpful, concise, and transparent about what you know and don't know."""


KNOWLEDGE_AGENT_PROMPT = """You are a knowledge specialist agent in an adaptive AI system.

Your role:
- Answer factual questions using your training knowledge
- Provide detailed, well-structured responses
- Cite sources when possible
- Admit when you don't have enough information rather than guessing

## Important:
- Follow any Active Business Rules injected into your context — these were learned from user feedback
- Be concise but thorough"""


FEEDBACK_ANALYST_PROMPT = """You are a feedback analysis specialist.

Your role:
- Help users understand the feedback that has been collected
- Explain what skills/rules the system has learned
- Show decision traces and explain agent behavior
- Suggest how the system could be improved based on feedback patterns

Use the get_feedback_summary and get_decision_trace tools to answer questions about
the system's learning and behavior."""


# ─── Orchestrator Factory ────────────────────────────────────────────────────

async def initialize_orchestrator(
    thread_id: str,
    user_id: Optional[str] = None,
) -> WorkflowAgent:
    """Initialize the multi-agent swarm with feedback loop.
    
    Returns a WorkflowAgent ready to process messages.
    """

    # Create feedback tools (session-scoped)
    feedback_tools = create_adaptive_feedback_tools(
        session_id=thread_id,
        user_id=user_id,
    )

    # ── Create Agents ────────────────────────────────────────────────────

    orchestrator = _create_agent(
        name="orchestrator",
        instructions=ORCHESTRATOR_PROMPT,
        tools=feedback_tools,
        description="Main orchestrator — routes queries, collects feedback, manages skills",
    )

    knowledge_agent = _create_agent(
        name="knowledge_agent",
        instructions=KNOWLEDGE_AGENT_PROMPT,
        tools=[],
        description="Knowledge specialist for factual questions and detailed analysis",
    )

    feedback_analyst = _create_agent(
        name="feedback_analyst",
        instructions=FEEDBACK_ANALYST_PROMPT,
        tools=feedback_tools[1:],  # get_feedback_summary + get_decision_trace only
        description="Feedback analysis specialist — shows what the system has learned",
    )

    # ── Build Handoff Workflow ───────────────────────────────────────────

    specialists = [knowledge_agent, feedback_analyst]
    participants = [orchestrator] + specialists

    workflow = (
        HandoffBuilder(
            name=f"adaptive_{thread_id}",
            participants=participants,
        )
        .with_start_agent(orchestrator)
        .add_handoff(orchestrator, specialists)
        .add_handoff(knowledge_agent, [orchestrator])
        .add_handoff(feedback_analyst, [orchestrator])
        .build()
    )

    # ── Wrap with Context Providers ──────────────────────────────────────

    outer_providers = [
        ChatHistoryProvider(thread_id=thread_id),              # Chat history (conversation continuity)
        WorkingMemoryProvider(session_id=thread_id),           # Working memory (current turn signals, context rot fix)
        FeedbackContextProvider(),                              # Procedural memory (self-evolving skills)
        EpisodicMemoryProvider(session_id=thread_id),           # Episodic memory (session history)
        DecisionTracingProvider(agent_name="orchestrator", session_id=thread_id),  # Decision tracing
    ]

    workflow_agent = WorkflowAgent(
        workflow=workflow,
        name=f"AdaptiveWorkflow_{thread_id}",
        context_providers=outer_providers,
    )

    logger.info(f"✅ Adaptive workflow initialized for {thread_id}")
    logger.info(f"   Agents: orchestrator, knowledge_agent, feedback_analyst")
    logger.info(f"   Context providers: FeedbackContext, EpisodicMemory, DecisionTracing")

    return workflow_agent
