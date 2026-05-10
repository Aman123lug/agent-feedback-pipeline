# Adaptive Feedback Pipeline — POC

> **How would you design a real-time, context-aware AI system that runs continuously and adapts to user behavior?**

A proof-of-concept implementing the exact architecture described in the interview — a **streaming-first, multi-agent system with layered memory, self-evolving skills, and decision tracing**.

## Core Design Principles (from the interview answer)

1. **Streaming-first, not request-response** — Instead of fetching context at query time, the system continuously takes in signals (queries, feedback, tool results) and keeps updating the agent's context in real time.

2. **Layered Memory Architecture**:
   - **Working Memory** (`WorkingMemory`) — current-turn signals, relevance-scored, with decay. Fixes the "context rot at turn 15" problem by curating what enters the context window.
   - **Episodic Memory** (`EpisodicMemory`) — compressed session history summaries. Every N turns, the conversation is summarized and the raw history is replaced.
   - **Procedural Memory** (`SkillsRegistry`) — self-evolving business rules extracted from feedback. Not static notes — executable instructions that auto-rewrite based on feedback signals.

3. **Signal Collection Pipeline** (`SignalCollector`) — sense → estimate → act:
   - **Sense**: Collect raw signals (queries, feedback, tool results, implicit patterns)
   - **Estimate**: Score relevance, detect repeated topics, measure feedback velocity
   - **Act**: Update working memory with only high-signal context

4. **Feedback Loop** — User feedback is collected, LLM-classified (bug/reprompting/business_context/other), and business_context feedback auto-registers as runtime skills that are hot-swapped into every subsequent turn.

5. **Decision Tracing** — Every agent action is logged with a snapshot of exactly what context it had. You can replay what it saw and understand why it went wrong.

6. **Multi-Agent with Shared Context** — Different agents (orchestrator, knowledge specialist, feedback analyst) share the same state layer via ContextProviders. When one agent evolves a skill, it's instantly available to all.

## Architecture

```
              ┌─────────────────────────────────────────────────┐
              │         Signal Collector (sense → estimate → act)│
User ────────▶│  query, feedback, tool results, implicit signals │
              └───────────────────┬─────────────────────────────┘
                                  │
                                  ▼
              ┌─────────────────────────────────────────────────┐
              │              Working Memory                      │
              │  (relevance-scored, decaying signals — NOT a log)│
              │  Fixes context rot at turn 15+                   │
              └───────────────────┬─────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
          ┌──────────────┐ ┌──────────┐ ┌────────────────┐
          │ Procedural   │ │ Episodic │ │ Decision       │
          │ Memory       │ │ Memory   │ │ Tracer         │
          │ (Skills from │ │ (Session │ │ (Context       │
          │  feedback)   │ │  history)│ │  snapshots)    │
          └──────┬───────┘ └────┬─────┘ └───────┬────────┘
                 │              │                │
                 └──────────────┼────────────────┘
                                │
              ┌─────────────────┴────────────────────────────┐
              │     WorkflowAgent (HandoffBuilder swarm)      │
              │  ┌────────────┐ ┌──────────┐ ┌────────────┐  │
              │  │Orchestrator│ │Knowledge │ │ Feedback   │  │
              │  │(routes +   │ │ Agent    │ │ Analyst    │  │
              │  │ feedback)  │ │          │ │            │  │
              │  └────────────┘ └──────────┘ └────────────┘  │
              └──────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — FastAPI + gagent_core UI, signal collection, streaming |
| `orchestrator.py` | Multi-agent swarm (HandoffBuilder), all 4 context providers |
| `context_providers.py` | WorkingMemory, FeedbackContext, EpisodicMemory, DecisionTracing |
| `tools.py` | Agent tools: provide_feedback, get_feedback_summary, get_decision_trace |
| `stores.py` | All memory layers: WorkingMemory, SkillsRegistry, EpisodicMemory, FeedbackStore, SignalCollector, DecisionTracer |

## Running

```bash
# From the parent project's venv (which has gagent_core installed)
cd adaptive_feedback_poc
python main.py
```

Open `http://localhost:8001` (gagent_core built-in UI) and try:

1. **Normal query**: "What is machine learning?" → routed to knowledge agent
2. **Give feedback**: "Always respond in bullet points" → classified as business_context, auto-registered as skill
3. **Next query**: Ask anything → system now responds in bullet points (skill injected)
4. **Inspect**: "Show me what the system has learned" → feedback analyst shows skills + traces
5. **API**: `http://localhost:8001/api/system_state` → full state dump

## Environment Variables

Uses the same `.env` as the parent project. Key vars:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY` 
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`
- `APPLICATION_ID` (defaults to `adaptive-feedback-poc`)
