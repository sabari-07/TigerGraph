"""The orchestrator: chooses the next action from the question + evidence.

This is what makes the system agentic. Rather than a fixed retrieval sequence,
the orchestrator inspects the current state (what's been resolved, what
evidence exists, what's still missing) and selects the next tool. It changes
strategy when a plan stalls (e.g. an ambiguous multi-hop result triggers
disambiguation, then document retrieval to confirm), and it stops as soon as it
has a confident, grounded answer, keeping token cost proportional to difficulty.

The `plan_next` method returns (tool_name, kwargs, rationale) or None when the
investigation should stop. It is deliberately a transparent policy so the
investigation path is fully explainable; an LLM planner can be swapped in via
`LLMPlanner` without changing the harness.
"""
from __future__ import annotations

from typing import Optional

from .state import AgentState
from .tools import Tools


class Orchestrator:
    """Rule-based planner with dynamic, evidence-driven action selection."""

    def __init__(self, tools: Tools, max_steps: int = 8) -> None:
        self.tools = tools
        self.max_steps = max_steps

    def plan_next(self, s: AgentState) -> Optional[tuple[str, dict, str]]:
        intent = s.parsed.intent

        # Global stop: confident proposed answer already grounded in evidence.
        if s.proposed_answer not in (None, "") and s.confidence >= 0.75 and s.evidence:
            return None

        # Always start by linking entities to the graph.
        if not s.has_run("entity_linking"):
            return ("entity_linking", {}, "First resolve question slots to graph entities.")

        # Route by intent, but adapt based on what evidence we already have.
        if intent == "aggregation":
            if not s.has_run("aggregation"):
                return ("aggregation", {}, "Count matching events over the graph (exact).")
        elif intent == "superlative":
            if not s.has_run("superlative"):
                return ("superlative", {}, "Find the extremum event over the graph (exact).")
        elif intent == "temporal":
            if not s.has_run("temporal"):
                return ("temporal", {}, "Resolve the prior edition then read the medalist.")
            # If temporal couldn't match the event, back off to similarity search.
            if s.confidence < 0.75 and not s.has_run("similarity_search"):
                return ("similarity_search", {"k": 5},
                        "Temporal match weak; gather supporting text to disambiguate the event.")
        elif intent == "lookup":
            if not s.has_run("lookup"):
                return ("lookup", {}, "Read the requested attribute directly from the event node.")
            if s.confidence < 0.75 and not s.has_run("similarity_search"):
                return ("similarity_search", {"k": 5},
                        "Exact title not found; try semantic retrieval to locate the event.")
        elif intent == "multi_hop":
            if not s.has_run("multi_hop"):
                return ("multi_hop", {},
                        "Traverse venue -> event (filter by date), then to the medalist.")
            # Strategy change: multi-hop was ambiguous -> disambiguate by date.
            if s.confidence < 0.75 and len(s.candidates) > 1 and not s.has_run("disambiguate_by_date"):
                s.strategy_changed = True
                return ("disambiguate_by_date", {},
                        "Multiple candidate events share the date; prefer the exact single-day "
                        "match. (strategy change)")
            # Still unsure -> pull the candidate documents ONCE so the LLM can
            # read the actual results and pick the right event.
            if s.confidence < 0.75 and s.candidates and not s.has_run("document_retrieval_multi"):
                s.strategy_changed = True
                return ("document_retrieval_multi", {"limit": 4},
                        "Several events share the venue/date; fetch their documents so the "
                        "answer can be grounded in the actual results. (strategy change)")
            # Last resort: genuinely ambiguous (e.g. two events on the exact same
            # day, no sport named). Commit to the best candidate's medalist
            # rather than returning 'unknown' - a grounded guess beats a blank.
            if s.candidates and not s.has_run("commit_best_candidate"):
                return ("commit_best_candidate", {},
                        "Ambiguity unresolved; commit to the top candidate's medalist.")

        # Fallback for any intent: if we have nothing, try vector search.
        if not s.evidence and not s.has_run("similarity_search"):
            return ("similarity_search", {"k": 5},
                    "No structured path available; fall back to semantic retrieval.")

        return None  # nothing left to try -> stop
