"""LLM-backed planner: a genuinely agentic tool-calling loop.

Where the rule-based `Orchestrator` encodes the policy directly, this planner
hands the decision to an LLM. At each step the model sees:
  - the original question,
  - the tool catalog (name + when to use it + args),
  - what has been tried, the evidence gathered, and any proposed answer,
and returns the next action as JSON: {"tool": ..., "args": {...},
"rationale": ..., "stop": false}.

This makes the system robust to novel phrasing (the LLM reasons about intent
rather than matching regexes) and demonstrates real agentic planning. It shares
the `plan_next(state)` interface with the rule-based orchestrator, so the
harness is agnostic to which planner runs.

Offline (mock LLM) it falls back to a competent heuristic planner so the whole
system still runs deterministically with no API key.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ..llm import LLMClient
from .state import AgentState
from .tools import Tools

# Tool catalog shown to the planner. Kept in sync with Tools methods.
TOOL_CATALOG = [
    {"tool": "entity_linking", "args": [],
     "use": "Resolve the question's year/season/sport/venue to graph entities. Run first."},
    {"tool": "aggregation", "args": [],
     "use": "Count events in a games edition that exceed a competitor threshold."},
    {"tool": "superlative", "args": [],
     "use": "Find the event with the most competitors in a games edition."},
    {"tool": "temporal", "args": [],
     "use": "Resolve the games edition before a given year, then read the medalist."},
    {"tool": "multi_hop", "args": [],
     "use": "From a venue (+ date), find the event, then the medalist."},
    {"tool": "disambiguate_by_date", "args": [],
     "use": "When multi_hop returns several candidate events, pick the exact single-day one."},
    {"tool": "document_retrieval_multi", "args": ["limit"],
     "use": "Fetch full documents of remaining candidate events to ground the choice."},
    {"tool": "lookup", "args": [],
     "use": "Read a scalar attribute (e.g. number of nations) from a named event."},
    {"tool": "similarity_search", "args": ["k"],
     "use": "Vector search for supporting text when no structured path fits."},
]

_PLANNER_SYSTEM = (
    "You are the orchestrator of an investigation over an Olympic-events "
    "knowledge graph. Choose the single best next tool to move toward a "
    "grounded answer. Prefer exact graph tools over vector search. Stop as soon "
    "as you have a confident answer. Respond ONLY with a JSON object: "
    '{"tool": "<name>", "args": {}, "rationale": "<why>", "stop": <bool>}. '
    "If you already have a confident grounded answer, set stop=true."
)


class LLMPlanner:
    """Agentic planner that asks an LLM for the next action each step."""

    def __init__(self, tools: Tools, llm: LLMClient, max_steps: int = 8) -> None:
        self.tools = tools
        self.llm = llm
        self.max_steps = max_steps
        self.tokens_used = 0  # planner overhead, surfaced in the trace

    def plan_next(self, s: AgentState) -> Optional[tuple[str, dict, str]]:
        # Cheap invariant: always link entities first (saves an LLM call).
        if not s.has_run("entity_linking"):
            return ("entity_linking", {}, "Resolve question slots to graph entities first.")

        # Stop if we already hold a confident, grounded answer.
        if s.proposed_answer not in (None, "") and s.confidence >= 0.75 and s.evidence:
            return None

        prompt = self._build_prompt(s)
        resp = self.llm.complete(_PLANNER_SYSTEM, prompt, max_tokens=120)
        self.tokens_used += resp.tokens.total
        decision = _parse_decision(resp.text)

        # If the LLM (or mock) could not decide, defer to a safe heuristic.
        if decision is None:
            return _heuristic_next(s)

        if decision.get("stop"):
            return None
        tool = decision.get("tool", "")
        if not tool or not hasattr(self.tools, tool) or s.has_run(tool):
            # avoid loops / invalid tools -> heuristic
            return _heuristic_next(s)
        args = decision.get("args") or {}
        rationale = decision.get("rationale", "") + " (LLM-planned)"
        # Mark strategy changes for transparency.
        if tool in ("disambiguate_by_date", "document_retrieval_multi"):
            s.strategy_changed = True
        return (tool, args, rationale)

    def _build_prompt(self, s: AgentState) -> str:
        catalog = "\n".join(
            f"- {t['tool']}({', '.join(t['args'])}): {t['use']}" for t in TOOL_CATALOG
        )
        ev = "; ".join(e.doc_id for e in s.evidence[:8]) or "none"
        cands = len(s.candidates)
        return (
            f"Question: {s.question}\n\n"
            f"Tools:\n{catalog}\n\n"
            f"Parsed intent (hint): {s.parsed.intent}\n"
            f"Resolved facts: {s.facts}\n"
            f"Tools already run: {s.history or 'none'}\n"
            f"Candidate events under consideration: {cands}\n"
            f"Evidence so far: {ev}\n"
            f"Current proposed answer: {s.proposed_answer!r} (confidence {s.confidence:.2f})\n\n"
            f"What is the single best next action?"
        )


# --------------------------------------------------------------------------- #
def _parse_decision(text: str) -> Optional[dict]:
    """Extract the first JSON object from the model's reply."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _heuristic_next(s: AgentState) -> Optional[tuple[str, dict, str]]:
    """Safe fallback policy mirroring the rule-based orchestrator's routing.

    Used when the (mock) LLM does not return a usable decision, so the agentic
    loop still terminates sensibly offline.
    """
    intent = s.parsed.intent
    routing = {
        "aggregation": "aggregation",
        "superlative": "superlative",
        "temporal": "temporal",
        "multi_hop": "multi_hop",
        "lookup": "lookup",
    }
    primary = routing.get(intent)
    if primary and not s.has_run(primary):
        return (primary, {}, f"Heuristic: run {primary} for intent={intent}.")
    if intent == "multi_hop" and s.confidence < 0.75 and len(s.candidates) > 1:
        if not s.has_run("disambiguate_by_date"):
            s.strategy_changed = True
            return ("disambiguate_by_date", {}, "Heuristic: disambiguate ambiguous candidates.")
        if not s.has_run("document_retrieval_multi"):
            s.strategy_changed = True
            return ("document_retrieval_multi", {"limit": 4},
                    "Heuristic: fetch candidate docs to ground the choice.")
    if not s.evidence and not s.has_run("similarity_search"):
        return ("similarity_search", {"k": 5}, "Heuristic: fall back to semantic retrieval.")
    return None
