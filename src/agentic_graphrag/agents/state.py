"""Agent state: the working memory of an investigation.

Holds the parsed question, accumulated evidence, derived facts (slots the agent
has resolved, e.g. the concrete Games edition), a running token budget, and the
list of actions still worth trying. The orchestrator reads/writes this state to
decide the next move and when to stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..graph.base import Node
from ..graph.entity_linking import ParsedQuestion
from ..models import Evidence, TokenUsage


@dataclass
class AgentState:
    question: str
    parsed: ParsedQuestion
    # resolved facts the agent has established during the investigation
    facts: dict[str, Any] = field(default_factory=dict)
    # candidate graph nodes under consideration (e.g. events at a venue)
    candidates: list[Node] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    # actions already taken (tool names), to avoid loops
    history: list[str] = field(default_factory=list)
    # a candidate answer proposed but not yet confirmed
    proposed_answer: Optional[str] = None
    confidence: float = 0.0
    strategy_changed: bool = False
    done: bool = False
    stop_reason: str = ""

    def add_evidence(self, items: list[Evidence]) -> None:
        seen = {(e.doc_id, e.source) for e in self.evidence}
        for e in items:
            if (e.doc_id, e.source) not in seen:
                self.evidence.append(e)
                seen.add((e.doc_id, e.source))

    def note(self, action: str) -> None:
        self.history.append(action)

    def has_run(self, action: str) -> bool:
        return action in self.history
