"""Shared data models used across pipelines, retrieval, and benchmarking.

These types are the contract between components. Every pipeline returns a
`PipelineResult`, so the benchmark harness can score all three uniformly and
the dashboard can render them side by side.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Corpus + questions
# --------------------------------------------------------------------------- #
@dataclass
class Document:
    """A single corpus document (one line of corpus.jsonl)."""

    doc_id: str
    title: str
    text: str
    url: str = ""
    wikidata_qid: str = ""
    approx_tokens: int = 0
    # Parsed infobox fields (populated by the ingestion parser).
    infobox: dict[str, str] = field(default_factory=dict)

    @property
    def is_olympic_event(self) -> bool:
        return "[Infobox Olympic event]" in self.text


class QType(str, Enum):
    LOOKUP = "lookup"
    MULTI_HOP = "multi_hop"
    AGGREGATION = "aggregation"
    SUPERLATIVE = "superlative"
    TEMPORAL = "temporal"
    OTHER = "other"


@dataclass
class Question:
    qid: str
    question: str
    qtype: QType = QType.OTHER
    gold_doc_ids: list[str] = field(default_factory=list)
    answer: list[str] = field(default_factory=list)  # empty for hidden set
    answer_named_in_question: bool = False
    answer_verified: bool = False

    @property
    def has_gold(self) -> bool:
        return bool(self.answer)


# --------------------------------------------------------------------------- #
# Evidence + trace (explainability)
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    """A piece of retrieved evidence with a citation back to a source doc."""

    doc_id: str
    title: str
    snippet: str
    score: float = 0.0
    source: str = ""  # which retriever/tool produced it (e.g. "vector", "graph")
    url: str = ""     # source URL for citation resolution

    def citation(self) -> str:
        return f"[{self.doc_id}] {self.title}"


@dataclass
class TokenUsage:
    context_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.context_tokens + self.input_tokens + self.output_tokens

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.context_tokens += other.context_tokens
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self

    def as_dict(self) -> dict[str, int]:
        return {
            "context_tokens": self.context_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total,
        }


@dataclass
class TraceStep:
    """One step in an investigation: a tool/agent call and its outcome.

    The trace is the backbone of the explainability score (15%) and the
    agentic-effectiveness score (15%). Each step records what was done, why,
    what came back, how long it took, and how many tokens it cost.
    """

    step_index: int
    agent: str  # e.g. "orchestrator", "graph_traversal", "aggregation"
    action: str  # tool/method invoked
    rationale: str = ""  # why the orchestrator chose this action
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs_summary: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "agent": self.agent,
            "action": self.action,
            "rationale": self.rationale,
            "inputs": self.inputs,
            "outputs_summary": self.outputs_summary,
            "evidence_ids": self.evidence_ids,
            "tokens": self.tokens.as_dict(),
            "duration_s": round(self.duration_s, 4),
        }


@dataclass
class PipelineResult:
    """Uniform output of every pipeline for a single question."""

    qid: str
    pipeline: str  # "rag" | "graphrag" | "agentic"
    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    latency_s: float = 0.0
    stopped_reason: str = ""  # why the agent decided it had enough
    strategy_changed: bool = False  # did the agent switch strategy mid-run
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cited_doc_ids(self) -> list[str]:
        seen: list[str] = []
        for e in self.evidence:
            if e.doc_id not in seen:
                seen.append(e.doc_id)
        return seen

    @property
    def num_steps(self) -> int:
        return len(self.trace)

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "pipeline": self.pipeline,
            "answer": self.answer,
            "citations": [e.citation() for e in self.evidence],
            "cited_doc_ids": self.cited_doc_ids,
            "num_steps": self.num_steps,
            "num_chunks": len(self.evidence),
            "tokens": self.tokens.as_dict(),
            "latency_s": round(self.latency_s, 4),
            "stopped_reason": self.stopped_reason,
            "strategy_changed": self.strategy_changed,
            "trace": [s.as_dict() for s in self.trace],
            "meta": self.meta,
        }


class Timer:
    """Small context manager for step timing."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
