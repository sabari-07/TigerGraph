"""Scoring metrics for the three-way benchmark.

The hackathon scores accuracy, completeness, and token efficiency, plus agentic
trace behavior. This module turns a PipelineResult + gold answer into scored
records and aggregates them per pipeline and per question type.

Accuracy is measured with a normalized match (exact or containment either way),
which is robust to punctuation and phrasing while staying strict on the
substantive answer. Completeness measures how much of the gold-supporting
document set the pipeline actually cited (evidence grounding). Token efficiency
is total tokens per answer, and we also report an accuracy-per-1k-tokens ratio
that captures the "is agentic worth the cost" question directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import PipelineResult, Question


def normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


def answer_hit(answer: str, golds: list[str]) -> bool:
    a = normalize(answer)
    if not a:
        return False
    for g in golds:
        gg = normalize(g)
        if not gg:
            continue
        if a == gg or gg in a or a in gg:
            return True
    return False


def citation_completeness(result: PipelineResult, gold_doc_ids: list[str]) -> float:
    """Fraction of gold supporting documents that appear in the citations."""
    if not gold_doc_ids:
        return 1.0 if result.cited_doc_ids else 0.0
    cited = set(result.cited_doc_ids)
    hit = sum(1 for d in gold_doc_ids if d in cited)
    return hit / len(gold_doc_ids)


@dataclass
class ScoredResult:
    qid: str
    pipeline: str
    qtype: str
    correct: bool
    completeness: float
    total_tokens: int
    num_steps: int
    num_chunks: int
    latency_s: float
    strategy_changed: bool
    stopped_reason: str
    answer: str
    gold: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid, "pipeline": self.pipeline, "qtype": self.qtype,
            "correct": self.correct, "completeness": round(self.completeness, 3),
            "total_tokens": self.total_tokens, "num_steps": self.num_steps,
            "num_chunks": self.num_chunks, "latency_s": round(self.latency_s, 4),
            "strategy_changed": self.strategy_changed,
            "stopped_reason": self.stopped_reason,
            "answer": self.answer, "gold": self.gold,
        }


def score_result(question: Question, result: PipelineResult) -> ScoredResult:
    return ScoredResult(
        qid=question.qid,
        pipeline=result.pipeline,
        qtype=question.qtype.value,
        correct=answer_hit(result.answer, question.answer) if question.has_gold else False,
        completeness=citation_completeness(result, question.gold_doc_ids),
        total_tokens=result.tokens.total,
        num_steps=result.num_steps,
        num_chunks=len(result.evidence),
        latency_s=result.latency_s,
        strategy_changed=result.strategy_changed,
        stopped_reason=result.stopped_reason,
        answer=result.answer,
        gold=question.answer,
    )


@dataclass
class PipelineSummary:
    pipeline: str
    n: int = 0
    correct: int = 0
    completeness_sum: float = 0.0
    tokens_sum: int = 0
    steps_sum: int = 0
    chunks_sum: int = 0
    latency_sum: float = 0.0
    strategy_changes: int = 0
    by_qtype: dict[str, list[int]] = field(default_factory=dict)  # qtype -> [correct, n]

    def add(self, sr: ScoredResult) -> None:
        self.n += 1
        self.correct += int(sr.correct)
        self.completeness_sum += sr.completeness
        self.tokens_sum += sr.total_tokens
        self.steps_sum += sr.num_steps
        self.chunks_sum += sr.num_chunks
        self.latency_sum += sr.latency_s
        self.strategy_changes += int(sr.strategy_changed)
        bucket = self.by_qtype.setdefault(sr.qtype, [0, 0])
        bucket[0] += int(sr.correct)
        bucket[1] += 1

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def avg_completeness(self) -> float:
        return self.completeness_sum / self.n if self.n else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.tokens_sum / self.n if self.n else 0.0

    @property
    def avg_steps(self) -> float:
        return self.steps_sum / self.n if self.n else 0.0

    @property
    def avg_latency(self) -> float:
        return self.latency_sum / self.n if self.n else 0.0

    @property
    def accuracy_per_1k_tokens(self) -> float:
        """Accuracy earned per 1000 tokens spent - the efficiency headline."""
        if self.tokens_sum == 0:
            return 0.0
        return (self.correct / self.tokens_sum) * 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "correct": self.correct,
            "avg_completeness": round(self.avg_completeness, 4),
            "avg_tokens": round(self.avg_tokens, 1),
            "total_tokens": self.tokens_sum,
            "avg_steps": round(self.avg_steps, 2),
            "avg_chunks": round(self.chunks_sum / self.n, 2) if self.n else 0,
            "avg_latency_s": round(self.avg_latency, 4),
            "strategy_changes": self.strategy_changes,
            "accuracy_per_1k_tokens": round(self.accuracy_per_1k_tokens, 4),
            "by_qtype": {
                qt: {"correct": c, "n": n, "accuracy": round(c / n, 3) if n else 0}
                for qt, (c, n) in sorted(self.by_qtype.items())
            },
        }


def summarize(scored: list[ScoredResult]) -> dict[str, PipelineSummary]:
    summaries: dict[str, PipelineSummary] = {}
    for sr in scored:
        summaries.setdefault(sr.pipeline, PipelineSummary(sr.pipeline)).add(sr)
    return summaries
