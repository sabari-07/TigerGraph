"""Common pipeline interface and answer-prompt helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Evidence, Question, PipelineResult

ANSWER_SYSTEM = (
    "You are a precise question-answering assistant. Answer ONLY using the "
    "provided context from a fixed corpus of Olympic-event documents. The "
    "corpus is the sole source of truth. Give the shortest correct answer: a "
    "name, a number, or an event title. Do not explain. If the context does "
    "not contain the answer, reply exactly 'unknown'. Always ground your answer "
    "in the cited documents."
)


def format_context(evidence: list[Evidence], max_chars: int = 6000) -> str:
    """Render evidence into a compact, citation-tagged context block."""
    parts: list[str] = []
    used = 0
    for ev in evidence:
        block = f"[{ev.doc_id}] {ev.title}\n{ev.snippet}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def build_answer_prompt(question: str, evidence: list[Evidence]) -> str:
    ctx = format_context(evidence)
    return (
        f"Context documents:\n{ctx}\n\n"
        f"Question: {question}\n"
        f"Answer (shortest correct form):"
    )


class Pipeline(ABC):
    name: str = "base"

    @abstractmethod
    def answer(self, question: Question) -> PipelineResult:
        ...
