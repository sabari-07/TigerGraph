"""Pipeline 1: RAG (vector-only baseline).

Single retrieval step: embed the question, pull top-k chunks by cosine
similarity, stuff them into the prompt, and generate. No graph, no iteration.
This is the baseline the other two pipelines must beat, and the reference for
"where a single retrieval is enough".
"""
from __future__ import annotations

import time

from ..llm import LLMClient
from ..models import PipelineResult, Question, TraceStep, Timer
from ..retrieval.vector_store import VectorStore
from .base import ANSWER_SYSTEM, Pipeline, build_answer_prompt


class RAGPipeline(Pipeline):
    name = "rag"

    def __init__(self, vector_store: VectorStore, llm: LLMClient, top_k: int = 5) -> None:
        self.vs = vector_store
        self.llm = llm
        self.top_k = top_k

    def answer(self, question: Question) -> PipelineResult:
        start = time.perf_counter()
        result = PipelineResult(qid=question.qid, pipeline=self.name, answer="")

        # --- Step 1: similarity search --- #
        with Timer() as t:
            evidence = self.vs.search(question.question, k=self.top_k)
        result.evidence = evidence
        result.trace.append(
            TraceStep(
                step_index=0,
                agent="rag",
                action="similarity_search",
                rationale=f"Retrieve top-{self.top_k} chunks by vector similarity.",
                inputs={"query": question.question, "k": self.top_k},
                outputs_summary=f"{len(evidence)} chunks",
                evidence_ids=[e.doc_id for e in evidence],
                duration_s=t.elapsed,
            )
        )

        # --- Step 2: generate --- #
        prompt = build_answer_prompt(question.question, evidence)
        with Timer() as t:
            resp = self.llm.complete(ANSWER_SYSTEM, prompt)
        result.answer = resp.text.strip()
        result.tokens += resp.tokens
        result.trace.append(
            TraceStep(
                step_index=1,
                agent="rag",
                action="generate",
                rationale="Generate answer grounded in retrieved chunks.",
                outputs_summary=result.answer[:120],
                evidence_ids=[e.doc_id for e in evidence],
                tokens=resp.tokens,
                duration_s=t.elapsed,
            )
        )

        result.stopped_reason = "single-shot retrieval (no iteration)"
        result.latency_s = time.perf_counter() - start
        return result
