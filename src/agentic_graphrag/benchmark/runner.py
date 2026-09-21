"""Benchmark runner: execute all three pipelines over a question set.

Builds the shared graph + vector store once, runs RAG, GraphRAG, and Agentic
GraphRAG on every question, scores each result, and writes:
  - artifacts/results_raw.jsonl   : every PipelineResult (full trace, tokens)
  - artifacts/scored.jsonl        : scored records for analysis
  - artifacts/summary.json        : per-pipeline aggregate metrics
  - artifacts/submission_<set>.jsonl : hidden-set answers + trace for scoring

The hidden set has no gold answers, so accuracy is skipped there; we still emit
the required raw outputs (answer, tokens, agentic trace) for the organizers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..config import Config, load_config
from ..data.loader import load_documents, load_questions
from ..graph.factory import build_backend
from ..llm import LLMClient
from ..models import Question
from ..pipelines.agentic import AgenticPipeline
from ..pipelines.base import Pipeline
from ..pipelines.graphrag import GraphRAGPipeline
from ..pipelines.rag import RAGPipeline
from ..retrieval.embeddings import build_embedder
from ..retrieval.vector_store import VectorStore
from .metrics import ScoredResult, score_result, summarize


@dataclass
class BenchmarkEngine:
    config: Config
    graph: object
    vector_store: VectorStore
    llm: LLMClient
    pipelines: dict[str, Pipeline]

    @classmethod
    def build(cls, config: Optional[Config] = None, top_k: int = 5, max_steps: int = 8) -> "BenchmarkEngine":
        cfg = config or load_config()
        print("[benchmark] loading corpus...")
        docs = load_documents(cfg.corpus_path)
        print(f"[benchmark] {len(docs)} docs loaded; building graph...")
        graph = build_backend(cfg, docs)
        print(f"[benchmark] graph: {graph.stats()}")
        embedder = build_embedder(cfg.embedding)

        # Vector store: use TigerGraph Vector DB natively when on TigerGraph
        # (vectors are already indexed there - NO local rebuild). Only build a
        # local index for explicit offline 'local' mode.
        if (cfg.graph_backend or "").lower() == "tigergraph" and hasattr(graph, "vector_search"):
            from ..retrieval.tg_vector_store import TigerGraphVectorStore
            vs = TigerGraphVectorStore(graph, embedder)
            print("[benchmark] vector store: TigerGraph Vector DB (native, no local build)")
        else:
            print("[benchmark] building local vector store (embedding chunks)...")
            vs = VectorStore(embedder).build(docs)
            print(f"[benchmark] {len(vs)} chunks indexed with {type(embedder).__name__}")
        llm = LLMClient(cfg.llm)
        pipelines: dict[str, Pipeline] = {
            "rag": RAGPipeline(vs, llm, top_k=top_k),
            "graphrag": GraphRAGPipeline(graph, vs, llm, top_k=3),
            "agentic": AgenticPipeline(graph, vs, llm, max_steps=max_steps,
                                       planner=cfg.planner, embedder=embedder),
        }
        print(f"[benchmark] agent planner: {cfg.planner}")
        return cls(cfg, graph, vs, llm, pipelines)

    def run(
        self,
        questions: list[Question],
        which: Optional[list[str]] = None,
        label: str = "public",
    ) -> dict:
        which = which or ["rag", "graphrag", "agentic"]
        art = self.config.artifacts_dir
        art.mkdir(parents=True, exist_ok=True)

        raw_path = art / f"results_raw_{label}.jsonl"
        scored: list[ScoredResult] = []
        raw_fh = open(raw_path, "w", encoding="utf-8")

        for q in tqdm(questions, desc=f"benchmark[{label}]"):
            for name in which:
                result = self.pipelines[name].answer(q)
                raw_fh.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
                scored.append(score_result(q, result))
        raw_fh.close()

        # scored records
        with open(art / f"scored_{label}.jsonl", "w", encoding="utf-8") as fh:
            for sr in scored:
                fh.write(json.dumps(sr.as_dict(), ensure_ascii=False) + "\n")

        # summary
        summaries = summarize(scored)
        summary_dict = {name: s.as_dict() for name, s in summaries.items()}
        with open(art / f"summary_{label}.json", "w", encoding="utf-8") as fh:
            json.dump(summary_dict, fh, indent=2, ensure_ascii=False)

        return {"summaries": summary_dict, "scored": scored, "raw_path": str(raw_path)}

    def emit_submission(self, questions: list[Question], pipeline: str = "agentic", label: str = "hidden") -> str:
        """Emit the required hidden-set submission: answer + tokens + trace."""
        art = self.config.artifacts_dir
        out = art / f"submission_{label}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for q in tqdm(questions, desc=f"submission[{label}]"):
                result = self.pipelines[pipeline].answer(q)
                fh.write(json.dumps({
                    "qid": q.qid,
                    "question": q.question,
                    "qtype": q.qtype.value,
                    "answer": result.answer,
                    "pipeline": pipeline,
                    "tokens": result.tokens.as_dict(),
                    "num_steps": result.num_steps,
                    "citations": result.cited_doc_ids,
                    "strategy_changed": result.strategy_changed,
                    "stopped_reason": result.stopped_reason,
                    "trace": [s.as_dict() for s in result.trace],
                }, ensure_ascii=False) + "\n")
        return str(out)
