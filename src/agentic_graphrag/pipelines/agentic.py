"""Pipeline 3: Agentic GraphRAG.

Thin adapter that exposes the agent harness through the common Pipeline
interface, so the benchmark treats all three pipelines identically.
"""
from __future__ import annotations

from ..agents.harness import AgentHarness
from ..graph.base import GraphBackend
from ..llm import LLMClient
from ..models import PipelineResult, Question
from ..retrieval.vector_store import VectorStore
from .base import Pipeline


class AgenticPipeline(Pipeline):
    name = "agentic"

    def __init__(
        self,
        graph: GraphBackend,
        vector_store: VectorStore,
        llm: LLMClient,
        max_steps: int = 8,
        planner: str = "rule",
        fact_store: object = None,
        embedder: object = None,
    ) -> None:
        self.harness = AgentHarness(
            graph, vector_store, llm, max_steps=max_steps,
            planner=planner, fact_store=fact_store, embedder=embedder,
        )

    def answer(self, question: Question) -> PipelineResult:
        return self.harness.investigate(question)
