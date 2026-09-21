"""Agent harness: runs the orchestrator loop and manages the investigation.

Responsibilities:
  - drive the plan -> act -> observe -> evaluate loop
  - accumulate evidence and resolved facts into AgentState
  - enforce stopping criteria (confident answer, no more actions, step budget)
  - record a full TraceStep per action for explainability + agentic scoring
  - synthesize the final answer (deterministic when the graph produced one,
    otherwise via the LLM over gathered evidence)
"""
from __future__ import annotations

import time
from typing import Optional

from ..graph.base import GraphBackend
from ..graph.parser_fallback import enrich_parse
from ..llm import LLMClient
from ..models import PipelineResult, Question, TraceStep, Timer
from ..pipelines.base import ANSWER_SYSTEM, build_answer_prompt
from ..retrieval.vector_store import VectorStore
from .llm_planner import LLMPlanner
from .orchestrator import Orchestrator
from .state import AgentState
from .tools import ToolResult, Tools


class AgentHarness:
    def __init__(
        self,
        graph: GraphBackend,
        vector_store: VectorStore,
        llm: LLMClient,
        max_steps: int = 8,
        fact_store: object = None,
        planner: str = "rule",
        embedder: object = None,
    ) -> None:
        self.tools = Tools(graph, vector_store, fact_store=fact_store, embedder=embedder)
        self.planner_kind = planner
        if planner == "llm":
            self.orchestrator = LLMPlanner(self.tools, llm, max_steps=max_steps)
        else:
            self.orchestrator = Orchestrator(self.tools, max_steps=max_steps)
        self.llm = llm
        self.max_steps = max_steps

    def investigate(self, question: Question) -> PipelineResult:
        start = time.perf_counter()
        result = PipelineResult(qid=question.qid, pipeline="agentic", answer="")

        # Reset per-investigation planner token counter (LLM planner reuses instance).
        if hasattr(self.orchestrator, "tokens_used"):
            self.orchestrator.tokens_used = 0

        # Parse with the fast regex path, repairing weak results via fallbacks
        # (graph-grounded slot resolution + optional LLM). This keeps common
        # questions token-free while staying robust to novel phrasing. The LLM
        # slot extractor is only consulted when a real LLM is configured (not
        # the offline mock), so the default path stays deterministic.
        parse_llm = self.llm if getattr(self.llm, "_client", None) is not None else None
        pq = enrich_parse(question.question, self.tools.g, llm=parse_llm)
        state = AgentState(question=question.question, parsed=pq)

        step_idx = 0
        # ---- plan/act loop ---- #
        while step_idx < self.max_steps:
            plan = self.orchestrator.plan_next(state)
            if plan is None:
                state.done = True
                state.stop_reason = state.stop_reason or "confident answer / no further actions"
                break
            tool_name, kwargs, rationale = plan

            with Timer() as t:
                tr = self._dispatch(tool_name, state, kwargs)
            self._apply(state, tr)

            result.trace.append(
                TraceStep(
                    step_index=step_idx,
                    agent=tool_name,
                    action=tool_name,
                    rationale=rationale + (" | " + tr.rationale if tr.rationale else ""),
                    inputs=kwargs,
                    outputs_summary=tr.outputs_summary,
                    evidence_ids=[e.doc_id for e in tr.evidence],
                    duration_s=t.elapsed,
                )
            )
            state.note(tool_name)
            step_idx += 1
        else:
            state.stop_reason = f"reached step budget ({self.max_steps})"

        # ---- synthesize final answer ---- #
        result.evidence = state.evidence
        result.strategy_changed = state.strategy_changed
        result.stopped_reason = state.stop_reason
        result.meta["intent"] = pq.intent
        result.meta["confidence"] = round(state.confidence, 3)
        result.meta["num_candidates"] = len(state.candidates)
        result.meta["planner"] = self.planner_kind
        # Account for planner LLM overhead (LLM planner only).
        planner_tokens = getattr(self.orchestrator, "tokens_used", 0)
        if planner_tokens:
            result.tokens.input_tokens += planner_tokens
            result.meta["planner_tokens"] = planner_tokens

        with Timer() as t:
            answer = self._finalize(question, state, result)
        result.answer = answer
        # record the synthesis step
        result.trace.append(
            TraceStep(
                step_index=step_idx,
                agent="evidence_evaluation",
                action="synthesize_answer",
                rationale="Evaluate gathered evidence and produce the grounded final answer.",
                outputs_summary=answer[:120],
                duration_s=t.elapsed,
            )
        )

        result.latency_s = time.perf_counter() - start
        return result

    # ------------------------------------------------------------------ #
    def _dispatch(self, tool_name: str, state: AgentState, kwargs: dict) -> ToolResult:
        fn = getattr(self.tools, tool_name)
        return fn(state, **kwargs)

    def _apply(self, state: AgentState, tr: ToolResult) -> None:
        state.add_evidence(tr.evidence)
        if tr.facts:
            state.facts.update(tr.facts)
        if tr.candidates is not None:
            state.candidates = tr.candidates
        # Adopt a proposed answer if it improves confidence.
        if tr.proposed_answer not in (None, "") and tr.confidence >= state.confidence:
            state.proposed_answer = tr.proposed_answer
            state.confidence = tr.confidence

    def _finalize(self, question: Question, state: AgentState, result: PipelineResult) -> str:
        # If a tool produced a confident grounded answer, use it deterministically.
        if state.proposed_answer not in (None, "") and state.confidence >= 0.6:
            # Still pass through the LLM for natural phrasing but seed the answer,
            # keeping tokens low and the answer grounded.
            prompt = (
                build_answer_prompt(question.question, state.evidence)
                + f"\n\nComputed by agent: FINAL_ANSWER: {state.proposed_answer}"
            )
            resp = self.llm.complete(ANSWER_SYSTEM, prompt, max_tokens=64)
            result.tokens += resp.tokens
            return state.proposed_answer.strip()

        # Otherwise let the LLM reason over whatever evidence we gathered.
        prompt = build_answer_prompt(question.question, state.evidence)
        resp = self.llm.complete(ANSWER_SYSTEM, prompt)
        result.tokens += resp.tokens
        return resp.text.strip()
