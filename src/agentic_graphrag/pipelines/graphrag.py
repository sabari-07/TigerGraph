"""Pipeline 2: GraphRAG (graph traversal + vector hybrid).

A FIXED two-stage flow (no dynamic replanning):
  1. Parse the question -> link entities -> run the graph traversal that the
     intent implies, computing a grounded answer + gold citations.
  2. Augment with vector-retrieved supporting chunks for the LLM context.

The graph gives structure and exact answers for aggregation/superlative/
temporal/lookup/multi_hop; vector search supplies natural-language support.
This beats pure RAG on connected questions but, unlike the agent, it cannot
recover when its single planned strategy comes up empty.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from ..graph.base import GraphBackend, Node
from ..graph.builder import _games_id
from ..graph.entity_linking import ParsedQuestion, match_event_by_date, parse_question
from ..llm import LLMClient
from ..models import Evidence, PipelineResult, Question, TokenUsage, TraceStep, Timer
from ..retrieval.vector_store import VectorStore
from .base import ANSWER_SYSTEM, Pipeline, build_answer_prompt


def _event_to_evidence(n: Node, source: str = "graph") -> Evidence:
    ib_lines = []
    for k in ("event_name", "year", "season", "competitors", "nations", "date_raw"):
        v = n.props.get(k)
        if v:
            ib_lines.append(f"{k}={v}")
    return Evidence(
        doc_id=n.id,
        title=n.props.get("title", ""),
        snippet="; ".join(ib_lines),
        score=1.0,
        source=source,
        url=n.props.get("url", ""),
    )


class GraphRAGPipeline(Pipeline):
    name = "graphrag"

    def __init__(
        self,
        graph: GraphBackend,
        vector_store: VectorStore,
        llm: LLMClient,
        top_k: int = 3,
    ) -> None:
        self.g = graph
        self.vs = vector_store
        self.llm = llm
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    def answer(self, question: Question) -> PipelineResult:
        start = time.perf_counter()
        result = PipelineResult(qid=question.qid, pipeline=self.name, answer="")

        # --- Step 0: parse + entity link --- #
        with Timer() as t:
            pq = parse_question(question.question, self.g)
        result.trace.append(
            TraceStep(
                step_index=0, agent="graphrag", action="entity_link",
                rationale="Parse question into structured slots and link to graph entities.",
                inputs={"question": question.question},
                outputs_summary=f"intent={pq.intent} slots={pq.slots}",
                duration_s=t.elapsed,
            )
        )

        # --- Step 1: graph traversal by intent --- #
        with Timer() as t:
            answer, evidence, detail = self._graph_answer(pq)
        result.evidence = evidence
        result.trace.append(
            TraceStep(
                step_index=1, agent="graph_traversal", action=f"traverse:{pq.intent}",
                rationale=detail,
                outputs_summary=f"answer={answer!r} from {len(evidence)} graph nodes",
                evidence_ids=[e.doc_id for e in evidence],
                duration_s=t.elapsed,
            )
        )

        # --- Step 2: vector augmentation (supporting docs) --- #
        with Timer() as t:
            vec_ev = self.vs.search(question.question, k=self.top_k)
        # merge, keeping graph evidence first
        seen = {e.doc_id for e in evidence}
        for e in vec_ev:
            if e.doc_id not in seen:
                evidence.append(e)
                seen.add(e.doc_id)
        result.evidence = evidence
        result.trace.append(
            TraceStep(
                step_index=2, agent="graphrag", action="similarity_search",
                rationale="Augment graph facts with vector-retrieved supporting text.",
                outputs_summary=f"{len(vec_ev)} supporting chunks",
                evidence_ids=[e.doc_id for e in vec_ev],
                duration_s=t.elapsed,
            )
        )

        # --- Step 3: generate (or use deterministic graph answer) --- #
        with Timer() as t:
            if answer is not None:
                # Deterministic graph answer: still pass through LLM for phrasing,
                # but seed it so token cost stays low and answer stays grounded.
                prompt = (
                    build_answer_prompt(question.question, evidence)
                    + f"\n\nComputed from graph: FINAL_ANSWER: {answer}"
                )
            else:
                prompt = build_answer_prompt(question.question, evidence)
            resp = self.llm.complete(ANSWER_SYSTEM, prompt)
        result.answer = (answer if answer is not None else resp.text).strip()
        result.tokens += resp.tokens
        result.trace.append(
            TraceStep(
                step_index=3, agent="graphrag", action="generate",
                rationale="Produce grounded final answer.",
                outputs_summary=result.answer[:120],
                tokens=resp.tokens, duration_s=t.elapsed,
            )
        )

        result.stopped_reason = "fixed graph->vector plan complete"
        result.meta["intent"] = pq.intent
        result.latency_s = time.perf_counter() - start
        return result

    # ------------------------------------------------------------------ #
    def _graph_answer(
        self, pq: ParsedQuestion
    ) -> tuple[Optional[str], list[Evidence], str]:
        """Return (answer_or_None, evidence, rationale) via structured traversal."""
        if pq.intent == "aggregation":
            return self._aggregation(pq)
        if pq.intent == "superlative":
            return self._superlative(pq)
        if pq.intent == "temporal":
            return self._temporal(pq)
        if pq.intent == "multi_hop":
            return self._multi_hop(pq)
        if pq.intent == "lookup":
            return self._lookup(pq)
        return None, [], "no structured strategy for this intent"

    def _games_node(self, pq: ParsedQuestion, year: Optional[int] = None):
        yr = year or pq.year
        if yr and pq.season:
            return self.g.games_by_year_season(yr, pq.season)
        return None

    def _aggregation(self, pq: ParsedQuestion):
        games = self._games_node(pq)
        if not games or pq.threshold is None:
            return None, [], "aggregation: missing games or threshold"
        events = self.g.events_in_games(games.id, sport=pq.sport)
        matches = [e for e in events if e.props.get("competitors", 0) > pq.threshold]
        ev = [_event_to_evidence(e) for e in matches]
        return (
            str(len(matches)),
            ev,
            f"aggregation: {len(events)} {pq.sport} events in {pq.year} {pq.season}, "
            f"{len(matches)} with >{pq.threshold} competitors",
        )

    def _superlative(self, pq: ParsedQuestion):
        games = self._games_node(pq)
        if not games:
            return None, [], "superlative: missing games"
        events = self.g.events_in_games(games.id, sport=pq.sport)
        if not events:
            return None, [], "superlative: no events found"
        top = max(events, key=lambda e: e.props.get("competitors", 0))
        return (
            top.props.get("title", ""),
            [_event_to_evidence(top)],
            f"superlative: max competitors among {len(events)} {pq.sport} events "
            f"= {top.props.get('competitors')}",
        )

    def _temporal(self, pq: ParsedQuestion):
        if not pq.before_year or not pq.season:
            return None, [], "temporal: missing before-year or season"
        # find the games edition strictly before `before_year`
        target = self.g.games_by_year_season(pq.before_year, pq.season)
        prev = None
        if target:
            prev = self.g.adjacent_games(target.id, "prev")
        else:
            # before_year may not be an actual edition; scan editions < year
            candidates = [
                n for n in self.g.find_nodes("Games", season=pq.season)
                if n.props.get("year", 0) < pq.before_year
            ]
            prev = max(candidates, key=lambda n: n.props.get("year", 0), default=None)
        if not prev:
            return None, [], "temporal: no prior games edition"
        # within that edition, find the event matching sport + descriptor, read gold
        events = self.g.events_in_games(prev.id, sport=pq.sport)
        best = self._match_descriptor(events, pq)
        if not best:
            return None, [], f"temporal: resolved to {prev.props.get('label')} but no matching event"
        gold = self._medal_name(best, pq.medal)
        return (
            gold,
            [_event_to_evidence(best)],
            f"temporal: {pq.season} before {pq.before_year} -> {prev.props.get('label')}; "
            f"event={best.props.get('title')}",
        )

    def _multi_hop(self, pq: ParsedQuestion):
        if not pq.venue:
            return None, [], "multi_hop: missing venue"
        events = self.g.events_at_venue(pq.venue)
        if pq.year and pq.season:
            gid = _games_id(pq.year, pq.season)
            events = [
                e for e in events
                if any(g.id == gid for g in self.g.neighbors(e.id, "PART_OF", "out"))
            ] or events
        if pq.date_text:
            events = match_event_by_date(events, pq.date_text)
        if not events:
            return None, [], f"multi_hop: no event at {pq.venue} matching date"
        best = events[0]
        gold = self._medal_name(best, pq.medal)
        return (
            gold,
            [_event_to_evidence(best)],
            f"multi_hop: {pq.venue} + {pq.date_text!r} -> {best.props.get('title')}",
        )

    def _lookup(self, pq: ParsedQuestion):
        if not pq.event_title:
            return None, [], "lookup: missing event title"
        # find Event whose title matches
        node = self._find_event_by_title(pq.event_title)
        if not node:
            return None, [], f"lookup: event title not found: {pq.event_title}"
        nations = node.props.get("nations", 0)
        return (
            str(nations),
            [_event_to_evidence(node)],
            f"lookup: {node.props.get('title')} nations={nations}",
        )

    # ---- helpers ---- #
    def _find_event_by_title(self, title: str) -> Optional[Node]:
        t = title.strip().lower().replace("–", "-")
        best = None
        for nid in getattr(self.g, "_by_type", {}).get("Event", []):  # type: ignore[attr-defined]
            n = self.g._nodes[nid]  # type: ignore[attr-defined]
            nt = n.props.get("title", "").lower().replace("–", "-")
            if nt == t:
                return n
            if best is None and t in nt:
                best = n
        return best

    def _match_descriptor(self, events: list[Node], pq: ParsedQuestion) -> Optional[Node]:
        """Pick the event whose title/event_name best matches the descriptor."""
        if not events:
            return None
        desc = (pq.event_descriptor or "").lower()
        desc_tokens = set(re.findall(r"[a-z0-9]+", desc)) if desc else set()
        if not desc_tokens:
            return events[0]
        scored = []
        for e in events:
            hay = (e.props.get("title", "") + " " + e.props.get("event_name", "")).lower()
            hay_tokens = set(re.findall(r"[a-z0-9]+", hay))
            scored.append((len(desc_tokens & hay_tokens), e))
        scored.sort(key=lambda x: -x[0])
        return scored[0][1] if scored and scored[0][0] > 0 else None

    def _medal_name(self, event: Node, medal: str) -> str:
        et = {"gold": "WON_GOLD", "silver": "WON_SILVER", "bronze": "WON_BRONZE"}[medal]
        winners = self.g.neighbors(event.id, edge_type=et, direction="out")
        return winners[0].props.get("name", "") if winners else ""
