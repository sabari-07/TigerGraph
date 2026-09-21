"""Specialized agents, exposed as tools the orchestrator can call.

Each tool is a focused capability from the hackathon spec:
  entity_linking, graph_traversal, similarity_search, document_retrieval,
  aggregation, multi_hop_reasoning, evidence_evaluation.

Every tool takes the shared AgentState and returns a ToolResult carrying new
evidence, resolved facts, an optional proposed answer, and a rationale that
feeds the explainability trace. Tools never call the LLM directly (except the
optional evidence evaluator); they operate on graph + vector primitives so the
agent's reasoning is grounded and cheap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..graph.base import GraphBackend, Node
from ..graph.builder import _games_id
from ..graph.entity_linking import match_event_by_date
from ..models import Evidence
from ..retrieval.vector_store import VectorStore
from .state import AgentState


@dataclass
class ToolResult:
    tool: str
    rationale: str
    evidence: list[Evidence] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    proposed_answer: Optional[str] = None
    confidence: float = 0.0
    candidates: Optional[list[Node]] = None
    outputs_summary: str = ""


def _event_evidence(n: Node, source: str = "graph") -> Evidence:
    parts = []
    for k in ("event_name", "year", "season", "competitors", "nations", "date_raw"):
        v = n.props.get(k)
        if v:
            parts.append(f"{k}={v}")
    return Evidence(
        doc_id=n.id, title=n.props.get("title", ""),
        snippet="; ".join(parts), score=1.0, source=source,
        url=n.props.get("url", ""),
    )


def _medal_winner(g: GraphBackend, event: Node, medal: str) -> str:
    et = {"gold": "WON_GOLD", "silver": "WON_SILVER", "bronze": "WON_BRONZE"}[medal]
    winners = g.neighbors(event.id, edge_type=et, direction="out")
    return winners[0].props.get("name", "") if winners else ""


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
class Tools:
    def __init__(self, graph: GraphBackend, vector_store: VectorStore,
                 fact_store: "Optional[object]" = None, embedder: "Optional[object]" = None) -> None:
        self.g = graph
        self.vs = vector_store
        # Optional bi-temporal fact store for Round 2 conflict reasoning.
        self.fact_store = fact_store
        # Optional embedder to query TigerGraph Vector DB (native path).
        self.embedder = embedder

    # ---- entity linking ---- #
    def entity_linking(self, s: AgentState) -> ToolResult:
        pq = s.parsed
        facts: dict[str, Any] = {}
        rationale_bits = []
        if pq.year and pq.season:
            games = self.g.games_by_year_season(pq.year, pq.season)
            if games:
                facts["games_id"] = games.id
                rationale_bits.append(f"linked games={games.props.get('label')}")
        if pq.sport:
            facts["sport"] = pq.sport
            rationale_bits.append(f"sport={pq.sport}")
        if pq.venue:
            facts["venue"] = pq.venue
            rationale_bits.append(f"venue={pq.venue}")
        return ToolResult(
            tool="entity_linking",
            rationale="Resolve question slots to graph entities. " + "; ".join(rationale_bits),
            facts=facts,
            outputs_summary=str(facts),
        )

    # ---- graph traversal (generic neighborhood) ---- #
    def graph_traversal(self, s: AgentState) -> ToolResult:
        pq = s.parsed
        games_id = s.facts.get("games_id")
        if not games_id and pq.year and pq.season:
            g = self.g.games_by_year_season(pq.year, pq.season)
            games_id = g.id if g else None
        if not games_id:
            return ToolResult("graph_traversal", "no games edition to traverse", confidence=0.0)
        events = self.g.events_in_games(games_id, sport=pq.sport)
        s.candidates = events
        return ToolResult(
            tool="graph_traversal",
            rationale=f"Traverse events PART_OF {games_id}"
            + (f" filtered by sport={pq.sport}" if pq.sport else ""),
            candidates=events,
            evidence=[_event_evidence(e) for e in events[:10]],
            outputs_summary=f"{len(events)} events",
        )

    # ---- similarity search ---- #
    def similarity_search(self, s: AgentState, k: int = 5) -> ToolResult:
        # Prefer TigerGraph Vector DB (native) when the backend + embedder allow.
        if self.embedder is not None and hasattr(self.g, "vector_search"):
            try:
                qv = self.embedder.encode([s.question])[0]
                nodes = self.g.vector_search([float(x) for x in qv], k=k)  # type: ignore[attr-defined]
                if nodes:
                    ev = [_event_evidence(n, source="tg_vector") for n in nodes]
                    return ToolResult(
                        tool="similarity_search",
                        rationale=f"TigerGraph Vector DB top-{k} search over Event.emb.",
                        evidence=ev,
                        outputs_summary=f"{len(ev)} events (TigerGraph vector)",
                    )
            except Exception:
                pass  # fall back to local vector store
        ev = self.vs.search(s.question, k=k)
        return ToolResult(
            tool="similarity_search",
            rationale=f"Vector top-{k} retrieval for supporting context.",
            evidence=ev,
            outputs_summary=f"{len(ev)} chunks",
        )

    # ---- document retrieval (fetch a specific doc's full text) ---- #
    def document_retrieval(self, s: AgentState, doc_id: str) -> ToolResult:
        n = self.g.get_node(doc_id)
        if not n:
            return ToolResult("document_retrieval", f"doc {doc_id} not found")
        text = n.props.get("text", "")
        return ToolResult(
            tool="document_retrieval",
            rationale=f"Fetch full document {doc_id} to confirm details.",
            evidence=[Evidence(doc_id=doc_id, title=n.props.get("title", ""),
                               snippet=text[:800], score=1.0, source="document")],
            outputs_summary=f"fetched {doc_id} ({len(text)} chars)",
        )

    # ---- document retrieval for several candidates ---- #
    def document_retrieval_multi(self, s: AgentState, limit: int = 4) -> ToolResult:
        """Fetch the full documents for the remaining candidate events.

        Used to resolve genuine venue/date collisions: the medal results in the
        document text let the LLM (or evidence evaluator) pick the correct
        event. Attaches each candidate's gold as a hint for grounding.
        """
        cands = s.candidates[:limit]
        if not cands:
            return ToolResult("document_retrieval_multi", "no candidates to fetch", confidence=0.0)
        ev: list[Evidence] = []
        for e in cands:
            text = e.props.get("text", "")
            gold = _medal_winner(self.g, e, s.parsed.medal)
            ev.append(Evidence(
                doc_id=e.id, title=e.props.get("title", ""),
                snippet=f"[{s.parsed.medal} medalist: {gold}] " + text[:600],
                score=1.0, source="document",
            ))
        return ToolResult(
            tool="document_retrieval_multi",
            rationale=f"Fetched {len(ev)} candidate documents to ground the final choice.",
            evidence=ev,
            # leave answer to the LLM synthesis over these docs
            confidence=0.5,
            outputs_summary=f"{len(ev)} docs",
        )

    # ---- aggregation ---- #
    def aggregation(self, s: AgentState) -> ToolResult:
        pq = s.parsed
        games_id = s.facts.get("games_id")
        if not games_id or pq.threshold is None:
            return ToolResult("aggregation", "missing games or threshold", confidence=0.0)
        events = self.g.events_in_games(games_id, sport=pq.sport)
        matches = [e for e in events if e.props.get("competitors", 0) > pq.threshold]
        return ToolResult(
            tool="aggregation",
            rationale=f"Count {pq.sport} events with >{pq.threshold} competitors "
            f"among {len(events)} events.",
            evidence=[_event_evidence(e) for e in matches],
            proposed_answer=str(len(matches)),
            confidence=1.0,
            outputs_summary=f"{len(matches)}/{len(events)} over threshold",
        )

    # ---- superlative ---- #
    def superlative(self, s: AgentState) -> ToolResult:
        pq = s.parsed
        games_id = s.facts.get("games_id")
        if not games_id:
            return ToolResult("superlative", "missing games", confidence=0.0)
        events = self.g.events_in_games(games_id, sport=pq.sport)
        if not events:
            return ToolResult("superlative", "no events found", confidence=0.0)
        top = max(events, key=lambda e: e.props.get("competitors", 0))
        return ToolResult(
            tool="superlative",
            rationale=f"Select max-competitor event among {len(events)} {pq.sport} events "
            f"(={top.props.get('competitors')}).",
            evidence=[_event_evidence(top)],
            proposed_answer=top.props.get("title", ""),
            confidence=1.0,
            outputs_summary=top.props.get("title", ""),
        )

    # ---- temporal reasoning ---- #
    def temporal(self, s: AgentState) -> ToolResult:
        pq = s.parsed
        if not pq.before_year or not pq.season:
            return ToolResult("temporal", "missing before-year or season", confidence=0.0)
        target = self.g.games_by_year_season(pq.before_year, pq.season)
        prev = self.g.adjacent_games(target.id, "prev") if target else None
        if not prev:
            candidates = [
                n for n in self.g.find_nodes("Games", season=pq.season)
                if n.props.get("year", 0) < pq.before_year
            ]
            prev = max(candidates, key=lambda n: n.props.get("year", 0), default=None)
        if not prev:
            return ToolResult("temporal", "no prior games edition", confidence=0.0)
        s.facts["games_id"] = prev.id
        events = self.g.events_in_games(prev.id, sport=pq.sport)
        best = _match_descriptor(events, pq.event_descriptor)
        if not best:
            return ToolResult(
                "temporal",
                f"resolved to {prev.props.get('label')} but event descriptor unmatched",
                candidates=events, confidence=0.3,
            )
        winner = _medal_winner(self.g, best, pq.medal)
        return ToolResult(
            tool="temporal",
            rationale=f"{pq.season} before {pq.before_year} -> {prev.props.get('label')}; "
            f"matched event '{best.props.get('title')}'.",
            evidence=[_event_evidence(best)],
            proposed_answer=winner,
            confidence=0.9 if winner else 0.4,
            outputs_summary=winner,
        )

    # ---- multi-hop reasoning (venue + date -> event -> medalist) ---- #
    def multi_hop(self, s: AgentState) -> ToolResult:
        pq = s.parsed
        if not pq.venue:
            return ToolResult("multi_hop", "missing venue", confidence=0.0)
        events = self.g.events_at_venue(pq.venue)
        note = f"{len(events)} events at {pq.venue}"
        # scope by games edition if present
        if pq.year and pq.season:
            gid = _games_id(pq.year, pq.season)
            scoped = [
                e for e in events
                if any(g.id == gid for g in self.g.neighbors(e.id, "PART_OF", "out"))
            ]
            if scoped:
                events = scoped
                note += f"; scoped to {pq.year} {pq.season} -> {len(events)}"
        # date filtering with disambiguation
        if pq.date_text:
            dated = match_event_by_date(events, pq.date_text)
            note += f"; date-match -> {len(dated)}"
            events = dated
        s.candidates = events
        if not events:
            return ToolResult("multi_hop", note + "; no candidate", confidence=0.0)
        if len(events) == 1:
            best = events[0]
            winner = _medal_winner(self.g, best, pq.medal)
            return ToolResult(
                tool="multi_hop",
                rationale=note + f"; unique event '{best.props.get('title')}'.",
                evidence=[_event_evidence(best)],
                proposed_answer=winner, confidence=0.95, outputs_summary=winner,
            )
        # ambiguous: leave for disambiguation tool
        return ToolResult(
            tool="multi_hop",
            rationale=note + f"; AMBIGUOUS ({len(events)} candidates) - needs disambiguation.",
            evidence=[_event_evidence(e) for e in events],
            candidates=events, confidence=0.4,
            outputs_summary=f"{len(events)} candidates",
        )

    # ---- disambiguation (the agentic edge over fixed GraphRAG) ---- #
    def disambiguate_by_date(self, s: AgentState) -> ToolResult:
        """Prefer an exact single-day date match over multi-day ranges.

        This resolves venue+date collisions where several events share date
        tokens but only one occurred on the exact day named in the question.
        """
        pq = s.parsed
        cands = s.candidates
        if not cands or not pq.date_text:
            return ToolResult("disambiguate_by_date", "nothing to disambiguate", confidence=0.0)

        # 1) Filter by sport if the question named one (a biathlon question
        #    should not resolve to a cross-country event at the same venue/date).
        if pq.sport:
            by_sport = [e for e in cands if (e.props.get("sport", "").lower() == pq.sport)]
            if len(by_sport) == 1:
                best = by_sport[0]
                winner = _medal_winner(self.g, best, pq.medal)
                return ToolResult(
                    tool="disambiguate_by_date",
                    rationale=f"Filtered {len(cands)} candidates to the one matching "
                    f"sport={pq.sport}: '{best.props.get('title')}'.",
                    evidence=[_event_evidence(best)],
                    proposed_answer=winner, confidence=0.9, outputs_summary=winner,
                )
            if by_sport:
                cands = by_sport

        # 2) Prefer the candidate held on exactly the single day named, with no
        #    other phases/dates (the finals-on-that-day event). We count how many
        #    distinct calendar days each candidate's date spans.
        want_day = _single_day_key(pq.date_text)
        pure = []
        for e in cands:
            raw = e.props.get("date_raw", "")
            days = _distinct_days(raw)
            if want_day and want_day in days and len(days) == 1:
                pure.append(e)
        if len(pure) == 1:
            best = pure[0]
            winner = _medal_winner(self.g, best, pq.medal)
            return ToolResult(
                tool="disambiguate_by_date",
                rationale=f"Selected the event occurring solely on {pq.date_text} "
                f"('{best.props.get('date_raw')}') among {len(s.candidates)} candidates.",
                evidence=[_event_evidence(best)],
                proposed_answer=winner, confidence=0.9, outputs_summary=winner,
            )

        # 3) Fallback: single non-range candidate.
        single = [e for e in cands if not _is_range(e.props.get("date_raw", ""))]
        if len(single) == 1:
            best = single[0]
            winner = _medal_winner(self.g, best, pq.medal)
            return ToolResult(
                tool="disambiguate_by_date",
                rationale=f"Single-day event preferred over range events "
                f"({len(s.candidates)} candidates).",
                evidence=[_event_evidence(best)],
                proposed_answer=winner, confidence=0.75, outputs_summary=winner,
            )
        return ToolResult(
            "disambiguate_by_date",
            f"could not uniquely disambiguate {len(cands)} candidates",
            candidates=cands, confidence=0.3,
        )

    # ---- conflict resolution (Round 2: evolving/conflicting facts) ---- #
    def conflict_resolution(self, s: AgentState, subject: str, predicate: str,
                            as_of: "Optional[Any]" = None) -> ToolResult:
        """Resolve conflicting/evolving assertions about (subject, predicate).

        Uses the bi-temporal fact store to pick the surviving value by
        supersession + source authority, reporting uncertainty when sources
        are close. This is the Round 2 capability.
        """
        if self.fact_store is None:
            return ToolResult("conflict_resolution", "no fact store configured", confidence=0.0)
        res = self.fact_store.resolve(subject, predicate, as_of=as_of)
        if res is None:
            return ToolResult("conflict_resolution",
                              f"no assertions for ({subject},{predicate})", confidence=0.0)
        ev = [Evidence(doc_id=f.doc_id or f.source, title=f"{f.source} ({f.source_type})",
                       snippet=f"{predicate}={f.object}", score=f.authority(),
                       source="fact_store")
              for f in ([res.winning] + res.conflicts)]
        return ToolResult(
            tool="conflict_resolution",
            rationale=res.rationale,
            evidence=ev,
            proposed_answer=res.value,
            confidence=res.confidence,
            outputs_summary=f"{res.value} (conf {res.confidence:.2f}, "
            f"{len(res.conflicts)} conflicts, {len(res.superseded)} superseded)",
        )

    # ---- last-resort commit (avoid 'unknown' on true ambiguity) ---- #
    def commit_best_candidate(self, s: AgentState) -> ToolResult:
        """Commit to the top remaining candidate's medalist.

        For genuinely unresolvable ties (e.g. two events on the exact same day
        at one venue with no sport named), a grounded best-guess scores better
        in expectation than returning 'unknown'.
        """
        if not s.candidates:
            return ToolResult("commit_best_candidate", "no candidates to commit to", confidence=0.0)
        best = s.candidates[0]
        winner = _medal_winner(self.g, best, s.parsed.medal)
        if not winner:
            return ToolResult("commit_best_candidate", "top candidate has no medalist", confidence=0.0)
        return ToolResult(
            tool="commit_best_candidate",
            rationale=f"Ambiguity unresolved among {len(s.candidates)} candidates; "
            f"committing to '{best.props.get('title')}' -> {winner}.",
            evidence=[_event_evidence(best)],
            proposed_answer=winner, confidence=0.6, outputs_summary=winner,
        )

    # ---- lookup ---- #
    def lookup(self, s: AgentState) -> ToolResult:
        pq = s.parsed
        if not pq.event_title:
            return ToolResult("lookup", "missing event title", confidence=0.0)
        node = _find_event_by_title(self.g, pq.event_title)
        if not node:
            return ToolResult("lookup", f"event title not found: {pq.event_title}", confidence=0.0)
        nations = node.props.get("nations", 0)
        return ToolResult(
            tool="lookup",
            rationale=f"Read 'nations' from event '{node.props.get('title')}' = {nations}.",
            evidence=[_event_evidence(node)],
            proposed_answer=str(nations), confidence=1.0, outputs_summary=str(nations),
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _find_event_by_title(g: GraphBackend, title: str) -> Optional[Node]:
    # Prefer a backend-provided method (works for both local and TigerGraph).
    if hasattr(g, "find_event_by_title"):
        return g.find_event_by_title(title)  # type: ignore[attr-defined]
    # Fallback for local graph internals.
    t = title.strip().lower().replace("–", "-")
    best = None
    for nid in getattr(g, "_by_type", {}).get("Event", []):  # type: ignore[attr-defined]
        n = g._nodes[nid]  # type: ignore[attr-defined]
        nt = n.props.get("title", "").lower().replace("–", "-")
        if nt == t:
            return n
        if best is None and t in nt:
            best = n
    return best


def _canon_event_phrase(text: str) -> str:
    """Canonicalize an event phrase for exact comparison.

    Keeps distinguishing markers like '+' (as in '+80 kg') and gender, drops
    the sport name and filler so "men's 80 kg taekwondo" -> "mens 80 kg".
    """
    t = text.lower()
    # normalize weight like "+80 kg" -> "plus80kg", "80 kg" -> "80kg"
    t = re.sub(r"\+\s*(\d+)\s*kg", r"plus\1kg", t)
    t = re.sub(r"(\d+)\s*kg", r"\1kg", t)
    # drop sport names / filler words
    for w in ("event", "olympics", "summer", "winter", "the", "at"):
        t = re.sub(rf"\b{w}\b", " ", t)
    t = re.sub(r"[^a-z0-9+ ]", " ", t)
    return " ".join(t.split())


def _match_descriptor(events: list[Node], descriptor: Optional[str]) -> Optional[Node]:
    if not events:
        return None
    if not descriptor:
        return events[0]
    want = _canon_event_phrase(descriptor)
    want_tokens = set(want.split())

    scored = []
    for e in events:
        hay_raw = e.props.get("event_name", "") + " " + e.props.get("title", "")
        hay = _canon_event_phrase(hay_raw)
        hay_tokens = set(hay.split())
        # Exact weight token must match if present (guards 80kg vs plus80kg vs 68kg).
        weight = next((w for w in want_tokens if w.endswith("kg")), None)
        if weight and weight not in hay_tokens:
            score = -1
        else:
            score = len(want_tokens & hay_tokens)
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1] if scored and scored[0][0] > 0 else None


_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")


def _is_range(date_raw: str) -> bool:
    t = date_raw.lower()
    return (" to " in t) or ("–" in t) or ("-" in t and any(m in t for m in _MONTHS))


def _single_day_key(date_text: str) -> str:
    """Return 'D month' for a clean single-day date string, else ''."""
    t = date_text.lower().replace(",", "")
    month = next((m for m in _MONTHS if m in t), "")
    nums = re.findall(r"\d+", t)
    day = next((n for n in nums if 1 <= int(n) <= 31 and len(n) <= 2), "")
    if month and day:
        return f"{int(day)} {month}"
    return ""


def _distinct_days(date_raw: str) -> set[str]:
    """Extract the set of distinct 'D month' calendar days a date string spans.

    Handles phase annotations like '15 August 2008 (heats)16 August 2008 (final)'
    and ranges like '6 to 8 August'. A pure single-day event yields one entry.
    """
    t = date_raw.lower().replace(",", "")
    # Corpus dates sometimes concatenate two dates with no separator, gluing a
    # 4-digit year to the next day, e.g. "5 August 20126 August 2012". Split a
    # run of >=5 digits back into a 4-digit year + following day.
    t = re.sub(r"(\d{4})(\d{1,2})(?=\s+(?:january|february|march|april|may|june|"
               r"july|august|september|october|november|december))", r"\1 \2", t)
    days: set[str] = set()
    _MON = (r"(january|february|march|april|may|june|july|"
            r"august|september|october|november|december)")
    # Pattern A: explicit "<day> <month>" pairs (day not part of a longer number).
    for m in re.finditer(rf"(?<!\d)(\d{{1,2}})\s+{_MON}", t):
        days.add(f"{int(m.group(1))} {m.group(2)}")
    # Pattern B: "<month> <day>" pairs (day not followed by more digits => not a year).
    for m in re.finditer(rf"{_MON}\s+(\d{{1,2}})(?!\d)", t):
        days.add(f"{int(m.group(2))} {m.group(1)}")
    # Pattern C: numeric range like "6 to 8 august" or "23-26 august".
    rng = re.search(rf"(?<!\d)(\d{{1,2}})\s*(?:to|–|-)\s*(\d{{1,2}})\s+{_MON}", t)
    if rng:
        month = rng.group(3)
        for d in range(int(rng.group(1)), int(rng.group(2)) + 1):
            days.add(f"{d} {month}")
    return days


def _norm_day(date_raw: str) -> str:
    """Normalize a single-day date to 'D month YYYY' if possible, else ''."""
    if not date_raw or _is_range(date_raw):
        # A range has no single day; return empty so exact-match fails for ranges.
        if _is_range(date_raw):
            return ""
    t = date_raw.lower().replace(",", "")
    month = next((m for m in _MONTHS if m in t), "")
    nums = re.findall(r"\d+", t)
    day = next((n for n in nums if 1 <= int(n) <= 31 and len(n) <= 2), "")
    year = next((n for n in nums if len(n) == 4), "")
    if month and day:
        return f"{int(day)} {month} {year}".strip()
    return ""
