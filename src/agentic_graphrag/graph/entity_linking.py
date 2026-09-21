"""Question parsing + entity linking.

Extracts structured slots from a natural-language question and links them to
graph entities (Games, Sport, Venue, Event). Shared by the GraphRAG pipeline
and the agent's entity-linking tool.

The parser is deterministic and pattern-based because the corpus vocabulary is
closed and the phrasings are regular. An LLM-backed fallback can refine slots
when patterns miss, but the deterministic path keeps token cost near zero for
the common cases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .base import GraphBackend, Node

SEASONS = ("Summer", "Winter")

# A closed vocabulary of sports (lower-cased) is built from the graph at
# runtime; this list is a fallback for when the graph is unavailable.
_FALLBACK_SPORTS = [
    "alpine skiing", "archery", "athletics", "badminton", "beach volleyball",
    "biathlon", "bobsleigh", "boxing", "canoeing", "cross-country skiing",
    "curling", "cycling", "diving", "equestrian", "fencing", "figure skating",
    "freestyle skiing", "golf", "gymnastics", "ice hockey", "judo", "luge",
    "marathon swimming", "modern pentathlon", "nordic combined", "rowing",
    "sailing", "shooting", "short-track speed skating", "skateboarding",
    "skeleton", "ski jumping", "snowboarding", "speed skating", "swimming",
    "synchronized swimming", "table tennis", "taekwondo", "tennis",
    "triathlon", "weightlifting", "wrestling",
]

_YEAR_SEASON_RE = re.compile(r"(\d{4})\s+(Summer|Winter)\s+Olympics", re.I)
_BEFORE_RE = re.compile(r"immediately before\s+(\d{4})", re.I)
_SEASON_RE = re.compile(r"\b(Summer|Winter)\s+Olympics", re.I)
_THRESHOLD_RE = re.compile(r"more than\s+(\d+)\s+competitors", re.I)
_VENUE_ON_RE = re.compile(r"held at\s+(.+?)\s+on\s+(.+?)(?:\s+at the|\?|$)", re.I)
_NATIONS_TITLE_RE = re.compile(r"how many nations competed in\s+(.+?)\??$", re.I)


@dataclass
class ParsedQuestion:
    raw: str
    intent: str = "unknown"  # aggregation|superlative|temporal|multi_hop|lookup|unknown
    year: Optional[int] = None
    season: Optional[str] = None
    sport: Optional[str] = None
    threshold: Optional[int] = None
    before_year: Optional[int] = None
    venue: Optional[str] = None
    date_text: Optional[str] = None
    event_title: Optional[str] = None
    medal: str = "gold"
    event_descriptor: Optional[str] = None  # e.g. "men's pole vault"
    slots: dict = field(default_factory=dict)


def _known_sports(graph: Optional[GraphBackend]) -> list[str]:
    if graph is not None and hasattr(graph, "_by_type"):
        names = [
            graph._nodes[nid].props.get("name", "").lower()
            for nid in graph._by_type.get("Sport", [])  # type: ignore[attr-defined]
        ]
        names = [n for n in names if n]
        if names:
            # longest first so "cross-country skiing" matches before "skiing"
            return sorted(set(names), key=len, reverse=True)
    return sorted(_FALLBACK_SPORTS, key=len, reverse=True)


def parse_question(question: str, graph: Optional[GraphBackend] = None) -> ParsedQuestion:
    q = question.strip()
    ql = q.lower()
    pq = ParsedQuestion(raw=q)

    # Year + season.
    ys = _YEAR_SEASON_RE.search(q)
    if ys:
        pq.year = int(ys.group(1))
        pq.season = ys.group(2).capitalize()
    else:
        s = _SEASON_RE.search(q)
        if s:
            pq.season = s.group(1).capitalize()

    # Sport (from closed vocab).
    for sport in _known_sports(graph):
        if re.search(rf"\b{re.escape(sport)}\b", ql):
            pq.sport = sport
            break

    # Threshold (aggregation).
    th = _THRESHOLD_RE.search(q)
    if th:
        pq.threshold = int(th.group(1))

    # Temporal "immediately before".
    bef = _BEFORE_RE.search(q)
    if bef:
        pq.before_year = int(bef.group(1))

    # Venue + date (multi_hop).
    vm = _VENUE_ON_RE.search(q)
    if vm:
        pq.venue = vm.group(1).strip()
        pq.date_text = vm.group(2).strip()

    # Full event title (lookup: "how many nations competed in <title>").
    nm = _NATIONS_TITLE_RE.search(q)
    if nm:
        pq.event_title = nm.group(1).strip()

    # Event descriptor for temporal (between "in the" and "event").
    desc = re.search(r"in the\s+(.+?)\s+event", ql)
    if desc:
        pq.event_descriptor = desc.group(1).strip()

    # Medal.
    for m in ("gold", "silver", "bronze"):
        if m in ql:
            pq.medal = m
            break

    # ---- classify intent ---- #
    if pq.threshold is not None and "how many" in ql:
        pq.intent = "aggregation"
    elif "highest number of competitors" in ql or "most competitors" in ql:
        pq.intent = "superlative"
    elif pq.before_year is not None:
        pq.intent = "temporal"
    elif pq.venue and pq.date_text:
        pq.intent = "multi_hop"
    elif pq.event_title and "how many nations" in ql:
        pq.intent = "lookup"
    elif "how many nations" in ql:
        pq.intent = "lookup"

    pq.slots = {
        "year": pq.year, "season": pq.season, "sport": pq.sport,
        "threshold": pq.threshold, "before_year": pq.before_year,
        "venue": pq.venue, "date_text": pq.date_text,
        "event_title": pq.event_title, "medal": pq.medal,
    }
    return pq


# --------------------------------------------------------------------------- #
# Date matching for multi_hop (venue + date -> single event)
# --------------------------------------------------------------------------- #
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


def _date_tokens(text: str) -> set[str]:
    """Normalize a date string into comparable tokens (day, month name, year).

    Handles the many corpus formats: "13 February 2010", "August 21-22",
    "6 to 8 August", "February 22, 1992". We compare token overlap rather than
    parse strictly, which is robust to ranges and ordering.
    """
    if not text:
        return set()
    t = text.lower()
    toks: set[str] = set()
    for mname, _ in _MONTHS.items():
        if mname in t:
            toks.add(mname)
    for num in re.findall(r"\d+", t):
        toks.add(num)
    return toks


def match_event_by_date(events: list[Node], date_text: str) -> list[Node]:
    """Filter candidate events (same venue) to those matching the date."""
    want = _date_tokens(date_text)
    if not want:
        return events
    scored: list[tuple[int, Node]] = []
    for e in events:
        have = _date_tokens(e.props.get("date_raw", ""))
        overlap = len(want & have)
        if overlap:
            scored.append((overlap, e))
    if not scored:
        return events
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0]
    return [e for score, e in scored if score == best]
