"""Robust fallback for question parsing when the fast regex path is weak.

The primary `parse_question` uses tight regexes tuned to the corpus's regular
phrasing. That is fast and token-free, but brittle to rewording. This module
recovers gracefully when the regex parse returns `intent=unknown` or leaves key
slots empty, using three cheap, non-brittle strategies:

  1. Intent inference from semantic cues (verb/keyword families), not exact
     template strings: "how many ... more than N" family -> aggregation,
     "most/largest/biggest field" -> superlative, "before/prior/preceding" ->
     temporal, "who won ... at <venue>" -> multi_hop, attribute questions ->
     lookup.
  2. Graph-grounded slot resolution: match the sport against the ACTUAL sport
     nodes (fuzzy), and detect a venue by scanning the question for any known
     venue name. This works even when wording differs from the templates.
  3. Optional LLM slot extraction: when an LLM is available and cues are still
     ambiguous, ask it for {intent, year, season, sport, venue, date,
     threshold, before_year} as JSON. Kept off the hot path (only on miss).

The result is that reworded questions still route to the right tools instead of
silently falling back to weak vector search.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .base import GraphBackend
from .entity_linking import ParsedQuestion, parse_question

# Semantic cue families (substring match, order-independent).
_AGG_CUES = ("how many", "number of", "count of")
_MORE_THAN = re.compile(r"(?:more than|over|above|at least|greater than|exceed\w*)\s+(\d+)", re.I)
_SUP_CUES = ("most", "highest", "largest", "biggest", "greatest", "maximum", "top ")
_TMP_CUES = ("before", "prior to", "preceding", "previous", "earlier than", "ahead of")
_WON_CUES = ("who won", "gold medal", "medal", "winner", "champion")
_LOOKUP_ATTR = ("how many nations", "number of nations", "how many countries",
                "how many competitors", "number of competitors")

_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")
_DATE_NEAR = re.compile(
    r"\bon\s+(.+?)(?:\?|$|\s+at the)", re.I
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _needs_fallback(pq: ParsedQuestion) -> bool:
    if pq.intent == "unknown":
        return True
    # Intent found but a slot the intent depends on is missing.
    if pq.intent == "aggregation" and (pq.threshold is None or not pq.sport):
        return True
    if pq.intent == "superlative" and not pq.sport:
        return True
    if pq.intent == "temporal" and not pq.before_year:
        return True
    if pq.intent == "multi_hop" and not pq.venue:
        return True
    if pq.intent == "lookup" and not pq.event_title:
        return True
    return False


def _known_sports(graph: Optional[GraphBackend]) -> list[str]:
    if graph is not None and hasattr(graph, "_by_type"):
        names = [graph._nodes[nid].props.get("name", "").lower()
                 for nid in graph._by_type.get("Sport", [])]  # type: ignore[attr-defined]
        return sorted({n for n in names if n}, key=len, reverse=True)
    return []


def _known_venues(graph: Optional[GraphBackend]) -> list[str]:
    if graph is not None and hasattr(graph, "_by_type"):
        names = [graph._nodes[nid].props.get("name", "")
                 for nid in graph._by_type.get("Venue", [])]  # type: ignore[attr-defined]
        return sorted({n for n in names if n}, key=len, reverse=True)
    return []


def _infer_intent(ql: str) -> str:
    if any(c in ql for c in _AGG_CUES) and _MORE_THAN.search(ql):
        return "aggregation"
    if any(c in ql for c in _SUP_CUES) and ("competitor" in ql or "field" in ql or "event" in ql):
        return "superlative"
    if any(c in ql for c in _TMP_CUES) and _YEAR_RE.search(ql):
        return "temporal"
    if any(c in ql for c in _LOOKUP_ATTR):
        return "lookup"
    if any(c in ql for c in _WON_CUES):
        return "multi_hop"
    return "unknown"


def enrich_parse(
    question: str, graph: Optional[GraphBackend] = None, llm: object = None
) -> ParsedQuestion:
    """Parse with the fast path, then repair weak results using fallbacks."""
    pq = parse_question(question, graph)
    if not _needs_fallback(pq):
        return pq

    ql = question.lower()

    # --- 2a: intent from semantic cues ---
    if pq.intent == "unknown":
        pq.intent = _infer_intent(ql)

    # --- 2b: graph-grounded sport ---
    if not pq.sport:
        for sport in _known_sports(graph):
            if sport in ql:
                pq.sport = sport
                break

    # --- threshold via broader phrasing ---
    if pq.threshold is None:
        m = _MORE_THAN.search(ql)
        if m:
            pq.threshold = int(m.group(1))

    # --- before-year via broader temporal phrasing ---
    if pq.before_year is None and pq.intent == "temporal":
        # "the Games before 2016", "prior to 2016"
        yrs = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", question)]
        if yrs:
            pq.before_year = yrs[-1]

    # --- venue by scanning known venue names ---
    if not pq.venue and pq.intent == "multi_hop":
        for venue in _known_venues(graph):
            if venue.lower() in ql:
                pq.venue = venue
                break
        # date near "on ..."
        if pq.venue and not pq.date_text:
            dm = _DATE_NEAR.search(question)
            if dm:
                pq.date_text = dm.group(1).strip()

    # --- event descriptor for temporal (robust to phrasing) ---
    if pq.intent == "temporal" and not pq.event_descriptor:
        # Grab the phrase between a lead-in ("in the"/"in"/"for the") and "event"/"at".
        m = re.search(r"(?:in the|in|for the|for)\s+(.+?)\s+(?:event\b|at the|at )", ql)
        if not m:
            m = re.search(r"gold(?:\s+medal)?\s+in\s+(.+?)\s+(?:at|before|preceding)", ql)
        if m:
            pq.event_descriptor = m.group(1).strip()

    # --- lookup event title (robust: nations OR countries; "competed in <title>") ---
    if pq.intent == "lookup" and not pq.event_title:
        m = re.search(r"(?:competed in|competing in|were in)\s+(.+?)\??$", question, re.I)
        if m:
            pq.event_title = m.group(1).strip()

    # --- year/season if still missing but present in text ---
    if not pq.year:
        ym = _YEAR_RE.search(question)
        if ym:
            pq.year = int(ym.group(0))
    if not pq.season:
        if "winter" in ql:
            pq.season = "Winter"
        elif "summer" in ql:
            pq.season = "Summer"

    # --- 2c: LLM slot extraction as a last resort ---
    if llm is not None and _needs_fallback(pq):
        _llm_fill(question, pq, llm)

    # refresh slots dict
    pq.slots = {
        "year": pq.year, "season": pq.season, "sport": pq.sport,
        "threshold": pq.threshold, "before_year": pq.before_year,
        "venue": pq.venue, "date_text": pq.date_text,
        "event_title": pq.event_title, "medal": pq.medal,
    }
    return pq


_LLM_SLOT_SYSTEM = (
    "Extract structured slots from an Olympic question. Respond ONLY with JSON: "
    '{"intent": "aggregation|superlative|temporal|multi_hop|lookup", '
    '"year": int|null, "season": "Summer|Winter|null", "sport": str|null, '
    '"threshold": int|null, "before_year": int|null, "venue": str|null, '
    '"date": str|null, "event_title": str|null}.'
)


def _llm_fill(question: str, pq: ParsedQuestion, llm: object) -> None:
    try:
        resp = llm.complete(_LLM_SLOT_SYSTEM, f"Question: {question}", max_tokens=160)
        m = re.search(r"\{.*\}", resp.text, re.S)
        if not m:
            return
        data = json.loads(m.group(0))
    except Exception:
        return
    pq.intent = data.get("intent") or pq.intent
    pq.year = pq.year or data.get("year")
    pq.season = pq.season or (data.get("season") if data.get("season") in ("Summer", "Winter") else None)
    pq.sport = pq.sport or (data.get("sport") or None)
    pq.threshold = pq.threshold if pq.threshold is not None else data.get("threshold")
    pq.before_year = pq.before_year or data.get("before_year")
    pq.venue = pq.venue or (data.get("venue") or None)
    pq.date_text = pq.date_text or (data.get("date") or None)
    pq.event_title = pq.event_title or (data.get("event_title") or None)
