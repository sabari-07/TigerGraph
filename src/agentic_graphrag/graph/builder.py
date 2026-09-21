"""Build the knowledge graph from parsed Document objects.

Reads the infobox fields and title of each Olympic-event document and creates:
- Event, Games, Venue, Athlete, NOC, Sport nodes
- All edges per the ontology

Also stores the full document text on the Event node (for fallback retrieval)
and builds the Games temporal chain (PREV/NEXT) from the infobox links +
sorting.
"""
from __future__ import annotations

import re
from typing import Optional

from ..models import Document
from .local_backend import LocalGraph
from .ontology import NodeType, EdgeType, MEDAL_EDGE

# Regex to extract sport from title: "Canoeing at the 2012 Summer Olympics – Men's ..."
_TITLE_RE = re.compile(r"^(.+?)\s+at the\s+(\d{4})\s+(Summer|Winter)\s+Olympics")

# The ordered GAMES temporal chain: all known editions by year+season
# The prev/next infobox fields are just year numbers, but we'll also compute
# the chain via sorting to fill any gaps.


def _games_id(year: int, season: str) -> str:
    return f"games:{year}_{season.lower()}"


def _venue_id(name: str) -> str:
    return f"venue:{name.strip().lower()}"


def _sport_id(name: str) -> str:
    return f"sport:{name.strip().lower()}"


def _athlete_id(name: str, noc: str) -> str:
    # Include NOC to disambiguate homonyms across countries.
    return f"athlete:{name.strip().lower()}_{noc.strip().lower()}"


def _noc_id(code: str) -> str:
    return f"noc:{code.strip().upper()}"


def _safe_int(val: str) -> int:
    try:
        return int(val.replace(",", "").strip())
    except (ValueError, TypeError):
        return 0


def build_graph(docs: list[Document]) -> LocalGraph:
    g = LocalGraph()
    seen_games: dict[str, tuple[int, str]] = {}  # games_id -> (year, season)

    for doc in docs:
        if not doc.is_olympic_event:
            continue

        ib = doc.infobox
        title_match = _TITLE_RE.match(doc.title)

        # ---- derive key attributes ---- #
        games_raw = ib.get("games", "")
        sport_name = title_match.group(1).strip() if title_match else ""
        year = int(title_match.group(2)) if title_match else 0
        season = title_match.group(3) if title_match else ""

        if not year or not season:
            # Fall back to the 'games' field: "2012 Summer"
            parts = games_raw.split()
            if len(parts) >= 2:
                year = _safe_int(parts[0]) or year
                season = parts[1] if parts[1] in ("Summer", "Winter") else season

        games_id = _games_id(year, season) if year and season else ""
        date_raw = ib.get("date") or ib.get("dates", "")
        competitors = _safe_int(ib.get("competitors", ""))
        nations = _safe_int(ib.get("nations", ""))

        # ---- create Event node ---- #
        g.add_node(
            doc.doc_id, NodeType.EVENT,
            title=doc.title, url=doc.url,
            event_name=ib.get("event", ""),
            competitors=competitors, nations=nations,
            year=year, season=season,
            date_raw=date_raw,
            win_value=ib.get("win_value", ""),
            sport=sport_name,
            text=doc.text,
        )

        # ---- Games node + PART_OF ---- #
        if games_id:
            # Derive the label from year+season (canonical) rather than the raw
            # infobox 'games' field, which is occasionally wrong in the source.
            canonical_label = f"{year} {season}"
            g.add_node(games_id, NodeType.GAMES, label=canonical_label, year=year, season=season)
            g.add_edge(doc.doc_id, EdgeType.PART_OF, games_id)
            seen_games[games_id] = (year, season)

        # ---- Venue node + HELD_AT ---- #
        venue_name = ib.get("venue", "")
        if venue_name:
            vid = _venue_id(venue_name)
            g.add_node(vid, NodeType.VENUE, name=venue_name)
            g.add_edge(doc.doc_id, EdgeType.HELD_AT, vid)

        # ---- Sport node + HAS_SPORT ---- #
        if sport_name:
            sid = _sport_id(sport_name)
            g.add_node(sid, NodeType.SPORT, name=sport_name)
            g.add_edge(doc.doc_id, EdgeType.HAS_SPORT, sid)

        # ---- Medal edges: Athlete + NOC ---- #
        for medal in ("gold", "silver", "bronze"):
            athlete_name = ib.get(medal, "")
            noc_code = ib.get(f"{medal}NOC", "")
            if not athlete_name:
                continue
            aid = _athlete_id(athlete_name, noc_code)
            g.add_node(aid, NodeType.ATHLETE, name=athlete_name)
            g.add_edge(doc.doc_id, MEDAL_EDGE[medal], aid)
            if noc_code:
                nid = _noc_id(noc_code)
                g.add_node(nid, NodeType.NOC, code=noc_code)
                g.add_edge(aid, EdgeType.REPRESENTS, nid)

    # ---- Build temporal chain: NEXT / PREV between Games ---- #
    _build_temporal_chain(g, seen_games)

    return g


def _build_temporal_chain(
    g: LocalGraph, seen_games: dict[str, tuple[int, str]]
) -> None:
    """Link Games nodes chronologically within each season."""
    by_season: dict[str, list[tuple[int, str]]] = {}
    for gid, (year, season) in seen_games.items():
        by_season.setdefault(season, []).append((year, gid))

    for season, entries in by_season.items():
        entries.sort()  # by year
        for i in range(len(entries) - 1):
            cur_gid = entries[i][1]
            nxt_gid = entries[i + 1][1]
            g.add_edge(cur_gid, EdgeType.NEXT, nxt_gid)
            g.add_edge(nxt_gid, EdgeType.PREV, cur_gid)
