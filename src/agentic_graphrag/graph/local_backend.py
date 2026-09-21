"""In-memory graph backend. Zero external dependencies; fast for dev + CI.

Mirrors the TigerGraph schema and traversal surface exactly, so switching
GRAPH_BACKEND between 'local' and 'tigergraph' does not change pipeline code.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from .base import GraphBackend, Node


class LocalGraph(GraphBackend):
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        # adjacency: node_id -> edge_type -> list[node_id]
        self._out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._in: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        # secondary indexes
        self._by_type: dict[str, list[str]] = defaultdict(list)

    # ----- build API ----- #
    def add_node(self, node_id: str, node_type: str, **props: Any) -> None:
        if node_id not in self._nodes:
            self._nodes[node_id] = Node(id=node_id, type=node_type, props=dict(props))
            self._by_type[node_type].append(node_id)
        else:
            # merge props (first non-empty wins for stability)
            existing = self._nodes[node_id].props
            for k, v in props.items():
                if k not in existing or existing[k] in (None, "", 0):
                    existing[k] = v

    def add_edge(self, src: str, edge_type: str, dst: str) -> None:
        if dst not in self._out[src][edge_type]:
            self._out[src][edge_type].append(dst)
        if src not in self._in[dst][edge_type]:
            self._in[dst][edge_type].append(src)

    # ----- query API ----- #
    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def neighbors(
        self, node_id: str, edge_type: Optional[str] = None, direction: str = "out"
    ) -> list[Node]:
        out: list[Node] = []
        maps = []
        if direction in ("out", "any"):
            maps.append(self._out.get(node_id, {}))
        if direction in ("in", "any"):
            maps.append(self._in.get(node_id, {}))
        for m in maps:
            etypes = [edge_type] if edge_type else list(m.keys())
            for et in etypes:
                for nid in m.get(et, []):
                    n = self._nodes.get(nid)
                    if n:
                        out.append(n)
        return out

    def find_nodes(self, node_type: str, **prop_filters: Any) -> list[Node]:
        result: list[Node] = []
        for nid in self._by_type.get(node_type, []):
            n = self._nodes[nid]
            if all(n.props.get(k) == v for k, v in prop_filters.items()):
                result.append(n)
        return result

    def events_in_games(self, games_id: str, sport: Optional[str] = None) -> list[Node]:
        events = self.neighbors(games_id, edge_type="PART_OF", direction="in")
        if sport:
            s = sport.strip().lower()
            events = [
                e for e in events
                if any(sp.props.get("name", "").lower() == s
                       for sp in self.neighbors(e.id, "HAS_SPORT", "out"))
            ]
        return events

    def games_by_year_season(self, year: int, season: str) -> Optional[Node]:
        matches = self.find_nodes("Games", year=year, season=season)
        return matches[0] if matches else None

    def adjacent_games(self, games_id: str, direction: str) -> Optional[Node]:
        et = "NEXT" if direction == "next" else "PREV"
        ns = self.neighbors(games_id, edge_type=et, direction="out")
        return ns[0] if ns else None

    def find_event_by_title(self, title: str) -> Optional[Node]:
        t = title.strip().lower().replace("\u2013", "-")
        best = None
        for nid in self._by_type.get("Event", []):
            n = self._nodes[nid]
            nt = n.props.get("title", "").lower().replace("\u2013", "-")
            if nt == t:
                return n
            if best is None and t in nt:
                best = n
        return best

    def events_at_venue(self, venue_name: str) -> list[Node]:
        v = venue_name.strip().lower()
        venue_ids = [
            nid for nid in self._by_type.get("Venue", [])
            if self._nodes[nid].props.get("name", "").lower() == v
        ]
        events: list[Node] = []
        for vid in venue_ids:
            events.extend(self.neighbors(vid, edge_type="HELD_AT", direction="in"))
        return events

    def stats(self) -> dict[str, int]:
        s = {t: len(ids) for t, ids in self._by_type.items()}
        s["_edges"] = sum(
            len(dsts) for em in self._out.values() for dsts in em.values()
        )
        return s
