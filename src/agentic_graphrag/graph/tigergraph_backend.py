"""TigerGraph Savanna backend implementing the GraphBackend interface.

Mirrors the LocalGraph query surface using GSQL / the pyTigerGraph REST client,
so pipelines and agents run unchanged against Savanna by setting
GRAPH_BACKEND=tigergraph.

The traversals map to the ontology in ontology.py:
  events_in_games     -> SELECT Event -(PART_OF)-> Games, filter by HAS_SPORT
  games_by_year_season-> lookup Games vertex by (year, season)
  adjacent_games      -> SELECT along NEXT / PREV edges
  events_at_venue     -> SELECT Event -(HELD_AT)-> Venue

Requires TG_* env vars. If pyTigerGraph is unavailable or credentials are
missing, construction raises a clear error and the caller should fall back to
the local backend (see graph.factory.build_backend).
"""
from __future__ import annotations

from typing import Any, Optional

from ..config import TigerGraphConfig
from .base import GraphBackend, Node


class TigerGraphBackend(GraphBackend):
    def __init__(self, cfg: TigerGraphConfig) -> None:
        try:
            import pyTigerGraph as tg
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "pyTigerGraph is not installed. `pip install pyTigerGraph` "
                "or set GRAPH_BACKEND=local."
            ) from exc

        if not cfg.host:
            raise RuntimeError("TG_HOST is not set; cannot connect to TigerGraph.")

        self.cfg = cfg
        # Savanna/Cloud auth: connect with the gsqlSecret, then getToken().
        # (username/password are not needed for Cloud instances.)
        if cfg.secret:
            self.conn = tg.TigerGraphConnection(
                host=cfg.host,
                graphname=cfg.graph_name,
                gsqlSecret=cfg.secret,
            )
            self.conn.getToken(cfg.secret)
        else:
            self.conn = tg.TigerGraphConnection(
                host=cfg.host,
                graphname=cfg.graph_name,
                username=cfg.username,
                password=cfg.password,
            )

    # ------------------------------------------------------------------ #
    def _vertex_to_node(self, vtype: str, vid: str, attrs: dict) -> Node:
        return Node(id=vid, type=vtype, props=dict(attrs))

    # Vertex types this graph knows about (for get_node lookup order).
    _VTYPES = ("Event", "Games", "Venue", "Athlete", "NOC", "Sport")

    def get_node(self, node_id: str) -> Optional[Node]:
        # Route to the right vertex type by id prefix to avoid 6 round-trips.
        order = self._VTYPES
        if node_id.startswith("games:"):
            order = ("Games",) + self._VTYPES
        elif node_id.startswith("venue:"):
            order = ("Venue",) + self._VTYPES
        elif node_id.startswith("athlete:"):
            order = ("Athlete",) + self._VTYPES
        elif node_id.startswith("noc:"):
            order = ("NOC",) + self._VTYPES
        elif node_id.startswith("sport:"):
            order = ("Sport",) + self._VTYPES
        for vtype in order:
            try:
                res = self.conn.getVerticesById(vtype, node_id)
            except Exception:
                res = []
            if res:
                v = res[0]
                return self._vertex_to_node(vtype, v.get("v_id", node_id), v.get("attributes", {}))
        return None

    def _node_vtype(self, node_id: str) -> str:
        if node_id.startswith("games:"):
            return "Games"
        if node_id.startswith("venue:"):
            return "Venue"
        if node_id.startswith("athlete:"):
            return "Athlete"
        if node_id.startswith("noc:"):
            return "NOC"
        if node_id.startswith("sport:"):
            return "Sport"
        return "Event"

    def neighbors(
        self, node_id: str, edge_type: Optional[str] = None, direction: str = "out"
    ) -> list[Node]:
        src_type = self._node_vtype(node_id)
        try:
            res = self.conn.getEdges(src_type, node_id, edgeType=edge_type)
        except Exception:
            return []
        out: list[Node] = []
        for e in res:
            # For out-edges we want to_id; getEdges on a source returns its
            # out-edges, so to_id is the neighbor.
            tgt = e.get("to_id")
            if tgt:
                n = self.get_node(tgt)
                if n:
                    out.append(n)
        return out

    def find_nodes(self, node_type: str, **prop_filters: Any) -> list[Node]:
        try:
            verts = self.conn.getVertices(node_type)
        except Exception:
            verts = []
        result = []
        for v in verts:
            attrs = v.get("attributes", {})
            if all(str(attrs.get(k)) == str(val) for k, val in prop_filters.items()):
                result.append(self._vertex_to_node(node_type, v.get("v_id", ""), attrs))
        return result

    def _interpret(self, gsql: str) -> list[dict]:
        """Run an interpreted GSQL query and return the first result set rows."""
        res = self.conn.runInterpretedQuery(gsql)
        rows: list[dict] = []
        if isinstance(res, list):
            for block in res:
                if isinstance(block, dict):
                    for _, val in block.items():
                        if isinstance(val, list):
                            rows.extend(v for v in val if isinstance(v, dict) and "v_id" in v)
        return rows

    def events_in_games(self, games_id: str, sport: Optional[str] = None) -> list[Node]:
        # PART_OF is directed Event->Games. Gather events in this edition, then
        # (if a sport is named) filter in-graph via HAS_SPORT in a single query.
        # Event has no sport attribute; sport is a Sport node (sport:<name>).
        if sport:
            sport_id = "sport:" + sport.strip().lower().replace('"', "")
            gsql = (
                "INTERPRET QUERY () FOR GRAPH " + self.cfg.graph_name + " {\n"
                '  start = {Event.*};\n'
                f'  inGames = SELECT s FROM start:s -(PART_OF:e)-> Games:g WHERE g.id == "{games_id}";\n'
                f'  res = SELECT s FROM inGames:s -(HAS_SPORT:h)-> Sport:sp WHERE sp.id == "{sport_id}";\n'
                "  PRINT res;\n"
                "}"
            )
        else:
            gsql = (
                "INTERPRET QUERY () FOR GRAPH " + self.cfg.graph_name + " {\n"
                '  start = {Event.*};\n'
                f'  res = SELECT s FROM start:s -(PART_OF:e)-> Games:g WHERE g.id == "{games_id}";\n'
                "  PRINT res;\n"
                "}"
            )
        rows = self._interpret(gsql)
        return [self._vertex_to_node("Event", r["v_id"], r.get("attributes", {})) for r in rows]

    def games_by_year_season(self, year: int, season: str) -> Optional[Node]:
        matches = self.find_nodes("Games", year=year, season=season)
        return matches[0] if matches else None

    def adjacent_games(self, games_id: str, direction: str) -> Optional[Node]:
        et = "NEXT" if direction == "next" else "PREV"
        ns = self.neighbors(games_id, edge_type=et, direction="out")
        return ns[0] if ns else None

    def events_at_venue(self, venue_name: str) -> list[Node]:
        # HELD_AT is Event->Venue; match Venue by name, gather source Events.
        # HELD_AT is directed Event->Venue. Start from Event, keep those whose
        # HELD_AT target venue name matches.
        safe = venue_name.replace('"', "")
        gsql = (
            "INTERPRET QUERY () FOR GRAPH " + self.cfg.graph_name + " {\n"
            '  start = {Event.*};\n'
            f'  res = SELECT s FROM start:s -(HELD_AT:e)-> Venue:v WHERE v.name == "{safe}";\n'
            "  PRINT res;\n"
            "}"
        )
        rows = self._interpret(gsql)
        return [self._vertex_to_node("Event", r["v_id"], r.get("attributes", {})) for r in rows]

    def vector_search(self, query_vector: list[float], k: int = 5) -> list[Node]:
        """Native TigerGraph Vector DB search over Event.emb (installed query)."""
        try:
            res = self.conn.runInstalledQuery(
                "event_vector_search",
                {"query_vector": [float(x) for x in query_vector], "k": k},
            )
        except Exception:
            return []
        nodes: list[Node] = []
        if isinstance(res, list):
            for block in res:
                if isinstance(block, dict) and "results" in block:
                    for v in block["results"]:
                        if isinstance(v, dict) and "v_id" in v:
                            nodes.append(
                                self._vertex_to_node("Event", v["v_id"], v.get("attributes", {}))
                            )
        return nodes

    def find_event_by_title(self, title: str) -> Optional[Node]:
        # Exact title match via interpreted query; try both dash variants.
        raw = title.strip().replace('"', "")
        for variant in (raw, raw.replace("-", "\u2013"), raw.replace("\u2013", "-")):
            gsql = (
                "INTERPRET QUERY () FOR GRAPH " + self.cfg.graph_name + " {\n"
                '  start = {Event.*};\n'
                f'  res = SELECT s FROM start:s WHERE s.title == "{variant}";\n'
                "  PRINT res;\n"
                "}"
            )
            try:
                rows = self._interpret(gsql)
            except Exception:
                rows = []
            if rows:
                r = rows[0]
                return self._vertex_to_node("Event", r["v_id"], r.get("attributes", {}))
        return None

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for vtype in self._VTYPES:
            try:
                out[vtype] = self.conn.getVertexCount(vtype)
            except Exception:
                out[vtype] = -1
        return out
