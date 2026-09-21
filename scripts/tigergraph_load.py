"""One-shot loader: push the generated CSVs into the live TigerGraph graph.

Prereqs:
  - Schema already created on the graph (savanna_01_schema.gsql was run).
  - artifacts/tg/*.csv generated (scripts/tigergraph_ingest.py).
  - .env has TG_HOST + TG_SECRET (fresh secret from Admin Portal) + TG_GRAPH_NAME.

Connects with the Cloud/Savanna pattern (gsqlSecret + getToken), then upserts
vertices and edges directly via the pyTigerGraph upsert APIs (no manual UI
mapping). Vertices first, then edges.

Run:  python scripts/tigergraph_load.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_graphrag.config import load_config  # noqa: E402

TG = Path("artifacts/tg")

VERTEX_FILES = {
    "Event": "event.csv",
    "Games": "games.csv",
    "Venue": "venue.csv",
    "Athlete": "athlete.csv",
    "NOC": "noc.csv",
    "Sport": "sport.csv",
}
INT_ATTRS = {"competitors", "nations", "year"}

EDGE_FILES = {
    "PART_OF": ("Event", "Games", "edge_PART_OF.csv"),
    "HELD_AT": ("Event", "Venue", "edge_HELD_AT.csv"),
    "HAS_SPORT": ("Event", "Sport", "edge_HAS_SPORT.csv"),
    "WON_GOLD": ("Event", "Athlete", "edge_WON_GOLD.csv"),
    "WON_SILVER": ("Event", "Athlete", "edge_WON_SILVER.csv"),
    "WON_BRONZE": ("Event", "Athlete", "edge_WON_BRONZE.csv"),
    "REPRESENTS": ("Athlete", "NOC", "edge_REPRESENTS.csv"),
    "NEXT": ("Games", "Games", "edge_NEXT.csv"),
    "PREV": ("Games", "Games", "edge_PREV.csv"),
}


def _to_int(v):
    try:
        return int(str(v).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0


def connect():
    from pyTigerGraph import TigerGraphConnection

    cfg = load_config().tigergraph
    if not cfg.host or not cfg.secret:
        raise SystemExit("TG_HOST and TG_SECRET must be set in .env")
    conn = TigerGraphConnection(host=cfg.host, graphname=cfg.graph_name, gsqlSecret=cfg.secret)
    conn.getToken(cfg.secret)
    print(f"connected to {cfg.host} graph={cfg.graph_name}")
    return conn


def load_vertices(conn):
    for vtype, fname in VERTEX_FILES.items():
        path = TG / fname
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        batch = {}
        for r in rows:
            vid = r["id"]
            attrs = {}
            for k, v in r.items():
                if k in ("id", "sport"):  # 'sport' col in event.csv is not a schema attr
                    continue
                attrs[k] = _to_int(v) if k in INT_ATTRS else v
            batch[vid] = attrs
        n = conn.upsertVertices(vtype, [(vid, a) for vid, a in batch.items()])
        print(f"  {vtype}: upserted {n} (from {len(rows)} rows)")


def load_edges(conn):
    for etype, (src_t, tgt_t, fname) in EDGE_FILES.items():
        path = TG / fname
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        edges = [(r["from"], r["to"], {}) for r in rows]
        n = conn.upsertEdges(src_t, etype, tgt_t, edges)
        print(f"  {etype}: upserted {n} (from {len(rows)} rows)")


def main():
    conn = connect()
    print("loading vertices...")
    load_vertices(conn)
    print("loading edges...")
    load_edges(conn)
    print("\nvertex counts:", conn.getVertexCount("*"))
    print("done.")


if __name__ == "__main__":
    main()
