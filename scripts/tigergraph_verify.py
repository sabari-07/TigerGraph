"""Verify the live TigerGraph load with real counts + a sample traversal."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_graphrag.config import load_config
from pyTigerGraph import TigerGraphConnection

cfg = load_config().tigergraph
conn = TigerGraphConnection(host=cfg.host, graphname=cfg.graph_name, gsqlSecret=cfg.secret)
conn.getToken(cfg.secret)

print("=== per-type vertex counts ===")
for vt in ("Event", "Games", "Venue", "Athlete", "NOC", "Sport"):
    try:
        print(f"  {vt}: {conn.getVertexCount(vt)}")
    except Exception as e:
        print(f"  {vt}: error {e}")

print("\n=== edge counts ===")
for et in ("PART_OF", "HELD_AT", "HAS_SPORT", "WON_GOLD", "REPRESENTS", "NEXT"):
    try:
        print(f"  {et}: {conn.getEdgeCount(et)}")
    except Exception as e:
        print(f"  {et}: error {e}")

print("\n=== sample vertex: a 2010 speed skating event ===")
try:
    v = conn.getVerticesById("Event", "Q607635")
    print(" ", v[0]["attributes"].get("title") if v else "not found")
except Exception as e:
    print("  error", e)
