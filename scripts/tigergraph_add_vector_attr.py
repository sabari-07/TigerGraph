"""Add the Event.emb vector attribute programmatically via GSQL (with -N)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_graphrag.config import load_config
from pyTigerGraph import TigerGraphConnection

cfg = load_config().tigergraph
conn = TigerGraphConnection(host=cfg.host, graphname=cfg.graph_name, gsqlSecret=cfg.secret)
conn.getToken(cfg.secret)

create_job = f"""
USE GRAPH {cfg.graph_name}
CREATE SCHEMA_CHANGE JOB add_event_embedding FOR GRAPH {cfg.graph_name} {{
  ALTER VERTEX Event ADD VECTOR ATTRIBUTE emb(DIMENSION=384, METRIC="COSINE");
}}
"""

print("creating schema-change job...")
print(conn.gsql(create_job))

print("running schema-change job...")
print(conn.gsql(f"USE GRAPH {cfg.graph_name}\nRUN SCHEMA_CHANGE JOB add_event_embedding"))
