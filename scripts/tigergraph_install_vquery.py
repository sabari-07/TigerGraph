"""Create + install the event_vector_search GSQL query programmatically."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_graphrag.config import load_config
from pyTigerGraph import TigerGraphConnection

cfg = load_config().tigergraph
conn = TigerGraphConnection(host=cfg.host, graphname=cfg.graph_name, gsqlSecret=cfg.secret)
conn.getToken(cfg.secret)

create_q = f"""
USE GRAPH {cfg.graph_name}
CREATE OR REPLACE QUERY event_vector_search(LIST<float> query_vector, INT k) SYNTAX v3 {{
  MapAccum<Vertex, Float> @@distances;
  v = vectorSearch({{Event.emb}}, query_vector, k, {{distance_map: @@distances}});
  PRINT v AS results;
  PRINT @@distances AS distances;
}}
"""
print("creating query...")
print(conn.gsql(create_q))

print("installing query...")
print(conn.gsql(f"USE GRAPH {cfg.graph_name}\nINSTALL QUERY event_vector_search"))
