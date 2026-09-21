"""Generate vertex/edge CSVs for loading the corpus into TigerGraph.

Builds the graph locally from the corpus and writes one CSV per vertex type and
one per edge type to artifacts/tg/. These CSVs are consumed by
scripts/tigergraph_load.py (programmatic upsert) or can be uploaded via the
Savanna Load Data UI.

The GSQL schema/queries are committed separately in tigergraph/:
  tigergraph/01_schema.gsql          vertex + edge types
  tigergraph/02_vector_attribute.gsql  Event.emb vector attribute
  tigergraph/03_vector_query.gsql    event_vector_search installed query

Usage:
  python scripts/tigergraph_ingest.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_graphrag.data.loader import load_documents  # noqa: E402
from agentic_graphrag.graph.builder import build_graph  # noqa: E402

OUT = Path("artifacts/tg")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    docs = load_documents("corpus/corpus.jsonl")
    g = build_graph(docs)

    # Vertex CSVs (one per type). Column names match schema attribute names so
    # the loader / Savanna Quick Map aligns them automatically. 'text' is
    # skipped (too large for CSV; not a graph attribute).
    for vtype in ("Event", "Games", "Venue", "Athlete", "NOC", "Sport"):
        ids = g._by_type.get(vtype, [])
        path = OUT / f"{vtype.lower()}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            keys: list[str] = []
            for nid in ids:
                for k in g._nodes[nid].props:
                    if k not in keys and k != "text":
                        keys.append(k)
            w.writerow(["id"] + keys)
            for nid in ids:
                props = g._nodes[nid].props
                w.writerow([nid] + [props.get(k, "") for k in keys])
        print(f"wrote {path} ({len(ids)} rows)")

    # Edge CSVs (one per type), columns: from, to.
    edge_rows: dict[str, list[tuple[str, str]]] = {}
    for src, emap in g._out.items():
        for et, dsts in emap.items():
            for dst in dsts:
                edge_rows.setdefault(et, []).append((src, dst))
    for et, rows in edge_rows.items():
        et_name = et.value if hasattr(et, "value") else str(et)
        path = OUT / f"edge_{et_name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["from", "to"])
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

    print("\nCSVs written to artifacts/tg/. Next: run scripts/tigergraph_load.py "
          "(schema first via tigergraph/01_schema.gsql).")


if __name__ == "__main__":
    main()
