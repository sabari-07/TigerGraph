"""Embed each Event locally and upsert the vectors into TigerGraph Vector DB.

Prereqs:
  - Vector attribute added: run artifacts/tg/savanna_02_vector.gsql first.
  - .env has TG_HOST + TG_SECRET + TG_GRAPH_NAME.
  - sentence-transformers installed (EMBEDDING_MODEL, 384-dim).

We embed "title + infobox summary" per event (same text the local vector store
chunked), then upsert the 384-dim vector to the Event.emb attribute via the
pyTigerGraph upsert API. Batches to keep requests reasonable.

Run:  python scripts/tigergraph_load_vectors.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_graphrag.config import load_config
from agentic_graphrag.data.loader import load_documents
from agentic_graphrag.retrieval.embeddings import build_embedder


def event_text(doc) -> str:
    # Compact, information-dense text: title + infobox key facts.
    ib = doc.infobox
    facts = []
    for k in ("games", "event", "venue", "date", "dates", "competitors",
              "nations", "gold", "silver", "bronze"):
        if ib.get(k):
            facts.append(f"{k}: {ib[k]}")
    return doc.title + "\n" + "\n".join(facts)


def main():
    cfg = load_config()
    from pyTigerGraph import TigerGraphConnection

    conn = TigerGraphConnection(host=cfg.tigergraph.host,
                                graphname=cfg.tigergraph.graph_name,
                                gsqlSecret=cfg.tigergraph.secret)
    conn.getToken(cfg.tigergraph.secret)
    print("connected.")

    docs = [d for d in load_documents(cfg.corpus_path) if d.is_olympic_event]
    print(f"embedding {len(docs)} events with {cfg.embedding.model} ...")
    embedder = build_embedder(cfg.embedding)
    print(f"embedder: {type(embedder).__name__} dim={embedder.dim}")

    texts = [event_text(d) for d in docs]
    # Batch encode.
    import numpy as np
    vecs = []
    B = 256
    for i in range(0, len(texts), B):
        vecs.append(embedder.encode(texts[i:i+B]))
        print(f"  embedded {min(i+B, len(texts))}/{len(texts)}")
    mat = np.vstack(vecs)

    # Upsert vectors via REST upsert API in batches.
    print("upserting vectors to Event.emb ...")
    total = 0
    B2 = 200
    for i in range(0, len(docs), B2):
        batch = {}
        for d, v in zip(docs[i:i+B2], mat[i:i+B2]):
            batch[d.doc_id] = {"emb": {"value": [float(x) for x in v]}}
        payload = {"vertices": {"Event": batch}}
        conn.upsertData(payload)
        total += len(batch)
        print(f"  upserted {total}/{len(docs)}")

    print("done. Check index status; vectors index incrementally.")
    try:
        print("vector status:", conn.get("/restpp/vector/status"))
    except Exception:
        pass


if __name__ == "__main__":
    main()
