"""Vector store backed by TigerGraph Vector DB (no local index rebuild).

The embeddings already live in TigerGraph (Event.emb). This adapter exposes the
same `.search(query, k)` interface as the local VectorStore, but delegates to
the graph backend's native `vector_search` (the installed `event_vector_search`
GSQL query, HNSW + cosine). It embeds only the *query* locally (one short
encode), never the whole corpus — so there is no 21k-chunk build step.
"""
from __future__ import annotations

from ..models import Evidence
from .embeddings import Embedder


class TigerGraphVectorStore:
    def __init__(self, graph, embedder: Embedder) -> None:
        self.g = graph
        self.embedder = embedder

    def build(self, docs=None, **kwargs) -> "TigerGraphVectorStore":
        # No-op: vectors are already indexed in TigerGraph. Kept for interface
        # parity with the local VectorStore.
        return self

    def search(self, query: str, k: int = 5) -> list[Evidence]:
        qv = self.embedder.encode([query])[0]
        nodes = self.g.vector_search([float(x) for x in qv], k=k)
        results: list[Evidence] = []
        for n in nodes:
            title = n.props.get("title", "")
            parts = []
            for key in ("event_name", "year", "season", "competitors", "nations", "date_raw"):
                v = n.props.get(key)
                if v:
                    parts.append(f"{key}={v}")
            results.append(
                Evidence(
                    doc_id=n.id, title=title,
                    snippet="; ".join(parts), score=1.0,
                    source="tg_vector", url=n.props.get("url", ""),
                )
            )
        return results

    def __len__(self) -> int:
        # Number of indexed vectors on the graph (Event count is a good proxy).
        try:
            return self.g.stats().get("Event", 0)
        except Exception:
            return 0
