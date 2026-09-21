"""Select and build the graph backend from config.

This project runs on TigerGraph. GRAPH_BACKEND=tigergraph connects to the live
Savanna graph and is the intended path; there is NO silent fallback to a local
graph — if TigerGraph cannot be reached the system fails loudly so results are
never quietly produced from a different backend.

GRAPH_BACKEND=local exists only as an explicit developer/offline escape hatch
and must be chosen deliberately.
"""
from __future__ import annotations

from ..config import Config
from ..models import Document
from .base import GraphBackend


def build_backend(cfg: Config, docs: list[Document]) -> GraphBackend:
    backend_kind = (cfg.graph_backend or "tigergraph").lower()

    if backend_kind == "local":
        # Explicit, deliberate offline mode only.
        from .builder import build_graph
        print("[graph] GRAPH_BACKEND=local (explicit offline mode) - NOT TigerGraph.")
        return build_graph(docs)

    # Default and intended path: TigerGraph. No fallback.
    from .tigergraph_backend import TigerGraphBackend

    backend = TigerGraphBackend(cfg.tigergraph)
    print(f"[graph] using TigerGraph backend @ {cfg.tigergraph.host}")
    return backend
