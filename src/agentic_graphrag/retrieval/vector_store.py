"""Vector store: chunk documents, embed, and run cosine top-k search.

Used directly by the RAG pipeline and as the "similarity_search" tool for
GraphRAG and the agent. Kept dependency-light (numpy only) so it runs anywhere;
the same interface would be backed by TigerGraph Vector DB in the Savanna
deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..models import Document, Evidence
from .embeddings import Embedder


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str
    url: str = ""


def chunk_document(doc: Document, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Split a document into overlapping character windows.

    The infobox at the top is information-dense, so we keep the first window
    intact and slide over the rest.
    """
    text = doc.text
    if len(text) <= max_chars:
        return [Chunk(doc.doc_id, doc.title, text, doc.url)]
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(Chunk(doc.doc_id, doc.title, text[start:end], doc.url))
        if end == len(text):
            break
        start = end - overlap
    return chunks


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._matrix: Optional[np.ndarray] = None

    def build(self, docs: list[Document], max_chars: int = 1200, overlap: int = 150) -> "VectorStore":
        for doc in docs:
            self.chunks.extend(chunk_document(doc, max_chars, overlap))
        texts = [f"{c.title}\n{c.text}" for c in self.chunks]
        # Batch encode to keep memory reasonable.
        mats: list[np.ndarray] = []
        batch = 512
        for i in range(0, len(texts), batch):
            mats.append(self.embedder.encode(texts[i : i + batch]))
        self._matrix = np.vstack(mats) if mats else np.zeros((0, self.embedder.dim))
        return self

    def search(self, query: str, k: int = 5) -> list[Evidence]:
        if self._matrix is None or len(self.chunks) == 0:
            return []
        q = self.embedder.encode([query])[0]
        scores = self._matrix @ q  # cosine (vectors are normalized)
        top_idx = np.argsort(-scores)[:k]
        results: list[Evidence] = []
        for idx in top_idx:
            c = self.chunks[idx]
            results.append(
                Evidence(
                    doc_id=c.doc_id,
                    title=c.title,
                    snippet=c.text[:500],
                    score=float(scores[idx]),
                    source="vector",
                    url=c.url,
                )
            )
        return results

    def __len__(self) -> int:
        return len(self.chunks)
