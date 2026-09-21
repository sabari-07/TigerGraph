"""Embedding providers.

Default is sentence-transformers (local, free). A deterministic hashing-based
'mock' provider is available so the whole system runs in CI / offline with no
model download, while still giving reproducible, lexically-sensitive vectors.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Optional

import numpy as np

from ..config import EmbeddingConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    """Base embedder. Subclasses implement `_encode`."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._encode(texts)
        # L2-normalize for cosine == dot product.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def _encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder (no dependencies).

    Not as strong as a neural model, but reproducible and fully offline, which
    keeps the benchmark runnable anywhere. Swap to SentenceTransformerEmbedder
    for best accuracy.
    """

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                # sublinear tf weighting
                out[i, idx] += sign
        # log-scale magnitudes
        out = np.sign(out) * np.log1p(np.abs(out))
        return out


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, dim: int) -> None:
        super().__init__(dim)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True),
            dtype=np.float32,
        )


def build_embedder(cfg: Optional[EmbeddingConfig] = None) -> Embedder:
    cfg = cfg or EmbeddingConfig()
    if cfg.provider == "sentence_transformers":
        try:
            return SentenceTransformerEmbedder(cfg.model, cfg.dim)
        except Exception as exc:  # graceful offline fallback
            print(f"[embeddings] falling back to hashing embedder: {exc}")
            return HashingEmbedder(cfg.dim)
    return HashingEmbedder(cfg.dim)
