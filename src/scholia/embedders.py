"""Embedding backends. FakeEmbedder for tests; NomicEmbedder for production."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into L2-normalized float32 vectors of shape (n, dim)."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class FakeEmbedder:
    """Deterministic hash-based embedder. No model, no RNG. Test-only."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for j in range(self.dim):
                h = hashlib.sha256(f"{j}:{text}".encode("utf-8")).digest()
                # Map first 4 bytes to a float in [-1, 1).
                val = int.from_bytes(h[:4], "big") / 2**31 - 1.0
                vecs[i, j] = val
        return _l2_normalize(vecs)


class NomicEmbedder:
    """Real CPU embedder backed by sentence-transformers (loaded lazily)."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self.dim: int = 0

    def _load_backend(self):
        # Imported lazily so unit tests never trigger a model download.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.model_name, device=self.device, trust_remote_code=True
        )

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self._load_backend()
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)
