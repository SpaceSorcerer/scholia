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
