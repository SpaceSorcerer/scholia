"""Embedding backends. FakeEmbedder for tests; NomicEmbedder for production."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into L2-normalized float32 vectors of shape (n, dim).

    The required surface is ONLY ``dim: int`` and ``embed(texts) -> ndarray``.
    This is the v0.1.0 public extension API; third-party embedders implementing
    only these two members satisfy ``isinstance(obj, Embedder)``.

    ``embed_documents`` (corpus side) and ``embed_query`` (query side) are
    OPTIONAL refinements that let an embedder apply asymmetric task instructions
    (e.g. nomic-embed-v1.5's ``search_document:`` / ``search_query:`` prefixes).
    When absent, callers fall through to ``embed`` unchanged.
    """

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class _DefaultPrefixMixin:
    """Default doc/query embedding: no prefixes, just delegate to ``embed``.

    Embedders that need asymmetric instructions (NomicEmbedder) override
    ``embed_documents`` / ``embed_query``; everyone else (FakeEmbedder, the
    MiniLM path) inherits this no-op behaviour, so they are unaffected.
    """

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class FakeEmbedder(_DefaultPrefixMixin):
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


# nomic-embed-v1.5 is trained to require these task instructions; without them
# the similarity floor is inflated (an empty string scores ~0.58) and ranking
# degrades. See https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
_NOMIC_DOC_PREFIX = "search_document: "
_NOMIC_QUERY_PREFIX = "search_query: "


class NomicEmbedder(_DefaultPrefixMixin):
    """Real CPU embedder backed by sentence-transformers (loaded lazily).

    For genuine nomic-embed models, ``embed_documents``/``embed_query`` prepend
    the required ``search_document:``/``search_query:`` task prefixes. For any
    other model loaded through this class (e.g. all-MiniLM-L6-v2), no prefix is
    applied and the inherited default (plain ``embed``) is used unchanged.
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self.dim: int = 0

    @property
    def _uses_nomic_prefixes(self) -> bool:
        return "nomic" in self.model_name.lower()

    def _load_backend(self):
        # Imported lazily so unit tests never trigger a model download.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.model_name, device=self.device, trust_remote_code=True
        )

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self._load_backend()
            # get_sentence_embedding_dimension() was renamed in sentence-transformers
            # >=3.x; fall back gracefully so either version works.
            _dim_fn = getattr(
                self._model,
                "get_embedding_dimension",
                None,
            ) or getattr(
                self._model,
                "get_sentence_embedding_dimension",
                None,
            )
            self.dim = int(_dim_fn()) if _dim_fn is not None else 768

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._uses_nomic_prefixes:
            texts = [_NOMIC_DOC_PREFIX + t for t in texts]
        return self.embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if self._uses_nomic_prefixes:
            text = _NOMIC_QUERY_PREFIX + text
        return self.embed([text])[0]
