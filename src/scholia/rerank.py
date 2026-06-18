"""Re-ranking backends. FakeReranker for tests; CrossEncoderReranker for production.

A bi-encoder (FAISS cosine over independently embedded query/document vectors)
gives a fast but coarse relevance signal with tight, fragile score margins. A
cross-encoder scores each ``(query, document)`` pair *jointly*, which is far more
discriminative and widens the SUPPORTED/UNSUPPORTED gap. The pipeline uses the
bi-encoder to fetch a candidate pool (FAISS top-``candidate_k``) and the
cross-encoder to re-rank that pool down to ``top_k`` (the classic
retrieve-then-rerank pattern).

This mirrors ``embedders.py``: a ``Reranker`` Protocol, a deterministic
model-free ``FakeReranker`` for unit tests, and a lazily-loaded real
``CrossEncoderReranker`` whose ``sentence_transformers`` import lives INSIDE the
load method so unit tests never trigger a download.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import numpy as np

from scholia.models import Paper
from scholia.retrieval import Hit


@runtime_checkable
class Reranker(Protocol):
    """Re-scores candidate papers against a query and returns the top_k.

    The required surface is a single method::

        rerank(query: str, papers: list[Paper], top_k: int) -> list[Hit]

    Each returned ``Hit.score`` is the cross-encoder relevance for that paper
    (NOT a cosine similarity — a different, model-specific scale; often a logit),
    and the list is sorted by descending score, truncated to ``top_k``.
    """

    def rerank(self, query: str, papers: list[Paper], top_k: int) -> list[Hit]:
        ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _sorted_top_k(
    papers: list[Paper], scores: list[float], top_k: int
) -> list[Hit]:
    """Pair papers with scores and return the top_k Hits (descending, stable).

    Ties keep the input (candidate) order: Python's sort is stable and we sort
    only on the negated score, so equal scores preserve their original sequence.
    """
    order = sorted(range(len(papers)), key=lambda i: -scores[i])
    hits = [Hit(paper=papers[i], score=float(scores[i])) for i in order]
    if top_k is not None and top_k >= 0:
        hits = hits[:top_k]
    return hits


class FakeReranker:
    """Deterministic, model-free reranker for unit tests.

    Scores each paper by lexical token overlap between the query and the paper's
    ``embedding_text`` (Jaccard-like: shared unique tokens / query token count).
    No model, no RNG, no network — stable across processes. Ordering is
    query-aware so tests can assert that a genuine match is promoted over a
    bi-encoder false positive.
    """

    def rerank(self, query: str, papers: list[Paper], top_k: int) -> list[Hit]:
        if not papers:
            return []
        q_tokens = set(_tokenize(query))
        n_q = max(len(q_tokens), 1)
        scores: list[float] = []
        for p in papers:
            doc_tokens = set(_tokenize(p.embedding_text))
            overlap = len(q_tokens & doc_tokens)
            scores.append(overlap / n_q)
        return _sorted_top_k(papers, scores, top_k)


# Per-reranker-model context. MiniLM ms-marco cross-encoders emit a single
# relevance logit (roughly -11..+11); bge-reranker-v2-m3 likewise emits a
# relevance logit. The SUPPORTED/UNSUPPORTED cutoff is derived empirically per
# model and lives in cli.py (default_reranker_threshold_for), exactly like the
# embedder-aware cosine thresholds.
_DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Real CPU cross-encoder reranker, backed by sentence-transformers.

    Lazily loads ``sentence_transformers.CrossEncoder`` on first ``rerank`` call;
    the import lives inside ``_load_backend`` so importing this module (and the
    unit-test suite) never triggers a model download. One-time model download on
    first real use, same posture as ``NomicEmbedder``. Local CPU only.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_backend(self):
        # Imported lazily so unit tests never trigger a model download.
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, device=self.device)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self._load_backend()

    def rerank(self, query: str, papers: list[Paper], top_k: int) -> list[Hit]:
        if not papers:
            return []
        self._ensure_loaded()
        pairs = [[query, p.embedding_text] for p in papers]
        raw = self._model.predict(pairs)
        scores = [float(s) for s in np.asarray(raw, dtype=np.float32).ravel()]
        return _sorted_top_k(papers, scores, top_k)


def rerank_hits(
    query: str, hits: list[Hit], reranker: Reranker, top_k: int = 5
) -> list[Hit]:
    """Re-rank an existing list of (bi-encoder) Hits and return the top_k.

    Thin adapter so the retrieval layer can hand its FAISS candidate Hits to any
    Reranker without the Reranker needing to know about Hits. The returned Hit
    scores are the cross-encoder relevance scores, NOT the input cosine scores.
    """
    if not hits:
        return []
    papers = [h.paper for h in hits]
    return reranker.rerank(query, papers, top_k)
