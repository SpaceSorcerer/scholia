"""Embed a query passage and retrieve the top-k matching Papers.

``retrieve`` is the bi-encoder path (FAISS cosine). ``retrieve_reranked`` adds an
optional cross-encoder re-rank stage on top: it pulls the FAISS top-``candidate_k``
and re-scores them down to ``top_k`` with a Reranker, producing a cleaner
relevance signal and a wider SUPPORTED/UNSUPPORTED margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scholia.embedders import Embedder
from scholia.index import ScholiaIndex
from scholia.models import Paper

if TYPE_CHECKING:  # avoid a runtime import cycle (rerank imports Hit from here)
    from scholia.rerank import Reranker


@dataclass(frozen=True)
class Hit:
    paper: Paper
    score: float


def retrieve(
    passage: str, embedder: Embedder, index: ScholiaIndex, k: int = 5
) -> list[Hit]:
    """Return up to k Hits for a passage, sorted by descending cosine score.

    An empty or whitespace-only passage carries no claim, so it returns ``[]``
    (claim-check -> UNSUPPORTED). This also avoids embedding degenerate blank
    text, whose vector floats near the corpus centroid and otherwise produces a
    spuriously high similarity (a documented nomic false-positive).
    """
    if not passage or not passage.strip():
        return []
    _embed_q = getattr(embedder, "embed_query", None)
    if _embed_q is not None:
        query_vector = _embed_q(passage)
    else:
        query_vector = embedder.embed([passage])[0]
    results = index.search(query_vector, k)
    return [Hit(paper=p, score=s) for p, s in results]


def retrieve_reranked(
    passage: str,
    embedder: Embedder,
    index: ScholiaIndex,
    reranker: "Reranker",
    candidate_k: int = 30,
    top_k: int = 5,
) -> list[Hit]:
    """Retrieve FAISS top-``candidate_k`` then cross-encoder re-rank to ``top_k``.

    The bi-encoder fetches a wide candidate pool cheaply; the (more expensive but
    more discriminative) cross-encoder re-scores only that pool. Returned Hit
    scores are the reranker's relevance scores — a DIFFERENT scale than cosine
    (see ``rerank.py``), so the claim-check threshold must be the reranker's, not
    the bi-encoder's.

    An empty/whitespace-only passage carries no claim and short-circuits to ``[]``
    (claim-check -> UNSUPPORTED), matching ``retrieve``.
    """
    # Local import avoids a module-level cycle (rerank imports Hit from here).
    from scholia.rerank import rerank_hits

    candidates = retrieve(passage, embedder, index, k=candidate_k)
    if not candidates:
        return []
    return rerank_hits(passage, candidates, reranker, top_k=top_k)
