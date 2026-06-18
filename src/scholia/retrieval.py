"""Embed a query passage and retrieve the top-k matching Papers."""

from __future__ import annotations

from dataclasses import dataclass

from scholia.embedders import Embedder
from scholia.index import ScholiaIndex
from scholia.models import Paper


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
