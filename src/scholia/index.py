"""FAISS-backed vector index with a JSON metadata sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from scholia.embedders import Embedder
from scholia.models import Paper

_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.json"

_META_FIELDS = (
    "id", "title", "year", "doi", "zotero_key", "zotero_link", "authors", "tags",
)


def _paper_to_meta(p: Paper) -> dict:
    return {f: getattr(p, f) for f in _META_FIELDS}


def _meta_to_paper(d: dict) -> Paper:
    # abstract is not persisted in metadata (not needed downstream); empty is fine.
    return Paper(
        id=d["id"],
        title=d.get("title", ""),
        authors=list(d.get("authors", [])),
        year=d.get("year", ""),
        doi=d.get("doi", ""),
        zotero_key=d.get("zotero_key", ""),
        zotero_link=d.get("zotero_link", ""),
        abstract="",
        tags=list(d.get("tags", [])),
    )


class ScholiaIndex:
    """An in-memory FAISS index plus the Papers it indexes, in row order.

    ``embedder_model`` / ``dim`` record the embedder used at build time so
    ``cite`` can adopt the same embedder and pick an embedder-appropriate
    threshold. Both may be empty/0 for legacy indices built before this was
    persisted.
    """

    def __init__(
        self,
        faiss_index: "faiss.Index",
        papers: list[Paper],
        embedder_model: str = "",
        dim: int = 0,
    ) -> None:
        self._index = faiss_index
        self._papers = papers
        self.embedder_model = embedder_model
        self.dim = dim

    @classmethod
    def load(cls, index_dir: Path) -> "ScholiaIndex":
        index_dir = Path(index_dir)
        faiss_path = index_dir / _INDEX_FILE
        meta_path = index_dir / _META_FILE
        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No index at {index_dir}. Run `scholia index` first."
            )
        faiss_index = faiss.read_index(str(faiss_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # New format: {"embedder_model", "dim", "papers": [...]}.
        # Legacy format: a bare list of paper dicts.
        if isinstance(meta, dict):
            paper_dicts = meta.get("papers", [])
            embedder_model = str(meta.get("embedder_model", ""))
            dim = int(meta.get("dim", 0) or 0)
        else:
            paper_dicts = meta
            embedder_model = ""
            dim = 0
        papers = [_meta_to_paper(d) for d in paper_dicts]
        return cls(faiss_index, papers, embedder_model, dim)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Paper, float]]:
        if len(self._papers) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(k, len(self._papers))
        scores, ids = self._index.search(q, k)
        hits: list[tuple[Paper, float]] = []
        for row_id, score in zip(ids[0], scores[0]):
            if row_id < 0:
                continue
            hits.append((self._papers[int(row_id)], float(score)))
        return hits


def _embedder_model_name(embedder: Embedder) -> str:
    """Best-effort identity for the embedder used to build an index.

    NomicEmbedder exposes ``model_name``; FakeEmbedder (test) has none, so we
    fall back to its class name.
    """
    name = getattr(embedder, "model_name", None)
    return str(name) if name else type(embedder).__name__


def build_index(
    papers: list[Paper], embedder: Embedder, index_dir: Path
) -> ScholiaIndex:
    if not papers:
        raise ValueError("Cannot build index: corpus is empty (no papers to index).")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    _embed_docs = getattr(embedder, "embed_documents", None) or embedder.embed
    vectors = _embed_docs([p.embedding_text for p in papers])
    vectors = np.asarray(vectors, dtype=np.float32)
    dim = int(vectors.shape[1])
    embedder_model = _embedder_model_name(embedder)

    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)

    faiss.write_index(faiss_index, str(index_dir / _INDEX_FILE))
    meta = {
        "embedder_model": embedder_model,
        "dim": dim,
        "papers": [_paper_to_meta(p) for p in papers],
    }
    (index_dir / _META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ScholiaIndex(faiss_index, papers, embedder_model, dim)
