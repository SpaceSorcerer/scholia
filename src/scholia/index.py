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
    """An in-memory FAISS index plus the Papers it indexes, in row order."""

    def __init__(self, faiss_index: "faiss.Index", papers: list[Paper]) -> None:
        self._index = faiss_index
        self._papers = papers

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
        papers = [_meta_to_paper(d) for d in meta]
        return cls(faiss_index, papers)

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


def build_index(
    papers: list[Paper], embedder: Embedder, index_dir: Path
) -> ScholiaIndex:
    if not papers:
        raise ValueError("Cannot build index: corpus is empty (no papers to index).")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    vectors = embedder.embed([p.embedding_text for p in papers])
    vectors = np.asarray(vectors, dtype=np.float32)
    dim = vectors.shape[1]

    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)

    faiss.write_index(faiss_index, str(index_dir / _INDEX_FILE))
    meta = [_paper_to_meta(p) for p in papers]
    (index_dir / _META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ScholiaIndex(faiss_index, papers)
