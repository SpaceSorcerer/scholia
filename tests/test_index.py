from pathlib import Path

import numpy as np
import pytest

from scholia.corpus import load_corpus
from scholia.embedders import FakeEmbedder
from scholia.index import build_index, ScholiaIndex

FIXTURES = Path(__file__).parent / "fixtures"


def _build(tmp_path):
    papers = load_corpus(FIXTURES / "corpus")
    emb = FakeEmbedder(dim=16)
    idx = build_index(papers, emb, tmp_path / "idx")
    return papers, emb, idx


def test_build_index_writes_files(tmp_path):
    _build(tmp_path)
    assert (tmp_path / "idx" / "index.faiss").exists()
    assert (tmp_path / "idx" / "metadata.json").exists()


def test_search_returns_self_as_top_hit(tmp_path):
    papers, emb, idx = _build(tmp_path)
    target = papers[0]
    qvec = emb.embed([target.embedding_text])[0]
    hits = idx.search(qvec, k=3)
    assert hits[0][0].id == target.id
    assert hits[0][1] > 0.99  # near-identical cosine to itself


def test_loaded_index_matches_built_index(tmp_path):
    papers, emb, _ = _build(tmp_path)
    reloaded = ScholiaIndex.load(tmp_path / "idx")
    qvec = emb.embed([papers[1].embedding_text])[0]
    hits = reloaded.search(qvec, k=1)
    assert hits[0][0].id == papers[1].id


def test_search_respects_k(tmp_path):
    _, emb, idx = _build(tmp_path)
    qvec = emb.embed(["unrelated query text"])[0]
    hits = idx.search(qvec, k=2)
    assert len(hits) == 2


# --- Finding A: empty-corpus guard ---

def test_build_index_empty_raises_value_error(tmp_path):
    emb = FakeEmbedder(dim=16)
    with pytest.raises(ValueError, match="corpus is empty"):
        build_index([], emb, tmp_path / "idx")


# --- Finding B: missing-dir guard and empty-index search ---

def test_load_missing_dir_raises_file_not_found(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="No index at"):
        ScholiaIndex.load(missing)


def test_load_empty_dir_raises_file_not_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No index at"):
        ScholiaIndex.load(empty)


def test_search_on_empty_index_returns_empty_list(tmp_path):
    """A ScholiaIndex with 0 vectors must return [] without crashing."""
    import faiss as _faiss
    fi = _faiss.IndexFlatIP(16)
    empty_idx = ScholiaIndex(fi, [])
    qvec = np.zeros(16, dtype=np.float32)
    hits = empty_idx.search(qvec, k=5)
    assert hits == []
