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


# --- Item 4: embedder identity persisted in metadata ---

def test_build_index_persists_embedder_model_and_dim(tmp_path):
    import json
    papers, emb, idx = _build(tmp_path)
    assert idx.embedder_model == "FakeEmbedder"
    assert idx.dim == 16
    meta = json.loads((tmp_path / "idx" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["embedder_model"] == "FakeEmbedder"
    assert meta["dim"] == 16
    assert len(meta["papers"]) == len(papers)


def test_loaded_index_exposes_embedder_model_and_dim(tmp_path):
    _build(tmp_path)
    reloaded = ScholiaIndex.load(tmp_path / "idx")
    assert reloaded.embedder_model == "FakeEmbedder"
    assert reloaded.dim == 16


def test_load_legacy_list_metadata_still_works(tmp_path):
    """A pre-v0.1.1 metadata.json (bare list of papers) must still load."""
    import json
    import faiss as _faiss
    papers = load_corpus(FIXTURES / "corpus")
    emb = FakeEmbedder(dim=16)
    vecs = emb.embed_documents([p.embedding_text for p in papers])
    idx_dir = tmp_path / "legacy"
    idx_dir.mkdir()
    fi = _faiss.IndexFlatIP(16)
    fi.add(np.asarray(vecs, dtype=np.float32))
    _faiss.write_index(fi, str(idx_dir / "index.faiss"))
    # Legacy: bare list of paper dicts (no embedder_model/dim wrapper).
    from scholia.index import _paper_to_meta
    legacy = [_paper_to_meta(p) for p in papers]
    (idx_dir / "metadata.json").write_text(
        json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
    )
    reloaded = ScholiaIndex.load(idx_dir)
    assert len(reloaded._papers) == len(papers)
    assert reloaded.embedder_model == ""
    assert reloaded.dim == 0


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
