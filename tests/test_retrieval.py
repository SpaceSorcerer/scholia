from pathlib import Path

from scholia.corpus import load_corpus
from scholia.embedders import FakeEmbedder
from scholia.index import build_index
from scholia.retrieval import retrieve, Hit

FIXTURES = Path(__file__).parent / "fixtures"


def _index(tmp_path):
    papers = load_corpus(FIXTURES / "corpus")
    emb = FakeEmbedder(dim=16)
    return build_index(papers, emb, tmp_path / "idx"), emb, papers


def test_retrieve_returns_hits(tmp_path):
    idx, emb, papers = _index(tmp_path)
    hits = retrieve(papers[0].embedding_text, emb, idx, k=3)
    assert all(isinstance(h, Hit) for h in hits)
    assert hits[0].paper.id == papers[0].id


def test_retrieve_scores_are_descending(tmp_path):
    idx, emb, _ = _index(tmp_path)
    hits = retrieve("some query about splicing", emb, idx, k=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_default_k(tmp_path):
    idx, emb, _ = _index(tmp_path)
    hits = retrieve("query", emb, idx)  # default k=5, corpus has 3
    assert len(hits) == 3
