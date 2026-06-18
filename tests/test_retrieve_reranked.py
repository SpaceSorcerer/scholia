"""Pipeline test: retrieve_reranked() takes FAISS top-candidate_k and re-ranks
to top_k. Backward compatibility of retrieve() is covered in test_retrieval.py."""

from pathlib import Path

from scholia.corpus import load_corpus
from scholia.embedders import FakeEmbedder
from scholia.index import build_index
from scholia.rerank import FakeReranker
from scholia.retrieval import retrieve, retrieve_reranked, Hit

FIXTURES = Path(__file__).parent / "fixtures"


def _index(tmp_path):
    papers = load_corpus(FIXTURES / "corpus")
    emb = FakeEmbedder(dim=16)
    return build_index(papers, emb, tmp_path / "idx"), emb, papers


def test_retrieve_reranked_returns_hits(tmp_path):
    idx, emb, papers = _index(tmp_path)
    out = retrieve_reranked(
        "QKI alternative splicing", emb, idx, FakeReranker(),
        candidate_k=30, top_k=3,
    )
    assert all(isinstance(h, Hit) for h in out)
    assert len(out) <= 3


def test_retrieve_reranked_top_k_bounds_results(tmp_path):
    idx, emb, _ = _index(tmp_path)
    out = retrieve_reranked(
        "splicing", emb, idx, FakeReranker(), candidate_k=30, top_k=2,
    )
    assert len(out) == 2


def test_retrieve_reranked_scores_descending(tmp_path):
    idx, emb, _ = _index(tmp_path)
    out = retrieve_reranked(
        "QKI splicing in cardiomyocytes", emb, idx, FakeReranker(),
        candidate_k=30, top_k=3,
    )
    scores = [h.score for h in out]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_reranked_empty_passage_returns_empty(tmp_path):
    idx, emb, _ = _index(tmp_path)
    assert retrieve_reranked("", emb, idx, FakeReranker()) == []
    assert retrieve_reranked("   ", emb, idx, FakeReranker()) == []


def test_retrieve_reranked_reorders_vs_bi_encoder(tmp_path):
    """The reranked order must be able to differ from the raw FAISS order.

    With the FakeEmbedder (hash-based, semantically meaningless) the bi-encoder
    order is effectively arbitrary; the FakeReranker (lexical overlap) imposes a
    query-aware order. We assert the reranked top-1 is the strongest lexical
    match for a QKI/splicing query among the candidates."""
    idx, emb, papers = _index(tmp_path)
    query = "QKI RNA-binding protein controls alternative splicing"
    reranked = retrieve_reranked(
        query, emb, idx, FakeReranker(), candidate_k=30, top_k=3,
    )
    # FakeReranker scores by lexical overlap; the QKI paper (paperA / AAAAAAAA)
    # has the most query tokens, so it must lead after reranking.
    assert reranked[0].paper.id == "AAAAAAAA"


def test_retrieve_reranked_candidate_k_limits_pool(tmp_path):
    """Only the FAISS top-candidate_k feed the reranker. With candidate_k=1 the
    reranker can only ever see (and thus return) a single candidate."""
    idx, emb, _ = _index(tmp_path)
    out = retrieve_reranked(
        "splicing", emb, idx, FakeReranker(), candidate_k=1, top_k=5,
    )
    assert len(out) == 1


def test_retrieve_without_reranker_is_unchanged(tmp_path):
    """Backward-compat: plain retrieve() still returns bi-encoder hits."""
    idx, emb, papers = _index(tmp_path)
    hits = retrieve(papers[0].embedding_text, emb, idx, k=3)
    assert hits[0].paper.id == papers[0].id
