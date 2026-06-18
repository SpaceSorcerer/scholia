"""Reranker unit tests. FakeReranker is deterministic and model-free; the real
CrossEncoderReranker is exercised only behind @pytest.mark.integration."""

from pathlib import Path

import numpy as np
import pytest

from scholia.models import Paper
from scholia.rerank import (
    FakeReranker,
    CrossEncoderReranker,
    Reranker,
    rerank_hits,
)
from scholia.retrieval import Hit


def _paper(key: str, title: str, abstract: str = "") -> Paper:
    return Paper(
        id=key,
        title=title,
        authors=["Doe, Jane"],
        year="2021",
        doi=f"10.1/{key}",
        zotero_key=key,
        zotero_link=f"zotero://select/library/items/{key}",
        abstract=abstract,
        tags=[],
    )


# --- FakeReranker: protocol + determinism + lexical-overlap ordering ---


def test_fake_reranker_satisfies_protocol():
    assert isinstance(FakeReranker(), Reranker)


def test_fake_reranker_returns_hits_sorted_descending():
    papers = [
        _paper("A", "QKI splicing in heart", "QKI controls alternative splicing"),
        _paper("B", "Mediterranean diet and olive oil", "olive oil lowers risk"),
        _paper("C", "splicing of pre-mRNA by QKI", "QKI binds pre-mRNA"),
    ]
    rr = FakeReranker()
    out = rr.rerank("QKI splicing", papers, top_k=3)
    assert all(isinstance(h, Hit) for h in out)
    scores = [h.score for h in out]
    assert scores == sorted(scores, reverse=True)
    # The two QKI/splicing papers must outrank the diet paper.
    assert out[-1].paper.id == "B"


def test_fake_reranker_is_deterministic():
    papers = [
        _paper("A", "QKI splicing in heart"),
        _paper("B", "olive oil and diet"),
    ]
    a = FakeReranker().rerank("QKI splicing", papers, top_k=2)
    b = FakeReranker().rerank("QKI splicing", papers, top_k=2)
    assert [(h.paper.id, h.score) for h in a] == [(h.paper.id, h.score) for h in b]


def test_fake_reranker_truncates_to_top_k():
    papers = [_paper(str(i), f"paper {i} splicing") for i in range(10)]
    out = FakeReranker().rerank("splicing", papers, top_k=3)
    assert len(out) == 3


def test_fake_reranker_top_k_larger_than_corpus_returns_all():
    papers = [_paper("A", "x"), _paper("B", "y")]
    out = FakeReranker().rerank("query", papers, top_k=5)
    assert len(out) == 2


def test_fake_reranker_empty_papers_returns_empty():
    assert FakeReranker().rerank("anything", [], top_k=5) == []


def test_fake_reranker_reorders_relative_to_input_order():
    """The strongest lexical match should move to the front regardless of input order."""
    papers = [
        _paper("diet", "olive oil mediterranean diet"),
        _paper("qki", "QKI regulates splicing splicing splicing"),
    ]
    out = FakeReranker().rerank("QKI splicing", papers, top_k=2)
    assert out[0].paper.id == "qki"


def test_fake_reranker_stable_for_zero_overlap():
    """Zero-overlap query keeps input order (stable) and assigns finite scores."""
    papers = [_paper("A", "alpha"), _paper("B", "beta")]
    out = FakeReranker().rerank("zzzzz nomatch", papers, top_k=2)
    assert [h.paper.id for h in out] == ["A", "B"]
    assert all(np.isfinite(h.score) for h in out)


# --- rerank_hits: pipeline helper takes Hits, returns reranked Hits ---


def test_rerank_hits_reorders_candidate_hits():
    hits = [
        Hit(_paper("diet", "olive oil diet"), 0.80),  # bi-encoder ranked #1
        Hit(_paper("qki", "QKI splicing splicing"), 0.78),
    ]
    out = rerank_hits("QKI splicing", hits, FakeReranker(), top_k=2)
    # Cross-encoder should promote the genuine match over the bi-encoder #1.
    assert out[0].paper.id == "qki"
    assert out[0].score >= out[1].score


def test_rerank_hits_empty_returns_empty():
    assert rerank_hits("q", [], FakeReranker(), top_k=5) == []


# --- CrossEncoderReranker: structure without downloading a model ---


def test_cross_encoder_reranker_satisfies_protocol():
    assert isinstance(CrossEncoderReranker(), Reranker)


def test_cross_encoder_reranker_does_not_load_on_construction():
    """Construction must be lazy: no model object until first rerank()."""
    rr = CrossEncoderReranker()
    assert rr._model is None


class _StubCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder; scores by string length
    so ordering is deterministic and asserted without a download."""

    def __init__(self, model_name, device="cpu", **kwargs):
        self.model_name = model_name

    def predict(self, pairs, **kwargs):
        # Score = negative length of the document side (shorter docs score higher),
        # purely to make ordering deterministic and testable.
        return np.array([-float(len(doc)) for _q, doc in pairs], dtype=np.float32)


def test_cross_encoder_reranker_uses_loaded_model(monkeypatch):
    rr = CrossEncoderReranker(model_name="dummy-reranker")
    monkeypatch.setattr(rr, "_load_backend", lambda: _StubCrossEncoder("dummy-reranker"))
    papers = [
        _paper("long", "a very long title that is quite lengthy indeed yes"),
        _paper("short", "short"),
    ]
    out = rr.rerank("query", papers, top_k=2)
    # Stub scores shorter docs higher, so "short" must come first.
    assert out[0].paper.id == "short"
    assert out[0].score >= out[1].score
    assert rr._model is not None  # loaded on first use


def test_cross_encoder_reranker_truncates_top_k(monkeypatch):
    rr = CrossEncoderReranker(model_name="dummy-reranker")
    monkeypatch.setattr(rr, "_load_backend", lambda: _StubCrossEncoder("dummy"))
    papers = [_paper(str(i), "x" * (i + 1)) for i in range(6)]
    out = rr.rerank("q", papers, top_k=2)
    assert len(out) == 2


def test_cross_encoder_reranker_empty_returns_empty(monkeypatch):
    rr = CrossEncoderReranker(model_name="dummy-reranker")
    monkeypatch.setattr(rr, "_load_backend", lambda: _StubCrossEncoder("dummy"))
    assert rr.rerank("q", [], top_k=5) == []
