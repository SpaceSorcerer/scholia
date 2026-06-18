"""Real cross-encoder reranker end-to-end test. Run explicitly:

    pytest -m integration

Downloads the cross-encoder model on first run; deselected by default via the
addopts filter in pyproject.toml (same posture as the embedder integration test).
"""

from pathlib import Path

import pytest

from scholia.corpus import load_corpus
from scholia.embedders import NomicEmbedder
from scholia.index import build_index
from scholia.rerank import CrossEncoderReranker
from scholia.retrieval import retrieve_reranked

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.integration
def test_real_cross_encoder_reranks_topical_paper(tmp_path):
    papers = load_corpus(FIXTURES / "corpus")
    embedder = NomicEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = build_index(papers, embedder, tmp_path / "idx")
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
    )
    out = retrieve_reranked(
        "How does the RNA-binding protein QKI control splicing during heart development?",
        embedder, index, reranker, candidate_k=3, top_k=3,
    )
    # The QKI paper (AAAAAAAA) must lead after cross-encoder reranking.
    assert out[0].paper.id == "AAAAAAAA"
