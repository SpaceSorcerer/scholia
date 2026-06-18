"""Real-model end-to-end test. Run explicitly: pytest -m integration

Downloads the embedding model on first run; deselected by default via the
addopts filter in pyproject.toml.
"""

from pathlib import Path

import pytest

from scholia.corpus import load_corpus
from scholia.embedders import NomicEmbedder
from scholia.index import build_index
from scholia.retrieval import retrieve
from scholia.grounding import claim_check

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.integration
def test_real_model_retrieves_topical_paper(tmp_path):
    papers = load_corpus(FIXTURES / "corpus")
    embedder = NomicEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = build_index(papers, embedder, tmp_path / "idx")

    # A QKI/splicing query should rank the QKI paper (AAAAAAAA) first.
    hits = retrieve(
        "How does the RNA-binding protein QKI control splicing during heart development?",
        embedder, index, k=3,
    )
    assert hits[0].paper.id == "AAAAAAAA"
    assert claim_check(hits, threshold=0.30).supported is True
