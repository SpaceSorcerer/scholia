"""READ-ONLY integration test for the real S2 + PubMed discovery sources.

Deselected by default (``-m 'not integration'`` in pyproject). Run with
``pytest -m integration``. This performs a live, READ-ONLY search against the
free Semantic Scholar Academic Graph API and PubMed E-utilities — it NEVER adds
anything to Zotero. Network failures (offline / rate-limited) skip rather than
fail, so the suite stays green without connectivity.
"""

from __future__ import annotations

import pytest

from scholia.discovery import (
    Candidate,
    PubMedSource,
    SemanticScholarSource,
    discover,
)


@pytest.mark.integration
def test_semantic_scholar_real_search_readonly():
    src = SemanticScholarSource()
    try:
        results = src.search("QKI RNA-binding protein alternative splicing", limit=5)
    except Exception as exc:  # noqa: BLE001 - offline / rate-limited
        pytest.skip(f"Semantic Scholar unavailable: {exc}")
    assert all(isinstance(c, Candidate) for c in results)
    if results:
        assert results[0].source == "semanticscholar"
        assert results[0].title


@pytest.mark.integration
def test_pubmed_real_search_readonly():
    src = PubMedSource()
    try:
        results = src.search("QKI alternative splicing cardiomyocyte", limit=5)
    except Exception as exc:  # noqa: BLE001 - offline / rate-limited
        pytest.skip(f"PubMed unavailable: {exc}")
    assert all(isinstance(c, Candidate) for c in results)
    if results:
        assert results[0].source == "pubmed"
        assert results[0].title


@pytest.mark.integration
def test_discover_real_sources_merge_readonly():
    sources = [SemanticScholarSource(), PubMedSource()]
    try:
        out = discover(
            "QKI controls alternative splicing during cardiomyocyte differentiation",
            sources=sources,
            library=[],
            limit=8,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"discovery sources unavailable: {exc}")
    # No duplicate DOIs across merged sources; all are Candidates.
    dois = [c.doi for c in out if c.doi]
    assert len(dois) == len(set(dois))
    assert all(isinstance(c, Candidate) for c in out)
