"""Unit tests for the discovery feature (find papers NOT in your library).

All tests here are deterministic and offline: they use ``FakeDiscoverySource``
and mock the ingest subprocess. The real Semantic Scholar / PubMed search lives
behind ``@pytest.mark.integration`` in test_integration_discovery.py (deselected
by default), and performs a READ-ONLY search only — never an add.
"""

from __future__ import annotations

from scholia.discovery import (
    Candidate,
    DiscoverySource,
    FakeDiscoverySource,
    _contact_email,
    _user_agent,
    build_query,
    dedupe_against_library,
    discover,
)
from scholia.models import Paper


# --- Contact email (privacy: no personal address shipped) ---

def test_contact_email_default_is_generic_not_personal(monkeypatch):
    """The shipped default contact is a generic project address, never personal."""
    monkeypatch.delenv("SCHOLIA_CONTACT_EMAIL", raising=False)
    email = _contact_email()
    assert "@" in email
    assert "utmb" not in email.lower()
    assert "gsdewson" not in email.lower()


def test_contact_email_env_override(monkeypatch):
    """SCHOLIA_CONTACT_EMAIL overrides the generic default."""
    monkeypatch.setenv("SCHOLIA_CONTACT_EMAIL", "me@example.org")
    assert _contact_email() == "me@example.org"
    assert "me@example.org" in _user_agent()


# --- Candidate dataclass ---

def test_candidate_is_a_small_record():
    c = Candidate(
        title="A paper",
        authors=["Doe, Jane"],
        year="2021",
        doi="10.1000/xyz",
        abstract_snippet="snippet",
        source="semanticscholar",
        score=0.9,
    )
    assert c.title == "A paper"
    assert c.doi == "10.1000/xyz"
    assert c.source == "semanticscholar"
    assert c.score == 0.9


# --- FakeDiscoverySource determinism ---

def test_fake_source_is_deterministic():
    """Same query + limit -> identical results across calls (no RNG, no network)."""
    src = FakeDiscoverySource()
    a = src.search("QKI alternative splicing", limit=5)
    b = src.search("QKI alternative splicing", limit=5)
    assert a == b
    assert all(isinstance(c, Candidate) for c in a)


def test_fake_source_satisfies_protocol():
    assert isinstance(FakeDiscoverySource(), DiscoverySource)


def test_fake_source_respects_limit():
    src = FakeDiscoverySource()
    assert len(src.search("anything", limit=2)) <= 2
    assert len(src.search("anything", limit=0)) == 0


def test_fake_source_query_influences_results():
    """Different queries yield different (query-aware) candidates."""
    src = FakeDiscoverySource()
    a = src.search("QKI splicing", limit=5)
    b = src.search("ribosome biogenesis", limit=5)
    assert a != b


# --- Query construction (privacy-aware) ---

def test_build_query_extracts_key_terms_not_raw_passage():
    """The query is a short keyword string, NOT the verbatim draft passage."""
    passage = (
        "In this study we found that the protein QKI clearly regulates "
        "alternative splicing of many transcripts during the differentiation "
        "of cardiomyocytes, and this is very important for the field."
    )
    q = build_query(passage)
    assert q != passage
    assert len(q) < len(passage)
    # Content words survive; common stopwords/filler do not.
    low = q.lower()
    assert "qki" in low
    assert "splicing" in low
    assert " the " not in f" {low} "
    assert "very" not in low.split()


def test_build_query_is_deterministic():
    passage = "QKI regulates alternative splicing in cardiomyocytes."
    assert build_query(passage) == build_query(passage)


def test_build_query_empty_passage():
    assert build_query("") == ""
    assert build_query("   \n\t") == ""


def test_build_query_caps_term_count():
    passage = " ".join(f"term{i}" for i in range(50))
    q = build_query(passage, max_terms=6)
    assert len(q.split()) <= 6


# --- Dedup against the library ---

def _lib_paper(doi="", title=""):
    return Paper(id="x", title=title, doi=doi)


def test_dedupe_drops_candidate_already_in_library_by_doi():
    library = [_lib_paper(doi="10.1000/known", title="Known paper")]
    cands = [
        Candidate("Known paper", [], "2020", "10.1000/known", "", "pubmed", 1.0),
        Candidate("Novel paper", [], "2021", "10.1000/novel", "", "pubmed", 0.9),
    ]
    out = dedupe_against_library(cands, library)
    assert [c.doi for c in out] == ["10.1000/novel"]


def test_dedupe_doi_match_is_case_and_prefix_insensitive():
    """DOIs compare case-insensitively and ignore a https://doi.org/ prefix."""
    library = [_lib_paper(doi="10.1000/Known")]
    cands = [
        Candidate("A", [], "", "https://doi.org/10.1000/known", "", "s2", 1.0),
        Candidate("B", [], "", "10.1000/OTHER", "", "s2", 0.5),
    ]
    out = dedupe_against_library(cands, library)
    assert [c.doi for c in out] == ["10.1000/OTHER"]


def test_dedupe_drops_candidate_already_in_library_by_title():
    """A DOI-less candidate already present (title match) is filtered out."""
    library = [_lib_paper(title="QKI regulates alternative splicing")]
    cands = [
        Candidate("QKI Regulates Alternative Splicing", [], "", "", "", "pubmed", 1.0),
        Candidate("A different paper entirely", [], "", "", "", "pubmed", 0.5),
    ]
    out = dedupe_against_library(cands, library)
    assert [c.title for c in out] == ["A different paper entirely"]


def test_dedupe_dedupes_candidates_among_themselves():
    """Two candidates with the same DOI collapse to one (merge across sources)."""
    cands = [
        Candidate("P", [], "2020", "10.1000/dup", "", "semanticscholar", 0.8),
        Candidate("P", [], "2020", "10.1000/dup", "", "pubmed", 0.7),
    ]
    out = dedupe_against_library(cands, [])
    assert len(out) == 1


def test_dedupe_empty_library_keeps_all():
    cands = [
        Candidate("A", [], "", "10.1/a", "", "s2", 1.0),
        Candidate("B", [], "", "10.1/b", "", "s2", 0.9),
    ]
    assert len(dedupe_against_library(cands, [])) == 2


# --- discover() orchestration (with a Fake source + a library) ---

def test_discover_returns_only_new_candidates_ranked():
    """discover surfaces only candidates NOT in the library, ranked by score desc."""
    # The FakeDiscoverySource derives its DOIs deterministically from the query;
    # we put one of its candidates into the 'library' and assert it is filtered.
    src = FakeDiscoverySource()
    raw = src.search(build_query("QKI splicing"), limit=8)
    assert raw, "fixture sanity: fake source returns candidates"
    already = raw[0]
    library = [_lib_paper(doi=already.doi, title=already.title)]

    out = discover("QKI splicing", sources=[src], library=library, limit=8)
    dois = [c.doi for c in out]
    assert already.doi not in dois
    # Ranked by descending score.
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)


def test_discover_merges_multiple_sources_and_dedupes():
    src1 = FakeDiscoverySource(source_name="semanticscholar")
    src2 = FakeDiscoverySource(source_name="pubmed")
    out = discover("QKI splicing", sources=[src1, src2], library=[], limit=8)
    dois = [c.doi for c in out]
    # No duplicate DOIs survive the merge.
    assert len(dois) == len(set(dois))


def test_discover_respects_limit():
    src = FakeDiscoverySource()
    out = discover("QKI splicing", sources=[src], library=[], limit=3)
    assert len(out) <= 3
