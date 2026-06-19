from scholia.entailment import EntailmentChecker, EntailmentResult, FakeEntailmentChecker
from scholia.models import Paper
from scholia.retrieval import Hit
from scholia.grounding import (
    format_citation_suggestions,
    claim_check,
    ClaimVerdict,
    VerifiedClaimVerdict,
    verified_claim_check,
)


def _hit(score, key="AAAAAAAA", title="QKI regulates splicing", abstract=""):
    p = Paper(
        id=key,
        title=title,
        authors=["Chen, Xinyun"],
        year="2021",
        doi="10.1038/aaa",
        zotero_key=key,
        zotero_link=f"zotero://select/library/items/{key}",
        abstract=abstract,
        tags=[],
    )
    return Hit(paper=p, score=score)


class _AlwaysSupports:
    """EntailmentChecker stub: every claim is supported (score 1.0)."""

    def verify(self, claim: str, evidence: str) -> EntailmentResult:
        return EntailmentResult(supported=True, score=1.0)


class _NeverSupports:
    """EntailmentChecker stub: nothing is supported (score 0.0)."""

    def verify(self, claim: str, evidence: str) -> EntailmentResult:
        return EntailmentResult(supported=False, score=0.0)


def test_format_lists_each_hit_with_links():
    out = format_citation_suggestions("QKI controls splicing.", [_hit(0.91)])
    assert "QKI controls splicing." in out
    assert "0.910" in out
    assert "Chen" in out and "2021" in out
    assert "AAAAAAAA" in out
    assert "zotero://select/library/items/AAAAAAAA" in out
    assert "10.1038/aaa" in out


def test_format_handles_no_hits():
    out = format_citation_suggestions("Unfounded claim.", [])
    assert "No supporting papers found in your library." in out


def test_claim_check_supported_above_threshold():
    v = claim_check([_hit(0.80)], threshold=0.45)
    assert isinstance(v, ClaimVerdict)
    assert v.supported is True
    assert v.top_score == 0.80
    assert v.top_paper.id == "AAAAAAAA"


def test_claim_check_unsupported_below_threshold():
    v = claim_check([_hit(0.20)], threshold=0.45)
    assert v.supported is False
    assert v.top_score == 0.20


def test_claim_check_unsupported_when_no_hits():
    v = claim_check([], threshold=0.45)
    assert v.supported is False
    assert v.top_score == 0.0
    assert v.top_paper is None


# --- verified_claim_check: similarity verdict + textual support verification ---


def test_verified_supported_when_similarity_and_entailment_agree():
    """Similarity says SUPPORTED and entailment confirms -> SUPPORTED."""
    v = verified_claim_check(
        [_hit(0.80)], _AlwaysSupports(), claim="QKI regulates splicing",
        threshold=0.45, entail_threshold=0.5,
    )
    assert isinstance(v, VerifiedClaimVerdict)
    assert v.base.supported is True
    assert v.checked is True
    assert v.entailed is True
    assert v.supported is True
    assert v.retrieved_but_not_supported is False
    assert v.status == "SUPPORTED"


def test_verified_flags_retrieved_but_not_supported():
    """Similarity says SUPPORTED but entailment does NOT -> the honest flag.

    This is the whole point of the layer: a high-similarity hit whose text does
    not actually support the claim is flagged, never silently passed."""
    v = verified_claim_check(
        [_hit(0.90)], _NeverSupports(), claim="QKI causes disease X",
        threshold=0.45, entail_threshold=0.5,
    )
    assert v.base.supported is True
    assert v.entailed is False
    assert v.retrieved_but_not_supported is True
    assert v.supported is False
    assert v.status == "RETRIEVED_NOT_SUPPORTED"


def test_verified_unsupported_when_similarity_below_threshold():
    """Similarity says UNSUPPORTED -> UNSUPPORTED regardless of entailment."""
    v = verified_claim_check(
        [_hit(0.20)], _AlwaysSupports(), claim="unrelated claim",
        threshold=0.45, entail_threshold=0.5,
    )
    assert v.base.supported is False
    assert v.supported is False
    assert v.retrieved_but_not_supported is False
    assert v.status == "UNSUPPORTED"


def test_verified_no_hits_does_not_run_entailment():
    """No retrieved paper -> no entailment pass; defer to similarity verdict."""
    v = verified_claim_check(
        [], _AlwaysSupports(), claim="anything", threshold=0.45,
    )
    assert v.checked is False
    assert v.base.supported is False
    assert v.supported is False
    assert v.status == "UNSUPPORTED"


def test_verified_uses_top_paper_text_for_entailment():
    """Entailment is run against the TOP hit's title+abstract (embedding_text).

    A real FakeEntailmentChecker over a topical abstract should support; a
    high-similarity hit with an off-topic abstract should be flagged."""
    topical = _hit(0.90, key="GOOD",
                   title="QKI controls splicing",
                   abstract="QKI regulates alternative splicing of pre-mRNA")
    v_good = verified_claim_check(
        [topical], FakeEntailmentChecker(), claim="QKI regulates splicing",
        threshold=0.45, entail_threshold=0.5,
    )
    assert v_good.status == "SUPPORTED"

    off_topic = _hit(0.90, key="BAD",
                     title="Olive oil and diet",
                     abstract="the mediterranean diet lowers cardiovascular risk")
    v_bad = verified_claim_check(
        [off_topic], FakeEntailmentChecker(), claim="QKI regulates splicing",
        threshold=0.45, entail_threshold=0.5,
    )
    assert v_bad.status == "RETRIEVED_NOT_SUPPORTED"


def test_verified_records_entailment_score_and_threshold():
    v = verified_claim_check(
        [_hit(0.80)], _AlwaysSupports(), claim="c", threshold=0.45,
        entail_threshold=0.6,
    )
    assert v.entail_score == 1.0
    assert v.entail_threshold == 0.6


# --- top-k aggregation: SUPPORTED if ANY of the top-k papers entails ---


class _SupportsSecondOnly:
    """EntailmentChecker stub: only supports when evidence contains 'GOOD'."""

    def verify(self, claim: str, evidence: str) -> EntailmentResult:
        if "GOOD" in evidence:
            return EntailmentResult(supported=True, score=0.95)
        return EntailmentResult(supported=False, score=0.02)


def test_topk_supported_when_second_hit_entails():
    """Top-1 doesn't entail but top-2 does -> claim is SUPPORTED (not flagged)."""
    miss = _hit(0.90, key="MISS", title="Olive oil diet", abstract="MISS evidence")
    hit_good = _hit(0.85, key="GOOD", title="QKI splicing", abstract="GOOD abstract")
    v = verified_claim_check(
        [miss, hit_good], _SupportsSecondOnly(), claim="QKI splicing",
        threshold=0.45, entail_threshold=0.5,
    )
    assert v.base.supported is True
    assert v.entailed is True
    assert v.status == "SUPPORTED"
    assert v.retrieved_but_not_supported is False


def test_topk_supporting_papers_reports_entailing_paper():
    """supporting_papers lists only the paper(s) that crossed the threshold."""
    miss = _hit(0.90, key="MISS", title="Off-topic paper", abstract="MISS evidence")
    hit_good = _hit(0.85, key="GOOD", title="QKI splicing", abstract="GOOD abstract")
    v = verified_claim_check(
        [miss, hit_good], _SupportsSecondOnly(), claim="QKI splicing",
        threshold=0.45, entail_threshold=0.5,
    )
    assert len(v.supporting_papers) == 1
    assert v.supporting_papers[0].id == "GOOD"


def test_topk_flags_when_no_hit_entails():
    """None of the top-k papers entail -> RETRIEVED_NOT_SUPPORTED."""
    hits = [_hit(0.90, key="A"), _hit(0.80, key="B"), _hit(0.70, key="C")]
    v = verified_claim_check(
        hits, _NeverSupports(), claim="QKI regulates splicing",
        threshold=0.45, entail_threshold=0.5,
    )
    assert v.entailed is False
    assert v.status == "RETRIEVED_NOT_SUPPORTED"
    assert v.supporting_papers == ()


def test_topk_entail_score_is_best_across_all_hits():
    """entail_score is the BEST (highest) score across all top-k hits, not top-1."""
    hits = [
        _hit(0.9, key="A", title="Alpha paper", abstract="ALPHA content"),
        _hit(0.8, key="B", title="Beta paper", abstract="BETA content"),
        _hit(0.7, key="C", title="Gamma paper", abstract="GAMMA content"),
    ]

    class _SupportsBetaOnly:
        def verify(self, claim: str, evidence: str) -> EntailmentResult:
            if "BETA" in evidence:
                return EntailmentResult(supported=True, score=0.92)
            return EntailmentResult(supported=False, score=0.05)

    v = verified_claim_check(
        hits, _SupportsBetaOnly(), claim="c",
        threshold=0.45, entail_threshold=0.5,
    )
    assert v.entail_score == 0.92
    assert v.entailed is True


def test_topk_supporting_papers_ordered_by_descending_score():
    """When multiple papers entail, supporting_papers is sorted best-first."""

    class _ScoresByKey:
        _scores = {"HIGH": 0.99, "MED": 0.75, "LOW": 0.10}

        def verify(self, claim: str, evidence: str) -> EntailmentResult:
            for key, score in self._scores.items():
                if key in evidence:
                    return EntailmentResult(supported=score >= 0.5, score=score)
            return EntailmentResult(supported=False, score=0.0)

    hits = [
        _hit(0.9, key="HIGH", title="HIGH hit", abstract="HIGH evidence"),
        _hit(0.85, key="MED", title="MED hit", abstract="MED evidence"),
        _hit(0.80, key="LOW", title="LOW hit", abstract="LOW evidence"),
    ]
    v = verified_claim_check(
        hits, _ScoresByKey(), claim="c",
        threshold=0.45, entail_threshold=0.5,
    )
    keys = [p.id for p in v.supporting_papers]
    assert keys == ["HIGH", "MED"]  # LOW (0.10) below threshold; HIGH before MED
    assert v.entail_score == 0.99


def test_topk_single_hit_behaves_same_as_before():
    """Single-hit path (k=1) remains backward-compatible."""
    v = verified_claim_check(
        [_hit(0.80)], _AlwaysSupports(), claim="QKI regulates splicing",
        threshold=0.45, entail_threshold=0.5,
    )
    assert v.status == "SUPPORTED"
    assert len(v.supporting_papers) == 1


def test_topk_no_hits_returns_unchecked():
    """Empty hits list -> checked=False, supporting_papers=()."""
    v = verified_claim_check(
        [], _AlwaysSupports(), claim="anything",
        threshold=0.45, entail_threshold=0.5,
    )
    assert v.checked is False
    assert v.supporting_papers == ()
