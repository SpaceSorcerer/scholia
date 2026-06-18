from scholia.models import Paper
from scholia.retrieval import Hit
from scholia.grounding import (
    format_citation_suggestions,
    claim_check,
    ClaimVerdict,
)


def _hit(score, key="AAAAAAAA"):
    p = Paper(
        id=key,
        title="QKI regulates splicing",
        authors=["Chen, Xinyun"],
        year="2021",
        doi="10.1038/aaa",
        zotero_key=key,
        zotero_link=f"zotero://select/library/items/{key}",
        abstract="",
        tags=[],
    )
    return Hit(paper=p, score=score)


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
