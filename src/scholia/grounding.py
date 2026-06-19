"""Format retrieval hits as cite-grounding suggestions and a claim-check verdict."""

from __future__ import annotations

from dataclasses import dataclass

from scholia.entailment import EntailmentChecker
from scholia.models import Paper
from scholia.retrieval import Hit


def _first_author(paper: Paper) -> str:
    if not paper.authors:
        return "Unknown"
    return paper.authors[0].split(",")[0].strip()


def format_citation_suggestions(passage: str, hits: list[Hit]) -> str:
    lines = [f"Passage: {passage}", ""]
    if not hits:
        lines.append("No supporting papers found in your library.")
        return "\n".join(lines)
    lines.append("Supporting papers from your library:")
    for rank, hit in enumerate(hits, 1):
        p = hit.paper
        lines.append(
            f"  {rank}. [{hit.score:.3f}] {_first_author(p)} ({p.year}) — {p.title}"
        )
        lines.append(f"     key={p.zotero_key}  {p.zotero_link}")
        if p.doi:
            lines.append(f"     doi: https://doi.org/{p.doi}")
    return "\n".join(lines)


@dataclass(frozen=True)
class ClaimVerdict:
    supported: bool
    top_score: float
    threshold: float
    top_paper: Paper | None


def claim_check(hits: list[Hit], threshold: float = 0.45) -> ClaimVerdict:
    if not hits:
        return ClaimVerdict(False, 0.0, threshold, None)
    top = hits[0]
    return ClaimVerdict(top.score >= threshold, top.score, threshold, top.paper)


@dataclass(frozen=True)
class VerifiedClaimVerdict:
    """A claim-check verdict enriched with a textual support-verification pass.

    ``base`` is the similarity/rerank verdict (unchanged from ``claim_check``).
    The entailment fields describe the second, independent check:

    - ``checked``      — whether an entailment pass actually ran (False when
                          there is no retrieved paper to verify against).
    - ``entailed``     — the entailment checker's boolean support verdict.
    - ``entail_score`` — the support probability in ``[0, 1]``.
    - ``entail_threshold`` — the cutoff used for ``entailed``.

    ``retrieved_but_not_supported`` is the honest flag this whole layer exists
    for: retrieval said SUPPORTED, but the top paper's text does not clearly
    support the claim. It is NEVER a "contradicts" claim — only "verify the
    source." ``status`` collapses the two checks into one of:
    ``"SUPPORTED"`` (both agree), ``"RETRIEVED_NOT_SUPPORTED"`` (similarity yes,
    entailment no), or ``"UNSUPPORTED"`` (similarity no).
    """

    base: ClaimVerdict
    checked: bool
    entailed: bool
    entail_score: float
    entail_threshold: float

    @property
    def retrieved_but_not_supported(self) -> bool:
        return self.base.supported and self.checked and not self.entailed

    @property
    def supported(self) -> bool:
        """True only when similarity AND entailment agree (or entailment did not
        run, in which case we defer to the similarity verdict)."""
        if not self.base.supported:
            return False
        if not self.checked:
            return True
        return self.entailed

    @property
    def status(self) -> str:
        if not self.base.supported:
            return "UNSUPPORTED"
        if self.retrieved_but_not_supported:
            return "RETRIEVED_NOT_SUPPORTED"
        return "SUPPORTED"


def verified_claim_check(
    hits: list[Hit],
    checker: EntailmentChecker,
    claim: str,
    threshold: float = 0.45,
    entail_threshold: float = 0.5,
) -> VerifiedClaimVerdict:
    """Run the similarity/rerank ``claim_check`` then a textual support check.

    The entailment pass verifies the TOP hit only: it asks whether that paper's
    ``embedding_text`` (title + abstract) actually supports ``claim``, rather than
    merely scoring similar. When similarity says SUPPORTED but the top paper does
    not clearly support the claim, ``retrieved_but_not_supported`` is set so the
    caller can surface the honest "verify the source" flag. When there is no
    retrieved paper, no entailment pass runs (``checked=False``) and the verdict
    defers to similarity.
    """
    base = claim_check(hits, threshold=threshold)
    if base.top_paper is None:
        return VerifiedClaimVerdict(
            base=base, checked=False, entailed=False,
            entail_score=0.0, entail_threshold=entail_threshold,
        )
    result = checker.verify(claim, base.top_paper.embedding_text)
    entailed = result.score >= entail_threshold
    return VerifiedClaimVerdict(
        base=base,
        checked=True,
        entailed=entailed,
        entail_score=result.score,
        entail_threshold=entail_threshold,
    )
