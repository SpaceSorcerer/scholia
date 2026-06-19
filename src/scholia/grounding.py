"""Format retrieval hits as cite-grounding suggestions and a claim-check verdict."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    The entailment fields describe the second, independent check, now run over
    **all retrieved hits** (top-k aggregation):

    - ``checked``      — whether an entailment pass actually ran (False when
                          there are no retrieved papers to verify against).
    - ``entailed``     — True if ANY of the top-k papers' texts supports the
                          claim above ``entail_threshold``.
    - ``entail_score`` — the BEST (highest) support score across all verified
                          papers (in ``[0, 1]``).
    - ``entail_threshold`` — the cutoff used for ``entailed``.
    - ``supporting_papers`` — the subset of retrieved papers whose text scored
                          >= ``entail_threshold``; ordered by descending support
                          score. Empty when none entail.

    ``retrieved_but_not_supported`` is the honest flag this whole layer exists
    for: retrieval said SUPPORTED, but NONE of the top-k papers' texts clearly
    support the claim. It is NEVER a "contradicts" claim — only "verify the
    source." ``status`` collapses the two checks into one of:
    ``"SUPPORTED"`` (similarity yes AND any paper entails),
    ``"RETRIEVED_NOT_SUPPORTED"`` (similarity yes, no paper entails), or
    ``"UNSUPPORTED"`` (similarity no).
    """

    base: ClaimVerdict
    checked: bool
    entailed: bool
    entail_score: float
    entail_threshold: float
    supporting_papers: tuple[Paper, ...] = field(default_factory=tuple)

    @property
    def retrieved_but_not_supported(self) -> bool:
        return self.base.supported and self.checked and not self.entailed

    @property
    def supported(self) -> bool:
        """True only when similarity AND any top-k paper's entailment agree (or
        entailment did not run, in which case we defer to the similarity verdict)."""
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
    """Run the similarity/rerank ``claim_check`` then a top-k textual support check.

    Rather than verifying only the single top hit, this aggregates entailment
    across ALL retrieved hits (up to the ``k`` already fetched). The claim is
    deemed SUPPORTED if ANY hit's ``embedding_text`` (title + abstract) scores
    >= ``entail_threshold``. Only when NONE of the top-k papers entail is the
    honest "retrieved but not clearly supported — verify the source" flag set.

    ``VerifiedClaimVerdict.supporting_papers`` lists the papers (ordered by
    descending support score) that crossed the threshold, so the caller can
    show the user which paper actually supports the claim, not just which ranks
    highest by similarity. When there is no retrieved paper at all, no entailment
    pass runs (``checked=False``) and the verdict defers to similarity.
    """
    base = claim_check(hits, threshold=threshold)
    if not hits:
        return VerifiedClaimVerdict(
            base=base, checked=False, entailed=False,
            entail_score=0.0, entail_threshold=entail_threshold,
            supporting_papers=(),
        )

    # Verify each hit independently; collect (score, paper) pairs.
    scored: list[tuple[float, Paper]] = []
    for hit in hits:
        result = checker.verify(claim, hit.paper.embedding_text)
        scored.append((result.score, hit.paper))

    best_score = max(s for s, _ in scored)
    supporting = sorted(
        ((s, p) for s, p in scored if s >= entail_threshold),
        key=lambda x: -x[0],
    )
    entailed = len(supporting) > 0
    supporting_papers = tuple(p for _, p in supporting)

    return VerifiedClaimVerdict(
        base=base,
        checked=True,
        entailed=entailed,
        entail_score=best_score,
        entail_threshold=entail_threshold,
        supporting_papers=supporting_papers,
    )
