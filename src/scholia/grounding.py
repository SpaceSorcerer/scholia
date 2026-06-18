"""Format retrieval hits as cite-grounding suggestions and a claim-check verdict."""

from __future__ import annotations

from dataclasses import dataclass

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
