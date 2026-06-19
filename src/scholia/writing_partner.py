"""Writing-partner: structure/gap SUGGESTIONS grounded in the user's own library.

``suggest_gaps`` helps the user see what their draft is MISSING — missing-but-
library-covered subtopics, claims that appear to need a citation, and suggested
next angles — each as a short POINTER, grounded in the papers they actually have.

HARD INTEGRITY RULE (enforced in the system prompt + the parser)
----------------------------------------------------------------
This feature SUGGESTS and FLAGS; it NEVER generates manuscript prose and NEVER
rewrites the user's sentences. The system prompt below explicitly forbids writing
prose/sentences for the user. No codepath here emits drafted text: the model's
output is parsed ONLY into lists of short pointer strings (see ``_parse_report``),
and any line that looks like a multi-sentence draft is dropped rather than echoed.

PRIVACY
-------
Retrieval and prompt-building are local. WHICH backend runs the prompt is the
caller's choice; the default is the on-device ``LocalLLM`` (and ``FakeLLM`` in
tests). The cloud path is opt-in/off and gated at the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scholia.embedders import Embedder
from scholia.index import ScholiaIndex
from scholia.llm import LanguageModel
from scholia.models import Paper
from scholia.retrieval import Hit, retrieve

# --- The integrity contract: the no-prose system prompt ---------------------
#
# This string is load-bearing. The test-suite asserts that it contains the
# explicit no-draft instruction; the CLI/feature relies on it to keep the model
# in "suggest, don't write" mode. Edit with care.
SYSTEM_PROMPT = (
    "You are a scientific writing PARTNER that helps an author find gaps in a "
    "draft passage, grounded ONLY in the author's own reference library.\n"
    "\n"
    "STRICT INTEGRITY RULE: You suggest what to address and which of their papers "
    "is relevant; do not draft text. You MUST NOT write prose or sentences for "
    "the user, MUST NOT rewrite, paraphrase, or continue their passage, and MUST "
    "NOT ghost-write any manuscript text. You only point at what is missing.\n"
    "\n"
    "Return SHORT POINTERS ONLY (a few words each, not sentences of manuscript "
    "prose), under exactly these three headers, each pointer on its own line "
    "beginning with '- ':\n"
    "MISSING TOPICS:\n"
    "  (subtopics the passage omits that the author's library DOES cover)\n"
    "NEEDS CITATION:\n"
    "  (specific claims in the passage that appear to need a supporting citation)\n"
    "NEXT ANGLES:\n"
    "  (directions the author could address next, grounded in their library)\n"
    "\n"
    "Base every pointer on the passage and the provided LIBRARY CONTEXT. Do not "
    "invent papers, findings, numbers, or citations. If the library does not "
    "cover something, do not claim it does."
)

_HEADERS = {
    "MISSING TOPICS": "missing_topics",
    "NEEDS CITATION": "needs_citation",
    "NEXT ANGLES": "next_angles",
}

# A pointer should be a short flag, not a drafted sentence. We defensively cap
# length and sentence-count so a model that ignores the contract cannot smuggle
# manuscript prose through the parser into the user's view.
_MAX_POINTER_CHARS = 240
_MAX_POINTER_SENTENCES = 2


@dataclass(frozen=True)
class GapReport:
    """Structured, prose-free gap suggestions for a passage.

    Every field is a list of short pointer strings — never drafted manuscript
    text. ``supporting_papers`` are the library papers retrieved as context for
    the suggestions (the grounding), each with the bi-encoder relevance score.
    """

    passage: str
    missing_topics: list[str] = field(default_factory=list)
    needs_citation: list[str] = field(default_factory=list)
    next_angles: list[str] = field(default_factory=list)
    supporting_papers: list[Hit] = field(default_factory=list)

    @property
    def all_suggestions(self) -> list[str]:
        return [*self.missing_topics, *self.needs_citation, *self.next_angles]

    @property
    def is_empty(self) -> bool:
        return not self.all_suggestions


def _format_library_context(hits: list[Hit]) -> str:
    """Render retrieved library papers as a compact, grounded context block."""
    if not hits:
        return "(The author's library returned no relevant papers for this passage.)"
    lines = []
    for i, hit in enumerate(hits, 1):
        p = hit.paper
        first_author = (
            p.authors[0].split(",")[0].strip() if p.authors else "Unknown"
        )
        lines.append(f"[{i}] {first_author} ({p.year}) — {p.title}")
    return "\n".join(lines)


def _build_user_prompt(passage: str, hits: list[Hit]) -> str:
    """Assemble the user payload: the passage + the grounded library context."""
    return (
        "PASSAGE (the author's draft — do NOT rewrite it; only suggest gaps):\n"
        f"{passage}\n"
        "\n"
        "LIBRARY CONTEXT (the author's own papers most relevant to this passage):\n"
        f"{_format_library_context(hits)}\n"
        "\n"
        "Give SHORT POINTERS ONLY under MISSING TOPICS / NEEDS CITATION / NEXT "
        "ANGLES, as instructed. Do not write any prose for the author."
    )


def _looks_like_prose(text: str) -> bool:
    """Heuristic guard: reject lines that look like drafted manuscript prose.

    A genuine pointer is short and flag-like. A drafted sentence is long and/or
    multi-sentence. This is a defensive backstop so that even a misbehaving model
    cannot route manuscript prose through the parser to the user.
    """
    if len(text) > _MAX_POINTER_CHARS:
        return True
    # Count sentence-terminators followed by a space + capital (rough sentence
    # boundary). More than _MAX_POINTER_SENTENCES => treat as prose, drop it.
    sentence_breaks = 0
    for i in range(1, len(text) - 1):
        if text[i] in ".!?" and text[i + 1] == " ":
            sentence_breaks += 1
    return sentence_breaks >= _MAX_POINTER_SENTENCES


def _parse_report(passage: str, raw: str, hits: list[Hit]) -> GapReport:
    """Parse the model's text into a GapReport of short pointers only.

    Splits on the three known headers; under each header, collects ``- `` bullet
    lines as pointers, dropping anything that looks like drafted prose. Anything
    that is not a recognized header or a bullet is ignored — the parser NEVER
    passes through free-form model prose.
    """
    buckets: dict[str, list[str]] = {v: [] for v in _HEADERS.values()}
    current: str | None = None
    for line in (raw or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Header line? (tolerate a trailing colon / surrounding markdown).
        header_key = stripped.rstrip(":").upper().strip("# ").strip()
        if header_key in _HEADERS:
            current = _HEADERS[header_key]
            continue
        if current is None:
            continue
        # Pointer line: must be a bullet to count (keeps stray prose out).
        if stripped[0] in "-*•":
            pointer = stripped[1:].strip()
            if pointer and not _looks_like_prose(pointer):
                buckets[current].append(pointer)
    return GapReport(
        passage=passage,
        missing_topics=buckets["missing_topics"],
        needs_citation=buckets["needs_citation"],
        next_angles=buckets["next_angles"],
        supporting_papers=hits,
    )


def suggest_gaps(
    passage: str,
    index: ScholiaIndex,
    embedder: Embedder,
    model: LanguageModel,
    *,
    k: int = 8,
) -> GapReport:
    """Suggest gaps in ``passage``, grounded in the user's library.

    Pipeline:
      1. Retrieve the ``k`` most relevant library papers for the passage (reuses
         ``retrieve`` — the same on-device bi-encoder path as ``cite``).
      2. Build a prompt giving the model the passage + that library context, with
         a system prompt that forbids writing prose (suggest, don't draft).
      3. Parse the model output into a ``GapReport`` of short pointers, dropping
         anything that looks like drafted prose.

    An empty/whitespace passage carries no draft to analyze and returns an empty
    ``GapReport`` without calling the model (matches ``retrieve``'s short-circuit).
    """
    if not passage or not passage.strip():
        return GapReport(passage=passage)
    hits = retrieve(passage, embedder, index, k=k)
    raw = model.complete(SYSTEM_PROMPT, _build_user_prompt(passage, hits))
    return _parse_report(passage, raw, hits)


def _first_author(paper: Paper) -> str:
    if not paper.authors:
        return "Unknown"
    return paper.authors[0].split(",")[0].strip()


def format_gap_report(report: GapReport) -> str:
    """Render a GapReport for the terminal: pointers + supporting papers.

    NEVER prints rewritten prose — only the parsed pointer lists and the library
    papers that ground them.
    """
    lines = [f"Passage: {report.passage}", ""]
    if report.is_empty:
        lines.append(
            "No gap suggestions (empty passage, or the model returned nothing "
            "actionable)."
        )
    else:
        sections = [
            ("Missing topics (your library covers, your draft omits):",
             report.missing_topics),
            ("Claims that appear to need a citation:", report.needs_citation),
            ("Suggested next angles:", report.next_angles),
        ]
        for title, items in sections:
            if not items:
                continue
            lines.append(title)
            for it in items:
                lines.append(f"  - {it}")
            lines.append("")

    lines.append("Supporting papers from your library:")
    if not report.supporting_papers:
        lines.append("  (none retrieved)")
    else:
        for rank, hit in enumerate(report.supporting_papers, 1):
            p = hit.paper
            lines.append(
                f"  {rank}. [{hit.score:.3f}] {_first_author(p)} "
                f"({p.year}) — {p.title}"
            )
            if p.zotero_key:
                lines.append(f"     key={p.zotero_key}  {p.zotero_link}")
    lines.append("")
    lines.append(
        "Note: these are SUGGESTIONS (gaps + where citations are needed), not "
        "drafted text. Scholia never writes manuscript prose for you."
    )
    return "\n".join(lines)
