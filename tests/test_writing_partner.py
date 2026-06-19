"""Unit tests for the writing-partner gap/structure suggestion feature.

Fully offline + model-free: FakeEmbedder + FakeLLM, an in-memory index over the
fixture corpus. Asserts (1) suggest_gaps retrieves library context and returns a
GapReport of short pointers grounded in the user's papers, and (2) the INTEGRITY
contract: the system prompt forbids drafting prose, and no codepath emits drafted
manuscript text.
"""

from __future__ import annotations

from pathlib import Path

from scholia.corpus import load_corpus
from scholia.embedders import FakeEmbedder
from scholia.index import build_index
from scholia.llm import FakeLLM
from scholia.writing_partner import (
    SYSTEM_PROMPT,
    GapReport,
    format_gap_report,
    suggest_gaps,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _index(tmp_path):
    papers = load_corpus(FIXTURES / "corpus")
    embedder = FakeEmbedder()
    index = build_index(papers, embedder, tmp_path / "idx")
    return index, embedder


# --- suggest_gaps: retrieval + structured GapReport ---

def test_suggest_gaps_returns_gap_report(tmp_path):
    index, embedder = _index(tmp_path)
    report = suggest_gaps(
        "QKI regulates alternative splicing in cardiomyocytes",
        index, embedder, FakeLLM(), k=3,
    )
    assert isinstance(report, GapReport)


def test_suggest_gaps_retrieves_library_context(tmp_path):
    """The report is GROUNDED: it carries the retrieved library papers as support."""
    index, embedder = _index(tmp_path)
    report = suggest_gaps(
        "QKI regulates alternative splicing in cardiomyocytes",
        index, embedder, FakeLLM(), k=3,
    )
    assert report.supporting_papers, "expected retrieved library papers"
    assert len(report.supporting_papers) <= 3
    # Each support is a real library Paper with an id from the fixture corpus.
    assert all(h.paper.id for h in report.supporting_papers)


def test_suggest_gaps_produces_pointer_suggestions(tmp_path):
    """The report contains short pointer suggestions parsed from the model."""
    index, embedder = _index(tmp_path)
    report = suggest_gaps(
        "QKI regulates alternative splicing in cardiomyocytes",
        index, embedder, FakeLLM(), k=3,
    )
    assert report.all_suggestions, "expected at least one suggestion pointer"
    # Pointers are short flags, NOT drafted manuscript prose.
    for s in report.all_suggestions:
        assert isinstance(s, str) and s.strip()
        assert len(s) <= 240
    # The three categories are wired up from the structured stub.
    assert report.missing_topics
    assert report.needs_citation
    assert report.next_angles


def test_suggest_gaps_is_deterministic(tmp_path):
    index, embedder = _index(tmp_path)
    a = suggest_gaps("QKI splicing", index, embedder, FakeLLM(), k=3)
    b = suggest_gaps("QKI splicing", index, embedder, FakeLLM(), k=3)
    assert a.all_suggestions == b.all_suggestions


def test_suggest_gaps_empty_passage_short_circuits(tmp_path):
    """An empty passage returns an empty report WITHOUT calling the model."""

    class _ExplodingLLM:
        def complete(self, system: str, user: str) -> str:  # pragma: no cover
            raise AssertionError("model must not be called for an empty passage")

    index, embedder = _index(tmp_path)
    report = suggest_gaps("   \n\t", index, embedder, _ExplodingLLM())
    assert report.is_empty
    assert report.supporting_papers == []


# --- INTEGRITY contract: suggest, never write prose ---

def test_system_prompt_forbids_drafting_prose():
    """The no-prose instruction MUST be present in the system prompt (the contract)."""
    low = SYSTEM_PROMPT.lower()
    assert "do not draft text" in low
    assert "must not write prose" in low or "not write prose" in low
    # Names the suggest-not-write framing explicitly.
    assert "suggest what to address and which of their papers is relevant" in low


def test_user_prompt_passes_passage_and_library_without_requesting_prose(tmp_path):
    """The user payload tells the model to suggest gaps, not to rewrite the draft."""
    from scholia.writing_partner import _build_user_prompt
    from scholia.retrieval import retrieve

    index, embedder = _index(tmp_path)
    hits = retrieve("QKI splicing", embedder, index, k=2)
    user = _build_user_prompt("QKI splicing", hits)
    low = user.lower()
    assert "do not write any prose" in low or "do not rewrite" in low
    # The library context is included (grounding).
    assert "library context" in low


def test_parser_drops_prose_lines_keeps_pointers():
    """A model that smuggles a long drafted sentence has it DROPPED, not echoed."""
    from scholia.writing_partner import _parse_report

    drafted = (
        "Here is a fully drafted manuscript paragraph that the model should never "
        "have written for the user. It contains multiple sentences of polished "
        "prose. It even concludes with a flourish that reads like real text."
    )
    raw = (
        "MISSING TOPICS:\n"
        "- mechanism of QKI binding\n"
        f"- {drafted}\n"
        "NEEDS CITATION:\n"
        "- the splicing claim needs a source\n"
    )
    report = _parse_report("p", raw, [])
    # Short pointers survive; the drafted prose line is filtered out.
    assert "mechanism of QKI binding" in report.missing_topics
    assert drafted not in report.missing_topics
    assert "the splicing claim needs a source" in report.needs_citation


def test_format_gap_report_emits_pointers_and_papers_not_prose(tmp_path):
    """The terminal renderer prints pointers + supporting papers, never drafted prose,
    and states the assist-not-ghostwrite note."""
    index, embedder = _index(tmp_path)
    report = suggest_gaps(
        "QKI regulates alternative splicing in cardiomyocytes",
        index, embedder, FakeLLM(), k=3,
    )
    text = format_gap_report(report)
    assert "Supporting papers from your library:" in text
    assert "never writes manuscript prose" in text.lower()
    # The pointers appear; no rewritten-passage block is emitted.
    assert "Missing topics" in text
