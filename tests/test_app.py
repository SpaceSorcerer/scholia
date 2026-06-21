"""Unit tests for scholia.app non-GUI logic.

Tests cover:
- build_card_data: top-N cap, field extraction, link building
- build_verdict: SUPPORTED / NOT CLEARLY SUPPORTED labels
- build_discover_cards: card field extraction
- build_query_label: query extraction
- _results_html: loading/error/idle/results states (no Qt needed)
- _card_html: HTML structure (no Qt needed)
- GroundingEngine: constructor, ready flag, n_papers

No real bridge, no Qt, no model download.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared fake payloads (same structure as the real /cite and /discover responses)
# ---------------------------------------------------------------------------

_SUGGESTIONS_5 = [
    {
        "rank": i,
        "score": 5.0 - i * 0.5,
        "first_author": f"Author{i}",
        "year": str(2020 + i),
        "title": f"Paper number {i}",
        "zotero_key": f"KEY{i:04d}",
        "zotero_link": f"zotero://select/library/items/KEY{i:04d}",
        "doi": f"10.9999/paper{i}",
    }
    for i in range(1, 6)
]

_SUGGESTIONS_8 = _SUGGESTIONS_5 + [
    {
        "rank": i,
        "score": 1.0 - i * 0.1,
        "first_author": f"Extra{i}",
        "year": "2019",
        "title": f"Extra paper {i}",
        "zotero_key": f"XKEY{i:04d}",
        "zotero_link": f"zotero://select/library/items/XKEY{i:04d}",
        "doi": None,
    }
    for i in range(6, 9)
]

_CITE_SUPPORTED: dict[str, Any] = {
    "suggestions": _SUGGESTIONS_5,
    "claim_check": {"supported": True, "top_score": 4.5, "threshold": 0.0},
    "ranking_signal": "reranked (cross-encoder)",
}

_CITE_UNSUPPORTED: dict[str, Any] = {
    "suggestions": [],
    "claim_check": {"supported": False, "top_score": -2.1, "threshold": 0.0},
    "ranking_signal": "reranked (cross-encoder)",
}

_DISCOVER_RESULT: dict[str, Any] = {
    "candidates": [
        {
            "title": "New lncRNA study",
            "authors": ["Jones, A.", "Brown, B."],
            "year": "2023",
            "doi": "10.9999/new",
            "snippet": "QKI is important.",
            "source": "semanticscholar",
        },
        {
            "title": "Another paper",
            "authors": [],
            "year": None,
            "doi": None,
            "snippet": "",
            "source": "pubmed",
        },
    ],
    "query": "QKI splicing",
}


# ---------------------------------------------------------------------------
# Import the pure functions (no Qt, no ML, no network)
# ---------------------------------------------------------------------------

from scholia.app import (
    TOP_N,
    build_card_data,
    build_discover_cards,
    build_query_label,
    build_verdict,
    _results_html,
    _card_html,
    GroundingEngine,
)


# ---------------------------------------------------------------------------
# build_card_data
# ---------------------------------------------------------------------------


class TestBuildCardData:
    def test_top_n_cap_applied(self):
        cards = build_card_data(_SUGGESTIONS_8, top_n=5)
        assert len(cards) == 5

    def test_fewer_than_top_n(self):
        cards = build_card_data(_SUGGESTIONS_5[:3], top_n=5)
        assert len(cards) == 3

    def test_empty_suggestions(self):
        assert build_card_data([]) == []

    def test_rank_extracted(self):
        cards = build_card_data(_SUGGESTIONS_5)
        assert cards[0]["rank"] == 1
        assert cards[4]["rank"] == 5

    def test_title_extracted(self):
        cards = build_card_data(_SUGGESTIONS_5)
        assert cards[0]["title"] == "Paper number 1"

    def test_first_author_extracted(self):
        cards = build_card_data(_SUGGESTIONS_5)
        assert cards[0]["first_author"] == "Author1"

    def test_year_extracted_as_string(self):
        cards = build_card_data(_SUGGESTIONS_5)
        assert cards[0]["year"] == "2021"

    def test_doi_url_built_from_doi(self):
        cards = build_card_data(_SUGGESTIONS_5)
        assert cards[0]["doi_url"] == "https://doi.org/10.9999/paper1"

    def test_empty_doi_gives_empty_url(self):
        sug = [
            {
                "rank": 1,
                "score": 1.0,
                "first_author": "Doe",
                "year": "2020",
                "title": "No DOI paper",
                "zotero_key": "ZK",
                "zotero_link": "zotero://select/library/items/ZK",
                "doi": None,
            }
        ]
        cards = build_card_data(sug)
        assert cards[0]["doi_url"] == ""

    def test_zotero_url_extracted(self):
        cards = build_card_data(_SUGGESTIONS_5)
        assert cards[0]["zotero_url"] == "zotero://select/library/items/KEY0001"

    def test_score_formatted_to_3dp(self):
        cards = build_card_data(_SUGGESTIONS_5)
        # score is "4.500" or similar — 3 decimal places
        assert "." in cards[0]["score"]
        assert len(cards[0]["score"].split(".")[1]) == 3

    def test_default_top_n_constant(self):
        cards = build_card_data(_SUGGESTIONS_8)
        assert len(cards) == TOP_N

    def test_no_doi_no_url(self):
        sug = [{"rank": 1, "score": 0.5, "first_author": "X", "year": "2020",
                 "title": "T", "zotero_key": "K", "zotero_link": "", "doi": ""}]
        cards = build_card_data(sug)
        assert cards[0]["doi_url"] == ""


# ---------------------------------------------------------------------------
# build_verdict
# ---------------------------------------------------------------------------


class TestBuildVerdict:
    def test_supported_returns_true_and_label(self):
        supported, label = build_verdict({"supported": True, "top_score": 3.5, "threshold": 0.0})
        assert supported is True
        assert "SUPPORTED" in label
        assert "NOT" not in label

    def test_unsupported_returns_false_and_label(self):
        supported, label = build_verdict({"supported": False, "top_score": -2.0, "threshold": 0.0})
        assert supported is False
        assert "NOT CLEARLY SUPPORTED" in label or "⚠" in label

    def test_score_shown_in_label(self):
        _, label = build_verdict({"supported": True, "top_score": 2.345, "threshold": 0.0})
        assert "2.345" in label

    def test_threshold_shown_in_label(self):
        _, label = build_verdict({"supported": False, "top_score": -1.0, "threshold": 0.5})
        assert "0.5" in label or "0.500" in label

    def test_empty_dict_gives_unsupported(self):
        supported, _ = build_verdict({})
        assert supported is False

    def test_returns_tuple(self):
        result = build_verdict({"supported": True, "top_score": 1.0, "threshold": 0.0})
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# build_discover_cards
# ---------------------------------------------------------------------------


class TestBuildDiscoverCards:
    def test_count_matches_candidates(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert len(cards) == 2

    def test_rank_starts_at_1(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert cards[0]["rank"] == 1
        assert cards[1]["rank"] == 2

    def test_title_extracted(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert cards[0]["title"] == "New lncRNA study"

    def test_first_author_extracted(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert cards[0]["first_author"] == "Jones"  # last name from "Jones, A."

    def test_year_extracted_as_string(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert cards[0]["year"] == "2023"

    def test_doi_url_built(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert cards[0]["doi_url"] == "https://doi.org/10.9999/new"

    def test_no_doi_gives_empty_url(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert cards[1]["doi_url"] == ""

    def test_source_extracted(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert cards[0]["source"] == "semanticscholar"

    def test_snippet_extracted(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert "QKI" in cards[0]["snippet"]

    def test_empty_authors_gives_unknown(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert cards[1]["first_author"] == "Unknown"

    def test_year_none_gives_nd(self):
        cards = build_discover_cards(_DISCOVER_RESULT["candidates"])
        assert cards[1]["year"] == "n.d."

    def test_empty_list(self):
        assert build_discover_cards([]) == []


# ---------------------------------------------------------------------------
# build_query_label
# ---------------------------------------------------------------------------


class TestBuildQueryLabel:
    def test_returns_query(self):
        assert build_query_label({"query": "QKI splicing"}) == "QKI splicing"

    def test_missing_query_returns_empty(self):
        assert build_query_label({}) == ""

    def test_none_query_returns_empty(self):
        assert build_query_label({"query": None}) == ""

    def test_returns_string(self):
        assert isinstance(build_query_label({"query": "x"}), str)


# ---------------------------------------------------------------------------
# _results_html and _card_html (pure HTML generation, no Qt)
# ---------------------------------------------------------------------------


class TestResultsHtml:
    def test_loading_state(self):
        html = _results_html(loading=True)
        assert "Thinking" in html

    def test_error_state(self):
        html = _results_html(error="Something went wrong")
        assert "Something went wrong" in html
        assert "Error" in html

    def test_idle_state(self):
        html = _results_html()
        # Should show the idle/hint text, no data
        assert "Ctrl+Alt+G" in html or "Ground" in html

    def test_supported_badge_present(self):
        html = _results_html(cite_result=_CITE_SUPPORTED)
        assert "SUPPORTED" in html

    def test_unsupported_badge_present(self):
        html = _results_html(cite_result=_CITE_UNSUPPORTED)
        assert "NOT CLEARLY SUPPORTED" in html or "⚠" in html

    def test_paper_title_in_html(self):
        html = _results_html(cite_result=_CITE_SUPPORTED)
        assert "Paper number 1" in html

    def test_discover_section_present(self):
        html = _results_html(discover_result=_DISCOVER_RESULT)
        assert "Discovery" in html or "New lncRNA" in html

    def test_doi_link_in_html(self):
        html = _results_html(cite_result=_CITE_SUPPORTED)
        assert "https://doi.org/10.9999/paper1" in html

    def test_zotero_link_in_html(self):
        html = _results_html(cite_result=_CITE_SUPPORTED)
        assert "zotero://select/library/items/KEY0001" in html

    def test_top_n_cap_honoured(self):
        # 8 suggestions but only TOP_N should appear as cards
        cite_8 = {**_CITE_SUPPORTED, "suggestions": _SUGGESTIONS_8}
        html = _results_html(cite_result=cite_8)
        # TOP_N = 5; paper 6+ should not appear
        assert "Extra paper 6" not in html

    def test_no_papers_message_when_empty(self):
        html = _results_html(cite_result=_CITE_UNSUPPORTED)
        assert "No matching papers" in html

    def test_no_new_candidates_message(self):
        html = _results_html(discover_result={"candidates": [], "query": "x"})
        assert "No new candidates" in html or "No NEW candidate" in html or "already be in your library" in html

    def test_returns_string(self):
        assert isinstance(_results_html(), str)


class TestCardHtml:
    def test_title_in_card(self):
        card = build_card_data(_SUGGESTIONS_5)[0]
        html = _card_html(card, is_cite=True)
        assert "Paper number 1" in html

    def test_doi_link_in_card(self):
        card = build_card_data(_SUGGESTIONS_5)[0]
        html = _card_html(card, is_cite=True)
        assert "https://doi.org/10.9999/paper1" in html

    def test_zotero_link_in_cite_card(self):
        card = build_card_data(_SUGGESTIONS_5)[0]
        html = _card_html(card, is_cite=True)
        assert "zotero://" in html

    def test_zotero_link_absent_in_discover_card(self):
        card = build_discover_cards(_DISCOVER_RESULT["candidates"])[0]
        html = _card_html(card, is_cite=False)
        assert "zotero://" not in html

    def test_html_escaped(self):
        card = {
            "rank": 1,
            "title": "<script>alert('xss')</script>",
            "first_author": "Hacker",
            "year": "2024",
            "score": "0.999",
            "doi_url": "",
            "zotero_url": "",
        }
        html = _card_html(card, is_cite=True)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_returns_string(self):
        card = build_card_data(_SUGGESTIONS_5)[0]
        assert isinstance(_card_html(card), str)


# ---------------------------------------------------------------------------
# GroundingEngine — no model download
# ---------------------------------------------------------------------------


class TestGroundingEngine:
    def test_initially_not_ready(self, tmp_path):
        engine = GroundingEngine(tmp_path)
        assert not engine.ready

    def test_n_papers_zero_before_load(self, tmp_path):
        engine = GroundingEngine(tmp_path)
        assert engine.n_papers == 0

    def test_load_async_calls_callback(self, tmp_path):
        """load_async must call on_done (with error since no real index)."""
        engine = GroundingEngine(tmp_path)
        results = []
        done = threading.Event()

        def _cb(err):
            results.append(err)
            done.set()

        engine.load_async(_cb)
        done.wait(timeout=5.0)

        assert len(results) == 1
        # No real index in tmp_path → should get an error message
        assert results[0] is not None
        assert "index" in results[0].lower() or "load" in results[0].lower() or "No" in results[0]

    def test_load_async_with_mocked_state(self, tmp_path):
        """load_async with a mocked load_state that succeeds."""
        engine = GroundingEngine(tmp_path)
        done_event = __import__("threading").Event()
        results = []

        mock_state = MagicMock()
        mock_state.index._papers = [1, 2, 3]  # 3 fake papers

        def _cb(err):
            results.append(err)
            done_event.set()

        with patch("scholia.server.load_state", return_value=mock_state):
            engine.load_async(_cb)
            done_event.wait(timeout=5.0)

        assert results[0] is None  # no error
        assert engine.ready
        assert engine.n_papers == 3

    def test_cite_raises_when_not_ready(self, tmp_path):
        engine = GroundingEngine(tmp_path)
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.cite("test")

    def test_discover_raises_when_not_ready(self, tmp_path):
        engine = GroundingEngine(tmp_path)
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.discover("test")

    def test_cite_calls_handle_cite(self, tmp_path):
        """cite() should call handle_cite with the passage."""
        engine = GroundingEngine(tmp_path)
        mock_state = MagicMock()
        mock_state.index._papers = []
        engine._state = mock_state
        engine._loaded.set()

        mock_result = {"suggestions": [], "claim_check": {}, "ranking_signal": ""}
        with patch("scholia.server.handle_cite", return_value=mock_result) as m:
            engine.cite("QKI splicing")

        m.assert_called_once()
        call_req = m.call_args[0][0]
        assert call_req["passage"] == "QKI splicing"

    def test_discover_calls_handle_discover(self, tmp_path):
        """discover() should call handle_discover with the passage."""
        engine = GroundingEngine(tmp_path)
        mock_state = MagicMock()
        mock_state.index._papers = []
        engine._state = mock_state
        engine._loaded.set()

        mock_result = {"candidates": [], "query": "QKI"}
        with patch("scholia.server.handle_discover", return_value=mock_result) as m:
            engine.discover("QKI splicing")

        m.assert_called_once()
        call_req = m.call_args[0][0]
        assert call_req["passage"] == "QKI splicing"


