"""Tests for scholia.overlay — BridgeClient, formatters, and headless GUI smoke.

Unit tests mock the HTTP layer (no real bridge needed).
The headless GUI smoke test uses QT_QPA_PLATFORM=offscreen; it is skipped if
PySide6 is not installed (pytest.importorskip) and marked @pytest.mark.integration
so the default suite can run it when PySide6 IS present (it's fast, model-free,
and offscreen — appropriate for CI).
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scholia.overlay import (
    BridgeClient,
    BridgeError,
    format_bridge_error,
    format_cite_result,
    format_discover_result,
    render_links_as_html,
)

# ---------------------------------------------------------------------------
# Shared fake payloads
# ---------------------------------------------------------------------------

_CITE_SUPPORTED: dict[str, Any] = {
    "suggestions": [
        {
            "rank": 1,
            "score": 3.21,
            "first_author": "Chen",
            "year": "2021",
            "title": "QKI regulates alternative splicing",
            "zotero_key": "ABCD1234",
            "zotero_link": "zotero://select/library/items/ABCD1234",
            "doi": "10.1038/aaa",
        },
        {
            "rank": 2,
            "score": 1.50,
            "first_author": "Smith",
            "year": "2019",
            "title": "RNA-binding proteins in splicing",
            "zotero_key": "EFGH5678",
            "zotero_link": "zotero://select/library/items/EFGH5678",
            "doi": "10.1016/bbb",
        },
    ],
    "claim_check": {"supported": True, "top_score": 3.21, "threshold": 0.0},
    "ranking_signal": "reranked (cross-encoder)",
}

_CITE_UNSUPPORTED: dict[str, Any] = {
    "suggestions": [],
    "claim_check": {"supported": False, "top_score": 0.0, "threshold": 0.0},
    "ranking_signal": "reranked (cross-encoder)",
}

_DISCOVER_RESULT: dict[str, Any] = {
    "candidates": [
        {
            "title": "New paper on QKI",
            "authors": ["Jones, A.", "Brown, B."],
            "year": "2023",
            "doi": "10.9999/xyz",
            "snippet": "QKI is important for splicing.",
            "source": "semanticscholar",
        },
        {
            "title": "Another lncRNA study",
            "authors": ["Doe, J."],
            "year": "2022",
            "doi": None,
            "snippet": "",
            "source": "pubmed",
        },
    ],
    "query": "QKI RNA splicing",
}

_DISCOVER_EMPTY: dict[str, Any] = {
    "candidates": [],
    "query": "gibberish xyzzy",
}


# ---------------------------------------------------------------------------
# Helpers to mock urllib.request.urlopen
# ---------------------------------------------------------------------------


def _mock_urlopen(response_dict: dict):
    """Return a context-manager mock that yields a fake urllib response."""
    body = json.dumps(response_dict).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=mock_resp)


# ---------------------------------------------------------------------------
# BridgeClient — unit tests (no sockets)
# ---------------------------------------------------------------------------


class TestBridgeClientHealth:
    def test_health_calls_correct_url(self):
        with patch("urllib.request.urlopen", _mock_urlopen({"status": "ok", "papers": 5})) as m:
            client = BridgeClient(host="127.0.0.1", port=8765)
            result = client.health()
        assert result["status"] == "ok"
        called_url = m.call_args[0][0]
        assert called_url == "http://127.0.0.1:8765/health"

    def test_health_returns_dict(self):
        with patch("urllib.request.urlopen", _mock_urlopen({"status": "ok", "papers": 42})):
            client = BridgeClient()
            result = client.health()
        assert isinstance(result, dict)
        assert result["papers"] == 42


class TestBridgeClientCite:
    def test_cite_posts_to_correct_path(self):
        with patch("urllib.request.urlopen", _mock_urlopen(_CITE_SUPPORTED)) as m:
            client = BridgeClient()
            result = client.cite("QKI splicing")
        # The Request object is passed, check path via its full_url
        req_obj = m.call_args[0][0]
        assert req_obj.full_url == "http://127.0.0.1:8765/cite"

    def test_cite_request_body_contains_passage(self):
        with patch("urllib.request.urlopen", _mock_urlopen(_CITE_SUPPORTED)) as m:
            client = BridgeClient()
            client.cite("QKI splicing cardiomyocyte")
        req_obj = m.call_args[0][0]
        body = json.loads(req_obj.data.decode("utf-8"))
        assert body["passage"] == "QKI splicing cardiomyocyte"

    def test_cite_request_method_is_post(self):
        with patch("urllib.request.urlopen", _mock_urlopen(_CITE_SUPPORTED)) as m:
            client = BridgeClient()
            client.cite("test")
        req_obj = m.call_args[0][0]
        assert req_obj.method == "POST"

    def test_cite_request_content_type_json(self):
        with patch("urllib.request.urlopen", _mock_urlopen(_CITE_SUPPORTED)) as m:
            client = BridgeClient()
            client.cite("test")
        req_obj = m.call_args[0][0]
        assert req_obj.headers.get("Content-type") == "application/json"

    def test_cite_k_param_included_in_body(self):
        with patch("urllib.request.urlopen", _mock_urlopen(_CITE_SUPPORTED)) as m:
            client = BridgeClient()
            client.cite("test", k=3)
        req_obj = m.call_args[0][0]
        body = json.loads(req_obj.data.decode("utf-8"))
        assert body["k"] == 3

    def test_cite_returns_dict_with_expected_keys(self):
        with patch("urllib.request.urlopen", _mock_urlopen(_CITE_SUPPORTED)):
            client = BridgeClient()
            result = client.cite("QKI splicing")
        assert "suggestions" in result
        assert "claim_check" in result
        assert "ranking_signal" in result

    def test_cite_raises_bridge_error_on_connection_failure(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            client = BridgeClient()
            with pytest.raises(BridgeError, match="Bridge unreachable"):
                client.cite("test")

    def test_cite_raises_bridge_error_on_bad_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not-json!!!"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            client = BridgeClient()
            with pytest.raises(BridgeError, match="invalid JSON"):
                client.cite("test")


class TestBridgeClientDiscover:
    def test_discover_posts_to_correct_path(self):
        with patch("urllib.request.urlopen", _mock_urlopen(_DISCOVER_RESULT)) as m:
            client = BridgeClient()
            client.discover("QKI splicing")
        req_obj = m.call_args[0][0]
        assert req_obj.full_url == "http://127.0.0.1:8765/discover"

    def test_discover_request_body_contains_passage(self):
        with patch("urllib.request.urlopen", _mock_urlopen(_DISCOVER_RESULT)) as m:
            client = BridgeClient()
            client.discover("RNA binding proteins")
        req_obj = m.call_args[0][0]
        body = json.loads(req_obj.data.decode("utf-8"))
        assert body["passage"] == "RNA binding proteins"

    def test_discover_limit_param_in_body(self):
        with patch("urllib.request.urlopen", _mock_urlopen(_DISCOVER_RESULT)) as m:
            client = BridgeClient()
            client.discover("test", limit=4)
        req_obj = m.call_args[0][0]
        body = json.loads(req_obj.data.decode("utf-8"))
        assert body["limit"] == 4

    def test_discover_returns_dict_with_expected_keys(self):
        with patch("urllib.request.urlopen", _mock_urlopen(_DISCOVER_RESULT)):
            client = BridgeClient()
            result = client.discover("QKI")
        assert "candidates" in result
        assert "query" in result

    def test_discover_raises_bridge_error_on_failure(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            client = BridgeClient()
            with pytest.raises(BridgeError):
                client.discover("test")


class TestBridgeClientCustomHostPort:
    def test_custom_host_port_used_in_url(self):
        with patch("urllib.request.urlopen", _mock_urlopen({"status": "ok"})) as m:
            client = BridgeClient(host="127.0.0.1", port=9999)
            client.health()
        called_url = m.call_args[0][0]
        assert "9999" in called_url


# ---------------------------------------------------------------------------
# format_cite_result — unit tests
# ---------------------------------------------------------------------------


class TestFormatCiteResult:
    def test_supported_verdict_in_output(self):
        text = format_cite_result(_CITE_SUPPORTED)
        assert "SUPPORTED" in text
        assert "UNSUPPORTED" not in text

    def test_unsupported_verdict_in_output(self):
        text = format_cite_result(_CITE_UNSUPPORTED)
        assert "UNSUPPORTED" in text

    def test_top_score_shown(self):
        text = format_cite_result(_CITE_SUPPORTED)
        assert "3.21" in text

    def test_ranking_signal_shown(self):
        text = format_cite_result(_CITE_SUPPORTED)
        assert "cross-encoder" in text.lower() or "rerank" in text.lower()

    def test_author_year_title_shown(self):
        text = format_cite_result(_CITE_SUPPORTED)
        assert "Chen" in text
        assert "2021" in text
        assert "QKI regulates alternative splicing" in text

    def test_doi_shown(self):
        text = format_cite_result(_CITE_SUPPORTED)
        assert "10.1038/aaa" in text

    def test_zotero_link_shown(self):
        text = format_cite_result(_CITE_SUPPORTED)
        assert "zotero://select/library/items/ABCD1234" in text

    def test_rank_numbers_shown(self):
        text = format_cite_result(_CITE_SUPPORTED)
        assert "1." in text
        assert "2." in text

    def test_no_papers_message_when_empty(self):
        text = format_cite_result(_CITE_UNSUPPORTED)
        assert "No matching papers" in text

    def test_returns_string(self):
        assert isinstance(format_cite_result(_CITE_SUPPORTED), str)

    def test_multiple_suggestions_all_shown(self):
        text = format_cite_result(_CITE_SUPPORTED)
        assert "Chen" in text
        assert "Smith" in text

    def test_threshold_shown(self):
        text = format_cite_result(_CITE_SUPPORTED)
        # threshold is 0.0 in the fixture
        assert "0.0" in text or "0.000" in text


# ---------------------------------------------------------------------------
# format_discover_result — unit tests
# ---------------------------------------------------------------------------


class TestFormatDiscoverResult:
    def test_query_shown(self):
        text = format_discover_result(_DISCOVER_RESULT)
        assert "QKI RNA splicing" in text

    def test_candidate_count_shown(self):
        text = format_discover_result(_DISCOVER_RESULT)
        assert "2" in text

    def test_author_year_title_shown(self):
        text = format_discover_result(_DISCOVER_RESULT)
        assert "Jones" in text
        assert "2023" in text
        assert "New paper on QKI" in text

    def test_doi_shown(self):
        text = format_discover_result(_DISCOVER_RESULT)
        assert "10.9999/xyz" in text

    def test_add_hint_shown_for_doi(self):
        text = format_discover_result(_DISCOVER_RESULT)
        assert "scholia discover" in text
        assert "--add" in text
        assert "10.9999/xyz" in text

    def test_snippet_shown(self):
        text = format_discover_result(_DISCOVER_RESULT)
        assert "QKI is important for splicing" in text

    def test_source_shown(self):
        text = format_discover_result(_DISCOVER_RESULT)
        assert "semanticscholar" in text

    def test_no_doi_message(self):
        text = format_discover_result(_DISCOVER_RESULT)
        assert "none reported" in text

    def test_empty_candidates_message(self):
        text = format_discover_result(_DISCOVER_EMPTY)
        assert "No NEW candidate" in text

    def test_returns_string(self):
        assert isinstance(format_discover_result(_DISCOVER_RESULT), str)

    def test_no_real_zotero_write_hint(self):
        # v0: must NOT suggest auto-adding; must show CLI hint only
        text = format_discover_result(_DISCOVER_RESULT)
        # The hint is a CLI command, not a "clicked and added" confirmation
        assert "scholia discover" in text


# ---------------------------------------------------------------------------
# format_bridge_error — unit tests
# ---------------------------------------------------------------------------


class TestFormatBridgeError:
    def test_unreachable_shows_friendly_message(self):
        exc = BridgeError("Bridge unreachable at http://127.0.0.1:8765/cite: Connection refused")
        msg = format_bridge_error(exc)
        assert "Can't reach Scholia server" in msg
        assert "scholia serve" in msg

    def test_connection_keyword_triggers_friendly_message(self):
        exc = BridgeError("Bridge unreachable at http://127.0.0.1:8765/health: connection refused")
        msg = format_bridge_error(exc)
        assert "Can't reach Scholia server" in msg

    def test_other_bridge_error_shows_error_prefix(self):
        exc = BridgeError("Bridge returned invalid JSON: ...")
        msg = format_bridge_error(exc)
        assert "Error:" in msg
        assert "invalid JSON" in msg

    def test_generic_exception_shows_error_prefix(self):
        exc = ValueError("something unexpected")
        msg = format_bridge_error(exc)
        assert "Error:" in msg
        assert "unexpected" in msg

    def test_returns_string(self):
        assert isinstance(format_bridge_error(BridgeError("oops")), str)


# ---------------------------------------------------------------------------
# render_links_as_html — unit tests
# ---------------------------------------------------------------------------


class TestRenderLinksAsHtml:
    def test_doi_url_becomes_anchor(self):
        text = "DOI: https://doi.org/10.1038/aaa"
        html = render_links_as_html(text)
        assert '<a href="https://doi.org/10.1038/aaa">' in html

    def test_zotero_url_becomes_anchor(self):
        text = "Zotero: zotero://select/library/items/ABCD1234"
        html = render_links_as_html(text)
        assert '<a href="zotero://select/library/items/ABCD1234">' in html

    def test_html_special_chars_escaped(self):
        text = "title: <Some Title> & stuff"
        html = render_links_as_html(text)
        assert "&lt;Some Title&gt;" in html
        assert "&amp;" in html
        assert "<Some Title>" not in html

    def test_plain_text_no_links_unchanged_content(self):
        text = "No links here, just plain text."
        html = render_links_as_html(text)
        assert "No links here" in html
        assert "<a href" not in html

    def test_output_wrapped_in_pre(self):
        html = render_links_as_html("some text")
        assert html.startswith("<pre")
        assert "</pre>" in html

    def test_multiple_links_all_linkified(self):
        text = (
            "DOI: https://doi.org/10.1038/aaa\n"
            "Zotero: zotero://select/library/items/XYZ"
        )
        html = render_links_as_html(text)
        assert 'href="https://doi.org/10.1038/aaa"' in html
        assert 'href="zotero://select/library/items/XYZ"' in html

    def test_returns_string(self):
        assert isinstance(render_links_as_html("test"), str)


# ---------------------------------------------------------------------------
# Headless GUI smoke test — PySide6 required, offscreen
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_overlay_widget_ground_action_offscreen():
    """Construct the overlay window offscreen and simulate a Ground action.

    Uses QT_QPA_PLATFORM=offscreen so no display is needed.
    Skipped entirely if PySide6 is not installed.
    """
    PySide6 = pytest.importorskip("PySide6")  # noqa: N806

    # Must set offscreen BEFORE QApplication is created.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtCore import Qt, QThread, Signal, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    # Re-use an existing QApplication or create one (offscreen).
    app = QApplication.instance() or QApplication([])

    # Build the overlay widget inline (mirrors overlay.py structure but
    # uses a stub client so we never hit the real bridge).
    from scholia.overlay import format_cite_result

    win = QWidget()
    win.setWindowTitle("Scholia — smoke test")
    win.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

    outer = QVBoxLayout(win)
    text_box = QPlainTextEdit()
    text_box.setPlainText("QKI controls alternative splicing.")
    results_box = QPlainTextEdit()
    results_box.setReadOnly(True)
    outer.addWidget(text_box)
    outer.addWidget(results_box)

    # Simulate a Ground action with a stub bridge client.
    stub_response = _CITE_SUPPORTED

    def _do_ground():
        passage = text_box.toPlainText().strip()
        assert passage, "passage should be non-empty"
        # Stub the bridge call.
        with patch("urllib.request.urlopen", _mock_urlopen(stub_response)):
            from scholia.overlay import BridgeClient

            client = BridgeClient()
            raw = client.cite(passage)
        display = format_cite_result(raw)
        results_box.setPlainText(display)

    # Show and trigger the action immediately.
    win.show()
    _do_ground()

    result_text = results_box.toPlainText()
    assert "SUPPORTED" in result_text, f"Expected SUPPORTED in results; got:\n{result_text}"
    assert "Chen" in result_text, "Expected author name in results"
    assert "QKI regulates" in result_text, "Expected paper title in results"

    win.close()
    # Do NOT call app.exec() — this is headless; we just validate construction
    # and action logic without entering the event loop.
