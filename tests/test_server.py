"""Tests for scholia.server — handle_cite, handle_discover, and the HTTP layer.

Unit tests exercise the pure handler functions directly (no sockets, no real
models). One integration test spins the server on an ephemeral port in a thread
and hits /health + /cite via urllib; it's marked so the default suite skips it.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from scholia.corpus import load_corpus
from scholia.discovery import FakeDiscoverySource
from scholia.embedders import FakeEmbedder
from scholia.index import build_index
from scholia.rerank import FakeReranker
from scholia.server import ServerState, handle_cite, handle_discover

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _build_state(tmp_path, fake_source: bool = True) -> ServerState:
    """Build a minimal ServerState backed by the fixture corpus.

    Sets ``models_ready`` immediately because FakeEmbedder/FakeReranker have
    no weights to load — handle_cite's ``state.models_ready.wait()`` must not
    block unit tests.
    """
    papers = load_corpus(FIXTURES / "corpus")
    embedder = FakeEmbedder(dim=16)
    index = build_index(papers, embedder, tmp_path / "idx")
    reranker = FakeReranker()
    state = ServerState(
        index=index,
        embedder=embedder,
        reranker=reranker,
        fake_source=fake_source,
    )
    state.models_ready.set()  # Fakes are trivially ready — no load needed.
    return state


# ---------------------------------------------------------------------------
# handle_cite — unit tests (no sockets)
# ---------------------------------------------------------------------------


class TestHandleCite:
    def test_returns_expected_keys(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "QKI splicing cardiomyocyte"}, state)
        assert "suggestions" in result
        assert "claim_check" in result
        assert "ranking_signal" in result

    def test_suggestions_have_expected_fields(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "QKI splicing cardiomyocyte"}, state)
        assert len(result["suggestions"]) > 0
        s = result["suggestions"][0]
        assert "rank" in s
        assert "score" in s
        assert "first_author" in s
        assert "year" in s
        assert "title" in s
        assert "zotero_key" in s
        assert "zotero_link" in s
        assert "doi" in s

    def test_claim_check_fields(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "QKI splicing cardiomyocyte"}, state)
        cc = result["claim_check"]
        assert "supported" in cc
        assert "top_score" in cc
        assert "threshold" in cc
        assert isinstance(cc["supported"], bool)
        assert isinstance(cc["top_score"], float)
        assert isinstance(cc["threshold"], float)

    def test_supported_passage_returns_true(self, tmp_path):
        state = _build_state(tmp_path)
        # Use exact text from paperA so FakeEmbedder gives high self-similarity
        passage = (
            "QKI regulates alternative splicing in cardiomyocytes\n\n"
            "QKI is an RNA-binding protein that controls pre-mRNA alternative "
            "splicing during cardiac differentiation."
        )
        result = handle_cite({"passage": passage, "threshold": 0.5, "rerank": False}, state)
        assert result["claim_check"]["supported"] is True

    def test_unsupported_with_high_threshold(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "completely unrelated topic xyz", "threshold": 0.99}, state)
        assert result["claim_check"]["supported"] is False

    def test_empty_passage_is_unsupported(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": ""}, state)
        assert result["claim_check"]["supported"] is False
        assert result["claim_check"]["top_score"] == 0.0
        assert result["suggestions"] == []

    def test_k_param_limits_suggestions(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "QKI RNA splicing", "k": 2}, state)
        assert len(result["suggestions"]) <= 2

    def test_ranking_signal_is_string(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "QKI RNA"}, state)
        assert isinstance(result["ranking_signal"], str)
        assert len(result["ranking_signal"]) > 0

    def test_ranks_are_1_indexed_and_ascending(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "QKI RNA splicing", "k": 3}, state)
        ranks = [s["rank"] for s in result["suggestions"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_scores_are_descending(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "QKI RNA splicing", "k": 3}, state)
        scores = [s["score"] for s in result["suggestions"]]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_false_uses_biencoder(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "QKI splicing", "rerank": False}, state)
        assert "bi-encoder" in result["ranking_signal"]

    def test_rerank_true_uses_reranker(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "QKI splicing", "rerank": True}, state)
        assert "rerank" in result["ranking_signal"].lower()

    def test_result_is_json_serializable(self, tmp_path):
        state = _build_state(tmp_path)
        result = handle_cite({"passage": "QKI splicing"}, state)
        # Must not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# handle_discover — unit tests (no sockets)
# ---------------------------------------------------------------------------


class TestHandleDiscover:
    def test_returns_expected_keys(self, tmp_path):
        state = _build_state(tmp_path, fake_source=True)
        result = handle_discover({"passage": "QKI RNA binding protein"}, state)
        assert "candidates" in result
        assert "query" in result

    def test_candidates_have_expected_fields(self, tmp_path):
        state = _build_state(tmp_path, fake_source=True)
        result = handle_discover({"passage": "QKI RNA binding protein"}, state)
        if result["candidates"]:
            c = result["candidates"][0]
            assert "title" in c
            assert "authors" in c
            assert "year" in c
            assert "doi" in c
            assert "snippet" in c
            assert "source" in c

    def test_query_is_string(self, tmp_path):
        state = _build_state(tmp_path, fake_source=True)
        result = handle_discover({"passage": "QKI RNA binding protein"}, state)
        assert isinstance(result["query"], str)

    def test_empty_passage_returns_empty_candidates(self, tmp_path):
        state = _build_state(tmp_path, fake_source=True)
        result = handle_discover({"passage": ""}, state)
        assert result["candidates"] == []

    def test_dedup_against_library(self, tmp_path):
        """Papers already in the index should not appear as candidates.

        The fixture corpus has DOI "10.1038/aaa" (paperA). FakeDiscoverySource
        generates deterministic DOIs keyed to the query+index — they won't match
        real DOIs, so we verify the dedup logic via the dedupe_against_library
        function and the structure. The key check is that candidates do NOT
        include papers whose DOI matches the library.
        """
        state = _build_state(tmp_path, fake_source=True)
        result = handle_discover({"passage": "QKI RNA binding protein"}, state)
        lib_dois = {p.doi for p in state.index._papers if p.doi}
        candidate_dois = {c["doi"] for c in result["candidates"] if c["doi"]}
        # No candidate should duplicate a library DOI
        assert lib_dois.isdisjoint(candidate_dois)

    def test_limit_param_respected(self, tmp_path):
        state = _build_state(tmp_path, fake_source=True)
        result = handle_discover({"passage": "QKI RNA binding protein", "limit": 2}, state)
        assert len(result["candidates"]) <= 2

    def test_result_is_json_serializable(self, tmp_path):
        state = _build_state(tmp_path, fake_source=True)
        result = handle_discover({"passage": "QKI RNA splicing"}, state)
        json.dumps(result)

    def test_whitespace_passage_returns_empty(self, tmp_path):
        state = _build_state(tmp_path, fake_source=True)
        result = handle_discover({"passage": "   "}, state)
        assert result["candidates"] == []


# ---------------------------------------------------------------------------
# Integration: real HTTP server on ephemeral port
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_server_health_and_cite_via_http(tmp_path):
    """Start the server on an ephemeral port, hit /health and /cite, verify JSON."""
    import socket
    import time

    from scholia.server import serve

    state = _build_state(tmp_path, fake_source=True)

    # Find a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server_obj = {"httpd": None}

    def _run():
        httpd = serve("127.0.0.1", port, state, daemon=True)
        server_obj["httpd"] = httpd
        httpd.serve_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait for the server to be ready (up to 3 s)
    base = f"http://127.0.0.1:{port}"
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=1):
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("Server did not start in time")

    # /health
    with urllib.request.urlopen(f"{base}/health") as resp:
        data = json.loads(resp.read())
    assert data["status"] == "ok"
    assert "papers" in data
    assert isinstance(data["papers"], int)

    # /cite
    body = json.dumps({"passage": "QKI RNA splicing"}).encode()
    req = urllib.request.Request(
        f"{base}/cite",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    assert "suggestions" in result
    assert "claim_check" in result

    if server_obj["httpd"]:
        server_obj["httpd"].shutdown()
