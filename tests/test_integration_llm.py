"""Integration tests for the REAL LLM backends. Run explicitly: pytest -m integration

Deselected by default (``-m 'not integration'`` in pyproject). These require a
running LOCAL OpenAI-compatible server (LM Studio / Ollama) and/or a configured
ANTHROPIC_API_KEY + the optional ``[cloud]`` extra — none of which exist in the
build — so they SKIP rather than fail when their backend is unavailable.

The cloud test, if it runs, sends prose to Anthropic; it is opt-in by design and
only executes when a key is present.
"""

from __future__ import annotations

import os

import pytest

from scholia.llm import CloudClaudeLLM, LLMUnavailable, LocalLLM


@pytest.mark.integration
def test_local_llm_real_completion():
    """Talks to a real local OpenAI-compatible server if one is running."""
    m = LocalLLM()  # default LM Studio http://localhost:1234/v1
    try:
        out = m.complete(
            "You output a single short pointer line.",
            "Suggest one gap for: QKI alternative splicing.",
        )
    except LLMUnavailable as exc:
        pytest.skip(f"No local LLM server running: {exc}")
    assert isinstance(out, str)


@pytest.mark.integration
def test_cloud_claude_real_completion():
    """Sends a tiny prompt to Anthropic IF a key + the [cloud] extra are present."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; cloud path is opt-in")
    m = CloudClaudeLLM()
    try:
        out = m.complete(
            "You output a single short pointer line; never prose.",
            "Suggest one gap for: QKI alternative splicing.",
        )
    except LLMUnavailable as exc:
        pytest.skip(f"Cloud backend unavailable: {exc}")
    assert isinstance(out, str)
