"""Unit tests for the pluggable language-model backends (offline, model-free).

FakeLLM is deterministic and needs no network/model. LocalLLM/CloudClaudeLLM are
exercised only for their guarded-failure behaviour here (server down / SDK
missing); the real-server / real-API paths live behind @pytest.mark.integration
in test_integration_llm.py (deselected by default).
"""

from __future__ import annotations

import pytest

from scholia.llm import (
    CloudClaudeLLM,
    FakeLLM,
    LanguageModel,
    LLMUnavailable,
    LocalLLM,
)


# --- FakeLLM determinism + protocol ---

def test_fake_llm_satisfies_protocol():
    assert isinstance(FakeLLM(), LanguageModel)


def test_fake_llm_is_deterministic():
    """Same (system, user) -> identical output across calls (no RNG, no network)."""
    m = FakeLLM()
    a = m.complete("sys", "QKI regulates alternative splicing")
    b = m.complete("sys", "QKI regulates alternative splicing")
    assert a == b
    assert isinstance(a, str) and a


def test_fake_llm_input_influences_output():
    """Different user payloads yield different (input-derived) stubs."""
    m = FakeLLM()
    a = m.complete("sys", "QKI splicing")
    b = m.complete("sys", "ribosome biogenesis")
    assert a != b


def test_fake_llm_returns_structured_pointer_sections():
    """The stub is structured pointers (not prose) under the expected headers."""
    out = FakeLLM().complete("sys", "QKI splicing in cardiomyocytes")
    assert "MISSING TOPICS:" in out
    assert "NEEDS CITATION:" in out
    assert "NEXT ANGLES:" in out
    # Pointers are bullet lines, not paragraphs of prose.
    assert "- " in out


# --- LocalLLM (guarded failure; no real server in the build) ---

def test_local_llm_construction_normalizes_base_url():
    m = LocalLLM(base_url="http://localhost:1234/v1/", model="x")
    assert m._endpoint() == "http://localhost:1234/v1/chat/completions"


def test_local_llm_raises_clean_error_when_server_down():
    """With no local server, complete() raises a clear LLMUnavailable (no traceback
    leakage), so callers can degrade gracefully."""
    # A port we don't serve on -> connection refused -> LLMUnavailable.
    m = LocalLLM(base_url="http://127.0.0.1:9/v1", model="x", timeout=2)
    with pytest.raises(LLMUnavailable):
        m.complete("system", "user")


# --- CloudClaudeLLM (guarded failure; no SDK/key required in the build) ---

def test_cloud_llm_is_constructible_without_anthropic_installed():
    """Constructing CloudClaudeLLM must NOT require the optional 'anthropic' SDK
    (it is imported lazily inside complete())."""
    m = CloudClaudeLLM(model="claude-opus-4-8")
    assert m.model == "claude-opus-4-8"


def test_cloud_llm_complete_raises_clean_error_without_sdk(monkeypatch):
    """If 'anthropic' is not importable, complete() raises LLMUnavailable telling
    the user to install the [cloud] extra — never a bare ImportError traceback."""
    import builtins

    real_import = builtins.__import__

    def _no_anthropic(name, *args, **kwargs):
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_anthropic)
    with pytest.raises(LLMUnavailable) as exc:
        CloudClaudeLLM().complete("system", "user")
    assert "cloud" in str(exc.value).lower()
