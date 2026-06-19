"""Pluggable language-model backends for the writing-partner layer.

This mirrors ``embedders.py`` / ``rerank.py``: a minimal ``LanguageModel``
Protocol, a deterministic model-free ``FakeLLM`` for unit tests, a lazily-loaded
``LocalLLM`` that talks to a LOCAL OpenAI-compatible server (LM Studio / Ollama)
over stdlib ``urllib`` (no new core dependency), and a ``CloudClaudeLLM`` that
uses the ``anthropic`` SDK (an OPTIONAL ``[cloud]`` extra, imported lazily).

INTEGRITY BOUNDARY
------------------
These backends are dumb text-in/text-out pipes. The writing-partner's integrity
contract (suggest, never write prose) is enforced by the SYSTEM PROMPT in
``writing_partner.py`` plus the parser there — not by the transport here.

PRIVACY
-------
``FakeLLM`` and ``LocalLLM`` are fully on-device: nothing leaves the machine.
``CloudClaudeLLM`` is the ONLY backend that transmits the user's prose off-box
(to Anthropic). It is opt-in, default-off, and gated at the CLI behind an explicit
``--allow-cloud`` flag with a printed warning, because sending unpublished
manuscript text to a cloud provider requires the user's institutional sign-off.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class LanguageModel(Protocol):
    """A single-shot text-in/text-out chat model.

    The entire required surface is one method::

        complete(system: str, user: str) -> str

    ``system`` carries the role/instruction prompt (e.g. the no-prose integrity
    contract); ``user`` carries the task payload (the passage + retrieved library
    context). The return is the model's raw text response. Implementations should
    be lazy about loading/network and raise a clear error when unavailable.
    """

    def complete(self, system: str, user: str) -> str:
        ...


class LLMUnavailable(RuntimeError):
    """Raised when a backend cannot reach its model (server down / no SDK / no key)."""


class FakeLLM:
    """Deterministic, model-free language model for unit tests.

    No network, no model, no RNG. It echoes a fixed STRUCTURED stub derived from
    the input so tests can assert determinism and parsing without any download or
    server. The stub is intentionally shaped like the writing-partner's expected
    suggestion format (pointer lines under labeled sections) so the parser can be
    exercised end-to-end — and it is pointers ONLY, never drafted prose, which is
    what the integrity contract requires.
    """

    def complete(self, system: str, user: str) -> str:
        # Pull the first SUBSTANTIVE content line of the user payload as a stable,
        # input-derived token so different passages yield different (but
        # deterministic) stubs. Skip the writing-partner prompt scaffolding
        # (UPPERCASE section labels and the "PASSAGE (...)" header) so the stub
        # keys off real content, not the template.
        tag = "passage"
        for line in user.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.endswith(":") or stripped.upper() == stripped:
                continue
            if stripped.upper().startswith("PASSAGE"):
                continue
            tag = stripped[:60]
            break
        return (
            "MISSING TOPICS:\n"
            f"- consider addressing the mechanism behind: {tag}\n"
            "- a comparison/control condition appears unaddressed\n"
            "NEEDS CITATION:\n"
            f"- the central claim about {tag} appears to need a citation\n"
            "NEXT ANGLES:\n"
            "- relate this to a downstream functional readout\n"
        )


# --- LocalLLM: OpenAI-compatible localhost server (LM Studio / Ollama) -------

_DEFAULT_LOCAL_URL = "http://localhost:1234/v1"
_DEFAULT_LOCAL_MODEL = "local-model"


class LocalLLM:
    """Talks to a LOCAL OpenAI-compatible chat server (LM Studio / Ollama).

    LM Studio and Ollama both expose ``POST {base_url}/chat/completions`` on
    localhost. Transport is stdlib ``urllib`` only — NO new core dependency, the
    same discipline as ``discovery.py``. Network is lazy and guarded: a clear
    ``LLMUnavailable`` is raised when the local server isn't running, so callers
    can degrade gracefully instead of crashing.

    Fully on-device: nothing leaves the machine.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_LOCAL_URL,
        model: str = _DEFAULT_LOCAL_MODEL,
        timeout: int = 120,
    ) -> None:
        # Normalize a trailing slash so f"{base_url}/chat/completions" is clean.
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # Low temperature: we want stable, structured pointers, not prose.
                "temperature": 0.2,
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError, ValueError) as exc:
            raise LLMUnavailable(
                f"Local LLM at {self._endpoint()!r} is unreachable — is LM Studio "
                f"or Ollama running and serving an OpenAI-compatible API? ({exc})"
            ) from exc
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(
                f"Local LLM returned an unexpected response shape: {exc}"
            ) from exc


# --- CloudClaudeLLM: Anthropic SDK (OPTIONAL [cloud] extra) ------------------

_DEFAULT_CLOUD_MODEL = "claude-opus-4-8"


class CloudClaudeLLM:
    """Cloud backend using the ``anthropic`` SDK (OPTIONAL ``[cloud]`` extra).

    The ``anthropic`` import lives INSIDE ``complete`` so the package (and the
    unit-test suite) never requires the SDK to be installed — same lazy-import
    discipline as ``sentence_transformers`` in ``NomicEmbedder``. A clear
    ``LLMUnavailable`` is raised if the SDK is missing or the API key is unset.

    PRIVACY: this is the ONLY backend that sends the user's prose off the machine
    (to Anthropic). The CLI gates it behind ``--allow-cloud`` + a printed warning.
    """

    def __init__(
        self,
        model: str = _DEFAULT_CLOUD_MODEL,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        try:
            import anthropic  # noqa: PLC0415 - lazy: optional [cloud] extra
        except ImportError as exc:
            raise LLMUnavailable(
                "The cloud backend needs the optional 'anthropic' package. "
                "Install it with:  pip install \"scholia[cloud]\""
            ) from exc
        try:
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - SDK/auth/network failure
            raise LLMUnavailable(
                f"Cloud Claude call failed ({type(exc).__name__}: {exc}). Check "
                f"ANTHROPIC_API_KEY and your connection."
            ) from exc
        # Concatenate text blocks from the response content.
        parts = []
        for block in getattr(resp, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)
