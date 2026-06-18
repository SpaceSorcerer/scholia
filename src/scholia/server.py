"""Scholia local JSON bridge: ``scholia serve``.

Loads the index + embedder + reranker ONCE at startup and exposes cite/discover
over localhost as a small JSON API. Binds 127.0.0.1 ONLY — never 0.0.0.0.

Endpoints
---------
GET  /health   → {"status":"ok","papers":N,"embedder":str}
POST /cite     → {"passage":str,"k"?:int,"threshold"?:float,"rerank"?:bool}
POST /discover → {"passage":str,"limit"?:int}

No new dependencies — stdlib http.server/json/urllib only.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from scholia.discovery import (
    FakeDiscoverySource,
    PubMedSource,
    SemanticScholarSource,
    build_query,
    dedupe_against_library,
    discover,
)
from scholia.embedders import FakeEmbedder, NomicEmbedder
from scholia.grounding import claim_check
from scholia.index import ScholiaIndex
from scholia.models import Paper
from scholia.rerank import CrossEncoderReranker, FakeReranker
from scholia.retrieval import Hit, retrieve, retrieve_reranked

_DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_THRESHOLD = 0.45
_DEFAULT_K = 5
_DEFAULT_CANDIDATE_K = 30
_DEFAULT_LIMIT = 8


@dataclass
class ServerState:
    """All models and index loaded once at startup."""

    index: ScholiaIndex
    embedder: Any
    reranker: Any
    fake_source: bool = False


def load_state(
    index_dir: Path,
    no_rerank: bool = False,
    fake_embedder: bool = False,
    fake_source: bool = False,
) -> ServerState:
    """Load index + embedder + reranker from ``index_dir``.

    Exits with a clear message if no index exists (run ``scholia index`` first).
    The reranker falls back gracefully if it cannot load.
    """
    try:
        index = ScholiaIndex.load(index_dir)
    except FileNotFoundError:
        print(
            f"No index at {index_dir}. Run `scholia index` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Adopt stored embedder model (same logic as cli.py cite command).
    model = index.embedder_model or _DEFAULT_MODEL
    embedder = FakeEmbedder() if fake_embedder else NomicEmbedder(model_name=model)

    if no_rerank:
        reranker = None
    elif fake_embedder:
        # Keep the run fully model-free in test/offline mode.
        reranker = FakeReranker()
    else:
        try:
            reranker = CrossEncoderReranker(model_name=_DEFAULT_RERANKER_MODEL)
        except Exception:  # noqa: BLE001
            reranker = None

    return ServerState(
        index=index,
        embedder=embedder,
        reranker=reranker,
        fake_source=fake_source,
    )


# ---------------------------------------------------------------------------
# Pure handler functions — unit-testable, no sockets
# ---------------------------------------------------------------------------


def _hits_to_suggestions(hits: list[Hit]) -> list[dict]:
    out = []
    for rank, h in enumerate(hits, 1):
        p = h.paper
        first_author = (
            p.authors[0].split(",")[0].strip() if p.authors else "Unknown"
        )
        out.append(
            {
                "rank": rank,
                "score": float(h.score),
                "first_author": first_author,
                "year": p.year,
                "title": p.title,
                "zotero_key": p.zotero_key,
                "zotero_link": p.zotero_link,
                "doi": p.doi,
            }
        )
    return out


def handle_cite(req: dict, state: ServerState) -> dict:
    """Pure cite handler: takes a parsed JSON request dict, returns a JSON-able dict.

    Uses the reranked path by default (same as cli.py); pass ``rerank: false`` in
    the request body to force the bi-encoder path.
    """
    passage = req.get("passage", "")
    k = int(req.get("k", _DEFAULT_K))
    threshold = req.get("threshold")
    use_rerank = bool(req.get("rerank", state.reranker is not None))

    reranked = False
    hits: list[Hit] = []

    if use_rerank and state.reranker is not None:
        try:
            hits = retrieve_reranked(
                passage,
                state.embedder,
                state.index,
                state.reranker,
                candidate_k=_DEFAULT_CANDIDATE_K,
                top_k=k,
            )
            reranked = True
        except Exception:  # noqa: BLE001
            pass  # fall through to bi-encoder

    if not reranked:
        hits = retrieve(passage, state.embedder, state.index, k=k)

    # Pick threshold: request overrides > default based on signal.
    if threshold is None:
        if reranked:
            # FakeReranker scores are in [0,1]; use small positive cutoff.
            is_fake_reranker = isinstance(state.reranker, FakeReranker)
            threshold = 0.001 if is_fake_reranker else 0.0
        else:
            threshold = _DEFAULT_THRESHOLD
    else:
        threshold = float(threshold)

    verdict = claim_check(hits, threshold=threshold)
    ranking_signal = "reranked (cross-encoder)" if reranked else "bi-encoder (cosine)"

    return {
        "suggestions": _hits_to_suggestions(hits),
        "claim_check": {
            "supported": verdict.supported,
            "top_score": float(verdict.top_score),
            "threshold": float(verdict.threshold),
        },
        "ranking_signal": ranking_signal,
    }


def handle_discover(req: dict, state: ServerState) -> dict:
    """Pure discover handler: takes a parsed JSON request dict, returns a JSON-able dict."""
    passage = req.get("passage", "")
    limit = int(req.get("limit", _DEFAULT_LIMIT))

    query = build_query(passage)
    if not query:
        return {"candidates": [], "query": ""}

    if state.fake_source:
        sources = [
            FakeDiscoverySource(source_name="semanticscholar"),
            FakeDiscoverySource(source_name="pubmed"),
        ]
    else:
        sources = [SemanticScholarSource(), PubMedSource()]

    library: list[Paper] = list(state.index._papers)

    candidates = discover(passage, sources=sources, library=library, limit=limit)

    out = []
    for c in candidates:
        out.append(
            {
                "title": c.title,
                "authors": list(c.authors),
                "year": c.year,
                "doi": c.doi,
                "snippet": c.abstract_snippet,
                "source": c.source,
            }
        )
    return {"candidates": out, "query": query}


# ---------------------------------------------------------------------------
# HTTP wrapper — thin stdlib dispatcher
# ---------------------------------------------------------------------------


class _ScholiaHandler(BaseHTTPRequestHandler):
    """Minimal HTTP/1.1 handler: routes GET /health and POST /cite,/discover."""

    # Injected by serve() so handlers share state without globals.
    state: ServerState

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        # Suppress the default Apache-style log noise; keep startup clean.
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "papers": len(self.state.index._papers),
                    "embedder": self.state.index.embedder_model or "unknown",
                }
            )
        else:
            self._send_json({"error": f"Not found: {self.path}"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        req = self._read_json_body()
        if req is None:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        if self.path == "/cite":
            try:
                result = handle_cite(req, self.state)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=500)

        elif self.path == "/discover":
            try:
                result = handle_discover(req, self.state)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=500)

        else:
            self._send_json({"error": f"Not found: {self.path}"}, status=404)


def serve(
    host: str,
    port: int,
    state: ServerState,
    daemon: bool = False,
) -> HTTPServer:
    """Create and return an HTTPServer bound to ``host:port`` with ``state`` injected.

    The caller is responsible for calling ``serve_forever()`` or ``handle_request()``.
    ``host`` MUST be ``127.0.0.1`` (enforced by the CLI; this function accepts it
    as a parameter for test flexibility but never defaults to 0.0.0.0).
    """

    # Inject state via a per-request handler subclass (avoids a global).
    class _Handler(_ScholiaHandler):
        pass

    _Handler.state = state  # type: ignore[assignment]

    httpd = HTTPServer((host, port), _Handler)
    return httpd
