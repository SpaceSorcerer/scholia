"""Scholia local JSON bridge: ``scholia serve``.

Loads the index + embedder + reranker ONCE at startup and exposes cite/discover
over localhost as a small JSON API. Binds 127.0.0.1 ONLY — never 0.0.0.0.

Endpoints
---------
GET  /health            → {"status":"ok","papers":N,"embedder":str}
POST /cite              → {"passage":str,"k"?:int,"threshold"?:float,"rerank"?:bool}
POST /discover          → {"passage":str,"limit"?:int}
GET  /taskpane.html     → task-pane static file (only in --serve-addin mode)
GET  /taskpane.js       → task-pane static file (only in --serve-addin mode)
GET  /taskpane.css      → task-pane static file (only in --serve-addin mode)
GET  /commands.html     → command shim (only in --serve-addin mode)
GET  /commands.js       → command shim (only in --serve-addin mode)
GET  /assets/<file>     → icon assets (only in --serve-addin mode)

--serve-addin mode: wraps the HTTPServer in an SSLContext so the pane loads over
HTTPS and calls /cite same-origin — no CORS, no mixed-content block.

Core dependencies: stdlib only.  HTTPS cert generation requires the ``cryptography``
package (already in the environment; NOT added to core deps — the plain-HTTP path
works without it).
"""

from __future__ import annotations

import json
import mimetypes
import ssl
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# The directory containing the word-addin static files, relative to this module.
# Resolved at import-time so tests can patch it easily.
_ADDIN_DIR = Path(__file__).parent.parent.parent / "word-addin"

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
    """All models and index loaded once at startup.

    ``models_ready`` is set by ``warm_models()`` once the embedder (and optional
    reranker) have finished loading their weights.  Handlers that need a warm
    model can wait on it with a short timeout rather than blocking the caller.
    """

    index: ScholiaIndex
    embedder: Any
    reranker: Any
    fake_source: bool = False
    addin_dir: Path | None = None  # set when --serve-addin is active
    models_ready: threading.Event = field(default_factory=threading.Event)


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


def warm_models(state: ServerState) -> None:
    """Pre-load (warm) the embedder and reranker weights into RAM synchronously.

    Call this once after ``load_state()`` — on a background thread for the
    bridge/serve path so the server socket is already listening; from the app's
    ``load_async`` background thread so the window paints before models load.

    When done it sets ``state.models_ready`` so waiting callers can proceed.
    Safe to call multiple times (the underlying ``_ensure_loaded`` is idempotent).
    """
    try:
        _ensure = getattr(state.embedder, "_ensure_loaded", None)
        if _ensure is not None:
            _ensure()
    except Exception:  # noqa: BLE001 — never crash a background warm thread
        pass

    if state.reranker is not None:
        try:
            _ensure_r = getattr(state.reranker, "_ensure_loaded", None)
            if _ensure_r is not None:
                _ensure_r()
        except Exception:  # noqa: BLE001
            pass

    state.models_ready.set()


def warm_models_async(
    state: ServerState,
    on_done: "callable[[], None] | None" = None,
) -> threading.Thread:
    """Warm the embedder + reranker on a daemon thread; call ``on_done`` when ready.

    Returns the Thread so the caller can join if needed.  The thread is daemonic
    so it never prevents interpreter exit.
    """

    def _work() -> None:
        warm_models(state)
        if on_done is not None:
            try:
                on_done()
            except Exception:  # noqa: BLE001
                pass

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    return t


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

    If the embedder/reranker have not finished warming yet (``state.models_ready``
    is unset) this call will block until they are ready — the models WILL load,
    so the wait is finite.  Callers on a non-UI thread (bridge request handler,
    app Worker QThread) are safe to block here.
    """
    # Wait for background model-warm to finish; 300 s is a very generous upper
    # bound for a first-ever download on a slow connection.  In practice warm
    # loads finish in < 20 s and cached loads in < 5 s.
    state.models_ready.wait(timeout=300)

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
        elif self.state.addin_dir is not None and self._is_static_path(self.path):
            self._serve_static(self.path)
        else:
            self._send_json({"error": f"Not found: {self.path}"}, status=404)

    # -- Static file helpers (task pane) -----------------------------------

    @staticmethod
    def _is_static_path(path: str) -> bool:
        """Return True if ``path`` is in the static-file allowlist.

        Only paths in this allowlist are ever served from the addin_dir;
        everything else falls through to 404.
        """
        _STATIC_ALLOWLIST = (
            "/taskpane.html",
            "/taskpane.js",
            "/taskpane.css",
            "/commands.html",
            "/commands.js",
            "/assets/",
        )
        return any(path == p or path.startswith(p) for p in _STATIC_ALLOWLIST)

    def _serve_static(self, url_path: str) -> None:
        """Serve a file from ``self.state.addin_dir`` matching ``url_path``.

        Only paths in ``_STATIC_ALLOWLIST`` reach here (enforced by the caller).
        Path traversal is prevented by resolving the final path inside addin_dir
        and checking it stays within that root.
        """
        addin_dir = self.state.addin_dir
        assert addin_dir is not None  # caller guarantees this

        # Strip the leading "/" and resolve safely.
        rel = url_path.lstrip("/")
        try:
            target = (addin_dir / rel).resolve()
        except Exception:
            self._send_json({"error": "Bad path"}, status=400)
            return

        # Security: target must stay inside addin_dir.
        try:
            target.relative_to(addin_dir.resolve())
        except ValueError:
            self._send_json({"error": "Forbidden"}, status=403)
            return

        if not target.is_file():
            self._send_json({"error": f"Not found: {url_path}"}, status=404)
            return

        content_type, _ = mimetypes.guess_type(str(target))
        if content_type is None:
            content_type = "application/octet-stream"

        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # CORS header for same-origin (https://127.0.0.1:8765 → /cite etc.)
        self.send_header("Access-Control-Allow-Origin", "https://127.0.0.1:8765")
        self.end_headers()
        self.wfile.write(body)

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
    ssl_certfile: Path | None = None,
    ssl_keyfile: Path | None = None,
) -> HTTPServer:
    """Create and return an HTTPServer bound to ``host:port`` with ``state`` injected.

    The caller is responsible for calling ``serve_forever()`` or ``handle_request()``.
    ``host`` MUST be ``127.0.0.1`` (enforced by the CLI; this function accepts it
    as a parameter for test flexibility but never defaults to 0.0.0.0).

    When ``ssl_certfile`` and ``ssl_keyfile`` are provided the socket is wrapped in
    an SSLContext so the server runs over HTTPS.  The task-pane add-in requires this
    (Office.js loads the pane from an HTTPS URL; mixed-content blocks plain HTTP).
    The plain-HTTP path (default) is unchanged — no SSL dependency.
    """

    # Inject state via a per-request handler subclass (avoids a global).
    class _Handler(_ScholiaHandler):
        pass

    _Handler.state = state  # type: ignore[assignment]

    httpd = HTTPServer((host, port), _Handler)

    if ssl_certfile and ssl_keyfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(ssl_certfile), keyfile=str(ssl_keyfile))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    return httpd


# ---------------------------------------------------------------------------
# Certificate generation helpers (--serve-addin / --https mode)
# ---------------------------------------------------------------------------


def generate_localhost_cert(cert_path: Path, key_path: Path) -> None:
    """Generate a self-signed localhost certificate using the ``cryptography`` package.

    The cert is valid for 825 days (Apple/Chrome limit) and covers both
    ``127.0.0.1`` and ``localhost`` as Subject Alternative Names.  This is
    functionally equivalent to what ``office-addin-dev-certs`` generates but
    uses only Python so Node is not required at runtime.

    The caller must TRUST the certificate in the OS/browser certificate store
    before the task pane will load without a security warning.  See
    ``word-addin/SIDELOAD_WORD.md`` for the trust step.

    Raises ``ImportError`` if the ``cryptography`` package is not installed.
    Raises ``OSError`` if the cert/key cannot be written.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:
        raise ImportError(
            "The 'cryptography' package is required for --serve-addin / --https.\n"
            "Install it:  pip install cryptography"
        ) from exc

    import datetime
    import ipaddress

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "Scholia localhost")]
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .sign(key, hashes.SHA256())
    )

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    cert_path.write_bytes(
        cert.public_bytes(serialization.Encoding.PEM)
    )
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
