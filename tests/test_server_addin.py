"""Tests for the add-in / HTTPS extensions to scholia.server.

Covers:
- generate_localhost_cert writes a valid PEM cert + key pair
- serve() with ssl_certfile/ssl_keyfile wraps the socket (smoke test)
- static-file serving via _is_static_path and _serve_static
- ServerState.addin_dir field
- HTTPS /health round-trip (integration, marked to skip by default)
"""
from __future__ import annotations

import json
import socket
import ssl
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from scholia.corpus import load_corpus
from scholia.embedders import FakeEmbedder
from scholia.index import build_index
from scholia.rerank import FakeReranker
from scholia.server import (
    ServerState,
    _ScholiaHandler,
    generate_localhost_cert,
    handle_cite,
    serve,
)

FIXTURES = Path(__file__).parent / "fixtures"
ADDIN_DIR = Path(__file__).parent.parent / "word-addin"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_state(tmp_path, addin_dir=None) -> ServerState:
    papers = load_corpus(FIXTURES / "corpus")
    embedder = FakeEmbedder(dim=16)
    index = build_index(papers, embedder, tmp_path / "idx")
    reranker = FakeReranker()
    return ServerState(
        index=index,
        embedder=embedder,
        reranker=reranker,
        fake_source=True,
        addin_dir=addin_dir,
    )


# ---------------------------------------------------------------------------
# Certificate generation
# ---------------------------------------------------------------------------


class TestGenerateLocalhostCert:
    def test_writes_cert_and_key(self, tmp_path):
        cert = tmp_path / "localhost.crt"
        key  = tmp_path / "localhost.key"
        generate_localhost_cert(cert, key)
        assert cert.exists(), "cert file must be written"
        assert key.exists(),  "key file must be written"

    def test_cert_is_pem(self, tmp_path):
        cert = tmp_path / "localhost.crt"
        key  = tmp_path / "localhost.key"
        generate_localhost_cert(cert, key)
        text = cert.read_text()
        assert "-----BEGIN CERTIFICATE-----" in text

    def test_key_is_pem(self, tmp_path):
        cert = tmp_path / "localhost.crt"
        key  = tmp_path / "localhost.key"
        generate_localhost_cert(cert, key)
        text = key.read_text()
        assert "-----BEGIN RSA PRIVATE KEY-----" in text or \
               "-----BEGIN PRIVATE KEY-----" in text

    def test_ssl_context_accepts_cert(self, tmp_path):
        """The generated cert/key must load into an SSLContext without error."""
        cert = tmp_path / "localhost.crt"
        key  = tmp_path / "localhost.key"
        generate_localhost_cert(cert, key)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        # Should not raise
        ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))

    def test_idempotent_different_serial(self, tmp_path):
        """Running twice should produce different certs (fresh serial each time)."""
        cert = tmp_path / "localhost.crt"
        key  = tmp_path / "localhost.key"
        generate_localhost_cert(cert, key)
        data1 = cert.read_bytes()
        generate_localhost_cert(cert, key)
        data2 = cert.read_bytes()
        # A freshly generated cert has a random serial — bytes will differ.
        assert data1 != data2

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        cert = nested / "localhost.crt"
        key  = nested / "localhost.key"
        generate_localhost_cert(cert, key)
        assert cert.exists()
        assert key.exists()


# ---------------------------------------------------------------------------
# ServerState.addin_dir field
# ---------------------------------------------------------------------------


class TestServerStateAddinDir:
    def test_default_is_none(self, tmp_path):
        state = _build_state(tmp_path)
        assert state.addin_dir is None

    def test_can_set_addin_dir(self, tmp_path):
        state = _build_state(tmp_path, addin_dir=ADDIN_DIR)
        assert state.addin_dir == ADDIN_DIR


# ---------------------------------------------------------------------------
# Static-path allowlist
# ---------------------------------------------------------------------------


class TestIsStaticPath:
    """_ScholiaHandler._is_static_path checks the allowlist."""

    # _is_static_path is a static method — call it directly on the class.
    def _check(self, path: str) -> bool:
        return _ScholiaHandler._is_static_path(path)

    def test_taskpane_html_allowed(self):
        assert self._check("/taskpane.html")

    def test_taskpane_js_allowed(self):
        assert self._check("/taskpane.js")

    def test_taskpane_css_allowed(self):
        assert self._check("/taskpane.css")

    def test_commands_html_allowed(self):
        assert self._check("/commands.html")

    def test_commands_js_allowed(self):
        assert self._check("/commands.js")

    def test_assets_prefix_allowed(self):
        assert self._check("/assets/icon-32.png")

    def test_health_not_static(self):
        assert not self._check("/health")

    def test_cite_not_static(self):
        assert not self._check("/cite")

    def test_random_not_static(self):
        assert not self._check("/arbitrary/path")

    def test_traversal_attempt_not_static(self):
        assert not self._check("/../etc/passwd")


# ---------------------------------------------------------------------------
# HTTP serve() with HTTPS (smoke — no real network needed)
# ---------------------------------------------------------------------------


class TestServeWithSSL:
    def test_serve_returns_httpserver(self, tmp_path):
        """serve() with ssl args returns an HTTPServer; checking socket type."""
        cert = tmp_path / "localhost.crt"
        key  = tmp_path / "localhost.key"
        generate_localhost_cert(cert, key)

        state = _build_state(tmp_path)

        # Find a free port.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        httpd = serve("127.0.0.1", port, state,
                      ssl_certfile=cert, ssl_keyfile=key)
        # The socket should be an SSLSocket after wrapping.
        assert isinstance(httpd.socket, ssl.SSLSocket)
        httpd.server_close()

    def test_serve_plain_http_unchanged(self, tmp_path):
        """serve() without ssl args returns a plain HTTPServer (no regression)."""
        state = _build_state(tmp_path)

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        httpd = serve("127.0.0.1", port, state)
        # Plain socket — NOT an SSLSocket.
        assert not isinstance(httpd.socket, ssl.SSLSocket)
        httpd.server_close()


# ---------------------------------------------------------------------------
# Static file serving — unit-level via a plain HTTP server + addin_dir
# ---------------------------------------------------------------------------


class TestStaticFileServing:
    """Spin an HTTP (not HTTPS) server with addin_dir set and check GET /taskpane.html."""

    def test_taskpane_html_served(self, tmp_path):
        """GET /taskpane.html returns 200 when addin_dir is set."""
        if not (ADDIN_DIR / "taskpane.html").exists():
            pytest.skip("word-addin directory not built yet")

        state = _build_state(tmp_path, addin_dir=ADDIN_DIR)

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        httpd = serve("127.0.0.1", port, state)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        base = f"http://127.0.0.1:{port}"
        try:
            _wait_for_server(base)
            with urllib.request.urlopen(f"{base}/taskpane.html") as resp:
                body = resp.read().decode()
            assert "Scholia" in body
            assert resp.status == 200
        finally:
            httpd.shutdown()

    def test_taskpane_js_served(self, tmp_path):
        if not (ADDIN_DIR / "taskpane.js").exists():
            pytest.skip("word-addin directory not built yet")

        state = _build_state(tmp_path, addin_dir=ADDIN_DIR)

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        httpd = serve("127.0.0.1", port, state)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        base = f"http://127.0.0.1:{port}"
        try:
            _wait_for_server(base)
            with urllib.request.urlopen(f"{base}/taskpane.js") as resp:
                body = resp.read().decode()
            assert "Office.onReady" in body
        finally:
            httpd.shutdown()

    def test_arbitrary_path_not_served(self, tmp_path):
        """GET /etc/passwd is rejected (not in allowlist)."""
        state = _build_state(tmp_path, addin_dir=ADDIN_DIR)

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        httpd = serve("127.0.0.1", port, state)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        base = f"http://127.0.0.1:{port}"
        try:
            _wait_for_server(base)
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"{base}/etc/passwd")
            assert exc_info.value.code == 404
        finally:
            httpd.shutdown()

    def test_static_not_served_without_addin_dir(self, tmp_path):
        """When addin_dir is None, static paths return 404."""
        state = _build_state(tmp_path, addin_dir=None)

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        httpd = serve("127.0.0.1", port, state)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        base = f"http://127.0.0.1:{port}"
        try:
            _wait_for_server(base)
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"{base}/taskpane.html")
            assert exc_info.value.code == 404
        finally:
            httpd.shutdown()


# ---------------------------------------------------------------------------
# Integration: real HTTPS /health + /cite round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_https_health_and_cite(tmp_path):
    """Start the HTTPS server on an ephemeral port; verify /health and /cite."""
    cert = tmp_path / "localhost.crt"
    key  = tmp_path / "localhost.key"
    generate_localhost_cert(cert, key)

    state = _build_state(tmp_path, addin_dir=ADDIN_DIR)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    httpd = serve("127.0.0.1", port, state,
                  ssl_certfile=cert, ssl_keyfile=key)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    base = f"https://127.0.0.1:{port}"
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(str(cert))  # trust our own self-signed cert

    try:
        _wait_for_server_https(base, ctx)

        # /health
        req = urllib.request.Request(f"{base}/health")
        with urllib.request.urlopen(req, context=ctx) as resp:
            data = json.loads(resp.read())
        assert data["status"] == "ok"
        assert "papers" in data

        # /cite
        body = json.dumps({"passage": "QKI RNA splicing"}).encode()
        req = urllib.request.Request(
            f"{base}/cite",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, context=ctx) as resp:
            result = json.loads(resp.read())
        assert "suggestions" in result
        assert "claim_check" in result

        # /taskpane.html (static)
        req = urllib.request.Request(f"{base}/taskpane.html")
        with urllib.request.urlopen(req, context=ctx) as resp:
            html = resp.read().decode()
        assert "Scholia" in html

    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_server(base: str, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=0.5):
                return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"Server at {base} did not start in {timeout}s")


def _wait_for_server_https(base: str, ctx: ssl.SSLContext,
                           timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base}/health")
            with urllib.request.urlopen(req, context=ctx, timeout=0.5):
                return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"HTTPS server at {base} did not start in {timeout}s")
