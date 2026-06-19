"""Scholia desktop overlay — always-on-top grounding/discovery client.

Thin PySide6 window that talks to ``scholia serve`` (127.0.0.1).  The bridge
HTTP calls and display formatting live in pure functions/classes (BridgeClient,
format_cite_result, format_discover_result) that are unit-testable without a
display.  The QWidget layer is glue only.

Install the optional extra before running:
    pip install "scholia[overlay]"

Then launch:
    scholia overlay           # bridge must be running: scholia serve
    scholia overlay --start-server   # auto-launches the bridge if not up
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


# ---------------------------------------------------------------------------
# Pure bridge client — testable without Qt
# ---------------------------------------------------------------------------


class BridgeError(Exception):
    """Raised when the bridge is unreachable or returns an error response."""


class BridgeClient:
    """Thin HTTP client for the scholia serve bridge.

    Parameters
    ----------
    host:
        Bridge hostname (default ``127.0.0.1``).
    port:
        Bridge port (default ``8765``).
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise BridgeError(f"Bridge unreachable at {url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise BridgeError(f"Bridge returned invalid JSON: {exc}") from exc

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise BridgeError(f"Bridge unreachable at {url}: {exc.reason}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """GET /health → dict with status/papers/embedder."""
        return self._get("/health")

    def cite(self, passage: str, k: int = 5) -> dict:
        """POST /cite → cite result dict (suggestions + claim_check + ranking_signal)."""
        return self._post("/cite", {"passage": passage, "k": k})

    def discover(self, passage: str, limit: int = 8) -> dict:
        """POST /discover → discover result dict (candidates + query)."""
        return self._post("/discover", {"passage": passage, "limit": limit})


# ---------------------------------------------------------------------------
# Pure display formatters — testable without Qt
# ---------------------------------------------------------------------------

import html as _html


def format_bridge_error(exc: Exception) -> str:
    """Return a user-facing error string for a BridgeError or unexpected exception."""
    msg = str(exc)
    if isinstance(exc, BridgeError) and ("unreachable" in msg or "Connection" in msg.lower()):
        return "Can't reach Scholia server — is `scholia serve` running?"
    return f"Error: {msg}"


def render_links_as_html(text: str) -> str:
    """Convert a plain-text result string to HTML with clickable DOI and Zotero links.

    Escapes all HTML, then linkifies ``https://doi.org/...`` and
    ``zotero://select/...`` URLs.  Output is a ``<pre>``-wrapped block so
    monospace formatting is preserved.
    """
    import re
    escaped = _html.escape(text)
    # Match DOI URLs and zotero:// links
    url_pat = re.compile(
        r'(https://doi\.org/[^\s<>"]+|zotero://[^\s<>"]+)'
    )

    def _linkify(m: re.Match) -> str:
        url = m.group(1)
        return f'<a href="{url}">{url}</a>'

    linked = url_pat.sub(_linkify, escaped)
    return f"<pre style='white-space:pre-wrap; font-family:monospace;'>{linked}</pre>"


def format_cite_result(result: dict[str, Any]) -> str:
    """Turn a /cite response dict into a human-readable display string."""
    lines: list[str] = []

    cc = result.get("claim_check", {})
    supported: bool = cc.get("supported", False)
    top_score: float = cc.get("top_score", 0.0)
    threshold: float = cc.get("threshold", 0.0)
    signal: str = result.get("ranking_signal", "")

    if supported:
        lines.append(
            f"SUPPORTED  (top score {top_score:.3f} ≥ {threshold})"
        )
    else:
        lines.append(
            f"UNSUPPORTED by your library  (top score {top_score:.3f} < {threshold})"
        )

    if signal:
        lines.append(f"Ranking: {signal}")
    lines.append("")

    suggestions = result.get("suggestions", [])
    if not suggestions:
        lines.append("No matching papers found in your library.")
        return "\n".join(lines)

    lines.append(f"Supporting papers ({len(suggestions)}):")
    for s in suggestions:
        rank = s.get("rank", "?")
        author = s.get("first_author", "Unknown")
        year = s.get("year") or "n.d."
        title = s.get("title", "(no title)")
        doi = s.get("doi") or ""
        zlink = s.get("zotero_link") or ""
        score = s.get("score", 0.0)

        lines.append(f"  {rank}. {author} ({year}) — {title}  [{score:.3f}]")
        if doi:
            lines.append(f"     DOI: https://doi.org/{doi}")
        if zlink:
            lines.append(f"     Zotero: {zlink}")

    return "\n".join(lines)


def format_discover_result(result: dict[str, Any]) -> str:
    """Turn a /discover response dict into a human-readable display string."""
    lines: list[str] = []

    query = result.get("query", "")
    candidates = result.get("candidates", [])

    if query:
        lines.append(f"Search query sent to APIs: {query}")
        lines.append("")

    if not candidates:
        lines.append(
            "No NEW candidate papers found "
            "(everything relevant is already in your library, or the search "
            "returned nothing)."
        )
        return "\n".join(lines)

    lines.append(
        f"{len(candidates)} candidate paper(s) NOT in your library "
        "(suggestions only — validate before adding):"
    )
    for rank, c in enumerate(candidates, 1):
        authors = c.get("authors") or []
        first_author = authors[0].split(",")[0].strip() if authors else "Unknown"
        year = c.get("year") or "n.d."
        title = c.get("title", "(no title)")
        doi = c.get("doi") or ""
        snippet = c.get("snippet") or ""
        source = c.get("source", "")

        lines.append(
            f"  {rank}. [{source}] {first_author} ({year}) — {title}"
        )
        if doi:
            lines.append(f"     DOI: https://doi.org/{doi}")
            lines.append(
                f"     To add: scholia discover \"<passage>\" --add {doi}"
            )
        else:
            lines.append("     DOI: (none reported)")
        if snippet:
            lines.append(f"     {snippet}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Qt overlay window
# ---------------------------------------------------------------------------


def _require_pyside6() -> None:
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print(
            "PySide6 is not installed.\n"
            "Run:  pip install \"scholia[overlay]\"\n"
            "Then: scholia overlay",
            file=sys.stderr,
        )
        sys.exit(1)


def _is_bridge_up(host: str, port: int) -> bool:
    try:
        client = BridgeClient(host=host, port=port, timeout=1.0)
        client.health()
        return True
    except BridgeError:
        return False


def _launch_bridge(host: str, port: int) -> subprocess.Popen | None:
    """Start ``scholia serve`` in a subprocess. Returns the Popen object."""
    cmd = [
        sys.executable,
        "-m",
        "scholia.cli",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None

    # Wait up to 5 s for it to become healthy
    for _ in range(25):
        time.sleep(0.2)
        if _is_bridge_up(host, port):
            return proc
    return proc


def run_overlay(host: str = "127.0.0.1", port: int = 8765, start_server: bool = False) -> None:
    """Launch the Scholia overlay window.

    Parameters
    ----------
    host:
        Bridge host (must be 127.0.0.1 for privacy).
    port:
        Bridge port.
    start_server:
        If True and the bridge is not running, auto-launch ``scholia serve``.
    """
    _require_pyside6()

    # Lazy imports — PySide6 is optional.
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QClipboard
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )

    _bridge_proc: list[subprocess.Popen] = []  # mutable container for closure

    if start_server and not _is_bridge_up(host, port):
        proc = _launch_bridge(host, port)
        if proc:
            _bridge_proc.append(proc)

    client = BridgeClient(host=host, port=port)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Scholia")

    # ------------------------------------------------------------------
    # Worker thread — keeps the UI responsive during bridge calls
    # ------------------------------------------------------------------

    class _Worker(QThread):
        finished = Signal(str)
        error = Signal(str)

        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self._fn = fn
            self._args = args
            self._kwargs = kwargs

        def run(self):
            try:
                result = self._fn(*self._args, **self._kwargs)
                self.finished.emit(result)
            except BridgeError as exc:
                self.error.emit(str(exc))

    # ------------------------------------------------------------------
    # Main window
    # ------------------------------------------------------------------

    win = QWidget()
    win.setWindowTitle("Scholia — live grounding")
    win.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
    win.resize(620, 520)

    outer = QVBoxLayout(win)
    outer.setContentsMargins(8, 8, 8, 8)
    outer.setSpacing(6)

    # Status bar (bridge health)
    status_label = QLabel("Bridge: checking…")
    status_label.setStyleSheet("color: gray; font-size: 11px;")
    outer.addWidget(status_label)

    # Splitter: input on top, results on bottom
    splitter = QSplitter(Qt.Orientation.Vertical)
    outer.addWidget(splitter, 1)

    # --- Input pane ---
    input_widget = QWidget()
    input_layout = QVBoxLayout(input_widget)
    input_layout.setContentsMargins(0, 0, 0, 0)
    input_layout.setSpacing(4)

    input_label = QLabel("Passage (type, paste, or use clipboard button):")
    input_label.setStyleSheet("font-size: 12px; font-weight: bold;")
    input_layout.addWidget(input_label)

    text_box = QPlainTextEdit()
    text_box.setPlaceholderText(
        "Paste or type a sentence or passage here…"
    )
    text_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    input_layout.addWidget(text_box)

    btn_row = QHBoxLayout()
    btn_row.setSpacing(6)

    btn_ground = QPushButton("Ground  (POST /cite)")
    btn_ground.setToolTip("Check this passage against your library (POST /cite)")
    btn_row.addWidget(btn_ground)

    btn_discover = QPushButton("Discover  (POST /discover)")
    btn_discover.setToolTip("Find papers NOT yet in your library (POST /discover)")
    btn_row.addWidget(btn_discover)

    btn_clipboard = QPushButton("Ground clipboard")
    btn_clipboard.setToolTip(
        "Grab text from your clipboard and ground it immediately"
    )
    btn_row.addWidget(btn_clipboard)

    btn_row.addStretch()
    input_layout.addLayout(btn_row)
    splitter.addWidget(input_widget)

    # --- Results pane (QTextBrowser renders HTML with clickable links) ---
    results_box = QTextBrowser()
    results_box.setReadOnly(True)
    results_box.setOpenExternalLinks(True)
    results_box.setPlaceholderText("Results appear here…")
    results_box.setStyleSheet("background: #f9f9f9;")
    splitter.addWidget(results_box)
    splitter.setSizes([200, 300])

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    _active_worker: list[_Worker] = []

    def _set_busy(busy: bool) -> None:
        btn_ground.setEnabled(not busy)
        btn_discover.setEnabled(not busy)
        btn_clipboard.setEnabled(not busy)
        if busy:
            results_box.setHtml("<pre>Working…</pre>")

    def _on_error(msg: str) -> None:
        _set_busy(False)
        safe = _html.escape(msg)
        results_box.setHtml(
            f"<p style='color:red; font-weight:bold;'>"
            f"Can't reach Scholia server — is <code>scholia serve</code> running?</p>"
            f"<pre style='color:gray;'>{safe}</pre>"
            if "unreachable" in msg.lower() or "connection" in msg.lower()
            else f"<p style='color:red;'>Error:</p><pre>{safe}</pre>"
        )

    def _run_action(api_fn, formatter):
        passage = text_box.toPlainText().strip()
        if not passage:
            results_box.setHtml("<pre>Enter a passage first.</pre>")
            return
        _set_busy(True)

        def _call():
            raw = api_fn(passage)
            plain = formatter(raw)
            return render_links_as_html(plain)

        worker = _Worker(_call)
        worker.finished.connect(lambda html: (_set_busy(False), results_box.setHtml(html)))
        worker.error.connect(_on_error)
        _active_worker.clear()
        _active_worker.append(worker)
        worker.start()

    def _ground():
        _run_action(client.cite, format_cite_result)

    def _discover():
        _run_action(client.discover, format_discover_result)

    def _ground_clipboard():
        clipboard = QApplication.clipboard()
        text = clipboard.text(QClipboard.Mode.Clipboard).strip()
        if not text:
            results_box.setHtml("<pre>Clipboard is empty.</pre>")
            return
        text_box.setPlainText(text)
        _ground()

    btn_ground.clicked.connect(_ground)
    btn_discover.clicked.connect(_discover)
    btn_clipboard.clicked.connect(_ground_clipboard)

    # Allow Ctrl+Enter to trigger Ground
    from PySide6.QtGui import QKeySequence, QShortcut
    shortcut = QShortcut(QKeySequence("Ctrl+Return"), win)
    shortcut.activated.connect(_ground)

    # ------------------------------------------------------------------
    # Bridge health check (non-blocking)
    # ------------------------------------------------------------------

    def _check_health():
        try:
            info = client.health()
            n = info.get("papers", "?")
            emb = info.get("embedder", "?")
            status_label.setText(f"Bridge: OK  |  {n} papers  |  {emb}")
            status_label.setStyleSheet("color: green; font-size: 11px;")
        except BridgeError:
            status_label.setText(
                f"Bridge: unreachable at {host}:{port}  —  run: scholia serve"
            )
            status_label.setStyleSheet("color: red; font-size: 11px;")

    # Check health on startup in a one-shot thread so the window opens immediately
    health_worker = _Worker(_check_health)
    # health check updates labels in-place via closure (already on main thread via
    # direct call inside _Worker.run; label updates must be on the main thread,
    # so we check health synchronously after a small delay using a QTimer)
    from PySide6.QtCore import QTimer
    QTimer.singleShot(200, _check_health)

    # ------------------------------------------------------------------
    # Show and run
    # ------------------------------------------------------------------

    win.show()
    exit_code = app.exec()

    # Clean up auto-launched bridge (best-effort)
    for proc in _bridge_proc:
        try:
            proc.terminate()
        except OSError:
            pass

    sys.exit(exit_code)
