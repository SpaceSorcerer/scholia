"""Scholia desktop app — system-tray + results panel + global hotkey.

A clean GUI that replaces the black-console batch experience.  The app lives
in the system tray; pressing Ctrl+Alt+G (or using the tray menu) grounds the
current selection / clipboard and pops a polished results panel.

Hard rules (never relax):
  - ASSIST, never ghostwrite — the panel shows supporting papers, the
    SUPPORTED/⚠ verdict, and Discovery results ONLY; it NEVER drafts or
    rewrites prose.
  - LOCAL-FIRST — grounding runs in-process (same engine as ``scholia cite``);
    only discovery keyword queries leave the machine.

Install the overlay + hotkey extras before running:
    pip install "scholia[overlay]"   # PySide6
    pip install pynput               # global hotkey, no admin required

Then launch:
    scholia app              # index must exist: scholia index
    python -m scholia.app    # equivalent
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Silence library noise BEFORE any ML imports.
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Global hotkey constant — change this one line to rebind.
# ---------------------------------------------------------------------------
HOTKEY_COMBO = "<ctrl>+<alt>+g"

# ---------------------------------------------------------------------------
# Top-N cap for the panel (only show the best N papers as cards).
# ---------------------------------------------------------------------------
TOP_N = 5


# ---------------------------------------------------------------------------
# Pure non-GUI helpers — unit-testable without a display.
# ---------------------------------------------------------------------------


def build_card_data(suggestions: list[dict[str, Any]], top_n: int = TOP_N) -> list[dict[str, Any]]:
    """Extract the top-N card fields from a /cite suggestions list.

    Returns a list of dicts with keys: rank, title, first_author, year,
    score, doi_url, zotero_url.  All values are plain strings (or empty string).
    """
    cards = []
    for s in suggestions[:top_n]:
        doi = s.get("doi") or ""
        doi_url = f"https://doi.org/{doi}" if doi else ""
        zk = s.get("zotero_link") or ""
        cards.append(
            {
                "rank": s.get("rank", "?"),
                "title": s.get("title") or "(no title)",
                "first_author": s.get("first_author") or "Unknown",
                "year": str(s.get("year") or "n.d."),
                "score": f"{s.get('score', 0.0):.3f}",
                "doi_url": doi_url,
                "zotero_url": zk,
            }
        )
    return cards


def build_verdict(claim_check: dict[str, Any]) -> tuple[bool, str]:
    """Return (supported: bool, label: str) from a claim_check dict."""
    supported = bool(claim_check.get("supported", False))
    top = float(claim_check.get("top_score", 0.0))
    thr = float(claim_check.get("threshold", 0.0))
    if supported:
        label = f"SUPPORTED  (score {top:.3f} ≥ {thr:.3f})"
    else:
        label = f"⚠ NOT CLEARLY SUPPORTED  (score {top:.3f} < {thr:.3f})"
    return supported, label


def build_discover_cards(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract card fields from a /discover candidates list."""
    cards = []
    for rank, c in enumerate(candidates, 1):
        authors = c.get("authors") or []
        first_author = authors[0].split(",")[0].strip() if authors else "Unknown"
        doi = c.get("doi") or ""
        doi_url = f"https://doi.org/{doi}" if doi else ""
        cards.append(
            {
                "rank": rank,
                "title": c.get("title") or "(no title)",
                "first_author": first_author,
                "year": str(c.get("year") or "n.d."),
                "doi_url": doi_url,
                "source": c.get("source") or "",
                "snippet": c.get("snippet") or "",
            }
        )
    return cards


def build_query_label(result: dict[str, Any]) -> str:
    """Extract the discovery query for display."""
    return result.get("query") or ""


# ---------------------------------------------------------------------------
# In-process grounding engine (loads index + models once, reuses on every call).
# ---------------------------------------------------------------------------


class GroundingEngine:
    """Loads the Scholia index + models once; exposes cite() and discover().

    All heavy ML work runs on a background thread; the Qt panel stays responsive.
    """

    def __init__(self, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._state: Any = None  # ServerState once loaded
        self._load_error: str | None = None
        self._loaded = threading.Event()

    def load_async(self, on_done: "callable[[str | None], None]") -> None:
        """Load the engine on a background thread.  Calls on_done(error_msg|None)."""
        def _work():
            try:
                from scholia.server import load_state
                self._state = load_state(self._index_dir)
                self._loaded.set()
                on_done(None)
            except SystemExit:
                # load_state calls sys.exit(1) if no index exists.
                self._load_error = (
                    "No index found. Build it first:\n\n"
                    "    scholia index --corpus <your-corpus-dir>\n\n"
                    "See QUICKSTART_TESTING.md for details."
                )
                self._loaded.set()
                on_done(self._load_error)
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                self._loaded.set()
                on_done(self._load_error)

        t = threading.Thread(target=_work, daemon=True)
        t.start()

    @property
    def ready(self) -> bool:
        return self._loaded.is_set() and self._state is not None

    @property
    def n_papers(self) -> int:
        if self._state is None:
            return 0
        return len(self._state.index._papers)

    def cite(self, passage: str, k: int = TOP_N) -> dict[str, Any]:
        """Ground passage in-process (blocking; call from a background thread)."""
        if not self.ready:
            raise RuntimeError("Engine not loaded yet.")
        from scholia.server import handle_cite
        return handle_cite({"passage": passage, "k": k}, self._state)

    def discover(self, passage: str, limit: int = 8) -> dict[str, Any]:
        """Run discovery in-process (blocking; call from a background thread)."""
        if not self.ready:
            raise RuntimeError("Engine not loaded yet.")
        from scholia.server import handle_discover
        return handle_discover({"passage": passage, "limit": limit}, self._state)


# ---------------------------------------------------------------------------
# Qt app — launched from run_app(); everything below imports PySide6 lazily.
# ---------------------------------------------------------------------------


def _require_pyside6() -> None:
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print(
            "PySide6 is not installed.\n"
            "Run:  pip install \"scholia[overlay]\"\n"
            "Then: scholia app",
            file=sys.stderr,
        )
        sys.exit(1)


def _require_pynput() -> None:
    try:
        import pynput  # noqa: F401
    except ImportError:
        print(
            "pynput is not installed (needed for the global hotkey).\n"
            "Run:  pip install pynput\n"
            "The rest of the app works without it; hotkey will be disabled.",
            file=sys.stderr,
        )
        # Non-fatal — we continue without the hotkey.


def _get_tray_icon() -> "QIcon":
    """Return a simple coloured tray icon (blue circle 'S') without needing a PNG."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap

    px = QPixmap(64, 64)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QBrush(QColor("#2563EB")))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2, 2, 60, 60)
    p.setPen(QColor("white"))
    f = QFont("Arial", 28, QFont.Weight.Bold)
    p.setFont(f)
    p.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, "S")
    p.end()
    return QIcon(px)


def _card_html(card: dict[str, Any], is_cite: bool = True) -> str:
    """Render a single paper card as HTML."""
    import html

    title = html.escape(card["title"])
    author = html.escape(card["first_author"])
    year = html.escape(card["year"])

    links = ""
    if card.get("doi_url"):
        doi_esc = html.escape(card["doi_url"])
        links += f'<a href="{doi_esc}" style="color:#2563EB;">DOI</a>'
    if is_cite and card.get("zotero_url"):
        z_esc = html.escape(card["zotero_url"])
        if links:
            links += ' &nbsp;·&nbsp; '
        links += f'<a href="{z_esc}" style="color:#7C3AED;">Zotero</a>'
    if is_cite:
        score_html = f' <span style="color:#6B7280; font-size:10px;">[{html.escape(card["score"])}]</span>'
    else:
        src = html.escape(card.get("source", ""))
        snip = html.escape((card.get("snippet") or "")[:120])
        score_html = f' <span style="color:#6B7280; font-size:10px;">[{src}]</span>'
        if snip:
            score_html += f'<br><span style="color:#374151; font-size:10px; font-style:italic;">{snip}</span>'

    rank = card.get("rank", "")

    return (
        f'<div style="background:#F9FAFB; border:1px solid #E5E7EB; border-radius:6px;'
        f' padding:8px 10px; margin-bottom:6px;">'
        f'<span style="color:#374151; font-size:11px; font-weight:600;">'
        f'{rank}. {title}</span>'
        f'<br>'
        f'<span style="color:#6B7280; font-size:10px;">{author} ({year})</span>'
        f'{score_html}'
        f'{"<br>" + links if links else ""}'
        f'</div>'
    )


def _results_html(
    cite_result: dict[str, Any] | None = None,
    discover_result: dict[str, Any] | None = None,
    passage: str = "",
    loading: bool = False,
    error: str | None = None,
) -> str:
    """Build the full results-pane HTML."""
    import html

    if loading:
        return (
            '<div style="padding:20px; text-align:center; color:#6B7280;">'
            '<span style="font-size:14px;">Thinking…</span>'
            '</div>'
        )

    if error:
        err_esc = html.escape(error)
        return (
            f'<div style="padding:12px; background:#FEF2F2; border-radius:6px;">'
            f'<span style="color:#DC2626; font-weight:600;">Error</span><br>'
            f'<span style="color:#7F1D1D; font-size:11px;">{err_esc}</span>'
            f'</div>'
        )

    parts = []

    # --- Grounding result ---
    if cite_result is not None:
        supported, verdict_label = build_verdict(cite_result.get("claim_check", {}))
        badge_bg = "#D1FAE5" if supported else "#FEF3C7"
        badge_color = "#065F46" if supported else "#92400E"

        parts.append(
            f'<div style="background:{badge_bg}; border-radius:6px; padding:8px 12px;'
            f' margin-bottom:10px; font-weight:700; color:{badge_color}; font-size:12px;">'
            f'{html.escape(verdict_label)}'
            f'</div>'
        )

        cards = build_card_data(cite_result.get("suggestions", []), top_n=TOP_N)
        if cards:
            parts.append(
                f'<p style="color:#374151; font-size:11px; font-weight:600;'
                f' margin:0 0 6px 0;">Top {len(cards)} supporting papers:</p>'
            )
            for c in cards:
                parts.append(_card_html(c, is_cite=True))
        else:
            parts.append(
                '<p style="color:#6B7280; font-size:11px;">No matching papers found in your library.</p>'
            )

    # --- Discovery result ---
    if discover_result is not None:
        query = html.escape(build_query_label(discover_result))
        discover_cards = build_discover_cards(discover_result.get("candidates", []))

        if discover_cards:
            parts.append(
                f'<p style="color:#374151; font-size:11px; font-weight:600;'
                f' margin:10px 0 6px 0;">Discovery ({len(discover_cards)} new):</p>'
            )
            if query:
                parts.append(
                    f'<p style="color:#6B7280; font-size:10px; margin:0 0 6px 0;">'
                    f'Query sent: {query}</p>'
                )
            for c in discover_cards:
                parts.append(_card_html(c, is_cite=False))
        else:
            parts.append(
                '<p style="color:#6B7280; font-size:11px;">'
                'No new candidates found (everything relevant may already be in your library).</p>'
            )

    if not parts:
        return (
            '<div style="padding:20px; text-align:center; color:#9CA3AF;">'
            '<span style="font-size:12px;">Press Ctrl+Alt+G (or Ground Clipboard) to start.</span>'
            '</div>'
        )

    return "".join(parts)


def run_app(index_dir: Path | None = None) -> None:
    """Launch the Scholia tray app.  Does not return until the user quits."""
    _require_pyside6()

    from PySide6.QtCore import QObject, Qt, QThread, Signal, QTimer
    from PySide6.QtGui import QAction, QCloseEvent, QFont, QKeySequence, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMenu,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSystemTrayIcon,
        QVBoxLayout,
        QWidget,
    )
    from PySide6.QtGui import QTextBrowser
    try:
        from PySide6.QtWidgets import QTextBrowser as _TB  # noqa: F401
    except ImportError:
        pass

    resolved_index_dir = index_dir or (Path.home() / ".scholia" / "index")

    # -- Qt application ---------------------------------------------------
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Scholia")
    app.setQuitOnLastWindowClosed(False)  # keep tray alive after panel close

    # -- Engine -----------------------------------------------------------
    engine = GroundingEngine(resolved_index_dir)

    # -- Worker thread helper ---------------------------------------------
    class Worker(QObject):
        finished = Signal(object, object)  # (cite_result|None, discover_result|None)
        error = Signal(str)

        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self._fn = fn
            self._args = args
            self._kwargs = kwargs
            self._thread = QThread()
            self.moveToThread(self._thread)
            self._thread.started.connect(self._run)

        def start(self):
            self._thread.start()

        def _run(self):
            try:
                result = self._fn(*self._args, **self._kwargs)
                self.finished.emit(result[0], result[1])
            except Exception as exc:  # noqa: BLE001
                self.error.emit(str(exc))
            finally:
                self._thread.quit()

    # -- Results panel window ---------------------------------------------
    panel = QWidget()
    panel.setWindowTitle("Scholia")
    panel.setWindowFlags(
        Qt.WindowType.Window
        | Qt.WindowType.WindowStaysOnTopHint
    )
    panel.resize(480, 560)

    panel_layout = QVBoxLayout(panel)
    panel_layout.setContentsMargins(10, 10, 10, 10)
    panel_layout.setSpacing(8)

    # Header row
    hdr = QHBoxLayout()
    title_lbl = QLabel("Scholia")
    title_font = QFont()
    title_font.setPointSize(13)
    title_font.setBold(True)
    title_lbl.setFont(title_font)
    title_lbl.setStyleSheet("color: #1E3A5F;")
    hdr.addWidget(title_lbl)
    hdr.addStretch()

    always_on_top_cb = QCheckBox("Always on top")
    always_on_top_cb.setChecked(True)
    always_on_top_cb.setStyleSheet("font-size:10px; color:#6B7280;")
    hdr.addWidget(always_on_top_cb)
    panel_layout.addLayout(hdr)

    def _toggle_always_on_top(checked: bool):
        flags = panel.windowFlags()
        if checked:
            panel.setWindowFlags(flags | Qt.WindowType.WindowStaysOnTopHint)
        else:
            panel.setWindowFlags(flags & ~Qt.WindowType.WindowStaysOnTopHint)
        panel.show()

    always_on_top_cb.toggled.connect(_toggle_always_on_top)

    # Passage input area
    passage_label = QLabel("Passage (type, paste, or press Ctrl+Alt+G):")
    passage_label.setStyleSheet("font-size:11px; color:#374151; font-weight:600;")
    panel_layout.addWidget(passage_label)

    text_input = QPlainTextEdit()
    text_input.setPlaceholderText("Paste or type a sentence to ground…")
    text_input.setFixedHeight(70)
    text_input.setStyleSheet(
        "border: 1px solid #D1D5DB; border-radius: 4px; padding: 4px; font-size:11px;"
    )
    panel_layout.addWidget(text_input)

    # Button row
    btn_row = QHBoxLayout()
    btn_row.setSpacing(6)

    btn_ground = QPushButton("Ground Clipboard")
    btn_ground.setStyleSheet(
        "QPushButton { background:#2563EB; color:white; border-radius:5px;"
        " padding:5px 12px; font-size:11px; font-weight:600; }"
        "QPushButton:hover { background:#1D4ED8; }"
        "QPushButton:disabled { background:#93C5FD; }"
    )
    btn_row.addWidget(btn_ground)

    btn_ground_text = QPushButton("Ground Text")
    btn_ground_text.setStyleSheet(
        "QPushButton { background:#059669; color:white; border-radius:5px;"
        " padding:5px 12px; font-size:11px; font-weight:600; }"
        "QPushButton:hover { background:#047857; }"
        "QPushButton:disabled { background:#6EE7B7; }"
    )
    btn_row.addWidget(btn_ground_text)

    btn_discover = QPushButton("Discover")
    btn_discover.setStyleSheet(
        "QPushButton { background:#7C3AED; color:white; border-radius:5px;"
        " padding:5px 12px; font-size:11px; font-weight:600; }"
        "QPushButton:hover { background:#6D28D9; }"
        "QPushButton:disabled { background:#C4B5FD; }"
    )
    btn_row.addWidget(btn_discover)

    btn_row.addStretch()
    panel_layout.addLayout(btn_row)

    # Status bar
    status_lbl = QLabel("Loading engine…")
    status_lbl.setStyleSheet("font-size:10px; color:#9CA3AF;")
    panel_layout.addWidget(status_lbl)

    # Results area (QTextBrowser for clickable links)
    results_browser = QTextBrowser()
    results_browser.setOpenExternalLinks(True)
    results_browser.setStyleSheet(
        "border: 1px solid #E5E7EB; border-radius:4px; background:#FFFFFF;"
        " font-size:11px;"
    )
    results_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # Show idle state initially
    results_browser.setHtml(_results_html())
    panel_layout.addWidget(results_browser, 1)

    # Hotkey label at the bottom
    hotkey_lbl = QLabel(f"Hotkey: Ctrl+Alt+G — captures selection from any app")
    hotkey_lbl.setStyleSheet("font-size:9px; color:#9CA3AF; margin-top:2px;")
    panel_layout.addWidget(hotkey_lbl)

    # -- Ctrl+Enter shortcut inside the panel ----------------------------
    ground_shortcut = QShortcut(QKeySequence("Ctrl+Return"), panel)

    # -- State management ------------------------------------------------
    _active_worker: list[Worker] = []

    def _set_busy(busy: bool):
        btn_ground.setEnabled(not busy)
        btn_ground_text.setEnabled(not busy)
        btn_discover.setEnabled(not busy)
        if busy:
            results_browser.setHtml(_results_html(loading=True))

    def _show_error(msg: str):
        _set_busy(False)
        results_browser.setHtml(_results_html(error=msg))

    def _show_results(cite_result, discover_result):
        _set_busy(False)
        html_str = _results_html(cite_result=cite_result, discover_result=discover_result)
        results_browser.setHtml(html_str)

    def _run_ground(passage: str, also_discover: bool = False):
        if not engine.ready:
            _show_error("Engine still loading — please wait a moment.")
            return
        if not passage.strip():
            _show_error("Nothing to ground — enter a passage first.")
            return

        _set_busy(True)
        text_input.setPlainText(passage)

        def _work():
            cite_res = engine.cite(passage)
            disc_res = engine.discover(passage) if also_discover else None
            return cite_res, disc_res

        w = Worker(_work)
        w.finished.connect(lambda c, d: _show_results(c, d))
        w.error.connect(_show_error)
        _active_worker.clear()
        _active_worker.append(w)
        w.start()

    def _run_discover(passage: str):
        if not engine.ready:
            _show_error("Engine still loading — please wait a moment.")
            return
        if not passage.strip():
            _show_error("Nothing to discover from — enter a passage first.")
            return

        _set_busy(True)

        def _work():
            disc_res = engine.discover(passage)
            return None, disc_res

        w = Worker(_work)
        w.finished.connect(lambda c, d: _show_results(c, d))
        w.error.connect(_show_error)
        _active_worker.clear()
        _active_worker.append(w)
        w.start()

    def _on_ground_clipboard():
        """Ground whatever is currently in the clipboard."""
        from PySide6.QtWidgets import QApplication as _App
        from PySide6.QtGui import QClipboard
        text = _App.clipboard().text(QClipboard.Mode.Clipboard).strip()
        if not text:
            _show_error("Clipboard is empty — copy some text first.")
            return
        _run_ground(text)

    def _on_ground_text():
        passage = text_input.toPlainText().strip()
        _run_ground(passage)

    def _on_discover():
        passage = text_input.toPlainText().strip()
        _run_discover(passage)

    btn_ground.clicked.connect(_on_ground_clipboard)
    btn_ground_text.clicked.connect(_on_ground_text)
    btn_discover.clicked.connect(_on_discover)
    ground_shortcut.activated.connect(_on_ground_text)

    # -- Global hotkey via pynput ----------------------------------------
    _hotkey_listener = None

    def _hotkey_triggered():
        """Called from the pynput thread — must dispatch to Qt main thread."""
        # Robust selection capture: send Ctrl+C, then read clipboard.
        # We do this on a background thread so we don't block pynput.
        def _capture_and_ground():
            try:
                from pynput.keyboard import Controller as KbController, Key
                kb = KbController()
                # Save current clipboard
                from PySide6.QtWidgets import QApplication as _App
                from PySide6.QtGui import QClipboard
                old_clip = _App.clipboard().text(QClipboard.Mode.Clipboard)
                # Send Ctrl+C to the focused window
                kb.press(Key.ctrl)
                kb.press("c")
                time.sleep(0.12)  # let the copy settle
                kb.release("c")
                kb.release(Key.ctrl)
                time.sleep(0.1)
                new_clip = _App.clipboard().text(QClipboard.Mode.Clipboard).strip()
                passage = new_clip if new_clip and new_clip != old_clip else new_clip
            except Exception:  # noqa: BLE001
                passage = ""

            # Schedule grounding on Qt main thread via QTimer.
            def _on_main():
                if passage:
                    _run_ground(passage)
                    panel.show()
                    panel.raise_()
                    panel.activateWindow()
                else:
                    panel.show()
                    panel.raise_()
                    panel.activateWindow()
                    _show_error("No text captured. Try copying text first, then press Ctrl+Alt+G.")

            QTimer.singleShot(0, _on_main)

        t = threading.Thread(target=_capture_and_ground, daemon=True)
        t.start()

    try:
        from pynput import keyboard as _kb

        # Parse the combo string into pynput HotKey format.
        _hotkey = _kb.HotKey(
            _kb.HotKey.parse(HOTKEY_COMBO),
            _hotkey_triggered,
        )

        def _for_canonical(f):
            def _inner(key):
                f(_listener.canonical(key))
            return _inner

        _listener = _kb.Listener(
            on_press=_for_canonical(_hotkey.press),
            on_release=_for_canonical(_hotkey.release),
        )
        _listener.daemon = True
        _listener.start()
        _hotkey_listener = _listener
        hotkey_lbl.setText(f"Hotkey: Ctrl+Alt+G  (captures selection from any app)")
    except Exception as exc:  # noqa: BLE001
        hotkey_lbl.setText(f"Hotkey unavailable: {exc}")
        hotkey_lbl.setStyleSheet("font-size:9px; color:#DC2626;")

    # -- System tray icon ------------------------------------------------
    tray_icon = _get_tray_icon()
    tray = QSystemTrayIcon(tray_icon, app)
    tray.setToolTip("Scholia — citation grounding")

    tray_menu = QMenu()
    act_open = QAction("Open Panel", app)
    act_ground = QAction("Ground Clipboard", app)
    act_sep = tray_menu.addSeparator()
    act_quit = QAction("Quit Scholia", app)

    tray_menu.addAction(act_open)
    tray_menu.addAction(act_ground)
    tray_menu.addSeparator()
    tray_menu.addAction(act_quit)
    tray.setContextMenu(tray_menu)

    def _show_panel():
        panel.show()
        panel.raise_()
        panel.activateWindow()

    act_open.triggered.connect(_show_panel)
    act_ground.triggered.connect(lambda: (_show_panel(), _on_ground_clipboard()))
    act_quit.triggered.connect(app.quit)
    tray.activated.connect(
        lambda reason: _show_panel()
        if reason == QSystemTrayIcon.ActivationReason.Trigger
        else None
    )

    tray.show()

    # -- Engine loading ---------------------------------------------------
    def _on_engine_loaded(error: str | None):
        """Called on engine background thread — schedule Qt update via QTimer."""
        def _update():
            if error:
                status_lbl.setText(f"Engine error: {error}")
                status_lbl.setStyleSheet("font-size:10px; color:#DC2626;")
                results_browser.setHtml(_results_html(error=error))
                btn_ground.setEnabled(False)
                btn_ground_text.setEnabled(False)
                btn_discover.setEnabled(False)
                tray.showMessage(
                    "Scholia — error",
                    f"Failed to load: {error}",
                    QSystemTrayIcon.MessageIcon.Critical,
                    4000,
                )
            else:
                n = engine.n_papers
                status_lbl.setText(f"Ready  |  {n} papers  |  Ctrl+Alt+G to ground")
                status_lbl.setStyleSheet("font-size:10px; color:#059669;")
                results_browser.setHtml(_results_html())
                btn_ground.setEnabled(True)
                btn_ground_text.setEnabled(True)
                btn_discover.setEnabled(True)
                tray.showMessage(
                    "Scholia ready",
                    f"{n} papers indexed. Press Ctrl+Alt+G to ground any selection.",
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )

        QTimer.singleShot(0, _update)

    # Disable buttons until engine is ready
    btn_ground.setEnabled(False)
    btn_ground_text.setEnabled(False)
    btn_discover.setEnabled(False)

    # Show panel immediately (loading state), then start engine
    panel.show()
    engine.load_async(_on_engine_loaded)

    # -- Run the event loop -----------------------------------------------
    exit_code = app.exec()

    # Cleanup
    if _hotkey_listener is not None:
        try:
            _hotkey_listener.stop()
        except Exception:  # noqa: BLE001
            pass

    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Module entry point: python -m scholia.app
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_app()
