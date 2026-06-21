# Scholia Packaging & Release Guide

## What "installing" Scholia means

Scholia is a Python application with heavy ML dependencies (torch, sentence-transformers,
faiss). Two packaging approaches are supported:

### Approach 1 — Installer + Shortcut (PRIMARY, recommended for GitHub releases)

**What it is:** A PowerShell installer script (`installer/install_scholia.ps1`) that:
1. Verifies Python + scholia are installed.
2. Copies the icon to `~/.scholia/scholia.ico`.
3. Creates a real Windows `.lnk` shortcut in the Start Menu (and optionally Desktop)
   pointing to `pythonw.exe` with the Scholia icon — **no console window, no visible script**.

**Why this is the right approach:** torch alone is ~500 MB; a self-contained PyInstaller
bundle would be 1.5-2 GB and is impractical as a GitHub release asset. The installer
is tiny (two files: `.ps1` + `.ico`), reproducible, and produces a result that is
indistinguishable from a native app for the user.

**User experience:** Start Menu shows "Scholia" with the blue-circle-S icon. Double-click
launches the tray app cleanly — no console window, no flash, icon in the system tray
within 10-30 seconds (ML model load; cached after first use).

### Approach 2 — PyInstaller onedir (OPTIONAL, large, for offline/standalone use)

Run: `.\build.ps1 -PyInstaller`

Output: `dist/Scholia/Scholia.exe` + support tree.
**VERIFIED 2026-06-21:** builds successfully (23,660 files, ~1.86 GB total; the
`Scholia.exe` bootloader is 86 MB, the rest is torch + ML deps). PE subsystem is
GUI (2) = no console window. Smoke-tested: the exe launches the tray app and
stays running with no crash. Self-contained (no Python required) but too large
for GitHub release assets — suitable for local use or enterprise distribution
via network share.

---

## Building locally

```powershell
# From repo root — runs tests + creates Start Menu shortcut (fast):
.\build.ps1

# Also attempt PyInstaller onedir (slow, ~1.5-2 GB):
.\build.ps1 -PyInstaller
```

The installer can also be run standalone:
```powershell
# Install (Start Menu + Desktop):
powershell -ExecutionPolicy Bypass -File installer\install_scholia.ps1

# Install (Start Menu only):
powershell -ExecutionPolicy Bypass -File installer\install_scholia.ps1 -SkipDesktop

# Uninstall shortcuts:
powershell -ExecutionPolicy Bypass -File installer\install_scholia.ps1 -Uninstall
```

---

## GitHub Release (when Brian visually validates the app)

### Prerequisites
- Brian has confirmed the tray icon appears and the panel opens correctly.
- CI is green on GitHub Actions.

### Release artifacts
The release should include:
- `install_scholia.ps1` — the installer script
- `scholia.ico` — the icon (bundled alongside the installer)
- `scholia.spec` — for anyone who wants to build the PyInstaller exe locally

### The exact command to cut the release

```bash
# From repo root, after confirming Brian's visual validation:
gh release create v0.2.0 \
  installer/install_scholia.ps1 \
  installer/scholia.ico \
  scholia.spec \
  --title "Scholia v0.2.0 — Desktop App" \
  --notes "## Scholia v0.2.0

### What's new
- System-tray desktop app (Ctrl+Alt+G global hotkey, results panel, no console)
- Word Office.js Add-in (sideload via Word Desktop or Word Online)
- Browser extension for Word Online (MV3, Edge/Chrome)
- Full local citation grounding + discovery (361-paper library)
- Self-signed HTTPS bridge for the Office Add-in

### Installation (Windows)

1. Install Python 3.11+ and scholia:
   \`\`\`
   pip install \"scholia[overlay]\"
   \`\`\`
2. Build your index:
   \`\`\`
   scholia index --corpus <your-zotero-mirror-dir>
   \`\`\`
3. Run the installer (creates Start Menu + Desktop shortcut with icon):
   \`\`\`
   powershell -ExecutionPolicy Bypass -File install_scholia.ps1
   \`\`\`
4. Click **Scholia** in the Start Menu (or Desktop) — no console window.

### Requirements
- Windows 10/11, Python 3.11–3.13
- pip install 'scholia[overlay]'  (PySide6 + pynput)
- A built Scholia index (~/.scholia/index/)
"
```

---

## Reproducing the PyInstaller build

For contributors who want to build a self-contained `.exe`:

```powershell
# Install build dep
pip install pyinstaller

# Build (from repo root)
python -m PyInstaller scholia.spec --clean --noconfirm

# Output: dist/Scholia/Scholia.exe (~1.5-2 GB directory)
```

The spec (`scholia.spec`) includes all necessary `collect_all()` calls for torch,
sentence-transformers, transformers, faiss, PySide6, pynput, and scholia itself.
UPX compression is enabled to reduce size.

---

## Files committed for packaging

| File | Purpose |
|------|---------|
| `scholia.spec` | PyInstaller spec (reproducible exe build) |
| `build.ps1` | One-command build (tests + installer + optional PyInstaller) |
| `installer/install_scholia.ps1` | Installer: creates Start Menu/Desktop .lnk with icon |
| `installer/scholia.ico` | Icon bundled with installer (gitignored copy; source: `assets/scholia.ico`) |
| `_pyinstaller_entry.py` | PyInstaller entry point (sets env vars, calls run_app) |
| `assets/scholia.ico` | Master icon source |

## Files NOT committed (gitignored)

| Pattern | Why |
|---------|-----|
| `dist/` | PyInstaller output (large binaries) |
| `build/` | PyInstaller build cache |
| `*.exe` | Compiled executables |
| `installer/scholia.ico` | Generated copy (source in `assets/`) |
