# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Scholia desktop app (Windows, onedir, no-console).
#
# Build with:
#   python -m PyInstaller scholia.spec
#
# Output: dist/Scholia/Scholia.exe  (+ the support tree alongside it)
#
# NOTE on bundle size: torch alone is ~500 MB on disk; the full onedir bundle
# will be ~1.5–2 GB.  This is intentional — it is a fully self-contained,
# dependency-free directory that any Windows user can run without Python.
# For a lightweight GitHub release, prefer the installer/install_scholia.ps1
# path instead (requires Python + scholia installed; creates a proper shortcut).

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# ── Collect heavy ML stack ──────────────────────────────────────────────────
torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
st_datas, st_binaries, st_hiddenimports = collect_all("sentence_transformers")
tf_datas, tf_binaries, tf_hiddenimports = collect_all("transformers")
faiss_datas, faiss_binaries, faiss_hiddenimports = collect_all("faiss")

# einops is required by the nomic-embed-text-v1.5 trust_remote_code path
einops_datas, einops_binaries, einops_hiddenimports = collect_all("einops")

# tokenizers (Rust-backed HF tokenizer — must come along with transformers)
tok_datas, tok_binaries, tok_hiddenimports = collect_all("tokenizers")

# huggingface_hub needed for model download / cache resolution
hub_datas, hub_binaries, hub_hiddenimports = collect_all("huggingface_hub")

# cryptography (TLS cert generation for --serve-addin)
crypto_datas, crypto_binaries, crypto_hiddenimports = collect_all("cryptography")

# PySide6 — collect_all handles Qt plugins, platform DLLs, etc.
pyside6_datas, pyside6_binaries, pyside6_hiddenimports = collect_all("PySide6")

# pynput (global hotkey)
pynput_datas, pynput_binaries, pynput_hiddenimports = collect_all("pynput")

# scholia itself
scholia_datas, scholia_binaries, scholia_hiddenimports = collect_all("scholia")

# ── Aggregate ───────────────────────────────────────────────────────────────
all_datas = (
    torch_datas + st_datas + tf_datas + faiss_datas
    + einops_datas + tok_datas + hub_datas + crypto_datas
    + pyside6_datas + pynput_datas + scholia_datas
)
all_binaries = (
    torch_binaries + st_binaries + tf_binaries + faiss_binaries
    + einops_binaries + tok_binaries + hub_binaries + crypto_binaries
    + pyside6_binaries + pynput_binaries + scholia_binaries
)
all_hiddenimports = (
    torch_hiddenimports + st_hiddenimports + tf_hiddenimports + faiss_hiddenimports
    + einops_hiddenimports + tok_hiddenimports + hub_hiddenimports + crypto_hiddenimports
    + pyside6_hiddenimports + pynput_hiddenimports + scholia_hiddenimports
    + [
        # Explicit extras that collect_all sometimes misses
        "scholia.app",
        "scholia.cli",
        "scholia.server",
        "scholia.embedder",
        "scholia.corpus",
        "scholia.index",
        "scholia.retrieval",
        "scholia.cite",
        "scholia.discovery",
        "scholia.mirror",
        "scholia.writing",
        "scholia.entailment",
        # torch internals
        "torch._C",
        "torch._C._VariableFunctions",
        "torch.jit",
        "torch.nn",
        "torch.nn.functional",
        # numpy / scipy helpers loaded at runtime
        "numpy",
        "numpy.core",
        "scipy",
        "scipy.special",
        # faiss extras
        "faiss",
        "_swigfaiss",
        # yaml
        "yaml",
        # click
        "click",
    ]
)

# ── Analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    # Entry point: dedicated GUI launcher script (keeps __main__.py as CLI)
    ["_pyinstaller_entry.py"],
    pathex=["src"],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude test frameworks from the frozen app
        "pytest",
        "_pytest",
        "pytest_asyncio",
        # Exclude heavy unused torch extras
        "torch.distributed",
        "torch.testing",
        # Exclude IPython / Jupyter (not needed at runtime)
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# ── Executable ───────────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Scholia",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # <-- NO console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/scholia.ico",  # <-- our logo
    version_file=None,
)

# ── One-directory bundle ──────────────────────────────────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Scholia",
)
