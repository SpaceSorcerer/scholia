"""PyInstaller entry point for the Scholia desktop app.

This file is ONLY used by PyInstaller (scholia.spec) to produce Scholia.exe.
It is not a Python package file and is gitignored after the first commit of
scholia.spec. Do not import or run this file directly — use ``scholia app``
or ``python -m scholia.app`` instead.
"""

import os

# Thread caps and HF-quiet flags — set before any import touches them.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scholia.app import run_app  # noqa: E402

if __name__ == "__main__":
    run_app()
