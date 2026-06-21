@echo off
:: Scholia Desktop App — system tray + results panel + Ctrl+Alt+G hotkey
:: Double-click to launch. No console window after startup.
:: Requirements: pip install "scholia[overlay]" pynput

set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set PYTHONUTF8=1
set TRANSFORMERS_VERBOSITY=error
set HF_HUB_DISABLE_PROGRESS_BARS=1
set TOKENIZERS_PARALLELISM=false

:: Launch without a visible console window (pythonw = no console).
:: Falls back to python if pythonw is not found.
start "" pythonw -m scholia.app
if errorlevel 1 (
    start "" python -m scholia.app
)
