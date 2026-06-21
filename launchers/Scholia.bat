@echo off
:: Scholia Desktop App — system tray + results panel + Ctrl+Alt+G hotkey
:: Double-click this to start Scholia. A tray icon will appear.
:: Press Ctrl+Alt+G from Word (or anywhere) to ground selected text.

set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set PYTHONUTF8=1
set TRANSFORMERS_VERBOSITY=error
set HF_HUB_DISABLE_PROGRESS_BARS=1
set TOKENIZERS_PARALLELISM=false

:: Start without a visible console window.
start "" pythonw -m scholia.app
if errorlevel 1 (
    start "" python -m scholia.app
)
