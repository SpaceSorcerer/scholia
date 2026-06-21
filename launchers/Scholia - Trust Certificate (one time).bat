@echo off
REM Scholia — Trust Localhost Certificate (one-time setup)
REM Installs the self-signed localhost cert into the CurrentUser Trusted Root
REM store so the Word add-in task pane loads without a security warning.
REM No admin elevation required. Safe to re-run (idempotent).
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set PYTHONUTF8=1
cd /d E:\Claude\scholia
python -m scholia trust-cert
pause
