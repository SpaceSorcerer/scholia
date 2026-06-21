@echo off
REM Scholia — Start Bridge for Word Add-in
REM Serves the task pane over HTTPS (127.0.0.1:8765) so the Office add-in can load.
REM First run: generates a self-signed cert in %USERPROFILE%\.scholia\
REM You must trust that cert in Windows before the add-in pane will load.
REM See word-addin\SIDELOAD_WORD.md for full instructions.
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python -m scholia serve --serve-addin
pause
