#Requires -Version 5.1
<#
.SYNOPSIS
    Scholia Installer -- creates Start Menu and Desktop shortcuts.

.DESCRIPTION
    Creates a proper Windows .lnk shortcut with the Scholia icon that launches
    the tray app via pythonw.exe (no console window, no visible script).

.PARAMETER SkipDesktop
    Skip creating the Desktop shortcut (Start Menu only).

.PARAMETER Uninstall
    Remove Start Menu and Desktop shortcuts.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install_scholia.ps1
    powershell -ExecutionPolicy Bypass -File install_scholia.ps1 -SkipDesktop
    powershell -ExecutionPolicy Bypass -File install_scholia.ps1 -Uninstall
#>
param(
    [switch]$SkipDesktop,
    [switch]$Uninstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$AppName    = "Scholia"
$IconDir    = Join-Path $env:USERPROFILE ".scholia"
$IconDest   = Join-Path $IconDir "scholia.ico"
$StartMenu  = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$LnkStart   = Join-Path $StartMenu "$AppName.lnk"
$LnkDesktop = Join-Path ([Environment]::GetFolderPath("Desktop")) "$AppName.lnk"

# ── Uninstall path ─────────────────────────────────────────────────────────────
if ($Uninstall) {
    Write-Host "Removing Scholia shortcuts..."
    if (Test-Path $LnkStart)   { Remove-Item $LnkStart   -Force; Write-Host "  Removed: $LnkStart" }
    if (Test-Path $LnkDesktop) { Remove-Item $LnkDesktop -Force; Write-Host "  Removed: $LnkDesktop" }
    Write-Host "Scholia shortcuts removed."
    exit 0
}

# ── Locate pythonw.exe ─────────────────────────────────────────────────────────
Write-Host "Locating Python..."
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    Write-Host "ERROR: Python not in PATH. Install Python 3.11+ from python.org." -ForegroundColor Red
    exit 1
}
$PythonDir  = Split-Path $PythonExe
$PythonwExe = Join-Path $PythonDir "pythonw.exe"
if (-not (Test-Path $PythonwExe)) {
    $pw = Get-Command pythonw -ErrorAction SilentlyContinue
    if ($pw) { $PythonwExe = $pw.Source }
}
if (-not (Test-Path $PythonwExe)) {
    Write-Host "ERROR: pythonw.exe not found. Cannot create a no-console shortcut." -ForegroundColor Red
    exit 1
}
Write-Host "  pythonw.exe: $PythonwExe"

# ── Verify scholia is installed ────────────────────────────────────────────────
Write-Host "Verifying scholia..."
$check = & python -c "import scholia.app; print('ok')" 2>&1
if ($check -notmatch "ok") {
    Write-Host "ERROR: scholia not installed. Run: pip install scholia[overlay]" -ForegroundColor Red
    exit 1
}
Write-Host "  scholia: OK"

# ── Warn if index is missing ───────────────────────────────────────────────────
$IndexDir = Join-Path $env:USERPROFILE ".scholia\index"
if (-not (Test-Path $IndexDir)) {
    Write-Host "  WARNING: No index found. Build it: scholia index --corpus DIR" -ForegroundColor Yellow
}

# ── Copy icon to stable location ──────────────────────────────────────────────
Write-Host "Installing icon..."
if (-not (Test-Path $IconDir)) { New-Item -ItemType Directory -Force -Path $IconDir | Out-Null }

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$IconSource = Join-Path $ScriptDir "..\assets\scholia.ico"
if (-not (Test-Path $IconSource)) {
    $IconSource = Join-Path $ScriptDir "scholia.ico"
}
if (Test-Path $IconSource) {
    Copy-Item $IconSource $IconDest -Force
    Write-Host "  Icon: $IconDest"
} else {
    Write-Host "  WARNING: scholia.ico not found -- using default icon." -ForegroundColor Yellow
    $IconDest = $null
}

# ── Build shortcut arguments ───────────────────────────────────────────────────
# pythonw -c "..." sets env vars then calls run_app() -- fully no-console.
$code = "import os; os.environ.setdefault('OPENBLAS_NUM_THREADS','4'); " +
        "os.environ.setdefault('OMP_NUM_THREADS','4'); " +
        "os.environ.setdefault('MKL_NUM_THREADS','4'); " +
        "os.environ.setdefault('TRANSFORMERS_VERBOSITY','error'); " +
        "os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1'); " +
        "os.environ.setdefault('TOKENIZERS_PARALLELISM','false'); " +
        "from scholia.app import run_app; run_app()"
$ShortcutArgs = "-c `"$code`""

# ── Helper: create a .lnk via WScript.Shell ───────────────────────────────────
function New-WindowsShortcut {
    param([string]$Path, [string]$Target, [string]$Arguments, [string]$Icon, [string]$Desc, [string]$WorkDir)
    $WS = New-Object -ComObject WScript.Shell
    $S  = $WS.CreateShortcut($Path)
    $S.TargetPath       = $Target
    $S.Arguments        = $Arguments
    $S.WorkingDirectory = $WorkDir
    $S.Description      = $Desc
    if ($Icon) { $S.IconLocation = $Icon }
    $S.WindowStyle = 7   # minimized = suppresses flash; Qt takes over immediately
    $S.Save()
    Write-Host "  Created: $Path"
}

$WorkDir = Join-Path $env:USERPROFILE ".scholia"
$Desc    = "Scholia - local citation grounding (Ctrl+Alt+G)"
$Icon    = if ($IconDest -and (Test-Path $IconDest)) { "$IconDest,0" } else { "" }

Write-Host "Creating shortcuts..."
New-WindowsShortcut -Path $LnkStart -Target $PythonwExe -Arguments $ShortcutArgs `
                    -Icon $Icon -Desc $Desc -WorkDir $WorkDir

if (-not $SkipDesktop) {
    New-WindowsShortcut -Path $LnkDesktop -Target $PythonwExe -Arguments $ShortcutArgs `
                        -Icon $Icon -Desc $Desc -WorkDir $WorkDir
}

Write-Host ""
Write-Host "Scholia installed!" -ForegroundColor Green
Write-Host "  Start Menu: search 'Scholia' in the Start Menu"
if (-not $SkipDesktop) { Write-Host "  Desktop: Scholia icon on your Desktop" }
Write-Host ""
Write-Host "First run takes 10-30s to load ML models (cached after that)."
