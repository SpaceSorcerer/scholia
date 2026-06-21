#Requires -Version 5.1
<#
.SYNOPSIS
    Scholia packaging build script.

.DESCRIPTION
    Runs tests, prepares the installer package, runs the local installer
    (creates Start Menu shortcut), and optionally builds the PyInstaller onedir.

    Usage:
        .\build.ps1              # installer only (fast, recommended)
        .\build.ps1 -PyInstaller # also build PyInstaller onedir (~1.5-2 GB)

.PARAMETER PyInstaller
    Also run the PyInstaller onedir build (slow, large).
#>
param([switch]$PyInstaller)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=== Scholia Packaging Build ===" -ForegroundColor Cyan

# ── Tests ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "1. Running test suite..." -ForegroundColor Yellow
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS      = "1"
$env:MKL_NUM_THREADS      = "1"
$result = python -m pytest -q 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests FAILED. Fix before packaging." -ForegroundColor Red
    $result
    exit 1
}
# Print last few lines (pass/fail summary)
($result | Select-Object -Last 3) | Write-Host
Write-Host "   Tests: PASSED" -ForegroundColor Green

# ── Prepare installer package ──────────────────────────────────────────────────
Write-Host ""
Write-Host "2. Preparing installer package..." -ForegroundColor Yellow
$InstallerDir = Join-Path $RepoRoot "installer"
if (-not (Test-Path $InstallerDir)) { New-Item -ItemType Directory -Path $InstallerDir | Out-Null }

$IconSrc  = Join-Path $RepoRoot "assets\scholia.ico"
$IconDest = Join-Path $InstallerDir "scholia.ico"
if (Test-Path $IconSrc) {
    Copy-Item $IconSrc $IconDest -Force
    Write-Host "   Icon staged: installer\scholia.ico"
} else {
    Write-Host "   WARNING: assets\scholia.ico not found." -ForegroundColor Yellow
}
Write-Host "   Installer ready: installer\install_scholia.ps1"

# ── Run local installer (creates Start Menu shortcut) ─────────────────────────
Write-Host ""
Write-Host "3. Installing (creates Start Menu shortcut)..." -ForegroundColor Yellow
$installerScript = Join-Path $InstallerDir "install_scholia.ps1"
powershell -ExecutionPolicy Bypass -File $installerScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "   WARNING: installer exited $LASTEXITCODE" -ForegroundColor Yellow
} else {
    Write-Host "   Start Menu shortcut created." -ForegroundColor Green
}

# ── Optional PyInstaller build ─────────────────────────────────────────────────
if ($PyInstaller) {
    Write-Host ""
    Write-Host "4. PyInstaller onedir build..." -ForegroundColor Yellow
    Write-Host "   NOTE: output will be ~1.5-2 GB. This takes several minutes." -ForegroundColor DarkYellow

    $SpecFile = Join-Path $RepoRoot "scholia.spec"
    python -m PyInstaller $SpecFile --clean --noconfirm
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   PyInstaller FAILED." -ForegroundColor Red
        exit 1
    }

    $ExePath = Join-Path $RepoRoot "dist\Scholia\Scholia.exe"
    if (Test-Path $ExePath) {
        $SizeMB = [math]::Round(
            (Get-ChildItem (Join-Path $RepoRoot "dist\Scholia") -Recurse |
             Measure-Object -Property Length -Sum).Sum / 1MB, 0)
        Write-Host "   Build: SUCCESS  ($SizeMB MB)" -ForegroundColor Green
        Write-Host "   Artifact: dist\Scholia\Scholia.exe"

        # 10-second smoke test
        Write-Host "   Smoke test (10s)..."
        $proc = Start-Process -FilePath $ExePath -PassThru
        Start-Sleep -Seconds 10
        if ($proc.HasExited) {
            Write-Host "   WARNING: Scholia.exe exited early ($($proc.ExitCode))." -ForegroundColor Yellow
        } else {
            Write-Host "   Smoke test: RUNNING (no crash)." -ForegroundColor Green
            $proc.Kill()
        }
    } else {
        Write-Host "   dist\Scholia\Scholia.exe not found after build." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "PRIMARY RESULT: Start Menu 'Scholia' shortcut -- no console window." -ForegroundColor Green
Write-Host "See PACKAGING.md for the gh release create command."
