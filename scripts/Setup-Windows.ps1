# Setup-Windows.ps1 - set up the MIP tool on Windows with a plain Python venv (NO conda) and put a
# "MIP tool" shortcut on the Desktop.
#
# Run once, from the repo root, in Windows PowerShell:
#     powershell -ExecutionPolicy Bypass -File scripts\Setup-Windows.ps1
#
# Requires Python 3.10+ and git. The venv is self-contained, so the shortcut runs the venv's own
# pythonw directly - no activation, nothing global touched.

# NOTE: not using -ErrorAction Stop globally, because native tools (py, pip) legitimately write to
# stderr and that would otherwise abort the script. We check exit codes explicitly and Die on failure.
$ErrorActionPreference = "Continue"
$AppName = "MIP tool"
$Module  = "squidmip._viewer"
$repo = Split-Path $PSScriptRoot -Parent

function Die($msg) { Write-Host ""; Write-Host ("ERROR: " + $msg) -ForegroundColor Red; exit 1 }

# 1. Pick a known-good Python from what's INSTALLED (parse 'py --list'; avoid launching missing ones,
#    and prefer 3.11/3.10/3.12 over a brand-new default that may lack wheels).
$pyExe = $null; $pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $listing = (cmd /c "py --list 2>&1" | Out-String)
    $avail = @()
    foreach ($m in [regex]::Matches($listing, '3\.(1[0-9])')) { $avail += [int]$m.Groups[1].Value }
    $avail = $avail | Sort-Object -Unique
    $pick = @(11, 10, 12, 13) | Where-Object { $avail -contains $_ } | Select-Object -First 1
    if (-not $pick -and $avail.Count -gt 0) { $pick = ($avail | Sort-Object | Select-Object -First 1) }
    if ($pick) { $pyExe = "py"; $pyArgs = @("-3.$pick") }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pyExe = (Get-Command python).Source
}
if (-not $pyExe) {
    Die "No Python 3.10+ found. Install Python 3.11 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then re-run."
}
$ver = (& $pyExe @pyArgs --version 2>&1 | Out-String).Trim()
Write-Host ("Using " + $ver)

# 2. Create the venv (once).
$venv = Join-Path $env:LOCALAPPDATA "squidmip\venv"
$vpy  = Join-Path $venv "Scripts\python.exe"
$vpyw = Join-Path $venv "Scripts\pythonw.exe"
if (-not (Test-Path $vpy)) {
    Write-Host ("Creating virtual environment at " + $venv + " ...")
    & $pyExe @pyArgs -m venv $venv
    if (-not (Test-Path $vpy)) { Die "Could not create the virtual environment." }
}

# 3. Install the app + GUI deps. First run downloads a few packages.
Write-Host "Installing the MIP tool and its dependencies (first time takes a few minutes) ..."
& $vpy -m pip install --upgrade pip
# EDITABLE install of the app: after this, a `git pull` in the repo takes effect on the next launch
# with no reinstall (only the pinned deps below are a fixed snapshot).
& $vpy -m pip install -e ($repo + "[gui]")
if ($LASTEXITCODE -ne 0) {
    Die "pip install failed (see the errors above). This usually means a package has no wheel for this Python version; tell Julio the error and we'll pin a version."
}

# 4. Desktop shortcut -> venv python.exe -m module. Uses python.exe (NOT pythonw) ON PURPOSE so a
#    console window opens alongside the app, showing logs/errors + the [footprint] lines. Close that
#    window to quit the app.
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop ($AppName + ".lnk")
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $vpy
$sc.Arguments = "-m " + $Module
$sc.WorkingDirectory = $env:USERPROFILE
$sc.IconLocation = $vpy + ",0"
$sc.Description = $AppName
$sc.Save()

Write-Host ""
Write-Host ("Done. '" + $AppName + "' is on your Desktop - double-click it, then drop an acquisition folder.")
