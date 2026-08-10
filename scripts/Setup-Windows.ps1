# Setup-Windows.ps1 - set up SquidXplorer on Windows with a plain Python venv (NO conda) and put a
# "SquidXplorer" shortcut on the Desktop.
#
# Run once, from the repo root, in Windows PowerShell:
#     powershell -ExecutionPolicy Bypass -File scripts\Setup-Windows.ps1
#
# Requires Python 3.10+ and git. The venv is self-contained, so the shortcut runs the venv's own
# pythonw directly - no activation, nothing global touched.

# NOTE: not using -ErrorAction Stop globally, because native tools (py, pip) legitimately write to
# stderr and that would otherwise abort the script. We check exit codes explicitly and Die on failure.
$ErrorActionPreference = "Continue"
# The Desktop icon's name. Was "MIP tool", which the README had already stopped calling it; the
# launcher below owns the module name now, so there is no $Module here any more.
$AppName = "SquidXplorer"
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
Write-Host "Installing SquidXplorer and its dependencies (first time takes a few minutes) ..."
& $vpy -m pip install --upgrade pip
# EDITABLE install of the app: after this, a `git pull` in the repo takes effect on the next launch
# with no reinstall (only the pinned deps below are a fixed snapshot).
& $vpy -m pip install -e ($repo + "[gui]")
if ($LASTEXITCODE -ne 0) {
    Die "pip install failed (see the errors above). This usually means a package has no wheel for this Python version; tell Julio the error and we'll pin a version."
}

# 4. Install the launcher + icon NEXT TO the venv, not in the checkout. The shortcut points at the
#    installed copy, so a branch switch, `git clean`, or an upstream sync inside the repo can never
#    delete or revert what the Desktop icon runs. __REPO__ is stamped with this checkout's path.
$launcherSrc = Join-Path $PSScriptRoot "launch-squidxplorer.cmd"
$iconSrc     = Join-Path $PSScriptRoot "squidxplorer.ico"
$appDir      = Join-Path $env:LOCALAPPDATA "squidmip"
$launcher    = Join-Path $appDir "launch-squidxplorer.cmd"
$icon        = Join-Path $appDir "squidxplorer.ico"

if (-not (Test-Path $launcherSrc)) { Die ("Missing " + $launcherSrc) }
(Get-Content $launcherSrc -Raw).Replace("__REPO__", $repo) |
    Set-Content $launcher -Encoding ASCII
if (Test-Path $iconSrc) { Copy-Item $iconSrc $icon -Force }

# 5. Desktop shortcut -> the launcher. It runs python.exe (NOT pythonw) ON PURPOSE so a console
#    window opens alongside the app with the logs/errors + [footprint] lines, and the launcher holds
#    that window open when the app exits nonzero so a traceback cannot flash past.
#
#    Because the target is a .cmd taking %*, you can also DRAG AN ACQUISITION FOLDER ONTO THE ICON
#    to open it directly, quoting intact.
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop ($AppName + ".lnk")
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $launcher
$sc.WorkingDirectory = $repo
if (Test-Path $icon) { $sc.IconLocation = $icon + ",0" } else { $sc.IconLocation = $vpy + ",0" }
$sc.Description = "SquidXplorer - local viewer for Squid HCS acquisitions"
$sc.WindowStyle = 1
$sc.Save()

# Retire the pre-rename icon so the Desktop does not keep two that do the same thing.
$stale = Join-Path $desktop "MIP tool.lnk"
if (($stale -ne $lnk) -and (Test-Path $stale)) { Remove-Item $stale -Force -ErrorAction SilentlyContinue }

Write-Host ""
Write-Host ("Done. '" + $AppName + "' is on your Desktop - double-click it, or drag an acquisition")
Write-Host "folder straight onto the icon to open that acquisition."
