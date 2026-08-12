---
name: run-squidxplorer
description: Launch and drive the SquidXplorer GUI on Windows, capture its windows, and run its test suite. Use when asked to run, start, screenshot, or verify the app on this machine.
---

Launch the SquidXplorer desktop GUI and interact with it. Windows-specific; the
recipe below was cold-started and verified on Windows 11 / Python 3.12.

## The interpreter to use

**Always use the venv `Setup-Windows.ps1` created — never bare `python`.**

```
C:\Users\<user>\AppData\Local\squidxplorer\venv\Scripts\python.exe
```

The repo checkout is installed into it editable (`pip install -e .[gui]`), so a
`git pull` applies live with no reinstall. Bare `python` on this machine fails at
`import squidxplorer` with `ModuleNotFoundError: No module named 'tensorstore'`.

Note `sys.executable` reports the venv path while `sys.base_prefix` is the system
Python312. Launching produces **two OS processes** — the venv redirector and the
base interpreter that owns the windows. That is normal, not a re-exec by the app.
Kill both when stopping.

## Launch

```powershell
$v = Join-Path $env:LOCALAPPDATA 'squidxplorer\venv\Scripts\python.exe'
$p = Start-Process -FilePath $v -ArgumentList '-m','squidxplorer._viewer' `
     -WorkingDirectory 'C:\Users\<user>\Desktop\SquidXplorer' `
     -RedirectStandardOutput out.txt -RedirectStandardError err.txt -PassThru
```

Optionally pass an acquisition folder as a trailing argument; with no argument the
app opens empty and you use **File → Open acquisition folder**. A small console
window appearing alongside is expected and documented in the README.

**QUOTE THE DATASET PATH.** `Start-Process -ArgumentList` does NOT quote array
elements, so a path with spaces arrives as several argv entries and `sys.argv[1]`
becomes just `D:\HCS_VIEWER`. The app then opens EMPTY with no error — the plate
title stays "well plate" instead of the acquisition name, which reads exactly like
a load failure. Pass it pre-quoted:

```powershell
$ds = '"D:\HCS_VIEWER TEST DATA\WELLPLATE 2026-07-23_15-03-59.976699"'
```

Confirm it took by checking the plate title, or `Get-CimInstance Win32_Process`
and reading the actual `CommandLine`.

Startup takes several seconds (napari import). `Get-Process -Id <pid>` reporting
`MainWindowTitle=''` and `MainWindowHandle=0` during that window is normal —
enumerate the child's windows instead of trusting `MainWindowTitle`.

Expected windows once up:

| Title | Size | What it is |
|---|---|---|
| `SquidXplorer` | 311x461 | compact portrait root: menus + Window navigator |
| `Log` | 393x156 | log panel |
| `[N] <well>` | varies | one per open view, e.g. `[1] A1` |
| `python` | 129x59 | the console the README mentions |

## Required dependency that is NOT declared

`napari` is imported by `_napari_pane.py`, `_napari3d.py`, and `_napari_view.py`
but is **absent from `pyproject.toml`** (the `[gui]` extra is only `PyQt5` +
`ndviewer_light`) and from CI's install lines. Without it every view window renders
a red panel reading *"napari viewer unavailable (NapariBindingError…)"* while the
rest of the shell works — an easy failure to misread as a Windows bug.

```powershell
& $v -m pip install napari      # PyQt5 is already present, so plain napari is enough
```

Verify the API surface without launching anything:

```powershell
& $v -c "from squidxplorer._napari_view import verify_napari_bindings; verify_napari_bindings(); print('OK')"
```

`psutil` is also undeclared. Without it `_rss_mb()` falls back to the POSIX-only
`resource` module, so the memory bar reads `[footprint] peak 0 MB` forever on
Windows. Installing napari pulls psutil in as a transitive dep and fixes it.

## Driving it — capture windows WITHOUT stealing focus

This is a real workstation that someone may be using. Do not call
`SetForegroundWindow`/`BringWindowToTop` to get a screenshot; it will interrupt
whatever they are typing. Use `PrintWindow` with `PW_RENDERFULLCONTENT` (flag `2`),
which renders an occluded window straight to a bitmap:

```powershell
[DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hw, IntPtr hdc, uint flags);
# ... EnumWindows -> filter by pid -> GetWindowRect -> PrintWindow(hw, hdc, 2) -> Bitmap.Save
```

Enumerate by PID via `EnumWindows` + `GetWindowThreadProcessId`; `MainWindowHandle`
alone misses the secondary windows.

**PowerShell gotcha:** variables are case-insensitive, so a loop variable named
`$h` silently clobbers a `-H` height parameter. Name loop variables `$hwnd`/`$wnd`
in any capture script that also takes `-W`/`-H`.

**CALL `SetProcessDPIAware()` IN ANY MEASURING SCRIPT.** The monitors are 4K at
200%, and a DPI-unaware PowerShell process is virtualised: `GetWindowRect` reports
every window at HALF its true size, and `MoveWindow` arguments are doubled. A
window truly 1192x1700 reads back as 596x850, which looks like the app ignoring a
resize when nothing is wrong. Costly to misdiagnose — add:

```powershell
[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
```

Two monitors, each 3840x2160 physical = 1920x1080 logical, so the virtual desktop
is 3840x1080 LOGICAL with `VirtualScreen.Left = -1920`. Region captures use
virtual coordinates.

## Stopping it

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*squidxplorer*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## Test datasets on this machine

- `C:\Users\<user>\Downloads\3DSet_BrainSection_*` — z-stacks, exercises the 3D/ROI paths
- `D:\Cukierman Fox Chase Cancer Center\20x_*` — multi-channel 20x slides
- `C:\Users\<user>\Desktop\LaserAF_Test\*`

An acquisition folder is the one holding the `0` subfolder plus `coordinates.csv`.
A `coordinates.csv is unusable … refusing to place FOVs` warning is non-fatal: the
acquisition still opens, but multi-FOV wells render as a single tile.

## Running the test suite

CI's invocation:

```powershell
$env:QT_QPA_PLATFORM='offscreen'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& $v -m pytest -q --ignore=tests/test_performance.py
```

Two things to know:

- The venv has no `pytest` by default (`Setup-Windows.ps1` installs only `.[gui]`).
- **Run the suite in chunks.** Qt-heavy files (`test_viewer`, `test_gui_commands`,
  `test_nav_wiring`, `test_viewer_3d`, `test_viewer_region_ids`) hard-crash at
  teardown with `0xC0000005`, killing the run before pytest prints its summary. Run
  per-file and collect exit codes, or the failures you care about are invisible.
- `tilefusion` and `bgsub` (Julio's separate repos) are not in this venv, so
  `test_stitch`, `test_minerva`, `test_background`, and `test_agave` fail on import
  here. That is an environment gap, not a Windows bug — CI installs them.

`QT_QPA_PLATFORM=offscreen` also exempts the app from its single-instance GUI cap
(`_gui_cap_applies()`), so offscreen runs never contend for a slot.
