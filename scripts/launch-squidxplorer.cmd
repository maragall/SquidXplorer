@echo off
rem ---------------------------------------------------------------------------
rem  SquidXplorer launcher -- what the Desktop shortcut actually runs.
rem
rem  Setup-Windows.ps1 COPIES this file to %LOCALAPPDATA%\squidmip and stamps the
rem  checkout's path into the REPO= line below. The installed copy deliberately lives
rem  OUTSIDE the checkout: a branch switch, `git clean`, or an upstream sync can
rem  then never delete or revert the thing the Desktop icon points at. The icon
rem  keeps working across any code change because the venv has the app installed
rem  EDITABLE against the checkout -- edits apply on the next launch, no reinstall.
rem
rem  Unstamped (i.e. run straight out of scripts\ in the repo) it falls back to
rem  its own parent directory, so the one file works in both places.
rem ---------------------------------------------------------------------------

setlocal

set "VENV_PY=%LOCALAPPDATA%\squidmip\venv\Scripts\python.exe"
set "REPO=__REPO__"
if not exist "%REPO%\squidmip\_viewer.py" set "REPO=%~dp0.."

if not exist "%VENV_PY%" (
    echo.
    echo   Cannot find the SquidXplorer virtual environment:
    echo     %VENV_PY%
    echo.
    echo   Create it by running, from the repo root:
    echo     powershell -ExecutionPolicy Bypass -File scripts\Setup-Windows.ps1
    echo.
    pause
    exit /b 9009
)

if not exist "%REPO%\squidmip\_viewer.py" (
    echo.
    echo   Cannot find the SquidXplorer checkout:
    echo     %REPO%
    echo.
    echo   If you moved the repo, re-run scripts\Setup-Windows.ps1 from its new
    echo   location to re-stamp this launcher.
    echo.
    pause
    exit /b 9009
)

cd /d "%REPO%"

rem %* and not %1: it keeps the ORIGINAL quoting of a dropped folder. A path with
rem spaces that loses its quotes arrives as several argv entries, and the app then
rem opens EMPTY with no error at all -- the plate title just stays "well plate",
rem which reads exactly like a load failure. Drag-and-drop onto the icon is the
rem main way this argument ever gets passed, so the quoting is load-bearing.
"%VENV_PY%" -m squidmip._viewer %*
set "RC=%ERRORLEVEL%"

rem Hold the window open ONLY on a failure. A clean exit closes silently; a crash
rem keeps its traceback on screen instead of flashing past. python.exe and not
rem pythonw.exe for the same reason -- the console is where the log and the
rem [footprint] lines go.
if not "%RC%"=="0" (
    echo.
    echo   SquidXplorer exited with code %RC%.
    echo.
    pause
)
exit /b %RC%
