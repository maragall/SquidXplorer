# Installers — one file per platform

Three artifacts from the latest **Build Installers** Actions run, each a SINGLE file:
the squidxplorer wheel and a `uv` binary ride inside the frozen setup program
(PyInstaller `--add-data`; bootstrap reads them back from `sys._MEIPASS/payload`).
Click it and it installs the whole thing: a private env
(`%LOCALAPPDATA%\SquidXplorer\env` on Windows, `~/.local/share/squidxplorer/env`
elsewhere) with extras `gui` + `stitch`, plus `decon` when the CUDA-12 probe passes,
then a double-clickable launcher pointing at the viewer.

Rerun the same file later with `--extras` to add operators (Fiji model; the env is
upgraded in place). Flags override every default; `--dry-run` prints the exact uv
commands. Installing needs network: dependencies come from PyPI and the two pinned
git SHAs.

## Windows — `SquidXplorer-Setup.exe`

Download, double-click. Done: a **SquidXplorer** desktop shortcut points at the viewer.

## macOS (Apple Silicon) — `SquidXplorer-Setup.zip` → one executable

Double-click the zip (Archive Utility restores the executable bit), then run
`SquidXplorer-Setup`. **First launch only**: the binary is not notarized, so macOS
refuses a plain double-click — right-click → Open → Open (or
`xattr -d com.apple.quarantine SquidXplorer-Setup`). On success
**~/Applications/SquidXplorer.app** launches the viewer; the app bundle is built
locally by the installer, so it carries no quarantine flag and double-clicks cleanly.

`decon` never installs here: petakit is cupy-cuda12x and no Mac passes the probe. The
installer says so rather than failing.

## Linux (x86_64) — `SquidXplorer-Setup-x86_64.AppImage`

Download, `chmod +x` once (or file manager → Properties → "Allow executing"), then
double-click or `./SquidXplorer-Setup-x86_64.AppImage`. On success **SquidXplorer**
appears in the application menu. Needs glibc ≥ 2.35 (ubuntu-22.04 build machine); on a
system without FUSE, run it with `--appimage-extract-and-run`.

## The launcher, the icon, and drag-and-open

A finished install leaves a double-clickable **SquidXplorer** launcher carrying the wellplate
icon (`make-icon.py` draws it once; the `.ico`, the `.png` and the `.app`'s `.icns` are three
encodings of the same art). Opening an acquisition by drag:

- **Windows**: drop the acquisition folder onto the desktop shortcut — a `.lnk` passes a dropped
  path as the target's argument, and the viewer opens `sys.argv[1]`.
- **Linux**: the menu entry declares `%f`, so "Open with SquidXplorer" (or dropping the folder
  onto the entry) opens it directly.
- **macOS**: dropping onto the `.app` is NOT wired (a shell-script bundle receives documents as
  Apple events, which nothing here handles). Open the app, then drop the folder onto the plate.

## Building by hand

Built by `.github/workflows/build-installer.yml` on all three platforms, which also
performs a real core install into a scratch env and imports the result. By hand, on
the target platform:

    uv build --wheel --out-dir dist
    mkdir payload && cp dist/squidxplorer-*.whl payload/ && cp "$(command -v uv)" payload/
    uvx pyinstaller --onefile --console --name SquidXplorer-Setup \
      --add-data "payload:payload" scripts/installer/bootstrap.py   # ';' separator on Windows

A wheel placed beside the program and uv on PATH still work as fallbacks when there is
no embedded payload (running `bootstrap.py` unfrozen, for instance).
