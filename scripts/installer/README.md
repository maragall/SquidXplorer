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
