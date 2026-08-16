# Installers — Windows, macOS, Linux

One artifact per platform from the latest **Build Installers** Actions run, each carrying
the setup program, the squidxplorer wheel it installs, and `uv` itself (a frozen setup
binary cannot pip-install into itself — that is why the env is private and uv-managed).

With no arguments every flag defaults itself — the wheel beside the program, a private
env (`%LOCALAPPDATA%\SquidXplorer\env` on Windows, `~/.local/share/squidxplorer/env`
elsewhere), extras `gui` plus `stitch`, plus `decon` when the CUDA-12 probe passes — and
on success a double-clickable launcher points at the viewer. Rerun the same program later
with `--extras` to add operators (Fiji model; the env is upgraded in place). Flags
override every default; `--dry-run` prints the exact uv commands.

Installing needs network: dependencies come from PyPI and the two pinned git SHAs.

## Windows — `squidxplorer-windows-installer`

Extract the artifact and **double-click `SquidXplorer-Setup.exe`**. On success a
**SquidXplorer** desktop shortcut points at the viewer.

## macOS (Apple Silicon) — `squidxplorer-macos-arm64-installer`

Extract the artifact, then extract `….tar.gz` inside it (double-click; the tar keeps the
executable bits) and run `SquidXplorer-Setup`. **First launch only**: the binary is not
notarized, so macOS refuses a plain double-click — right-click → Open → Open (or
`xattr -d com.apple.quarantine SquidXplorer-Setup`). On success
**~/Applications/SquidXplorer.app** launches the viewer; the app bundle is built locally
by the installer, so it carries no quarantine flag and double-clicks cleanly.

`decon` never installs here: petakit is cupy-cuda12x and no Mac passes the probe. The
installer says so rather than failing.

## Linux (x86_64) — `squidxplorer-linux-x86_64-installer`

Extract the artifact, then `tar xzf ….tar.gz` and run `./SquidXplorer-Setup` (the tar
preserves the executable bit; built on ubuntu-22.04, so it needs glibc ≥ 2.35). On
success **SquidXplorer** appears in the application menu via
`~/.local/share/applications/squidxplorer.desktop`.

## Building by hand

Built by `.github/workflows/build-installer.yml` on all three platforms, which also
performs a real core install into a scratch env and imports the result. By hand, on the
target platform:

    uvx pyinstaller --onefile --console --name SquidXplorer-Setup scripts/installer/bootstrap.py

then place the squidxplorer wheel (and, for a machine without uv, a `uv` binary) beside it.
