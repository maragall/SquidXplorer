# Windows installer

The user story: download the `squidxplorer-windows-installer` artifact from the latest
Actions run, extract it, and **double-click `SquidXplorer-Setup.exe`**. With no arguments
every flag defaults itself — the wheel beside the exe, an env under
`%LOCALAPPDATA%\SquidXplorer\env`, extras `stitch` plus `decon` when the CUDA-12 probe
passes — and on success a **SquidXplorer** desktop shortcut points at the viewer. Rerun
the same exe later with `--extras` to add operators (Fiji model; the env is upgraded in
place). Flags override every default; `--dry-run` prints the exact uv commands.

Installing needs network: dependencies come from PyPI and the two pinned git SHAs.

Built by `.github/workflows/build-installer.yml`. On a Windows machine by hand:
`uvx pyinstaller --onefile --console --name SquidXplorer-Setup scripts\installer\bootstrap.py`,
shipped beside the squidxplorer wheel and a `uv.exe`, because a frozen exe cannot
pip-install into itself — that is why the env is private and uv-managed.
