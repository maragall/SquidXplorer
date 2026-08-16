"""Create or update SquidXplorer's private environment with uv, from chosen extras.

Idempotent on purpose (the Fiji model): re-running with more extras upgrades the one
env in place, and the app restarts with the operator there. Stdlib only; uv is never
downloaded here — the frozen one-file binary CARRIES the squidxplorer wheel and a uv
binary as its payload (PyInstaller --add-data, surfacing under sys._MEIPASS/payload),
with a wheel beside the program and uv on PATH as the fallbacks.

Double-clicked with no arguments, every flag has a default: the payload wheel, an env
under the user's local app data, and the default extras with decon gated by the
CUDA-12 probe. Flags override everything, so the scripted path is unchanged.

A finished install leaves a double-clickable launcher, per platform: a desktop shortcut
on Windows, a ~/Applications app bundle on macOS, an XDG menu entry on Linux.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

UV_HINT = ("uv not found on PATH or beside this program. Install it first "
           "(https://docs.astral.sh/uv/getting-started/installation/): "
           "macOS/Linux `curl -LsSf https://astral.sh/uv/install.sh | sh`, "
           "Windows `winget install astral-sh.uv`.")

# gui is not an operator extra but a desktop install is nothing without it;
# decon is subject to the CUDA-12 probe.
DEFAULT_EXTRAS = ("gui", "stitch", "decon")


def cuda12_available() -> tuple[bool, str]:
    """``(ok, reason_if_not)`` — can this machine run a cupy-cuda12x payload right now?"""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False, "nvidia-smi not on PATH: no NVIDIA driver visible (petakit needs CUDA 12)"
    try:
        out = subprocess.run([smi], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"nvidia-smi failed: {exc}"
    version = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if version is None:
        return False, "nvidia-smi reported no CUDA version: driver too old for CUDA 12"
    if int(version.group(1)) < 12:
        return False, (f"driver speaks CUDA {version.group(1)}.{version.group(2)}, "
                       "petakit needs 12")
    return True, ""


def program_dir() -> Path:
    """Where this program lives: the frozen exe's folder, else this file's."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def payload_dirs() -> list[Path]:
    """Where the wheel and uv may live: the one-file payload first, then beside the program."""
    bundled = getattr(sys, "_MEIPASS", None)
    dirs = [Path(bundled) / "payload"] if bundled else []
    return dirs + [program_dir()]


def default_source(beside: Optional[Path] = None) -> Optional[Path]:
    """The newest squidxplorer wheel from the payload or beside the program, or None."""
    for candidate in [beside] if beside is not None else payload_dirs():
        wheels = sorted(candidate.glob("squidxplorer-*.whl"))
        if wheels:
            return wheels[-1]
    return None


def default_env() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "SquidXplorer" / "env"
    return Path.home() / ".local" / "share" / "squidxplorer" / "env"


def default_extras(probe=cuda12_available) -> tuple[list[str], str]:
    """The default-checked extras, with decon's shading reason when the probe fails."""
    ok, reason = probe()
    extras = [e for e in DEFAULT_EXTRAS if e != "decon" or ok]
    return extras, ("" if ok else f"decon left out — {reason}")


def env_python(env_dir: Path) -> Path:
    """The env's interpreter, per platform layout."""
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def find_uv() -> Optional[str]:
    """The uv shipped in the payload or beside this program, else uv on PATH."""
    name = "uv.exe" if sys.platform == "win32" else "uv"
    for candidate in payload_dirs():
        shipped = candidate / name
        if shipped.exists():
            if os.name == "posix" and not os.access(shipped, os.X_OK):
                shipped.chmod(0o755)    # onefile extraction may drop the mode
            return str(shipped)
    return shutil.which("uv")


def install_spec(source: str, extras: Sequence[str]) -> str:
    """The PEP 508 requirement for *source* with *extras* (``core`` is not an extra)."""
    chosen = sorted(set(extras) - {"core"})
    name = f"squidxplorer[{','.join(chosen)}]" if chosen else "squidxplorer"
    path = Path(source)
    return f"{name} @ {path.resolve().as_uri() if path.exists() else source}"


def commands(uv: str, extras: Sequence[str], source: str, env_dir: Path) -> list[list[str]]:
    """The exact uv invocations a run would make; the venv step only when the env is missing."""
    cmds = []
    if not env_python(env_dir).exists():
        cmds.append([uv, "venv", str(env_dir)])
    cmds.append([uv, "pip", "install", "--python", str(env_python(env_dir)),
                 install_spec(source, extras)])
    return cmds


def _windows_shortcut(env_dir: Path) -> Optional[str]:
    """A desktop shortcut to the viewer; a failure is said, never fatal."""
    target = env_dir / "Scripts" / "squidxplorer-view.exe"
    script = ("$ws = New-Object -ComObject WScript.Shell; "
              "$s = $ws.CreateShortcut(\"$([Environment]::GetFolderPath('Desktop'))"
              "\\SquidXplorer.lnk\"); "
              f"$s.TargetPath = '{target}'; $s.Save()")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       check=True, capture_output=True, timeout=30)
        return "desktop shortcut created: SquidXplorer"
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"no desktop shortcut ({exc}); launch {target} directly", file=sys.stderr)
        return None


_INFO_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>SquidXplorer</string>
  <key>CFBundleIdentifier</key><string>com.cephla.squidxplorer</string>
  <key>CFBundleExecutable</key><string>SquidXplorer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
</dict></plist>
"""


def _macos_app(env_dir: Path, apps_dir: Optional[Path] = None) -> Optional[str]:
    """A minimal app bundle in ~/Applications wrapping the env's viewer.

    Built locally on this machine, so it carries no Gatekeeper quarantine flag (only the
    downloaded setup binary does) and a plain double-click launches it. The executable is
    one exec line, so a rerun that upgrades the env leaves the bundle correct.
    """
    target = env_dir / "bin" / "squidxplorer-view"
    app = (apps_dir or Path.home() / "Applications") / "SquidXplorer.app"
    try:
        macos_dir = app / "Contents" / "MacOS"
        macos_dir.mkdir(parents=True, exist_ok=True)
        (app / "Contents" / "Info.plist").write_text(_INFO_PLIST)
        runner = macos_dir / "SquidXplorer"
        runner.write_text(f'#!/bin/sh\nexec "{target}" "$@"\n')
        runner.chmod(0o755)
        return f"app created: {app}"
    except OSError as exc:
        print(f"no app bundle ({exc}); launch {target} directly", file=sys.stderr)
        return None


def _linux_desktop_entry(env_dir: Path, apps_dir: Optional[Path] = None) -> Optional[str]:
    """An XDG menu entry pointing at the env's viewer; a failure is said, never fatal."""
    target = env_dir / "bin" / "squidxplorer-view"
    entry_dir = apps_dir or Path.home() / ".local" / "share" / "applications"
    exec_line = f'"{target}"' if " " in str(target) else str(target)
    text = ("[Desktop Entry]\nType=Application\nName=SquidXplorer\n"
            "Comment=Post-acquisition HCS plate viewer\n"
            f"Exec={exec_line}\nTerminal=false\nCategories=Science;Graphics;\n")
    try:
        entry_dir.mkdir(parents=True, exist_ok=True)
        entry = entry_dir / "squidxplorer.desktop"
        entry.write_text(text)
        entry.chmod(0o755)
        return f"menu entry created: {entry}"
    except OSError as exc:
        print(f"no menu entry ({exc}); launch {target} directly", file=sys.stderr)
        return None


def create_launcher(env_dir: Path, platform: str = sys.platform) -> Optional[str]:
    """A double-clickable launcher for the installed viewer, per platform.

    Returns the sentence describing what was made, or None after a said failure —
    the install itself has already succeeded either way.
    """
    if platform == "win32":
        return _windows_shortcut(env_dir)
    if platform == "darwin":
        return _macos_app(env_dir)
    return _linux_desktop_entry(env_dir)


def _interactive(argv: Optional[Sequence[str]]) -> bool:
    """Double-clicked: no arguments and a real console — hold the window open at the end.

    The tty check keeps a piped or CI run from blocking on Enter (input() would EOFError
    there anyway, turning a green install into a crash at the last line).
    """
    return (argv is None and len(sys.argv) == 1
            and sys.stdin is not None and sys.stdin.isatty())


def main(argv: Optional[Sequence[str]] = None) -> int:
    interactive = _interactive(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", default=None,
                        help="comma-separated extras, e.g. stitch,decon; '' for core only "
                             "(default: stitch, plus decon when the machine has CUDA 12)")
    parser.add_argument("--source", default=None,
                        help="wheel, sdist or project directory to install "
                             "(default: the squidxplorer wheel beside this program)")
    parser.add_argument("--env", default=None, type=Path,
                        help=f"the private env directory (default: {default_env()})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact commands, run nothing")
    args = parser.parse_args(argv)
    try:
        return _run(args)
    finally:
        if interactive:
            input("Press Enter to close.")


def _run(args: argparse.Namespace) -> int:
    source = args.source or default_source()
    if source is None:
        print("no squidxplorer wheel in the payload or beside this program; pass --source",
              file=sys.stderr)
        return 2
    env_dir = args.env or default_env()
    if args.extras is None:
        extras, note = default_extras()
        if note:
            print(note)
    else:
        extras = [e for e in args.extras.split(",") if e]
    print(f"installing squidxplorer[{','.join(extras) or 'core'}] -> {env_dir}")

    uv = find_uv()
    if uv is None:
        print(UV_HINT, file=sys.stderr)
        return 2
    for cmd in commands(uv, extras, str(source), env_dir):
        print(shlex.join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    if not args.dry_run:
        made = create_launcher(env_dir)
        if made:
            print(made)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
