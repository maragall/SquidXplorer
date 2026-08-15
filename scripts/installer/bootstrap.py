"""Create or update SquidXplorer's private environment with uv, from chosen extras.

Idempotent on purpose (the Fiji model): re-running with more extras upgrades the one
env in place, and the app restarts with the operator there. Stdlib only; uv is located
on PATH or beside this program, never downloaded here.

Double-clicked with no arguments, every flag has a default: the wheel beside the exe,
an env under the user's local app data, and the default extras with decon gated by the
CUDA-12 probe. Flags override everything, so the scripted path is unchanged.
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


def default_source(beside: Optional[Path] = None) -> Optional[Path]:
    """The one squidxplorer wheel shipped beside the program, or None."""
    wheels = sorted((beside or program_dir()).glob("squidxplorer-*.whl"))
    return wheels[-1] if wheels else None


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
    """uv on PATH, else the uv shipped beside this program (the artifact carries one)."""
    on_path = shutil.which("uv")
    if on_path:
        return on_path
    shipped = program_dir() / ("uv.exe" if sys.platform == "win32" else "uv")
    return str(shipped) if shipped.exists() else None


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


def create_shortcut(env_dir: Path) -> bool:
    """A desktop shortcut to the viewer, Windows only; a failure is said, never fatal."""
    if sys.platform != "win32":
        return False
    target = env_dir / "Scripts" / "squidxplorer-view.exe"
    script = ("$ws = New-Object -ComObject WScript.Shell; "
              "$s = $ws.CreateShortcut(\"$([Environment]::GetFolderPath('Desktop'))"
              "\\SquidXplorer.lnk\"); "
              f"$s.TargetPath = '{target}'; $s.Save()")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       check=True, capture_output=True, timeout=30)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"no desktop shortcut ({exc}); launch {target} directly", file=sys.stderr)
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    interactive = argv is None and len(sys.argv) == 1
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
        print("no squidxplorer wheel beside this program; pass --source", file=sys.stderr)
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
    if not args.dry_run and create_shortcut(env_dir):
        print("desktop shortcut created: SquidXplorer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
