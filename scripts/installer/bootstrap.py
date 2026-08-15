"""Create or update SquidXplorer's private environment with uv, from chosen extras.

Idempotent on purpose (the Fiji model): re-running with more extras upgrades the one
env in place, and the app restarts with the operator there. Stdlib only; uv is located
on PATH and never downloaded here.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

UV_HINT = ("uv not found on PATH. Install it first "
           "(https://docs.astral.sh/uv/getting-started/installation/): "
           "macOS/Linux `curl -LsSf https://astral.sh/uv/install.sh | sh`, "
           "Windows `winget install astral-sh.uv`.")


def env_python(env_dir: Path) -> Path:
    """The env's interpreter, per platform layout."""
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", default="",
                        help="comma-separated extras from the menu, e.g. stitch,decon")
    parser.add_argument("--source", required=True,
                        help="the squidxplorer wheel, sdist or project directory to install")
    parser.add_argument("--env", required=True, type=Path, help="the private env directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact commands, run nothing")
    args = parser.parse_args(argv)

    uv = shutil.which("uv")
    if uv is None:
        print(UV_HINT, file=sys.stderr)
        return 2
    extras = [e for e in args.extras.split(",") if e]
    for cmd in commands(uv, extras, args.source, args.env):
        print(shlex.join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
