"""Create or update SquidXplorer's private environment with uv, from chosen extras.

Idempotent on purpose (the Fiji model): re-running with more extras upgrades the one
env in place, and the app restarts with the operator there. Stdlib only; uv is never
downloaded here — the frozen one-file binary CARRIES the squidxplorer wheel and a uv
binary as its payload (PyInstaller --add-data, surfacing under sys._MEIPASS/payload),
with a wheel beside the program and uv on PATH as the fallbacks.

Double-clicked with no arguments, every flag has a default: the payload wheel, an env
under the user's local app data, and the default extras (gui, stitch, decon everywhere,
plus decon's CUDA payload when the GPU probe sees a CUDA-12 driver). Flags override
everything, so the scripted path is unchanged.

A finished install leaves a double-clickable launcher, per platform: a desktop shortcut
on Windows, a ~/Applications app bundle on macOS, an XDG menu entry on Linux.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

UV_HINT = ("uv not found on PATH or beside this program. Install it first "
           "(https://docs.astral.sh/uv/getting-started/installation/): "
           "macOS/Linux `curl -LsSf https://astral.sh/uv/install.sh | sh`, "
           "Windows `winget install astral-sh.uv`.")

# gui is not an operator extra but a desktop install is nothing without it. decon installs
# on every machine (petakit's CPU path, torch MPS on Apple Silicon by pyproject marker).
DEFAULT_EXTRAS = ("gui", "stitch", "decon")

# decon's CUDA payload (cupy-cuda12x, petakit's own GPU path), added ONLY when the probe sees
# an NVIDIA driver speaking CUDA 12: that wheel exists for no other machine.
CUDA_EXTRA = "decon-cuda"


def cuda12_available() -> tuple[bool, str]:
    """``(ok, reason_if_not)``: can this machine run a cupy-cuda12x payload right now?"""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False, "nvidia-smi not on PATH, no NVIDIA driver visible"
    try:
        out = subprocess.run([smi], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"nvidia-smi failed: {exc}"
    version = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if version is None:
        return False, "nvidia-smi reported no CUDA version, driver too old for CUDA 12"
    if int(version.group(1)) < 12:
        return False, (f"driver speaks CUDA {version.group(1)}.{version.group(2)}, "
                       "cupy-cuda12x needs 12")
    return True, ""


def gpu_backend(system: str = platform.system(), machine: str = platform.machine(),
                cuda: Callable[[], tuple[bool, str]] = cuda12_available) -> tuple[str, str]:
    """``(kind, note)``: which decon backend this machine will run, as the menu row's note.

    A three-way FACT, never a shade: ``"cuda"`` (an NVIDIA driver speaking CUDA 12, the
    cupy payload is added), ``"mps"`` (Apple Silicon, torch's Metal backend rides the decon
    extra by marker), or ``"cpu"`` (everything else, Intel Macs included, and the note says
    why CUDA was not seen).
    """
    ok, reason = cuda()
    if ok:
        return "cuda", "GPU: CUDA (petakit)"
    if system == "Darwin" and machine == "arm64":
        return "mps", "GPU: Apple (torch MPS)"
    return "cpu", f"CPU only: {reason}"


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


def default_extras(probe=gpu_backend) -> tuple[list[str], str]:
    """The default-checked extras plus decon's CUDA payload on a CUDA machine, and the
    probe's note naming the backend decon will run on."""
    kind, note = probe()
    extras = list(DEFAULT_EXTRAS) + ([CUDA_EXTRA] if kind == "cuda" else [])
    return extras, f"decon: {note}"


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


#: The interpreter the private env is built with. PASSED TO ``uv venv`` EXPLICITLY, because uv
#: otherwise builds from whatever python the MACHINE happens to have — measured failure,
#: 2026-08-16, first customer install on the Katana rig: Ubuntu 22.04's system python is 3.10.12,
#: the wheel requires >=3.11, and dependency resolution died on the very first run. With the pin,
#: uv DOWNLOADS a self-contained managed CPython when the machine has none (or the wrong one), so
#: the installer works on a machine with no Python at all — which is the one-file promise.
#:
#: 3.11 EXACTLY, and the number is chosen by the WHEELS, not by newness: it is the newest Python
#: on which every pack's whole dependency closure is wheels-only. Measured failure, 2026-08-18,
#: second customer install (Windows CUDA rig): under 3.12 the decon pack pulled psfmodels 0.3.3,
#: whose wheels top out at cp311 (checked on PyPI: cp311 win+linux exist, no cp312 anywhere), so
#: uv built its C extension from source and died on "Microsoft Visual C++ 14.0 or greater is
#: required" — on a machine that must need NO compiler. CI's runners have MSVC/gcc, which is why
#: the install-only decon test stayed green; the wheels-only guard in the workflow now fails on
#: any silently-compiled sdist. Raise this only when psfmodels (or its successor in petakit's
#: dependencies) ships wheels for the newer Python.
ENV_PYTHON = "3.11"


def stale_env(env_dir: Path) -> Optional[str]:
    """A sentence naming why the existing env cannot be reused, or None when it can (or is absent).

    The venv step is skipped whenever the env exists, so without this check an env built wrong
    once is broken FOREVER: the Katana rig's 3.10 env would have failed every future install of
    a fixed installer identically. Asking the env's own interpreter is the one honest probe —
    the directory name says nothing about what built it.

    EXACT-MINOR match with :data:`ENV_PYTHON`, not a floor. A 3.12 env (built by the one-day
    window when that was the pin) satisfies squidxplorer's own requires-python, but adding the
    decon pack to it later would source-build psfmodels all over again — the env's interpreter
    decides which wheels exist, so one installer version means one interpreter, recreated on
    mismatch rather than left as a per-machine wheel lottery.
    """
    py = env_python(env_dir)
    if not py.exists():
        return None
    try:
        out = subprocess.run(
            [str(py), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=60, check=True).stdout.strip()
        int(out.split(".")[0]), int(out.split(".")[1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return f"its interpreter would not answer ({type(exc).__name__})"
    if out != ENV_PYTHON:
        return f"its Python is {out} and this installer builds envs with {ENV_PYTHON}"
    return None


def commands(uv: str, extras: Sequence[str], source: str, env_dir: Path) -> list[list[str]]:
    """The exact uv invocations a run would make; the venv step only when the env is missing."""
    cmds = []
    if not env_python(env_dir).exists():
        cmds.append([uv, "venv", "--python", ENV_PYTHON, str(env_dir)])
    # --reinstall-package: a rerun with a newer installer must update the app even at the same
    # version string, so updating = run the installer again, never pip surgery in the env.
    cmds.append([uv, "pip", "install", "--python", str(env_python(env_dir)),
                 "--reinstall-package", "squidxplorer", install_spec(source, extras)])
    return cmds


def _installed_icon(env_dir: Path, name: str) -> Optional[Path]:
    """Copy the payload's icon file *name* beside the env, returning its lasting path.

    COPIED, not referenced in place: a one-file build extracts its payload to a temp dir that
    vanishes when the installer exits, so a shortcut pointing there would lose its art on the
    next reboot. Beside the env is the one place that lives exactly as long as the install.
    """
    for candidate in payload_dirs():
        src = candidate / name
        if src.exists():
            try:
                dest = env_dir.parent / name
                shutil.copyfile(src, dest)
                return dest
            except OSError:
                return None
    return None


def _windows_shortcut(env_dir: Path) -> Optional[str]:
    """A desktop shortcut to the viewer; a failure is said, never fatal.

    DRAG-AND-OPEN COMES FREE: dropping a folder onto a .lnk passes its path as an argument to
    the target, and ``squidxplorer-view`` opens ``sys.argv[1]``. The icon is the wellplate art
    (scripts/installer/make-icon.py), installed beside the env so it outlives the one-file
    extraction dir.
    """
    target = env_dir / "Scripts" / "squidxplorer-view.exe"
    icon = _installed_icon(env_dir, "squidxplorer.ico")
    icon_line = f"$s.IconLocation = '{icon},0'; " if icon is not None else ""
    script = ("$ws = New-Object -ComObject WScript.Shell; "
              "$s = $ws.CreateShortcut(\"$([Environment]::GetFolderPath('Desktop'))"
              "\\SquidXplorer.lnk\"); "
              f"$s.TargetPath = '{target}'; {icon_line}$s.Save()")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       check=True, capture_output=True, timeout=30)
        return "desktop shortcut created: SquidXplorer (drop an acquisition folder onto it to open)"
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
  <key>CFBundleIconFile</key><string>SquidXplorer</string>
  <!-- DRAG-ONTO-THE-APP. Declaring public.folder is what makes Finder accept a dropped
       acquisition folder at all; LaunchServices then delivers it as an odoc Apple event, Qt's
       Cocoa plugin turns that into a QFileOpenEvent, and squidxplorer._viewer._FileOpenFilter
       routes it into PlateWindow.ingest — the same entry as a drop on the plate. Role Viewer +
       rank Alternate: this app never becomes the system's default handler for folders. -->
  <key>CFBundleDocumentTypes</key><array><dict>
    <key>CFBundleTypeName</key><string>Squid acquisition folder</string>
    <key>CFBundleTypeRole</key><string>Viewer</string>
    <key>LSItemContentTypes</key><array><string>public.folder</string></array>
    <key>LSHandlerRank</key><string>Alternate</string>
  </dict></array>
</dict></plist>
"""


def _macos_icns(env_dir: Path, resources_dir: Path) -> bool:
    """Write Resources/SquidXplorer.icns from the payload's png, through the ENV's Pillow.

    The frozen bootstrapper carries no imaging library, but the env it just installed carries
    Pillow (napari depends on it) — so the conversion is delegated to the interpreter that has
    it. A miss is cosmetic: the bundle works iconless.
    """
    png = None
    for candidate in payload_dirs():
        if (candidate / "squidxplorer.png").exists():
            png = candidate / "squidxplorer.png"
            break
    if png is None:
        return False
    code = ("import sys; from PIL import Image; "
            "Image.open(sys.argv[1]).save(sys.argv[2], format='ICNS')")
    try:
        subprocess.run([str(env_python(env_dir)), "-c", code, str(png),
                        str(resources_dir / "SquidXplorer.icns")],
                       check=True, capture_output=True, timeout=60)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


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
        resources = app / "Contents" / "Resources"
        resources.mkdir(parents=True, exist_ok=True)
        (app / "Contents" / "Info.plist").write_text(_INFO_PLIST)
        _macos_icns(env_dir, resources)   # cosmetic; the bundle works iconless
        runner = macos_dir / "SquidXplorer"
        runner.write_text(f'#!/bin/sh\nexec "{target}" "$@"\n')
        runner.chmod(0o755)
        return f"app created: {app}"
    except OSError as exc:
        print(f"no app bundle ({exc}); launch {target} directly", file=sys.stderr)
        return None


def _linux_desktop_entry(env_dir: Path, apps_dir: Optional[Path] = None) -> Optional[str]:
    """An XDG menu entry pointing at the env's viewer; a failure is said, never fatal.

    ``%f`` is the drag-and-open half: a folder dropped on (or opened with) the entry arrives as
    ``sys.argv[1]``. The icon is the payload's png, installed beside the env so it outlives the
    one-file extraction dir.
    """
    target = env_dir / "bin" / "squidxplorer-view"
    entry_dir = apps_dir or Path.home() / ".local" / "share" / "applications"
    exec_line = f'"{target}"' if " " in str(target) else str(target)
    icon = _installed_icon(env_dir, "squidxplorer.png")
    icon_line = f"Icon={icon}\n" if icon is not None else ""
    text = ("[Desktop Entry]\nType=Application\nName=SquidXplorer\n"
            "Comment=Post-acquisition HCS plate viewer\n"
            f"Exec={exec_line} %f\n{icon_line}Terminal=false\nCategories=Science;Graphics;\n")
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


def _linux_gui_libs_note(platform: str = sys.platform) -> Optional[str]:
    """Name the system libraries Qt will want at FIRST LAUNCH on Linux, when they are missing.

    The install itself needs none of them — PyQt6's wheel carries Qt — but Qt's xcb platform
    plugin dlopens a handful of system libraries at startup, and ``libxcb-cursor0`` in
    particular is absent from a default Ubuntu 22.04 (Qt >= 6.5 grew the requirement). Without
    this note the failure is a successful install followed by a launch that dies with an
    inscrutable "could not load the Qt xcb platform plugin". Installing them needs sudo, which
    this installer must not assume — so it SAYS the fix instead of attempting it.
    """
    if not platform.startswith("linux"):
        return None
    import ctypes

    try:
        ctypes.CDLL("libxcb-cursor.so.0")
        return None
    except OSError:
        return ("note: the viewer needs a system library that this machine is missing "
                "(Qt's xcb plugin). Before the first launch, run:\n"
                "    sudo apt install libxcb-cursor0")


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
                             f"(default: gui,stitch,decon, plus {CUDA_EXTRA} when the "
                             "machine has a CUDA-12 driver)")
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
    if not args.dry_run:
        reason = stale_env(env_dir)
        if reason:
            # The env is THIS APP'S OWN private directory, so recreating it destroys nothing of
            # the user's — and reusing it would fail every install forever (see stale_env).
            print(f"recreating {env_dir}: {reason}")
            try:
                shutil.rmtree(env_dir)
            except OSError as exc:
                print(f"could not remove the stale env ({exc}); close anything using it and "
                      f"rerun, or delete {env_dir} by hand.", file=sys.stderr)
                return 2
    for cmd in commands(uv, extras, str(source), env_dir):
        print(shlex.join(cmd))
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                # A SENTENCE, not a traceback. The first customer failure ended in an unhandled
                # CalledProcessError and PyInstaller's "Failed to execute script" banner — the
                # real message (uv's, printed just above) was the least visible thing on screen.
                print(f"\ninstall failed (exit {exc.returncode}) — the message above this line "
                      f"is uv's own account of why.", file=sys.stderr)
                return exc.returncode or 1
    if not args.dry_run:
        made = create_launcher(env_dir)
        if made:
            print(made)
        note = _linux_gui_libs_note()
        if note:
            print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
