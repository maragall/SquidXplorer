"""Odon bridge: samplesheet over the existing plate + arm's-length launch.

Odon is a native desktop GUI viewer with no plate model; the samplesheet CSV flattens
the plate to a row per field. Nothing here writes pixels.
"""

from __future__ import annotations

import csv
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from squidxplorer._output import _row_sort_key, parse_well_id

from squidxplorer._logpane import get_logger

logger = get_logger("odon")

ODON_RELEASES_URL = "https://github.com/alexcoulton/odon/releases"
ODON_VERIFIED_VERSION = "0.1.5"

PLATE_DIRNAME = "plate.ome.zarr"
SAMPLESHEET_NAME = "odon_samplesheet.csv"

# Odon's contract: header required, first two columns positional (id, path);
# later columns are free-form metadata.
SAMPLESHEET_COLUMNS = ("id", "path", "well", "fov")

_CRASH_CHECK_DELAY_S = 1.0   # how long to let odon live before deciding it crashed
_CHECK_TIMEOUT_S = 120.0


def _plate_dir(hcs_dir: Path) -> Path:
    """Resolve the ``plate.ome.zarr`` inside *hcs_dir* (or *hcs_dir* itself if it IS one)."""
    hcs_dir = Path(hcs_dir).expanduser()
    plate = hcs_dir if hcs_dir.name == PLATE_DIRNAME else hcs_dir / PLATE_DIRNAME
    if not plate.is_dir():
        raise FileNotFoundError(
            f"no {PLATE_DIRNAME} under {hcs_dir} — point this at a squidxplorer output "
            "directory (the one containing plate.ome.zarr), or run squidxplorer first."
        )
    return plate


def _field_sort_key(row: str, col: str, fov: str):
    """Natural plate order: A..Z then AA.., column numerically, fov numerically."""
    return (_row_sort_key(row), int(col), int(fov))


def iter_fields(hcs_dir):
    """Yield ``(row, col, fov, field_dir)`` for every complete field group, in plate order."""
    plate = _plate_dir(hcs_dir)
    found = []
    for zj in plate.glob("*/*/*/zarr.json"):
        field_dir = zj.parent
        row, col, fov = field_dir.parts[-3:]
        try:
            parse_well_id(row + col)          # canonical <letters><digits> well id
            key = _field_sort_key(row, col, fov)
        except (ValueError, TypeError):
            # Not a plate-shaped path: skip, but say so.
            logger.warning("odon: skipping non-plate-shaped path %s", field_dir)
            continue
        found.append((key, row, col, fov, field_dir))
    found.sort(key=lambda t: t[0])
    for _, row, col, fov, field_dir in found:
        yield row, col, fov, field_dir


def write_samplesheet(hcs_dir, out_csv=None) -> Path:
    """Write Odon's samplesheet CSV over an existing squidxplorer output directory.

    Paths are written relative to the CSV, which is how Odon resolves them.
    """
    plate = _plate_dir(hcs_dir)
    out_csv = Path(out_csv) if out_csv is not None else plate.parent / SAMPLESHEET_NAME
    csv_dir = out_csv.parent

    rows = []
    for row, col, fov, field_dir in iter_fields(plate):
        well = f"{row}{col}"
        try:
            rel = field_dir.relative_to(csv_dir)
        except ValueError:
            # CSV lives outside the plate tree: fall back to an absolute path.
            rel = field_dir.resolve()
            logger.warning("odon: samplesheet is outside the plate tree; writing absolute paths")
        rows.append({"id": f"{well}_{fov}", "path": rel.as_posix(), "well": well, "fov": fov})

    if not rows:
        raise ValueError(
            f"no complete field groups under {plate} — nothing to put in a samplesheet. "
            "(A field is complete once its zarr.json exists; a run killed mid-write leaves "
            "field directories without one.) Odon's import fails on an empty sheet anyway."
        )

    csv_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SAMPLESHEET_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("odon samplesheet: %d field(s) -> %s", len(rows), out_csv)
    return out_csv


def _platform_default() -> Optional[Path]:
    """The install location for this platform's released artifact, if there is one."""
    if sys.platform == "darwin":
        # The .dmg installs an .app bundle and does not put odon on PATH.
        return Path("/Applications/odon.app/Contents/MacOS/odon")
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        return Path(program_files) / "odon" / "odon.exe"
    if sys.platform.startswith("linux"):
        return Path("/usr/bin/odon")          # from odon_<ver>_amd64.deb
    return None


def _no_build_for_this_platform() -> Optional[str]:
    """Explain why no release exists here, or None if one does."""
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in ("aarch64", "arm64"):
        return "there is no Linux arm64 build of odon (releases are macOS arm64, Windows x86_64, Linux amd64)"
    if sys.platform == "darwin" and machine == "x86_64":
        return "the macOS odon build is Apple Silicon only (no Intel build is published)"
    return None


def find_odon() -> Path:
    """Locate the odon binary: ``$ODON_BIN`` -> ``PATH`` -> this platform's install location.

    Never downloads or vendors anything: Odon is GPL-3.0-only, SquidXplorer is BSD-3-Clause.
    """
    override = os.environ.get("ODON_BIN")
    if override:
        # Fail loud rather than falling through to a different binary than the user named.
        path = Path(override).expanduser()
        if not (path.is_file() and os.access(path, os.X_OK)):
            raise FileNotFoundError(
                f"$ODON_BIN is set to {override!r} but that is not an executable file. "
                "Unset it to fall back to PATH, or point it at the odon binary "
                "(macOS: /Applications/odon.app/Contents/MacOS/odon)."
            )
        return path

    on_path = shutil.which("odon")
    if on_path:
        return Path(on_path)

    default = _platform_default()
    if default is not None and default.is_file() and os.access(default, os.X_OK):
        return default

    reason = _no_build_for_this_platform()
    detail = f" Note: {reason}." if reason else ""
    raise FileNotFoundError(
        f"odon not found. Looked at $ODON_BIN, PATH, and {default or 'no known install location'}."
        f"{detail} Install it from {ODON_RELEASES_URL} (verified against v{ODON_VERIFIED_VERSION}), "
        "or set $ODON_BIN to the binary. SquidXplorer never bundles it: odon is GPL-3.0-only."
    )


def check_odon(field_dir, odon_bin=None) -> bool:
    """Run ``odon --check`` on one field group — the only headless path Odon has."""
    binary = Path(odon_bin) if odon_bin is not None else find_odon()
    try:
        proc = subprocess.run(
            [str(binary), "--check", str(Path(field_dir))],
            capture_output=True, text=True, timeout=_CHECK_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        logger.warning("odon --check timed out after %.0fs on %s", _CHECK_TIMEOUT_S, field_dir)
        return False
    output = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0 and "OK: loaded tile" in output
    if not ok:
        logger.warning("odon --check failed on %s (rc=%s): %s",
                       field_dir, proc.returncode, output.strip()[:400] or "<no output>")
    return ok


def launch_odon(samplesheet, *, mosaic_cols: Optional[int] = None, odon_bin=None,
                crash_check_delay: float = _CRASH_CHECK_DELAY_S) -> subprocess.Popen:
    """Launch Odon on *samplesheet* as a detached process and return the handle.

    Detached, never waited on; polled once after a short delay so a crash is not silent.
    """
    binary = Path(odon_bin) if odon_bin is not None else find_odon()
    samplesheet = Path(samplesheet)
    argv = [str(binary), "--mosaic-samplesheet", str(samplesheet)]
    if mosaic_cols is not None:
        argv += ["--mosaic-cols", str(int(mosaic_cols))]

    kwargs = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True          # survive the parent shell exiting
    logger.info("launching odon: %s", " ".join(argv))
    proc = subprocess.Popen(argv, **kwargs)

    if crash_check_delay > 0:
        time.sleep(crash_check_delay)
    rc = proc.poll()
    if rc is not None and rc != 0:
        logger.warning(
            "odon exited immediately with code %s — no window will appear. This usually "
            "means no usable GPU/display, or a samplesheet odon could not import. The "
            "plate itself is written and unaffected: %s", rc, samplesheet)
    return proc
