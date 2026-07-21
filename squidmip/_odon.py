"""IMA-212 odon bridge — samplesheet over the existing plate + arm's-length launch.

Odon (https://github.com/alexcoulton/odon) is a native Rust **desktop GUI** viewer.
It is NOT a web or remote renderer: every code path except ``--check`` calls
``eframe::run_native`` and opens an OpenGL window, there is no HTTP server, no WASM
build, and no headless render-to-file. "Remote" in Odon means it can *read* zarr over
HTTP/S3, not that it *serves* pixels. This module evaluates it as a fast **local**
alternative to the embedded ndviewer_light detail viewer.

**Verified against Odon v0.1.5.** ``--mosaic-samplesheet`` and the positional
``id,path`` column contract both come from that version's docs; ``find_odon`` proves a
file exists, not that its flags match, so a future rename would surface as an opaque
launch failure. If that happens, check this version pin first.

Nothing here writes pixels. ``squidmip._output.write_plate`` already emits OME-Zarr
that Odon can open as-is::

    Odon requires                              SquidMIP writes
    ─────────────────────────────────────────  ──────────────────────────────────
    zarr v2 or v3 (.zattrs, else zarr.json)    zarr v3            _zarr_store.py:90
    NGFF 0.4/0.5, unwraps {"ome": {...}}       0.5, ome-nested    _zarr_store.py:92
    multiscales[0] (hard error if absent)      always written     _output.py:249
    per-level scale of exactly len(axes)       5 elems, 5 axes    _output.py:180
    axes named y and x (hard error if absent)  t,c,z,y,x          _output.py:184
    uint8 / uint16 only                        reader enforces    reader.py:55

What Odon canNOT do, and what this module therefore works around:

  * **No HCS/plate model at all** (zero ``plate``/``well``/``hcs`` hits in its source).
    It opens ONE image group at a time; the samplesheet CSV is its own flattening
    mechanism. So the plate becomes a row-per-field list and well identity survives
    only as a metadata column.
  * **Ignores ``omero.color``.** Its ``OmeroChannel`` has no color field — colors come
    from a fixed 8-color cycle in file order, plus a hardcoded case forcing any channel
    whose name contains "dapi" to blue. Only channel 0 is visible by default. So the
    same data renders in DIFFERENT colors here than in ndviewer_light (which does read
    omero.color). That is expected, not a defect in our output. We deliberately do NOT
    rename channels to game that heuristic.
  * **No ``t`` axis handling.** Any axis that isn't c/z/y/x is pinned to range 0..1, so
    a T axis is silently fixed at timepoint 0. Harmless while n_t == 1.

Flow::

    <acq>.hcs/
      plate.ome.zarr/{row}/{col}/{fov}/zarr.json   ─┐  glob: disk IS the source of truth
      odon_samplesheet.csv                         ◄┘  id,path,well,fov (paths RELATIVE)
              │
              └─► odon --mosaic-samplesheet <csv>      detached; poll() catches a crash

Enumeration is a directory glob, NOT a second call to ``select_fovs``. Re-deriving the
field list from metadata with independently-passed ``n_fovs``/``regions`` can silently
disagree with what was actually written, and it would force a live reader — which
forecloses pointing Odon at a previously-written ``.hcs``, the likeliest real use.

The glob matches ``zarr.json``, never the directory: ``_write_field`` writes the pyramid
arrays FIRST and calls ``write_group`` LAST (_output.py:241-252), so a run killed by
``stop()``, a crash, or a full disk leaves a field directory that exists with no
``multiscales`` — precisely Odon's documented hard error. Testing ``dir.exists()`` would
pass that straight through.
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

from squidmip._output import _row_sort_key, parse_well_id

logger = logging.getLogger("squidmip")

ODON_RELEASES_URL = "https://github.com/alexcoulton/odon/releases"
ODON_VERIFIED_VERSION = "0.1.5"

PLATE_DIRNAME = "plate.ome.zarr"
SAMPLESHEET_NAME = "odon_samplesheet.csv"

# Odon's samplesheet contract: header required, first TWO columns positional (id, path);
# every later column is free-form ROI metadata usable for mosaic grouping/sorting.
# `well` is the conventional one and is the only place well identity survives, since
# Odon has no plate model. `row`/`col` are deliberately omitted — they are
# parse_well_id(well) and would be redundant.
SAMPLESHEET_COLUMNS = ("id", "path", "well", "fov")

_CRASH_CHECK_DELAY_S = 1.0   # how long to let odon live before deciding it crashed
_CHECK_TIMEOUT_S = 120.0     # `odon --check` reads one coarse tile; generous ceiling


# --- plate enumeration -----------------------------------------------------------------

def _plate_dir(hcs_dir: Path) -> Path:
    """Resolve the ``plate.ome.zarr`` inside *hcs_dir* (or *hcs_dir* itself if it IS one)."""
    hcs_dir = Path(hcs_dir).expanduser()
    plate = hcs_dir if hcs_dir.name == PLATE_DIRNAME else hcs_dir / PLATE_DIRNAME
    if not plate.is_dir():
        raise FileNotFoundError(
            f"no {PLATE_DIRNAME} under {hcs_dir} — point this at a squidmip output "
            "directory (the one containing plate.ome.zarr), or run squidmip first."
        )
    return plate


def _field_sort_key(row: str, col: str, fov: str):
    """Natural plate order: A..Z then AA.., column numerically, fov numerically.

    Without this a plain lexicographic sort gives B10, B2, B3 — which becomes the mosaic's
    display order, since Odon renders the samplesheet as a linear sequence.
    """
    return (_row_sort_key(row), int(col), int(fov))


def iter_fields(hcs_dir):
    """Yield ``(row, col, fov, field_dir)`` for every COMPLETE field group, in plate order.

    A field group is exactly three levels below ``plate.ome.zarr`` — the plate, row, and
    well groups sit at depths 0, 1 and 2 and are not matched, and the pyramid arrays
    (``{fov}/0/zarr.json``) sit at depth 4. Incomplete fields have no ``zarr.json`` yet
    and are skipped by construction.
    """
    plate = _plate_dir(hcs_dir)
    found = []
    for zj in plate.glob("*/*/*/zarr.json"):
        field_dir = zj.parent
        row, col, fov = field_dir.parts[-3:]
        try:
            parse_well_id(row + col)          # canonical <letters><digits> well id
            key = _field_sort_key(row, col, fov)
        except (ValueError, TypeError):
            # Not a plate-shaped path (stray directory). Skip rather than emit a row that
            # would make Odon error, but say so — a silent drop hides a real layout bug.
            logger.warning("odon: skipping non-plate-shaped path %s", field_dir)
            continue
        found.append((key, row, col, fov, field_dir))
    found.sort(key=lambda t: t[0])
    for _, row, col, fov, field_dir in found:
        yield row, col, fov, field_dir


# --- samplesheet -----------------------------------------------------------------------

def write_samplesheet(hcs_dir, out_csv=None) -> Path:
    """Write Odon's samplesheet CSV over an existing squidmip output directory.

    Takes a DIRECTORY, not metadata — so it works on a plate written days ago, with no
    reader and no acquisition on disk.

    Paths are written RELATIVE to the CSV, because Odon resolves relative paths against
    the samplesheet's own location; absolute paths would break the moment the output is
    copied to a share.
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
            # CSV lives outside the plate tree: fall back to an absolute path, which Odon
            # uses as written. Correct, just not portable — so say so once.
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


# --- binary discovery ------------------------------------------------------------------

def _platform_default() -> Optional[Path]:
    """The install location for this platform's released artifact, if there is one."""
    if sys.platform == "darwin":
        # The .dmg installs an .app bundle and does NOT put odon on PATH — so this, not
        # shutil.which, is the normal case on macOS.
        return Path("/Applications/odon.app/Contents/MacOS/odon")
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        return Path(program_files) / "odon" / "odon.exe"
    if sys.platform.startswith("linux"):
        return Path("/usr/bin/odon")          # from odon_<ver>_amd64.deb
    return None


def _no_build_for_this_platform() -> Optional[str]:
    """Explain why no release exists here, or None if one does.

    Odon v0.1.5 ships exactly three artifacts: odon-macos.dmg (Apple Silicon only),
    OdonSetup-<ver>-windows-x86_64.exe, and odon_<ver>_amd64.deb.
    """
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in ("aarch64", "arm64"):
        return "there is no Linux arm64 build of odon (releases are macOS arm64, Windows x86_64, Linux amd64)"
    if sys.platform == "darwin" and machine == "x86_64":
        return "the macOS odon build is Apple Silicon only (no Intel build is published)"
    return None


def find_odon() -> Path:
    """Locate the odon binary: ``$ODON_BIN`` -> ``PATH`` -> this platform's install location.

    Never downloads or vendors anything. Odon is GPL-3.0-only and SquidMIP is
    BSD-3-Clause: launching a separately-installed, unmodified binary with CLI arguments
    is mere aggregation and leaves our license alone. Shipping the binary would not.
    """
    override = os.environ.get("ODON_BIN")
    if override:
        # An override that is silently ignored is worse than no override — fail loud
        # rather than falling through to a different binary than the user named.
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
        "or set $ODON_BIN to the binary. SquidMIP never bundles it: odon is GPL-3.0-only."
    )


# --- headless probe --------------------------------------------------------------------

def check_odon(field_dir, odon_bin=None) -> bool:
    """Run ``odon --check`` on one field group — the ONLY headless path Odon has.

    It loads a single coarse tile, prints ``OK: loaded tile level N ...`` and returns
    before any window is created. It is single-dataset and local-only, and does NOT
    accept a samplesheet — which is why the samplesheet half of the oracle stays manual.
    """
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


# --- launch ----------------------------------------------------------------------------

def launch_odon(samplesheet, *, mosaic_cols: Optional[int] = None, odon_bin=None,
                crash_check_delay: float = _CRASH_CHECK_DELAY_S) -> subprocess.Popen:
    """Launch Odon on *samplesheet* as a detached process and return the handle.

    Detached and never waited on: Odon is a GUI that should outlive the CLI invocation,
    and blocking would hang a batch run until a human closed a window.

    Detachment means a crash is otherwise SILENT — the user just sees no window appear.
    So poll once after a short delay and report a non-zero exit. This is the common
    failure on a machine with no GPU or a broken driver.
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
