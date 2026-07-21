"""IMA-184 output: canonical multiscale OME-zarr HCS plate + individual-TIFF export.

Consumes IMA-188's ``project_plate`` stream (single-thread — the engine parallelises the
projection internally and hands results back one at a time, so the writer needs no locking)
and writes each well as it arrives. Two outputs from one pass:

  1. ``<out>/plate.ome.zarr``  — OME-NGFF v0.5 HCS *plate* (zarr v3), Squid's canonical
     ``control/core/zarr_writer.py`` layout EXTENDED with a per-FOV pyramid (levels 0..L, each a 2x
     block-mean of the previous) so a pyramid-aware reader / plate navigator can show a field without
     pulling full-res. Level 0 stays full-res and pixel-exact, so canonical single-level consumers are
     unchanged; fields <= 256 px keep just level 0:
        plate.ome.zarr/                     zarr.json  = plate group (rows/columns/wells)
          {row}/                            zarr.json  = row group (bare)
            {col}/                          zarr.json  = well group (images -> raw fov ids)
              {fov}/                        zarr.json  = image group (multiscales + omero)
                0/                          array: full-res (T, C, 1, Y, X), native dtype
                1/ 2/ ...                   array: 2x-downsampled pyramid levels (native dtype)
     Opens in ndviewer_light (directory-walk -> array ``0`` + ``omero`` colors; it reads only level 0)
     AND validates as a spec plate (plate/well group metadata) under an independent reader (zarr-python).

  2. ``<out>/tiff/{t}/{region}_{fov}_0_{channel}.tiff`` — individual per-plane TIFFs in Squid's
     filename convention, z collapsed to ``0`` (the projection), native dtype. You Yan's
     "individual tiff output": channel identity lives in the filename, no OME-XML, so it drops
     straight into Nick's existing Squid-reading workflow.

Flow::

    reader.metadata ─► select_fovs ─► plate/row/well GROUP metadata written UP FRONT
                                       (full layout known from metadata, so the stream's
                                        completion-order arrival needs no ordering logic)
    project_plate(reader, ...) ─► (region, fov, (T,C,1,Y,X))
                                       │  per well, as it arrives:
                                       ├─► field group: array 0 (full-res) + multiscales + omero
                                       └─► individual TIFFs (one per channel, per timepoint)

IMA-217 adds ONE thing to the metadata: each dataset also carries an NGFF ``translation``
transform — the field's top-left corner in stage MICROMETRES, derived from
``metadata["fov_positions_um"]`` (which records FOV *centres*, so half a frame is subtracted;
see :func:`field_origin_um`). Per the NGFF v0.4/v0.5 spec a dataset MUST have exactly one
``scale`` and MAY have one ``translation`` listed *after* it, each with one entry per axis —
which is exactly what is written, so stock readers (ome-zarr-py, napari-ome-zarr) are unaffected
while the plate becomes self-describing in world space. ``squidmip._tilesource`` rebuilds the
whole plate layout from it without ever re-reading coordinates.csv. Acquisitions with no stage
positions get no translation and are byte-identical to the pre-IMA-217 output.

IMA-230 puts a DISK PRE-FLIGHT in front of the whole thing: the write is estimated from the real
numbers (fields x n_t x channels x frame bytes x the exact pyramid factor) and refused up front,
naming the estimate and the free space, instead of dying most of the way through and leaving a
half-written store. While it runs, the store carries a ``.squidmip-incomplete`` marker, every
field is published by atomic rename (so a field is never half-visible) and each region's
intermediates are swept as it finishes. See :func:`estimate_write_bytes` / :func:`check_disk_space`.

IMA-231 adds, per WELL and only on this persist path (the live viewer already has
coordinates.csv), a Fractal/ngio ``tables/FOV_ROI_table``: an AnnData-encoded ROI table giving
every FOV's box in µm, so an external tool can recover FOV boundaries after the region is fused.
Its corners are :func:`field_origin_um` — the same top-left corner as the NGFF ``translation`` and
as ``_tilesource.fov_bboxes_um``. See :func:`fov_roi_records_um`.

Colors come from ``metadata.channels[].display_color`` (IMA-189 already resolves them, mapped
by name, raising on an unrecognised channel) — the writer never re-parses the acquisition YAML.
Channel order in ``omero`` and in the TIFF filenames follows ``metadata.channels`` order, which
is exactly the array's C-axis order (IMA-183 builds the C axis from that list).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import tifffile

from squidmip._engine import _default_workers, project_plate
from squidmip._zarr_store import create_array, write_array, write_group
from squidmip.projection import resolve_n_fovs, select_fovs

_NGFF_VERSION = "0.5"
_WAVELENGTH_RE = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")  # a standalone 3-4 digit nm in a channel name

# Pyramid: halve (Y, X) per level until the coarsest level fits in a screen-sized tile, capped at a
# few levels. A per-FOV pyramid IS worthwhile at HCS scale (4168x4168 fields): the coarse levels let
# a plate navigator / pyramid-aware reader (napari, a future LOD viewer) show a well without pulling
# the full-res plane. Small fields (<= _PYRAMID_MIN_YX, e.g. test frames) collapse to level 0 alone,
# so the canonical single-level output is unchanged for them.
_PYRAMID_MIN_YX = 256
_PYRAMID_MAX_LEVELS = 6
_WRITE_WORKERS = min(4, _default_workers())   # bounded writer pool overlapping pyramid-build + zstd
#                            (~75% of end-to-end wall time when serial) with projection. Adapt to the
#                            machine like the engine (never more writer threads than usable cores);
#                            4 is plenty — the write stage is I/O + compress bound, not CPU-scaling.


# --- IMA-230: disk pre-flight guard ----------------------------------------------------------
#
# Squid already stops an acquisition BEFORE it overflows the disk, in
# ``control/widgets.py::check_space_available_with_error_dialog`` +
# ``control/core/multi_point_controller.py::get_estimated_acquisition_disk_storage``. Four things
# are reused here, deliberately, rather than a new policy being invented:
#   1. count the units of work first (Squid: Nt x NZ x FOVs x configs = images), multiply by a
#      per-unit byte cost;
#   2. a FACTOR OF SAFETY on the estimate before the comparison (Squid: 1.03) — an over-estimate
#      only ever asks for a roomier disk, which is the safe way to be wrong;
#   3. a small fixed allowance for the non-image files (Squid: 100 kB of metadata/JSON);
#   4. ``shutil.disk_usage(dir).free`` on the SAVE DIRECTORY (Squid's ``utils.get_available_disk_space``)
#      and refuse up front with a message naming BOTH numbers.
# SquidMIP adds what Squid cannot know: the pyramid tail (:func:`plate_pyramid_factor`, the exact
# level-shape sum this module writes, not a 4/3 approximation) and the optional second TIFF copy.
#
# This is the SAME policy the plate viewer's pre-flight check applies (``_viewer._check_disk``,
# which refused a ~6 GB write during IMA-206): field count x n_t x channels x frame bytes x
# pyramid, refused when it would eat past the headroom. That check is the GUI's early warning —
# it can only warn a user before a run starts. This one is the ENFORCEMENT point on the write
# path itself, so the CLI, the tests and any programmatic caller get the same refusal; it uses
# the exact pyramid factor where the viewer uses the 1.34 closed form, so it is never the looser
# of the two. ``_viewer._check_disk`` should be reduced to a call to
# :func:`estimate_write_bytes` / :func:`check_disk_space` (that file is owned elsewhere today).

_DISK_SAFETY_FACTOR = 1.03          # Squid's `factor_of_safecty`
_DISK_NON_IMAGE_BYTES = 100 * 1024  # Squid's `non_image_file_size`: zarr.json/omero/attrs allowance
_DISK_HEADROOM = 0.10               # keep this FRACTION of free space free (viewer: est > free*0.9)
_DISK_MIN_FREE_BYTES = 256 * 1024 ** 2   # ...and never less than this, however small the disk is.
#                                          A percentage alone lets an almost-full disk approve a
#                                          write that lands it at a few MB free, where everything
#                                          else on the machine then fails. Both are overridable.
_INCOMPLETE_MARKER = ".squidmip-incomplete"
_PARTIAL_PREFIX = "."               # in-progress field dirs are ".{fov}.partial" — a leading dot
#                                     keeps them out of ndviewer's digit-named field discovery.
_PARTIAL_SUFFIX = ".partial"
_GB = 1024 ** 3


class InsufficientDiskSpaceError(OSError):
    """Refusing to start a write that would not fit (raised BEFORE anything is created)."""


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def plate_pyramid_factor(frame_shape, **kw) -> float:
    """Total written pixels per level-0 pixel, i.e. the exact pyramid overhead of one field.

    ``1 + 1/4 + 1/16 + ...`` truncated at the real level ladder :func:`pyramid_shapes` produces
    (and, for odd axes, at the real cropped shapes) — not the 4/3 infinite-sum approximation. A
    field too small to pyramid returns exactly 1.0.
    """
    shapes = pyramid_shapes(frame_shape, **kw)
    y0, x0 = shapes[0]
    return float(sum(y * x for y, x in shapes) / (y0 * x0))


def estimate_write_bytes(metadata: dict, *, n_fovs: Optional[int] = 1, regions=None,
                         tiff: bool = False, n_z: int = 1) -> int:
    """Bytes :func:`write_from_stream` will need for this acquisition, from the real numbers.

    ``n_regions x n_fovs x n_channels x n_z x frame_bytes``, times the pyramid factor, times the
    safety factor, plus the fixed non-image allowance — and times ``n_t``, which is the term whose
    omission let a time-lapse fill the disk mid-write.

    ``n_z`` is the number of Z planes *written per field*, which is 1 on this path: the MIP
    collapses Z (the writer asserts ``shape[2] == 1``). It is a parameter, not a constant, so a
    caller that persists a stack passes the stack depth and gets the same arithmetic — but it
    defaults to what this module actually writes, because a 10x over-estimate refuses runs that
    would have fit and is no kinder than an under-estimate.

    UNCOMPRESSED throughout: real fluorescence zstds unpredictably (often < 1.2x), so discounting
    for compression would under-estimate. Returns 0 when the metadata cannot support an estimate
    (no ``frame_shape``); the caller then skips the check rather than blocking on a guess.
    """
    frame_shape = metadata.get("frame_shape")
    channels = metadata.get("channels") or []
    if not frame_shape or not channels:
        return 0
    ny, nx = int(frame_shape[0]), int(frame_shape[1])
    itemsize = np.dtype(metadata.get("dtype", "uint16")).itemsize
    fovs_per_region = metadata.get("fovs_per_region") or {}

    scoped = list(fovs_per_region) if regions is None else [r for r in regions if r in fovs_per_region]
    if n_fovs is None:
        n_fields = sum(len(fovs_per_region[r]) for r in scoped)
    else:                                   # select_fovs takes at most n_fovs per region
        n_fields = sum(min(int(n_fovs), len(fovs_per_region[r])) for r in scoped)

    frame_bytes = n_fields * int(metadata.get("n_t", 1) or 1) * len(channels) * int(n_z) * ny * nx * itemsize
    total = frame_bytes * plate_pyramid_factor((ny, nx))
    if tiff:
        total += frame_bytes                # a second, uncompressed, pyramid-free copy
    return int(total * _DISK_SAFETY_FACTOR) + _DISK_NON_IMAGE_BYTES


def free_bytes(path) -> int:
    """Free bytes on the filesystem that will hold *path* — Squid's ``get_available_disk_space``.

    Squid requires the directory to exist; a write destination legitimately does not yet, so the
    nearest EXISTING ancestor is stat-ed instead (same filesystem, same answer). Returns -1 when
    the filesystem cannot be stat-ed at all, which the caller reads as "don't block".
    """
    p = Path(path).absolute()
    for candidate in (p, *p.parents):
        if candidate.is_dir():
            try:
                return int(shutil.disk_usage(candidate).free)
            except OSError:
                return -1
    return -1


def check_disk_space(out_dir, required_bytes: int, *, headroom: Optional[float] = None,
                     min_free_bytes: Optional[int] = None, what: str = "this write") -> None:
    """Raise :class:`InsufficientDiskSpaceError` unless *required_bytes* fits with headroom.

    Headroom defaults to :data:`_DISK_HEADROOM` of the free space but never less than
    :data:`_DISK_MIN_FREE_BYTES`, so an estimate can never consume the last of the disk; both are
    overridable per call and via ``SQUIDMIP_DISK_HEADROOM`` / ``SQUIDMIP_MIN_FREE_BYTES``.
    """
    if required_bytes <= 0:
        return
    frac = _env_float("SQUIDMIP_DISK_HEADROOM", _DISK_HEADROOM) if headroom is None else float(headroom)
    floor = (int(_env_float("SQUIDMIP_MIN_FREE_BYTES", _DISK_MIN_FREE_BYTES))
             if min_free_bytes is None else int(min_free_bytes))
    free = free_bytes(out_dir)
    if free < 0:
        return                                    # can't stat the disk -> don't block the run
    reserve = max(int(free * frac), floor)
    budget = free - reserve
    if required_bytes > budget:
        raise InsufficientDiskSpaceError(
            f"refusing to start: {what} needs ~{required_bytes / _GB:.2f} GB but "
            f"{Path(out_dir).absolute()} has {free / _GB:.2f} GB free "
            f"(keeping {reserve / _GB:.2f} GB headroom, so {max(budget, 0) / _GB:.2f} GB usable). "
            "Free space, pick another disk, or lower the headroom "
            "(disk_headroom= / SQUIDMIP_DISK_HEADROOM)."
        )


# --- IMA-230: partial-write hygiene ----------------------------------------------------------

def is_incomplete(plate_dir) -> bool:
    """True while a plate store is mid-write, or if the write that made it never finished."""
    return (Path(plate_dir) / _INCOMPLETE_MARKER).exists()


def _mark_incomplete(plate_dir: Path, info: dict) -> None:
    (plate_dir / _INCOMPLETE_MARKER).write_text(json.dumps(info, indent=2))


def _clear_incomplete(plate_dir: Path) -> None:
    try:
        (plate_dir / _INCOMPLETE_MARKER).unlink()
    except FileNotFoundError:
        pass


def _partial_dir(well_dir: Path, fov) -> Path:
    return well_dir / f"{_PARTIAL_PREFIX}{fov}{_PARTIAL_SUFFIX}"


def _publish(tmp: Path, final: Path) -> None:
    """Atomically make *tmp* visible as *final* — a field appears whole or not at all.

    ``os.replace`` on a directory needs the target absent (or an empty dir), so a rerun's old
    field is removed first; the window between the two is the only moment a reader could see the
    field missing, and a missing field is honest where a half-written one is not.
    """
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)


def _cleanup_partials(directory: Path) -> int:
    """Remove any leftover ``.{fov}.partial`` intermediates under *directory*. Returns the count.

    Called per REGION as its last field lands (and once at the end), so a long multi-region run
    never accumulates every region's intermediates at once — a crashed field's temp directory is
    reclaimed while the next region is still being projected, not hours later.
    """
    n = 0
    if not directory.is_dir():
        return 0
    for child in directory.iterdir():
        if child.is_dir() and child.name.startswith(_PARTIAL_PREFIX) and child.name.endswith(_PARTIAL_SUFFIX):
            shutil.rmtree(child, ignore_errors=True)
            n += 1
    return n


# --- well id <-> row/col --------------------------------------------------------------------

def parse_well_id(region: str) -> tuple[str, str]:
    """Split a well id into (row_letters, col_digits) — vendored from Squid ``utils.parse_well_id``.

    Squid's canonical parser upper-cases then partitions alphabetic vs numeric characters
    (``"aa3" -> ("AA", "3")``); the HCS layout is ``plate.ome.zarr/{row}/{col}/{fov}/0`` and
    ndviewer_light rebuilds ``well_id = row_dir + col_dir`` by concatenation. So the column is
    NOT zero-padded — ``B2 -> B/2`` (``B/02`` would still be discovered, ``"02".isdigit()`` is
    True, but report the well as ``B02`` != the real id ``B2``, breaking well-id fidelity).

    We match Squid's accepted inputs exactly (uppercase, multi-letter rows, no padding) but,
    for a scientific tool, additionally ASSERT the canonical ``<letters><digits>`` shape and
    fail loud: a manual/no-plate region (Squid would silently accumulate stray chars into the
    column) must not be written to a mislabelled directory.
    """
    s = str(region).upper()
    letters = "".join(c for c in s if c.isalpha())
    digits = "".join(c for c in s if not c.isalpha())
    if not letters or not digits.isdigit() or letters + digits != s:
        raise ValueError(
            f"region {region!r} is not a canonical <letters><digits> well id (e.g. 'B2', 'AA3'); "
            "the HCS plate layout needs a row/column split. Manual/no-plate acquisitions are out "
            "of scope (IMA-189: well-plate layout only)."
        )
    return letters, digits


# Back-compat alias for the earlier name used in this module's history.
split_well = parse_well_id


def _row_sort_key(row: str):
    # Plate row order: A..Z then AA..AF (shorter labels first, then lexicographic).
    return (len(row), row)


# --- NGFF metadata builders -----------------------------------------------------------------

def plate_metadata(regions: Iterable[str], field_count: int, name: str = "plate") -> dict:
    """OME-NGFF v0.5 ``plate`` group metadata from the well ids (rows/columns/wells)."""
    splits = [(r, *split_well(r)) for r in regions]
    rows = sorted({row for _, row, _ in splits}, key=_row_sort_key)
    cols = sorted({col for _, _, col in splits}, key=int)
    wells = [
        {"path": f"{row}/{col}", "rowIndex": rows.index(row), "columnIndex": cols.index(col)}
        for _, row, col in splits
    ]
    return {
        "version": _NGFF_VERSION,
        "plate": {
            "name": name,
            "rows": [{"name": r} for r in rows],
            "columns": [{"name": c} for c in cols],
            "wells": wells,
            "field_count": int(field_count),
        },
    }


def _downsample_yx(image: np.ndarray) -> np.ndarray:
    """Halve a ``(T, C, Z, Y, X)`` field in Y and X by 2x2 block-mean, native dtype kept.

    Each spatial axis is halved only when it has >= 2 px — a size-1 axis is left intact, so a narrow
    strip never collapses to a zero-width level (which would divide-by-zero in ``_multiscales``). Odd
    axes are cropped by one before halving. Vectorised reshape+mean over the whole 5-D field is ~3x
    faster than looping ``_area_downsample`` per plane (measured 250ms vs 670ms for a 4168x4168x4ch
    field), which matters because every written well pays this per level. Rounded back to the source
    dtype (clamped for integers). mean in float32 (not float64) halves the transient and is exact for
    a 2x2 mean of uint16 (max sum 4*65535 is within float32's integer range).
    """
    fy = 2 if image.shape[-2] >= 2 else 1
    fx = 2 if image.shape[-1] >= 2 else 1
    y = (image.shape[-2] // fy) * fy                       # crop to a multiple of the axis factor
    x = (image.shape[-1] // fx) * fx
    cropped = image[..., :y, :x]
    ds = cropped.reshape(*cropped.shape[:-2], y // fy, fy, x // fx, fx).mean(axis=(-3, -1), dtype=np.float32)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        np.rint(ds, out=ds)                       # round + clip IN PLACE — no extra float buffers
        np.clip(ds, info.min, info.max, out=ds)
    return ds.astype(image.dtype)


def _pyramid(image: np.ndarray, *, min_yx: int = _PYRAMID_MIN_YX,
             max_levels: int = _PYRAMID_MAX_LEVELS) -> list[np.ndarray]:
    """Level list ``[full-res, /2, /4, ...]`` — halving until the coarsest fits *min_yx*
    (or *max_levels*). A field already <= the floor yields just ``[image]`` (level 0).

    The stopping rule is duplicated, shape-only, in :func:`pyramid_shapes`; the two are pinned
    together by a test, because IMA-217 needs the level ladder BEFORE any pixels exist (it builds
    the viewer's :class:`~squidmip._tiling.Geometry` from metadata alone).
    """
    levels = [image]
    while (max(levels[-1].shape[-2:]) > int(min_yx) and len(levels) < int(max_levels)):
        levels.append(_downsample_yx(levels[-1]))
    return levels


def pyramid_shapes(frame_shape, *, min_yx: int = _PYRAMID_MIN_YX,
                   max_levels: int = _PYRAMID_MAX_LEVELS) -> list[tuple[int, int]]:
    """The ``(Y, X)`` of every pyramid level :func:`_pyramid` would write, from the shape alone.

    Pure arithmetic — no array, no I/O — so a viewer can build its level ladder from acquisition
    metadata before a single field has been projected. Mirrors ``_downsample_yx``: an axis with
    < 2 px is left intact, an odd axis is cropped by one before halving (so 521 -> 260, not 261).
    """
    y, x = int(frame_shape[0]), int(frame_shape[1])
    if y < 1 or x < 1:
        raise ValueError(f"frame_shape must be positive, got {frame_shape!r}")
    shapes = [(y, x)]
    while max(shapes[-1]) > int(min_yx) and len(shapes) < int(max_levels):
        y, x = shapes[-1]
        fy, fx = (2 if y >= 2 else 1), (2 if x >= 2 else 1)
        shapes.append((y // fy, x // fx))
    return shapes


def _multiscales(level_shapes: list[tuple], pixel_size_um: Optional[float], dz_um: Optional[float] = None,
                 position_um: Optional[tuple] = None) -> dict:
    """multiscales metadata for a per-FOV pyramid: one ``datasets`` entry per level, its scale the
    real downsample factor (level 0's Y,X over this level's Y,X) so physical coordinates stay true.

    ``level_shapes`` is the (Y, X) of each written level, level 0 first. A single-element list gives
    the canonical single-dataset ``0`` output (unchanged for small fields). Axes mirror Squid's
    zarr_writer.

    ``position_um`` is the field's TOP-LEFT corner in stage MICROMETRES — the world coordinate of
    pixel (0, 0). Given it, each dataset also carries an NGFF ``translation`` transform (after the
    scale, as the spec requires: scale then translation, one entry per axis), which is what makes
    the plate self-describing in world space: a pyramid-aware reader (IMA-217's tile source, napari,
    ome-zarr-py) can place every field on the plate WITHOUT re-reading coordinates.csv. Omitted when
    the acquisition has no stage positions, keeping the canonical output byte-identical for those.

    Corner convention: every level's translation is the same corner. Area-averaged downsampling
    nudges the sample *centre* by half a coarse pixel; carrying that half-pixel here would make the
    levels of one field disagree with each other by less than one coarse pixel while breaking the
    "levels share an origin" assumption every mosaic compositor makes. Corner it is, documented.
    """
    p = float(pixel_size_um) if pixel_size_um else 1.0
    dz = float(dz_um) if dz_um else 1.0
    y0, x0 = level_shapes[0]
    datasets = []
    for i, (y, x) in enumerate(level_shapes):
        sy, sx = p * (y0 / y), p * (x0 / x)   # coarse levels have a larger physical pixel
        xforms: list[dict] = [{"type": "scale", "scale": [1.0, 1.0, dz, sy, sx]}]
        if position_um is not None:
            xforms.append({"type": "translation",
                           "translation": [0.0, 0.0, 0.0, float(position_um[1]), float(position_um[0])]})
        datasets.append({"path": str(i), "coordinateTransformations": xforms})
    doc = {
        "version": _NGFF_VERSION,
        "name": "0",
        # NGFF SHOULDs: name the downscaling method and record its provenance, so a consumer knows
        # the coarse levels are area means (not decimation) without reverse-engineering the pixels.
        "type": "mean",
        "metadata": {"method": "squidmip._output._downsample_yx", "description": "2x2 block mean"},
        "axes": [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets,
    }
    return doc


def _wavelength_nm(channel: dict) -> Optional[int]:
    """Best-effort emission wavelength (nm) parsed from the channel name, else None."""
    m = _WAVELENGTH_RE.search(channel.get("name", ""))
    return int(m.group(1)) if m else None


def _omero(channels: list[dict], dtype) -> dict:
    """omero rendering metadata (Squid shape): label, hex color (no '#'), window, wavelength."""
    dmax = float(np.iinfo(np.dtype(dtype)).max)
    out = []
    for ch in channels:
        entry = {
            "label": ch.get("display_name") or ch["name"],
            "color": str(ch["display_color"]).lstrip("#"),
            "active": True,
            "window": {"min": 0.0, "max": dmax, "start": 0.0, "end": dmax},
        }
        wl = _wavelength_nm(ch)
        if wl is not None:
            entry["emission_wavelength"] = {"value": wl, "unit": "nanometer"}
        out.append(entry)
    return {"channels": out}


# --- field + tiff writers --------------------------------------------------------------------

def _validate_image(image: np.ndarray, channels: list[dict]) -> None:
    """Fail loud on anything that isn't a projected ``(T, C, 1, Y, X)`` frame for these channels."""
    if image.ndim != 5 or image.shape[2] != 1:
        raise ValueError(
            f"expected a projected (T, C, 1, Y, X) array (z collapsed to 1), got shape {image.shape}. "
            "IMA-184 writes the projection output of IMA-188; a non-5D or Z>1 array is a seam bug."
        )
    if image.shape[1] != len(channels):
        raise ValueError(
            f"image has C={image.shape[1]} channels but metadata lists {len(channels)} "
            f"({[c['name'] for c in channels]}); channel/axis mismatch — refusing to mislabel omero."
        )


def field_origin_um(centre_um, frame_shape, pixel_size_um) -> Optional[tuple[float, float]]:
    """Stage-µm ``(x, y)`` of a field's TOP-LEFT pixel, from its recorded CENTRE position.

    ``metadata["fov_positions_um"]`` records where the stage was, i.e. the middle of the frame;
    NGFF ``translation`` places pixel (0, 0). Half a frame apart — 388 µm on a 2084 px 20x field,
    which is half an FOV of mosaic shear if it is skipped. Returns None when the position or the
    pixel size is unknown, so the writer simply omits the translation instead of guessing an origin.
    """
    if centre_um is None or not pixel_size_um or frame_shape is None:
        return None
    p = float(pixel_size_um)
    if not p > 0:
        return None
    h, w = int(frame_shape[0]), int(frame_shape[1])
    return (float(centre_um[0]) - w * p / 2.0, float(centre_um[1]) - h * p / 2.0)


def _write_field(field_dir: Path, image: np.ndarray, channels: list[dict], pixel_size_um, dz_um=None,
                 position_um: Optional[tuple] = None) -> int:
    """Write one field: pyramid levels ``0..L`` (0 = full-res, pixel-exact) + multiscales + omero.

    ``position_um`` is the field's top-left corner in stage µm (see :func:`field_origin_um`); it
    becomes the NGFF ``translation`` on every dataset, so the plate carries its own world layout.

    Returns the number of levels written (1 for a small field with no pyramid)."""
    _validate_image(image, channels)
    levels = _pyramid(image)
    for i, lvl in enumerate(levels):
        store = create_array(field_dir / str(i), lvl.shape, lvl.dtype)
        write_array(store, lvl)
    level_shapes = [(int(lvl.shape[-2]), int(lvl.shape[-1])) for lvl in levels]
    write_group(
        field_dir,
        {
            "version": _NGFF_VERSION,
            "multiscales": [_multiscales(level_shapes, pixel_size_um, dz_um, position_um)],
            "omero": _omero(channels, image.dtype),
        },
    )
    return len(levels)


def _write_tiffs(tiff_root: Path, region: str, fov: int, image: np.ndarray, channel_names: list[str]) -> None:
    """Individual per-plane TIFFs: tiff/{t}/{region}_{fov}_0_{channel}.tiff, native dtype.

    Each plane is written to a ``.partial`` sibling and atomically renamed into place (IMA-230):
    a run killed by a full disk used to leave an 8-byte ``.tiff`` that every downstream tool
    happily opened as a real file. A rename either happens or does not, so a published TIFF is
    always a complete one; the temp file is removed on the way out of a failure.
    """
    n_t = image.shape[0]
    for t in range(n_t):
        tdir = tiff_root / str(t)
        tdir.mkdir(parents=True, exist_ok=True)
        for c_i, channel in enumerate(channel_names):
            plane = image[t, c_i, 0]  # (Y, X), native dtype, z collapsed
            final = tdir / f"{region}_{fov}_0_{channel}.tiff"
            tmp = final.with_name(final.name + _PARTIAL_SUFFIX)
            try:
                tifffile.imwrite(tmp, plane)
                os.replace(tmp, final)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise


# --- IMA-231: FOV_ROI_table (Fractal / ngio ROI-table convention) -----------------------------
#
# WHAT THE SPEC ACTUALLY SAYS (read from ngio's source — the fractal-tasks-core "Table
# specifications" page 404s now, table I/O moved to ngio, which fractal-tasks-core 2.x depends on):
#
#   * location   — ``<image group>/tables/<table_name>``; the ``tables`` group's attrs list the
#                  table names: ``{"tables": ["FOV_ROI_table"]}``.
#   * name       — ``FOV_ROI_table`` (``fractal_tasks_core.roi.v1.prepare_FOV_ROI_table``).
#   * payload    — an AnnData object; the six REQUIRED columns are, exactly
#                  (``ngio/tables/v1/_roi_table.py::REQUIRED_COLUMNS``):
#                     x_micrometer, y_micrometer, z_micrometer,
#                     len_x_micrometer, len_y_micrometer, len_z_micrometer
#                  optional ones used here: x_micrometer_original / y_micrometer_original /
#                  z_micrometer_original (ORIGIN_COLUMNS) and path_in_well (PLATE_COLUMNS).
#   * index      — ``FieldIndex``, string, values ``FOV_1``, ``FOV_2``, ... (ngio
#                  ``RoiTableV1Meta.index_key`` default; fractal's prepare_FOV_ROI_table does
#                  ``adata.obs_names = "FOV_" + adata.obs.index``).
#   * attrs      — ngio writes ``{"type": "roi_table", "table_version": "1",
#                  "backend": "anndata_v1", "index_key": "FieldIndex", "index_type": "str"}``;
#                  fractal-tasks-core <= 1.6 requires ``fractal_table_version: "1"`` instead.
#                  Both are written — ngio's model is ``extra="allow"``, so old and new readers work.
#   * origin     — "the axes origin for the ROI positions corresponds to the top-left corner of the
#                  image (for the YX axes) and to the lowest Z plane", i.e. the LOWER BOUND of the
#                  interval, a CORNER, never a centre; and ``prepare_FOV_ROI_table`` calls
#                  ``reset_origin()``, so x/y_micrometer are relative to the image (here: the well)
#                  while the absolute stage coordinate is kept in x/y_micrometer_original.
#
# UNITS. SquidMIP's contract is micrometres with a ``_um`` suffix on every key; ngio's contract is
# micrometres with a ``_micrometer`` suffix. Same unit, different spelling — so the two vocabularies
# are joined by ONE explicit table (:data:`_NGIO_COLUMN`) and nothing is renamed by coincidence.
# There is no scale factor in that map, and there must never be one: a millimetre anywhere here is
# a bug (positions arrive from ``metadata["fov_positions_um"]``, which the reader already converted
# from coordinates.csv's mm).

_ROI_TABLE_NAME = "FOV_ROI_table"
_ROI_INDEX_KEY = "FieldIndex"

# SquidMIP key (µm) -> ngio/fractal column (µm). A RENAME, never a conversion.
_NGIO_COLUMN = {
    "x_um": "x_micrometer",
    "y_um": "y_micrometer",
    "z_um": "z_micrometer",
    "len_x_um": "len_x_micrometer",
    "len_y_um": "len_y_micrometer",
    "len_z_um": "len_z_micrometer",
    "x_original_um": "x_micrometer_original",
    "y_original_um": "y_micrometer_original",
    "z_original_um": "z_micrometer_original",
}
_ROI_NUMERIC_UM = list(_NGIO_COLUMN)          # X columns, in written order
_ROI_OBS_COLUMNS = ("path_in_well",)          # string columns -> AnnData obs (ngio PLATE_COLUMNS)


def fov_roi_records_um(fovs, positions_um, frame_shape, pixel_size_um, *,
                       dz_um: Optional[float] = None, n_z: int = 1) -> list[dict]:
    """One ROI record per FOV of one region — all lengths and positions in MICROMETRES (``_um``).

    ``x_um``/``y_um`` are the TOP-LEFT CORNER, derived by :func:`field_origin_um` from the recorded
    FOV **centre** — the same function that produces the NGFF ``translation`` this writer stamps on
    every dataset, and the same corner ``squidmip._tilesource.fov_bboxes_um`` computes. One
    definition, so a fused region's ROI boxes cannot drift half an FOV from the pixels.

    ``*_original_um`` is that corner in ABSOLUTE stage µm; ``x_um``/``y_um`` are relative to the
    region's own top-left corner (min over its FOVs), which is fractal's ``reset_origin`` — after a
    region is fused, pixel (0, 0) of the fused image is exactly that minimum, so a consumer can use
    the ROI boxes as pixel offsets without knowing where the stage was.

    FOVs with no recorded position are skipped; an empty list means "no table" (the acquisition had
    no coordinates.csv), never a table full of zeros.
    """
    p = float(pixel_size_um or 0.0)
    if not p > 0:
        return []
    h, w = int(frame_shape[0]), int(frame_shape[1])
    len_x_um, len_y_um = w * p, h * p
    # A projected field is one plane, but the ROI describes the physical volume it came from:
    # z-spacing x n planes (the ngio spec's own rule for a 2D table). Unknown spacing -> one plane.
    len_z_um = float(dz_um) * max(1, int(n_z)) if dz_um else 1.0

    raw = []
    for fov in fovs:
        corner = field_origin_um(positions_um.get(fov), (h, w), p)
        if corner is None:
            continue
        raw.append((fov, float(corner[0]), float(corner[1])))
    if not raw:
        return []
    x0 = min(x for _, x, _ in raw)      # the region's own origin: its top-left corner
    y0 = min(y for _, _, y in raw)
    return [
        {
            "FieldIndex": f"FOV_{fov}",   # ngio index convention; the RAW fov id, so the name
            "path_in_well": str(fov),     # and path_in_well point at the field dir on disk
            "x_um": x - x0, "y_um": y - y0, "z_um": 0.0,
            "len_x_um": len_x_um, "len_y_um": len_y_um, "len_z_um": len_z_um,
            "x_original_um": x, "y_original_um": y, "z_original_um": 0.0,
        }
        for fov, x, y in raw
    ]


def _check_roi_micrometres(records: list[dict], frame_extent_um: float) -> None:
    """Fail loud if the FOV pitch says the positions were millimetres wearing a ``_um`` key.

    Same invariant (and the same 1000x defect) as ``_tilesource._check_micrometres``, applied at
    the other end of the pipe: a table is a durable artifact an external tool will trust, so it
    must not be the place a unit bug becomes permanent.
    """
    xs = sorted({round(r["x_um"], 6) for r in records})
    ys = sorted({round(r["y_um"], 6) for r in records})
    gaps = [b - a for v in (xs, ys) for a, b in zip(v, v[1:]) if b > a]
    if gaps and min(gaps) < frame_extent_um / 100.0:
        raise ValueError(
            f"FOV pitch is {min(gaps):.4g} µm for a {frame_extent_um:.4g} µm frame — that is "
            "millimetres in a `_um` key (1000x). Refusing to write an FOV_ROI_table that would "
            "put every downstream tool 1000x off; positions come from metadata['fov_positions_um']."
        )


def _zarr_write_anndata_roi_table(table_dir: Path, records: list[dict]) -> None:
    """Write *records* as an AnnData-encoded zarr v3 group at *table_dir* (no anndata dependency).

    The AnnData zarr encoding is a documented on-disk contract (``encoding-type`` /
    ``encoding-version`` attrs), so it is written directly with zarr-python rather than taking a
    dependency on anndata + h5py for nine columns. Numeric ``_micrometer`` columns go in ``X``
    (``var`` names them, which is where ngio and fractal look for them); the string columns go in
    ``obs``, whose ``_index`` is the ``FieldIndex``.
    """
    import zarr

    root = zarr.open_group(str(table_dir), mode="w", zarr_format=3)
    root.attrs.update({
        "encoding-type": "anndata", "encoding-version": "0.1.0",
        "type": "roi_table",
        "fractal_table_version": "1",     # fractal-tasks-core <= 1.6 refuses a table without it
        "table_version": "1",             # ngio's spelling (extra="allow" keeps both)
        "backend": "anndata_v1",
        "index_key": _ROI_INDEX_KEY, "index_type": "str",
    })

    x = np.array([[float(r[k]) for k in _ROI_NUMERIC_UM] for r in records], dtype=np.float64)
    arr = root.create_array("X", shape=x.shape, dtype="float64")
    arr[...] = x
    arr.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})

    def _string_column(group, name: str, values: list[str]) -> None:
        a = group.create_array(name, shape=(len(values),), dtype=str)
        a[...] = np.array(values, dtype=object)
        a.attrs.update({"encoding-type": "string-array", "encoding-version": "0.2.0"})

    obs = root.create_group("obs")
    obs.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0",
                      "_index": _ROI_INDEX_KEY, "column-order": list(_ROI_OBS_COLUMNS)})
    _string_column(obs, _ROI_INDEX_KEY, [str(r[_ROI_INDEX_KEY]) for r in records])
    for col in _ROI_OBS_COLUMNS:
        _string_column(obs, col, [str(r[col]) for r in records])

    var = root.create_group("var")
    var.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0",
                      "_index": "_index", "column-order": []})
    _string_column(var, "_index", [_NGIO_COLUMN[k] for k in _ROI_NUMERIC_UM])

    for empty in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
        g = root.create_group(empty)
        g.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})


def write_fov_roi_table(image_dir, records: list[dict], *, table_name: str = _ROI_TABLE_NAME) -> Optional[Path]:
    """Write ``<image_dir>/tables/<table_name>`` from :func:`fov_roi_records_um` records.

    Returns the table path, or None when there is nothing to write. The ``tables`` group's attrs
    accumulate the table names (``{"tables": [...]}``), which is how ngio discovers them.
    """
    if not records:
        return None
    _check_roi_micrometres(records, float(records[0]["len_x_um"]))
    image_dir = Path(image_dir)
    tables_dir = image_dir / "tables"
    table_dir = tables_dir / table_name
    tmp = tables_dir / f"{_PARTIAL_PREFIX}{table_name}{_PARTIAL_SUFFIX}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        _zarr_write_anndata_roi_table(tmp, records)
        _publish(tmp, table_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    # The tables group is a plain zarr group whose attrs index its members.
    existing = []
    zj = tables_dir / "zarr.json"
    if zj.exists():
        try:
            existing = json.loads(zj.read_text()).get("attributes", {}).get("tables", [])
        except (OSError, ValueError):
            existing = []
    names = list(dict.fromkeys([*existing, table_name]))
    zj.write_text(json.dumps({"zarr_format": 3, "node_type": "group",
                              "attributes": {"tables": names}}, indent=2))
    return table_dir


# --- orchestration ---------------------------------------------------------------------------

def write_from_stream(
    metadata: dict,
    stream: Iterator[tuple[str, int, np.ndarray]],
    out_dir,
    *,
    n_fovs: Optional[int] = 1,
    tiff: bool = False,
    on_well=None,
    write_workers: int = _WRITE_WORKERS,
    stop=None,
    regions=None,
    check_disk: bool = True,
    disk_headroom: Optional[float] = None,
    min_free_bytes: Optional[int] = None,
    roi_table: bool = True,
) -> dict:
    """Write the plate + (optionally) TIFFs from a ``(region, fov, image)`` stream and *metadata*.

    The core of :func:`write_plate`, split out so it can be driven clean-room in tests with a
    fabricated metadata dict + a hand-built stream (no reader, no data on disk).

    Each projected well is handed to a bounded writer POOL (``write_workers`` threads) so the disk
    write — pyramid build + zstd compress, ~75% of end-to-end wall time when serial — overlaps the
    projection engine instead of starving it. Wells write to disjoint directories, so parallel
    writes never contend; at most ~``write_workers`` wells are in flight, so peak memory stays
    O(engine workers + write_workers), never the whole plate.

    ``on_well(region, fov, image)`` is an optional callback invoked after each well is written.
    NOTE: it runs on a WRITER THREAD and several may overlap — it MUST be thread-safe (the plate
    viewer guards its shared contrast/tiles with a lock). ``stop()`` is an optional predicate polled
    before each submit; when it returns True the stream is abandoned and in-flight writes are drained
    — a clean partial-plate stop for a cancelled GUI run.

    IMA-230: before anything is created, the write is estimated (:func:`estimate_write_bytes`) and
    refused with :class:`InsufficientDiskSpaceError` if it would not fit with headroom — nothing
    is written at all, rather than dying 94% of the way through. ``check_disk=False`` opts out;
    ``disk_headroom`` / ``min_free_bytes`` tune the reserve. While the store is being written it
    carries a ``.squidmip-incomplete`` marker (:func:`is_incomplete`), each field is published by
    atomic rename so a half-written one is never visible, and each region's intermediates are
    swept as its last field lands.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from threading import Lock

    out_dir = Path(out_dir)
    plate_dir = out_dir / "plate.ome.zarr"
    tiff_root = out_dir / "tiff"

    if check_disk:
        need = estimate_write_bytes(metadata, n_fovs=n_fovs, regions=regions, tiff=tiff)
        scope = "this plate write" if regions is None else f"this {len(list(regions))}-well write"
        check_disk_space(out_dir, need, headroom=disk_headroom, min_free_bytes=min_free_bytes,
                         what=scope)

    wells = select_fovs(metadata, n_fovs=n_fovs)  # {region: [fov, ...]}, deterministic
    # NGFF field_count is a single plate-level scalar and is int()-ed below, so n_fovs=None
    # (= all FOVs) MUST be resolved to a concrete number here or the write raises TypeError
    # deep in plate_metadata. A ragged plate reports the max, the only value that does not
    # under-describe some well.
    field_count = resolve_n_fovs(metadata, n_fovs)
    if regions is not None:   # subset: write only these wells (keep the requested order), for previews
        keep = list(dict.fromkeys(regions))
        wells = {r: wells[r] for r in keep if r in wells}

    # Full plate/row/well group metadata written UP FRONT (layout is fully known from metadata).
    write_group(plate_dir, plate_metadata(wells.keys(), field_count=field_count))
    # ...and the store declares itself unfinished from its first byte until its last (IMA-230), so
    # a run killed mid-write leaves something a reader can TELL is incomplete.
    _mark_incomplete(plate_dir, {"wells": list(wells), "fields": sum(len(f) for f in wells.values())})
    for region, fovs in wells.items():
        row, col = parse_well_id(region)
        write_group(plate_dir / row)  # bare row group
        # well.images paths are the RAW fov ids (Squid uses {fov} as the field dir + image path),
        # not a re-indexed 0-based field index — so a non-contiguous fov set stays faithful.
        write_group(
            plate_dir / row / col,
            {"version": _NGFF_VERSION, "well": {"images": [{"path": str(f)} for f in fovs]}},
        )

    channels = metadata["channels"]
    channel_names = [c["name"] for c in channels]
    pixel_size_um = metadata.get("pixel_size_um")
    dz_um = metadata.get("dz_um")
    positions_um = metadata.get("fov_positions_um") or {}   # {} when there is no coordinates.csv

    remaining = {r: len(f) for r, f in wells.items()}   # fields still owed per region
    remaining_lock = Lock()

    def _write_one(region, fov, image):
        row, col = parse_well_id(region)
        well_dir = plate_dir / row / col
        # Stage µm of the field's top-left pixel -> the NGFF translation. Frame shape comes from the
        # IMAGE, not the metadata, so a cropped/binned field is placed at its true extent.
        origin_um = field_origin_um(positions_um.get((region, fov)), image.shape[-2:], pixel_size_um)
        # Build the field in a ".{fov}.partial" directory and RENAME it into place: the field
        # directory is the RAW fov id (Squid convention, digit-named for ndviewer), and it only
        # ever appears complete. A crash leaves a dot-named temp, which no reader walks.
        tmp = _partial_dir(well_dir, fov)
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            levels = _write_field(tmp, image, channels, pixel_size_um, dz_um, position_um=origin_um)
            if tiff:
                _write_tiffs(tiff_root, region, fov, image, channel_names)
            _publish(tmp, well_dir / str(fov))
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)     # never leave this field's intermediate behind
            raise
        with remaining_lock:                            # per-REGION cleanup as the region finishes,
            remaining[region] -= 1                      # so a long run doesn't hoard every region's
            done_region = remaining[region] <= 0        # intermediates until the very end
        if done_region:
            _cleanup_partials(well_dir)
            if roi_table:
                # IMA-231: the region is complete, so its FOV boundaries can be published for
                # whoever fuses it later (Fractal). Persist path ONLY — the live viewer has
                # coordinates.csv already. Frame shape comes from the IMAGE, like the translation.
                fov_pos = {f: positions_um[(region, f)] for f in wells[region]
                           if (region, f) in positions_um}
                write_fov_roi_table(well_dir, fov_roi_records_um(
                    wells[region], fov_pos, image.shape[-2:], pixel_size_um,
                    dz_um=dz_um, n_z=int(metadata.get("n_z", 1) or 1)))
        if on_well is not None:  # live consumer (plate viewer): render tile + push to ndviewer
            on_well(region, fov, image)
        return levels

    n_written = 0
    n_levels = 1
    stopped = False
    n_writers = max(1, int(write_workers))
    try:
        with ThreadPoolExecutor(max_workers=n_writers, thread_name_prefix="squidmip-write") as ex:
            pending: set = set()
            for region, fov, image in stream:
                if stop is not None and stop():
                    stopped = True
                    break
                pending.add(ex.submit(_write_one, region, fov, image))
                if len(pending) >= n_writers:    # keep <= n_writers wells in flight (bounded memory)
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        n_levels = f.result()    # re-raises a writer-thread exception here
                        n_written += 1
            for f in pending:                     # drain the tail (and any in-flight after a stop)
                n_levels = f.result()
                n_written += 1
    finally:
        # Close the producer promptly on a stop/exception (don't wait for GC) so project_plate's
        # own thread pool shuts down now. Guarded: a plain iterator (used in tests) has no close().
        close = getattr(stream, "close", None)
        if callable(close):
            close()
        # Sweep every well's leftovers, however this run ended (a failure that skipped the
        # per-region sweep, or a stop that abandoned regions mid-flight).
        for region in wells:
            row, col = parse_well_id(region)
            _cleanup_partials(plate_dir / row / col)

    complete = not stopped
    if complete:
        _clear_incomplete(plate_dir)   # last act of a finished write: the store is now trustworthy

    return {
        "plate": str(plate_dir),
        "tiff": str(tiff_root) if tiff else None,
        "n_wells": len(wells),
        "n_fields_written": n_written,
        "levels": n_levels,
        "complete": complete,
    }


def write_plate(
    reader,
    out_dir,
    *,
    n_fovs: Optional[int] = 1,
    workers: Optional[int] = None,
    projector: str = "mip",
    tiff: bool = False,
    on_well=None,
    write_workers: int = _WRITE_WORKERS,
    stop=None,
    on_error=None,
    regions=None,
    check_disk: bool = True,
    disk_headroom: Optional[float] = None,
    min_free_bytes: Optional[int] = None,
    roi_table: bool = True,
) -> dict:
    """Project a plate (IMA-188) and write the canonical OME-zarr + individual TIFFs.

    Consumes :func:`squidmip.project_plate` lazily — each projected well is written as it
    arrives, so peak memory stays at the engine's bounded window, never the whole plate.

    Parameters
    ----------
    reader:
        An IMA-189 ``SquidReader`` (from ``open_reader``).
    out_dir:
        Destination directory; receives ``plate.ome.zarr/`` and (if *tiff*) ``tiff/``.
    n_fovs, workers, projector:
        Passed straight to :func:`squidmip.project_plate`.
    tiff:
        Also write the individual per-plane TIFF export (default False — opt in). This is a SECOND,
        UNCOMPRESSED copy of the output in Squid's ``{region}_{fov}_0_{channel}.tiff`` filename
        convention (You Yan's "individual tiff output"), for tools that read Squid TIFFs directly and
        can't open OME-Zarr. It roughly DOUBLES on-disk size, so it's off unless a caller asks for it.

    Returns
    -------
    dict
        Manifest: output paths, well/field counts, pyramid level count.
    """
    metadata = reader.metadata
    stream = project_plate(reader, n_fovs=n_fovs, workers=workers, projector=projector,
                           on_error=on_error, regions=regions)
    return write_from_stream(metadata, stream, out_dir, n_fovs=n_fovs, tiff=tiff, on_well=on_well,
                             write_workers=write_workers, stop=stop, regions=regions,
                             check_disk=check_disk, disk_headroom=disk_headroom,
                             min_free_bytes=min_free_bytes, roi_table=roi_table)
