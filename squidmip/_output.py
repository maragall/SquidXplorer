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
                                       │  per field, as it arrives:
                                       ├─ stop()? ────────────────► clean USER CANCEL
                                       ├─ storage guard (IMA-230):
                                       │    free < min_free + (in_flight+1) * est_per_field
                                       │                     └────► InsufficientDiskSpace
                                       ├─► field group: array 0 (full-res) + multiscales + omero
                                       ├─► individual TIFFs (one per channel, per timepoint)
                                       └─► record bytes written (prices the NEXT guard check)

    on abort (guard fired, or a real out-of-space error beat it):
        discard half-written field dirs  ─► rewrite plate + well metadata to what survived
                                         ─► raise InsufficientDiskSpace (never a silent partial)

The guard is armed only when a caller passes ``min_free_bytes``; the estimate is measured from the
bytes this writer actually wrote, never assumed from a compression ratio. See ``squidmip._storage``.

Colors come from ``metadata.channels[].display_color`` (IMA-189 already resolves them, mapped
by name, raising on an unrecognised channel) — the writer never re-parses the acquisition YAML.
Channel order in ``omero`` and in the TIFF filenames follows ``metadata.channels`` order, which
is exactly the array's C-axis order (IMA-183 builds the C axis from that list).
"""

from __future__ import annotations

import logging
import os
import re
import shutil as _shutil
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import tifffile

from squidmip._engine import _default_workers, project_plate
from squidmip._storage import (
    InsufficientDiskSpace,
    WrittenBytes,
    estimate_field_bytes,
    free_bytes,
    is_out_of_space,
)
from squidmip._zarr_store import create_array, write_array, write_group
from squidmip.projection import select_fovs

logger = logging.getLogger("squidmip")

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


def _pyramid(image: np.ndarray) -> list[np.ndarray]:
    """Level list ``[full-res, /2, /4, ...]`` — halving until the coarsest fits _PYRAMID_MIN_YX
    (or _PYRAMID_MAX_LEVELS). A field already <= the floor yields just ``[image]`` (level 0)."""
    levels = [image]
    while (max(levels[-1].shape[-2:]) > _PYRAMID_MIN_YX
           and len(levels) < _PYRAMID_MAX_LEVELS):
        levels.append(_downsample_yx(levels[-1]))
    return levels


def _multiscales(level_shapes: list[tuple], pixel_size_um: Optional[float], dz_um: Optional[float] = None) -> dict:
    """multiscales metadata for a per-FOV pyramid: one ``datasets`` entry per level, its scale the
    real downsample factor (level 0's Y,X over this level's Y,X) so physical coordinates stay true.

    ``level_shapes`` is the (Y, X) of each written level, level 0 first. A single-element list gives
    the canonical single-dataset ``0`` output (unchanged for small fields). Axes mirror Squid's
    zarr_writer.
    """
    p = float(pixel_size_um) if pixel_size_um else 1.0
    dz = float(dz_um) if dz_um else 1.0
    y0, x0 = level_shapes[0]
    datasets = []
    for i, (y, x) in enumerate(level_shapes):
        sy, sx = p * (y0 / y), p * (x0 / x)   # coarse levels have a larger physical pixel
        datasets.append({"path": str(i),
                         "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, dz, sy, sx]}]})
    return {
        "version": _NGFF_VERSION,
        "name": "0",
        "axes": [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets,
    }


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


def _write_field(field_dir: Path, image: np.ndarray, channels: list[dict], pixel_size_um, dz_um=None) -> tuple[int, list[Path]]:
    """Write one field: pyramid levels ``0..L`` (0 = full-res, pixel-exact) + multiscales + omero.

    Returns ``(levels_written, paths)`` — *paths* being every file this field put on disk, so the
    storage guard can price the field exactly (IMA-230) instead of walking the output tree."""
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
            "multiscales": [_multiscales(level_shapes, pixel_size_um, dz_um)],
            "omero": _omero(channels, image.dtype),
        },
    )
    return len(levels), _files_under(field_dir)


def _files_under(root: Path) -> list[Path]:
    """Every file under *root* (chunks + zarr.json). Scoped to ONE field directory, so this is
    O(that field), never O(plate) — see squidmip._storage on why the guard must not walk the tree."""
    if not root.exists():
        return []
    return [Path(dp) / f for dp, _, fns in os.walk(root) for f in fns]


def _write_tiffs(tiff_root: Path, region: str, fov: int, image: np.ndarray, channel_names: list[str]) -> list[Path]:
    """Individual per-plane TIFFs: tiff/{t}/{region}_{fov}_0_{channel}.tiff, native dtype.

    Returns the written paths. These live in a tree that is a SIBLING of the plate, so a guard that
    measured only ``plate.ome.zarr`` would miss them entirely and under-reserve by ~2x."""
    n_t = image.shape[0]
    written: list[Path] = []
    for t in range(n_t):
        tdir = tiff_root / str(t)
        tdir.mkdir(parents=True, exist_ok=True)
        for c_i, channel in enumerate(channel_names):
            plane = image[t, c_i, 0]  # (Y, X), native dtype, z collapsed
            path = tdir / f"{region}_{fov}_0_{channel}.tiff"
            tifffile.imwrite(path, plane)
            written.append(path)
    return written


# --- honest partial output (IMA-230) ---------------------------------------------------------

def _truncate_plate(plate_dir: Path, written: dict[str, list[int]], intended: dict[str, list[int]],
                    n_fovs: int) -> None:
    """Rewrite the plate + well metadata to describe EXACTLY what is on disk.

    The full layout is declared up front (see :func:`write_from_stream`), because the writer needs no
    ordering logic when wells arrive in completion order. That declaration becomes a LIE the moment a
    run stops early: it names wells that hold no arrays. ndviewer_light dir-walks so it silently hides
    the gap and shows a plausible-looking partial plate; a reader that follows the declared layout
    breaks on the missing wells. Either way the operator cannot tell a truncated plate from a
    complete one, which is the failure this whole module-level guard exists to prevent.

    Truncation happens at BOTH levels. The stream is per FIELD ``(region, fov)`` and ``select_fovs``
    can return several FOVs per region, so a well can be half-written — plate-level truncation alone
    would leave a well group declaring FOVs with no arrays.

    ``intended`` is preserved under ``ome.squidmip``: :func:`plate_metadata` recomputes rows/columns
    from whatever region list it is handed, renumbering indices and destroying the record of what was
    MEANT to be written. Without it a resume cannot tell "never attempted" from "not in this plate".

    Ordered so the honesty flag rides on the rewrite itself rather than a separate file: a new inode
    is the write most likely to fail on a full disk, whereas this rewrite replaces a document with a
    SMALLER one in place and is the most likely to succeed.
    """
    regions = [r for r in intended if written.get(r)]

    # The up-front declaration also CREATED a directory for every intended well. Leaving those
    # behind would re-run the same lie one level down: ndviewer_light discovers wells by walking
    # directories, so an empty well dir shows up as a real (blank) well no matter what the metadata
    # says. Remove the wells that never got a field, then any row left with no wells.
    for region in intended:
        if written.get(region):
            continue
        try:
            row, col = parse_well_id(region)
        except ValueError:
            continue
        _shutil.rmtree(plate_dir / row / col, ignore_errors=True)
    for row_dir in [p for p in plate_dir.iterdir() if p.is_dir()] if plate_dir.exists() else []:
        if not any(p.is_dir() for p in row_dir.iterdir()):
            _shutil.rmtree(row_dir, ignore_errors=True)

    meta = plate_metadata(regions, field_count=n_fovs)
    meta["squidmip"] = {
        "truncated": True,
        "intended_regions": list(intended),
        "written_regions": regions,
        "n_fields_written": sum(len(v) for v in written.values()),
        "n_fields_intended": sum(len(v) for v in intended.values()),
    }
    write_group(plate_dir, meta)
    for region in regions:
        row, col = parse_well_id(region)
        write_group(
            plate_dir / row / col,
            {"version": _NGFF_VERSION,
             "well": {"images": [{"path": str(f)} for f in sorted(written[region])]}},
        )


def _discard_incomplete_fields(plate_dir: Path, written: dict[str, list[int]],
                               attempted: list[tuple[str, int]]) -> None:
    """Delete field directories that were started but never completed.

    The guard fires per SUBMIT, but one field is 100+ chunk writes across up to 6 pyramid levels, so
    a real out-of-space error lands mid-field. ``create_array`` opens with ``delete_existing=True``,
    which writes the array ``zarr.json`` BEFORE its chunks — so an interrupted field leaves a
    structurally valid, data-missing array that ndviewer will happily display as if it were real.
    That is "looks finished but silently isn't" reintroduced at field granularity, so the directory
    has to go rather than merely be undeclared.
    """
    for region, fov in attempted:
        if fov in written.get(region, []):
            continue
        try:
            row, col = parse_well_id(region)
        except ValueError:
            continue
        _shutil.rmtree(plate_dir / row / col / str(fov), ignore_errors=True)


# --- orchestration ---------------------------------------------------------------------------

def write_from_stream(
    metadata: dict,
    stream: Iterator[tuple[str, int, np.ndarray]],
    out_dir,
    *,
    n_fovs: int = 1,
    tiff: bool = False,
    on_well=None,
    write_workers: int = _WRITE_WORKERS,
    stop=None,
    regions=None,
    min_free_bytes: Optional[int] = None,
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

    ``min_free_bytes`` arms the IMA-230 storage guard: bytes to keep free on the output filesystem.
    ``None`` (default) disables it entirely, so existing callers are unchanged. When armed, the run
    stops before the submit that would breach the reserve and raises
    :class:`~squidmip._storage.InsufficientDiskSpace` — deliberately NOT via ``stop()``, which means
    "the operator cancelled" and is read that way by the viewer. A real out-of-space error that beats
    the guard is translated into the same exception. Either way the output is left honest: partial
    field directories are removed and the plate/well metadata is rewritten to exactly what survived.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    out_dir = Path(out_dir)
    plate_dir = out_dir / "plate.ome.zarr"
    tiff_root = out_dir / "tiff"

    wells = select_fovs(metadata, n_fovs=n_fovs)  # {region: [fov, ...]}, deterministic
    if regions is not None:   # subset: write only these wells (keep the requested order), for previews
        keep = list(dict.fromkeys(regions))
        wells = {r: wells[r] for r in keep if r in wells}

    # Full plate/row/well group metadata written UP FRONT (layout is fully known from metadata).
    write_group(plate_dir, plate_metadata(wells.keys(), field_count=n_fovs))
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

    def _write_one(region, fov, image):
        row, col = parse_well_id(region)
        # field directory is the RAW fov id (Squid convention), digit-named for ndviewer.
        levels, paths = _write_field(plate_dir / row / col / str(fov), image, channels,
                                     pixel_size_um, dz_um)
        if tiff:
            paths += _write_tiffs(tiff_root, region, fov, image, channel_names)
        # Price the field from the files it just wrote — exact, spans BOTH trees, and O(this field).
        written_bytes.record_field(paths)
        if on_well is not None:  # live consumer (plate viewer): render tile + push to ndviewer
            on_well(region, fov, image)
        return levels

    written_bytes = WrittenBytes()
    # Conservative stand-in until a field completes: for the first ``n_writers`` submits there is no
    # measurement yet, and a nearly-full disk is exactly when that window matters.
    bound_per_field = estimate_field_bytes(metadata, tiff=tiff)
    min_free = max(0, int(min_free_bytes or 0))

    def _headroom_shortfall(n_in_flight: int) -> Optional[tuple[int, int]]:
        """(free, needed) when the next submit would breach the reserve, else None.

        Reserves for the fields ALREADY committed to writer threads, not just the next one: the pool
        keeps up to ``n_writers`` in flight, so checking room for a single field would pass and then
        let several land at once and blow straight through. Deriving the count from ``pending``
        rather than a tuned constant keeps this correct if ``_WRITE_WORKERS`` or the core count
        changes.
        """
        if min_free <= 0 and not guard_enabled:
            return None
        per_field = written_bytes.per_field() or bound_per_field
        needed = min_free + per_field * (n_in_flight + 1)
        free = free_bytes(out_dir)
        return (free, needed) if free < needed else None

    guard_enabled = min_free_bytes is not None
    n_written = 0
    n_levels = 1
    n_writers = max(1, int(write_workers))
    written_fovs: dict[str, list[int]] = {}
    attempted: list[tuple[str, int]] = []
    shortfall: Optional[tuple[int, int]] = None
    out_of_space_exc: Optional[BaseException] = None

    def _harvest(futures):
        """Collect finished futures, recording which fields actually landed."""
        nonlocal n_written, n_levels, out_of_space_exc
        for f in futures:
            region, fov = pending[f]
            try:
                n_levels = f.result()          # re-raises a writer-thread exception here
            except BaseException as exc:       # noqa: BLE001 - re-raised below unless out-of-space
                if is_out_of_space(exc):
                    # The guard predicts; predictions lose to another process on the same disk, or
                    # to a field that compressed worse than its predecessors. Funnel it into the
                    # SAME typed failure so the operator gets one error and the same honest cleanup.
                    out_of_space_exc = exc
                    continue
                raise
            written_fovs.setdefault(region, []).append(fov)
            n_written += 1

    try:
        with ThreadPoolExecutor(max_workers=n_writers, thread_name_prefix="squidmip-write") as ex:
            pending: dict = {}   # future -> (region, fov), so the abort path knows what landed
            for region, fov, image in stream:
                if stop is not None and stop():
                    break
                if guard_enabled:
                    shortfall = _headroom_shortfall(len(pending))
                    if shortfall is not None:
                        break
                attempted.append((region, fov))
                pending[ex.submit(_write_one, region, fov, image)] = (region, fov)
                if len(pending) >= n_writers:    # keep <= n_writers wells in flight (bounded memory)
                    done, still = wait(set(pending), return_when=FIRST_COMPLETED)
                    _harvest(done)
                    pending = {f: pending[f] for f in still}
                    if out_of_space_exc is not None:
                        break
            _harvest(list(pending))              # drain the tail (and any in-flight after a stop)
            pending = {}
    finally:
        # Close the producer promptly on a stop/exception (don't wait for GC) so project_plate's
        # own thread pool shuts down now. Guarded: a plain iterator (used in tests) has no close().
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    if shortfall is not None or out_of_space_exc is not None:
        # Leave an artifact that tells the truth: drop half-written fields, then rewrite the plate
        # and well metadata to exactly what survived.
        _discard_incomplete_fields(plate_dir, written_fovs, attempted)
        _truncate_plate(plate_dir, written_fovs, wells, n_fovs)
        free, needed = shortfall or (free_bytes(out_dir), written_bytes.per_field() or bound_per_field)
        raise InsufficientDiskSpace(
            bytes_free=free,
            bytes_needed=needed,
            path=out_dir,
            fields_written=n_written,
            truncated=True,
            detail=None if shortfall is not None else f"writer failed: {out_of_space_exc}",
        ) from out_of_space_exc

    return {
        "plate": str(plate_dir),
        "tiff": str(tiff_root) if tiff else None,
        "n_wells": len(wells),
        "n_fields_written": n_written,
        "levels": n_levels,
        "truncated": False,
        "bytes_written": written_bytes.total,
    }


def write_plate(
    reader,
    out_dir,
    *,
    n_fovs: int = 1,
    workers: Optional[int] = None,
    projector: str = "mip",
    tiff: bool = False,
    on_well=None,
    write_workers: int = _WRITE_WORKERS,
    stop=None,
    on_error=None,
    regions=None,
    min_free_bytes: Optional[int] = None,
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
                             min_free_bytes=min_free_bytes)
