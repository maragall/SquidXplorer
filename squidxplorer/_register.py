"""Register a well's FOVs and write the solved positions out — the stitcher's solve, no fusion.

``register`` is a region operator over the stitcher's own :func:`_stitch.solve_offsets_px`
(untouched): it solves per-FOV offsets from the overlaps, shows a decimated paste at the
corrected positions (a LOOK through the one placement rule, later-overwrites-earlier — not the
fused mosaic of record), and with ``copy=True`` writes ``stitched_<folder>`` beside the
acquisition: a copy whose image files are HARDLINKED (a second directory entry for the same
bytes — no privileges needed, nothing to dangle, indistinguishable from a regular file to every
tool; per-file copy fallback where a filesystem refuses) and whose coordinates.csv carries the
registered positions. Sidecars (csv/yaml/json/log/txt) are always REAL copies, so rewriting one
can never write through a link into the source; the source acquisition is never written.

The elastic distortion residual is fusion-only (a per-seam warp cannot ride in one translation
per FOV), so the copy carries the global solve exactly and nothing else.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import threading
from pathlib import Path

import numpy as np

from squidxplorer._engine import Param, add_region_operator
from squidxplorer._mosaic_source import _MAX_FUSED_PX
from squidxplorer._placement import Placement, PlacedArray, fov_offsets_px, mosaic_extent_px
from squidxplorer._stitch import _NullTimer, _pixel_size, _positions_yx_um, solve_offsets_px
from squidxplorer.reader import _coord_columns, _fov_column

_log = logging.getLogger(__name__)

#: Sidecars are always real copies: some are rewritten later, and a rewrite through a hardlink
#: would edit the source acquisition.
_SIDECAR_SUFFIXES = {".csv", ".yaml", ".yml", ".json", ".log", ".txt"}

_MM_PER_UM = 1e-3

#: One writer at a time per process: a bulk run updates one copy's csvs region by region, and
#: the region loop may hold several regions in flight.
_COPY_LOCK = threading.Lock()


# -----------------------------------------------------------------------------------------
# the registered copy
# -----------------------------------------------------------------------------------------

def registered_copy_root(src) -> Path:
    """``stitched_<folder>`` beside the acquisition."""
    src = Path(src)
    return src.parent / f"stitched_{src.name}"


def link_or_copy(src, dst):
    """A second directory entry for *src*'s bytes at *dst*; a real copy where the filesystem
    refuses (cross-volume, exFAT, SMB). Returns the refusal (``OSError``) or ``None`` (linked)."""
    try:
        os.link(src, dst)
        return None
    except OSError as exc:
        shutil.copy2(src, dst)
        return exc


def ensure_registered_copy(src) -> tuple[Path, int, int]:
    """The copy, created if absent: images hardlinked (copy fallback), sidecars real copies.

    Built under a ``.partial`` name and renamed whole, so a killed run never leaves something
    that reads as a finished copy. Returns ``(root, n_linked, n_copied)``; ``(root, 0, 0)``
    when it already existed.
    """
    src = Path(src)
    if not src.is_dir():
        raise ValueError(f"cannot copy {src}: not a directory.")
    dst = registered_copy_root(src)
    with _COPY_LOCK:
        if dst.exists():
            return dst, 0, 0
        tmp = dst.with_name(dst.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)             # a dead run's leftovers; links never reach the source
        linked = copied = 0
        first_refusal = None
        for dirpath, _dirnames, filenames in os.walk(src):
            rel = Path(dirpath).relative_to(src)
            (tmp / rel).mkdir(parents=True, exist_ok=True)
            for name in filenames:
                s, d = Path(dirpath) / name, tmp / rel / name
                if s.suffix.lower() in _SIDECAR_SUFFIXES:
                    shutil.copy2(s, d)
                    copied += 1
                    continue
                refusal = link_or_copy(s, d)
                if refusal is None:
                    linked += 1
                else:
                    copied += 1
                    if first_refusal is None:
                        first_refusal = refusal
        os.rename(tmp, dst)
    if first_refusal is not None:
        _log.info("registered copy: this filesystem refuses hardlinks (%s); image files were "
                  "copied in full.", first_refusal)
    _log.info("registered copy: %s - %d file(s) hardlinked, %d copied.", dst.name, linked, copied)
    return dst, linked, copied


def write_registered_rows(root, region: str, positions_um: dict) -> int:
    """Rewrite *region*'s x/y cells (mm) in every coordinates.csv under *root*.

    ``positions_um`` is ``{fov: (x_um, y_um)}``. Rows of other regions are untouched; a
    region's repeated rows (one per z/t) all move. Returns rows rewritten; refuses when no csv
    carried the region — the copy would silently keep its stage positions.
    """
    root = Path(root)
    total = 0
    with _COPY_LOCK:
        for path in sorted(root.rglob("coordinates.csv")):
            total += _rewrite_csv(path, region, positions_um)
    if total == 0:
        raise ValueError(
            f"no coordinates.csv under {root} has rows for region {region!r}; the registered "
            "copy would silently keep its stage positions.")
    return total


def _rewrite_csv(path: Path, region: str, positions_um: dict) -> int:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0
    header = rows[0]
    x_col, y_col = _coord_columns(header)
    xi, yi = header.index(x_col), header.index(y_col)
    fov_col = _fov_column(header)
    fi = header.index(fov_col) if fov_col else None
    try:
        ri = header.index("region")
    except ValueError:
        return 0
    changed = 0
    # Without a fov column, row order per DISTINCT position is the id (the reader's own rule:
    # repeated rows are one per z/t at the same position). Keyed on the original cell text.
    seen_keys: dict = {}
    for row in rows[1:]:
        if len(row) <= max(xi, yi, ri):
            continue
        if (row[ri] or "").strip() != region:
            continue
        if fi is not None:
            try:
                fov = int((row[fi] or "").strip())
            except ValueError:
                continue
        else:
            key = ((row[xi] or "").strip(), (row[yi] or "").strip())
            if key not in seen_keys:
                seen_keys[key] = len(seen_keys)
            fov = seen_keys[key]
        if fov not in positions_um:
            continue
        x_um, y_um = positions_um[fov]
        # Shortest round-trip repr, never rounded: a rounded copy of an unmoved stage point
        # reads as a 0.1 nm neighbour of its unrewritten siblings and poisons pitch derivations.
        row[xi] = repr(float(x_um * _MM_PER_UM))
        row[yi] = repr(float(y_um * _MM_PER_UM))
        changed += 1
    if not changed:
        return 0
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    os.replace(tmp, path)   # never in-place: an in-place write through a link edits the source
    return changed


# -----------------------------------------------------------------------------------------
# the operator
# -----------------------------------------------------------------------------------------

def register_region(reader, region, fovs, *, registration_channel=0, registration_t=0,
                    copy=False, timer=None) -> PlacedArray:
    """Solve one region's per-FOV offsets and return a decimated paste at the corrected
    positions. ``copy=True`` also writes/updates ``stitched_<folder>``'s coordinates.csv."""
    meta = reader.metadata
    all_channels = [c["name"] for c in meta["channels"]]
    if isinstance(registration_channel, str) and registration_channel in all_channels:
        reg_c = all_channels.index(registration_channel)
    else:
        reg_c = int(registration_channel)
    if not 0 <= reg_c < len(all_channels):
        raise ValueError(
            f"registration_channel {registration_channel!r} is not one of {all_channels} "
            f"(or an index below {len(all_channels)}).")
    fovs = list(fovs)
    positions = _positions_yx_um(meta, region, fovs)
    pixel_size = _pixel_size(meta)
    tile_shape = tuple(int(v) for v in meta["frame_shape"])
    reg_z = int(meta["n_z"]) // 2                     # parity with the stitcher's solve
    reg_t = int(registration_t)

    tiles = np.stack([np.asarray(reader.read(region, f, all_channels[reg_c], reg_z, reg_t))
                      for f in fovs])[:, None]
    # A tile with no pixels is not a measurement: a PADDED slot of a stopped run reads as
    # zeros, and solving it hands the affine fallback hundreds of phantoms whose "registered
    # positions" are model outputs. Solve and rewrite the content-bearing FOVs only; blanks
    # keep their recorded positions.
    has_pixels = [bool(tiles[i].any()) for i in range(len(fovs))]
    measured = [i for i, ok in enumerate(has_pixels) if ok]
    if not measured:
        raise ValueError(
            f"region {region!r}: every registration tile is blank at z={reg_z}, t={reg_t} "
            f"(channel {all_channels[reg_c]}); there is nothing to register.")
    offsets = np.zeros((len(fovs), 2), dtype=np.float64)
    if len(measured) > 1:
        sub = solve_offsets_px(tiles[measured], [positions[i] for i in measured],
                               pixel_size, tile_shape,
                               registration_channel=0, timer=timer or _NullTimer())
        offsets[measured] = sub
    if len(measured) < len(fovs):
        _log.info("register: region %s - %d of %d FOV(s) carry pixels; the %d blank (padded) "
                  "FOV(s) keep their recorded positions.",
                  region, len(measured), len(fovs), len(fovs) - len(measured))
    del tiles

    registered = [(y + float(o[0]) * pixel_size[0], x + float(o[1]) * pixel_size[1])
                  for (y, x), o in zip(positions, offsets)]

    if len(measured) > 1:
        # SAY the corrections: a good stage solves to a few um, which moves tiles by
        # fractions of a frame — without the numbers a working solve reads as a no-op in
        # the decimated whole-region preview (measured on the 900-FOV 20x tissue: median
        # 5 px on a 1900 px frame, pasted at step 6).
        mags_um = [float(np.hypot(offsets[i][0] * pixel_size[0],
                                  offsets[i][1] * pixel_size[1])) for i in measured]
        _log.info("register: region %s - %d FOV(s) solved; corrections median %.2f um, "
                  "max %.2f um (%.1f px max on a %d px frame).",
                  region, len(measured), float(np.median(mags_um)), max(mags_um),
                  max(float(np.hypot(*offsets[i])) for i in measured), tile_shape[0])

    if copy:
        src = getattr(reader, "source_id", None)
        if not src or not Path(str(src)).is_dir():
            raise ValueError(
                f"copy=True needs an on-disk acquisition to copy, and this reader's source "
                f"({src!r}) is not a directory.")
        dst, linked, copied = ensure_registered_copy(Path(str(src)))
        rows = write_registered_rows(
            dst, region, {fovs[i]: (registered[i][1], registered[i][0]) for i in measured})
        _log.info("register: %s - %d row(s) of region %s now carry the registered positions%s.",
                  dst.name, rows, region,
                  f" ({linked} file(s) hardlinked, {copied} copied)" if linked or copied else "")

    # The LOOK: a decimated paste at the corrected positions, through the one placement rule.
    pos_map = {(region, f): (x, y) for f, (y, x) in zip(fovs, registered)}
    px = float(pixel_size[0])
    off_px = fov_offsets_px(pos_map, region, fovs, px)
    full_h, full_w = mosaic_extent_px(off_px, tile_shape)
    step = max(1, int(np.ceil(max(full_h, full_w) / float(_MAX_FUSED_PX))))
    out_h = int(np.ceil(full_h / step))
    out_w = int(np.ceil(full_w / step))
    out = np.zeros((1, len(all_channels), 1, out_h, out_w), np.dtype(meta["dtype"]))
    for ci, ch in enumerate(all_channels):
        for f in fovs:
            plane = np.asarray(reader.read(region, f, ch, reg_z, reg_t))[::step, ::step]
            r0, c0 = off_px[f][0] // step, off_px[f][1] // step
            r1, c1 = min(out_h, r0 + plane.shape[0]), min(out_w, c0 + plane.shape[1])
            out[0, ci, 0, r0:r1, c0:c1] = plane[: r1 - r0, : c1 - c0]

    placement = Placement(
        origin_um=(min(y for y, _ in registered), min(x for _, x in registered)),
        pixel_size_um=px * step,                      # the PASTE's pitch, so bbox_um is exact
        z_step_um=meta.get("dz_um"),
        shape=(out_h, out_w),
        tile_shape=(int(np.ceil(tile_shape[0] / step)), int(np.ceil(tile_shape[1] / step))),
        fovs=tuple(fovs),
        offsets_px=tuple((float(o[0]) / step, float(o[1]) / step) for o in offsets),
        origins_px=tuple((off_px[f][0] / step, off_px[f][1] / step) for f in fovs),
        reg_channel=all_channels[reg_c],
        reg_t=reg_t,
        reg_z=reg_z,
    )
    return PlacedArray(out, placement)


_REGISTER_PARAMS = (
    Param("registration_channel", 0,
          "the channel the pose graph is solved on, by index or name"),
    Param("registration_t", 0, "the timepoint the pose graph is solved on"),
)


def _register_factory(**params):
    """The registered object: returns the region operator at the declared parameters."""
    def register_fovs(reader, region, fovs, *, copy=False, timer=None):
        # Explicit keywords ON PURPOSE: a **kwargs here would read as "takes flatfield" to the
        # region loop's _accepts_kwarg probe and buy every run a plate-wide BaSiC estimate.
        return register_region(reader, region, fovs, copy=copy, timer=timer, **params)
    return register_fovs


add_region_operator("register", _register_factory, params=_REGISTER_PARAMS,
                    requires=("tilefusion",), extra="stitch",
                    accepts=("copy", "timer"))
