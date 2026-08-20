"""Squid's downsampled per-well mosaics (SAVE_DOWNSAMPLED_WELL_IMAGES): find, read, backfill.

Layout, written by Squid's unified mosaic widget (Cephla-Lab/Squid PR #418,
control/widgets_mosaic.py ``_write_per_well_tiffs``), one file per well per timepoint::

    <acquisition>/<t>/mosaic_view/wells/<well_id>_<N>um.tiff     # (C, H, W) or (H, W)

``N = int(round(pixel_size_um * factor))`` where ``factor = max(1, round(target_um /
pixel_size_um))``, Squid's integer plate-mode downsample (target defaults to 2 µm, so a
0.752 µm 10x set gets factor 3, 20x factor 6, 40x factor 12). The TIFF carries tifffile
shaped metadata: ``axes CYX``, ``PhysicalSizeX/Y`` (the file's own µm/px), ``Channel Name``
(the widget's layer names) and ``well_id``.

The reading rules here:

- The FACTOR IS DERIVED FROM THE FILE'S OWN SIZE against the fused mosaic extent
  (``fov_offsets_px`` + ``mosaic_extent_px``), never from the objective table: the file is
  ground truth about itself. A file whose two axes disagree about the factor is foreign or
  corrupt and reads as absent, with a named log line, never a crash, never wrong pixels.
- GEOMETRY: Squid blits each tile at ``round((tile_tl_mm - well_origin_mm) / eff_px)`` where
  the origin is the min tile top-left, so well-image pixel (0, 0) is the region's mosaic
  origin, the same origin every fused rung uses (positions are centres there and top-lefts
  here, but the half-frame shift cancels in the differences).
- A well image is ONE z (the widget keeps the LAST plane blitted per layer), so it may only
  stand in for an acquisition with a single z level. ``n_z > 1`` reads as absent, by design.
- A backfill through a PADDED reader (a stopped run explored at its planned state) bakes the
  padded FOVs in as black, so the file carries a ``padded_fovs`` stamp naming them. Once any
  stamped FOV has data on disk the file is a lie about the acquisition: ``load_well_stack``
  DELETES it — the one file this package deletes, provably ours (Squid never writes the stamp)
  and provably stale — and the next backfill rewrites it from the real pixels.

``SQUIDXPLORER_WELL_IMAGES=0`` turns the whole feature off (reading and backfill).
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from squidxplorer._logpane import get_logger

_log = get_logger("wellimage")

#: Env kill-switch for the whole feature; "0"/"false"/"no"/"off" disable it.
ENV_ENABLED = "SQUIDXPLORER_WELL_IMAGES"

#: Squid's MOSAIC_VIEW_TARGET_PIXEL_SIZE_UM default (control/_def.py). The save resolution is
#: coupled to the on-screen target there, and 2 µm is what every rig ships with.
TARGET_PIXEL_SIZE_UM = 2.0

#: ``<well_id>_<N>um.tiff``, Squid's per-well filename, N in whole micrometres.
_FILE_RE = re.compile(r"^(?P<well>[A-Za-z]+\d+)_(?P<um>\d+)um\.tiff?$", re.IGNORECASE)

#: Regions Squid's plate mode can save: well ids only, 1-2 row letters and a column >= 1
#: (its parse_well_id refuses column 0, so "manual0" is not a well).
_WELL_ID_RE = re.compile(r"^[A-Za-z]{1,2}[1-9]\d*$")

#: Slot-vs-extent tolerance in pixels: Squid rounds mm->px once per axis and its slot is sized
#: to the LARGEST well, so a matching file lands within a couple of pixels of extent/factor.
_DIM_TOL_PX = 2.0

#: Decoded well stacks kept in RAM; they are small (a few MB each).
_MAX_CACHED = 64


def enabled(env: Optional[dict] = None) -> bool:
    """Whether the well-image feature is on. Off is a supported state, not a broken one."""
    src = os.environ if env is None else env
    return str(src.get(ENV_ENABLED, "1")).strip().lower() not in ("0", "false", "no", "off")


def _norm_channel(name: Any) -> str:
    """One spelling for a channel name: Squid layer names use spaces, filenames underscores."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()


def acquisition_root(reader: Any) -> Optional[Path]:
    """The acquisition folder a reader reads, or ``None`` when it has no path identity."""
    path = getattr(reader, "source_id", None) or getattr(reader, "_path", None)
    if path is None:
        return None
    p = Path(str(path))
    return p if p.is_dir() else None


def _timepoint_dir(root: Path, time_point: int) -> Optional[Path]:
    """The folder Squid saved this timepoint's mosaic_view under (reader._time_folders rule)."""
    from squidxplorer.reader import _time_folders

    try:
        folders = _time_folders(Path(root))
    except OSError:
        return None
    t = int(time_point)
    return folders[t] if 0 <= t < len(folders) else None


def wells_dir(root, time_point: int = 0) -> Optional[Path]:
    """``<t>/mosaic_view/wells`` for this acquisition and timepoint; may not exist yet."""
    tdir = _timepoint_dir(Path(root), time_point)
    return None if tdir is None else tdir / "mosaic_view" / "wells"


def well_image_paths(root, region: str, time_point: int = 0) -> list[Path]:
    """Existing ``<region>_<N>um.tiff`` files for one well, finest (smallest N) first."""
    d = wells_dir(root, time_point)
    if d is None or not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir()):
        m = _FILE_RE.match(f.name)
        if m and m.group("well") == str(region) and f.is_file():
            out.append((int(m.group("um")), f))
    return [f for _n, f in sorted(out, key=lambda p: p[0])]


class WellStack(np.ndarray):
    """A decoded well image ``(C, H, W)`` carrying its ``factor`` and ``channel_names``."""

    def __new__(cls, array, factor: int, channel_names: Sequence[str]):
        obj = np.asarray(array).view(cls)
        obj.factor = int(factor)
        obj.channel_names = [str(n) for n in channel_names]
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.factor = getattr(obj, "factor", 1)
            self.channel_names = getattr(obj, "channel_names", [])

    def channel_plane(self, channel: str) -> Optional[np.ndarray]:
        """The 2-D plane for *channel* (normalised-name match), or ``None``."""
        want = _norm_channel(channel)
        for i, name in enumerate(self.channel_names):
            if _norm_channel(name) == want:
                return np.asarray(self[i])
        return None


# One small process-wide cache of decoded stacks; entries are a few MB each.
_STACKS: "OrderedDict[tuple, Optional[WellStack]]" = OrderedDict()
_STACKS_LOCK = threading.Lock()


def clear_cache() -> None:
    """Drop the decoded-stack cache (tests, and a backfill that just rewrote the files)."""
    with _STACKS_LOCK:
        _STACKS.clear()


def _derive_factor(meta: dict, region: str, shape_hw: tuple) -> Optional[int]:
    """The integer downsample factor THIS FILE has, from its size vs the fused mosaic extent.

    Never from the objective table: the file is ground truth about itself. ``None`` when the
    two axes disagree or the size fits no integer factor, a foreign or stale file.
    """
    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    px = meta.get("pixel_size_um")
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not positions or px in (None, 0) or not fovs:
        return None
    try:
        offsets = fov_offsets_px(positions, region, fovs, px)
        full_h, full_w = mosaic_extent_px(offsets, tuple(int(v) for v in meta["frame_shape"]))
    except (KeyError, ValueError):
        return None

    ih, iw = int(shape_hw[0]), int(shape_hw[1])
    if ih < 1 or iw < 1:
        return None
    f = int(round(full_w / iw))
    if f < 1 or int(round(full_h / ih)) != f:
        return None
    # The slot may pad a couple of px past extent/factor (Squid rounds mm once per axis and
    # sizes the slot to the largest well); more than that is not this acquisition's file.
    if abs(ih - full_h / f) > _DIM_TOL_PX or abs(iw - full_w / f) > _DIM_TOL_PX:
        return None
    return f


def _read_stack(path: Path, meta: dict) -> tuple[Optional[np.ndarray], Optional[list], list]:
    """``(array (C, H, W), channel_names_or_None, padded_fovs_stamp)`` off one TIFF.

    Raises on a corrupt file. The stamp is the backfill's record of the FOVs it wrote as
    black through a padded reader; empty on Squid's own files.
    """
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        arr = tif.series[0].asarray()
        names = None
        stamp: list = []
        shaped = getattr(tif, "shaped_metadata", None)
        if shaped:
            channel = shaped[0].get("Channel") or {}
            got = channel.get("Name")
            if isinstance(got, (list, tuple)):
                names = [str(n) for n in got]
            elif got is not None:
                names = [str(got)]
            raw = shaped[0].get("padded_fovs")
            if isinstance(raw, (list, tuple)):
                stamp = [int(f) for f in raw]
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"rank {arr.ndim} is not a (C, H, W) well image")
    if names is None or len(names) != arr.shape[0]:
        meta_names = [c["name"] for c in meta.get("channels", [])]
        if len(meta_names) == arr.shape[0]:
            names = meta_names          # positional: the widget saved every layer, in order
        else:
            raise ValueError(
                f"{arr.shape[0]} plane(s) but no usable channel names "
                f"(metadata named {names}, the acquisition has {len(meta_names)})"
            )
    return arr, names, stamp


def _arrived_padded_fovs(reader: Any, meta: dict, region: str, stamp: list) -> list[int]:
    """The stamped-black FOVs that NOW have data on disk; non-empty means the file is stale.

    A padded reader's ``fovs_per_region`` is the PLAN, so presence is "in the plan and not in
    the reader's own padded set"; an unpadded reader lists only on-disk FOVs and needs no set.
    """
    slots = getattr(reader, "padded_slots", None)
    still_missing = set((slots.fovs if slots else {}).get(str(region), ()))
    have = set((meta.get("fovs_per_region") or {}).get(region) or ())
    return sorted(int(f) for f in stamp if f in have and f not in still_missing)


def load_well_stack(reader: Any, meta: dict, region: str,
                    time_point: int = 0) -> Optional[WellStack]:
    """This well's downsampled mosaic at *time_point*, or ``None`` (reason logged, never raised).

    ``None`` covers: feature off, no folder, ``n_z > 1`` (the saved image is the widget's LAST
    z, and serving it under another z would be the wrong-image failure), a corrupt file, or a
    file whose size fits no integer factor of this acquisition's geometry.
    """
    if not enabled():
        return None
    if int(meta.get("n_z") or 1) > 1:
        return None                     # one saved z cannot stand in for a stack
    root = acquisition_root(reader)
    if root is None:
        return None

    # The reader's own padded set is part of the key: a same-session reopen after the
    # acquisition completed must re-read the file so the stale pad-derived check runs.
    slots = getattr(reader, "padded_slots", None)
    key = (str(root), str(region), int(time_point),
           frozenset((slots.fovs if slots else {}).get(str(region), ())))
    with _STACKS_LOCK:
        if key in _STACKS:
            _STACKS.move_to_end(key)
            return _STACKS[key]

    stack = None
    for path in well_image_paths(root, region, time_point):
        try:
            arr, names, stamp = _read_stack(path, meta)
        except Exception as exc:        # noqa: BLE001 - a bad file falls back to fusing, named
            _log.warning("well image %s is unreadable (%s: %s); falling back to fusing "
                         "this well from its FOVs.", path, type(exc).__name__, exc)
            continue
        if stamp:
            arrived = _arrived_padded_fovs(reader, meta, region, stamp)
            if arrived:
                try:
                    path.unlink()
                    _log.info("well image %s was backfilled from a padded (stopped) run and "
                              "%d of its black FOV(s) now have data; deleted — this well "
                              "fuses fresh and the next backfill rewrites the file.",
                              path, len(arrived))
                except OSError as exc:
                    _log.warning("well image %s is a stale pad-derived mosaic (FOV(s) %s now "
                                 "have data) but could not be deleted (%s); ignoring it "
                                 "rather than serving black over real pixels.",
                                 path, arrived, exc)
                continue
        factor = _derive_factor(meta, region, arr.shape[-2:])
        if factor is None:
            _log.warning("well image %s is %sx%s px, which fits no integer downsample of this "
                         "acquisition's %s geometry; falling back to fusing this well.",
                         path, arr.shape[-1], arr.shape[-2], region)
            continue
        stack = WellStack(arr, factor, names)
        break

    with _STACKS_LOCK:
        _STACKS[key] = stack            # negative results cached too: one directory scan per well
        while len(_STACKS) > _MAX_CACHED:
            _STACKS.popitem(last=False)
    return stack


def downsampled_well(reader: Any, meta: dict, region: str, channel: str,
                     time_point: int = 0) -> Optional[tuple[np.ndarray, int]]:
    """``(plane (H, W), factor)`` for one well/channel, or ``None`` (absent / unusable, logged)."""
    stack = load_well_stack(reader, meta, region, time_point)
    if stack is None:
        return None
    plane = stack.channel_plane(channel)
    if plane is None:
        _log.info("well image for %s carries channels %s, not %r; fusing that channel.",
                  region, stack.channel_names, channel)
        return None
    return plane, stack.factor


def resample_plane(plane: np.ndarray, factor: int, step: int,
                   out_h: int, out_w: int) -> np.ndarray:
    """Map a factor-µm well plane onto a step-µm pyramid level, same origin, top-left samples.

    A fused rung strides ``frame[::step]``, a top-left sample, so the level pixel ``r`` maps
    to full-resolution row ``r * step``, which lives in well-image row ``(r * step) // factor``.
    """
    ri = np.minimum((np.arange(int(out_h)) * int(step)) // int(factor), plane.shape[0] - 1)
    ci = np.minimum((np.arange(int(out_w)) * int(step)) // int(factor), plane.shape[1] - 1)
    return np.ascontiguousarray(plane[ri][:, ci])


# --- the backfill: write what Squid would have written -------------------------------------


def downsample_plane(plane: np.ndarray, factor: int) -> np.ndarray:
    """Integer-factor area downsample of one plane.

    Vendored from Squid ``control/core/mosaic_utils.downsample_tile`` (Cephla-Lab/Squid
    PR #418) so SquidXplorer can backfill without cv2: output dims are its ``dim // factor``,
    and a factor-sized block mean equals ``cv2.INTER_AREA`` at integer factors (Squid's
    INTER_AREA_FAST pyrDown chain approximates the same mean). Squid will grow a standalone
    downsampler later; this stays the minimal read-side copy until then.
    """
    f = int(factor)
    if f <= 1:
        return plane
    h, w = plane.shape[0] // f, plane.shape[1] // f
    if h < 1 or w < 1:
        return plane
    crop = plane[: h * f, : w * f].astype(np.float32)
    out = crop.reshape(h, f, w, f).mean(axis=(1, 3))
    return out.astype(plane.dtype)


def write_well_images(reader: Any, meta: dict, *, time_point: int = 0,
                      z_level: Optional[int] = None,
                      should_stop: Optional[Callable[[], bool]] = None,
                      target_um: float = TARGET_PIXEL_SIZE_UM) -> int:
    """Backfill ``<t>/mosaic_view/wells`` with what Squid's widget would have written.

    Same filenames, same (C, H, W) TIFF metadata, same integer factor and slot geometry, so
    the acquisition afterwards looks microscope-produced. The default z is the LAST level,
    the plane Squid's widget keeps, since every z blits over the same slot. A well that
    already has a file — at ANY resolution, Squid's own or an earlier backfill's — is left
    alone. A PADDED reader taints what it writes: a well whose FOVs are all padded (or a
    padded z/t plane) gets no file, and a partially padded well's file is stamped with
    ``padded_fovs`` so ``load_well_stack`` can delete it once the data arrives. Best-effort:
    a read-only mount or a bad region is logged and skipped, never raised. Returns the number
    of well files written. Call it off the Qt thread; every FOV of the timepoint is decoded.
    """
    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    if not enabled():
        return 0
    root = acquisition_root(reader)
    positions = meta.get("fov_positions_um") or {}
    px = meta.get("pixel_size_um")
    if root is None or not positions or px in (None, 0):
        _log.info("well-image backfill skipped: no acquisition path or no stage geometry.")
        return 0
    tdir = _timepoint_dir(root, time_point)
    if tdir is None:
        return 0

    factor = max(1, int(round(float(target_um) / float(px))))
    res_tag = f"{int(round(float(px) * factor))}um"
    channels = [c["name"] for c in meta.get("channels", [])]
    dtype = np.dtype(meta.get("dtype", "uint16"))
    frame_h, frame_w = (int(v) for v in meta["frame_shape"])
    nz = int(meta.get("n_z") or 1)
    z = (nz - 1) if z_level is None else int(z_level)

    slots = getattr(reader, "padded_slots", None)
    if slots and (z in slots.z_levels or int(time_point) in slots.time_points):
        _log.info("well-image backfill skipped: z=%d at t=%d is itself padded — there is no "
                  "data at that plane to write.", z, time_point)
        return 0
    padded_by_region = dict(slots.fovs) if slots else {}

    # Squid sizes ONE slot for the whole plate: the largest well's extent, floored at a tile.
    per_region: dict[str, tuple] = {}
    slot_h = max(1, int(round(frame_h / factor)))
    slot_w = max(1, int(round(frame_w / factor)))
    for region in meta.get("regions", []):
        if not _WELL_ID_RE.match(str(region)):
            _log.info("well-image backfill: region %r is not a well id; skipping it, as Squid "
                      "would.", region)
            continue
        fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
        if not fovs:
            continue
        if set(fovs) <= set(padded_by_region.get(str(region), ())):
            _log.info("well-image backfill: %s has no data at all (every FOV is padded); "
                      "no file.", region)
            continue
        try:
            offsets = fov_offsets_px(positions, region, fovs, px)
            full_h, full_w = mosaic_extent_px(offsets, (frame_h, frame_w))
        except (KeyError, ValueError) as exc:
            _log.info("well-image backfill: %s has no placeable geometry (%s); skipping it.",
                      region, exc)
            continue
        per_region[str(region)] = (fovs, offsets)
        slot_h = max(slot_h, int(round(full_h / factor)))
        slot_w = max(slot_w, int(round(full_w / factor)))
    if not per_region:
        return 0

    wells = tdir / "mosaic_view" / "wells"
    try:
        # The empty directory is part of the contract: Squid creates it before the first file.
        wells.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log.info("well-image backfill skipped: cannot create %s (%s); a read-only "
                  "acquisition stays untouched.", wells, exc)
        return 0

    import tifffile

    try:
        # A well that has ANY file — Squid's own at another resolution included — is theirs.
        existing = {m.group("well") for f in wells.iterdir()
                    for m in (_FILE_RE.match(f.name),) if m}
    except OSError:
        existing = set()

    written = 0
    for region, (fovs, offsets) in per_region.items():
        if should_stop is not None and should_stop():
            break
        path = wells / f"{region}_{res_tag}.tiff"
        if region in existing or path.exists():
            continue
        canvas = np.zeros((len(channels), slot_h, slot_w), dtype=dtype)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            for fov in fovs:
                row, col = offsets[fov]
                r0 = int(round(row / factor))
                c0 = int(round(col / factor))
                for ci, ch in enumerate(channels):
                    tile = downsample_plane(np.asarray(reader.read(region, fov, ch, z,
                                                                   time_point)), factor)
                    r1 = min(r0 + tile.shape[0], slot_h)
                    c1 = min(c0 + tile.shape[1], slot_w)
                    if r1 > r0 and c1 > c0:
                        canvas[ci, r0:r1, c0:c1] = tile[: r1 - r0, : c1 - c0]
            md = {
                "axes": "CYX",
                "PhysicalSizeX": float(px) * factor,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": float(px) * factor,
                "PhysicalSizeYUnit": "µm",
                "Channel": {"Name": channels},
                "well_id": region,
            }
            missing = sorted(set(fovs) & set(padded_by_region.get(region, ())))
            if missing:
                # The stamp is the delete licence: load_well_stack removes this file once
                # any of these FOVs has data on disk.
                md["padded_fovs"] = [int(f) for f in missing]
            tifffile.imwrite(tmp, canvas, photometric="minisblack", metadata=md)
            os.replace(tmp, path)       # atomic publish, like every other writer in this repo
            written += 1
        except Exception as exc:        # noqa: BLE001 - one bad well must not lose the rest
            _log.warning("well-image backfill: %s could not be written (%s: %s); continuing.",
                         region, type(exc).__name__, exc)
            try:
                tmp.unlink(missing_ok=True)     # noqa: SIM105
            except OSError:
                pass
    if written:
        clear_cache()                   # decoded absences are stale now
        _log.info("well-image backfill: wrote %d per-well mosaic(s) at %s into %s.",
                  written, res_tag, wells)
    return written
