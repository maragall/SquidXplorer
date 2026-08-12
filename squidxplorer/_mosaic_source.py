"""Fused mosaics for the napari pane — the unit displayed is a MOSAIC, never a single FOV.

Written OME-Zarr is read lazily as a dask pyramid; a raw acquisition is fused by pasting
frames at their stage-position offsets.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from squidxplorer._budget import cache_budget
from squidxplorer._logpane import get_logger

# Carries no Qt, so a fuser can report unreadable FOVs in a headless run.
_log = get_logger("mosaic")


def level_paths(group: Path) -> list[Path]:
    """Every resolution level of an OME-NGFF image group, highest resolution first."""
    group = Path(group)
    doc = json.loads((group / "zarr.json").read_text())
    attrs = doc.get("attributes", {})
    ome = attrs.get("ome", attrs)
    multiscales = ome.get("multiscales") or []
    if not multiscales:
        raise ValueError(f"{group}: no 'multiscales' metadata; not an OME-NGFF image group.")
    datasets = multiscales[0].get("datasets") or []
    if not datasets:
        raise ValueError(f"{group}: multiscales carries no 'datasets' (no resolution levels).")
    return [group / str(d["path"]) for d in datasets]


def open_pyramid(group, *, t: int = 0, c: int = 0, z: int = 0) -> list:
    """Lazy 2-D dask pyramid for one (t, c, z) of a written field/mosaic group."""
    import dask.array as da
    import zarr

    out = []
    for path in level_paths(group):
        arr = zarr.open_array(str(path), mode="r")
        d = da.from_array(arr, chunks=arr.chunks)
        # Squid canonical order is (t, c, z, y, x); 2-D/3-D stores are legal NGFF too.
        if d.ndim == 5:
            d = d[t, c, z]
        elif d.ndim == 4:
            d = d[t, c]
        elif d.ndim == 3:
            d = d[z]
        elif d.ndim != 2:
            raise ValueError(f"{path}: unsupported rank {d.ndim} for a mosaic plane.")
        out.append(d)

    return strictly_decreasing_levels(out)


def strictly_decreasing_levels(levels: list) -> list:
    """Drop any level that does not shrink in BOTH displayed axes. The one pyramid guard."""
    if not levels:
        raise ValueError("a pyramid needs at least one level; got none.")
    kept = [levels[0]]
    for d in levels[1:]:
        if d.shape[-2] < kept[-1].shape[-2] and d.shape[-1] < kept[-1].shape[-1]:
            kept.append(d)
    return kept


# Bound on a fused raw mosaic materialised in RAM; frames are decimated on read beyond it.
_MAX_FUSED_PX = 8192


def fuse_region_mosaic(
    reader: Any,
    meta: dict,
    region: str,
    channel: str,
    *,
    z: int = 0,
    t: int = 0,
    max_px: int = _MAX_FUSED_PX,
) -> Optional[tuple[np.ndarray, float]]:
    """Paste a region's FOVs into ONE plane, placed by stage position.

    Returns ``(mosaic, step)`` where ``step`` is the decimation factor, or ``None`` when the
    acquisition carries no stage positions / no pixel size. An unreadable FOV stays a zeroed
    hole and is logged by name.
    """
    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    pixel_size = meta.get("pixel_size_um")
    if not positions or pixel_size in (None, 0):
        return None

    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        return None

    frame_h, frame_w = (int(v) for v in meta["frame_shape"])
    try:
        offsets = fov_offsets_px(positions, region, fovs, pixel_size)
        full_h, full_w = mosaic_extent_px(offsets, (frame_h, frame_w))
    except (KeyError, ValueError):
        return None

    step = max(1, int(np.ceil(max(full_h, full_w) / float(max_px))))
    out_h = int(np.ceil(full_h / step))
    out_w = int(np.ceil(full_w / step))

    dtype = np.dtype(meta.get("dtype", "uint16"))
    mosaic = np.zeros((out_h, out_w), dtype=dtype)

    for fov in fovs:
        row, col = offsets[fov]
        try:
            frame = reader.read(region, fov, channel, z, t)
        except Exception as exc:                 # noqa: BLE001 - one bad FOV must not lose a well
            _log.warning("mosaic %s/%s: FOV %s could not be read (%s: %s) — it is a BLACK HOLE in "
                         "this mosaic, not an empty field.", region, channel, fov,
                         type(exc).__name__, exc)
            continue          # leave zeros: the hole stays put, and the line above says it is one
        if frame is None:
            _log.warning("mosaic %s/%s: FOV %s read as None — it is a BLACK HOLE in this mosaic, "
                         "not an empty field.", region, channel, fov)
            continue
        frame = np.asarray(frame)
        if frame.ndim != 2:
            frame = frame.reshape(frame.shape[-2:])
        sub = frame[::step, ::step]
        r0, c0 = row // step, col // step
        r1, c1 = min(r0 + sub.shape[0], out_h), min(c0 + sub.shape[1], out_w)
        if r1 > r0 and c1 > c0:
            # Later FOVs overwrite earlier ones in the overlap: a preview placement, not a stitch.
            mosaic[r0:r1, c0:c1] = sub[: r1 - r0, : c1 - c0]

    return mosaic, float(step)


#: Refuse to materialise more than this in one slice request.
_PLANE_BUDGET_BYTES = 2 * 1024 ** 3


def _planned_plane(meta: dict, region: str, max_px: int):
    """``(out_h, out_w, step, dtype)`` a fused plane WOULD have. Pure geometry, reads nothing."""
    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    pixel_size = meta.get("pixel_size_um")
    if not positions or pixel_size in (None, 0):
        return None
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        return None
    try:
        offsets = fov_offsets_px(positions, region, fovs, pixel_size)
        full_h, full_w = mosaic_extent_px(offsets, tuple(int(v) for v in meta["frame_shape"]))
    except (KeyError, ValueError):
        return None

    step = max(1, int(np.ceil(max(full_h, full_w) / float(max_px))))
    return (int(np.ceil(full_h / step)), int(np.ceil(full_w / step)),
            float(step), np.dtype(meta.get("dtype", "uint16")))


def fuse_region_stack(
    reader: Any,
    meta: dict,
    region: str,
    channel: str,
    *,
    t: int = 0,
    max_px: int = _MAX_FUSED_PX,
):
    """A LAZY ``(z, y, x)`` mosaic stack — one fused plane materialised per visible z."""
    import dask.array as da

    nz = int(meta.get("n_z") or 1)

    # Size the plane from geometry, before anything is allocated.
    planned = _planned_plane(meta, region, max_px)
    if planned is None:
        return None
    h, w, step, dtype = planned

    # The budget is per plane, not for the stack: only the visible z is ever in RAM.
    per_plane = h * w * dtype.itemsize
    if per_plane > _PLANE_BUDGET_BYTES:
        raise MemoryError(
            f"{region}/{channel}: one fused plane is {per_plane / 1e9:.1f} GB "
            f"({h}x{w} {dtype}), over the {_PLANE_BUDGET_BYTES / 1e9:.1f} GB plane budget. "
            "Lower max_px rather than letting this page the machine."
        )

    if nz <= 1:
        # No singleton z axis: return the plane so napari draws no one-position slider.
        probe = fuse_region_mosaic(reader, meta, region, channel, z=0, t=t, max_px=max_px)
        if probe is None:
            return None
        return probe[0], probe[1], 1

    # One-plane cache: the same z is sliced twice (contrast sample + napari draw) per region change.
    _cache: dict = {"z": None, "plane": None}
    _cache_lock = __import__("threading").Lock()

    def _plane(z: int):
        z = int(z)
        with _cache_lock:
            if _cache["z"] == z and _cache["plane"] is not None:
                return _cache["plane"]
        got = fuse_region_mosaic(reader, meta, region, channel, z=int(z), t=t, max_px=max_px)
        if got is None:
            out = np.zeros((h, w), dtype=dtype)
        else:
            arr = got[0]
            if arr.shape != (h, w):    # a ragged z would silently misalign the stack
                out = np.zeros((h, w), dtype=dtype)
                out[: min(h, arr.shape[0]), : min(w, arr.shape[1])] = \
                    arr[: min(h, arr.shape[0]), : min(w, arr.shape[1])]
            else:
                out = arr
        with _cache_lock:
            _cache["z"], _cache["plane"] = z, out
        return out

    from dask import delayed

    blocks = [
        da.from_delayed(delayed(_plane)(z), shape=(h, w), dtype=dtype)[None, ...]
        for z in range(nz)
    ]
    return da.concatenate(blocks, axis=0), step, nz


#: Byte bound on the fused-plane cache, shared across every region/channel/level/z.
PYRAMID_CACHE_BYTES = cache_budget()

#: Runaway guard on level count; levels stop earlier once a level stops shrinking.
_MAX_PREVIEW_LEVELS = 12

#: Below this the level is smaller than a thumbnail and buys nothing.
_MIN_LEVEL_PX = 128


class MemoryBoundedLRUCache:
    """Thread-safe LRU cache bounded by BYTES, not by entry count.

    An item bigger than the whole budget raises rather than being silently dropped.
    """

    def __init__(self, max_memory_bytes: int):
        max_memory_bytes = int(max_memory_bytes)
        if max_memory_bytes <= 0:
            raise ValueError(f"cache capacity must be positive, got {max_memory_bytes} bytes.")
        self._max_memory = max_memory_bytes
        self._current_memory = 0
        self._cache: "OrderedDict[tuple, Any]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def capacity_bytes(self) -> int:
        return self._max_memory

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._current_memory

    def get(self, key: tuple):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: tuple, value) -> None:
        item_size = int(value.nbytes)
        if item_size > self._max_memory:
            raise ValueError(
                f"cannot cache a {item_size / 1e6:.1f} MB plane: that is larger than the whole "
                f"{self._max_memory / 1e6:.1f} MB cache budget. Raise the budget or lower "
                "max_px — a cache that silently stores nothing is just a slow viewer."
            )
        with self._lock:
            if key in self._cache:
                self._current_memory -= self._cache.pop(key).nbytes
            while self._current_memory + item_size > self._max_memory and self._cache:
                _oldest_key, oldest = self._cache.popitem(last=False)
                self._current_memory -= oldest.nbytes
            self._cache[key] = value
            self._current_memory += item_size

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

    def invalidate(self, key: tuple) -> bool:
        with self._lock:
            if key in self._cache:
                self._current_memory -= self._cache.pop(key).nbytes
                return True
            return False

    def __contains__(self, key: tuple) -> bool:
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


#: The process-wide preview cache — one instance, so the bound covers the whole preview path.
_PLANE_CACHE = MemoryBoundedLRUCache(PYRAMID_CACHE_BYTES)


def plane_cache() -> MemoryBoundedLRUCache:
    """THE process-wide preview cache, by name, for producers outside this module."""
    return _PLANE_CACHE


def source_token(reader: Any) -> str:
    """Public name for :func:`_source_token` — see there. Raises when a reader has no identity."""
    return _source_token(reader)


def _source_token(reader: Any) -> str:
    """Stable identity of the acquisition a reader reads, for cache keys."""
    path = getattr(reader, "_path", None)
    if path is None:
        raise ValueError(
            f"{type(reader).__name__} exposes no '_path', so its cache entries cannot be told "
            "apart from another acquisition's. Refusing to risk serving the wrong pixels."
        )
    return str(path)


def _level_max_px(max_px: int, k: int) -> int:
    return max(_MIN_LEVEL_PX, int(max_px) >> k)


def _fuse_levels(reader: Any, meta: dict, region: str, channel: str, z: int, t: int, plans: list):
    """Fuse ONE z into SEVERAL pyramid levels in a single pass over the FOV frames.

    TIFF decode is whole-frame, so every level coarser than the one asked for is pasted from
    the frames already in hand; nothing finer is built.
    """
    from squidxplorer._placement import fov_offsets_px

    positions = meta.get("fov_positions_um") or {}
    pixel_size = meta.get("pixel_size_um")
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    offsets = fov_offsets_px(positions, region, fovs, pixel_size)

    outs = {px: np.zeros((h, w), dtype=dt) for px, h, w, _st, dt in plans}
    unreadable = []
    for fov in fovs:
        try:
            frame = reader.read(region, fov, channel, int(z), int(t))
        except Exception as exc:                 # noqa: BLE001 - collected, then reported
            unreadable.append((fov, f"{type(exc).__name__}: {exc}"))
            continue
        if frame is None:
            unreadable.append((fov, "reader returned None"))
            continue
        frame = np.asarray(frame)
        if frame.ndim != 2:
            frame = frame.reshape(frame.shape[-2:])
        row, col = offsets[fov]
        for px, h, w, st, _dt in plans:
            step = int(st)
            sub = frame[::step, ::step]
            r0, c0 = row // step, col // step
            r1, c1 = min(r0 + sub.shape[0], h), min(c0 + sub.shape[1], w)
            if r1 > r0 and c1 > c0:
                # Later FOVs overwrite earlier ones in the overlap: a preview placement, not a stitch.
                outs[px][r0:r1, c0:c1] = sub[: r1 - r0, : c1 - c0]

    if unreadable and len(unreadable) < len(fovs):
        _log.warning("mosaic %s/%s z=%s: %d of %d FOV(s) could not be read — they are BLACK HOLES "
                     "in this mosaic, not empty fields. First: fov %s: %s",
                     region, channel, z, len(unreadable), len(fovs),
                     unreadable[0][0], unreadable[0][1])
    if unreadable and len(unreadable) == len(fovs):
        # Every FOV bad is not a picture at all; a black plane would report a read failure as
        # empty tissue.
        why = "; ".join(f"fov {f}: {m}" for f, m in unreadable[:3])
        raise ValueError(
            f"{region}/{channel} z={z}: no FOV in the region could be read "
            f"({len(unreadable)} of {len(fovs)} failed) — {why}"
        )
    return outs


def fuse_region_pyramid(
    reader: Any,
    meta: dict,
    region: str,
    channel: str,
    *,
    t: int = 0,
    max_px: int = _MAX_FUSED_PX,
    cache_bytes: Optional[int] = None,
):
    """A lazy multiscale pyramid over the fused region mosaic — what napari wants.

    Returns ``(levels, step, nz)``; each level is fused directly from the FOV tiles at its own
    decimation, y and x only (z is never coarsened). Returns ``None`` when geometry is
    underivable.
    """
    import dask.array as da
    from dask import delayed

    base = _planned_plane(meta, region, max_px)
    if base is None:
        return None
    _h0, _w0, step0, dtype = base

    # The plane budget guards level 0: it is still materialised at full zoom.
    per_plane = _h0 * _w0 * dtype.itemsize
    if per_plane > _PLANE_BUDGET_BYTES:
        raise MemoryError(
            f"{region}/{channel}: one fused plane is {per_plane / 1e9:.1f} GB "
            f"({_h0}x{_w0} {dtype}), over the {_PLANE_BUDGET_BYTES / 1e9:.1f} GB plane budget. "
            "Lower max_px rather than letting this page the machine."
        )

    cache = _PLANE_CACHE if cache_bytes is None else MemoryBoundedLRUCache(cache_bytes)
    token = _source_token(reader)
    nz = int(meta.get("n_z") or 1)

    # Plan every rung up front: pure geometry, reads nothing.
    plans: list = []
    for k in range(_MAX_PREVIEW_LEVELS):
        level_px = _level_max_px(max_px, k)
        plan = _planned_plane(meta, region, level_px)
        if plan is None:
            # Level 0 resolved above, so a later level failing means inconsistent geometry.
            raise ValueError(
                f"{region}/{channel}: level {k} (max_px={level_px}) has no derivable geometry "
                f"although level 0 does. Refusing to hand napari a partial pyramid."
            )
        h, w, step, dt = plan
        plans.append((level_px, h, w, step, dt))
        if level_px <= _MIN_LEVEL_PX:
            break

    def _plane(i: int, z: int):
        """Level ``i`` at ``z``, from the cache or one decode pass that also fills coarser levels."""
        level_px, h, w, step, dt = plans[i]
        key = (token, region, channel, int(t), float(step), int(z))
        hit = cache.get(key)
        if hit is not None:
            return hit

        # This level and every coarser one; nothing finer is built.
        wanted = plans[i:]
        outs = _fuse_levels(reader, meta, region, channel, int(z), int(t), wanted)
        for px, ph, pw, pstep, _pdt in wanted:
            arr = outs[px]
            if arr.shape != (ph, pw):
                raise ValueError(
                    f"{region}/{channel}: z={z} fused to {arr.shape}, but this pyramid level is "
                    f"{(ph, pw)}. A ragged z would misalign the stack and misregister the level."
                )
            cache.put((token, region, channel, int(t), float(pstep), int(z)), arr)
        return outs[level_px]

    levels = []
    for i, (_px, h, w, _step, dt) in enumerate(plans):
        if nz <= 1:
            lv = da.from_delayed(delayed(_plane)(i, 0), shape=(h, w), dtype=dt)
        else:
            lv = da.concatenate(
                [da.from_delayed(delayed(_plane)(i, z), shape=(h, w), dtype=dt)[None, ...]
                 for z in range(nz)],
                axis=0,
            )
        levels.append(lv)

    # One guard, shared with open_pyramid: a level that did not shrink is dropped.
    return strictly_decreasing_levels(levels), step0, nz


def mosaic_bbox_um(meta: dict, region: str) -> Optional[tuple[float, float, float, float]]:
    """``(x0, y0, x1, y1)`` stage micrometres covered by a region's mosaic."""
    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    pixel_size = meta.get("pixel_size_um")
    if not positions or pixel_size in (None, 0):
        return None
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        return None
    try:
        offsets = fov_offsets_px(positions, region, fovs, pixel_size)
        h, w = mosaic_extent_px(offsets, tuple(int(v) for v in meta["frame_shape"]))
    except (KeyError, ValueError):
        return None

    xs = [float(positions[(region, f)][0]) for f in fovs]
    ys = [float(positions[(region, f)][1]) for f in fovs]
    x0, y0 = min(xs), min(ys)
    return (x0, y0, x0 + w * float(pixel_size), y0 + h * float(pixel_size))


def fovs_overlapping_bbox(meta: dict, region: str,
                          bbox_um: "Optional[tuple]") -> "Optional[list[int]]":
    """The WHOLE FOVs of *region* whose footprint overlaps *bbox_um*, in acquisition order.

    Returns ``None`` when the question cannot be answered or no field overlaps, so a caller
    falls back to the whole region rather than running on a silently empty set.
    """
    from squidxplorer._placement import fov_offsets_px

    if bbox_um is None:
        return None
    positions = meta.get("fov_positions_um") or {}
    pixel_size = meta.get("pixel_size_um")
    if not positions or pixel_size in (None, 0):
        return None
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        return None
    try:
        offsets = fov_offsets_px(positions, region, fovs, pixel_size)
        fh, fw = (int(v) for v in meta["frame_shape"])
    except (KeyError, ValueError, TypeError):
        return None
    x0 = min(float(positions[(region, f)][0]) for f in fovs)
    y0 = min(float(positions[(region, f)][1]) for f in fovs)
    p = float(pixel_size)

    rx0, ry0, rx1, ry1 = (float(v) for v in bbox_um)
    rx0, rx1 = min(rx0, rx1), max(rx0, rx1)
    ry0, ry1 = min(ry0, ry1), max(ry0, ry1)

    hit = []
    for fov in fovs:
        row, col = offsets[fov]
        fx0, fy0 = x0 + col * p, y0 + row * p
        # Half-open on the far edge: a box that stops exactly on a seam belongs to the field it
        # is inside, not to both.
        if fx0 < rx1 and (fx0 + fw * p) > rx0 and fy0 < ry1 and (fy0 + fh * p) > ry0:
            hit.append(int(fov))
    return hit or None


def fov_at_point(meta: dict, region: str, x_um: float, y_um: float) -> "Optional[int]":
    """The FOV under one stage-micrometre point, or ``None`` off the mosaic."""
    hit = fovs_overlapping_bbox(meta, region, (float(x_um), float(y_um),
                                               float(x_um), float(y_um)))
    return None if not hit else hit[0]
