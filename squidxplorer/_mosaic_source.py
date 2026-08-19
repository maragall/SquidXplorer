"""Fused mosaics for the napari pane — the unit displayed is a MOSAIC, never a single FOV.

Written OME-Zarr is read lazily as a dask pyramid; a raw acquisition is fused by pasting
frames at their stage-position offsets.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from squidxplorer import _bitdepth
from squidxplorer._budget import cache_budget
from squidxplorer._logpane import get_logger

# Carries no Qt, so a fuser can report unreadable FOVs in a headless run.
_log = get_logger("mosaic")


# THE one store walk (contract.store); the name stays because callers import it from here.
from squidxplorer.contract.store import level_paths  # noqa: F401


def open_pyramid(group, *, time_point: int = 0, c: int = 0, z_level: int = 0) -> list:
    """Lazy 2-D dask pyramid for one (t, c, z) of a written field/mosaic group."""
    import dask.array as da
    import zarr

    out = []
    for path in level_paths(group):
        arr = zarr.open_array(str(path), mode="r")
        d = da.from_array(arr, chunks=arr.chunks)
        # Squid canonical order is (t, c, z, y, x); 2-D/3-D stores are legal NGFF too.
        if d.ndim == 5:
            d = d[time_point, c, z_level]
        elif d.ndim == 4:
            d = d[time_point, c]
        elif d.ndim == 3:
            d = d[z_level]
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
    z_level: int = 0,
    time_point: int = 0,
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
            frame = reader.read(region, fov, channel, z_level, time_point)
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
        # The contrast slider's ceiling, measured HERE because this is full-resolution camera
        # data. Anything downstream is strided (`sub`, below) or averaged, and a maximum taken
        # from a decimated plane UNDER-states the real one -- which snaps the ceiling too low and
        # clips. ~0.06 ms on a 2048x2048 frame against a 2.6 ms decode.
        _bitdepth.depth().observe_array(frame)
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
    time_point: int = 0,
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
        probe = fuse_region_mosaic(reader, meta, region, channel, z_level=0, time_point=time_point,
                                   max_px=max_px)
        if probe is None:
            return None
        return probe[0], probe[1], 1

    # One-plane cache: the same z is sliced twice (contrast sample + napari draw) per region change.
    _cache: dict = {"z": None, "plane": None}
    _cache_lock = __import__("threading").Lock()

    def _plane(z_level: int):
        z_level = int(z_level)
        with _cache_lock:
            if _cache["z"] == z_level and _cache["plane"] is not None:
                return _cache["plane"]
        got = fuse_region_mosaic(reader, meta, region, channel, z_level=int(z_level), time_point=time_point, max_px=max_px)
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
            _cache["z"], _cache["plane"] = z_level, out
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

#: Chunk edge of the on-demand fine rungs: what one deep-zoom slice materialises.
_FINE_CHUNK_PX = 2048


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
    """Stable identity of the acquisition a reader reads, for cache keys.

    ``source_id`` is the contract's identity member (for the Zarr reader it is the acquisition
    ROOT, so the staleness token stats sidecars that exist); ``_path`` is the fallback for
    doubles written before it was declared.
    """
    path = getattr(reader, "source_id", None) or getattr(reader, "_path", None)
    if path is None:
        raise ValueError(
            f"{type(reader).__name__} exposes neither 'source_id' nor '_path', so its cache "
            "entries cannot be told apart from another acquisition's. Refusing to risk serving "
            "the wrong pixels."
        )
    return str(path)


def _level_max_px(max_px: int, k: int) -> int:
    return max(_MIN_LEVEL_PX, int(max_px) >> k)


def _norm_window(idx, shape) -> "tuple[slice, slice]":
    """Normalise a 2-D index to unit-step ``(rows, cols)`` slices over *shape*."""
    if not isinstance(idx, tuple):
        idx = (idx,)
    idx = idx + (slice(None),) * (2 - len(idx))
    out = []
    for s, n in zip(idx, shape):
        if isinstance(s, slice):
            if s.step not in (None, 1):
                raise ValueError(f"a fine mosaic level slices at unit step only, got {s!r}")
            out.append(slice(*s.indices(int(n))[:2]))
        else:
            i = int(s)
            out.append(slice(i, i + 1))
    return out[0], out[1]


class _WindowedLevel:
    """A pyramid rung materialised BY WINDOW: a slice pastes only the FOVs it touches.

    Same paste rule as :func:`_fuse_levels` — ``frame[::step, ::step]`` at ``offset // step``,
    later FOVs overwriting earlier in the overlap — so a window is pixel-identical to the
    whole-plane fusion it stands in for (tests pin it bit-exact at every stride). A slice reads
    a few files, never the region; decoded frames land in the shared plane cache, so panning
    re-reads nothing.

    EVERY rung is served this way, coarse included (2026-08-19). The coarse rungs used to be
    one ``delayed`` whole-region fuse each, and napari's draw blocks synchronously on the slice
    it asks for — so on a 452-FOV region every zoom notch decoded all 452 frames to show a
    viewport covering a dozen (measured 0.35–3.7 s per rung per channel: "stopped responding").

    ``well`` (coarse rungs only) is the Squid well-image resolver: a rung at least as coarse as
    the saved file's factor is derived WHOLE from it — one small file read, no FOV decode — and
    cached under this rung's plane key, which any window then slices.
    """

    def __init__(self, reader, meta, region, channel, z_level, time_point,
                 step, shape, dtype, cache, token, well=None):
        from squidxplorer._placement import fov_offsets_px

        self._reader, self._meta = reader, meta
        self._region, self._channel = str(region), str(channel)
        self._z, self._t = int(z_level), int(time_point)
        self._step = int(step)
        self.shape = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = 2
        self._cache, self._token = cache, token
        self._well = well
        # The whole-plane key _plane used before the conversion: same tuple, so a plane fused
        # once (full-window compute, well image) is a hit for every later window of this rung.
        self._plane_key = (token, self._region, self._channel, self._t,
                           float(self._step), self._z)
        self._fovs = list((meta.get("fovs_per_region") or {}).get(self._region) or [])
        self._offsets = fov_offsets_px(meta["fov_positions_um"], self._region, self._fovs,
                                       meta["pixel_size_um"])
        self._frame_hw = tuple(int(v) for v in meta["frame_shape"])

    def _frame(self, fov: int) -> np.ndarray:
        key = (self._token, "fovplane", self._region, int(fov), self._channel, self._z, self._t)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        frame = np.asarray(self._reader.read(self._region, int(fov), self._channel,
                                             self._z, self._t))
        if frame.ndim != 2:
            frame = frame.reshape(frame.shape[-2:])
        _bitdepth.depth().observe_array(frame)   # same full-resolution observation _fuse_levels makes
        self._cache.put(key, frame)
        return frame

    def _whole_plane(self) -> Optional[np.ndarray]:
        """This rung's WHOLE plane when one is already in hand — cached, or well-image-derived."""
        hit = self._cache.get(self._plane_key)
        if hit is not None:
            return hit
        if self._well is None:
            return None
        got = self._well()
        if got is None:
            return None
        plane_w, factor = got
        if self._step < int(factor):
            return None                          # finer than the file: fuse from the FOVs
        from squidxplorer import _wellimage

        arr = _wellimage.resample_plane(plane_w, int(factor), self._step,
                                        self.shape[0], self.shape[1]).astype(self.dtype,
                                                                             copy=False)
        # An area-averaged plane UNDER-states the true ceiling, but the range only ever
        # widens (_bitdepth): this seeds a floor until the first full-resolution decode.
        _bitdepth.depth().observe_array(arr)
        self._cache.put(self._plane_key, arr)
        return arr

    def __getitem__(self, idx) -> np.ndarray:
        ys, xs = _norm_window(idx, self.shape)
        whole = self._whole_plane()
        if whole is not None:
            return whole[ys, xs]
        out = np.zeros((ys.stop - ys.start, xs.stop - xs.start), self.dtype)
        step = self._step
        fh = -(-self._frame_hw[0] // step)       # a decimated frame's own extent
        fw = -(-self._frame_hw[1] // step)
        touched, failed = 0, []
        for fov in self._fovs:                   # _fuse_levels' order: later overwrites earlier
            row, col = self._offsets[fov]
            r0, c0 = row // step, col // step
            if r0 >= ys.stop or c0 >= xs.stop or r0 + fh <= ys.start or c0 + fw <= xs.start:
                continue
            touched += 1
            try:
                sub = self._frame(fov)[::step, ::step]
            except Exception as exc:             # noqa: BLE001 - a black hole, as in _fuse_levels
                _log.warning("mosaic %s/%s z=%s step=%s: fov %s unreadable (%s) — a BLACK HOLE.",
                             self._region, self._channel, self._z, step, fov, exc)
                failed.append((fov, exc))
                continue
            rr0, cc0 = max(r0, ys.start), max(c0, xs.start)
            rr1 = min(r0 + sub.shape[0], ys.stop)
            cc1 = min(c0 + sub.shape[1], xs.stop)
            if rr1 <= rr0 or cc1 <= cc0:
                continue
            out[rr0 - ys.start:rr1 - ys.start, cc0 - xs.start:cc1 - xs.start] = \
                sub[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
        if touched and len(failed) == touched:
            # Every FOV under the window bad is not a picture: a black window would report a
            # read failure as empty tissue (same refusal _fuse_levels made for a whole region).
            why = "; ".join(f"fov {f}: {type(e).__name__}: {e}" for f, e in failed[:3])
            raise ValueError(
                f"{self._region}/{self._channel} z={self._z}: no FOV under this window could "
                f"be read ({len(failed)} of {touched} failed) — {why}")
        # A full-window compute IS the whole plane: cache it so the next pull (napari asks for
        # the coarsest rung per thumbnail) costs a lookup even after the frames evict.
        if (ys, xs) == (slice(0, self.shape[0]), slice(0, self.shape[1])) \
                and out.nbytes <= self._cache.capacity_bytes:
            self._cache.put(self._plane_key, out)
        return out


def _fuse_levels(reader: Any, meta: dict, region: str, channel: str, z_level: int, time_point: int, plans: list):
    """Fuse ONE z into SEVERAL pyramid levels in a single pass over the FOV frames.

    THE REFERENCE PASTE RULE, no longer on the interactive path: every displayed rung is a
    :class:`_WindowedLevel` (2026-08-19), and the parity tests pin each windowed rung
    bit-exact against this function's product at the same stride. TIFF decode is whole-frame,
    so every level coarser than the one asked for is pasted from the frames already in hand;
    nothing finer is built.
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
            frame = reader.read(region, fov, channel, int(z_level), int(time_point))
        except Exception as exc:                 # noqa: BLE001 - collected, then reported
            unreadable.append((fov, f"{type(exc).__name__}: {exc}"))
            continue
        if frame is None:
            unreadable.append((fov, "reader returned None"))
            continue
        frame = np.asarray(frame)
        if frame.ndim != 2:
            frame = frame.reshape(frame.shape[-2:])
        # See `fuse_region_mosaic`: the ceiling is measured on the FULL-RESOLUTION frame, before
        # any `frame[::step, ::step]` below has a chance to hide the brightest pixel. This is the
        # observation that covers every region the app displays, and it lands on the worker
        # thread BEFORE `ready` is emitted -- so a region's layer is built already knowing it.
        _bitdepth.depth().observe_array(frame)
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
                     region, channel, z_level, len(unreadable), len(fovs),
                     unreadable[0][0], unreadable[0][1])
    if unreadable and len(unreadable) == len(fovs):
        # Every FOV bad is not a picture at all; a black plane would report a read failure as
        # empty tissue.
        why = "; ".join(f"fov {f}: {m}" for f, m in unreadable[:3])
        raise ValueError(
            f"{region}/{channel} z={z_level}: no FOV in the region could be read "
            f"({len(unreadable)} of {len(fovs)} failed) — {why}"
        )
    return outs


def fuse_region_pyramid(
    reader: Any,
    meta: dict,
    region: str,
    channel: str,
    *,
    time_point: int = 0,
    max_px: int = _MAX_FUSED_PX,
    cache_bytes: Optional[int] = None,
):
    """A lazy multiscale pyramid over the fused region mosaic — what napari wants.

    Returns ``(levels, step, nz)``; EVERY level is a chunked windowed source
    (:class:`_WindowedLevel`), so a viewport slice pastes only the FOVs under it, at any zoom
    — fused at the rung's own decimation, y and x only (z is never coarsened). Returns
    ``None`` when geometry is underivable.

    When Squid saved a downsampled well mosaic for this (region, channel, t)
    (``mosaic_view/wells``, see :mod:`squidxplorer._wellimage`), every rung at least as coarse
    as that file's own factor is derived from IT instead of fused from FOV decodes: first
    paint then costs one small file read. Finer rungs are fused exactly as before; an absent,
    corrupt or multi-z well image falls back to fusing, with the reason logged.
    """
    import dask.array as da

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

    # Squid's saved well mosaic, resolved once and shared by every rung's windowed source.
    _well_lock = threading.Lock()
    _well: list = []                # [] unresolved; [(plane, factor)] found; [None] absent

    def _well_source():
        with _well_lock:
            if not _well:
                from squidxplorer import _wellimage

                _well.append(_wellimage.downsampled_well(reader, meta, region, channel,
                                                         int(time_point)))
            return _well[0]

    def _rung(s: int, h: int, w: int, dt, well):
        """One windowed dask rung: a viewport slice pastes only the FOVs under it."""
        def one_z(z: int):
            src = _WindowedLevel(reader, meta, region, channel, z, time_point,
                                 s, (h, w), dt, cache, token, well=well)
            return da.from_array(src, chunks=_FINE_CHUNK_PX, asarray=False,
                                 meta=np.empty((0, 0), dtype=dt),
                                 name=f"raw-win-{token}-{region}-{channel}-s{s}-z{z}")
        if nz <= 1:
            return one_z(0)
        return da.concatenate([one_z(z)[None, ...] for z in range(nz)], axis=0)

    # EVERY planned rung is windowed (2026-08-19). They used to be one whole-region
    # ``delayed`` fuse each, and napari's draw blocks synchronously on the slice it asks for:
    # on a 452-FOV single-z region every zoom notch decoded all 452 frames to show a viewport
    # covering a dozen (0.35–3.7 s per rung per channel, measured — "stopped responding").
    # The well-image short-circuit lives on inside ``_WindowedLevel._whole_plane``.
    levels = [_rung(int(step), h, w, dt, _well_source) for _px, h, w, step, dt in plans]

    # FINER RUNGS, ON DEMAND, down to native. Budget-gated per rung because the
    # full-materialisation consumers (the 3-D full-res swap, _full_res_mip) still take level 0
    # whole. ``well=None`` on purpose: a fine rung's pixels stay strided camera data, never
    # the area-averaged well image.
    native = _planned_plane(meta, region, 10 ** 9)
    fine: list = []
    if native is not None and int(step0) > 1:
        nh, nw, _s1, _dt = native
        s = 1
        while s < int(step0):
            h_s, w_s = int(np.ceil(nh / s)), int(np.ceil(nw / s))
            if h_s * w_s * dtype.itemsize * max(1, nz) <= _PLANE_BUDGET_BYTES:
                fine.append((s, h_s, w_s))
            s *= 2
        fine.sort()                                        # ascending step = descending resolution
        levels = [_rung(s, h_s, w_s, dtype, None) for s, h_s, w_s in fine] + levels

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


def mosaic_fov_bboxes_um(meta: dict, region: str) -> "dict[int, tuple[float, float, float, float]]":
    """``{fov: (x0, y0, x1, y1)}`` in stage micrometres for every FOV of *region*, in acquisition
    order.

    THE PLACEMENT THAT DREW THE PIXELS, not a second derivation of it. The origin is ``min`` over
    the region's recorded positions and each offset is
    :func:`~squidxplorer._placement.fov_offsets_px` -- exactly what :func:`mosaic_bbox_um` above
    uses, and therefore exactly what ``_workers._MosaicWorker`` placed this window's layer with.
    So a box this returns lands on the pixels it names, in napari's world, with no conversion.

    IT IS NOT ``_tilesource.fov_bboxes_um``, and the difference is not cosmetic. That one treats a
    recorded position as the frame's CENTRE (as ``_output.field_origin_um`` does, for the NGFF
    translation); everything on the mosaic path treats it as the frame's TOP-LEFT. The two are
    half a frame apart -- measured, 195.9 um on the 40x AF-sweep set -- and half a frame of
    UNIFORM shear renders as a perfectly plausible picture of the wrong tissue. Each convention is
    right for its own surface (that one places the PLATE, this one places a WINDOW's mosaic), so
    neither is a bug; picking the wrong one HERE would be. The check that catches it is free and
    already on screen: the centre of every box this returns must satisfy ``fov_at_point`` -> that
    same fov.

    RAISES rather than returning ``None``, which is the opposite of what
    :func:`fovs_overlapping_bbox` below does, and the asymmetry is the point. That function has an
    honest fallback -- "the user boxed nothing, so run on the whole region" -- so a caller that
    cannot tell "no overlap" from "no positions" still lands somewhere sensible. A caller that
    wants to DRAW every FOV has no such fallback: fifteen boxes out of sixteen is a picture of a
    region with a hole in it, and the fifteen look exactly as convincing as sixteen would. So the
    reason comes back as a sentence and the caller says it out loud. ``fov_offsets_px``'s own
    ``KeyError`` is let through unwrapped, because its message ("coordinates.csv and the image
    filenames disagree; refusing to draw a mosaic with holes") is better than anything added here.
    """
    from squidxplorer._placement import fov_offsets_px

    positions = meta.get("fov_positions_um") or {}
    if not positions:
        raise ValueError(
            f"{region}: this acquisition records no stage positions, so its FOVs cannot be "
            f"located -- coordinates.csv is missing or unreadable.")
    pixel_size = meta.get("pixel_size_um")
    if pixel_size in (None, 0):
        raise ValueError(
            f"{region}: this acquisition records no pixel size, so a FOV's extent in micrometres "
            f"cannot be derived.")
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        raise ValueError(f"{region}: no FOVs -- is that a region of this acquisition?")

    offsets = fov_offsets_px(positions, region, fovs, pixel_size)   # KeyError: let it speak
    fh, fw = (int(v) for v in meta["frame_shape"])
    p = float(pixel_size)
    # The region origin, spelled exactly as `mosaic_bbox_um` spells it. Two spellings of one
    # origin is how the boxes and the mosaic they sit on drift apart by a sub-pixel nobody
    # notices until it is a whole frame.
    x0 = min(float(positions[(region, f)][0]) for f in fovs)
    y0 = min(float(positions[(region, f)][1]) for f in fovs)

    out: "dict[int, tuple[float, float, float, float]]" = {}
    for fov in fovs:
        row, col = offsets[fov]
        fx0, fy0 = x0 + col * p, y0 + row * p
        out[int(fov)] = (fx0, fy0, fx0 + fw * p, fy0 + fh * p)
    return out


def fovs_overlapping_bbox(meta: dict, region: str,
                          bbox_um: "Optional[tuple]") -> "Optional[list[int]]":
    """The WHOLE FOVs of *region* whose footprint overlaps *bbox_um*, in acquisition order.

    ONE GEOMETRY, and it is one FUNCTION: the boxes come from :func:`mosaic_fov_bboxes_um` above,
    which is the same placement rule that drew the pixels the user boxed. This used to build them
    inline, and so did :func:`fov_at_point` below -- two copies of "where is FOV 7" sitting in one
    file under a docstring warning against exactly that. A third consumer (the FOV walk) was what
    made the cost of keeping them visible.

    Returns ``None`` when the question cannot be answered or no field overlaps, so a caller
    falls back to the whole region rather than running on a silently empty set. That is why the
    raise from ``mosaic_fov_bboxes_um`` is caught here and nowhere else -- see its docstring for
    why the two contracts differ.
    """
    if bbox_um is None:
        return None
    try:
        boxes = mosaic_fov_bboxes_um(meta, region)
    except (KeyError, ValueError, TypeError):
        return None

    rx0, ry0, rx1, ry1 = (float(v) for v in bbox_um)
    rx0, rx1 = min(rx0, rx1), max(rx0, rx1)
    ry0, ry1 = min(ry0, ry1), max(ry0, ry1)

    hit = []
    for fov, (fx0, fy0, fx1, fy1) in boxes.items():
        # Half-open on the far edge: a box that stops exactly on a seam belongs to the field it
        # is inside, not to both.
        if fx0 < rx1 and fx1 > rx0 and fy0 < ry1 and fy1 > ry0:
            hit.append(int(fov))
    return hit or None


def fov_pixel_at_point(meta: dict, region: str, x_um: float,
                       y_um: float) -> "Optional[tuple[int, float, float]]":
    """``(fov, py, px)`` — which FOV a stage-micrometre point is in, and where inside THAT FRAME
    it lands, in level-0 acquisition pixels. ``None`` off the mosaic.

    THE SAME GEOMETRY AS :func:`fovs_overlapping_bbox`, which is why it lives here and not in the
    loupe: the field the loupe magnifies and the field a stitch run would select must be the same
    field. A loupe that derived "where is FOV 7" for itself would agree with itself and with
    nothing else, and the disagreement would be invisible — both surfaces would show a sharp,
    plausible field.

    THE LAST OVERLAPPING FIELD WINS, not the first. Fields overlap at the seams, and the preview
    fuser is later-overwrites-earlier (``fuse_region_mosaic``; CLAUDE.md, "Two producers of a
    region's pixels"), so the field whose pixels are actually ON TOP at a seam is the last one.
    ``_plate_overview._fov_box_at`` settled this for the plate in the same words and for the same
    reason. Magnifying the field underneath the one the user can see is exactly the wrong-image
    failure this whole module is careful about.
    """
    try:
        boxes = mosaic_fov_bboxes_um(meta, region)
    except (KeyError, ValueError, TypeError):
        return None
    pixel_size = float(meta["pixel_size_um"])
    hit = None
    for fov, (fx0, fy0, fx1, fy1) in boxes.items():
        if fx0 <= x_um < fx1 and fy0 <= y_um < fy1:
            hit = (int(fov), (y_um - fy0) / pixel_size, (x_um - fx0) / pixel_size)
    return hit


def fov_at_point(meta: dict, region: str, x_um: float, y_um: float) -> "Optional[int]":
    """The FOV under one stage-micrometre point, or ``None`` off the mosaic."""
    hit = fovs_overlapping_bbox(meta, region, (float(x_um), float(y_um),
                                               float(x_um), float(y_um)))
    return None if not hit else hit[0]
