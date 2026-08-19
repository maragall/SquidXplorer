"""Static whole-plate thumbnail montage rendered from the canonical OME-zarr HCS plate.

Single streaming pass (one well resident at a time), global-per-channel contrast so wells
stay comparable, additive RGB composite, plus JSON sidecar and self-contained HTML viewer.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

# Montage cell size (downsampled well thumbnail, px).
_DEFAULT_CELL_PX = 128
# Per-channel contrast percentiles across all wells; clips hot pixels.
_DEFAULT_PERCENTILES = (1.0, 99.8)


# THE one store walk (contract.store): v0.4/v0.5-normalising attrs and plate-dir resolution.
from squidxplorer.contract.store import ome_attrs as _read_group_ome
from squidxplorer.contract.store import resolve_plate_dir as _resolve_plate_dir


_PCT = (1.0, 99.8)


def _pct_window(a: np.ndarray, pct=_PCT) -> tuple[float, float]:
    """EXACT percentile window over *a*. THE percentile rule — there is no second one.

    Exactness is the point. ``_RunningContrast`` quantizes to a bin ~dmax/bins wide (~128 counts
    on uint16), so a dim region spanning a few hundred counts collapses into two or three bins and
    its window comes out garbage — precisely the region PER_REGION exists to rescue. The histogram
    stays the live during-run approximation; this is what the final render uses.

    A degenerate result (hi <= lo) is returned as-is on purpose: ``_window`` renders it black.
    Widening it to ``(lo, lo + 1)`` is a real bug wearing a helpful face — ``+1`` is one DATA
    unit, so ``(v - lo) / 1`` clips to 1.0 and a blank or saturated channel renders FULL WHITE and
    reads as signal. The loupe carried a private copy that DID widen, and that is the defect
    IMA-242 collapsed (see the note in ``_plate_overview``).

    IT LIVES HERE, and not in a widget module, because it is a numpy percentile and because the
    things that must not disagree with it are here: :func:`composite` is the one compositor and
    :func:`_area_downsample` the one resampler. A rule reachable only by importing a QWidget
    module is a rule the modules below the GUI boundary will copy.
    """
    if a.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(a, pct)
    return float(lo), float(hi)


def _area_downsample(plane: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Area-average *plane* (Y, X) down to at most (out_h, out_w).

    Never upsamples; the target is clamped per axis, so the returned shape is
    ``(min(out_h, Y), min(out_w, X))`` and callers needing an exact shape must guard.
    """
    y, x = plane.shape
    out_h, out_w = min(int(out_h), y), min(int(out_w), x)   # per axis: no bin count can be 0
    if out_h == y and out_w == x:
        return plane.astype(np.float32, copy=False)
    row_edges = (np.arange(out_h) * y) // out_h
    col_edges = (np.arange(out_w) * x) // out_w
    row_counts = np.diff(np.append(row_edges, y))
    col_counts = np.diff(np.append(col_edges, x))
    summed = np.add.reduceat(plane.astype(np.float32), row_edges, axis=0)
    summed = np.add.reduceat(summed, col_edges, axis=1)
    return summed / (row_counts[:, None] * col_counts[None, :])


def _window(channel_plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear contrast window [lo, hi] -> [0, 1] as float32; guards a degenerate channel."""
    span = hi - lo
    if span <= 0:  # empty / flat channel — avoid divide-by-zero
        return np.zeros_like(channel_plane, dtype=np.float32)
    out = (channel_plane.astype(np.float32, copy=False) - np.float32(lo)) / np.float32(span)
    return np.clip(out, 0.0, 1.0, out=out)


_LUT_MAX_ITEMSIZE = 2            # uint8 and uint16 only; a 32-bit table is 4 G entries
_LUT_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_LUT_CACHE_MAX = 64
_LUT_LOCK = threading.Lock()


def _window_lut(dtype: np.dtype, lo: float, hi: float) -> Optional[np.ndarray]:
    """``_window`` memoised over every value *dtype* can hold, or None if that is not finite."""
    dt = np.dtype(dtype)
    if dt.kind != "u" or dt.itemsize > _LUT_MAX_ITEMSIZE:
        return None
    key = (dt.str, float(lo), float(hi))
    with _LUT_LOCK:
        hit = _LUT_CACHE.get(key)
        if hit is not None:
            _LUT_CACHE.move_to_end(key)
            return hit
    table = _window(np.arange(1 << (8 * dt.itemsize), dtype=dt), lo, hi)
    with _LUT_LOCK:
        _LUT_CACHE[key] = table
        while len(_LUT_CACHE) > _LUT_CACHE_MAX:
            _LUT_CACHE.popitem(last=False)
    return table


_COMPOSITE_MIN_PX_PER_BAND = 120_000   # below this a band costs more in dispatch than it saves
_COMPOSITE_POOL: "Optional[ThreadPoolExecutor]" = None
_COMPOSITE_POOL_LOCK = threading.Lock()


def _composite_pool() -> "ThreadPoolExecutor":
    """A small, process-wide pool for banded compositing.

    Created lazily under a lock: composite() is called from two threads and a racing
    loser's pool would leak its workers.
    """
    global _COMPOSITE_POOL
    with _COMPOSITE_POOL_LOCK:
        if _COMPOSITE_POOL is None:
            _COMPOSITE_POOL = ThreadPoolExecutor(
                max_workers=max(1, min(8, (os.cpu_count() or 1))),
                thread_name_prefix="composite")
        return _COMPOSITE_POOL


def _hex_to_rgb01(hex_color: str) -> np.ndarray:
    """'#20ADF8' / '20ADF8' -> float RGB in [0, 1]. Fail loud on a malformed color."""
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        raise ValueError(f"channel display color {hex_color!r} is not a 6-digit hex RGB.")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def channel_tint01(channel) -> np.ndarray:
    """A channel's composite tint in [0, 1]: the measured stain LUT's mid stop when one exists
    (a color channel recorded gray — see ``_stain``), else its resolved ``display_color``."""
    lut = channel.get("display_lut") if hasattr(channel, "get") else None
    if lut:
        return np.asarray(lut[len(lut) // 2][:3], dtype=np.float32)
    return _hex_to_rgb01(channel["display_color"])


def composite(store: np.ndarray, colors: np.ndarray, windows, mask=None, luts=None) -> np.ndarray:
    """Window each channel of a ``(C, H, W)`` stack and add it into one ``(H, W, 3)`` uint8 RGB.

    The single home of the window-multiply-sum loop. *windows* is one ``(lo, hi)`` per
    channel; *mask* is a per-channel bool (None = every channel on). *luts* is one optional
    per-channel colormap (an ``(N, 3)``-shaped sequence of stops): a channel carrying one maps
    each windowed value THROUGH it instead of tinting — the brightfield/stain mode, where the
    background belongs at white and additive-from-black would wash the cell out.
    """
    n_ch, h, w = store.shape
    if h == 0 or w == 0:
        return np.zeros((h, w, 3), np.uint8)
    colors = np.ascontiguousarray(colors[:n_ch], dtype=np.float32)
    lut_arrs = [None] * n_ch
    for ch in range(n_ch):
        lut = luts[ch] if luts is not None and ch < len(luts) else None
        if lut is not None:
            lut_arrs[ch] = np.ascontiguousarray(np.asarray(lut, dtype=np.float32)[:, :3])
    out = np.empty((h, w, 3), np.uint8)
    n_bands = max(1, min(_composite_pool()._max_workers, (h * w) // _COMPOSITE_MIN_PX_PER_BAND))
    n_bands = min(n_bands, h)
    edges = [(i * h) // n_bands for i in range(n_bands)] + [h]
    rows = [slice(edges[i], edges[i + 1]) for i in range(n_bands)]
    work = lambda r: _composite_band(store, colors, windows, mask, lut_arrs, out, r)   # noqa: E731
    if n_bands == 1:
        work(rows[0])
    else:
        # Bands write disjoint row slices; list() re-raises any band's exception here.
        list(_composite_pool().map(work, rows))
    return out


def _composite_band(store, colors, windows, mask, lut_arrs, out, rows: slice) -> None:
    """Composite one horizontal band of rows into ``out[rows]``."""
    n_ch = store.shape[0]
    sub = store[:, rows]
    bh, bw = sub.shape[1], sub.shape[2]
    n = bh * bw
    gray = np.zeros((n_ch, n), np.float32)          # zero == "masked off contributes nothing"
    rgb = np.zeros((n, 3), np.float32)
    lut_dtype = store.dtype
    for ch in range(n_ch):
        if mask is not None and not mask[ch]:
            continue
        lo, hi = windows[ch]
        table = _window_lut(lut_dtype, lo, hi)
        plane = sub[ch]
        if table is None:
            norm = _window(plane, lo, hi).reshape(-1)
        else:
            # table[idx] beats np.take here: take carries a bounds-check path.
            norm = table[plane.reshape(-1)]
        lut = lut_arrs[ch]
        if lut is None:
            gray[ch] = norm
        else:                                       # per-pixel colormap, not a tint
            # nan_to_num + clip: a degenerate window (lo == hi) yields NaN norms, and a NaN
            # cast to intp is an out-of-range index.
            idx = np.clip(np.nan_to_num(norm) * (lut.shape[0] - 1),
                          0, lut.shape[0] - 1).astype(np.intp)
            rgb += lut[idx]
    # einsum, NEVER a BLAS gemm: many band threads calling OpenBLAS at once exhausted its
    # internal buffer pool on a many-core Windows machine (access violation at this line).
    rgb += np.einsum("cn,cd->nd", gray, colors)     # (n, 3) float32
    np.clip(rgb, 0.0, 1.0, out=rgb)
    rgb *= 255.0
    out[rows] = rgb.reshape(bh, bw, 3).astype(np.uint8)
