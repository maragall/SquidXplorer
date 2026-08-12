"""The contrast window for fluorescence: background peak to black, 99.9th percentile on top."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

#: Bins for the background-peak histogram.
_BINS = 256

#: Pixels sampled before computing the window.
_SAMPLE = 100_000

#: A window narrower than this is treated as degenerate.
_MIN_SPAN = 10.0
_FALLBACK_SPAN = 100.0


def auto_contrast(data: Any, pmax: float = 99.9,
                  rng: Optional[np.random.Generator] = None) -> Optional[tuple[float, float]]:
    """``(lo, hi)`` for one fluorescence plane, or None when the plane carries no usable window."""
    a = np.asarray(data)
    if a.size == 0:
        return None
    flat = a.ravel()
    # Drop non-finite values before sampling: one NaN poisons the histogram and percentile.
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    if flat.size > _SAMPLE:
        # Seeded, so the same plane always yields the same window.
        gen = rng if rng is not None else np.random.default_rng(0)
        flat = gen.choice(flat, _SAMPLE, replace=False)

    hist, edges = np.histogram(flat, bins=_BINS)
    mode_idx = int(np.argmax(hist))
    mode_val = float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)

    background = flat[flat <= np.median(flat)]
    bg_std = float(np.std(background)) if background.size else abs(mode_val) * 0.1

    lo = mode_val + 2.0 * bg_std
    hi = float(np.percentile(flat, pmax))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi - lo < _MIN_SPAN:
        # Flat or near-flat plane: refuse rather than hand a blank channel a noise-amplifying window.
        if float(np.ptp(flat)) < _MIN_SPAN:
            return None
        hi = lo + _FALLBACK_SPAN
    return float(lo), float(hi)


def dtype_range(dtype: Any) -> tuple[float, float]:
    """The full display range of a dtype, for napari's ``contrast_limits_range``."""
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return float(info.min), float(info.max)
    if np.issubdtype(dt, np.floating):
        return 0.0, 1.0
    return 0.0, 65535.0


def opening_z(n_planes: int) -> int:
    """The z index a window opens on: napari's centring rule, ``(n - 1) // 2``."""
    return max(0, (int(n_planes) - 1) // 2)


def sample_plane(levels: Any) -> Optional[np.ndarray]:
    """The cheapest representative plane to derive a window from: coarsest level, opening z."""
    if levels is None:
        return None
    arr = levels[-1] if isinstance(levels, (list, tuple)) else levels
    if arr is None:
        return None
    a = np.asarray(arr[opening_z(arr.shape[0])]) if getattr(arr, "ndim", 0) == 3 \
        else np.asarray(arr)
    return a if a.size else None
