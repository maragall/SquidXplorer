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
#: Foreground membership for the ceiling: this many background sigmas above the mode, and
#: at least this many sampled pixels (a lone hot pixel is not a population).
_FOREGROUND_SIGMA = 5.0
#: The background population for the floor: pixels under this many sigmas above the mode,
#: and the percentile of it the floor sits at (< 1% of background renders above black).
_BACKGROUND_SIGMA = 6.0
_BACKGROUND_PCT = 99.5
_FOREGROUND_MIN_PX = 16
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
    # THE FLOOR SITS ABOVE THE BACKGROUND (Julio, 2026-08-25: the 561 background rendered as
    # yellow speckle). Measured on G7 561 / FOV 1 / z 7 (mode 896, sigma 18.7): mode + 2 sigma
    # leaves 16.7% of background pixels above black, + 3 sigma 7.7%, + 4 sigma 3.0%; the 99th
    # percentile of the background population (pixels under mode + 6 sigma) leaves 0.99%.
    # The below-median sigma is a half-normal under-estimate, so the percentile is the rule.
    background_population = flat[flat < mode_val + _BACKGROUND_SIGMA * bg_std]
    if background_population.size:
        lo = max(lo, float(np.percentile(background_population, _BACKGROUND_PCT)))
    hi = float(np.percentile(flat, pmax))
    # THE CEILING IS FOREGROUND-AWARE (Julio, 2026-08-25: "the napari autocontrast SUCKS for
    # the G7 dataset"; a sparse field: bright cells on black). On a plane whose objects are
    # a fraction of a percent of the pixels, a plain 99.9th percentile sits in the noise and
    # clips every object; napari's min/max lets one hot pixel set the ceiling and renders the
    # objects dim. The ceiling is the 99th percentile of the pixels ABOVE the background floor
    # when that is higher: a lone hot pixel cannot carry it, an object population can.
    # The population well clear of the background (5 sigma: a 2 sigma cut is mostly the
    # noise tail on a sparse field), and large enough that a lone hot pixel cannot form it.
    foreground = flat[flat > mode_val + _FOREGROUND_SIGMA * bg_std]
    if foreground.size >= _FOREGROUND_MIN_PX:
        hi = max(hi, float(np.percentile(foreground, 99.5)))

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


#: The finest rung a SEED window may be measured on: a ~2k x 2k plane (one full-resolution
#: field). Julio, 2026-08-25: a sparse field's objects are attenuated up to 1/s^2 on the
#: coarsest rung, so the first paint of G7 was dim; the seed reads the finest rung this
#: budget allows, off the Qt thread like every window.
SEED_MAX_PX = 4_200_000


def sample_plane(levels: Any, max_px: Optional[int] = None) -> Optional[np.ndarray]:
    """The cheapest representative plane to derive a window from: coarsest level, opening z,
    or, given *max_px*, the FINEST level whose plane fits that many pixels.

    RIGHT FOR A WINDOW, WRONG FOR A CEILING. The coarsest level is mean-downsampled, so its
    maximum is an UNDER-estimate of level 0's -- a hot pixel at stride ``s`` is attenuated by up
    to ``1/s**2``. That is harmless for a 99.9th-percentile window and fatal for
    ``contrast_limits_range``, where an under-estimate snaps the slider below real data and clips
    it. :mod:`squidxplorer._bitdepth` therefore measures the ceiling from full-resolution frames
    in :mod:`squidxplorer._mosaic_source` and never from here. Do not wire this function to it.
    """
    if levels is None:
        return None
    arr = levels[-1] if isinstance(levels, (list, tuple)) else levels
    if max_px is not None and isinstance(levels, (list, tuple)):
        for candidate in levels:                 # finest first
            shape = getattr(candidate, "shape", None)
            if shape is None or len(shape) < 2:
                continue
            if int(shape[-1]) * int(shape[-2]) <= int(max_px):
                arr = candidate
                break
    if arr is None:
        return None
    a = np.asarray(arr[opening_z(arr.shape[0])]) if getattr(arr, "ndim", 0) == 3 \
        else np.asarray(arr)
    return a if a.size else None
