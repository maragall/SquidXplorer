"""The contrast window for FLUORESCENCE, ported from maragall/stitcher.

Julio: "both panes still look blown out... port it."

WHY A PERCENTILE WINDOW WASHES FLUORESCENCE OUT
-----------------------------------------------
A fluorescence plane is mostly background. On the 10x tissue set the background peak sits well
above zero (camera offset plus autofluorescence), and the signal is a thin tail above it. Take a
1st-percentile low end and you land INSIDE the background distribution: everything from the
background peak upward is then mapped into the visible range, so the whole field lifts off black
and the tissue saturates. Composite four such channels additively and they sum to white. That is
exactly the "channel blending still sucks" / "both panes look blown out" complaint.

THE RULE, WHICH IS NOT OURS
---------------------------
``maragall/stitcher`` (``gui/app.py:auto_contrast``) already solved this and is the tool Julio has
been using, so this is a PORT, not a design:

    lo = histogram mode + 2 * std(background)     background peak pushed to BLACK
    hi = 99.9th percentile                        signal top, hot pixels clipped

The mode of a 256-bin histogram is the background peak. Adding two standard deviations of the
below-median pixels puts the low end just above the noise, so background renders black and only
real signal lifts. Both numbers and the pmax default are the stitcher's; the only changes are the
ones noted in the docstrings below, each of which turns a silent behaviour into a stated one.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

#: Bins for the background-peak histogram. The stitcher's number.
_BINS = 256

#: Pixels sampled before computing the window. The stitcher's number: enough for a stable mode
#: on a 256-bin histogram, small enough that this never shows up in a profile.
_SAMPLE = 100_000

#: If the resulting window is narrower than this, it is treated as degenerate. The stitcher's.
_MIN_SPAN = 10.0
_FALLBACK_SPAN = 100.0


def auto_contrast(data: Any, pmax: float = 99.9,
                  rng: Optional[np.random.Generator] = None) -> Optional[tuple[float, float]]:
    """``(lo, hi)`` for one fluorescence plane: background peak to black, 99.9th percentile on top.

    Returns None -- rather than a guess -- when the plane carries no usable window: empty, all
    NaN, or completely flat. This is the one deliberate divergence from the stitcher, which
    returns ``[lo, lo + 100]`` for a flat plane. A blank channel given a 100-wide window renders
    as full-intensity noise, i.e. it reads as SIGNAL; None lets the caller leave napari to its own
    autoscale and say nothing false. `_pct_window` in `_viewer` makes the same choice for the same
    reason and its docstring says so.

    *rng* lets a caller override the sampling; production passes nothing and gets a seeded
    generator, so the same plane always yields the same window.
    """
    a = np.asarray(data)
    if a.size == 0:
        return None
    flat = a.ravel()
    # Drop non-finite values BEFORE sampling: one NaN poisons the histogram, the median and the
    # percentile, and the result is a window of (nan, nan) that napari accepts and renders black.
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    if flat.size > _SAMPLE:
        # SEEDED, unlike the stitcher's bare np.random.choice. Same plane -> same window, every
        # time. Unseeded sampling moved the low end by ~6% between calls on a 1M-pixel plane
        # (measured), which means the same region could come up looking different on two loads
        # and a screenshot could not be reproduced. The sample is still unbiased; it is just no
        # longer a source of run-to-run variation nobody can account for.
        gen = rng if rng is not None else np.random.default_rng(0)
        flat = gen.choice(flat, _SAMPLE, replace=False)

    hist, edges = np.histogram(flat, bins=_BINS)
    mode_idx = int(np.argmax(hist))
    mode_val = float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)

    background = flat[flat <= np.median(flat)]
    bg_std = float(np.std(background)) if background.size else abs(mode_val) * 0.1

    lo = mode_val + 2.0 * bg_std
    hi = float(np.percentile(flat, pmax))          # the SAMPLE, not the whole plane -- see below

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi - lo < _MIN_SPAN:
        # Flat or near-flat. The stitcher widens to lo+100 here; we refuse instead, so a blank
        # channel is not handed a window that makes noise look like signal.
        if float(np.ptp(flat)) < _MIN_SPAN:
            return None
        hi = lo + _FALLBACK_SPAN
    return float(lo), float(hi)


def dtype_range(dtype: Any) -> tuple[float, float]:
    """The full display range of a dtype, for napari's ``contrast_limits_range``.

    Without this the slider spans only the window we set, so a user who wants to open the window
    up cannot: the control silently bounds them to the range we chose. Ported from the stitcher's
    ``dtype_range``, and the reason it sets ``contrast_limits_range`` right after every add.
    """
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return float(info.min), float(info.max)
    if np.issubdtype(dt, np.floating):
        return 0.0, 1.0
    return 0.0, 65535.0


def sample_plane(levels: Any) -> Optional[np.ndarray]:
    """The cheapest representative plane to derive a window from: coarsest level, middle z.

    The stitcher reads ``pyramid[-1]`` (the smallest level), middle T, then middle Z, precisely so
    that computing a window costs nothing against a lazy store. We have the same pyramid, so we
    make the same choice: on the 10x set the coarsest level is ~956x799 against level 0's
    5731x4793, i.e. ~36x fewer pixels, and it is the level napari is already fetching to draw the
    layer thumbnail.

    MIDDLE z, not z=0: the first plane of a z-stack is routinely out of focus, and a window
    derived from an out-of-focus plane is derived from blur.

    Accepts a level list (multiscale) or a single array, and returns None if nothing usable is
    there -- a caller must not have to sniff which shape it was handed.
    """
    if levels is None:
        return None
    arr = levels[-1] if isinstance(levels, (list, tuple)) else levels
    if arr is None:
        return None
    a = np.asarray(arr[arr.shape[0] // 2]) if getattr(arr, "ndim", 0) == 3 else np.asarray(arr)
    return a if a.size else None
