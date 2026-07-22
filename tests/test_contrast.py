"""The fluorescence contrast rule, ported from maragall/stitcher.

Julio, looking at the live GUI with both panes finally agreeing: "the channels are not well
contrast-adjusted (background looks colored)". They agreed on a window that was wrong in the same
way in both places.

The property under test is the one that matters on screen: **background renders BLACK**. A
percentile low end lands inside the background distribution of a fluorescence plane, so the whole
field lifts off black and four additive channels sum to white. These tests build a plane with a
known background and a known signal, and check the window separates them.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip._contrast import auto_contrast, dtype_range, sample_plane


def _fluorescence(bg=500.0, bg_noise=30.0, signal=8000.0, frac=0.02, shape=(256, 256), seed=0):
    """A plane shaped like real fluorescence: a big noisy background pedestal, a sparse bright tail.

    The pedestal is the point. Real fluorescence background is NOT near zero -- camera offset plus
    autofluorescence put it at some few hundred counts -- which is exactly why a 1st percentile
    low end fails: the 1st percentile of this plane is still background.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(bg, bg_noise, shape)
    n = int(a.size * frac)
    idx = rng.choice(a.size, n, replace=False)
    a.flat[idx] = rng.normal(signal, signal * 0.1, n)
    return np.clip(a, 0, 65535).astype(np.uint16)


def test_the_low_end_lands_ABOVE_the_background_so_it_renders_black():
    """THE defect, stated as a number.

    A window whose low end sits inside the background maps background to visible grey. Here the
    background is 500 +/- 30, so a correct low end is above ~500 and the tissue is what lifts.
    """
    plane = _fluorescence(bg=500.0, bg_noise=30.0)
    lo, hi = auto_contrast(plane)

    assert lo > 500.0, f"low end {lo:.0f} is inside the background (mean 500) — it will render grey"
    assert lo < 800.0, f"low end {lo:.0f} is so high it will clip real signal"
    assert hi > 5000.0, f"high end {hi:.0f} is below the signal — the tissue will saturate"


def test_a_percentile_window_is_what_it_beats():
    """Pins the comparison rather than asserting the port is better in prose.

    The 1st percentile of a fluorescence plane is BACKGROUND, and that is the whole problem.
    """
    plane = _fluorescence(bg=500.0, bg_noise=30.0)
    pct_lo = float(np.percentile(plane, 1.0))
    auto_lo, _ = auto_contrast(plane)

    assert pct_lo < 500.0, "fixture is wrong: the 1st percentile should be inside the background"
    assert auto_lo > pct_lo, (
        f"the ported rule ({auto_lo:.0f}) must start ABOVE the percentile rule ({pct_lo:.0f}); "
        "starting below is what made the background visible"
    )


def test_the_fraction_of_pixels_left_visible_is_small():
    """The end result, measured the way the eye sees it: on a plane that is 2% signal, only a
    few percent of pixels should be above the low end. A washed-out window leaves most of the
    field visible, which is precisely what 'background looks colored' means."""
    plane = _fluorescence(frac=0.02)
    lo, _hi = auto_contrast(plane)
    visible = float((plane > lo).mean())
    assert visible < 0.10, f"{visible:.0%} of the field is above black; the background is showing"


def test_a_brighter_background_moves_the_window_with_it():
    """It tracks the data, not a constant. Two planes differing only in pedestal must get
    windows that differ by about that pedestal."""
    lo_dim, _ = auto_contrast(_fluorescence(bg=300.0))
    lo_bright, _ = auto_contrast(_fluorescence(bg=3000.0))
    assert lo_bright - lo_dim > 2000.0, "the window ignored a 2700-count shift in background"


def test_a_blank_channel_gets_NO_window_rather_than_a_guess():
    """The deliberate divergence from the stitcher, which returns lo..lo+100 here.

    A blank channel handed a 100-wide window renders its own read noise at full intensity, i.e.
    it reads as SIGNAL. Refusing lets napari autoscale and says nothing false. `_pct_window` in
    `_viewer` makes the same call for the same reason.
    """
    assert auto_contrast(np.full((64, 64), 700, dtype=np.uint16)) is None
    assert auto_contrast(np.zeros((64, 64), dtype=np.uint16)) is None
    assert auto_contrast(np.zeros((0,), dtype=np.uint16)) is None


def test_a_nan_does_not_poison_the_window():
    """One NaN makes the histogram, the median and the percentile all NaN, and a (nan, nan)
    window is accepted by napari and renders BLACK — a blank pane with no error."""
    plane = _fluorescence().astype(np.float32)
    plane[0, 0] = np.nan
    win = auto_contrast(plane)
    assert win is not None and np.isfinite(win[0]) and np.isfinite(win[1])


def test_the_same_plane_always_gets_the_SAME_window():
    """Sampling is seeded, so a window is reproducible.

    Found by this test failing: with the stitcher's unseeded np.random.choice the low end moved
    ~6% between calls on a 1M-pixel plane. Small enough to be invisible, big enough that the same
    region could come up looking different on two loads with nothing to account for it, and that
    a screenshot could not be reproduced.

    MUTATION: take the seed out of default_rng() in _contrast and this goes red.
    """
    plane = _fluorescence(shape=(1024, 1024))          # ~1M px, so sampling really happens
    windows = [auto_contrast(plane) for _ in range(5)]
    assert len(set(windows)) == 1, f"the same plane produced different windows: {set(windows)}"


def test_dtype_range_spans_the_whole_type():
    assert dtype_range(np.uint16) == (0.0, 65535.0)
    assert dtype_range(np.uint8) == (0.0, 255.0)
    assert dtype_range(np.float32) == (0.0, 1.0)


# --- picking the plane to measure ---------------------------------------------------------

def test_the_sample_is_the_COARSEST_level_of_a_pyramid():
    """Seeding must cost nothing: measure the small level, not the 5731x4793 one."""
    levels = [np.zeros((64, 64), np.uint16), np.ones((16, 16), np.uint16)]
    got = sample_plane(levels)
    assert got.shape == (16, 16)


def test_the_sample_is_the_MIDDLE_z_not_the_first():
    """The first plane of a z-stack is routinely out of focus, and a window derived from an
    out-of-focus plane is derived from blur."""
    stack = np.stack([np.full((8, 8), i, np.uint16) for i in range(5)])
    assert sample_plane([stack])[0, 0] == 2          # middle of 5, not 0


def test_a_plain_array_works_without_the_caller_sniffing_the_shape():
    assert sample_plane(np.ones((8, 8), np.uint16)).shape == (8, 8)
    assert sample_plane(None) is None
