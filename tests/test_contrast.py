"""The fluorescence contrast rule, ported from maragall/stitcher."""

from __future__ import annotations

import numpy as np

from squidxplorer._contrast import auto_contrast, dtype_range, sample_plane


def _fluorescence(bg=500.0, bg_noise=30.0, signal=8000.0, frac=0.02, shape=(256, 256), seed=0):
    """A plane shaped like real fluorescence: a noisy background pedestal, a sparse bright tail."""
    rng = np.random.default_rng(seed)
    a = rng.normal(bg, bg_noise, shape)
    n = int(a.size * frac)
    idx = rng.choice(a.size, n, replace=False)
    a.flat[idx] = rng.normal(signal, signal * 0.1, n)
    return np.clip(a, 0, 65535).astype(np.uint16)


def test_the_low_end_lands_ABOVE_the_background_so_it_renders_black():
    plane = _fluorescence(bg=500.0, bg_noise=30.0)
    lo, hi = auto_contrast(plane)

    assert lo > 500.0, f"low end {lo:.0f} is inside the background (mean 500) — it will render grey"
    assert lo < 800.0, f"low end {lo:.0f} is so high it will clip real signal"
    assert hi > 5000.0, f"high end {hi:.0f} is below the signal — the tissue will saturate"


def test_a_brighter_background_moves_the_window_with_it():
    lo_dim, _ = auto_contrast(_fluorescence(bg=300.0))
    lo_bright, _ = auto_contrast(_fluorescence(bg=3000.0))
    assert lo_bright - lo_dim > 2000.0, "the window ignored a 2700-count shift in background"


def test_a_blank_channel_gets_NO_window_rather_than_a_guess():
    """A blank channel handed a 100-wide window would render its own read noise as signal; refusing lets napari autoscale instead."""
    assert auto_contrast(np.full((64, 64), 700, dtype=np.uint16)) is None
    assert auto_contrast(np.zeros((64, 64), dtype=np.uint16)) is None
    assert auto_contrast(np.zeros((0,), dtype=np.uint16)) is None


def test_a_nan_does_not_poison_the_window():
    """A NaN makes the histogram/median/percentile all NaN, and (nan, nan) renders BLACK silently."""
    plane = _fluorescence().astype(np.float32)
    plane[0, 0] = np.nan
    win = auto_contrast(plane)
    assert win is not None and np.isfinite(win[0]) and np.isfinite(win[1])


def test_the_same_plane_always_gets_the_SAME_window():
    """Sampling is seeded. MUTATION: drop the seed from default_rng() in _contrast and this fails."""
    plane = _fluorescence(shape=(1024, 1024))          # ~1M px, so sampling really happens
    windows = [auto_contrast(plane) for _ in range(5)]
    assert len(set(windows)) == 1, f"the same plane produced different windows: {set(windows)}"


def test_dtype_range_spans_the_whole_type():
    assert dtype_range(np.uint16) == (0.0, 65535.0)
    assert dtype_range(np.uint8) == (0.0, 255.0)
    assert dtype_range(np.float32) == (0.0, 1.0)


def test_the_sample_is_the_COARSEST_level_of_a_pyramid():
    """Seeding must cost nothing: measure the small level, not the 5731x4793 one."""
    levels = [np.zeros((64, 64), np.uint16), np.ones((16, 16), np.uint16)]
    got = sample_plane(levels)
    assert got.shape == (16, 16)


def test_the_sample_is_the_MIDDLE_z_not_the_first():
    """The first plane of a z-stack is routinely out of focus."""
    stack = np.stack([np.full((8, 8), i, np.uint16) for i in range(5)])
    assert sample_plane([stack])[0, 0] == 2          # middle of 5, not 0


def test_a_plain_array_works_without_the_caller_sniffing_the_shape():
    assert sample_plane(np.ones((8, 8), np.uint16)).shape == (8, 8)
    assert sample_plane(None) is None
