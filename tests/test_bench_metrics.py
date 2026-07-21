"""The seam-residual metric — the ruler every stitcher is measured with.

The load-bearing test here is `test_injected_misalignment_is_recovered`: we hand the
metric positions that are deliberately wrong by a known amount and assert it reports
that amount. A metric that only ever returns "small" would pass a smoke test and be
useless for ranking stitchers.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.metrics import (
    MIN_BLOCK_PX,
    adjacent_pairs,
    block_shifts,
    overlap_strips,
    phase_correlate,
    seam_residual,
)
from tests.conftest_bench import make_canvas


# --------------------------------------------------------------------------- phase


@pytest.mark.parametrize("dy,dx", [(0, 0), (3, 0), (0, -4), (5, 7), (-6, 2)])
def test_phase_correlate_recovers_integer_shift(dy, dx):
    canvas = make_canvas(256, 256, seed=1)
    a = canvas[64:192, 64:192]
    b = canvas[64 + dy : 192 + dy, 64 + dx : 192 + dx]
    got_dy, got_dx = phase_correlate(a, b)
    assert got_dy == pytest.approx(dy, abs=0.35)
    assert got_dx == pytest.approx(dx, abs=0.35)


def test_phase_correlate_is_brightness_invariant():
    """Neighbouring FOVs differ in illumination; alignment must not depend on it."""
    canvas = make_canvas(256, 256, seed=2)
    a = canvas[50:178, 50:178].astype(np.float64)
    b = canvas[52:180, 53:181].astype(np.float64) * 2.5 + 300.0
    dy, dx = phase_correlate(a, b)
    assert dy == pytest.approx(2, abs=0.35)
    assert dx == pytest.approx(3, abs=0.35)


def test_phase_correlate_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        phase_correlate(np.zeros((8, 8)), np.zeros((8, 9)))


def test_phase_correlate_on_flat_input_does_not_crash():
    dy, dx = phase_correlate(np.zeros((32, 32)), np.zeros((32, 32)))
    assert np.isfinite(dy) and np.isfinite(dx)


# ------------------------------------------------------------------------- overlap


def test_overlap_strips_are_the_same_pixels_when_positions_are_exact():
    canvas = make_canvas(200, 300, seed=3)
    tile_a = canvas[0:128, 0:128]
    tile_b = canvas[0:128, 96:224]  # 32 px overlap, offset (0, 96)
    strips = overlap_strips(tile_a, tile_b, (0, 96))
    assert strips is not None
    sa, sb = strips
    assert sa.shape == sb.shape
    np.testing.assert_array_equal(sa, sb)


def test_overlap_strips_handles_negative_offset():
    canvas = make_canvas(200, 300, seed=4)
    tile_a = canvas[0:128, 96:224]
    tile_b = canvas[0:128, 0:128]
    strips = overlap_strips(tile_a, tile_b, (0, -96))
    assert strips is not None
    np.testing.assert_array_equal(*strips)


def test_overlap_strips_returns_none_when_tiles_are_disjoint():
    a = np.zeros((64, 64), dtype=np.uint16)
    assert overlap_strips(a, a, (0, 1000)) is None


def test_overlap_strips_returns_none_when_sliver_is_too_thin():
    a = np.zeros((64, 64), dtype=np.uint16)
    assert overlap_strips(a, a, (0, 64 - (MIN_BLOCK_PX - 1))) is None


# -------------------------------------------------------------------------- blocks


def test_block_shifts_reports_near_zero_for_aligned_strips():
    canvas = make_canvas(64, 512, seed=5)
    shifts = block_shifts(canvas, canvas.copy(), n_blocks=4)
    assert shifts.size >= 2
    assert np.median(shifts) < 0.5


def test_block_shifts_recovers_a_known_offset():
    canvas = make_canvas(64, 600, seed=6)
    a = canvas[:, 0:512]
    b = canvas[:, 3:515]  # shifted by 3 px
    shifts = block_shifts(a, b, n_blocks=4)
    assert shifts.size >= 2
    assert np.median(shifts) == pytest.approx(3.0, abs=0.5)


def test_block_shifts_drops_blank_blocks_instead_of_scoring_them_zero():
    """A blank block correlates to nothing. Counting it as 0 px would flatter the tool."""
    flat = np.zeros((64, 512), dtype=np.uint16)
    assert block_shifts(flat, flat.copy(), n_blocks=4).size == 0


def test_block_shifts_gate_rejects_uncorrelated_content():
    a = make_canvas(64, 512, seed=7)
    b = make_canvas(64, 512, seed=8)  # unrelated texture
    assert block_shifts(a, b, n_blocks=4, ncc_min=0.9).size == 0


def test_block_shifts_returns_empty_when_strip_is_too_small():
    tiny = make_canvas(4, 4, seed=9)
    assert block_shifts(tiny, tiny.copy(), n_blocks=8).size == 0


# --------------------------------------------------------------------------- pairs


def test_adjacent_pairs_finds_overlapping_neighbours_only():
    positions = {0: (0.0, 0.0), 1: (0.0, 96.0), 2: (0.0, 5000.0)}
    pairs = adjacent_pairs(positions, (128, 128))
    assert (0, 1) in pairs
    assert (0, 2) not in pairs and (1, 2) not in pairs


def test_adjacent_pairs_respects_the_minimum_overlap_fraction():
    positions = {0: (0.0, 0.0), 1: (0.0, 127.0)}  # 1 px sliver
    assert adjacent_pairs(positions, (128, 128), min_overlap_fraction=0.02) == []


def test_adjacent_pairs_on_a_full_grid():
    positions = {i * 3 + j: (i * 96.0, j * 96.0) for i in range(2) for j in range(3)}
    pairs = adjacent_pairs(positions, (128, 128))
    assert len(pairs) >= 7  # 4 horizontal + 3 vertical at minimum


# ------------------------------------------------------------------- aggregate


def _grid_reader(canvas, truth, tile):
    th, tw = tile

    def read(fov):
        y, x = truth[fov]
        return canvas[int(y) : int(y) + th, int(x) : int(x) + tw]

    return read


def test_seam_residual_is_near_zero_for_perfect_positions():
    tile, step, grid = (128, 128), (96, 96), (2, 3)
    canvas = make_canvas(128 + 96, 128 + 2 * 96, seed=10)
    truth = {i * 3 + j: (i * 96.0, j * 96.0) for i in range(2) for j in range(3)}
    stats = seam_residual(_grid_reader(canvas, truth, tile), truth, frame_shape=tile)
    assert stats["n_seams_measured"] >= 4
    assert stats["resid_median_px"] < 0.5


def test_injected_misalignment_is_recovered():
    """The proof the ruler works: lie about one tile's position by a known amount and
    assert the metric reports that amount. Without this, the metric could be reporting
    a constant and nobody would notice."""
    tile = (128, 128)
    canvas = make_canvas(128 + 96, 128 + 2 * 96, seed=11)
    truth = {i * 3 + j: (i * 96.0, j * 96.0) for i in range(2) for j in range(3)}
    read = _grid_reader(canvas, truth, tile)

    wrong = dict(truth)
    wrong[1] = (truth[1][0] + 4.0, truth[1][1] - 3.0)  # |(4,-3)| = 5 px

    perfect = seam_residual(read, truth, frame_shape=tile)
    degraded = seam_residual(read, wrong, frame_shape=tile)

    assert perfect["resid_median_px"] < 0.5
    assert degraded["resid_p90_px"] == pytest.approx(5.0, abs=1.0)
    assert degraded["resid_median_px"] > perfect["resid_median_px"]


def test_seam_residual_reports_zero_seams_on_blank_data():
    """Blank tiles must produce 'could not measure', never a flattering 0 px."""
    tile = (128, 128)
    canvas = np.zeros((128 + 96, 128 + 2 * 96), dtype=np.uint16)
    truth = {i * 3 + j: (i * 96.0, j * 96.0) for i in range(2) for j in range(3)}
    stats = seam_residual(_grid_reader(canvas, truth, tile), truth, frame_shape=tile)
    assert stats["n_seams_measured"] == 0
    assert np.isnan(stats["resid_median_px"])


def test_seam_residual_with_no_overlapping_pairs():
    positions = {0: (0.0, 0.0), 1: (0.0, 9999.0)}
    stats = seam_residual(lambda f: make_canvas(64, 64), positions, frame_shape=(64, 64))
    assert stats["n_seams_measured"] == 0
    assert stats["n_pairs_candidate"] == 0


def test_seam_residual_requires_pairs_or_frame_shape():
    with pytest.raises(ValueError, match="pairs or frame_shape"):
        seam_residual(lambda f: None, {0: (0.0, 0.0)})


def test_seam_residual_skips_fovs_absent_from_positions():
    tile = (128, 128)
    canvas = make_canvas(128, 128 + 96, seed=12)
    truth = {0: (0.0, 0.0), 1: (0.0, 96.0)}
    stats = seam_residual(
        _grid_reader(canvas, truth, tile), {0: (0.0, 0.0)}, pairs=[(0, 1)]
    )
    assert stats["n_seams_measured"] == 0
