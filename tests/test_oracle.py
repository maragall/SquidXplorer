"""Unit tests for IMA-211's stitch acceptance oracle (``squidmip/_oracle.py``).

The oracle grades a stitcher; these tests grade the oracle. The load-bearing property is
**discrimination**: a metric that passes everything is worse than no metric, so every
positive assertion here has a matching negative one proving the gate actually fails when it
should.

Covered:
  * cut_fixture — geometry, dtype preservation, ground truth, loud refusals.
  * paste — bit-exact reconstruction at truth; dtype preserved; loud refusals.
  * seam_ratio — near 1 when aligned, high when misplaced (the discrimination proof).
  * overlap_texture — flags a blank overlap strip (oracle #5 precondition).
  * grade_positions — passes a perfect stitcher, fails a lazy one that trusts nominal,
    and names every reason.

Fixtures are synthetic but shaped like the real thing: uint16, and the default overlap is
0.09, the fraction measured on ~/Downloads/20x_scan_2025-09-05_17-57-50.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip._oracle import (
    coverage,
    cut_fixture,
    grade_positions,
    overlap_texture,
    paste,
    seam_ratio,
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _textured(h=400, w=400, dtype=np.uint16, seed=7):
    """An image with structure at every scale — what registration needs to exist."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    smooth = (
        np.sin(xx / 11.0) * np.cos(yy / 13.0) * 4000.0
        + np.sin((xx + yy) / 29.0) * 3000.0
        + 20000.0
    )
    grain = rng.normal(0.0, 700.0, size=(h, w))
    return np.clip(smooth + grain, 0, 65535).astype(dtype)


# --------------------------------------------------------------------------------------
# cut_fixture
# --------------------------------------------------------------------------------------
def test_cut_fixture_shape_dtype_and_truth():
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=20, seed=1)

    assert len(fx.tiles) == 4
    assert fx.grid == (2, 2)
    assert all(t.dtype == np.uint16 for t in fx.tiles), "dtype must never be upcast"
    assert all(t.shape == fx.tiles[0].shape for t in fx.tiles)
    assert fx.nominal.shape == (4, 2) and fx.truth.shape == (4, 2)
    # The injected error is exactly truth - nominal, and it is non-trivial.
    assert np.array_equal(fx.offsets, fx.truth - fx.nominal)
    assert np.abs(fx.offsets).max() > 0, "a fixture with no error tests nothing"
    assert np.abs(fx.offsets).max() <= 20


def test_cut_fixture_overlap_is_nonzero_and_matches_request():
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.25, max_offset_px=10, seed=2)
    th, tw = fx.tiles[0].shape
    oy, ox = fx.overlap_px
    assert oy > 0 and ox > 0
    # overlap_px should be ~25% of the tile, within rounding of one pixel.
    assert abs(ox - round(tw * 0.25)) <= 1
    assert abs(oy - round(th * 0.25)) <= 1


def test_cut_fixture_explicit_zero_offsets_is_perfectly_aligned():
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), offsets=np.zeros((4, 2), dtype=int))
    assert np.array_equal(fx.truth, fx.nominal)
    assert np.abs(fx.offsets).max() == 0


def test_cut_fixture_six_by_six_matches_real_acquisition_shape():
    # The 20x scan is 36 FOV/well in a 6x6 grid — the real case, not the 2x2 the first
    # review pass assumed. The oracle must handle it.
    src = _textured(h=900, w=900)
    fx = cut_fixture(src, grid=(6, 6), overlap_frac=0.25, max_offset_px=8, seed=3)
    assert len(fx.tiles) == 36
    assert fx.grid == (6, 6)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(overlap_frac=1.0), "overlap_frac"),
        (dict(overlap_frac=-0.1), "overlap_frac"),
        (dict(grid=(0, 2)), "grid"),
    ],
)
def test_cut_fixture_refuses_bad_parameters_loudly(kwargs, match):
    src = _textured(h=200, w=200)
    with pytest.raises(ValueError, match=match):
        cut_fixture(src, **kwargs)


def test_cut_fixture_refuses_non_2d_source():
    with pytest.raises(ValueError, match="2-D"):
        cut_fixture(np.zeros((4, 4, 3), dtype=np.uint16))


def test_cut_fixture_refuses_source_too_small():
    with pytest.raises(ValueError, match="too small"):
        cut_fixture(np.zeros((40, 40), dtype=np.uint16), grid=(6, 6), max_offset_px=20)


def test_cut_fixture_refuses_wrong_offsets_shape():
    src = _textured(h=200, w=200)
    with pytest.raises(ValueError, match="offsets"):
        cut_fixture(src, grid=(2, 2), offsets=np.zeros((3, 2), dtype=int))


# --------------------------------------------------------------------------------------
# paste — oracle #1's bit-exactness
# --------------------------------------------------------------------------------------
def test_paste_at_truth_is_bit_exact_to_source():
    """Oracle #1: recover the true positions and the source comes back exactly.

    Judged over the covered region — offsetting tiles leaves the canvas corners as padding,
    which is not a reconstruction failure.
    """
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=15, seed=5)
    out = paste(fx.tiles, fx.truth)
    mask = coverage(fx.tiles, fx.truth)
    ref = fx.source[: out.shape[0], : out.shape[1]]
    assert out.dtype == np.uint16
    assert np.array_equal(out[mask], ref[mask])


def test_paste_at_nominal_is_not_bit_exact_when_offsets_injected():
    """The negative half: trusting the nominal grid must NOT reproduce the source."""
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=15, seed=5)
    out = paste(fx.tiles, fx.nominal)
    mask = coverage(fx.tiles, fx.nominal)
    ref = fx.source[: out.shape[0], : out.shape[1]]
    assert not np.array_equal(out[mask], ref[mask])


def test_paste_preserves_dtype_and_refuses_mismatches():
    a = np.ones((4, 4), dtype=np.uint16)
    b = np.ones((4, 4), dtype=np.uint8)
    assert paste([a, a], [[0, 0], [0, 4]]).dtype == np.uint16
    with pytest.raises(ValueError, match="dtype"):
        paste([a, b], [[0, 0], [0, 4]])


def test_paste_refuses_empty_negative_and_non_2d():
    a = np.ones((4, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match="no tiles"):
        paste([], np.zeros((0, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        paste([a], [[-1, 0]])
    with pytest.raises(ValueError, match="2-D"):
        paste([np.ones((2, 2, 2), dtype=np.uint16)], [[0, 0]])


# --------------------------------------------------------------------------------------
# seam_ratio — oracle #2, and the proof it discriminates
# --------------------------------------------------------------------------------------
def test_seam_ratio_near_one_when_aligned():
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=15, seed=9)
    th, tw = fx.tiles[0].shape
    ratio = seam_ratio(
        paste(fx.tiles, fx.truth), fx.truth, (th, tw), mask=coverage(fx.tiles, fx.truth)
    )
    assert ratio <= 1.5, f"a correctly stitched mosaic must pass the gate, got {ratio:.2f}"


def test_seam_ratio_spikes_when_misplaced():
    """If this ever stops failing, the gate is worthless — that is the point of the test."""
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=15, seed=9)
    th, tw = fx.tiles[0].shape
    aligned = seam_ratio(
        paste(fx.tiles, fx.truth), fx.truth, (th, tw), mask=coverage(fx.tiles, fx.truth)
    )
    misplaced = seam_ratio(
        paste(fx.tiles, fx.nominal), fx.nominal, (th, tw), mask=coverage(fx.tiles, fx.nominal)
    )
    assert misplaced > aligned
    assert misplaced > 1.5, f"misplaced mosaic must FAIL the gate, got {misplaced:.2f}"


def test_seam_ratio_flat_image_reports_inf_not_a_fudged_number():
    flat = np.full((60, 60), 1000, dtype=np.uint16)
    pos = np.array([[0, 0], [0, 30]])
    assert seam_ratio(flat, pos, (60, 30)) == float("inf")


def test_seam_ratio_single_tile_has_no_seams():
    src = _textured(h=100, w=100)
    assert seam_ratio(src, np.array([[0, 0]]), (100, 100)) == 1.0


# --------------------------------------------------------------------------------------
# overlap_texture — oracle #5's precondition
# --------------------------------------------------------------------------------------
def test_overlap_texture_positive_on_textured_data():
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=10, seed=11)
    assert (overlap_texture(fx) > 0).all()


def test_overlap_texture_zero_on_blank_overlap():
    """A blank seam carries no registration signal — the oracle must say so, loudly."""
    flat = np.full((400, 400), 500, dtype=np.uint16)
    fx = cut_fixture(flat, grid=(2, 2), overlap_frac=0.20, max_offset_px=10, seed=12)
    tex = overlap_texture(fx)
    assert (tex == 0).all(), "blank overlap must report zero texture, not a small number"


# --------------------------------------------------------------------------------------
# grade_positions — the gate itself
# --------------------------------------------------------------------------------------
def test_grade_passes_a_perfect_stitcher():
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=15, seed=13)
    g = grade_positions(fx.truth, fx)
    assert g.passed, g.reasons
    assert g.max_error_px == 0.0
    assert g.bit_exact


def test_grade_fails_a_stitcher_that_just_trusts_nominal():
    """The lazy implementation — ignore the pixels, return the grid — must not pass."""
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=15, seed=13)
    g = grade_positions(fx.nominal, fx)
    assert not g.passed
    assert g.max_error_px > 1.0
    assert not g.bit_exact
    assert any("placement error" in r for r in g.reasons)


def test_grade_tolerates_sub_pixel_error_within_threshold():
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=15, seed=14)
    nudged = fx.truth.copy()
    nudged[1, 1] += 1  # 1 px out — exactly at the documented tolerance
    g = grade_positions(nudged, fx)
    assert g.max_error_px == 1.0
    assert not any("placement error" in r for r in g.reasons)


def test_grade_reports_every_failed_gate_not_just_the_first():
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.25, max_offset_px=20, seed=15)
    g = grade_positions(fx.nominal, fx, max_error_px=1.0, max_seam_ratio=1.5)
    assert not g.passed
    assert len(g.reasons) >= 2, f"expected placement AND seam failures, got {g.reasons}"


def test_grade_refuses_wrong_shaped_estimate():
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=10)
    with pytest.raises(ValueError, match="estimated"):
        grade_positions(np.zeros((3, 2), dtype=int), fx)


def test_grade_str_is_readable():
    src = _textured()
    fx = cut_fixture(src, grid=(2, 2), overlap_frac=0.20, max_offset_px=10, seed=16)
    assert str(grade_positions(fx.truth, fx)).startswith("PASS")
    assert str(grade_positions(fx.nominal, fx)).startswith("FAIL")
