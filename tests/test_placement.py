"""IMA-187 coordinate placement — exact pixel arithmetic, no GUI.

These are the tests that matter most for the mosaic, because placement bugs do not raise:
they draw a plausible-but-wrong picture. A test that counts tiles passes just as happily on a
vertically mirrored or uniformly mis-scaled mosaic. So every assertion here compares integer
pixel offsets against values computed by hand from the stage coordinates.

The three silent failure modes, each with a dedicated test:
    scale error  -> test_offsets_scale_exactly_with_pixel_size
    Y-axis flip  -> test_y_axis_increases_downward
    wrong origin -> test_origin_is_per_region_not_global
"""

from __future__ import annotations

import pytest

from squidxplorer._placement import (
    cell_boxes,
    fov_offsets_px,
    mosaic_extent_px,
)

PX_UM = 0.5          # 0.5 um/px -> 2000 px per mm; every expected value below is exact
FRAME = (100, 100)


def _grid_positions(region="A1", n=3, pitch_um=50.0, x0=10_000.0, y0=20_000.0):
    """n x n grid of stage positions in MICROMETRES, raster order (x fastest) — the Squid
    scan pattern. Units are um because that is what metadata["fov_positions_um"] carries;
    a mm fixture here would silently pass while the production key is 1000x different."""
    pos = {}
    fov = 0
    for r in range(n):
        for c in range(n):
            pos[(region, fov)] = (x0 + c * pitch_um, y0 + r * pitch_um)
            fov += 1
    return pos


# --- the three silent bugs ------------------------------------------------------------------

def test_offsets_scale_exactly_with_pixel_size():
    """A 50 um pitch at 0.5 um/px is exactly 100 px. Catches any scale-factor error."""
    pos = _grid_positions(n=2, pitch_um=50.0)
    off = fov_offsets_px(pos, "A1", [0, 1, 2, 3], PX_UM)
    assert off[0] == (0, 0)
    assert off[1] == (0, 100)      # +50 um in x -> +100 px in column
    assert off[2] == (100, 0)      # +50 um in y -> +100 px in row
    assert off[3] == (100, 100)


def test_offsets_halve_when_pixel_size_doubles():
    """Same stage geometry at 1.0 um/px must give exactly half the pixel offsets."""
    pos = _grid_positions(n=2, pitch_um=50.0)
    fine = fov_offsets_px(pos, "A1", [0, 1, 2, 3], PX_UM)
    coarse = fov_offsets_px(pos, "A1", [0, 1, 2, 3], PX_UM * 2)
    assert set(fine) == {0, 1, 2, 3}, sorted(fine)
    for fov in fine:
        assert coarse[fov] == (fine[fov][0] // 2, fine[fov][1] // 2)


def test_y_axis_increases_downward():
    """Larger stage y MUST map to a larger row index. A flip here mirrors the whole mosaic."""
    pos = {("A1", 0): (10_000.0, 20_000.0), ("A1", 1): (10_000.0, 20_100.0)}  # fov 1 further +y
    off = fov_offsets_px(pos, "A1", [0, 1], PX_UM)
    assert off[0][0] == 0
    assert off[1][0] > 0, "stage +y must map to +row; a negative/zero row means a Y-axis flip"
    assert off[1][0] == 200                                     # 100 um / 0.5 um = 200 px


def test_origin_is_per_region_not_global():
    """Each region is laid out in its OWN frame: both wells start at (0, 0) despite different
    absolute stage coordinates. A plate-wide origin would shift one well by a huge constant."""
    pos = {
        ("A1", 0): (10_000.0, 20_000.0), ("A1", 1): (10_050.0, 20_000.0),
        ("B2", 0): (80_000.0, 60_000.0), ("B2", 1): (80_050.0, 60_000.0),  # far away on the plate
    }
    a1 = fov_offsets_px(pos, "A1", [0, 1], PX_UM)
    b2 = fov_offsets_px(pos, "B2", [0, 1], PX_UM)
    assert a1 == b2 == {0: (0, 0), 1: (0, 100)}


# --- ordering / geometry --------------------------------------------------------------------

def test_raster_grid_lays_out_row_major():
    """A 3x3 raster acquisition places fov 0..8 left-to-right, top-to-bottom."""
    off = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM)
    assert off[0] == (0, 0)
    assert off[2] == (0, 200)      # end of the first row
    assert off[6] == (200, 0)      # start of the last row
    assert off[8] == (200, 200)


def test_offsets_are_non_negative_and_anchored_at_zero():
    off = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM)
    assert min(r for r, _ in off.values()) == 0
    assert min(c for _, c in off.values()) == 0
    assert all(r >= 0 and c >= 0 for r, c in off.values())


def test_negative_stage_coordinates_still_anchor_at_zero():
    """Stage coordinates can be negative; only relative geometry matters."""
    pos = {("A1", 0): (-5_000.0, -3_000.0), ("A1", 1): (-4_950.0, -3_000.0)}
    off = fov_offsets_px(pos, "A1", [0, 1], PX_UM)
    assert off == {0: (0, 0), 1: (0, 100)}


def test_mosaic_extent_is_bounding_box_of_placed_frames():
    off = fov_offsets_px(_grid_positions(n=2, pitch_um=50.0), "A1", [0, 1, 2, 3], PX_UM)
    assert mosaic_extent_px(off, FRAME) == (200, 200)   # 100 px offset + 100 px frame


def test_mosaic_extent_accounts_for_overlap():
    """Overlapping FOVs produce a SMALLER extent than a dense grid would — the extent is a real
    bounding box, not pitch x count."""
    pos = {("A1", 0): (10_000.0, 20_000.0), ("A1", 1): (10_025.0, 20_000.0)}  # 50 px pitch, 100 px frame
    off = fov_offsets_px(pos, "A1", [0, 1], PX_UM)
    assert mosaic_extent_px(off, FRAME) == (100, 150)


# --- error paths ----------------------------------------------------------------------------

def test_missing_pixel_size_raises_named():
    with pytest.raises(ValueError, match="pixel_size_um is required"):
        fov_offsets_px(_grid_positions(n=2), "A1", [0, 1, 2, 3], None)


@pytest.mark.parametrize("bad", [0, -1.0])
def test_non_positive_pixel_size_raises(bad):
    with pytest.raises(ValueError, match="must be > 0"):
        fov_offsets_px(_grid_positions(n=2), "A1", [0, 1, 2, 3], bad)


def test_missing_position_raises_rather_than_leaving_a_hole():
    pos = {("A1", 0): (10.0, 20.0)}
    with pytest.raises(KeyError, match="no stage position"):
        fov_offsets_px(pos, "A1", [0, 1], PX_UM)


def test_empty_fov_list_raises():
    with pytest.raises(ValueError, match="no FOVs to place"):
        fov_offsets_px(_grid_positions(), "A1", [], PX_UM)


# --- cell boxes (thumbnail scale) -----------------------------------------------------------

def test_cell_boxes_fit_inside_the_cell():
    off = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM)
    boxes = cell_boxes(off, FRAME, 88)
    assert len(boxes) == 9
    for top, left, h, w in boxes.values():
        assert 0 <= top < 88 and 0 <= left < 88
        assert h >= 1 and w >= 1
        assert top + h <= 88 and left + w <= 88


def test_cell_boxes_preserve_raster_ordering():
    """Thumbnail scaling must not reorder tiles: fov 0 top-left, fov 8 bottom-right."""
    off = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM)
    boxes = cell_boxes(off, FRAME, 88)
    assert boxes[0][0] <= boxes[6][0]      # row: first row above last row
    assert boxes[0][1] <= boxes[2][1]      # col: first col left of last col
    assert boxes[8][0] >= boxes[0][0] and boxes[8][1] >= boxes[0][1]


def test_cell_boxes_survive_a_tiny_cell():
    """At an absurdly small cell every FOV still gets a >=1px box — none silently vanish."""
    off = fov_offsets_px(_grid_positions(n=6), "A1", range(36), PX_UM)
    boxes = cell_boxes(off, FRAME, 8)
    assert len(boxes) == 36
    assert all(h >= 1 and w >= 1 for _, _, h, w in boxes.values())


def test_cell_boxes_reject_zero_cell():
    off = fov_offsets_px(_grid_positions(n=2), "A1", [0, 1, 2, 3], PX_UM)
    with pytest.raises(ValueError, match="cell_px must be >= 1"):
        cell_boxes(off, FRAME, 0)


def test_single_fov_box_fills_the_cell():
    """N=1 must not shrink: one FOV occupies the whole cell, matching pre-IMA-187 rendering."""
    off = fov_offsets_px({("A1", 0): (1_000.0, 2_000.0)}, "A1", [0], PX_UM)
    (top, left, h, w) = cell_boxes(off, FRAME, 88)[0]
    assert (top, left, h, w) == (0, 0, 88, 88)
