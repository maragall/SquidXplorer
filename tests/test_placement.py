"""Coordinate placement: exact pixel arithmetic, no GUI."""

from __future__ import annotations

import pytest

from squidxplorer._placement import (
    cell_boxes,
    fov_offsets_px,
    mosaic_extent_px,
)

PX_UM = 0.5
FRAME = (100, 100)


def _grid_positions(region="A1", n=3, pitch_um=50.0, x0=10_000.0, y0=20_000.0):
    """n x n grid of stage positions in micrometres, raster order (x fastest)."""
    pos = {}
    fov = 0
    for r in range(n):
        for c in range(n):
            pos[(region, fov)] = (x0 + c * pitch_um, y0 + r * pitch_um)
            fov += 1
    return pos


def test_offsets_scale_exactly_with_pixel_size():
    """A 50 um pitch at 0.5 um/px is exactly 100 px."""
    pos = _grid_positions(n=2, pitch_um=50.0)
    off = fov_offsets_px(pos, "A1", [0, 1, 2, 3], PX_UM)
    assert off[0] == (0, 0)
    assert off[1] == (0, 100)
    assert off[2] == (100, 0)
    assert off[3] == (100, 100)


def test_offsets_halve_when_pixel_size_doubles():
    pos = _grid_positions(n=2, pitch_um=50.0)
    fine = fov_offsets_px(pos, "A1", [0, 1, 2, 3], PX_UM)
    coarse = fov_offsets_px(pos, "A1", [0, 1, 2, 3], PX_UM * 2)
    assert set(fine) == {0, 1, 2, 3}, sorted(fine)
    for fov in fine:
        assert coarse[fov] == (fine[fov][0] // 2, fine[fov][1] // 2)


def test_y_axis_increases_downward():
    """Larger stage y must map to a larger row index."""
    pos = {("A1", 0): (10_000.0, 20_000.0), ("A1", 1): (10_000.0, 20_100.0)}
    off = fov_offsets_px(pos, "A1", [0, 1], PX_UM)
    assert off[0][0] == 0
    assert off[1][0] > 0, "stage +y must map to +row; a negative/zero row means a Y-axis flip"
    assert off[1][0] == 200


def test_origin_is_per_region_not_global():
    """Each region is laid out in its own frame: both wells start at (0, 0)."""
    pos = {
        ("A1", 0): (10_000.0, 20_000.0), ("A1", 1): (10_050.0, 20_000.0),
        ("B2", 0): (80_000.0, 60_000.0), ("B2", 1): (80_050.0, 60_000.0),
    }
    a1 = fov_offsets_px(pos, "A1", [0, 1], PX_UM)
    b2 = fov_offsets_px(pos, "B2", [0, 1], PX_UM)
    assert a1 == b2 == {0: (0, 0), 1: (0, 100)}


def test_raster_grid_lays_out_row_major():
    off = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM)
    assert off[0] == (0, 0)
    assert off[2] == (0, 200)
    assert off[6] == (200, 0)
    assert off[8] == (200, 200)


def test_offsets_are_non_negative_and_anchored_at_zero():
    off = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM)
    assert min(r for r, _ in off.values()) == 0
    assert min(c for _, c in off.values()) == 0
    assert all(r >= 0 and c >= 0 for r, c in off.values())


def test_negative_stage_coordinates_still_anchor_at_zero():
    pos = {("A1", 0): (-5_000.0, -3_000.0), ("A1", 1): (-4_950.0, -3_000.0)}
    off = fov_offsets_px(pos, "A1", [0, 1], PX_UM)
    assert off == {0: (0, 0), 1: (0, 100)}


def test_mosaic_extent_is_bounding_box_of_placed_frames():
    off = fov_offsets_px(_grid_positions(n=2, pitch_um=50.0), "A1", [0, 1, 2, 3], PX_UM)
    assert mosaic_extent_px(off, FRAME) == (200, 200)


def test_mosaic_extent_accounts_for_overlap():
    """The extent is a real bounding box, not pitch x count."""
    pos = {("A1", 0): (10_000.0, 20_000.0), ("A1", 1): (10_025.0, 20_000.0)}
    off = fov_offsets_px(pos, "A1", [0, 1], PX_UM)
    assert mosaic_extent_px(off, FRAME) == (100, 150)


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


def test_cell_boxes_fit_inside_the_cell():
    off = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM)
    boxes = cell_boxes(off, FRAME, 88)
    assert len(boxes) == 9
    for top, left, h, w in boxes.values():
        assert 0 <= top < 88 and 0 <= left < 88
        assert h >= 1 and w >= 1
        assert top + h <= 88 and left + w <= 88


def test_cell_boxes_preserve_raster_ordering():
    off = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM)
    boxes = cell_boxes(off, FRAME, 88)
    assert boxes[0][0] <= boxes[6][0]
    assert boxes[0][1] <= boxes[2][1]
    assert boxes[8][0] >= boxes[0][0] and boxes[8][1] >= boxes[0][1]


def test_cell_boxes_survive_a_tiny_cell():
    """Every FOV still gets a >=1px box — none silently vanish."""
    off = fov_offsets_px(_grid_positions(n=6), "A1", range(36), PX_UM)
    boxes = cell_boxes(off, FRAME, 8)
    assert len(boxes) == 36
    assert all(h >= 1 and w >= 1 for _, _, h, w in boxes.values())


def test_cell_boxes_reject_zero_cell():
    off = fov_offsets_px(_grid_positions(n=2), "A1", [0, 1, 2, 3], PX_UM)
    with pytest.raises(ValueError, match="cell_px must be >= 1"):
        cell_boxes(off, FRAME, 0)


def test_single_fov_box_fills_the_cell():
    """N=1 must not shrink: one FOV occupies the whole cell."""
    off = fov_offsets_px({("A1", 0): (1_000.0, 2_000.0)}, "A1", [0], PX_UM)
    (top, left, h, w) = cell_boxes(off, FRAME, 88)[0]
    assert (top, left, h, w) == (0, 0, 88, 88)
