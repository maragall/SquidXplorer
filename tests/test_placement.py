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


def test_offsets_are_exact_row_major_pixels_anchored_at_zero():
    """A 50 um pitch at 0.5 um/px is exactly 100 px; stage +y maps to +row; per-region origin."""
    off = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM)
    assert off[0] == (0, 0) and off[2] == (0, 200) and off[6] == (200, 0) and off[8] == (200, 200)
    assert min(r for r, _ in off.values()) == 0 and min(c for _, c in off.values()) == 0
    coarse = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM * 2)
    assert all(coarse[f] == (off[f][0] // 2, off[f][1] // 2) for f in off)
    pos = {
        ("A1", 0): (10_000.0, 20_000.0), ("A1", 1): (10_000.0, 20_100.0),
        ("B2", 0): (-5_000.0, -3_000.0), ("B2", 1): (-5_000.0, -2_900.0),
    }
    a1 = fov_offsets_px(pos, "A1", [0, 1], PX_UM)
    assert a1[1][0] == 200, "stage +y must map to +row; a negative/zero row means a Y-axis flip"
    assert fov_offsets_px(pos, "B2", [0, 1], PX_UM) == a1 == {0: (0, 0), 1: (200, 0)}


def test_mosaic_extent_is_the_bounding_box_of_placed_frames():
    off = fov_offsets_px(_grid_positions(n=2, pitch_um=50.0), "A1", [0, 1, 2, 3], PX_UM)
    assert mosaic_extent_px(off, FRAME) == (200, 200)
    pos = {("A1", 0): (10_000.0, 20_000.0), ("A1", 1): (10_025.0, 20_000.0)}
    off = fov_offsets_px(pos, "A1", [0, 1], PX_UM)
    assert mosaic_extent_px(off, FRAME) == (100, 150), "a real bounding box, not pitch x count"


def test_placement_refuses_bad_input_by_name():
    with pytest.raises(ValueError, match="pixel_size_um is required"):
        fov_offsets_px(_grid_positions(n=2), "A1", [0, 1, 2, 3], None)
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="must be > 0"):
            fov_offsets_px(_grid_positions(n=2), "A1", [0, 1, 2, 3], bad)
    with pytest.raises(KeyError, match="no stage position"):
        fov_offsets_px({("A1", 0): (10.0, 20.0)}, "A1", [0, 1], PX_UM)
    with pytest.raises(ValueError, match="no FOVs to place"):
        fov_offsets_px(_grid_positions(), "A1", [], PX_UM)
    with pytest.raises(ValueError, match="cell_px must be >= 1"):
        cell_boxes(fov_offsets_px(_grid_positions(n=2), "A1", [0, 1, 2, 3], PX_UM), FRAME, 0)


def test_cell_boxes_fit_inside_the_cell_in_raster_order_and_none_vanish():
    off = fov_offsets_px(_grid_positions(n=3), "A1", range(9), PX_UM)
    boxes = cell_boxes(off, FRAME, 88)
    assert len(boxes) == 9
    for top, left, h, w in boxes.values():
        assert 0 <= top < 88 and 0 <= left < 88 and h >= 1 and w >= 1
        assert top + h <= 88 and left + w <= 88
    assert boxes[0][0] <= boxes[6][0] and boxes[0][1] <= boxes[2][1]
    assert boxes[8][0] >= boxes[0][0] and boxes[8][1] >= boxes[0][1]
    tiny = cell_boxes(fov_offsets_px(_grid_positions(n=6), "A1", range(36), PX_UM), FRAME, 8)
    assert len(tiny) == 36 and all(h >= 1 and w >= 1 for _, _, h, w in tiny.values())
    one = fov_offsets_px({("A1", 0): (1_000.0, 2_000.0)}, "A1", [0], PX_UM)
    assert cell_boxes(one, FRAME, 88)[0] == (0, 0, 88, 88), "one FOV fills the whole cell"
