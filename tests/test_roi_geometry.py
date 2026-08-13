"""Coordinate round-trips: every A -> B -> A this app performs must be the identity, checked
against the EXACT set of FOVs a box covers, never a count — a test that counts tiles passes on a
mosaic that is uniformly wrong. Real acquisition numbers are used throughout since a rule that
only holds near the origin is not a rule."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._napari3d import region_origin_um, roi_window_px
from squidxplorer._napari_view import scale_translate_from_bbox_um
from squidxplorer._placement import Placement, PlacedArray, fov_offsets_px, mosaic_extent_px
from squidxplorer._stitch import _mosaic_geometry
from squidxplorer._tilesource import fov_bboxes_um, plate_ladder

# The real 10x tissue set.
STAGE_X0_UM = 96813.688
STAGE_Y0_UM = 7870.0
PX_10X = 0.752
FRAME = 2084

# The synthetic 1536 plate: 2x2 fields, 400 um step on a 777.034 um field.
PX_1536 = 0.3728571351101784
STEP_1536 = 400.0


def test_a_one_tile_stitch_canvas_is_exactly_the_tile():
    """A single field's canvas must be exactly that field, at any stage position. It used to gain
    one extra row/col from a `ceil` computed around a five-figure stage coordinate."""
    (h, w), origins = _mosaic_geometry(
        [(STAGE_Y0_UM, STAGE_X0_UM)], (PX_10X, PX_10X), (FRAME, FRAME))
    assert (h, w) == (FRAME, FRAME), (
        f"a single {FRAME} px tile needs a {FRAME}x{FRAME} canvas, got {h}x{w}")
    assert origins == [(0.0, 0.0)]


@pytest.mark.parametrize("shift_um", [0.0, 1e3, 1e5, 1e6, 96813.688])
def test_the_stitch_canvas_does_not_depend_on_the_stage_origin(shift_um):
    """Canvas shape is a function of the spans between tiles, not the absolute stage coordinate."""
    rel = [(0.0, 0.0), (0.0, STEP_1536), (STEP_1536, 0.0), (STEP_1536, STEP_1536)]
    moved = [(y + shift_um, x + shift_um) for y, x in rel]
    at_origin, _ = _mosaic_geometry(rel, (PX_1536, PX_1536), (FRAME, FRAME))
    shifted, _ = _mosaic_geometry(moved, (PX_1536, PX_1536), (FRAME, FRAME))
    assert shifted == at_origin, (
        f"the same 2x2 mosaic fused to {at_origin} at the origin and {shifted} "
        f"{shift_um} um away")


@pytest.mark.parametrize("px", [0.752, 0.3728571351101784, 0.325, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_tiles", [1, 2, 9])
def test_the_stitch_canvas_holds_every_tile_and_is_minimal(px, seed, n_tiles):
    """Tiles sit at fractional (sub-pixel-registered) origins rather than a grid, so the canvas
    must cover max(origin) + tile and nothing beyond it."""
    rng = np.random.default_rng(seed)
    pos = [(STAGE_Y0_UM + float(y), STAGE_X0_UM + float(x))
           for y, x in rng.uniform(0.0, 5000.0, size=(n_tiles, 2))]
    (h, w), origins = _mosaic_geometry(pos, (px, px), (FRAME, FRAME))
    max_r = max(oy for oy, _ in origins) + FRAME
    max_c = max(ox for _, ox in origins) + FRAME
    assert h >= max_r - 1e-9 and w >= max_c - 1e-9, "a tile falls outside the canvas"
    assert h - 1 < max_r and w - 1 < max_c, (
        f"canvas {h}x{w} is larger than the {max_r:.4f}x{max_c:.4f} the tiles need")


def _placement(shape=(2084, 3157), px=PX_1536) -> Placement:
    return Placement(
        origin_um=(STAGE_Y0_UM, STAGE_X0_UM), pixel_size_um=px, z_step_um=1.5,
        shape=shape, tile_shape=(FRAME, FRAME), fovs=(0, 1),
        offsets_px=((0.0, 0.0), (0.0, 0.0)), origins_px=((0.0, 0.0), (0.0, 1072.8)),
        reg_channel=None, reg_t=None)


def test_placement_bbox_round_trips_through_napari_placement():
    """Shape is deliberately non-square (2084x3157): a square mosaic would hide an X/Y transpose."""
    p = _placement()
    scale, translate = scale_translate_from_bbox_um(p.bbox_um, p.shape)
    assert scale == pytest.approx((p.pixel_size_um, p.pixel_size_um), rel=0, abs=1e-12)
    assert translate == (STAGE_Y0_UM, STAGE_X0_UM), "napari translate is (y, x); the bbox is x-first"


def test_placement_bbox_is_x_first_and_the_flip_happens_once():
    """``origin_um`` is ``(y, x)`` and ``bbox_um`` is ``(x0, y0, x1, y1)``. Opposite orders."""
    p = _placement(shape=(100, 300), px=2.0)
    assert p.bbox_um == (STAGE_X0_UM, STAGE_Y0_UM, STAGE_X0_UM + 600.0, STAGE_Y0_UM + 200.0)


def _subset_meta() -> dict:
    """The synthetic 1536 plate's A1: four fields, 2x2 at a 400 um step on a 777.034 um field."""
    pos = {}
    for i, (dx, dy) in enumerate([(0, 0), (STEP_1536, 0), (0, STEP_1536),
                                  (STEP_1536, STEP_1536)]):
        pos[("A1", i)] = (STAGE_X0_UM + dx, STAGE_Y0_UM + dy)
    return {
        "pixel_size_um": PX_1536,
        "frame_shape": (FRAME, FRAME),
        "z_levels": [0],
        "channels": [{"name": "c0"}],
        "fovs_per_region": {"A1": [0, 1, 2, 3]},
        "fov_positions_um": pos,
    }


def test_a_stitched_fov_subset_is_not_stretched_over_the_whole_well():
    """A FOV-subset stitch must be placed by its own fused canvas, not by ``mosaic_bbox_um``
    (which always spans the whole region) — that mismatch was a 1.515x vertical stretch on real
    geometry, Julio's "detected 2 FOVs instead of 4"."""
    from squidxplorer._mosaic_source import mosaic_bbox_um
    from squidxplorer._op_result import RegionResultAccumulator

    meta = _subset_meta()
    subset = [0, 1]
    offsets = fov_offsets_px(meta["fov_positions_um"], "A1", subset, PX_1536)
    h, w = mosaic_extent_px(offsets, (FRAME, FRAME))
    placement = Placement(
        origin_um=(STAGE_Y0_UM, STAGE_X0_UM), pixel_size_um=PX_1536, z_step_um=None,
        shape=(h, w), tile_shape=(FRAME, FRAME), fovs=tuple(subset),
        offsets_px=((0.0, 0.0),) * 2, origins_px=((0.0, 0.0), (0.0, float(offsets[1][1]))),
        reg_channel=None, reg_t=None)
    stack = PlacedArray(np.zeros((1, h, w), np.uint16), placement)

    acc = RegionResultAccumulator("stitch", "A1", meta, ["c0"], region_operator=True)
    acc.add(0, stack)
    result = acc.result()

    scale, translate = scale_translate_from_bbox_um(result.extent.bbox_um,
                                                    result.plane("c0").shape)
    assert scale == pytest.approx((PX_1536, PX_1536), rel=0, abs=1e-12), (
        f"a two-field stitch was placed at {scale} um/px; the acquisition is {PX_1536}")
    assert translate == (STAGE_Y0_UM, STAGE_X0_UM)
    whole = mosaic_bbox_um(meta, "A1")
    assert result.extent.bbox_um[3] < whole[3], (
        "two of four fields must not claim the whole well's footprint")


def test_a_per_fov_operator_still_uses_the_preview_footprint():
    """The fallback isn't dead: a per-FOV operator's planes are fused by the preview's own code,
    so its footprint stays right for them and must not change."""
    from squidxplorer._mosaic_source import mosaic_bbox_um
    from squidxplorer._op_result import RegionResultAccumulator

    meta = _subset_meta()
    acc = RegionResultAccumulator("bgsub", "A1", meta, ["c0"])
    for fov in meta["fovs_per_region"]["A1"]:
        acc.add(fov, np.zeros((1, FRAME, FRAME), np.uint16))
    assert acc.result().extent.bbox_um == mosaic_bbox_um(meta, "A1")


def _bboxes():
    meta = _subset_meta()
    return fov_bboxes_um(meta["fov_positions_um"], (FRAME, FRAME), PX_1536)


def _overlapping(box) -> set:
    """``PlateLadder.fovs_overlapping`` asked through the real ladder rather than re-implemented
    here."""
    return {k[1] for k in plate_ladder(_subset_meta()).fovs_overlapping(tuple(box))}


def test_a_box_over_the_whole_well_contains_every_field():
    b = _bboxes()
    xs = [v[0] for v in b.values()] + [v[2] for v in b.values()]
    ys = [v[1] for v in b.values()] + [v[3] for v in b.values()]
    assert _overlapping((min(xs), min(ys), max(xs), max(ys))) == {0, 1, 2, 3}


def test_a_box_inside_one_field_only_contains_that_field():
    """Fields overlap by 377 um here, so the probe sits in field 0's far corner, away from any
    neighbour."""
    x0, y0, _x1, _y1 = _bboxes()[("A1", 0)]
    assert _overlapping((x0 + 1.0, y0 + 1.0, x0 + 11.0, y0 + 11.0)) == {0}


def test_the_overlap_test_is_half_open_so_a_touching_edge_is_out():
    """Half-open on both ends: inclusive on both would count 6 fields, exclusive on both would
    count 2."""
    x0, y0, x1, y1 = _bboxes()[("A1", 0)]
    assert 0 not in _overlapping((x1, y0 + 1.0, x1 + 50.0, y0 + 11.0)), "touching is not overlapping"
    assert 0 in _overlapping((x1 - 1e-6, y0 + 1.0, x1 + 50.0, y0 + 11.0)), "a sliver still overlaps"


@pytest.mark.parametrize("box", [
    (0.0, 0.0, 300.0, 300.0),
    (137.0, 41.0, 637.0, 941.0),
    (700.5, 700.5, 1100.25, 900.75),
])
def test_roi_window_px_round_trips_to_within_half_a_pixel(box):
    """Pins ``roi_window_px`` together with ``region_origin_um``, the origin it converts against —
    the same pair the bricked 3D ROI reads through."""
    meta = _subset_meta()
    x0, y0 = region_origin_um(meta, "A1")
    shifted = (x0 + box[0], y0 + box[1], x0 + box[2], y0 + box[3])
    window = roi_window_px(meta, "A1", shifted)
    assert window is not None
    r0, r1, c0, c1 = window
    back = (x0 + c0 * PX_1536, y0 + r0 * PX_1536, x0 + c1 * PX_1536, y0 + r1 * PX_1536)
    assert back == pytest.approx(shifted, abs=PX_1536 / 2 + 1e-9)


def test_roi_window_px_is_a_half_open_window_of_the_right_size():
    """``(r1 - r0, c1 - c0)`` is the pixel count the crop will have, so it must equal the box's
    own span — not one more (inclusive) or one less (exclusive at both ends)."""
    meta = _subset_meta()
    x0, y0 = region_origin_um(meta, "A1")
    span_px = 512
    span_um = span_px * PX_1536
    r0, r1, c0, c1 = roi_window_px(
        meta, "A1", (x0 + 100.0, y0 + 100.0, x0 + 100.0 + span_um, y0 + 100.0 + span_um))
    assert (r1 - r0, c1 - c0) == (span_px, span_px)


def test_the_preview_extent_and_the_offsets_agree_on_where_the_last_field_ends():
    """``mosaic_extent_px`` must equal exactly ``max(offset) + frame`` — the bound the preview
    pastes into."""
    meta = _subset_meta()
    fovs = meta["fovs_per_region"]["A1"]
    offsets = fov_offsets_px(meta["fov_positions_um"], "A1", fovs, PX_1536)
    h, w = mosaic_extent_px(offsets, (FRAME, FRAME))
    assert h == max(r for r, _ in offsets.values()) + FRAME
    assert w == max(c for _, c in offsets.values()) + FRAME
    assert min(offsets.values()) == (0, 0), "the top-left field anchors the mosaic at (0, 0)"
