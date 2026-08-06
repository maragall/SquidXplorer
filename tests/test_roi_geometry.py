"""Coordinate ROUND-TRIPS: every A -> B -> A this app performs must be the identity.

A geometry defect in this codebase does not raise. It draws a plausible picture of the wrong
tissue, or of half of it. So the assertions here are all of one shape -- convert, convert back,
compare to the number that went in -- plus the EXACT SET of FOVs a box covers, never a count. A
test that counts tiles passes on a mosaic that is uniformly wrong; a test that compares integers
does not.

Nothing here needs tilefusion, a GL context or Qt: every function under test is pure arithmetic on
stage micrometres, which is exactly why they are worth pinning at this level.

The numbers are the real ones. The 10x tissue acquisition starts at stage x = 96813.688 um with
0.752 um/px and 2084 px fields; the synthetic 1536 plate steps 400 um on a 777.034 um field at
0.3728571351101784 um/px. A rule that only holds near the origin is not a rule -- half the defects
this file was written against are cancellation against a five-figure stage coordinate.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip._napari3d import region_origin_um, roi_window_px
from squidmip._napari_view import scale_translate_from_bbox_um
from squidmip._placement import Placement, PlacedArray, fov_offsets_px, mosaic_extent_px
from squidmip._stitch import _mosaic_geometry
from squidmip._tilesource import fov_bboxes_um, plate_ladder

# The real 10x tissue set.
STAGE_X0_UM = 96813.688
STAGE_Y0_UM = 7870.0
PX_10X = 0.752
FRAME = 2084

# The synthetic 1536 plate: 2x2 fields, 400 um step on a 777.034 um field.
PX_1536 = 0.3728571351101784
STEP_1536 = 400.0


# ══════════════════════════════════════════════════════════════════════════════════════════
# 1. The stitch canvas: exactly big enough, and independent of where the stage happens to be.
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_a_one_tile_stitch_canvas_is_exactly_the_tile():
    """One field fused alone must produce a canvas of exactly that field, at any stage position.

    It used to produce one MORE row and column, because the canvas was written as
    ``ceil((max + Y*py - min) / py)``: the tile's height went to micrometres and back around a
    stage coordinate of order 1e5 um, and the cancellation left the quotient one ULP above the
    integer. Measured on the real 10x acquisition, all 55 of its single-FOV stitches came back
    2085 px on an axis holding exactly 2084 px of data -- an all-zero row along the bottom of a
    stitched mosaic that the raw preview of the same field does not have.
    """
    (h, w), origins = _mosaic_geometry(
        [(STAGE_Y0_UM, STAGE_X0_UM)], (PX_10X, PX_10X), (FRAME, FRAME))
    assert (h, w) == (FRAME, FRAME), (
        f"a single {FRAME} px tile needs a {FRAME}x{FRAME} canvas, got {h}x{w}")
    assert origins == [(0.0, 0.0)]


@pytest.mark.parametrize("shift_um", [0.0, 1e3, 1e5, 1e6, 96813.688])
def test_the_stitch_canvas_does_not_depend_on_the_stage_origin(shift_um):
    """Translating the whole acquisition must not change the mosaic it fuses into.

    The canvas is a function of the SPANS between tiles. Any dependence on the absolute stage
    coordinate is floating-point noise, and noise that changes an array shape changes what a
    consumer draws.
    """
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
    """Every tile fits inside the canvas, and one pixel less would not fit either axis.

    The tiles are placed at FRACTIONAL origins (that is what sub-pixel registration buys), so the
    canvas must cover ``max(origin) + tile`` and nothing beyond it. Freeform positions, because
    ``manual0`` / ``manual1`` on the real set are not a grid.
    """
    rng = np.random.default_rng(seed)
    pos = [(STAGE_Y0_UM + float(y), STAGE_X0_UM + float(x))
           for y, x in rng.uniform(0.0, 5000.0, size=(n_tiles, 2))]
    (h, w), origins = _mosaic_geometry(pos, (px, px), (FRAME, FRAME))
    max_r = max(oy for oy, _ in origins) + FRAME
    max_c = max(ox for _, ox in origins) + FRAME
    assert h >= max_r - 1e-9 and w >= max_c - 1e-9, "a tile falls outside the canvas"
    assert h - 1 < max_r and w - 1 < max_c, (
        f"canvas {h}x{w} is larger than the {max_r:.4f}x{max_c:.4f} the tiles need")


# ══════════════════════════════════════════════════════════════════════════════════════════
# 2. Placement -> napari and back. X-FIRST becomes Y-FIRST exactly once.
# ══════════════════════════════════════════════════════════════════════════════════════════

def _placement(shape=(2084, 3157), px=PX_1536) -> Placement:
    return Placement(
        origin_um=(STAGE_Y0_UM, STAGE_X0_UM), pixel_size_um=px, z_step_um=1.5,
        shape=shape, tile_shape=(FRAME, FRAME), fovs=(0, 1),
        offsets_px=((0.0, 0.0), (0.0, 0.0)), origins_px=((0.0, 0.0), (0.0, 1072.8)),
        reg_channel=None, reg_t=None)


def test_placement_bbox_round_trips_through_napari_placement():
    """``Placement -> bbox_um -> (scale, translate)`` must give back the pixel size and the origin.

    The shape is deliberately NON-SQUARE (2084 x 3157). With a square mosaic an X/Y transpose is
    invisible, which is the whole reason ``scale_translate_from_bbox_um`` exists as one function.
    """
    p = _placement()
    scale, translate = scale_translate_from_bbox_um(p.bbox_um, p.shape)
    assert scale == pytest.approx((p.pixel_size_um, p.pixel_size_um), rel=0, abs=1e-12)
    assert translate == (STAGE_Y0_UM, STAGE_X0_UM), "napari translate is (y, x); the bbox is x-first"


def test_placement_bbox_is_x_first_and_the_flip_happens_once():
    """``origin_um`` is ``(y, x)`` and ``bbox_um`` is ``(x0, y0, x1, y1)``. Opposite orders."""
    p = _placement(shape=(100, 300), px=2.0)
    assert p.bbox_um == (STAGE_X0_UM, STAGE_Y0_UM, STAGE_X0_UM + 600.0, STAGE_Y0_UM + 200.0)


# ══════════════════════════════════════════════════════════════════════════════════════════
# 3. A region operator's pixels are placed by the canvas that fused them, never by the preview's.
# ══════════════════════════════════════════════════════════════════════════════════════════

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
    """Two of four fields must be drawn across two fields' worth of stage, not four.

    ``stitch_plate(regions={region: [fov, ...]})`` fuses exactly the fields it is handed, so its
    mosaic spans only those. The result used to be placed by ``mosaic_bbox_um``, which always spans
    the WHOLE region: on this geometry a 2084 x 3157 px two-field mosaic was drawn into the four-
    field 1177.11 x 1177.11 um box, i.e. at 0.5648 um/px against a true 0.3729 -- a 1.515x vertical
    stretch, and Julio's "the stitched view is not exactly the same as that of raw (detected 2 FOVs
    instead of 4)".
    """
    from squidmip._mosaic_source import mosaic_bbox_um
    from squidmip._op_result import RegionResultAccumulator

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

    scale, translate = scale_translate_from_bbox_um(result.bbox_um, result.planes[0].shape)
    assert scale == pytest.approx((PX_1536, PX_1536), rel=0, abs=1e-12), (
        f"a two-field stitch was placed at {scale} um/px; the acquisition is {PX_1536}")
    assert translate == (STAGE_Y0_UM, STAGE_X0_UM)
    whole = mosaic_bbox_um(meta, "A1")
    assert result.bbox_um[3] < whole[3], (
        "two of four fields must not claim the whole well's footprint")


def test_a_per_fov_operator_still_uses_the_preview_footprint():
    """The fallback is not dead: a per-FOV operator's planes ARE fused by the preview's own code,
    so the preview's footprint is exactly right for them and must not change."""
    from squidmip._mosaic_source import mosaic_bbox_um
    from squidmip._op_result import RegionResultAccumulator

    meta = _subset_meta()
    acc = RegionResultAccumulator("bgsub", "A1", meta, ["c0"])
    for fov in meta["fovs_per_region"]["A1"]:
        acc.add(fov, np.zeros((1, FRAME, FRAME), np.uint16))
    assert acc.result().bbox_um == mosaic_bbox_um(meta, "A1")


# ══════════════════════════════════════════════════════════════════════════════════════════
# 4. Which FOVs does this box contain? The EXACT set, never a count.
# ══════════════════════════════════════════════════════════════════════════════════════════

def _bboxes():
    meta = _subset_meta()
    return fov_bboxes_um(meta["fov_positions_um"], (FRAME, FRAME), PX_1536)


def _overlapping(box) -> set:
    """``PlateLadder.fovs_overlapping`` — the production answer to "which fields does this box
    contain", asked through the real ladder rather than re-implemented here."""
    return {k[1] for k in plate_ladder(_subset_meta()).fovs_overlapping(tuple(box))}


def test_a_box_over_the_whole_well_contains_every_field():
    b = _bboxes()
    xs = [v[0] for v in b.values()] + [v[2] for v in b.values()]
    ys = [v[1] for v in b.values()] + [v[3] for v in b.values()]
    assert _overlapping((min(xs), min(ys), max(xs), max(ys))) == {0, 1, 2, 3}


def test_a_box_inside_one_field_only_contains_that_field():
    """A 10 um box at field 0's centre. Fields overlap by 377 um here, so the box has to be placed
    where only one field reaches -- the far corner of field 0, away from every neighbour."""
    x0, y0, _x1, _y1 = _bboxes()[("A1", 0)]
    assert _overlapping((x0 + 1.0, y0 + 1.0, x0 + 11.0, y0 + 11.0)) == {0}


def test_the_overlap_test_is_half_open_so_a_touching_edge_is_out():
    """A box that only SHARES AN EDGE with a field contains no pixel of it.

    Inclusive on both ends is how a box that should contain 4 fields comes back with 6, and
    exclusive on both is how it comes back with 2. Pinned in both directions.
    """
    x0, y0, x1, y1 = _bboxes()[("A1", 0)]
    assert 0 not in _overlapping((x1, y0 + 1.0, x1 + 50.0, y0 + 11.0)), "touching is not overlapping"
    assert 0 in _overlapping((x1 - 1e-6, y0 + 1.0, x1 + 50.0, y0 + 11.0)), "a sliver still overlaps"


# ══════════════════════════════════════════════════════════════════════════════════════════
# 5. ROI box (stage um) -> mosaic pixels -> ROI box. The 3D crop's own round-trip.
# ══════════════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("box", [
    (0.0, 0.0, 300.0, 300.0),
    (137.0, 41.0, 637.0, 941.0),
    (700.5, 700.5, 1100.25, 900.75),
])
def test_roi_window_px_round_trips_to_within_half_a_pixel(box):
    """``bbox um -> (r0, r1, c0, c1) -> bbox um`` must return the box, within the pixel it rounded
    to. This is the conversion the bricked 3D ROI reads through, and the origin it uses is
    ``region_origin_um`` -- so this pins the two together rather than each alone."""
    meta = _subset_meta()
    x0, y0 = region_origin_um(meta, "A1")
    shifted = (x0 + box[0], y0 + box[1], x0 + box[2], y0 + box[3])
    window = roi_window_px(meta, "A1", shifted)
    assert window is not None
    r0, r1, c0, c1 = window
    back = (x0 + c0 * PX_1536, y0 + r0 * PX_1536, x0 + c1 * PX_1536, y0 + r1 * PX_1536)
    assert back == pytest.approx(shifted, abs=PX_1536 / 2 + 1e-9)


def test_roi_window_px_is_a_half_open_window_of_the_right_size():
    """``(r1 - r0, c1 - c0)`` is the pixel COUNT the crop will have, so it must be the box's own
    span in pixels -- not one more (inclusive) and not one less (exclusive at both ends)."""
    meta = _subset_meta()
    x0, y0 = region_origin_um(meta, "A1")
    span_px = 512
    span_um = span_px * PX_1536
    r0, r1, c0, c1 = roi_window_px(
        meta, "A1", (x0 + 100.0, y0 + 100.0, x0 + 100.0 + span_um, y0 + 100.0 + span_um))
    assert (r1 - r0, c1 - c0) == (span_px, span_px)


def test_the_preview_extent_and_the_offsets_agree_on_where_the_last_field_ends():
    """``mosaic_extent_px`` must be exactly ``max(offset) + frame`` -- the bound the preview pastes
    into. One pixel short and the bottom-right field is clipped; one long and it is letterboxed."""
    meta = _subset_meta()
    fovs = meta["fovs_per_region"]["A1"]
    offsets = fov_offsets_px(meta["fov_positions_um"], "A1", fovs, PX_1536)
    h, w = mosaic_extent_px(offsets, (FRAME, FRAME))
    assert h == max(r for r, _ in offsets.values()) + FRAME
    assert w == max(c for _, c in offsets.values()) + FRAME
    assert min(offsets.values()) == (0, 0), "the top-left field anchors the mosaic at (0, 0)"
