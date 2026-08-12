"""Defect 3: an operator's output is a RESULT TYPE, and a result becomes a layer group.

Julio: "what if we want to see stitched AND deconvolved AND background subed. That's why we
need the toggles." And: "when I run the MIP or background sub or flatfield correction or the
stitcher or decon, like these are also reflected in the plate view and in my central viewer
and that's why I turn layers on and off."

Before this module there was no result type at all. Every operator emitted a bare
``(region, fov, ndarray)`` triple, and the only sink was ``register_array`` -- the ndviewer
push path -- so NO operator's pixels ever reached pane 2's napari ``MosaicLayers``. The group
toggle UI (``_layer_tree.MosaicTree``) was already built and working; it had nothing to show
because the producer side did not exist.

The unit under test is deliberately Qt-free and napari-free: accumulating an operator's
per-FOV planes into the region mosaic is arithmetic over placement, and it is the part that
can be wrong in a way a screenshot cannot reveal (a mosaic in a DIFFERENT frame from raw
still looks like a picture -- it just makes the before/after toggle a lie).
"""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._op_result import OperatorResult, RegionResultAccumulator

CHANNELS = ("Fluorescence_405_nm_Ex", "Fluorescence_488_nm_Ex")


def _meta(frame=(8, 8), step=6.0):
    """Two FOVs side by side, overlapping, 1 um/px."""
    return {
        "fovs_per_region": {"A1": [0, 1]},
        "fov_positions_um": {("A1", 0): (0.0, 0.0), ("A1", 1): (step, 0.0)},
        "pixel_size_um": 1.0,
        "frame_shape": frame,
        "dtype": "uint16",
        "channels": [{"name": c} for c in CHANNELS],
    }


# ---------------------------------------------------------------------------------------
# the accumulator: per-FOV operator output -> one region mosaic, in the RAW frame
# ---------------------------------------------------------------------------------------

def test_a_plane_op_s_fovs_accumulate_into_one_region_mosaic():
    acc = RegionResultAccumulator("bgsub", "A1", _meta(), CHANNELS)
    assert not acc.complete()
    acc.add(0, np.full((2, 8, 8), 11, np.uint16))
    assert not acc.complete()                      # one FOV is not a region
    acc.add(1, np.full((2, 8, 8), 22, np.uint16))
    assert acc.complete()

    res = acc.result()
    assert isinstance(res, OperatorResult)
    assert res.op == "bgsub"
    assert res.region == "A1"
    assert res.channels == CHANNELS
    # 2 FOVs, 8 px frames, 6 px step -> 14 px wide, 8 tall.
    assert res.plane(CHANNELS[0]).shape == (8, 14)


def test_the_operator_mosaic_lands_in_THE_SAME_FRAME_as_the_raw_mosaic():
    """The whole point of a toggle. If the operator layer had its own extent or its own
    placement rule, flipping between raw and processed would MOVE the picture, and any
    difference the user saw would be registration error rather than the operator's effect.

    So this asserts against the raw path's own geometry helpers, not against a number I
    typed in: same offsets, same extent, one source of truth."""
    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    meta = _meta()
    offsets = fov_offsets_px(meta["fov_positions_um"], "A1", [0, 1], 1.0)
    raw_shape = mosaic_extent_px(offsets, (8, 8))

    acc = RegionResultAccumulator("bgsub", "A1", meta, CHANNELS)
    acc.add(0, np.zeros((2, 8, 8), np.uint16))
    acc.add(1, np.zeros((2, 8, 8), np.uint16))
    assert acc.result().plane(CHANNELS[0]).shape == tuple(raw_shape)


def test_the_pixels_are_the_OPERATOR_S_not_the_reader_s():
    """A mosaic that quietly re-read the raw file would produce a beautiful, identical
    'processed' layer. Distinct constants per FOV make that substitution visible."""
    acc = RegionResultAccumulator("bgsub", "A1", _meta(), CHANNELS)
    acc.add(0, np.full((2, 8, 8), 11, np.uint16))
    acc.add(1, np.full((2, 8, 8), 22, np.uint16))
    plane = acc.result().plane(CHANNELS[0])
    assert plane[0, 0] == 11                       # first FOV's own value
    assert plane[0, -1] == 22                      # second FOV's own value


def test_each_channel_keeps_its_own_pixels():
    """One layer per channel. A channel mix-up here is the defect that makes a 4-channel
    composite look plausible and be wrong."""
    acc = RegionResultAccumulator("bgsub", "A1", _meta(), CHANNELS)
    for fov in (0, 1):
        planes = np.stack([np.full((8, 8), 7, np.uint16), np.full((8, 8), 99, np.uint16)])
        acc.add(fov, planes)
    res = acc.result()
    assert res.plane(CHANNELS[0])[0, 0] == 7
    assert res.plane(CHANNELS[1])[0, 0] == 99


def test_a_region_operator_s_result_IS_the_mosaic_and_is_not_re_placed():
    """stitch already returns the fused region. Running it back through FOV placement would
    tile a mosaic as if it were a FOV -- so a region op is a single, whole result."""
    acc = RegionResultAccumulator("stitch", "A1", _meta(), CHANNELS, region_operator=True)
    assert not acc.complete()
    acc.add(0, np.full((2, 20, 30), 5, np.uint16))
    assert acc.complete()
    plane = acc.result().plane(CHANNELS[0])
    assert plane.shape == (20, 30)                 # untouched, not re-tiled
    assert plane[0, 0] == 5


def test_an_incomplete_region_refuses_to_produce_a_result():
    """NO SILENT FAILURES: half a region is not a result. Returning the half would put a
    layer on screen with holes the user would read as the operator's output."""
    acc = RegionResultAccumulator("bgsub", "A1", _meta(), CHANNELS)
    acc.add(0, np.zeros((2, 8, 8), np.uint16))
    with pytest.raises(ValueError, match="1 of 2"):
        acc.result()


def test_a_channel_count_mismatch_is_named_not_broadcast():
    acc = RegionResultAccumulator("bgsub", "A1", _meta(), CHANNELS)
    with pytest.raises(ValueError, match="channel"):
        acc.add(0, np.zeros((1, 8, 8), np.uint16))


def test_an_unknown_fov_is_refused_rather_than_placed_at_the_origin():
    acc = RegionResultAccumulator("bgsub", "A1", _meta(), CHANNELS)
    with pytest.raises(ValueError, match="99"):
        acc.add(99, np.zeros((2, 8, 8), np.uint16))


def test_the_result_carries_the_bbox_so_napari_places_it_over_the_raw_layer():
    """add_mosaic takes bbox_um; without it the operator group would sit at the origin in
    stage space and the toggle would jump."""
    from squidxplorer._mosaic_source import mosaic_bbox_um

    meta = _meta()
    acc = RegionResultAccumulator("bgsub", "A1", meta, CHANNELS)
    acc.add(0, np.zeros((2, 8, 8), np.uint16))
    acc.add(1, np.zeros((2, 8, 8), np.uint16))
    assert acc.result().bbox_um == mosaic_bbox_um(meta, "A1")


def test_the_group_key_is_the_operator_so_two_operators_are_two_groups():
    """'stitched AND deconvolved AND background subed' -- three groups, three toggles. The
    group key must be the OPERATOR, not the region, or a second region would overwrite the
    first operator's group instead of adding to it."""
    a = RegionResultAccumulator("bgsub", "A1", _meta(), CHANNELS)
    b = RegionResultAccumulator("decon", "A1", _meta(), CHANNELS)
    for acc in (a, b):
        acc.add(0, np.zeros((2, 8, 8), np.uint16))
        acc.add(1, np.zeros((2, 8, 8), np.uint16))
    assert a.result().op != b.result().op
