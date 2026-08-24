"""RegionResultAccumulator: per-FOV operator output becomes one region's self-describing Result."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._op_result import RegionResultAccumulator
from squidxplorer._result import Result

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


def test_a_plane_op_s_fovs_accumulate_into_one_region_mosaic():
    acc = RegionResultAccumulator("demo", "A1", _meta(), CHANNELS)
    assert not acc.complete()
    acc.add(0, np.full((2, 8, 8), 11, np.uint16))
    assert not acc.complete()
    acc.add(1, np.full((2, 8, 8), 22, np.uint16))
    assert acc.complete()

    res = acc.result()
    assert isinstance(res, Result)
    assert res.region_id == "A1"
    assert res.channels == CHANNELS
    # 2 FOVs, 8 px frames, 6 px step -> 14 px wide, 8 tall.
    assert res.plane(CHANNELS[0]).shape == (8, 14)
    # ...and the result DECLARES what it is, so no sink has to re-derive it.
    assert res.z_depth == 1
    assert res.dtype == "uint16"
    assert res.pixel_size_um == 1.0
    assert res.kind == "intensity"


def test_the_operator_mosaic_lands_in_THE_SAME_FRAME_as_the_raw_mosaic():
    """Asserted against the raw path's own geometry helpers: same offsets, same extent."""
    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    meta = _meta()
    offsets = fov_offsets_px(meta["fov_positions_um"], "A1", [0, 1], 1.0)
    raw_shape = mosaic_extent_px(offsets, (8, 8))

    acc = RegionResultAccumulator("demo", "A1", meta, CHANNELS)
    acc.add(0, np.zeros((2, 8, 8), np.uint16))
    acc.add(1, np.zeros((2, 8, 8), np.uint16))
    assert acc.result().plane(CHANNELS[0]).shape == tuple(raw_shape)


def test_the_pixels_are_the_OPERATOR_S_not_the_reader_s():
    """Distinct constants per FOV make a quiet re-read of the raw file visible."""
    acc = RegionResultAccumulator("demo", "A1", _meta(), CHANNELS)
    acc.add(0, np.full((2, 8, 8), 11, np.uint16))
    acc.add(1, np.full((2, 8, 8), 22, np.uint16))
    plane = acc.result().plane(CHANNELS[0])
    assert plane[0, 0] == 11
    assert plane[0, -1] == 22


def test_each_channel_keeps_its_own_pixels():
    acc = RegionResultAccumulator("demo", "A1", _meta(), CHANNELS)
    for fov in (0, 1):
        planes = np.stack([np.full((8, 8), 7, np.uint16), np.full((8, 8), 99, np.uint16)])
        acc.add(fov, planes)
    res = acc.result()
    assert res.plane(CHANNELS[0])[0, 0] == 7
    assert res.plane(CHANNELS[1])[0, 0] == 99


def test_a_region_operator_s_result_IS_the_mosaic_and_is_not_re_placed():
    """A region op already returns the fused region; it must not be re-placed as a FOV."""
    acc = RegionResultAccumulator("stitch", "A1", _meta(), CHANNELS, region_operator=True)
    assert not acc.complete()
    acc.add(0, np.full((2, 20, 30), 5, np.uint16))
    assert acc.complete()
    plane = acc.result().plane(CHANNELS[0])
    assert plane.shape == (20, 30)
    assert plane[0, 0] == 5


def test_an_incomplete_region_refuses_to_produce_a_result():
    acc = RegionResultAccumulator("demo", "A1", _meta(), CHANNELS)
    acc.add(0, np.zeros((2, 8, 8), np.uint16))
    with pytest.raises(ValueError, match="1 of 2"):
        acc.result()


def test_a_channel_count_mismatch_is_named_not_broadcast():
    acc = RegionResultAccumulator("demo", "A1", _meta(), CHANNELS)
    with pytest.raises(ValueError, match="channel"):
        acc.add(0, np.zeros((1, 8, 8), np.uint16))


def test_an_unknown_fov_is_refused_rather_than_placed_at_the_origin():
    acc = RegionResultAccumulator("demo", "A1", _meta(), CHANNELS)
    with pytest.raises(ValueError, match="99"):
        acc.add(99, np.zeros((2, 8, 8), np.uint16))


def test_an_acquisition_without_a_pixel_size_is_refused_not_guessed():
    """A result that cannot say its own scale is not self-describing."""
    meta = _meta()
    del meta["pixel_size_um"]
    acc = RegionResultAccumulator("stitch", "A1", meta, CHANNELS, region_operator=True)
    acc.add(0, np.zeros((2, 8, 8), np.uint16))
    with pytest.raises(ValueError, match="pixel size"):
        acc.result()


def test_the_result_carries_the_bbox_so_napari_places_it_over_the_raw_layer():
    """Without the bbox the operator group would sit at the origin in stage space."""
    from squidxplorer._mosaic_source import mosaic_bbox_um

    meta = _meta()
    acc = RegionResultAccumulator("demo", "A1", meta, CHANNELS)
    acc.add(0, np.zeros((2, 8, 8), np.uint16))
    acc.add(1, np.zeros((2, 8, 8), np.uint16))
    assert acc.result().extent.bbox_um == mosaic_bbox_um(meta, "A1")


def test_the_result_carries_the_kind_its_operator_declares(blob_operator):
    """``produces`` is read off the registry ONCE, here, and rides on the Result to every sink."""
    acc = RegionResultAccumulator(blob_operator, "A1", _meta(), CHANNELS)
    acc.add(0, np.zeros((2, 8, 8), np.uint16))
    acc.add(1, np.zeros((2, 8, 8), np.uint16))
    assert acc.result().kind == "labels"
