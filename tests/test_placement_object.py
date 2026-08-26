"""ONE source of truth for "where are these pixels"."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._placement import Placement, PlacedArray


def _p(**over):
    kw = dict(
        origin_um=(10.0, 20.0),
        pixel_size_um=0.5,
        z_step_um=1.5,
        shape=(400, 600),
        tile_shape=(256, 256),
        fovs=(0, 1, 2, 3),
        offsets_px=((0.0, 0.0), (1.5, -2.0), (0.0, 0.5), (-1.0, 0.0)),
        origins_px=((0.0, 0.0), (0.0, 192.0), (192.0, 0.0), (192.0, 192.0)),
        reg_channel="Fluorescence_488_nm_Ex",
        reg_t=0,
    )
    kw.update(over)
    return Placement(**kw)


def test_the_placement_records_by_NAME_which_channel_and_timepoint_solved_it_and_is_immutable():
    p = _p()
    assert p.reg_channel == "Fluorescence_488_nm_Ex" and isinstance(p.reg_channel, str)
    assert p.reg_t == 0
    with pytest.raises(Exception):
        p.pixel_size_um = 999.0
    unreg = _p(reg_channel=None, reg_t=None, offsets_px=((0.0, 0.0),) * 4)
    assert unreg.reg_channel is None and unreg.reg_t is None and not unreg.registered
    assert _p(offsets_px=((0.0, 0.0),) * 4).registered, "registered is declared, not inferred from zero offsets"


def test_the_placement_refuses_a_mismatched_fov_count_or_a_nonpositive_pixel_size():
    with pytest.raises(ValueError, match="fovs"):
        _p(fovs=(0, 1))
    with pytest.raises(ValueError, match="pixel_size_um"):
        _p(pixel_size_um=0.0)


def test_a_placed_array_is_an_ndarray_whose_placement_survives_a_slice():
    p = _p()
    arr = PlacedArray(np.zeros((2, 3, 1, 8, 8), dtype=np.uint16), p)
    assert isinstance(arr, np.ndarray) and arr.shape == (2, 3, 1, 8, 8) and arr.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(arr), np.zeros((2, 3, 1, 8, 8)))
    assert arr.placement is p and arr[0].placement is p and arr[:, 0].placement is p
    with pytest.raises(TypeError, match="Placement"):
        PlacedArray(np.zeros((1, 1, 1, 4, 4)), {"pixel_size_um": 0.5})
