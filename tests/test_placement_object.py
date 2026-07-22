"""Defect 3: ONE source of truth for "where are these pixels".

The solved transform was computed once at t=0 and then, by default, THROWN AWAY — ``geometry``
was an optional out-dict, so unless a caller remembered to pass one, the answer to "where did
this mosaic land, and what solved it" simply did not exist. Meanwhile the viewer re-derived
placement independently from a stage bounding box. Two mechanisms, two answers, and no way to
tell when they disagreed.

``Placement`` is that answer as a value object, and it travels WITH the array (see
``PlacedArray``) rather than in a side channel a caller can forget. Critically it records
``reg_channel`` and ``reg_t`` — WHICH channel and timepoint solved the transform — so the data
carries its own provenance. That is the other half of Defect 2: once registration is guaranteed
to run on the requested channel, the output should be able to say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip._placement import Placement, PlacedArray


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


# --- the value object -------------------------------------------------------------------

def test_it_records_which_channel_and_timepoint_solved_the_transform():
    p = _p()
    assert p.reg_channel == "Fluorescence_488_nm_Ex"
    assert p.reg_t == 0


def test_an_unregistered_placement_says_so_rather_than_naming_a_channel_it_did_not_use():
    # register=False is pure coordinate placement. Claiming a reg_channel there would be a
    # provenance lie, so it is None and the offsets are zero.
    p = _p(reg_channel=None, reg_t=None, offsets_px=((0.0, 0.0),) * 4)
    assert p.reg_channel is None and p.reg_t is None
    assert not p.registered


def test_registered_is_not_inferred_from_the_offsets_being_nonzero():
    # A real solve can legitimately return all-zero offsets (no overlap, every pair rejected).
    # That is still a REGISTERED placement, and calling it unregistered would misreport it.
    p = _p(offsets_px=((0.0, 0.0),) * 4)
    assert p.registered


def test_it_is_immutable_so_it_cannot_drift_from_the_array_it_describes():
    p = _p()
    with pytest.raises(Exception):
        p.pixel_size_um = 999.0


def test_the_channel_is_a_NAME_not_an_index():
    # An index re-breaks the moment the channel selection or the reader's axis order changes,
    # which is precisely the Defect 2 bug one level up. Names survive both.
    assert isinstance(_p().reg_channel, str)


def test_fov_count_must_match_the_offsets_and_origins():
    with pytest.raises(ValueError, match="fovs"):
        _p(fovs=(0, 1))


def test_a_nonpositive_pixel_size_is_refused():
    with pytest.raises(ValueError, match="pixel_size_um"):
        _p(pixel_size_um=0.0)


# --- travelling WITH the array ----------------------------------------------------------

def test_a_placed_array_is_still_an_ndarray_for_every_existing_consumer():
    """The whole point of the subclass: nothing downstream has to know it exists.

    stitch_plate yields these straight into the viewer's worker and the OME-Zarr writer,
    neither of which is being changed. If a PlacedArray were not substitutable for an
    ndarray, this change would have to touch both.
    """
    arr = PlacedArray(np.zeros((1, 2, 1, 8, 8), dtype=np.uint16), _p())
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 2, 1, 8, 8)
    assert arr.dtype == np.uint16
    assert float(arr.sum()) == 0.0
    np.testing.assert_array_equal(np.asarray(arr), np.zeros((1, 2, 1, 8, 8)))


def test_the_placement_rides_along_and_cannot_be_forgotten():
    p = _p()
    arr = PlacedArray(np.zeros((1, 1, 1, 4, 4)), p)
    assert arr.placement is p


def test_the_placement_survives_a_slice_because_a_view_is_still_those_pixels():
    p = _p()
    arr = PlacedArray(np.zeros((2, 3, 1, 4, 4)), p)
    assert arr[0].placement is p
    assert arr[:, 0].placement is p


def test_asking_a_plain_array_for_its_placement_fails_loudly():
    # Never a silent None: a consumer that needs placement and got a bare ndarray must find
    # out here, not by rendering the mosaic in the wrong spot.
    with pytest.raises(AttributeError):
        np.zeros((1, 1, 1, 4, 4)).placement          # noqa: B018


def test_the_placement_must_actually_be_one():
    with pytest.raises(TypeError, match="Placement"):
        PlacedArray(np.zeros((1, 1, 1, 4, 4)), {"pixel_size_um": 0.5})
