"""The contrast slider's ceiling: measured, monotone, and biased to over-cover.

The two datasets these tests are written from are real and on disk. Their maxima are the
parameters below, so a rule that stops working on them fails here rather than on screen.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer import _bitdepth


@pytest.fixture(autouse=True)
def _fresh_dataset():
    """Every test starts on an unmeasured uint16 dataset. The module holds ONE, so it leaks."""
    _bitdepth.new_dataset(np.uint16)
    yield
    _bitdepth.new_dataset(None)


# The measured maxima from the two reference acquisitions. In the 14-bit set, region C3 alone
# reads 3437 and looks 12-bit while the SAME acquisition reaches 16380 at E7/F8. Both sets are
# here because both have to come out right.
_A_DIM_REGION = 3437.0          # one region of the 14-bit set, in isolation
_A_DATASET = 16380.0            # = 4095 * 4, i.e. 12-bit shifted left by 2
_B_DATASET = 65520.0            # = 4095 * 16, 12-bit shifted left by 4, then binned 2x2


@pytest.mark.parametrize("observed, expected", [
    (0.0, 4095.0),
    (_A_DIM_REGION, 4095.0),    # region C3 in isolation
    (4095.0, 4095.0),           # exactly full 12-bit
    (4096.0, 16383.0),          # one count over
    (_A_DATASET, 16383.0),      # the 14-bit set, all regions
    (16383.0, 16383.0),
    (16384.0, 65535.0),
    (_B_DATASET, 65535.0),      # the 16-bit set
    (65535.0, 65535.0),
])
def test_it_snaps_to_the_smallest_depth_that_CONTAINS_the_data(observed, expected):
    """Containment is the invariant. A ceiling below any pixel clips it; above it never can."""
    assert _bitdepth.snap(observed) == expected


def test_a_uint16_dataset_opens_at_the_full_container_range():
    """Before any pixel is seen there is no evidence for narrowing, so nothing under-covers."""
    assert _bitdepth.range_for(np.uint16) == (0.0, 65535.0)
    assert _bitdepth.depth().observed is None


def test_the_ceiling_only_ever_RISES():
    """C3 then E7: the 12-bit conclusion must give way. E7 then C3: it must not come back."""
    d = _bitdepth.depth()
    assert d.observe(_A_DIM_REGION) is True
    assert d.ceiling == 4095.0
    assert d.observe(_A_DATASET) is True
    assert d.ceiling == 16383.0

    assert d.observe(_A_DIM_REGION) is False    # a dimmer region afterwards
    assert d.ceiling == 16383.0

    assert d.observe(2.0) is False
    assert d.ceiling == 16383.0


def test_observing_within_the_current_ceiling_reports_no_change():
    """`observe` returning True is what fires the widen broadcast, so it must mean something."""
    d = _bitdepth.depth()
    assert d.observe(1000.0) is True            # 65535 -> 4095 is a change
    assert d.observe(2000.0) is False           # still 12-bit; no layer needs touching
    assert d.ceiling == 4095.0


def test_a_settled_dataset_stops_measuring():
    """Once 16-bit is proved nothing can widen it further, and the hot path says so."""
    d = _bitdepth.depth()
    d.observe(_B_DATASET)
    assert d.ceiling == 65535.0
    assert d.settled is True
    assert d.observe_array(np.array([65535], dtype=np.uint16)) is False


def test_uint8_and_float_never_consult_the_datasets_depth():
    """The gate is the LAYER's dtype. A float result has no bit depth to apply one to."""
    _bitdepth.depth().observe(_A_DIM_REGION)    # dataset now says 12-bit
    assert _bitdepth.range_for(np.uint8) == (0.0, 255.0)
    assert _bitdepth.range_for(np.float32) == (0.0, 1.0)
    assert _bitdepth.range_for(np.uint16) == (0.0, 4095.0)


def test_a_uint16_layer_in_a_non_uint16_dataset_gets_the_container_range():
    """`_current` is measuring something else; its ceiling says nothing about this layer."""
    _bitdepth.new_dataset(np.uint8)
    assert _bitdepth.range_for(np.uint8) == (0.0, 255.0)
    assert _bitdepth.range_for(np.uint16) == (0.0, 65535.0)


def test_an_unknown_dataset_dtype_still_measures():
    """A reader that reported no dtype must not silently switch the measurement off."""
    _bitdepth.new_dataset(None)
    assert _bitdepth.depth().tracks_uint16 is True
    assert _bitdepth.depth().observe(_A_DIM_REGION) is True
    assert _bitdepth.range_for(np.uint16) == (0.0, 4095.0)


def test_observe_array_takes_the_max_and_survives_junk():
    d = _bitdepth.depth()
    assert d.observe_array(np.array([[10, 3000], [40, _A_DIM_REGION]], dtype=np.uint16)) is True
    assert d.ceiling == 4095.0
    assert d.observe_array(np.array([], dtype=np.uint16)) is False
    assert d.observe_array(None) is False
    assert d.ceiling == 4095.0


def test_nan_does_not_move_the_ceiling():
    """A bad FOV read must not be able to pin a dataset to 16-bit."""
    d = _bitdepth.depth()
    d.observe(_A_DIM_REGION)
    assert d.observe(float("nan")) is False
    assert d.observe_array(np.array([np.nan, 100.0])) is False
    assert d.ceiling == 4095.0


def test_subscribers_hear_every_rise_and_nothing_else():
    seen: list[tuple[float, float]] = []
    d = _bitdepth.depth()
    d.on_change(lambda lo, hi: seen.append((lo, hi)))

    d.observe(_A_DIM_REGION)                    # 65535 -> 4095
    d.observe(3000.0)                           # no change
    d.observe(_A_DATASET)                       # 4095 -> 16383
    assert seen == [(0.0, 4095.0), (0.0, 16383.0)]


def test_a_raising_subscriber_does_not_lose_the_ceiling():
    """A broken listener is a broken listener, not a reason to mis-render the data."""
    d = _bitdepth.depth()
    d.on_change(lambda lo, hi: (_ for _ in ()).throw(RuntimeError("boom")))
    assert d.observe(_A_DATASET) is True
    assert d.ceiling == 16383.0


def test_the_env_override_pins_the_ceiling(monkeypatch):
    monkeypatch.setenv(_bitdepth.ENV_BIT_DEPTH, "12")
    _bitdepth.new_dataset(np.uint16)
    assert _bitdepth.range_for(np.uint16) == (0.0, 4095.0)


def test_the_env_override_does_not_move_even_when_the_data_exceeds_it(monkeypatch, caplog):
    """An override that lies has to SAY so, or it silently reproduces the bug being fixed."""
    monkeypatch.setenv(_bitdepth.ENV_BIT_DEPTH, "12")
    _bitdepth.new_dataset(np.uint16)
    d = _bitdepth.depth()

    with caplog.at_level("WARNING"):
        assert d.observe(_A_DATASET) is False
    assert d.ceiling == 4095.0
    assert "16380" in caplog.text


@pytest.mark.parametrize("junk", ["", "  ", "twelve", "0", "17", "-3"])
def test_a_junk_override_is_ignored_and_the_data_is_measured(monkeypatch, junk):
    monkeypatch.setenv(_bitdepth.ENV_BIT_DEPTH, junk)
    _bitdepth.new_dataset(np.uint16)
    assert _bitdepth.depth().observe(_A_DIM_REGION) is True
    assert _bitdepth.range_for(np.uint16) == (0.0, 4095.0)


def test_a_new_dataset_forgets_the_last_ones_ceiling():
    _bitdepth.depth().observe(_B_DATASET)
    assert _bitdepth.range_for(np.uint16) == (0.0, 65535.0)
    _bitdepth.new_dataset(np.uint16)
    assert _bitdepth.depth().observed is None
    assert _bitdepth.range_for(np.uint16) == (0.0, 65535.0)
    assert _bitdepth.depth().observe(_A_DIM_REGION) is True
    assert _bitdepth.range_for(np.uint16) == (0.0, 4095.0)


def test_a_gcd_of_four_does_NOT_become_a_two_bit_shift():
    """A STANDING REFUSAL of the trailing-zero-bits heuristic, so it is not reinvented.

    The 16-bit set is 12-bit shifted left by 4 -- max 65520 = 4095 * 16 -- but the
    camera binned 2x2, so four such samples were averaged and the file's gcd is 4. Reading that
    gcd as the shift gives 14-bit and clips 65520 to 16383, destroying the top 4x of the range.
    The ONLY thing that decides the ceiling is the largest pixel.
    """
    data = (np.array([0, 4, 16, 1024, 65520], dtype=np.uint16))
    assert int(np.gcd.reduce(data[data > 0])) == 4          # the tempting evidence
    d = _bitdepth.depth()
    d.observe_array(data)
    assert d.ceiling == 65535.0                             # the correct answer
