"""``projection.cast_like``: round (half-to-even), clip (not wrap), monotone — pinned."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer.projection import cast_like


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_cast_like_is_monotone_non_decreasing_over_random_floats(dtype):
    """A larger input may never map to a smaller output, through the rounding AND the clip."""
    rng = np.random.default_rng(7)
    values = np.sort(rng.uniform(-500.0, 70000.0, 4096))
    out = cast_like(values, dtype).astype(np.int64)
    assert np.all(np.diff(out) >= 0)


def test_cast_like_clips_at_both_dtype_ends_instead_of_wrapping():
    values = np.array([-1e9, -1.0, -0.4, 70000.0, 1e9])
    assert list(cast_like(values, np.uint8)) == [0, 0, 0, 255, 255]
    assert list(cast_like(values, np.uint16)) == [0, 0, 0, 65535, 65535]


def test_cast_like_rounds_rather_than_truncating():
    values = np.array([10.4, 10.6, 99.9])
    assert list(cast_like(values, np.uint16)) == [10, 11, 100]


def test_the_halfway_cases_are_HALF_TO_EVEN_by_name():
    """``np.rint``'s tie-breaking, kept deliberately: numpy's default, statistically unbiased across a plate (half-up would brighten every image ~0.5 count"""
    values = np.array([0.5, 1.5, 2.5, 254.5, 255.5])
    assert list(cast_like(values, np.uint16)) == [0, 2, 2, 254, 256]
    assert list(cast_like(values, np.uint8)) == [0, 2, 2, 254, 255]


def test_copy_false_refuses_a_non_float_buffer():
    """In-place ``rint`` on an integer array is a silent no-op, so it is refused, named."""
    with pytest.raises(ValueError, match="floating"):
        cast_like(np.array([1, 2, 3], np.int32), np.uint16, copy=False)


def test_copy_false_rounds_the_caller_s_own_buffer():
    buf = np.array([1.4, 2.6], np.float64)
    assert list(cast_like(buf, np.uint16, copy=False)) == [1, 3]
    assert list(buf) == [1.0, 3.0]
