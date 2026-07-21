"""IMA-211: the third-party-stitcher wrappers, on synthetic arrays only.

Same rule as ``test_benchmark.py``: these lock CONTRACTS, never numbers. The numbers come
from ``tools/benchmark.py --stitchers`` on the real acquisition; asserting a seam NCC
against a synthetic fixture would be measuring the fixture.

What must not silently break:

* the ashlar adapters hand ashlar the units it expects (zero-based PIXEL positions),
  because a micron/pixel mix-up produces a plausible mosaic that is simply wrong;
* registering the challengers is idempotent and never clobbers ``stitch``;
* every unavailable stitcher carries the four fields the report prints, so a missing tool
  can never degrade into a missing row.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip import _bench_stitchers as bs
from squidmip import _stitch


@pytest.fixture(autouse=True)
def _restore_region_operators():
    """``add_region_operator`` mutates a module-level table for the life of the process.

    That is correct at runtime — ``tools/benchmark.py --stitchers`` wants the challengers
    to stay registered — but a test that leaks them makes
    ``test_stitch.py::test_default_operators_present`` fail depending on collection order.
    Snapshot and restore, so these tests are order-independent.
    """
    saved = dict(_stitch._REGION_OPERATORS)
    try:
        yield
    finally:
        _stitch._REGION_OPERATORS.clear()
        _stitch._REGION_OPERATORS.update(saved)


def test_array_metadata_is_zero_based_in_pixels():
    """ashlar's own BioformatsMetadata.tile_position divides microns by pixel_size before
    returning, so the adapter must too — or EdgeAligner searches the wrong neighbourhood."""
    tiles = np.zeros((3, 2, 16, 24), dtype=np.uint16)
    md = bs._ArrayMetadata(tiles, [[100.0, 50.0], [100.0, 70.0], [120.0, 50.0]], 0.75)
    assert md.num_images == 3
    assert md.num_channels == 2
    assert md.pixel_size == pytest.approx(0.75)
    assert md.pixel_dtype == np.uint16
    assert tuple(md.size) == (16, 24)
    assert md.positions.min() == 0.0
    np.testing.assert_allclose(md.tile_position(1), [0.0, 20.0])
    np.testing.assert_allclose(md.origin, [0.0, 0.0])
    np.testing.assert_allclose(md.centers[0], [8.0, 12.0])


def test_array_reader_returns_the_requested_plane():
    tiles = np.arange(2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    reader = bs._ArrayReader(bs._ArrayMetadata(tiles, [[0, 0], [0, 2]], 1.0))
    np.testing.assert_array_equal(reader.read(series=1, c=0), tiles[1, 0])
    np.testing.assert_array_equal(reader.read(1, 1), tiles[1, 1])


def test_register_challengers_is_idempotent_and_keeps_stitch():
    from squidmip._stitch import available_region_operators

    first = bs.register_challengers()
    second = bs.register_challengers()
    assert first == second
    ops = available_region_operators()
    assert "stitch" in ops and "coordinate" in ops
    for name in first:
        assert name in ops


def test_challengers_are_only_registered_when_importable():
    """A stitcher that is not installed must be absent, never a stub that would produce a
    fabricated row."""
    for name in bs.CHALLENGERS:
        try:
            bs._probe(name)
        except Exception:
            assert name not in bs.register_challengers()


def test_every_unavailable_stitcher_states_need_reason_and_cost():
    for name, info in bs.UNAVAILABLE.items():
        for key in ("what", "needs", "why_not", "cost"):
            assert info.get(key), f"{name} is missing {key}"


def test_availability_report_mentions_every_unavailable_stitcher():
    text = bs.availability_report()
    for name in bs.UNAVAILABLE:
        assert name in text


def test_petakit_python_package_is_not_petakit5d():
    """The importable ``petakit`` is Julio's deconvolution repo, not the Betzig-lab MATLAB
    toolkit, and it has no stitching entry point. If that ever changes this test should
    fail LOUD rather than let the report keep asserting it."""
    petakit = pytest.importorskip("petakit")
    exported = set(getattr(petakit, "__all__", ())) | set(dir(petakit))
    assert not [n for n in exported if "stitch" in n.lower() or "mosaic" in n.lower()]
    assert "deconvolve" in exported
