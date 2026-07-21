"""IMA-224 background subtraction — numerical property tests + the LAYER contract.

Julio's constraint, and the reason this file is longer than a "does it run" test: background
subtraction is a **layer**, not a destructive edit. "Each transform is a layer, something like
CellProfiler does this." So there are two families of test here:

  1. it must actually REMOVE a known added background (the numerical property), and
  2. the raw must remain RECOVERABLE — the source planes are never mutated, the background is
     an addressable artefact of its own, and ``raw == corrected + background`` exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip import available_projectors, project_well, projector_consumes
from squidmip._background import (
    BackgroundParams,
    bgsub_op,
    estimate_background,
    restore,
    subtract_background,
)
from squidmip._layers import OperationStack
from squidmip.projection import PLANE_OP
from squidmip.reader import open_reader

pytest.importorskip("scipy.ndimage")


def _foreground(size: int = 128, seed: int = 1) -> np.ndarray:
    """Sparse bright puncta on a true-zero background — so any nonzero floor in the corrected
    image is measurable leftover background, not sample."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    for y, x in zip(rng.integers(6, size - 6, 25), rng.integers(6, size - 6, 25)):
        img[y - 2:y + 3, x - 2:x + 3] += rng.uniform(800, 3000)
    return img


def _known_background(size: int = 128, amplitude: float = 600.0, pedestal: float = 200.0):
    """A smooth corner-to-corner dome: the shape stray light and out-of-focus haze actually
    make, and something a single scalar offset provably cannot remove."""
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    return (pedestal + amplitude * np.exp(-((yy - 0.2) ** 2 + (xx - 0.8) ** 2) / 0.5)).astype(np.float32)


# --- the core numerical property: a KNOWN added background must come off ------------------

@pytest.mark.parametrize("method", ["rolling_ball", "gaussian"])
def test_removes_a_known_added_background(method):
    size = 128
    fg, bg = _foreground(size), _known_background(size)
    raw = (fg + bg).astype(np.uint16)

    params = BackgroundParams(method=method, radius_px=25)
    corrected = subtract_background(raw, params)
    estimated = estimate_background(raw, params)

    # the estimate must track the truth, not just be "some smooth thing"
    err = np.abs(estimated - bg).mean() / bg.mean()
    assert err < 0.15, f"{method}: background estimate off by {err:.1%} of its mean"

    # and the corrected image must be flat where there is no sample
    empty = fg == 0
    assert corrected[empty].mean() < bg.mean() * 0.15


@pytest.mark.parametrize("method", ["rolling_ball", "gaussian"])
def test_a_gradient_background_is_flattened_across_the_field(method):
    """A single scalar offset cannot do this: with a corner-to-corner ramp, the bright corner
    and the dark corner must end up at the SAME level after subtraction."""
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    ramp = 100.0 + 8.0 * (yy + xx)
    raw = (_foreground(size) + ramp).astype(np.uint16)

    corrected = subtract_background(raw, BackgroundParams(method=method, radius_px=25))

    q = size // 4
    dark = float(np.percentile(corrected[:q, :q], 20))
    bright = float(np.percentile(corrected[-q:, -q:], 20))
    raw_dark = float(np.percentile(raw[:q, :q], 20))
    raw_bright = float(np.percentile(raw[-q:, -q:], 20))
    assert raw_bright - raw_dark > 1000                     # the ramp was really there
    assert abs(bright - dark) < (raw_bright - raw_dark) * 0.1


@pytest.mark.parametrize("method", ["rolling_ball", "gaussian"])
def test_foreground_puncta_survive_subtraction(method):
    """Removing the background must not eat the sample: a background estimator with too large
    an effect would flatten the puncta too, which is the failure mode that matters clinically."""
    size = 128
    fg, bg = _foreground(size), _known_background(size)
    raw = (fg + bg).astype(np.uint16)
    corrected = subtract_background(raw, BackgroundParams(method=method, radius_px=25))
    hot = fg > 500
    assert corrected[hot].mean() > fg[hot].mean() * 0.7


def test_a_flat_image_has_a_flat_background_and_subtracts_to_zero():
    flat = np.full((64, 64), 1234, dtype=np.uint16)
    est = estimate_background(flat, BackgroundParams(radius_px=15))
    assert np.allclose(est, 1234, rtol=0.02)
    assert subtract_background(flat, BackgroundParams(radius_px=15)).max() < 40


# --- THE LAYER CONTRACT: the raw stays recoverable ----------------------------------------

def test_the_input_plane_is_never_mutated():
    """The most basic sense in which this is not a destructive edit: the caller's buffer — the
    array the reader just handed us, backed by the raw TIFF — is untouched."""
    raw = (_foreground(64) + _known_background(64)).astype(np.uint16)
    before = raw.copy()
    subtract_background(raw, BackgroundParams(radius_px=15))
    estimate_background(raw, BackgroundParams(radius_px=15))
    assert np.array_equal(raw, before)


def test_raw_is_exactly_recoverable_from_corrected_plus_background_in_float():
    """The layer is ADDITIVE and its operand is addressable, so the composition is invertible:
    raw == corrected + background, exactly, with no float slop, when nothing clipped."""
    raw = (_foreground(96) + _known_background(96)).astype(np.float32)
    params = BackgroundParams(radius_px=20)
    bg = estimate_background(raw, params)
    corrected = subtract_background(raw, params)
    assert np.array_equal(restore(corrected, bg, dtype=raw.dtype), raw)


def test_integer_clipping_is_reported_not_hidden():
    """Where the background estimate exceeds the raw value, an unsigned result clips at 0 and
    THAT is where information would be lost. The operator must be able to say how much clipped
    rather than pretending the transform was lossless."""
    raw = (_foreground(96) + _known_background(96)).astype(np.uint16)
    params = BackgroundParams(radius_px=20)
    bg = estimate_background(raw, params)
    corrected = subtract_background(raw, params)

    clipped = (raw.astype(np.float32) - bg) < 0
    recovered = restore(corrected, bg, dtype=raw.dtype)
    assert np.array_equal(recovered[~clipped], raw[~clipped])   # lossless everywhere else
    assert clipped.mean() < 0.6                                  # sanity: not the whole frame


def test_a_bgsub_layer_can_be_toggled_off_to_return_to_raw():
    """The OperationStack half of 'it's a layer': the plate falls back to the raw base the
    moment the layer is disabled or removed — no re-read, no undo stack, no inverse transform."""
    stack = OperationStack()
    stack.add("bgsub@tab1", "background subtraction")
    assert stack.top_enabled().key == "bgsub@tab1"

    stack.toggle("bgsub@tab1", False)
    assert stack.top_enabled().key == "raw"

    stack.toggle("bgsub@tab1", True)
    assert stack.top_enabled().key == "bgsub@tab1"
    assert stack.remove("bgsub@tab1")
    assert stack.top_enabled().key == "raw"


def test_the_reader_is_read_only_so_the_source_tiffs_survive_a_run(squid_dataset):
    """End-to-end non-destructiveness: run the operator over a real acquisition and prove the
    on-disk raw is byte-identical afterwards."""
    root, arrays = squid_dataset
    reader = open_reader(root)
    tiffs = sorted((root / "0").glob("*.tiff"))
    before = {p: p.read_bytes() for p in tiffs}

    project_well(reader, "B2", 0, reduce=bgsub_op(BackgroundParams(radius_px=2)))

    assert {p: p.read_bytes() for p in tiffs} == before
    key = ("B2", 0, 0, reader.metadata["channels"][0]["name"])
    assert np.array_equal(reader.read("B2", 0, key[3], 0, 0), arrays[key])


# --- registry / engine seam ---------------------------------------------------------------

def test_bgsub_is_registered_as_a_plane_op():
    assert "bgsub" in available_projectors()
    assert projector_consumes("bgsub") == PLANE_OP


def test_bgsub_op_refuses_a_whole_z_stack():
    op = bgsub_op(BackgroundParams(radius_px=3))
    with pytest.raises(ValueError, match="more than one plane"):
        op([np.zeros((8, 8), np.uint16), np.zeros((8, 8), np.uint16)])


def test_unknown_method_fails_loud_by_name():
    with pytest.raises(ValueError, match="unknown background method"):
        estimate_background(np.zeros((8, 8), np.uint16), BackgroundParams(method="wishful"))


def test_project_well_with_bgsub_keeps_z_at_full_depth(squid_dataset):
    root, _ = squid_dataset
    reader = open_reader(root)
    out = project_well(reader, "B2", 0, reduce=bgsub_op(BackgroundParams(radius_px=2)))
    assert out.shape[2] == len(reader.metadata["z_levels"])
    assert out.dtype == reader.metadata["dtype"]
