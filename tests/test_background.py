"""IMA-224 background subtraction — numerical property tests + the LAYER contract.

Julio's constraint, and the reason this file is longer than a "does it run" test: background
subtraction is a **layer**, not a destructive edit. "Each transform is a layer, something like
CellProfiler does this." So there are two families of test here:

  1. it must actually REMOVE a known added background (the numerical property), and
  2. the raw must remain RECOVERABLE — the source planes are never mutated, the background is
     an addressable artefact of its own, and ``raw == corrected + background`` exactly.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from squidmip import available_projectors, project_well, projector_consumes
from squidmip._background import (
    BackgroundParams,
    bgsub_op,
    clipped_fraction,
    estimate_background,
    restore,
    subtract_background,
)
from squidmip._layers import OperationStack
from squidmip.projection import PLANE_OP
from squidmip.reader import open_reader

pytest.importorskip("scipy.ndimage")

# Julio's bgsub (the 'sep' estimator, IMA-247) is an OPTIONAL package, not a dependency, so a clean
# install has no `bgsub`. test_sep_method_is_julios_bgsub_implementation_not_a_reimplementation
# already guards itself with importorskip("bgsub.core") -- and is the one sep test that did NOT
# fail on CI. The sep PARAMETRISATIONS and the two dedicated sep tests below never got the same
# guard, so on every runner they failed with ModuleNotFoundError instead of skipping. Same
# reasoning as the tilefusion guard in tests/test_stitch.py.
#
# Stated plainly: THE SEP ESTIMATOR IS NOT COVERED IN CI. A skip is not a pass. rolling_ball and
# gaussian, which are the default and the one that ships, keep running everywhere.
_NEEDS_BGSUB = pytest.mark.skipif(
    importlib.util.find_spec("bgsub") is None,
    reason="bgsub (Julio's sep estimator) not installed: the sep path is UNTESTED here, not passing")
#: Use in place of the bare "sep" string so the parametrised cases skip rather than error.
_SEP = pytest.param("sep", marks=_NEEDS_BGSUB)


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

@pytest.mark.parametrize("method", ["rolling_ball", "gaussian", _SEP])
def test_removes_the_structure_of_a_known_added_background(method):
    """The estimate must reproduce the SHAPE of the planted dome, and subtracting it must
    flatten the empty field.

    Measured against shape, not against absolute level, because a constant offset in the
    estimate is not an error here — see ``test_rolling_ball_bias_is_conservative``. What would
    be an error is leaving the dome's *structure* behind, and that is what is asserted."""
    size = 128
    fg, bg = _foreground(size), _known_background(size)
    raw = (fg + bg).astype(np.uint16)
    span = float(bg.max() - bg.min())

    params = BackgroundParams(method=method, radius_px=15)
    corrected = subtract_background(raw, params)
    estimated = estimate_background(raw, params)

    shape_err = float(np.abs((estimated - estimated.mean()) - (bg - bg.mean())).mean()) / span
    assert shape_err < 0.15, f"{method}: estimate's SHAPE is off by {shape_err:.1%} of the span"

    # and the corrected image must be flat where there is no sample: the dome's spread must be
    # gone, not merely reduced.
    empty = fg == 0
    residual_spread = float(np.percentile(corrected[empty], 90) - np.percentile(corrected[empty], 10))
    assert residual_spread < span * 0.35, (
        f"{method}: {residual_spread:.0f} counts of dome left out of a planted span of {span:.0f}"
    )


def test_rolling_ball_bias_is_conservative_and_measured():
    """Sternberg's ball rolls UNDER the surface, so its estimate is systematically LOW by
    roughly the ball's sagitta — it never subtracts more signal than is there. That is a
    property of the algorithm (ImageJ behaves the same way), not a bug, and it is pinned here
    with a number so a future change that flips the sign cannot pass silently."""
    size = 128
    fg, bg = _foreground(size), _known_background(size)
    raw = (fg + bg).astype(np.uint16)

    bias = {r: float((estimate_background(raw, BackgroundParams(radius_px=r)) - bg).mean() / bg.mean())
            for r in (15, 25, 40)}
    assert all(b < 0 for b in bias.values()), f"rolling ball over-estimated the background: {bias}"
    assert bias[15] > bias[25] > bias[40], f"bias should deepen with radius: {bias}"
    assert abs(bias[15]) < 0.15


@pytest.mark.parametrize("method", ["rolling_ball", "gaussian", _SEP])
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


@pytest.mark.parametrize("method", ["rolling_ball", "gaussian", _SEP])
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


@pytest.mark.parametrize("method", ["rolling_ball", "gaussian", _SEP])
@pytest.mark.parametrize("dtype", [np.uint16, np.uint8])
def test_raw_is_exactly_recoverable_wherever_the_result_did_not_clip_for_every_method(dtype, method):
    """Property 3 of the layer contract, held for EVERY estimator including Julio's sep.

    ``restore()`` must give the raw back exactly (``array_equal``, no tolerance) wherever the
    subtraction did not clip. This is what keeps 'background subtraction is a layer' true no
    matter which background was subtracted — the invertibility comes from the additive
    composition and the rounding cast, not from the choice of estimator.
    """
    scale = 255 / 4000 if dtype is np.uint8 else 1.0
    raw = ((_foreground(96) + _known_background(96)) * scale).astype(dtype)
    params = BackgroundParams(method=method, radius_px=20)
    bg = estimate_background(raw, params)
    corrected = subtract_background(raw, params)

    clipped = np.rint(raw.astype(np.float32) - bg) < 0
    recovered = restore(corrected, bg, dtype=raw.dtype)
    assert np.array_equal(recovered[~clipped], raw[~clipped])
    assert not clipped.all(), f"{method}: every pixel clipped; the test proves nothing"


@pytest.mark.parametrize("dtype", [np.uint16, np.uint8])
def test_raw_is_exactly_recoverable_wherever_the_result_did_not_clip(dtype):
    """The layer is ADDITIVE and its operand is addressable, so the composition is INVERTIBLE:
    ``raw == corrected + background`` exactly — ``array_equal``, no tolerance — on the
    acquisition dtypes (uint8/uint16). This is the mechanical form of "the raw is preserved".

    It works because the integer cast ROUNDS: the residual of ``round(raw - bg)`` is under half
    a count, so adding ``bg`` back and rounding lands on ``raw`` itself. Truncating instead
    (plain ``astype``) would lose the raw by one count everywhere, which is precisely the kind
    of quiet destructive edit this operator must not be.
    """
    scale = 255 / 4000 if dtype is np.uint8 else 1.0
    raw = ((_foreground(96) + _known_background(96)) * scale).astype(dtype)
    params = BackgroundParams(radius_px=20)
    bg = estimate_background(raw, params)
    corrected = subtract_background(raw, params)

    clipped = np.rint(raw.astype(np.float32) - bg) < 0
    recovered = restore(corrected, bg, dtype=raw.dtype)
    assert np.array_equal(recovered[~clipped], raw[~clipped])


def test_integer_clipping_is_reported_not_hidden():
    """Clipping at the dtype floor is the ONE place this transform loses information — where
    the background estimate exceeds the raw value. The operator must be able to say how much,
    rather than presenting a lossy transform as a lossless one."""
    raw = (_foreground(96) + _known_background(96)).astype(np.uint16)
    params = BackgroundParams(radius_px=20)
    bg = estimate_background(raw, params)

    reported = clipped_fraction(raw, params)
    measured = float(np.mean(np.rint(raw.astype(np.float32) - bg) < 0))
    assert reported == pytest.approx(measured)

    # rolling_ball rolls UNDER the surface, so its estimate never exceeds the raw and NOTHING
    # clips: with the default method the layer is fully lossless, and it can say so.
    assert reported == 0.0

    # the gaussian method has a positive bias (bright objects leak into their own background),
    # so it DOES clip — and the operator reports a nonzero fraction rather than hiding it.
    leaky = BackgroundParams(method="gaussian", radius_px=20)
    assert clipped_fraction(raw, leaky) > 0.0
    assert clipped_fraction(raw.astype(np.float32), leaky) == 0.0   # float has no floor to clip at


# --- IMA-247: Julio's sep estimator is wired in, and is deliberately not the default --------

def test_sep_method_is_julios_bgsub_implementation_not_a_reimplementation():
    """The 'sep' method must be a call INTO bgsub.core, not a local copy of it.

    Pinned by substitution: monkeypatch bgsub.core._run_sep and assert our estimate changes.
    If this module ever grew its own box-estimator, this test would keep passing while the
    reuse silently disappeared — so it also asserts the sentinel value flows through.
    """
    bgsub_core = pytest.importorskip("bgsub.core")
    raw = (_foreground(96) + _known_background(96)).astype(np.uint16)
    params = BackgroundParams(method="sep", radius_px=20)

    sentinel = np.full((96, 96), 1234.0, dtype=np.float32)
    real = bgsub_core._run_sep
    bgsub_core._run_sep = lambda img, box: (img - sentinel, sentinel)
    try:
        assert np.array_equal(estimate_background(raw, params), sentinel)
    finally:
        bgsub_core._run_sep = real

    # and with his real implementation restored it produces a sane, plane-shaped estimate
    bg = estimate_background(raw, params)
    assert bg.shape == raw.shape and bg.dtype == np.float32
    assert 0 < float(bg.mean()) < float(raw.max())


@_NEEDS_BGSUB
def test_sep_is_not_the_default_because_it_clips_far_more_of_the_frame():
    """The measured reason sep stays opt-in.

    sep is a CENTRAL background estimator (sigma-clipped box mean), so roughly half a plane
    sits below its estimate and clips at zero on an unsigned dtype. rolling_ball is a lower
    ENVELOPE and clips almost nothing. Clipping is the one place this layer destroys
    information, so the default must be the estimator that destroys least.
    """
    raw = (_foreground(96) + _known_background(96)).astype(np.uint16)

    assert BackgroundParams().method == "rolling_ball", "the default estimator changed"

    ball = clipped_fraction(raw, BackgroundParams(method="rolling_ball", radius_px=20))
    sep = clipped_fraction(raw, BackgroundParams(method="sep", radius_px=20))

    assert ball == 0.0
    assert sep > ball, f"sep ({sep:.3f}) no longer clips more than rolling_ball ({ball:.3f})"
    assert sep > 0.1, (
        "sep stopped clipping heavily — if that is real, re-evaluate whether it should be the "
        f"default; measured {sep:.3f}"
    )


@_NEEDS_BGSUB
def test_sep_never_writes_to_disk_and_never_mutates_the_caller(tmp_path):
    """Property 1 of the layer contract for the sep path specifically.

    bgsub ships a BackgroundSubtractor that WRITES corrected TIFFs into an output directory.
    That orchestrator is deliberately not used: only the pure array-level estimator is called.
    Asserted by running the operator with the CWD inside an empty directory and checking that
    nothing appeared, and that the input plane is byte-identical afterwards.
    """
    import os

    raw = (_foreground(64) + _known_background(64)).astype(np.uint16)
    before = raw.copy()
    params = BackgroundParams(method="sep", radius_px=20)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        subtract_background(raw, params)
        estimate_background(raw, params)
        clipped_fraction(raw, params)
    finally:
        os.chdir(cwd)

    assert list(tmp_path.iterdir()) == [], "the sep path wrote files; a layer must not"
    assert np.array_equal(raw, before), "the sep path mutated the caller's plane"


def test_a_missing_bgsub_fails_loud_and_never_silently_falls_back_to_rolling_ball():
    """No silent substitution between estimators: they disagree by design (see the clipping
    numbers above), so a missing dependency must be an error, not a different algorithm."""
    import builtins

    real_import = builtins.__import__

    def _no_bgsub(name, *args, **kwargs):
        if name.startswith("bgsub"):
            raise ImportError("simulated: bgsub not installed")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _no_bgsub
    try:
        with pytest.raises(ImportError, match="background_subtraction|NO fallback"):
            estimate_background(np.zeros((16, 16), np.uint16),
                                BackgroundParams(method="sep", radius_px=8))
    finally:
        builtins.__import__ = real_import


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
