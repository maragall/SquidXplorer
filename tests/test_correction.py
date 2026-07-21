"""IMA-224 background subtraction: the two traps, the commutation, and the regression guard.

The traps this file exists to pin:
  1. uint16 underflow — `100 - 200` wraps to 65436, and because the reducer is a MAX those
     wrapped pixels win every comparison and invert the whole plate.
  2. float64 promotion — np.percentile returns float64, and np.maximum(uint16, float64)
     promotes the RESULT, which breaks the native-dtype output contract.
"""

import json

import numpy as np
import pytest

from squidmip._correction import (AFTER, BEFORE, background_corrector, estimate_background,
                                  subtract_background, with_correction, write_provenance)
from squidmip.projection import project


# --- the underflow trap -----------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_subtract_background_never_underflows(dtype):
    """Values below the background clip to 0 — they must NOT wrap to near the dtype max."""
    info = np.iinfo(dtype)
    plane = np.array([[0, 5, 20], [50, 100, 200]], dtype=dtype)   # fits uint8 too
    out = subtract_background(plane, 50)
    assert out.max() <= plane.max(), "wrapped: a dark pixel became bright"
    assert out.max() < info.max
    np.testing.assert_array_equal(out, np.array([[0, 0, 0], [0, 50, 150]], dtype=dtype))


def test_underflow_would_corrupt_a_max_projection():
    """The regression that motivates clamping: naive subtraction inverts the MIP."""
    planes = [np.full((4, 4), 50, np.uint16), np.full((4, 4), 500, np.uint16)]
    naive = project([p - np.uint16(200) for p in planes])       # the bug
    fixed = project([subtract_background(p, 200) for p in planes])
    assert naive.max() > 60000, "expected the naive form to wrap (this is the bug being guarded)"
    assert fixed.max() == 300


# --- the float64 promotion trap ---------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_subtract_background_preserves_dtype(dtype):
    plane = np.array([[10, 20], [30, 40]], dtype=dtype)
    assert subtract_background(plane, 15).dtype == np.dtype(dtype)


def test_dtype_survives_a_raw_numpy_percentile():
    """np.percentile returns float64; passing it through must not widen the result."""
    plane = (np.arange(256, dtype=np.uint16) * 7).reshape(16, 16)
    bg = np.percentile(plane, 10)
    assert isinstance(bg, np.floating) and np.dtype(type(bg)) == np.float64
    assert subtract_background(plane, bg).dtype == np.uint16


# --- purity / null cases ----------------------------------------------------------------

def test_does_not_mutate_the_callers_plane():
    plane = np.array([[10, 200]], dtype=np.uint16)
    before = plane.copy()
    subtract_background(plane, 50)
    np.testing.assert_array_equal(plane, before)


def test_zero_and_none_background_are_no_ops():
    plane = np.array([[10, 200]], dtype=np.uint16)
    np.testing.assert_array_equal(subtract_background(plane, 0), plane)
    np.testing.assert_array_equal(subtract_background(plane, None), plane)


def test_background_at_or_above_max_yields_all_zeros():
    plane = np.array([[10, 200]], dtype=np.uint16)
    out = subtract_background(plane, 200)
    np.testing.assert_array_equal(out, np.zeros_like(plane))


# --- the composition seam ---------------------------------------------------------------

def test_with_correction_none_returns_the_reducer_untouched():
    assert with_correction(project, None) is project


def test_before_equals_explicit_map_then_reduce():
    planes = [np.full((3, 3), v, np.uint16) for v in (100, 250, 400)]
    got = with_correction(project, background_corrector(120), BEFORE)(iter(planes))
    want = project([subtract_background(p, 120) for p in planes])
    np.testing.assert_array_equal(got, want)


def test_after_is_bit_identical_to_before_for_mip():
    """The commutation that licenses AFTER: max_z(p - b) == max_z(p) - b for z-invariant b."""
    rng = np.random.default_rng(0)
    planes = [rng.integers(0, 5000, (32, 32)).astype(np.uint16) for _ in range(6)]
    corr = background_corrector(137)
    before = with_correction(project, corr, BEFORE)(iter(planes))
    after = with_correction(project, corr, AFTER)(iter(planes))
    np.testing.assert_array_equal(before, after)
    assert after.dtype == np.uint16


def test_with_correction_streams_in_one_pass():
    """A generator input must not be materialised — bounded memory is the engine's contract."""
    planes = [np.full((2, 2), v, np.uint16) for v in (10, 20, 30)]
    consumed = []

    def gen():
        for p in planes:
            consumed.append(p)
            yield p

    with_correction(project, background_corrector(5), BEFORE)(gen())
    assert len(consumed) == 3


def test_with_correction_rejects_a_bad_side():
    with pytest.raises(ValueError, match="side must be"):
        with_correction(project, background_corrector(1), "sideways")


def test_empty_planes_still_raise_through_the_decorator():
    with pytest.raises(ValueError):
        with_correction(project, background_corrector(1), BEFORE)(iter([]))


# --- estimation -------------------------------------------------------------------------

def test_estimate_background_is_one_scalar_for_the_plate(squid_dataset):
    from squidmip import open_reader

    path, _ = squid_dataset
    reader = open_reader(str(path))
    bg = estimate_background(reader, 10.0)
    assert isinstance(bg, float)
    assert bg == estimate_background(reader, 10.0), "estimate must be deterministic"


def test_estimate_background_rejects_bad_arguments(squid_dataset):
    from squidmip import open_reader

    path, _ = squid_dataset
    reader = open_reader(str(path))
    with pytest.raises(ValueError, match="percentile"):
        estimate_background(reader, 101.0)
    with pytest.raises(ValueError, match="sample_planes"):
        estimate_background(reader, 10.0, sample_planes=0)


def test_estimate_background_degrades_to_zero_when_nothing_is_readable():
    class Dud:
        metadata = {"channels": [], "z_levels": [0], "n_z": 1}

    assert estimate_background(Dud(), 10.0) == 0.0


# --- provenance -------------------------------------------------------------------------

def test_write_provenance_records_the_level(tmp_path):
    path = write_provenance(tmp_path / "out.hcs", {"background_percentile": 10, "background_level": 42})
    got = json.loads(path.read_text())
    assert got["background_level"] == 42
    assert path.name == "squidmip-provenance.json"


# --- wiring: engine, CLI, viewer ---------------------------------------------------------

def test_engine_resolves_a_callable_projector():
    """The widening that lets a parameterised correction reach the name-keyed seam."""
    from squidmip._engine import _resolve_projector

    corrected = with_correction(project, background_corrector(5), AFTER)
    assert _resolve_projector(corrected) is corrected
    assert _resolve_projector("mip") is project


def test_engine_still_rejects_an_unknown_name():
    from squidmip._engine import _resolve_projector

    with pytest.raises(KeyError, match="unknown projector"):
        _resolve_projector("nope")


def test_plate_run_with_background_is_a_regression_free_null_when_off(squid_dataset):
    """REGRESSION GUARD: no correction must be byte-identical to a plain MIP run."""
    from squidmip import open_reader, project_plate

    path, _ = squid_dataset
    plain = {(r, f): im for r, f, im in project_plate(open_reader(str(path)), projector="mip")}
    passthrough = with_correction(project, None)
    got = {(r, f): im for r, f, im in project_plate(open_reader(str(path)), projector=passthrough)}
    assert set(plain) == set(got)
    for k in plain:
        np.testing.assert_array_equal(plain[k], got[k])


def test_plate_run_with_background_subtracts_and_keeps_dtype(squid_dataset):
    from squidmip import open_reader, project_plate

    path, _ = squid_dataset
    reader = open_reader(str(path))
    plain = {(r, f): im for r, f, im in project_plate(reader, projector="mip")}
    corrected_proj = with_correction(project, background_corrector(50), AFTER)
    got = {(r, f): im for r, f, im in project_plate(open_reader(str(path)), projector=corrected_proj)}
    for k in plain:
        assert got[k].dtype == plain[k].dtype
        assert got[k].max() <= plain[k].max()


def test_cli_rejects_an_out_of_range_background(squid_dataset):
    from squidmip._cli import ProcessParameters

    path, _ = squid_dataset
    with pytest.raises(Exception, match="percentile"):
        ProcessParameters(input_folder=str(path), background=101.0)


def test_cli_accepts_a_valid_background_and_defaults_to_none(squid_dataset):
    from squidmip._cli import ProcessParameters

    path, _ = squid_dataset
    assert ProcessParameters(input_folder=str(path)).background is None
    assert ProcessParameters(input_folder=str(path), background=10.0).background == 10.0


def test_worker_projector_is_the_plain_name_when_background_is_off():
    """Toggle off must not build a correction at all — the null case stays the old path."""
    from squidmip._viewer import _OperatorWorker

    w = _OperatorWorker.__new__(_OperatorWorker)
    w._operator, w._background = "mip", None
    projector, level = w._projector()
    assert projector == "mip" and level is None


def test_worker_projector_decorates_once_when_background_is_on(squid_dataset):
    """Toggle on: ONE estimate for the run (not per well), and a callable handed to the engine."""
    from squidmip import open_reader
    from squidmip._viewer import _OperatorWorker

    path, _ = squid_dataset
    w = _OperatorWorker.__new__(_OperatorWorker)
    w._operator, w._background = "mip", 10.0
    w._reader = open_reader(str(path))
    projector, level = w._projector()
    assert callable(projector) and projector is not project
    assert isinstance(level, float)
