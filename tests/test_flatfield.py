"""IMA-225 flat-field correction — numerical property tests + the MIP-commutation shortcut.

Two things are proven here on data rather than asserted in a comment:

1. **It flattens a known vignette.** A synthetic dome is applied to a flat field and the
   correction must bring the corner-to-centre ratio back to ~1.
2. **Flat-field commutes with the MIP.** Correction is applied per pixel by a MONOTONE
   non-decreasing map f, and ``max(f(a), f(b)) == f(max(a, b))`` for monotone f. So
   flat-fielding AFTER a MIP is bit-identical to flat-fielding every plane before it, at
   1/Nz the cost. Integer rounding and clipping do NOT break this (both are themselves
   monotone), and that is exactly what these tests check — on real 10x data, not just
   synthetic — because "assume it holds" is how a rounding bug ships.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from squidmip import available_projectors, project, project_well, projector_consumes
from squidmip._flatfield import (
    FlatfieldProfile,
    active_profile,
    clear_profile,
    correct_flatfield,
    estimate_profile,
    flatfield_op,
    set_profile,
)
from squidmip.projection import PLANE_OP
from squidmip.reader import open_reader

pytest.importorskip("scipy.ndimage")

REAL = Path("/Users/julioamaragall/Downloads/"
            "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")


@pytest.fixture
def laser_af_dataset():
    if not REAL.is_dir():
        pytest.skip("10x laser-AF z-stack dataset not present")
    return REAL


@pytest.fixture(autouse=True)
def _no_leaked_active_profile():
    """The registered ``flatfield`` operator reads module-level state; make sure no test leaks
    a profile into another one."""
    clear_profile()
    yield
    clear_profile()


def _vignette(size: int = 128, depth: float = 0.55) -> np.ndarray:
    """A radial dome normalised to mean 1 — bright centre, dim corners, like a real objective."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1) - 0.5
    field = 1.0 - depth * (yy ** 2 + xx ** 2) / 0.5
    return (field / field.mean()).astype(np.float32)


# --- the core numerical property: a KNOWN vignette must be flattened ----------------------

def test_flattens_a_known_vignette():
    size = 128
    flat_truth = np.full((size, size), 3000.0, dtype=np.float32)
    ff = _vignette(size)
    vignetted = (flat_truth * ff).astype(np.uint16)

    c = size // 8
    corner = float(vignetted[:c, :c].mean())
    centre = float(vignetted[size // 2 - c:size // 2 + c, size // 2 - c:size // 2 + c].mean())
    assert corner / centre < 0.85, "the synthetic vignette was too weak to test anything"

    corrected = correct_flatfield(vignetted, FlatfieldProfile(ff))

    corner_c = float(corrected[:c, :c].mean())
    centre_c = float(corrected[size // 2 - c:size // 2 + c, size // 2 - c:size // 2 + c].mean())
    assert 0.98 < corner_c / centre_c < 1.02
    assert np.allclose(corrected, 3000, atol=2)


def test_darkfield_pedestal_is_removed_before_the_gain_divide():
    """Order matters: (raw - dark) / gain, not raw / gain - dark. Applying them in the wrong
    order leaves a residual gradient proportional to the pedestal."""
    size, pedestal = 96, 400.0
    ff = _vignette(size)
    raw = (np.float32(2000.0) * ff + pedestal).astype(np.uint16)
    df = np.full((size, size), pedestal, np.float32)

    corrected = correct_flatfield(raw, FlatfieldProfile(ff, df))
    assert np.allclose(corrected, 2000, atol=3)


def test_estimate_profile_recovers_a_vignette_from_tiles():
    """The estimator is the stitcher's BaSiC (reused, not reimplemented); this pins that the
    reuse is wired up correctly by recovering a field we planted."""
    pytest.importorskip("tilefusion.flatfield")
    size, n = 64, 24
    rng = np.random.default_rng(3)
    ff = _vignette(size)
    tiles = np.stack([
        ((rng.uniform(200, 400) + rng.normal(0, 30, (size, size)).clip(0)) * ff).astype(np.uint16)
        for _ in range(n)
    ])
    est = estimate_profile(tiles)
    assert est.flatfield.shape == (size, size)
    assert abs(float(est.flatfield.mean()) - 1.0) < 1e-3
    corr = np.corrcoef(est.flatfield.ravel(), ff.ravel())[0, 1]
    assert corr > 0.9, f"estimated field does not track the planted one (r={corr:.2f})"


def test_dtype_preserved_input_not_mutated_and_no_integer_wrap():
    ff = _vignette(64)
    raw = (np.float32(60000) * ff).astype(np.uint16)
    before = raw.copy()
    out = correct_flatfield(raw, FlatfieldProfile(ff))
    assert out.dtype == np.uint16
    assert np.array_equal(raw, before)
    assert out.max() <= 65535 and out.min() >= 0
    # the dim corners get divided UP past the ceiling -> must clip, never wrap to black
    assert out[:4, :4].mean() > raw[:4, :4].mean()


def test_shape_mismatch_fails_loud():
    with pytest.raises(ValueError, match="shape"):
        correct_flatfield(np.ones((32, 32), np.uint16), FlatfieldProfile(_vignette(64)))


def test_a_profile_that_is_not_mean_one_is_refused():
    """A profile whose mean is not 1 silently rescales the whole image — a brightness change
    masquerading as a correction. Refuse it by name."""
    with pytest.raises(ValueError, match="mean"):
        FlatfieldProfile(np.full((16, 16), 2.0, np.float32))


# --- THE ALGEBRAIC SHORTCUT: flat-field commutes with the MIP ------------------------------

def _monotone_map_is_exact(ff, planes):
    per_plane = project([correct_flatfield(p, FlatfieldProfile(ff)) for p in planes])
    after_mip = correct_flatfield(project(planes), FlatfieldProfile(ff))
    return per_plane, after_mip


def test_flatfield_commutes_with_the_mip_exactly_on_synthetic_uint16():
    size, nz = 96, 10
    rng = np.random.default_rng(7)
    ff = _vignette(size)
    planes = [(rng.uniform(500, 4000, (size, size)) * ff).astype(np.uint16) for _ in range(nz)]
    per_plane, after_mip = _monotone_map_is_exact(ff, planes)
    assert np.array_equal(per_plane, after_mip), (
        f"commutation broken: max |diff| = {np.abs(per_plane.astype(int) - after_mip.astype(int)).max()}"
    )


def test_commutation_survives_clipping_at_the_uint16_ceiling():
    """Clipping is monotone too, so saturating the result must not break it. This is the case
    a careless float-then-cast implementation gets wrong."""
    size = 64
    ff = _vignette(size, depth=0.9)
    rng = np.random.default_rng(11)
    planes = [rng.integers(50000, 65535, (size, size)).astype(np.uint16) for _ in range(6)]
    per_plane, after_mip = _monotone_map_is_exact(ff, planes)
    assert after_mip.max() == 65535, "this test needs to actually saturate"
    assert np.array_equal(per_plane, after_mip)


def test_commutation_survives_a_darkfield_pedestal_and_clipping_at_zero():
    size = 64
    ff = _vignette(size)
    df = np.full((size, size), 800.0, np.float32)
    rng = np.random.default_rng(13)
    planes = [rng.integers(0, 2000, (size, size)).astype(np.uint16) for _ in range(6)]
    prof = FlatfieldProfile(ff, df)
    per_plane = project([correct_flatfield(p, prof) for p in planes])
    after_mip = correct_flatfield(project(planes), prof)
    assert (after_mip == 0).any(), "this test needs to actually clip at zero"
    assert np.array_equal(per_plane, after_mip)


@pytest.mark.integration
def test_flatfield_commutes_with_the_mip_on_real_10x_data(laser_af_dataset, capsys):
    """The measurement the ticket asks for: bit-exactness AND the measured speedup on real
    2000px, Nz=10 planes — not a synthetic proxy."""
    reader = open_reader(laser_af_dataset)
    meta = reader.metadata
    region = meta["regions"][0]
    fov = meta["fovs_per_region"][region][0]
    channel = meta["channels"][0]["name"]
    z_levels = meta["z_levels"]
    planes = [reader.read(region, fov, channel, z, 0) for z in z_levels]
    assert len(planes) > 1

    ff = _vignette(planes[0].shape[0]) if planes[0].shape[0] == planes[0].shape[1] else None
    if ff is None:  # non-square frame: build the dome at the real aspect
        yy, xx = (np.mgrid[0:planes[0].shape[0], 0:planes[0].shape[1]].astype(np.float32)
                  / np.array(planes[0].shape, np.float32)[:, None, None] - 0.5)
        f = 1.0 - 0.55 * (yy ** 2 + xx ** 2) / 0.5
        ff = (f / f.mean()).astype(np.float32)
    prof = FlatfieldProfile(ff)

    t0 = time.perf_counter()
    per_plane = project([correct_flatfield(p, prof) for p in planes])
    t_per_plane = time.perf_counter() - t0

    t0 = time.perf_counter()
    after_mip = correct_flatfield(project(planes), prof)
    t_after_mip = time.perf_counter() - t0

    diff = np.abs(per_plane.astype(np.int64) - after_mip.astype(np.int64)).max()
    print(f"\n[IMA-225] real data {planes[0].shape} Nz={len(planes)} dtype={planes[0].dtype}: "
          f"max|diff|={diff}  per-plane {t_per_plane * 1000:.1f} ms  "
          f"after-MIP {t_after_mip * 1000:.1f} ms  speedup {t_per_plane / t_after_mip:.1f}x")
    assert diff == 0, "flat-field does NOT commute with the MIP on real data"
    assert t_per_plane > t_after_mip


# --- registry / engine seam ----------------------------------------------------------------

def test_flatfield_is_registered_as_a_plane_op():
    assert "flatfield" in available_projectors()
    assert projector_consumes("flatfield") == PLANE_OP


def test_the_registered_operator_fails_loud_and_actionable_with_no_profile_set():
    """A flat-field with no profile has no sane default — an identity field would silently do
    nothing while the UI said 'flat-field applied'."""
    from squidmip._engine import _resolve_projector
    op = _resolve_projector("flatfield").fn
    with pytest.raises(ValueError, match="no flat-field profile"):
        op([np.ones((8, 8), np.uint16)])


def test_set_profile_activates_the_registered_operator():
    from squidmip._engine import _resolve_projector
    ff = _vignette(32)
    set_profile(FlatfieldProfile(ff))
    assert active_profile() is not None
    raw = (np.float32(1000) * ff).astype(np.uint16)
    out = _resolve_projector("flatfield").fn([raw])
    assert np.allclose(out, 1000, atol=2)


def test_flatfield_op_refuses_a_whole_z_stack():
    op = flatfield_op(FlatfieldProfile(_vignette(8)))
    with pytest.raises(ValueError, match="more than one plane"):
        op([np.ones((8, 8), np.uint16), np.ones((8, 8), np.uint16)])


def test_project_well_with_flatfield_keeps_z_at_full_depth(squid_dataset):
    root, _ = squid_dataset
    reader = open_reader(root)
    ff = np.ones((4, 4), np.float32)
    out = project_well(reader, "B2", 0, reduce=flatfield_op(FlatfieldProfile(ff)))
    assert out.shape[2] == len(reader.metadata["z_levels"])
    assert out.dtype == reader.metadata["dtype"]
