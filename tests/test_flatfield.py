"""Flat-field machinery: numerical property tests + the MIP-commutation shortcut.

The standalone `flatfield` OPERATOR was shelved 2026-08-24; what this file covers is the
machinery STITCH rides (profile record, BaSiC estimate, per-channel .npy parse, the
correction arithmetic, the installed-profile store) plus the absence pins.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from squidxplorer import available_plane_operators, project
from squidxplorer._flatfield import (
    FlatfieldProfile,
    active_profiles,
    clear_profile,
    correct_flatfield,
    estimate_profile,
    set_profile,
    set_profiles,
)
from squidxplorer.reader import open_reader

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
    """The registered operator reads module-level state; don't leak a profile across tests."""
    clear_profile()
    yield
    clear_profile()


def _vignette(size: int = 128, depth: float = 0.55) -> np.ndarray:
    """A radial dome normalised to mean 1."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1) - 0.5
    field = 1.0 - depth * (yy ** 2 + xx ** 2) / 0.5
    return (field / field.mean()).astype(np.float32)


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
    """Order matters: (raw - dark) / gain, not raw / gain - dark."""
    size, pedestal = 96, 400.0
    ff = _vignette(size)
    raw = (np.float32(2000.0) * ff + pedestal).astype(np.uint16)
    df = np.full((size, size), pedestal, np.float32)

    corrected = correct_flatfield(raw, FlatfieldProfile(ff, df))
    assert np.allclose(corrected, 2000, atol=3)


def test_estimate_profile_recovers_a_vignette_from_tiles():
    """Pins that the reused BaSiC estimator is wired up correctly."""
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


def test_estimate_profile_normalises_a_field_the_constructor_would_have_refused():
    """With few tiles BaSiC's gain can land outside FlatfieldProfile's 1e-3 mean tolerance;
    estimate_profile must renormalise it rather than let the constructor raise."""
    import squidxplorer._flatfield as F

    off_by = np.full((8, 8), 1.0, dtype=np.float32)
    off_by[0, 0] = 1.32
    assert abs(float(off_by.mean()) - 1.0) > 1e-3
    with pytest.raises(ValueError, match="normalised to mean 1.0"):
        FlatfieldProfile(off_by)

    def stub(stack, use_darkfield=False):
        return off_by.copy(), None

    # Only the import may skip; an ImportError inside estimate_profile itself must not be absorbed.
    try:
        import tilefusion.flatfield as tff
    except ImportError:
        pytest.skip("tilefusion not installed")
    real, tff.estimate_flatfield_channel = tff.estimate_flatfield_channel, stub
    try:
        est = F.estimate_profile(np.zeros((2, 8, 8), dtype=np.uint16))
    finally:
        tff.estimate_flatfield_channel = real

    assert abs(float(est.flatfield.mean()) - 1.0) < 1e-6
    assert np.corrcoef(est.flatfield.ravel(), off_by.ravel())[0, 1] > 0.999


def test_dtype_preserved_input_not_mutated_and_no_integer_wrap():
    ff = _vignette(64)
    raw = (np.float32(60000) * ff).astype(np.uint16)
    before = raw.copy()
    out = correct_flatfield(raw, FlatfieldProfile(ff))
    assert out.dtype == np.uint16
    assert np.array_equal(raw, before)
    assert out.max() <= 65535 and out.min() >= 0
    # dim corners divide UP past the ceiling: must clip, never wrap to black
    assert out[:4, :4].mean() > raw[:4, :4].mean()


def test_shape_mismatch_fails_loud():
    with pytest.raises(ValueError, match="shape"):
        correct_flatfield(np.ones((32, 32), np.uint16), FlatfieldProfile(_vignette(64)))


def test_a_profile_that_is_not_mean_one_is_refused():
    """A mean != 1 profile would silently rescale brightness rather than correct it."""
    with pytest.raises(ValueError, match="mean"):
        FlatfieldProfile(np.full((16, 16), 2.0, np.float32))


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
    """Clipping is monotone too, so saturating the result must not break commutation."""
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
    """Bit-exactness and speedup on real data, not a synthetic proxy."""
    reader = open_reader(laser_af_dataset)
    meta = reader.metadata
    region = meta["regions"][0]
    fov = meta["fovs_per_region"][region][0]
    channel = meta["channels"][0]["name"]
    z_levels = meta["z_levels"]
    planes = [reader.read(region, fov, channel, z, 0) for z in z_levels]
    assert len(planes) > 1

    ff = _vignette(planes[0].shape[0]) if planes[0].shape[0] == planes[0].shape[1] else None
    if ff is None:
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


def test_the_flatfield_operator_is_shelved_whole():
    """Absence pin (2026-08-24): no registered operator, no callable surface, no export.
    The machinery stitch rides — FlatfieldProfile, estimate_profile, correct_flatfield and
    the set_profile(s)/active_profiles store — deliberately survives, tested above."""
    import squidxplorer
    import squidxplorer._flatfield as ff

    assert "flatfield" not in squidxplorer.runnable_operators()
    assert "flatfield" not in available_plane_operators()
    for gone in ("flatfield_op", "_ACTIVE_OP", "_correct_with_active", "_profile_for",
                 "LAYER_KEY", "LAYER_LABEL"):
        assert not hasattr(ff, gone), f"{gone} is back; the flatfield operator was shelved"
    assert not hasattr(squidxplorer, "flatfield_op")


def test_a_profile_cannot_be_installed_without_saying_which_channel_measured_it():
    """channel is keyword-only and required, so a call with no channel no longer type-checks."""
    with pytest.raises(TypeError):
        set_profile(FlatfieldProfile(_vignette(8)))          # type: ignore[call-arg]
    with pytest.raises(ValueError, match="CHANNEL NAME"):
        set_profile(FlatfieldProfile(_vignette(8)), channel="")


def test_the_profile_store_round_trips_per_channel():
    """set_profile / set_profiles / active_profiles — what _stitch._selected_profiles reads."""
    a, b = FlatfieldProfile(_vignette(8)), FlatfieldProfile(_vignette(8, 0.2))
    set_profile(a, channel="405")
    assert list(active_profiles()) == ["405"]
    set_profiles({"405": a, "488": b})
    got = active_profiles()
    assert sorted(got) == ["405", "488"] and got["488"] is b
    clear_profile()
    assert active_profiles() == {}


@pytest.mark.integration
def test_per_channel_npy_parse_on_the_real_stored_profile(laser_af_dataset, capsys):
    """per_channel_from_npy maps channel NAMES to plane INDICES of the stored (C, Y, X) file;
    correcting a channel with plane 0's field is measurably wrong (the 2026-08-06 defect)."""
    npy = laser_af_dataset / f"{laser_af_dataset.name}_flatfield.npy"
    if not npy.exists():
        pytest.skip("this acquisition carries no stored flat-field profile")
    reader = open_reader(laser_af_dataset)
    meta = reader.metadata
    names = [c["name"] for c in meta["channels"]]
    region = meta["regions"][0]
    fov = meta["fovs_per_region"][region][0]
    z = meta["z_levels"][len(meta["z_levels"]) // 2]

    profiles = FlatfieldProfile.per_channel_from_npy(npy, names)
    print(f"\n[per-channel flat-field] {npy.name}")
    any_differs = False
    for i, name in enumerate(names):
        raw = reader.read(region, fov, name, z, 0)
        right = correct_flatfield(raw, profiles[name])
        by_index = correct_flatfield(raw, FlatfieldProfile.from_npy(npy, channel=i))
        assert np.array_equal(right, by_index), (
            f"{name}: per_channel_from_npy did not map this NAME to plane {i}")
        wrong = correct_flatfield(raw, FlatfieldProfile.from_npy(npy, channel=0))
        d = np.abs(wrong.astype(np.int64) - right.astype(np.int64))
        any_differs = any_differs or bool((d > 0).any())
        print(f"  {name:30s} one-profile mean {wrong.mean():9.2f}  per-channel mean "
              f"{right.mean():9.2f}  differing {100.0 * (d > 0).mean():7.3f}%  max {d.max()}")
    assert any_differs, "every channel matched plane 0's field; this file cannot see the defect"
