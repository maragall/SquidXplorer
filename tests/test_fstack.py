"""fstack: the noise-robust selective all-in-focus fusion (Pertuz SAF), as an operator.

Pins the ported math (gfocus, gauss3P's end-swap quirk, the weight path), the algorithm's
defining properties against MIP, the native-dtype divergence, the registry declaration
(advanced params included) and the acquisition-format save round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as ndi
import tifffile

import squidxplorer as s
from squidxplorer._fstack import (
    MIN_PLANES,
    fstack_op,
    fuse_stack,
    gauss3P,
    gfocus,
)
from squidxplorer.projection import project, project_well


# ------------------------------------------------------------------ 1. the algorithm pins

def test_gfocus_hand_checked_on_a_delta():
    """3x3 zeros with a 9 at the centre, window 3, replicate borders: FM is 8 everywhere.

    Mean is 1 at every position (the 9 is in every window exactly once under nearest
    padding), so (im-U)^2 is 64 centre / 1 elsewhere and every 3x3 window sums to 72.
    """
    im = np.zeros((3, 3))
    im[1, 1] = 9.0
    np.testing.assert_allclose(gfocus(im, 3), np.full((3, 3), 8.0))


def test_gfocus_is_zero_on_a_constant_and_scales_as_variance():
    im = np.arange(25.0).reshape(5, 5)
    np.testing.assert_allclose(gfocus(np.full((5, 5), 7.0), 3), 0.0, atol=1e-12)
    np.testing.assert_allclose(gfocus(2.0 * im, 3), 4.0 * gfocus(im, 3), rtol=1e-12)


def test_gauss3P_end_swaps_peak_the_fit_at_the_argmax_frame():
    """The MATLAB's Index1/Index3 end-swaps make y3 == y1 at EVERY pixel, so the fitted
    Gaussian's mean is the (clamped) argmax frame index — ported verbatim, not repaired."""
    P = 9
    focus = np.arange(P, dtype=np.float64)
    profile = np.exp(-((focus - 4.3) ** 2) / (2 * 1.7 ** 2))    # true peak between frames
    fm = np.tile(profile[:, None, None], (1, 4, 4))
    u, sig, big_a, fmax = gauss3P(focus, fm)
    np.testing.assert_allclose(u, 4.0)               # the argmax frame, never 4.3
    np.testing.assert_allclose(fmax, profile.max())
    assert np.all(np.isfinite(sig)) and np.all(sig > 0)
    assert np.all(np.isfinite(big_a))


def test_gauss3P_clamps_an_edge_peak_into_the_interior():
    P = 7
    focus = np.arange(P, dtype=np.float64)
    fm = np.tile(np.array([5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0])[:, None, None], (1, 2, 2))
    u, _, _, fmax = gauss3P(focus, fm)               # argmax 0 clamps to STEP = 2
    np.testing.assert_allclose(u, 2.0)
    np.testing.assert_allclose(fmax, 5.0)


def test_identical_planes_fuse_to_themselves_exactly():
    """fm/fmax == 1 in every frame -> every weight is 0.5+0.5*tanh(0); the fusion is a
    convex combination, so equal weights return the plane bit-exactly."""
    rng = np.random.default_rng(3)
    plane = rng.integers(100, 40000, (32, 32)).astype(np.uint16)
    np.testing.assert_array_equal(fuse_stack([plane] * MIN_PLANES), plane)


def test_the_weights_favor_the_sharp_frame():
    rng = np.random.default_rng(7)
    sharp = rng.integers(500, 4000, (48, 48)).astype(np.float64)
    stack = [ndi.gaussian_filter(sharp, 4.0) if k != 2 else sharp for k in range(6)]
    fused = fuse_stack([p.astype(np.uint16) for p in stack])
    err_sharp = np.abs(fused.astype(np.float64) - sharp).mean()
    err_mean = np.abs(np.mean(stack, axis=0) - sharp).mean()
    assert err_sharp < 0.5 * err_mean, (
        f"fusion sits {err_sharp:.1f} counts from the sharp frame vs the plain mean's "
        f"{err_mean:.1f}; the selectivity weights are not selecting")


# ------------------------------------------------- 2. two regions, two sharp frames

def _two_region_stack(P=10, size=96):
    """Region A is sharp in frame 2, region B in frame 7, background flat; the known
    all-in-focus composite is the unblurred scene."""
    rng = np.random.default_rng(11)
    scene = np.full((size, size), 300.0)
    scene[8:40, 8:40] = rng.integers(100, 4000, (32, 32))
    scene[56:88, 56:88] = rng.integers(100, 4000, (32, 32))
    mask_b = np.zeros((size, size), bool)
    mask_b[48:96, 48:96] = True
    frames = []
    for k in range(P):
        fa = ndi.gaussian_filter(scene, 1.2 * abs(k - 2)) if k != 2 else scene
        fb = ndi.gaussian_filter(scene, 1.2 * abs(k - 7)) if k != 7 else scene
        frame = np.where(mask_b, fb, fa)
        frames.append(np.clip(frame, 0, 65535).astype(np.uint16))
    return frames, scene.astype(np.float64)


def test_each_region_is_taken_from_its_own_sharp_frame_and_beats_mip():
    frames, composite = _two_region_stack()
    fused = fuse_stack(frames).astype(np.float64)
    mip = project(frames).astype(np.float64)

    def rmse(a):
        return float(np.sqrt(np.mean((a - composite) ** 2)))

    assert rmse(fused) < rmse(mip), (
        f"fstack RMSE {rmse(fused):.1f} vs MIP {rmse(mip):.1f} against the known composite")
    # and per region, against each region's own sharp frame
    for sl in (np.s_[8:40, 8:40], np.s_[56:88, 56:88]):
        region_err = np.abs(fused[sl] - composite[sl]).mean()
        mip_err = np.abs(mip[sl] - composite[sl]).mean()
        assert region_err < mip_err, (
            f"region {sl}: fstack {region_err:.1f} vs MIP {mip_err:.1f}")
    corr_f = np.corrcoef(fused.ravel(), composite.ravel())[0, 1]
    corr_m = np.corrcoef(mip.ravel(), composite.ravel())[0, 1]
    assert corr_f > corr_m


# ------------------------------------------------------------------ 3. noise robustness

def test_the_fused_background_is_smoother_than_mip_under_noise():
    rng = np.random.default_rng(23)
    P, size = 6, 96
    scene = np.full((size, size), 1000.0)
    scene[8:24, 8:24] = 4000.0                       # some structure so focus exists
    frames = []
    for k in range(P):
        f = ndi.gaussian_filter(scene, 1.0 * abs(k - 3)) if k != 3 else scene
        f = f + rng.normal(0.0, 50.0, (size, size))
        frames.append(np.clip(f, 0, 65535).astype(np.uint16))
    flat = np.s_[48:92, 8:92]                        # far from the structure
    fused_std = float(np.std(fuse_stack(frames)[flat].astype(np.float64)))
    mip_std = float(np.std(project(frames)[flat].astype(np.float64)))
    assert fused_std < mip_std, (
        f"background std {fused_std:.1f} (fstack) vs {mip_std:.1f} (MIP); the noise-robust "
        "fusion must not amplify background noise the way a max projection does")


# ------------------------------------------------------------------ 4. dtype preservation

def test_uint16_in_uint16_out_no_wrap_and_within_the_stack_range():
    rng = np.random.default_rng(31)
    frames = [rng.integers(60000, 65535, (24, 24)).astype(np.uint16) for _ in range(5)]
    fused = fuse_stack(frames)
    assert fused.dtype == np.uint16
    lo = np.min(np.stack(frames), axis=0)
    hi = np.max(np.stack(frames), axis=0)
    assert np.all(fused.astype(np.int64) >= lo.astype(np.int64) - 1)
    assert np.all(fused.astype(np.int64) <= hi.astype(np.int64) + 1), (
        "the fusion is a convex combination per pixel; a value outside the stack's own "
        "range means an overflow or wrap")


def test_float32_in_float32_out():
    rng = np.random.default_rng(37)
    frames = [rng.random((16, 16)).astype(np.float32) for _ in range(5)]
    assert fuse_stack(frames).dtype == np.float32


# ------------------------------------------------------------------ 5. the operator contract

class _StackReader:
    """Five z planes per (region, fov, channel, t): sharp texture at z=2, blur elsewhere."""

    def __init__(self, n_t=2, size=24):
        self.metadata = {
            "regions": ["A1"],
            "channels": [{"name": "405"}, {"name": "488"}],
            "n_z": 5, "z_levels": [0, 1, 2, 3, 4], "n_t": n_t, "dz_um": 1.0,
            "dtype": "uint16", "frame_shape": (size, size), "pixel_size_um": 1.0,
            "fovs_per_region": {"A1": [0]},
        }
        rng = np.random.default_rng(41)
        self._sharp = {c: rng.integers(200, 4000, (size, size)).astype(np.float64)
                       for c in ("405", "488")}

    def read(self, region, fov, channel, z_level, time_point=0):
        sharp = self._sharp[str(channel)] + 10.0 * int(time_point)
        z = int(z_level)
        plane = sharp if z == 2 else ndi.gaussian_filter(sharp, 1.5 * abs(z - 2))
        return np.clip(plane, 0, 65535).astype(np.uint16)


def test_fstack_is_registered_as_a_core_z_reducer_with_advanced_params():
    assert "fstack" in s.runnable_operators()
    assert s.operator_consumes("fstack") == frozenset({"z"})
    assert s.operator_produces("fstack") == "intensity"
    assert not s.is_region_operator("fstack")
    assert s.operator_extra("fstack") is None                 # core: no install extra
    params = {p.name: p for p in s.operator_params("fstack")}
    assert set(params) == {"nhsize", "alpha", "sth"}
    for p in params.values():
        assert p.advanced is True, f"{p.name} must be an advanced knob"
    # the field's default leaves every existing declaration untouched
    assert s.Param("x", 1).advanced is False


def test_fstack_runs_through_project_well_with_the_z_reducer_shape():
    reader = _StackReader()
    out = project_well(reader, "A1", 0, reduce=s.bind_operator("fstack"))
    assert out.shape == (2, 2, 1, 24, 24)
    assert out.dtype == np.uint16
    # per-channel independence: each channel's fusion comes from its own planes
    ch0 = fuse_stack([reader.read("A1", 0, "405", z, 0) for z in range(5)])
    ch1 = fuse_stack([reader.read("A1", 0, "488", z, 0) for z in range(5)])
    np.testing.assert_array_equal(out[0, 0, 0], ch0)
    np.testing.assert_array_equal(out[0, 1, 0], ch1)
    assert not np.array_equal(ch0, ch1)


def test_a_declared_parameter_reaches_the_pixels_through_bind_operator():
    reader = _StackReader()
    planes = [reader.read("A1", 0, "405", z) for z in range(5)]
    base = s.bind_operator("fstack")(iter(planes))
    tuned = s.bind_operator("fstack", {"nhsize": 3})(iter(planes))
    assert not np.array_equal(base, tuned)


def test_too_few_planes_is_a_named_refusal():
    plane = np.zeros((8, 8), np.uint16)
    with pytest.raises(ValueError, match="at least 5 z planes"):
        fuse_stack([plane] * 4)
    with pytest.raises(ValueError, match="'mip'"):
        fuse_stack([plane] * 2)


def test_the_factory_refuses_bad_knobs_by_name():
    with pytest.raises(ValueError, match="alpha"):
        fstack_op(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        fstack_op(alpha=1.5)
    with pytest.raises(ValueError, match="nhsize"):
        fstack_op(nhsize=0)


def test_mismatched_planes_are_refused():
    a = np.zeros((8, 8), np.uint16)
    with pytest.raises(ValueError, match="shape"):
        fuse_stack([a] * 4 + [np.zeros((8, 9), np.uint16)])
    with pytest.raises(ValueError, match="dtype"):
        fuse_stack([a] * 4 + [np.zeros((8, 8), np.uint8)])


# ------------------------------------------------------------------ 6. the acquisition save

_FS_REGION = "B2"
_FS_CH = "Fluorescence_638_nm_-_Penta"
_FS_NZ = 5

_FS_CHANNELS_YAML = """\
version: 1
channels:
- name: Fluorescence 638 nm - Penta
  camera_settings:
    '1':
      display_color: '#FF0000'
      exposure_time_ms: 50.0
"""

_FS_ACQ_YAML = f"""\
objective:
  pixel_size_um: 0.325
  magnification: 20.0
  sensor_pixel_size_um: 3.76
sample:
  wellplate_format: 96 well plate
z_stack:
  nz: {_FS_NZ}
  delta_z_mm: 0.0015
time_series:
  nt: 1
"""


@pytest.fixture
def fstack_dataset(tmp_path):
    """A tiny on-disk acquisition with the 5 z planes fstack needs; returns (root, planes)."""
    import json

    root = tmp_path / "acq5"
    folder = root / "0"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(53)
    sharp = rng.integers(200, 4000, (12, 12)).astype(np.float64)
    planes = []
    for z in range(_FS_NZ):
        plane = sharp if z == 2 else ndi.gaussian_filter(sharp, 1.5 * abs(z - 2))
        plane = np.clip(plane, 0, 65535).astype(np.uint16)
        planes.append(plane)
        tifffile.imwrite(folder / f"{_FS_REGION}_0_{z}_{_FS_CH}.tiff", plane)
    (root / "acquisition_channels.yaml").write_text(_FS_CHANNELS_YAML)
    (root / "acquisition.yaml").write_text(_FS_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps({
        "Nz": _FS_NZ, "Nt": 1, "dz(um)": 1.5,
        "objective": {"magnification": 20.0, "NA": 0.8},
        "sensor_pixel_size_um": 3.76,
    }))
    rows = ["region,x (mm),y (mm),z (mm)"]
    rows += [f"{_FS_REGION},10.0,20.0," for _ in range(_FS_NZ)]
    (root / "coordinates.csv").write_text("\n".join(rows) + "\n")
    return root, planes


def test_the_save_routes_to_acquisition_format_and_round_trips(fstack_dataset, tmp_path):
    from squidxplorer._acq_output import acquisition_format_dst, write_acquisition_planes

    root, planes = fstack_dataset
    reader = s.open_reader(root)
    dst = acquisition_format_dst(reader, "fstack")
    assert dst == root.parent / "fstack_acq5", (
        "a per-FOV intensity z-reducer's save lands in acquisition format automatically")

    summary = write_acquisition_planes(reader, "fstack", dst)
    assert summary["complete"] and not summary["stopped"]
    assert summary["n_fields_written"] == 1

    # one plane per FOV/channel at z filename index 0
    assert sorted(p.name for p in (dst / "0").iterdir()) == [
        f"{_FS_REGION}_0_0_{_FS_CH}.tiff"]
    # pixels are exactly the fusion of the source planes at the defaults
    written = tifffile.imread(dst / "0" / f"{_FS_REGION}_0_0_{_FS_CH}.tiff")
    np.testing.assert_array_equal(written, fuse_stack(planes))

    out = s.open_reader(dst)
    meta = out.metadata
    assert meta["n_z"] == 1 and list(meta["z_levels"]) == [0]
    np.testing.assert_array_equal(out.read(_FS_REGION, 0, _FS_CH, 0), fuse_stack(planes))
