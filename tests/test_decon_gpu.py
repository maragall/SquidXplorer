"""The GPU decon backend must be a backend: same algorithm, same numbers, different device.

The load-bearing test is "the GPU path returns petakit's array"; the rest covers the
device decision, which must be deterministic and degrade to CPU rather than raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer import _decon_gpu
from squidxplorer._decon import (
    DEFAULT_OPTICS,
    METHOD,
    OpticsParams,
    deconvolve_stack,
    make_psf,
)


def _psf_1z(optics):
    """The in-focus PSF plane, ``(1, Y, X)`` sum 1 — what the shelved ``make_psf_2d`` built,
    derived exactly as the nz=1 volume solve derives it (PSF depth follows the stack depth)."""
    import numpy as _np

    psf = make_psf(OpticsParams(optics.na, optics.wavelength_um, optics.dxy_um,
                                optics.dz_um, 1, optics.ni))
    centre = psf[psf.shape[0] // 2]
    return _np.ascontiguousarray((centre / centre.sum())[None, ...], dtype=_np.float32)


def _plane_solve(plane, optics, iterations):
    """RL on ONE plane through the one code path (the shelved ``deconvolve_plane``'s heir)."""
    import numpy as _np

    return deconvolve_stack(_np.asarray(plane)[None, ...], optics, iterations,
                            project=False)[0]
from squidxplorer.projection import cast_like

petakit = pytest.importorskip("petakit")

#: Small and 7-smooth (256 = 2^8), so the test exercises the accepted branch and stays fast.
SIZE = 256
ITERATIONS = 3

#: Relative-to-peak agreement between the two backends.
TOLERANCE = 1e-4

#: After the cast to the acquisition dtype: at most one count, on hardly any pixels.
MAX_COUNTS = 1
MAX_DIFFERING_FRACTION = 1e-3


def _phantom(shape, seed=0):
    """Puncta on a background: structure with real dynamic range, which is what RL amplifies."""
    rng = np.random.default_rng(seed)
    volume = rng.random(shape).astype(np.float32) * 200.0 + 100.0
    for _ in range(24):
        z = rng.integers(0, shape[0])
        y = rng.integers(8, shape[1] - 8)
        x = rng.integers(8, shape[2] - 8)
        volume[z, y - 2:y + 3, x - 2:x + 3] += 4000.0
    return np.ascontiguousarray(volume, dtype=np.float32)


def _device_or_skip():
    device = _decon_gpu._torch_device()
    if device is None:
        pytest.skip("no torch GPU device (MPS/CUDA) on this machine")
    return device


# --- the equivalence ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, SIZE, SIZE), (5, SIZE, SIZE)])
def test_gpu_backend_reproduces_petakit_cpu_within_tolerance(shape):
    """2-D plane and 3-D stack alike: the device result is petakit's result."""
    device = _device_or_skip()
    psf = _psf_1z(DEFAULT_OPTICS) if shape[0] == 1 else make_psf(DEFAULT_OPTICS)
    volume = _phantom(shape)

    cpu = petakit.deconvolve(volume, psf, method=METHOD, iterations=ITERATIONS, gpu=False)
    gpu = _decon_gpu.rl(volume, psf, ITERATIONS, device)

    assert gpu.shape == cpu.shape
    assert gpu.dtype == np.float32
    peak = float(np.abs(cpu).max())
    assert peak > 0, "the CPU reference is all zero; the phantom is not exercising anything"
    relative = float(np.abs(gpu - cpu).max()) / peak
    assert relative <= TOLERANCE, f"backends disagree by {relative:.3e} of peak (limit {TOLERANCE})"


def _assert_quantised_agreement(gpu, cpu):
    """No pixel moves by more than one count, and hardly any pixel moves at all."""
    assert cpu.dtype == gpu.dtype == np.uint16
    delta = np.abs(gpu.astype(np.int32) - cpu.astype(np.int32))
    worst = int(delta.max())
    fraction = float(np.count_nonzero(delta) / delta.size)
    assert worst <= MAX_COUNTS, f"a pixel moved by {worst} counts (limit {MAX_COUNTS})"
    assert fraction <= MAX_DIFFERING_FRACTION, (
        f"{fraction:.4%} of pixels moved (limit {MAX_DIFFERING_FRACTION:.2%}); "
        "that is too many to be quantisation and looks like real drift"
    )


def test_the_uint16_planes_the_two_backends_write_agree_to_the_quantisation_step():
    """The same pixels on disk, to within one count."""
    device = _device_or_skip()
    psf = _psf_1z(DEFAULT_OPTICS)
    volume = _phantom((1, SIZE, SIZE), seed=3)

    cpu = cast_like(petakit.deconvolve(volume, psf, method=METHOD,
                                        iterations=ITERATIONS, gpu=False)[0], np.dtype(np.uint16))
    gpu = cast_like(_decon_gpu.rl(volume, psf, ITERATIONS, device)[0], np.dtype(np.uint16))

    _assert_quantised_agreement(gpu, cpu)


def test_the_plane_solve_goes_through_the_device_and_still_matches_the_cpu_path(monkeypatch):
    """``_run``'s fork must not change the nz=1 volume solve."""
    _device_or_skip()
    from squidxplorer import _decon

    plane = _phantom((1, SIZE, SIZE), seed=7)[0].astype(np.uint16)

    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    on_cpu = _plane_solve(plane, DEFAULT_OPTICS, ITERATIONS)
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "auto")
    on_gpu = _plane_solve(plane, DEFAULT_OPTICS, ITERATIONS)

    assert on_gpu.dtype == plane.dtype
    _assert_quantised_agreement(on_gpu, on_cpu)


# --- the device decision -----------------------------------------------------------------------


@pytest.mark.parametrize("n,smooth", [
    (2048, True),       # 2^11
    (2160, True),       # 2^4 x 3^3 x 5
    (2800, True),       # 2^4 x 5^2 x 7
    (3000, True),       # 2^3 x 3 x 5^3
    (2084, False),      # 2^2 x 521, this instrument's frame width
    (3036, False),      # 2^2 x 3 x 11 x 23
    (4024, False),      # 2^3 x 503
    (1, True),
])
def test_is_smooth_matches_the_measured_fast_and_slow_transform_lengths(n, smooth):
    assert _decon_gpu.is_smooth(n) is smooth


def test_the_guard_now_tests_the_padded_length_not_the_raw_one():
    """Without a PSF shape the guard refuses 2084; with one it sees the padded 2160."""
    _device_or_skip()
    assert _decon_gpu.select_device((1, AWKWARD, AWKWARD)) is None
    assert _decon_gpu.select_device((1, AWKWARD, AWKWARD), psf_shape=(1, 19, 19)) is not None
    assert _decon_gpu.select_device((1, 2048, 2048)) is not None


def test_with_padding_off_the_guard_refuses_the_awkward_width_again(monkeypatch):
    _device_or_skip()
    monkeypatch.setenv(_decon_gpu.PAD_ENV_VAR, "off")
    assert _decon_gpu.effective_shape((1, AWKWARD, AWKWARD), (1, 19, 19)) == (1, AWKWARD, AWKWARD)
    assert _decon_gpu.select_device((1, AWKWARD, AWKWARD), psf_shape=(1, 19, 19)) is None


def test_effective_shape_reports_what_the_fft_will_really_run_at():
    assert _decon_gpu.effective_shape((1, AWKWARD, AWKWARD), (1, 19, 19)) == (1, 2160, 2160)
    assert _decon_gpu.effective_shape((10, 2048, 2048), (10, 19, 19)) == (10, 2048, 2048)
    assert _decon_gpu.effective_shape((1, AWKWARD, AWKWARD)) == (1, AWKWARD, AWKWARD)


def test_gpu_false_is_honoured_verbatim():
    assert _decon_gpu.select_device((1, 256, 256), gpu=False) is None


@pytest.mark.parametrize("value", ["cpu", "off", "0", "none", "false", "CPU", " Off "])
def test_the_env_var_can_turn_the_backend_off(monkeypatch, value):
    monkeypatch.setenv(_decon_gpu.ENV_VAR, value)
    assert _decon_gpu.select_device((1, 256, 256)) is None


def test_the_env_var_can_force_a_device_past_the_smoothness_guard(monkeypatch):
    device = _device_or_skip()
    monkeypatch.setenv(_decon_gpu.ENV_VAR, device)
    assert _decon_gpu.select_device((1, 2084, 2084)) == device


# --- transform-length padding -----------------------------------------------------------------
# A padding bug shows up as a rim, so assertions below target the border explicitly.


#: Non-7-smooth, and the width of this instrument's actual camera frames.
AWKWARD = 2084


def _rim_phantom(shape, seed=11):
    """Puncta plus a bright rim clamped to the outermost pixels."""
    volume = _phantom(shape, seed=seed)
    volume[:, :4, :] = volume[:, -4:, :] = 3500.0
    volume[:, :, :4] = volume[:, :, -4:] = 3500.0
    return np.ascontiguousarray(volume, dtype=np.float32)


def _border_max(a, width):
    mask = np.zeros(a.shape, bool)
    mask[..., :width, :] = mask[..., -width:, :] = True
    mask[..., :, :width] = mask[..., :, -width:] = True
    return float(np.abs(a[mask]).max())


def test_fast_len_returns_the_shortest_7_smooth_length_and_never_an_11_smooth_one():
    assert _decon_gpu.fast_len(2084) == 2100
    assert _decon_gpu.fast_len(2122) == 2160
    assert _decon_gpu.fast_len(2048) == 2048          # already fast, unchanged
    for n in (1, 2, 97, 1000, 2084, 3036, 4024):
        got = _decon_gpu.fast_len(n)
        assert got >= n and _decon_gpu.is_smooth(got)
    # scipy would answer 2112 = 2^6 x 3 x 11 here; 11 must not be accepted
    assert not _decon_gpu.is_smooth(2112)
    assert _decon_gpu.fast_len(2102) == 2160


def test_pad_plan_leaves_already_fast_axes_alone_including_z():
    widths = _decon_gpu.pad_plan((10, AWKWARD, AWKWARD), (10, 19, 19))
    assert widths[0] == 0, "z was already 7-smooth and must not be padded"
    assert widths[1] == widths[2] > 0
    assert AWKWARD + 2 * widths[1] == 2160
    assert _decon_gpu.pad_plan((1, 2048, 2048), (1, 19, 19)) == (0, 0, 0)


def test_pad_plan_gives_at_least_the_psf_extent_which_is_what_makes_the_wrap_exact():
    """A pad narrower than the PSF leaks into the border."""
    widths = _decon_gpu.pad_plan((1, AWKWARD, AWKWARD), (1, 19, 19))
    assert widths[1] >= 19, f"pad {widths[1]} is under the PSF extent, the wrap will leak"


def test_pad_plan_declines_rather_than_growing_an_axis_without_bound(monkeypatch):
    monkeypatch.setattr(_decon_gpu, "MAX_PAD_GROWTH", 1.0001)
    assert _decon_gpu.pad_plan((1, AWKWARD, AWKWARD), (1, 19, 19)) == (0, 0, 0)


def test_wrap_pad_reproduces_the_opposite_edge_and_not_zeros_or_a_replicated_edge():
    """Only wrap matches the circular convolution petakit already performs."""
    a = np.arange(1, 7, dtype=np.float32).reshape(1, 2, 3)
    out = _decon_gpu._wrap_pad(a, (0, 0, 2))
    assert out.shape == (1, 2, 7)
    np.testing.assert_array_equal(out[0, 0], [2, 3, 1, 2, 3, 1, 2])
    assert _decon_gpu._wrap_pad(a, (0, 0, 0)) is a


def test_padding_leaves_the_border_pixels_alone_on_an_awkward_width(monkeypatch):
    """Pad, run, crop, and check the rim against petakit's unpadded CPU output."""
    device = _device_or_skip()
    volume = _rim_phantom((1, AWKWARD, AWKWARD))
    psf = _psf_1z(DEFAULT_OPTICS)

    assert any(_decon_gpu.pad_plan(volume.shape, psf.shape)), "this width must actually pad"
    reference = petakit.deconvolve(volume, psf, method=METHOD, iterations=ITERATIONS, gpu=False)
    got = _decon_gpu.rl(volume, psf, ITERATIONS, device)

    assert got.shape == volume.shape, "the crop must restore the exact input extent"
    peak = float(np.abs(reference).max())
    delta = np.abs(got - reference)
    assert float(delta.max()) / peak <= TOLERANCE
    assert _border_max(delta, 64) / peak <= TOLERANCE


def test_end_to_end_at_the_real_camera_width_cpu_and_gpu_write_the_same_plane(monkeypatch):
    """uint16 in, uint16 out, through the nz=1 volume solve, at the real camera width."""
    _device_or_skip()
    from squidxplorer import _decon

    plane = _rim_phantom((1, AWKWARD, AWKWARD), seed=13)[0].astype(np.uint16)
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    monkeypatch.delenv(_decon_gpu.PAD_CPU_ENV_VAR, raising=False)
    on_cpu = _plane_solve(plane, DEFAULT_OPTICS, ITERATIONS)
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "auto")
    on_gpu = _plane_solve(plane, DEFAULT_OPTICS, ITERATIONS)

    assert on_gpu.shape == plane.shape and on_gpu.dtype == plane.dtype
    _assert_quantised_agreement(on_gpu, on_cpu)
    # and the rim specifically, which is where a padding bug would live
    rim = np.abs(on_gpu.astype(np.int32) - on_cpu.astype(np.int32))
    assert _border_max(rim, 64) <= MAX_COUNTS


def test_a_3d_stack_pads_yx_but_never_the_acquired_depth(monkeypatch):
    """decon3d must not inflate z: the stack depth is the acquisition's."""
    device = _device_or_skip()
    volume = _rim_phantom((5, 514, 514), seed=17)
    psf = make_psf(DEFAULT_OPTICS)
    widths = _decon_gpu.pad_plan(volume.shape, psf.shape)
    assert widths[0] == 0, "z (5) is 7-smooth and must be left exactly as acquired"
    assert widths[1] > 0 and widths[2] > 0

    reference = petakit.deconvolve(volume, psf, method=METHOD, iterations=ITERATIONS, gpu=False)
    got = _decon_gpu.rl(volume, psf, ITERATIONS, device)
    assert got.shape == volume.shape
    peak = float(np.abs(reference).max())
    assert float(np.abs(got - reference).max()) / peak <= TOLERANCE
    assert _border_max(np.abs(got - reference), 32) / peak <= TOLERANCE


def test_the_padded_result_is_not_merely_the_unpadded_one_by_accident(monkeypatch):
    """Break the pad and the border assertion must fire, or that test measures nothing."""
    device = _device_or_skip()
    volume = _rim_phantom((1, AWKWARD, AWKWARD))
    psf = _psf_1z(DEFAULT_OPTICS)
    reference = petakit.deconvolve(volume, psf, method=METHOD, iterations=ITERATIONS, gpu=False)
    peak = float(np.abs(reference).max())

    monkeypatch.setattr(_decon_gpu, "_wrap_pad",
                        lambda v, w: np.pad(v, [(x, x) for x in w]) if any(w) else v)
    broken = _decon_gpu.rl(volume, psf, ITERATIONS, device)
    border = _border_max(np.abs(broken - reference), 64) / peak
    assert border > TOLERANCE, (
        f"zero-padding only moved the border by {border:.2e}, under the {TOLERANCE} limit; "
        "the border test cannot be detecting a padding bug"
    )


def test_restricting_lambda_to_the_true_region_is_load_bearing():
    """A lambda reduced over the padded array shifts every pixel; the restricted one must win."""
    device = _device_or_skip()
    volume = _rim_phantom((1, AWKWARD, AWKWARD), seed=5)
    psf = _psf_1z(DEFAULT_OPTICS)
    widths = _decon_gpu.pad_plan(volume.shape, psf.shape)
    core = tuple(slice(w, w + n) for w, n in zip(widths, volume.shape))

    reference = petakit.deconvolve(volume, psf, method=METHOD, iterations=ITERATIONS, gpu=False)
    peak = float(np.abs(reference).max())

    ours = _decon_gpu.rl(volume, psf, ITERATIONS, device)
    padded_input = np.ascontiguousarray(_decon_gpu._wrap_pad(volume, widths))
    global_lambda = petakit.deconvolve(padded_input, psf, method=METHOD,
                                       iterations=ITERATIONS, gpu=False)[core]

    ours_err = float(np.abs(ours - reference).max()) / peak
    global_err = float(np.abs(global_lambda - reference).max()) / peak
    assert ours_err <= TOLERANCE
    assert global_err > ours_err, (
        f"a global lambda scored {global_err:.2e} against our {ours_err:.2e}; if it is not "
        "worse, restricting the reduction is not doing anything and the code should be simpler"
    )


def test_padding_can_be_switched_off_and_then_nothing_is_padded(monkeypatch):
    monkeypatch.setenv(_decon_gpu.PAD_ENV_VAR, "off")
    assert _decon_gpu.padding_disabled()
    assert not _decon_gpu.cpu_padding_enabled()
    assert "padded" not in _decon_gpu.describe((1, AWKWARD, AWKWARD), psf_shape=(1, 19, 19))


def test_the_cpu_path_is_not_padded_unless_explicitly_asked(monkeypatch):
    """CPU stays bit-stable by default."""
    monkeypatch.delenv(_decon_gpu.PAD_CPU_ENV_VAR, raising=False)
    assert not _decon_gpu.cpu_padding_enabled()
    monkeypatch.setenv(_decon_gpu.PAD_CPU_ENV_VAR, "1")
    assert _decon_gpu.cpu_padding_enabled()


#: Opt-in CPU padding is held to a looser bar: petakit's lambda is reduced over the
#: whole padded array and cannot be restricted from this layer.
CPU_PAD_MAX_COUNTS = 3


def test_opting_the_cpu_path_into_padding_keeps_the_extent_and_stays_below_shot_noise(monkeypatch):
    """Opt-in CPU padding must return the right shape and stay a few counts from unpadded."""
    from squidxplorer import _decon

    # 514 = 2 x 257, not 7-smooth, so the pad plan is non-trivial
    assert not _decon_gpu.is_smooth(514)
    plane = _rim_phantom((1, 514, 514), seed=9)[0].astype(np.uint16)
    assert any(_decon_gpu.pad_plan((1, 514, 514), (1, 19, 19)))
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    monkeypatch.delenv(_decon_gpu.PAD_CPU_ENV_VAR, raising=False)
    plain = _plane_solve(plane, DEFAULT_OPTICS, ITERATIONS)
    monkeypatch.setenv(_decon_gpu.PAD_CPU_ENV_VAR, "1")
    padded = _plane_solve(plane, DEFAULT_OPTICS, ITERATIONS)

    assert padded.shape == plane.shape
    delta = np.abs(padded.astype(np.int32) - plain.astype(np.int32))
    worst = int(delta.max())
    assert worst <= CPU_PAD_MAX_COUNTS, f"padded CPU moved a pixel by {worst} counts"
    assert worst >= 1, (
        "padded and unpadded CPU came out identical, so either the pad did not happen or "
        "petakit's lambda is not global after all; both would make this test vacuous"
    )


def test_forcing_a_device_this_machine_does_not_have_falls_back_instead_of_raising(monkeypatch):
    """A forced device this machine lacks must degrade, not explode from inside torch."""
    monkeypatch.setattr(_decon_gpu, "_torch_device", lambda: "cuda")
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "mps")
    assert _decon_gpu.select_device((1, 256, 256)) is None


def test_an_unknown_env_value_fails_loud_rather_than_guessing(monkeypatch):
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "metal")
    with pytest.raises(ValueError, match="is not a device"):
        _decon_gpu.select_device((1, 256, 256))


# --- the device budget and z tiling ------------------------------------------------------------
# The budget is the DEVICE's, and a volume over it is solved in z tiles with petakit's own
# overlap rule (tests/test_engine.py in petakit pins the CPU half of the same rule).

#: One 256 x 256 plane's working set, petakit's estimate.
PLANE_BYTES = 256 * 256 * _decon_gpu.BYTES_PER_VOXEL


def _budget_for(planes: int) -> float:
    """A budget that fits exactly *planes* planes of 256 x 256 and not twice as many."""
    return planes * PLANE_BYTES * 1.05


def _force_budget(monkeypatch, budget: float) -> None:
    monkeypatch.setattr(_decon_gpu, "_device_free_bytes",
                        lambda device: budget / _decon_gpu.MEMORY_FRACTION)


def test_the_budget_is_the_devices_own_not_the_systems(monkeypatch):
    monkeypatch.setattr(_decon_gpu, "_free_bytes", lambda: 20e9)
    monkeypatch.setattr(_decon_gpu, "_mps_recommended_bytes", lambda: 8e9)
    monkeypatch.setattr(_decon_gpu, "_cuda_free_bytes", lambda: 5e9)
    assert _decon_gpu._device_free_bytes("mps") == 8e9
    assert _decon_gpu._device_free_bytes("cuda") == 5e9
    assert _decon_gpu._device_free_bytes(None) == 20e9
    assert _decon_gpu.budget_bytes("cuda") == 5e9 * _decon_gpu.MEMORY_FRACTION


def test_mps_is_unified_memory_so_the_smaller_of_the_two_readings_wins(monkeypatch):
    monkeypatch.setattr(_decon_gpu, "_mps_recommended_bytes", lambda: 12e9)
    monkeypatch.setattr(_decon_gpu, "_free_bytes", lambda: 3e9)
    assert _decon_gpu._device_free_bytes("mps") == 3e9


def test_without_the_mps_query_the_system_ram_rule_stands_in(monkeypatch):
    monkeypatch.setattr(_decon_gpu, "_mps_recommended_bytes", lambda: None)
    monkeypatch.setattr(_decon_gpu, "_free_bytes", lambda: 6e9)
    assert _decon_gpu._device_free_bytes("mps") == 6e9


def test_the_tile_plan_is_petakits_halve_then_border_rule():
    """(48, 256, 256) against a budget of 24 planes: petakit halves 48 -> 24, borders the
    tile by min(support // 2 + 2, 24 // 3) = 8 and keeps 8 planes per tile, 6 tiles."""
    shape = (48, 256, 256)
    assert _decon_gpu.tile_plan(shape, budget=_budget_for(24), support_z=95) == (8, 8)
    assert _decon_gpu.tile_count(48, (8, 8)) == 6
    assert _decon_gpu.tile_plan(shape, budget=_budget_for(12), support_z=95) == (4, 4)
    assert _decon_gpu.tile_count(48, (4, 4)) == 12
    # a narrow PSF asks for a narrower border, never under three planes
    assert _decon_gpu.tile_plan(shape, budget=_budget_for(24), support_z=1) == (18, 3)
    assert _decon_gpu.tile_plan(shape, budget=_budget_for(48), support_z=95) is None
    assert _decon_gpu.tile_count(48, None) == 1


def test_a_volume_whose_smallest_tile_is_still_over_budget_goes_to_petakit(monkeypatch):
    """A 4-plane tile is petakit's floor; under it the volume is not this device's to run."""
    pytest.importorskip("torch")
    _force_budget(monkeypatch, _budget_for(2))
    with pytest.raises(MemoryError, match="4-plane tile"):
        _decon_gpu.tile_plan((64, 256, 256), budget=_budget_for(2), support_z=9)
    assert _decon_gpu.select_device((64, 256, 256)) is None
    assert "z-tiler" in _decon_gpu.describe((64, 256, 256))


def test_a_volume_over_budget_is_no_longer_declined_but_tiled(monkeypatch):
    _device_or_skip()
    _force_budget(monkeypatch, _budget_for(24))
    assert _decon_gpu.select_device((48, 256, 256)) is not None
    assert not _decon_gpu.fits_in_memory((48, 256, 256), _decon_gpu._torch_device())


def test_describe_names_the_device_and_the_tile_count(monkeypatch):
    device = _device_or_skip()
    psf = make_psf(DEFAULT_OPTICS)
    _force_budget(monkeypatch, _budget_for(24))
    line = _decon_gpu.describe((48, 256, 256), psf_shape=psf.shape, psf=psf)
    assert f"torch/{device}, 6 z tiles" in line
    assert "approximate" in line, "a z-tiled RL is not the whole solve and the line must say so"
    whole = _decon_gpu.describe((24, 256, 256), psf_shape=psf.shape, psf=psf)
    assert "z tiles" not in whole


def test_a_tiled_solve_on_the_device_is_petakits_tiled_solve():
    """Same plan, same tiles, same numbers: the backend property holds for the tiled path.

    petakit tiles with budget ``avail * 0.7``, so its ``avail_memory_gb`` is our budget / 0.7."""
    device = _device_or_skip()
    psf = make_psf(DEFAULT_OPTICS)
    volume = _phantom((48, 256, 256), seed=2)
    budget = _budget_for(24)
    import unittest.mock as mock

    with mock.patch.object(_decon_gpu, "_device_free_bytes",
                           lambda d: budget / _decon_gpu.MEMORY_FRACTION):
        assert _decon_gpu.tile_plan(volume.shape, budget=_decon_gpu.budget_bytes(device),
                                    support_z=_decon_gpu.psf_support_z(psf),
                                    psf_shape=psf.shape) == (8, 8)
        ours = _decon_gpu.rl(volume, psf, ITERATIONS, device)
    theirs = petakit.deconvolve(volume, psf, method=METHOD, iterations=ITERATIONS, gpu=False,
                                avail_memory_gb=budget / 0.7 / 1e9)
    peak = float(np.abs(theirs).max())
    assert float(np.abs(ours - theirs).max()) / peak <= TOLERANCE
    _assert_quantised_agreement(cast_like(ours, np.dtype(np.uint16)),
                                cast_like(theirs, np.dtype(np.uint16)))


def test_tiling_is_not_the_whole_solve_and_nobody_may_claim_it_is(monkeypatch):
    """Measured: with petakit's rule (border capped at a third of the tile, here 8 planes
    against a 95-plane PSF support) the tiled result differs from the whole solve by up to
    25% of peak on the puncta planes. That is petakit's CPU behaviour too; the log line
    names the tiling so a user can tell. This test keeps the number honest."""
    device = _device_or_skip()
    psf = make_psf(DEFAULT_OPTICS)
    volume = _phantom((48, 256, 256), seed=2)
    _force_budget(monkeypatch, _budget_for(48))
    whole = _decon_gpu.rl(volume, psf, ITERATIONS, device)
    _force_budget(monkeypatch, _budget_for(24))
    tiled = _decon_gpu.rl(volume, psf, ITERATIONS, device)
    gap = float(np.abs(tiled - whole).max()) / float(np.abs(whole).max())
    assert gap > TOLERANCE, (
        f"tiled and whole agree to {gap:.2e} of peak; if tiling became exact the "
        "'approximate' wording in describe() is a lie and should go")


def test_selection_degrades_silently_when_torch_is_absent(monkeypatch):
    """The normal case on a clean install: torch is not a squidxplorer dependency."""
    monkeypatch.setattr(_decon_gpu, "_torch_device", lambda: None)
    assert _decon_gpu.select_device((1, 256, 256)) is None
    assert "no torch GPU device" in _decon_gpu.describe((1, 256, 256))


def test_the_device_choice_is_logged_once_and_not_once_per_plane(caplog, monkeypatch):
    from squidxplorer import _decon

    monkeypatch.setattr(_decon_gpu, "_LOG_ONCE", set())
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    plane = _phantom((1, 64, 64), seed=1)[0].astype(np.uint16)
    with caplog.at_level("INFO"):
        _plane_solve(plane, DEFAULT_OPTICS, 1)
        _plane_solve(plane, DEFAULT_OPTICS, 1)
    lines = [r.message for r in caplog.records if "decon backend" in r.message]
    assert len(lines) == 1, f"expected one backend line, got {lines}"
    assert "CPU" in lines[0]


def test_a_cupy_cuda_machine_is_left_entirely_alone(monkeypatch):
    """petakit's own CuPy path already covers NVIDIA."""
    monkeypatch.setattr(_decon_gpu, "_cupy_cuda_present", lambda: True)
    monkeypatch.setattr(_decon_gpu, "_torch_device", lambda: "cuda")
    assert _decon_gpu.select_device((1, 256, 256)) is None
    assert "petakit/CuPy" in _decon_gpu.describe((1, 256, 256))
