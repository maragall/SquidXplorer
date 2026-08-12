"""The GPU decon backend must be a BACKEND: same algorithm, same numbers, different device.

The whole risk of ``squidxplorer/_decon_gpu.py`` is that it is a second transcription of
``petakit.engine._rl_core`` and could silently drift from it. So the load-bearing test here is
not "the GPU path returns an array", it is "the GPU path returns petakit's array". Everything
else in this file is about the DECISION (which device, and when not to), which must be
deterministic and must degrade to CPU rather than raise.

WHY THE TOLERANCE IS WHAT IT IS. Two assertions are made, in increasing strength:

1. ``max |gpu - cpu| / max |cpu|  <=  1e-4``. float32 eps is 1.2e-7; three RL iterations put
   six full-volume transforms and a global float32 dot-product reduction (the Biggs-Andrews
   ``lambda``) between input and output, and neither the FFT butterfly order nor the reduction
   order is the same on a GPU as on ducc's CPU kernel. Measured on an Apple M4 across eight
   plane sizes from 2048^2 to 4024^2 the observed figure was 1.1e-06 to 4.3e-06, so 1e-4 is
   roughly two orders of margin over what the hardware actually does, loose enough not to be
   flaky, tight enough that a real algorithmic drift (a dropped clamp, a wrong conjugate, an
   off-by-one in the circshift) cannot hide under it. Those mistakes move pixels by percent, not
   by parts per million.

2. **The uint16 planes an operator run actually writes differ by at most ONE count, on a
   handful of pixels.** This is the assertion that matters to a user, and it is the one the
   measurement supports, not the stronger "identical" claim it is tempting to write. Run here:
   ``6 of 65536 pixels (0.009%) differ, all by exactly 1``. Those are pixels whose float value
   sat within a part-per-million of a ``.5`` boundary, so ``cast_like``'s ``rint`` rounded them
   opposite ways; that is the arithmetic of quantisation, not drift in the deconvolution. The
   test therefore pins BOTH the magnitude (<= 1 count, i.e. the quantisation step itself) and
   the population (< 0.1% of pixels). Real algorithmic drift moves whole structures by percent
   and would blow through both bounds at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer import _decon_gpu
from squidxplorer._decon import DEFAULT_OPTICS, METHOD, make_psf, make_psf_2d
from squidxplorer.projection import cast_like

petakit = pytest.importorskip("petakit")

#: Small and 7-smooth (256 = 2^8), so the test exercises the accepted branch and stays fast.
SIZE = 256
ITERATIONS = 3

#: Relative-to-peak agreement between the two backends. Defended in the module docstring.
TOLERANCE = 1e-4

#: After the cast to the acquisition dtype: no pixel may move by more than the quantisation step
#: itself, and hardly any pixel may move at all. Measured 1 count on 0.009% of pixels.
MAX_COUNTS = 1
MAX_DIFFERING_FRACTION = 1e-3


def _phantom(shape, seed=0):
    """Puncta on a background, structure with real dynamic range, which is what RL amplifies.

    Uniform noise would let a broken backend pass by returning something bland; a sparse bright
    field is where a wrong kernel or a lost clamp shows up as a visible halo.
    """
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


# --- the equivalence, which is the point of the file ------------------------------------------


@pytest.mark.parametrize("shape", [(1, SIZE, SIZE), (5, SIZE, SIZE)])
def test_gpu_backend_reproduces_petakit_cpu_within_tolerance(shape):
    """2-D plane and 3-D stack alike: the device result IS petakit's result."""
    device = _device_or_skip()
    psf = make_psf_2d(DEFAULT_OPTICS) if shape[0] == 1 else make_psf(DEFAULT_OPTICS)
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
    """The assertion a user can feel: the same pixels on disk, to within one count."""
    device = _device_or_skip()
    psf = make_psf_2d(DEFAULT_OPTICS)
    volume = _phantom((1, SIZE, SIZE), seed=3)

    cpu = cast_like(petakit.deconvolve(volume, psf, method=METHOD,
                                        iterations=ITERATIONS, gpu=False)[0], np.dtype(np.uint16))
    gpu = cast_like(_decon_gpu.rl(volume, psf, ITERATIONS, device)[0], np.dtype(np.uint16))

    _assert_quantised_agreement(gpu, cpu)


def test_deconvolve_plane_goes_through_the_device_and_still_matches_the_cpu_path(monkeypatch):
    """The seam, not just the kernel: ``_run``'s fork must not change ``deconvolve_plane``."""
    _device_or_skip()
    from squidxplorer import _decon

    plane = _phantom((1, SIZE, SIZE), seed=7)[0].astype(np.uint16)

    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    on_cpu = _decon.deconvolve_plane(plane, DEFAULT_OPTICS, ITERATIONS)
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "auto")
    on_gpu = _decon.deconvolve_plane(plane, DEFAULT_OPTICS, ITERATIONS)

    assert on_gpu.dtype == plane.dtype
    _assert_quantised_agreement(on_gpu, on_cpu)


# --- the decision, which must be deterministic and must never raise ---------------------------


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
    """The guard is unchanged in spirit and moved in target, and both halves matter.

    Told nothing about the PSF it cannot know a pad is coming, so it still refuses 2084 (the
    conservative reading). Told the PSF shape it sees the transform will really run at 2160 and
    allows it. The refusal is not obsolete: MPS at a raw 2084 measured 0.72x against the CPU
    pool on an idle machine, i.e. a real loss, so the case the guard covers is real whenever
    padding is off or declined.
    """
    _device_or_skip()
    assert _decon_gpu.select_device((1, AWKWARD, AWKWARD)) is None
    assert _decon_gpu.select_device((1, AWKWARD, AWKWARD), psf_shape=(1, 19, 19)) is not None
    assert _decon_gpu.select_device((1, 2048, 2048)) is not None


def test_with_padding_off_the_guard_refuses_the_awkward_width_again(monkeypatch):
    """PAD=off puts the 0.72x loss back on the table, so the guard must fire again."""
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
#
# A padding bug does not show up in the middle of the frame. It shows up as a rim, so every
# assertion below either targets the border explicitly or uses a phantom whose content runs
# hard into the edge. The interior is the easy case and proves nothing on its own.


#: Non-7-smooth, and the width of this instrument's actual camera frames.
AWKWARD = 2084


def _rim_phantom(shape, seed=11):
    """Puncta PLUS a bright rim clamped to the outermost pixels.

    A phantom that fades to background at the edge cannot detect a boundary bug, because the
    wrong answer and the right answer are both ~0 there. This one puts the brightest structure
    in the frame exactly where a bad pad would corrupt it.
    """
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
    # scipy would answer 2112 = 2^6 x 3 x 11 here; 11 is exactly what we must not accept.
    assert not _decon_gpu.is_smooth(2112)
    assert _decon_gpu.fast_len(2102) == 2160


def test_pad_plan_leaves_already_fast_axes_alone_including_z():
    """A 10-plane stack is already smooth, so a 3-D run must not inflate the acquired depth."""
    widths = _decon_gpu.pad_plan((10, AWKWARD, AWKWARD), (10, 19, 19))
    assert widths[0] == 0, "z was already 7-smooth and must not be padded"
    assert widths[1] == widths[2] > 0
    assert AWKWARD + 2 * widths[1] == 2160
    assert _decon_gpu.pad_plan((1, 2048, 2048), (1, 19, 19)) == (0, 0, 0)


def test_pad_plan_gives_at_least_the_psf_extent_which_is_what_makes_the_wrap_exact():
    """Measured: a pad narrower than the PSF leaks. W=8 with a 19-px PSF cost 3.0e-05 of peak,
    all of it in the border; W>=16 reached the 7e-07 floor. The plan must never choose W=8."""
    widths = _decon_gpu.pad_plan((1, AWKWARD, AWKWARD), (1, 19, 19))
    assert widths[1] >= 19, f"pad {widths[1]} is under the PSF extent, the wrap will leak"


def test_pad_plan_declines_rather_than_growing_an_axis_without_bound(monkeypatch):
    monkeypatch.setattr(_decon_gpu, "MAX_PAD_GROWTH", 1.0001)
    assert _decon_gpu.pad_plan((1, AWKWARD, AWKWARD), (1, 19, 19)) == (0, 0, 0)


def test_wrap_pad_reproduces_the_opposite_edge_and_not_zeros_or_a_replicated_edge():
    """The mode is the correctness argument, so pin the mode itself.

    Measured on the real plane: zero-pad moved the border by 4.4e-02 of peak and edge-replication
    by 1.9e-02, against a 1e-04 tolerance. Only wrap matches the circular convolution petakit
    already performs.
    """
    a = np.arange(1, 7, dtype=np.float32).reshape(1, 2, 3)
    out = _decon_gpu._wrap_pad(a, (0, 0, 2))
    assert out.shape == (1, 2, 7)
    np.testing.assert_array_equal(out[0, 0], [2, 3, 1, 2, 3, 1, 2])
    assert _decon_gpu._wrap_pad(a, (0, 0, 0)) is a


def test_padding_leaves_the_border_pixels_alone_on_an_awkward_width(monkeypatch):
    """THE test this whole change lives or dies by: pad, run, crop, and check the RIM.

    Reference is petakit's own unpadded CPU output at the awkward width, i.e. exactly what the
    user gets today. The padded GPU run must reproduce it everywhere, and the border band is
    checked separately and more tightly than the frame as a whole, because that is the only
    place a padding bug can hide.
    """
    device = _device_or_skip()
    volume = _rim_phantom((1, AWKWARD, AWKWARD))
    psf = make_psf_2d(DEFAULT_OPTICS)

    assert any(_decon_gpu.pad_plan(volume.shape, psf.shape)), "this width must actually pad"
    reference = petakit.deconvolve(volume, psf, method=METHOD, iterations=ITERATIONS, gpu=False)
    got = _decon_gpu.rl(volume, psf, ITERATIONS, device)

    assert got.shape == volume.shape, "the crop must restore the exact input extent"
    peak = float(np.abs(reference).max())
    delta = np.abs(got - reference)
    assert float(delta.max()) / peak <= TOLERANCE
    assert _border_max(delta, 64) / peak <= TOLERANCE


def test_end_to_end_at_the_real_camera_width_cpu_and_gpu_write_the_same_plane(monkeypatch):
    """The user-facing assertion, at the width this instrument actually produces.

    uint16 in, uint16 out, through ``deconvolve_plane``, with the GPU padding 2084 -> 2160 under
    the covers and the CPU arm not padding at all. Held to the same one-count bar as every other
    backend comparison in this file.
    """
    _device_or_skip()
    from squidxplorer import _decon

    plane = _rim_phantom((1, AWKWARD, AWKWARD), seed=13)[0].astype(np.uint16)
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    monkeypatch.delenv(_decon_gpu.PAD_CPU_ENV_VAR, raising=False)
    on_cpu = _decon.deconvolve_plane(plane, DEFAULT_OPTICS, ITERATIONS)
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "auto")
    on_gpu = _decon.deconvolve_plane(plane, DEFAULT_OPTICS, ITERATIONS)

    assert on_gpu.shape == plane.shape and on_gpu.dtype == plane.dtype
    _assert_quantised_agreement(on_gpu, on_cpu)
    # and the rim specifically, which is where a padding bug would live
    rim = np.abs(on_gpu.astype(np.int32) - on_cpu.astype(np.int32))
    assert _border_max(rim, 64) <= MAX_COUNTS


def test_a_3d_stack_pads_yx_but_never_the_acquired_depth(monkeypatch):
    """decon3d must not inflate z: the stack depth is the acquisition's, not ours to round up."""
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
    """Mutation check on the test above: break the pad and the border assertion must FIRE.

    Substituting zeros for the wrap is the exact bug the border test exists to catch. If this
    passes, the border test is measuring nothing.
    """
    device = _device_or_skip()
    volume = _rim_phantom((1, AWKWARD, AWKWARD))
    psf = make_psf_2d(DEFAULT_OPTICS)
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
    """The subtle failure: pad correctly but reduce lambda over the PADDED array, as petakit does.

    Unlike a pad-mode bug this one does NOT show up at the border; it shifts every pixel a
    little, because lambda is a single global scalar. The contrast is measured against a run
    that genuinely has a global lambda: petakit's own RL, handed the same wrap-padded volume.
    Our region-restricted kernel must land inside tolerance where that one does not.
    """
    device = _device_or_skip()
    volume = _rim_phantom((1, AWKWARD, AWKWARD), seed=5)
    psf = make_psf_2d(DEFAULT_OPTICS)
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
    """CPU stays bit-stable by default: its lambda cannot be corrected from this layer."""
    monkeypatch.delenv(_decon_gpu.PAD_CPU_ENV_VAR, raising=False)
    assert not _decon_gpu.cpu_padding_enabled()
    monkeypatch.setenv(_decon_gpu.PAD_CPU_ENV_VAR, "1")
    assert _decon_gpu.cpu_padding_enabled()


#: Opt-in CPU padding is held to a LOOSER bar than the GPU path, and the gap is the point.
#: petakit reduces the Biggs-Andrews lambda over the whole padded array and this layer cannot
#: reach inside to restrict it, so a padded CPU run is not the same numbers. Measured: 1 count
#: on 0.30%-0.41% of pixels on the real 2084^2 plane, and 2 counts on a 514^2 frame, where the
#: pad is a larger fraction of the array (9% of the pixels against 3.6%) and so perturbs lambda
#: more. Three is the assertion, with the measurements above as its justification. This is
#: precisely why the switch is off by default while GPU padding is on.
CPU_PAD_MAX_COUNTS = 3


def test_opting_the_cpu_path_into_padding_keeps_the_extent_and_stays_below_shot_noise(monkeypatch):
    """Opt-in CPU padding must return the right SHAPE and stay a few counts from unpadded.

    It does NOT meet the one-count bar the GPU path meets; see :data:`CPU_PAD_MAX_COUNTS`.
    """
    from squidxplorer import _decon

    # 514 = 2 x 257, NOT 7-smooth, so the pad plan is non-trivial. 512 would be 2^9 and the
    # test would silently assert nothing.
    assert not _decon_gpu.is_smooth(514)
    plane = _rim_phantom((1, 514, 514), seed=9)[0].astype(np.uint16)
    assert any(_decon_gpu.pad_plan((1, 514, 514), (1, 19, 19)))
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    monkeypatch.delenv(_decon_gpu.PAD_CPU_ENV_VAR, raising=False)
    plain = _decon.deconvolve_plane(plane, DEFAULT_OPTICS, ITERATIONS)
    monkeypatch.setenv(_decon_gpu.PAD_CPU_ENV_VAR, "1")
    padded = _decon.deconvolve_plane(plane, DEFAULT_OPTICS, ITERATIONS)

    assert padded.shape == plane.shape
    delta = np.abs(padded.astype(np.int32) - plain.astype(np.int32))
    worst = int(delta.max())
    assert worst <= CPU_PAD_MAX_COUNTS, f"padded CPU moved a pixel by {worst} counts"
    assert worst >= 1, (
        "padded and unpadded CPU came out identical, so either the pad did not happen or "
        "petakit's lambda is not global after all; both would make this test vacuous"
    )


def test_forcing_a_device_this_machine_does_not_have_falls_back_instead_of_raising(monkeypatch):
    """``SQUIDXPLORER_DECON_DEVICE=mps`` on a CUDA box (or vice versa) must degrade, not explode
    from inside torch halfway through a plate run."""
    monkeypatch.setattr(_decon_gpu, "_torch_device", lambda: "cuda")
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "mps")
    assert _decon_gpu.select_device((1, 256, 256)) is None


def test_an_unknown_env_value_fails_loud_rather_than_guessing(monkeypatch):
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "metal")
    with pytest.raises(ValueError, match="is not a device"):
        _decon_gpu.select_device((1, 256, 256))


def test_a_volume_too_large_for_a_tiler_less_backend_is_declined(monkeypatch):
    """petakit answers an oversized volume by tiling along z; this backend must answer by
    declining, never by allocating something that does not fit.

    NEEDS TORCH, which is not a squidxplorer dependency -- the test below this one calls its absence
    "the normal case on a clean install". With no GPU device to size against, `describe` answers
    "CPU (no torch GPU device)" and never reaches the size reasoning asserted here, so on CI and in
    any clean venv this failure was about the environment and not about the backend. It was one of
    exactly two failures in the first clean-venv run of this suite (2026-08-05).
    """
    pytest.importorskip("torch")
    monkeypatch.setattr(_decon_gpu, "_free_bytes", lambda: 1e9)
    assert not _decon_gpu.fits_in_memory((64, 4096, 4096))
    assert _decon_gpu.select_device((64, 4096, 4096)) is None
    assert "z-tiler" in _decon_gpu.describe((64, 4096, 4096))


def test_selection_degrades_silently_when_torch_is_absent(monkeypatch):
    """The normal case on a clean install: torch is not a squidxplorer dependency at all."""
    monkeypatch.setattr(_decon_gpu, "_torch_device", lambda: None)
    assert _decon_gpu.select_device((1, 256, 256)) is None
    assert "no torch GPU device" in _decon_gpu.describe((1, 256, 256))


def test_the_device_choice_is_logged_once_and_not_once_per_plane(caplog, monkeypatch):
    """A decon that silently declined the GPU must be visible in the log panel, not a mystery."""
    from squidxplorer import _decon

    monkeypatch.setattr(_decon_gpu, "_LOG_ONCE", set())
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    plane = _phantom((1, 64, 64), seed=1)[0].astype(np.uint16)
    with caplog.at_level("INFO"):
        _decon.deconvolve_plane(plane, DEFAULT_OPTICS, 1)
        _decon.deconvolve_plane(plane, DEFAULT_OPTICS, 1)
    lines = [r.message for r in caplog.records if "decon backend" in r.message]
    assert len(lines) == 1, f"expected one backend line, got {lines}"
    assert "CPU" in lines[0]


def test_a_cupy_cuda_machine_is_left_entirely_alone(monkeypatch):
    """petakit's own CuPy path already covers NVIDIA; this module must not intercept it."""
    monkeypatch.setattr(_decon_gpu, "_cupy_cuda_present", lambda: True)
    monkeypatch.setattr(_decon_gpu, "_torch_device", lambda: "cuda")
    assert _decon_gpu.select_device((1, 256, 256)) is None
    assert "petakit/CuPy" in _decon_gpu.describe((1, 256, 256))
