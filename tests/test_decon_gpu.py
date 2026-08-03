"""The GPU decon backend must be a BACKEND: same algorithm, same numbers, different device.

The whole risk of ``squidmip/_decon_gpu.py`` is that it is a second transcription of
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
   sat within a part-per-million of a ``.5`` boundary, so ``_cast_like``'s ``rint`` rounded them
   opposite ways; that is the arithmetic of quantisation, not drift in the deconvolution. The
   test therefore pins BOTH the magnitude (<= 1 count, i.e. the quantisation step itself) and
   the population (< 0.1% of pixels). Real algorithmic drift moves whole structures by percent
   and would blow through both bounds at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip import _decon_gpu
from squidmip._decon import DEFAULT_OPTICS, METHOD, _cast_like, make_psf, make_psf_2d

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

    cpu = _cast_like(petakit.deconvolve(volume, psf, method=METHOD,
                                        iterations=ITERATIONS, gpu=False)[0], np.dtype(np.uint16))
    gpu = _cast_like(_decon_gpu.rl(volume, psf, ITERATIONS, device)[0], np.dtype(np.uint16))

    _assert_quantised_agreement(gpu, cpu)


def test_deconvolve_plane_goes_through_the_device_and_still_matches_the_cpu_path(monkeypatch):
    """The seam, not just the kernel: ``_run``'s fork must not change ``deconvolve_plane``."""
    _device_or_skip()
    from squidmip import _decon

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
    (2084, False),      # 2^2 x 521; measured 0.55x on MPS, i.e. a REGRESSION
    (3036, False),      # 2^2 x 3 x 11 x 23; measured 0.81x
    (4024, False),      # 2^3 x 503
    (1, True),
])
def test_is_smooth_matches_the_measured_fast_and_slow_transform_lengths(n, smooth):
    assert _decon_gpu.is_smooth(n) is smooth


def test_a_non_smooth_plane_width_declines_the_gpu():
    """The guard that stops this being a 1.8x SLOWDOWN on a 2084-wide sensor."""
    _device_or_skip()
    assert _decon_gpu.select_device((1, 2084, 2084)) is None
    assert "non-7-smooth" in _decon_gpu.describe((1, 2084, 2084))


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


def test_forcing_a_device_this_machine_does_not_have_falls_back_instead_of_raising(monkeypatch):
    """``SQUIDMIP_DECON_DEVICE=mps`` on a CUDA box (or vice versa) must degrade, not explode
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
    declining, never by allocating something that does not fit."""
    monkeypatch.setattr(_decon_gpu, "_free_bytes", lambda: 1e9)
    assert not _decon_gpu.fits_in_memory((64, 4096, 4096))
    assert _decon_gpu.select_device((64, 4096, 4096)) is None
    assert "z-tiler" in _decon_gpu.describe((64, 4096, 4096))


def test_selection_degrades_silently_when_torch_is_absent(monkeypatch):
    """The normal case on a clean install: torch is not a squidmip dependency at all."""
    monkeypatch.setattr(_decon_gpu, "_torch_device", lambda: None)
    assert _decon_gpu.select_device((1, 256, 256)) is None
    assert "no torch GPU device" in _decon_gpu.describe((1, 256, 256))


def test_the_device_choice_is_logged_once_and_not_once_per_plane(caplog, monkeypatch):
    """A decon that silently declined the GPU must be visible in the log panel, not a mystery."""
    from squidmip import _decon

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
