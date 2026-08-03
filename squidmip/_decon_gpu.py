"""Apple-Silicon (Metal/MPS) and torch-CUDA BACKEND for the Richardson-Lucy update.

WHY THIS EXISTS. ``_decon.py`` hands its volumes to ``petakit.deconvolve``, whose GPU path is
**CuPy only** (``petakit/engine.py:33 _get_array_module`` -> ``import cupy``). CuPy is CUDA, so on
an Apple Silicon Mac that branch can never be taken and every deconvolution runs on
``scipy.fft``. Measured on this repo, one 2084x2084 plane at ``iterations=3``: 0.785 s total, of
which 0.683 s (87%) is ``scipy.fft._duccfft.pyduccfft.c2c``, thirteen complex FFTs. The cost is
the FFT and nothing else, which is exactly the thing a GPU is for.

THIS IS A BACKEND, NOT AN ALGORITHM. Every line below is a transcription of
``petakit.engine._rl_core``, the same Biggs-Andrews accelerated Richardson-Lucy, the same
``psf2otf`` (crop, zero-pad, circshift), the same clamps, the same ``lambda`` update, the same
float32. IMA-247 forbids silently substituting a *different* method; it does not forbid running
the *same* method on a different device, which is precisely the distinction
``deconvolve_plane``'s ``gpu=`` docstring already draws. ``tests/test_decon_gpu.py`` pins the
equivalence against petakit's own CPU output.

ONE DELIBERATE DEPARTURE: rfftn INSTEAD OF fftn
-----------------------------------------------
petakit takes the full complex ``fftn`` of a real array and then throws the imaginary half away
(``real(ifftn(...))``). The half-spectrum ``rfftn``/``irfftn`` pair computes the identical real
result from half the coefficients. It is not an approximation, it is the Hermitian symmetry a
real signal already has. Measured on this machine it is where most of the remaining win comes
from: 3000x3000 MPS ``fftn`` 14.1 ms vs ``rfftn`` 7.4 ms. Agreement with petakit's CPU output is
reported in the table below.

THE MEASURED WIN
----------------
The honest baseline is NOT one CPU core. ``project_plate`` fans out over ``(region, fov)`` pairs
on a ``ThreadPoolExecutor`` sized to the CPU count (``_engine.py:319``), so the status quo on a
10-core M4 is already ten planes deconvolving in parallel on ten cores. Measured through
``_decon.deconvolve_plane`` / ``deconvolve_stack`` at ``iterations=3`` on an Apple M4 / 16 GB,
medians of six A/B rounds with the run order alternated each round:

    workload                CPU x10     MPS      speedup
    --------------------    -------    ------    -------
    2-D 2048^2  x12 planes   3.04 s    1.22 s      2.48x
    2-D 3000^2  x12 planes   5.02 s    2.30 s      2.18x
    3-D 10x1024^2 x4 stacks  1.68 s    0.68 s      2.47x
    2-D 2084^2  x12 planes   3.65 s    3.62 s      1.01x   <- CONTROL, guard declines (below)

**About 2.2x to 2.5x, not 10x.** An M4's GPU is not a discrete NVIDIA card, and the ten CPU cores
it is racing were already all busy. Anyone expecting a CUDA-shaped number should read the CPU
column first. The last row is the control that makes the other three trustworthy: the guard sends
2084^2 to the CPU on both arms, so identical code runs twice and the row must read ~1.00x. It
reads 1.01x, which is this harness's noise floor. (An earlier, uncontrolled version of this
benchmark reported figures between 1.8x and 6.2x for the same work; almost all of that spread was
run-order and warm-up bias. The alternated numbers above are the ones to quote.)

AND THE MEASURED LOSS, WHICH IS WHY THERE IS A GUARD
----------------------------------------------------
Metal's FFT is fast on lengths that factor into small primes and falls onto a Bluestein path
otherwise. Single-shot calibration sweep, twelve planes each, GPU against the same 10-thread CPU
pool (these rows are un-alternated and so are good enough only to sort winners from losers, which
is all they are used for):

    plane      largest prime    speedup    max |diff| / peak
    -------    -------------    -------    -----------------
    2048^2                 2      2.44x             1.10e-06
    2160^2                 5      2.06x             1.49e-06
    2560^2                 5      2.02x             1.30e-06
    2800^2                 7      1.56x             1.39e-06
    3000^2                 5      1.46x             1.43e-06
    2084^2               521      0.55x             2.73e-06   <- SLOWER than the CPU pool
    3036^2                23      0.81x             2.46e-06   <- SLOWER
    4024^2               503      1.55x             2.46e-06

So this module does **not** hand every volume to the GPU. :func:`select_device` requires every
transform length to be 7-smooth (its only prime factors are 2, 3, 5, 7), which admits all five
clean winners and rejects both losers. 4024 = 2^3 x 503 is rejected too and is the one win given
up; that row's "win" is against a CPU baseline that was itself thrashing (ten threads x 16 MP at
40 B/voxel does not fit in 16 GB), so declining it is the conservative call rather than a missed
opportunity. A rejected shape costs nothing: it runs exactly the code it ran before, as the
control row above demonstrates.

If your camera's frame width is not 7-smooth and you want the GPU anyway, force it::

    SQUIDMIP_DECON_DEVICE=mps

and if the GPU misbehaves, turn it off the same way::

    SQUIDMIP_DECON_DEVICE=cpu

NOT TOUCHED: CUDA MACHINES THAT ALREADY WORK. If CuPy is importable and sees a CUDA device,
petakit's existing GPU path is already running and :func:`select_device` returns ``None`` so that
nothing changes for the Windows/NVIDIA install. torch-CUDA is used only where CuPy is absent.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

#: Device override, same spelling convention as ``SQUIDMIP_CACHE_MB`` (``_budget.py:56``) and
#: ``SQUIDMIP_PLATE_CACHE`` (``_platecache.py:196``). Values:
#:
#: * ``auto`` / unset, pick a device if one is present AND the transform sizes are 7-smooth.
#: * ``cpu`` / ``off`` / ``0`` / ``none``, never use this backend; petakit's CPU path runs.
#: * ``mps`` / ``cuda``, force that device, skipping the smoothness guard. The escape hatch for
#:   a sensor whose frame width this module's guard rejects.
ENV_VAR = "SQUIDMIP_DECON_DEVICE"

#: Prime factors Metal's FFT handles on its fast path. A length whose factorisation contains
#: anything larger falls onto Bluestein and loses to ten CPU cores, measured, see the module
#: docstring's table.
SMOOTH_PRIMES = (2, 3, 5, 7)

#: Bytes of working set per voxel, petakit's own estimate (``engine.py:305 _estimate_peak_gb``):
#: two float32 volumes plus four complex64 spectra. This module's ``rfftn`` spectra are half that
#: size, so 40 is deliberately pessimistic, the point of the number is to decline a volume, not
#: to size an allocator.
BYTES_PER_VOXEL = 40

#: Fraction of free system memory a single GPU volume may claim. petakit answers an oversized
#: volume by TILING ALONG Z (``engine.py:345 _tile_z``); this backend has no tiler, so it must
#: answer the same question by declining and letting petakit's tiler do its job on the CPU. Half,
#: not all, because Apple Silicon memory is unified: the tensors below are competing with the same
#: pool the CPU worker threads are reading planes into.
MEMORY_FRACTION = 0.5

#: One GPU, one volume at a time. ``project_plate`` runs ten worker threads and each would
#: otherwise hold a full working set (image, J_2, J_3, J_4, Y, ratio, ReBlurred + spectra) in
#: unified memory at once, for no throughput gain: measured at 2048^2, four threads submitting
#: to MPS ran 1.28 s against 1.34 s for one thread. Serialising costs ~nothing and keeps the
#: footprint at one volume instead of ten. The lock is held only around device work, so the
#: other threads keep reading planes from disk meanwhile.
_device_lock = threading.Lock()

_LOG_ONCE: set[str] = set()
_log_once_lock = threading.Lock()


def is_smooth(n: int, primes=SMOOTH_PRIMES) -> bool:
    """True when *n* factors entirely into *primes*, i.e. the FFT fast path applies.

    ``is_smooth(2048)`` -> True (2^11). ``is_smooth(2084)`` -> False (2^2 x 521).
    """
    if n < 1:
        return False
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1


def _cupy_cuda_present() -> bool:
    """True when petakit's OWN GPU path is live, in which case this module stands down."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _free_bytes() -> float:
    """Free system memory in bytes. psutil is already a ``[gui]`` dependency; without it, assume
    a small machine rather than a large one, so the unknown case declines rather than thrashes."""
    try:
        import psutil

        return float(psutil.virtual_memory().available)
    except Exception:
        return 4e9


def fits_in_memory(shape) -> bool:
    """True when one volume of *shape* is small enough for this tiler-less backend to accept."""
    return int(np.prod(shape)) * BYTES_PER_VOXEL < _free_bytes() * MEMORY_FRACTION


def _torch_device() -> Optional[str]:
    """The best torch device present, or ``None``. Never raises, torch is an optional import.

    torch is NOT a declared dependency of squidmip (``pyproject.toml`` has no torch line); it
    arrives with the optional Cellpose extra. So its absence is the normal case and must be
    silent, exactly as ``_cellpose.py`` treats Cellpose itself.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return None
    return None


def select_device(shape, *, gpu: bool = True) -> Optional[str]:
    """Which device should run a volume of *shape*, or ``None`` for "petakit's CPU path".

    Parameters
    ----------
    shape:
        The volume's ``(Z, Y, X)``. Every axis is an FFT length, so every axis is checked
        against :func:`is_smooth`, including Z, because a 3-D run transforms the stack depth
        too and an 11-plane stack is no friendlier to Metal than an 11-pixel-wide image.
    gpu:
        The caller's existing ``gpu=`` flag (``deconvolve_plane``/``deconvolve_stack``).
        ``False`` means "CPU please" and is honoured verbatim.

    Returns
    -------
    str or None
        ``"mps"``, ``"cuda"``, or ``None``.
    """
    override = os.environ.get(ENV_VAR, "").strip().lower()
    if override in {"cpu", "off", "0", "none", "false"}:
        return None
    if override in {"mps", "cuda"}:
        # An explicit device skips the smoothness guard (that is what the escape hatch is FOR)
        # but not the memory guard: no override should be able to ask a tiler-less backend to
        # hold a volume that does not fit. Nor may it name a device this machine does not have:
        # SQUIDMIP_DECON_DEVICE=mps on a CUDA box must fall back, not raise from inside torch
        # halfway through a plate run.
        if _torch_device() != override or not fits_in_memory(shape):
            return None
        return override
    if override not in {"", "auto"}:
        raise ValueError(
            f"{ENV_VAR}={override!r} is not a device. Use one of: auto, cpu, mps, cuda."
        )

    if not gpu:
        return None
    if _cupy_cuda_present():
        return None                     # petakit already has this machine covered; change nothing.
    if not all(is_smooth(int(n)) for n in shape):
        return None                     # Bluestein; measured slower than ten CPU cores.
    if not fits_in_memory(shape):
        return None                     # petakit's z-tiler owns this one.
    return _torch_device()


def _otf_source(psf: np.ndarray, out_shape) -> np.ndarray:
    """petakit's ``crop_psf_to_image`` + ``psf2otf`` padding/circshift, before the transform.

    Kept on CPU numpy and transcribed line for line from ``petakit/engine.py:81-118`` so the
    kernel that reaches the device is the same array petakit would have transformed. The PSF is
    small (19x19 for this instrument's optics) so doing it host-side costs nothing.
    """
    p = np.asarray(psf, dtype=np.float32)
    p = p / p.sum()
    slices = []
    for have, want in zip(p.shape, out_shape):
        if have > want:
            start = (have - want) // 2
            slices.append(slice(start, start + want))
        else:
            slices.append(slice(None))
    p = p[tuple(slices)]
    pad = np.array(out_shape) - np.array(p.shape)
    padded = np.pad(p, [(0, int(x)) for x in pad])
    return np.roll(padded, -(np.array(p.shape) // 2), axis=(0, 1, 2))


def rl(volume: np.ndarray, psf: np.ndarray, iterations: int, device: str) -> np.ndarray:
    """Biggs-Andrews accelerated Richardson-Lucy on *device*. Returns float32 ``(Z, Y, X)``.

    A transcription of ``petakit.engine._rl_core``; see the module docstring for the one
    departure (``rfftn`` for ``fftn``) and for why that is an identity rather than an
    approximation.
    """
    import torch

    dims = (-3, -2, -1)
    image_np = np.maximum(np.asarray(volume, dtype=np.float32), 0)
    shape = image_np.shape
    dev = torch.device(device)

    with _device_lock:
        image = torch.from_numpy(image_np).to(dev)
        H = torch.fft.rfftn(torch.from_numpy(_otf_source(psf, shape)).to(dev), dim=dims)
        H_conj = torch.conj(H)

        J_2 = image.clone()
        J_3 = torch.zeros_like(image)
        J_4 = torch.zeros(image.numel(), dtype=torch.float32, device=dev)
        lam = 0.0
        eps = float(np.finfo(np.float32).eps)
        Y = None

        for k in range(1, iterations + 1):
            if k > 2 and Y is not None:
                diff = (J_2 - Y).ravel()
                num = float(torch.dot(diff, J_4))
                den = float(torch.dot(J_4, J_4))
                lam = max(min(num / (den + eps), 1.0), 0.0)
                J_4 = diff
            elif k == 2 and Y is not None:
                J_4 = (J_2 - Y).ravel()

            Y = torch.clamp(J_2 + lam * (J_2 - J_3), min=0)
            reblurred = torch.clamp(
                torch.fft.irfftn(H * torch.fft.rfftn(Y, dim=dims), s=shape, dim=dims), min=eps
            )
            ratio = image / reblurred
            J_3 = J_2.clone()
            J_2 = torch.clamp(
                Y * torch.fft.irfftn(H_conj * torch.fft.rfftn(ratio, dim=dims), s=shape, dim=dims),
                min=0,
            )

        out = J_2.to("cpu").numpy()

    return out.astype(np.float32, copy=False)


def describe(shape, *, gpu: bool = True) -> str:
    """A one-line, human-readable account of the device decision, for logs, not for control flow."""
    device = select_device(shape, gpu=gpu)
    if device is not None:
        return f"decon backend: torch/{device} for shape {tuple(shape)}"
    if not gpu:
        return "decon backend: CPU (caller passed gpu=False)"
    override = os.environ.get(ENV_VAR, "").strip().lower()
    if override in {"cpu", "off", "0", "none", "false"}:
        return f"decon backend: CPU ({ENV_VAR}={override})"
    if _cupy_cuda_present():
        return "decon backend: petakit/CuPy CUDA (unchanged)"
    if _torch_device() is None:
        return "decon backend: CPU (no torch GPU device)"
    if not fits_in_memory(shape):
        need = int(np.prod(shape)) * BYTES_PER_VOXEL / 1e9
        return (f"decon backend: CPU (a {tuple(shape)} volume wants ~{need:.1f} GB; petakit's "
                "z-tiler handles it, this backend does not tile)")
    bad = [int(n) for n in shape if not is_smooth(int(n))]
    return (
        f"decon backend: CPU (shape {tuple(shape)} has non-7-smooth length(s) {bad}; "
        f"Metal's FFT is slower than the CPU pool there, override with {ENV_VAR}=mps)"
    )


def log_choice(shape, *, gpu: bool = True) -> None:
    """Say which device a run picked, ONCE per distinct answer.

    ``_cellpose.py`` states the rule this follows: ask for the GPU, then log what was actually
    chosen, "so a demo that is silently on CPU is visible in the log panel rather than a
    mystery-slow run". A deconvolution that quietly declined the GPU because the camera's frame
    width is 2084 is exactly that failure, and the operator has no other way to find out.

    Once per distinct message, not once per plane: ``project_plate`` calls the operator thousands
    of times per run and all but the first would be the same sentence.
    """
    message = describe(shape, gpu=gpu)
    with _log_once_lock:
        if message in _LOG_ONCE:
            return
        _LOG_ONCE.add(message)
    from squidmip._logpane import get_logger

    get_logger("decon").info(message)
