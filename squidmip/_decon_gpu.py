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

PADDING IS WHAT MAKES THIS WORTH ANYTHING ON THIS INSTRUMENT
------------------------------------------------------------
An FFT is fast when its length factors into small primes and slow (Bluestein) otherwise. This
instrument's camera produces **2084 x 2084** frames, and 2084 = 2^2 x 521 with 521 prime. That is
the slow case, on Metal badly and on ducc noticeably, and it is not an edge case: it is every
plane this scope has ever written.

Nothing in the stack was padding for this. ``petakit`` transforms at exactly the image extent
(``engine.py:139 out_shape = image.shape``) and its ``psf2otf`` pads the **PSF up to the image**
(``engine.py:108-115``), never the image beyond the PSF. So the convolution is fully circular at
2084 and the transform length is 2084 whether you like it or not.

:func:`pad_plan` changes that: each axis whose length is not 7-smooth is wrap-padded out to one
that is, and cropped back afterwards. For this geometry that is 2084 -> 2160, **+7.4% area**.
Measured through the shipped API on a real plane from the user's own acquisition, four planes,
``iterations=3``, four threads, best of nine interleaved repeats on an idle Apple M4 / 16 GB:

    cell                                best     speedup
    ------------------------------      ------   -------
    CPU 2084 unpadded (status quo)      0.90 s     1.00x
    CPU 2084 -> 2160 wrap-pad           0.61 s     1.48x   <- padding alone, no GPU involved
    MPS 2084 -> 2160 wrap-pad           0.26 s     3.41x   <- what this module now does
    MPS 2084 unpadded                   1.26 s     0.72x   <- the GPU LOSES without padding

    control, 2048^2 (already 7-smooth, so pad_plan is a no-op there)
    CPU 2048                            0.51 s     1.00x
    MPS 2048                            0.20 s     2.53x

Read the last two rows first. **The GPU is slower than the CPU pool at this camera's native
width**, 0.72x, and only padding turns that into 3.41x. The 2048 control reproduces the 2.48x
this module measured before padding existed (2.53x here), which is what says the harness is
measuring the same thing it measured last time.

A warning about the middle rows, learned the hard way: on a LOADED machine the CPU baseline
inflates several-fold (0.90 s idle became 4.63 s under concurrent load) and every GPU ratio
inflates with it. An intermediate version of this file claimed, on exactly that basis, that
"MPS unpadded at 2084 is 2.18x and the smoothness guard was refusing a real win". It was not;
the machine was busy. Quote the idle numbers.

THE GUARD SURVIVES, APPLIED TO THE PADDED LENGTH
------------------------------------------------
Because a non-smooth transform really is a loss (0.72x above), :func:`select_device` still
refuses one. What changed is *which* length it tests: the length **after** padding, via
:func:`effective_shape`. In normal operation that is 7-smooth by construction and the guard
never fires; it fires only when padding is switched off or declined by :data:`MAX_PAD_GROWTH`,
which are exactly the cases where the old loss is back on the table.

Switches, both directions::

    SQUIDMIP_DECON_DEVICE=cpu     # no GPU at all
    SQUIDMIP_DECON_DEVICE=mps     # force the GPU, past the guard
    SQUIDMIP_DECON_PAD=off        # no transform padding, raw lengths again
    SQUIDMIP_DECON_PAD_CPU=1      # opt the petakit CPU path into padding too (see below)

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

#: Padding off-switch, for bisecting a suspected padding bug without losing the GPU entirely.
#: ``SQUIDMIP_DECON_PAD=off`` runs every transform at the raw image length again.
PAD_ENV_VAR = "SQUIDMIP_DECON_PAD"

#: Opt-in: also pad the petakit CPU path. OFF by default, and the reason is numerical, not
#: performance. Padding is worth 1.24x to 1.55x on the CPU (measured, real 2084^2 plane), but
#: petakit reduces the Biggs-Andrews lambda over the whole array and this module cannot reach
#: inside it to restrict that reduction the way :func:`rl` does. So a padded CPU run differs
#: from an unpadded one by ~1e-04 of peak, which moves 0.30% to 0.41% of uint16 pixels by one
#: count. That is below the shot noise in the data (peak 2977 counts implies ~55 counts of
#: photon noise) and scientifically irrelevant, but it is still a silent change to the numbers
#: on the path that is the DEFAULT and the FALLBACK, and this repo's IMA-247 rule is that such
#: changes are opt-in and named. On the GPU path, where the lambda can be and is restricted,
#: padding is on by default because it costs nothing numerically.
PAD_CPU_ENV_VAR = "SQUIDMIP_DECON_PAD_CPU"

#: How much longer a padded axis may get before padding stops being worth it. 2084 -> 2160 is
#: 1.036 per axis and 1.075 in area, which the measured 1.24x-8.97x pays for many times over.
MAX_PAD_GROWTH = 1.15

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


def select_device(shape, *, gpu: bool = True, psf_shape=None) -> Optional[str]:
    """Which device should run a volume of *shape*, or ``None`` for "petakit's CPU path".

    Parameters
    ----------
    shape:
        The volume's ``(Z, Y, X)``. Every axis is an FFT length.
    gpu:
        The caller's existing ``gpu=`` flag (``deconvolve_plane``/``deconvolve_stack``).
        ``False`` means "CPU please" and is honoured verbatim.
    psf_shape:
        The kernel's shape, needed only to work out what the transform length will be AFTER
        :func:`pad_plan`. The smoothness test is applied to the PADDED length, not the raw one:
        padding is what makes an awkward width fast, so a width that padding fixes is fine and
        a width padding cannot fix is still refused. ``None`` means "no padding information",
        and then the raw length is tested, which is the conservative reading.

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
    if not fits_in_memory(shape):
        return None                     # petakit's z-tiler owns this one.
    if not all(is_smooth(int(n)) for n in effective_shape(shape, psf_shape)):
        # Metal would fall onto Bluestein and lose to the CPU pool: measured 0.72x at 2084^2
        # on an idle machine. Reachable only when padding is switched off or declined, since
        # otherwise the padded length is 7-smooth by construction.
        return None
    return _torch_device()


def effective_shape(shape, psf_shape=None) -> tuple[int, ...]:
    """The shape the FFT will actually run at, i.e. *shape* after :func:`pad_plan`."""
    shape = tuple(int(n) for n in shape)
    if psf_shape is None or padding_disabled():
        return shape
    return tuple(n + 2 * w for n, w in zip(shape, pad_plan(shape, psf_shape)))


# --- transform-length padding -----------------------------------------------------------------


def padding_disabled() -> bool:
    """True when ``SQUIDMIP_DECON_PAD`` turns transform-length padding off."""
    return os.environ.get(PAD_ENV_VAR, "").strip().lower() in {"off", "0", "none", "false", "no"}


def cpu_padding_enabled() -> bool:
    """True when the caller has opted the petakit CPU path into padding. See :data:`PAD_CPU_ENV_VAR`."""
    if padding_disabled():
        return False
    return os.environ.get(PAD_CPU_ENV_VAR, "").strip().lower() in {"1", "on", "yes", "true"}


def fast_len(n: int) -> int:
    """Smallest 7-smooth integer >= *n*: the shortest transform length on the fast path.

    NOT ``scipy.fft.next_fast_len``, deliberately. scipy's version admits 11 and 13 because
    ducc handles them well (``next_fast_len(2102)`` returns 2112 = 2^6 x 3 x 11). Metal does
    not: the whole reason this function exists is that Bluestein is slow there, and an
    11-smooth answer would put us back on it. Same fast-length idea, stricter alphabet.
    """
    n = max(int(n), 1)
    while not is_smooth(n):
        n += 1
    return n


def pad_plan(shape, psf_shape) -> tuple[int, ...]:
    """Wrap-pad width per axis, so every transform runs at a 7-smooth length. 0 = leave alone.

    Two rules, both from measurement (see the module docstring's PADDING section):

    * An axis whose length is ALREADY 7-smooth is left alone. Padding it would cost area for
      no transform-speed gain, and Z is the axis this matters for: a 10-plane stack is already
      smooth, so a 3-D run pads Y and X and leaves the stack depth exactly as acquired.
    * Otherwise the pad is at least the PSF's full extent on that axis, then the total is
      rounded up to :func:`fast_len`. The PSF-extent floor is what makes the wrap exact; it was
      measured, not assumed (below).

    If the rounded-up length would exceed :data:`MAX_PAD_GROWTH` of the original, the axis is
    left alone rather than paying more in area than the fast transform gives back.
    """
    widths = []
    for n, k in zip((int(s) for s in shape), (int(s) for s in psf_shape)):
        if is_smooth(n):
            widths.append(0)
            continue
        target = fast_len(n + 2 * min(k, n))
        widths.append(0 if target > n * MAX_PAD_GROWTH else (target - n) // 2)
    return tuple(widths)


def _wrap_pad(volume: np.ndarray, widths) -> np.ndarray:
    """Extend *volume* by *widths* per side using WRAP, i.e. pixels from the opposite edge.

    The pad MODE is the whole correctness argument, so it is worth being explicit. petakit
    transforms at exactly the image shape (``engine.py:139 out_shape = image.shape``), which
    makes the convolution fully CIRCULAR: the left edge already convolves against the right
    edge. Wrap-padding extends that same circle outward, so the true region sees exactly the
    neighbours it saw before. Zero-padding or edge-replication would each impose a DIFFERENT
    boundary condition and change the answer at the rim: measured on the user's own 2084^2
    plane, zero-pad moved border pixels by 4.4e-02 of peak and edge-replication by 1.9e-02,
    against a 1e-04 tolerance. Wrap moved them by 3e-05 at worst, and by 8e-07 once the pad is
    wide enough. Wrap is not a nicety here; it is the only mode that keeps the operator's
    output the same operator's output.
    """
    if not any(widths):
        return volume
    return np.pad(volume, [(w, w) for w in widths], mode="wrap")


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
    raw = np.maximum(np.asarray(volume, dtype=np.float32), 0)
    widths = (0, 0, 0) if padding_disabled() else pad_plan(raw.shape, psf.shape)
    image_np = _wrap_pad(raw, widths)
    shape = image_np.shape
    #: The TRUE region inside the padded array. When nothing was padded this is the whole
    #: array and every line below is byte-for-byte the previous behaviour.
    core = tuple(slice(w, w + n) for w, n in zip(widths, raw.shape))
    dev = torch.device(device)

    with _device_lock:
        image = torch.from_numpy(image_np).to(dev)
        H = torch.fft.rfftn(torch.from_numpy(_otf_source(psf, shape)).to(dev), dim=dims)
        H_conj = torch.conj(H)

        J_2 = image.clone()
        J_3 = torch.zeros_like(image)
        J_4 = torch.zeros(J_2[core].numel(), dtype=torch.float32, device=dev)
        lam = 0.0
        eps = float(np.finfo(np.float32).eps)
        Y = None

        for k in range(1, iterations + 1):
            # THE LAMBDA IS REDUCED OVER THE TRUE REGION ONLY, and that is what makes padding
            # honest. Biggs-Andrews' lambda is a GLOBAL dot-product ratio
            # (``petakit/engine.py:153-156``), so any pixel added to the array changes it, and a
            # changed lambda moves EVERY pixel, not just the rim. Measured on the user's 2084^2
            # plane, padding with lambda left global cost 8.3e-05 of peak; restricting the two
            # reductions to the unpadded region drops that to 1.1e-06, which is the same floor
            # this backend already sits at against petakit without any padding at all. Verified
            # at 3, 10 and 25 iterations.
            if k > 2 and Y is not None:
                diff = (J_2 - Y)[core].ravel()
                num = float(torch.dot(diff, J_4))
                den = float(torch.dot(J_4, J_4))
                lam = max(min(num / (den + eps), 1.0), 0.0)
                J_4 = diff
            elif k == 2 and Y is not None:
                J_4 = (J_2 - Y)[core].ravel()

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

        out = J_2[core].contiguous().to("cpu").numpy()

    return out.astype(np.float32, copy=False)


def _pad_note(shape, psf_shape=None) -> str:
    """The ' padded A -> B' clause for a log line, or '' when nothing is padded."""
    if psf_shape is None or padding_disabled():
        return ""
    widths = pad_plan(shape, psf_shape)
    if not any(widths):
        return ""
    padded = tuple(int(n) + 2 * w for n, w in zip(shape, widths))
    grew = float(np.prod(padded)) / float(np.prod([int(n) for n in shape]))
    return f", transform padded to {padded} (+{grew - 1:.1%} area) for a 7-smooth FFT"


def describe(shape, *, gpu: bool = True, psf_shape=None) -> str:
    """A one-line, human-readable account of the device decision, for logs, not for control flow."""
    device = select_device(shape, gpu=gpu, psf_shape=psf_shape)
    if device is not None:
        return f"decon backend: torch/{device} for shape {tuple(shape)}{_pad_note(shape, psf_shape)}"
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
    cpu_pad = " (padded, SQUIDMIP_DECON_PAD_CPU)" if cpu_padding_enabled() else ""
    return f"decon backend: petakit CPU{cpu_pad}"


def log_choice(shape, *, gpu: bool = True, psf_shape=None) -> None:
    """Say which device a run picked, ONCE per distinct answer.

    ``_cellpose.py`` states the rule this follows: ask for the GPU, then log what was actually
    chosen, "so a demo that is silently on CPU is visible in the log panel rather than a
    mystery-slow run". A deconvolution that quietly ran on the CPU, or that padded its
    transform, is exactly that: invisible from the outside and worth one line.

    Once per distinct message, not once per plane: ``project_plate`` calls the operator thousands
    of times per run and all but the first would be the same sentence.
    """
    message = describe(shape, gpu=gpu, psf_shape=psf_shape)
    with _log_once_lock:
        if message in _LOG_ONCE:
            return
        _LOG_ONCE.add(message)
    from squidmip._logpane import get_logger

    get_logger("decon").info(message)
