"""Apple-Silicon (Metal/MPS) and torch-CUDA backend for the Richardson-Lucy update.

A backend, not an algorithm: a transcription of ``petakit.engine._rl_core`` with one departure
(``rfftn`` for ``fftn``, an identity for real input). Non-7-smooth transform lengths are
wrap-padded to fast lengths; CUDA machines where CuPy works are left to petakit.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

#: Device override: auto/unset, cpu/off/0/none, or an explicit mps/cuda (skips the smoothness guard).
ENV_VAR = "SQUIDXPLORER_DECON_DEVICE"

#: Prime factors Metal's FFT handles on its fast path; anything larger falls onto Bluestein.
SMOOTH_PRIMES = (2, 3, 5, 7)

#: Padding off-switch: ``SQUIDXPLORER_DECON_PAD=off`` runs every transform at the raw length.
PAD_ENV_VAR = "SQUIDXPLORER_DECON_PAD"

#: Opt-in: also pad the petakit CPU path (off by default — it changes the numbers by ~1e-04 of peak).
PAD_CPU_ENV_VAR = "SQUIDXPLORER_DECON_PAD_CPU"

#: How much longer a padded axis may get before padding stops being worth it.
MAX_PAD_GROWTH = 1.15

#: Bytes of working set per voxel (petakit's own estimate, deliberately pessimistic).
BYTES_PER_VOXEL = 40

#: Fraction of free system memory a single GPU volume may claim; oversized volumes are declined
#: so petakit's z-tiler handles them on the CPU.
MEMORY_FRACTION = 0.5

# One GPU, one volume at a time: parallel submission gains nothing and multiplies the footprint.
_device_lock = threading.Lock()

_LOG_ONCE: set[str] = set()
_log_once_lock = threading.Lock()


def is_smooth(n: int, primes=SMOOTH_PRIMES) -> bool:
    """True when *n* factors entirely into *primes*, i.e. the FFT fast path applies."""
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
    """Free system memory in bytes; without psutil, assume a small machine."""
    try:
        import psutil

        return float(psutil.virtual_memory().available)
    except Exception:
        return 4e9


def fits_in_memory(shape) -> bool:
    """True when one volume of *shape* is small enough for this tiler-less backend to accept."""
    return int(np.prod(shape)) * BYTES_PER_VOXEL < _free_bytes() * MEMORY_FRACTION


def _torch_device() -> Optional[str]:
    """The best torch device present, or ``None``. Never raises — torch is an optional import."""
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
    """Which device should run a volume of *shape* — ``"mps"``, ``"cuda"``, or ``None`` for CPU."""
    override = os.environ.get(ENV_VAR, "").strip().lower()
    if override in {"cpu", "off", "0", "none", "false"}:
        return None
    if override in {"mps", "cuda"}:
        # An explicit device skips the smoothness guard but not the memory guard, and must fall
        # back rather than name a device this machine does not have.
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
        # Metal would fall onto Bluestein and lose to the CPU pool.
        return None
    return _torch_device()


def effective_shape(shape, psf_shape=None) -> tuple[int, ...]:
    """The shape the FFT will actually run at, i.e. *shape* after :func:`pad_plan`."""
    shape = tuple(int(n) for n in shape)
    if psf_shape is None or padding_disabled():
        return shape
    return tuple(n + 2 * w for n, w in zip(shape, pad_plan(shape, psf_shape)))


def padding_disabled() -> bool:
    """True when ``SQUIDXPLORER_DECON_PAD`` turns transform-length padding off."""
    return os.environ.get(PAD_ENV_VAR, "").strip().lower() in {"off", "0", "none", "false", "no"}


def cpu_padding_enabled() -> bool:
    """True when the caller has opted the petakit CPU path into padding. See :data:`PAD_CPU_ENV_VAR`."""
    if padding_disabled():
        return False
    return os.environ.get(PAD_CPU_ENV_VAR, "").strip().lower() in {"1", "on", "yes", "true"}


def fast_len(n: int) -> int:
    """Smallest 7-smooth integer >= *n* (stricter than scipy's next_fast_len: Metal has no fast 11/13)."""
    n = max(int(n), 1)
    while not is_smooth(n):
        n += 1
    return n


def pad_plan(shape, psf_shape) -> tuple[int, ...]:
    """Wrap-pad width per axis, so every transform runs at a 7-smooth length. 0 = leave alone.

    Already-smooth axes are untouched; otherwise the pad is at least the PSF's extent (which
    makes the wrap exact), rounded up to :func:`fast_len`, unless growth exceeds
    :data:`MAX_PAD_GROWTH`.
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
    """Extend *volume* by *widths* per side using WRAP.

    Wrap is the only mode that preserves petakit's circular convolution; zero-pad or
    edge-replication would change the answer at the rim.
    """
    if not any(widths):
        return volume
    return np.pad(volume, [(w, w) for w in widths], mode="wrap")


def _otf_source(psf: np.ndarray, out_shape) -> np.ndarray:
    """petakit's ``crop_psf_to_image`` + ``psf2otf`` padding/circshift, before the transform."""
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


def rl(volume: np.ndarray, psf: np.ndarray, iterations: int, device: str,
       snapshot_iters=None):
    """Biggs-Andrews accelerated Richardson-Lucy on *device*. Returns float32 ``(Z, Y, X)``.

    ``snapshot_iters`` (petakit's own contract): an iterable of iteration counts; the return
    becomes ``{iter: volume}`` capturing the estimate after each requested iteration of ONE
    solve. The loop runs to ``max(iterations, max(snapshot_iters))``.
    """
    import torch

    snaps = sorted({int(i) for i in snapshot_iters}) if snapshot_iters else None
    if snaps is not None:
        iterations = max(int(iterations), snaps[-1])
    captured: Optional[dict] = {} if snaps is not None else None
    dims = (-3, -2, -1)
    raw = np.maximum(np.asarray(volume, dtype=np.float32), 0)
    widths = (0, 0, 0) if padding_disabled() else pad_plan(raw.shape, psf.shape)
    image_np = _wrap_pad(raw, widths)
    shape = image_np.shape
    # The TRUE region inside the padded array (the whole array when nothing was padded).
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
            # The lambda is reduced over the TRUE region only, which is what makes padding honest:
            # Biggs-Andrews' lambda is a global dot-product ratio, so padded pixels would move it.
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

            if snaps is not None and k in snaps:
                captured[k] = (J_2[core].contiguous().to("cpu").numpy()
                               .astype(np.float32, copy=False))

        if snaps is not None:
            return captured
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
    cpu_pad = " (padded, SQUIDXPLORER_DECON_PAD_CPU)" if cpu_padding_enabled() else ""
    return f"decon backend: petakit CPU{cpu_pad}"


def log_choice(shape, *, gpu: bool = True, psf_shape=None) -> None:
    """Say which device a run picked, once per distinct answer (not once per plane)."""
    message = describe(shape, gpu=gpu, psf_shape=psf_shape)
    with _log_once_lock:
        if message in _LOG_ONCE:
            return
        _LOG_ONCE.add(message)
    from squidxplorer._logpane import get_logger

    get_logger("decon").info(message)
