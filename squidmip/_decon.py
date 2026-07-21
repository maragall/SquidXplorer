"""Deconvolution as a PLANE-OP, on Julio's PetaKit engine and a REAL PSF (IMA-223, IMA-247).

One ``add_projector`` call, zero engine edits: this module declares ``consumes=frozenset()``
via :func:`squidmip.plane_op`, so ``project_well``'s existing group-by-then-reduce loop hands
it ONE plane at a time and z survives at full depth (IMA-210). Nothing in ``_engine.py`` knows
this operator exists.

IMA-247: the reimplementation is gone
-------------------------------------
This module used to carry a from-scratch Richardson-Lucy that assumed a **Gaussian** PSF of a
hardcoded ``sigma = 1.5 px``, justified in a docstring by "NA 0.4". The scope this runs on is
NA **0.3** — the constant was not merely an approximation, it was an approximation of the wrong
instrument. Meanwhile ``/Users/julioamaragall/CEPHLA/projects/deconvolution`` (the ``petakit``
package) already modelled the PSF from the acquisition optics and already shipped the
PetaKit5D RL engine. That repo is now the runtime path and the reimplementation is deleted, not
kept as a fallback: see IMA-247 for the audit.

What is reused, and from where
------------------------------
* **``petakit.generate_psf``** — a *vectorial* PSF from ``psfmodels``, given NA, emission
  wavelength, pixel size, z-step and immersion index. This is the whole point of the ticket:
  the kernel is derived from the instrument, not assumed to be a Gaussian.
* **``petakit.compute_psf_size``** — how big that kernel has to be (6 Airy radii laterally,
  6 axial FWHM or the ``2*Nz-1`` signal-processing floor axially). Not guessed here.
* **``petakit.deconvolve(method="rl")``** — PetaKit5D's ``decon_lucy_function.m`` ported, with
  Biggs-Andrews acceleration. GPU via CuPy when present, CPU via scipy otherwise.
* **``petakit.open_acquisition`` / ``infer_immersion_index`` / ``wavelength_from_channel``** —
  how :meth:`OpticsParams.from_acquisition` turns an acquisition folder into optics. The
  metadata parse (``sensor_pixel_size_um / magnification``, ``dz(um)``, ``objective.NA``) is
  his, not re-derived here.

Measured against the deleted implementation (real 10x data, ``manual0_0``, channel 488): the
real PSF's second-moment-equivalent sigma is **1.165 px**, not the 1.5 px that was hardcoded —
the old kernel was ~29% too wide. The two results correlate 0.9932 and both conserve flux
exactly, so the old code was not *broken*; it was sharpening with the wrong kernel, which is
precisely the difference that matters on a scientific tool.

TWO TRAPS FOUND IN petakit, BOTH PINNED BY TESTS HERE
-----------------------------------------------------
1. **``method="omw"`` is petakit's default and it returns an ALL-ZERO volume on this data.**
   (Measured: ``petakit.deconvolve(stack, psf, method="omw")`` -> every pixel 0.0 on the 10x
   NA-0.3 stack, where ``method="rl"`` returns a sane 1170..10513 range.) OMW's masked-Wiener
   back-projector does not survive this PSF/volume geometry. So this module **pins
   ``method="rl"``** and never inherits the default, and :func:`_run` raises if the engine
   hands back a degenerate all-zero result rather than writing black tiles to disk.
2. **A real PSF is 3-D, but the plane-op seam is 2-D.** A plane-op maps plane -> plane; it
   never sees the z-stack, so it cannot do true 3-D deconvolution. The registered ``decon``
   therefore convolves with the **in-focus plane** of the vectorial PSF — a genuine 2-D
   widefield deconvolution with real optics, and a strict improvement on a made-up Gaussian at
   the same seam. Where a real PSF actually pays off is in 3-D, so :func:`decon3d_op` is also
   provided: it declares ``consumes={"z"}``, receives the whole stack, deconvolves in 3-D and
   then projects. That is still ZERO engine edits — ``add_projector`` has taken a ``consumes``
   declaration since IMA-210. Measured on a 10x512x512 crop: 3-D RL doubles gradient-energy
   sharpness (0.2526 -> 0.5282) where the 2-D path cannot, because out-of-focus light from
   neighbouring planes is exactly what a 3-D PSF removes.

Where the optics come from
--------------------------
PSF parameters are **acquisition metadata**, not constants::

    set_optics(OpticsParams.from_acquisition(dataset_path, channel="488"))

The plane-op seam carries no metadata alongside the plane (the same limitation
``_flatfield.py`` documents for per-channel profiles), so the registered ``decon`` reads a
module-level *active* optics record, exactly as ``_flatfield.py`` does for its profile. Unlike
a flat-field, decon has a defensible default — :data:`DEFAULT_OPTICS` is the **measured
configuration of the 10x scope this tool ships against** (NA 0.3, 10x, 7.52 um sensor pitch,
1.5 um z-step, 525 nm emission), transcribed from that instrument's own
``acquisition parameters.json`` rather than invented — so ``decon`` keeps working with no setup
for the benchmark and the walkthrough. It is a *starting point that names its instrument*, and
:func:`set_optics` overrides it from the data actually loaded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Optional

import numpy as np

from squidmip._engine import add_projector
from squidmip.projection import plane_op

# RL iterations. Julio's working point on this instrument, not a textbook default.
#
# Richardson-Lucy is SEMI-CONVERGENT: error against the truth falls, reaches a minimum, then
# rises again as the algorithm starts fitting noise. Ringing re-grows and point-like structures
# develop a bright core with an expanding halo. So more iterations is not more deconvolved, it
# is eventually less, and there is no universally correct count - it depends on SNR and on the
# PSF, which is why IMA-252 puts a turbo XZ/YZ view in front of a human to judge the turn.
#
# 3 is the default and 2 is where the QC loop starts. The previous value here was 10, justified
# as "the widefield working point"; that was a generic number, and this instrument (NA 0.3,
# 0.752 um/px, Nz=10) is not the generic case.
DEFAULT_ITERATIONS: int = 3
QC_START_ITERATIONS: int = 2

# PINNED, never inherited from petakit's default. petakit's default is "omw", which returns an
# all-zero volume on this instrument's data (see the module docstring, trap 1).
METHOD: str = "rl"

_MISSING = (
    "deconvolution needs Julio's petakit engine, which is not importable.\n"
    "  repo:    /Users/julioamaragall/CEPHLA/projects/deconvolution\n"
    "  install: pip install --no-deps -e /Users/julioamaragall/CEPHLA/projects/deconvolution\n"
    "           pip install psfmodels          # the vectorial PSF model\n"
    "There is deliberately NO fallback: this module used to approximate the PSF with a "
    "Gaussian, and silently reverting to that would mean a user asking for deconvolution "
    "could not tell which kernel actually ran (IMA-247)."
)


def _petakit():
    """Import petakit LAZILY and fail LOUD — never silently substitute another algorithm.

    Lazy for the same reason ``_flatfield.py`` imports tilefusion lazily: ``import petakit``
    pulls scipy.fft, psfmodels and (where present) CuPy, and the headless import graph should
    not pay for that unless deconvolution is actually asked for.
    """
    try:
        import petakit
    except ImportError as exc:                      # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from exc
    return petakit


def _as_channel_name(channel) -> str:
    """Normalise a channel label into the form ``petakit.wavelength_from_channel`` can parse.

    A THIRD petakit seam that does not quite meet itself: its OME-TIFF reader returns channel
    names as bare wavelength numbers (``"488"`` — ``_parse_channels_from_ome`` keeps only the
    digits), but ``wavelength_from_channel`` matches ``r"(\\d{3})\\s*nm"`` and raises on a bare
    number. Feeding his reader's output straight into his own parser therefore fails. Rather
    than duplicate the Stokes-shift table here, re-spell the label and keep using his table.
    """
    text = str(channel).strip()
    if text.isdigit():
        return f"{text} nm"
    # Squid's own channel labels are UNDERSCORED ("Fluorescence_405_nm_Ex"), which is what
    # squidmip's reader reports and therefore what a caller has in hand. petakit's regex wants
    # whitespace before "nm", so the underscored form raises there too (IMA-252). Same one-line
    # re-spelling, same table.
    return text.replace("_", " ")


@dataclass(frozen=True)
class OpticsParams:
    """The acquisition optics a PSF is computed from. A frozen, hashable record.

    Hashable on purpose: :func:`make_psf` is ``lru_cache``d on it, so a plate run generates the
    vectorial PSF once rather than once per plane.

    na:
        Objective numerical aperture (``objective.NA``).
    wavelength_um:
        **Emission** wavelength in um. Emission, not excitation — the PSF is formed by the
        light that reaches the sensor. :meth:`from_acquisition` applies petakit's Stokes-shift
        table (488 nm excitation -> 0.525 um emission).
    dxy_um:
        Pixel size in the sample plane: ``sensor_pixel_size_um / magnification``.
    dz_um:
        Z-step in um.
    nz:
        Number of acquired z-planes; sets the axial PSF extent floor (``2*Nz-1``).
    ni:
        Immersion refractive index. ``None`` infers it from NA via petakit
        (<=1.0 air, <=1.33 water, else oil).
    """
    na: float
    wavelength_um: float
    dxy_um: float
    dz_um: float = 1.5
    nz: int = 1
    ni: Optional[float] = None

    def __post_init__(self) -> None:
        for field in ("na", "wavelength_um", "dxy_um", "dz_um"):
            value = getattr(self, field)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be a positive finite number, got {value!r}")
        if self.nz < 1:
            raise ValueError(f"nz must be >= 1, got {self.nz}")
        if self.ni is not None and (not np.isfinite(self.ni) or self.ni < 1.0):
            raise ValueError(f"ni (immersion index) must be >= 1.0, got {self.ni!r}")

    @property
    def immersion_index(self) -> float:
        """The immersion index, inferring it from NA when it was not given (petakit's rule)."""
        if self.ni is not None:
            return float(self.ni)
        return float(_petakit().infer_immersion_index(self.na))

    @classmethod
    def from_acquisition(cls, path, channel: Optional[str] = None) -> "OpticsParams":
        """Read the optics off a real acquisition — **the intended way to build this**.

        Uses petakit's own reader, so the metadata parse (and the format detection across
        OME-TIFF / individual TIFF / CurrentStack) is his code, not a second parser that can
        drift from it.

        *channel* picks the emission wavelength; ``None`` takes the acquisition's first
        channel. A channel name may be either petakit's short form (``"488"``) or the long
        Squid form (``"Fluorescence 488 nm Ex"``) — both parse to the same wavelength.
        """
        petakit = _petakit()
        acq = petakit.open_acquisition(path)
        meta = acq.metadata
        if channel is None:
            if not meta.channels:
                raise ValueError(
                    f"the acquisition at {path} declares no channels, so there is no emission "
                    "wavelength to build a PSF from. Pass channel= explicitly."
                )
            channel = meta.channels[0]
        return cls(
            na=float(meta.na),
            wavelength_um=float(petakit.wavelength_from_channel(_as_channel_name(channel))),
            dxy_um=float(meta.dxy),
            dz_um=float(meta.dz),
            nz=int(meta.nz),
        )


# The 10x scope this tool ships against, transcribed from its own acquisition metadata
# (`objective: {magnification: 10.0, NA: 0.3}`, `sensor_pixel_size_um: 7.52`, `dz(um): 1.5`)
# with the 488 nm line's ~525 nm emission. A named instrument, not a tuning constant — and
# overridden by set_optics() the moment a real dataset is loaded.
DEFAULT_OPTICS = OpticsParams(na=0.3, wavelength_um=0.525, dxy_um=7.52 / 10.0, dz_um=1.5, nz=10)


@lru_cache(maxsize=8)
def make_psf(optics: OpticsParams) -> np.ndarray:
    """The 3-D vectorial PSF for *optics*, ``(Z, Y, X)`` float32 normalised to sum 1.

    Both the sizing and the model are petakit's (``compute_psf_size`` then ``generate_psf``).
    Cached, because the plate engine calls the operator once per plane while the PSF depends
    only on the optics.
    """
    petakit = _petakit()
    ni = optics.immersion_index
    nz_psf, nxy_psf = petakit.compute_psf_size(
        optics.nz, optics.dxy_um, optics.dz_um,
        wavelength=optics.wavelength_um, na=optics.na, ni=ni,
    )
    psf = petakit.generate_psf(
        nz=nz_psf, nxy=nxy_psf,
        dxy=optics.dxy_um, dz=optics.dz_um,
        wavelength=optics.wavelength_um, na=optics.na, ni=ni,
    )
    return np.ascontiguousarray(psf, dtype=np.float32)


@lru_cache(maxsize=8)
def make_psf_2d(optics: OpticsParams) -> np.ndarray:
    """The **in-focus plane** of the 3-D PSF, shaped ``(1, Y, X)`` and renormalised to sum 1.

    This is the kernel the plane-op seam can actually use (module docstring, trap 2). It is a
    real widefield 2-D PSF from real optics — not a Gaussian standing in for one.
    """
    psf3 = make_psf(optics)
    centre = psf3[psf3.shape[0] // 2]
    total = float(centre.sum())
    if total <= 0:
        raise ValueError(f"the in-focus PSF plane for {optics!r} sums to {total}; cannot normalise")
    return np.ascontiguousarray((centre / total)[None, ...], dtype=np.float32)


def _run(volume: np.ndarray, psf: np.ndarray, iterations: int, gpu: bool) -> np.ndarray:
    """One call into petakit's RL, with the all-zero guard from trap 1."""
    petakit = _petakit()
    out = petakit.deconvolve(
        np.ascontiguousarray(volume, dtype=np.float32), psf,
        method=METHOD, iterations=iterations, gpu=gpu,
    )
    if np.any(volume) and not np.any(out):
        raise RuntimeError(
            "petakit returned an all-zero result for a non-empty input. That is the failure "
            f"mode method='omw' shows on this instrument's geometry; this call used "
            f"method={METHOD!r} with a PSF of shape {psf.shape}. Refusing to hand back a black "
            "image that would look like a successful deconvolution."
        )
    return out


def deconvolve_plane(
    plane: np.ndarray,
    optics: Optional[OpticsParams] = None,
    iterations: int = DEFAULT_ITERATIONS,
    *,
    gpu: bool = True,
) -> np.ndarray:
    """Deconvolve ONE plane with the real in-focus PSF for *optics*. Same shape and dtype.

    Parameters
    ----------
    plane:
        2-D image, any dtype. The caller's array is never mutated.
    optics:
        Acquisition optics. ``None`` uses the active optics (:func:`set_optics`), which falls
        back to :data:`DEFAULT_OPTICS`.
    iterations:
        RL iterations. ``0`` is the identity (a plain copy), so "no deconvolution" has an
        unambiguous spelling and a benchmark has a zero point.
    gpu:
        Hand the volume to CuPy when a CUDA device is present. This selects a *backend*, not an
        algorithm — the RL update is identical either way — so falling back to CPU is not the
        kind of silent substitution IMA-247 forbids.

    Returns
    -------
    np.ndarray
        Same shape and dtype as *plane*. Integer dtypes are **clipped** to the dtype range
        before the cast, never wrapped — an RL overshoot on a saturated punctum would otherwise
        turn the brightest pixel in the frame into a black one.
    """
    if plane.ndim != 2:
        raise ValueError(f"deconvolve_plane takes ONE 2-D plane; got shape {plane.shape}")
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if iterations == 0:
        return np.array(plane, copy=True)

    optics = optics or active_optics()
    out = _run(plane[None, ...], make_psf_2d(optics), iterations, gpu)[0]
    return _cast_like(out, plane.dtype)


def deconvolve_stack(
    planes: Iterable[np.ndarray],
    optics: Optional[OpticsParams] = None,
    iterations: int = DEFAULT_ITERATIONS,
    *,
    gpu: bool = True,
) -> np.ndarray:
    """TRUE 3-D deconvolution of a whole z-stack with the full 3-D PSF, then a MIP.

    This is where modelling a real PSF actually earns its keep: the out-of-focus light in each
    plane comes from its neighbours, and only a 3-D kernel can put it back. Returns ONE plane,
    which is the z-reducer contract — see :func:`decon3d_op`.
    """
    stack = planes if isinstance(planes, np.ndarray) else np.asarray(list(planes))
    if stack.ndim != 3 or stack.shape[0] < 1:
        raise ValueError(f"deconvolve_stack needs (Z, Y, X); got shape {stack.shape}")
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    dtype = stack.dtype
    if iterations == 0:
        return stack.max(axis=0)

    optics = optics or active_optics()
    # The acquired depth sets the PSF's axial extent, so bind it to the actual stack rather
    # than to whatever nz the optics record happened to carry.
    if optics.nz != stack.shape[0]:
        optics = OpticsParams(optics.na, optics.wavelength_um, optics.dxy_um,
                              optics.dz_um, int(stack.shape[0]), optics.ni)
    out = _run(stack, make_psf(optics), iterations, gpu)
    return _cast_like(out.max(axis=0), dtype)


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast back to the acquisition dtype, ROUNDING and clipping integers rather than
    truncating and wrapping them. ``astype`` alone truncates toward zero (half a count of
    systematic dimming on every pixel) and wraps on overflow (the brightest pixel in the frame
    becomes the darkest)."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    return values.astype(dtype, copy=False)


# --- the ACTIVE optics, for the registry entry -------------------------------------------------
#
# Same shape as _flatfield.py's active profile and for the same reason: the registered operator
# is selected by NAME, so it cannot take arguments, and the plane-op seam does not carry
# acquisition metadata alongside the plane. Guarded by a lock because project_plate runs the
# operator on a thread pool.
_lock = threading.Lock()
_active: Optional[OpticsParams] = None


def set_optics(optics: OpticsParams) -> None:
    """Install the optics the registered ``decon``/``decon3d`` operators compute their PSF from.

    THE intended entry point for real work::

        set_optics(OpticsParams.from_acquisition(dataset_path, channel="488"))
    """
    global _active
    if not isinstance(optics, OpticsParams):
        raise ValueError(f"set_optics needs an OpticsParams, got {type(optics).__name__}")
    with _lock:
        _active = optics


def active_optics() -> OpticsParams:
    """The installed optics, or :data:`DEFAULT_OPTICS` when none has been set."""
    with _lock:
        return _active if _active is not None else DEFAULT_OPTICS


def clear_optics() -> None:
    """Uninstall the optics; the operators go back to :data:`DEFAULT_OPTICS`."""
    global _active
    with _lock:
        _active = None


def deconvolve(plane: np.ndarray) -> np.ndarray:
    """Deconvolve one plane with the active optics — the function behind the ``decon`` name."""
    return deconvolve_plane(plane, None, DEFAULT_ITERATIONS)


def decon_op(
    optics: Optional[OpticsParams] = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build a parameterised deconvolution **plane-op**, ready for ``add_projector``::

        add_projector("decon_sharp", decon_op(iterations=25))

    The returned callable carries ``consumes = frozenset()`` (stamped by
    :func:`squidmip.plane_op`), so the registry infers the declaration and z survives.
    """
    def _decon(p: np.ndarray) -> np.ndarray:
        return deconvolve_plane(p, optics, iterations)

    _decon.__name__ = f"decon(rl,iterations={iterations})"
    return plane_op(_decon)


def decon3d_op(
    optics: Optional[OpticsParams] = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build a TRUE 3-D deconvolution operator: a **z-reducer** (``consumes={"z"}``).

    Registered through the same ``add_projector`` seam with no engine edit — ``consumes`` has
    been part of that signature since IMA-210. Unlike the plane-op, this one collapses z (it
    deconvolves the volume and then projects), which is the honest shape for an operator that
    needs the whole stack to do its job.
    """
    def _decon3d(planes: Iterable[np.ndarray]) -> np.ndarray:
        return deconvolve_stack(planes, optics, iterations)

    _decon3d.__name__ = f"decon3d(rl,iterations={iterations})"
    _decon3d.consumes = frozenset({"z"})
    return _decon3d


# The whole registration. No engine edit — that is the IMA-210 seam working as designed.
add_projector("decon", plane_op(deconvolve))
add_projector("decon3d", decon3d_op(), consumes=frozenset({"z"}))
