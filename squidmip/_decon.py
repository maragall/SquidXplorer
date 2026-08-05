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

  CuPy is CUDA, so "GPU when present" has always meant "never, on a Mac". :mod:`squidmip._decon_gpu`
  adds Metal: the SAME Biggs-Andrews RL update, transcribed onto torch, chosen by :func:`_run`
  only when it can be shown to beat the CPU thread pool, and disabled by
  ``SQUIDMIP_DECON_DEVICE=cpu``.

  It also **pads the transform to a fast FFT length**, and on this instrument that is not a
  detail, it is the whole result. Frames are 2084 = 2^2 x 521 wide, which is the slow Bluestein
  case, and at that raw width the GPU is 0.72x, i.e. LOSES to the CPU pool. Wrap-padded to 2160
  (+7.4% area) it is 3.41x, and the CPU alone, padded the same way, is 1.48x. Measured on a real
  plane from the acquisition, idle Apple M4, best of nine interleaved repeats. CPU padding is
  opt-in (``SQUIDMIP_DECON_PAD_CPU=1``) because petakit's Biggs-Andrews lambda is reduced over
  the whole array and cannot be corrected from this layer; the GPU path corrects it and so pads
  by default. That module's docstring carries the table and the traps.
* **``petakit.infer_immersion_index`` / ``wavelength_from_channel``** — two pure lookups: NA to
  immersion index, and an excitation line to its Stokes-shifted emission (488 -> 525). Tables,
  not parsers; neither touches the acquisition.

  ``petakit.open_acquisition`` USED TO BE ON THIS LIST and is not any more — see the section on
  where the optics come from, below.

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

Where the optics come from: PER CHANNEL, AT THE CALL SITE
--------------------------------------------------------
PSF parameters are **acquisition metadata**, not constants — and the one that varies WITHIN an
acquisition is the emission wavelength, because it is a property of the CHANNEL::

    Fluorescence_405_nm_Ex -> 0.450 um      Fluorescence_561_nm_Ex -> 0.590 um
    Fluorescence_488_nm_Ex -> 0.525 um      Fluorescence_638_nm_Ex -> 0.670 um

(measured, ``OpticsParams.from_acquisition`` on the 10x tissue acquisition this ships against.)

That is 1.49x of wavelength across the four, and the PSF width follows it: measured second-moment
sigma 1.165 px at 525 nm against 1.441 px at 670 nm (kernels 19x19 and 23x23 px), so the 525 nm
kernel is **19% narrower** than the one the 638 channel needs. Deconvolving every channel with one
PSF is not an approximation of the instrument, it is three of the four channels sharpened with
another channel's optics.

THE DEFECT THIS SECTION USED TO DESCRIBE, stated plainly because it shipped: the registered
``decon`` read a module-level *active* optics record and fell back to :data:`DEFAULT_OPTICS`
(525 nm), and ``set_optics`` — the function this docstring called "THE intended entry point" —
**was never called anywhere in the package**. Measured on one real 638 plane (manual0/fov 0/z 5,
3 RL iterations): 97.28% of the 4.34 M pixels differ between the 525 nm PSF and the correct
670 nm one, mean 13.8 counts, and 190 counts mean (1684 max) over the brightest 0.1% — the
puncta a biologist is looking at.

So the optics are now derived **per channel, at the call site**. ``project_well`` knows the
channel it is about to read and the acquisition it is reading from; an operator that can be
specialised to a channel says so by carrying a ``for_channel`` attribute (the same kind of
declaration as ``consumes``/``produces``/``select_index``, never a branch on the operator's
name), and ``project_well`` calls it once per channel. :func:`optics_for_channel` is what
``decon``/``decon3d`` hand it, cached by ``(acquisition, channel)``.

...FROM THIS PACKAGE'S OWN PARSE, which cost a user a whole operator (2026-08-05)
--------------------------------------------------------------------------------
:meth:`OpticsParams.from_acquisition` used to answer "what wavelength is this channel" by
calling ``petakit.open_acquisition(path)`` — a SECOND acquisition reader, re-detecting the
format and re-parsing the metadata of a folder squidmip had already opened. It detects
individual-TIFF acquisitions by globbing ``*/*_Fluorescence_*_nm_Ex.tiff``
(``petakit/readers/detect.py:28``), and a real Squid multi-band channel is named
``Fluorescence_638_nm_-_Penta``, which does not end in ``_nm_Ex``. So the whole acquisition came
back "Unknown acquisition format", no optics came back with it, and **decon refused outright**
for every user with a Penta cube — including this repo's own fixture, whose decon-on-plate test
was skipped for exactly that reason.

Two parsers, one question, and the narrower one was the one being asked. Every field now comes
from squidmip's own read of the same folder:

    wavelength  ``_channels.excitation_nm`` off the channel name, then petakit's Stokes table
                (the table is a dict; asking it 638 -> 670 opens nothing)
    dxy_um      ``acquisition.yaml`` ``objective.pixel_size_um`` via ``load_acquisition_metadata``
                — the binning-aware object-space number the rest of the app already places
                mosaics with, not ``sensor_pixel_size_um / magnification`` recomputed
    dz_um       ``acquisition.yaml`` ``z_stack.delta_z_mm`` x 1000, same parse
    nz          ``acquisition.yaml`` ``z_stack.nz``, same parse
    na          ``acquisition parameters.json`` ``objective.NA`` via ``load_objective_na``,
                because NO acquisition.yaml carries an aperture — see that function

and none of them re-detects a format. A channel name a multi-band filter made unrecognisable to
a glob is read by the module that already reads channel names for their COLOUR.

WHAT DID NOT CHANGE: the refusal. A channel whose excitation the name does not state
(``BF_LED_matrix_full``) still gets no PSF and no default — silently deconvolving every channel
at 525 nm is the defect above, and widening the parse must not be the door it comes back
through.

:func:`set_optics` survives as a **deliberate override** — set it and every channel uses it, which
is what a user who is re-deriving optics by hand wants. It is checked FIRST, so it still wins. It
is no longer the default path, and :data:`DEFAULT_OPTICS` is no longer what a plate run silently
gets.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Optional

import numpy as np

from squidmip import _decon_gpu
from squidmip._acquisition import load_acquisition_metadata, load_objective_na
from squidmip._channels import excitation_nm
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


def emission_um_for(channel) -> float:
    """The EMISSION wavelength (um) a PSF is formed at, for one channel. Or raise, naming it.

    Two steps, and each is owned by whoever already knows the answer:

    1. ``_channels.excitation_nm`` reads the excitation line off the channel name — squidmip's
       own channel parse, the same one that resolves the channel's display colour, and the one
       that copes with every spelling Squid writes (``Fluorescence_638_nm_-_Penta`` included);
    2. petakit's ``wavelength_from_channel`` applies the Stokes shift (638 -> 670). It is fed a
       canonical ``"638 nm"`` built here, so it is used as the TABLE it is and never as a parser
       of a name it might not recognise.

    No acquisition is opened, and no emission is invented for a channel that states no line.
    """
    excitation = excitation_nm(channel)
    if excitation is None:
        raise ValueError(
            f"channel {str(channel)!r} states no excitation wavelength, so it has no emission "
            "line and no PSF can be derived from it. Broadband channels (brightfield, "
            "darkfield) are the usual case."
        )
    return float(_petakit().wavelength_from_channel(f"{excitation:.0f} nm"))


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
        Pixel size in the sample plane. :meth:`from_acquisition` takes ``acquisition.yaml``'s
        ``objective.pixel_size_um``, which Squid has ALREADY computed for the objective and the
        camera binning — not ``sensor_pixel_size_um / magnification`` recomputed here, which
        ignores binning and disagrees with the number the rest of this package places mosaics
        with (0.376 vs 0.373 on the 20x scan).
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
    def from_acquisition(cls, path, channel: str) -> "OpticsParams":
        """Read the optics off a real acquisition — **the intended way to build this**.

        Every field comes from SQUIDMIP'S OWN read of the acquisition (module docstring): the
        scalars from ``_acquisition``, the wavelength from ``_channels`` via
        :func:`emission_um_for`. Nothing here re-detects the acquisition format, and there is no
        second metadata parser to drift from the one every other read path in this package uses.

        *channel* is required. It used to default to "the acquisition's first channel", which
        was a property of the other reader's channel list; a PSF is per channel, and there is no
        such thing as the right wavelength for an unspecified one.

        Raises
        ------
        ValueError
            Naming the missing field AND the file that supplies it, for every one of NA,
            pixel size, z step and channel wavelength. Never a default: each of these sets the
            width of the kernel, so a substituted value is a different measurement rather than
            a rougher one.
        """
        meta = load_acquisition_metadata(path)           # acquisition.yaml, squidmip's own parse
        na = load_objective_na(path)                     # acquisition parameters.json
        if na is None:
            raise ValueError(
                f"the acquisition at {path} states no objective NA. The aperture sets the width "
                "of the PSF, so it cannot be guessed. It is written as objective.NA in "
                "'acquisition parameters.json' (acquisition.yaml has no aperture field at all)."
            )
        if meta["pixel_size_um"] is None:
            raise ValueError(
                f"the acquisition at {path} states no objective pixel size, so the PSF has no "
                "sampling to be computed on. Add objective.pixel_size_um to acquisition.yaml."
            )
        if not meta["dz_um"]:
            raise ValueError(
                f"the acquisition at {path} states dz_um={meta['dz_um']!r}. A z step of zero or "
                "none cannot scale the axial PSF. Add z_stack.delta_z_mm to acquisition.yaml."
            )
        return cls(
            na=na,
            wavelength_um=emission_um_for(channel),
            dxy_um=float(meta["pixel_size_um"]),
            dz_um=float(meta["dz_um"]),
            nz=int(meta["n_z_declared"] or 1),
        )


# The 10x scope this tool ships against, transcribed from its own acquisition metadata
# (`objective: {magnification: 10.0, NA: 0.3}`, `sensor_pixel_size_um: 7.52`, `dz(um): 1.5`)
# with the 488 nm line's ~525 nm emission. A named instrument, not a tuning constant.
#
# THE COMMENT THAT USED TO BE HERE SAID it was "overridden by set_optics() the moment a real
# dataset is loaded". THAT WAS FALSE, and it is the whole of the defect the module docstring now
# opens with: `set_optics` was called from NOWHERE in this package, so this 525 nm record WAS the
# PSF for every channel of every run, including 405, 561 and 638.
#
# What it is now: the value for a caller who deconvolves a bare array with no acquisition behind
# it (`deconvolve_plane(plane)`, the benchmark, a doctest). Any operator running through
# `project_well` derives its optics per channel instead — see `optics_for_channel` — so this
# constant is never what a plate run gets.
DEFAULT_OPTICS = OpticsParams(na=0.3, wavelength_um=0.525, dxy_um=7.52 / 10.0, dz_um=1.5, nz=10)


# PSF cache size. The kernel depends ONLY on the optics record, and the optics record is now
# per channel, so a plate run needs one live entry per (channel, form) — 4 channels x {3-D, the
# in-focus plane} = 8 on this instrument, and `deconvolve_stack` mints one more per distinct
# stack depth. 32 leaves headroom for a 6-channel plate running decon and decon3d in one session
# without evicting a kernel that is about to be asked for again, and the kernels are small (this
# acquisition: 2-D 1.1-2.1 KB, 3-D 121x23x23 float32 = 250 KB), so the bound is arithmetic.
#
# MEASURED, this machine, on the 10x tissue acquisition (idle Apple M4, one FOV's operator calls
# for 4 channels x 10 z = 40 make_psf_2d calls):
#   cold build, per distinct optics    0.022 s (405) .. 0.045 s (638)   petakit.generate_psf
#   cached lookup                      0.18 us
#   40 calls WITHOUT the cache         1.341 s   (a vectorial-PSF rebuild per plane)
#   40 calls WITH the cache            0.092 s   (4 builds, 36 hits)  ->  14.6x
# The 3-D kernel decon3d uses costs the same 0.046 s to build and is rebuilt on the same schedule,
# so the same cache serves both paths.
_PSF_CACHE_SIZE = 32


@lru_cache(maxsize=_PSF_CACHE_SIZE)
def make_psf(optics: OpticsParams) -> np.ndarray:
    """The 3-D vectorial PSF for *optics*, ``(Z, Y, X)`` float32 normalised to sum 1.

    Both the sizing and the model are petakit's (``compute_psf_size`` then ``generate_psf``).
    Cached on the optics TUPLE (``OpticsParams`` is a frozen dataclass, so it hashes by value):
    the plate engine calls the operator once per plane, and with per-channel optics that is
    4 channels x Nz calls per FOV against 4 distinct kernels. See :data:`_PSF_CACHE_SIZE` for
    the measured cost of getting this wrong.
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


@lru_cache(maxsize=_PSF_CACHE_SIZE)
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
    """One call into RL, with the all-zero guard from trap 1.

    THE DEVICE FORK LIVES HERE, and it is a fork between *backends of the same algorithm*, never
    between algorithms. petakit's own GPU branch is CuPy, i.e. CUDA only, so on Apple Silicon it
    can never be taken and every plane runs on ``scipy.fft``: 87% of a plane's wall clock,
    measured. :mod:`squidmip._decon_gpu` runs the identical Biggs-Andrews RL update on Metal
    (or on torch-CUDA where CuPy is absent) and returns ``None`` for a device whenever it cannot
    beat the CPU thread pool, so the CPU path below stays the default and the fallback.
    ``SQUIDMIP_DECON_DEVICE`` overrides the choice in both directions.

    TRANSFORM-LENGTH PADDING. Both branches run an FFT whose length is the volume's own extent
    (``petakit/engine.py:139``), and a length with a large prime factor falls onto Bluestein and
    is slow on BOTH backends. This instrument's frames are 2084 = 2^2 x 521 wide, which is the
    bad case. The GPU branch therefore wrap-pads to the next 7-smooth length inside
    :func:`squidmip._decon_gpu.rl`, where it can also restrict the Biggs-Andrews reduction to
    the true region and so keep the answer unchanged. The CPU branch can be padded the same way
    but NOT with that correction, since the reduction lives inside petakit, so it is opt-in via
    ``SQUIDMIP_DECON_PAD_CPU=1`` and is off here by default.
    """
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    device = _decon_gpu.select_device(volume.shape, gpu=gpu, psf_shape=psf.shape)
    _decon_gpu.log_choice(volume.shape, gpu=gpu, psf_shape=psf.shape)
    if device is not None:
        out = _decon_gpu.rl(volume, psf, iterations, device)
    else:
        petakit = _petakit()
        widths = (_decon_gpu.pad_plan(volume.shape, psf.shape)
                  if _decon_gpu.cpu_padding_enabled() else (0, 0, 0))
        padded = _decon_gpu._wrap_pad(volume, widths)
        out = petakit.deconvolve(
            np.ascontiguousarray(padded), psf,
            method=METHOD, iterations=iterations, gpu=gpu,
        )
        if any(widths):
            out = out[tuple(slice(w, w + n) for w, n in zip(widths, volume.shape))]
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
        Acquisition optics. ``None`` uses the override (:func:`set_optics`) and otherwise
        :data:`DEFAULT_OPTICS` — the answer for a bare plane with no acquisition behind it. A
        caller who knows the acquisition and the channel should pass
        ``optics_for_channel(path, channel)``, which is what the registered operators do.
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


# --- the OVERRIDE optics ------------------------------------------------------------------------
#
# Same shape as _flatfield.py's active profile, but NO LONGER THE SOURCE OF TRUTH. It was, and the
# result was one 525 nm PSF for every channel of every run (module docstring). What it is now: a
# deliberate override for a caller who has derived optics by hand and wants them used for every
# channel — an escape hatch, checked FIRST by `optics_for_channel` and empty by default.
# Guarded by a lock because project_plate runs the operator on a thread pool.
_lock = threading.Lock()
_active: Optional[OpticsParams] = None


def set_optics(optics: OpticsParams) -> None:
    """Install an optics OVERRIDE, used for every channel until :func:`clear_optics`.

    Use it when you have optics the acquisition's own metadata does not describe (a filter set
    the Stokes-shift table does not know, a re-measured NA)::

        set_optics(OpticsParams(na=0.3, wavelength_um=0.610, dxy_um=0.752))

    It is NOT how a plate run gets its optics. ``decon``/``decon3d`` derive those per channel from
    the acquisition being read (:func:`optics_for_channel`), because the emission wavelength is a
    property of the channel and a single installed record cannot be right for four of them.
    """
    global _active
    if not isinstance(optics, OpticsParams):
        raise ValueError(f"set_optics needs an OpticsParams, got {type(optics).__name__}")
    with _lock:
        _active = optics


def optics_override() -> Optional[OpticsParams]:
    """The override installed by :func:`set_optics`, or ``None``. The honest question.

    :func:`active_optics` cannot answer it: it substitutes :data:`DEFAULT_OPTICS` for "nothing
    installed", which is exactly the conflation that let a default masquerade as a measurement.
    """
    with _lock:
        return _active


def active_optics() -> OpticsParams:
    """The override, or :data:`DEFAULT_OPTICS` when none has been set.

    For a caller holding a bare plane and no acquisition. Anything that knows which acquisition
    and which channel a plane came from should call :func:`optics_for_channel` instead.
    """
    with _lock:
        return _active if _active is not None else DEFAULT_OPTICS


def clear_optics() -> None:
    """Remove the override; per-channel derivation resumes."""
    global _active
    with _lock:
        _active = None


@lru_cache(maxsize=64)
def _acquisition_optics(path: str, channel: str) -> OpticsParams:
    """``OpticsParams.from_acquisition`` memoised on ``(acquisition, channel)``.

    ``from_acquisition`` re-reads and re-parses both sidecars, and the binding happens once per
    channel per FOV — 55 FOVs x 4 channels = 220 parses of the tissue plate for 4 distinct
    answers. Re-measured on this machine after the parse moved off petakit's reader onto this
    package's own (idle Apple M4, the 10x tissue acquisition): 2.565 ms cold, 0.04 us cached, so
    0.564 s of re-parsing per plate becomes 0.00001 s. Keyed by strings so two readers over the
    same folder share the entry.
    """
    return OpticsParams.from_acquisition(path, channel=channel)


def optics_for_channel(path, channel: str) -> OpticsParams:
    """The optics for ONE channel of ONE acquisition. **The per-channel seam.**

    Resolution order, and there is no fourth case:

    1. an override installed with :func:`set_optics` — deliberate, so it wins;
    2. the acquisition's own metadata for *this channel* (squidmip's parse, memoised);
    3. a refusal, naming the channel.

    No fall-through to :data:`DEFAULT_OPTICS`. That fall-through is what made every 638 plane
    deconvolve at 525 nm with nothing in the log to say so, and a wrong PSF is not a degraded
    result but a different measurement. A channel whose emission cannot be derived (brightfield
    has no emission line) must be answered by a human — with ``set_optics``, or by registering
    ``decon_op(optics=...)`` under its own name — not by this module picking a wavelength.
    """
    override = optics_override()
    if override is not None:
        return override
    if path is None:
        raise ValueError(
            f"cannot derive the PSF for channel {channel!r}: the operator was not told which "
            "acquisition it is reading, so the emission wavelength is unknown. Pass optics "
            "explicitly (decon_op(optics=...)) or install an override with set_optics()."
        )
    try:
        return _acquisition_optics(str(path), str(channel))
    except Exception as exc:
        raise ValueError(
            f"cannot derive the PSF for channel {channel!r} of {path}: "
            f"{type(exc).__name__}: {exc}. Deconvolution needs this channel's EMISSION "
            "wavelength; refusing to substitute another channel's optics, which is a different "
            "measurement and not a degraded one. Pass optics explicitly (decon_op(optics=...)) "
            "or install an override with set_optics()."
        ) from exc


def deconvolve(plane: np.ndarray, optics: Optional[OpticsParams] = None) -> np.ndarray:
    """Deconvolve one plane at the module defaults. *optics* ``None`` -> :func:`active_optics`.

    Kept for callers holding a bare plane. The registered ``decon`` is built by :func:`decon_op`,
    which is specialised to a channel by ``project_well`` before it ever sees a plane.
    """
    return deconvolve_plane(plane, optics, DEFAULT_ITERATIONS)


def decon_op(
    optics: Optional[OpticsParams] = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build a parameterised deconvolution **plane-op**, ready for ``add_projector``::

        add_projector("decon_sharp", decon_op(iterations=25))

    A SECOND NAME IS NO LONGER THE ONLY WAY to change a number. Since 2026-08-03 a registry entry
    can declare its own ``params`` and be RUN at a different value
    (``project_plate(..., operator_kwargs={"iterations": 25})``); see ``_engine.Param`` and
    ``_spots.segmentation_operator`` for the shape. ``decon`` has not been migrated -- its iteration
    count is chosen by eye against the QC panel, which is its own UI seam -- so the recipe above is
    still what this function serves today.

    The returned callable carries ``consumes = frozenset()`` (stamped by
    :func:`squidmip.plane_op`), so the registry infers the declaration and z survives.

    PER-CHANNEL OPTICS. With ``optics=None`` the returned callable also carries ``for_channel``,
    the declaration :func:`squidmip.projection.bind_channel` reads: ``project_well`` calls it once
    per channel with the acquisition it is reading, and runs the specialised operator it gets
    back. Given explicit *optics* the attribute is absent, because an explicit argument is an
    instruction and must not be silently re-derived per channel.
    """
    def _decon(p: np.ndarray) -> np.ndarray:
        return deconvolve_plane(p, optics, iterations)

    _decon.__name__ = f"decon(rl,iterations={iterations})"
    op = plane_op(_decon)
    if optics is None:
        op.for_channel = lambda path, channel: decon_op(
            optics_for_channel(path, channel), iterations)
    return op


def decon3d_op(
    optics: Optional[OpticsParams] = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build a TRUE 3-D deconvolution operator: a **z-reducer** (``consumes={"z"}``).

    Registered through the same ``add_projector`` seam with no engine edit — ``consumes`` has
    been part of that signature since IMA-210. Unlike the plane-op, this one collapses z (it
    deconvolves the volume and then projects), which is the honest shape for an operator that
    needs the whole stack to do its job.

    Like :func:`decon_op` it carries ``for_channel`` when *optics* is ``None``, so the 3-D kernel
    is the one for the channel actually being deconvolved. ``decon`` and ``decon3d`` are the same
    algorithm registered twice, and a per-channel PSF is not optional in one of them.
    """
    def _decon3d(planes: Iterable[np.ndarray]) -> np.ndarray:
        return deconvolve_stack(planes, optics, iterations)

    _decon3d.__name__ = f"decon3d(rl,iterations={iterations})"
    _decon3d.consumes = frozenset({"z"})
    if optics is None:
        _decon3d.for_channel = lambda path, channel: decon3d_op(
            optics_for_channel(path, channel), iterations)
    return _decon3d


# The whole registration. No engine edit — that is the IMA-210 seam working as designed, and the
# per-channel optics ride the same rails: a declaration on the callable (`for_channel`), read by
# project_well. Nothing here is named "decon" anywhere outside this line.
#
# `requires=("petakit",)` is the DECLARATION of what `_petakit()` imports lazily further down. It
# was undeclared, and that was a measured silent success rather than a tidiness point: on a stock
# `pip install .[gui]` petakit is absent (it is not in [project.dependencies]), the ImportError
# `_petakit` raises so carefully was recorded by `project_plate(on_error=...)` as a per-well skip,
# and a whole-plate deconvolution finished green having written nothing. Declared, the run now
# refuses BY NAME at bind time, before a well is read.
add_projector("decon", decon_op(), requires=("petakit",))
add_projector("decon3d", decon3d_op(), consumes=frozenset({"z"}), requires=("petakit",))
