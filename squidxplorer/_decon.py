"""THE deconvolution operator (``decon``) on the petakit engine with a real vectorial PSF.

``decon`` is the volume solve: true 3-D RL over the whole z-stack, every plane kept. It is
the ONE deconvolution since 2026-08-24 (Julio: the 2-D per-plane variant was shelved — both
iterate per z-plane and only this one uses the complete PSF; on a 1-plane stack "3D decon
would still use a 2D PSF, since there is no more to draw from"). ONE code path,
:func:`deconvolve_stack`, whose PSF depth follows the stack depth — measured on n_z=1: the
volume solve equals the old per-plane solve (float32 max abs diff 0.00195, uint16 max 1
count, and the depth-1 3-D PSF's central plane renormalises exactly to the old in-focus
slice). Optics are derived per channel via ``for_channel``. The old registered name
``decon3d`` is refused BY NAME with a pointer here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Optional

import numpy as np

from squidxplorer import _decon_gpu
from squidxplorer._acquisition import load_acquisition_metadata, load_objective_na
from squidxplorer._channels import excitation_nm
from squidxplorer._engine import Param, add_operator
from squidxplorer.projection import cast_like

# RL is semi-convergent; the working point on this instrument, not a textbook default.
DEFAULT_ITERATIONS: int = 3
QC_START_ITERATIONS: int = 2

# Pinned: petakit's default "omw" returns an all-zero volume on this instrument's data.
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
    """Import petakit lazily and fail loud — never silently substitute another algorithm."""
    try:
        import petakit
    except ImportError as exc:                      # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from exc
    return petakit


def emission_um_for(channel) -> float:
    """The emission wavelength (um) a PSF is formed at, for one channel; raises for broadband channels."""
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
    """The acquisition optics a PSF is computed from; frozen and hashable so PSFs cache on it."""
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
        """Read the optics off a real acquisition; missing fields raise, never default."""
        meta = load_acquisition_metadata(path)           # acquisition.yaml, squidxplorer's own parse
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


# The 10x scope this tool ships against; the value for a bare array with no acquisition behind it.
# Operators running through project_well derive optics per channel instead.
DEFAULT_OPTICS = OpticsParams(na=0.3, wavelength_um=0.525, dxy_um=7.52 / 10.0, dz_um=1.5, nz=10)

_PSF_CACHE_SIZE = 32


@lru_cache(maxsize=_PSF_CACHE_SIZE)
def make_psf(optics: OpticsParams) -> np.ndarray:
    """The 3-D vectorial PSF for *optics*, ``(Z, Y, X)`` float32 normalised to sum 1."""
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


def _run(volume: np.ndarray, psf: np.ndarray, iterations: int, gpu: bool,
         snapshot_iters=None):
    """One call into RL: device selection, optional FFT-length padding, and an all-zero result guard.

    ``snapshot_iters`` (petakit's own contract) asks ONE solve to capture the estimate after
    each named iteration; the return is then ``{iter: volume}`` instead of one array. The
    QC sweep steps those captures back and forth for free — never a re-solve per count.
    """
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    snaps = sorted({int(i) for i in snapshot_iters}) if snapshot_iters else None
    if snaps is not None:
        iterations = max(int(iterations), snaps[-1])
    device = _decon_gpu.select_device(volume.shape, gpu=gpu, psf_shape=psf.shape)
    _decon_gpu.log_choice(volume.shape, gpu=gpu, psf_shape=psf.shape)
    if device is not None:
        out = _decon_gpu.rl(volume, psf, iterations, device, snapshot_iters=snaps)
    else:
        petakit = _petakit()
        widths = (_decon_gpu.pad_plan(volume.shape, psf.shape)
                  if _decon_gpu.cpu_padding_enabled() else (0, 0, 0))
        padded = _decon_gpu._wrap_pad(volume, widths)
        if snaps is not None:
            import inspect

            if "snapshot_iters" not in inspect.signature(petakit.engine.rl).parameters:
                raise RuntimeError(
                    "this petakit build's engine.rl takes no snapshot_iters, so a "
                    "per-iteration QC sweep cannot capture inside one solve. Update petakit "
                    "(the pinned SHA in pyproject carries it); refusing to fall back to one "
                    "full re-solve per iteration count without saying so.")
            out = petakit.engine.rl(
                np.ascontiguousarray(padded), psf,
                n_iter=iterations, gpu=gpu, snapshot_iters=snaps,
            )
        else:
            out = petakit.deconvolve(
                np.ascontiguousarray(padded), psf,
                method=METHOD, iterations=iterations, gpu=gpu,
            )
        if any(widths):
            core = tuple(slice(w, w + n) for w, n in zip(widths, volume.shape))
            out = ({k: v[core] for k, v in out.items()} if snaps is not None
                   else out[core])
    final = out[snaps[-1]] if snaps is not None else out
    if np.any(volume) and not np.any(final):
        raise RuntimeError(
            "petakit returned an all-zero result for a non-empty input. That is the failure "
            f"mode method='omw' shows on this instrument's geometry; this call used "
            f"method={METHOD!r} with a PSF of shape {psf.shape}. Refusing to hand back a black "
            "image that would look like a successful deconvolution."
        )
    return out


def deconvolve_stack(
    planes: Iterable[np.ndarray],
    optics: Optional[OpticsParams] = None,
    iterations: int = DEFAULT_ITERATIONS,
    *,
    gpu: bool = True,
    project: bool = True,
) -> np.ndarray:
    """True 3-D deconvolution of a whole z-stack with the full 3-D PSF.

    ``project=True`` (the historical default) returns the MIP; ``project=False`` returns the
    whole deconvolved stack — same shape as the input, the output the user examines plane by
    plane (the format contract: SquidXplorer writes in the format it ingests).
    """
    stack = planes if isinstance(planes, np.ndarray) else np.asarray(list(planes))
    if stack.ndim != 3 or stack.shape[0] < 1:
        raise ValueError(f"deconvolve_stack needs (Z, Y, X); got shape {stack.shape}")
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    dtype = stack.dtype
    if iterations == 0:
        return stack.max(axis=0) if project else stack

    optics = optics or active_optics()
    # Bind the PSF's axial extent to the actual stack depth.
    if optics.nz != stack.shape[0]:
        optics = OpticsParams(optics.na, optics.wavelength_um, optics.dxy_um,
                              optics.dz_um, int(stack.shape[0]), optics.ni)
    out = _run(stack, make_psf(optics), iterations, gpu)
    return cast_like(out.max(axis=0) if project else out, dtype)


# The override optics: an escape hatch checked first by optics_for_channel, empty by default.
# Locked because the per-FOV loop runs the operator on a thread pool.
_lock = threading.Lock()
_active: Optional[OpticsParams] = None

#: The standard immersion indices, value first, medium beside it (Julio, 2026-08-24: "the
#: typical NI values"). Air is the assumed default in the panel; a user picks the real medium.
IMMERSION_MEDIA: tuple[tuple[float, str], ...] = (
    (1.000, "air"),
    (1.333, "water"),
    (1.406, "silicone oil"),
    (1.473, "glycerol"),
    (1.515, "oil"),
)

# Session choices from the decon panel's optics row: ONE source of truth for both the QC
# preview solve and the real run, applied by optics_for_channel on acquisition-derived optics.
# Session-scoped on purpose (no prefs file); locked like the optics override above.
_session_ni: Optional[float] = None
_session_na: Optional[float] = None


def medium_for_ni(ni: float) -> str:
    """The medium name for an immersion index, when it is one of the standard values."""
    for value, medium in IMMERSION_MEDIA:
        if abs(float(ni) - value) < 5e-3:
            return medium
    return f"ni {float(ni):.3f}"


def set_session_ni(ni: Optional[float]) -> None:
    """Install the session's immersion index (None clears it; NA-based inference resumes)."""
    global _session_ni
    if ni is not None and (not np.isfinite(ni) or ni < 1.0):
        raise ValueError(f"an immersion index must be >= 1.0 (air), got {ni!r}")
    with _lock:
        _session_ni = None if ni is None else float(ni)


def session_ni() -> Optional[float]:
    with _lock:
        return _session_ni


def set_session_na(na: Optional[float]) -> None:
    """Install a session NA override (None clears it; the acquisition's recorded NA resumes)."""
    global _session_na
    if na is not None and (not np.isfinite(na) or na <= 0):
        raise ValueError(f"NA must be a positive finite number, got {na!r}")
    with _lock:
        _session_na = None if na is None else float(na)


def session_na() -> Optional[float]:
    with _lock:
        return _session_na


def apply_session_optics(optics: OpticsParams) -> OpticsParams:
    """The session's NI / NA choices applied to acquisition-derived *optics*.

    NA <= ni is physics, not preference: a lens cannot collect a cone wider than its medium
    carries, so NA 1.40 under air is refused BY NAME rather than solved into an impossible PSF.
    """
    with _lock:
        ni, na = _session_ni, _session_na
    if ni is None and na is None:
        return optics
    new_na = float(na) if na is not None else optics.na
    new_ni = float(ni) if ni is not None else optics.ni
    if new_ni is not None and new_na > new_ni + 1e-9:
        raise ValueError(
            f"NA {new_na:.2f} is impossible in {medium_for_ni(new_ni)} (ni {new_ni:.3f}) - "
            "pick the objective's actual immersion in the decon panel's optics row.")
    return OpticsParams(new_na, optics.wavelength_um, optics.dxy_um,
                        optics.dz_um, optics.nz, new_ni)


def set_optics(optics: OpticsParams) -> None:
    """Install an optics override, used for every channel until :func:`clear_optics`."""
    global _active
    if not isinstance(optics, OpticsParams):
        raise ValueError(f"set_optics needs an OpticsParams, got {type(optics).__name__}")
    with _lock:
        _active = optics


def optics_override() -> Optional[OpticsParams]:
    """The override installed by :func:`set_optics`, or ``None``."""
    with _lock:
        return _active


def active_optics() -> OpticsParams:
    """The override, or :data:`DEFAULT_OPTICS` when none has been set."""
    with _lock:
        return _active if _active is not None else DEFAULT_OPTICS


def clear_optics() -> None:
    """Remove the override; per-channel derivation resumes."""
    global _active
    with _lock:
        _active = None


@lru_cache(maxsize=64)
def _acquisition_optics(path: str, channel: str) -> OpticsParams:
    """``OpticsParams.from_acquisition`` memoised on ``(acquisition, channel)``."""
    return OpticsParams.from_acquisition(path, channel=channel)


def optics_for_channel(path, channel: str) -> OpticsParams:
    """The optics for one channel of one acquisition: override, then acquisition metadata, then a refusal."""
    override = optics_override()
    if override is not None:
        return override                     # the full escape hatch wins whole, session edits and all
    if path is None:
        raise ValueError(
            f"cannot derive the PSF for channel {channel!r}: the operator was not told which "
            "acquisition it is reading, so the emission wavelength is unknown. Pass optics "
            "explicitly or install an override with set_optics()."
        )
    try:
        optics = _acquisition_optics(str(path), str(channel))
    except Exception as exc:
        raise ValueError(
            f"cannot derive the PSF for channel {channel!r} of {path}: "
            f"{type(exc).__name__}: {exc}. Deconvolution needs this channel's EMISSION "
            "wavelength; refusing to substitute another channel's optics, which is a different "
            "measurement and not a degraded one. Pass optics explicitly "
            "or install an override with set_optics()."
        ) from exc
    # Session NI/NA (the decon panel's optics row) is applied AFTER the cached read, so the QC
    # preview and the real run agree on the PSF's medium — one source of truth. Its NA-vs-ni
    # refusal travels un-wrapped: it names its own way out.
    return apply_session_optics(optics)


def decon_op(
    optics: Optional[OpticsParams] = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build THE deconvolution operator: the volume solve, z-consuming, depth-keeping.

    The whole stack is deconvolved in one 3-D solve and EVERY plane comes back (Julio,
    2026-08-21: the output is the same size as the input — the user examines the planes).
    The PSF's depth follows the stack's, so an n_z=1 acquisition gets the 2-D in-focus solve
    as the volume solve's own degenerate case (measured equal; see the module docstring).
    ``keeps_depth`` on the callable is the declaration ``project_well`` and the acquisition
    writer honour.
    """
    def _decon(planes: Iterable[np.ndarray]) -> np.ndarray:
        return deconvolve_stack(planes, optics, iterations, project=False)

    _decon.__name__ = f"decon(rl,iterations={iterations})"
    _decon.consumes = frozenset({"z"})
    _decon.keeps_depth = True
    if optics is None:
        _decon.for_channel = lambda path, channel: decon_op(
            optics_for_channel(path, channel), iterations)
    return _decon


# `iterations` is DECLARED (a Param, so the factory rebinds per run): it is THE place the QC
# sweep's chosen count lands — DeconQCPanel.kwargs() feeds it through operator_kwargs_for into
# every run launched while the panel is open. Before 2026-08-24 nothing could change a run's
# iteration count at all; the QC tool existed to choose one and had nowhere to put the answer.
_ITERATIONS_PARAM = Param(
    "iterations", DEFAULT_ITERATIONS,
    "Richardson-Lucy iterations. RL is semi-convergent, so pick the count by eye in the decon "
    "panel's turbo x-z / y-z sweep; its 'use k iterations' button writes the choice here.")

add_operator("decon", decon_op, consumes=frozenset({"z"}), params=(_ITERATIONS_PARAM,),
             requires=("petakit",), extra="decon")
