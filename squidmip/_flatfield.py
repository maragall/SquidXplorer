"""Flat-field (illumination) correction as a PLANE-OP (IMA-225).

One ``add_projector`` call, zero engine edits: ``consumes=frozenset()`` via
:func:`squidmip.plane_op`, so ``project_well``'s existing loop hands this ONE plane at a time
and z survives at full depth (IMA-210).

THE ALGEBRAIC SHORTCUT: flat-field commutes with the MIP
--------------------------------------------------------
Correction is applied INDEPENDENTLY PER PIXEL by

    f(v) = clip(round((v - darkfield) / flatfield), dtype_min, dtype_max)

and every step of that — subtract a constant, divide by a positive constant, round, clip — is
**monotone non-decreasing** in ``v``. For any monotone non-decreasing ``f``:

    max(f(a), f(b)) == f(max(a, b))

so flat-fielding every plane and THEN taking the maximum-intensity projection is **bit-for-bit
identical** to taking the MIP first and flat-fielding the single result — at 1/Nz the cost.
Integer rounding and clipping do not break it, because both are themselves monotone; that is
the part people assume and it is measured here instead (``tests/test_flatfield.py`` pins
``np.array_equal`` on real 10x data, on saturating data, and on data that clips at zero).

This holds for **any** z-reduction that is a max, a min, or a plane SELECTION (IMA-183's
``reference``): each picks a value rather than blending, so a per-pixel monotone map commutes
with it. It does NOT hold for a MEAN projector — an average of rounded values is not the
rounded average — so if a mean/sum reduction is ever added, this shortcut must not be applied
to it. That is the whole condition, stated once.

Prior art: reused, not reimplemented
------------------------------------
* **Julio's ``tilefusion.flatfield`` (the stitcher, /Users/.../projects/stitcher)** — the
  estimator AND the on-disk profile format come from there verbatim: ``estimate_flatfield_channel``
  (a pure numpy/scipy port of BaSiC's inexact-augmented-Lagrangian low-rank + sparse solver,
  Peng et al. Nat. Commun. 2017) and ``load_flatfield``/``save_flatfield`` (the ``.npy``
  dict-with-``flatfield``/``darkfield``-keys format, including its numpy-1.x pickle compat
  shim). Imported LAZILY inside the functions, exactly as ``_stitch.py`` does it, because
  ``import tilefusion`` runs a heavy package ``__init__``.
* **BaSiC / BaSiCPy** — the algorithm, reached through the stitcher's port rather than the
  package: BaSiCPy exists only to provide a jax/torch GPU backend for the same solver, and the
  numpy port is already on this machine and already validated against Julio's own data.
* **CellProfiler ``CorrectIlluminationApply``** — TAKEN: the divide-vs-subtract distinction.
  Flat-field is the MULTIPLICATIVE correction (sensor/objective gain); the ADDITIVE haze is
  IMA-224's background subtraction, a separate operator. Also taken: apply the additive
  darkfield BEFORE the multiplicative divide, which is the order the physics has
  (``(raw - dark) / gain``) and the order that leaves no residual gradient.

The one seam limitation, stated loud
------------------------------------
A plane-op's callable shape is ``Iterable[plane] -> plane``: it never sees which CHANNEL the
plane came from. Illumination profiles are per-channel in reality (and ``tilefusion`` stores
them as ``(C, Y, X)``), so :meth:`FlatfieldProfile.from_npy` takes an explicit ``channel=``
index and the active profile applies to every channel of a run. Per-channel dispatch needs the
operator signature to carry channel identity, which is an IMA-210 change, not this ticket.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from squidmip._engine import add_projector
from squidmip.projection import plane_op

# A gain below this is treated as 1.0 rather than dividing by ~0 and exploding a dead pixel to
# the dtype ceiling. Same threshold as tilefusion.flatfield.apply_flatfield, deliberately.
_MIN_GAIN = 1e-6


@dataclass(frozen=True)
class FlatfieldProfile:
    """An illumination profile: a multiplicative gain field, optionally an additive pedestal.

    flatfield:
        ``(Y, X)`` float32 gain, **normalised to mean 1.0**. The normalisation is enforced, not
        assumed: a field with mean 0.5 would double the brightness of every image while calling
        itself a correction, and nothing downstream would flag it.
    darkfield:
        ``(Y, X)`` additive pedestal (dark current / stray light offset), or ``None``. Applied
        BEFORE the gain divide — ``(raw - dark) / gain`` — which is the order the physics has.
    """
    flatfield: np.ndarray
    darkfield: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        ff = np.asarray(self.flatfield, dtype=np.float32)
        if ff.ndim != 2:
            raise ValueError(f"flatfield must be a 2-D (Y, X) field; got shape {ff.shape}")
        mean = float(ff.mean())
        if not np.isfinite(mean) or abs(mean - 1.0) > 1e-3:
            raise ValueError(
                f"flatfield must be normalised to mean 1.0 (got {mean:.4f}); a profile whose "
                "mean is not 1 rescales the whole image — a brightness change masquerading as "
                "a correction. Divide by its mean first."
            )
        object.__setattr__(self, "flatfield", ff)
        if self.darkfield is not None:
            df = np.asarray(self.darkfield, dtype=np.float32)
            if df.shape != ff.shape:
                raise ValueError(f"darkfield shape {df.shape} != flatfield shape {ff.shape}")
            object.__setattr__(self, "darkfield", df)

    @property
    def shape(self) -> tuple[int, int]:
        return self.flatfield.shape

    @classmethod
    def from_npy(cls, path, channel: int = 0) -> "FlatfieldProfile":
        """Load a profile written by the stitcher's ``save_flatfield`` (a ``.npy`` holding a
        dict with ``(C, Y, X)`` ``flatfield``/``darkfield``). Reused verbatim — including its
        numpy-1.x pickle compatibility shim, which real Squid-era profiles need.

        *channel* selects the plane of a multi-channel profile; see the module docstring's note
        on why the plane-op seam cannot pick it automatically.
        """
        from tilefusion.flatfield import load_flatfield   # lazy: heavy package __init__

        ff, df = load_flatfield(Path(path))
        ff = np.asarray(ff, dtype=np.float32)
        if ff.ndim == 3:
            if not 0 <= channel < ff.shape[0]:
                raise ValueError(
                    f"channel {channel} out of range for a profile with {ff.shape[0]} channel(s)"
                )
            ff = ff[channel]
            df = None if df is None else np.asarray(df, dtype=np.float32)[channel]
        mean = float(ff.mean())
        if abs(mean) > _MIN_GAIN:
            ff = ff / mean          # tolerate a profile stored un-normalised; never silently scale
        return cls(ff, None if df is None else np.asarray(df, dtype=np.float32))

    def to_npy(self, path) -> None:
        """Write this profile in the stitcher's format, so the two tools read each other's files."""
        from tilefusion.flatfield import save_flatfield

        save_flatfield(Path(path), self.flatfield[None, ...],
                       None if self.darkfield is None else self.darkfield[None, ...])


def estimate_profile(planes, *, use_darkfield: bool = False) -> FlatfieldProfile:
    """Estimate a profile from a stack of tiles with the stitcher's BaSiC estimator.

    *planes* is ``(n_tiles, Y, X)`` (or any iterable of equal-shape planes) — the more tiles and
    the more decorrelated their content, the better the low-rank/sparse split. Not
    reimplemented: this is ``tilefusion.flatfield.estimate_flatfield_channel``.
    """
    from tilefusion.flatfield import estimate_flatfield_channel

    stack = np.asarray(planes if isinstance(planes, np.ndarray) else list(planes),
                       dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 1:
        raise ValueError(f"estimate_profile needs (n_tiles, Y, X); got shape {stack.shape}")
    ff, df = estimate_flatfield_channel(stack, use_darkfield=use_darkfield)
    return FlatfieldProfile(ff, df)


def correct_flatfield(plane: np.ndarray, profile: FlatfieldProfile) -> np.ndarray:
    """Apply ``(raw - darkfield) / flatfield`` to ONE plane. Same shape and dtype; input untouched.

    Every step is monotone non-decreasing in the input value, which is what makes this commute
    with the MIP (see the module docstring). Integer results are ROUNDED and CLIPPED — a dim
    corner divided up past the dtype ceiling must saturate, never wrap to black.
    """
    if plane.ndim != 2:
        raise ValueError(f"correct_flatfield takes ONE 2-D plane; got shape {plane.shape}")
    if plane.shape != profile.shape:
        raise ValueError(
            f"plane shape {plane.shape} does not match flat-field profile shape {profile.shape}"
        )
    gain = np.where(profile.flatfield > _MIN_GAIN, profile.flatfield, np.float32(1.0))
    values = plane.astype(np.float32, copy=True)
    if profile.darkfield is not None:
        values -= profile.darkfield
    values /= gain
    return _cast_like(values, plane.dtype)


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast to the acquisition dtype, ROUNDING and clipping integers (never truncate, never wrap).

    Both operations are monotone non-decreasing, which is exactly why they do not break the
    commutation with the MIP — the property people assume breaks here and it does not.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    return values.astype(dtype, copy=False)


def flatfield_op(profile: FlatfieldProfile) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build a **plane-op** bound to *profile*, ready for ``add_projector``::

        add_projector("flatfield_638", flatfield_op(FlatfieldProfile.from_npy(p, channel=1)))
    """
    if not isinstance(profile, FlatfieldProfile):
        raise ValueError(f"flatfield_op needs a FlatfieldProfile, got {type(profile).__name__}")

    def _flatfield(p: np.ndarray) -> np.ndarray:
        return correct_flatfield(p, profile)

    _flatfield.__name__ = f"flatfield{profile.shape}"
    op = plane_op(_flatfield)
    op.corrects_illumination = True     # see CORRECTS_ILLUMINATION below
    return op


# --- the DOUBLE-APPLY declaration --------------------------------------------------------------
#
# ``corrects_illumination = True`` on an operator callable means: THESE PIXELS COME OUT
# FLAT-FIELDED. It exists because there are now two places that can apply the correction and only
# one of them may run per pass:
#
#   * the READ path — ``_stitch._FlatfieldReader``, which is where TileFusion applies it and where
#     it has to be applied for registration to see corrected strips. ON by default;
#   * this OPERATOR, which corrects the pixels a projector emits.
#
# Until IMA-277 they could not meet: stitching refused every plane-op outright, so the ``flatfield``
# projector could not reach a stitch. Removing that refusal makes the combination reachable, and
# the correction is NOT idempotent — measured on the 10x tissue set, correcting twice changes 88.6%
# of pixels, by up to 23 counts. Silently. ``_stitch.stitch_region`` reads this attribute and
# refuses the combination; see tests/test_stitch_zplanes.py.
#
# An ATTRIBUTE ON THE CALLABLE, in the same style as ``consumes`` (``projection.plane_op``) and
# ``select_index``, rather than a name comparison: this module already registers the correction
# under one name and hands out others through ``flatfield_op()``, and a guard keyed on the string
# "flatfield" would miss every one of those. ``_engine.Operator`` reads it off the callable exactly
# as it reads ``consumes``.
CORRECTS_ILLUMINATION = "corrects_illumination"


# --- the ACTIVE profile, for the registry entry ------------------------------------------------
#
# STILL TRUE, BUT NO LONGER FORCED (2026-08-03). A registry entry can now declare its own
# ``params`` and be run with them (``_engine.Param`` / ``Operator.bind``), so "selected by name,
# therefore cannot take an argument" has stopped being a property of the table. This global is what
# remains of that limit, and it is the reason for the only two name comparisons left in the package
# (``_viewer.run_operator`` and ``_benchmark._prepare``, both spelled ``== "flatfield"``, both
# recorded in tests/test_operator_declaration.py::KNOWN_NAME_BRANCHES). Migrating the profile onto a
# declared parameter is what deletes them, and it is deliberately NOT done here: it changes how the
# GUI's auto-estimate worker hands its result to a run, which is a separate change.
#
# WHO READS IT (2026-08-04). No longer only the plane-op below. ``_stitch._selected_profiles``
# reads it too, so a profile chosen in the GUI's flat-field tab is also the one STITCHING corrects
# by — it used to have zero effect there, because ``resolve_flatfield`` went straight to the
# ``.npy`` and estimated its own. This global is now the single owner of "the profile the user
# chose"; the per-call ``stitch_region(flatfield=...)`` argument still outranks it. Keep it that
# way: a second place that remembers a chosen profile is the defect that was just removed.
#
# The registered ``flatfield`` operator is selected by NAME (``project_plate(projector=...)``),
# so it cannot take a profile argument — and unlike decon's sigma or bgsub's radius, a flat-field
# has no sane default: an identity field would silently do nothing while the UI said "flat-field
# applied". So the profile is set once (from a file or an estimate) and the named operator reads
# it, failing LOUD and actionable when it is unset. Guarded by a lock because ``project_plate``
# runs the operator on a thread pool.
_lock = threading.Lock()
_active: Optional[FlatfieldProfile] = None


def set_profile(profile: FlatfieldProfile) -> None:
    """Install the profile the registered ``flatfield`` operator will use."""
    global _active
    if not isinstance(profile, FlatfieldProfile):
        raise ValueError(f"set_profile needs a FlatfieldProfile, got {type(profile).__name__}")
    with _lock:
        _active = profile


def active_profile() -> Optional[FlatfieldProfile]:
    """The installed profile, or ``None``."""
    with _lock:
        return _active


def clear_profile() -> None:
    """Uninstall the profile (the named operator goes back to failing loud)."""
    global _active
    with _lock:
        _active = None


def _correct_with_active(plane: np.ndarray) -> np.ndarray:
    profile = active_profile()
    if profile is None:
        raise ValueError(
            "no flat-field profile is loaded, so 'flatfield' has nothing to apply. Load one "
            "with squidmip._flatfield.set_profile(FlatfieldProfile.from_npy(path)) or estimate "
            "one from tiles with estimate_profile(planes). (A flat-field has no meaningful "
            "default: an identity field would silently do nothing.)"
        )
    return correct_flatfield(plane, profile)


LAYER_KEY: str = "flatfield"
LAYER_LABEL: str = "flat-field correction"

# The whole registration. No engine edit — the IMA-210 seam working as designed.
_ACTIVE_OP = plane_op(_correct_with_active)
_ACTIVE_OP.corrects_illumination = True    # see CORRECTS_ILLUMINATION above
# `requires=("tilefusion",)` states what every route to a profile actually imports: `from_npy`
# loads one through `tilefusion.flatfield.load_flatfield` and `estimate_profile` derives one
# through `estimate_flatfield_channel`. tilefusion is not in [project.dependencies], so on a stock
# install this operator could only ever raise from a lazy import one call deep — which
# `project_plate(on_error=...)` then filed as a per-well skip. Declared, it refuses by name first.
add_projector(LAYER_KEY, _ACTIVE_OP, requires=("tilefusion",))
