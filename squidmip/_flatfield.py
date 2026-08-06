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

PER-CHANNEL, through ``for_channel`` (2026-08-06)
-------------------------------------------------
This section used to say the opposite — that a plane-op's ``Iterable[plane] -> plane`` shape
"never sees which CHANNEL the plane came from", so one profile applied to every channel of a
run and per-channel dispatch was somebody else's ticket. That stopped being true when ``decon``
hit the identical defect (all four channels deconvolved with the 488 line's PSF) and was fixed
with a THIRD declaration on the callable: ``for_channel(acquisition_path, channel) -> operator``,
read by :func:`squidmip.projection.bind_channel` and called by ``project_well`` ONCE PER CHANNEL
before the (t, z) loops. Flat-field was left behind on those rails for a day, and the comment
promising a limitation the seam no longer had is what hid it.

What it cost, measured on ``test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy`` (whose own
stored ``…_flatfield.npy`` is ``(4, 2084, 2084)`` with four genuinely different fields: c0
0.974–1.020, c1 0.645–1.102, c3 0.840–1.096) — one profile from ``from_npy(path)`` versus each
channel's own, region ``manual0`` fov 0 at the middle z::

                              WRONG(one profile)   RIGHT(per channel)   differing   mean|d|   max
    Fluorescence_405_nm_Ex        799.76               799.76             0.000%      0.00      0
    Fluorescence_488_nm_Ex       3128.63              3120.88            99.792%    155.68   1799
    Fluorescence_561_nm_Ex        792.53               792.54            88.684%      2.89     20
    Fluorescence_638_nm_Ex       2118.68              2129.37            99.578%     67.32   1307

405 is channel 0 of that file, so the channel anyone looks at first was bit-identical and the
three that were wrong were wrong by up to 1799 counts, silently.

So the active profile is a MAP KEYED BY CHANNEL NAME (:func:`set_profiles`,
:func:`active_profiles`), :func:`set_profile` REQUIRES the channel it was measured from — a gain
field with no channel attached is the defect above, made unsayable — and the registered operator
carries ``for_channel``, which hands back that channel's profile or refuses BY NAME, listing the
channels that do have one. The ``.npy`` is still ``(C, Y, X)``; :meth:`FlatfieldProfile.from_npy`
still takes a ``channel=`` INDEX into it, and :meth:`FlatfieldProfile.per_channel_from_npy` is
the one place that maps channel NAMES onto those planes.
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

        *channel* is an INDEX into the file's ``(C, Y, X)`` stack, not a channel name — the
        ``.npy`` carries no names. :meth:`per_channel_from_npy` is what turns an acquisition's
        channel names into those indices, and it is the only place that mapping is written.
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

    @classmethod
    def per_channel_from_npy(cls, path, names: Iterable[str]) -> dict[str, "FlatfieldProfile"]:
        """``{channel_name: profile}`` for *names*, in order, from one ``(C, Y, X)`` ``.npy``.

        THE ONE PLACE a channel NAME becomes a plane INDEX of the stored profile. It was written
        out longhand in ``_stitch.resolve_flatfield`` and nowhere else, so every other route to a
        stored profile — the GUI's "Load illumination profile", which is the one a user clicks —
        took plane 0 for all four channels and corrected 488, 561 and 638 with the 405 field.
        The mapping is positional because the file has no names in it: ``tilefusion``'s
        ``save_flatfield`` writes the acquisition's channels in the acquisition's own order, and
        both tools read ``reader.metadata["channels"]`` in that same order.

        A single-channel (``(Y, X)``) file gives every name the same field: there is one
        measurement in it and no per-channel claim to get wrong.
        """
        names = [str(n) for n in names]
        return {n: cls.from_npy(path, channel=i) for i, n in enumerate(names)}

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

    THE ESTIMATE IS RE-NORMALISED TO MEAN 1 HERE, and that is a fix, not a formality. BaSiC's
    low-rank/sparse split fixes the gain field only up to a scale, and it converges to mean 1
    only when it has enough decorrelated tiles: measured on ``sim_5d_2x2_t3``, four tiles give
    1.000000, three give 1.000000, TWO give 1.0030 and ONE gives 1.0053 — both outside
    :class:`FlatfieldProfile`'s 1e-3 tolerance, so the constructor raised and the whole run died
    with a message telling the caller to "divide by its mean first". Nobody could: the estimate
    is produced inside ``resolve_flatfield`` and never surfaces. Flat-field is ON by default, so
    that made every stitch and every Minerva export of a ONE- or TWO-FOV selection fail outright.

    Doing it here and not by loosening the tolerance is deliberate. The tolerance is what
    protects us from a FOREIGN profile (a user's ``.npy``) that would rescale every image while
    calling itself a correction; that check must stay exact. This is the one place we own an
    estimator's raw output, so this is the one place the documented remedy belongs. The divide
    changes overall brightness by whatever the estimate was off by — 0.3 % at two tiles — and the
    shape of the correction, which is the whole content of a gain field, is untouched.
    """
    from tilefusion.flatfield import estimate_flatfield_channel

    stack = np.asarray(planes if isinstance(planes, np.ndarray) else list(planes),
                       dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 1:
        raise ValueError(f"estimate_profile needs (n_tiles, Y, X); got shape {stack.shape}")
    ff, df = estimate_flatfield_channel(stack, use_darkfield=use_darkfield)
    ff = np.asarray(ff, dtype=np.float32)
    mean = float(ff.mean())
    if np.isfinite(mean) and mean > _MIN_GAIN:
        ff = ff / np.float32(mean)
    # A non-finite or ~zero mean is NOT normalised: there is nothing to divide by and pretending
    # otherwise would manufacture a field. FlatfieldProfile still refuses it, by name, which is
    # the right outcome for an estimate that did not converge at all.
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


# --- the ACTIVE profiles, ONE PER CHANNEL, for the registry entry -------------------------------
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
# WHAT CHANGED (2026-08-06): it is a MAP, ``{channel name: profile}``, and there is no way left to
# install a gain field without saying which channel measured it. It was ONE ``FlatfieldProfile``,
# and every consumer of it therefore corrected every channel with one channel's field — 99.8% of
# pixels wrong by up to 1799 counts on the 10x set (the table in the module docstring), and
# ``_stitch._selected_profiles`` broadcast the same single field over the stitcher's per-channel
# dict while the file on disk held four different ones. A dict cannot express that mistake.
#
# WHO READS IT (2026-08-04, still true). Not only the operator below: ``_stitch._selected_profiles``
# reads it too, so a profile chosen in the GUI's flat-field tab is also the one STITCHING corrects
# by — it used to have zero effect there, because ``resolve_flatfield`` went straight to the
# ``.npy`` and estimated its own. This global is the single owner of "the profiles the user chose";
# the per-call ``stitch_region(flatfield=...)`` argument still outranks it. Keep it that way: a
# second place that remembers a chosen profile is the defect that was removed then.
#
# The registered ``flatfield`` operator is selected by NAME (``project_plate(projector=...)``), so
# it cannot take a profile argument — and unlike decon's sigma or bgsub's radius, a flat-field has
# no sane default: an identity field would silently do nothing while the UI said "flat-field
# applied". So the profiles are set once (from a file or an estimate), the named operator's
# ``for_channel`` picks THIS channel's out of the map, and anything it cannot answer fails LOUD and
# actionable. Guarded by a lock because ``project_plate`` runs the operator on a thread pool.
_lock = threading.Lock()
_active: dict[str, FlatfieldProfile] = {}


def set_profile(profile: FlatfieldProfile, *, channel: str) -> None:
    """Install *profile* as the gain field for ONE *channel*, by name.

    ``channel`` is keyword-only and REQUIRED. Installing a gain field without saying which
    channel measured it is exactly how every channel came to be corrected by channel 0's field;
    the argument makes that unsayable rather than merely discouraged.
    """
    if not isinstance(profile, FlatfieldProfile):
        raise ValueError(f"set_profile needs a FlatfieldProfile, got {type(profile).__name__}")
    if not isinstance(channel, str) or not channel:
        raise ValueError(
            f"set_profile needs the CHANNEL NAME the profile was measured from, got {channel!r}. "
            "An illumination profile belongs to one channel of one optical path; applying it to "
            "the others is a different measurement, not a degraded one."
        )
    with _lock:
        _active[channel] = profile


def set_profiles(mapping: dict) -> None:
    """Install a whole ``{channel name: profile}`` map, REPLACING what was installed.

    The normal case: a stored ``.npy`` holds one field per channel
    (``FlatfieldProfile.per_channel_from_npy``), and installing them one at a time would leave a
    half-installed map visible to a run that started in between.
    """
    clean = {}
    for channel, profile in dict(mapping).items():
        if not isinstance(profile, FlatfieldProfile):
            raise ValueError(f"set_profiles needs FlatfieldProfile values; channel {channel!r} "
                             f"got {type(profile).__name__}")
        if not isinstance(channel, str) or not channel:
            raise ValueError(f"set_profiles needs channel NAMES as keys; got {channel!r}")
        clean[channel] = profile
    with _lock:
        _active.clear()
        _active.update(clean)


def active_profiles() -> dict:
    """A copy of the installed ``{channel name: profile}`` map; ``{}`` when nothing is installed.

    A copy, and no singular ``active_profile()`` beside it: the one-value query is what every
    caller collapsed the four fields through.
    """
    with _lock:
        return dict(_active)


def clear_profile() -> None:
    """Uninstall every channel's profile (the named operator goes back to failing loud)."""
    with _lock:
        _active.clear()


def _profile_for(channel: str) -> FlatfieldProfile:
    """THIS channel's installed profile, or a refusal that names the channel and what is installed.

    The refusal is the whole point of the per-channel map: a channel with no measured field must
    stop the run, not borrow another channel's. Two distinguishable messages, because the fixes
    differ — nothing installed at all is "load or estimate one"; something installed for other
    channels is "that file/estimate does not cover this one".
    """
    profiles = active_profiles()
    if not profiles:
        raise ValueError(
            f"no flat-field profile is loaded, so 'flatfield' has nothing to apply to channel "
            f"{channel!r}. Load an acquisition's stored profile with "
            "squidmip._flatfield.set_profiles(FlatfieldProfile.per_channel_from_npy(path, names)) "
            "or estimate one from tiles with estimate_profile(planes) and install it with "
            "set_profile(profile, channel=...). (A flat-field has no meaningful default: an "
            "identity field would silently do nothing.)"
        )
    if channel not in profiles:
        raise ValueError(
            f"no flat-field profile is installed for channel {channel!r}; the installed "
            f"profile(s) are for {sorted(profiles)}. Refusing to correct {channel!r} with another "
            "channel's gain field — that is a different measurement, not a degraded one. Install "
            f"one with set_profile(profile, channel={channel!r}), or load the acquisition's "
            "stored per-channel profile with set_profiles(per_channel_from_npy(path, names))."
        )
    return profiles[channel]


def _correct_with_active(plane: np.ndarray) -> np.ndarray:
    """The UNBOUND path: the operator run without ``bind_channel`` ever telling it a channel.

    ``project_well`` binds every channel before its (t, z) loops, so nothing in the engine, the
    GUI or the stitcher reaches this — it is what a direct ``_resolve_operator('flatfield').fn(…)``
    call gets, which is how the registry conformance suite runs every operator over one plane.

    With NOTHING installed it refuses, as it always did. With SEVERAL channels installed it
    refuses too, naming them: picking one of four measured fields for a plane whose channel was
    never stated is the defect this module was rebuilt to make unsayable. With exactly ONE
    installed it applies it, because then there is no choice being made — one profile is the whole
    of what the caller installed, and refusing it would only mean no caller could run the operator
    without going through the engine.
    """
    profiles = active_profiles()
    if len(profiles) == 1:
        return correct_flatfield(plane, next(iter(profiles.values())))
    if not profiles:
        raise ValueError(
            "no flat-field profile is loaded, so 'flatfield' has nothing to apply. Load an "
            "acquisition's stored profile with squidmip._flatfield.set_profiles("
            "FlatfieldProfile.per_channel_from_npy(path, names)) or estimate one from tiles with "
            "estimate_profile(planes) and install it with set_profile(profile, channel=...). "
            "(A flat-field has no meaningful default: an identity field would silently do "
            "nothing.)"
        )
    raise ValueError(
        f"'flatfield' was handed a plane without being told which channel it is, and profiles "
        f"for {sorted(profiles)} are installed. Refusing to pick one of them: correcting a "
        "channel with another channel's gain field is a different measurement, not a degraded "
        "one. Run it through project_well/project_plate (which specialises the operator per "
        "channel via for_channel), or call flatfield_op(profile) with the profile you mean."
    )


LAYER_KEY: str = "flatfield"
LAYER_LABEL: str = "flat-field correction"

# The whole registration. No engine edit — the IMA-210 seam working as designed, and the
# per-channel profile rides the same rails ``decon``'s per-channel PSF does: a declaration on the
# callable (``for_channel``), read by ``projection.bind_channel``, called by ``project_well`` once
# per channel before the (t, z) loops. Nothing branches on the string "flatfield" to get here.
_ACTIVE_OP = plane_op(_correct_with_active)
_ACTIVE_OP.corrects_illumination = True    # see CORRECTS_ILLUMINATION above
# The specialised operator comes from ``flatfield_op``, which stamps ``corrects_illumination``
# itself — so the double-apply guard in ``_stitch.stitch_region`` still sees a corrected-pixels
# declaration after binding, which is when it actually matters. The acquisition path is unused:
# unlike an emission wavelength, a gain field cannot be derived from metadata; it is measured, and
# what is installed is what there is.
_ACTIVE_OP.for_channel = lambda path, channel: flatfield_op(_profile_for(str(channel)))
# `requires=("tilefusion",)` states what every route to a profile actually imports: `from_npy`
# loads one through `tilefusion.flatfield.load_flatfield` and `estimate_profile` derives one
# through `estimate_flatfield_channel`. tilefusion is not in [project.dependencies], so on a stock
# install this operator could only ever raise from a lazy import one call deep — which
# `project_plate(on_error=...)` then filed as a per-well skip. Declared, it refuses by name first.
add_projector(LAYER_KEY, _ACTIVE_OP, requires=("tilefusion",))
