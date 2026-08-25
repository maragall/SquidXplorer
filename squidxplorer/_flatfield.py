"""Flat-field (illumination) machinery: the profile record, estimate, parse and correction.

The correction is per-pixel monotone, so it commutes with max/min/selection z-reductions
(NOT with a mean). Estimator and .npy format are reused from tilefusion.flatfield (BaSiC).
Stitch is the consumer (its read path corrects tiles; the GUI's profile loader installs the
selection it reads); the standalone ``flatfield`` OPERATOR was shelved 2026-08-24.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from squidxplorer.projection import cast_like

# A gain below this is treated as 1.0 rather than dividing a dead pixel to the dtype ceiling.
_MIN_GAIN = 1e-6


@dataclass(frozen=True)
class FlatfieldProfile:
    """A multiplicative gain field (mean 1.0, enforced), optionally an additive darkfield."""
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
                "mean is not 1 rescales the whole image - a brightness change masquerading as "
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
        """Load one channel (an index, not a name) of a stitcher-format ``.npy`` profile."""
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
            ff = ff / mean          # tolerate a profile stored un-normalised
        return cls(ff, None if df is None else np.asarray(df, dtype=np.float32))

    @classmethod
    def per_channel_from_npy(cls, path, names: Iterable[str]) -> dict[str, "FlatfieldProfile"]:
        """``{channel_name: profile}`` for *names*, in order, from one ``(C, Y, X)`` ``.npy``.

        The one place a channel NAME becomes a plane INDEX of the stored file; the mapping is
        positional because the file carries no names.
        """
        names = [str(n) for n in names]
        return {n: cls.from_npy(path, channel=i) for i, n in enumerate(names)}


def estimate_profile(planes, *, use_darkfield: bool = False) -> FlatfieldProfile:
    """Estimate a profile from ``(n_tiles, Y, X)`` tiles with the stitcher's BaSiC estimator.

    The estimate is re-normalised to mean 1 here: BaSiC fixes the gain field only up to a scale,
    and a one- or two-tile estimate lands outside FlatfieldProfile's tolerance.
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
    # A non-finite or ~zero mean is not normalised; FlatfieldProfile refuses it by name.
    return FlatfieldProfile(ff, df)


def correct_flatfield(plane: np.ndarray, profile: FlatfieldProfile) -> np.ndarray:
    """Apply ``(raw - darkfield) / flatfield`` to one plane; same shape and dtype, input untouched."""
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
    return cast_like(values, plane.dtype)


# ``corrects_illumination = True`` on a callable means: these pixels come out flat-fielded.
# ``_stitch.stitch_region`` reads it to refuse a double apply (the correction is not idempotent).
# An attribute on the callable, like ``consumes``, never a name comparison.


# The active profiles, one per channel — what ``_stitch._selected_profiles`` reads and the
# GUI's "Load illumination profile" / estimate-from-plate flows install; the per-call
# ``stitch_region(flatfield=...)`` argument still outranks it. Locked because stitch's
# read path applies profiles on a thread pool.
_lock = threading.Lock()
_active: dict[str, FlatfieldProfile] = {}


def set_profile(profile: FlatfieldProfile, *, channel: str) -> None:
    """Install *profile* as the gain field for one *channel*, by name (required)."""
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
    """Install a whole ``{channel name: profile}`` map, replacing what was installed."""
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
    """A copy of the installed ``{channel name: profile}`` map; ``{}`` when nothing is installed."""
    with _lock:
        return dict(_active)


def clear_profile() -> None:
    """Uninstall every channel's profile."""
    with _lock:
        _active.clear()


# (The registered ``flatfield`` OPERATOR — ``_ACTIVE_OP``, ``_correct_with_active``,
# ``_profile_for``, ``flatfield_op`` — was shelved 2026-08-24. This module keeps the
# MACHINERY stitch rides: the profile record, the BaSiC estimate, the per-channel .npy
# parse, the correction arithmetic and the installed-profile store above. Git history
# reinstates the operator.)
