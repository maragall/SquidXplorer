"""Flat-field (illumination) correction as a plane-op, with per-channel profiles via for_channel.

The correction is per-pixel monotone, so it commutes with max/min/selection z-reductions
(NOT with a mean). Estimator and .npy format are reused from tilefusion.flatfield (BaSiC).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from squidxplorer._engine import add_operator
from squidxplorer.projection import cast_like, plane_op

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

    def to_npy(self, path) -> None:
        """Write this profile in the stitcher's format."""
        from tilefusion.flatfield import save_flatfield

        save_flatfield(Path(path), self.flatfield[None, ...],
                       None if self.darkfield is None else self.darkfield[None, ...])


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


def flatfield_op(profile: FlatfieldProfile) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build a plane-op bound to *profile*, ready for ``add_operator``."""
    if not isinstance(profile, FlatfieldProfile):
        raise ValueError(f"flatfield_op needs a FlatfieldProfile, got {type(profile).__name__}")

    def _flatfield(p: np.ndarray) -> np.ndarray:
        return correct_flatfield(p, profile)

    _flatfield.__name__ = f"flatfield{profile.shape}"
    op = plane_op(_flatfield)
    op.corrects_illumination = True
    return op


# ``corrects_illumination = True`` on a callable means: these pixels come out flat-fielded.
# ``_stitch.stitch_region`` reads it to refuse a double apply (the correction is not idempotent).
# An attribute on the callable, like ``consumes``, never a name comparison.
CORRECTS_ILLUMINATION = "corrects_illumination"


# The active profiles, one per channel. ``_stitch._selected_profiles`` reads this too; the
# per-call ``stitch_region(flatfield=...)`` argument still outranks it. Locked because
# The per-FOV loop runs the operator on a thread pool.
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


def _profile_for(channel: str) -> FlatfieldProfile:
    """This channel's installed profile, or a refusal naming the channel and what is installed."""
    profiles = active_profiles()
    if not profiles:
        raise ValueError(
            f"no flat-field profile is loaded, so 'flatfield' has nothing to apply to channel "
            f"{channel!r}. Load an acquisition's stored profile with "
            "squidxplorer._flatfield.set_profiles(FlatfieldProfile.per_channel_from_npy(path, names)) "
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
    """The unbound path: applies a lone installed profile, refuses on none or on several."""
    profiles = active_profiles()
    if len(profiles) == 1:
        return correct_flatfield(plane, next(iter(profiles.values())))
    if not profiles:
        raise ValueError(
            "no flat-field profile is loaded, so 'flatfield' has nothing to apply. Load an "
            "acquisition's stored profile with squidxplorer._flatfield.set_profiles("
            "FlatfieldProfile.per_channel_from_npy(path, names)) or estimate one from tiles with "
            "estimate_profile(planes) and install it with set_profile(profile, channel=...). "
            "(A flat-field has no meaningful default: an identity field would silently do "
            "nothing.)"
        )
    raise ValueError(
        f"'flatfield' was handed a plane without being told which channel it is, and profiles "
        f"for {sorted(profiles)} are installed. Refusing to pick one of them: correcting a "
        "channel with another channel's gain field is a different measurement, not a degraded "
        "one. Run it through project_well/run_plate (which specialises the operator per "
        "channel via for_channel), or call flatfield_op(profile) with the profile you mean."
    )


LAYER_KEY: str = "flatfield"
LAYER_LABEL: str = "flat-field correction"

_ACTIVE_OP = plane_op(_correct_with_active)
_ACTIVE_OP.corrects_illumination = True
# The acquisition path is unused: a gain field is measured, never derived from metadata.
_ACTIVE_OP.for_channel = lambda path, channel: flatfield_op(_profile_for(str(channel)))
add_operator(LAYER_KEY, _ACTIVE_OP, requires=("tilefusion",), extra="stitch")
