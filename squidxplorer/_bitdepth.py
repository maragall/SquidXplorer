"""How wide the contrast slider goes: the bit depth of the data, not of its container.

WHY THIS IS NOT ``np.iinfo``
----------------------------
Squid writes MONO12 and MONO16 into the SAME uint16 container -- ``reader.py`` says so at its
``_SUPPORTED_DTYPES`` and refuses everything else. So ``np.iinfo(uint16).max`` answers 65535 for
both, and on a 12-bit acquisition that is a slider whose useful travel is the bottom 1/16th.
``_contrast.dtype_range`` still answers the container question and is still right for uint8 and
for float; this module answers the DATA question for uint16, and delegates everything else to it.

WHY IT IS MEASURED AND NOT DECLARED
-----------------------------------
There is no bit depth on disk. ``acquisition.yaml`` has ``camera_settings.pixel_format``, and it
is ``null`` on every channel of every acquisition on this machine; ``acquisition parameters.json``
and ``configurations.xml`` carry nothing. Even when Squid is fixed to stamp it, MONO16 is what
both rigs report while the data is 12-bit shifted left -- so the field is an upper bound and a
cross-check, never the answer. The answer is the largest pixel actually seen.

WHY THE ESTIMATE ONLY EVER RISES
--------------------------------
Measured on the 14-bit reference set -- ONE acquisition, ONE channel, per-region maxima:

    C3 3437   C4 3504   C5 2737   E7 16380   F8 16380

Read C3 first and "the dataset is 12-bit" is the obvious conclusion; then E7 arrives at
16380 = 4095 x 4 and a 4095 ceiling clips it by 4x -- which is the exact bug this whole change
exists to remove. So there is no "infer once per dataset": the ceiling is monotone for the life
of the dataset. Choosing a ceiling ABOVE the truth costs a slider that travels further than it
needs to. Choosing one BELOW it destroys data on screen. Those are not comparable, and every
rule here is biased accordingly.

WHY NOT DETECT THE LEFT SHIFT
-----------------------------
Tempting and wrong. The 16-bit reference set is 12-bit shifted left by 4 (max 65520 =
4095 x 16), but the camera binned 2x2, so four of those samples were AVERAGED and the gcd of the
file is 4, not 16. A trailing-zero-bits heuristic reads that as a 2-bit shift, concludes 14-bit,
and clips 65520 down to 16383. ``test_bitdepth.py`` refuses the heuristic by name so it does not
get reinvented.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Optional

import numpy as np

from squidxplorer._contrast import dtype_range
from squidxplorer._logpane import get_logger

_log = get_logger("bitdepth")

#: The depths a uint16 buffer is allowed to be, smallest first. 12 is the FLOOR, not a
#: measurement: a dim acquisition whose brightest pixel is 400 is far more likely to be a 12-bit
#: sensor pointed at something faint than a genuine 10-bit one, and we do not trust our own
#: sampling enough to hand it a 1023 ceiling on that evidence. 16 is the default for the same
#: reason in the other direction -- anything above 16383 can only be the whole container.
_DEPTHS: tuple[int, ...] = (12, 14, 16)

#: Full-scale value for each of the above. Precomputed so the hot path is a tuple scan.
_FULL_SCALE: tuple[float, ...] = tuple(float((1 << d) - 1) for d in _DEPTHS)

#: Override, for when the measurement is wrong and you know better. Pins the ceiling and warns
#: on every observation that exceeds it -- see `DatasetDepth.observe`.
ENV_BIT_DEPTH = "SQUIDMIP_BIT_DEPTH"

_UINT16 = np.dtype("uint16")


def snap(observed_max: float) -> float:
    """The smallest standard full scale that CONTAINS ``observed_max``.

    Containment is the whole invariant: a ceiling >= every pixel cannot clip, and one below any
    pixel does. Floor-at-12 and default-16 are not branches here -- they fall out of `_DEPTHS`
    beginning at 12 and this returning the last entry when nothing fits.
    """
    for full in _FULL_SCALE:
        if observed_max <= full:
            return full
    return _FULL_SCALE[-1]


def _env_ceiling(env: Optional[dict] = None) -> Optional[float]:
    """``SQUIDMIP_BIT_DEPTH`` as a full-scale value, or None. Junk is ignored, loudly."""
    raw = (os.environ if env is None else env).get(ENV_BIT_DEPTH)
    if raw is None or not str(raw).strip():
        return None
    try:
        bits = int(str(raw).strip())
    except ValueError:
        _log.warning("%s=%r is not an integer -- ignoring it and measuring the data instead.",
                     ENV_BIT_DEPTH, raw)
        return None
    if not 1 <= bits <= 16:
        _log.warning("%s=%d is outside 1..16 -- ignoring it and measuring the data instead.",
                     ENV_BIT_DEPTH, bits)
        return None
    return float((1 << bits) - 1)


class DatasetDepth:
    """The ceiling for ONE open acquisition. Rises, never falls.

    ``uint16`` starts at the full container range rather than at nothing: until a pixel has been
    seen there is no evidence for a narrower ceiling, and a too-wide slider is a cosmetic
    complaint while a too-narrow one is lost data. The first observation is what earns the
    narrowing, and it lands before the first layer of a region is built (see `_mosaic_source`).
    """

    def __init__(self, dtype: Any = None) -> None:
        self._dtype = None if dtype is None else np.dtype(dtype)
        # An UNKNOWN dtype tracks as uint16. That is the container every Squid acquisition this
        # question applies to lands in, and a reader that did not report one must not silently
        # turn the measurement off.
        self._is_uint16 = self._dtype is None or self._dtype == _UINT16
        self._pinned = _env_ceiling() if self._is_uint16 else None
        self._observed: Optional[float] = None
        self._ceiling = self._pinned if self._pinned is not None else self._base_ceiling()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[str, float, float], None]] = []
        self._per_channel: dict[str, float] = {}
        if self._pinned is not None:
            _log.info("%s pins the contrast slider to (0, %.0f); the data will not widen it.",
                      ENV_BIT_DEPTH, self._pinned)

    def _base_ceiling(self) -> float:
        """The answer before any pixel has been seen."""
        if self._is_uint16:
            return _FULL_SCALE[-1]
        return dtype_range(self._dtype)[1]

    @property
    def tracks_uint16(self) -> bool:
        """True when this dataset's ceiling describes uint16 data and may be narrowed."""
        return self._is_uint16

    @property
    def ceiling(self) -> float:
        return self._ceiling

    @property
    def range(self) -> tuple[float, float]:
        return 0.0, self._ceiling

    @property
    def observed(self) -> Optional[float]:
        """The largest pixel seen so far, for logs and tests. None before the first observation."""
        return self._observed

    @property
    def settled(self) -> bool:
        """True when nothing can move the ceiling again -- the lock-free fast path.

        The unmeasured opening state is NOT settled even though its ceiling is already the
        container maximum: that 65535 is an absence of evidence, and the first observation is
        what turns it into an answer (usually a narrower one).
        """
        if self._pinned is not None or not self._is_uint16:
            return True
        return self._observed is not None and self._ceiling >= _FULL_SCALE[-1]

    def observe(self, value: Any) -> bool:
        """Feed one measured maximum. Returns True when the ceiling MOVED.

        The FIRST observation may narrow -- that is the whole point, and it is the only
        narrowing in the life of a dataset. Every observation after it can only widen, because
        `_observed` is monotone and `snap` is monotone in it.

        Callers are worker threads, so this takes a lock -- but `settled` is checked first and
        without one, which is what keeps this off the profile once a dataset has proved itself
        16-bit (the common case, and the case where this is called most often).
        """
        if self._pinned is not None:
            try:
                if value is not None and float(value) > self._pinned:
                    _log.warning(
                        "%s pins the slider at %.0f but this data reaches %.0f -- %.0f and above "
                        "are being clipped ON PURPOSE because you asked. Unset it to measure.",
                        ENV_BIT_DEPTH, self._pinned, float(value), self._pinned)
            except (TypeError, ValueError):
                pass
            return False
        if not self._is_uint16 or value is None:
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(v):
            return False
        with self._lock:
            if self._observed is not None and v <= self._observed:
                return False
            self._observed = v if self._observed is None else max(self._observed, v)
            new = snap(self._observed)
            if new == self._ceiling:
                return False
            old, self._ceiling = self._ceiling, new
        # An automatic adjustment the sliders already show; it can fire several times as
        # frames decode (log diet, 2026-08-25).
        _log.debug("contrast slider now (0, %.0f): this dataset reaches %.0f.", new, v)
        return old != new

    def observe_array(self, data: Any, channel: Optional[str] = None) -> bool:
        """`observe` the maximum of an array. Cheap: ~0.06 ms on a 2048x2048 uint16 frame.

        NaN-safe and empty-safe, because a bad FOV must not be able to move the ceiling.
        With *channel*, the maximum is also booked PER CHANNEL (Julio, 2026-08-25: "look how
        close to each other are the contrast limits" - a dim channel's slider spanned the
        saturated channel's 16 bits), and a rise of that channel's own ceiling is published
        as ``callback(channel, lo, hi)``.
        """
        if data is None:
            return False
        try:
            arr = np.asanyarray(data)
            if arr.size == 0:
                return False
            peak = float(np.nanmax(arr))
        except (TypeError, ValueError):
            return False
        moved = self.observe(peak) if not self.settled else False
        if channel is not None:
            with self._lock:
                old = self._per_channel.get(str(channel))
                rose = old is None or peak > old
                if rose:
                    self._per_channel[str(channel)] = peak
                    hi = channel_ceiling(peak)
                    callbacks = list(self._callbacks)
            if rose:
                for cb in callbacks:
                    try:
                        cb(str(channel), 0.0, hi)
                    except Exception:            # noqa: BLE001 - a bad subscriber must not
                        _log.exception("a depth subscriber raised; %s's ceiling stands at %.0f.",
                                       channel, hi)
        return moved

    def channel_range(self, channel: Optional[str]) -> Optional[tuple[float, float]]:
        """``(0, ceiling)`` for one channel from ITS OWN observed maximum, or None before
        that channel was seen (the dataset ceiling is the caller's fallback)."""
        with self._lock:
            peak = self._per_channel.get(str(channel)) if channel is not None else None
        return None if peak is None else (0.0, channel_ceiling(peak))

    def on_change(self, callback: Callable[[str, float, float], None]) -> None:
        """Subscribe to a CHANNEL's ceiling rises. Fired as ``callback(channel, lo, hi)`` ON
        THE OBSERVING THREAD.

        That thread is a worker, so a Qt subscriber must do nothing here but emit a signal --
        the queued connection is what marshals it to the GUI thread.
        """
        with self._lock:
            self._callbacks.append(callback)


#: The one open acquisition ("one app, one dataset"), re-armed in `ViewerManager.set_dataset`.
_current = DatasetDepth(None)


def depth() -> DatasetDepth:
    """The current dataset's depth object."""
    return _current


def new_dataset(dtype: Any = None) -> DatasetDepth:
    """Forget the last acquisition's ceiling and start measuring a new one."""
    global _current
    _current = DatasetDepth(dtype)
    return _current


#: A channel whose maximum reaches this fraction of a full-scale ceiling gets that ceiling.
_REACHES = 0.95
#: Headroom above a channel's observed maximum otherwise.
_HEADROOM = 1.05


def channel_ceiling(observed_max: float) -> float:
    """A channel's own slider top: the full-scale ceiling it actually reaches, else its
    observed maximum with 5% headroom."""
    peak = float(observed_max)
    for fs in _FULL_SCALE:
        if peak >= _REACHES * fs and peak <= fs:
            return float(fs)
    return max(1.0, peak * _HEADROOM)


def range_for(dtype: Any, channel: Optional[str] = None) -> tuple[float, float]:
    """``(lo, hi)`` for ``contrast_limits_range`` on a layer holding ``dtype``. THE call-site API.

    The gate is the LAYER's own dtype, not the dataset's: an operator that returns float32
    or uint8 must keep its own range, or a "bit depth" is being applied to data that has none.
    """
    if dtype is None:
        return _current.range
    dt = np.dtype(dtype)
    if dt != _UINT16:
        return dtype_range(dt)
    # A uint16 LAYER inside a dataset whose own dtype is not uint16 (a uint8 acquisition with a
    # uint16 operator result, say) gets the container range. `_current` is measuring something
    # else entirely and its ceiling says nothing about this layer.
    if not _current.tracks_uint16:
        return dtype_range(dt)
    own = _current.channel_range(channel)          # this channel's own ceiling, when observed
    return own if own is not None else _current.range
