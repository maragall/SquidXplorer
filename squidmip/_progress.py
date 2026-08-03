"""How far an operator run has got, and how much longer it has to go.

Julio, on a decon run over ONE region: "there's nothing on the child window that tells me how much
is left what's the progress, or that it is working. It only tells me that it worked after layers
populated, but how long is that?" That run took 433 seconds. For all 433 of them the region window
showed a busy dot and the log printed a memory line every 5 seconds, which is a memory printer
being read as a progress bar.

WHY THE COUNT WAS NOT ALREADY THERE, EVEN THOUGH A ``progress`` SIGNAL WAS
-------------------------------------------------------------------------
``_OperatorWorker.progress`` counts WELLS. That is the right denominator for a plate run over 1536
of them and completely useless for the case above, where the whole run is one well: it says ``0 of
1`` for seven minutes and then ``1 of 1``. The work inside that well is countable -- ``project_plate``
iterates FOVs and a 27-FOV region is 27 units -- so this module counts the ENGINE'S OWN UNIT rather
than the well, and the well counter is left exactly as it was for the plate's status header.

    per-FOV operator (decon, mip, bgsub, flatfield)   unit = FOV     total = FOVs in scope
    region operator  (stitch, coordinate)             unit = region  total = regions in scope

Both totals are known BEFORE the run starts, from ``metadata["fovs_per_region"]``, so the bar is
determinate from its first frame rather than growing a denominator as it goes. When the metadata
cannot supply one, :func:`unit_plan` returns ``None`` and the caller shows an INDETERMINATE bar --
the same rule ``squidmip._activity.Activity`` already follows, and for the same reason: a progress
bar that invents a denominator is a lie that gets believed.

TIME REMAINING IS A RUNNING MEAN OF COMPLETED-UNIT INTERVALS, AND THE FIRST UNIT IS EXCLUDED
-------------------------------------------------------------------------------------------
The estimate is ``remaining / rate`` where ``rate`` is measured from the FIRST completion onwards::

    rate = (done - 1) / (t_last_completion - t_first_completion)

Two deliberate choices in that one line.

*Wall-clock intervals, not per-unit durations.* The engine runs ``workers`` units at once, so a
unit's own compute time is ``workers`` times longer than the interval between arrivals. Dividing
the remaining work by a per-unit duration would over-state the wait by that factor. Arrival
intervals ARE throughput, which is what a "left" number has to be built from.

*The first unit is excluded.* The interval from the user's click to the first arrival pays the
reader's metadata warm, the thread pool's priming and the first cache-cold read. Extrapolating a
whole run from it is the classic dishonest ETA, so the clock for the rate starts at the first
completion and ``eta`` is ``None`` until a second one lands. Until then the caller shows the count
with no time, which is a weaker statement rather than a false one.

The estimate is still a rate measured over a small sample early on, so :func:`format_eta` rounds it
COARSELY -- "~4 min left", never "247 s left". Reporting a jumpy number to the second claims a
precision the sample does not have.

Pure Python, no Qt, like :mod:`squidmip._activity` and :mod:`squidmip._footprint`: the arithmetic is
the part that can be wrong, and it must be testable without a window and without an event loop
quietly swallowing an exception raised inside a slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

__all__ = ["ProgressReport", "RunProgress", "format_eta", "unit_plan"]

#: What a per-FOV operator and a region operator each count. Kept as constants rather than as bare
#: strings at the call sites so the two spellings cannot drift, and so ``CONTEXT.md``'s vocabulary
#: (FOV, region) is the one that reaches the screen.
FOV_UNIT = "FOV"
REGION_UNIT = "region"


def unit_plan(metadata: dict, regions, *, region_op: bool,
              n_fovs: Optional[int]) -> "tuple[Optional[int], str]":
    """``(total_units, unit_noun)`` for a run, or ``(None, noun)`` when the total is unknowable.

    ``regions=None`` means the whole acquisition; a list means exactly those regions (unknown names
    are dropped, exactly as the engine drops them, so the denominator matches the work).

    A REGION operator's unit is the region, so the total is just the scope's size. A per-FOV
    operator's unit is the FOV, so the total is what ``select_fovs`` would select: every FOV of each
    region for ``n_fovs=None`` (the viewer's path), otherwise ``n_fovs`` per region, clamped to what
    each region actually has so a ragged acquisition cannot produce a denominator larger than the
    work that will run.

    Returns ``None`` for the total -- never a guess -- when ``metadata`` carries no
    ``fovs_per_region``, which is the one case where the count genuinely is not known up front.
    """
    known = metadata.get("fovs_per_region") or {}
    if regions is None:
        scope = list(metadata.get("regions") or known.keys())
    else:
        seen = dict.fromkeys(regions)                      # de-dup, keep the caller's order
        scope = [r for r in seen if r in known] if known else list(seen)
    if region_op:
        return len(scope), REGION_UNIT
    if not known:
        return None, FOV_UNIT                              # no FOV table -> say so, do not invent
    total = 0
    for region in scope:
        available = len(known.get(region) or ())
        total += available if n_fovs is None else min(int(n_fovs), available)
    return total, FOV_UNIT


def format_eta(seconds: Optional[float]) -> str:
    """A COARSE "time left" phrase, or ``""`` when there is no honest estimate.

    Rounded up to the next bucket on purpose. The rate behind it is a small sample, and a number
    quoted to the second ("247 s left") claims a precision it does not have; rounding up also means
    the run tends to finish sooner than promised rather than later.
    """
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return ""
    if seconds < 10:
        return "a few seconds left"
    if seconds < 60:
        return f"~{int(math.ceil(seconds / 10.0)) * 10} s left"
    if seconds < 3600:
        return f"~{int(math.ceil(seconds / 60.0))} min left"
    hours, rest = divmod(int(math.ceil(seconds / 60.0)), 60)
    return f"~{hours} h {rest} min left" if rest else f"~{hours} h left"


@dataclass(frozen=True)
class ProgressReport:
    """One immutable snapshot of a run's progress, as it crosses a thread boundary.

    Frozen because it is emitted from a worker thread and read on the GUI thread. Handing the GUI a
    live, mutating tracker is how a label comes to show a count from one moment and a total from
    another; a snapshot cannot disagree with itself.
    """

    label: str
    done: int
    total: Optional[int]
    unit: str
    eta_seconds: Optional[float] = None

    @property
    def determinate(self) -> bool:
        """True when there is a real denominator to draw a bar against."""
        return self.total is not None and self.total > 0

    @property
    def remaining(self) -> Optional[int]:
        return None if self.total is None else max(0, self.total - self.done)

    @property
    def percent(self) -> Optional[int]:
        """0..100, or ``None`` when indeterminate. Never invents a denominator."""
        if not self.determinate:
            return None
        return int(min(100, max(0, round(100.0 * self.done / float(self.total)))))

    def sentence(self) -> str:
        """What to show a human: ``decon · 12 of 27 FOVs · ~4 min left``.

        An indeterminate run says what it is doing and how many units it has finished, and says
        nothing about how many there are -- because it does not know.
        """
        counted = self.done if self.total is None else self.total
        unit = self.unit if counted == 1 else f"{self.unit}s"
        if self.determinate:
            body = f"{self.done} of {self.total} {unit}"
        else:
            body = f"{self.done} {unit} so far"
        eta = format_eta(self.eta_seconds)
        parts = [p for p in (self.label, body, eta) if p]
        return " · ".join(parts)


class RunProgress:
    """The mutable side: counts completions and turns their arrival times into a rate.

    NOT thread-safe by itself, deliberately. Its one caller (``_OperatorWorker``) already holds a
    lock around the shared counter it updates in the same breath, and giving this a second lock
    would be a second thing to get right for no gain. The rule is stated rather than assumed: every
    ``tick`` happens under the caller's lock.
    """

    def __init__(self, label: str, total: Optional[int], unit: str = FOV_UNIT) -> None:
        self.label = str(label)
        self.total = None if total is None else max(0, int(total))
        self.unit = str(unit)
        self.done = 0
        self._first: Optional[float] = None      # arrival time of the FIRST completion
        self._last: Optional[float] = None       # arrival time of the most recent completion

    def tick(self, now: float) -> None:
        """One unit completed at *now* (a monotonic clock)."""
        self.done += 1
        if self._first is None:
            self._first = now
        self._last = now

    def eta(self) -> Optional[float]:
        """Seconds remaining, or ``None`` when no honest estimate exists yet.

        ``None`` on any of: an unknown total, fewer than two completions (the first one pays
        warm-up and is not extrapolated), a degenerate elapsed time, or nothing left to do.
        """
        if self.total is None or self.done < 2 or self._first is None or self._last is None:
            return None
        remaining = self.total - self.done
        if remaining <= 0:
            return 0.0
        elapsed = self._last - self._first
        if elapsed <= 0:
            return None                          # two arrivals in the same instant say nothing
        rate = (self.done - 1) / elapsed         # units per second, warm-up excluded
        if rate <= 0:
            return None
        return remaining / rate

    def report(self) -> ProgressReport:
        """An immutable snapshot to hand across the thread boundary."""
        return ProgressReport(label=self.label, done=self.done, total=self.total,
                              unit=self.unit, eta_seconds=self.eta())
