"""How far an operator run has got, and how much longer it has to go."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

__all__ = ["PREVIEW_LABEL", "ProgressReport", "RunProgress", "format_eta", "unit_plan"]

FOV_UNIT = "FOV"
REGION_UNIT = "region"

PREVIEW_LABEL = "preview"


def unit_plan(metadata: dict, regions, *, region_op: bool,
              n_fovs: Optional[int]) -> "tuple[Optional[int], str]":
    """(total_units, unit_noun) for a run, or (None, noun) when the total is unknowable.

    A region operator's unit is the region; a per-FOV operator's unit is the FOV, clamped to
    what each region actually has. Returns None for the total, never a guess, when metadata
    carries no fovs_per_region.
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
    """A coarse "time left" phrase, or "" when there is no honest estimate."""
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
    """One immutable snapshot of a run's progress, as it crosses a thread boundary."""

    label: str
    done: int
    total: Optional[int]
    unit: str
    eta_seconds: Optional[float] = None

    @property
    def determinate(self) -> bool:
        return self.total is not None and self.total > 0

    @property
    def remaining(self) -> Optional[int]:
        return None if self.total is None else max(0, self.total - self.done)

    @property
    def percent(self) -> Optional[int]:
        if not self.determinate:
            return None
        return int(min(100, max(0, round(100.0 * self.done / float(self.total)))))

    def sentence(self) -> str:
        """"decon · 12 of 27 FOVs · ~4 min left"."""
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
    """Mutable side: counts completions and turns arrival times into a rate.

    Not thread-safe by itself: the caller (_OperatorWorker) already holds a lock around the
    shared counter, and every tick happens under that lock.
    """

    def __init__(self, label: str, total: Optional[int], unit: str = FOV_UNIT) -> None:
        self.label = str(label)
        self.total = None if total is None else max(0, int(total))
        self.unit = str(unit)
        self.done = 0
        self._first: Optional[float] = None      # arrival time of the first completion
        self._last: Optional[float] = None       # arrival time of the most recent completion

    def tick(self, now: float) -> None:
        self.done += 1
        if self._first is None:
            self._first = now
        self._last = now

    def eta(self) -> Optional[float]:
        """Seconds remaining, or None when no honest estimate exists yet."""
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
        return ProgressReport(label=self.label, done=self.done, total=self.total,
                              unit=self.unit, eta_seconds=self.eta())
