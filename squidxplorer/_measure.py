"""What one operator run cost: wall clock and peak resident memory."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Sequence

__all__ = [
    "RunMetrics",
    "MetricsLog",
    "METRICS",
    "measure_run",
    "verdict",
    "rss_bytes",
    "human_bytes",
    "human_seconds",
    "compare",
    "SAMPLE_INTERVAL_S",
    "run_log_path",
    "persist_runs",
    "WindowOpen",
    "WINDOW_OPEN",
]

from squidxplorer._logpane import get_logger

logger = get_logger("measure")

#: How often the poller reads RSS while a run is in flight.
SAMPLE_INTERVAL_S = 0.05

#: Runs retained in memory; bounded to avoid an unbounded in-session leak.
MAX_RUNS = 500

#: "Finished" and "finished having skipped everything" are different facts, not the same duration.
OK = "ok"
PARTIAL = "partial"
FAILED = "failed"
STOPPED = "stopped"
OUTCOMES = (OK, PARTIAL, FAILED, STOPPED)


def verdict(landed: int, owed: int, skipped: int, stopped: bool) -> "tuple[str, str]":
    """``(outcome, detail)`` for a finished run — THE one computation, for every surface.

    ``stopped`` wins and comes with an empty detail, because how a run was stopped (a manifest's
    own flag, a window's stop event) is the caller's sentence to write. A run that landed nothing
    while owing targets is PARTIAL, as is any skip. ``landed`` is in the caller's own unit
    (fields or wells) — only zero is read here — while ``owed`` and ``skipped`` count target
    wells. FAILED never comes from counts; it is the exception path's word.
    """
    if stopped:
        return STOPPED, ""
    if landed == 0 and owed:
        return PARTIAL, f"produced nothing — all {owed} target(s) skipped"
    if skipped:
        return PARTIAL, f"{skipped} well(s) skipped"
    return OK, ""


def rss_bytes() -> Optional[int]:
    """This process's resident set size, or None (never 0) when it cannot be measured."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:                       # noqa: BLE001 - psutil missing/refused; say so by None
        return None


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "peak unknown"
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024.0 or unit == "TiB":
            return f"{v:.0f} {unit}" if unit in ("B", "KiB") else f"{v:.2f} {unit}"
        v /= 1024.0
    return f"{v:.2f} TiB"                   # pragma: no cover - unreachable, the loop returns


def human_seconds(s: float) -> str:
    """Duration a human reads at a glance."""
    s = float(s)
    if s < 1.0:
        return f"{s * 1000:.0f} ms"
    if s < 60.0:
        return f"{s:.1f} s"
    m, rem = divmod(s, 60.0)
    if m < 60:
        return f"{int(m)}m {rem:04.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h {m:02d}m {rem:04.1f}s"


@dataclass(frozen=True)
class RunMetrics:
    """One operator run's cost, as a record."""

    operator: str
    target: str
    n_targets: Optional[int]
    seconds: float
    peak_rss: Optional[int]
    start_rss: Optional[int]
    outcome: str
    detail: str = ""
    started_at: float = 0.0
    #: Time from run start to the first tile DRAWN (not emitted); None means none was drawn.
    first_paint_seconds: Optional[float] = None
    extra: dict = field(default_factory=dict)

    @property
    def peak_over_start(self) -> Optional[int]:
        """How much the run added at its peak."""
        if self.peak_rss is None or self.start_rss is None:
            return None
        return max(0, self.peak_rss - self.start_rss)

    def line(self) -> str:
        """The one line per run that goes to the log panel."""
        parts = [
            self.operator,
            self.target,
            human_seconds(self.seconds),
            f"peak {human_bytes(self.peak_rss)}",
        ]
        if self.peak_over_start is not None:
            parts.append(f"+{human_bytes(self.peak_over_start)}")
        # Absent rather than 0.0 s: a zero would read as the best result in the log.
        if self.first_paint_seconds is not None:
            parts.append(f"first paint {human_seconds(self.first_paint_seconds)}")
        parts.append(self.outcome if not self.detail else f"{self.outcome} — {self.detail}")
        return " · ".join(parts)

    def as_dict(self) -> dict:
        """Serialisable form for the CSAT record, a manifest, or a command result."""
        return {
            "operator": self.operator,
            "target": self.target,
            "n_targets": self.n_targets,
            "seconds": round(float(self.seconds), 4),
            "peak_rss": self.peak_rss,
            "start_rss": self.start_rss,
            "peak_over_start": self.peak_over_start,
            "outcome": self.outcome,
            "detail": self.detail,
            "started_at": self.started_at,
            "first_paint_seconds": self.first_paint_seconds,
            **({"extra": dict(self.extra)} if self.extra else {}),
        }


class MetricsLog:
    """Bounded registry of finished runs, with subscribers."""

    def __init__(self, maxlen: int = MAX_RUNS) -> None:
        self._runs: deque = deque(maxlen=int(maxlen))
        self._subs: list[Callable[[RunMetrics], None]] = []

    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self) -> Iterator[RunMetrics]:
        return iter(list(self._runs))

    def record(self, m: RunMetrics) -> RunMetrics:
        self._runs.append(m)
        for cb in list(self._subs):
            try:
                cb(m)
            except Exception:               # noqa: BLE001 - a sink's bug must not fail the run
                pass
        return m

    def subscribe(self, callback: Callable[[RunMetrics], None]) -> None:
        self._subs.append(callback)

    def clear(self) -> None:
        self._runs.clear()

    def for_operator(self, operator: str) -> list:
        return [m for m in self._runs if m.operator == operator]

    def last(self) -> Optional[RunMetrics]:
        return self._runs[-1] if self._runs else None


#: One history per process.
METRICS = MetricsLog()

#: Marks a log that already has a JSONL sink attached, so persist_runs() stays idempotent.
_PERSISTED = "_squidxplorer_run_log_installed"


def run_log_path():
    """Where finished runs are appended, one JSON object per line."""
    from pathlib import Path

    from squidxplorer._platecache import cache_root

    return Path(cache_root()) / "runs.jsonl"


def persist_runs(metrics: Optional[MetricsLog] = None, path=None):
    """Append every finished run to ``path`` as one JSON line, idempotently. Returns the path."""
    from pathlib import Path

    log = metrics if metrics is not None else METRICS
    target = Path(path) if path is not None else run_log_path()
    if getattr(log, _PERSISTED, None) is not None:
        return getattr(log, _PERSISTED)

    def _write(m: RunMetrics) -> None:
        import json

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(m.as_dict()) + "\n")
        except OSError as exc:                          # noqa: BLE001 - named, not swallowed
            logger.warning("run log unwritable at %s: %s", target, exc)

    log.subscribe(_write)
    setattr(log, _PERSISTED, target)
    return target


class _Sampler:
    """Polls RSS on a daemon thread and keeps the maximum. Best effort, never fatal."""

    def __init__(self, interval: float = SAMPLE_INTERVAL_S) -> None:
        self.interval = float(interval)
        self.peak: Optional[int] = None
        #: RSS at t=0, kept separate from peak so peak_over_start reports only this run.
        self.start_rss: Optional[int] = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _note(self, value: Optional[int]) -> None:
        if value is None:
            return
        self.samples += 1
        if self.peak is None or value > self.peak:
            self.peak = value

    def start(self) -> "_Sampler":
        # Seed at t=0 so a run shorter than one interval still reports a peak, not None.
        self.start_rss = rss_bytes()
        self._note(self.start_rss)

        def _loop() -> None:
            while not self._stop.wait(self.interval):
                self._note(rss_bytes())

        try:
            self._thread = threading.Thread(target=_loop, name="squidxplorer-rss", daemon=True)
            self._thread.start()
        except Exception:                   # noqa: BLE001 - no thread available; the seed stands
            self._thread = None
        return self

    def stop(self) -> Optional[int]:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            # Bounded join: the poller only waits on the stop event, so it returns within one interval.
            t.join(timeout=max(1.0, self.interval * 20))
        self._note(rss_bytes())             # the last moment before teardown counts too
        return self.peak


class RunRecorder:
    """The handle :func:`measure_run` yields; the caller reports the outcome.

    An unset outcome defaults to ``ok`` on a clean exit and ``failed`` on an exception.
    """

    def __init__(self, operator: str, target: str, n_targets: Optional[int]) -> None:
        self.operator = str(operator)
        self.target = str(target)
        self.n_targets = n_targets
        self.outcome: Optional[str] = None
        self.detail: str = ""
        self.extra: dict = {}
        self.metrics: Optional[RunMetrics] = None
        self.first_paint_seconds: Optional[float] = None
        self._closed = False

    def first_paint(self, seconds: float) -> None:
        """Report this run's first tile as DRAWN, ``seconds`` after it was asked for. First call wins;
        a negative interval or a report after the block has exited is dropped."""
        if self._closed or self.first_paint_seconds is not None:
            return
        s = float(seconds)
        if s < 0.0:
            return
        self.first_paint_seconds = s

    def finish(self, outcome: str, detail: str = "") -> None:
        """Name how this run ended. Last call wins."""
        self.outcome = str(outcome)
        self.detail = str(detail)

    def note(self, **kwargs) -> None:
        """Attach extra data (dataset, version, params) to the recorded run."""
        self.extra.update(kwargs)


class measure_run:
    """Time one operator run and record its peak RSS. A context manager.

    ::

        with measure_run("mip", "28 regions", n_targets=28) as run:
            ...
            run.finish("partial", "3 wells skipped")

    An exception inside the block is recorded as ``failed`` and re-raised.
    """

    def __init__(self, operator: str, target: str = "", *, n_targets: Optional[int] = None,
                 log: Optional[logging.Logger] = None, metrics: Optional[MetricsLog] = None,
                 interval: float = SAMPLE_INTERVAL_S, announce: bool = True) -> None:
        self._recorder = RunRecorder(operator, target, n_targets)
        self._log = log if log is not None else logger
        self._metrics = metrics if metrics is not None else METRICS
        self._sampler = _Sampler(interval)
        self._announce = bool(announce)
        self._t0 = 0.0

    def __enter__(self) -> RunRecorder:
        if self._announce:
            self._log.info("%s: starting — %s", self._recorder.operator,
                           self._recorder.target or "no target named")
        self._sampler.start()
        # perf_counter, not time.time: immune to backwards NTP clock steps.
        self._t0 = time.perf_counter()
        return self._recorder

    def __exit__(self, exc_type, exc, tb) -> bool:
        seconds = max(0.0, time.perf_counter() - self._t0)
        peak = self._sampler.stop()
        r = self._recorder
        if exc_type is not None:
            outcome, detail = FAILED, f"{exc_type.__name__}: {exc}"
        else:
            outcome = r.outcome or OK
            detail = r.detail
        m = RunMetrics(
            operator=r.operator, target=r.target, n_targets=r.n_targets,
            seconds=seconds, peak_rss=peak, start_rss=self._sampler.start_rss,
            outcome=outcome, detail=detail, started_at=time.time() - seconds,
            extra=dict(r.extra), first_paint_seconds=r.first_paint_seconds,
        )
        r._closed = True        # a tile arriving after this cannot alter a record already written
        r.metrics = m
        self._metrics.record(m)
        level = logging.WARNING if outcome in (FAILED, PARTIAL) else logging.INFO
        self._log.log(level, "%s", m.line())
        return False                        # never swallow: an exception here is the run's, not ours


#: Pseudo-operator name so window opens are comparable via compare() alongside real operators.
WINDOW_OPEN = "window open"


class WindowOpen:
    """The clock for ONE region or ROI window: requested -> first mosaic layer -> loaded.

    Records a RunMetrics (operator=WINDOW_OPEN) so window opens share one history and one log
    file with operator runs. ``peak_rss`` is always None: this class is a clock, not a sampler.
    """

    def __init__(self, target: str, *, n_targets: Optional[int] = None,
                 metrics: Optional[MetricsLog] = None, log: Optional[logging.Logger] = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.target = str(target)
        self.n_targets = n_targets
        self.first_paint_seconds: Optional[float] = None
        self.metrics: Optional[RunMetrics] = None
        self._clock = clock
        self._metrics = metrics if metrics is not None else METRICS
        self._log = log if log is not None else logger
        # perf_counter: immune to backwards NTP clock steps.
        self._t0 = float(self._clock())
        self._start_rss = rss_bytes()
        self._closed = False

    def _elapsed(self) -> float:
        return max(0.0, float(self._clock()) - self._t0)

    def first_layer(self) -> None:
        """The window's first mosaic layer just went in; first call wins."""
        if self._closed or self.first_paint_seconds is not None:
            return
        self.first_paint_seconds = self._elapsed()

    def finish(self, outcome: str = OK, detail: str = "") -> Optional[RunMetrics]:
        """Close the clock and record it, idempotently. Returns None if already closed."""
        if self._closed:
            return None
        self._closed = True
        seconds = self._elapsed()
        m = RunMetrics(
            operator=WINDOW_OPEN, target=self.target, n_targets=self.n_targets,
            seconds=seconds, peak_rss=None, start_rss=self._start_rss,
            outcome=str(outcome), detail=str(detail), started_at=time.time() - seconds,
            first_paint_seconds=self.first_paint_seconds,
        )
        self.metrics = m
        self._metrics.record(m)
        level = logging.WARNING if m.outcome in (FAILED, PARTIAL) else logging.INFO
        self._log.log(level, "%s", m.line())
        return m


def compare(metrics: Optional[MetricsLog] = None, operators: Optional[Sequence[str]] = None) -> list:
    """The n-algorithms table: one row per operator, over the runs recorded so far.

    Only an ``ok`` run is timed — a failed/partial/stopped run's duration is not a speed, so it
    counts in ``runs`` (and, if failed, ``failures``) but never in ``timed`` or ``median_seconds``.
    Median, not mean; ``peak_rss`` is the worst peak over the timed runs only.
    """
    log = metrics if metrics is not None else METRICS
    rows: dict[str, dict] = {}
    for m in log:
        if operators is not None and m.operator not in operators:
            continue
        row = rows.setdefault(m.operator, {"operator": m.operator, "runs": 0, "timed": 0,
                                           "failures": 0, "_secs": [], "peak_rss": None})
        row["runs"] += 1
        if m.outcome == FAILED:
            row["failures"] += 1
        if m.outcome != OK:
            continue
        row["timed"] += 1
        row["_secs"].append(float(m.seconds))
        if m.peak_rss is not None and (row["peak_rss"] is None or m.peak_rss > row["peak_rss"]):
            row["peak_rss"] = m.peak_rss
    out = []
    for row in rows.values():
        secs = sorted(row.pop("_secs"))
        row["median_seconds"] = _median(secs)
        row["best_seconds"] = secs[0] if secs else None
        out.append(row)
    # A row with no timed run sorts last, not first (None would otherwise sort as 0).
    return sorted(out, key=lambda r: (r["median_seconds"] is None, r["median_seconds"] or 0.0))


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def compare_table(metrics: Optional[MetricsLog] = None) -> str:
    """compare() as fixed-width text, for a terminal and the log panel."""
    rows = compare(metrics)
    if not rows:
        return "no operator runs recorded yet"
    head = (f"{'operator':<16}{'runs':>6}{'timed':>6}{'fail':>6}"
            f"{'median':>12}{'best':>12}{'peak RSS':>14}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['operator']:<16}{r['runs']:>6}{r['timed']:>6}{r['failures']:>6}"
            f"{(human_seconds(r['median_seconds']) if r['median_seconds'] is not None else '—'):>12}"
            f"{(human_seconds(r['best_seconds']) if r['best_seconds'] is not None else '—'):>12}"
            f"{human_bytes(r['peak_rss']):>14}"
        )
    return "\n".join(lines)
