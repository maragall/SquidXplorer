"""What one operator run COST: wall clock and peak resident memory. Measured once, read thrice.

Julio: "part of the logger is having a timer that times the wallclock run of each operator, it
also measures memory footprint. Like this is great for integrating new implementations and
variants of the operator, assuming you're using the registry and relations to scale to n
algorithms."

That last clause is the whole design. The registries (``add_projector``, ``add_region_operator``,
``add_segmenter``) already make "n algorithms for one job" cheap to ADD. What they do not make
cheap is CHOOSING between them: two deconvolutions that both produce a plausible picture are
distinguished by what they cost, and nothing recorded what they cost. So a run's cost is a
first-class record, produced by the run itself, keyed by the operator's registry name — which is
what makes two implementations of the same job comparable without anybody writing a benchmark.

ONE MEASUREMENT, THREE CONSUMERS
--------------------------------
``docs/NAUTILUS.md`` names them, and they are deliberately the same number rather than three
similar ones:

1. **The log panel** — one line per run, so the user sees that the GUI did something and how long
   it took. A progress bar says "working"; "MIP · 28 regions · 41.2 s · peak 1.9 GiB" says what
   the machine actually did.
2. **The CSAT / Nautilus loop** — "the survey response is stored WITH the adapter version, the
   dataset it ran on, and the measured wall clock and peak RSS. 'It looked wrong' is not
   actionable; 'it looked wrong on this dataset, at this version, and took 4 minutes' is."
3. **A comparison table** — :func:`compare` groups the recorded runs by operator, which is the
   n-algorithms question asked directly.

Three separate stopwatches would drift, and the one in the GUI would be the only one anybody
maintained. So the measurement is here, and the three consumers READ.

WHY PEAK RSS AND NOT THE FINAL NUMBER
-------------------------------------
The final RSS after a run is close to useless: Python's allocator returns almost nothing to the
OS, and the engine's whole point is a BOUNDED in-flight window, so the interesting number is how
high the window actually got — that is the number that decides whether a machine can run this
operator at all. A before/after difference reports the leak, not the footprint, and on a run that
frees everything it reports zero for a job that peaked at 4 GiB.

So peak is SAMPLED, by a poller thread, at :data:`SAMPLE_INTERVAL_S`. That is an approximation and
it is stated as one: a spike shorter than the interval is missed. The alternatives are worse —
``tracemalloc`` sees only Python allocations (not the numpy buffers or the decode arenas that
dominate here, and it costs a large constant factor), and ``resource.getrusage(RUSAGE_SELF)``'s
``ru_maxrss`` is high-water-mark for the WHOLE PROCESS LIFETIME, so after one big run every later
run reports the same number forever. A 50 ms sampler on a job measured in seconds is both accurate
enough to compare implementations and cheap enough to leave on always (see
``tests/test_measure.py::test_the_sampler_overhead_is_negligible``).

WHY IT NEVER RAISES
-------------------
Instrumentation that can fail a run is worse than no instrumentation. psutil may be absent or
refused (a sandboxed/containerised process can be denied its own memory info), a poller thread may
not start under an exhausted thread limit. Every failure here degrades to ``None`` — a peak that is
honestly unknown — and never to a number that looks measured, and never to an exception in the
caller's path. That is the same rule ``_budget.available_bytes`` follows, for the same reason.
"""

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
    "rss_bytes",
    "human_bytes",
    "human_seconds",
    "compare",
    "SAMPLE_INTERVAL_S",
]

logger = logging.getLogger("squidmip.measure")

#: How often the poller reads RSS while a run is in flight. 50 ms: two orders of magnitude below
#: the runs it measures (seconds to minutes), so the sample count is in the hundreds rather than
#: the millions, and each sample is one cheap syscall.
SAMPLE_INTERVAL_S = 0.05

#: Runs retained in memory. Bounded for the same reason the log view is: a record of every run in
#: a long session is a slow leak with a nice name. 500 records is far more than any comparison
#: table shows and still a few hundred KB.
MAX_RUNS = 500

#: Outcomes a run can end with. Named, because "it finished" and "it finished having skipped every
#: well" are the same duration and completely different facts — this codebase has already shipped
#: a "done" message over an empty plate.
OK = "ok"
PARTIAL = "partial"
FAILED = "failed"
STOPPED = "stopped"
OUTCOMES = (OK, PARTIAL, FAILED, STOPPED)


def rss_bytes() -> Optional[int]:
    """This process's resident set size, or None when it cannot be measured.

    None rather than 0: a caller that cannot measure must be able to say "unknown", and a zero
    would be averaged into a comparison table as a real, very good, result.
    """
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
    """Duration a human reads at a glance. Sub-second runs keep their milliseconds — that is the
    resolution at which two implementations of a plane-op differ."""
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
    """One operator run's cost, as a record. Frozen: a measurement that can be edited after the
    fact is a claim, not a measurement.

    ``target`` is a SENTENCE naming what the run was aimed at, not a count, because "41 s" means
    nothing without it — the same operator over 4 wells and over 1536 is the comparison people
    actually get wrong. It is produced by ``_explore.describe_run_target``, the existing owner of
    that sentence, so the log line and the pre-run confirmation cannot disagree.
    """

    operator: str
    target: str
    n_targets: Optional[int]
    seconds: float
    peak_rss: Optional[int]
    start_rss: Optional[int]
    outcome: str
    detail: str = ""
    started_at: float = 0.0
    #: Anything the caller wants carried alongside — the dataset id, the adapter version, the
    #: operator's parameters. The CSAT loop needs exactly this and should not need a schema change
    #: here to record it.
    extra: dict = field(default_factory=dict)

    @property
    def peak_over_start(self) -> Optional[int]:
        """How much the run ADDED at its peak. The number that answers 'will this fit'."""
        if self.peak_rss is None or self.start_rss is None:
            return None
        return max(0, self.peak_rss - self.start_rss)

    def line(self) -> str:
        """THE one line per run that goes to the log panel. Fixed field order on purpose: a log
        a human scans vertically is only scannable if the columns do not move."""
        parts = [
            self.operator,
            self.target,
            human_seconds(self.seconds),
            f"peak {human_bytes(self.peak_rss)}",
        ]
        if self.peak_over_start is not None:
            parts.append(f"+{human_bytes(self.peak_over_start)}")
        parts.append(self.outcome if not self.detail else f"{self.outcome} — {self.detail}")
        return " · ".join(parts)

    def as_dict(self) -> dict:
        """Serialisable form — for the CSAT record, a manifest, or a command result. Same rule as
        the command layer: anything a machine consumes is plain data, never an object graph."""
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
            **({"extra": dict(self.extra)} if self.extra else {}),
        }


class MetricsLog:
    """Bounded registry of finished runs, with subscribers. The same shape as ``ActivityLog``.

    Deliberately NOT a list somebody appends to from three places: the comparison table, the log
    panel and the CSAT record must be looking at one history, and a second list is how this
    codebase's dominant defect (two representations of one truth) gets in.
    """

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


#: THE history. One per process, because the process is what the RSS measurement is about.
METRICS = MetricsLog()


class _Sampler:
    """Polls RSS on a daemon thread and keeps the maximum. Best effort, never fatal."""

    def __init__(self, interval: float = SAMPLE_INTERVAL_S) -> None:
        self.interval = float(interval)
        self.peak: Optional[int] = None
        #: RSS at t=0, kept SEPARATE from the peak so ``peak_over_start`` means what it says.
        #: Folding the two together would report the whole process's footprint as this run's cost.
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
        # Seed with the value at t=0 so a run that finishes inside one interval still reports a
        # real peak rather than None. Without this every fast operator was "peak unknown", which
        # is precisely the class of run people compare most often.
        self.start_rss = rss_bytes()
        self._note(self.start_rss)

        def _loop() -> None:
            while not self._stop.wait(self.interval):
                self._note(rss_bytes())

        try:
            self._thread = threading.Thread(target=_loop, name="squidmip-rss", daemon=True)
            self._thread.start()
        except Exception:                   # noqa: BLE001 - no thread available; the seed stands
            self._thread = None
        return self

    def stop(self) -> Optional[int]:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            # Bounded join: the poller only ever waits on the stop event, so it returns within one
            # interval. A generous multiple, and then we give up rather than hang a run's teardown
            # on instrumentation.
            t.join(timeout=max(1.0, self.interval * 20))
        self._note(rss_bytes())             # the last moment before teardown counts too
        return self.peak


class RunRecorder:
    """The handle :func:`measure_run` yields. The caller reports the OUTCOME; it owns nothing else.

    An outcome the caller does not set defaults to ``ok`` on a clean exit and ``failed`` on an
    exception, which is the honest default pair: a run that raised did not succeed, whatever it
    meant to record.
    """

    def __init__(self, operator: str, target: str, n_targets: Optional[int]) -> None:
        self.operator = str(operator)
        self.target = str(target)
        self.n_targets = n_targets
        self.outcome: Optional[str] = None
        self.detail: str = ""
        self.extra: dict = {}
        self.metrics: Optional[RunMetrics] = None

    def finish(self, outcome: str, detail: str = "") -> None:
        """Name how this run ended. Last call wins, so a partial result reported late is the one
        recorded."""
        self.outcome = str(outcome)
        self.detail = str(detail)

    def note(self, **kwargs) -> None:
        """Attach anything the CSAT record or a manifest will want (dataset, version, params)."""
        self.extra.update(kwargs)


class measure_run:
    """Time one operator run and record its peak RSS. A context manager.

    ::

        with measure_run("mip", "28 regions", n_targets=28) as run:
            ...                                  # the actual work
            run.finish("partial", "3 wells skipped")

    On exit it logs ONE line at INFO — which is what puts it in the log panel, because the panel
    listens to the root logger — and appends a :class:`RunMetrics` to :data:`METRICS`.

    An exception inside the block is recorded as ``failed`` with the exception's name and then
    RE-RAISED. Instrumentation must never swallow a failure: this codebase's stated worst defect
    shape is silent partial failure, and a stopwatch that eats an exception is that shape wearing
    a hat.
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
        # perf_counter, not time.time: a wall clock that an NTP step can move backwards produces
        # negative durations, and a negative duration in a comparison table is silently the best
        # result in it.
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
            extra=dict(r.extra),
        )
        r.metrics = m
        self._metrics.record(m)
        level = logging.WARNING if outcome in (FAILED, PARTIAL) else logging.INFO
        self._log.log(level, "%s", m.line())
        return False                        # never swallow: an exception here is the run's, not ours


def compare(metrics: Optional[MetricsLog] = None, operators: Optional[Sequence[str]] = None) -> list:
    """The n-algorithms table: one row per operator, over the runs recorded so far.

    Julio's reason for the whole module — "great for integrating new implementations and variants
    of the operator, assuming you're using the registry and relations to scale to n algorithms".
    The registry says WHICH algorithms exist; this says what each of them cost.

    Rows are ``{"operator", "runs", "median_seconds", "best_seconds", "peak_rss", "failures"}``,
    sorted by median wall clock (fastest first) so the table reads as a ranking. MEDIAN and not
    mean: one cold-cache first run doubles a mean over three runs and is not what anybody is
    trying to learn.

    Only runs that produced something are timed — a ``failed`` run's duration is how long it took
    to break, which is not a speed. They are counted in ``failures`` instead, because an operator
    that is fast and fails half the time must not out-rank a slow one that works.
    """
    log = metrics if metrics is not None else METRICS
    rows: dict[str, dict] = {}
    for m in log:
        if operators is not None and m.operator not in operators:
            continue
        row = rows.setdefault(m.operator, {"operator": m.operator, "runs": 0, "failures": 0,
                                           "_secs": [], "peak_rss": None})
        row["runs"] += 1
        if m.outcome == FAILED:
            row["failures"] += 1
            continue
        row["_secs"].append(float(m.seconds))
        if m.peak_rss is not None and (row["peak_rss"] is None or m.peak_rss > row["peak_rss"]):
            row["peak_rss"] = m.peak_rss
    out = []
    for row in rows.values():
        secs = sorted(row.pop("_secs"))
        row["median_seconds"] = _median(secs)
        row["best_seconds"] = secs[0] if secs else None
        out.append(row)
    # A row with no timed run sorts LAST rather than first: sorting None as 0 would put the
    # operator that never once succeeded at the top of a speed ranking.
    return sorted(out, key=lambda r: (r["median_seconds"] is None, r["median_seconds"] or 0.0))


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def compare_table(metrics: Optional[MetricsLog] = None) -> str:
    """:func:`compare` as fixed-width text, for a terminal and for the log panel."""
    rows = compare(metrics)
    if not rows:
        return "no operator runs recorded yet"
    head = f"{'operator':<16}{'runs':>6}{'fail':>6}{'median':>12}{'best':>12}{'peak RSS':>14}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['operator']:<16}{r['runs']:>6}{r['failures']:>6}"
            f"{(human_seconds(r['median_seconds']) if r['median_seconds'] is not None else '—'):>12}"
            f"{(human_seconds(r['best_seconds']) if r['best_seconds'] is not None else '—'):>12}"
            f"{human_bytes(r['peak_rss']):>14}"
        )
    return "\n".join(lines)
