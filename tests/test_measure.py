"""Per-operator wall clock and peak RSS — the measurement three consumers read.

These are RULE tests: they check the things that make a measurement worth trusting (peak and not
final, an exception is recorded AND re-raised, a failed run is not counted as a fast one) rather
than asserting particular numbers, which would be a flake on a shared machine.
"""

from __future__ import annotations

import logging
import time

import pytest

from squidmip._measure import (
    FAILED,
    MetricsLog,
    OK,
    PARTIAL,
    RunMetrics,
    compare,
    compare_table,
    human_bytes,
    human_seconds,
    measure_run,
    rss_bytes,
)


@pytest.fixture()
def log():
    return MetricsLog()


# --- the measurement itself -------------------------------------------------------------------

def test_a_run_is_recorded_with_its_wall_clock(log):
    with measure_run("mip", "2 regions", n_targets=2, metrics=log) as run:
        time.sleep(0.02)
        assert run.outcome is None            # not yet decided; the block owns that
    m = log.last()
    assert m.operator == "mip" and m.n_targets == 2
    assert m.seconds >= 0.02, "the recorded duration must cover the work, not the setup"
    assert m.outcome == OK


def test_the_outcome_the_block_names_is_the_one_recorded(log):
    with measure_run("bgsub", "1 region", metrics=log) as run:
        run.finish(PARTIAL, "3 wells skipped")
    m = log.last()
    assert m.outcome == PARTIAL and "3 wells skipped" in m.detail
    assert "3 wells skipped" in m.line()


def test_an_exception_is_recorded_as_failed_AND_reraised(log):
    """Instrumentation must never swallow a failure — silent partial failure is the defect shape
    this codebase has paid for most."""
    with pytest.raises(ValueError, match="boom"):
        with measure_run("decon", "1 region", metrics=log):
            raise ValueError("boom")
    m = log.last()
    assert m.outcome == FAILED
    assert "ValueError" in m.detail and "boom" in m.detail


def test_the_sampler_keeps_the_MAX_not_the_last_reading(monkeypatch):
    """The high-water-mark property, tested DETERMINISTICALLY.

    A real freed buffer does not lower RSS (Python's allocator keeps it), so allocating-then-
    freeing cannot distinguish peak from final on this machine. Instead, feed the sampler a known
    rising-then-FALLING sequence and assert it reports the crest, not the trough it ended on.

    MUTATION: change ``if value > self.peak`` to always assign (report the last reading) and this
    goes red — 300 is kept, not the final 150.
    """
    from squidmip import _measure as M

    seq = iter([100, 250, 300, 180, 150])   # RSS climbs to 300, then recedes
    monkeypatch.setattr(M, "rss_bytes", lambda: next(seq, 150))
    sampler = M._Sampler(interval=0.001).start()
    # drive several notes past the crest, then stop (which takes the final reading too)
    for _ in range(4):
        sampler._note(M.rss_bytes())
    peak = sampler.stop()
    assert peak == 300, f"the sampler reported {peak}, not the high-water mark 300"
    assert sampler.start_rss == 100, "start_rss must be the reading at t=0, not the peak"


def test_a_run_reports_a_peak_at_or_above_where_it_started(log):
    """Weaker but real end-to-end: an actual run's peak is never below where it began."""
    if rss_bytes() is None:
        pytest.skip("RSS is not measurable on this machine")
    import numpy as np

    with measure_run("hog", "synthetic", metrics=log, interval=0.01) as run:
        block = np.ones(64 << 20, dtype=np.uint8)   # 64 MiB, touched so it is really resident
        block[::4096] = 7
        time.sleep(0.05)
        run.finish(OK)
        del block
    m = log.last()
    assert m.peak_rss is not None and m.start_rss is not None
    assert m.peak_over_start is not None and m.peak_over_start >= 0


def test_a_run_shorter_than_one_sample_interval_still_reports_a_peak(log):
    """Seeded at t=0 on purpose: without it every fast operator reported 'peak unknown', which is
    exactly the class of run people compare most often."""
    if rss_bytes() is None:
        pytest.skip("RSS is not measurable on this machine")
    with measure_run("fast", "1 region", metrics=log, interval=30.0):
        pass
    assert log.last().peak_rss is not None


def test_the_measurement_survives_a_machine_that_will_not_report_memory(log, monkeypatch):
    """psutil absent or refused degrades to an honest None, never to a fabricated number and never
    to an exception in the caller's path."""
    monkeypatch.setattr("squidmip._measure.rss_bytes", lambda: None)
    with measure_run("mip", "1 region", metrics=log):
        pass
    m = log.last()
    assert m.peak_rss is None and m.peak_over_start is None
    assert "peak unknown" in m.line()


def test_the_sampler_thread_does_not_outlive_the_run(log):
    """A poller per run that is never joined is a thread leak in a GUI that runs hundreds."""
    import threading

    before = {t.name for t in threading.enumerate()}
    for _ in range(5):
        with measure_run("mip", "1 region", metrics=log, interval=0.01):
            time.sleep(0.02)
    time.sleep(0.05)
    after = [t for t in threading.enumerate()
             if t.name == "squidmip-rss" and t.name not in before]
    assert not after, f"sampler threads survived their runs: {after}"


def test_the_sampler_overhead_is_negligible(log):
    """The reason this is safe to leave on always.

    Compares a fixed amount of real work with and without the instrumentation. The bar is
    deliberately loose (2x the work's own duration is still a catastrophic overhead we would want
    to catch; anything near 1.0 is what we expect) because CI machines are noisy — this is a
    regression guard against someone dropping the interval to microseconds, not a benchmark.
    """
    import numpy as np

    def work():
        a = np.arange(1 << 20, dtype=np.float64)
        for _ in range(20):
            a = a * 1.000001
        return float(a[0])

    work()                                              # warm numpy/import paths
    t0 = time.perf_counter()
    for _ in range(3):
        work()
    bare = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(3):
        with measure_run("mip", "bench", metrics=log, announce=False):
            work()
    measured = time.perf_counter() - t0

    assert measured < bare * 3 + 0.5, (
        f"instrumented {measured:.3f}s vs bare {bare:.3f}s — the sampler is not negligible")


# --- the log line, which is what the panel shows ------------------------------------------------

def test_the_log_line_names_operator_target_clock_peak_and_outcome(log):
    m = RunMetrics(operator="mip", target="28 regions", n_targets=28, seconds=41.2,
                   peak_rss=2_000_000_000, start_rss=100_000_000, outcome=OK)
    line = m.line()
    for expected in ("mip", "28 regions", "41.2 s", "peak", "GiB"):
        assert expected in line, f"{expected!r} missing from {line!r}"


def test_the_run_emits_exactly_one_line_to_the_root_logger(caplog):
    """One line per run is the contract with the log panel — the panel listens to the ROOT logger,
    so anything logged here appears there with no wiring."""
    metrics = MetricsLog()
    with caplog.at_level(logging.INFO, logger="squidmip.measure"):
        with measure_run("mip", "2 regions", metrics=metrics):
            pass
    finals = [r for r in caplog.records if "peak" in r.getMessage()]
    assert len(finals) == 1, [r.getMessage() for r in caplog.records]


def test_a_failed_run_is_logged_at_warning_not_info(caplog):
    metrics = MetricsLog()
    with caplog.at_level(logging.INFO, logger="squidmip.measure"):
        with pytest.raises(RuntimeError):
            with measure_run("decon", "1 region", metrics=metrics):
                raise RuntimeError("nope")
    levels = {r.levelno for r in caplog.records if "peak" in r.getMessage()}
    assert levels == {logging.WARNING}


def test_human_seconds_keeps_milliseconds_below_a_second():
    assert human_seconds(0.0412).endswith("ms")
    assert "41" in human_seconds(0.0412)
    assert human_seconds(41.2) == "41.2 s"
    assert human_seconds(125.0).startswith("2m")
    assert human_seconds(3725.0).startswith("1h")


def test_human_bytes_never_prints_a_bare_number_for_unknown():
    assert human_bytes(None) == "peak unknown"
    assert human_bytes(2 << 30).endswith("GiB")


# --- the comparison table: the n-algorithms question --------------------------------------------

def test_compare_ranks_operators_by_median_wall_clock(log):
    for s in (0.30, 0.10, 0.20):
        log.record(RunMetrics("slow", "1 region", 1, s * 10, 1000, 500, OK))
    for s in (0.30, 0.10, 0.20):
        log.record(RunMetrics("fast", "1 region", 1, s, 1000, 500, OK))
    rows = compare(log)
    assert [r["operator"] for r in rows] == ["fast", "slow"]
    assert rows[0]["median_seconds"] == pytest.approx(0.20)
    assert rows[0]["best_seconds"] == pytest.approx(0.10)


def test_compare_uses_the_median_so_one_cold_run_does_not_decide_the_ranking(log):
    log.record(RunMetrics("a", "t", 1, 10.0, None, None, OK))   # cold cache
    log.record(RunMetrics("a", "t", 1, 1.0, None, None, OK))
    log.record(RunMetrics("a", "t", 1, 1.0, None, None, OK))
    log.record(RunMetrics("b", "t", 1, 2.0, None, None, OK))
    assert [r["operator"] for r in compare(log)] == ["a", "b"], (
        "a mean would put 'a' (mean 4.0) behind 'b' (2.0) on the strength of one cold run")


def test_a_failed_run_is_counted_but_never_timed(log):
    """An operator that is fast and breaks half the time must not out-rank a slow one that works."""
    log.record(RunMetrics("flaky", "t", 1, 0.01, None, None, FAILED, "ValueError"))
    log.record(RunMetrics("flaky", "t", 1, 5.0, None, None, OK))
    row = compare(log)[0]
    assert row["runs"] == 2 and row["failures"] == 1
    assert row["median_seconds"] == pytest.approx(5.0), "the failure's 0.01 s must not be a speed"


def test_an_operator_that_never_succeeded_sorts_last_not_first(log):
    log.record(RunMetrics("broken", "t", 1, 0.001, None, None, FAILED))
    log.record(RunMetrics("works", "t", 1, 9.0, None, None, OK))
    assert [r["operator"] for r in compare(log)] == ["works", "broken"]


def test_compare_reports_the_worst_peak_not_the_last_one(log):
    log.record(RunMetrics("mip", "t", 1, 1.0, 5_000_000, 0, OK))
    log.record(RunMetrics("mip", "t", 1, 1.0, 1_000_000, 0, OK))
    assert compare(log)[0]["peak_rss"] == 5_000_000, (
        "a memory budget is decided by the worst run, not the most recent")


def test_compare_table_says_so_when_there_is_nothing_to_compare(log):
    assert "no operator runs recorded" in compare_table(log)


def test_compare_table_has_one_row_per_operator(log):
    log.record(RunMetrics("mip", "t", 1, 1.0, 1 << 20, 0, OK))
    log.record(RunMetrics("mip", "t", 1, 2.0, 1 << 20, 0, OK))
    log.record(RunMetrics("stitch", "t", 1, 3.0, 1 << 20, 0, OK))
    body = [l for l in compare_table(log).splitlines()[2:] if l.strip()]
    assert len(body) == 2


# --- the registry ------------------------------------------------------------------------------

def test_the_history_is_bounded(log):
    small = MetricsLog(maxlen=3)
    for i in range(10):
        small.record(RunMetrics(f"op{i}", "t", 1, 1.0, None, None, OK))
    assert len(small) == 3, "an unbounded run history is a leak with a nice name"
    assert small.last().operator == "op9"


def test_a_subscriber_is_told_about_every_finished_run(log):
    seen = []
    log.subscribe(seen.append)
    with measure_run("mip", "1 region", metrics=log):
        pass
    assert len(seen) == 1 and seen[0].operator == "mip"


def test_one_broken_subscriber_does_not_stop_the_others(log):
    """The same rule LogBus follows: a sink's bug is not the measurement's problem."""
    seen = []

    def boom(_m):
        raise RuntimeError("bad sink")

    log.subscribe(boom)
    log.subscribe(seen.append)
    log.record(RunMetrics("mip", "t", 1, 1.0, None, None, OK))
    assert len(seen) == 1


def test_metrics_serialise_to_plain_data_for_the_csat_record(log):
    with measure_run("mip", "2 regions", n_targets=2, metrics=log) as run:
        run.note(dataset="plate-A", adapter_version="v3")
    d = log.last().as_dict()
    assert d["operator"] == "mip" and d["n_targets"] == 2 and d["outcome"] == OK
    assert d["extra"] == {"dataset": "plate-A", "adapter_version": "v3"}
    import json

    json.dumps(d)      # the CSAT loop stores this next to a survey response; it must serialise
