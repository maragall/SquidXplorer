"""First paint, and the run log that survives the process.

Two things this suite pins, both from the operator-run-measurement design:

1. **First paint** is the interval from a user starting an operator run to the first tile of that
   run being drawn. It is distinct from the run's total duration, which ``RunMetrics.seconds``
   already carries. A run that emits promptly while the user sees nothing is the failure being
   investigated, so the number must be settable from wherever the drawing happens rather than
   derived on the producer's side.

2. **Persistence.** ``METRICS`` is a bounded in-memory deque, so every measurement the app takes is
   discarded when the process exits. That is why two eng-review documents can disagree about the
   per-region cost with no way to settle it. One JSON line per finished run, appended, outside the
   repo and outside the acquisition.

Per docs/adr/0001-ci-gates-work-not-time.md, nothing here asserts a duration. These tests assert
that a number was recorded and carried, never that it was small.
"""

import json

import pytest

from squidmip._measure import (
    FAILED,
    OK,
    MetricsLog,
    RunMetrics,
    measure_run,
    persist_runs,
    run_log_path,
)


def _run(metrics, operator="mip", target="2 regions", first_paint=None, boom=False):
    """Drive one measured run and return the record it produced."""
    try:
        with measure_run(operator, target, n_targets=2, metrics=metrics, announce=False) as run:
            if first_paint is not None:
                run.first_paint(first_paint)
            if boom:
                raise ValueError("nope")
    except ValueError:
        pass
    return metrics.last()


# ---------------------------------------------------------------------------------------
# First paint
# ---------------------------------------------------------------------------------------

def test_a_run_that_never_painted_reports_no_first_paint():
    """None, not 0.0. A run whose first tile never appeared did not paint instantly."""
    m = _run(MetricsLog())
    assert m.first_paint_seconds is None


def test_first_paint_reported_by_the_caller_lands_on_the_record():
    m = _run(MetricsLog(), first_paint=0.25)
    assert m.first_paint_seconds == pytest.approx(0.25)


def test_first_paint_survives_a_failed_run():
    """A run can paint its first tile and then fail. Losing the number there would hide exactly
    the case where someone waited a long time AND got nothing."""
    m = _run(MetricsLog(), first_paint=1.5, boom=True)
    assert m.outcome == FAILED
    assert m.first_paint_seconds == pytest.approx(1.5)


def test_only_the_first_report_counts():
    """Every tile of a run reaches the same handler; the first one is the measurement. Last-wins
    would silently turn first paint into last paint on a long run."""
    metrics = MetricsLog()
    with measure_run("mip", "2 regions", metrics=metrics, announce=False) as run:
        run.first_paint(0.4)
        run.first_paint(9.9)
    assert metrics.last().first_paint_seconds == pytest.approx(0.4)


def test_a_negative_report_is_refused_rather_than_recorded():
    """A clock that runs backwards produces a negative interval, and a negative duration in a
    comparison table is silently the best result in it."""
    metrics = MetricsLog()
    with measure_run("mip", "2 regions", metrics=metrics, announce=False) as run:
        run.first_paint(-3.0)
    assert metrics.last().first_paint_seconds is None


def test_the_log_line_names_first_paint_when_there_is_one():
    m = _run(MetricsLog(), first_paint=0.25)
    assert "first paint" in m.line()


def test_the_log_line_says_nothing_about_first_paint_when_there_is_none():
    """Absent, not zero. The line is scanned vertically, so a column that means 'we did not
    measure this' must not look like a measurement."""
    assert "first paint" not in _run(MetricsLog()).line()


def test_the_serialised_form_carries_first_paint():
    d = _run(MetricsLog(), first_paint=0.25).as_dict()
    assert d["first_paint_seconds"] == pytest.approx(0.25)


def test_first_paint_is_not_the_run_duration():
    """Distinct fields. Conflating them is the whole reason this metric exists: the total was
    always measured and never answered the complaint."""
    m = _run(MetricsLog(), first_paint=0.01)
    assert m.first_paint_seconds != m.seconds


# ---------------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------------

def test_the_run_log_sits_under_the_user_cache_root(tmp_path, monkeypatch):
    """Never in the repo, never in the acquisition. `_platecache.cache_root` already owns that
    rule and its override, so this reuses it rather than deriving a second location."""
    from squidmip import _platecache

    monkeypatch.setenv(_platecache.ENV_DIR, str(tmp_path))
    assert run_log_path().parent == tmp_path


def test_a_finished_run_appends_one_line(tmp_path):
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)
    _run(metrics, first_paint=0.25)

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["first_paint_seconds"] == pytest.approx(0.25)


def test_runs_accumulate_rather_than_replace(tmp_path):
    """Append-only: the comparison you most want is against a session weeks ago."""
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)
    _run(metrics, operator="mip")
    _run(metrics, operator="stitch")

    operators = [json.loads(line)["operator"] for line in path.read_text().splitlines()]
    assert operators == ["mip", "stitch"]


def test_a_failed_run_is_written_too(tmp_path):
    """The slow and broken runs are the ones worth having a record of."""
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)
    _run(metrics, boom=True)

    assert json.loads(path.read_text().splitlines()[0])["outcome"] == FAILED


def test_installing_twice_does_not_double_write(tmp_path):
    """Every PlateWindow construction would otherwise add another sink, and a plate opened eight
    times in one process would write eight lines per run."""
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)
    persist_runs(metrics=metrics, path=path)
    _run(metrics)

    assert len(path.read_text().splitlines()) == 1


def test_an_unwritable_log_does_not_fail_the_run(tmp_path):
    """Instrumentation must never take down the work it measures. The run still records in
    memory; only the copy on disk is lost."""
    metrics = MetricsLog()
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("i am a file")
    persist_runs(metrics=metrics, path=blocked / "runs.jsonl")

    m = _run(metrics)
    assert m.outcome == OK
    assert len(metrics) == 1


def test_every_line_is_one_json_object(tmp_path):
    """Line-delimited so two sessions can be compared with ordinary tools."""
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)
    for _ in range(3):
        _run(metrics)

    for line in path.read_text().splitlines():
        assert isinstance(json.loads(line), dict)


def test_the_record_is_still_frozen():
    """A measurement that can be edited after the fact is a claim, not a measurement. Adding a
    field must not have opened the record up."""
    m = _run(MetricsLog(), first_paint=0.25)
    with pytest.raises(Exception):
        m.first_paint_seconds = 99.0  # type: ignore[misc]


def test_a_first_paint_report_after_the_run_ended_is_ignored():
    """The recorder outlives the block; a late tile must not mutate a record already written."""
    metrics = MetricsLog()
    with measure_run("mip", "2 regions", metrics=metrics, announce=False) as run:
        pass
    run.first_paint(0.5)
    assert metrics.last().first_paint_seconds is None


def test_RunMetrics_can_still_be_built_without_first_paint():
    """The field is optional, so existing construction sites keep working."""
    m = RunMetrics(operator="mip", target="1 region", n_targets=1, seconds=1.0,
                   peak_rss=None, start_rss=None, outcome=OK)
    assert m.first_paint_seconds is None
