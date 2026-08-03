"""First paint, the run log that survives the process, and the clock on a window opening.

Three things this suite pins. The first two are the operator-run-measurement design's; the third
is the ROI-viewport design's increment 0 (section 3.6), which asks for a second clock on the SAME
seam rather than a second mechanism beside it:

1. **First paint** is the interval from a user starting an operator run to the first tile of that
   run being drawn. It is distinct from the run's total duration, which ``RunMetrics.seconds``
   already carries. A run that emits promptly while the user sees nothing is the failure being
   investigated, so the number must be settable from wherever the drawing happens rather than
   derived on the producer's side.

2. **Persistence.** ``METRICS`` is a bounded in-memory deque, so every measurement the app takes is
   discarded when the process exits. That is why two eng-review documents can disagree about the
   per-region cost with no way to settle it. One JSON line per finished run, appended, outside the
   repo and outside the acquisition.

3. **The window-open clock.** Julio: "If we can speed up window loading time, that would be good."
   Nothing could say whether it had, because the only measured interval in the app started on a Run
   button that opening a window never presses. Requested -> first mosaic layer -> loaded, recorded
   into the same log and therefore the same file.

Per docs/adr/0001-ci-gates-work-not-time.md, nothing here asserts a duration. These tests assert
that a number was recorded and carried, never that it was small.
"""

import json

import pytest

from squidmip._measure import (
    FAILED,
    OK,
    STOPPED,
    WINDOW_OPEN,
    MetricsLog,
    RunMetrics,
    WindowOpen,
    compare,
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


# ---------------------------------------------------------------------------------------
# The window-open clock
# ---------------------------------------------------------------------------------------
# Julio: "If we can speed up window loading time, that would be good." The operator metric above
# cannot answer that: its clock starts on a Run button a window open never presses. This is the
# second clock, and these tests pin what it records, never how long anything took (ADR-0001).
#
# The wiring -- that a real window opening actually starts and stops one -- is in
# tests/test_window_open_measurement.py, which needs Qt and is gated for it.


class _Clock:
    """A clock the test moves by hand, so an interval can be asserted exactly.

    Sleeping to make a real clock advance would put a duration in the assertion, which is the thing
    ADR-0001 forbids, and would make the suite slower for the privilege.
    """

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_a_window_open_records_the_wait_from_request_to_loaded():
    """The interval the user experiences: asked for a window, saw it finish loading."""
    metrics, clock = MetricsLog(), _Clock()
    w = WindowOpen("1 region: A1", metrics=metrics, clock=clock)
    clock.t = 3.0
    w.finish()

    m = metrics.last()
    assert m.operator == WINDOW_OPEN
    assert m.target == "1 region: A1"
    assert m.seconds == pytest.approx(3.0)
    assert m.outcome == OK


def test_window_open_first_paint_is_the_first_layer_not_the_last():
    """Every channel of every region reaches the same handler. Last-wins would report the last
    channel of whatever region the user navigated to and call it first paint."""
    metrics, clock = MetricsLog(), _Clock()
    w = WindowOpen("1 region: A1", metrics=metrics, clock=clock)
    clock.t = 1.0
    w.first_layer()
    clock.t = 2.0
    w.first_layer()
    clock.t = 3.0
    w.finish()

    m = metrics.last()
    assert m.first_paint_seconds == pytest.approx(1.0)
    assert m.seconds == pytest.approx(3.0), "first paint and the whole open are different facts"


def test_a_window_that_never_showed_a_layer_records_no_first_paint():
    """None, not 0.0. A window that showed nothing did not show it instantly."""
    metrics, clock = MetricsLog(), _Clock()
    WindowOpen("1 region: A1", metrics=metrics, clock=clock).finish(FAILED, "no mosaic")

    assert metrics.last().first_paint_seconds is None
    assert metrics.last().outcome == FAILED


def test_a_window_closed_before_its_mosaic_landed_still_leaves_a_record():
    """The wait somebody GAVE UP ON is the most interesting one there is, and it is the only one
    with no natural end. Dropping it would leave the log listing only the opens that went well."""
    metrics, clock = MetricsLog(), _Clock()
    w = WindowOpen("1 region: A1", metrics=metrics, clock=clock)
    clock.t = 9.0
    w.finish(STOPPED, "closed before its mosaic landed")

    m = metrics.last()
    assert m.outcome == STOPPED
    assert m.seconds == pytest.approx(9.0)
    assert m.first_paint_seconds is None


def test_an_open_is_recorded_once_however_many_times_it_ends():
    """Three paths end an open -- it loads, it cannot load, the user closes it -- and a window that
    loads is closed later too. Recording both would double every open in the log."""
    metrics, clock = MetricsLog(), _Clock()
    w = WindowOpen("1 region: A1", metrics=metrics, clock=clock)
    clock.t = 2.0
    w.finish()
    clock.t = 60.0
    assert w.finish(STOPPED, "closed") is None

    assert len(metrics) == 1
    assert metrics.last().outcome == OK
    assert metrics.last().seconds == pytest.approx(2.0)


def test_a_layer_arriving_after_the_open_ended_does_not_alter_it():
    """A window that was closed, or that failed, can still have a layer land afterwards. The record
    is written and frozen by then, so the clock must not go on collecting a first paint that no
    record will ever carry and that any later reader would take for one."""
    metrics, clock = MetricsLog(), _Clock()
    w = WindowOpen("1 region: A1", metrics=metrics, clock=clock)
    w.finish()
    clock.t = 5.0
    w.first_layer()

    assert metrics.last().first_paint_seconds is None
    assert w.first_paint_seconds is None


def test_a_window_open_is_appended_to_the_same_run_log(tmp_path):
    """One history and one file. A separate log for opens would need its own subscriber, its own
    rotation and its own reader, and the next person would have two places to look."""
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)

    clock = _Clock()
    w = WindowOpen("ROI in 1 region: A1", metrics=metrics, clock=clock)
    clock.t = 0.5
    w.first_layer()
    clock.t = 1.25
    w.finish()

    line = json.loads(path.read_text().splitlines()[0])
    assert line["operator"] == WINDOW_OPEN
    assert line["target"] == "ROI in 1 region: A1"
    assert line["first_paint_seconds"] == pytest.approx(0.5)
    assert line["seconds"] == pytest.approx(1.25)


def test_a_window_open_does_not_claim_a_peak_it_never_sampled():
    """This is a CLOCK. A single reading taken at the end is not a peak, and labelling it one puts
    an invented number in the column the hardware-budget work will read."""
    metrics = MetricsLog()
    WindowOpen("1 region: A1", metrics=metrics, clock=_Clock()).finish()

    m = metrics.last()
    assert m.peak_rss is None
    assert m.peak_over_start is None
    assert "peak unknown" in m.line()


def test_window_opens_are_selectable_apart_from_operator_runs():
    """Recording opens into the shared log is only safe because the ``operator`` field already
    separates them: 'how long do windows take' and 'how fast is decon' stay different questions."""
    metrics, clock = MetricsLog(), _Clock()
    _run(metrics, operator="mip")
    WindowOpen("1 region: A1", metrics=metrics, clock=clock).finish()

    assert [r["operator"] for r in compare(metrics, operators=[WINDOW_OPEN])] == [WINDOW_OPEN]
    assert [r["operator"] for r in compare(metrics, operators=["mip"])] == ["mip"]
