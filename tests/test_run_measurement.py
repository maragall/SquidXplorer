"""First paint, the persistent run log, and the window-open clock.

Per docs/adr/0001-ci-gates-work-not-time.md, nothing here asserts a duration.
"""

import json

import pytest

from squidxplorer._measure import (
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


def test_a_run_that_never_painted_reports_no_first_paint():
    """None, not 0.0."""
    m = _run(MetricsLog())
    assert m.first_paint_seconds is None


def test_first_paint_reported_by_the_caller_lands_on_the_record():
    m = _run(MetricsLog(), first_paint=0.25)
    assert m.first_paint_seconds == pytest.approx(0.25)


def test_first_paint_survives_a_failed_run():
    m = _run(MetricsLog(), first_paint=1.5, boom=True)
    assert m.outcome == FAILED
    assert m.first_paint_seconds == pytest.approx(1.5)


def test_only_the_first_report_counts():
    """Last-wins would silently turn first paint into last paint on a long run."""
    metrics = MetricsLog()
    with measure_run("mip", "2 regions", metrics=metrics, announce=False) as run:
        run.first_paint(0.4)
        run.first_paint(9.9)
    assert metrics.last().first_paint_seconds == pytest.approx(0.4)


def test_a_negative_report_is_refused_rather_than_recorded():
    """A negative duration in a comparison table is silently the best result in it."""
    metrics = MetricsLog()
    with measure_run("mip", "2 regions", metrics=metrics, announce=False) as run:
        run.first_paint(-3.0)
    assert metrics.last().first_paint_seconds is None


def test_the_log_line_names_first_paint_when_there_is_one():
    m = _run(MetricsLog(), first_paint=0.25)
    assert "first paint" in m.line()


def test_the_log_line_says_nothing_about_first_paint_when_there_is_none():
    assert "first paint" not in _run(MetricsLog()).line()


def test_the_serialised_form_carries_first_paint():
    d = _run(MetricsLog(), first_paint=0.25).as_dict()
    assert d["first_paint_seconds"] == pytest.approx(0.25)


def test_first_paint_is_not_the_run_duration():
    m = _run(MetricsLog(), first_paint=0.01)
    assert m.first_paint_seconds != m.seconds


def test_the_run_log_sits_under_the_user_cache_root(tmp_path, monkeypatch):
    """Never in the repo, never in the acquisition."""
    from squidxplorer import _platecache

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
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)
    _run(metrics, operator="mip")
    _run(metrics, operator="stitch")

    operators = [json.loads(line)["operator"] for line in path.read_text().splitlines()]
    assert operators == ["mip", "stitch"]


def test_a_failed_run_is_written_too(tmp_path):
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)
    _run(metrics, boom=True)

    assert json.loads(path.read_text().splitlines()[0])["outcome"] == FAILED


def test_installing_twice_does_not_double_write(tmp_path):
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)
    persist_runs(metrics=metrics, path=path)
    _run(metrics)

    assert len(path.read_text().splitlines()) == 1


def test_an_unwritable_log_does_not_fail_the_run(tmp_path):
    """Instrumentation must never take down the work it measures."""
    metrics = MetricsLog()
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("i am a file")
    persist_runs(metrics=metrics, path=blocked / "runs.jsonl")

    m = _run(metrics)
    assert m.outcome == OK
    assert len(metrics) == 1


def test_every_line_is_one_json_object(tmp_path):
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)
    for _ in range(3):
        _run(metrics)

    lines = path.read_text().splitlines()
    # The count guards against persist_runs writing nothing and the loop iterating zero times.
    assert len(lines) == 3, lines
    for line in lines:
        assert isinstance(json.loads(line), dict)


def test_the_record_is_still_frozen():
    m = _run(MetricsLog(), first_paint=0.25)
    with pytest.raises(Exception):
        m.first_paint_seconds = 99.0  # type: ignore[misc]


def test_a_first_paint_report_after_the_run_ended_is_ignored():
    metrics = MetricsLog()
    with measure_run("mip", "2 regions", metrics=metrics, announce=False) as run:
        pass
    run.first_paint(0.5)
    assert metrics.last().first_paint_seconds is None


def test_RunMetrics_can_still_be_built_without_first_paint():
    m = RunMetrics(operator="mip", target="1 region", n_targets=1, seconds=1.0,
                   peak_rss=None, start_rss=None, outcome=OK)
    assert m.first_paint_seconds is None


# The window-open clock. The Qt wiring is pinned in tests/test_window_open_measurement.py.


class _Clock:
    """A clock the test moves by hand, so an interval can be asserted exactly."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_a_window_open_records_the_wait_from_request_to_loaded():
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
    """None, not 0.0."""
    metrics, clock = MetricsLog(), _Clock()
    WindowOpen("1 region: A1", metrics=metrics, clock=clock).finish(FAILED, "no mosaic")

    assert metrics.last().first_paint_seconds is None
    assert metrics.last().outcome == FAILED


def test_a_window_closed_before_its_mosaic_landed_still_leaves_a_record():
    """The wait somebody gave up on is the most interesting one there is."""
    metrics, clock = MetricsLog(), _Clock()
    w = WindowOpen("1 region: A1", metrics=metrics, clock=clock)
    clock.t = 9.0
    w.finish(STOPPED, "closed before its mosaic landed")

    m = metrics.last()
    assert m.outcome == STOPPED
    assert m.seconds == pytest.approx(9.0)
    assert m.first_paint_seconds is None


def test_an_open_is_recorded_once_however_many_times_it_ends():
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
    metrics, clock = MetricsLog(), _Clock()
    w = WindowOpen("1 region: A1", metrics=metrics, clock=clock)
    w.finish()
    clock.t = 5.0
    w.first_layer()

    assert metrics.last().first_paint_seconds is None
    assert w.first_paint_seconds is None


def test_a_window_open_is_appended_to_the_same_run_log(tmp_path):
    """One history and one file."""
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
    """A single reading taken at the end is not a peak."""
    metrics = MetricsLog()
    WindowOpen("1 region: A1", metrics=metrics, clock=_Clock()).finish()

    m = metrics.last()
    assert m.peak_rss is None
    assert m.peak_over_start is None
    assert "peak unknown" in m.line()


def test_window_opens_are_selectable_apart_from_operator_runs():
    metrics, clock = MetricsLog(), _Clock()
    _run(metrics, operator="mip")
    WindowOpen("1 region: A1", metrics=metrics, clock=clock).finish()

    assert [r["operator"] for r in compare(metrics, operators=[WINDOW_OPEN])] == [WINDOW_OPEN]
    assert [r["operator"] for r in compare(metrics, operators=["mip"])] == ["mip"]
