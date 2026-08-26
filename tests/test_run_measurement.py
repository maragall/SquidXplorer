"""First paint, the persistent run log, and the window-open clock."""

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


def test_first_paint_is_the_first_valid_report_inside_the_run_or_none():
    """None (not 0.0) when never painted; first wins; negative and post-run reports are ignored."""
    assert _run(MetricsLog()).first_paint_seconds is None
    m = _run(MetricsLog(), first_paint=0.01)
    assert m.first_paint_seconds == pytest.approx(0.01) and m.first_paint_seconds != m.seconds
    failed = _run(MetricsLog(), first_paint=1.5, boom=True)
    assert failed.outcome == FAILED and failed.first_paint_seconds == pytest.approx(1.5)

    metrics = MetricsLog()
    with measure_run("mip", "2 regions", metrics=metrics, announce=False) as run:
        run.first_paint(0.4)
        run.first_paint(9.9)
    assert metrics.last().first_paint_seconds == pytest.approx(0.4), "last-wins turns first paint into last paint"
    with measure_run("mip", "2 regions", metrics=metrics, announce=False) as run:
        run.first_paint(-3.0)
    assert metrics.last().first_paint_seconds is None, "a negative duration is silently the best result in a table"
    with measure_run("mip", "2 regions", metrics=metrics, announce=False) as run:
        pass
    run.first_paint(0.5)
    assert metrics.last().first_paint_seconds is None


def test_first_paint_reaches_the_line_and_the_serialised_form_and_the_record_stays_frozen():
    m = _run(MetricsLog(), first_paint=0.25)
    assert "first paint" in m.line()
    assert m.as_dict()["first_paint_seconds"] == pytest.approx(0.25)
    assert "first paint" not in _run(MetricsLog()).line()
    with pytest.raises(Exception):
        m.first_paint_seconds = 99.0  # type: ignore[misc]
    bare = RunMetrics(operator="mip", target="1 region", n_targets=1, seconds=1.0,
                      peak_rss=None, start_rss=None, outcome=OK)
    assert bare.first_paint_seconds is None


def test_the_run_log_sits_under_the_user_cache_root(tmp_path, monkeypatch):
    """Never in the repo, never in the acquisition."""
    from squidxplorer import _platecache

    monkeypatch.setenv(_platecache.ENV_DIR, str(tmp_path))
    assert run_log_path().parent == tmp_path


def test_the_run_log_appends_one_json_line_per_run_failures_included(tmp_path):
    metrics = MetricsLog()
    path = tmp_path / "runs.jsonl"
    persist_runs(metrics=metrics, path=path)
    persist_runs(metrics=metrics, path=path)          # installing twice must not double-write
    _run(metrics, operator="mip", first_paint=0.25)
    _run(metrics, operator="stitch")
    _run(metrics, operator="decon", boom=True)

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [d["operator"] for d in lines] == ["mip", "stitch", "decon"]
    assert lines[0]["first_paint_seconds"] == pytest.approx(0.25)
    assert lines[2]["outcome"] == FAILED


def test_an_unwritable_log_does_not_fail_the_run(tmp_path):
    """Instrumentation must never take down the work it measures."""
    metrics = MetricsLog()
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("i am a file")
    persist_runs(metrics=metrics, path=blocked / "runs.jsonl")

    m = _run(metrics)
    assert m.outcome == OK
    assert len(metrics) == 1


# The window-open clock. The Qt wiring is pinned in tests/test_window_open_measurement.py.


class _Clock:
    """A clock the test moves by hand, so an interval can be asserted exactly."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_a_window_open_records_the_wait_and_its_first_layer_separately():
    metrics, clock = MetricsLog(), _Clock()
    w = WindowOpen("1 region: A1", metrics=metrics, clock=clock)
    clock.t = 1.0
    w.first_layer()
    clock.t = 2.0
    w.first_layer()
    clock.t = 3.0
    w.finish()

    m = metrics.last()
    assert m.operator == WINDOW_OPEN and m.target == "1 region: A1" and m.outcome == OK
    assert m.first_paint_seconds == pytest.approx(1.0)
    assert m.seconds == pytest.approx(3.0), "first paint and the whole open are different facts"


@pytest.mark.parametrize("outcome, detail, t", [(FAILED, "no mosaic", 0.0),
                                                (STOPPED, "closed before its mosaic landed", 9.0)])
def test_a_window_that_never_showed_a_layer_still_leaves_a_record(outcome, detail, t):
    """The wait somebody gave up on is the most interesting one there is; first paint is None, not 0.0."""
    metrics, clock = MetricsLog(), _Clock()
    w = WindowOpen("1 region: A1", metrics=metrics, clock=clock)
    clock.t = t
    w.finish(outcome, detail)

    m = metrics.last()
    assert m.outcome == outcome and m.seconds == pytest.approx(t)
    assert m.first_paint_seconds is None


def test_an_open_is_recorded_once_and_nothing_after_the_end_alters_it():
    metrics, clock = MetricsLog(), _Clock()
    w = WindowOpen("1 region: A1", metrics=metrics, clock=clock)
    clock.t = 2.0
    w.finish()
    clock.t = 60.0
    assert w.finish(STOPPED, "closed") is None
    w.first_layer()

    assert len(metrics) == 1
    assert metrics.last().outcome == OK and metrics.last().seconds == pytest.approx(2.0)
    assert metrics.last().first_paint_seconds is None and w.first_paint_seconds is None


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
