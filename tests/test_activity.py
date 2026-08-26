"""ActivityLog: the busy indicator's rules, tested without a window."""

from __future__ import annotations

import pytest

from squidxplorer._activity import ActivityLog


@pytest.fixture
def log():
    return ActivityLog()


def test_busy_follows_start_and_end_and_clear_and_a_stray_end_is_harmless(log):
    assert not log.busy and log.sentence() == "" and log.current() is None
    log.start("fuse", "loading mosaic")
    assert log.busy
    log.end("fuse")
    assert not log.busy, "the indicator is still on with nothing running"
    log.end("never-started")
    assert not log.busy
    log.start("a", "one")
    log.start("b", "two")
    log.clear()
    assert not log.busy


def test_an_unknown_total_stays_unknown_until_the_worker_revises_it(log):
    """Unknown must stay unknown so the indicator can be indeterminate; the worker, not the click, knows the total."""
    a = log.start("fuse", "loading mosaic")
    assert a.total is None and not a.determinate
    assert a.sentence() == "loading mosaic …"
    log.start("run", "MIP")
    assert not log.current().determinate
    log.advance("run", 1, 28)
    assert log.current().determinate
    assert log.current().sentence() == "MIP · 1/28"
    log.advance("run", 3)
    assert log.current().sentence() == "MIP · 3/28"


def test_progress_for_work_that_already_ended_is_ignored_not_fatal(log):
    """A worker's last `progress` can be delivered after its `finished`."""
    log.start("run", "MIP", total=4)
    log.end("run")
    log.advance("run", 4, 4)          # must not raise
    assert not log.busy


def test_two_activities_are_both_tracked_and_the_determinate_one_is_shown(log):
    log.start("fuse", "loading mosaic")
    log.start("run", "MIP", total=28)
    log.advance("run", 5)
    assert len(log) == 2
    assert log.current().key == "run", "show the one that can say something real"
    assert log.sentence() == "MIP · 5/28  (+1 more)"


def test_restarting_the_same_key_replaces_it_rather_than_stacking(log):
    """Stacking would need two end()s to clear one visible activity."""
    log.start("fuse", "loading A1")
    log.start("fuse", "loading A2")
    assert len(log) == 1
    log.end("fuse")
    assert not log.busy


def test_subscribers_hear_every_change_and_a_late_one_hears_the_current_state(log):
    seen = []
    log.subscribe(lambda lg: seen.append(lg.sentence()))
    log.start("run", "MIP", total=2)
    log.advance("run", 1)
    log.end("run")
    assert seen == ["", "MIP · 0/2", "MIP · 1/2", ""]
    log.start("run", "MIP", total=9)
    log.advance("run", 4)
    late = []
    log.subscribe(lambda lg: late.append(lg.sentence()))
    assert late == ["MIP · 4/9"], "a late subscriber was not told what is already running"
