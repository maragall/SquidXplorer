"""The busy indicator's rules, tested without a window.

Spencer Schwarz: "responsiveness is important. And an indicator when its working."

The property under test is not "a bar appears". It is that the indicator CANNOT LIE: it is on
exactly while work is in flight, it never invents a percentage it does not have, and it cannot
be left running by a path that forgot to clear a flag -- because there is no flag, there is one
registry and every widget is a sink of it.
"""

from __future__ import annotations

import pytest

from squidxplorer._activity import ActivityLog


@pytest.fixture
def log():
    return ActivityLog()


def test_a_fresh_log_is_not_busy(log):
    assert not log.busy
    assert log.sentence() == ""
    assert log.current() is None


def test_starting_work_makes_it_busy_and_ending_it_stops(log):
    log.start("fuse", "loading mosaic")
    assert log.busy
    log.end("fuse")
    assert not log.busy, "the indicator is still on with nothing running"


def test_unknown_size_is_reported_as_unknown_not_as_zero_percent(log):
    """A mosaic fuse is ONE lazy graph, not N countable steps.

    A bar that shows 0/1 or 0% for it is inventing a denominator, and an invented denominator is
    believed: the user reads a stuck bar as a hung app. Unknown must stay unknown so the
    indicator can be indeterminate.
    """
    a = log.start("fuse", "loading mosaic")
    assert a.total is None
    assert not a.determinate
    assert a.sentence() == "loading mosaic …"


def test_known_size_counts(log):
    log.start("run", "MIP", total=28)
    log.advance("run", 3)
    assert log.current().sentence() == "MIP · 3/28"


def test_advance_can_revise_a_total_it_did_not_know_at_the_start(log):
    """The well count is known by the worker, not by the click that started it."""
    log.start("run", "MIP")
    assert not log.current().determinate
    log.advance("run", 1, 28)
    assert log.current().determinate
    assert log.current().sentence() == "MIP · 1/28"


def test_progress_for_work_that_already_ended_is_ignored_not_fatal(log):
    """A worker's last `progress` can be delivered after its `finished`. Raising there would
    turn a benign Qt delivery race into a crash inside a slot."""
    log.start("run", "MIP", total=4)
    log.end("run")
    log.advance("run", 4, 4)          # must not raise
    assert not log.busy


def test_ending_something_that_never_started_is_not_an_error(log):
    """Teardown paths call end() defensively; they must not raise on the way out."""
    log.end("never-started")
    assert not log.busy


def test_two_activities_are_both_tracked_and_the_bar_says_how_many(log):
    log.start("fuse", "loading mosaic")
    log.start("run", "MIP", total=28)
    log.advance("run", 5)
    assert len(log) == 2
    assert log.sentence() == "MIP · 5/28  (+1 more)"


def test_a_determinate_activity_is_preferred_for_display(log):
    """When only one line can be shown, show the one that can say something real."""
    log.start("fuse", "loading mosaic")            # indeterminate, started first
    log.start("run", "MIP", total=28)
    assert log.current().key == "run"


def test_restarting_the_same_key_replaces_it_rather_than_stacking(log):
    """A region change while the previous fuse is still draining fires start() twice.

    Stacking would need two end()s to clear one visible activity, and the second would never
    come -- the indicator would be stuck on over an idle app, which is the exact failure this
    design exists to prevent.
    """
    log.start("fuse", "loading A1")
    log.start("fuse", "loading A2")
    assert len(log) == 1
    log.end("fuse")
    assert not log.busy


def test_subscribers_hear_every_change(log):
    seen = []
    log.subscribe(lambda lg: seen.append(lg.sentence()))
    log.start("run", "MIP", total=2)
    log.advance("run", 1)
    log.end("run")
    assert seen == ["", "MIP · 0/2", "MIP · 1/2", ""]


def test_a_subscriber_added_late_is_told_the_current_state_immediately(log):
    """A widget built while work is already running must not start out blank and wrong."""
    log.start("run", "MIP", total=9)
    log.advance("run", 4)
    seen = []
    log.subscribe(lambda lg: seen.append(lg.sentence()))
    assert seen == ["MIP · 4/9"], "a late subscriber was not told what is already running"


def test_clear_stops_everything(log):
    log.start("a", "one")
    log.start("b", "two")
    log.clear()
    assert not log.busy
