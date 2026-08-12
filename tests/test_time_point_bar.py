"""The timepoint control: one definition, hidden when there is nothing to navigate.

Task 4b, 2026-07-29. The widget half of closing a SILENT CORRECTNESS BUG: every consumer read
timepoint 0 and presented it as the whole dataset, so a 40-timepoint plate was indistinguishable
from a 1-timepoint plate with no error anywhere. Nothing caught it because every fixture on this
machine was ``Nt = 1``, so the bug was invisible by construction.

This file tests the CONTROL in isolation and deliberately imports nothing heavy: no napari, no
`PlateWindow`, no acquisition fixture. That is not laziness, it is why this file can run at all. The
original GUI-level test file for this feature aborts the interpreter on import, because it pulls
napari in through `test_viewer`'s fixture chain, and an abort takes pytest's summary line with it,
which is the exact failure mode that hid 51 failures in this repo for weeks. A control this small
should be provable without that.

What is deliberately NOT covered here, and is honestly still unwired: the bar being mounted into
`PlateWindow` and each `RegionViewer`, and the three preview reads that hardcode ``[0, :, 0]``. The
widget exists and is correct; the wiring is not done, and `tests/test_time_point.py` still documents
the read bug as live.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication  # noqa: E402

from squidxplorer._time_point import TimePointBar  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    # Held in a module global by the fixture cache's own reference AND returned: see
    # tests/test_window_lifetime.py for why a QApplication owned only by a fixture cache segfaults.
    return QApplication.instance() or QApplication([])


def test_a_single_timepoint_bar_is_hidden(qapp):
    """A slider with one position is clutter, so it is built and hidden rather than not built.

    Built-and-hidden keeps every call site unconditional: no caller has to ask whether the control
    exists before talking to it, and `isHidden()` becomes the one honest question about it.
    """
    bar = TimePointBar()
    bar.set_count(1)
    assert bar.isHidden()
    assert bar.count == 1


def test_a_multi_timepoint_bar_is_shown_and_sized_to_the_series(qapp):
    bar = TimePointBar()
    bar.set_count(3)
    assert not bar.isHidden()
    assert bar.count == 3
    assert bar.slider.maximum() == 2, "the slider is not the length of what it navigates"


def test_the_control_is_named_squids_way(qapp):
    """The naming law: a timepoint exists in the microscope, so it takes Squid's exact spelling."""
    bar = TimePointBar()
    bar.set_count(3)
    assert "time_point" in bar.label.text(), f"not Squid's word: {bar.label.text()!r}"
    assert "t " not in bar.label.text().replace("time_point", ""), "abbreviated somewhere"


def test_the_label_says_where_in_the_series_you_are(qapp):
    bar = TimePointBar()
    bar.set_count(3)
    bar.set_time_point(1)
    text = bar.label.text()
    assert "2" in text and "3" in text, f"the label does not locate you: {text!r}"


def test_a_user_gesture_fires_the_callback(qapp):
    seen = []
    bar = TimePointBar(on_change=seen.append)
    bar.set_count(3)
    bar.set_time_point_from_user(2)
    assert seen == [2]
    assert bar.time_point == 2


def test_a_PROGRAMMATIC_move_does_NOT_fire_the_callback(qapp):
    """The distinction that stops a control fighting the thing it follows.

    Same rule as the plate's contrast sink, and for the same reason: treating our own write as a
    user gesture is what "latched every channel MANUAL on open and killed the plate's running
    auto-contrast from the first frame". A bar that echoes its own programmatic moves would loop.
    """
    seen = []
    bar = TimePointBar(on_change=seen.append)
    bar.set_count(3)
    bar.set_time_point(2)
    assert seen == [], "following someone else was reported as a user gesture"
    assert bar.time_point == 2, "the bar did not move"


def test_resizing_down_clamps_the_position(qapp):
    """Re-ingesting a shorter acquisition must not leave the bar past the end."""
    bar = TimePointBar()
    bar.set_count(5)
    bar.set_time_point(4)
    bar.set_count(2)
    assert bar.time_point <= 1
    assert bar.slider.maximum() == 1


def test_resizing_does_not_fire_the_callback(qapp):
    """An ingest is not a gesture."""
    seen = []
    bar = TimePointBar(on_change=seen.append)
    bar.set_count(5)
    bar.set_time_point(4)
    seen.clear()
    bar.set_count(2)
    assert seen == [], "a re-ingest was reported as a user gesture"


def test_a_gesture_past_the_end_is_clamped_not_refused(qapp):
    seen = []
    bar = TimePointBar(on_change=seen.append)
    bar.set_count(3)
    bar.set_time_point_from_user(99)
    assert bar.time_point == 2
    assert seen == [2]
