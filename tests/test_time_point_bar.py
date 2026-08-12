"""The timepoint control: one definition, hidden when there is nothing to navigate."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication  # noqa: E402

from squidxplorer._time_point import TimePointBar  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_a_single_timepoint_bar_is_hidden(qapp):
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
    seen = []
    bar = TimePointBar(on_change=seen.append)
    bar.set_count(3)
    bar.set_time_point(2)
    assert seen == [], "following someone else was reported as a user gesture"
    assert bar.time_point == 2, "the bar did not move"


def test_resizing_down_clamps_the_position(qapp):
    bar = TimePointBar()
    bar.set_count(5)
    bar.set_time_point(4)
    bar.set_count(2)
    assert bar.time_point <= 1
    assert bar.slider.maximum() == 1


def test_resizing_does_not_fire_the_callback(qapp):
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
