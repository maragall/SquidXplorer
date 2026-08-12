"""The plate takes its views with it, after asking once.

Two facts, tested separately because they fail separately: closing the plate must close every
region window (Qt has no `quitOnLastWindowClosed` here, so otherwise the process outlives the
plate and holds the single-instance flock), and "don't show me this again" must survive a
restart, so it has to reach disk.

The dialog itself is deliberately NOT driven here: it is suppressed under the test harness
(`app.property("_squidxplorer_test")`) because a modal `QMessageBox` in a suite hangs with nobody
to dismiss it. What IS driven is everything either side of it -- `_confirm_close_all`'s decision
table, and the sweep it gates.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("qtpy")

from squidxplorer import _prefs                                          # noqa: E402
from squidxplorer._viewer import PlateWindow                             # noqa: E402


# --------------------------------------------------------------------------- the preference file


def test_a_preference_that_was_never_set_reads_the_callers_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(tmp_path / "prefs.json"))
    assert _prefs.get(PlateWindow.WARN_CLOSE_ALL, True) is True
    assert _prefs.get("nothing-has-ever-set-this", "fallback") == "fallback"


def test_setting_a_preference_puts_it_on_disk_and_reads_back(tmp_path, monkeypatch):
    path = tmp_path / "prefs.json"
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(path))

    assert _prefs.set(PlateWindow.WARN_CLOSE_ALL, False) is True
    assert json.loads(path.read_text()) == {PlateWindow.WARN_CLOSE_ALL: False}
    assert _prefs.get(PlateWindow.WARN_CLOSE_ALL, True) is False


def test_an_unwritable_location_is_reported_not_raised(tmp_path, monkeypatch):
    """A read-only home must cost a checkbox, never the ability to close.

    MUTATION: let `_prefs.set` propagate and this raises out of `closeEvent`, aborting the process.
    """
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(blocker / "prefs.json"))
    assert _prefs.set(PlateWindow.WARN_CLOSE_ALL, False) is False


def test_a_corrupt_preferences_file_falls_back_instead_of_failing_the_launch(tmp_path, monkeypatch):
    path = tmp_path / "prefs.json"
    path.write_text("{ this is not json")
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(path))
    assert _prefs.get(PlateWindow.WARN_CLOSE_ALL, True) is True


# ------------------------------------------------------------------- what the confirmation decides


class _FakeManager:
    def __init__(self, n: int) -> None:
        self._open = list(range(1, n + 1))
        self.closed: list[int] = []

    @property
    def windows(self):
        return [type("W", (), {"window_id": i})() for i in self._open]

    def close(self, wid: int) -> None:
        self.closed.append(int(wid))
        if int(wid) in self._open:
            self._open.remove(int(wid))


def _plate_with(n_views: int) -> PlateWindow:
    """A PlateWindow shell carrying only what the close path reads."""
    win = PlateWindow.__new__(PlateWindow)
    win._viewer_manager = _FakeManager(n_views)
    return win


def test_no_open_views_means_no_question_to_ask(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(tmp_path / "prefs.json"))
    win = _plate_with(0)
    assert PlateWindow._open_view_count(win) == 0
    assert PlateWindow._confirm_close_all(win, 0) is True


def test_the_preference_suppresses_the_question_entirely(tmp_path, monkeypatch):
    """MUTATION: drop the `_prefs.get` check in `_confirm_close_all` and this hangs on a modal box."""
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(tmp_path / "prefs.json"))
    _prefs.set(PlateWindow.WARN_CLOSE_ALL, False)
    win = _plate_with(3)
    assert PlateWindow._confirm_close_all(win, 3) is True


def test_the_view_count_is_read_off_the_manager_not_tracked(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(tmp_path / "prefs.json"))
    win = _plate_with(4)
    assert PlateWindow._open_view_count(win) == 4
    win._viewer_manager.close(1)
    assert PlateWindow._open_view_count(win) == 3


def test_a_torn_down_manager_reports_no_views_rather_than_raising():
    """`closeEvent` runs during teardown, where the manager may already be gone."""
    win = PlateWindow.__new__(PlateWindow)
    win._viewer_manager = None
    assert PlateWindow._open_view_count(win) == 0
