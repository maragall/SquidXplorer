"""THE PLATE TAKES ITS VIEWS WITH IT, after asking once.

Julio, 2026-08-06: *"closing the plate window should close all the other windosws make sure that
you pop up the warning, with the 'don't show me this again'."*

Two facts, tested separately because they fail separately:

* the SWEEP -- closing the plate closes every region window. Without it Qt keeps the process alive
  (a ``RegionViewer`` is a top-level and nothing sets ``quitOnLastWindowClosed``), so the plate
  disappears while a headless remainder goes on holding the single-instance flock and the next
  launch is refused by a process with no visible plate.
* the PREFERENCE -- "don't show me this again" has to survive a restart, which means it has to
  reach disk. A checkbox that only lives in memory is a lie with a checkbox on it.

The dialog itself is deliberately NOT driven here: it is suppressed under the test harness
(``app.property("_squidxplorer_test")``) because a modal ``QMessageBox`` in a suite hangs with nobody
to dismiss it. What IS driven is everything either side of it -- ``_confirm_close_all``'s decision
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
    """It has to REACH DISK. The whole point of the checkbox is the next session."""
    path = tmp_path / "prefs.json"
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(path))

    assert _prefs.set(PlateWindow.WARN_CLOSE_ALL, False) is True
    assert json.loads(path.read_text()) == {PlateWindow.WARN_CLOSE_ALL: False}
    assert _prefs.get(PlateWindow.WARN_CLOSE_ALL, True) is False


def test_an_unwritable_location_is_reported_not_raised(tmp_path, monkeypatch):
    """A read-only home must cost a checkbox, never the application's ability to close.

    MUTATION: let `_prefs.set` propagate and this raises out of `closeEvent` -- an unhandled
    exception in a Qt event handler aborts the process (SIGABRT, no usable traceback).
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
    """A PlateWindow shell carrying only what the close path reads. Methods are called UNBOUND off
    the real class, so what is exercised is production code and not a restatement of it."""
    win = PlateWindow.__new__(PlateWindow)
    win._viewer_manager = _FakeManager(n_views)
    return win


def test_no_open_views_means_no_question_to_ask(tmp_path, monkeypatch):
    """The dialog must not appear when there is nothing to confirm -- the overwhelmingly common
    close, and the one where an extra click would be pure friction."""
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(tmp_path / "prefs.json"))
    win = _plate_with(0)
    assert PlateWindow._open_view_count(win) == 0
    assert PlateWindow._confirm_close_all(win, 0) is True


def test_the_preference_suppresses_the_question_entirely(tmp_path, monkeypatch):
    """Once "don't show me this again" is on disk, a plate with views closes without asking.

    MUTATION: drop the `_prefs.get` check in `_confirm_close_all` and this hangs on a modal box
    (or, under the harness, silently stops proving anything) -- so it is asserted through the
    return value, which is the decision the caller acts on.
    """
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(tmp_path / "prefs.json"))
    _prefs.set(PlateWindow.WARN_CLOSE_ALL, False)
    win = _plate_with(3)
    assert PlateWindow._confirm_close_all(win, 3) is True


def test_the_view_count_is_read_off_the_manager_not_tracked(tmp_path, monkeypatch):
    """Derived, never bookkept: a count kept alongside the manager is the drift this repo keeps
    deleting, and here it would decide whether the user is warned at all."""
    monkeypatch.setenv("SQUIDXPLORER_PREFS", str(tmp_path / "prefs.json"))
    win = _plate_with(4)
    assert PlateWindow._open_view_count(win) == 4
    win._viewer_manager.close(1)
    assert PlateWindow._open_view_count(win) == 3


def test_a_torn_down_manager_reports_no_views_rather_than_raising():
    """`closeEvent` runs during teardown, where the manager may already be gone. Raising there
    aborts the process rather than closing the window."""
    win = PlateWindow.__new__(PlateWindow)
    win._viewer_manager = None
    assert PlateWindow._open_view_count(win) == 0
