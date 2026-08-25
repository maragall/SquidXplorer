"""The plate takes its views with it, after asking once."""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from squidxplorer._viewer import PlateWindow                             # noqa: E402


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


def test_the_view_count_is_read_off_the_manager_and_zero_means_no_question():
    win = _plate_with(4)
    assert PlateWindow._open_view_count(win) == 4
    win._viewer_manager.close(1)
    assert PlateWindow._open_view_count(win) == 3
    assert PlateWindow._confirm_close_all(_plate_with(0), 0) is True


def test_the_session_flag_suppresses_the_question_entirely(monkeypatch):
    """MUTATION: drop the `_warn_close_all` check in `_confirm_close_all` and this hangs on a modal box."""
    monkeypatch.setattr(PlateWindow, "_warn_close_all", False)
    win = _plate_with(3)
    assert PlateWindow._confirm_close_all(win, 3) is True


def test_a_torn_down_manager_reports_no_views_rather_than_raising():
    """`closeEvent` runs during teardown, where the manager may already be gone."""
    win = PlateWindow.__new__(PlateWindow)
    win._viewer_manager = None
    assert PlateWindow._open_view_count(win) == 0
