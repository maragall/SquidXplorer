"""Destroying a PlateWindow, and ending a test module that built one, must not corrupt the heap."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication  # noqa: E402

import squidxplorer._viewer as V  # noqa: E402
from squidxplorer._command import OpenAcquisition  # noqa: E402


#: Comfortably past the 6-window failure point measured on macOS.
_N_WINDOWS = 25

#: Windows built through a real fixture teardown, which is the only way bug 2 shows up: the
#: crash is pytest releasing a test's funcargs, so a loop inside one test cannot reach it.
_N_TEARDOWNS = 30


@pytest.fixture(scope="module")
def qapp():
    # Deliberately unsafe, the way every other GUI test module writes it: the library must
    # survive a caller who keeps the application only in a fixture cache.
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


def test_the_fusion_style_outlives_any_single_window(qapp):
    first = V.PlateWindow(None)
    style_a = first._fusion_style
    if style_a is None:
        pytest.skip("no Fusion style on this Qt build")
    second = V.PlateWindow(None)
    try:
        assert second._fusion_style is style_a, "each window created its own QStyle"
    finally:
        first.close()
        second.close()


def test_the_qapplication_is_pinned_for_the_process_not_by_the_caller(qapp):
    win = V.PlateWindow(None)
    try:
        assert V._APP is qapp, "the window did not pin the process's QApplication"
        assert V.qt_app() is qapp, "qt_app() handed back a different application"
    finally:
        win.close()


def test_many_windows_can_be_built_and_destroyed_in_one_process(qapp):
    for i in range(_N_WINDOWS):
        win = V.PlateWindow(None)
        titles = [a.text() for a in win.menuBar().actions()]
        assert titles, f"window {i} came up with an empty menu bar"
        win.close()
        del win          # the trigger: PyQt deletes the C++ object here


def test_no_timer_on_the_plate_window_survives_its_close(qapp, squid_dataset):
    """Whatever timers this window owns, none may still be running once it is closed."""
    from qtpy.QtCore import QTimer

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    try:
        assert win.commands.execute(OpenAcquisition(path=str(root))).ok
        qapp.processEvents()
        win.close()
        qapp.processEvents()
        still_running = [repr(t) for t in win.findChildren(QTimer)
                         if t.parent() is win and t.isActive()]
        assert not still_running, (
            f"closeEvent left {len(still_running)} timer(s) armed past the close: {still_running}")
    finally:
        win.close()


# bug 2 needs real per-test teardowns: a loop inside one test never reaches
# `item.funcargs = None` more than once.

@pytest.fixture
def win(qapp):
    w = V.PlateWindow(None)
    yield w
    w.close()


@pytest.mark.parametrize("i", range(_N_TEARDOWNS))
def test_a_window_survives_being_released_by_the_test_framework(win, squid_dataset, i):
    """The assertion is incidental; what matters is that the interpreter and pytest survive the fixture release that follows."""
    root, _ = squid_dataset
    assert win.commands.execute(OpenAcquisition(path=str(root))).ok, f"window {i} did not open"
