"""Destroying a PlateWindow, and ending a test module that built one, must not corrupt the heap.

IMA regression, 2026-07-28. The whole pytest run died with SIGSEGV partway through
``tests/test_gui_commands.py``, taking the summary line with it -- so a run that crashed and a
run that passed were indistinguishable, on macOS here and as ``0xC0000005`` on Windows.

BUG 1: THE PER-WINDOW QSTYLE
----------------------------
Root cause, bisected: ``QStyleFactory.create("Fusion")`` returns a QStyle that PYTHON owns, and
``QWidget.setStyle()`` does NOT take ownership -- the widget keeps a bare pointer and Qt never
tells it when that style dies. The window stored the only reference on ``self``, handed it to
four widgets (one of them ``self._log_window``, a separate TOP-LEVEL, not even a child -- that one
is gone since 2026-07-29, the log is a fixed tab now, but the hazard below is unchanged), and
then, when the last Python reference to the window went away, the attribute died with the
``__dict__`` while ``~QWidget`` was still to run. Every widget's destructor calls
``style()->unpolish(this)``, i.e. through freed memory.

The evidence that pinned it: with the created QStyle held alive artificially, 40 windows built
and destroyed cleanly; without, the interpreter died on the 6th. ``close()`` was measured to be
irrelevant -- holding a Python reference and never closing survived, closing and dropping the
reference crashed -- because the trigger is destruction, not closing.

BUG 2: THE QAPPLICATION HELD ONLY BY A PYTEST FIXTURE CACHE
-----------------------------------------------------------
With the style fixed, ``tests/test_gui_commands.py`` ran all 15 tests green and then STILL
segfaulted, inside pytest's own ``runtestprotocol``. The crashing line is
``_pytest/runner.py:147``, ``item.funcargs = None``: the moment pytest releases a test's fixture
values. It reproduced with a SINGLE test selected, so it was never about accumulation, and it
always struck on the LAST test of the module, which is the one teardown where the MODULE-scoped
fixtures are released too.

Root cause: every GUI test module holds the QApplication like this, and only like this::

    @pytest.fixture(scope="module")
    def qapp():
        return QApplication.instance() or QApplication([])

PyQt gives the QApplication to PYTHON to own. That fixture cache was the only Python reference in
the process, so at the last teardown pytest dropped it and the QApplication was deleted while
widgets, the shared QStyle and posted events still pointed at it. Same shape as bug 1, one level
up: a Python-owned Qt object handed to Qt-land and freed on someone else's schedule.
``tests/test_channel_bar.py`` had already been bitten by this and pinned its own copy in a module
global, with a comment saying so; the lesson had just never been moved into the library.

``main()`` had the same latent defect: it kept the app in a LOCAL and then returned the WINDOW.

Evidence. A window fixture parametrised to build, open, close and drop 60 windows per run, 12 runs
per configuration, macOS offscreen. The measure is whether pytest's summary line printed at all,
because that is exactly what the crash destroys:

    configuration                                            summary line printed
    ------------------------------------------------------  ---------------------
    as found                                                 0 / 12
    pin ONLY the QApplication alive, change nothing else     12 / 12
    pin every funcarg EXCEPT the QApplication                0 / 12

Bisected down to that one object: pinning the whole funcargs dict fixed it, pinning all of it but
the app did not, pinning the app alone did. The fix is ``squidmip._viewer.qt_app()``, which holds
the QApplication in a module global for the life of the process, exactly as ``_fusion_style()``
holds the style. ``PlateWindow.__init__`` calls it, so every window pins its own application
however that window was constructed.

BUG 3: THE REGION DEBOUNCE LEFT ARMED ACROSS THE CLOSE
------------------------------------------------------
Found while bisecting bug 2, in a standalone loop where the QApplication was already safe.
``PlateWindow._on_region_changed`` debounces the expensive mosaic fuse behind a single-shot
``QTimer`` armed for 140 ms, and NOTHING disarmed it. Two consequences, each independently
sufficient to corrupt the heap:

* the slot was ``lambda: self._load_mosaic(region=self._pending_region)``. PyQt keeps a lambda
  alive in a slot proxy parented to the sender, and the sender is parented to this window, so the
  closure's ``self`` closed the loop window -> timer -> proxy -> lambda -> window.
* the timer was still ACTIVE at close, so it fired ``_load_mosaic`` on a window that had already
  been closed and dropped. Observed directly, not inferred: instrumenting ``_load_mosaic`` to name
  its receiver printed a call against an already-closed window one iteration before the crash.

Measured in that loop, 40 windows per run: as found, the interpreter died at window 9, 13, 14 and
21 on four runs. Stopping the timer at close alone survived 40; using a bound-method slot alone
survived 40; both together survived 40 on 3 of 3 runs. Both are shipped, because they remove
different halves of the hazard, the zombie window and the armed callback. ``PlateOverview``
already stops its own coalescing timer in ``hideEvent`` for precisely this reason.

KNOWN AND NOT FIXED HERE
------------------------
Constructing a PlateWindow at all leaves the PROCESS exiting with status 139 about half the time,
in C++ with no Python frame, after pytest has printed its summary and finished. It is pre-existing
and independent of everything above: measured at 4 of 8 runs for a window that is merely built and
never even closed, and at the same 4 of 8 with the QApplication pin bypassed entirely. The cause
is that a PlateWindow is never freed at all: the window's own widgets connect dozens of
``self``-capturing lambdas, those links live in C++ where Python's cyclic collector cannot see
them, so ``gc.collect()`` frees nothing and the windows pile up until interpreter shutdown tears
them down against a half-finalised Qt. Plain Qt does the same thing with a menu bar, buttons and
lambda slots without crashing (0 of 8), so it is this window's object graph, not the toolkit.
That is a real leak and a real bug, but it is a different bug from the ones above and it does not
affect any test result, so it is recorded here rather than half-fixed.

These tests drive well past every observed failure point through the real constructor. A
regression does not fail an assertion here, it kills the interpreter.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication  # noqa: E402

import squidmip._viewer as V  # noqa: E402
from squidmip._command import OpenAcquisition  # noqa: E402


#: Comfortably past the 6-window failure point measured on macOS, so a reintroduction segfaults
#: rather than passing by luck. Cheap: these windows open no acquisition.
_N_WINDOWS = 25

#: Windows built through a REAL fixture teardown, which is the only way bug 2 shows up: the crash
#: is pytest releasing a test's funcargs, so a loop inside one test cannot reach it. Each of these
#: is a separate test item, and the last one is the one that also releases the module-scoped
#: ``qapp``. Measured unfixed at 0 summary lines out of 12 runs at this count.
_N_TEARDOWNS = 30


class _StubDetail:
    """The detail viewer is a heavy third-party pane and is not what this test is about."""

    def __getattr__(self, _name):
        return lambda *a, **k: None


@pytest.fixture(scope="module")
def qapp():
    # Deliberately written the UNSAFE way, the way every other GUI test module writes it: the
    # library must survive a caller who keeps the application only in a fixture cache. Pinning it
    # here would test the fixture instead of the fix.
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)
    return app


@pytest.fixture
def stub_detail(monkeypatch):
    monkeypatch.setattr(V.PlateWindow, "_make_detail_viewer", lambda self: _StubDetail())


def test_the_fusion_style_outlives_any_single_window(qapp, stub_detail):
    """The style must not be per-window state, or its lifetime is tied to the wrong object."""
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


def test_the_qapplication_is_pinned_for_the_process_not_by_the_caller(qapp, stub_detail):
    """Building a window must take the application out of the caller's hands.

    The concrete failure this prevents: pytest drops the module-scoped ``qapp`` fixture at the last
    teardown, and with no other reference the QApplication is deleted underneath live widgets.
    """
    win = V.PlateWindow(None)
    try:
        assert V._APP is qapp, "the window did not pin the process's QApplication"
        assert V.qt_app() is qapp, "qt_app() handed back a different application"
    finally:
        win.close()


def test_many_windows_can_be_built_and_destroyed_in_one_process(qapp, stub_detail):
    """Bug 1: dropping the last reference to a window used to corrupt the heap."""
    for i in range(_N_WINDOWS):
        win = V.PlateWindow(None)
        assert win.menuBar() is not None, f"window {i} came up wrong"
        win.close()
        del win          # the trigger: PyQt deletes the C++ object here


def test_the_region_debounce_is_disarmed_and_owns_no_closure_over_the_window(qapp, stub_detail,
                                                                            squid_dataset):
    """Bug 3, both halves, asserted directly rather than only through a crash.

    A pending single-shot that survives the close will call back into a torn-down window, and a
    ``self``-capturing lambda on a timer parented to that same window is a reference cycle Python's
    collector cannot even see, because the link that closes it lives in C++.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    try:
        assert win.commands.execute(OpenAcquisition(path=str(root))).ok
        timer = win._region_load_timer
        assert timer.isActive(), "the region debounce never armed; this test proves nothing"
        assert win._fire_region_load.__self__ is win, \
            "the debounce slot must be a bound method, not a closure over the window"
        win.close()
        assert not timer.isActive(), "closeEvent left the region debounce armed past the close"
    finally:
        win.close()


# --- bug 2: the crash is pytest RELEASING the fixtures, so it needs real teardowns --------------
#
# One window per test item, deliberately. A loop inside a single test never reaches
# ``item.funcargs = None`` more than once and passed happily while the module still segfaulted.

@pytest.fixture
def win(qapp, stub_detail):
    w = V.PlateWindow(None)
    yield w
    w.close()


@pytest.mark.parametrize("i", range(_N_TEARDOWNS))
def test_a_window_survives_being_released_by_the_test_framework(win, squid_dataset, i):
    """Open an acquisition, then let pytest drop every fixture value. That release is the bug.

    The assertion is incidental. What matters is that the interpreter is still alive to run the
    next parameter, and that pytest lives long enough to print a summary line at the end.
    """
    root, _ = squid_dataset
    assert win.commands.execute(OpenAcquisition(path=str(root))).ok, f"window {i} did not open"
