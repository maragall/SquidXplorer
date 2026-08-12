"""A running ``QThread`` must never be destroyable. The ownership rule, pinned.

The failure this prevents
------------------------
Qt's ``~QThread`` calls ``qFatal`` when ``isRunning()``: the process aborts with no Python
traceback, no pytest summary, and no chance to report anything. Measured on 2026-08-06 with a
20-line script (a ``QThread`` parented to a ``QWidget``, started, the parent dropped,
``gc.collect()``): ``QThread: Destroyed while thread is still running``, exit code 134.

It was reachable in this app because ``PlateOverview`` owns two worker threads whose only stop was
somebody ELSE's ``closeEvent``. Reproduced end to end: ``pytest tests/test_integration.py
tests/test_nav_wiring.py`` in one process aborted the interpreter — the first nav test failed
before its ``shutdown_plate_window``, leaving a live ``_TileFetcher`` and ``_LoupeWorker``, and
the next test's ``gc.collect()`` reaped them. **69 of the 100 tests in Linux CI's chunk 24 never
ran** for the same species of reason. After the fix the same command runs all 30 tests and REPORTS
the one real failure instead of taking the process down with it.

So these do not test "does close() work". They test the two structural facts that make the abort
impossible even when nobody closes anything:

1. no worker QThread is a Qt CHILD of the widget (a child is deleted by Qt, running or not);
2. a worker that will not stop in time is DETACHED and kept referenced, never dropped.
"""

from __future__ import annotations

import os
import sys
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded — Qt binding conflict", allow_module_level=True)

from squidxplorer import _plate_overview as PO  # noqa: E402

from .test_viewer import qapp  # noqa: E402,F401  (fixture)


class _NeverStops(PO.QThread):
    """A worker that ignores ``stop()`` for a bounded time — i.e. every real long read."""

    def __init__(self):
        super().__init__()
        self._release = threading.Event()

    def stop(self):
        pass                      # deliberately deaf: this is what a wait() timeout looks like

    def run(self):
        self._release.wait(10)


def test_the_overview_builds_its_workers_with_no_qt_parent(qapp):
    """A parented QThread is deleted by Qt when the parent goes, running or not.

    ``_TileFetcher`` used to be built as ``_TileFetcher(self._tile_src, self)``. Three call sites
    destroy a ``PlateOverview`` and only one of them stopped the thread first, so the other two
    handed a live thread to ``deleteLater()``.

    This reads the CONSTRUCTION SITE, not a hand-built instance: the defect was an argument at one
    call site, and a test that constructs its own object cannot see that argument. The scenario
    test — destroy the widget and observe — cannot be written at all, because the observation is
    ``SIGABRT`` and nothing survives to report it.
    """
    import inspect

    for owner, method in ((PO.PlateOverview, "set_tile_source"),
                          (PO.PlateOverview, "set_loupe_source")):
        src = inspect.getsource(getattr(owner, method))
        for worker in ("_TileFetcher(", "_LoupeWorker("):
            for line in src.splitlines():
                idx = line.find(worker)
                if idx < 0:
                    continue
                args = line[idx + len(worker):]
                assert "self)" not in args and "self," not in args, (
                    f"{method} parents a worker QThread to the widget: {line.strip()!r}. Qt will "
                    "delete it on the widget's destruction whether or not it is running, and "
                    "that is SIGABRT, not an exception.")


def test_a_worker_that_will_not_stop_is_detached_not_dropped(qapp):
    """The last reference to a running QThread must never be the one we overwrite."""
    worker = _NeverStops()
    worker.start()
    assert worker.isRunning()
    try:
        PO._detach(worker)
        assert worker in PO._DETACHED, (
            "a running worker was not parked; dropping its last reference frees a running "
            "QThread, which aborts the process")
        assert worker.parent() is None
    finally:
        worker._release.set()
        worker.wait(5000)


def test_a_detached_worker_removes_itself_once_it_finishes(qapp):
    """Parking must not be a leak: the set is a lifeline, not a graveyard."""
    worker = _NeverStops()
    worker.start()
    PO._detach(worker)
    assert worker in PO._DETACHED

    worker._release.set()
    assert worker.wait(5000), "the worker never finished"
    for _ in range(50):           # `finished` is delivered on this thread's event loop
        qapp.processEvents()
        if worker not in PO._DETACHED:
            break
    assert worker not in PO._DETACHED, (
        "a finished worker stayed parked — every stop-timeout would then leak one thread object")


def test_a_worker_that_stopped_cleanly_is_not_parked(qapp):
    """`_detach` on an already-finished thread is a no-op, so the normal path costs nothing."""
    worker = _NeverStops()
    worker.start()
    worker._release.set()
    assert worker.wait(5000)
    PO._detach(worker)
    assert worker not in PO._DETACHED


def test_the_overview_has_one_name_for_stopping_both_of_its_threads():
    """Two threads stopped in two places by two owners is how one of them got forgotten."""
    assert hasattr(PO.PlateOverview, "shutdown"), (
        "PlateOverview.shutdown() is the single call every destroyer makes; without it a caller "
        "has to remember clear_tile_source() AND set_loupe_source(None), and two of the three "
        "call sites remembered neither")
