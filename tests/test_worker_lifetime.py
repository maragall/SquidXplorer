"""A running QThread must never be destroyable: workers have no Qt parent, and a worker that will not stop in time is detached and kept referenced, never dropped."""

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
    """Reads the construction sites: a parented QThread is deleted by Qt with the parent."""
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
