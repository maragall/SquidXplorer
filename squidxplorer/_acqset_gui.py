"""The QThread that drives ``_acqset.run_over_set`` off the Qt thread.

Its own module: ``_acqset`` stays Qt-free and ``_workers`` is not this feature's file. The
worker holds ITS OWN readers (one per acquisition, opened and dropped inside the loop), so the
plate window may re-ingest or cycle while a set run is in flight without pulling the reader out
from under it. Per-acquisition lines go straight to the log bus; logging is thread-safe.
"""

from __future__ import annotations

import threading

from qtpy.QtCore import QThread, Signal

from squidxplorer._logpane import get_logger

log = get_logger("acqset")


class SetRunWorker(QThread):
    """One SAVE of one operator over every acquisition of a set, sequentially."""

    done = Signal(dict)      # the run_over_set summary
    problem = Signal(str)    # a failure of the LOOP itself; one acquisition failing is not one

    def __init__(self, paths, operator: str, parameters, out_parent, parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._operator = operator
        self._parameters = dict(parameters) if parameters else None
        self._out_parent = out_parent
        self._stop = threading.Event()

    def stop(self) -> None:
        """Ask the loop to end; run_over_set polls this between fields and acquisitions."""
        self._stop.set()

    def run(self) -> None:
        from squidxplorer._acqset import run_over_set

        try:
            summary = run_over_set(
                self._paths, operator=self._operator, out_parent=self._out_parent,
                parameters=self._parameters, log=log.info, stop=self._stop.is_set)
        except Exception as exc:   # noqa: BLE001, named to the window, never swallowed
            self.problem.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(summary)
