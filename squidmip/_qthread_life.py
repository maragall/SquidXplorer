"""The QThread OWNERSHIP rule: what to do with a worker that will not stop in time.

Extracted from ``_plate_overview`` on 2026-08-13, unchanged, when the canvas loupe
(``squidmip._napari_loupe``) needed the same rule. It could not simply be imported from there:
``_plate_overview`` is a 3000-line QWidget module that the loupe engine must not depend on, and it
could not go in ``_workers`` either, because ``_workers`` imports ``_plate_overview`` and that is
the cycle again. It is not a plate rule and it is not a loupe rule — it is a rule about owning a
``QThread`` — so it lives on its own, below both.

Three workers obey it today: ``_plate_overview._TileFetcher``, the plate's ``_LoupeWorker`` and the
canvas loupe's. Every one of them can be asked to stop in the middle of a decode that cannot be
interrupted, which is precisely the situation this exists for.
"""

from __future__ import annotations

from squidmip._logpane import get_logger

log = get_logger("qthread_life")

#: Workers that outlived the join that asked them to stop. Parking one here is the whole
#: mechanism behind :func:`detach`; see its docstring for why a set and not a `wait()`.
_DETACHED: "set" = set()


def detach(worker) -> None:
    """Cut a still-running worker loose instead of destroying it. The ownership rule.

    A ``QThread`` whose C++ half is destroyed while ``isRunning()`` calls ``qFatal`` — the process
    aborts, with no Python traceback and no chance to report anything. Measured on 2026-08-06 with
    a 20-line script (`QThread` parented to a `QWidget`, started, parent dropped, `gc.collect`):
    ``QThread: Destroyed while thread is still running``, exit code 134.

    That was reachable two ways, and both are closed by this function rather than by a longer
    timeout — a longer wait is a bet, and the losing side of the bet is the whole process:

    * ``_TileFetcher`` used to be PARENTED to the overview, so Qt deleted it whenever the widget
      was destroyed, whether or not anyone had stopped it. Three call sites destroy the overview
      and only one of them joined the thread first. It is now unparented, so Qt cannot;
    * ``_LoupeWorker`` is unparented but its ONLY reference was the ``_loupe_worker`` slot, which
      was overwritten on a ``wait()`` timeout — dropping the last reference to a running thread,
      which is the same crash by the other door.

    So: a worker that will not stop in time is *reparented to nobody and kept referenced* until it
    finishes on its own, at which point it removes itself. The cost of a straggler is one idle
    thread and its buffers for as long as its current read takes; the cost of the alternative is
    SIGABRT. Nothing waits on this set — waiting is what we are declining to do.
    """
    if worker is None:
        return
    try:
        if not worker.isRunning():
            return
        worker.setParent(None)          # Qt must not delete it on our behalf
    except RuntimeError:                # C++ half already gone: nothing of ours to keep alive
        return
    _DETACHED.add(worker)
    log.warning("%s did not stop in time; detached rather than destroyed (it is still reading)",
                type(worker).__name__)
    worker.finished.connect(lambda w=worker: _DETACHED.discard(w))
