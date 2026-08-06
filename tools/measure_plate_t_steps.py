#!/usr/bin/env python
"""Step the PLATE's timepoint t=0 -> t=1 -> t=0 and report, per step, the time AND the pixels.

WHY BOTH. Timing this alone is how the bug it was written for hid: before 2026-08-05 the second
step was the FASTEST of the three, because ``_PreviewWorker``'s cell cache was keyed with no
timepoint and answered t=1 with t=0's cells. A pure benchmark would have called that a win. So
every step also reports a signature of the composited plate CELLS, and the last two lines are the
correctness claims -- "t=0 and t=1 differ" and "the revisit matches the first visit" -- without
which the milliseconds mean nothing.

WHAT IT DRIVES. The real ``_PreviewWorker`` over a real acquisition, through the real default cell
cache (``cache=_CACHE_AUTO``), with the RAM tier dropped between steps so each one is what a fresh
window would see. It runs the worker in-thread, so the number is the PASS and not the scheduler.

Works unmodified against a checkout that predates the fix: it detects whether ``_PreviewWorker``
takes a ``t`` and reports which shape it measured, which is what makes a before/after honest.

Usage::

    SQUIDMIP_CACHE_DIR=/tmp/plate-t python tools/measure_plate_t_steps.py ~/Downloads/sim_5d_2x2_t3
    REPEATS=9 SQUIDMIP_CACHE_DIR=/tmp/plate-t python tools/measure_plate_t_steps.py ACQ

``SQUIDMIP_CACHE_DIR`` is REQUIRED and gets a fresh suffix per repetition: this drives the real
cache path, and a measurement must not write into (or read from) the developer's own cells.

MEASURED on ``sim_5d_2x2_t3`` (4 regions x 4 FOVs x 2 channels, 256 px), median of 9, both columns
back to back on one machine with the OS page cache warm::

    step               before                              after
    t=0 first visit    13.9 ms, 4 wells read               13.6 ms, 4 wells read
    t=1                 6.4 ms, 4 HITS -- frame 0's cells  13.9 ms, 4 wells read
    t=0 again           5.3 ms, 4 hits                      7.0 ms, 4 hits
    t=0 != t=1          FALSE                               True
"""
from __future__ import annotations

import inspect
import os
import sys
import time

# THE CHECKOUT THIS FILE IS IN, not whichever squidmip is installed. Every other tool in here does
# the same line, and it is load-bearing for this one in particular: the editable install points at
# the main worktree, so without it a measurement run from a branch worktree silently measures MAIN
# -- which, for a script whose whole job is a before/after, would report the "before" twice.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                            # noqa: E402
from qtpy.QtWidgets import QApplication                       # noqa: E402

import squidmip._viewer as V                                  # noqa: E402
from squidmip import _platecache                              # noqa: E402
from squidmip.reader import open_reader                       # noqa: E402

CELL = 88                                                     # _workers._CELL
STEPS = (("t=0 (cold)", 0), ("t=1 (new)", 1), ("t=0 (revisit)", 0))


def one_pass(reader, meta, idx, order, t, takes_t):
    """One preview pass at *t*. Returns (seconds, hits, reads, {region: cell signature})."""
    worker = V._PreviewWorker(reader, meta, idx, order, **({"t": t} if takes_t else {}))
    cells: dict = {}

    def on_tile(_ri, _ci, region, tile, box=None):
        # Composited at its box, exactly as PlateOverview.add_tile does. A cold pass emits one
        # tile per FOV and a cache replay emits one per REGION (that is the reopen win), so the
        # raw emissions are not comparable between the two. The cell is.
        arr = np.asarray(tile)
        canvas = cells.get(region)
        if canvas is None:
            canvas = cells[region] = np.zeros((arr.shape[0], CELL, CELL), dtype=arr.dtype)
        top, left = (0, 0) if box is None else (int(box[0]), int(box[1]))
        canvas[:, top:top + arr.shape[1], left:left + arr.shape[2]] = arr

    worker.tileReady.connect(on_tile)
    t0 = time.perf_counter()
    worker.run()
    elapsed = time.perf_counter() - t0
    sig = {r: float(c.astype(np.float64).sum()) for r, c in cells.items()}
    return elapsed, worker.cache_hits, worker.cache_reads, sig


def main() -> int:
    fixture = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/sim_5d_2x2_t3")
    base = os.environ.get("SQUIDMIP_CACHE_DIR")
    if not base:
        print("set SQUIDMIP_CACHE_DIR to a disposable directory first (see the docstring)")
        return 2
    repeats = int(os.environ.get("REPEATS", "5"))
    takes_t = "t" in inspect.signature(V._PreviewWorker.__init__).parameters

    QApplication.instance() or QApplication([])
    reader = open_reader(fixture)
    meta = reader.metadata
    order = list(meta["regions"])
    idx = {r: {"rc": (i // 2, i % 2), "idx": i} for i, r in enumerate(order)}
    print(f"{fixture}: {len(order)} regions, n_t={meta.get('n_t')}   "
          f"_PreviewWorker takes t: {takes_t}   repeats: {repeats}")

    runs: dict = {label: [] for label, _ in STEPS}
    counts: dict = {}
    sigs: dict = {}
    for r in range(repeats):
        os.environ["SQUIDMIP_CACHE_DIR"] = f"{base}-{r}"      # a COLD cache per repetition
        for label, t in STEPS:
            _platecache.clear_memory_tier()   # every step is a fresh window's worth of RAM
            elapsed, hits, reads, sig = one_pass(reader, meta, idx, order, t, takes_t)
            runs[label].append(elapsed * 1000)
            counts[label], sigs[label] = (hits, reads), sig

    for label, _ in STEPS:
        v = sorted(runs[label])
        hits, reads = counts[label]
        print(f"  {label:<16} median {v[len(v) // 2]:7.1f} ms  (min {v[0]:.1f}, max {v[-1]:.1f})"
              f"   cache_hits={hits} cache_reads={reads}")
    print("  t=0 and t=1 show DIFFERENT pixels:", sigs["t=0 (cold)"] != sigs["t=1 (new)"])
    print("  the revisit shows the SAME pixels as the first visit:",
          sigs["t=0 (cold)"] == sigs["t=0 (revisit)"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
