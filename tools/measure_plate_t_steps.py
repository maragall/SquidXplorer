#!/usr/bin/env python
"""Step the plate's timepoint t=0 -> t=1 -> t=0, reporting per-step time and cell signatures.

Usage: SQUIDXPLORER_CACHE_DIR=/tmp/plate-t python tools/measure_plate_t_steps.py <acquisition>
"""
from __future__ import annotations

import inspect
import os
import sys
import time

# Measure the checkout this file is in, not whichever squidxplorer is installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                            # noqa: E402
from qtpy.QtWidgets import QApplication                       # noqa: E402

import squidxplorer._workers as W                                 # noqa: E402
from squidxplorer import _platecache                              # noqa: E402
from squidxplorer.reader import open_reader                       # noqa: E402

CELL = 88   # _workers._CELL
STEPS = (("t=0 (cold)", 0), ("t=1 (new)", 1), ("t=0 (revisit)", 0))


def one_pass(reader, meta, idx, order, t, takes_t):
    """One preview pass at *t*. Returns (seconds, hits, reads, {region: cell signature})."""
    worker = W._PreviewWorker(reader, meta, idx, order, **({"t": t} if takes_t else {}))
    cells: dict = {}

    def on_tile(_ri, _ci, region, tile, box=None):
        # Composite at its box, as PlateOverview.add_tile does; cells are comparable, emissions not.
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
    base = os.environ.get("SQUIDXPLORER_CACHE_DIR")
    if not base:
        print("set SQUIDXPLORER_CACHE_DIR to a disposable directory first (see the docstring)")
        return 2
    repeats = int(os.environ.get("REPEATS", "5"))
    takes_t = "t" in inspect.signature(W._PreviewWorker.__init__).parameters

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
        os.environ["SQUIDXPLORER_CACHE_DIR"] = f"{base}-{r}"      # a cold cache per repetition
        for label, t in STEPS:
            _platecache.clear_memory_tier()   # every step sees a fresh window's RAM
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
