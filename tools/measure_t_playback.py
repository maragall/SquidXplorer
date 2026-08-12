#!/usr/bin/env python
"""Measure timepoint playback in a real window: achieved rate, dropped frames, UI stalls, memory.

Drives the real RegionViewer / napari AnimationThread / _MosaicWorker on a real acquisition.

Usage::

    python tools/measure_t_playback.py ~/Downloads/sim_5d_2x2_t3
    python tools/measure_t_playback.py ACQ --fps 2,5,10,20 --seconds 6 --blocking-supersede
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:                                # noqa: BLE001 - measurement, never fatal
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


class _UiWatch:
    """Worst blockage of the GUI thread, measured by a timer living on it."""

    def __init__(self, parent) -> None:
        from qtpy.QtCore import QTimer

        self.gaps: list[float] = []
        self._last = time.perf_counter()
        self._timer = QTimer(parent)
        self._timer.setInterval(5)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        now = time.perf_counter()
        self.gaps.append(now - self._last)
        self._last = now

    def start(self) -> None:
        self.gaps.clear()
        self._last = time.perf_counter()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    @property
    def worst_ms(self) -> float:
        return max(self.gaps) * 1e3 if self.gaps else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.gaps:
            return 0.0
        ordered = sorted(self.gaps)
        return ordered[int(0.95 * (len(ordered) - 1))] * 1e3


def _pump(app, seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.002)


def measure(root: Path, fps_list, seconds: float, blocking: bool) -> int:
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    from squidxplorer import open_reader
    from squidxplorer._napari_pane import gl_available
    from squidxplorer._region_viewer import RegionViewer, ViewerManager

    ok, why = gl_available()
    print(f"napari canvas: {'REAL (GL available)' if ok else f'NOT AVAILABLE — {why}'}")

    reader = open_reader(root)
    meta = reader.metadata
    regions = list(meta["fovs_per_region"])
    n_t = int(meta.get("n_t", 1) or 1)
    print(f"acquisition: {root}")
    print(f"  {len(regions)} regions, {len(meta['channels'])} channels, "
          f"{len(meta['z_levels'])} z, n_t={n_t}, frame {meta.get('frame_shape')}")
    if n_t < 2:
        print("REFUSING: this acquisition has one timepoint. There is nothing to play.")
        return 2

    if blocking:
        # reproduce the pre-cancellation behaviour: block the GUI thread until the superseded
        # read gives up, kept as a flag so the improvement stays measurable
        RegionViewer._retire_worker = lambda self, worker: worker.wait(2000)   # type: ignore
        print("MODE: blocking supersede (the OLD behaviour), for comparison")

    mgr = ViewerManager(reader, meta)
    win = mgr.open([regions[0]])
    win.show()
    bar = win._time_point_bar
    if bar.playback is None:
        print("REFUSING: this window's timepoint bar has no playback.")
        return 2

    # `queued` counts a load started while another was still in flight, the failure the gate
    # exists to prevent; it must stay 0.
    requested: list[float] = []
    stepped: list[tuple[float, int]] = []
    landed: list[float] = []
    inflight = {"n": 0, "queued": 0}

    thread = bar.playback.qt_dims._animation_thread
    if thread is not None:
        thread.frame_requested.connect(lambda *_a: requested.append(time.perf_counter()))

    real_on_change = win._on_time_point_changed

    def _on_change(t):
        stepped.append((time.perf_counter(), int(t)))
        if inflight["n"] > 0:
            inflight["queued"] += 1
        inflight["n"] += 1
        real_on_change(t)

    bar._on_change = _on_change

    real_frame_done = win._frame_done

    def _frame_done():
        landed.append(time.perf_counter())
        inflight["n"] = max(0, inflight["n"] - 1)
        real_frame_done()

    win._frame_done = _frame_done

    watch = _UiWatch(win)
    _pump(app, 3.0)                                  # let the first mosaic land before timing

    print()
    print(f"{'req fps':>8} {'steps':>6} {'drop':>5} {'queued':>7} {'interval ms':>22} "
          f"{'achieved fps':>13} {'UI p95/worst ms':>17}  note")
    rss0 = _rss_mb()
    for fps in fps_list:
        # a window that is not active halts its own playback (RegionViewer.set_active), so
        # reclaim focus before timing and flag it in the row if it was lost anyway
        win.raise_()
        win.activateWindow()
        app.processEvents()
        requested.clear()
        stepped.clear()
        landed.clear()
        inflight["queued"] = 0
        watch.start()
        bar.play(fps=fps)
        _pump(app, seconds)
        still_playing = bar.is_playing
        bar.stop()
        _pump(app, 0.3)
        watch.stop()

        gaps = [b - a for a, b in zip(landed, landed[1:])]
        n_req, n_step = len(requested), len(stepped)
        interval = (f"{statistics.median(gaps) * 1e3:7.0f} med  {min(gaps) * 1e3:5.0f}-"
                    f"{max(gaps) * 1e3:5.0f}" if gaps else "        (none landed)")
        achieved = f"{1.0 / statistics.median(gaps):12.2f}" if gaps else "           -"
        note = "" if still_playing else "  window lost focus: playback halted itself"
        print(f"{fps:>8} {n_step:>6} {max(0, n_req - n_step):>5} {inflight['queued']:>7} "
              f"{interval:>22} {achieved} {watch.p95_ms:>8.0f} /{watch.worst_ms:>7.0f}{note}")

    print()
    print("memory over continuous playback (RSS MB / frames landed):")
    landed.clear()
    watch.start()
    bar.play(fps=max(fps_list))
    trace = []
    for _ in range(8):
        _pump(app, 5.0)
        # a window that loses focus stops its own playback on purpose; a flat frame count next
        # to "playing no" is the halt working, not a hang
        trace.append((_rss_mb(), len(landed), bar.is_playing))
    bar.stop()
    _pump(app, 0.5)
    watch.stop()
    print(f"  start {rss0:6.0f}")
    for rss, frames, playing in trace:
        print(f"        {rss:6.0f}   after {frames:4d} frames   playing={'yes' if playing else 'NO'}")
    import gc

    gc.collect()
    _pump(app, 0.5)
    print(f"  after gc.collect(): {_rss_mb():.0f} MB")

    mgr._mem_timer.stop()
    mgr.close_all()
    for _ in range(30):
        app.processEvents()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acquisition", type=Path)
    ap.add_argument("--fps", default="2,5,10,20", help="requested rates to try")
    ap.add_argument("--seconds", type=float, default=6.0, help="how long to play at each rate")
    ap.add_argument("--blocking-supersede", action="store_true",
                    help="restore the old blocking join, to measure what it cost")
    args = ap.parse_args(argv)
    fps_list = [float(x) for x in str(args.fps).split(",") if x.strip()]
    return measure(args.acquisition, fps_list, args.seconds, args.blocking_supersede)


if __name__ == "__main__":                           # pragma: no cover
    raise SystemExit(main())
