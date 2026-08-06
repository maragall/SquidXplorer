"""The window half of the .mp4 export: the chip, what it hands the worker, and what lands on disk.

``tests/test_video.py`` pins the recorder itself, Qt-free. This file pins the two things only the
window can get wrong, and both of them have precedent in this repo:

* **The click handler doing the work.** ``_FocusWorker``'s docstring records that the reference
  plane scan "used to run inside the button's clicked slot, which froze the window solid for the
  whole scan", and ``_MosaicWorker``'s records 128 ms of frozen UI for one contrast seed. A movie
  is 4 s of reads and encoding on the real 10x region. So the handler's own wall clock is
  ASSERTED here, with a real worker started, rather than left as an intention in a comment.
* **Recording something other than what is on screen.** The window knows the region, the visible
  channels, the timepoint and the z plane; the recorder knows none of them. A test that only
  checked "a worker was started" would pass on a movie of region A1 taken while looking at B2.

Both end in a decoded file: the last test runs the REAL worker to completion and reads the frames
back out of the .mp4 it wrote.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded — Qt binding conflict", allow_module_level=True)

from squidmip import _viewer as V  # noqa: E402
from squidmip._video import encoder_problem  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_video import _decode, _make_5d  # noqa: E402
from .test_viewer import _drain_until, qapp, stub_detail  # noqa: E402,F401  (fixtures)


@pytest.fixture
def five_d_root(tmp_path):
    """A tiny 5-D acquisition on disk: 2 regions x 1 FOV x 2 z x 2 ch x 3 t, 64 px.

    Two regions so "the region on screen" is a question with a wrong answer available.
    """
    root = tmp_path / "acq5d"
    _make_5d().build(root, ["A1", "A2"], n_fovs=1, nz=2, nt=3, size=64)
    return root


def _open_window(qapp, root, region_index=0):
    """A real ``RegionViewer`` over a real acquisition, the way the plate opens one."""
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = win._viewer_manager.open(list(win._order))
    assert w is not None, "no window was opened"
    if region_index:
        w._cursor.set_index(region_index)
        _drain_until(qapp, lambda: w.current_region() == win._order[region_index], timeout=5)
    return win, w


class _SpyWorker:
    """Stands in for ``_VideoWorker``: records its arguments, never touches a plane."""

    instances: list = []

    def __init__(self, reader, meta, region, out_path, **kw):
        self.reader, self.meta, self.region, self.out_path = reader, meta, region, out_path
        self.kw = kw
        self.started = False
        self.stopped = False
        _SpyWorker.instances.append(self)

    # the signal surface the handler wires
    class _Sig:
        def connect(self, *_a, **_k):
            pass

    progress = done = problem = cancelled = finished = _Sig()

    def start(self):
        self.started = True

    def isRunning(self):
        return self.started and not self.stopped

    def stop(self):
        self.stopped = True

    def wait(self, _ms=0):
        return True

    def deleteLater(self):
        pass


@pytest.fixture
def spy_worker(monkeypatch):
    _SpyWorker.instances = []
    monkeypatch.setattr(V, "_VideoWorker", _SpyWorker)
    return _SpyWorker


@pytest.fixture
def save_dialog(monkeypatch, tmp_path):
    """``QFileDialog.getSaveFileName`` answering with a path, so nothing modal blocks the run."""
    from qtpy.QtWidgets import QFileDialog

    chosen = {"path": str(tmp_path / "movie.mp4")}
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (chosen["path"], "")))
    return chosen


# --- the chip -------------------------------------------------------------------------------

def test_the_record_chip_is_enabled_on_an_acquisition_with_an_axis_to_sweep(
        qapp, stub_detail, napari_pane_stub, five_d_root):
    win, w = _open_window(qapp, five_d_root)
    assert w._btn_record.isEnabled(), "the record chip is dead on a 3-timepoint acquisition"
    assert "3 frames along the time axis" in w._btn_record.toolTip(), w._btn_record.toolTip()
    shutdown_plate_window(qapp, win)


def test_the_record_chip_says_why_when_there_is_nothing_to_sweep(
        qapp, stub_detail, napari_pane_stub, tmp_path):
    """n_t = 1 and one z plane: no axis, and the tooltip names that rather than going quiet."""
    root = tmp_path / "flat"
    _make_5d().build(root, ["A1"], n_fovs=1, nz=1, nt=1, size=64)
    win, w = _open_window(qapp, root)
    assert not w._btn_record.isEnabled()
    assert "single timepoint and a single z plane" in w._btn_record.toolTip()
    shutdown_plate_window(qapp, win)


# --- what the handler hands over --------------------------------------------------------------

def test_the_worker_is_given_the_region_the_window_is_actually_showing(
        qapp, stub_detail, napari_pane_stub, five_d_root, spy_worker, save_dialog):
    """A movie of A1 taken while looking at A2 would pass any state-only assertion."""
    win, w = _open_window(qapp, five_d_root, region_index=1)
    assert w.current_region() == "A2", "the fixture did not move; this test asserts nothing"

    w._record_movie()

    assert len(spy_worker.instances) == 1
    spy = spy_worker.instances[0]
    assert spy.region == "A2", f"recorded {spy.region}, window is showing {w.current_region()}"
    assert spy.kw["axis"] == "t", "a 3-timepoint acquisition must record its time axis"
    assert spy.started, "the worker was built and never started"
    assert str(spy.out_path).endswith(".mp4")
    shutdown_plate_window(qapp, win)


def test_a_hidden_channel_is_left_out_of_the_movie(
        qapp, stub_detail, napari_pane_stub, five_d_root, spy_worker, save_dialog):
    """Out of the view is out of the movie: the window passes the VISIBLE channels."""
    win, w = _open_window(qapp, five_d_root)
    _drain_until(qapp, lambda: len(w._pane.mosaic._layers) >= 2, timeout=20)
    names = [c["name"] for c in w._meta["channels"]]
    assert len(names) == 2

    w._pane.mosaic.find("raw", names[1]).visible = False
    w._record_movie()

    assert spy_worker.instances[-1].kw["channels"] == names[:1], (
        f"recorded {spy_worker.instances[-1].kw['channels']}, only {names[0]} is visible")
    shutdown_plate_window(qapp, win)


def test_a_second_click_cancels_instead_of_starting_a_second_export(
        qapp, stub_detail, napari_pane_stub, five_d_root, spy_worker, save_dialog):
    win, w = _open_window(qapp, five_d_root)
    w._record_movie()
    assert len(spy_worker.instances) == 1

    w._record_movie()

    assert len(spy_worker.instances) == 1, "a second click started a second export"
    assert spy_worker.instances[0].stopped, "a second click did not cancel the run in flight"
    shutdown_plate_window(qapp, win)


# --- the UI thread --------------------------------------------------------------------------

def test_the_click_handler_does_not_read_or_encode_on_the_ui_thread(
        qapp, stub_detail, napari_pane_stub, five_d_root, save_dialog, tmp_path):
    """The handler's own wall clock, with the REAL worker started.

    SELF-CALIBRATING, not a fixed millisecond budget. The same export is first run SYNCHRONOUSLY
    to find out what this machine and this fixture actually cost, and the click is then required
    to be a small fraction of that. A fixed budget would be either untrippable on a small fixture
    (a 3-frame 64 px encode is ~60 ms, which fits under any threshold loose enough to be
    non-flaky) or flaky on a slow one. The ratio says the thing that matters — the work is
    somewhere else — at any fixture size and on any machine.

    Measured on the real 10x acquisition (27 FOVs x 4 channels x 10 z), which is the size this
    guard exists for: 0.91 ms in the handler against 4.20 s of export.
    """
    from squidmip._video import record_region

    win, w = _open_window(qapp, five_d_root)
    _drain_until(qapp, lambda: bool(w._pane.mosaic._layers), timeout=20)

    t0 = time.perf_counter()
    record_region(w._reader, w._meta, w.current_region(), tmp_path / "reference.mp4",
                  axis="t", fps=6)
    synchronous = time.perf_counter() - t0

    started = time.perf_counter()
    w._record_movie()
    elapsed = time.perf_counter() - started

    assert w._video_worker is not None, "no export was started, so this timed nothing"
    assert elapsed < 0.2 * synchronous, (
        f"the record click blocked the UI thread for {elapsed * 1000:.1f} ms of a "
        f"{synchronous * 1000:.1f} ms export — reads and encoding belong to _VideoWorker")
    w._video_worker.stop()
    w._video_worker.wait(5000)
    shutdown_plate_window(qapp, win)


# --- the file the window produced ---------------------------------------------------------------

@pytest.mark.skipif(encoder_problem() is not None,
                    reason=f"no mp4 encoder: {encoder_problem()}")
def test_the_window_writes_a_real_movie_whose_frames_differ(
        qapp, stub_detail, napari_pane_stub, five_d_root, save_dialog, tmp_path):
    """The whole chain, no stub in the recording path: chip -> _VideoWorker -> .mp4 -> decode."""
    win, w = _open_window(qapp, five_d_root)
    _drain_until(qapp, lambda: bool(w._pane.mosaic._layers), timeout=20)
    out = Path(save_dialog["path"])

    w._record_movie()
    assert _drain_until(qapp, lambda: out.exists() and w._video_worker is None, timeout=60), (
        "the export never finished")

    assert out.stat().st_size > 0, "the window produced a 0-byte movie"
    frames, meta = _decode(out)
    assert len(frames) == 3, f"decoded {len(frames)} frames of a 3-timepoint acquisition"
    assert meta["fps"] == 6
    for i in range(len(frames) - 1):
        assert np.any(frames[i] != frames[i + 1]), (
            f"decoded frames {i} and {i + 1} are identical — the window recorded one timepoint "
            f"three times")
    shutdown_plate_window(qapp, win)
