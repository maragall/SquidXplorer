"""HCS viewer — headless (offscreen) tests.

Gates the viewer contract: pure hit-testing + fit-cell shape guard, ingest that LOADS a grey plate
without processing, the Process-well-plates operator that fills tiles + drives the hue status, the
raw-z-stack push into the embedded ndviewer on double-click (pointing at the acquisition's own
TIFFs — nothing copied), the FOV-slider -> red-box link, and second-open state reset. PyQt5 is
optional (the GUI is an extra), so this whole module skips when it isn't installed — the headless
pipeline never depends on Qt.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede the PyQt import

import time

import numpy as np
import pytest

pytest.importorskip("PyQt5")
# Guard the two-Qt-bindings segfault: if PySide is already in the process (napari / pytest-qt
# autoload it), importing PyQt5 GUI widgets on top crashes. Clean CI has neither. Locally, run
# `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_viewer.py` to load only PyQt5.
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )
from PyQt5.QtCore import QEvent, QPointF, Qt, pyqtSignal  # noqa: E402
from PyQt5.QtGui import QImage, QMouseEvent  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QPushButton, QSlider, QSpinBox, QWidget,
)

from squidmip import _viewer as V  # noqa: E402

from .conftest import CH_IN_YAML  # noqa: E402


class _StubDetail(QWidget):
    """Stand-in for the embedded ndviewer_light detail viewer.

    Records the push API (start_acquisition / register_image / go_to_well_fov) so we can assert
    the seam WITHOUT constructing ndviewer's real vispy/GL widget — which segfaults offscreen
    under pytest's PySide6/napari-loaded environment (a Qt-binding conflict, not a code bug).
    """

    # THE contrast window this viewer renders a channel with (IMA-261). The stub carries the
    # signal because the real ndviewer_light does: a stub that omits it would let the plate's
    # `contrastChanged` connection rot unobserved, which is precisely the class of dead wiring
    # this file exists to catch.
    contrastChanged = pyqtSignal(int, float, float)

    def __init__(self):
        super().__init__()
        self._fov_labels = []
        self._fov_slider = QSlider(Qt.Horizontal, self)
        self._windows = {}        # ch -> (lo, hi), exactly as ndv would have resolved them
        self.registered = []
        # (t, idx, z, ch, (h, w)) of every register_array push. The SHAPE is recorded because
        # IMA-245 is a shape defect: the handler was called with the right index and the wrong
        # rectangle, and a recorder that keeps only the index cannot tell those two apart.
        self.arrays = []
        self.nav = []
        self.acquisitions = []    # one entry per start_acquisition — the slider's label list
        self.canvases = []        # (h, w) declared by each start_acquisition
        # (pixel_size_um, dz_um) per start_acquisition. The real viewer needs both to give
        # ndv's 3D button the right voxel aspect (IMA-255); recording them is what lets
        # tests/test_viewer_3d.py prove the metadata actually reaches the seam.
        self.voxel_um = []

    def start_acquisition(self, channels, nz, h, w, labels,
                          pixel_size_um=None, dz_um=None):
        self._fov_labels = list(labels)
        self._fov_slider.setMaximum(max(0, len(labels) - 1))
        self.acquisitions.append(list(labels))
        self.canvases.append((h, w))
        self.voxel_um.append((pixel_size_um, dz_um))

    def register_image(self, t, idx, z, ch, path, page_idx=0):
        self.registered.append((t, idx, z, ch, path))

    def register_array(self, t, idx, z, ch, plane):
        """Record computed-well pushes. The real ndviewer indexes its slider by ``idx``, so a push
        whose idx exceeds the current label list would land out of range — recording it here is what
        makes the global->subset remap assertable at all (the push path was previously unobserved).

        IMA-245: a real ndviewer canvas is fixed by ``start_acquisition``, so a plane of the wrong
        SHAPE is as undisplayable as a push that never arrives. Refuse it here for the same reason
        the real viewer would, so a test cannot pass on a push the GUI would show as black."""
        shape = tuple(np.asarray(plane).shape)
        want = self.canvases[-1] if self.canvases else None
        if want is not None and shape != want:
            raise ValueError(f"plane {shape} does not fit the declared canvas {want}")
        self.arrays.append((t, idx, z, ch, shape))

    def go_to_well_fov(self, well_id, fov):
        self.nav.append((well_id, fov))
        return True

    # -- contrast: this viewer OWNS it and publishes it; the plate only follows (IMA-261) --
    def channel_windows(self):
        return dict(self._windows)

    def drag_contrast(self, ch, lo, hi):
        """What ndv does when its clim slider moves: record, then broadcast. Tests call THIS,
        never PlateWindow._on_detail_contrast, so the signal/slot connection is under test."""
        self._windows[ch] = (float(lo), float(hi))
        self.contrastChanged.emit(int(ch), float(lo), float(hi))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)  # main() won't call exec_/exit under test
    return app


@pytest.fixture
def stub_detail(monkeypatch):
    """Swap the real ndviewer for a recording stub (avoids the offscreen-GL segfault)."""
    monkeypatch.setattr(V.PlateWindow, "_make_detail_viewer", lambda self: _StubDetail())


def _drain_until(app, pred, timeout=60):
    t0 = time.time()
    while not pred() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    return pred()


def _close_exploration_pane(win):
    """Empty pane 3 through the REAL tab-close path, restoring a whole-plate detail slider.

    An exploration tab scopes the slider to its subset (IMA-205), and since IMA-237 that scope is
    owned by pane 3 alone — so "give me the whole plate back" means closing its tabs, not switching
    pane 1. Never pokes _push_index directly: the point is to drive what the user drives."""
    for i in range(win._explore_tabs.count() - 1, -1, -1):
        win._close_op_tab(i, win._explore_tabs)


def _press(x, y, button=Qt.LeftButton):
    """A synthetic left-press/release at (x, y) — the handlers only read button/pos."""
    return QMouseEvent(QEvent.MouseButtonPress, QPointF(x, y), button, button, Qt.NoModifier)


def _move(x, y, buttons=Qt.NoButton):
    return QMouseEvent(QEvent.MouseMove, QPointF(x, y), Qt.NoButton, buttons, Qt.NoModifier)


# --- pure helpers (no Qt display needed) ----------------------------------------------------

def test_well_at_maps_and_bounds():
    by_rc = {(0, 0): "A1", (1, 1): "B2"}
    assert V.well_at(["A", "B"], ["1", "2"], by_rc, 5, 5, 20.0)["well_id"] == "A1"
    assert V.well_at(["A", "B"], ["1", "2"], by_rc, 25, 25, 20.0)["well_id"] == "B2"
    assert V.well_at(["A", "B"], ["1", "2"], by_rc, 5, 25, 20.0)["well_id"] is None  # empty cell
    assert V.well_at(["A"], ["1"], {}, 9e9, 9e9, 20.0) is None                       # off-plate


def test_cells_in_rect_basic():
    by_rc = {(r, c): f"{'AB'[r]}{c + 1}" for r in range(2) for c in range(2)}
    rows, cols = ["A", "B"], ["1", "2"]
    assert V.cells_in_rect(rows, cols, by_rc, 0, 0, 39, 39, 20.0) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert V.cells_in_rect(rows, cols, by_rc, 0, 0, 5, 5, 20.0) == [(0, 0)]          # one cell
    assert V.cells_in_rect(rows, cols, by_rc, 25, 0, 35, 35, 20.0) == [(0, 1), (1, 1)]  # one column


def test_cells_in_rect_inverted_drag():
    """Dragging up-left must select the SAME cells as the equivalent down-right drag."""
    by_rc = {(r, c): f"{'AB'[r]}{c + 1}" for r in range(2) for c in range(2)}
    rows, cols = ["A", "B"], ["1", "2"]
    fwd = V.cells_in_rect(rows, cols, by_rc, 0, 0, 39, 39, 20.0)
    assert V.cells_in_rect(rows, cols, by_rc, 39, 39, 0, 0, 20.0) == fwd
    assert V.cells_in_rect(rows, cols, by_rc, 39, 0, 0, 39, 20.0) == fwd   # mixed inversion


def test_cells_in_rect_clamps_to_plate():
    by_rc = {(r, c): f"{'AB'[r]}{c + 1}" for r in range(2) for c in range(2)}
    rows, cols = ["A", "B"], ["1", "2"]
    # a rect running far past the last row/col clamps instead of inventing cells
    assert V.cells_in_rect(rows, cols, by_rc, 0, 0, 9999, 9999, 20.0) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    # ...and a rect starting at negative coords clamps at 0
    assert V.cells_in_rect(rows, cols, by_rc, -500, -500, 5, 5, 20.0) == [(0, 0)]


def test_cells_in_rect_off_plate_returns_empty():
    by_rc = {(0, 0): "A1"}
    rows, cols = ["A"], ["1"]
    assert V.cells_in_rect(rows, cols, by_rc, -900, -900, -100, -100, 20.0) == []   # above-left
    assert V.cells_in_rect(rows, cols, by_rc, 5000, 5000, 9000, 9000, 20.0) == []   # beyond extent


def test_cells_in_rect_zero_area_is_single_cell():
    by_rc = {(r, c): f"{'AB'[r]}{c + 1}" for r in range(2) for c in range(2)}
    assert V.cells_in_rect(["A", "B"], ["1", "2"], by_rc, 25, 25, 25, 25, 20.0) == [(1, 1)]


def test_cells_in_rect_excludes_unacquired():
    """A sparse plate: the marquee sweeps every position but only ACQUIRED wells are selected."""
    by_rc = {(0, 0): "A1", (1, 1): "B2"}          # A2 and B1 were never acquired
    assert V.cells_in_rect(["A", "B"], ["1", "2"], by_rc, 0, 0, 39, 39, 20.0) == [(0, 0), (1, 1)]


def test_fit_cell_always_returns_cell_shape():
    assert V._fit_cell(np.zeros((768, 768), np.float32)).shape == (V._CELL, V._CELL)
    assert V._fit_cell(np.zeros((V._CELL, V._CELL), np.float32)).shape == (V._CELL, V._CELL)
    assert V._fit_cell(np.zeros((40, 40), np.float32)).shape == (V._CELL, V._CELL)  # tiny frame upscaled


def test_running_contrast_latch_holds_against_new_wells():
    # IMA-206 D4: the running histogram must not stomp a window the user set. Channel 0 is latched
    # manual, channel 1 is left on auto; a new well then moves channel 1 and leaves channel 0 alone.
    rc = V._RunningContrast(2, 1000.0)
    for ch in (0, 1):
        rc.add(ch, np.full((8, 8), 100.0))
    rc.set_manual(0, 10.0, 20.0)
    assert rc.is_manual(0) and not rc.is_manual(1)
    before = rc.window(1)
    for ch in (0, 1):
        rc.add(ch, np.full((8, 8), 900.0))     # a much brighter well lands
    assert rc.window(0) == (10.0, 20.0)        # latched: untouched
    assert rc.window(1) != before              # auto: followed the new well
    rc.set_auto(0)                             # reset-to-auto -> back on the running window
    assert not rc.is_manual(0) and rc.window(0) == rc.window(1)


def test_running_contrast_manual_window_never_degenerate():
    # a user can drag both handles together; hi must stay above lo so _window can't divide by zero
    rc = V._RunningContrast(1, 1000.0)
    rc.set_manual(0, 500.0, 500.0)
    lo, hi = rc.window(0)
    assert hi > lo


def test_resolve_plate_root(tmp_path):
    (tmp_path / "plate.ome.zarr").mkdir()
    _, is_plate = V.resolve_plate_root(tmp_path)
    assert is_plate
    acq = tmp_path / "acq"
    acq.mkdir()
    _, is_plate = V.resolve_plate_root(acq)
    assert not is_plate


# --- per-channel plate store / channel toggle / contrast (IMA-206) --------------------------

_RED_BLUE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)   # a red and a blue channel


def _overview(qapp, n_ch=2):
    """A 1x2 plate (A1, A2) with *n_ch* channels declared — the store/mask/contrast are live."""
    ov = V.PlateOverview(["A"], ["1", "2"], {(0, 0): "A1", (0, 1): "A2"})
    ov.set_channels([f"c{i}" for i in range(n_ch)], _RED_BLUE[:n_ch], np.uint16)
    return ov


def _tile(levels):
    """(C, cell, cell) uint16 ramp per channel — a flat tile would window down to black."""
    grad = np.linspace(0.0, 1.0, V._CELL * V._CELL).reshape(V._CELL, V._CELL)
    return np.stack([(grad * lv).astype(np.uint16) for lv in levels])


def _rgb(ov) -> np.ndarray:
    """Whatever the plate is currently showing, as an (H, W, 3) uint8 array."""
    img = ov._active_source()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    row = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())
    return row[:, : img.width() * 3].reshape(img.height(), img.width(), 3)


def test_add_tile_retains_the_channel_axis(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 0]))
    store = ov._store["raw"]
    assert store.shape == (2, V._CELL, 2 * V._CELL) and store.dtype == np.uint16
    assert store[0, :, : V._CELL].max() > 0        # channel 0 landed in A1's cell
    assert store[1].max() == 0                     # channel 1 was dark, and stayed dark
    assert store[:, :, V._CELL :].max() == 0       # A2 never got a tile
    assert _rgb(ov)[:, : V._CELL].max() > 0        # ...and the cell composited onto the plate


def test_stale_or_foreign_cell_is_ignored(qapp):
    ov = _overview(qapp)
    ov.add_tile(9, 9, "Z9", _tile([1000, 1000]))   # a tile from a retired run / off-plate cell
    assert "raw" not in ov._store and not ov._tiles


def test_channel_toggle_removes_only_that_channel(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.recomposite()
    both = _rgb(ov).copy()
    assert both[:, :, 0].max() > 0 and both[:, :, 2].max() > 0
    ov.set_channel_visible(1, False)               # blue off -> the single-channel mosaic (P1)
    only_red = _rgb(ov)
    assert only_red[:, :, 2].max() == 0            # blue's contribution is gone
    np.testing.assert_array_equal(only_red[:, :, 0], both[:, :, 0])   # red is untouched
    ov.set_channel_visible(1, True)                # ...and it comes back
    np.testing.assert_array_equal(_rgb(ov), both)


def test_all_channels_off_is_black_and_does_not_crash(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    for ch in (0, 1):
        ov.set_channel_visible(ch, False)
    assert _rgb(ov).sum() == 0


def test_single_channel_acquisition_toggles_to_black(qapp):
    # C=1: turning the only channel off is allowed (a mask, not an exclusive swap) and is black.
    ov = _overview(qapp, n_ch=1)
    ov.add_tile(0, 0, "A1", _tile([1000]))
    assert _rgb(ov).max() > 0
    ov.set_channel_visible(0, False)
    assert _rgb(ov).sum() == 0


def test_rewindow_repaints_without_touching_the_store(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.recomposite()
    before_px, before_store = _rgb(ov).copy(), ov._store["raw"].copy()
    ov.set_channel_window(0, 0.0, 50.0)            # a much tighter window -> channel 0 saturates
    assert not np.array_equal(_rgb(ov), before_px)
    np.testing.assert_array_equal(ov._store["raw"], before_store)   # retained pixels, not re-read
    assert ov._contrast.is_manual(0) and not ov._contrast.is_manual(1)


def test_latched_channel_survives_a_new_well_and_auto_restores_it(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.set_channel_window(0, 0.0, 50.0)            # latch channel 0 mid-stream
    auto_before = ov._contrast.window(1)
    ov.add_tile(0, 1, "A2", _tile([60000, 60000]))  # a much brighter well lands
    assert ov.channel_windows()[0] == (0.0, 50.0)   # latched: the user's window held (D4)
    assert ov.channel_windows()[1] != auto_before   # unlatched: kept auto-scaling
    ov.set_channel_auto(0)
    assert ov.channel_windows()[0] == ov.channel_windows()[1]   # back on the running window


def test_recomposited_backing_array_outlives_its_qimage(qapp):
    # OV11: QImage WRAPS the numpy buffer. If the widget drops the reference the canvas is a
    # use-after-free, not a bug — so force a GC and read the plate back.
    import gc
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.recomposite()
    expected = _rgb(ov).copy()
    gc.collect()
    np.testing.assert_array_equal(_rgb(ov), expected)


def test_recomposite_is_global_so_wells_stay_comparable(qapp):
    # D6 regression: one bright well and one dim well must KEEP their relative brightness. A
    # per-well window (what the reopen path used to do) would wrongly equalize them.
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([4000, 0]))
    ov.add_tile(0, 1, "A2", _tile([400, 0]))
    ov.recomposite()
    rgb = _rgb(ov)
    assert rgb[:, : V._CELL].max() > rgb[:, V._CELL :].max()


def test_quick_recomposite_matches_the_full_one_at_fit_zoom(qapp):
    # A gesture composites a strided view at DISPLAY resolution; at 1:1 zoom that is the full pass.
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.recomposite(quick=True)
    quick = _rgb(ov).copy()
    ov.recomposite(quick=False)
    np.testing.assert_array_equal(_rgb(ov), quick)


# --- mosaic (IMA-187) x per-channel store (IMA-206) -----------------------------------------
#
# IMA-187 composites MANY FOVs into one 88px cell, zero-padding wherever no field lands. Those
# zeros are NOT data. If they reach the running histogram the 1st percentile pins to 0 for the
# WHOLE plate and every well renders washed out — silently, with the mosaic still looking correct.
# These tests hold that line, and hold the sub-cell placement the mosaic depends on.

def _box_tile(levels, h, w):
    """(C, h, w) uint16 ramp — one FIELD's worth of pixels, sized to its box, not to the cell."""
    grad = np.linspace(0.2, 1.0, h * w).reshape(h, w)
    return np.stack([(grad * lv).astype(np.uint16) for lv in levels])


def test_mosaic_tile_lands_at_its_box_offset(qapp):
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 3
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(h, w, h, w))   # the middle sub-cell
    store = ov._store["raw"]
    assert store[0, h:h + h, w:w + w].max() > 0          # the field landed inside its box...
    assert store[0, :h, :].max() == 0                    # ...and nowhere else in the cell
    assert store[0, :, :w].max() == 0


def test_mosaic_fields_accumulate_in_one_cell_and_seams_recomposite(qapp):
    # A 36-FOV well is built from 36 arrivals, not 36 overwrites, and each arrival re-composites
    # the WHOLE cell so the seam against its already-landed neighbour updates.
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 4
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(0, 0, h, w))
    first = _rgb(ov)[:, :V._CELL].copy()
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(0, w, h, w))   # the neighbour to its right
    store = ov._store["raw"]
    assert store[0, :h, :w].max() > 0 and store[0, :h, w:2 * w].max() > 0   # BOTH still present
    assert not np.array_equal(_rgb(ov)[:, :V._CELL], first)                # the cell repainted


def test_contrast_ignores_the_mosaic_zero_padding(qapp):
    # THE regression. A sparse mosaic: one small bright field in a mostly-empty 88px cell. The
    # window must be the one the FIELD's pixels alone imply — feeding the padded cell instead
    # drags the 1st percentile to 0 and washes the plate out.
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 4                      # the field covers 1/16 of the cell; 15/16 is padding
    tile = _box_tile([50000], h, w)
    ov.add_tile(0, 0, "A1", tile, box=(0, 0, h, w))
    got = ov.channel_windows()[0]

    ref = V._RunningContrast(1, float(np.iinfo(np.uint16).max))
    ref.add(0, tile[0])                       # the boxes alone — no padding
    assert got == ref.window(0)

    poisoned = V._RunningContrast(1, float(np.iinfo(np.uint16).max))
    poisoned.add(0, ov._store["raw"][0, :V._CELL, :V._CELL])   # the cell INCLUDING its zeros
    assert poisoned.window(0)[0] < got[0]     # ...which is strictly darker-pinned: the bug
    assert poisoned.window(0) != got


def test_dim_mosaic_well_is_not_washed_out_by_padding(qapp):
    # The user-visible consequence, end to end: a dim well next to a bright one, both sparse
    # mosaics. With the padding poisoning the histogram the dim well's rendered range collapses.
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 4
    ov.add_tile(0, 0, "A1", _box_tile([60000], h, w), box=(0, 0, h, w))    # bright well
    ov.add_tile(0, 1, "A2", _box_tile([3000], h, w), box=(0, 0, h, w))     # dim well
    ov.recomposite()
    rgb = _rgb(ov)
    dim = rgb[:h, V._CELL:V._CELL + w, 0]
    assert dim.max() > 0                      # the dim well is still visible at all...
    assert rgb[:h, :w, 0].max() > dim.max()   # ...and still reads as dimmer than the bright one


def test_reset_layer_frees_the_store_so_a_shorter_rerun_leaves_nothing(qapp):
    # A re-run that lands FEWER fields must not composite on top of the last run's pixels.
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 4
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(0, 0, h, w))
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(h, 0, h, w))
    ov.reset_layer("raw")
    assert "raw" not in ov._store and not ov._tiles_by_layer.get("raw")
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(0, 0, h, w))     # the shorter re-run
    assert ov._store["raw"][0, h:2 * h, :w].max() == 0      # the old second field is GONE


# --- GUI behavior (offscreen; embedded viewer stubbed) --------------------------------------

def test_ingest_bad_folder_does_not_crash(qapp, stub_detail, tmp_path):
    win = V.PlateWindow(None)
    bad = tmp_path / "not_squid"
    bad.mkdir()
    win.ingest(str(bad))          # must NOT raise / abort
    assert "not a readable" in win._readout.text().lower() or "no squid" in win._readout.text().lower()
    win.close()


def test_ingest_loads_plate_and_previews_without_processing(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset          # tiny real acquisition (B2, B3)
    win = V.PlateWindow(None)
    win.ingest(str(root))
    # the plate loads immediately with every acquired well; a raw PREVIEW fills thumbnails but
    # leaves status grey ("empty"); NO operator worker runs until the Process menu is used.
    assert win._overview is not None
    assert set(win._overview._by_rc.values()) == {"B2", "B3"}
    assert _drain_until(qapp, lambda: len(win._overview._tiles) == 2)   # preview filled thumbnails
    assert set(win._overview._status.values()) == {"empty"}            # ...but status stays grey
    assert win._worker is None
    assert all(a.isEnabled() for a in win._op_actions.values())        # operators enabled once loaded
    win.close()


def test_ingest_non_wellplate_region_opens_as_a_slide_carrier(qapp, stub_detail, tmp_path):
    # IMA-214 INVERTED THIS TEST. It used to assert that a readable acquisition whose region is
    # not a well id ("R2C3", "manual0", a glass slide) was REFUSED with "not a well-plate".
    # That refusal is exactly what blocked the real 18 GB tissue dataset from ever opening.
    #
    # A slide carrier IS a plate: a grid of cells where a cell holds 0, 1 or many FOVs. So the
    # acquisition must now OPEN, with the freeform region id as a carrier cell. The old contract
    # (never crash out of ingest/__init__) still holds -- it is just satisfied by succeeding
    # rather than by bailing out.
    #
    # "R2C3" is deliberate: it does NOT match <letters><digits>, so it is the case that used to
    # crash activate_well's parse_well_id outside its try. "manual0" survived only by luck.
    import tifffile
    root = tmp_path / "slide_acq"
    (root / "0").mkdir(parents=True)
    for z in (0, 1):
        tifffile.imwrite(root / "0" / f"R2C3_0_{z}_Fluorescence_638_nm_-_Penta.tiff",
                         np.zeros((4, 4), np.uint16))
    (root / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n- name: Fluorescence 638 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#FF0000'\n      exposure_time_ms: 50.0\n")
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.325\n  magnification: 20.0\n  sensor_pixel_size_um: 3.76\n"
        "sample:\n  wellplate_format: 1536 well plate\nz_stack:\n  nz: 2\n  delta_z_mm: 0.0015\n"
        "time_series:\n  nt: 1\n")

    win = V.PlateWindow(None)
    win.ingest(str(root))                        # must not raise
    assert win._reader is not None, win._readout.text()
    assert win._overview is not None, "a slide carrier must reach the plate widget"
    assert "R2C3" in win._fov_index, f"freeform region lost: {list(win._fov_index)}"
    assert "not a well-plate" not in win._readout.text().lower()
    # the initial-path route through __init__ must not crash either
    win2 = V.PlateWindow(str(root))
    assert win2._overview is not None
    win.close(); win2.close()


def test_run_operator_persists_via_write_plate(qapp, stub_detail, squid_dataset, monkeypatch, tmp_path):
    # run_operator now PERSISTS: it drives write_plate with the SELECTED projector, and the GUI must
    # NOT write the uncompressed individual-TIFF copy (tiff=False) — that would double disk use.
    import squidmip
    captured = {}

    def fake_write_plate(reader, out_dir, *, n_fovs=1, workers=None, projector="mip",
                         tiff=True, on_well=None, write_workers=4, stop=None, on_error=None,
                         regions=None, operator_kwargs=None):
        # operator_kwargs is real_write_plate's IMA-decon-stitch-ui parameter: a REGION
        # operator's per-run settings (registration, feather, thresholds) have to reach the
        # SAVE path too, not just the preview. A stub whose signature drifts from the real
        # function does not fail here by luck -- run_operator calls it with the keyword, so
        # omitting it raises TypeError and this test goes red, which is how it was found.
        captured.update(projector=projector, tiff=tiff, out_dir=str(out_dir), regions=regions,
                        operator_kwargs=operator_kwargs)
        return {"plate": str(out_dir), "levels": 1}      # no wells — we only assert the dispatch
    monkeypatch.setattr(squidmip, "write_plate", fake_write_plate)

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: "projector" in captured)
    assert captured["projector"] == "mip"
    assert captured["operator_kwargs"] is None      # a projector takes no per-run parameters
    assert captured["tiff"] is False                     # never the uncompressed TIFF duplicate
    assert captured["out_dir"].endswith(".hcs")          # persisted next to the acquisition
    win._stop_worker(); win.close()


def test_run_operator_fills_tiles_and_hue_status(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(
        qapp, lambda: win._overview is not None and len(win._overview._tiles) == 2
        and win._overview._final is not None
    )
    # both wells processed -> tiled + hue-coded "done"
    assert win._overview._tiles == set(win._fov_index[w]["rc"] for w in ("B2", "B3"))
    assert set(win._overview._status.values()) == {"done"}
    # bounded memory: the plate keeps one 88px per-channel tile per well, not the acquisition
    store = win._overview._store["mip"]
    assert store.shape == (len(win._meta["channels"]), win._overview._nr * V._CELL,
                           win._overview._nc * V._CELL)
    assert store.dtype == np.dtype(win._meta["dtype"])       # native dtype, not float32
    win._stop_worker()
    win.close()


def test_double_click_pushes_raw_zstack(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._detail.registered.clear()   # ignore the first well auto-opened on ingest
    win.activate_well("B3", 0)       # double-click B3 -> register its raw z-planes + navigate
    regs = win._detail.registered
    assert regs, "no raw planes registered"
    # every registration points at a real on-disk TIFF at B3's plate index, across both z-levels
    idx = win._fov_index["B3"]["idx"]
    assert {r[1] for r in regs} == {idx}
    assert {r[2] for r in regs} == {0, 1}                        # z-stack: both z-levels pushed
    assert all(r[4].endswith(".tiff") and os.path.exists(r[4]) for r in regs)
    assert win._detail.nav[-1] == ("B3", 0)                      # navigated to the well
    # second double-click doesn't re-register (idempotent push)
    n = len(regs)
    win.activate_well("B3", 0)
    assert len(win._detail.registered) == n
    win.close()


def test_fov_slider_moves_red_box(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    # drive the ndviewer FOV slider -> the plate's red box should select that well
    idx = win._detail._fov_labels.index("B3:0")
    win._detail._fov_slider.setValue(idx)
    qapp.processEvents()
    assert win._overview._sel == win._fov_index["B3"]["rc"]
    win.close()


# --- selection: marquee + click (IMA-221) ---------------------------------------------------
#
# Gesture matrix under test. Shift owns EVERY selection gesture, so plain drag/double-click
# (the landed navigator behavior) are untouched, and Qt's press->release->doubleclick ordering
# can never toggle a well as a side effect of opening it.
#
#   Shift+drag       -> marquee, REPLACES the selection
#   Shift+Alt+drag   -> marquee, UNIONS into the selection
#   Shift+click      -> toggles one well
#   plain drag       -> pans (unchanged)      plain double-click -> opens the well (unchanged)

def _sel_overview(cd=20.0):
    """A 2x2 plate with a sparse corner (B1 never acquired) and a FROZEN view.

    Freezing (_user_view + explicit _cd/_ox/_oy) keeps widget pixels deterministic — otherwise
    paintEvent's auto-fit would move the plate under the synthetic coordinates.
    """
    wells = {(0, 0): "A1", (0, 1): "A2", (1, 1): "B2"}     # (1,0) = B1 absent
    ov = V.PlateOverview(["A", "B"], ["1", "2"], wells)
    ov._user_view = True
    ov._cd, ov._ox, ov._oy = cd, 0.0, 0.0
    return ov


def _pt(ri, ci, cd=20.0):
    """Widget-space center of cell (ri, ci) — mirrors PlateOverview._cell's margin offsets."""
    from PyQt5.QtCore import QPointF
    return QPointF(V._HDR + ci * cd + cd / 2, V._COLH + ri * cd + cd / 2)


def _within(ri, ci, cd=20.0):
    """Two points INSIDE one cell, far enough apart to read as a drag (not a Shift+click)."""
    from PyQt5.QtCore import QPointF
    return (QPointF(V._HDR + ci * cd + 2, V._COLH + ri * cd + 2),
            QPointF(V._HDR + ci * cd + cd - 2, V._COLH + ri * cd + cd - 2))


def _mouse(kind, pos, mods=Qt.NoModifier, buttons=Qt.LeftButton, btn=Qt.LeftButton):
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent
    ev = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove,
          "release": QEvent.MouseButtonRelease, "dblclick": QEvent.MouseButtonDblClick}[kind]
    return QMouseEvent(ev, pos, btn, buttons, mods)


def _drag(ov, a, b, mods):
    ov.mousePressEvent(_mouse("press", a, mods))
    ov.mouseMoveEvent(_mouse("move", b, mods))
    ov.mouseReleaseEvent(_mouse("release", b, mods, buttons=Qt.NoButton))


def test_marquee_replaces_selection(qapp):
    ov = _sel_overview()
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.ShiftModifier)          # sweep the whole 2x2
    assert ov.selected_wells() == ["A1", "A2", "B2"]           # B1 never acquired -> excluded
    _drag(ov, *_within(0, 0), Qt.ShiftModifier)                # a fresh marquee over A1 only...
    assert ov.selected_wells() == ["A1"]                        # ...REPLACES, not unions


def test_additive_marquee_unions(qapp):
    ov = _sel_overview()
    _drag(ov, *_within(0, 0), Qt.ShiftModifier)                          # A1
    _drag(ov, *_within(1, 1), Qt.ShiftModifier | Qt.AltModifier)         # + B2
    assert ov.selected_wells() == ["A1", "B2"]


def test_shift_click_toggles_well(qapp):
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 1), Qt.ShiftModifier))
    ov.mouseReleaseEvent(_mouse("release", _pt(0, 1), Qt.ShiftModifier, buttons=Qt.NoButton))
    assert ov.selected_wells() == ["A2"]
    ov.mousePressEvent(_mouse("press", _pt(0, 1), Qt.ShiftModifier))     # click again -> off
    ov.mouseReleaseEvent(_mouse("release", _pt(0, 1), Qt.ShiftModifier, buttons=Qt.NoButton))
    assert ov.selected_wells() == []


def test_selection_emits_once_on_release(qapp):
    """The rubber band is the live feedback; the SIGNAL fires once per gesture, on release.
    A 1536-well plate would otherwise rebuild + emit a 1536-item list per mouse-move."""
    ov = _sel_overview()
    seen = []
    ov.selectionChanged.connect(lambda wells: seen.append(list(wells)))
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    for _ in range(5):                                          # five moves mid-drag...
        ov.mouseMoveEvent(_mouse("move", _pt(1, 1), Qt.ShiftModifier))
    assert seen == []                                           # ...emit NOTHING
    ov.mouseReleaseEvent(_mouse("release", _pt(1, 1), Qt.ShiftModifier, buttons=Qt.NoButton))
    assert seen == [["A1", "A2", "B2"]]                         # exactly one emission


def test_selection_excludes_empty_wells(qapp):
    ov = _sel_overview()
    _drag(ov, *_within(1, 0), Qt.ShiftModifier)                 # B1: a plate position, never acquired
    assert ov.selected_wells() == []


def test_wheel_ignored_during_marquee(qapp):
    """Zooming mid-marquee would move the plate under the drag, so the wheel is ignored."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtGui import QWheelEvent
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    cd_before = ov._cd
    ov.wheelEvent(QWheelEvent(QPoint(60, 60), QPoint(60, 60), QPoint(0, 0), QPoint(0, 120),
                              Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert ov._cd == cd_before                                  # zoom did NOT happen


def test_right_button_release_does_not_commit_a_selection(qapp):
    """A RIGHT release must not commit the gesture. Qt delivers a release for whichever button
    went up, so without an e.button() check a right-click during a Shift-drag silently toggled
    a well (and dropped the in-flight marquee) with no left release ever having happened."""
    ov = _sel_overview()
    seen = []
    ov.selectionChanged.connect(lambda wells: seen.append(list(wells)))
    ov.mousePressEvent(_mouse("press", _pt(0, 1), Qt.ShiftModifier))          # Shift-press on A2
    ov.mouseReleaseEvent(_mouse("release", _pt(0, 1), Qt.ShiftModifier,
                                buttons=Qt.NoButton, btn=Qt.RightButton))
    assert ov.selected_wells() == []                            # nothing selected
    assert seen == []                                           # and nothing emitted
    assert ov._marquee is not None                              # the gesture is still in flight
    ov.mouseReleaseEvent(_mouse("release", _pt(0, 1), Qt.ShiftModifier,       # the LEFT release...
                                buttons=Qt.NoButton))
    assert ov.selected_wells() == ["A2"]                        # ...is what commits it


def test_leave_clears_the_marquee_so_zoom_survives(qapp):
    """Losing the grab mid-drag (modal dialog, alt-tab) delivers a leave and NO release. A
    stranded _marquee would paint a dashed rect forever and trip wheelEvent's guard, disabling
    zoom permanently."""
    from PyQt5.QtCore import QEvent, QPoint
    from PyQt5.QtGui import QWheelEvent
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    assert ov._marquee is not None
    ov.leaveEvent(QEvent(QEvent.Leave))                         # grab lost; no release ever arrives
    assert ov._marquee is None
    cd_before = ov._cd
    ov.wheelEvent(QWheelEvent(QPoint(60, 60), QPoint(60, 60), QPoint(0, 0), QPoint(0, 120),
                              Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert ov._cd != cd_before                                  # zoom works again


# --- selection regressions: the landed navigator gestures must be untouched -----------------

def test_plain_drag_still_pans(qapp):
    ov = _sel_overview()
    ox0, oy0 = ov._ox, ov._oy
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.NoModifier)              # NO Shift
    assert (ov._ox, ov._oy) != (ox0, oy0), "plain drag no longer pans"
    assert ov.selected_wells() == [], "plain drag must not select"


def test_double_click_does_not_toggle_selection(qapp):
    """Qt delivers press+release BEFORE mouseDoubleClickEvent — opening a well must not select it."""
    ov = _sel_overview()
    opened = []
    ov.wellActivated.connect(lambda wid, fov: opened.append((wid, fov)))
    p = _pt(0, 0)
    ov.mousePressEvent(_mouse("press", p))
    ov.mouseReleaseEvent(_mouse("release", p, buttons=Qt.NoButton))
    ov.mouseDoubleClickEvent(_mouse("dblclick", p))
    assert opened == [("A1", 0)]                                # still opens the well
    assert ov.selected_wells() == []                            # ...and selects nothing


def test_selection_does_not_disturb_red_box(qapp):
    """_sel (ndviewer current well, red box) and _selection (operator's pick) stay independent."""
    ov = _sel_overview()
    ov.select(1, 1)
    _drag(ov, *_within(0, 0), Qt.ShiftModifier)
    assert ov._sel == (1, 1)                                    # red box unmoved
    assert ov.selected_wells() == ["A1"]


def test_clear_selection_emits_empty(qapp):
    ov = _sel_overview()
    seen = []
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.ShiftModifier)
    ov.selectionChanged.connect(lambda wells: seen.append(list(wells)))
    ov.clear_selection()
    assert ov.selected_wells() == [] and seen == [[]]


# --- window level: expansion to (region, fov) + run-on-selection ----------------------------

def test_selection_expands_to_region_fov_pairs(qapp, stub_detail, squid_dataset):
    """PlateOverview is display-only (it has no metadata), so PlateWindow does the expansion."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._overview.selectionChanged.emit(["B3"])
    qapp.processEvents()
    assert win._selected_regions == ["B3"]
    fovs = win._meta["fovs_per_region"]["B3"]
    assert win.selected_region_fovs() == [("B3", f) for f in fovs]
    win.close()


def test_run_operator_on_selection_only_processes_selected(qapp, stub_detail, squid_dataset,
                                                           monkeypatch, tmp_path):
    """The Accept gate: a selection SCOPES the operator run to just those wells."""
    import squidmip
    captured = {}

    def fake_write_plate(reader, out_dir, **kw):
        captured.update(regions=kw.get("regions"))
        return {"plate": str(out_dir), "levels": 1}
    monkeypatch.setattr(squidmip, "write_plate", fake_write_plate)

    root, _ = squid_dataset                       # B2, B3
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._overview.selectionChanged.emit(["B3"])   # select ONE of the two wells
    qapp.processEvents()
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: "regions" in captured)
    assert captured["regions"] == ["B3"], "the run was not scoped to the selection"
    # ...and only the selected well went amber
    assert win._overview._status[win._fov_index["B3"]["rc"]] == "processing"
    assert win._overview._status[win._fov_index["B2"]["rc"]] == "empty"
    win._stop_worker(); win.close()


def test_selection_clears_on_second_ingest(qapp, stub_detail, squid_dataset):
    """A stale selection must never point at wells from the previous acquisition."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._overview.selectionChanged.emit(["B3"])
    qapp.processEvents()
    assert win._selected_regions == ["B3"]
    win.ingest(str(root))                          # re-open
    qapp.processEvents()
    assert win._selected_regions == []
    assert win._overview.selected_wells() == []
    win._stop_worker(); win.close()


# --- tab detach / float / re-dock (IMA-209; offscreen drives the _detach_tab seam, not the drag) --

class _StubTab(QWidget):
    """A registry-registered tab standing in for a live terminal: records shutdown() calls."""

    def __init__(self):
        super().__init__()
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1


def _open_stub_tab(win, key="stub", title="Stub"):
    w = _StubTab()
    win._open_op_tab(key, title, lambda: w)
    return w


def test_detach_moves_widget_to_float_and_registry(qapp, stub_detail):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    fl = win._detach_tab(win._left_tabs.indexOf(w))
    assert fl is not None
    assert win._left_tabs.indexOf(w) == -1                   # gone from the bar...
    assert "stub" not in win._op_tabs and win._floating["stub"] is fl
    assert w.window() is fl                                  # ...and the SAME live widget floats
    win.close()


def test_detach_home_tab_refused(qapp, stub_detail):
    win = V.PlateWindow(None)
    assert win._detach_tab(0) is None                        # 'Process wells' never detaches
    assert win._left_tabs.count() >= 1 and win._left_tabs.widget(0) is not None
    win.close()


def test_open_op_tab_focuses_float_not_duplicate(qapp, stub_detail):
    # REGRESSION (eng review D4): with the key moved to _floating, an unpatched _open_op_tab
    # would rebuild the UI — for the CLI, a SECOND live shell. The opener must focus the float.
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    win._detach_tab(win._left_tabs.indexOf(w))
    built = []
    win._open_op_tab("stub", "Stub", lambda: built.append(1) or _StubTab())
    assert not built                                         # builder NOT re-called
    assert win._floating["stub"].isVisible()                 # float raised, not replaced
    win.close()


def test_close_float_disposes_widget(qapp, stub_detail):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    fl = win._detach_tab(win._left_tabs.indexOf(w))
    fl.close()                                               # user closes the floating window
    assert w.shutdowns == 1                                  # shell dead, via the ONE cleanup path
    assert "stub" not in win._floating and "stub" not in win._op_tabs
    w2 = _StubTab()
    win._open_op_tab("stub", "Stub", lambda: w2)             # reopening builds fresh
    assert win._op_tabs["stub"] is w2
    win.close()


def test_redock_returns_same_widget(qapp, stub_detail):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    win._detach_tab(win._left_tabs.indexOf(w))
    win._redock("stub")
    assert win._op_tabs["stub"] is w                         # SAME object — a live shell survives
    assert win._left_tabs.currentWidget() is w
    assert not win._floating
    assert w.shutdowns == 0                                  # re-dock never kills the shell
    win.close()


def test_main_close_with_float_open_shuts_down(qapp, stub_detail):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    win._detach_tab(win._left_tabs.indexOf(w))
    win.close()                                              # app exit with a float open
    assert w.shutdowns == 1                                  # drained: no leaked shell...
    assert not win._floating                                 # ...no orphan window blocking exit


def test_detached_layers_keeps_refreshing_until_dispose(qapp, stub_detail):
    win = V.PlateWindow(None)
    win._open_op_tab("layers", "Layers", win._build_layers_tab)
    lw = win._op_tabs["layers"]
    fl = win._detach_tab(win._left_tabs.indexOf(lw))
    assert win._layers_tab is lw                             # refs NOT cleared on detach...
    win._refresh_layers_tab()                                # ...so refresh still writes the float
    assert win._layers_box.count() >= 2                      # rebuilt (title + stretch at minimum)
    fl.close()
    assert win._layers_tab is None and win._layers_box is None   # cleared on dispose ONLY
    win.close()


def test_float_survives_second_ingest(qapp, stub_detail, squid_dataset):
    # Floats follow docked-tab semantics across a plate swap: they persist (staleness of op tabs
    # on re-ingest is a pre-existing, tab-wide behavior — tracked in TODOS.md, not 209's scope).
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = _open_stub_tab(win)
    win._detach_tab(win._left_tabs.indexOf(w))
    win.ingest(str(root))                                    # plate swap with a float open
    qapp.processEvents()
    assert win._floating["stub"].isVisible()                 # still floating, registry intact
    assert "stub" not in win._op_tabs
    win.close()


def test_channel_toggle_after_preview_reads_nothing(qapp, stub_detail, squid_dataset):
    # OV10 defines "no recompute": no reader I/O and no projection. Assert it with a SPY on the
    # reader, not by timing — the toggle must recomposite purely from the retained store.
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert _drain_until(qapp, lambda: len(win._overview._tiles) == 2)   # preview filled the store
    win._stop_preview()                       # the preview owns the only other reader traffic
    qapp.processEvents()

    reads = []
    real_read = win._reader.read
    win._reader.read = lambda *a, **k: (reads.append(a), real_read(*a, **k))[1]

    before = _rgb(win._overview).copy()
    win._overview.set_channel_visible(0, False)
    qapp.processEvents()
    assert not np.array_equal(_rgb(win._overview), before)   # the plate really changed
    assert reads == []                                       # ...and nothing was read/projected
    assert win._worker is None                               # no operator run was triggered
    win.close()


def test_napari_visibility_drives_the_plate_and_the_strip_only_reports_it(qapp, stub_detail,
                                                                          squid_dataset):
    """Julio: "there shouldn't be any controls for the plate view. It just reacts to toggles and
    contrast adjustments in napari."

    The strip's checkboxes used to be the seam: click a box, mask the channel out of the plate
    composite. napari's eye icon over the SAME channel was a second control over the same
    question, and the two could disagree on screen. Now the eye icon is the only control and the
    plate is a sink -- so this drives the sink and checks the plate followed.

    MUTATION: drop the `on_user_visibility` binding in `_bind_napari_contrast` and this goes red.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    # what napari reports when the user clicks an eye icon off, and back on
    win._overview.set_channel_visible(0, False)
    assert win._overview._mask[0] == False        # noqa: E712 — numpy bool, not python bool
    win._overview.set_channel_visible(0, True)
    assert win._overview._mask[0] == True         # noqa: E712
    win.close()


# --- IMA-261: contrast has exactly ONE owner, and it is the central array viewer --------------
#
# The plate used to carry its own low/high slider pair and an "auto" button per channel, two
# hand-widths from ndviewer_light's contrast slider over the same channel. Two controls over one
# quantity is this project's second-most-common defect shape, and here it had already gone wrong:
# the same channel was displayed at two different windows, side by side, on one screen.
#
# These tests pin the resolution in both directions — the duplicate control is GONE, and the plate
# genuinely FOLLOWS the surviving owner.

def test_array_viewer_contrast_drag_repaints_the_plate(qapp, stub_detail, squid_dataset):
    """The user's actual complaint, at the seam: ndv re-windows, the plate must follow.

    Emits the REAL `contrastChanged` signal rather than calling `_on_detail_contrast`, so the
    signal/slot connection itself is under test — a handler-level test passes with dead wiring,
    which is how the Re-dock button shipped broken.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview

    win._detail.drag_contrast(0, 700.0, 5000.0)
    qapp.processEvents()
    assert ov.channel_windows()[0] == (700.0, 5000.0), "the plate ignored the array viewer"
    win.close()


def test_the_array_viewer_never_latches_the_plate_manual(qapp, stub_detail, squid_dataset):
    """THE regression this nearly shipped with.

    ndv autoscales on its own — at open, and again on every data change — so the first version of
    this sync, which recorded each broadcast with `set_manual`, came up with EVERY channel latched
    manual before the user had touched anything. That killed the plate's running auto-contrast
    from the first frame and, because a manual latch outranks everything, made SCOPE_PER_REGION
    paint every well under one global window while the plate still drew the amber "wells NOT
    comparable" badge over the top.

    A sink records what the owner resolved. Only the user sets policy.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview
    n = len(ov._labels)

    assert not any(ov._contrast.is_manual(c) for c in range(n)), (
        "a channel was latched MANUAL on open, before any user gesture")

    win._detail.drag_contrast(0, 700.0, 5000.0)     # ndv autoscaling, or the user in ndv's pane
    qapp.processEvents()
    assert ov.channel_windows()[0] == (700.0, 5000.0), "the plate did not follow the viewer"
    assert not ov._contrast.is_manual(0), (
        "following the array viewer latched the channel MANUAL — the sink wrote policy back")
    assert ov._contrast.is_followed(0), "the window was not recorded as followed either"
    win.close()


def test_a_user_latch_still_outranks_the_viewer(qapp, stub_detail, squid_dataset):
    """`resolve` is still ONE precedence rule, now over three inputs:

        user latch  >  the owning viewer's window  >  whatever the caller computed.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview

    win._detail.drag_contrast(0, 111.0, 9999.0)
    qapp.processEvents()
    assert ov._contrast.resolve(0, (0.0, 1.0)) == (111.0, 9999.0)   # viewer beats the auto window

    ov.set_channel_window(0, 40.0, 80.0)                            # a real user gesture
    assert ov._contrast.is_manual(0)
    assert ov._contrast.resolve(0, (0.0, 1.0)) == (40.0, 80.0), "the user lost to the viewer"

    ov.set_channel_auto(0)                                          # release the user's latch
    assert ov._contrast.resolve(0, (0.0, 1.0)) == (111.0, 9999.0), (
        "releasing the user latch did not fall back to the viewer's window")
    win.close()


def test_a_channel_the_plate_does_not_have_is_ignored_not_a_crash(qapp, stub_detail, squid_dataset):
    """ndv draws RGB mode and re-ingests; it can broadcast a channel index the plate lacks."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n = len(win._overview._labels)

    win._detail.drag_contrast(n + 3, 1.0, 2.0)      # out of range: must be dropped silently
    qapp.processEvents()
    win._detail.drag_contrast(-1, 1.0, 2.0)
    qapp.processEvents()
    assert not win._overview._contrast.is_manual(0)
    win.close()


def test_a_fresh_plate_adopts_the_viewers_current_windows(qapp, stub_detail, squid_dataset):
    """Opening a second acquisition must not show a window the array viewer is not showing.

    The viewer keeps whatever contrast it had; a plate that only synced from the NEXT gesture
    onward would open disagreeing with the picture already on screen.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._detail.drag_contrast(0, 42.0, 4242.0)
    qapp.processEvents()

    win.ingest(str(root))                            # re-open: a brand-new plate store
    qapp.processEvents()
    assert win._overview.channel_windows()[0] == (42.0, 4242.0), (
        "the re-opened plate did not adopt the window the array viewer is still showing")
    win.close()


def test_a_contrast_change_keeps_the_thumbnail_but_new_pixels_drop_it(qapp):
    """The cache split that makes the drag fast, and the stale frame it could cause.

    `_disp` holds a display-resolution copy of the store. It is keyed on PIXELS, so a contrast
    change must NOT drop it (that is the whole speedup) while a tile landing MUST (otherwise the
    plate keeps compositing the thumbnail taken before the well arrived, and the new well never
    appears). Two invalidation reasons, deliberately not merged — so both directions are pinned.
    """
    # A plate big enough that the screen cannot show it 1:1 — the thumbnail only exists when the
    # composite is sub-sampled, which is exactly the case a drag hits.
    rows = [chr(ord("A") + i) for i in range(8)]
    cols = [str(i + 1) for i in range(12)]
    ov = V.PlateOverview(rows, cols, {(r, c): f"{rows[r]}{cols[c]}"
                                      for r in range(8) for c in range(12)})
    ov.set_channels(["c0", "c1"], _RED_BLUE, np.uint16)
    ov.resize(360, 240)
    ov._fit()                                 # the fit an unshown widget never gets an event for
    assert ov._cd < V._CELL, "the plate fits 1:1 here, so a quick repaint would not sub-sample"
    ov.add_tile(0, 0, "A1", _tile([1000, 2000]))
    ov.recomposite(quick=True)
    cached = ov._disp.get("raw")
    assert cached is not None, "no display thumbnail was cached, so the drag has nothing to reuse"

    ov.set_channel_window(0, 10.0, 900.0)
    assert ov._disp.get("raw") is cached, "a contrast change threw away the thumbnail cache"

    ov.add_tile(7, 11, "H12", _tile([3000, 4000]))    # far corner: new pixels
    assert ov._disp.get("raw") is not cached, "new pixels did not invalidate the thumbnail"
    ov.recomposite(quick=True)
    shown = _rgb(ov)
    assert shown[shown.shape[0] // 2:, shown.shape[1] // 2:].any(), (
        "the newly added well never appeared — the plate composited a stale thumbnail")


def test_contrast_is_connected_once_not_once_per_ingest(qapp, stub_detail, squid_dataset):
    """The detail viewer is a singleton that outlives every ingest.

    A per-ingest `connect` stacks duplicate slots, so the Nth ingest re-runs the handler N times
    per drag. Counted through the plate's own setter, because the visible symptom of a stacked
    slot is work done N times, not a wrong final value.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    for _ in range(3):
        win.ingest(str(root))
    qapp.processEvents()

    calls = []
    real = win._overview.follow_channel_window
    win._overview.follow_channel_window = lambda ch, lo, hi: (calls.append((ch, lo, hi)),
                                                              real(ch, lo, hi))
    win._detail.drag_contrast(0, 5.0, 500.0)
    qapp.processEvents()
    assert len(calls) == 1, f"one drag reached the plate {len(calls)} times — slots have stacked"
    win.close()


def test_channel_store_survives_an_operator_run(qapp, stub_detail, squid_dataset, tmp_path):
    # D3: the store lives in the widget, so the toggle works on the operator layer too — not just
    # on the raw preview. Both layers keep their own (C, H, W) store.
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: "mip" in win._overview._store
                        and len(win._overview._tiles_by_layer.get("mip", ())) == 2)
    assert set(win._overview._store) >= {"mip"}        # the operator layer has its own store
    before = _rgb(win._overview).copy()
    win._overview.set_channel_visible(0, False)
    assert not np.array_equal(_rgb(win._overview), before)
    win._stop_worker()
    win.close()


# --- IMA-205: exploration tabs ---------------------------------------------------------------
#
# The tab is a multi-instance container scoped to a region subset. Identity is content-addressed,
# operator results are filed per-tab, and closing a tab must stop its run and free its canvas.

def test_exploration_tab_key_is_order_independent_and_set_based():
    k = V.exploration_tab_key("acq", ["B3", "B2"])
    assert k == V.exploration_tab_key("acq", ["B2", "B3"])        # drag order must not matter
    assert k == V.exploration_tab_key("acq", ["B2", "B3", "B2"])  # duplicates collapse
    assert k != V.exploration_tab_key("acq", ["B2"])              # a different set is a different tab
    assert k.startswith("exp:")


def test_exploration_tab_key_includes_acquisition_identity():
    # the SAME well ids on a DIFFERENT plate must never dedupe onto a stale tab
    assert V.exploration_tab_key("plate_a", ["B2"]) != V.exploration_tab_key("plate_b", ["B2"])


def test_exploration_tab_key_rejects_empty():
    with pytest.raises(ValueError):
        V.exploration_tab_key("acq", [])


def test_exploration_tab_key_stable_at_plate_scale():
    many = [f"B{i}" for i in range(1, 1537)]
    k = V.exploration_tab_key("acq", many)
    assert k == V.exploration_tab_key("acq", list(reversed(many)))
    assert len(k) < 32                       # bounded — it is a tab key, not a serialized set


def test_exploration_tab_label_is_human_readable():
    assert V.exploration_tab_label(["B2"]) == "B2"
    assert V.exploration_tab_label(["B5", "B2", "B3"]) == "B2–B5 (3)"
    assert "exp:" not in V.exploration_tab_label(["B2", "B3"])   # never show the hash as a title


def test_operator_layer_key_namespaces_only_when_scoped():
    assert V.operator_layer_key("mip", None) == "mip"             # plate-wide: unchanged behavior
    assert V.operator_layer_key("mip", "exp:ab12") == "mip@exp:ab12"


def test_open_exploration_tab_lists_exactly_the_selection(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B3"])
    assert key is not None
    tab = win._op_tabs[key]
    assert isinstance(tab, V._ExplorationTab)
    assert tab.regions == ["B3"]
    assert tab.listing.text() == "B3"                 # the tab shows exactly what it is scoped to
    assert win._explore_tabs.currentWidget() is tab   # ...in PANE 3, not the process console
    assert win._left_tabs.indexOf(tab) == -1
    win.close()


def test_open_exploration_tab_same_selection_focuses_not_duplicates(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n0 = win._explore_tabs.count()
    k1 = win.open_exploration_tab(["B2", "B3"])
    k2 = win.open_exploration_tab(["B3", "B2"])       # same SET, different order
    assert k1 == k2
    assert win._explore_tabs.count() == n0 + 1        # one tab, not two
    k3 = win.open_exploration_tab(["B3"])             # a different set DOES open another
    assert k3 != k1
    assert win._explore_tabs.count() == n0 + 2
    win.close()


def test_open_exploration_tab_rejects_empty_and_unknown(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n0 = win._explore_tabs.count()
    assert win.open_exploration_tab([]) is None
    assert "empty selection" in win._readout.text().lower()
    assert win.open_exploration_tab(["ZZ99"]) is None          # named, not a raw KeyError
    assert "not in this acquisition" in win._readout.text().lower()
    assert win._explore_tabs.count() == n0
    assert win._explore_tabs.isHidden()          # a REFUSED drag must not reveal pane 3
    win.close()


def test_open_exploration_tab_needs_an_acquisition(qapp, stub_detail):
    win = V.PlateWindow(None)
    assert win.open_exploration_tab(["B2"]) is None
    assert "acquisition" in win._readout.text().lower()
    win.close()


def test_run_operator_rejects_empty_and_unknown_regions(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path), regions=[])
    assert "empty selection" in win._readout.text().lower()
    assert win._worker is None
    win.run_operator("mip", out_parent=str(tmp_path), regions=["ZZ99"])
    assert "not in this acquisition" in win._readout.text().lower()
    assert win._worker is None                                  # never started
    win.close()


def test_subset_run_scopes_slider_and_remaps_push_index(qapp, stub_detail, squid_dataset):
    """The regression that decision 3 would have introduced without the remap.

    B3 is plate index 1, but in a ['B3'] subset its slider position is 0. The worker emits the
    GLOBAL index, so an unremapped push would address slot 1 of a 1-entry slider."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._fov_index["B3"]["idx"] == 1                     # global index is 1
    win._detail.arrays.clear()
    win.run_operator("mip", regions=["B3"], save=False)
    assert win._detail._fov_labels == ["B3:0"]                  # slider is the SUBSET, not the plate
    assert _drain_until(qapp, lambda: len(win._detail.arrays) > 0)
    pushed = {a[1] for a in win._detail.arrays}
    assert pushed == {0}, f"push landed at {pushed}, expected subset position 0"
    assert max(pushed) < len(win._detail._fov_labels)           # never out of range
    win._stop_worker(); win.close()


def test_whole_plate_run_keeps_identity_indexing(qapp, stub_detail, squid_dataset, tmp_path):
    """Regression guard: the remap must not disturb the shipped whole-plate path."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._detail.arrays.clear()
    win.run_operator("mip", out_parent=str(tmp_path))
    assert win._push_index is None                              # identity for a full plate
    assert _drain_until(qapp, lambda: len(win._detail.arrays) >= 2)
    assert {a[1] for a in win._detail.arrays} == {0, 1}         # both plate indices, unchanged
    win._stop_worker(); win.close()


def test_preview_spinner_still_runs_first_n_wells(qapp, stub_detail, squid_dataset, monkeypatch):
    """REGRESSION for the preview_limit -> regions= collapse: the shipped spinner call site."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    seen = {}
    real = V.PlateWindow.run_operator

    def spy(self, key, out_parent=None, regions=None, save=True, tab_key=None):
        seen["regions"] = regions
        return real(self, key, out_parent=out_parent, regions=regions, save=save, tab_key=tab_key)
    monkeypatch.setattr(V.PlateWindow, "run_operator", spy)

    tab = win._build_run_tab(V._OPERATIONS_BY_KEY["mip"])       # the real MIP tab
    prev = [b for b in tab.findChildren(QPushButton) if b.text() == "Preview"][0]
    spin = tab.findChildren(QSpinBox)[0]
    spin.setValue(1)
    prev.click()
    assert seen["regions"] == ["B2"], "preview must still run the FIRST N wells"
    win._stop_worker(); win.close()


def test_operator_tab_opened_twice_is_one_tab(qapp, stub_detail, squid_dataset):
    """REGRESSION: exploration tabs are multi-instance, operator tabs must stay singletons."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n0 = win._left_tabs.count()
    win._activate_operator("mip")
    win._activate_operator("mip")
    assert win._left_tabs.count() == n0 + 1
    win.close()


def test_two_tabs_same_operator_get_separate_layers(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    k1 = win.open_exploration_tab(["B2"])
    win.run_operator("mip", regions=["B2"], save=False, tab_key=k1)
    assert _drain_until(qapp, lambda: not win._busy())
    k2 = win.open_exploration_tab(["B3"])
    win.run_operator("mip", regions=["B3"], save=False, tab_key=k2)
    assert _drain_until(qapp, lambda: not win._busy())
    keys = {ly.key for ly in win._op_stack.layers()}
    assert f"mip@{k1}" in keys and f"mip@{k2}" in keys          # distinct layers, no collision
    assert f"mip@{k1}" in win._overview._op_canvas
    assert f"mip@{k2}" in win._overview._op_canvas
    win._stop_worker(); win.close()


def test_closing_tab_mid_run_stops_worker_and_frees_canvas(qapp, stub_detail, squid_dataset):
    """CRITICAL gap: no test, no error handling, and silent memory growth before this change."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B2", "B3"])
    win.run_operator("mip", regions=["B2", "B3"], save=False, tab_key=key)
    layer = f"mip@{key}"
    assert win._active_op_key == layer
    idx = win._explore_tabs.indexOf(win._op_tabs[key])
    win._close_op_tab(idx, win._explore_tabs)                    # close it, possibly mid-run
    assert _drain_until(qapp, lambda: not win._busy())
    assert layer not in {ly.key for ly in win._op_stack.layers()}   # layer dropped
    assert layer not in win._overview._op_canvas                    # ~plate-sized canvas freed
    assert layer not in win._overview._op_final
    assert layer not in win._overview._store                        # ...and the ~95 MB per-channel
    assert layer not in win._overview._final_arr                    #    store with it (IMA-206)
    assert win._active_op_key is None
    win.close()


def test_closing_idle_exploration_tab_is_clean(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B2"])
    n = win._explore_tabs.count()
    idx = win._explore_tabs.indexOf(win._op_tabs[key])
    win._close_op_tab(idx, win._explore_tabs)
    assert win._explore_tabs.count() == n - 1
    assert key not in win._op_tabs
    assert win._explore_tabs.isHidden()          # last tab gone -> pane 3 collapses again
    win.close()


def test_home_tab_is_never_closable(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n = win._left_tabs.count()
    win._close_op_tab(0)
    assert win._left_tabs.count() == n
    win.close()


def test_busy_guard_covers_retired_workers(qapp, stub_detail, squid_dataset, tmp_path):
    """_stop_worker clears self._worker while the retired thread drains — the guard must still
    refuse a new run, or closing a tab lets two workers hit the same reader at once."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    win._stop_worker()
    assert win._worker is None
    if win._busy():                                    # still draining -> a new run must be refused
        win.run_operator("mip", out_parent=str(tmp_path))
        assert win._worker is None
        assert "already processing" in win._readout.text().lower()
    assert _drain_until(qapp, lambda: not win._busy())
    win.close()


def test_tab_switch_repoints_detail_and_closing_it_restores_plate(qapp, stub_detail, squid_dataset):
    """IMA-237 moved exploration into PANE 3, so scope is owned by pane 3's front tab, not by
    whatever is in front of the process console. Switching pane 1 back to 'Process wells' must
    therefore NOT un-scope the viewer (pane 3 is still right there, still showing the subset);
    closing the exploration tab is what restores the whole plate."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B3"])
    qapp.processEvents()
    assert win._detail._fov_labels == ["B3:0"]                  # follows the exploration tab
    assert win._active_exploration is win._op_tabs[key]
    win._left_tabs.setCurrentIndex(0)                           # pane 1 back to "Process wells"
    qapp.processEvents()
    assert win._detail._fov_labels == ["B3:0"]                  # pane 3 still owns the scope
    assert win._active_exploration is win._op_tabs[key]
    win._close_op_tab(win._explore_tabs.indexOf(win._op_tabs[key]), win._explore_tabs)
    qapp.processEvents()
    assert win._detail._fov_labels == ["B2:0", "B3:0"]          # whole plate restored
    assert win._active_exploration is None
    win.close()


def test_subset_tab_registers_raw_paths_at_subset_positions(qapp, stub_detail, squid_dataset):
    """The raw bulk-register path indexes the slider too — it must use subset positions, not the
    global plate index, or B3 (plate idx 1) would register past the end of a 1-entry slider."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._detail.registered.clear()
    win.open_exploration_tab(["B3"])
    qapp.processEvents()
    if win._detail.registered:
        assert {r[1] for r in win._detail.registered} == {0}
    win.activate_well("B3", 0)
    assert all(r[1] < len(win._detail._fov_labels) for r in win._detail.registered)
    win.close()


def test_ingest_closes_exploration_tabs(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B2", "B3"])
    assert key in win._op_tabs
    win.ingest(str(root))                              # re-open: tabs belong to the old _fov_index
    qapp.processEvents()
    assert key not in win._op_tabs
    assert win._explore_tabs.count() == 0
    assert win._explore_tabs.isHidden()           # ...and pane 3 collapses with them
    assert win._active_exploration is None
    win.close()


def test_subset_save_is_disk_guarded(qapp, stub_detail, squid_dataset, monkeypatch, tmp_path):
    """The guard used to be skipped entirely for subsets (`if not ok and regions is None`)."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    class _Tiny:
        free = 1                                        # a byte free: everything must be refused
    monkeypatch.setattr("shutil.disk_usage", lambda p: _Tiny())
    win.run_operator("mip", out_parent=str(tmp_path), regions=["B3"], save=True)
    assert win._worker is None, "a subset save must be blocked when the disk can't hold it"
    assert "free space" in win._readout.text().lower()
    win.close()


def test_check_disk_scales_with_subset_size(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _, full_gb, _ = win._check_disk(tmp_path / "x.hcs")
    one = win._order[:1]
    _, one_gb, _ = win._check_disk(tmp_path / "x.hcs", regions=one)
    assert one_gb < full_gb                             # a 1-well run is not a whole-plate estimate
    # the estimate counts FIELDS (IMA-187 runs every FOV), so scale by this well's share of them
    total_fields = sum(len(win._meta["fovs_per_region"][r]) for r in win._fov_index)
    share = len(win._meta["fovs_per_region"][one[0]]) / total_fields
    assert one_gb == pytest.approx(full_gb * share, rel=0.01)
    win.close()


def test_note_partial_output_marks_a_stopped_plate(qapp, stub_detail, squid_dataset, tmp_path):
    """A save run that is stopped mid-write leaves a real-looking plate.ome.zarr holding only some
    wells. Mark it, so 'Open a computed MIP' can refuse it instead of showing a truncated plate as
    a finished one."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    out = tmp_path / "acq.hcs"
    (out / "plate.ome.zarr").mkdir(parents=True)
    win._run_out_dir = str(out)
    win._note_partial_output()
    assert (out / "INCOMPLETE").exists()
    assert win._run_out_dir is None                     # consumed, so it can't leak to a later run
    win.close()


def test_open_computed_refuses_an_incomplete_plate(qapp, stub_detail, tmp_path, monkeypatch):
    base = tmp_path / "acq.hcs"
    (base / "plate.ome.zarr").mkdir(parents=True)
    (base / "plate.ome.zarr" / "zarr.json").write_text("{}")
    (base / "INCOMPLETE").write_text("stopped\n")
    win = V.PlateWindow(None)
    monkeypatch.setattr(V.QFileDialog, "getExistingDirectory", lambda *a, **k: str(base))
    win._open_computed()
    assert "incomplete" in win._readout.text().lower()
    win.close()


def test_completed_save_run_is_not_marked_incomplete(qapp, stub_detail, squid_dataset, tmp_path):
    """The other half of the invariant: a run that finishes must NOT be flagged."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B2", "B3"])
    win.run_operator("mip", out_parent=str(tmp_path), regions=["B2", "B3"], save=True, tab_key=key)
    assert _drain_until(qapp, lambda: not win._busy(), timeout=90)
    out = tmp_path / f"{win._acq_name}.hcs"
    idx = win._explore_tabs.indexOf(win._op_tabs[key])
    win._close_op_tab(idx, win._explore_tabs)           # close AFTER it finished
    assert not (out / "INCOMPLETE").exists(), "a completed plate must not be flagged incomplete"
    win.close()


def test_operation_stack_remove_and_remove_suffix():
    from squidmip._layers import OperationStack
    st = OperationStack()
    st.add("mip@exp:a", "MIP · a")
    st.add("mip@exp:b", "MIP · b")
    st.add("mip", "MIP")
    assert st.remove_suffix("@exp:a") == ["mip@exp:a"]
    keys = {ly.key for ly in st.layers()}
    assert keys == {"raw", "mip@exp:b", "mip"}
    assert st.remove("raw") is False                    # the base layer is never removable
    assert st.remove("mip") is True
    assert st.remove("mip") is False


def test_second_ingest_resets_state(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: win._overview is not None and len(win._overview._tiles) == 2)
    win.ingest(str(root))            # second open: must stop the old worker + reset state
    qapp.processEvents()
    time.sleep(0.1)
    qapp.processEvents()
    assert len(win._fov_index) == 2                              # rebuilt, not accumulated
    assert len(win.findChildren(V.PlateOverview)) == 1           # one overview, not stacked
    assert set(win._overview._status.values()) == {"empty"}     # fresh grey plate
    win._stop_worker()
    win.close()


# --- IMA-187 wiring guard -------------------------------------------------------------
# The mosaic half of IMA-187 shipped DEAD: `_OperatorWorker` was constructed without
# `n_fovs`, so it defaulted to 1 and `_boxes` was always {}; and `set_mosaic_boxes` had
# zero callers in the repo. Every inherited viewer test still passed, because they only
# exercise the single-tile path. These fail on that dead wiring, so the 227 -> 206 -> 187
# rebase cannot silently drop the feature again.

def test_operator_worker_is_constructed_for_multi_fov_not_defaulted_to_one(
        qapp, stub_detail, squid_dataset, tmp_path, monkeypatch):
    """run_operator must hand the worker a multi-FOV n_fovs, or the mosaic is unreachable."""
    seen = {}
    real_init = V._OperatorWorker.__init__

    def spy(self, *a, **kw):
        seen["n_fovs"] = kw.get("n_fovs", "NOT-PASSED")
        return real_init(self, *a, **kw)

    monkeypatch.setattr(V._OperatorWorker, "__init__", spy)
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: "n_fovs" in seen)

    try:
        assert seen.get("n_fovs") != "NOT-PASSED", (
            "run_operator constructed _OperatorWorker without n_fovs, so it defaults to 1, "
            "_boxes is always {}, and the coordinate-placed mosaic can never render.")
        assert seen["n_fovs"] != 1, (
            f"n_fovs={seen['n_fovs']!r}; the mosaic path requires n_fovs != 1 "
            "(_OperatorWorker: `_boxes = _mosaic_boxes(meta) if n_fovs != 1 else {}`).")
    finally:
        win._stop_worker(); win.close()


def test_set_mosaic_boxes_is_actually_called_by_the_viewer(
        qapp, stub_detail, squid_dataset, tmp_path, monkeypatch):
    """PlateOverview.set_mosaic_boxes exists but nothing calls it -- boxes never reach paint."""
    calls = []
    monkeypatch.setattr(V.PlateOverview, "set_mosaic_boxes",
                        lambda self, boxes: calls.append(boxes))
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: bool(calls))

    try:
        assert calls, (
            "set_mosaic_boxes was never called. PlateOverview._boxes stays empty, so _fov_at() "
            "always returns FOV 0 and the mosaic is invisible to hit-testing and paint.")
    finally:
        win._stop_worker(); win.close()


# --- IMA-218: the mosaic's PLACEMENT and PIXELS, not just its wiring --------------------------
#
# The two guards above prove the mosaic path is REACHED (n_fovs is passed, set_mosaic_boxes is
# called). Neither proves a FOV lands anywhere in particular, and neither ever looks at a pixel:
# a mosaic that stacked every field at (0, 0), or mirrored the well vertically, passes both.
# Those are precisely the failures `_placement.py`'s docstring is written against -- they do not
# raise, they draw a plausible-but-wrong picture. So these drive the REAL widget and assert on
# geometry and on rendered pixels.

def test_mosaic_places_each_fov_at_its_own_stage_offset(qapp, stub_detail, squid_dataset,
                                                        tmp_path):
    """FOVs must occupy DISTINCT boxes derived from stage coords, not pile up at the origin.

    The fixture's fov 1 is +0.5 mm in x from fov 0 at the same y, so the mosaic must place it to
    the RIGHT, on the same row. A collapsed placement, a scale error and a transposed axis each
    break one of these assertions.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: bool(win._overview._boxes))
    try:
        boxes = win._overview._boxes
        assert boxes, "no mosaic boxes reached the widget"
        (t0, l0, h0, w0) = boxes[("B2", 0)]
        (t1, l1, h1, w1) = boxes[("B2", 1)]
        assert (t0, l0) != (t1, l1), (
            f"both FOVs placed at the same spot {(t0, l0)}: the mosaic collapsed into one pile, "
            "which is what a dropped offset or a zeroed pixel_size_um looks like.")
        assert l1 > l0, (
            f"fov 1 is +0.5 mm in x from fov 0, so it must sit to the RIGHT: got left {l1} <= {l0}. "
            "A negated or transposed x axis mirrors every well horizontally.")
        assert t1 == t0, (
            f"the two FOVs share a stage y, so they must share a row: got top {t1} != {t0}.")
        assert h0 > 0 and w0 > 0 and l1 + w1 <= V._CELL, (
            f"box {(t1, l1, h1, w1)} escapes the {V._CELL}px cell and would bleed into its neighbour.")
    finally:
        win._stop_worker(); win.close()


def _cell_of(img, ri, ci):
    """Crop cell (ri, ci) out of the plate's composited montage (exactly _CELL px per cell)."""
    buf = img.constBits().asstring(img.byteCount())
    a = np.frombuffer(buf, np.uint8).reshape(img.height(), img.bytesPerLine() // 3, 3)
    a = a[:, :img.width(), :]
    return a[ri * V._CELL:(ri + 1) * V._CELL, ci * V._CELL:(ci + 1) * V._CELL].astype(int)


def test_mosaic_cell_composites_real_structured_pixels(qapp, stub_detail, squid_dataset, tmp_path):
    """Drive the real widget and LOOK at the acquired cell: it must hold real, varying imagery.

    Measured on the MONTAGE (``_active_source``), one cell of which is exactly _CELL x _CELL, and
    NOT on ``grab()`` of the whole widget. That is deliberate. The widget also paints row/column
    labels, a 3px grid, status dots, the red current-well box and (IMA-220) the carrier
    photograph; on this fixture the plate auto-fits to ~12 px per cell, so a cropped cell is
    almost entirely chrome and its variance stays high with the montage blanked out entirely.
    A whole-widget dynamic-range assertion therefore passes with the tiles deleted -- it was
    written that way first, and a mutation that returns a blank montage still passed it. The
    montage crop kills that mutant, which is the only reason to prefer it.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))

    def _mosaic_complete():
        """Every field this test asserts on has actually landed.

        The wait used to be `len(_tiles) >= 2`, i.e. "two CELLS have been touched" -- but a cell is
        a MOSAIC of FOVs (IMA-253) and the assertions below require BOTH of B2's fields, plus
        enough signal in the first tiled cell to show dynamic range. Waiting for a weaker condition
        than the one asserted makes the outcome depend on how far the background stream happened to
        get, so the test passed or failed according to how fast compositing was that day. It went
        red on IMA-261 purely because the repaint got faster.
        Timing out here does NOT pass the test: the assertions still run, and still fail.
        """
        ov = win._overview
        if len(ov._tiles) < 2 or not ov._boxes or "B2" not in win._fov_index:
            return False
        ri, ci = win._fov_index["B2"]["rc"]
        store = ov._store_for(ov._active)
        if store is None:
            return False
        cell = store[:, ri * V._CELL:(ri + 1) * V._CELL, ci * V._CELL:(ci + 1) * V._CELL]
        return all(np.count_nonzero(cell[:, t:t + h, l:l + w]) > 0
                   for t, l, h, w in (ov._boxes[("B2", f)] for f in (0, 1)))

    _drain_until(qapp, _mosaic_complete)
    try:
        ov = win._overview
        ov.recomposite(ov._active)
        qapp.processEvents()
        img = ov._active_source()
        tiled = sorted(ov._tiles_by_layer.get(ov._active, set()))
        assert tiled, "no cell has an image on the active layer"
        got = _cell_of(img, *tiled[0])
        assert got.size, "the acquired cell fell outside the montage"
        assert int(got.max()) - int(got.min()) > 30, (
            f"acquired-cell dynamic range is only {int(got.max()) - int(got.min())}: the cell is "
            "effectively blank (tiles never composited, or contrast collapsed the window).")
        # NOTE: no std-over-the-whole-cell assertion. This fixture's frames are 4x4 px, so their
        # boxes cover a few percent of the 88px cell and the rest is legitimate zero padding --
        # whole-cell std is ~0.4 even when the mosaic is perfect. Coverage of the cell is the
        # mosaic's business (asserted by box below), brightness is the tile's.
        # ...and it must differ from an UNACQUIRED cell, or "structure" could just be background.
        empty = [(r, c) for r in range(ov._nr) for c in range(ov._nc) if (r, c) not in tiled]
        if empty:
            ref = _cell_of(img, *empty[-1])
            # MAX, not mean: this fixture's fields cover a few percent of the cell, so a mean over
            # all 88x88 px is ~0.004 even for a perfectly drawn mosaic. Mean would only be asking
            # "does the tile fill the cell", which is not this assertion's question.
            assert int(np.abs(got - ref).max()) > 30, (
                "an acquired cell is indistinguishable from an empty one: nothing was drawn.")
        # ...and the mosaic must reach BOTH fields' sub-boxes, not just fov 0's.
        if ov._boxes:
            ri, ci = win._fov_index["B2"]["rc"]
            cell = ov._store_for(ov._active)[:, ri * V._CELL:(ri + 1) * V._CELL,
                                             ci * V._CELL:(ci + 1) * V._CELL]
            for fov in (0, 1):
                top, left, h, w = ov._boxes[("B2", fov)]
                assert int(np.count_nonzero(cell[:, top:top + h, left:left + w])) > 0, (
                    f"fov {fov}'s sub-box is entirely zero: only part of the mosaic was composited.")
    finally:
        win._stop_worker(); win.close()


# --- IMA-207: contrast scope, and the two contrast bugs found reviewing it --------------------
#
# Ported from the ima-207 branch onto IMA-206's architecture. The branch carried its own parallel
# `_raw_tiles` retention because PlateOverview did not yet own the pixels; it does now (the
# per-layer native-dtype store), so the scope re-composites THAT and no second copy of every tile
# is kept. The design decision is unchanged and is the whole point of the ticket: contrast scope
# is a DISPLAY control, never a run parameter.

def _plate_rgb(ov):
    """The overview's composited montage as (H, W, 3) uint8 — the pixels, not the chrome."""
    img = ov._active_source()
    buf = img.constBits().asstring(img.byteCount())
    a = np.frombuffer(buf, np.uint8).reshape(img.height(), img.bytesPerLine() // 3, 3)
    return a[:, :img.width(), :]


def _two_well_plate(bright_peak=40000, dim_peak=600):
    """A 1x2 plate: one BRIGHT well beside one DIM well, both spread (non-degenerate)."""
    ov = V.PlateOverview(["A"], ["1", "2"], {(0, 0): "A1", (0, 1): "A2"})
    ov.resize(400, 300)
    ov.set_channels(["c0"], np.array([[1.0, 1.0, 1.0]], np.float32), dtype=np.uint16)
    for (rc, wid), peak in zip({(0, 0): "A1", (0, 1): "A2"}.items(), (bright_peak, dim_peak)):
        tile = np.linspace(peak * 0.4, peak, V._CELL * V._CELL).astype(np.uint16)
        ov.add_tile(rc[0], rc[1], wid, tile.reshape(1, V._CELL, V._CELL))
    return ov


def _cell_mean(ov, ri, ci):
    ov.recomposite(ov._active)
    return float(_plate_rgb(ov)[ri * V._CELL:(ri + 1) * V._CELL,
                                ci * V._CELL:(ci + 1) * V._CELL].mean())


def test_running_contrast_flat_channel_yields_degenerate_window():
    """A flat channel has no contrast to show, so the window must be DEGENERATE (span <= 0) and
    _window must render it BLACK.

    Regression: window() returned ``max(hi, lo + 1)`` — a 1 data-unit span against a
    ``65535/512 ~= 128``-unit histogram bin. ``(v - lo) / 1`` then clipped to 1.0, so a blank,
    dead or saturated well rendered FULL WHITE and read as signal. Blank wells are normal on a
    partially acquired plate, so this was on screen constantly.
    """
    from squidmip._montage import _window

    rc = V._RunningContrast(1, float(np.iinfo(np.uint16).max))
    flat = np.full((8, 8), 500.0, dtype=np.float32)
    rc.add(0, flat)
    lo, hi = rc.window(0)
    assert hi - lo <= 0, "a flat channel must produce a degenerate window, not a 1-unit span"
    assert np.all(_window(flat, lo, hi) == 0.0), "a flat channel must render black, not white"


def test_running_contrast_saturated_channel_renders_black():
    """The same guard at the top of the range: a fully saturated well is flat too."""
    from squidmip._montage import _window

    dmax = float(np.iinfo(np.uint16).max)
    rc = V._RunningContrast(1, dmax)
    sat = np.full((8, 8), dmax, dtype=np.float32)
    rc.add(0, sat)
    lo, hi = rc.window(0)
    assert np.all(_window(sat, lo, hi) == 0.0)


def test_running_contrast_spread_channel_still_windows():
    """The degenerate guard must not eat a real channel: a ramp keeps an ordered, usable window."""
    rc = V._RunningContrast(1, float(np.iinfo(np.uint16).max))
    rc.add(0, np.linspace(0, 60000, 64 * 64).astype(np.float32).reshape(64, 64))
    lo, hi = rc.window(0)
    assert hi > lo


def test_running_contrast_empty_histogram_is_full_range():
    """No tiles yet -> full range, unchanged behaviour."""
    rc = V._RunningContrast(2, 65535.0)
    assert rc.window(0) == (0.0, 65535.0)


def test_blank_well_renders_black_not_white_through_the_widget(qapp):
    """End to end: the bug was visible on the PLATE, so assert on the plate's pixels."""
    ov = V.PlateOverview(["A"], ["1"], {(0, 0): "A1"})
    ov.set_channels(["c0"], np.array([[1.0, 1.0, 1.0]], np.float32), dtype=np.uint16)
    ov.add_tile(0, 0, "A1", np.full((1, V._CELL, V._CELL), 7, np.uint16))   # a blank channel
    ov.recomposite(ov._active)
    cell = _plate_rgb(ov)[:V._CELL, :V._CELL]
    assert float(cell.mean()) < 1.0, (
        f"a blank well rendered at mean {cell.mean():.1f} — it must be black, not white.")


def test_reopened_plate_windows_globally_like_the_run_that_wrote_it(qapp):
    """A reopened plate.ome.zarr must agree with the run that wrote it.

    _ComputedPlateWorker used to window each tile independently with its own percentiles — that is
    per-region contrast applied unconditionally, so a dim well and a bright well came back
    indistinguishable and the reopened plate looked nothing like the run. It now emits NATIVE
    per-channel tiles and lets PlateOverview window them, exactly like every other producer, which
    is what makes the plate look the same however it got filled.
    """
    import inspect

    src = inspect.getsource(V._ComputedPlateWorker)
    assert "np.percentile" not in src, (
        "_ComputedPlateWorker is windowing tiles itself again — that is per-region contrast "
        "imposed on the reopen path, and it makes a reopened plate disagree with its own run.")
    assert "_window(" not in src, "the reopen path must emit native tiles, not pre-windowed RGB"
    sig = inspect.signature(V._ComputedPlateWorker.__init__)
    assert "colors" not in sig.parameters, (
        "a worker that needs colours is compositing; the widget owns compositing (IMA-206).")


# --- IMA-205 + IMA-221: the SHIFT GESTURE opens the exploration tab ---------------------------
#
# This is the user's verbatim sentence, end to end: "hold shift to open an 'exploration' tab with
# the selected FOV subset". IMA-221 landed the marquee; before this wiring `open_exploration_tab`
# had no UI entry point at all and was reachable only programmatically.

def _freeze(ov, cd=20.0):
    """Freeze the plate view so synthetic widget coordinates hit the cells we mean (paintEvent's
    auto-fit would otherwise move the plate under the drag)."""
    ov._user_view = True
    ov._cd, ov._ox, ov._oy = cd, 0.0, 0.0
    return ov


def _shift_drag_over(win, wells, cd=20.0):
    """Shift-drag a marquee across exactly `wells` on the window's own plate."""
    ov = _freeze(win._overview, cd)
    rcs = [win._fov_index[w]["rc"] for w in wells]
    r0, c0 = min(r for r, _ in rcs), min(c for _, c in rcs)
    r1, c1 = max(r for r, _ in rcs), max(c for _, c in rcs)
    if (r0, c0) == (r1, c1):                      # one cell: still a DRAG, not a Shift+click
        a, b = _within(r0, c0, cd)
    else:
        a, b = _pt(r0, c0, cd), _pt(r1, c1, cd)
    _drag(ov, a, b, Qt.ShiftModifier)


def test_shift_drag_opens_an_exploration_tab_scoped_to_the_selected_wells(
        qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset                            # B2, B3
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _shift_drag_over(win, ["B3"])                      # marquee over ONE of the two wells
    qapp.processEvents()

    tabs = [win._explore_tabs.widget(i) for i in range(win._explore_tabs.count())
            if isinstance(win._explore_tabs.widget(i), V._ExplorationTab)]
    assert len(tabs) == 1, "the shift-drag gesture did not open an exploration tab"
    tab = tabs[0]
    assert tab.regions == ["B3"]                       # scoped to EXACTLY the selected wells
    assert tab.listing.text() == "B3"
    assert win._explore_tabs.currentWidget() is tab    # ...brought to the front of PANE 3
    assert not win._explore_tabs.isHidden(), "the shift-drag did not reveal pane 3"
    assert win._active_exploration is tab
    assert win._detail._fov_labels == ["B3:0"]         # the viewer follows the subset
    assert win._selected_regions == ["B3"]             # IMA-221 scoping is untouched
    win.close()


def test_shift_drag_over_several_wells_scopes_the_tab_to_all_of_them(
        qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _shift_drag_over(win, ["B2", "B3"])
    qapp.processEvents()
    tab = win._explore_tabs.currentWidget()
    assert isinstance(tab, V._ExplorationTab)
    assert tab.regions == ["B2", "B3"]
    assert win._detail._fov_labels == ["B2:0", "B3:0"]
    win.close()


def test_repeating_the_same_shift_drag_focuses_the_same_tab(qapp, stub_detail, squid_dataset):
    """Content-addressed identity, driven through the GESTURE: dragging the same wells again must
    focus the open tab, not pile up duplicates on every stray drag."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _shift_drag_over(win, ["B3"])
    qapp.processEvents()
    first = win._explore_tabs.currentWidget()
    _shift_drag_over(win, ["B3"])
    qapp.processEvents()
    tabs = [win._explore_tabs.widget(i) for i in range(win._explore_tabs.count())
            if isinstance(win._explore_tabs.widget(i), V._ExplorationTab)]
    assert tabs == [first]
    win.close()


def test_shift_click_refines_the_selection_without_opening_a_tab(qapp, stub_detail, squid_dataset):
    """Only the DRAG opens a tab. Shift+click is the refine-one-well gesture, and since every
    distinct set is a distinct tab, opening one per corrective click would bury the real one."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = _freeze(win._overview)
    rc = win._fov_index["B3"]["rc"]
    ov.mousePressEvent(_mouse("press", _pt(*rc), Qt.ShiftModifier))
    ov.mouseReleaseEvent(_mouse("release", _pt(*rc), Qt.ShiftModifier, buttons=Qt.NoButton))
    qapp.processEvents()
    assert ov.selected_wells() == ["B3"]                       # selection still happens...
    assert win._explore_tabs.count() == 0 and win._explore_tabs.isHidden()   # ...no tab
    win.close()


def test_shift_drag_over_empty_plate_opens_nothing_and_says_nothing(
        qapp, stub_detail, squid_dataset):
    """A miss is a miss: no tab, and no 'empty selection' text stomping the readout."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    before = win._readout.text()
    ov = _freeze(win._overview)
    _drag(ov, _pt(0, 0), _pt(0, 0.5), Qt.ShiftModifier)        # row A: no acquired wells
    qapp.processEvents()
    assert win._explore_tabs.count() == 0 and win._explore_tabs.isHidden()
    assert win._readout.text() == before
    win.close()


# --- IMA-205 bug 1: closing an exploration tab MID-RUN stranded the whole view ----------------
# --- IMA-205 bug 2: a second tab opened MID-RUN sat in front lying about what was shown --------
#
# Same root cause: `_on_tab_changed` returned early while `_busy()` and nothing re-delivered the
# switch when the run drained. These use a worker that blocks until the test releases it, so the
# mid-run window is deterministic instead of a race against a 2-well run.

class _BlockingWorker(V.QThread):
    """An _OperatorWorker stand-in that stays RUNNING until stop() (or the test) releases it."""
    tileReady = V.pyqtSignal(int, int, str, object)
    pushReady = V.pyqtSignal(int, object)
    resultReady = V.pyqtSignal(str, int, object)     # full-res result -> napari layer group
    progress = V.pyqtSignal(int, int)
    finalReady = V.pyqtSignal(object)
    writtenReady = V.pyqtSignal(str)
    wellFailed = V.pyqtSignal(int, int)
    failed = V.pyqtSignal(str)
    finished_ok = V.pyqtSignal()
    streamEnded = V.pyqtSignal()

    def __init__(self, *a, **kw):
        super().__init__()
        import threading
        self.mosaic_boxes = {}
        # IMA-245: the window declares the array viewer's canvas from the worker's push geometry,
        # so a worker double has to carry it too — the frame square is what a per-FOV run reports.
        self.push_shape = (V._PUSH_PX, V._PUSH_PX)
        self.push_shape_estimated = False
        self._go = threading.Event()

    def run(self):
        self._go.wait(20)          # bounded: a hung test must not hang the suite

    def stop(self):
        self._go.set()

    release = stop


@pytest.fixture
def blocking_worker(monkeypatch):
    made = []
    monkeypatch.setattr(V, "_OperatorWorker", lambda *a, **kw: made.append(_BlockingWorker()) or made[-1])
    return made


def test_closing_the_front_tab_mid_run_restores_a_coherent_plate_view(
        qapp, stub_detail, squid_dataset, blocking_worker):
    """BUG 1 — click one well, see another.

    Closing the front exploration tab while its run is live used to strand everything: the switch
    back was dropped (`_busy()`), nothing re-emitted it when the run drained, so the FOV slider
    stayed pinned to the closed tab's subset, `_push_index` stayed stale and `_active_exploration`
    pointed at a deleted widget. Double-clicking another well then moved the red box but showed the
    OLD well's pixels — silently."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B3"])
    qapp.processEvents()
    win.run_operator("mip", regions=["B3"], save=False, tab_key=key)
    assert win._busy()                                        # the run is live and blocked
    assert win._push_index == {win._fov_index["B3"]["idx"]: 0}

    idx = win._explore_tabs.indexOf(win._op_tabs[key])
    win._close_op_tab(idx, win._explore_tabs)                 # close it MID-RUN
    assert _drain_until(qapp, lambda: not win._busy())
    qapp.processEvents()

    assert win._active_exploration is None                    # no dangling deleted widget
    assert win._push_index is None                            # back to identity plate indexing
    assert win._detail._fov_labels == ["B2:0", "B3:0"]        # slider is the whole plate again
    # ...and the double-click path is coherent again: the red box and the pixels agree.
    win._detail.nav.clear()
    win.activate_well("B2", 0)
    assert win._overview._sel == win._fov_index["B2"]["rc"]
    assert win._detail.nav[-1][0] == "B2"
    win.close()


def test_double_click_never_moves_the_box_to_a_well_the_viewer_cannot_show(
        qapp, stub_detail, squid_dataset):
    """The other half of BUG 1's symptom, on its own: while a subset tab scopes the slider, a well
    outside it cannot be shown — so the red box must NOT claim it is being shown."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.open_exploration_tab(["B3"])
    qapp.processEvents()
    win.activate_well("B3", 0)
    box = win._overview._sel
    win.activate_well("B2", 0)                                 # B2 is NOT in this tab's slider
    assert win._overview._sel == box, "the red box moved to a well the viewer isn't showing"
    assert win._current_well == "B3"
    assert "not in this tab" in win._readout.text()            # and it says so, instead of silence
    win.close()


def test_a_second_tab_opened_mid_run_syncs_when_the_run_finishes(
        qapp, stub_detail, squid_dataset, blocking_worker):
    """BUG 2 — the front tab lies.

    Opening a second exploration tab while the first tab's run is live left the new tab in front
    while the slider and plate still showed the FIRST tab's run, and it never resynced when the run
    finished. The switch is now deferred (and said out loud), then delivered on drain."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key_a = win.open_exploration_tab(["B2"])
    qapp.processEvents()
    win.run_operator("mip", regions=["B2"], save=False, tab_key=key_a)
    assert win._detail._fov_labels == ["B2:0"]                 # the live run's slider

    key_b = win.open_exploration_tab(["B3"])                   # ...open a SECOND tab mid-run
    qapp.processEvents()
    tab_b = win._op_tabs[key_b]
    assert win._explore_tabs.currentWidget() is tab_b          # it is in front of pane 3...
    assert win._detail._fov_labels == ["B2:0"]                 # ...but the view still shows tab A
    assert tab_b.sync_pending, "the front tab shows another tab's run and says nothing"
    assert tab_b.sync_note.isVisibleTo(tab_b)
    assert win._pending_resync

    blocking_worker[-1].release()                              # let the run finish
    assert _drain_until(qapp, lambda: not win._busy())
    qapp.processEvents()

    assert win._detail._fov_labels == ["B3:0"]                 # the front tab is now the truth
    assert win._active_exploration is tab_b
    assert not tab_b.sync_pending
    assert not win._pending_resync
    win.close()


def test_deferred_resync_survives_a_failed_run(qapp, stub_detail, squid_dataset, blocking_worker):
    """The resync hangs off QThread.finished, not finished_ok, so a run that fails or is stopped
    still hands the view back instead of leaving it pinned to a dead run's subset."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key_a = win.open_exploration_tab(["B2"])
    qapp.processEvents()
    win.run_operator("mip", regions=["B2"], save=False, tab_key=key_a)
    # Defer a switch WITHOUT stopping the run: open a second tab in pane 3 mid-run. (Closing tab A
    # would also defer, but _discard_exploration stops the worker, so the run would no longer be
    # the failing one this test is about.)
    key_b = win.open_exploration_tab(["B3"])
    qapp.processEvents()
    assert win._pending_resync
    assert win._detail._fov_labels == ["B2:0"]                 # still tab A's live run
    blocking_worker[-1].failed.emit("boom")
    blocking_worker[-1].release()
    assert _drain_until(qapp, lambda: not win._busy())
    qapp.processEvents()
    assert win._detail._fov_labels == ["B3:0"]                 # handed to the front tab anyway
    assert win._active_exploration is win._op_tabs[key_b]
    assert not win._pending_resync
# --- loupe: press-and-hold magnifier (IMA-208) ----------------------------------------------
#
# The gesture and the geometry are tested separately from the I/O: the state machine needs no
# pixels, and the pure math needs no Qt. Only the read tests touch a real written pyramid —
# which is why `pyramid_dataset` exists (a 4x4 fixture writes ONE level, so it cannot prove
# level selection at all).

class _FakeLoupeSource(V._LoupeSource):
    """A source with known pixels, so gesture tests don't need zarr or TIFF decode.

    It HOLDS A FIELD and slices it with ordinary numpy semantics, which is the whole point: the
    original fake ignored y0/x0 and returned ``np.full((2, h, w), 500)``, so it could not
    produce an empty array no matter what rectangle it was handed — and every gesture test ran
    on it while the real raw source was returning nothing over ~75% of each well (a negative
    origin makes ``a[-427:1399]`` empty, not an error). A test double that cannot express the
    failure it is standing in for is worse than no double: it certifies the bug."""

    def __init__(self, well_px=1000, n_levels=3, pixel_size_um=0.325, missing=()):
        self.well_px, self.n_levels, self.pixel_size_um = well_px, n_levels, pixel_size_um
        self._missing = set(missing)
        self.reads = []
        self._fields = {}

    def _field(self, level):
        """A (2, span, span) field at ``level``, with a per-pixel ramp so a crop's CONTENT
        identifies where it came from (a constant fill would hide an off-by-one origin)."""
        span = max(1, self.well_px >> int(level))
        if span not in self._fields:
            yy, xx = np.mgrid[0:span, 0:span]
            plane = ((yy + xx) % 1000).astype(np.uint16) + 1        # never 0 -> "read pixels"
            self._fields[span] = np.stack([plane, plane[::-1]])
        return self._fields[span]

    def available(self, well_id):
        if well_id in self._missing:
            return False, "not written yet"
        return True, ""

    def read_crop(self, well_id, level, y0, x0, h, w):
        self.reads.append((well_id, level, y0, x0, h, w))
        f = self._field(level)
        span = f.shape[-1]
        y0, x0, h, w = V.loupe_clamp_crop(y0, x0, h, w, span, span)   # what a real source must do
        step = V.loupe_decimation(max(h, w))
        return f[:, y0:y0 + h:step, x0:x0 + w:step]

    def coarse(self, well_id):
        return self._field(max(0, self.n_levels - 1))


def _loupe_win(qapp, root, tmp_path=None):
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _drain_until(qapp, lambda: win._overview is not None)
    return win


def _cell_center(ov, ri=0, ci=0):
    """Widget coords at the middle of a cell (matches PlateOverview._cell's mapping)."""
    ov._fit()
    ax, ay = ov._ox + V._HDR, ov._oy + V._COLH
    return int(ax + (ci + 0.5) * ov._cd), int(ay + (ri + 0.5) * ov._cd)


# -- pure geometry (no Qt, no I/O) --

def test_loupe_scale_never_upsamples_past_native():
    """The cap: one screen px per level-0 image px is as far as honest magnification goes."""
    for cd, well in ((20, 4168), (200, 4168), (1000, 1024), (5000, 1024)):
        s, m = V.loupe_scale(cd, well)
        assert s <= 1.0 or s == pytest.approx(cd / well)   # only "past native" exceeds 1.0
        assert m >= 1.0                                    # and it never shrinks


def test_loupe_inset_shows_at_most_one_whole_well():
    """A fixed 8x does not survive a 1536wp: a well is ~10 screen px at fit, so 8x would fill
    a third of the inset and the rest would have to come from NEIGHBOURING wells. The scale is
    floored so the inset shows at most one well — which is what the gesture means."""
    well, inset = 1024, 240
    s, m = V.loupe_scale(cd=10.6, well_px=well, inset_px=inset)   # 1536wp at fit
    region = inset / s                                            # image px the inset covers
    assert region <= well + 1                                     # never more than one well
    assert m > 8.0                                                # ...so the real gain exceeds 8x
    # In the band where 8x both fills the inset and stays under native, the plain target holds.
    s2, m2 = V.loupe_scale(cd=100, well_px=well, inset_px=inset)
    assert m2 == pytest.approx(8.0)
    assert s2 <= 1.0


def test_loupe_never_demagnifies_past_native_plate_zoom():
    """Wheel-zoom the plate BEYOND native and the loupe must not shrink what it points at.

    The native cap alone would drop the inset below the plate's own scale (M < 1) — a
    magnifier that makes things smaller. The floor keeps M >= 1; at that point there is no
    detail left to reveal, and the inset is labelled 'native' rather than claiming a gain."""
    s, m = V.loupe_scale(cd=4096, well_px=1024)     # plate already at 4x native
    assert m == pytest.approx(1.0)
    assert s == pytest.approx(4.0)                  # inset matches the plate, never below it
    for cd in (1, 10, 100, 1024, 2048, 8192):
        assert V.loupe_scale(cd, 1024)[1] >= 1.0


def test_loupe_scale_is_dynamic_in_plate_zoom():
    """Magnification is derived from the CURRENT zoom, not a constant: zoom the plate in and
    the loupe's gain falls away, reaching 1.0 (native) when the plate is already there."""
    well = 4168
    mags = [V.loupe_scale(cd, well)[1] for cd in (10, 100, 1000, 4168)]
    assert mags == sorted(mags, reverse=True)
    assert mags[0] > mags[-1]
    assert mags[-1] == pytest.approx(1.0)


def test_loupe_level_picks_coarsest_adequate_and_clamps():
    assert V.loupe_level(1.0, 5) == 0             # native display -> full-res level
    assert V.loupe_level(0.5, 5) == 1             # half scale -> level 1 is exactly enough
    assert V.loupe_level(0.25, 5) == 2
    assert V.loupe_level(0.01, 3) == 2            # clamped to the levels that exist
    assert V.loupe_level(0.01, 1) == 0            # a single-level plate always reads level 0


def test_loupe_crop_px_shrinks_with_level():
    # Same inset, coarser level -> fewer pixels to read. This is what keeps a zoomed-out
    # loupe cheap instead of pulling a 4168px plane.
    assert V.loupe_crop_px(1.0, 0, inset_px=240) == 240
    assert V.loupe_crop_px(0.25, 2, inset_px=240) == 240
    assert V.loupe_crop_px(0.25, 0, inset_px=240) == 960


def test_loupe_clamp_crop_shifts_the_origin_in_and_keeps_the_extent():
    assert V.loupe_clamp_crop(-427, -427, 1826, 1826, 2084, 2084) == (0, 0, 1826, 1826)
    assert V.loupe_clamp_crop(-5, 10, 32, 32, 640, 640) == (0, 10, 32, 32)
    assert V.loupe_clamp_crop(630, 630, 32, 32, 640, 640) == (608, 608, 32, 32)  # not a 10px sliver
    assert V.loupe_clamp_crop(0, 0, 9999, 9999, 64, 64) == (0, 0, 64, 64)        # rect > field
    ny = nx = 100
    for y0 in range(-150, 150, 7):                  # never negative, never past the field
        cy, cx, h, w = V.loupe_clamp_crop(y0, y0, 40, 40, ny, nx)
        assert 0 <= cy <= ny - h and 0 <= cx <= nx - w and (h, w) == (40, 40)


def test_loupe_decimation_bounds_the_sample_count_by_powers_of_two():
    assert V.loupe_decimation(240) == 1                       # already inset-sized
    assert V.loupe_decimation(V._LOUPE_MAX_CROP) == 1         # exactly at the ceiling
    assert V.loupe_decimation(V._LOUPE_MAX_CROP + 1) == 2
    for px in (600, 1826, 4168, 10000):
        assert px // V.loupe_decimation(px) <= V._LOUPE_MAX_CROP


def test_loupe_um_per_screen_px_refuses_to_guess():
    assert V.loupe_um_per_screen_px(0.325, 1.0) == pytest.approx(0.325)
    assert V.loupe_um_per_screen_px(0.325, 0.5) == pytest.approx(0.65)
    assert V.loupe_um_per_screen_px(None, 1.0) is None      # unknown -> no bar, never a guess
    assert V.loupe_um_per_screen_px(0, 1.0) is None
    assert V.loupe_um_per_screen_px(float("nan"), 1.0) is None


def test_composite_rgb_matches_manual_windowing():
    # IMA-242: the loupe's private `_composite_rgb` is gone; `composite` is the one compositor and
    # the loupe goes through it, so this asserts against the survivor.
    from squidmip._montage import composite
    planes = np.array([[[0.0, 10.0]], [[5.0, 5.0]]])
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wins = [(0.0, 10.0), (0.0, 10.0)]
    out = composite(planes, colors, wins).astype(float) / 255.0
    assert out.shape == (1, 2, 3)
    assert out[0, 0, 0] == pytest.approx(0.0, abs=0.01)      # ch0 at its window floor
    assert out[0, 1, 0] == pytest.approx(1.0, abs=0.01)      # ch0 at its window ceiling
    assert out[0, 0, 1] == pytest.approx(0.5, abs=0.01)      # ch1 mid-window, in green


def test_ima242_one_contrast_model_resolves_manual_over_auto():
    """The precedence rule lives in ONE place and every renderer asks it the same question."""
    rc = V._RunningContrast(2, 65535.0)
    rc.add(0, np.full((8, 8), 1000, np.uint16))
    rc.add(1, np.full((8, 8), 2000, np.uint16))
    auto0 = rc._auto_window(0)
    assert rc.resolve(0, auto0) == auto0            # untouched -> the caller's auto window stands
    rc.set_manual(0, 111.0, 222.0)
    # A latched channel keeps the user's window WHATEVER auto the caller derived -- this is the
    # single rule the plate, the per-region cells and the loupe all consult.
    assert rc.resolve(0, (9.0, 9999.0)) == (111.0, 222.0)
    assert rc.window(0) == (111.0, 222.0)
    assert rc.resolve(1, (9.0, 9999.0)) == (9.0, 9999.0)     # ch1 is not latched
    rc.set_auto(0)
    assert rc.resolve(0, (9.0, 9999.0)) == (9.0, 9999.0)     # unlatched -> auto again


def test_ima242_no_second_contrast_implementation_survives():
    """Guard the collapse: the twins must not grow back."""
    assert not hasattr(V, "_composite_rgb"), "the loupe's private compositor came back"
    assert not hasattr(V, "_percentile_window"), "the second percentile rule came back"


def test_fov_seam_is_single_fov():
    # The plate resolves a WELL, never a FOV, so this is 0 today. When viewer-side multi-FOV
    # lands this test fails LOUDLY — which is the entire point of routing FOV lookups through
    # one helper instead of scattering bare 0 literals.
    assert V._fov_of_well("B2") == 0
    assert V._fov_of_well("B2", {"B2": [0]}) == 0
    assert V._fov_of_well("B2", {"B2": [3, 4]}) == 3


# -- gesture state machine --

def test_hold_raises_loupe_and_release_dismisses(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(), np.ones((2, 3), np.float32))
    x, y = _cell_center(ov, *ov._by_rc and list(ov._by_rc)[0])
    ov.mousePressEvent(_press(x, y))
    assert ov._loupe is None                       # not yet — the dwell hasn't elapsed
    ov._arm_loupe()                                # (what the hold timer does)
    assert ov._loupe is not None
    assert _drain_until(qapp, lambda: ov._loupe_img is not None)
    ov.mouseReleaseEvent(_press(x, y))
    assert ov._loupe is None and ov._loupe_img is None
    ov.set_loupe_source(None)
    win.close()


def test_loupe_follows_cursor_and_coalesces_to_newest(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    src = _FakeLoupeSource()
    ov.set_loupe_source(src, np.ones((2, 3), np.float32))
    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    ov.mousePressEvent(_press(x, y))
    ov._arm_loupe()
    gen0 = ov._loupe_gen
    ov.mouseMoveEvent(_move(x + 4, y + 4))
    assert ov._loupe["x"] == x + 4                 # the inset tracks the cursor
    assert ov._loupe_gen > gen0                    # ...and asks for the new position
    assert _drain_until(qapp, lambda: ov._loupe_img is not None)
    ov.mouseReleaseEvent(_press(x, y))
    ov.set_loupe_source(None)
    win.close()


def test_moving_before_the_dwell_pans_and_never_loupes(qapp, stub_detail, squid_dataset):
    """REGRESSION: drag-to-pan must survive the loupe."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(), np.ones((2, 3), np.float32))
    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    ox0 = ov._ox
    ov.mousePressEvent(_press(x, y))
    ov.mouseMoveEvent(_move(x + 40, y, buttons=Qt.LeftButton))
    assert ov._panning is True
    assert ov._ox == pytest.approx(ox0 + 40)       # the plate actually panned
    assert not ov._hold.isActive()                 # ...and the hold timer was killed
    ov._arm_loupe()                                # even if the timer had fired late:
    assert ov._loupe is None                       # a pan never becomes a loupe
    ov.mouseReleaseEvent(_press(x + 40, y))
    ov.set_loupe_source(None)
    win.close()


def test_slow_pan_stays_a_pan(qapp, stub_detail, squid_dataset):
    """REGRESSION: press, dwell PAST the timer, then drag — the classic deliberate pan.

    The obvious 'pan still works' test (press + immediate drag) passes even if this breaks,
    which is why it gets its own test."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(), np.ones((2, 3), np.float32))
    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    ov.mousePressEvent(_press(x, y))
    ov._arm_loupe()                                # the dwell elapses: loupe is up
    assert ov._loupe is not None
    ov.mouseReleaseEvent(_press(x, y))             # let go...
    assert ov._loupe is None
    ox0 = ov._ox                                   # ...and the next drag pans normally
    ov.mousePressEvent(_press(x, y))
    ov.mouseMoveEvent(_move(x + 25, y, buttons=Qt.LeftButton))
    assert ov._panning is True and ov._ox == pytest.approx(ox0 + 25)
    ov.mouseReleaseEvent(_press(x + 25, y))
    ov.set_loupe_source(None)
    win.close()


def test_double_click_cancels_the_hold_and_still_opens_the_well(qapp, stub_detail, squid_dataset):
    """REGRESSION: Qt sends press/release/dblclick, so the second press re-arms the timer.
    Without the cancel, one double-click both opens the detail viewer AND raises a loupe."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(), np.ones((2, 3), np.float32))
    opened = []
    ov.wellActivated.connect(lambda w, f: opened.append((w, f)))
    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    ov.mousePressEvent(_press(x, y))               # first click of the pair
    ov.mouseDoubleClickEvent(_press(x, y))
    assert not ov._hold.isActive()                 # timer cancelled
    assert ov._loupe is None                       # no loupe from a double-click
    assert opened and opened[0][1] == 0            # ...and the well still opens, fov 0
    ov.set_loupe_source(None)
    win.close()


def test_press_off_plate_never_arms(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(), np.ones((2, 3), np.float32))
    ov.mousePressEvent(_press(2, 2))               # in the label margin, off the grid
    assert not ov._hold.isActive()
    ov._arm_loupe()
    assert ov._loupe is None
    ov.set_loupe_source(None)
    win.close()


def test_leaving_the_widget_dismisses_a_live_loupe(qapp, stub_detail, squid_dataset):
    """Release may never arrive if the cursor leaves the widget mid-hold."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(), np.ones((2, 3), np.float32))
    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    ov.mousePressEvent(_press(x, y))
    ov._arm_loupe()
    assert ov._loupe is not None
    ov.leaveEvent(None)
    assert ov._loupe is None
    ov.set_loupe_source(None)
    win.close()


def test_unavailable_well_reports_instead_of_showing_other_pixels(qapp, stub_detail, squid_dataset):
    """A well the run hasn't written must say so — never magnify some other well's pixels."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    rc = sorted(ov._by_rc)[0]
    missing = ov._by_rc[rc]
    src = _FakeLoupeSource(missing=[missing])
    ov.set_loupe_source(src, np.ones((2, 3), np.float32))
    x, y = _cell_center(ov, *rc)
    ov.mousePressEvent(_press(x, y))
    ov._arm_loupe()
    qapp.processEvents()
    assert ov._loupe_img is None
    assert ov._loupe_note == "not written yet"
    assert src.reads == []                         # and we never even issued the read
    ov.set_loupe_source(None)
    win.close()


def test_no_source_means_the_gesture_never_arms(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(None)
    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    ov.mousePressEvent(_press(x, y))
    assert not ov._hold.isActive()
    win.close()


# -- source wiring --

def test_raw_layer_gets_a_loupe_source_on_ingest(qapp, stub_detail, squid_dataset):
    """The loupe works before ANY operator run — raw mode is where users actually are."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    assert isinstance(win._loupe_sources.get("raw"), V._RawLoupeSource)
    assert win._overview._loupe_src is win._loupe_sources["raw"]
    ok, _why = win._overview._loupe_src.available("B2")
    assert ok
    win.close()


def test_raw_source_reads_real_acquisition_pixels(qapp, stub_detail, squid_dataset):
    root, arrays = squid_dataset
    win = _loupe_win(qapp, root)
    src = win._loupe_sources["raw"]
    crop = src.read_crop("B2", 0, 0, 0, 4, 4)
    assert crop.shape[1:] == (4, 4)
    # Channel order is the metadata's, not the fixture's, so resolve the index rather than
    # assuming it; z=1 is the mid plane both the preview and the loupe read.
    names = [c["name"] for c in win._meta["channels"]]
    for ch in names:
        assert np.array_equal(crop[names.index(ch)], arrays[("B2", 0, 1, ch)])   # unmodified pixels
    win.close()


def test_raw_source_clamps_a_negative_crop_origin(qapp, stub_detail, squid_dataset):
    """THE bug (IMA-208): raw is the DEFAULT source on every folder open, and it did not clamp.

    A crop centred anywhere in the upper-left of a well starts at a negative origin, and
    ``plane[-3:1]`` is not an error in numpy — it is an EMPTY array. The inset drew "no pixels
    here" over roughly three quadrants of every well while the fourth worked, which is exactly
    what "broken over most of every well" looked like. The zarr source clamped; this one didn't."""
    root, arrays = squid_dataset
    win = _loupe_win(qapp, root)
    src = win._loupe_sources["raw"]
    names = [c["name"] for c in win._meta["channels"]]
    full = np.stack([arrays[("B2", 0, 1, ch)] for ch in names])    # 4x4 frames, mid z

    for y0, x0 in ((-3, -3), (-3, 1), (1, -3), (-100, -100)):
        crop = src.read_crop("B2", 0, y0, x0, 4, 4)
        assert crop.size > 0, f"empty crop at origin {(y0, x0)}"
        assert np.array_equal(crop, full)             # shifted in whole, not truncated to a sliver
    win.close()


def test_loupe_shows_pixels_in_every_quadrant_of_a_well(qapp, stub_detail, squid_dataset):
    """The user-visible contract, driven through the widget: hold anywhere in a well and pixels
    appear. Quadrant-by-quadrant because the failure was positional, not total."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(well_px=2084, n_levels=1), np.ones((2, 3), np.float32))
    rc = sorted(ov._by_rc)[0]
    ov._fit()
    ax, ay = ov._ox + V._HDR, ov._oy + V._COLH
    for fx, fy in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
        x = int(ax + (rc[1] + fx) * ov._cd)
        y = int(ay + (rc[0] + fy) * ov._cd)
        ov.mousePressEvent(_press(x, y))
        ov._arm_loupe()
        assert _drain_until(qapp, lambda: ov._loupe_img is not None), f"no pixels at {(fx, fy)}"
        assert ov._loupe_note == ""
        ov.mouseReleaseEvent(_press(x, y))
    ov.set_loupe_source(None)
    win.close()


def test_loupe_read_stays_bounded_when_the_source_has_no_pyramid(qapp, stub_detail, squid_dataset):
    """Raw has n_levels == 1, so level selection cannot shrink the read: at plate fit the rect
    IS the whole field. What crosses to the GUI thread must still be inset-sized."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    src = _FakeLoupeSource(well_px=4168, n_levels=1)
    ov.set_loupe_source(src, np.ones((2, 3), np.float32))
    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    _w, level, (y0, x0, h, w), _s, _m = ov._loupe_geometry(x, y)
    assert level == 0 and max(h, w) > V._LOUPE_MAX_CROP          # the rect really is huge
    crop = src.read_crop("B2", level, y0, x0, h, w)
    assert max(crop.shape[-2:]) <= V._LOUPE_MAX_CROP             # ...the ARRAY never is
    ov.set_loupe_source(None)
    win.close()


def test_raw_plane_cache_is_safe_across_threads(qapp, stub_detail, squid_dataset):
    """Two wells, many threads: a crop must never carry another well's pixels.

    ``_planes`` memoises ONE well and was mutated from both the loupe worker (read_crop) and the
    GUI thread (coarse, via the old window derivation). An interleave between the key test and
    the store returns the wrong well's data under the right well's label."""
    import concurrent.futures as cf

    root, arrays = squid_dataset
    win = _loupe_win(qapp, root)
    src = win._loupe_sources["raw"]
    names = [c["name"] for c in win._meta["channels"]]
    expect = {w: np.stack([arrays[(w, 0, 1, ch)] for ch in names]) for w in ("B2", "B3")}

    def one(i):
        well = "B2" if i % 2 == 0 else "B3"
        if i % 5 == 0:
            src.window(well)                       # the other caller, on the same cache
        return well, np.array(src.read_crop(well, 0, 0, 0, 4, 4))

    with cf.ThreadPoolExecutor(8) as ex:
        for well, got in ex.map(one, range(400)):
            assert np.array_equal(got, expect[well]), f"{well} came back as another well"
    win.close()


def test_opening_another_plate_joins_the_previous_loupe_thread(qapp, stub_detail, squid_dataset):
    """A _LoupeWorker QThread hangs off the OVERVIEW, so replacing the overview without stopping
    it leaked one thread (plus its plane cache) per plate open — _open_computed cleared
    ``_loupe_sources`` directly and never went near the thread."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    first = win._overview._loupe_worker
    assert first is not None and first.isRunning()
    win.ingest(str(root))                          # open a plate again: the overview is rebuilt
    _drain_until(qapp, lambda: win._overview is not None)
    assert not first.isRunning()                   # the old plate's reader thread is joined
    second = win._overview._loupe_worker
    assert second is not first
    win.close()
    assert not second.isRunning()                  # ...and closing joins the current one too


def test_dragging_off_the_widget_dismisses_a_live_loupe(qapp, stub_detail, squid_dataset):
    """Qt GRABS the mouse for the duration of a press, so no leaveEvent is delivered while the
    button is down — dragging off-widget mid-hold left the inset pinned on stale pixels. The
    move events keep coming (that is what the grab means), with coordinates outside rect()."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(), np.ones((2, 3), np.float32))
    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    ov.mousePressEvent(_press(x, y))
    ov._arm_loupe()
    assert ov._loupe is not None
    ov.mouseMoveEvent(_move(x, ov.height() + 40))   # dragged below the plate pane, button still down
    assert ov._loupe is None and ov._loupe_img is None
    ov.set_loupe_source(None)
    win.close()


def test_preview_run_gets_no_loupe_source(qapp, stub_detail, squid_dataset, tmp_path):
    """An unsaved preview writes nothing, so its layer must NOT inherit a zarr source — this is
    the stale-run trap: OperationStack dedupes by key, so the layer name alone proves nothing."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    win.run_operator("mip", out_parent=str(tmp_path), save=False, regions=win._order[:1])
    _drain_until(qapp, lambda: win._overview._active == "mip")
    assert win._loupe_sources.get("mip") is None
    assert win._overview._loupe_src is None        # the gesture is off, not showing raw
    win._stop_worker(); win.close()


def test_saved_run_registers_zarr_source_and_grows_written_set(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: isinstance(win._loupe_sources.get("mip"), V._ZarrLoupeSource))
    src = win._loupe_sources["mip"]
    assert src.available("B2") == (False, "not written yet")   # nothing written at run start
    assert _drain_until(qapp, lambda: src.available("B2")[0])  # ...available once the well lands
    win._stop_worker(); win.close()


def test_switching_back_to_raw_switches_the_source(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: win._overview._active == "mip")
    win._return_to_raw()
    assert win._overview._active == "raw"
    assert win._overview._loupe_src is win._loupe_sources["raw"]
    win._stop_worker(); win.close()


# -- the real read path, against a real pyramid --

def test_zarr_source_crop_read_against_a_real_pyramid(qapp, pyramid_dataset, tmp_path):
    """The load-bearing risk is path + level construction, so this hits real files.

    Uses `pyramid_dataset` because the 4x4 fixture writes a single level (_PYRAMID_MIN_YX),
    which would make level selection untestable."""
    import squidmip
    from squidmip.reader import open_reader

    root, region, size = pyramid_dataset
    out = tmp_path / "out.hcs"
    squidmip.write_plate(open_reader(str(root)), str(out), tiff=False)
    base = out / "plate.ome.zarr"
    assert base.is_dir()

    src = V._ZarrLoupeSource(
        str(base),
        path_of=lambda w: "/".join(str(x) for x in V.parse_well_id(w)),
        fov_of=lambda w: V._fov_of_well(w),
        levels=None,                                # discovered from the field, as in a live run
        well_px=size, pixel_size_um=0.325, written=None)

    assert src.available(region) == (True, "")
    levels = src._resolve_levels(region)
    assert len(levels) >= 2                         # the fixture really does build a pyramid
    assert src.n_levels == len(levels)

    crop = src.read_crop(region, 0, 100, 100, 32, 32)
    assert crop.shape[1:] == (32, 32)               # a WINDOW, not the whole 640px plane
    around = src.read_crop(region, 0, 100, 100, V._LOUPE_MAX_CROP, V._LOUPE_MAX_CROP)
    assert np.array_equal(crop[0], around[0][:32, :32])         # the crop is where we asked

    # A rect bigger than the ceiling comes back DECIMATED, not truncated: same region, fewer
    # samples. (A field with too few levels is the case that used to pull a whole plane.)
    full = src.read_crop(region, 0, 0, 0, size, size)
    assert max(full.shape[-2:]) <= V._LOUPE_MAX_CROP
    step = V.loupe_decimation(size)
    assert full.shape[-1] == size // step
    assert np.array_equal(full[0][:16, :16], src.read_crop(region, 0, 0, 0, size, size)[0][:16, :16])

    coarse = src.coarse(region)                     # coarsest level, for the contrast window
    assert coarse.shape[-1] < size

    deep = src.read_crop(region, len(levels) - 1, 0, 0, 8, 8)   # a coarser level is readable
    assert deep.shape[1:] == (8, 8)

    over = src.read_crop(region, 99, 0, 0, 8, 8)    # out-of-range level clamps, never raises
    assert over.shape[1:] == (8, 8)

    edge = src.read_crop(region, 0, size - 4, size - 4, 32, 32)  # clipped at the field edge
    assert edge.shape[1] <= 32 and edge.size > 0


def test_computed_plate_open_wires_a_loupe_source(qapp, stub_detail, pyramid_dataset, tmp_path, monkeypatch):
    """Opening a written plate: every well is on disk, so the loupe covers the whole plate."""
    import squidmip
    from squidmip.reader import open_reader

    root, region, size = pyramid_dataset
    out = tmp_path / "out.hcs"
    squidmip.write_plate(open_reader(str(root)), str(out), tiff=False)

    win = V.PlateWindow(None)
    monkeypatch.setattr(V.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(out)))
    win._open_computed()
    assert _drain_until(qapp, lambda: win._overview is not None)
    src = win._loupe_sources.get("computed")
    assert isinstance(src, V._ZarrLoupeSource)
    assert src.available(region) == (True, "")      # written plate: no per-well holes
    assert src.well_px == size                      # level-0 field size, not the push size
    # pixel size is recovered from the multiscales scale, so the µm bar has a real source
    assert win._meta["pixel_size_um"] == pytest.approx(0.325)
    assert V.loupe_um_per_screen_px(src.pixel_size_um, 1.0) == pytest.approx(0.325)
    win._stop_worker(); win.close()


def test_ambiguous_unit_pixel_size_is_treated_as_unknown():
    """_output writes 1.0 for BOTH 'unknown' and a genuine 1.0 µm/px, so a plate reporting
    exactly 1.0 must suppress the scale bar rather than assert a figure it can't back."""
    assert V.loupe_um_per_screen_px(None, 0.5) is None


def test_loupe_geometry_maps_cursor_to_the_right_well_and_crop(qapp, stub_detail, squid_dataset):
    """The cursor -> image-coordinate mapping: right well, crop centred where the user pointed,
    and a level chosen for the CURRENT zoom (coarse when zoomed out, level 0 when near native)."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(well_px=1024, n_levels=4), np.ones((2, 3), np.float32))

    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    well, level, (y0, x0, h, w), s_loupe, mag = ov._loupe_geometry(x, y)
    assert well == ov._by_rc[rc]                     # the well actually under the cursor
    span = 1024 >> level
    # The crop is centred on where the user pointed, to within the resolution the plate can
    # even express: on a 1536wp at fit, one screen pixel IS span/cd image pixels, so that is
    # the honest tolerance — a tighter bound would be testing int() rounding, not the mapping.
    slop = span / ov._cd + 2
    assert y0 + h // 2 == pytest.approx(span // 2, abs=slop)
    assert x0 + w // 2 == pytest.approx(span // 2, abs=slop)

    # Zoomed out, the plate scale is tiny, so the loupe reads a COARSE level: that is what keeps
    # a whole-plate hold cheap instead of pulling a full-res plane. (_user_view stops paint/_fit
    # from resetting the zoom under us.)
    ov._user_view = True
    ov._cd = 20.0
    ax, ay = ov._ox + V._HDR, ov._oy + V._COLH
    pt_out = (int(ax + (rc[1] + 0.5) * ov._cd), int(ay + (rc[0] + 0.5) * ov._cd))
    _w, lvl_out, _r, _s, mag_out = ov._loupe_geometry(*pt_out)
    # Zoomed in near native, it reads level 0 and stops claiming magnification.
    ov._cd = 4096.0
    ax, ay = ov._ox + V._HDR, ov._oy + V._COLH
    pt_in = (int(ax + (rc[1] + 0.5) * ov._cd), int(ay + (rc[0] + 0.5) * ov._cd))
    _w, lvl_in, _r, _s, mag_in = ov._loupe_geometry(*pt_in)
    assert lvl_out > lvl_in and lvl_in == 0
    assert mag_out > mag_in and mag_in == pytest.approx(1.0)

    off = ov._loupe_geometry(1, 1)                   # in the label margin: no geometry at all
    assert off is None
    ov.set_loupe_source(None)
    win.close()
# --- IMA-228: Minerva export -------------------------------------------------------------------

def test_minerva_is_a_registered_operation():
    """One registry entry buys the console card, the menu item and the tab — no scattered edits."""
    op = V._OPERATIONS_BY_KEY["minerva"]
    assert op.build_tab == "_build_minerva_tab"
    assert hasattr(V.PlateWindow, op.build_tab)


def test_minerva_tab_builds_and_lists_projectors(qapp, stub_detail, squid_dataset):
    """The projector choice must be real: squid2minerva's convert.py offers --mip/--z, so a
    hardcoded projection here would be a capability regression."""
    from PyQt5.QtWidgets import QComboBox
    from squidmip import available_projectors

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._open_op_tab("minerva", "Minerva", win._build_minerva_tab)
    tab = win._op_tabs["minerva"]

    combos = tab.findChildren(QComboBox)
    assert combos, "no projector selector in the Minerva tab"
    listed = [combos[0].itemText(i) for i in range(combos[0].count())]
    assert listed == available_projectors()
    assert combos[0].currentText() == "mip"
    win.close()


def test_run_minerva_export_writes_one_fused_mosaic_for_the_selected_region(
        qapp, stub_detail, squid_dataset, tmp_path):
    """Selecting a region must export ONE fused mosaic of it, not one file per FOV.

    Two bugs are pinned here, in the order they were found. The first was the GUI building its
    own 1-element selection pinned to fov 0, so a user who picked a well got 1 of its N FOVs.
    The fix for that produced N files — which is also wrong, and worse because it looks right:
    Minerva Author hardcodes ``"Layout": {"Grid": [["i0"]]}`` and opens only ``series[0]``, so
    it would have rendered one of the N and silently dropped the rest. The fixture has 2 FOVs
    per region; the export is 1 mosaic.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.activate_well("B2", 0)                                       # the user's selection
    win.run_minerva_export(out_dir=str(tmp_path), launch=False)      # launch=False: no server, no browser
    assert _drain_until(qapp, lambda: "✓ exported" in win._readout.text())
    names = sorted(p.name for p in tmp_path.glob("*.ome.tiff"))
    assert len(names) == 1, f"one mosaic per region, got {names}"
    assert "B2" in names[0]
    assert "fov" not in names[0], "a per-FOV filename means the per-FOV model came back"
    assert len(list(tmp_path.glob("*.story.json"))) == 1
    assert "1 mosaic" in win._readout.text()                         # honest unit + count
    assert "B2" in win._readout.text()
    win._stop_minerva(); win.close()


def test_run_minerva_export_with_nothing_selected_says_so(qapp, stub_detail, squid_dataset, tmp_path):
    """No selection must be a message, not a silent export of fov 0 of the first well."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._current_well = None            # nothing selected and nothing open in the detail viewer
    assert win.minerva_selection() == []
    win.run_minerva_export(out_dir=str(tmp_path), launch=False)
    assert "nothing selected" in win._readout.text()
    qapp.processEvents()
    assert not list(tmp_path.glob("*.ome.tiff"))
    assert win._minerva is None
    win.close()


def test_minerva_selection_reads_the_window_not_the_overview(qapp, stub_detail, squid_dataset):
    """The selection has ONE owner: ``PlateWindow``.

    ``PlateOverview`` is display-only — it maps grid cells to well ids and emits them; the
    expansion to (region, fov) needs ``fovs_per_region``, which lives on the window. The
    previous version of ``minerva_selection`` duck-typed a chain of three probes, two of them
    on the overview, and reached the right answer only through the last one
    (``selected_wells``) — the overview never had a ``selected_region_fovs`` at all. That
    accident is what this test forbids: attributes bolted onto the overview must be IGNORED,
    and the window's own selection must be what the export sees.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.activate_well("B2", 0)                       # detail well: the last-resort source

    # Decoys on the display-only widget. Reading either would be reading the wrong owner.
    win._overview.selected_wells = lambda: ["B3"]
    win._overview.selected_region_fovs = lambda: {"B3": [1]}
    assert win.minerva_selection() == [("B2", 0), ("B2", 1)], "read the overview, not the window"

    # The real owner. Setting it is what must move the export scope.
    win._selected_regions = ["B3"]
    assert win.minerva_selection() == [("B3", 0), ("B3", 1)]
    win._selected_regions = ["B3", "B2"]
    assert win.minerva_selection() == [("B3", 0), ("B3", 1), ("B2", 0), ("B2", 1)]

    # A selection naming things the acquisition does not have is dropped, never exported.
    win._selected_regions = ["ZZ"]
    assert win.minerva_selection() == [("B2", 0), ("B2", 1)]   # falls back to the detail well
    win.close()


def test_real_shift_drag_selection_is_what_minerva_exports(qapp, stub_detail, squid_dataset):
    """IMA-221 <-> IMA-228, end to end through the ACTUAL gesture — no stubbed selection API.

    Both halves shipped on separate branches and nothing joined them: IMA-221's per-FOV payload
    landed as ``PlateWindow.selected_region_fovs`` (the overview is display-only), so a
    ``minerva_selection`` that probed only the overview would silently skip the real API and
    reach the same answer by accident via ``selected_wells``. This drives a genuine Shift-drag
    marquee and pins that the export scope IS the dragged wells — not the detail well.
    """
    from PyQt5.QtCore import QEvent, QPoint
    from PyQt5.QtGui import QMouseEvent

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview
    ov.resize(600, 480)
    ov.show()
    qapp.processEvents()

    target = ov._by_rc[sorted(ov._by_rc)[0]]                  # first acquired well only: a subset
    assert len(ov._by_rc) > 1, "fixture must have >1 well or 'subset' means nothing"
    (r, c), = [rc for rc, w in ov._by_rc.items() if w == target]
    ax, ay = ov._ox + V._HDR, ov._oy + V._COLH
    cx, cy = ax + (c + 0.5) * ov._cd, ay + (r + 0.5) * ov._cd
    box = ov._cd * 0.3

    def send(kind, x, y, buttons):
        qapp.sendEvent(ov, QMouseEvent(kind, QPoint(int(x), int(y)), Qt.LeftButton,
                                       buttons, Qt.ShiftModifier))

    send(QEvent.MouseButtonPress, cx - box, cy - box, Qt.LeftButton)
    send(QEvent.MouseMove, cx, cy, Qt.LeftButton)
    send(QEvent.MouseButtonRelease, cx + box, cy + box, Qt.NoButton)
    qapp.processEvents()

    assert ov.selected_wells() == [target], "the Shift-drag itself did not select the well"
    expected = [(target, f) for f in win._meta["fovs_per_region"][target]]
    assert win.selected_region_fovs() == expected             # IMA-221's payload
    assert win.minerva_selection() == expected                # ...is what IMA-228 exports

    # The same Shift-drag also opens IMA-205's exploration tab in pane 3, which SCOPES the detail
    # slider to the subset — so empty pane 3 before opening a well outside it, or activate_well
    # correctly refuses and the precedence assertion below would pass vacuously.
    _close_exploration_pane(win)
    qapp.processEvents()
    other = ov._by_rc[sorted(ov._by_rc)[-1]]
    win.activate_well(other, 0)                               # a DIFFERENT well open in detail
    assert win._current_well == other, "the detail well never actually changed"
    assert win.minerva_selection() == expected, (
        "minerva_selection fell through to the detail well and ignored the plate selection")

    ov.clear_selection()
    qapp.processEvents()
    assert win.minerva_selection() == [(other, f) for f in win._meta["fovs_per_region"][other]]
    win.close()


def test_ingest_stops_a_running_minerva_export(qapp, stub_detail, squid_dataset, tmp_path):
    """Re-ingesting mid-export used to leave the worker running against the OLD reader."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.activate_well("B2", 0)
    win.run_minerva_export(out_dir=str(tmp_path), launch=False)
    worker = win._minerva
    assert worker is not None

    win.ingest(str(root))                       # open an acquisition again, mid-export
    assert win._minerva is None                 # ...the export is retired, not orphaned
    assert worker.wait(10000)
    win._stop_worker(); win._stop_preview(); win.close()


def test_run_minerva_export_refuses_a_second_concurrent_run(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    class _Busy:
        def isRunning(self):
            return True

    win._minerva = _Busy()
    win.run_minerva_export(out_dir=str(tmp_path), launch=False)
    assert "already exporting" in win._readout.text()
    assert not list(tmp_path.glob("*.ome.tiff"))
    win._minerva = None
    win.close()


def test_run_minerva_export_without_an_acquisition_is_a_message_not_a_crash(qapp, stub_detail):
    win = V.PlateWindow(None)
    win.run_minerva_export(launch=False)
    assert "open an acquisition" in win._readout.text()
    win.close()


def test_minerva_export_failure_surfaces_in_the_readout(qapp, stub_detail, squid_dataset, monkeypatch, tmp_path):
    """A worker never raises across the thread boundary; the user must still see why."""
    from squidmip import _minerva

    def boom(*a, **k):
        raise ValueError("no objective pixel size")

    monkeypatch.setattr(_minerva, "export_selection", boom)
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.activate_well("B2", 0)
    win.run_minerva_export(out_dir=str(tmp_path), launch=False)
    assert _drain_until(qapp, lambda: "failed" in win._readout.text())
    assert "no objective pixel size" in win._readout.text()
    win._stop_minerva(); win.close()


def test_minerva_reports_when_author_is_not_installed(qapp, stub_detail, squid_dataset, monkeypatch, tmp_path):
    """The export still succeeded — a missing sibling checkout must not read as a failure."""
    from squidmip import _minerva

    monkeypatch.setattr(_minerva, "is_running", lambda timeout=1.0: False)
    monkeypatch.setattr(_minerva, "minerva_home", lambda: None)
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.activate_well("B2", 0)
    win.run_minerva_export(out_dir=str(tmp_path), launch=True)
    assert _drain_until(qapp, lambda: "not found" in win._readout.text())
    assert "✓ exported" in win._readout.text()          # the files are still good
    assert list(tmp_path.glob("*.ome.tiff"))
    win._stop_minerva(); win.close()


def test_signal_names_discovers_every_worker_signal():
    """The regression guard: _retire used to disconnect a HARDCODED name list, so any worker
    declaring a signal outside it stayed connected through teardown and could paint onto the
    next plate. Introspection makes a new worker correct by construction."""
    names = set(V._signal_names(V._MinervaWorker))
    assert {"progress", "exported", "launched", "failed", "finished_ok"} <= names
    assert "finished" not in names and "started" not in names   # QThread's own — never torn down
    # the pre-existing worker keeps full coverage too
    assert {"tileReady", "pushReady", "streamEnded", "writtenReady", "wellFailed"} <= set(
        V._signal_names(V._OperatorWorker))


def test_retire_disconnects_every_declared_signal(qapp, stub_detail, squid_dataset, tmp_path):
    """_signal_names being right is worthless unless _retire USES it: this test failed to notice
    the loop being emptied, so it now drives _retire itself and emits every signal afterwards.
    Nothing may reach a handler connected before the retire."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    worker = V._MinervaWorker(win._reader, [("B2", 0)], str(tmp_path), "mip", t=0, launch=False)

    payload = {"progress": (1, 1), "exported": ([],), "launched": (False,), "failed": ("x",),
               "finished_ok": ()}
    seen = []
    names = [n for n in V._signal_names(V._MinervaWorker) if n in payload]
    assert set(names) == set(payload), "a declared worker signal is not covered here"
    for name in names:
        getattr(worker, name).connect(lambda *a, _n=name: seen.append(_n))

    win._retire(worker)                       # not running -> retire is pure disconnection

    for name in names:
        getattr(worker, name).emit(*payload[name])
    qapp.processEvents()
    assert seen == [], f"signals still connected after _retire: {sorted(set(seen))}"
    win._stop_worker(); win._stop_preview(); win.close()


def test_closing_mid_export_disconnects_the_worker(qapp, stub_detail, squid_dataset, tmp_path):
    """Close the window mid-export: no signal may reach the (now dead) window afterward."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.activate_well("B2", 0)
    win.run_minerva_export(out_dir=str(tmp_path), launch=False)
    worker = win._minerva
    win.close()

    seen = []
    worker.exported.connect(lambda p: seen.append(p))   # reconnect: proves the old ones are gone
    worker.wait(5000)
    qapp.processEvents()
    assert win._minerva is None


# --- IMA-237: three horizontal panes, the third revealed by the exploration gesture ------------
#
# Julio's requirement is a THREE-pane app on one monitor: pane 1 = plate + the tabbed controls,
# pane 2 = the initial viewer, pane 3 = exploration. The constraint that shapes the code is that
# pane 3 must not exist for a user who never Shift-drags, so these drive the REAL splitter and the
# REAL gesture rather than asserting on a builder's return value.

def test_outer_split_has_three_panes_and_pane3_opens_with_width(qapp, stub_detail):
    """IMA-260 reversed IMA-237's collapsed-until-revealed rule: the third pane is there from the
    first frame, because a pane that is not there cannot be found. It opens on its EXAMPLE page —
    holding no tabs is a state with copy in it, not a state with nothing in it."""
    win = V.PlateWindow(None)
    outer = win._split
    assert outer.count() == 3, "the window is not a three-pane layout"
    assert outer.widget(2) is win._explore_pane, "pane 3 is not the exploration pane"
    assert outer.sizes()[2] > 0, "pane 3 opened collapsed"
    assert win._explore_pane.currentWidget() is win._explore_empty
    assert win._explore_tabs.count() == 0
    win.close()


def test_window_resize_never_grows_pane3_at_the_plate_pane_s_expense(qapp, stub_detail, squid_dataset):
    """Requirement 5, measured rather than asserted on a stretch factor Qt won't read back:
    widen the window with pane 3 open and the extra pixels go to panes 1 and 2."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.resize(1400, 900)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    _shift_drag_over(win, ["B3"])
    qapp.processEvents()
    plate0, viewer0, explore0 = win._split.sizes()
    assert explore0 > 0

    win.resize(1900, 900)
    qapp.processEvents()
    plate1, viewer1, explore1 = win._split.sizes()
    assert plate1 > plate0, "the plate pane did not get its share of the new width"
    assert explore1 == explore0, "the exploration pane grew on resize instead of the plate"
    win.close()


def test_a_real_shift_drag_fills_pane3_without_moving_a_single_divider(
        qapp, stub_detail, squid_dataset):
    """Pane 3 already exists (IMA-260), so the gesture changes its CONTENT and nothing else.

    The requirement that outlived IMA-237's reveal is that pane 3 must never squash the plate:
    under a fixed-width third column that is now stronger than it was — no divider moves at all,
    so neither neighbour can lose a pixel to a tab opening.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.resize(1600, 900)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    qapp.processEvents()

    assert win._explore_tabs.count() == 0               # BEFORE: pane 3 is present but empty...
    assert win._explore_pane.currentWidget() is win._explore_empty     # ...showing the example
    before = win._split.sizes()

    _shift_drag_over(win, ["B3"])
    qapp.processEvents()

    assert win._explore_tabs.count() == 1               # POPULATED, and the page swapped to it
    assert win._explore_pane.currentWidget() is win._explore_tabs
    assert isinstance(win._explore_tabs.currentWidget(), V._ExplorationTab)
    assert win._split.sizes() == before, "opening a tab moved a divider"
    win.close()


def test_exploration_tabs_are_not_in_the_process_console(qapp, stub_detail, squid_dataset):
    """Requirement 3: exploration tabs moved OUT of pane 1's tab bar."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B2", "B3"])
    tab = win._op_tabs[key]
    assert win._explore_tabs.indexOf(tab) >= 0
    assert win._left_tabs.indexOf(tab) == -1
    assert not [i for i in range(win._left_tabs.count())
                if isinstance(win._left_tabs.widget(i), V._ExplorationTab)]
    win.close()


def test_a_tab_drags_out_of_pane3_through_the_same_detach_path_as_ima209(
        qapp, stub_detail, squid_dataset):
    """Requirement 4: ONE _detach_tab serves both bars — pane 3 floats out, and re-docks BACK to
    pane 3 rather than landing in the process console."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B3"])
    tab = win._op_tabs[key]

    fl = win._detach_tab(win._explore_tabs.indexOf(tab), win._explore_tabs)
    assert isinstance(fl, V._FloatWindow)               # the IMA-209 float, not a second class
    assert fl.content() is tab                          # the SAME widget, not a rebuild
    assert win._explore_tabs.indexOf(tab) == -1
    assert win._explore_tabs.isHidden()                 # its last tab left -> pane 3 collapses
    assert win._floating[key] is fl

    dock = next(b for b in fl.findChildren(QPushButton) if b.text() == "Re-dock")
    dock.click()                                        # the float's own Re-dock button
    qapp.processEvents()
    assert win._explore_tabs.indexOf(tab) >= 0, "re-dock did not return it to pane 3"
    assert win._left_tabs.indexOf(tab) == -1, "re-dock dumped it in the process console"
    assert not win._explore_tabs.isHidden()
    assert key not in win._floating
    win.close()


def test_pane3_index0_is_detachable_but_the_process_home_tab_is_not(qapp, stub_detail, squid_dataset):
    """The home-tab guard belongs to the process console, not to _detach_tab in general: pane 3's
    index 0 is an ordinary user-opened tab and must drag out like any other."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._detach_tab(0) is None                   # 'Process wells' — never detaches
    assert win._left_tabs.count() >= 1
    win.open_exploration_tab(["B3"])
    assert isinstance(win._detach_tab(0, win._explore_tabs), V._FloatWindow)
    assert win._explore_tabs.tabBar()._first_detachable == 0
    assert win._left_tabs.tabBar()._first_detachable == 1
    win.close()


def test_opening_a_pane1_tab_does_not_unscope_the_viewer(qapp, stub_detail, squid_dataset):
    """Both panes are visible at once now, so pane 1's front tab must not silently claim the
    viewer's scope: opening Layers while an exploration tab is up used to look like a tab switch."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B3"])
    qapp.processEvents()
    assert win._detail._fov_labels == ["B3:0"]
    win._open_op_tab("layers", "Layers", win._build_layers_tab)   # a PANE 1 tab
    qapp.processEvents()
    assert win._detail._fov_labels == ["B3:0"], "a pane 1 tab stole the viewer's scope"
    assert win._active_exploration is win._op_tabs[key]
    win.close()


def test_the_redock_BUTTON_works_not_just_the_method(qapp, stub_detail):
    """REGRESSION (found by IMA-237 driving the real widget): QPushButton.clicked passes
    `checked=False`, which bound to the `k=key` default of the on_redock lambda — so clicking
    Re-dock called _redock(False), missed _floating entirely, and did nothing. Every existing test
    called win._redock(key) directly, so the button was dead from IMA-209 until now."""
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    fl = win._detach_tab(win._left_tabs.indexOf(w))
    dock = next(b for b in fl.findChildren(QPushButton) if b.text() == "Re-dock")
    dock.click()                                             # the GESTURE, not the method
    qapp.processEvents()
    assert win._op_tabs.get("stub") is w, "the Re-dock button did nothing"
    assert win._left_tabs.indexOf(w) >= 0
    assert not win._floating
    win.close()


# --- IMA-223/224/225: the three plane-op cards -------------------------------------------------

def test_the_plane_op_cards_build_and_are_preview_only(qapp, stub_detail, squid_dataset):
    """DRIVEN, not read: open each plane-op tab through the real _open_op_tab path and inspect
    the widgets it actually produced. A plane-op keeps z at full depth and _validate_image accepts
    Z == 1 only, so the card must offer Preview and NO Save/destination half.

    DECON IS NO LONGER IN THIS LIST. It is still a plane-op in the engine, but its card is now
    the RL semi-convergence QC panel (iteration count, +1, turbo x-z / y-z view in pane 3), not
    the generic preview button -- the generic tab gave no way to choose an iteration count at
    all, which is what Julio was blocked on. See the decon-specific test below.
    """
    from squidmip import available_projectors
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    for key in ("bgsub", "flatfield"):
        assert key in available_projectors(), f"{key} is not registered in the engine"
        op = V._OPERATIONS_BY_KEY[key]
        win._open_op_tab(op.key, op.label, getattr(win, op.build_tab))
        qapp.processEvents()
        tab = win._op_tabs[key]
        texts = [b.text() for b in tab.findChildren(QPushButton)]
        assert "Preview" in texts, f"{key} card has no Preview button: {texts}"
        # the run-tab half must be ABSENT: no destination picker, no whole-plate run
        assert not [t for t in texts if "Choose" in t or "whole plate" in t.lower()], texts
        assert not [c for c in tab.findChildren(QCheckBox)], f"{key} exposed a Save checkbox"
        assert tab.findChildren(QSpinBox), f"{key} card has no 'first N wells' spinner"
    win.close()


def test_flatfield_card_gates_preview_on_a_profile(qapp, stub_detail, squid_dataset):
    """Flat-field is the one plane-op with no sane default: an identity field would silently do
    nothing while the UI said 'corrected'. So its Preview stays disabled until a profile loads,
    and decon/bgsub - which need no argument - are enabled from the start."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    prev = {}
    for key in ("bgsub", "flatfield"):
        op = V._OPERATIONS_BY_KEY[key]
        win._open_op_tab(op.key, op.label, getattr(win, op.build_tab))
        tab = win._op_tabs[key]
        prev[key] = next(b for b in tab.findChildren(QPushButton) if b.text() == "Preview")
    assert prev["bgsub"].isEnabled()
    assert not prev["flatfield"].isEnabled(), "flat-field ran without an illumination profile"
    ff = win._op_tabs["flatfield"]
    assert [b for b in ff.findChildren(QPushButton) if "illumination profile" in b.text()], \
        "flat-field card has no profile chooser"
    win.close()


# --- IMA-decon-stitch-ui: the two operator INTERFACES in pane 1 --------------------------

def test_the_decon_card_is_the_iteration_qc_panel_not_a_bare_preview(qapp, stub_detail,
                                                                    squid_dataset):
    """Julio: "The deconvolution is not showing the XZ/YZ strips on the turbo colormap ... so
    that we can choose the iterations." The card must therefore carry an iteration count and a
    way to add one, and must NOT have grown a profile chooser or a second contrast control."""
    from squidmip._decon import QC_START_ITERATIONS
    from squidmip._op_panels import DeconQCPanel

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    op = V._OPERATIONS_BY_KEY["decon"]
    win._open_op_tab(op.key, op.label, getattr(win, op.build_tab))
    qapp.processEvents()
    tab = win._op_tabs["decon"]
    assert isinstance(tab, DeconQCPanel)
    assert tab.iter_spin.value() == QC_START_ITERATIONS
    assert [b for b in tab.findChildren(QPushButton) if b.text() == "+1 iteration"]
    assert not [b for b in tab.findChildren(QPushButton)
                if "illumination profile" in b.text()], "decon grew a profile chooser"
    win.close()


def test_the_stitch_card_is_the_stitcher_control_surface(qapp, stub_detail, squid_dataset):
    """The blocking item: maragall/stitcher's Settings group, in the top-left subpane."""
    from squidmip._op_panels import StitcherPanel

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    op = V._OPERATIONS_BY_KEY["stitch"]
    win._open_op_tab(op.key, op.label, getattr(win, op.build_tab))
    qapp.processEvents()
    tab = win._op_tabs["stitch"]
    assert isinstance(tab, StitcherPanel)
    assert tab.register_cb.isChecked()
    assert tab.reg_channel_combo.count() == len(win._meta["channels"])
    assert not hasattr(tab, "scope_combo")      # Defect 2: the run selector owns scope
    win.close()


def test_the_stitcher_panel_kwargs_reach_the_worker(qapp, stub_detail, squid_dataset):
    """End to end through the REAL run_operator: a setting made in pane 1 has to survive into
    the object that actually runs the fuse. This is the seam a typo would break silently."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    op = V._OPERATIONS_BY_KEY["stitch"]
    win._open_op_tab(op.key, op.label, getattr(win, op.build_tab))
    tab = win._op_tabs["stitch"]
    # The fixture's frames are tiny, so pick a feather that fits inside them -- the panel
    # REFUSES a ramp as wide as the tile, and that refusal is asserted separately below.
    tab.blend_spin.setValue(2)
    tab.rel_spin.setValue(25)
    tab.run_btn.click()
    qapp.processEvents()
    assert win._worker is not None, f"the run did not start: {win._readout.text()}"
    assert win._worker._operator_kwargs["blend_px"] == 2
    assert win._worker._operator_kwargs["rel_thresh"] == 0.25
    win._stop_worker(); win.close()


def test_an_impossible_feather_is_refused_in_the_readout_not_at_the_end_of_a_fuse(
        qapp, stub_detail, squid_dataset):
    """No silent failure and no half-run: a ramp wider than the tile stops the run BEFORE it
    starts, with the reason in the status line."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    op = V._OPERATIONS_BY_KEY["stitch"]
    win._open_op_tab(op.key, op.label, getattr(win, op.build_tab))
    tab = win._op_tabs["stitch"]
    tab.blend_spin.setValue(999)
    tab.run_btn.click()
    qapp.processEvents()
    assert win._worker is None, "the run started with an impossible feather width"
    assert "blend" in win._readout.text().lower()
    win.close()


def test_panel_kwargs_reach_stitch_plate_on_the_PREVIEW_path(qapp, stub_detail, squid_dataset,
                                                             monkeypatch):
    """Not just "the worker stored them" -- they must reach the function that fuses.

    Storing a dict on the worker and then not forwarding it is invisible to any assertion
    made on the worker itself, so this spies on the engine call instead.
    """
    import squidmip
    seen = {}

    def fake_stitch_plate(reader, **kw):
        seen.update(kw)
        return iter(())

    monkeypatch.setattr(squidmip, "stitch_plate", fake_stitch_plate)
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("stitch", regions=["B2"], save=False,
                     operator_kwargs={"blend_px": 3, "register": False})
    _drain_until(qapp, lambda: "blend_px" in seen)
    assert seen["blend_px"] == 3 and seen["register"] is False
    win._stop_worker(); win.close()


def test_panel_kwargs_reach_write_plate_on_the_SAVE_path(qapp, stub_detail, squid_dataset,
                                                        monkeypatch, tmp_path):
    """The save path is the one that matters most: a registration tuned on a preview and then
    silently dropped is thrown away at exactly the moment it is written to disk."""
    import squidmip
    seen = {}

    def fake_write_plate(reader, out_dir, **kw):
        seen.update(kw)
        return {"plate": str(out_dir), "levels": 1}

    monkeypatch.setattr(squidmip, "write_plate", fake_write_plate)
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("stitch", out_parent=str(tmp_path), regions=["B2"], save=True,
                     operator_kwargs={"blend_px": 3, "register": False})
    _drain_until(qapp, lambda: "operator_kwargs" in seen)
    assert seen["operator_kwargs"]["blend_px"] == 3
    assert seen["operator_kwargs"]["register"] is False
    win._stop_worker(); win.close()


def test_a_decon_qc_result_opens_as_a_tab_in_pane_3(qapp, stub_detail, squid_dataset):
    """The seam with pane 3, driven: publish_qc_result must put the widget in the EXPLORE tab
    bar (pane 3), not in the pane-1 console, and re-publishing the same title must reuse it."""
    from squidmip._op_panels import DeconQCResultView

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    view = DeconQCResultView("B2/0/c0")
    win.publish_qc_result(view, "Decon QC · B2/0/c0")
    qapp.processEvents()
    assert win._explore_tabs.indexOf(view) >= 0, "the QC result did not land in pane 3"
    assert win._left_tabs.indexOf(view) < 0, "the QC result landed in pane 1"
    before = win._explore_tabs.count()
    # Publishing the SAME SUBJECT again must reuse its tab. Passing a DIFFERENT widget is the
    # point: keying on anything unique-per-call (a uuid, an iteration number) would stack a new
    # tab for every iteration of the QC loop, which is the loop's whole working rhythm. Passing
    # the same object again could not detect that, because a widget already in a tab bar is
    # merely moved rather than added twice.
    win.publish_qc_result(DeconQCResultView("B2/0/c0"), "Decon QC · B2/0/c0")
    qapp.processEvents()
    assert win._explore_tabs.count() == before, "a second tab was stacked for the same subject"
    assert win._explore_tabs.indexOf(view) >= 0, "the original tab was replaced, not reused"
    # A DIFFERENT subject does get its own tab.
    win.publish_qc_result(DeconQCResultView("B3/0/c0"), "Decon QC · B3/0/c0")
    qapp.processEvents()
    assert win._explore_tabs.count() == before + 1
    win.close()


# --- IMA-226: EVERY operator streams live to the plate and the ndviewer slider -----------------

def _run_live(qapp, win, key, regions=("B3",)):
    """Drive a real preview run to completion and return (tiles, pushes)."""
    win._detail.arrays.clear()
    tiles = []
    win.run_operator(key, regions=list(regions), save=False)
    if win._worker is None:
        return None, None
    win._worker.tileReady.connect(lambda *a: tiles.append(a))
    t0 = time.time()
    while win._worker.isRunning() and time.time() - t0 < 90:
        qapp.processEvents(); time.sleep(0.02)
    for _ in range(25):
        qapp.processEvents(); time.sleep(0.02)
    return tiles, list(win._detail.arrays)


@pytest.mark.parametrize("key", ["mip", "reference", "stitch", "decon", "bgsub", "coordinate"])
def test_every_operator_streams_live_to_plate_and_slider(qapp, stub_detail, squid_dataset, key):
    """IMA-226. Not 'MIP streams and the rest are TODO': every operator the ENGINE can run must
    reach the plate canvas and the ndviewer slider through the same _OperatorWorker.

    `reference` is the one this test was written for — a registered projector with NO card, so
    run_operator's `_OPERATIONS_BY_KEY[key].label` raised a bare KeyError out of the event loop
    and it could not be run live at all. `coordinate` is the same story on the region-operator
    side. Both stream here.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    tiles, pushes = _run_live(qapp, win, key)
    assert tiles is not None, f"{key}: no worker started — {win._readout.text()!r}"
    assert tiles, f"{key}: nothing reached the PLATE — {win._readout.text()!r}"
    assert pushes, f"{key}: nothing reached the ndviewer SLIDER — {win._readout.text()!r}"
    assert win._active_op_key == key, f"{key} streamed into layer {win._active_op_key!r}"
    assert win._readout.text().startswith("✓"), win._readout.text()
    # The tiles carry the operator's own pixels, not an empty canvas. Checked for the per-FOV
    # operators only: on this 4x4-frame fixture a REGION operator's blend weights divide by zero
    # (_montage.py:142) and the fused mosaic comes back NaN -> 0 on the uint16 cast. That is the
    # fixture's degenerate geometry, not the stream — the tile still arrives, which is what
    # IMA-226 is about, and test_ima222_* cover stitch's pixels on real extents.
    if key not in ("stitch", "coordinate"):
        assert any(np.asarray(t[3]).any() for t in tiles), f"{key} streamed all-zero tiles"
    win._stop_worker(); win.close()


def test_flatfield_streams_live_once_a_profile_is_installed(qapp, stub_detail, squid_dataset):
    """The last operator: flat-field cannot run without a profile, so with one installed it must
    stream exactly like the rest — and without one it must SAY it produced nothing."""
    from squidmip import FlatfieldProfile
    from squidmip._flatfield import set_profile
    import squidmip._flatfield as FF

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ny, nx = win._meta["frame_shape"]

    prev = FF.active_profile()
    try:
        FF._active = None                                   # no profile -> every field raises
        tiles, pushes = _run_live(qapp, win, "flatfield")
        assert not tiles and not pushes
        assert win._readout.text().startswith("⚠"), \
            f"a run that produced NOTHING reported success: {win._readout.text()!r}"
        assert "produced nothing" in win._readout.text()

        set_profile(FlatfieldProfile(np.ones((ny, nx), np.float32)))
        tiles, pushes = _run_live(qapp, win, "flatfield")
        assert tiles, f"flat-field with a profile still reached no tile: {win._readout.text()!r}"
        assert pushes, "flat-field reached the plate but not the ndviewer slider"
        assert win._readout.text().startswith("✓"), win._readout.text()
    finally:
        FF._active = prev
    win._stop_worker(); win.close()


def test_run_operator_refuses_a_non_operator_by_name(qapp, stub_detail, squid_dataset):
    """`minerva` is a CARD, not an operator — it is an export hand-off. Before IMA-226 the
    exploration tab built it a '(preview)' button from _OPERATIONS and clicking it handed
    'minerva' to the engine, which died with a raw KeyError printed into the status line."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("minerva", regions=["B3"], save=False)
    assert win._worker is None, "a non-operator started a run"
    assert "not a runnable operator" in win._readout.text()
    assert "KeyError" not in win._readout.text(), "raw engine exception leaked into the UI"
    win.run_operator("no_such_op", regions=["B3"], save=False)
    assert win._worker is None and "not a runnable operator" in win._readout.text()
    win.close()


def test_the_subset_tab_is_not_a_second_operator_launcher(qapp, stub_detail, squid_dataset):
    """There is ONE operator catalogue and ONE control panel, and they are in pane 1.

    This test used to assert the opposite: that the subset tab offered a preview button per
    entry of the ENGINE registry, while pane 1 offered a card per entry of ``_OPERATIONS``. Two
    registries, two control surfaces, one job — and they drifted, which is what the comment on
    the loop that built these buttons recorded. Julio: "we have the controls for the whole
    dataset on the left, but those controls are repeated for the subset on the right pane. Maybe
    it's not a good idea for there to be repetition of knowledge in our user interface."

    Running an operator on the subset is now a SCOPE on the one panel, so what is asserted here
    is the absence of the second surface and the presence of the scope that replaced it.
    """
    from squidmip import _explore as _E

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B3"])
    tab = win._op_tabs[key]
    texts = [b.text() for b in tab.findChildren(QPushButton)]
    assert not [t for t in texts if t.endswith(" (preview)")], \
        f"the subset tab still launches operators: {texts}"
    for op in V.runnable_operators():
        assert not [t for t in texts if V.operator_label(op).lower() in t.lower()], \
            f"the subset tab still launches {op!r}"
    # ...and the capability did not vanish with the buttons: it is a scope on pane 1.
    assert _E.SCOPE_SUBSET in [win._scope_run.itemText(i) for i in range(win._scope_run.count())]
    win.close()


# --- IMA-227: raw / MIP / stitched as toggleable, reorderable LAYERS ---------------------------

def _montage_px(qapp, ov):
    """The ACTIVE layer's montage pixels — cropped to the montage, never grab() of the widget.

    The widget paints labels, a 3px grid, status dots, the current-well box and the carrier
    photograph; on this fixture the plate fits to ~12 px per cell, so a whole-widget comparison
    stays 'different' (or 'identical') for reasons that have nothing to do with layers."""
    ov.recomposite(ov._active); qapp.processEvents()
    img = ov._active_source()
    a = np.frombuffer(img.constBits().asstring(img.byteCount()), np.uint8)
    a = a.reshape(img.height(), img.bytesPerLine() // (img.depth() // 8), -1)
    return a[:, :img.width(), :].copy()


def _run_to_completion(qapp, win, key, regions):
    win.run_operator(key, regions=regions, save=False)
    t0 = time.time()
    while win._worker is not None and win._worker.isRunning() and time.time() - t0 < 90:
        qapp.processEvents(); time.sleep(0.02)
    for _ in range(25):
        qapp.processEvents(); time.sleep(0.02)


def _layer_rows(win):
    """{layer label -> (checkbox, up, dn)} from the REAL Layers tab."""
    lw = win._op_tabs["layers"]
    rows = {}
    for cb in lw.findChildren(QCheckBox):
        row = cb.parentWidget()
        ups = [b for b in row.findChildren(QPushButton)]
        rows[cb.text()] = (cb, ups[0], ups[1])
    return rows


def test_layer_toggle_gives_back_raw_mip_and_stitched(qapp, stub_detail, squid_dataset):
    """IMA-227, driven through the real checkboxes: every operator is a LAYER, and the raw is
    recoverable by toggling — never destroyed. Measured on the montage, cropped.

    Julio's framing: "each transform is a LAYER, something like CellProfiler does this."
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _drain_until(qapp, lambda: len(win._overview._tiles) >= 2)
    ov = win._overview
    raw_before = _montage_px(qapp, ov)

    for key in ("mip", "stitch"):
        _run_to_completion(qapp, win, key, ["B2", "B3"])
    assert [ly.key for ly in win._op_stack.layers()] == ["raw", "mip", "stitch"]
    assert ov._active == "stitch"
    stitched = _montage_px(qapp, ov)

    win._open_op_tab("layers", "Layers", win._build_layers_tab)
    qapp.processEvents()
    rows = _layer_rows(win)

    # untick the top transform -> the one underneath shows. Nothing was destroyed to get there.
    rows["Stitch (register + fuse)"][0].setChecked(False)
    qapp.processEvents()
    assert ov._active == "mip", f"unticking stitch showed {ov._active!r}"
    mip_px = _montage_px(qapp, ov)
    assert not np.array_equal(mip_px, stitched), "the MIP layer renders the stitched pixels"

    # untick that too -> back to the RAW, byte for byte. This is the whole contract.
    rows["Maximum Intensity Projection"][0].setChecked(False)
    qapp.processEvents()
    assert ov._active == "raw", f"unticking every transform showed {ov._active!r}"
    assert win._plate_mode == "raw"
    raw_after = _montage_px(qapp, ov)
    assert raw_after.shape == raw_before.shape
    assert np.array_equal(raw_after, raw_before), \
        "the raw acquisition was not recovered by toggling — a transform destroyed it"
    assert not np.array_equal(raw_after, mip_px), "raw and MIP render identical pixels"

    # and re-ticking brings the transform straight back: the layers kept their pixels
    rows["Maximum Intensity Projection"][0].setChecked(True)
    qapp.processEvents()
    assert ov._active == "mip"
    assert np.array_equal(_montage_px(qapp, ov), mip_px), "re-enabling a layer lost its pixels"
    win.close()


def test_the_base_layer_can_be_neither_disabled_nor_reordered(qapp, stub_detail, squid_dataset):
    """The raw must ALWAYS remain recoverable. Two ways it used not to be:

    - ``toggle('raw', False)`` was accepted, so unticking every box left top_enabled() == None and
      _apply_layers no-opped: the plate kept painting the last operator with every checkbox OFF.
    - ``move('raw', +1)`` reordered the base like any other layer (and ``move('mip', -1)`` shoved
      it off index 0 from the other side), putting raw ABOVE an operator — which the plate then
      renders, hiding an enabled layer with no way to reach it.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _drain_until(qapp, lambda: len(win._overview._tiles) >= 2)
    _run_to_completion(qapp, win, "mip", ["B2", "B3"])
    win._open_op_tab("layers", "Layers", win._build_layers_tab)
    qapp.processEvents()
    rows = _layer_rows(win)
    raw_cb, raw_up, raw_dn = next(v for k, v in rows.items() if k.startswith("raw"))

    # the controls SAY it, rather than accepting a click the model then ignores
    assert not raw_cb.isEnabled() and raw_cb.isChecked(), "the base layer's checkbox is clickable"
    assert not raw_up.isEnabled() and not raw_dn.isEnabled(), "the base layer can be reordered"

    # ...and the model enforces it even when driven directly
    win._on_layer_toggle("raw", False)
    assert win._op_stack.top_enabled() is not None, "every layer got disabled"
    assert [ly for ly in win._op_stack.layers() if ly.key == "raw"][0].enabled
    win._on_layer_move("raw", +1)
    assert [ly.key for ly in win._op_stack.layers()][0] == "raw", "the base moved off the bottom"
    win._on_layer_move("mip", -1)
    assert [ly.key for ly in win._op_stack.layers()] == ["raw", "mip"], \
        "an operator was pushed below the base"
    assert win._overview._active == "mip"
    win.close()


def test_layer_reorder_changes_what_the_plate_shows(qapp, stub_detail, squid_dataset):
    """Reorder, not just toggle: the plate renders the TOPMOST enabled layer, so moving one up
    must change the pixels on screen — driven through the real ↑ button."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _drain_until(qapp, lambda: len(win._overview._tiles) >= 2)
    for key in ("mip", "stitch"):
        _run_to_completion(qapp, win, key, ["B2", "B3"])
    ov = win._overview
    win._open_op_tab("layers", "Layers", win._build_layers_tab)
    qapp.processEvents()
    assert ov._active == "stitch"
    stitched = _montage_px(qapp, ov)

    _layer_rows(win)["Maximum Intensity Projection"][1].click()   # the ↑ GESTURE
    qapp.processEvents()
    assert [ly.key for ly in win._op_stack.layers()] == ["raw", "stitch", "mip"]
    assert ov._active == "mip", f"MIP moved to the top but the plate shows {ov._active!r}"
    assert not np.array_equal(_montage_px(qapp, ov), stitched), \
        "the reorder changed the stack but not the plate"
    win.close()


def test_an_exploration_tab_rerun_never_wipes_the_plate_wide_layer(qapp, stub_detail, squid_dataset):
    """The layer-key bug that already bit once: reset_layer/_recomposite key off the LAYER key
    ('mip@<tab>'), not the operator key, so a tab re-running MIP must leave the plate-wide 'mip'
    layer's pixels standing."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _drain_until(qapp, lambda: len(win._overview._tiles) >= 2)
    _run_to_completion(qapp, win, "mip", ["B2", "B3"])
    ov = win._overview
    plate_wide = ov._store["mip"].copy()
    assert plate_wide.any()

    key = win.open_exploration_tab(["B3"])
    win.run_operator("mip", regions=["B3"], save=False, tab_key=key)
    t0 = time.time()
    while win._worker is not None and win._worker.isRunning() and time.time() - t0 < 90:
        qapp.processEvents(); time.sleep(0.02)
    for _ in range(25):
        qapp.processEvents(); time.sleep(0.02)

    assert f"mip@{key}" in ov._store, "the tab run did not get its own layer"
    assert "mip" in ov._store, "the tab run WIPED the plate-wide mip layer"
    assert np.array_equal(ov._store["mip"], plate_wide), \
        "the tab run overwrote the plate-wide mip layer's pixels"
    assert [ly.key for ly in win._op_stack.layers()] == ["raw", "mip", f"mip@{key}"]
    win._stop_worker(); win.close()


def test_dropping_a_layer_frees_its_store_and_composite(qapp, stub_detail, squid_dataset):
    """~95 MB per layer lives in _store/_final_arr. Dropping a layer must release BOTH — dropping
    only the canvas looks like a fix and leaks the majority of the memory."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _drain_until(qapp, lambda: len(win._overview._tiles) >= 2)
    _run_to_completion(qapp, win, "mip", ["B2", "B3"])
    ov = win._overview
    ov.recomposite("mip"); qapp.processEvents()
    for d in ("_store", "_final_arr", "_op_canvas", "_op_final", "_tiles_by_layer"):
        assert "mip" in getattr(ov, d), f"mip never reached {d}"
    ov.drop_layer("mip")
    for d in ("_store", "_final_arr", "_op_canvas", "_op_final", "_tiles_by_layer"):
        assert "mip" not in getattr(ov, d), f"drop_layer leaked {d}['mip']"
    assert ov._active == "raw", "dropping the shown layer left the plate on it"
    assert "raw" in ov._store, "dropping a layer took the raw with it"
    win.close()


# --- IMA-245: a region operator's result must reach the CENTRAL array viewer -------------------

_NONSQUARE_YAML = """\
version: 1
objective: 20x
channels:
- name: Fluorescence 638 nm - Penta
  camera_settings:
    '1':
      display_color: '#FF0000'
      exposure_time_ms: 50.0
"""

_NONSQUARE_ACQ_YAML = """\
objective:
  pixel_size_um: 0.325
  magnification: 20.0
  sensor_pixel_size_um: 3.76
sample:
  wellplate_format: 1536 well plate
z_stack:
  nz: 1
  delta_z_mm: 0.0015
time_series:
  nt: 1
"""


@pytest.fixture
def nonsquare_mosaic_dataset(tmp_path):
    """A real, stitchable Squid acquisition whose mosaic is deliberately NOT square.

    Six 256x256 fields on a 3-wide x 2-tall grid with a real 56 px (22%) overlap, cropped out of
    ONE noise image so registration has genuine, matchable content — this is a real acquisition
    that the real stitcher really fuses, not a stub.

    Non-square is the entire point. The frame is square and the mosaic is 456x656, so a viewer
    sized as a FRAME and a viewer sized as the MOSAIC produce different numbers, and a test can
    tell which one the array viewer actually got. On a square 6x6 plate well (the synthetic 2x2
    wellplate) both answers are 512x512 and the defect is invisible.

    Returns (root, region, frame_px, mosaic_extent_px).
    """
    import json

    import tifffile

    frame, step, cols, rows = 256, 200, 3, 2
    region, ch = "B2", CH_IN_YAML
    mh, mw = step * (rows - 1) + frame, step * (cols - 1) + frame     # 456 x 656
    rng = np.random.default_rng(245)
    source = rng.integers(0, 4000, size=(mh, mw), dtype=np.uint16)

    folder = tmp_path / "acq_nonsquare" / "0"
    folder.mkdir(parents=True)
    px_um, lines = 0.325, ["region,x (mm),y (mm),z (mm)"]
    for r in range(rows):
        for c in range(cols):
            fov = r * cols + c
            top, left = r * step, c * step
            tifffile.imwrite(folder / f"{region}_{fov}_0_{ch}.tiff",
                             source[top:top + frame, left:left + frame])
            # stage mm: px -> um -> mm. The reader turns these back into fov_positions_um, which
            # is what _placement (and therefore the push geometry) lays the mosaic out from.
            lines.append(f"{region},{left * px_um / 1000.0},{top * px_um / 1000.0},")
    root = tmp_path / "acq_nonsquare"
    (root / "acquisition_channels.yaml").write_text(_NONSQUARE_YAML)
    (root / "acquisition.yaml").write_text(_NONSQUARE_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(
        json.dumps({"Nz": 1, "Nt": 1, "dz(um)": 1.5,
                    "objective": {"magnification": 20.0}, "sensor_pixel_size_um": 3.76}))
    (root / "coordinates.csv").write_text("\n".join(lines) + "\n")
    return root, region, frame, (mh, mw)


def _stitch_into_central_viewer(qapp, win, region):
    """Run the REAL stitch operator on ``region`` and return (fused mosaic shape, pushes).

    ``_on_well`` is wrapped rather than mocked: it is the one place the fused mosaic exists as an
    array, and the test needs its true extent to compare against what the viewer was handed.
    """
    fused = []
    original = V._OperatorWorker._on_well

    def spy(worker, r, f, image):
        fused.append(np.asarray(image).shape)
        return original(worker, r, f, image)

    V._OperatorWorker._on_well = spy
    try:
        win._stop_preview()
        _drain_until(qapp, lambda: not win._busy(), timeout=120)
        win._detail.arrays.clear()
        win.run_operator("stitch", regions=[region], save=False)
        assert win._worker is not None, win._readout.text()
        t0 = time.time()
        while win._worker is not None and win._worker.isRunning() and time.time() - t0 < 300:
            qapp.processEvents(); time.sleep(0.02)
        for _ in range(25):
            qapp.processEvents(); time.sleep(0.02)
    finally:
        V._OperatorWorker._on_well = original
    assert fused, f"the stitch produced no fused mosaic: {win._readout.text()!r}"
    return fused[0][-2:], list(win._detail.arrays)


def test_ima245_region_operator_reaches_the_central_viewer_as_a_mosaic(
        qapp, stub_detail, nonsquare_mosaic_dataset):
    """A stitch's FUSED MOSAIC must arrive in the central array viewer, sized as the mosaic.

    Reported from the live GUI: "after I see the stitch, I cannot see it in my central array
    viewer — it is black". The plate showed the mosaic; the viewer beside it did not.

    This asserts on the RECTANGLE, not on the handler being called. A region operator yields one
    fused mosaic per region, and the viewer's canvas was declared as a FRAME (_PUSH_PX square) —
    so the mosaic was squashed into a shape it does not have, or refused for not fitting. Both
    fail here, and the aspect is checked against the real fused array rather than a constant.
    """
    root, region, frame_px, predicted = nonsquare_mosaic_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._meta["frame_shape"] == (frame_px, frame_px)
    # the coordinate-derived extent the viewer is sized from, and it is NOT the frame
    assert V.region_mosaic_extent_px(win._meta, [region]) == predicted

    mosaic_hw, pushes = _stitch_into_central_viewer(qapp, win, region)
    assert mosaic_hw != (frame_px, frame_px), \
        f"the stitch yielded a frame, not a mosaic ({mosaic_hw}) — the fixture is not stitching"
    assert pushes, f"NOTHING reached the central array viewer: {win._readout.text()!r}"

    got = pushes[0][4]
    canvas = win._detail.canvases[-1]
    assert got == canvas, f"the viewer was declared {canvas} and handed {got}"
    # the shape is the MOSAIC's, bounded by the push budget — not the frame's, and not a square
    assert max(got) == V._PUSH_PX and got[0] != got[1], f"push {got} is not a bounded mosaic"
    scale = V._PUSH_PX / max(mosaic_hw)
    assert abs(got[0] / got[1] - mosaic_hw[0] / mosaic_hw[1]) < 0.02, \
        f"push {got} does not have the fused mosaic's aspect {mosaic_hw}"
    assert abs(got[0] - mosaic_hw[0] * scale) <= 2 and abs(got[1] - mosaic_hw[1] * scale) <= 2, \
        f"push {got} is not the fused mosaic {mosaic_hw} scaled to fit {V._PUSH_PX}"
    # ...and nothing was thrown away getting there
    assert win._dropped_pushes == 0, \
        f"{win._dropped_pushes} push(es) dropped: {win._readout.text()!r}"
    assert win._readout.text().startswith("✓") and "⚠" not in win._readout.text()
    win._stop_worker(); win.close()


def test_ima245_every_region_operator_is_sized_as_a_region_not_a_frame(
        qapp, stub_detail, nonsquare_mosaic_dataset):
    """The category, not the name. Every operator in ``available_region_operators()`` yields a
    region mosaic, so every one of them sizes the viewer from the mosaic extent — and every
    per-FOV projector still sizes it from the frame. An `if operator == "stitch"` branch would
    pass the test above and fail this one."""
    from squidmip import available_projectors, available_region_operators

    root, region, frame_px, (mh, mw) = nonsquare_mosaic_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    mosaic = V.push_shape_for(win._meta, True, [region])
    frame = V.push_shape_for(win._meta, False, [region])
    assert mosaic != frame, "the region and per-FOV surfaces are the same rectangle"
    assert abs(mosaic[0] / mosaic[1] - mh / mw) < 0.01

    for op in available_region_operators():
        w = V._OperatorWorker(op, win._reader, win._meta, win._fov_index, "",
                              regions=[region], save=False, n_fovs=None)
        assert w.push_shape == mosaic, f"region operator {op!r} is sized {w.push_shape}, not {mosaic}"
    for op in available_projectors():
        w = V._OperatorWorker(op, win._reader, win._meta, win._fov_index, "",
                              regions=[region], save=False, n_fovs=None)
        assert w.push_shape == frame, f"projector {op!r} is sized {w.push_shape}, not {frame}"
    win.close()


def test_ima245_an_unshowable_push_is_counted_and_said_out_loud(
        qapp, stub_detail, nonsquare_mosaic_dataset):
    """A push that cannot be shown must never be silent. An ndviewer build with no
    ``register_array`` (the installed 0.1.0 has none) discarded every computed result at a
    ``hasattr`` guard: no counter, no message, a black viewer, and a human to find it."""
    root, region, _frame, _extent = nonsquare_mosaic_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._push_shape = (16, 16)
    win._push_index = None
    win._push_problem = None
    win._dropped_pushes = 0

    class _NoArrays:                       # an older ndviewer: no register_array at all
        pass

    detail, win._detail = win._detail, _NoArrays()
    win._on_push(0, [np.zeros((16, 16), np.uint16)])
    assert win._dropped_pushes == 1
    assert "register_array" in win._readout.text() and "⚠" in win._readout.text()

    win._detail, win._push_problem, win._dropped_pushes = detail, None, 0
    win._push_index = {}                   # an index this run's viewer has no slot for
    win._on_push(7, [np.zeros((16, 16), np.uint16)])
    assert win._dropped_pushes == 1 and "no slot" in win._readout.text()

    win._push_problem, win._dropped_pushes = None, 0
    win._push_index = None
    win._on_push(0, [np.zeros((99, 3), np.uint16)])      # not the declared canvas
    assert win._dropped_pushes == 1 and "99x3" in win._readout.text()
    # a success line must not be able to bury the warning
    win._run_readout("✓ done")
    assert win._readout.text().startswith("✓ done") and "99x3" in win._readout.text()
    win.close()


def test_ima245_real_tissue_stitch_reaches_the_central_viewer(qapp, stub_detail, real_dataset):
    """The honest case, on the acquisition the product is demoed on.

    ``manual0`` is 27 freeform 10x FOVs at ~10% overlap; the fused mosaic is 11535x9635 and the
    frame is 2084x2084, so nothing here can be mistaken for the other. This is the exact run the
    defect was reported from, and it costs ~13 s, so it runs by default (it skips only where the
    acquisition is absent). Measured on the fix: the array viewer was declared and handed
    (512, 428) — the fused 11535x9635 mosaic's aspect to within a pixel — with 0 dropped pushes."""
    win = V.PlateWindow(None)
    win.ingest(str(real_dataset))
    mosaic_hw, pushes = _stitch_into_central_viewer(qapp, win, "manual0")
    assert pushes, f"NOTHING reached the central array viewer: {win._readout.text()!r}"
    got = pushes[0][4]
    assert got == win._detail.canvases[-1]
    assert got != tuple(win._meta["frame_shape"]) and max(got) == V._PUSH_PX
    scale = V._PUSH_PX / max(mosaic_hw)
    assert abs(got[0] - mosaic_hw[0] * scale) <= 2 and abs(got[1] - mosaic_hw[1] * scale) <= 2, \
        f"push {got} is not the fused mosaic {mosaic_hw} scaled to fit {V._PUSH_PX}"
    assert win._dropped_pushes == 0, win._readout.text()
    win._stop_worker(); win.close()


# ============================================================================ IMA-253 / IMA-249
# A REGION IS A MOSAIC CONTAINING AN ARRAY OF FOVs, and it must look like one the moment the
# acquisition opens. Julio, on the real 10x tissue:
#
#   "I still don't see the mosaics in the plate at all. It doesn't look like a slide. It looks
#    like a bunch of squares overlapped with each other under different regions ... They look
#    like overlapping FOVs when they should actually be independent mosaics that have different
#    slots in the slide carrier."
#
# Two causes, one change: the layout came from ENUMERATION ORDER (a freeform id carries no
# position), and the raw preview read ONE representative FOV per region while `set_mosaic_boxes`
# was only ever called from `run_operator` -- so the mosaic was invisible until something ran.
# Everything here is measured on the montage, never on a whole-widget grab: labels, grid and
# status dots keep whole-frame variance high, so a widget-level check passes against a BLANK
# montage. That trap has been hit once already.

def _region_crop(ov, region):
    """The rendered pixels of ONE region's cell -- its own rectangle, not a grid square."""
    rc = next(k for k, v in ov._by_rc.items() if v == region)
    x, y, w, h = ov._cell_rect(*rc)
    img = ov.grab().toImage().convertToFormat(QImage.Format_RGB32)
    a = np.frombuffer(img.constBits().asstring(img.byteCount()), np.uint8)
    a = a.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width(), :3]
    return a[max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)]


def test_ima253_real_tissue_previews_both_regions_as_mosaics_before_any_operator_runs(
        qapp, stub_detail, real_dataset):
    """The acceptance number: 55 boxes and two composited mosaics, with nothing run.

    27 + 28 FOVs. Before the fix ``boxes on ov`` was 0 and each region showed ONE frame stretched
    over its cell, because ``set_mosaic_boxes`` was reachable only from ``run_operator``.
    """
    win = V.PlateWindow(None)
    win.ingest(str(real_dataset))
    ov = win._overview
    assert len(ov._boxes) == 55, (
        f"the mosaic geometry is pure arithmetic on coordinates.csv and is known at ingest, but "
        f"only {len(ov._boxes)} boxes reached the plate. This is IMA-249: the boxes existed and "
        f"were never handed to the widget until an operator ran.")
    per_region: dict = {}
    for region, _fov in ov._boxes:
        per_region[region] = per_region.get(region, 0) + 1
    assert per_region == {"manual0": 27, "manual1": 28}, per_region

    # ...and the preview really composites all of them, rather than one frame per region.
    assert _drain_until(qapp, lambda: win._preview is None or not win._preview.isRunning(), 180)
    assert win._worker is None, "no operator ran; the mosaic must be there without one"
    for region in ("manual0", "manual1"):
        crop = _region_crop(ov, region)
        assert crop.size and crop.std() > 3, f"{region} renders blank/uniform (std {crop.std():.2f})"
    win.close()


def test_ima253_preview_plan_reads_every_fov_of_a_region_but_only_one_of_a_single_fov_well(
        qapp, stub_detail, real_dataset, squid_dataset):
    """Cost is driven by the REAL FOV COUNT PER REGION -- the reason 1536x1 cannot get slower."""
    from squidmip import open_reader

    meta = open_reader(str(real_dataset)).metadata
    idx = {r: {"rc": (i, 0), "idx": i} for i, r in enumerate(meta["regions"])}
    plan = V._PreviewWorker(None, meta, idx, list(meta["regions"]))._plan()
    assert len(plan) == 55, f"the preview reads {len(plan)} planes/channel, not 55"
    assert all(box is not None for _r, _f, box in plan)

    root, _ = squid_dataset                       # 2 FOVs/region, but specks apart on this fixture
    m2 = open_reader(str(root)).metadata
    idx2 = {r: {"rc": (i, 0), "idx": i} for i, r in enumerate(m2["regions"])}
    plan2 = V._PreviewWorker(None, m2, idx2, list(m2["regions"]))._plan()
    assert all(box is None for _r, _f, box in plan2), \
        "sub-_MIN_PREVIEW_BOX_PX fields are specks: reading one plane each is cost with no picture"


def test_ima253_real_tissue_regions_are_stacked_in_y_and_offset_in_x(
        qapp, stub_detail, real_dataset):
    """The layout defect itself: two regions that are separated in Y rendered SIDE BY SIDE.

    manual0 spans stage y 10186..17238, manual1 21113..28165 -- no overlap at all -- while their
    x ranges (96814..102456 / 97937..103578) overlap heavily. The old carrier put them in columns
    0 and 1 of a "4 slide carrier" by enumeration order, which is why the plate did not look like
    a slide.
    """
    win = V.PlateWindow(None)
    win.ingest(str(real_dataset))
    ov = win._overview
    assert ov._layout is not None, "a freeform holder must be placed by geometry"
    r0 = ov._cell_rect(*next(k for k, v in ov._by_rc.items() if v == "manual0"))
    r1 = ov._cell_rect(*next(k for k, v in ov._by_rc.items() if v == "manual1"))
    assert r1[1] >= r0[1] + r0[3], f"manual1 must render BELOW manual0, got {r0} / {r1}"
    assert r1[0] < r0[0] + r0[2] and r0[0] < r1[0] + r1[2], "...and still overlap it in x"
    assert r1[0] > r0[0], "manual1 is further +x than manual0, as the stage records"
    # relative scale: the two mosaics have the same extent on this dataset, so must the cells
    assert r0[2] == pytest.approx(r1[2], rel=0.02)
    assert r0[3] == pytest.approx(r1[3], rel=0.02)
    win.close()


def test_ima253_shuffling_the_region_names_does_not_move_anything(qapp, stub_detail, real_dataset):
    """MUTATION-CHECK. This is the assertion that proves placement follows GEOMETRY.

    Reverse the order the acquisition reports its regions in. A layout driven by enumeration
    order flips; one driven by ``fov_positions_um`` cannot notice.
    """
    from squidmip import open_reader
    from squidmip._plate import build_plate

    meta = open_reader(str(real_dataset)).metadata
    ref = build_plate(meta)
    flipped = build_plate({**meta, "regions": list(reversed(meta["regions"]))})
    assert flipped.cell_layout() == ref.cell_layout()
    assert flipped.occupied_map == ref.occupied_map


def test_ima253_the_default_paint_path_loads_no_carrier_png(qapp, stub_detail, squid_dataset,
                                                            monkeypatch):
    """Art available or not, the plate renders IDENTICALLY -- because art is never consulted.

    The PNG needed three calibration constants (``a1_x_pixel``, ``a1_x_mm``, ``mm_per_pixel``) to
    agree with the geometry the cells are laid out from; when they did not, nothing raised and the
    wells were simply drawn in the wrong place. The registry is kept in ``_plate`` as an optional
    skin, so this asserts it is OFF the path, not that it is gone.
    """
    import squidmip._plate as P

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _drain_until(qapp, lambda: len(win._overview._tiles) >= 2)
    before = _montage_px(qapp, win._overview)
    shot = win._overview.grab().toImage()

    calls = []
    monkeypatch.setattr(P, "carrier_art", lambda *a, **k: calls.append(a) or None)
    win2 = V.PlateWindow(None)
    win2.ingest(str(root))
    _drain_until(qapp, lambda: len(win2._overview._tiles) >= 2)
    assert np.array_equal(_montage_px(qapp, win2._overview), before)
    assert win2._overview.grab().toImage() == shot, \
        "the render changed when the art registry was disabled -- art is still on the paint path"
    assert not calls, f"carrier_art() was called {len(calls)}x during a default open"
    assert not hasattr(win2._overview, "_art_img")
    win.close(); win2.close()


def test_ima253_empty_slots_are_visibly_distinct_from_occupied_ones(qapp, stub_detail):
    """Julio said the photograph was poor at exactly this, so it is now drawn and measured."""
    from squidmip._plate import SlideCarrier

    plate = SlideCarrier.from_format("4 slide carrier", occupancy={"manual0": [0]},
                                     cell_ids=["manual0"])
    ov = V.PlateOverview(plate.row_labels, plate.col_labels, plate.occupied_map)
    ov.set_carrier(plate)
    ov.resize(600, 240)
    img = ov.grab().toImage().convertToFormat(QImage.Format_RGB32)
    a = np.frombuffer(img.constBits().asstring(img.byteCount()), np.uint8)
    a = a.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width(), :3]

    def _cell_px(ci):
        x, y, w, h = ov._cell_rect(0, ci)
        return a[int(y) + 2:int(y + h) - 2, int(x) + 2:int(x + w) - 2]

    occupied, empty = _cell_px(0), _cell_px(1)
    assert occupied.size and empty.size
    assert abs(float(occupied.mean()) - float(empty.mean())) > 1.5, \
        "an empty slot must not look like an occupied one"
    ov.deleteLater()


# --- IMA-260: three panes on OPEN, and the empty third pane teaches by EXAMPLE ------------------
#
# IMA-237 shipped pane 3 collapsed until a Shift-drag revealed it, which made the whole feature
# undiscoverable: you cannot find a pane that is not there, and the only gesture that summoned it
# was itself invisible. IMA-260 opens with all three and fills the empty one with EXAMPLE USAGE.
#
# The earlier three-pane check passed FAKE-GREEN because it never showed the window: an unshown
# QSplitter reports whatever sizes it was handed and every child has zero geometry, so "the pane
# is there" was trivially true and "the pane has width" was unaskable. Everything below shows the
# window at a real size first and asserts on REAL widget geometry.

def _drain_preview(win, app, timeout_s=60):
    """Block until the raw preview worker has stopped streaming (tools/walkthrough's helper).

    A fixed settle() races it: the fill's duration is however many fields the acquisition has.
    """
    _drain_until(app, lambda: getattr(win, "_preview", None) is None or not win._preview.isRunning(),
                 timeout=timeout_s)
    for _ in range(20):                 # let the queued tileReady slots actually run
        app.processEvents()


def _shown(qapp, path=None, size=(1600, 900)):
    """A window the user could actually look at: real size, really shown, really ingested."""
    win = V.PlateWindow(None)
    win.resize(*size)
    win.show()
    qapp.processEvents()
    if path is not None:
        win.ingest(str(path))
        _drain_preview(win, qapp)
    return win


def _right_click(qapp, ov, pos):
    """A REAL right-click: a QContextMenuEvent through Qt's dispatch, not a direct handler call."""
    from PyQt5.QtGui import QContextMenuEvent
    p = pos.toPoint() if hasattr(pos, "toPoint") else pos
    ev = QContextMenuEvent(QContextMenuEvent.Mouse, p, ov.mapToGlobal(p))
    qapp.sendEvent(ov, ev)
    qapp.processEvents()
    return ov._context_menu


def _menu_action(menu, needle):
    return next((a for a in menu.actions() if needle.lower() in a.text().lower()), None)


def test_ima260_all_three_panes_have_real_width_on_open(qapp, stub_detail, squid_dataset):
    """The headline. Not 'the splitter has three children' — three panes a user can SEE."""
    root, _ = squid_dataset
    win = _shown(qapp, root)
    assert win._split.count() == 3
    assert win._split.widget(2) is win._explore_pane, "pane 3 is not the exploration pane"

    plate, viewer, explore = win._split.sizes()
    assert explore > 0, "the exploration pane opened collapsed — IMA-237's undiscoverable state"
    assert explore >= 360, f"pane 3 opened at {explore} px, too narrow to read its own copy"
    # ...and the same claim measured off the LIVE widget, which is what fake-green missed.
    assert win._explore_pane.isVisible()
    assert win._explore_pane.width() > 0
    assert win._explore_pane.height() > 0
    assert plate > 0 and viewer > 0, "opening pane 3 collapsed one of the other two"
    win.close()


def test_ima260_the_empty_pane_shows_example_usage_not_a_blank_strip(qapp, stub_detail,
                                                                     squid_dataset):
    root, _ = squid_dataset
    win = _shown(qapp, root)
    text = win.explore_empty_text()
    assert text.strip(), "the empty exploration pane is blank"

    low = text.lower()
    # PRIMARY: right-click -> Control Well, in that order (Julio's priority, and his correction:
    # the unit is the WELL, never the FOV).
    assert "right-click" in low and "control well" in low
    assert low.index("right-click") < low.index("hold shift"), \
        "the Shift line came before the right-click line — the primary message is right-click"
    assert "control fov" not in low, "the action is Control Well, not Control FOV"
    # SECONDARY: hold Shift to view a subset here.
    assert "hold shift" in low
    # ...and the whole thing is framed as an EXAMPLE, never as an instruction. This is Julio's
    # explicit second correction, so it is asserted rather than left to the reviewer's eye.
    assert "for example" in low
    assert "only examples" in low
    win.close()


def test_ima260_empty_state_copy_meets_the_legibility_floor(qapp, stub_detail, squid_dataset):
    """16 arcmin minimum = 17.3 px at 60 cm and (scaling 28.8 px @ 20 arcmin) 23.0 px at 1 m.

    Copy sized for the near case only is unreadable from the chair the big monitor is viewed
    from, which is the entire reason this pane carries text at all.
    """
    from PyQt5.QtWidgets import QLabel
    root, _ = squid_dataset
    win = _shown(qapp, root)
    labels = [w for w in win._explore_empty.findChildren(QLabel) if w.text().strip()]
    assert labels, "no copy to measure"
    # 14.0 px = 16 arcmin at ~40 cm, i.e. desk distance on a SMALL monitor.
    # The old 23.0 px floor scaled 16 arcmin to a 1 m viewing distance on a large monitor.
    # Julio is on a small monitor and called the result "huge text" -- the spec changed, so the
    # test changes with it rather than being deleted. The constraint still exists and still bites:
    # anything under 14 px is genuinely unreadable at the desk and fails here.
    floor = 14.0
    for lab in labels:
        px = lab.font().pixelSize()
        if px <= 0:                    # a point-sized font: convert through the screen's DPI
            px = lab.fontMetrics().height()
        assert px >= floor, f"{lab.text()[:32]!r} renders at {px} px, under the {floor} px floor"
        # and it is really on screen at that size, not merely configured
        assert lab.isVisible() and lab.height() >= floor
    win.close()


def test_ima260_the_example_goes_away_with_content_and_comes_back_when_empty(
        qapp, stub_detail, squid_dataset):
    """Both directions. A user who explores once and tidies up must not be left with the blank
    strip the empty state exists to prevent."""
    root, _ = squid_dataset
    win = _shown(qapp, root)
    assert win.explore_empty_text().strip()             # present on open

    _shift_drag_over(win, ["B3"])
    qapp.processEvents()
    assert win._explore_tabs.count() == 1
    assert win.explore_empty_text() == "", "the example copy stayed up behind real content"
    assert win._explore_pane.currentWidget() is win._explore_tabs

    _close_exploration_pane(win)                        # the REAL tab-close path
    qapp.processEvents()
    assert win._explore_tabs.count() == 0
    assert win.explore_empty_text().strip(), "the example never came back when the pane emptied"
    assert win._split.sizes()[2] > 0, "the pane collapsed instead of returning to its empty state"
    win.close()


def test_ima260_right_click_offers_control_well_and_setting_it_pins_the_reference(
        qapp, stub_detail, squid_dataset):
    """The example points at an action, so the action has to exist and has to work."""
    root, _ = squid_dataset
    win = _shown(qapp, root)
    ov = _freeze(win._overview)
    ri, ci = win._fov_index["B3"]["rc"]

    menu = _right_click(qapp, ov, _pt(ri, ci))
    act = _menu_action(menu, "control well")
    assert act is not None, "right-click offers no Control Well — the empty pane's example is a lie"
    assert act.isEnabled()
    act.trigger()
    qapp.processEvents()

    # ONE identity, read from the owner and from the plate — never compared across three copies.
    assert win.control_well() == "B3"
    assert win._overview.control_well() == "B3", "the plate disagrees with the window"
    # ...pinned FIRST in pane 3, and not closable by the normal affordance.
    assert win._explore_tabs.count() >= 1
    assert win._explore_tabs.widget(0) is win._op_tabs[V.PlateWindow.CONTROL_KEY]
    assert win._explore_tabs.widget(0).regions == ["B3"]
    assert "B3" in win._explore_tabs.tabText(0)
    from PyQt5.QtWidgets import QTabBar
    assert win._explore_tabs.tabBar().tabButton(0, QTabBar.RightSide) is None
    assert win._detach_tab(0, win._explore_tabs) is None, "the pinned control tab floated away"
    # ...and the pane now holds content, so the example stands down.
    assert win.explore_empty_text() == ""
    menu.close()
    win.close()


def test_ima260_a_second_control_releases_the_first(qapp, stub_detail, squid_dataset):
    """One control at a time. The release is implicit in there being one variable — this asserts
    there is no stale second tab and no stale second frame."""
    root, _ = squid_dataset
    win = _shown(qapp, root)
    win.set_control_well("B2")
    qapp.processEvents()
    win.set_control_well("B3")
    qapp.processEvents()

    assert win.control_well() == "B3"
    assert win._overview.control_well() == "B3"
    controls = [i for i in range(win._explore_tabs.count())
                if win._explore_tabs.tabText(i).startswith("Control")]
    assert controls == [0], f"expected exactly one pinned control tab, got {controls}"
    assert win._explore_tabs.widget(0).regions == ["B3"]
    win.close()


def test_ima260_clearing_the_control_drops_the_frame_the_tab_and_restores_the_example(
        qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = _shown(qapp, root)
    ov = _freeze(win._overview)
    ri, ci = win._fov_index["B3"]["rc"]
    win.set_control_well("B3")
    qapp.processEvents()

    menu = _right_click(qapp, ov, _pt(ri, ci))          # right-click the control itself
    clear = _menu_action(menu, "clear control well")
    assert clear is not None, "no way to release a control from the plate that set it"
    clear.trigger()
    qapp.processEvents()

    assert win.control_well() is None
    assert win._overview.control_well() is None, "a stale blue frame survived the clear"
    assert win._explore_tabs.count() == 0
    assert win.explore_empty_text().strip(), "the empty pane went blank instead of back to example"
    menu.close()
    win.close()


def test_ima260_the_control_frame_is_really_painted_and_is_not_the_red_box(qapp, stub_detail,
                                                                          squid_dataset):
    """Pixels, not state: a control the user cannot see on the plate is not a control."""
    from PyQt5.QtGui import QImage
    root, _ = squid_dataset
    win = _shown(qapp, root)
    ov = _freeze(win._overview, cd=60.0)
    ov.resize(500, 400)
    qapp.processEvents()

    def _grab():
        img = ov.grab().toImage().convertToFormat(QImage.Format_RGB32)
        a = np.frombuffer(img.constBits().asstring(img.byteCount()), np.uint8)
        return a.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width(), :3]

    before = _grab()
    win.set_control_well("B3")
    qapp.processEvents()
    after = _grab()
    assert not np.array_equal(before, after), "setting the control changed no pixels at all"

    ri, ci = win._fov_index["B3"]["rc"]
    x, y, w, h = ov._cell_rect(ri, ci)
    cell = after[int(y):int(y + h), int(x):int(x + w)].astype(int)
    blue = V._CONTROL_BLUE
    hit = (np.abs(cell[..., 2] - blue.red()) <= 24) & (np.abs(cell[..., 1] - blue.green()) <= 24) \
        & (np.abs(cell[..., 0] - blue.blue()) <= 24)
    assert hit.sum() > 20, "no light-blue control frame on the control well's cell"
    # ...and it is NOT the transient red current-FOV box wearing a different name.
    assert blue.red() < blue.blue(), "the control frame must be blue, not red"
    win.close()


# ------------------------------------------------- the raw mosaic preview hands over a PYRAMID
#
# The written-OME-Zarr path has always given napari a multiscale pyramid. The raw preview path
# gave it full-resolution fused planes: 54.9 MB per channel per z on the real 10x region, four
# channels composited, re-fused on every z step. These pin the wiring that closes that gap.


class _PyrReader:
    #: The plane cache keys on the acquisition a reader reads, so every reader must name it.
    def __init__(self, frame=(256, 256), path="/fake/acquisition/viewer"):
        self.frame = frame
        self._path = path

    def read(self, region, fov, channel, z, t=0):
        return np.full(self.frame, z + 1, dtype=np.uint16)


def _pyr_meta(nz=4, n=16, frame=(256, 256), px=1.0):
    return {
        "regions": ["A1"],
        "fovs_per_region": {"A1": list(range(n))},
        "fov_positions_um": {("A1", i): (i * frame[1] * px, 0.0) for i in range(n)},
        "pixel_size_um": px,
        "frame_shape": frame,
        "dtype": "uint16",
        "n_z": nz,
        "dz_um": 1.5,
        "channels": [{"name": "488"}, {"name": "561"}],
    }


def test_the_mosaic_worker_emits_a_pyramid_not_a_single_resolution_stack(qapp):
    """``_MosaicWorker`` is what feeds pane 2 on OPEN, before any operator runs."""
    meta = _pyr_meta()
    got, problems = [], []
    w = V._MosaicWorker(_PyrReader(), meta, "A1", ["488", "561"])
    w.ready.connect(lambda r, ch, data, bbox: got.append((ch, data)))
    w.problem.connect(problems.append)         # or a failure reads as a silent empty list
    w.run()                                    # synchronous; no thread, no event loop

    assert problems == [], f"the worker reported: {problems}"
    assert [ch for ch, _ in got] == ["488", "561"]
    for ch, data in got:
        assert isinstance(data, list), f"{ch}: napari's multiscale contract is a LIST of levels"
        assert len(data) > 1, f"{ch}: a 256x4096 mosaic has room for a pyramid; got one level"
        for above, below in zip(data, data[1:]):
            assert below.shape[-2] < above.shape[-2] and below.shape[-1] < above.shape[-1]
        assert all(lv.shape[0] == 4 for lv in data), "every level keeps the z axis"


def test_the_mosaic_worker_builds_the_pyramid_without_reading_anything(qapp):
    """Opening a region must not cost a fuse. Four channels x 10 z x 54.9 MB is 2.2 GB."""
    reads = []

    class _Counting(_PyrReader):
        def read(self, *a, **kw):
            reads.append(a)
            return super().read(*a, **kw)

    problems = []
    w = V._MosaicWorker(_Counting(), _pyr_meta(), "A1", ["488", "561"])
    w.ready.connect(lambda *a: None)
    w.problem.connect(problems.append)
    w.run()
    assert problems == [], f"the worker reported: {problems}"
    assert reads == [], f"building the pyramids read {len(reads)} frames; it must read none"


def test_on_mosaic_plane_tells_napari_the_data_is_multiscale(qapp):
    """A pyramid passed WITHOUT ``multiscale=True`` is just a list napari cannot use — it would
    either error or take level 0 and render exactly as slowly as before."""
    calls = []

    class _Mosaic:
        def add_mosaic(self, op, channel, data, **kw):
            calls.append((op, channel, data, kw))

    class _Pane:
        ok = True
        mosaic = _Mosaic()

        def say(self, msg):
            pass

    win = V.PlateWindow.__new__(V.PlateWindow)
    win._mosaic_pane = _Pane()
    # _mosaic_region is a read-only PROPERTY over the cursor now (one owner, no second copy to
    # drift), so the region is set by moving the cursor -- which is what production does too.
    from squidmip._region_nav import RegionCursor
    win._cursor = RegionCursor()
    win._cursor.set_order(["A1"])
    win._cursor.activate("A1")
    win._meta = _pyr_meta()

    levels = [np.zeros((4, 64, 48), "uint16"), np.zeros((4, 32, 24), "uint16")]
    V.PlateWindow._on_mosaic_plane(win, "raw", "A1", "488", levels, (0.0, 0.0, 10.0, 8.0))

    assert len(calls) == 1
    op, ch, data, kw = calls[0]
    assert kw.get("multiscale") is True, "napari must be told the data is a pyramid"
    assert data is levels
    # napari OWNS contrast: still no contrast_limits, pyramid or not.
    assert "contrast_limits" not in kw
    # the z scale commit 19cd491 established must survive the pyramid
    assert kw.get("z_scale_um") == 1.5
# ---------------------------------------------- Defect 4: ONE contract across the two registries
#
# _OPERATIONS (the card table) and runnable_operators() (the engine registry) are two lists that
# launch the same operators, and the comment at the exploration tab already recorded them
# diverging in production. They are not the same SET on purpose -- a card is presentation, an
# engine entry is capability -- but "not the same set on purpose" was written in a comment and
# enforced nowhere, so a card whose key is not runnable produced a dead button and said nothing.
#
# These pin the contract instead of restating it in prose.


def test_every_card_declares_whether_it_is_a_runnable_operator():
    """A card key that is not in the engine registry must be DECLARED non-runnable.

    `minerva` is the honest case: an export hand-off, not an operator - handing its key to the
    engine dies with a raw KeyError. That is fine, but it has to be said out loud, because the
    same shape produced by a typo'd key is a button that silently does nothing.
    """
    runnable = set(V.runnable_operators())
    for op in V._OPERATIONS:
        if op.runnable:
            assert op.key in runnable, (
                f"card {op.key!r} declares runnable=True but the engine cannot run it "
                f"(engine has: {sorted(runnable)}). Either it is not an operator - set "
                "runnable=False - or its key is a typo."
            )
        else:
            assert op.key not in runnable, (
                f"card {op.key!r} declares runnable=False but the engine CAN run it; "
                "the declaration is stale."
            )


def test_the_save_button_names_its_operator_instead_of_taking_the_first_card():
    """`_OPERATIONS[0].key` made 'Save this subset to disk' mean whatever happened to be first.

    Reordering the card table - a presentation edit - would then silently change which
    operator the save button RUNS. The operator is now named.
    """
    assert V._SAVE_OPERATOR == "mip"
    assert V._SAVE_OPERATOR in V.runnable_operators()
    # and it must not be a positional accident: reordering the cards must not change it
    assert V._SAVE_OPERATOR in V._OPERATIONS_BY_KEY


def test_operator_label_falls_back_to_the_key_for_a_cardless_operator():
    # `reference` is a registered projector with no card. It must still name itself rather
    # than raising a bare KeyError out of the event loop.
    assert V.operator_label("reference") == "reference"
    assert V.operator_label("mip") == V._OPERATIONS_BY_KEY["mip"].label




# -------------------------------- the mosaic worker's signal must reach the slot it is wired to
#
# This bit TWICE in one day and the suite never saw it, because every other test calls
# `_on_mosaic_plane` DIRECTLY. `_MosaicWorker.ready` lost an argument, the lambda in
# `_load_mosaic` kept five parameters, and nothing failed until the GUI ran -- PyQt raises
# inside emit(), the region never loads, and the only symptom is a black pane. A test that
# bypasses the CONNECTION cannot see a connection that is wrong, so this one goes through it.

def test_the_mosaic_workers_signal_actually_reaches_on_mosaic_plane(qapp, monkeypatch):
    """Drive the real `_load_mosaic` wiring, then emit the real signal down it.

    MUTATION: give the lambda in `_load_mosaic` a parameter the signal does not emit (which is
    exactly what the pyramid merge left behind) and this stops passing -- verified. It goes
    down as an ABORT rather than an assertion, because that is literally what PyQt does with an
    exception raised inside emit(); the point is that it is no longer green.
    """
    landed = []

    class _Mosaic:
        model = None            # no napari model here; `_napari_dims` reads through it

        def add_mosaic(self, *a, **kw):
            pass

        def remove_op(self, op):
            return []

    class _Pane:
        ok = True
        mosaic = _Mosaic()

        def say(self, msg):
            pass

    # _MosaicWorker is constructed with `parent=self`, and a QObject parent must be a live
    # QObject -- which the __new__ shell is not. Build a REAL window and this test passes, but
    # it also leaves a napari/GL window behind that segfaults a later test in the same process.
    # So: keep the shell, and drop only the Qt parentage. Everything under test -- the signal,
    # the lambda, the slot -- is untouched by that.
    class _NoParentWorker(V._MosaicWorker):
        def __init__(self, *a, parent=None, **k):
            super().__init__(*a, parent=None, **k)

    monkeypatch.setattr(V, "_MosaicWorker", _NoParentWorker)

    win = V.PlateWindow.__new__(V.PlateWindow)
    win._mosaic_pane = _Pane()
    win._reader = _PyrReader()
    win._meta = _pyr_meta()
    win._mosaic_worker = None
    win._pending_dims_step = None
    from squidmip._region_nav import RegionCursor
    win._cursor = RegionCursor()
    win._cursor.set_order(["A1"])
    win._cursor.activate("A1")
    monkeypatch.setattr(V.PlateWindow, "_napari_z_axis", lambda self: None)
    monkeypatch.setattr(V.PlateWindow, "_on_mosaic_plane",
                        lambda self, *a: landed.append(a))
    monkeypatch.setattr(V._MosaicWorker, "start", lambda self: None)   # no thread; wiring only

    V.PlateWindow._load_mosaic(win, region="A1")
    worker = win._mosaic_worker
    assert worker is not None, "_load_mosaic built no worker"

    # Emit exactly what the worker emits in `run()`. A lambda that does not match this
    # raises inside PyQt's emit and the mosaic silently never arrives.
    levels = [np.zeros((4, 8, 8), "uint16")]
    worker.ready.emit("A1", "488", levels, (0.0, 0.0, 8.0, 8.0))

    assert landed, "the ready signal never reached _on_mosaic_plane"
    op, region, channel, got_levels, bbox = landed[0]
    assert (op, region, channel) == ("raw", "A1", "488")
    assert got_levels is levels


def test_the_plate_adopts_napari_s_window_the_moment_a_region_lands(qapp, monkeypatch):
    """Julio, with a screenshot: "Look at contrast difference between napari window and plate view."

    The event sink was not broken. `on_user_contrast` reports a USER gesture and deliberately
    filters napari's own autoscale -- otherwise every channel latches MANUAL before anyone touches
    anything. But that filter also swallows the window napari picks when a region is FIRST shown,
    so the plate painted from its running percentile histogram, napari from its autoscale, and the
    panes disagreed from frame one until the user happened to drag a slider.

    An event tells you about a CHANGE; the initial state is not a change. So the plate pulls.

    MUTATION: drop the `_adopt_centre_view()` call in `_on_mosaic_done` and this goes red.
    """
    class _Mosaic:
        def show_op(self, op):
            pass

        model = type("M", (), {"reset_view": lambda self: None})()

        def contrast(self, ch):
            return {"488": (11.0, 222.0), "561": (33.0, 444.0)}.get(ch)

        def channel_rgb(self, ch):
            return {"488": (0.0, 1.0, 0.0), "561": (1.0, 1.0, 0.0)}.get(ch)

        def channel_visible(self, ch):
            return True

    class _Pane:
        ok = True
        mosaic = _Mosaic()

        def say(self, msg):
            pass

    followed, tinted = [], []

    class _Overview:
        _labels = ["488", "561"]

        def follow_channel_window(self, ch, lo, hi):
            followed.append((ch, lo, hi))

        def set_channel_color(self, ch, rgb):
            tinted.append((ch, tuple(rgb)))

        def set_channel_visible(self, ch, on):
            pass

    win = V.PlateWindow.__new__(V.PlateWindow)
    win._mosaic_pane = _Pane()
    win._overview = _Overview()
    win._meta = {"channels": [{"name": "488"}, {"name": "561"}]}
    monkeypatch.setattr(V.PlateWindow, "_restore_dims_step", lambda self: None)
    monkeypatch.setattr(V.PlateWindow, "_bind_napari_contrast", lambda self: None)
    monkeypatch.setattr(V.PlateWindow, "_region_frame_done", lambda self: None)

    V.PlateWindow._on_mosaic_done(win, "raw", "A1", 2)

    assert followed == [(0, 11.0, 222.0), (1, 33.0, 444.0)], (
        f"the plate did not adopt napari's windows on arrival: {followed}")
    assert tinted == [(0, (0.0, 1.0, 0.0)), (1, (1.0, 1.0, 0.0))], (
        f"the plate did not adopt napari's colours on arrival: {tinted}")


def test_closing_a_tab_restores_the_plate_even_while_the_raw_preview_streams(qapp, stub_detail,
                                                                             squid_dataset):
    """THE root cause of a ~50% flake, pinned as behaviour rather than as timing.

    Three gates asked `self._busy()` — "is ANY producer thread alive" — when the question they
    needed was "is an OPERATOR RUN alive". `_busy()` counts the raw plate preview, which is
    streaming almost all the time on a real plate. So closing an exploration tab deferred the
    restore, and the only thing that ever delivers a deferred restore is a worker thread exiting.
    The viewer stayed scoped to the subset of a tab that no longer existed until some unrelated
    thread happened to finish — the plate came back as ['B3:0'] instead of ['B2:0', 'B3:0'],
    one well silently missing.

    It read as a flake because it depended on whether the preview happened to still be running.
    It is not a flake: deferring on the preview is simply wrong. `_setup_raw_detail` re-scopes
    and restarts the preview itself, so a streaming preview is never a reason to postpone.

    MUTATION: change any of the three gates back to `self._busy()` and this goes red.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B3"])
    qapp.processEvents()
    assert win._detail._fov_labels == ["B3:0"]

    # Force the condition the flake depended on: a live raw preview at the moment of the close.
    class _StillStreaming:
        IS_PREVIEW = True

        def isRunning(self):
            return True

    win._retired.append(_StillStreaming())
    assert win._busy() is True, "fixture is wrong: the window must look busy for this to bite"

    win._close_op_tab(win._explore_tabs.indexOf(win._op_tabs[key]), win._explore_tabs)
    qapp.processEvents()

    assert win._detail._fov_labels == ["B2:0", "B3:0"], (
        "the plate was not restored while the raw preview was streaming — the restore is waiting "
        "on a thread that has nothing to do with it"
    )
    assert win._active_exploration is None
    win._retired.clear()
    win.close()


# --- Defect 3: an operator result becomes a toggleable LAYER GROUP in pane 2 -------------------
#
# Julio: "what if we want to see stitched AND deconvolved AND background subed. That's why we
# need the toggles." Before this, NO operator's pixels reached pane 2's napari: every result
# went to register_array (the ndviewer slider) and that was the whole of "it is visible".
# test_every_operator_streams_live_to_plate_and_slider above pins that slider path and is
# exactly the test that made the hole look covered -- it asserts nothing about layers.

class _RecordingMosaic:
    def __init__(self):
        self.calls = []

    def add_mosaic(self, op, channel, data, **kw):
        self.calls.append((op, channel, data, kw))

    # the real MosaicLayers' group view, over what was actually added
    def ops(self):
        return sorted({c[0] for c in self.calls})

    def group(self, op):
        return [c for c in self.calls if c[0] == op]


class _RecordingPane:
    ok = True

    def __init__(self):
        self.mosaic = _RecordingMosaic()

    def say(self, msg):
        pass


def _result_win(op="bgsub", region="A1", channels=("405", "488")):
    from squidmip._region_nav import RegionCursor

    win = V.PlateWindow.__new__(V.PlateWindow)
    win._mosaic_pane = _RecordingPane()
    win._cursor = RegionCursor()
    win._cursor.set_order([region])
    win._cursor.activate(region)
    win._active_op_key = op
    win._readout = type("R", (), {"setText": lambda self, t: setattr(self, "t", t),
                                  "text": lambda self: getattr(self, "t", "")})()
    win._meta = {
        # B7 is a REAL region here, with real positions. Without it the off-screen-drop test
        # would pass for the wrong reason: an unknown region cannot complete anyway, so the
        # guard it means to pin could be deleted and the test would stay green.
        "fovs_per_region": {region: [0, 1], "B7": [0, 1]},
        "fov_positions_um": {(region, 0): (0.0, 0.0), (region, 1): (6.0, 0.0),
                             ("B7", 0): (0.0, 0.0), ("B7", 1): (6.0, 0.0)},
        "pixel_size_um": 1.0,
        "frame_shape": (8, 8),
        "dtype": "uint16",
        "channels": [{"name": c} for c in channels],
        "dz_um": 1.0,
    }
    return win


def test_a_plane_op_result_becomes_a_layer_group_one_layer_per_channel(qapp):
    """The hole this branch exists to close: bgsub produced pixels and produced no layer."""
    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.full((2, 8, 8), 7, "uint16"))
    mos = win._mosaic_pane.mosaic
    assert mos.ops() == ["bgsub"]                    # one GROUP, keyed by the operator
    assert [c[1] for c in mos.group("bgsub")] == ["405", "488"]   # one LAYER per channel


def test_the_layer_group_is_not_drawn_until_the_region_is_whole(qapp):
    """Half a region drawn as a layer is a mosaic with holes, and the user reads holes as
    something the operator did."""
    win = _result_win("bgsub")
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((2, 8, 8), "uint16"))
    assert win._mosaic_pane.mosaic.calls == []


def test_the_operator_layer_lands_in_the_raw_mosaic_s_frame(qapp):
    """bbox_um is what puts the group ON TOP of raw. Without it the toggle would jump, and
    every difference the user saw would be misregistration, not the operator."""
    from squidmip._mosaic_source import mosaic_bbox_um

    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.zeros((2, 8, 8), "uint16"))
    kw = win._mosaic_pane.mosaic.group("bgsub")[0][3]
    assert kw["bbox_um"] == mosaic_bbox_um(win._meta, "A1")


def test_two_operators_make_TWO_groups_so_both_can_be_toggled(qapp):
    """'stitched AND deconvolved AND background subed'. A second operator must ADD a group,
    not replace the first one's."""
    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.zeros((2, 8, 8), "uint16"))
    win._active_op_key = "decon"
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.zeros((2, 8, 8), "uint16"))
    assert win._mosaic_pane.mosaic.ops() == ["bgsub", "decon"]


def test_a_result_for_a_region_that_is_not_on_screen_is_dropped_not_accumulated(qapp):
    """Pane 2 shows ONE region. Holding full-res mosaics for every well of a plate run would
    be gigabytes of layers nobody can look at -- the same rule the raw path already follows."""
    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "B7", fov, np.zeros((2, 8, 8), "uint16"))
    assert win._mosaic_pane.mosaic.calls == []


def test_a_result_that_cannot_be_placed_SAYS_SO_instead_of_vanishing(qapp):
    """NO SILENT FAILURES. A channel-count mismatch used to be impossible to notice because
    nothing was ever drawn from a result in the first place."""
    win = _result_win("bgsub")
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((1, 8, 8), "uint16"))
    assert "not shown as a layer" in win._readout.text()
    assert win._mosaic_pane.mosaic.calls == []


def test_a_region_operator_s_fused_mosaic_is_added_whole_not_re_tiled(qapp):
    """stitch already returns the fused region; running it back through FOV placement would
    tile a mosaic as if it were a FOV."""
    win = _result_win("stitch")
    V.PlateWindow._on_result(win, "A1", 0, np.full((2, 20, 30), 3, "uint16"))
    layers = win._mosaic_pane.mosaic.group("stitch")
    assert len(layers) == 2
    assert layers[0][2].shape == (20, 30)


def test_no_napari_pane_means_the_ndviewer_path_still_stands(qapp):
    """A window without napari must not raise out of the result slot."""
    win = _result_win("bgsub")
    win._mosaic_pane = None
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((2, 8, 8), "uint16"))
