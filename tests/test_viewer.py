"""HCS viewer — headless (offscreen) tests.

Gates the viewer contract: pure hit-testing + fit-cell shape guard, ingest that LOADS a grey plate
without processing, the Process-well-plates operator that fills tiles + drives the hue status, the
raw-z-stack push into the embedded ndviewer on double-click (pointing at the acquisition's own
TIFFs — nothing copied), the FOV-slider -> red-box link, and second-open state reset. PyQt5 is
optional (the GUI is an extra), so this whole module skips when it isn't installed — the headless
pipeline never depends on Qt.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede the PyQt import

import time

import numpy as np
import pytest

pytest.importorskip("qtpy")
# Guard the two-Qt-bindings segfault: if PySide is already in the process (napari / pytest-qt
# autoload it), importing PyQt5 GUI widgets on top crashes. Clean CI has neither. Locally, run
# `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_viewer.py` to load only PyQt5.
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )
from qtpy.QtCore import QEvent, QPointF, Qt, Signal  # noqa: E402
from qtpy.QtGui import QImage, QMouseEvent  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QPushButton, QSlider, QSpinBox, QWidget,
)

from squidmip import _viewer as V  # noqa: E402
from squidmip._napari_view import MosaicLayers as _MosaicLayers  # noqa: E402

from .conftest import CH_IN_YAML  # noqa: E402


def _needs(pkg: str):
    """Skip when an OPTIONAL operator backend is absent, instead of failing on an empty result.

    stitch and coordinate call into tilefusion (maragall/stitcher), decon into petakit, and Minerva
    export fuses through the same region-operator seam as the stitcher. None of the three is a
    dependency, and the engine's contract is to SKIP a well whose operator cannot run rather than
    to crash -- so a missing package arrives as "produced nothing, all 1 well(s) were skipped" and
    these tests failed on an assertion about pixels instead of saying the backend was absent.

    mip, reference and bgsub keep running everywhere: bgsub falls back to the scipy rolling_ball,
    which is the default and does ship.

    Stated plainly: THESE OPERATOR PATHS ARE NOT COVERED IN CI. A skip is not a pass.
    """
    return pytest.mark.skipif(
        importlib.util.find_spec(pkg) is None,
        reason=f"{pkg} not installed: this operator path is UNTESTED here, not passing")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)  # main() won't call exec_/exit under test
    return app


def _drain_until(app, pred, timeout=60):
    t0 = time.time()
    while not pred() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    return pred()


# REMOVED alongside commit 2b8fbc5's obsolete tests: the `_close_exploration_pane` helper, which
# emptied the exploration pane through the real tab-close path. The pane itself was removed on
# 2026-08-05 (see the section below), so there is nothing left for it to empty.


def _press(x, y, button=Qt.LeftButton):
    """A synthetic left-press/release at (x, y) — the handlers only read button/pos."""
    return QMouseEvent(QEvent.MouseButtonPress, QPointF(x, y), button, button, Qt.NoModifier)


def _move(x, y, buttons=Qt.NoButton):
    return QMouseEvent(QEvent.MouseMove, QPointF(x, y), Qt.NoButton, buttons, Qt.NoModifier)


# --- pure helpers (no Qt display needed) ----------------------------------------------------

def _plate_window_shell():
    """A PlateWindow with no super().__init__(), for testing a method in isolation.

    Why a helper and not four inline `__new__` calls: on a shell like this the C++ half was never
    constructed, so ANY attribute access that falls through to Qt raises
    `RuntimeError: super-class __init__() of type PlateWindow was never called`, and
    `getattr(self, name, default)` does NOT save you: the default is only used for a clean
    AttributeError, not for that RuntimeError. So every time a production path starts reading a new
    attribute, four tests break at once with an error that names Qt rather than the missing name.

    It happened twice on 2026-07-29 (`_spot_worker`, then `_viewer_manager`). Seeding the attributes
    here means it happens once and in one place.
    """
    win = V.PlateWindow.__new__(V.PlateWindow)
    # Attributes production paths read defensively. Seeded to their "absent" values so the code
    # under test takes its absent branch instead of dying in Qt.
    win._viewer_manager = None
    win._spot_worker = None
    win._overview = None
    return win


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
    """Whatever the plate is currently showing, as an (H, W, 3) uint8 array.

    ``sizeInBytes()`` rather than ``byteCount()``: byteCount was removed in Qt6, and sizeInBytes
    exists in both bindings (Qt 5.10+), so this reads pixels under either one. This single helper
    was 10 of the 19 Qt6 failures in this file, which is why the whole-suite count of "25" was
    never 25 problems.
    """
    img = ov._active_source()
    ptr = img.bits()
    ptr.setsize(img.sizeInBytes())
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


# --- a SUBSET run must replace only its own wells (Julio, 2026-08-03) -------------------------
#
# "When I preview an operator on a window, which contains a region subset, the plate view removes
# the thumbnails for all the regions rather than only those that are being processed."
#
# The plate blitted the ACTIVE layer and nothing else, so switching to an operator layer that
# covers four wells blanked the other 1532. A layer sits OVER the base; it does not replace it.


def _cell_slices(ov, rc, inset=0, inset_frac=0.0):
    """Index tuple for cell *rc* in a GRABBED frame, inset LOGICAL px on every side.

    THE ONE PLACE THAT KNOWS ABOUT THE DEVICE PIXEL RATIO. ``grab()`` renders at the screen's
    ratio -- measured 2.0 on this laptop's panel -- while ``_cell_rect`` answers in logical px. A
    crop that mixes the two lands a quarter of the plate away, so it reads a NEIGHBOURING well and
    compares it against the one it meant. That is silent on a 1x display and under
    ``QT_QPA_PLATFORM=offscreen``, which is why it survived so long: CI never has a retina panel,
    and the whole-widget checks nearby only take a std, which a wrong quadrant barely moves.

    *inset_frac* is a share of the cell (for keeping the grid pen and the centre status dot out of
    a thumbnail sample); *inset* is a flat logical count. Both are scaled here, once.
    """
    r = ov.devicePixelRatioF()
    x, y, cw, ch = (v * r for v in ov._cell_rect(*rc))
    ix = int(cw * inset_frac) + int(round(inset * r))
    iy = int(ch * inset_frac) + int(round(inset * r))
    return (slice(int(y) + iy, int(y + ch) - iy), slice(int(x) + ix, int(x + cw) - ix))


def _grab_bgr(ov) -> np.ndarray:
    """The widget's own paint as (H, W, 3) uint8 in BYTE order (B, G, R), at DEVICE resolution.

    ``Format_RGB32`` packs 0xffRRGGBB, so little-endian bytes come out B,G,R,A. Callers that only
    compare one grab against another do not care about the order; ``_grab_rgb`` reverses it for the
    ones that name an actual QColor.
    """
    img = ov.grab().toImage().convertToFormat(QImage.Format_RGB32)
    ptr = img.bits()
    ptr.setsize(img.sizeInBytes())
    row = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return row[:, : img.width(), :3]


def _painted_cell(ov, ri, ci, w=420, h=260):
    """The INTERIOR of one cell as the user sees it: the widget's own paint, not the canvas.

    Inset by a fifth of the cell so the 3 px grid pen and the status dot (drawn at the centre,
    capped at 15 px) are both outside the sample — this must read the thumbnail and nothing else.
    """
    ov.resize(w, h)
    return _grab_bgr(ov)[_cell_slices(ov, (ri, ci), inset_frac=0.2)].copy()


def test_a_subset_layer_leaves_the_other_wells_thumbnails_on_the_plate(qapp):
    ov = _overview(qapp, n_ch=1)
    ov.add_tile(0, 0, "A1", _tile([4000]))
    ov.add_tile(0, 1, "A2", _tile([4000]))          # both wells have a raw thumbnail
    raw_a2 = _painted_cell(ov, 0, 1)
    assert raw_a2.size and raw_a2.max() > 0         # the fixture really does paint something

    ov.add_tile(0, 0, "A1", _tile([9000]), layer="mip")   # the run covered A1 only
    ov.set_active_layer("mip")

    assert ov.shown_cells() == {(0, 0), (0, 1)}, "A2 lost its thumbnail when the layer switched"
    assert ov.underlay_cells() == {(0, 1)}, "A2 is not the one showing the base through"
    assert np.array_equal(_painted_cell(ov, 0, 1), raw_a2), (
        "A2 is outside the run, so the plate must still paint A2's raw thumbnail there")


def test_the_base_stops_showing_through_a_well_the_run_reaches(qapp):
    """The other half of the rule: once the operator lands on a well, its pixels win there."""
    ov = _overview(qapp, n_ch=1)
    ov.add_tile(0, 0, "A1", _tile([4000]))
    ov.add_tile(0, 1, "A2", _tile([4000]))
    ov.set_active_layer("mip")
    assert ov.underlay_cells() == {(0, 0), (0, 1)}      # nothing computed yet: all base

    ov.add_tile(0, 1, "A2", _tile([9000]), layer="mip")
    assert ov.underlay_cells() == {(0, 0)}
    assert ov.shown_cells() == {(0, 0), (0, 1)}


def test_the_base_never_shows_through_itself(qapp):
    """``raw`` active is the historical path and must stay byte-identical: no underlay, no second
    blit, and a well with no tile is still an empty slot rather than one borrowed from elsewhere."""
    ov = _overview(qapp, n_ch=1)
    ov.add_tile(0, 0, "A1", _tile([4000]))
    assert ov.underlay_cells() == set()
    assert ov.shown_cells() == {(0, 0)}


# --- GUI behavior (offscreen; embedded viewer stubbed) --------------------------------------

def test_ingest_bad_folder_does_not_crash(qapp, tmp_path):
    win = V.PlateWindow(None)
    bad = tmp_path / "not_squid"
    bad.mkdir()
    win.ingest(str(bad))          # must NOT raise / abort
    assert "not a readable" in win._readout.text().lower() or "no squid" in win._readout.text().lower()
    win.close()


def test_ingest_loads_plate_and_previews_without_processing(qapp, squid_dataset):
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


def test_ingest_non_wellplate_region_opens_as_a_slide_carrier(qapp, tmp_path):
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


def test_run_operator_persists_via_write_plate(qapp, squid_dataset, monkeypatch, tmp_path):
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


def test_run_operator_fills_tiles_and_hue_status(qapp, squid_dataset, tmp_path):
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


def test_double_click_opens_an_independent_window_on_that_region(qapp, squid_dataset):
    """RE-POINTED by commit 2b8fbc5 (was test_double_click_pushes_raw_zstack).

    A double-click used to push the well's raw z-stack into the EMBEDDED ndviewer
    (`_detail.register_image` per z-level, then `go_to_well_fov`). There is no embedded viewer any
    more: `PlateWindow._detail` is unconditionally None, so `activate_well` takes its
    decentralized branch instead — "double-click opens ONE independent window on this region (the
    single-region case of the shift-drag gesture)".

    So what a double-click has to do now is: move the cursor (which moves the red frame) and ask
    the ViewerManager for exactly one window over exactly that region. Both are asserted; the
    z-stack push half has no destination left and is not re-asserted anywhere.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    asked = []
    win._viewer_manager.open = lambda regions, **kw: asked.append(list(regions)) or object()

    win.activate_well("B3", 0)                    # double-click B3
    assert asked == [["B3"]], f"a double-click opened {asked}, not one window on B3"
    assert win._current_well == "B3"
    assert win._overview._sel == win._fov_index["B3"]["rc"], "the red frame did not follow"

    win.activate_well("B2", 0)                    # a second region -> its own window
    assert asked == [["B3"], ["B2"]]
    assert win._current_well == "B2"
    win.close()


# REMOVED BY commit 2b8fbc5 ("Decentralize GUI"): test_fov_slider_moves_red_box.
# It drove ndviewer_light's own FOV slider (`_detail._fov_slider`) and asserted the plate's red box
# followed. The embedded viewer and its slider are gone (`_detail` is unconditionally None), and
# `_on_fov_slider` documents itself as "the FALLBACK viewer's slider; under napari the navigation
# control is `_region_slider`". The surviving contract — moving the navigation slider moves what is
# shown, and the red frame follows the current region — is asserted against the controls that still
# exist in tests/test_nav_wiring.py (test_moving_a_windows_region_slider_reloads_that_windows_mosaic
# and test_double_clicking_the_plate_opens_a_window_on_that_region_and_moves_the_red_frame).


# --- selection: marquee + click (IMA-221, rebound by commit 2b8fbc5) -------------------------
#
# Gesture matrix under test, as commit 2b8fbc5 ("Decentralize GUI") left it:
#
#   Shift+drag       -> emits `marqueeSelected` (open an INDEPENDENT window over the box) and
#                       CLEARS any lingering batch wash. It no longer leaves a selection behind:
#                       the boxed set is visible in the new window's region slider, so a
#                       persistent highlight was just the "stays selected forever" clutter.
#   Shift+Alt+drag   -> marquee, UNIONS into the batch selection (unchanged)
#   Shift+click      -> toggles one well (unchanged)
#   Cmd/Ctrl+click   -> toggles one well (Linux-file-manager add/remove)
#   plain click      -> selects ONLY that well, or clears on an empty position (a REPLACE)
#   plain drag       -> pans (unchanged)      plain double-click -> opens the well (unchanged)


def _boxed(ov):
    """Record what a Shift-drag asks to be OPENED (`marqueeSelected`), plus every selection
    emission alongside it — the two halves the rebinding split apart."""
    opened, selected = [], []
    ov.marqueeSelected.connect(lambda wells: opened.append(list(wells)))
    ov.selectionChanged.connect(lambda wells: selected.append(list(wells)))
    return opened, selected

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
    from qtpy.QtCore import QPointF
    return QPointF(V._HDR + ci * cd + cd / 2, V._COLH + ri * cd + cd / 2)


def _within(ri, ci, cd=20.0):
    """Two points INSIDE one cell, far enough apart to read as a drag (not a Shift+click)."""
    from qtpy.QtCore import QPointF
    return (QPointF(V._HDR + ci * cd + 2, V._COLH + ri * cd + 2),
            QPointF(V._HDR + ci * cd + cd - 2, V._COLH + ri * cd + cd - 2))


def _mouse(kind, pos, mods=Qt.NoModifier, buttons=Qt.LeftButton, btn=Qt.LeftButton):
    from qtpy.QtCore import QEvent
    from qtpy.QtGui import QMouseEvent
    ev = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove,
          "release": QEvent.MouseButtonRelease, "dblclick": QEvent.MouseButtonDblClick}[kind]
    return QMouseEvent(ev, pos, btn, buttons, mods)


def _drag(ov, a, b, mods):
    ov.mousePressEvent(_mouse("press", a, mods))
    ov.mouseMoveEvent(_mouse("move", b, mods))
    ov.mouseReleaseEvent(_mouse("release", b, mods, buttons=Qt.NoButton))


def test_marquee_asks_for_a_window_over_exactly_the_boxed_wells(qapp):
    """REWRITTEN for commit 2b8fbc5. A Shift-drag used to REPLACE the batch selection; it now
    emits `marqueeSelected` so the window opens an independent viewer over the box instead.

    The surviving contract is the same one the old test guarded: the payload is exactly the
    acquired wells the box covers, and a second drag reports its own box rather than accumulating.
    """
    ov = _sel_overview()
    opened, _sel = _boxed(ov)
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.ShiftModifier)          # sweep the whole 2x2
    assert opened == [["A1", "A2", "B2"]]                      # B1 never acquired -> excluded
    assert ov.selected_wells() == [], "the drag left a lingering selection wash on the plate"
    _drag(ov, *_within(0, 0), Qt.ShiftModifier)                # a fresh marquee over A1 only...
    assert opened == [["A1", "A2", "B2"], ["A1"]]              # ...its own box, not a union


def test_additive_marquee_unions(qapp):
    """Shift+Alt is still the UNION into the batch selection, and still selects rather than opens.

    Seeded with a Shift+Alt drag rather than a plain Shift drag: since 2b8fbc5 a plain Shift drag
    opens a window and leaves no selection to union into, so seeding with one would have been
    testing the union against an empty set.
    """
    ov = _sel_overview()
    opened, _sel = _boxed(ov)
    _drag(ov, *_within(0, 0), Qt.ShiftModifier | Qt.AltModifier)         # A1
    _drag(ov, *_within(1, 1), Qt.ShiftModifier | Qt.AltModifier)         # + B2
    assert ov.selected_wells() == ["A1", "B2"]
    assert opened == [], "Shift+Alt opened a window instead of unioning into the selection"


def test_shift_click_toggles_well(qapp):
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 1), Qt.ShiftModifier))
    ov.mouseReleaseEvent(_mouse("release", _pt(0, 1), Qt.ShiftModifier, buttons=Qt.NoButton))
    assert ov.selected_wells() == ["A2"]
    ov.mousePressEvent(_mouse("press", _pt(0, 1), Qt.ShiftModifier))     # click again -> off
    ov.mouseReleaseEvent(_mouse("release", _pt(0, 1), Qt.ShiftModifier, buttons=Qt.NoButton))
    assert ov.selected_wells() == []


def test_marquee_emits_once_on_release(qapp):
    """The rubber band is the live feedback; the SIGNAL fires once per gesture, on release.
    A 1536-well plate would otherwise rebuild + emit a 1536-item list per mouse-move.

    Re-pointed by 2b8fbc5 from `selectionChanged` to `marqueeSelected` — the signal a Shift-drag
    now carries. The cost argument is unchanged, and so is the once-per-gesture guarantee: a
    per-move emission here would open a window per mouse-move.
    """
    ov = _sel_overview()
    opened, seen = _boxed(ov)
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    for _ in range(5):                                          # five moves mid-drag...
        ov.mouseMoveEvent(_mouse("move", _pt(1, 1), Qt.ShiftModifier))
    assert opened == [] and seen == []                          # ...emit NOTHING
    ov.mouseReleaseEvent(_mouse("release", _pt(1, 1), Qt.ShiftModifier, buttons=Qt.NoButton))
    assert opened == [["A1", "A2", "B2"]]                       # exactly one emission
    assert seen == [], "an empty batch selection was cleared it never had"


def test_selection_excludes_empty_wells(qapp):
    ov = _sel_overview()
    _drag(ov, *_within(1, 0), Qt.ShiftModifier)                 # B1: a plate position, never acquired
    assert ov.selected_wells() == []


def test_wheel_ignored_during_marquee(qapp):
    """Zooming mid-marquee would move the plate under the drag, so the wheel is ignored."""
    from qtpy.QtCore import QPoint
    from qtpy.QtGui import QWheelEvent
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    cd_before = ov._cd
    # QPointF, not QPoint: Qt6 dropped the QPoint overload for event positions. QPointF is
    # accepted by both bindings, so this stays binding-agnostic rather than becoming a cutover.
    ov.wheelEvent(QWheelEvent(QPointF(60, 60), QPointF(60, 60), QPoint(0, 0), QPoint(0, 120),
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
    from qtpy.QtCore import QEvent, QPoint
    from qtpy.QtGui import QWheelEvent
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    assert ov._marquee is not None
    ov.leaveEvent(QEvent(QEvent.Leave))                         # grab lost; no release ever arrives
    assert ov._marquee is None
    cd_before = ov._cd
    # QPointF, not QPoint: Qt6 dropped the QPoint overload for event positions. QPointF is
    # accepted by both bindings, so this stays binding-agnostic rather than becoming a cutover.
    ov.wheelEvent(QWheelEvent(QPointF(60, 60), QPointF(60, 60), QPoint(0, 0), QPoint(0, 120),
                              Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert ov._cd != cd_before                                  # zoom works again


# --- selection regressions: the landed navigator gestures must be untouched -----------------

def test_plain_drag_still_pans(qapp):
    ov = _sel_overview()
    ox0, oy0 = ov._ox, ov._oy
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.NoModifier)              # NO Shift
    assert (ov._ox, ov._oy) != (ox0, oy0), "plain drag no longer pans"
    assert ov.selected_wells() == [], "plain drag must not select"


def test_double_click_selects_only_the_well_it_opens(qapp):
    """INVERTED by commit 2b8fbc5 (Linux-file-manager selection).

    This used to assert that opening a well selected NOTHING: selection was a Shift-only gesture,
    so a plain click had to stay inert or Qt's press->release->doubleclick ordering would toggle a
    well as a side effect of opening it. A plain left click is now a REPLACE (`mouseReleaseEvent`:
    "select ONLY this well, or clear on empty"), which is idempotent — so press+release+dblclick on
    A1 leaves exactly {A1} selected however many times it repeats, and there is no toggle to flip.

    Both halves of the original intent are still pinned: the well still OPENS, and the selection
    that results is deterministic rather than a function of how many clicks Qt delivered.
    """
    ov = _sel_overview()
    opened = []
    ov.wellActivated.connect(lambda wid, fov: opened.append((wid, fov)))
    p = _pt(0, 0)
    ov.mousePressEvent(_mouse("press", p))
    ov.mouseReleaseEvent(_mouse("release", p, buttons=Qt.NoButton))
    ov.mouseDoubleClickEvent(_mouse("dblclick", p))
    assert opened == [("A1", 0)]                                # still opens the well
    assert ov.selected_wells() == ["A1"]                        # ...and selects exactly it
    # REPLACE, not toggle: repeating the gesture must not deselect the well you just opened.
    ov.mousePressEvent(_mouse("press", p))
    ov.mouseReleaseEvent(_mouse("release", p, buttons=Qt.NoButton))
    ov.mouseDoubleClickEvent(_mouse("dblclick", p))
    assert ov.selected_wells() == ["A1"], "a second plain click toggled the well off"
    # ...and a plain click on an EMPTY plate position clears, rather than leaving a stale pick.
    q = _pt(1, 0)                                               # B1: never acquired
    ov.mousePressEvent(_mouse("press", q))
    ov.mouseReleaseEvent(_mouse("release", q, buttons=Qt.NoButton))
    assert ov.selected_wells() == []


def test_marquee_does_not_disturb_red_box(qapp):
    """_sel (the current-region red box) and the marquee stay independent.

    Re-pointed by 2b8fbc5: the drag now reports its box through `marqueeSelected` instead of
    leaving a selection, but the claim under test is the same one — boxing wells must not move
    the frame that says "this is the region you are looking at".
    """
    ov = _sel_overview()
    opened, _sel = _boxed(ov)
    ov.select(1, 1)
    _drag(ov, *_within(0, 0), Qt.ShiftModifier)
    assert ov._sel == (1, 1)                                    # red box unmoved
    assert opened == [["A1"]]


def test_clear_selection_emits_empty(qapp):
    """Seeded through Shift+Alt, the gesture that still SELECTS since 2b8fbc5 (a plain Shift-drag
    opens a window and clears the wash, so it would leave nothing here to clear)."""
    ov = _sel_overview()
    seen = []
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.ShiftModifier | Qt.AltModifier)
    assert ov.selected_wells() == ["A1", "A2", "B2"]            # there is really something to clear
    ov.selectionChanged.connect(lambda wells: seen.append(list(wells)))
    ov.clear_selection()
    assert ov.selected_wells() == [] and seen == [[]]


# --- window level: expansion to (region, fov) + run-on-selection ----------------------------

def test_selection_expands_to_region_fov_pairs(qapp, squid_dataset):
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


def test_run_operator_on_selection_only_processes_selected(qapp, squid_dataset,
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


def test_selection_clears_on_second_ingest(qapp, squid_dataset):
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


def test_detach_moves_widget_to_float_and_registry(qapp):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    fl = win._detach_tab(win._left_tabs.indexOf(w))
    assert fl is not None
    assert win._left_tabs.indexOf(w) == -1                   # gone from the bar...
    assert "stub" not in win._op_tabs and win._floating["stub"] is fl
    assert w.window() is fl                                  # ...and the SAME live widget floats
    win.close()


def test_detach_home_tab_refused(qapp):
    win = V.PlateWindow(None)
    assert win._detach_tab(0) is None                        # 'Process wells' never detaches
    assert win._left_tabs.count() >= 1 and win._left_tabs.widget(0) is not None
    win.close()


def test_open_op_tab_focuses_float_not_duplicate(qapp):
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


def test_close_float_disposes_widget(qapp):
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


def test_redock_returns_same_widget(qapp):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    win._detach_tab(win._left_tabs.indexOf(w))
    win._redock("stub")
    assert win._op_tabs["stub"] is w                         # SAME object — a live shell survives
    assert win._left_tabs.currentWidget() is w
    assert not win._floating
    assert w.shutdowns == 0                                  # re-dock never kills the shell
    win.close()


def test_main_close_with_float_open_shuts_down(qapp):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    win._detach_tab(win._left_tabs.indexOf(w))
    win.close()                                              # app exit with a float open
    assert w.shutdowns == 1                                  # drained: no leaked shell...
    assert not win._floating                                 # ...no orphan window blocking exit


def test_detached_layers_keeps_refreshing_until_dispose(qapp):
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


def test_float_survives_second_ingest(qapp, squid_dataset):
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


def test_channel_toggle_after_preview_reads_nothing(qapp, squid_dataset):
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


def test_napari_visibility_drives_the_plate_and_the_strip_only_reports_it(qapp,
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

# REMOVED BY commit 2b8fbc5 ("Decentralize GUI"), three tests from this section:
#
#   test_array_viewer_contrast_drag_repaints_the_plate
#       — it existed to prove the SIGNAL/SLOT CONNECTION `_detail.contrastChanged ->
#         PlateWindow._on_detail_contrast` was live, by emitting the real signal. There is no
#         `_detail` to emit from any more (it is unconditionally None) and nothing connects to
#         that signal, so the connection under test does not exist. The behaviour it proved
#         through that connection — the plate re-windows a channel it is TOLD about — is asserted
#         directly against the surviving sink, `PlateOverview.follow_channel_window`, by the two
#         re-pointed tests below.
#
#   test_a_fresh_plate_adopts_the_viewers_current_windows
#       — a re-ingest used to pull `self._detail.channel_windows()` so the new plate opened
#         agreeing with the picture already on screen. That pull is `if self._detail is not None`
#         guarded and can never run; there is no single central viewer left holding "the" window
#         to adopt, because each independent RegionViewer carries its own.
#
#   test_contrast_is_connected_once_not_once_per_ingest
#       — it counted duplicate slots stacked on a per-ingest `connect` to `_detail.contrastChanged`.
#         Neither the singleton viewer nor the connection exists, so there is no slot to stack.
#
# The plate's contrast PRECEDENCE rule (`_RunningContrast.resolve`: user latch > followed window >
# the auto window) is untouched and is still fully covered — by the three tests below, which are
# re-pointed onto the seam a napari window actually uses, and by
# test_the_plate_adopts_napari_s_window_the_moment_a_region_lands, which drives
# `follow_channel_window` from `_on_mosaic_done`.


def test_following_a_viewer_window_never_latches_the_plate_manual(qapp, squid_dataset):
    """THE regression this nearly shipped with.

    A viewer autoscales on its own — at open, and again on every data change — so the first
    version of this sync, which recorded each broadcast with `set_manual`, came up with EVERY
    channel latched manual before the user had touched anything. That killed the plate's running
    auto-contrast from the first frame and, because a manual latch outranks everything, made
    SCOPE_PER_REGION paint every well under one global window while the plate still drew the
    amber "wells NOT comparable" badge over the top.

    A sink records what the owner resolved. Only the user sets policy.

    RE-POINTED by 2b8fbc5 from `_detail.drag_contrast` (the deleted central ndviewer's broadcast)
    onto `PlateOverview.follow_channel_window`, which is the same sink and is what a napari
    RegionViewer's window arrives through today (see `_on_mosaic_done`).
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview
    n = len(ov._labels)

    assert not any(ov._contrast.is_manual(c) for c in range(n)), (
        "a channel was latched MANUAL on open, before any user gesture")

    ov.follow_channel_window(0, 700.0, 5000.0)      # a viewer autoscaling, or the user inside it
    qapp.processEvents()
    assert ov.channel_windows()[0] == (700.0, 5000.0), "the plate did not follow the viewer"
    assert not ov._contrast.is_manual(0), (
        "following a viewer latched the channel MANUAL — the sink wrote policy back")
    assert ov._contrast.is_followed(0), "the window was not recorded as followed either"
    win.close()


def test_a_user_latch_still_outranks_the_viewer(qapp, squid_dataset):
    """`resolve` is still ONE precedence rule, now over three inputs:

        user latch  >  the owning viewer's window  >  whatever the caller computed.

    RE-POINTED by 2b8fbc5 onto `follow_channel_window`; the precedence rule itself is unchanged.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview

    ov.follow_channel_window(0, 111.0, 9999.0)
    qapp.processEvents()
    assert ov._contrast.resolve(0, (0.0, 1.0)) == (111.0, 9999.0)   # viewer beats the auto window

    ov.set_channel_window(0, 40.0, 80.0)                            # a real user gesture
    assert ov._contrast.is_manual(0)
    assert ov._contrast.resolve(0, (0.0, 1.0)) == (40.0, 80.0), "the user lost to the viewer"

    ov.set_channel_auto(0)                                          # release the user's latch
    assert ov._contrast.resolve(0, (0.0, 1.0)) == (111.0, 9999.0), (
        "releasing the user latch did not fall back to the viewer's window")
    win.close()


def test_a_channel_the_plate_does_not_have_is_ignored_not_a_crash(qapp, squid_dataset):
    """A viewer drawing RGB mode, or re-ingesting, can report a channel index the plate lacks.

    RE-POINTED by 2b8fbc5 onto `follow_channel_window`, which carries the same range guard.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview
    n = len(ov._labels)

    ov.follow_channel_window(n + 3, 1.0, 2.0)       # out of range: must be dropped silently
    qapp.processEvents()
    ov.follow_channel_window(-1, 1.0, 2.0)
    qapp.processEvents()
    assert not ov._contrast.is_manual(0)
    assert not ov._contrast.is_followed(0), "an out-of-range channel was recorded anyway"
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


def test_channel_store_survives_an_operator_run(qapp, squid_dataset, tmp_path):
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


# --- operator tabs ----------------------------------------------------------------------------
#
# REMOVED 2026-08-05 with the EXPLORATION PANE, which is what IMA-205 built: the content-addressed
# tab key, the per-tab layer namespace (`mip@exp:…`), the tab's own viewer and slider, and the
# tests for all of it. Nothing opened one — the Shift-drag that used to had been rebound to an
# independent window by 2b8fbc5, and the pane holding them was in no layout. What is left here is
# the operator tab: a singleton per operator, in the one bar.











def test_operator_layer_key_namespaces_only_when_scoped():
    assert V.operator_layer_key("mip", None) == "mip"             # plate-wide: unchanged behavior
    assert V.operator_layer_key("mip", "exp:ab12") == "mip@exp:ab12"










def test_run_operator_rejects_empty_and_unknown_regions(qapp, squid_dataset, tmp_path):
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


# REMOVED BY commit 2b8fbc5 ("Decentralize GUI"), two tests:
#   test_subset_run_scopes_slider_and_remaps_push_index
#   test_whole_plate_run_keeps_identity_indexing
#
# Both assert the GLOBAL-plate-index -> subset-slider-position remap by reading back what landed in
# the embedded ndviewer. There is no embedded viewer and, as of 2026-08-05, no remap either: the
# whole feed it belonged to was deleted with the viewer it fed.


def test_preview_spinner_still_runs_first_n_wells(qapp, squid_dataset, monkeypatch):
    """REGRESSION for the preview_limit -> regions= collapse: the shipped spinner call site."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    seen = {}
    real = V.PlateWindow.run_operator

    def spy(self, key, out_parent=None, regions=None, save=True):
        seen["regions"] = regions
        return real(self, key, out_parent=out_parent, regions=regions, save=save)
    monkeypatch.setattr(V.PlateWindow, "run_operator", spy)

    tab = win._build_run_tab(V._OPERATIONS_BY_KEY["mip"])       # the real MIP tab
    prev = [b for b in tab.findChildren(QPushButton) if b.text() == "Preview"][0]
    spin = tab.findChildren(QSpinBox)[0]
    spin.setValue(1)
    prev.click()
    assert seen["regions"] == ["B2"], "preview must still run the FIRST N wells"
    win._stop_worker(); win.close()


def test_a_preview_that_cannot_read_names_the_failure_instead_of_freezing_the_plate(qapp):
    """The raw preview used to `except Exception: pass` over its whole run, so a bad read left the
    plate half-grey forever — indistinguishable from 'still loading' — and streamEnded never fired.
    It must now finalise what landed AND name the failure."""
    class _BoomReader:
        def read(self, *a, **k):
            raise OSError("disk gone")

    meta = {"channels": [{"name": "c0"}], "dtype": "uint16", "z_levels": [0, 1, 2],
            "fovs_per_region": {"A1": [0]}, "frame_shape": (4, 4),
            "fov_positions_um": {}, "pixel_size_um": 1.0}
    w = V._PreviewWorker(_BoomReader(), meta, {"A1": {"rc": (0, 0)}}, ["A1"])
    failures, ended = [], []
    w.failed.connect(failures.append)
    w.streamEnded.connect(lambda: ended.append(True))
    w.run()                                    # in-thread: signal delivery is synchronous here
    assert failures and "disk gone" in failures[0]   # the failure is NAMED, not swallowed
    assert ended                                     # the plate still recomposites what it has


def test_operator_tab_opened_twice_is_one_tab(qapp, squid_dataset):
    """REGRESSION: an operator tab is a SINGLETON — opening it twice focuses the one tab."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n0 = win._left_tabs.count()
    win._activate_operator("mip")
    win._activate_operator("mip")
    assert win._left_tabs.count() == n0 + 1
    win.close()








def test_home_tab_is_never_closable(qapp, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n = win._left_tabs.count()
    win._close_op_tab(0)
    assert win._left_tabs.count() == n
    win.close()


def test_busy_guard_covers_retired_workers(qapp, squid_dataset, tmp_path):
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




# REMOVED BY commit 2b8fbc5 ("Decentralize GUI"):
# test_subset_tab_registers_raw_paths_at_subset_positions. It asserted that the raw bulk-register
# path indexed the embedded ndviewer's slider by SUBSET position rather than global plate index.
# `_setup_raw_detail`, its only caller, was unreachable and was deleted on 2026-08-05 with the
# rest of the embedded viewer: there is no slider to index and no registration to observe.




def test_subset_save_is_disk_guarded(qapp, squid_dataset, monkeypatch, tmp_path):
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


def test_check_disk_scales_with_subset_size(qapp, squid_dataset, tmp_path):
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


def test_the_window_asks_the_store_whether_it_is_incomplete_and_nothing_else(qapp, tmp_path,
                                                                            monkeypatch):
    """ONE marker answers "did this write finish", and it is the writer's own.

    Until 2026-08-06 this window wrote and read a SECOND marker -- a file named `INCOMPLETE` in the
    parent `.hcs` directory, dropped best-effort from a Qt teardown slot -- while `_output` wrote
    `.squidmip-incomplete` INSIDE `plate.ome.zarr` before the first byte. The two were blind to
    each other and did not answer the same question: the store's marker means *every field this
    run owed is on disk*, the window's meant only *somebody pressed stop*.

    Measured on a `write_from_stream` that put 2 of 3 fields on disk: `_output.is_incomplete` said
    True and this window's predicate said False, so "Open a computed .hcs plate" opened a plate
    with a third of its wells missing, silently, while its own plate metadata still advertised
    them.
    """
    from squidmip._output import INCOMPLETE_MARKER

    base = tmp_path / "acq.hcs"
    zroot = base / "plate.ome.zarr"
    zroot.mkdir(parents=True)
    (zroot / "zarr.json").write_text("{}")
    (zroot / INCOMPLETE_MARKER).write_text('{"fields": 3, "fields_written": 2}')

    win = V.PlateWindow(None)
    monkeypatch.setattr(V.QFileDialog, "getExistingDirectory", lambda *a, **k: str(base))
    win._open_computed()
    assert "incomplete" in win._readout.text().lower(), (
        f"a store carrying its own {INCOMPLETE_MARKER} opened without a word; the window was "
        "reading a second marker in a different directory, which only a GUI stop ever wrote"
    )
    win.close()


def test_there_is_no_second_incomplete_marker(qapp):
    """Structural: re-introducing the window's private marker would break this.

    `_note_partial_output` and `_run_out_dir` were the whole of it. They are gone because the
    store already knew, earlier and more accurately, and a second file in a second place is how
    one surface came to disagree with the other two.
    """
    assert not hasattr(V.PlateWindow, "_note_partial_output"), (
        "the window is writing its own incomplete marker again; ask `_output.is_incomplete`"
    )
    src = Path(V.__file__).read_text()
    assert '"INCOMPLETE"' not in src and "'INCOMPLETE'" not in src, (
        "a bare INCOMPLETE filename is back in _viewer.py; the ONE name is "
        "`_output.INCOMPLETE_MARKER`"
    )


def test_completed_save_run_is_not_marked_incomplete(qapp, squid_dataset, tmp_path):
    """The other half of the invariant: a run that finishes must NOT be flagged."""
    from squidmip._output import is_incomplete

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path), regions=["B2", "B3"], save=True)
    assert _drain_until(qapp, lambda: not win._busy(), timeout=90)
    out = tmp_path / f"{win._acq_name}.hcs"
    assert not is_incomplete(out / "plate.ome.zarr"), (
        "a completed plate must not be flagged incomplete"
    )
    win.close()


def test_open_computed_names_a_well_that_cannot_read_its_own_image_id(
        qapp, squid_dataset, tmp_path, monkeypatch):
    """A well whose own group metadata is unreadable falls back to well 0's image id. On this
    uniform plate that still reads correct pixels, but on a heterogeneous plate the loupe would
    magnify the WRONG field — so the substitution must be NAMED, never silent."""
    import json

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path), regions=["B2", "B3"], save=True)
    assert _drain_until(qapp, lambda: not win._busy(), timeout=90)
    out = tmp_path / f"{win._acq_name}.hcs"

    # Corrupt the group metadata of a NON-zero well, so its own image-id lookup fails.
    zroot = out / "plate.ome.zarr"
    plate = json.loads((zroot / "zarr.json").read_text())["attributes"]["ome"]["plate"]
    wells = sorted(plate["wells"], key=lambda w: (w["rowIndex"], w["columnIndex"]))
    assert len(wells) >= 2, "need a non-zero well to corrupt"
    (zroot / wells[-1]["path"] / "zarr.json").write_text("{ not valid json")

    monkeypatch.setattr(V.QFileDialog, "getExistingDirectory", lambda *a, **k: str(out))
    win._open_computed()
    assert _drain_until(qapp, lambda: "read-only" in win._readout.text(), timeout=90)
    assert "wrong field" in win._readout.text(), win._readout.text()  # named, not silently wrong
    win._stop_worker(); win.close()




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


def test_second_ingest_resets_state(qapp, squid_dataset, tmp_path):
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
        qapp, squid_dataset, tmp_path, monkeypatch):
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
        qapp, squid_dataset, tmp_path, monkeypatch):
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

def test_mosaic_places_each_fov_at_its_own_stage_offset(qapp, squid_dataset,
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
    buf = img.constBits().asstring(img.sizeInBytes())
    a = np.frombuffer(buf, np.uint8).reshape(img.height(), img.bytesPerLine() // 3, 3)
    a = a[:, :img.width(), :]
    return a[ri * V._CELL:(ri + 1) * V._CELL, ci * V._CELL:(ci + 1) * V._CELL].astype(int)


def test_mosaic_cell_composites_real_structured_pixels(qapp, squid_dataset, tmp_path):
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
        ri, ci = win._fov_index["B2"]["rc"]
        assert (ri, ci) in tiled, "B2 never landed on the active layer"
        got = _cell_of(img, ri, ci)
        assert got.size, "the acquired cell fell outside the montage"

        # THE `> 30` DYNAMIC-RANGE NUMBER IS GONE, ON PURPOSE (commit 2b8fbc5).
        #
        # It was derived from the OLD plate auto-window, a plain (1st, 99.8th) percentile stretch.
        # The plate now windows with the stitcher's fluorescence rule instead: low end at
        # `mode + 2*bg_std` (background peak pushed to black), high end at the 99.9th percentile
        # (`_RunningContrast._auto_window`). On THIS fixture the frames are 4x4 px inside an 88px
        # cell, so ~99.9% of the montage is legitimate zero padding — which means the 99.9th
        # percentile itself lands in the padding and all but the brightest handful of pixels are
        # correctly rendered black. Measured here: the acquired cell renders max 26 / min 0 with
        # 2 non-zero pixels, against an unacquired cell that is uniformly 0. Re-deriving a
        # threshold from the rule would just be re-recording those two numbers, and they are an
        # artefact of a 4x4 fixture, not of the contract.
        #
        # So this asserts the PROPERTY the number was standing in for: signal is present, it is
        # brighter than background, and it is where the mosaic geometry says it should be. That
        # kills the same mutants the old number did (a blank montage, or a contrast window that
        # collapses the cell to black, both render `got.max() == ref.max()`) and adds one it
        # never could: signal drawn in the wrong place inside the cell.
        empty = [(r, c) for r in range(ov._nr) for c in range(ov._nc) if (r, c) not in tiled]
        assert empty, "the fixture has no unacquired cell to compare against"
        ref = _cell_of(img, *empty[-1])
        assert int(ref.max()) == int(ref.min()), (
            f"the unacquired reference cell is not uniform background ({ref.min()}..{ref.max()}); "
            "it cannot be used as the background level.")
        assert int(got.max()) > int(ref.max()), (
            f"acquired cell max {int(got.max())} does not exceed the unacquired background "
            f"{int(ref.max())}: the cell is effectively blank (tiles never composited, or "
            "contrast collapsed the window).")
        # ...and every rendered pixel above background sits INSIDE one of B2's FOV boxes, so a
        # mosaic drawn at the wrong offset cannot pass by lighting up chrome or padding.
        boxes = [ov._boxes[("B2", f)] for f in (0, 1)]
        bright = np.argwhere(got.max(axis=2) > int(ref.max()))
        assert len(bright), "no pixel above background at all"
        assert all(any(t <= y < t + h and l <= x < l + w for t, l, h, w in boxes)
                   for y, x in bright), (
            f"signal rendered outside B2's FOV boxes {boxes}: the mosaic is misplaced.")
        # ...and the mosaic must reach BOTH fields' sub-boxes, not just fov 0's. Measured on the
        # STORE (native-dtype composited pixels, before any contrast window) so that a rule change
        # like the one above can never make this half of the test unaskable.
        if ov._boxes:
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
    buf = img.constBits().asstring(img.sizeInBytes())
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


# --- IMA-205 + IMA-221: what the SHIFT GESTURE opens -------------------------------------------
#
# This was the user's verbatim sentence, end to end: "hold shift to open an 'exploration' tab with
# the selected FOV subset". IMA-221 landed the marquee; before that wiring `open_exploration_tab`
# had no UI entry point at all and was reachable only programmatically.
#
# REMOVED BY commit 2b8fbc5 ("Decentralize GUI"), three tests:
#   test_shift_drag_opens_an_exploration_tab_scoped_to_the_selected_wells
#   test_shift_drag_over_several_wells_scopes_the_tab_to_all_of_them
#   test_repeating_the_same_shift_drag_focuses_the_same_tab
# ...along with the `_shift_drag_over` helper, whose every caller was one of the tests removed
# here or in the pane-3 section below.
#
# The gesture was REBOUND, not dropped: a Shift-drag now emits `marqueeSelected`, and
# `PlateWindow._on_marquee_selected` turns that into `ViewerManager.open(ordered)` — ONE
# independent napari window with a region slider over the boxed set, instead of a tab in a
# central pane that no longer exists. So there is no exploration tab to be scoped, brought to the
# front, or content-addressed any more.
#
# The surviving contract (the drag names exactly the boxed acquired wells, once, on release, and
# that set is what gets opened) is asserted at widget level by
# `test_marquee_asks_for_a_window_over_exactly_the_boxed_wells` and
# `test_marquee_emits_once_on_release`, and end-to-end through `ViewerManager.open` by
# `test_a_real_plate_gesture_is_what_minerva_exports`.

def _freeze(ov, cd=20.0):
    """Freeze the plate view so synthetic widget coordinates hit the cells we mean (paintEvent's
    auto-fit would otherwise move the plate under the drag)."""
    ov._user_view = True
    ov._cd, ov._ox, ov._oy = cd, 0.0, 0.0
    return ov


def test_shift_click_refines_the_selection_without_opening_anything(qapp,
                                                                    squid_dataset):
    """Only the DRAG opens a window. Shift+click is the refine-one-well gesture, and opening one
    window per corrective click would bury the one the user meant."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    opened = []
    win._viewer_manager.open = lambda regions, **kw: opened.append(list(regions)) or object()
    ov = _freeze(win._overview)
    rc = win._fov_index["B3"]["rc"]
    ov.mousePressEvent(_mouse("press", _pt(*rc), Qt.ShiftModifier))
    ov.mouseReleaseEvent(_mouse("release", _pt(*rc), Qt.ShiftModifier, buttons=Qt.NoButton))
    qapp.processEvents()
    assert ov.selected_wells() == ["B3"]                       # selection still happens...
    assert opened == []                                        # ...and nothing opened
    win.close()


def test_shift_drag_over_empty_plate_opens_nothing_and_says_nothing(
        qapp, squid_dataset):
    """A miss is a miss: nothing opens, and no 'empty selection' text stomps the readout."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    before = win._readout.text()
    opened = []
    win._viewer_manager.open = lambda regions, **kw: opened.append(list(regions)) or object()
    ov = _freeze(win._overview)
    _drag(ov, _pt(0, 0), _pt(0, 0.5), Qt.ShiftModifier)        # row A: no acquired wells
    qapp.processEvents()
    assert opened == []
    assert win._readout.text() == before
    win.close()


# --- a deferred view sync is DELIVERED, not dropped -------------------------------------------
#
# IMA-205 bugs 1 and 2 had one root cause: `_on_tab_changed` returned early while `_busy()` and
# nothing re-delivered the switch when the run drained. Their own tests went with the exploration
# tabs they drove (2026-08-05); the deferral machinery is unchanged and is still gated by
# test_the_plate_is_restored_even_while_the_raw_preview_streams. `_BlockingWorker` below survives
# because two thumbnail tests use it.

class _BlockingWorker(V.QThread):
    """An _OperatorWorker stand-in that stays RUNNING until stop() (or the test) releases it."""
    tileReady = V.Signal(int, int, str, object)
    pushReady = V.Signal(int, object)
    resultReady = V.Signal(str, int, object)     # full-res result -> napari layer group
    progress = V.Signal(int, int)
    runProgress = V.Signal(object)               # the engine-unit report (squidmip._progress)
    finalReady = V.Signal(object)
    writtenReady = V.Signal(str)
    wellFailed = V.Signal(int, int)
    failed = V.Signal(str)
    finished_ok = V.Signal()
    streamEnded = V.Signal()

    def __init__(self, *a, **kw):
        super().__init__()
        import threading
        self.mosaic_boxes = {}
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




# REMOVED BY commit 2b8fbc5 ("Decentralize GUI"):
# test_double_click_never_moves_the_box_to_a_well_the_viewer_cannot_show.
# Its premise no longer exists. It asserted that while a subset exploration tab SCOPED the shared
# central slider, double-clicking a well outside that subset must refuse: leave the red box where
# it was and say "not in this tab's subset". A double-click now opens an INDEPENDENT window on the
# region (`activate_well`'s `_detail is None` branch), so there is no shared slider to be outside
# of and no well the viewer cannot show — every region is openable, always. The refusal path it
# guarded (`_slider_pos` returning None) is behind `if self._detail is not None` and cannot run.




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

    def read_crop(self, well_id, level, y0, x0, h, w, time_point=0, fov=None):
        # fov before time_point so `r[-1]` stays the TIMEPOINT for the callers that read it.
        self.reads.append((well_id, level, y0, x0, h, w, fov, int(time_point)))
        # +fov as well as +t: a crop names the FIELD and the FRAME it came from, so a test can
        # tell a read of field 1 from a read of field 0 at the same rectangle.
        f = self._field(level) + int(time_point) + 1000 * int(fov or 0)
        span = f.shape[-1]
        y0, x0, h, w = V.loupe_clamp_crop(y0, x0, h, w, span, span)   # what a real source must do
        step = V.loupe_decimation(max(h, w))
        return f[:, y0:y0 + h:step, x0:x0 + w:step]

    def coarse(self, well_id, time_point=0):
        return self._field(max(0, self.n_levels - 1)) + int(time_point)


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

def test_hold_raises_loupe_and_release_dismisses(qapp, squid_dataset):
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


def test_loupe_follows_cursor_and_coalesces_to_newest(qapp, squid_dataset):
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


def test_moving_before_the_dwell_pans_and_never_loupes(qapp, squid_dataset):
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


def test_slow_pan_stays_a_pan(qapp, squid_dataset):
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


def test_double_click_cancels_the_hold_and_still_opens_the_well(qapp, squid_dataset):
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


def test_press_off_plate_never_arms(qapp, squid_dataset):
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


def test_leaving_the_widget_dismisses_a_live_loupe(qapp, squid_dataset):
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


def test_unavailable_well_reports_instead_of_showing_other_pixels(qapp, squid_dataset):
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


def test_no_source_means_the_gesture_never_arms(qapp, squid_dataset):
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

def test_raw_layer_gets_a_loupe_source_on_ingest(qapp, squid_dataset):
    """The loupe works before ANY operator run — raw mode is where users actually are."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    assert isinstance(win._loupe_sources.get("raw"), V._RawLoupeSource)
    assert win._overview._loupe_src is win._loupe_sources["raw"]
    ok, _why = win._overview._loupe_src.available("B2")
    assert ok
    win.close()


def test_raw_source_reads_real_acquisition_pixels(qapp, squid_dataset):
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


def test_raw_source_clamps_a_negative_crop_origin(qapp, squid_dataset):
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


def test_loupe_shows_pixels_in_every_quadrant_of_a_well(qapp, squid_dataset):
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


# -- current plate state: the cell under the cursor, the timepoint, the active layer -----------
#
# Three separate ways the inset showed real pixels from somewhere the plate was not. All three
# are silent: the loupe comes up, it is full of plausible data, and nothing says it is the wrong
# data. That is why they are pinned at the WIDGET seam and not on the pure helpers -- every one
# of them lived in the wiring between a helper that took the right argument and a caller that
# never passed it.

def _freeform_overview():
    """A slide-carrier plate: two regions, each placed by its own rectangle, each letterboxed.

    ``layout`` is in GRID UNITS and ``boxes`` in ``_CELL`` px, and the two carry the SAME aspect
    ratio per region -- which is what ``_mosaic_boxes`` and ``even_carrier_layout`` produce
    together, and what ``_cell_source``'s "recoverable from the rect alone" claim rests on.
    """
    layout = {(0, 0): (0.0, 0.0, 2.0, 1.0),        # A1: a 2-FOV row, bars top and bottom
              (0, 1): (2.2, 0.0, 1.0, 2.0)}        # A2: a 2-FOV column, bars left and right
    ov = V.PlateOverview(["A"], ["1", "2"], {(0, 0): "A1", (0, 1): "A2"}, layout=layout)
    ov.resize(800, 600)
    ov.set_mosaic_boxes({("A1", 0): (22, 0, 44, 44), ("A1", 1): (22, 44, 44, 44),
                         ("A2", 0): (0, 22, 44, 44), ("A2", 1): (44, 22, 44, 44)})
    ov._fit()
    return ov


def _widget_point_of_block(ov, ri, ci, bx, by):
    """The widget pixel that ``_cell_point(ri, ci, ...)`` maps back to block point (bx, by).

    The exact inverse of ``_cell_point``, written out rather than solved numerically so a
    mistake in it cannot cancel a mistake in the thing under test."""
    rx, ry, rw, rh = ov._cell_rect(ri, ci)
    sx, sy, sw, sh = ov._cell_source(ri, ci)
    return (rx + (bx - (sx - ci * V._CELL)) / sw * rw,
            ry + (by - (sy - ri * V._CELL)) / sh * rh)


def test_the_loupe_magnifies_the_field_the_cursor_is_over(qapp):
    """A freeform holder places each cell by its own rectangle, AND each cell holds a MOSAIC of
    several fields. Two transforms, and the loupe used to get both wrong in turn.

    The first was fixed: ``_cell_fraction`` now inverts the cell's own rect and its letterbox
    instead of the uniform grid. The second was not, and is what Julio still saw ("Coordinate map
    is off, as you can see in my mouse positioning"): that fraction is across the whole MOSAIC,
    and ``_loupe_geometry`` multiplied it by ONE field's pixel span and read the region's FIRST
    field. A 2-FOV region therefore mapped its entire width onto FOV 0 -- the middle of the cell,
    which is the LEFT EDGE of FOV 1, read the middle of FOV 0.

    The invariant that says it properly: the centre of a field's own box magnifies the centre of
    THAT field."""
    ov = _freeform_overview()
    ov.set_loupe_source(_FakeLoupeSource(well_px=1000, n_levels=1))
    try:
        for (region, fov), (top, left, bh, bw) in sorted(ov._boxes.items()):
            ri, ci = next(rc for rc, r in ov._by_rc.items() if r == region)
            x, y = _widget_point_of_block(ov, ri, ci, left + bw / 2, top + bh / 2)
            geo = ov._loupe_geometry(int(round(x)), int(round(y)))
            assert geo is not None, f"{region} fov {fov}: the cursor was over a field, got nothing"
            well, got_fov, _lvl, (y0, x0, h, w), _s, _m = geo
            assert (well, got_fov) == (region, fov), (
                f"the cursor was over {region} fov {fov} and the loupe went to {well} "
                f"fov {got_fov}")
            assert (y0 + h / 2, x0 + w / 2) == pytest.approx((500, 500), abs=2), (
                f"{region} fov {fov}: the middle of the FIELD read "
                f"{(y0 + h / 2, x0 + w / 2)} of a 1000 px field, not its middle")
        # ...and the corners of the drawn rect are still the corners of the MOSAIC, not of the
        # block: the letterbox inverse the earlier fix landed is untouched by any of the above.
        rx, ry, rw, rh = ov._cell_rect(0, 0)
        assert ov._cell_fraction(0, 0, rx, ry) == pytest.approx((0.0, 0.0), abs=0.02)
        assert ov._cell_fraction(0, 0, rx + rw, ry + rh) == pytest.approx((1.0, 1.0), abs=0.02)
    finally:
        ov.set_loupe_source(None)


def test_the_loupe_and_a_double_click_resolve_the_same_field(qapp):
    """One box lookup, so the field the inset showed and the field a double-click opens cannot
    be different fields. They were two loops over ``self._boxes`` before, and only one of them
    existed."""
    ov = _freeform_overview()
    ov.set_loupe_source(_FakeLoupeSource(well_px=1000, n_levels=1))
    try:
        for (region, fov), (top, left, bh, bw) in sorted(ov._boxes.items()):
            ri, ci = next(rc for rc, r in ov._by_rc.items() if r == region)
            x, y = _widget_point_of_block(ov, ri, ci, left + bw / 2, top + bh / 2)
            c = ov._cell(int(round(x)), int(round(y)))
            assert ov._fov_at(c, _press(int(round(x)), int(round(y)))) == fov
            assert ov._loupe_geometry(int(round(x)), int(round(y)))[1] == fov
    finally:
        ov.set_loupe_source(None)


def test_the_loupe_reads_the_field_the_cursor_is_over_not_the_regions_first(qapp):
    """The read that actually reaches the source carries the field, not just the geometry.

    ``docs/plate-contract.md`` records the shape of this exact failure for the TIMEPOINT: every
    read site took one and nothing passed one, so a signature test read as correct from both
    ends. Driven through the widget for the same reason."""
    ov = _freeform_overview()
    src = _FakeLoupeSource(well_px=1000, n_levels=1)
    ov.set_loupe_source(src, np.ones((2, 3), np.float32))
    try:
        seen = {}
        for (region, fov), (top, left, bh, bw) in sorted(ov._boxes.items()):
            ri, ci = next(rc for rc, r in ov._by_rc.items() if r == region)
            x, y = _widget_point_of_block(ov, ri, ci, left + bw / 2, top + bh / 2)
            src.reads.clear()
            ov.mousePressEvent(_press(int(round(x)), int(round(y))))
            ov._arm_loupe()
            assert _drain_until(qapp, lambda: bool(src.reads)), "the loupe never issued a read"
            ov.mouseReleaseEvent(_press(int(round(x)), int(round(y))))
            seen[(region, fov)] = src.reads[-1][6]          # the fov the SOURCE was asked for
        assert seen == {k: k[1] for k in seen}, (
            f"the source was asked for the wrong fields: {seen}")
    finally:
        ov.set_loupe_source(None)


def test_the_loupe_and_the_blit_share_one_content_box(qapp):
    """ONE letterbox formula, not two. ``_cell_source`` recovers the inner box from the cell
    RECT's aspect ratio (for the blit); ``_content_box`` recovers it from ``self._boxes`` (for the
    hit test and the loupe). They are the same rectangle by construction -- both are the mosaic's
    aspect centred in the same square -- and this is the assertion that keeps them so, since the
    two live at opposite ends of the file and neither calls the other."""
    ov = _freeform_overview()
    for ri, ci in ((0, 0), (0, 1)):
        sx, sy, sw, sh = ov._cell_source(ri, ci)
        assert (sx - ci * V._CELL, sy - ri * V._CELL, sw, sh) == pytest.approx(
            ov._content_box(ri, ci), abs=0.5), f"cell {(ri, ci)}: the blit and the loupe disagree"


def _loupe_rgb(ov):
    """The inset's pixels as (h, w, 3) uint8."""
    img = ov._loupe_img
    a = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
    a = a.reshape(img.height(), img.bytesPerLine() // 3, 3)
    return a[:, :img.width(), :].copy()


def _contrast_overview(colors):
    """A bare plate with a declared channel set, so it owns a contrast model and a LUT table."""
    ov = V.PlateOverview(["A"], ["1"], {(0, 0): "A1"})
    ov.set_channels(["c0", "c1"], np.asarray(colors, np.float32))
    ov._loupe = {"well": "A1", "x": 0, "y": 0}
    ov._loupe_colors = np.ones((2, 3), np.float32)     # the STALE display_color snapshot
    return ov


def test_the_loupe_paints_with_the_plates_own_contrast(qapp):
    """Julio, with a screenshot: "loupe not contrast synched with window ... the yellow vs green."

    The plate resolves a channel's window through ``_RunningContrast`` -- the user's latch, the
    napari window's followed LUT, else the running histogram under the maragall fluorescence rule
    (background mode + 2sigma to black). The SOURCE derived a second one, ``_pct_window`` over
    the well's coarse plane, and the loupe painted with that.

    They do not merely differ on a channel carrying no signal: they disagree about whether
    anything is there. The plate's rule returns a DEGENERATE window and renders it black on
    purpose; a 1/99.8 percentile window over pure background is a tight window ON the background,
    which lifts the whole field. Two channels doing that additively is the yellow.

    So: a plate whose channels both window to black must show a BLACK inset, whatever window the
    source computed for itself."""
    ov = _contrast_overview([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for ch in (0, 1):
        ov._contrast.add(ch, np.full((32, 32), 900, np.uint16))   # flat -> degenerate -> black
        assert ov.channel_windows()[ch][1] <= ov.channel_windows()[ch][0]
    crop = np.stack([np.full((8, 8), 5000, np.uint16)] * 2)
    # ...and the source insists the field is bright: the exact disagreement, handed in.
    ov._on_loupe_crop(ov._loupe_gen, "A1", crop, [(0.0, 6000.0)] * 2, None)
    assert _loupe_rgb(ov).max() == 0, (
        "the plate windows both channels to black and the inset lit them up: the loupe is still "
        "painting with the source's own percentile window")


def test_the_loupe_paints_with_the_plates_own_colours(qapp):
    """Julio: "I change channel colormap in napari and plate view doesn't react" was fixed for the
    PLATE (``set_channel_color`` -> ``self._colors``) and not for the loupe, whose ``_loupe_colors``
    is a snapshot of the acquisition's ``display_color`` taken once in ``set_loupe_source``. Recolour
    a channel in napari and the plate moved while the inset kept the acquisition's colour."""
    ov = _contrast_overview([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for ch in (0, 1):
        ov.set_channel_window(ch, 0.0, 10000.0)        # a latch, so the WINDOW is not the variable
    crop = np.stack([np.full((8, 8), 5000, np.uint16)] * 2)
    ov._on_loupe_crop(ov._loupe_gen, "A1", crop, [(0.0, 10000.0)] * 2, None)
    rgb = _loupe_rgb(ov)
    assert rgb[..., 2].max() == 0, (
        "no plate channel is blue and the inset has blue in it: the loupe is painting with the "
        "stale display_color snapshot, not the LUT napari owns")
    assert rgb[..., 0].min() > 0 and rgb[..., 1].min() > 0
    # ...and a recolour in napari moves the inset, not just the plate.
    ov.set_channel_color(0, np.array([0.0, 0.0, 1.0], np.float32))
    ov._on_loupe_crop(ov._loupe_gen, "A1", crop, [(0.0, 10000.0)] * 2, None)
    after = _loupe_rgb(ov)
    assert after[..., 0].max() == 0 and after[..., 2].min() > 0, (
        "channel 0 was recoloured blue and the inset is still painting it red")


def test_the_loupe_reads_the_timepoint_the_plate_is_showing(qapp, squid_dataset):
    """The sources have taken a ``time_point`` since 2026-07-29 (docs/plate-contract.md pins the
    signatures) and NOTHING passed one, so every read defaulted to frame 0. Move the plate's
    timepoint and the inset went on magnifying the first frame, silently.

    Driven through the widget on purpose: a signature test cannot see a caller that never calls."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    src = _FakeLoupeSource(well_px=2084, n_levels=1)
    ov.set_loupe_source(src, np.ones((2, 3), np.float32))
    try:
        rc = sorted(ov._by_rc)[0]
        x, y = _cell_center(ov, *rc)
        ov.set_time_point(3)
        ov.mousePressEvent(_press(x, y))
        ov._arm_loupe()
        assert _drain_until(qapp, lambda: ov._loupe_img is not None)
        assert src.reads, "the loupe never issued a read"
        assert [r[-1] for r in src.reads] == [3] * len(src.reads), (
            f"the plate is showing timepoint 3 and the loupe read {[r[-1] for r in src.reads]}")
        ov.mouseReleaseEvent(_press(x, y))
        # ...and moving the plate's timepoint under a LIVE inset re-reads rather than sitting on
        # the old frame: the crop cache is keyed by rectangle, so nothing else would notice.
        ov.mousePressEvent(_press(x, y))
        ov._arm_loupe()
        assert _drain_until(qapp, lambda: ov._loupe_img is not None)
        src.reads.clear()
        ov.set_time_point(5)
        assert _drain_until(qapp, lambda: bool(src.reads))
        assert src.reads[-1][-1] == 5
        ov.mouseReleaseEvent(_press(x, y))
    finally:
        ov.set_loupe_source(None)
        win.close()


def test_switching_the_active_layer_repoints_the_loupe(qapp, squid_dataset, tmp_path):
    """The plate follows which operator layer a window is showing (``set_active_layer``). The
    loupe's source is chosen BY that layer, and four of the six call sites happened to re-point it
    afterwards while the tab-change handler did not -- so clicking onto a window's tab moved the
    plate and left the inset reading the layer the plate had stopped showing.

    Pinned at ``set_active_layer`` rather than at the tab switch because that is the one place
    ``_active`` moves, and therefore the only place that can guarantee it for all six."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    try:
        raw = win._loupe_sources["raw"]
        assert win._overview._loupe_src is raw
        other = _FakeLoupeSource()
        win._loupe_sources["mip"] = other          # registered, but the plate still shows raw
        win._overview.set_active_layer("mip")
        assert win._overview._loupe_src is other, (
            "the plate moved to the 'mip' layer and the loupe kept reading raw")
        win._overview.set_active_layer("raw")
        assert win._overview._loupe_src is raw
    finally:
        win.close()


def test_loupe_read_stays_bounded_when_the_source_has_no_pyramid(qapp, squid_dataset):
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
    _w, _f, level, (y0, x0, h, w), _s, _m = ov._loupe_geometry(x, y)
    assert level == 0 and max(h, w) > V._LOUPE_MAX_CROP          # the rect really is huge
    crop = src.read_crop("B2", level, y0, x0, h, w)
    assert max(crop.shape[-2:]) <= V._LOUPE_MAX_CROP             # ...the ARRAY never is
    ov.set_loupe_source(None)
    win.close()


def test_raw_plane_cache_is_safe_across_threads(qapp, squid_dataset):
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


def test_opening_another_plate_joins_the_previous_loupe_thread(qapp, squid_dataset):
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


def test_dragging_off_the_widget_dismisses_a_live_loupe(qapp, squid_dataset):
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


def test_preview_run_gets_no_loupe_source(qapp, squid_dataset, tmp_path):
    """An unsaved preview writes nothing, so its layer must NOT inherit a zarr source — this is
    the stale-run trap: OperationStack dedupes by key, so the layer name alone proves nothing."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    win.run_operator("mip", out_parent=str(tmp_path), save=False, regions=win._order[:1])
    _drain_until(qapp, lambda: win._overview._active == "mip")
    assert win._loupe_sources.get("mip") is None
    assert win._overview._loupe_src is None        # the gesture is off, not showing raw
    win._stop_worker(); win.close()


def test_saved_run_registers_zarr_source_and_grows_written_set(qapp, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: isinstance(win._loupe_sources.get("mip"), V._ZarrLoupeSource))
    src = win._loupe_sources["mip"]
    assert src.available("B2") == (False, "not written yet")   # nothing written at run start
    assert _drain_until(qapp, lambda: src.available("B2")[0])  # ...available once the well lands
    win._stop_worker(); win.close()


def test_switching_back_to_raw_switches_the_source(qapp, squid_dataset, tmp_path):
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


def test_computed_plate_open_wires_a_loupe_source(qapp, pyramid_dataset, tmp_path, monkeypatch):
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


def test_loupe_geometry_maps_cursor_to_the_right_well_and_crop(qapp, squid_dataset):
    """The cursor -> image-coordinate mapping: right well, crop centred where the user pointed,
    and a level chosen for the CURRENT zoom (coarse when zoomed out, level 0 when near native)."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(well_px=1024, n_levels=4), np.ones((2, 3), np.float32))

    # The SINGLE-FIELD path on purpose. This fixture's two FOVs are 0.5 mm apart with 4 px frames,
    # so `_mosaic_boxes` places each field in ONE pixel of the 88 px cell -- a cell where a single
    # screen pixel of cursor motion spans the whole field, and "centred where the user pointed" is
    # not expressible at all. Multi-field centring is pinned on `_freeform_overview`, whose boxes
    # are 44 px. Here the question is the WELL, the crop centre and the level.
    ov.set_mosaic_boxes({})
    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    well, fov, level, (y0, x0, h, w), s_loupe, mag = ov._loupe_geometry(x, y)
    assert well == ov._by_rc[rc]                     # the well actually under the cursor
    assert fov is None                               # no mosaic to name a field from
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
    _w, _f, lvl_out, _r, _s, mag_out = ov._loupe_geometry(*pt_out)
    # Zoomed in near native, it reads level 0 and stops claiming magnification.
    ov._cd = 4096.0
    ax, ay = ov._ox + V._HDR, ov._oy + V._COLH
    pt_in = (int(ax + (rc[1] + 0.5) * ov._cd), int(ay + (rc[0] + 0.5) * ov._cd))
    _w, _f, lvl_in, _r, _s, mag_in = ov._loupe_geometry(*pt_in)
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


def test_minerva_tab_builds_and_lists_projectors(qapp, squid_dataset):
    """The projector choice must be real: squid2minerva's convert.py offers --mip/--z, so a
    hardcoded projection here would be a capability regression."""
    from qtpy.QtWidgets import QComboBox
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


@_needs("tilefusion")
def test_run_minerva_export_writes_one_fused_mosaic_for_the_selected_region(
        qapp, squid_dataset, tmp_path):
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


#: How many widget px a FOV box needs before a drag inside it clears ``_CLICK_SLOP`` and is a
#: DRAG rather than a Shift-click toggle. The gesture is only available zoomed in; this is what
#: "zoomed in enough" means, in the one unit the gesture is decided in.
_GRABBABLE_PX = 60.0


def _zoom_onto(ov, qapp, region):
    """Zoom the overview until *region*'s FOV boxes are big enough to drag inside.

    Derived from the geometry, not from a guessed factor: ``squid_dataset``'s tiles are 4 px
    and land as 1x1 boxes in an 88 px cell block, so a "reasonable" 8x zoom leaves a field
    ~1 px wide and every drag inside one is a click. Returns ``(row_index, col_index)``.
    """
    (rc,), = ([rc for rc, w in ov._by_rc.items() if w == region],)
    r, c = rc
    ov._user_view = True                       # stop paintEvent re-fitting under the gesture
    fov = sorted({f for rr, f in ov._boxes if rr == region})[0]
    _x, _y, w, _h = ov._block_rect(r, c, *ov._boxes[(region, fov)])
    ov._cd *= max(1.0, _GRABBABLE_PX / max(w, 1e-9))
    qapp.processEvents()
    return r, c


def _drag_px(qapp, ov, x0, y0, x1, y1, mods):
    """Press-move-release in WIDGET px. Distinct from `_drag` above, which takes QPointF cell
    corners: a FOV box is smaller than a cell and is addressed in raw px, not in cell units."""
    for kind, x, y, buttons in (
        (QEvent.MouseButtonPress, x0, y0, Qt.LeftButton),
        (QEvent.MouseMove, (x0 + x1) / 2, (y0 + y1) / 2, Qt.LeftButton),
        (QEvent.MouseButtonRelease, x1, y1, Qt.NoButton),
    ):
        qapp.sendEvent(ov, QMouseEvent(kind, QPointF(int(x), int(y)), Qt.LeftButton, buttons, mods))
    qapp.processEvents()


def test_a_shift_alt_box_inside_a_mosaic_selects_fovs_not_the_whole_well(
        qapp, squid_dataset):
    """THE gesture that makes a FOV subset expressible, driven as a real drag.

    ``stitch_plate(regions={region: [fov, ...]})`` has always cropped — it derives the mosaic
    canvas from the positions it is handed — but no gesture could say which fields, and every
    GUI caller expanded a well to all of them before the engine saw it
    (``selected_region_fovs``, ``minerva_selection``, ``_run_scope.subset_selection``, all three).
    So "run Minerva on a subset of the acquisition" could pick WELLS and never FIELDS.

    Zoomed out the same gesture covers every field of each well it touches, ``_fovs_in``
    reports no strict subset, and the behaviour is what it always was — that half is pinned
    below, because a gesture that quietly starts cropping whole-plate runs would be worse than
    the gap it closes.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview
    ov.resize(700, 560)
    ov.show()
    qapp.processEvents()
    assert ov._boxes, "fixture has no mosaic boxes; a FOV subset is not expressible on it"

    region = "B2"
    fovs = win._meta["fovs_per_region"][region]
    assert len(fovs) > 1, "a single-FOV region cannot be subset"
    r, c = _zoom_onto(ov, qapp, region)

    # A box over the FIRST field only.
    x, y, w, h = ov._block_rect(r, c, *ov._boxes[(region, fovs[0])])
    mods = Qt.ShiftModifier | Qt.AltModifier
    _drag_px(qapp, ov, x + w * 0.2, y + h * 0.2, x + w * 0.8, y + h * 0.8, mods)

    assert ov.selected_wells() == [region]
    assert ov.fov_subsets() == {region: [fovs[0]]}
    assert win.selected_region_fovs() == [(region, fovs[0])]
    assert win.minerva_selection() == [(region, fovs[0])], (
        "the export still expands the well to every FOV — the box never reached it")
    assert f"1/{len(fovs)} FOVs" in win._selection_label.text(), (
        "a cropped well reads exactly like a whole one in the Selection bar")

    # A SECOND box completing the region is back to "the whole region", with no special case.
    x2, y2, w2, h2 = ov._block_rect(r, c, *ov._boxes[(region, fovs[-1])])
    _drag_px(qapp, ov, x2 + w2 * 0.2, y2 + h2 * 0.2, x2 + w2 * 0.8, y2 + h2 * 0.8, mods)
    assert ov.fov_subsets() == {}, "a box over every field is the whole region, not a subset"
    assert win.minerva_selection() == [(region, f) for f in fovs]

    # ZOOMED OUT, the same gesture is the whole-well union it has always been.
    ov.clear_selection()
    ov._user_view = False
    qapp.processEvents()
    cd = ov._cd
    ax, ay = ov._ox + V._HDR, ov._oy + V._COLH
    _drag_px(qapp, ov, ax + c * cd + 2, ay + r * cd + 2,
          ax + (c + 1) * cd - 2, ay + (r + 1) * cd - 2, mods)
    assert ov.selected_wells() == [region]
    assert ov.fov_subsets() == {}
    assert win.minerva_selection() == [(region, f) for f in fovs]
    win.close()


@_needs("tilefusion")
def test_a_boxed_fov_subset_exports_a_smaller_mosaic_than_the_whole_region(
        qapp, squid_dataset, tmp_path):
    """...and it lands as ONE cropped mosaic, not N files and not the whole region.

    The end of the chain the test above starts: gesture -> selection -> export. Measured on the
    5-D fixture the same way (``sim_5d_2x2_t3``, A1): 2 of 4 fields fused to (2, 256, 448)
    against (2, 449, 448) whole, 460 KB against 806 KB, filename ``..._2fov.ome.tiff``.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview
    ov.resize(700, 560)
    ov.show()
    qapp.processEvents()

    region = "B2"
    fovs = win._meta["fovs_per_region"][region]
    r, c = _zoom_onto(ov, qapp, region)
    x, y, w, h = ov._block_rect(r, c, *ov._boxes[(region, fovs[0])])
    _drag_px(qapp, ov, x + w * 0.2, y + h * 0.2, x + w * 0.8, y + h * 0.8,
          Qt.ShiftModifier | Qt.AltModifier)
    assert ov.fov_subsets() == {region: [fovs[0]]}

    win.run_minerva_export(out_dir=str(tmp_path / "crop"), launch=False)
    assert _drain_until(qapp, lambda: "✓ exported" in win._readout.text())
    crop, = list((tmp_path / "crop").glob("*.ome.tiff"))
    assert "1fov" in crop.name, f"the filename does not say it is a crop: {crop.name}"
    assert "cropped" in win._readout.text(), "the readout does not say the mosaic was cropped"

    ov.clear_selection()
    ov._selection = {(r, c)}
    win._on_selection_changed(ov.selected_wells())
    assert win.minerva_selection() == [(region, f) for f in fovs]
    win._stop_minerva()
    win.run_minerva_export(out_dir=str(tmp_path / "whole"), launch=False)
    assert _drain_until(qapp, lambda: str(tmp_path / "whole") in win._readout.text())
    whole, = list((tmp_path / "whole").glob("*.ome.tiff"))

    import tifffile
    crop_px, whole_px = tifffile.imread(str(crop)), tifffile.imread(str(whole))
    assert crop_px.shape[0] == whole_px.shape[0], "the crop lost a channel"
    assert crop_px.shape[-1] < whole_px.shape[-1], (
        f"the boxed subset was not cropped: {crop_px.shape} vs {whole_px.shape}")
    win._stop_minerva(); win.close()


def test_a_user_drag_of_the_timepoint_bar_does_not_raise(qapp,
                                                         multi_time_point_dataset):
    """``_on_time_point_changed`` called ``self._say`` — a method that does not exist on this
    window and never did. It is the FIRST statement of the only slot the bar calls, so every
    user drag raised AttributeError out of Qt's slot dispatch and nothing below it ran: the
    plate never re-read at the new timepoint. Found 2026-08-05 while proving the Minerva export
    follows the slider; the export cannot follow a bar that cannot be moved.
    """
    root, _ = multi_time_point_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._time_point_bar.count > 1, "fixture is single-timepoint; the bar would be hidden"

    win._time_point_bar.set_time_point_from_user(1)     # exactly what a drag delivers
    qapp.processEvents()

    assert win.time_point == 1
    # ...and the statements BELOW the raising line ran. That is the whole claim: an
    # AttributeError on the slot's first statement is invisible in `time_point` (the bar moved
    # itself) and shows up only as the plate never being told. The readout is NOT asserted on:
    # `_return_to_raw()` at the end of the same slot legitimately overwrites it with "raw view".
    assert win._overview._time_point == 1, (
        "the slot died before it reached the plate — the timepoint moved on the bar only")
    win.close()


@_needs("tilefusion")
def test_the_exported_timepoint_is_the_one_the_plate_is_showing(
        qapp, multi_time_point_dataset, tmp_path, monkeypatch):
    """``run_minerva_export`` took ``t: int = 0`` and BOTH GUI call sites took the default, so a
    multi-timepoint acquisition always exported frame 0 whatever the bar said — the pixels on
    screen and the pixels in the OME-TIFF were different images and nothing said so.

    Asserted on what reaches the worker AND on what lands on disk: the worker argument alone
    would pass if the export then ignored it, and the file alone would not say where the value
    came from.
    """
    root, _ = multi_time_point_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    region = win._meta["regions"][0]
    win.activate_well(region, 0)                     # the user's selection; without one the
    #                                                  export is a message, not an export
    assert win.minerva_selection(), "the fixture region never became a selection"

    seen = []
    real = V._MinervaWorker

    class Spy(real):
        def __init__(self, reader, selection, out_dir, projector, t=0, **kw):
            seen.append(t)
            super().__init__(reader, selection, out_dir, projector, t=t, **kw)

    monkeypatch.setattr(V, "_MinervaWorker", Spy)

    for t in (1, 2):
        win._time_point_bar.set_time_point_from_user(t)
        qapp.processEvents()
        out = tmp_path / f"t{t}"
        win.run_minerva_export(out_dir=str(out), launch=False)
        assert _drain_until(qapp, lambda o=out: str(o) in win._readout.text()), \
            win._readout.text()
        win._stop_minerva()
        written = [p.name for p in out.glob("*.ome.tiff")]
        assert written and f"_t{t}_" in written[0], f"t={t} wrote {written}"

    assert seen == [1, 2], f"the window's timepoint never reached the worker: {seen}"
    assert region                                    # the selection was a real region, not a stub

    import tifffile
    px1 = tifffile.imread(str(next((tmp_path / "t1").glob("*.ome.tiff"))))
    px2 = tifffile.imread(str(next((tmp_path / "t2").glob("*.ome.tiff"))))
    assert not np.array_equal(px1, px2), (
        "two timepoints exported identical pixels — the slider is not reaching the export")
    win.close()


def test_run_minerva_export_with_nothing_selected_says_so(qapp, squid_dataset, tmp_path):
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


def test_minerva_selection_reads_the_window_not_the_overview(qapp, squid_dataset):
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


def test_a_real_plate_gesture_is_what_minerva_exports(qapp, squid_dataset):
    """IMA-221 <-> IMA-228, end to end through ACTUAL gestures — no stubbed selection API.

    Both halves shipped on separate branches and nothing joined them: IMA-221's per-FOV payload
    landed as ``PlateWindow.selected_region_fovs`` (the overview is display-only), so a
    ``minerva_selection`` that probed only the overview would silently skip the real API and
    reach the same answer by accident via ``selected_wells``. That is still the claim under test.

    REWRITTEN for commit 2b8fbc5, which split one gesture into two. A Shift-DRAG no longer
    selects: it emits ``marqueeSelected`` and the window turns that into
    ``ViewerManager.open(...)``, an independent viewer over the box. The gesture that scopes a
    BULK operation (which is what an export is) is now the plain click / Cmd-click selection. So
    both are driven here, and both claims are pinned: the drag asks for a window and moves no
    export scope, the click moves the export scope and asks for no window.
    """
    from qtpy.QtCore import QEvent, QPoint
    from qtpy.QtGui import QMouseEvent

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview
    ov.resize(600, 480)
    ov.show()
    qapp.processEvents()

    asked = []
    win._viewer_manager.open = lambda regions, **kw: asked.append(list(regions)) or object()

    target = ov._by_rc[sorted(ov._by_rc)[0]]                  # first acquired well only: a subset
    assert len(ov._by_rc) > 1, "fixture must have >1 well or 'subset' means nothing"
    (r, c), = [rc for rc, w in ov._by_rc.items() if w == target]
    ax, ay = ov._ox + V._HDR, ov._oy + V._COLH
    cx, cy = ax + (c + 0.5) * ov._cd, ay + (r + 0.5) * ov._cd
    box = ov._cd * 0.3

    def send(kind, x, y, buttons, mods=Qt.ShiftModifier):
        # QPointF: see the QWheelEvent note above, same Qt6 removal.
        qapp.sendEvent(ov, QMouseEvent(kind, QPointF(int(x), int(y)), Qt.LeftButton,
                                       buttons, mods))

    # 1. THE SHIFT-DRAG: opens a window over exactly the boxed well, and selects nothing.
    send(QEvent.MouseButtonPress, cx - box, cy - box, Qt.LeftButton)
    send(QEvent.MouseMove, cx, cy, Qt.LeftButton)
    send(QEvent.MouseButtonRelease, cx + box, cy + box, Qt.NoButton)
    qapp.processEvents()

    assert asked == [[target]], f"the Shift-drag did not open a window over {target}: {asked}"
    assert ov.selected_wells() == [], "the Shift-drag left a batch selection behind"

    # 2. THE PLAIN CLICK: the selection gesture, and the one an export scopes to.
    asked.clear()
    send(QEvent.MouseButtonPress, cx, cy, Qt.LeftButton, Qt.NoModifier)
    send(QEvent.MouseButtonRelease, cx, cy, Qt.NoButton, Qt.NoModifier)
    qapp.processEvents()

    assert ov.selected_wells() == [target], "the plain click did not select the well"
    assert asked == [], "a plain click opened a window; only the drag and double-click do that"
    expected = [(target, f) for f in win._meta["fovs_per_region"][target]]
    assert win.selected_region_fovs() == expected             # IMA-221's payload
    assert win.minerva_selection() == expected                # ...is what IMA-228 exports

    other = ov._by_rc[sorted(ov._by_rc)[-1]]
    win.activate_well(other, 0)                               # a DIFFERENT well is the current one
    assert win._current_well == other, "the current well never actually changed"
    assert win.minerva_selection() == expected, (
        "minerva_selection fell through to the current well and ignored the plate selection")

    ov.clear_selection()
    qapp.processEvents()
    assert win.minerva_selection() == [(other, f) for f in win._meta["fovs_per_region"][other]]
    win.close()


def test_ingest_stops_a_running_minerva_export(qapp, squid_dataset, tmp_path):
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


def test_run_minerva_export_refuses_a_second_concurrent_run(qapp, squid_dataset, tmp_path):
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


def test_run_minerva_export_without_an_acquisition_is_a_message_not_a_crash(qapp):
    win = V.PlateWindow(None)
    win.run_minerva_export(launch=False)
    assert "open an acquisition" in win._readout.text()
    win.close()


def test_minerva_export_failure_surfaces_in_the_readout(qapp, squid_dataset, monkeypatch, tmp_path):
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


@_needs("tilefusion")
def test_minerva_reports_when_author_is_not_installed(qapp, squid_dataset, monkeypatch, tmp_path):
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
    assert {"tileReady", "resultReady", "streamEnded", "writtenReady", "wellFailed"} <= set(
        V._signal_names(V._OperatorWorker))


def test_retire_disconnects_every_declared_signal(qapp, squid_dataset, tmp_path):
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


def test_closing_mid_export_disconnects_the_worker(qapp, squid_dataset, tmp_path):
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


# --- IMA-237: the pane layout -------------------------------------------------------------------
#
# Julio's requirement WAS a THREE-pane app on one monitor: pane 1 = plate + the tabbed controls,
# pane 2 = the initial viewer, pane 3 = exploration.
#
# REMOVED BY commit 2b8fbc5 ("Decentralize GUI"), three tests:
#   test_outer_split_has_three_panes_and_pane3_opens_with_width
#   test_window_resize_never_grows_pane3_at_the_plate_pane_s_expense
#   test_a_real_shift_drag_fills_pane3_without_moving_a_single_divider
#
# The three-pane layout is gone. `self._split` is now the compact top ROW of the portrait deck and
# holds exactly TWO widgets (`OpenViewList` | `_right_col`, the latter a vertical splitter of
# `_left_tabs` over the log panel since the 2026-08-03 restack); there is no `_explore_col` and the
# central pane was deleted. The exploration pane was constructed but parented into no layout, and
# it was deleted outright on 2026-08-05 — which is why the geometry these three measured is no
# longer measurable rather than merely different. The third test additionally asserted that the
# Shift-drag fills pane 3, which the gesture no longer does (see the marquee section above).
#
# What survives is asserted elsewhere: that the root window builds, sizes and tears down cleanly is
# tests/test_window_lifetime.py; that a Shift-drag opens an independent window is
# test_marquee_asks_for_a_window_over_exactly_the_boxed_wells and
# test_a_real_plate_gesture_is_what_minerva_exports.





def test_the_operators_home_tab_never_detaches(qapp, squid_dataset):
    """The home-tab guard is a property of the bar, not of _detach_tab in general: index 0 is the
    Operators tab and it cannot be dragged out, while everything above it can.

    It used to prove the other half against pane 3's bar, whose index 0 WAS detachable. That bar
    went with the exploration pane; `_first_detachable` is what still states the rule."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._detach_tab(0) is None                   # 'Operators' — never detaches
    assert win._left_tabs.count() >= 1
    assert win._left_tabs.tabBar()._first_detachable == 1
    win.close()




def test_the_redock_BUTTON_works_not_just_the_method(qapp):
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

def test_the_plane_op_cards_build_and_are_preview_only(qapp, squid_dataset):
    """DRIVEN, not read: open each plane-op tab through the real _open_op_tab path and inspect
    the widgets it actually produced. The card offers Preview and NO Save/destination half. That
    began as a consequence of _validate_image accepting Z == 1 only; IMA-277 lifted that, so what
    this now pins is the CARD's shape, not a writer limit (see _operations.py).

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


def test_flatfield_card_gates_preview_on_a_profile(qapp, squid_dataset):
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


def test_loading_a_profile_installs_one_field_per_channel_not_plane_zero(qapp, squid_dataset,
                                                                        tmp_path, monkeypatch):
    """The button a user actually clicks. ``FlatfieldProfile.from_npy(path)`` defaults to plane 0,
    so "Load illumination profile" installed channel 0's gain field and every other channel of the
    plate was corrected by it — measured on the real 10x set: 99.8% of 488's pixels wrong, by up
    to 1799 counts, while its own field sat unread in the same file."""
    pytest.importorskip("tilefusion.flatfield")
    from tilefusion.flatfield import save_flatfield

    import squidmip._flatfield as FF

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    names = [c["name"] for c in win._meta["channels"]]
    ny, nx = win._meta["frame_shape"]
    fields = []
    for i in range(len(names)):                    # one GENUINELY different field per channel
        f = np.ones((ny, nx), dtype=np.float32)
        f[: ny // 2] = 1.5 + 0.75 * i
        fields.append((f / f.mean()).astype(np.float32))
    npy = tmp_path / "profile.npy"
    save_flatfield(npy, np.stack(fields), None)

    op = V._OPERATIONS_BY_KEY["flatfield"]
    win._open_op_tab(op.key, op.label, getattr(win, op.build_tab))
    tab = win._op_tabs["flatfield"]
    button = next(b for b in tab.findChildren(QPushButton) if "illumination profile" in b.text())
    monkeypatch.setattr(V.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(npy), "")))
    before = FF.active_profiles()
    FF.clear_profile()
    try:
        button.click()
        installed = FF.active_profiles()
        assert set(installed) == set(names), (
            f"loading a {len(names)}-channel profile installed {sorted(installed)}")
        for i, name in enumerate(names):
            np.testing.assert_array_equal(
                installed[name].flatfield, fields[i],
                err_msg=f"{name} got plane {'0' if i else 'n'} of the file, not its own")
        assert not np.array_equal(installed[names[0]].flatfield, installed[names[1]].flatfield), (
            "the fixture's two channels carry the same field — this test could not fail")
    finally:
        FF.set_profiles(before) if before else FF.clear_profile()
    win.close()


# --- IMA-decon-stitch-ui: the two operator INTERFACES in pane 1 --------------------------

def test_the_decon_card_is_the_iteration_qc_panel_not_a_bare_preview(qapp,
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


def test_the_stitch_card_is_the_stitcher_control_surface(qapp, squid_dataset):
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


def test_the_stitcher_panel_kwargs_reach_the_worker(qapp, squid_dataset):
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
        qapp, squid_dataset):
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


def test_panel_kwargs_reach_stitch_plate_on_the_PREVIEW_path(qapp, squid_dataset,
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


def test_panel_kwargs_reach_write_plate_on_the_SAVE_path(qapp, squid_dataset,
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


def test_a_decon_qc_result_opens_as_a_tab_beside_the_operators(qapp, squid_dataset):
    """The seam, driven: publish_qc_result must put the widget in the operator tab bar, and
    re-publishing the same title must reuse its tab.

    It went to the exploration pane's bar until 2026-08-05, and that bar spent six weeks in no
    layout: the result was tabbed and shown to nobody. `tests/test_no_orphan_windows.py` pins the
    reachability half; this pins the identity and the reuse."""
    from squidmip._op_panels import DeconQCResultView

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    view = DeconQCResultView("B2/0/c0")
    win.publish_qc_result(view, "Decon QC · B2/0/c0")
    qapp.processEvents()
    assert win._left_tabs.indexOf(view) >= 0, "the QC result did not land beside the operators"
    before = win._left_tabs.count()
    # Publishing the SAME SUBJECT again must reuse its tab. Passing a DIFFERENT widget is the
    # point: keying on anything unique-per-call (a uuid, an iteration number) would stack a new
    # tab for every iteration of the QC loop, which is the loop's whole working rhythm. Passing
    # the same object again could not detect that, because a widget already in a tab bar is
    # merely moved rather than added twice.
    win.publish_qc_result(DeconQCResultView("B2/0/c0"), "Decon QC · B2/0/c0")
    qapp.processEvents()
    assert win._left_tabs.count() == before, "a second tab was stacked for the same subject"
    assert win._left_tabs.indexOf(view) >= 0, "the original tab was replaced, not reused"
    # A DIFFERENT subject does get its own tab.
    win.publish_qc_result(DeconQCResultView("B3/0/c0"), "Decon QC · B3/0/c0")
    qapp.processEvents()
    assert win._left_tabs.count() == before + 1
    win.close()


# --- IMA-226: EVERY operator streams live to the plate ------------------------------------------
#
# RE-POINTED by commit 2b8fbc5 ("Decentralize GUI"). This section used to assert TWO destinations
# per run: the plate canvas AND the embedded ndviewer's slider (`_detail.arrays`, the
# `register_array` pushes). The embedded viewer is gone — `_detail` is unconditionally None and
# and its feed was deleted with it on 2026-08-05 — so the slider half has no destination to reach
# and is not asserted any more. The plate half is untouched, is the half IMA-226 was really about
# ("every operator the ENGINE can run must reach the plate canvas through the same
# _OperatorWorker"), and is not covered anywhere else, so these are re-pointed rather than removed.

def _run_live(qapp, win, key, regions=("B3",)):
    """Drive a real preview run to completion and return the tiles that reached the plate."""
    tiles = []
    win.run_operator(key, regions=list(regions), save=False)
    if win._worker is None:
        return None
    win._worker.tileReady.connect(lambda *a: tiles.append(a))
    t0 = time.time()
    while win._worker.isRunning() and time.time() - t0 < 90:
        qapp.processEvents(); time.sleep(0.02)
    for _ in range(25):
        qapp.processEvents(); time.sleep(0.02)
    return tiles


@pytest.mark.parametrize("key", [
    "mip",
    "reference",
    pytest.param("stitch", marks=_needs("tilefusion")),
    pytest.param("decon", marks=_needs("petakit")),
    "bgsub",
    pytest.param("coordinate", marks=_needs("tilefusion")),
])
def test_every_operator_streams_live_to_the_plate(qapp, squid_dataset, key):
    """IMA-226. Not 'MIP streams and the rest are TODO': every operator the ENGINE can run must
    reach the plate canvas through the same _OperatorWorker.

    `reference` is the one this test was written for — a registered projector with NO card, so
    run_operator's `_OPERATIONS_BY_KEY[key].label` raised a bare KeyError out of the event loop
    and it could not be run live at all. `coordinate` is the same story on the region-operator
    side. Both stream here.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    tiles = _run_live(qapp, win, key)
    assert tiles is not None, f"{key}: no worker started — {win._readout.text()!r}"
    assert tiles, f"{key}: nothing reached the PLATE — {win._readout.text()!r}"
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


def test_flatfield_streams_live_once_a_profile_is_installed(qapp, squid_dataset):
    """The last operator: flat-field cannot run without a profile, so with one installed it must
    stream exactly like the rest — and without one it must SAY it produced nothing.

    RE-POINTED by 2b8fbc5 to the plate only; see the section note above.
    """
    from squidmip import FlatfieldProfile
    from squidmip._flatfield import set_profiles
    import squidmip._flatfield as FF

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ny, nx = win._meta["frame_shape"]

    prev = FF.active_profiles()
    try:
        # NO PROFILE. This half of the test also had a stale premise, independent of 2b8fbc5, and
        # it was invisible behind the `_detail` failure above: "no profile -> every field raises ->
        # the run produces nothing and says so". `run_operator` no longer lets that happen. It
        # intercepts flat-field with no active profile and AUTO-ESTIMATES one off-thread
        # (tilefusion BaSiC) instead, precisely so the plate does not fill with red x's — the
        # symptom Julio reported. So the surviving claim is the same one, one step earlier: a
        # flat-field run without a profile must NOT start and must SAY what it is doing.
        FF.clear_profile()
        tiles = _run_live(qapp, win, "flatfield")
        assert tiles is None, "the operator ran without an illumination profile"
        assert win._worker is None, "an operator worker started without a profile"
        assert "estimating an illumination profile" in win._readout.text(), \
            f"a flat-field run with no profile said nothing: {win._readout.text()!r}"
        est = getattr(win, "_ff_est_worker", None)
        assert isinstance(est, V._FlatfieldWorker), "no estimate was actually started"
        # Cut the estimate loose before waiting: `done` would install its profile and re-enter
        # run_operator, which is a second run this test is not about (and a thread still alive at
        # teardown). _FlatfieldWorker has no stop(), so _retire cannot be used on it.
        for sig in (est.done, est.problem, est.stage):
            try:
                sig.disconnect()
            except TypeError:
                pass
        assert _drain_until(qapp, lambda: not est.isRunning(), timeout=90)
        win._ff_est_worker = None

        # ONE PER CHANNEL. The operator is specialised per channel by project_well and refuses a
        # channel it has no measured field for, so a live run needs every channel covered.
        set_profiles({c["name"]: FlatfieldProfile(np.ones((ny, nx), np.float32))
                      for c in win._meta["channels"]})
        tiles = _run_live(qapp, win, "flatfield")
        assert tiles, f"flat-field with a profile still reached no tile: {win._readout.text()!r}"
        assert win._readout.text().startswith("✓"), win._readout.text()
    finally:
        FF.set_profiles(prev) if prev else FF.clear_profile()
    win._stop_worker(); win.close()


def test_run_operator_refuses_a_non_operator_by_name(qapp, squid_dataset):
    """`minerva` is a CARD, not an operator — it is an export hand-off. Before IMA-226 the
    exploration tab built it a '(preview)' button from _OPERATIONS and clicking it handed
    'minerva' to the engine, which died with a raw KeyError printed into the status line. That tab
    is gone; the refusal it forced is not."""
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




# --- IMA-227: raw / MIP / stitched as toggleable, reorderable LAYERS ---------------------------

def _montage_px(qapp, ov):
    """The ACTIVE layer's montage pixels — cropped to the montage, never grab() of the widget.

    The widget paints labels, a 3px grid, status dots, the current-well box and the carrier
    photograph; on this fixture the plate fits to ~12 px per cell, so a whole-widget comparison
    stays 'different' (or 'identical') for reasons that have nothing to do with layers."""
    ov.recomposite(ov._active); qapp.processEvents()
    img = ov._active_source()
    a = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
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


def test_layer_toggle_gives_back_raw_mip_and_stitched(qapp, squid_dataset):
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


def test_the_base_layer_can_be_neither_disabled_nor_reordered(qapp, squid_dataset):
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


def test_layer_reorder_changes_what_the_plate_shows(qapp, squid_dataset):
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




def test_previewing_a_subset_leaves_the_wells_outside_it_showing(qapp, squid_dataset):
    """END TO END, through the real run: preview MIP on B2 alone and B3 must keep its thumbnail.

    Julio: "when I preview an operator on a window, which contains a region subset, the plate view
    removes the thumbnails for all the regions rather than only those that are being processed."
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _drain_until(qapp, lambda: len(win._overview._tiles) >= 2)
    ov = win._overview
    b2, b3 = win._fov_index["B2"]["rc"], win._fov_index["B3"]["rc"]
    assert {b2, b3} <= ov._tiles_by_layer["raw"], "the raw preview did not fill both wells"

    _run_to_completion(qapp, win, "mip", ["B2"])

    assert ov._active == "mip" and ov._tiles_by_layer["mip"] == {b2}   # the run covered B2 only
    assert b3 in ov.shown_cells(), "B3 was not in the run and lost its thumbnail anyway"
    assert ov.underlay_cells() == {b3}
    win._stop_worker(); win.close()


def test_dropping_a_layer_frees_its_store_and_composite(qapp, squid_dataset):
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


# REMOVED BY commit 2b8fbc5 ("Decentralize GUI"), two tests and the `_stitch_into_central_viewer`
# helper they shared (no other caller):
#
#   test_ima245_region_operator_reaches_the_central_viewer_as_a_mosaic
#   test_ima245_real_tissue_stitch_reaches_the_central_viewer
#
# Both measure the RECTANGLE a fused mosaic arrives in AT THE CENTRAL ARRAY VIEWER: they read
# `_detail.canvases[-1]` (what `start_acquisition` declared) and the shape recorded by
# `_detail.register_array`. There is no central array viewer — `_detail` is unconditionally None,
# `start_acquisition` was never called and the push feed was deleted on 2026-08-05 — so neither
# number exists to compare.


# REMOVED with the array-viewer feed (ndviewer_light rot, 2026-08-05):
#
#   test_ima245_every_region_operator_is_sized_as_a_region_not_a_frame
#
# It asserted that `_OperatorWorker.push_shape` sized a REGION operator's push from the mosaic
# extent and a per-FOV projector's from the frame. `push_shape`, `push_shape_for` and
# `region_mosaic_extent_px` existed only to size ndviewer_light's canvas, and every plane they
# sized was letterboxed on the worker thread and dropped at `_on_push`'s first guard. They are
# gone; there is no rectangle left to be right about. The part of IMA-245 that is about the PLATE
# CELL -- a fused mosaic landing exactly where the raw mosaic does -- is `content_box`, which
# stays and is asserted below.


# --- the plate cell is ONE rectangle, whichever producer fills it ------------------------------
#
# Julio, from the running GUI: "Thumbnails in the plateview, say raw vs stitched, are not
# registered, meaning that they are the same dimensions but the subject image isn't, for stitching
# it gets warped on my particular example."
#
# Both halves of that sentence are one cause. The raw preview letterboxes a region's mosaic into
# its 88 px cell (`_placement.cell_boxes`: scale by min(cell/mh, cell/mw), then centre). A REGION
# operator has no per-FOV sub-boxes — the fused mosaic IS the cell — so it took the box=None
# branch, which resized to EXACTLY (_CELL, _CELL). On this fixture's 456x656 mosaic that fills the
# square: stretched 1.44x vertically (the warp) and moved off the raw cell's centred band (the
# missing registration). Same cell, same size, two geometries.

def test_a_stitched_cell_lands_exactly_where_the_raw_cell_does(
        qapp, nonsquare_mosaic_dataset):
    """A region operator's cell must be the SAME rectangle the raw preview draws for that well."""
    from functools import reduce

    from squidmip import available_region_operators

    root, region, _frame_px, (mh, mw) = nonsquare_mosaic_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    # RAW: the union of the boxes `cell_boxes` puts this region's FOVs in — what the preview paints.
    raw = reduce(V._box_union,
                 [b for (r, _f), b in V._mosaic_boxes(win._meta).items() if r == region], None)
    assert raw != (0, 0, V._CELL, V._CELL), \
        "the fixture stopped being non-square: it must letterbox, or this test proves nothing"

    for op in available_region_operators():
        w = V._OperatorWorker(op, win._reader, win._meta, win._fov_index, "",
                              regions=[region], save=False, n_fovs=None)
        assert w.mosaic_boxes == {}, "a region operator has no per-FOV sub-boxes; that is the trap"
        got: list = []
        w.tileReady.connect(lambda *a: got.append(a))
        # One fused mosaic per region, the shape `stitch_plate` yields: (T, C, 1, Y, X).
        fused = np.zeros((1, len(win._meta["channels"]), 1, mh, mw), win._meta["dtype"])
        w._on_well(region, 0, fused)
        _ri, _ci, _wid, tile, box = got[0]
        assert box == raw, f"{op!r} paints its cell at {box}, the raw preview at {raw}"
        assert tile.shape[1:] == (raw[2], raw[3]), \
            f"{op!r} emitted a {tile.shape[1:]} tile for a {raw[2]}x{raw[3]} box"
        # And the warp itself, named: the cell's aspect ratio is the mosaic's, not 1:1.
        assert abs(box[3] / box[2] - mw / mh) < 0.05
    win.close()


def test_content_box_is_a_no_op_on_a_square_field(qapp):
    """The historical single-FOV path must be untouched: a square field still fills its cell.

    `content_box` replaced `_fit_cell` on the whole-cell branch, so this is the guard that the
    fix costs nothing where there was nothing wrong.
    """
    assert V.content_box((2084, 2084)) == (0, 0, V._CELL, V._CELL)
    assert V.content_box((37, 37)) == (0, 0, V._CELL, V._CELL)
    # Wider than tall -> full width, centred vertically. Never taller than the cell.
    top, left, h, w = V.content_box((100, 400))
    assert (left, w) == (0, V._CELL) and h == V._CELL // 4 and top == (V._CELL - h) // 2


# REMOVED with the array-viewer feed (ndviewer_light rot, 2026-08-05):
#
#   test_ima245_an_unshowable_push_is_counted_and_said_out_loud
#
# It asserted that a push nobody could show was COUNTED and said out loud rather than swallowed.
# That was the right answer while the destination could exist. It could not: `_detail` had been
# unconditionally None since ndviewer_light was deleted, so `_on_push` dropped EVERY push at its
# first guard and every operator run ended with a sticky "⚠ there is no array viewer in this
# window to show the result in" on its readout. The producer, the router and the counter are all
# gone; the results themselves reach napari through `resultReady`, which is a different signal and
# was never part of this.


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
    return _grab_bgr(ov)[_cell_slices(ov, rc)]


def test_ima253_real_tissue_previews_both_regions_as_mosaics_before_any_operator_runs(
        qapp, real_dataset):
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
        qapp, real_dataset, squid_dataset):
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


def test_ima253_real_tissue_regions_are_laid_out_by_geometry_in_even_non_overlapping_cells(
        qapp, real_dataset):
    """REWRITTEN for commit 2b8fbc5. The RULE changed; the two properties worth guarding did not.

    This used to assert the STAGE-PROPORTIONAL layout: manual0 spans stage y 10186..17238 and
    manual1 21113..28165 (no overlap), while their x ranges overlap heavily, so the cells had to
    stack vertically and overlap in x. `even_carrier_layout` (squidmip/_plate.py:831, wired at
    :1060) deliberately replaced that, for the reason recorded in its docstring: true relative
    size and position "stacked two tissues into a tall, tiny, uneven column and wasted the
    viewer's horizontal space". Two regions now land in an EQUAL, landscape-biased 1x2 grid.

    So the y-stacking assertion is dead by design and is not re-asserted. What was actually being
    defended, and still is:

      1. PLACEMENT FOLLOWS GEOMETRY, NOT ENUMERATION ORDER. That was the defect ("the old carrier
         put them in columns 0 and 1 by enumeration order"). `even_carrier_layout` orders its
         cells by the stage box, so manual0 (the lower stage x) must land left of manual1, and
         reversing the order the acquisition reports its regions in must change nothing.
      2. CELLS NEVER OVERLAP, and are equal. That is the whole point of the even layout, and it is
         the property the old geometry violated in the other direction ("a bunch of squares
         overlapped with each other under different regions").
    """
    from squidmip._plate import even_carrier_layout, region_stage_boxes_um

    win = V.PlateWindow(None)
    win.ingest(str(real_dataset))
    ov = win._overview
    assert ov._layout is not None, "a freeform holder must be placed by geometry"
    boxes = region_stage_boxes_um(win._meta)          # the SAME key the carrier orders cells by
    assert boxes and boxes["manual0"][0] < boxes["manual1"][0], (
        f"fixture assumption broken: manual0 is no longer the lower stage x ({boxes})")

    r0 = ov._cell_rect(*next(k for k, v in ov._by_rc.items() if v == "manual0"))
    r1 = ov._cell_rect(*next(k for k, v in ov._by_rc.items() if v == "manual1"))

    # 1. geometry, not enumeration order: the lower stage x renders further left.
    assert r1[0] > r0[0], "manual1 is further +x than manual0, as the stage records"
    # ...and the MUTATION-CHECK for it, at the layout rule itself: reversing the reported order
    # cannot move a cell, because the ordering key is the stage box and not the report order.
    fwd = even_carrier_layout(["manual0", "manual1"], order_key=boxes)
    rev = even_carrier_layout(["manual1", "manual0"], order_key=boxes)
    assert fwd == rev, "the carrier layout follows enumeration order, not stage geometry"
    assert fwd[2]["manual0"][1] < fwd[2]["manual1"][1] or fwd[2]["manual0"][0] < fwd[2]["manual1"][0]

    # 2. equal cells that do not overlap. Rectangles are (x, y, w, h).
    assert r0[2] == pytest.approx(r1[2], rel=0.02)
    assert r0[3] == pytest.approx(r1[3], rel=0.02)
    sep_x = r1[0] >= r0[0] + r0[2] or r0[0] >= r1[0] + r1[2]
    sep_y = r1[1] >= r0[1] + r0[3] or r0[1] >= r1[1] + r1[3]
    assert sep_x or sep_y, (
        f"the two tissue cells overlap: {r0} / {r1} — this is the 'squares overlapped with each "
        "other under different regions' the even carrier exists to prevent")
    win.close()


def test_ima253_shuffling_the_region_names_does_not_move_anything(qapp, real_dataset):
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


def test_ima253_the_default_paint_path_loads_no_carrier_png(qapp, squid_dataset,
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


def test_ima253_empty_slots_are_visibly_distinct_from_occupied_ones(qapp):
    """Julio said the photograph was poor at exactly this, so it is now drawn and measured."""
    from squidmip._plate import SlideCarrier

    plate = SlideCarrier.from_format("4 slide carrier", occupancy={"manual0": [0]},
                                     cell_ids=["manual0"])
    ov = V.PlateOverview(plate.row_labels, plate.col_labels, plate.occupied_map)
    ov.set_carrier(plate)
    ov.resize(600, 240)
    a = _grab_bgr(ov)
    occupied, empty = (a[_cell_slices(ov, (0, ci), inset=2)] for ci in (0, 1))
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


# THE IMA-260 SECTION IS EMPTY, and this is what was in it.
#
# Five tests went with commit 2b8fbc5 ("Decentralize GUI"): three that measured the geometry of a
# three-pane split that no longer exists, and four that drove the CONTROL WELL feature, which
# 2b8fbc5 deleted outright (signal, menu, frame, state, tab) — there is no behaviour left to
# assert and nowhere it moved to.
#
# The remaining three went with the EXPLORATION PANE on 2026-08-05:
#   test_ima260_the_empty_pane_shows_example_usage_not_a_blank_strip
#   test_ima260_empty_state_copy_meets_the_legibility_floor
#   test_the_empty_pane_does_not_name_CONTROL_WELL_on_a_slide
#       — all three read `_build_explore_empty` / `explore_empty_text()`, the pane's empty-state
#         copy. IMA-260's whole argument was that a pane visible from open must teach the gesture
#         that fills it; the gesture was rebound to an independent window, the pane was removed,
#         and the copy went with it. The legibility floor the second one defended still lives in
#         `_qtstyle.EMPTY_BODY_PX` and is applied by every panel that sets empty-state copy.






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
    w.ready.connect(lambda r, ch, data, bbox, win: got.append((ch, data)))
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


def test_the_mosaic_worker_derives_the_contrast_seed_ITSELF(qapp):
    """The contrast seed is sampled on the WORKER thread, not in the ``ready`` slot.

    `add_mosaic` does not let napari autoscale: given no window it derives one with
    `_contrast.auto_contrast`, and `_contrast.sample_plane` has to materialise a pyramid level to
    do it. Every level of a region pyramid is fused from the FOV TIFFs at its own decimation, so
    even the coarsest rung decodes every FOV of the region — which is why sampling on the UI
    thread froze the window for the length of a whole region read (measured: 128 ms on a 27-FOV
    4-channel region here, 493-604 ms on the machine it was reported from).

    Two things are asserted and they are different: that a window comes over the wire at all, and
    that it is EXACTLY the window the UI thread used to derive. The second is what makes this a
    move rather than a second contrast rule — this project's most-repeated defect shape.
    """
    from squidmip._napari_view import _auto_window_for

    meta = _pyr_meta()
    got, problems = [], []
    w = V._MosaicWorker(_PyrReader(), meta, "A1", ["488", "561"])
    w.ready.connect(lambda r, ch, data, bbox, win: got.append((ch, data, win)))
    w.problem.connect(problems.append)
    w.run()

    assert problems == [], f"the worker reported: {problems}"
    assert [ch for ch, _d, _w in got] == ["488", "561"]
    for ch, data, window in got:
        assert window == _auto_window_for(data, True), (
            f"{ch}: the worker's seed is not the window add_mosaic would have derived")


def test_the_mosaic_worker_reads_exactly_the_coarsest_level_at_one_z(qapp):
    """Opening a region costs ONE coarse fuse per channel — the seed — and nothing else.

    This test used to assert ``reads == []``: the pyramid is lazy, so building it read nothing at
    all. That was true of the worker and false of the OPEN, because the very next thing that
    happened was `add_mosaic` sampling that pyramid for a contrast window on the Qt thread. The
    read did not go away when it was invisible here; it went somewhere this suite could not see
    it. Now the worker does it, so the bound is asserted where the work is:

      * ONE decode pass per channel — 16 FOVs x 2 channels — not one per pyramid level;
      * at ONE z (`_contrast.opening_z`, the plane the viewer opens on), not the whole 4-deep
        stack, which is what "four channels x 10 z x 54.9 MB is 2.2 GB" was guarding against;
      * and level 0 is still never materialised, which is the property the pyramid exists for.

    The plane cache is process-wide and keyed on the reader's path, so a stale entry from another
    test would make this pass while reading nothing. It is cleared first, deliberately.
    """
    from squidmip import _mosaic_source as MS
    from squidmip._contrast import opening_z

    reads = []

    class _Counting(_PyrReader):
        def read(self, *a, **kw):
            reads.append(a)
            return super().read(*a, **kw)

    meta = _pyr_meta()
    n_fovs, n_channels, nz = len(meta["fovs_per_region"]["A1"]), 2, meta["n_z"]
    problems = []
    MS._PLANE_CACHE.clear()
    w = V._MosaicWorker(_Counting(), meta, "A1", ["488", "561"])
    w.ready.connect(lambda *a: None)
    w.problem.connect(problems.append)
    w.run()

    assert problems == [], f"the worker reported: {problems}"
    assert len(reads) == n_fovs * n_channels, (
        f"the seed read {len(reads)} frames; one pass over {n_fovs} FOVs per channel is "
        f"{n_fovs * n_channels}")
    # read(region, fov, channel, z, t) -> a[3] is z, a[4] is t.
    assert {r[3] for r in reads} == {opening_z(nz)}, (
        "the seed must sample the ONE z the viewer opens on, not the whole stack")
    assert {r[4] for r in reads} == {0}, "and the timepoint this worker was built for"
    # The z axis is still 4 deep and level 0 still exists — the pyramid was not flattened to
    # make the seed cheap.
    assert MS._PLANE_CACHE.nbytes > 0, "the seed's decode must be cached, not thrown away"


# --- the mosaic reaches napari as a PYRAMID, with the worker's seed ---------------------------
#
# RETARGETED 2026-08-06, from ``PlateWindow._on_mosaic_plane`` to ``RegionViewer._on_plane``.
# These two used to install a fake pane on ``PlateWindow._mosaic_pane`` and call the plate's slot;
# that attribute has been unconditionally ``None`` since 2b8fbc5, so the slot returned at its first
# line in the app and was deleted. ``_on_plane`` is the surviving implementation of the same three
# rules and had no test of its own for them. The method is called UNBOUND on a duck shell, exactly
# as the ``PlateWindow`` tests above are: what is under test is the slot, not the widget.


class _PlaneView:
    """A ``RegionViewer`` reduced to what ``_on_plane`` reads."""

    _roi_bbox = None
    open_clock = None
    window_id = 1

    def __init__(self, pane, meta, region):
        from squidmip._region_nav import RegionCursor

        self._pane = pane
        self._meta = meta
        self._cursor = RegionCursor()
        self._cursor.set_order([region])
        self._cursor.activate(region)

    def _say(self, msg):
        pass

    def on_plane(self, *a, **kw):
        from squidmip._region_viewer import RegionViewer

        return RegionViewer._on_plane(self, *a, **kw)


def test_on_plane_tells_napari_the_data_is_multiscale(qapp):
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

    view = _PlaneView(_Pane(), _pyr_meta(), "A1")

    levels = [np.zeros((4, 64, 48), "uint16"), np.zeros((4, 32, 24), "uint16")]
    view.on_plane("A1", "488", levels, (0.0, 0.0, 10.0, 8.0), (12.0, 345.0))

    assert len(calls) == 1
    _op, _ch, data, kw = calls[0]
    assert kw.get("multiscale") is True, "napari must be told the data is a pyramid"
    assert data is levels
    # THE WORKER'S CONTRAST SEED IS PASSED THROUGH, unchanged.
    #
    # `add_mosaic` treats a missing / None window as "derive one" and derives
    # `_contrast.auto_contrast` from the pixels, on the calling thread. The window is the same
    # window either way; what moved off the UI thread is which thread samples for it
    # (`_MosaicWorker.run`). What napari still owns is contrast FROM HERE ON -- this is a seed and
    # nothing recomputes it behind the user.
    assert kw.get("contrast_limits") == (12.0, 345.0)
    # the z scale commit 19cd491 established must survive the pyramid
    assert kw.get("z_scale_um") == 1.5


def test_on_plane_without_a_window_still_lets_add_mosaic_derive_one(qapp):
    """``window=None`` means "derive one", which is what a missing argument always meant.

    Guards the degrade path: `_auto_window_for` returns None for a blank or unreadable plane, and
    a None on the wire must not become a contrast window of `(None, None)` — it must land as the
    same "you decide" `add_mosaic` has always answered to.
    """
    calls = []

    class _Mosaic:
        def add_mosaic(self, op, channel, data, **kw):
            calls.append(kw)

    class _Pane:
        ok = True
        mosaic = _Mosaic()

        def say(self, msg):
            pass

    view = _PlaneView(_Pane(), _pyr_meta(), "A1")
    view.on_plane("A1", "488", [np.zeros((4, 64, 48), "uint16")], (0.0, 0.0, 10.0, 8.0), None)
    assert calls and calls[0].get("contrast_limits") is None


# ---------------------------------------------- Defect 4: ONE contract across the two registries
#
# _OPERATIONS (the card table) and runnable_operators() (the engine registry) are two lists that
# launch the same operators, and the comment on the second control surface (the exploration tab,
# removed 2026-08-05) recorded them diverging in production. They are not the same SET on purpose -- a card is presentation, an
# engine entry is capability -- but "not the same set on purpose" was written in a comment and
# enforced nowhere, so a card whose key is not runnable produced a dead button and said nothing.
#
# These pin the contract instead of restating it in prose.


def test_a_cards_runnability_is_the_engines_answer_and_cannot_go_stale():
    """`Operation.runnable` used to be a hand-written bool, and this test checked it still agreed.

    That is the shape of the defect this whole change removes: a second table restating a fact the
    first one owns, plus a test to catch it drifting. It is a property over `runnable_operators()`
    now, so the only thing left to check is that the derivation is live -- register an operator
    named after a card and the card becomes runnable with no edit to `_OPERATIONS`.
    """
    from squidmip import add_projector, plane_op

    assert V._OPERATIONS_BY_KEY["minerva"].runnable is False   # nobody registered it
    assert V._OPERATIONS_BY_KEY["mip"].runnable is True

    card = V.Operation("card_only_key", "Card only", "no engine entry", "_build_mip_tab")
    assert card.runnable is False
    add_projector("card_only_key", plane_op(lambda p: p))
    assert card.runnable is True, "runnable is stale; it must be read, not stored"


def test_gallery_view_is_a_view_menu_command_and_not_an_operator(qapp):
    """"I guess I don't understand how this can be treated as an operator in bulk" (Julio).

    He is right. An operator here is something the engine runs over regions to produce derived
    data, declared by a `consumes` frozenset; "arrange the open windows in a grid" consumes no
    axis and produces no pixels. It was never in `_OPERATIONS`, but it sat in the operator card
    stack wearing the same card, which is what made it read as one.

    It is BUILT now (2026-08-05, `squidmip/_gallery.py` + `_gallery_window.py`), and the half of
    this test that pinned the "not implemented" status line is gone with the stub -- see
    `tests/test_gallery.py` for what it does instead. What survives here is the half that was never
    about the stub: Gallery View is a View-menu command, it is not a runnable operator, and it has
    no card. The remaining assertion below is the one the stub's status line stood in for: with NO
    acquisition open the command must say so and open nothing, rather than raising or opening an
    empty grid.
    """
    win = V.PlateWindow(None)
    try:
        assert "galleryview" not in win._op_cards
        assert "galleryview" not in win._op_actions
        assert "galleryview" not in {op.key for op in V._OPERATIONS}
        assert "galleryview" not in V.runnable_operators()

        act = win._gallery_act
        assert act.menu() is not None or act.parent() is not None
        assert [a for a in win.menuBar().actions()
                if a.text() == "&View" and act in a.menu().actions()], (
            "Gallery View is not in the View menu, so it is nowhere")
        # window management is not gated on an acquisition; the operator cards are
        assert act.isEnabled() is True

        act.trigger()
        assert win._gallery is None, "Gallery View opened a window with no acquisition to tile"
        assert "open an acquisition" in win._readout.text().lower(), (
            f"Gallery View with nothing open said {win._readout.text()!r}, which does not name "
            "the missing acquisition as the reason")
    finally:
        win.close()


# The OTHER direction of the same contract, and the one nothing checked.
#
# `test_every_card_declares_whether_it_is_a_runnable_operator` walks the CARDS and asks the engine.
# An operator the engine can run but that has NO card is invisible to that walk: there is no card
# to iterate, so nothing fails. That is exactly how `reference` -- a z-reduction to the sharpest
# plane, the capability Julio asked for twice -- stayed CLI-only for months while being in
# `available_projectors()` the whole time. It was in no dropdown, no menu and no card, and no test
# said a word.
#
# So: every runnable operator must either have a card or be DECLARED CLI-only here, with the
# reason written down. Registering a projector is one line anywhere in the package
# (`add_projector`), so without this the next one lands the same way: shipped, runnable, and
# unreachable from the GUI. The allowlist lives in the test rather than in `_operations.py`
# because "this operator has no card" is not a fact about the card table -- it is a decision, and
# the point is that the decision has to be made out loud when the operator is added.

#: Runnable operators that deliberately have no GUI card, and why. Adding an operator without
#: adding it here (or giving it a card) fails the test below. Removing a card without moving its
#: key here fails it too.
CLI_ONLY_OPERATORS = {
    "spot": "a LABELS overlay, not a plate result; it is driven from the spot-count controls "
            "on the mosaic, not from a card that writes an OME-Zarr plate.",
    "cellpose": "the same LABELS overlay as `spot`, with the model instead of the Otsu recipe. "
                "Same reason it has no card. (It is NOT because the result cannot be written -- "
                "that was true while _validate_image accepted Z == 1 only, and IMA-277 lifted "
                "it.) It is "
                "reachable from the CLI (--projector cellpose), the operator dropdown and the "
                "Detect-nuclei button, all of which read the registry.",
    "decon3d": "the volume-then-project variant of `decon`; the decon card's own panel is where "
               "an iteration count gets chosen, and a second card for the same operator with a "
               "different z contract is how a user picks the wrong one.",
    "coordinate": "the unregistered CONTROL for `stitch` (stage coordinates, no registration). "
                  "It exists to be the baseline a stitch is graded against in the benchmark, "
                  "not to be offered as a thing to run.",
}


def test_every_runnable_operator_is_either_carded_or_declared_cli_only():
    """An engine entry with no card is a capability the GUI cannot reach.

    The reverse of the card->engine check above. `reference` is the case that proves it: the
    engine has run it since IMA-210 and it appeared in no GUI surface at all.
    """
    carded = {op.key for op in V._OPERATIONS}
    for key in V.runnable_operators():
        assert key in carded or key in CLI_ONLY_OPERATORS, (
            f"the engine can run {key!r} but no card offers it and it is not declared CLI-only. "
            f"Either add an Operation for it to _OPERATIONS (plus its _build_<x>_tab), or add it "
            f"to CLI_ONLY_OPERATORS with the reason it is deliberately not in the GUI."
        )


def test_the_cli_only_declaration_cannot_go_stale():
    """A key that is no longer runnable, or that has since been given a card, must be removed.

    Without this the allowlist becomes a place where names go to be forgotten, and the test above
    would keep passing over an operator that has quietly gained a card or lost its registration.
    """
    runnable = set(V.runnable_operators())
    carded = {op.key for op in V._OPERATIONS}
    for key in CLI_ONLY_OPERATORS:
        assert key in runnable, (
            f"{key!r} is declared CLI-only but the engine no longer runs it; delete the entry.")
        assert key not in carded, (
            f"{key!r} is declared CLI-only but now HAS a card; delete the entry.")


def test_the_reference_plane_operator_is_reachable_from_the_gui():
    """The defect itself, pinned: `reference` has a card, and the card is wired to a real tab."""
    op = V._OPERATIONS_BY_KEY["reference"]
    assert op.runnable is True
    assert "reference" in V.runnable_operators()
    assert hasattr(V.PlateWindow, op.build_tab), (
        f"the reference card names {op.build_tab!r} and PlateWindow has no such method; "
        "clicking it would raise AttributeError out of the event loop.")


def test_the_save_button_names_its_operator_instead_of_taking_the_first_card():
    """`_OPERATIONS[0].key` made 'Save this subset to disk' mean whatever happened to be first.

    Reordering the card table - a presentation edit - would then silently change which
    operator the save button RUNS. The operator is now named.

    The button itself went with the exploration pane (2026-08-05); the constant and this rule
    stay for the next one. See the note at `_SAVE_OPERATOR`.
    """
    assert V._SAVE_OPERATOR == "mip"
    assert V._SAVE_OPERATOR in V.runnable_operators()
    # and it must not be a positional accident: reordering the cards must not change it
    assert V._SAVE_OPERATOR in V._OPERATIONS_BY_KEY


def test_a_cardless_operator_opens_a_panel_built_from_its_declaration(qapp,
                                                                     squid_dataset):
    """`_activate_operator` used to end at `if op is not None:` and do NOTHING for a key the card
    table did not know: no tab, no error, no line in the readout. Silence was the bug.

    `spot` is the case: a registered projector with four declared parameters, no card, and
    therefore no way to reach any of them from the GUI. It now opens the generic panel, and the
    panel's widgets ARE the declaration.
    """
    from squidmip._engine import operator_params
    from squidmip._param_panel import GenericOperatorPanel

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert "spot" not in V._OPERATIONS_BY_KEY, "spot has gained a card; pick another cardless one"
    win._activate_operator("spot")
    panel = win._op_tabs.get("spot")
    assert panel is not None, f"no panel opened; readout said {win._readout.text()!r}"
    assert isinstance(panel, GenericOperatorPanel)
    assert sorted(panel.widgets) == sorted(p.name for p in operator_params("spot"))
    win.close()


def test_a_key_that_has_no_panel_at_all_is_refused_by_name_never_silently(qapp,
                                                                         squid_dataset):
    """The other half: a key with neither a card nor a readable declaration must SAY SO. A click
    that lands on nothing is indistinguishable from one that is still working."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    before = dict(win._op_tabs)
    win._activate_operator("stitch_but_misspelled")
    assert win._op_tabs == before, "a refused operator must not open a tab"
    assert "stitch_but_misspelled" in win._readout.text()
    win.close()


def test_the_preview_path_carries_operator_kwargs_to_the_engine(qapp, squid_dataset, monkeypatch):
    """MEASURED, not read: `_OperatorWorker`'s PREVIEW branch called `project_plate` WITHOUT
    `operator_kwargs` while the save branch passed them and the class's own docstring said both
    branches carried them. It was true when written -- a projector's parameters were baked in at
    registration -- and stopped being true the moment `Operator.params`/`factory` landed.

    The symptom is the worst shape there is: the panel says min_area_px=400, the console line
    (`_action_label`) says min_area_px=400, and the pixels are the ones min_area_px=30 produces.
    Verified end to end on a real acquisition (57 labels vs 44 at min_area_px 30/400); this is the
    unit that keeps it from coming back.
    """
    import squidmip
    from squidmip.reader import open_reader

    root, _ = squid_dataset
    reader = open_reader(str(root))
    meta = reader.metadata
    seen = {}

    def fake_project_plate(_reader, **kw):
        seen.update(kw)
        return iter(())

    monkeypatch.setattr(squidmip, "project_plate", fake_project_plate)
    fov_index = {r: {"rc": (0, i), "idx": i, "well_id": r}
                 for i, r in enumerate(meta["regions"])}
    worker = V._OperatorWorker("spot", reader, meta, fov_index, "", regions=meta["regions"][:1],
                               save=False, n_fovs=None,
                               operator_kwargs={"min_area_px": 400})
    worker.run()
    assert seen.get("operator_kwargs") == {"min_area_px": 400}, (
        "the preview branch dropped the panel's parameters on the floor: "
        f"project_plate was called with {sorted(seen)}")


def test_every_uncarded_runnable_operator_is_offered_in_the_declaration_submenu(qapp):
    """A card is presentation and the engine is capability, and the gap between them used to be a
    capability the GUI could not reach AT ALL. The submenu is built off `runnable_operators()`, so
    an operator added by a plugin appears without an edit here."""
    win = V.PlateWindow(None)
    offered = {a.text() for a in win._declared_menu.actions()}
    expected = {V.operator_label(k) for k in V.runnable_operators()
                if k not in V._OPERATIONS_BY_KEY}
    assert offered == expected
    assert "spot" in offered and "cellpose" in offered
    win.close()


def test_operator_label_falls_back_to_the_key_for_a_cardless_operator():
    # `spot` is a registered projector with no card. It must still name itself rather than
    # raising a bare KeyError out of the event loop. (`reference` was this example until it
    # was given a card; the fallback is what makes a cardless operator survive, so it is
    # pinned against whichever operator is currently cardless.)
    assert V.operator_label("spot") == "spot"
    assert V.operator_label("mip") == V._OPERATIONS_BY_KEY["mip"].label
    # and the newly carded one now answers with its card
    assert V.operator_label("reference") == V._OPERATIONS_BY_KEY["reference"].label




# --- these two moved to the WINDOW, which is the only thing that draws a mosaic ----------------
#
# ``test_the_mosaic_workers_signal_actually_reaches_on_mosaic_plane`` and
# ``test_the_plate_adopts_napari_s_window_the_moment_a_region_lands`` lived here. Both drove
# ``PlateWindow._load_mosaic`` / ``_on_mosaic_done`` through a hand-installed ``_mosaic_pane``,
# and that attribute has been unconditionally ``None`` since 2b8fbc5: in the app neither method
# got past its first line. Deleted with the methods on 2026-08-06.
#
# Neither concern is lost, and neither is now untested:
#
# * the SIGNAL-TO-SLOT ARITY (a lambda that does not match ``_MosaicWorker.ready`` raises inside
#   PyQt's emit and the region silently never loads) is exercised on the live path by
#   ``tests/test_time_point_playback.py``, whose ``_shape_worker_class`` emits a real
#   ``Signal(str, str, object, object, object)`` down ``RegionViewer._load_mosaic``'s own lambda.
# * the PLATE ADOPTING NAPARI'S RESOLVED WINDOW on arrival -- Julio's "look at the contrast
#   difference between napari window and plate view" -- is ``_adopt_window_view``, called from
#   ``_bind_window_contrast``, and pinned with its own mutation note in
#   ``tests/test_plate_follows_windows.py``. ``_adopt_centre_view``, the pane-flavoured twin this
#   test drove, was the dead one of the pair.


def test_the_plate_is_restored_even_while_the_raw_preview_streams(qapp,
                                                                  squid_dataset):
    """THE root cause of a ~50% flake, pinned as behaviour rather than as timing.

    Three gates asked `self._busy()` — "is ANY producer thread alive" — when the question they
    needed was "is an OPERATOR RUN alive". `_busy()` counts the raw plate preview, which is
    streaming almost all the time on a real plate, so the restore of the plate-wide view was
    deferred — and the only thing that ever delivers a deferred restore is a worker thread
    exiting. The view stayed scoped to a subset until some unrelated thread happened to finish.

    It read as a flake because it depended on whether the preview happened to still be running.
    It is not a flake: deferring on the preview is simply wrong. The restore path re-scopes
    and restarts the preview itself, so a streaming preview is never a reason to postpone.

    MUTATION: change any of the three gates back to `self._busy()` and this goes red — the sync
    defers instead of running, and `_pending_resync` is left set.

    RE-POINTED twice. 2b8fbc5 moved the symptom off the deleted central viewer's slider; the
    exploration pane's removal (2026-08-05) took the tab CLOSE that used to trigger this, so the
    sync is asked for directly. The GATES are what this is about and they are unchanged.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    qapp.processEvents()

    # Force the condition the flake depended on: a live raw preview at the moment of the sync.
    class _StillStreaming:
        IS_PREVIEW = True

        def isRunning(self):
            return True

    win._retired.append(_StillStreaming())
    assert win._busy() is True, "fixture is wrong: the window must look busy for this to bite"
    assert not V._run_scope.operator_busy(win._worker, win._retired), (
        "fixture is wrong: a raw preview must not count as an operator run")

    win._on_tab_changed(force=True)
    qapp.processEvents()

    assert not win._pending_resync, "the restore was deferred on a thread it does not depend on"
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
    """A fake that records the TERMINAL layer calls and borrows the REAL kind dispatch.

    ``add_result`` and its ``_RESULT_ADDERS`` table are taken off ``MosaicLayers`` rather than
    re-implemented here. That is deliberate: a fake that reimplements the thing under test can
    agree with itself while disagreeing with the app, and the dispatch from a result's declared
    kind to a layer type is exactly what these tests are for. What is faked is only the napari
    boundary -- ``add_mosaic`` (an Image) and ``add_labels`` (a Labels).
    """

    add_result = _MosaicLayers.add_result
    _RESULT_ADDERS = _MosaicLayers._RESULT_ADDERS

    def __init__(self):
        self.calls = []
        self.labels = []

    def add_mosaic(self, op, channel, data, **kw):
        self.calls.append((op, channel, data, kw))

    def add_labels(self, op, channel, data, **kw):
        self.labels.append((op, channel, data, kw))
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


class _ResultView:
    """A ``RegionViewer`` reduced to what the RESULT SINK reads, running the REAL sink.

    RETARGETED 2026-08-06. These tests used to install a ``_RecordingPane`` on the plate's own
    ``_mosaic_pane`` and assert against ``PlateWindow._add_result_layers``. That pane has been
    unconditionally ``None`` since 2b8fbc5, so the method under test could not run in the app and
    the branch calling it could not be taken; both were deleted. The live sink is
    ``RegionViewer.deliver_result``, reached through ``_deliver_to_views``, and it applies the same
    three rules (``mosaic_bbox_um`` placement, ``add_result`` kind dispatch, a z scale only when
    the result declares depth). So the sink moved and the assertions did not have to.

    ``deliver_result`` is CALLED UNBOUND off the real class rather than reimplemented, for the
    same reason ``_RecordingMosaic`` borrows ``add_result``: a fake that restates the rule can
    agree with itself while disagreeing with the app.
    """

    _roi_bbox = None
    window_id = 1

    def __init__(self, meta, region):
        self._pane = _RecordingPane()
        self._meta = meta
        self._region = region
        self._result_region = None

    @property
    def mosaic(self):
        return self._pane.mosaic

    def current_region(self):
        return self._region

    def _say(self, msg):
        pass

    def deliver_result(self, op, result, *, visible):
        from squidmip._region_viewer import RegionViewer

        return RegionViewer.deliver_result(self, op, result, visible=visible)


class _ResultManager:
    """The one seam ``_deliver_to_views`` and ``_result_regions`` read: ``mgr.windows``."""

    def __init__(self, *views):
        self.windows = list(views)


def _result_win(op="bgsub", region="A1", channels=("405", "488")):
    from squidmip._region_nav import RegionCursor

    win = _plate_window_shell()
    win._cursor = RegionCursor()
    win._cursor.set_order([region])
    win._cursor.activate(region)
    win._active_op_key = op
    win._readout = type("R", (), {"setText": lambda self, t: setattr(self, "t", t),
                                  "text": lambda self: getattr(self, "t", "")})()
    # A finished result is now also FILED in `_recipe.RESULTS` so a window opened later can reuse
    # it instead of recomputing, and the key carries WHICH acquisition (every plate has a `B2`).
    # That identity comes from the reader's path, so the shell has to carry a reader -- the same
    # rule every other attribute on this shell follows.
    win._reader = type("R", (), {"_path": "/fake/acquisition/result-win"})()
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
    win._view = _ResultView(win._meta, region)
    win._viewer_manager = _ResultManager(win._view)
    return win


def test_a_plane_op_result_becomes_a_layer_group_one_layer_per_channel(qapp):
    """The hole this branch exists to close: bgsub produced pixels and produced no layer."""
    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.full((2, 8, 8), 7, "uint16"))
    mos = win._view.mosaic
    assert mos.ops() == ["bgsub"]                    # one GROUP, keyed by the operator
    assert [c[1] for c in mos.group("bgsub")] == ["405", "488"]   # one LAYER per channel


def test_the_layer_group_is_not_drawn_until_the_region_is_whole(qapp):
    """Half a region drawn as a layer is a mosaic with holes, and the user reads holes as
    something the operator did."""
    win = _result_win("bgsub")
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((2, 8, 8), "uint16"))
    assert win._view.mosaic.calls == []


def test_a_run_that_ends_with_a_half_read_region_SAYS_SO_instead_of_stranding_it(qapp):
    """THE MEASURED DEFECT, 2026-08-06, on the 10x acquisition Julio reported against.

    Two of ``manual0``'s 27 TIFFs failed to read, so ``_on_result`` never saw a complete region,
    ``acc.complete()`` was never True, and the accumulator sat in ``_result_accs`` for the rest of
    the process. Nothing ever flushed it, so NO LAYER WAS EVER DRAWN — while the plate printed
    "✓ Maximum Intensity Projection · 1 well" and the requester window was told "finished in
    4.6 s". ``⚙ controls`` then opened no tab, which is the symptom that got reported, because
    ``RegionViewer._window_operators()`` was honestly empty.

    The refusal to draw half a mosaic stands (the test above pins it). What must not stand is
    doing it in SILENCE at the end of the run.
    """
    win = _result_win("mip")
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((2, 8, 8), "uint16"))   # 1 of 2 FOVs
    assert win._result_accs, "the accumulator should be holding the half region"

    stranded = V.PlateWindow._settle_stranded_results(win)

    assert stranded == 1
    assert win._result_accs == {}, "the stranded accumulator was not resolved"
    assert win._view.mosaic.calls == [], "half a region must still not be drawn"
    said = win._readout.text()
    assert "A1" in said and "1 of 2" in said, f"the run did not say what happened: {said!r}"
    assert win._run_error, "the window that ASKED would still have been told the run finished"


def test_settling_a_run_with_every_region_complete_is_a_no_op(qapp):
    """The other half: a clean run must not gain a failure line for having nothing left over."""
    win = _result_win("mip")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.zeros((2, 8, 8), "uint16"))
    win._readout.setText("")

    assert V.PlateWindow._settle_stranded_results(win) == 0
    assert win._readout.text() == ""
    assert not win._run_error


def test_the_operator_layer_lands_in_the_raw_mosaic_s_frame(qapp):
    """bbox_um is what puts the group ON TOP of raw. Without it the toggle would jump, and
    every difference the user saw would be misregistration, not the operator."""
    from squidmip._mosaic_source import mosaic_bbox_um

    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.zeros((2, 8, 8), "uint16"))
    kw = win._view.mosaic.group("bgsub")[0][3]
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
    assert win._view.mosaic.ops() == ["bgsub", "decon"]


def test_a_result_for_a_region_that_is_not_on_screen_is_dropped_not_accumulated(qapp):
    """Pane 2 shows ONE region. Holding full-res mosaics for every well of a plate run would
    be gigabytes of layers nobody can look at -- the same rule the raw path already follows."""
    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "B7", fov, np.zeros((2, 8, 8), "uint16"))
    assert win._view.mosaic.calls == []


def test_a_result_that_cannot_be_placed_SAYS_SO_instead_of_vanishing(qapp):
    """NO SILENT FAILURES. A channel-count mismatch used to be impossible to notice because
    nothing was ever drawn from a result in the first place."""
    win = _result_win("bgsub")
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((1, 8, 8), "uint16"))
    assert "not shown as a layer" in win._readout.text()
    assert win._view.mosaic.calls == []


def test_a_region_operator_s_fused_mosaic_is_added_whole_not_re_tiled(qapp):
    """stitch already returns the fused region; running it back through FOV placement would
    tile a mosaic as if it were a FOV."""
    win = _result_win("stitch")
    V.PlateWindow._on_result(win, "A1", 0, np.full((2, 20, 30), 3, "uint16"))
    layers = win._view.mosaic.group("stitch")
    assert len(layers) == 2
    assert layers[0][2].shape == (20, 30)


def test_no_open_window_means_the_result_slot_still_stands(qapp):
    """A plate with nothing open must not raise out of the result slot.

    Retargeted from ``_mosaic_pane = None`` (which was the only value that attribute ever held) to
    the condition that can actually differ today: no window is open to deliver into.
    """
    win = _result_win("bgsub")
    win._viewer_manager = _ResultManager()
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((2, 8, 8), "uint16"))


# --- Minerva: the on-screen LUTs, and the second destination ----------------------------------

def test_the_minerva_export_hands_the_on_screen_luts_to_the_exporter(
        qapp, squid_dataset, tmp_path, monkeypatch):
    """Julio: "channels need to be set to specific colors". The colours are on SCREEN, and the
    export defaults (acquisition display_color + 1/99.9 percentiles) do not know about them.

    Asserted at the seam that actually carries them - what export_selection is called with - 
    rather than by inspecting the widget, because a checkbox wired to nothing looks identical.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    seen = {}

    def spy(reader, selection, out_dir, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr("squidmip._minerva.export_selection", spy)
    luts = {"ch": {"clim": (3.0, 4.0), "rgb": (9, 9, 9)}}

    win.run_minerva_export(out_dir=str(tmp_path), launch=False, selection=[("B2", 0)], luts=luts)
    win._minerva.wait(20000)
    qapp.processEvents()

    assert seen.get("luts") == luts, "the LUTs stopped somewhere between the tab and the export"
    win.close()


def test_the_minerva_export_defaults_to_no_luts_so_the_plate_path_is_unchanged(
        qapp, squid_dataset, tmp_path, monkeypatch):
    """The plate has no window and therefore no on-screen LUTs. Passing nothing must reach the
    exporter as nothing, so the percentile behaviour every existing caller relies on stands."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    seen = {}
    monkeypatch.setattr("squidmip._minerva.export_selection",
                        lambda reader, selection, out_dir, **kw: (seen.update(kw), [])[1])

    win.run_minerva_export(out_dir=str(tmp_path), launch=False, selection=[("B2", 0)])
    win._minerva.wait(20000)
    qapp.processEvents()

    assert seen.get("luts", "missing") is None
    win.close()


def test_on_screen_luts_is_none_when_no_view_window_is_open(qapp, squid_dataset):
    """Not a failure: it IS the plate-level export, and the percentile defaults are right for it
    precisely because there is no screen to match."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win.on_screen_luts() is None
    win.close()


def test_on_screen_luts_reaches_a_focused_window(qapp, squid_dataset):
    """The None path was pinned and the returns-something path was not, which is exactly where the
    bug lived: ``focused_id`` and ``windows`` are PROPERTIES on ViewerManager and were called with
    parentheses. The resulting TypeError was swallowed by a broad except, so this returned None
    every single time. Every other Minerva test passed ``luts`` in by hand, so a feature that could
    never gather them looked fully covered.

    The stub declares both as @property ON PURPOSE. A stub exposing them as methods would pass
    against the broken code and would pin nothing.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    expected = {"Fluorescence_488_nm_Ex": {"clim": (12.0, 340.0), "rgb": (0, 255, 0)}}

    class _Win:
        window_id = 7

        def _per_channel_luts(self):
            return expected

    class _Mgr:
        @property
        def focused_id(self):
            return 7

        @property
        def windows(self):
            return [_Win()]

        def set_run_progress(self, report):
            """The raw preview is still running while this test builds; the plate publishes its
            progress through the manager. Present so teardown does not raise over the assertion."""

    win._viewer_manager = _Mgr()
    assert win.on_screen_luts() == expected, "the focused window's LUTs never reached the exporter"
    win.close()


def test_the_render_destination_refuses_an_empty_export_instead_of_starting_a_worker(
        qapp, squid_dataset):
    """render.py takes minutes. Starting it on nothing would spend them and produce nothing."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_minerva_render([])
    assert win._minerva_render is None
    assert "nothing to render" in win._readout.text()
    win.close()


def test_a_render_worker_is_retired_with_the_export_worker(
        qapp, squid_dataset, tmp_path):
    """closeEvent joins these threads. A render left connected is a MEASURED 132 s hold on the
    close for one real region, which is worse than the 90 s port poll that motivated
    ``_stop_minerva`` in the first place."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._minerva_render = w = V._MinervaRenderWorker(
        [(tmp_path / "a.ome.tiff", tmp_path / "a.json")])
    seen = []
    w.failed.connect(lambda m: seen.append(m))

    win._stop_minerva()

    assert win._minerva_render is None
    w.failed.emit("x")
    qapp.processEvents()
    assert seen == [], "the render worker survived _stop_minerva still connected"
    win.close()


# ==================================================================== SELECTION = A BOUNDING BOX
# Julio, 2026-08: "Region highlights on plate view: alpha modification removed, replaced with
# bounding boxes for selected regions. Current alpha value too high, causes confusion. Window title
# already identifies open wells, so dual indication unnecessary." And: "No alpha-valued overlay on
# the thumbnail. Rather frames. Do for > 3x3 wellplate."
#
# The confusion is specific, and it is why these tests measure the CELL INTERIOR rather than the
# whole widget: a wash repaints the thumbnail, so the pixels the user is judging change colour when
# the well is merely selected. At 1536wp density -- a cell is ~25 px and dozens of wells get
# selected at once -- that reads as a difference in the DATA. A frame lands on the boundary and
# leaves the thumbnail byte for byte alone, which is the property asserted below.
#
# Sized at plate scale on purpose. Spencer's warning is that the 2-region tissue set gives no
# intuition for what an overlay does when cells are 25 px and there are 1536 of them.

def _grab_rgb(ov) -> np.ndarray:
    """The widget as actually PAINTED, (H, W, 3) uint8, in R,G,B order.

    ``_grab_bgr`` returns the buffer in BYTE order; the reverse here is what makes a comparison
    against a QColor mean anything. ``_region_crop`` reads the same buffer without reversing,
    which is harmless there because it only takes a std, and wrong the moment an actual ink is
    named.
    """
    return _grab_bgr(ov)[:, :, ::-1]


def _fitted_plate(nrows, ncols, w=1400, h=900):
    """An nrows x ncols overview, fitted to (w, h), one channel declared."""
    rows = [V._row_letter(i) for i in range(nrows)]
    cols = [str(i + 1) for i in range(ncols)]
    by_rc = {(r, c): f"{rows[r]}{cols[c]}" for r in range(nrows) for c in range(ncols)}
    ov = V.PlateOverview(rows, cols, by_rc)
    ov.set_channels(["c0"], _RED_BLUE[:1], np.uint16)
    ov.resize(w, h)
    ov._fit()
    return ov


#: This file's cell-cropping helper. It was the only DPR-correct one for a while, and the three
#: older helpers above have now been routed through the same implementation.
_cell = _cell_slices


def _carries_ink(frame, sl, color, tol=24) -> bool:
    """True when some pixel in *sl* is *color* at full strength (a 16% wash never gets there)."""
    band = frame[sl].reshape(-1, 3).astype(int)
    want = np.array([color.red(), color.green(), color.blue()])
    return bool(np.abs(band - want).sum(1).min() <= tol)


def test_selecting_a_well_on_a_1536wp_leaves_the_thumbnail_pixels_untouched(qapp):
    """THE regression: no alpha wash over a selected cell once the plate is bigger than 3x3."""
    ov = _fitted_plate(32, 48)
    rc = (16, 24)
    ov.add_tile(*rc, ov._by_rc[rc], _tile([3000]))
    ov.recomposite(quick=True)
    assert ov._cd > 14, f"cell is {ov._cd:.1f} px wide; the interior crop would be empty"

    before = _grab_rgb(ov).copy()
    ov.highlight_regions([ov._by_rc[rc]])
    after = _grab_rgb(ov)

    assert np.array_equal(before[_cell(ov, rc, 8)], after[_cell(ov, rc, 8)]), (
        "selecting the well repainted the pixels INSIDE it. That is the alpha wash Julio removed: "
        "the thumbnail's apparent contrast and hue change when the well is merely selected.")
    assert not np.array_equal(before[_cell(ov, rc)], after[_cell(ov, rc)]), \
        "the selection produced no visible mark on the cell at all"


def test_the_selection_mark_on_a_1536wp_is_full_strength_ink_on_the_cell_boundary(qapp):
    """A drawn box in the accent ink at FULL alpha. A 16% wash cannot reach that colour, and the
    3 px black grid line between wells would bury a box painted before it."""
    ov = _fitted_plate(32, 48)
    rc = (10, 30)
    ov.highlight_regions([ov._by_rc[rc]])
    frame = _grab_rgb(ov)
    assert _carries_ink(frame, _cell(ov, rc), V._SEL_FRAME), (
        "no pixel of the selected cell carries the accent ink at full strength: the mark is still "
        "a wash, or the grid lines were painted over the box")


def test_the_selection_box_is_a_frame_and_not_a_filled_rectangle(qapp):
    """A frame is a PERIMETER. Counting the ink separates "box" from "opaque fill", which the
    interior-unchanged test alone does not: a fill in the accent ink would also leave a tile-less
    cell "changed" and would still carry the ink."""
    ov = _fitted_plate(32, 48)
    rc = (4, 4)
    ov.highlight_regions([ov._by_rc[rc]])
    frame = _grab_rgb(ov)
    band = frame[_cell(ov, rc)].reshape(-1, 3).astype(int)
    want = np.array([V._SEL_FRAME.red(), V._SEL_FRAME.green(), V._SEL_FRAME.blue()])
    inked = int((np.abs(band - want).sum(1) <= 24).sum())
    assert 0 < inked < len(band) // 3, (
        f"{inked} of {len(band)} pixels in the cell carry the accent ink: that is a filled "
        f"rectangle, not a frame on the boundary")


def test_a_3x3_or_smaller_plate_keeps_the_wash_because_frames_were_scoped_to_bigger_plates(qapp):
    """"Do for > 3x3 wellplate": at that size the cells are huge, one or two are selected at a
    time, and the wash is unambiguous rather than confusing. Changing it was not asked for."""
    assert not V.frames_for_grid(3, 3) and not V.frames_for_grid(1, 2)
    assert V.frames_for_grid(3, 4) and V.frames_for_grid(4, 3) and V.frames_for_grid(32, 48)

    ov = _fitted_plate(3, 3, 600, 600)
    rc = (1, 1)
    ov.add_tile(*rc, ov._by_rc[rc], _tile([3000]))
    ov.recomposite(quick=True)
    before = _grab_rgb(ov).copy()
    ov.highlight_regions([ov._by_rc[rc]])
    after = _grab_rgb(ov)

    assert not np.array_equal(before[_cell(ov, rc, 40)], after[_cell(ov, rc, 40)]), \
        "the 3x3 plate lost its wash; frames were scoped to plates BIGGER than 3x3"


def test_the_selection_frame_stroke_is_clamped_at_both_ends(qapp):
    """Visible at 1536wp density (~25 px cells), not a slab on a 4-well plate (~200 px cells)."""
    assert V.selection_frame_pen_px(25.0) == pytest.approx(2.5)
    assert V.selection_frame_pen_px(4.0) == 1.0          # floor: still one drawn pixel
    assert V.selection_frame_pen_px(200.0) == 3.0        # ceiling


def test_the_red_current_fov_box_still_outranks_the_blue_selection_frame(qapp):
    """Two boxes now share the boundary. ``_sel`` (red, the well the detail viewer is showing) is
    painted last and must stay on top: it is the transient 'you are here', and it was already a
    box, so it is the one indication the bounding-box change must not have swallowed."""
    ov = _fitted_plate(32, 48)
    rc = (5, 5)
    ov.highlight_regions([ov._by_rc[rc]])
    blue = _grab_rgb(ov).copy()
    ov._sel = rc
    ov.update()
    red = _grab_rgb(ov)

    assert _carries_ink(red, _cell(ov, rc), V._RED), "the red current-FOV box was covered"
    assert not np.array_equal(blue, red)


def test_the_1536_fixture_opens_and_reports_1536_wells(sim_1536wp):
    """Task 1's fixture guard, at the reader level: 1536 regions, four channels, live symlinks.

    ``open_reader`` is what fails first when the plate is hollow -- it refuses with "contains no
    {region}_{fov}_{z}_{channel}.tiff" -- so this is the cheapest statement that the plate-scale
    data the selection work was validated on is really there."""
    from squidmip import open_reader

    meta = open_reader(str(sim_1536wp)).metadata
    assert len(meta["regions"]) == 1536, f"{len(meta['regions'])} regions, not 1536"
    assert meta["wellplate_format"] == "1536 well plate"
    assert len(meta["channels"]) == 4
# ==============================================================================================
# THE PLATE THUMBNAIL AND ITS PER-CHANNEL DOWNSAMPLE PASS
#
# Julio, from the running GUI: "mip layer causes incomplete thumbnail ... incomplete render,
# likely a process getting stuck and losing sync with downsampling."
#
# MEASURED on the real 10x tissue set (2 regions) before the fix: open the plate and run mip on
# ONE region while the raw preview is still walking it, and the OTHER well ends the session with
# no thumbnail at all -- `PlateOverview.shown_cells()` returned {(0,0)} out of {(0,0), (0,1)} and
# `_tiles_by_layer` held no "raw" entry whatsoever.
#
# `_PreviewWorker` IS the per-channel downsample pass, and it fills the BASE layer that shows
# through wherever a subset run has nothing (`PlateOverview.underlay_cells`). `_run_operator`
# retired it unconditionally, and `_retire` disconnects a worker's signals before stopping it, so
# the tiles already in flight went too. Nothing restarts it but the return-to-raw path.

class _GatedPreview(V.QThread):
    """A ``_PreviewWorker`` stand-in that is STILL WALKING the plate until the test releases it.

    Deterministic where the real pass is a race: the defect only shows when an operator run starts
    while wells are still to be downsampled, which on a 2-well fixture is a few hundred
    milliseconds wide. Carries ``IS_PREVIEW`` because ``_run_scope.operator_busy`` reads it.
    """

    tileReady = V.Signal(int, int, str, object, object)
    runProgress = V.Signal(object)
    streamEnded = V.Signal()
    failed = V.Signal(str)
    IS_PREVIEW = True

    def __init__(self):
        super().__init__()
        import threading
        self._go = threading.Event()

    def run(self):
        self._go.wait(20)            # bounded: a hung test must not hang the suite

    def stop(self):
        self._go.set()


def _gated_preview_on(win, qapp):
    """Put a preview that is provably mid-plate in front of *win*, wired as the real one is."""
    win._stop_preview()
    gate = _GatedPreview()
    win._preview = gate
    gate.tileReady.connect(win._on_preview_tile)
    gate.start()
    assert _drain_until(qapp, gate.isRunning, 5)
    return gate


def test_a_subset_operator_run_leaves_the_thumbnail_downsample_pass_running(
        qapp, squid_dataset, blocking_worker):
    """THE reported defect. A run over SOME wells must not stop the pass that fills the rest."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    regions = list(win._order)
    assert len(regions) >= 2, "this fixture must have a well outside the run, or it proves nothing"
    n_ch = len(win._meta["channels"])
    gate = _gated_preview_on(win, qapp)

    win.run_operator("mip", save=False, regions=[regions[0]])

    assert win._preview is gate and gate.isRunning(), (
        "the operator run retired the per-channel downsample pass, so every well it had not yet "
        "reached keeps no thumbnail for the rest of the session")
    # ...and it is still WIRED: _retire disconnects before it stops, so a live thread alone is not
    # enough -- the tiles it is about to produce have to still reach the plate.
    ri, ci = win._fov_index[regions[-1]]["rc"]
    gate.tileReady.emit(ri, ci, regions[-1],
                        np.full((n_ch, V._CELL, V._CELL), 1000, np.uint16), None)
    assert (ri, ci) in win._overview.shown_cells(), (
        "the downsample pass is running but its tiles no longer reach the plate")

    gate.stop()
    gate.wait(5000)
    win.close()


def test_a_plate_wide_run_still_supersedes_the_downsample_pass(
        qapp, squid_dataset, blocking_worker):
    """The other direction, so the fix is a scope and not a deletion. A plate-wide run gives every
    well an operator tile, so continuing to read the same planes twice is pure cost."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    gate = _gated_preview_on(win, qapp)

    win.run_operator("mip", save=False)                 # regions=None: the whole plate

    assert win._preview is None, "a plate-wide run no longer supersedes the raw preview"
    assert _drain_until(qapp, lambda: not gate.isRunning(), 5)
    win.close()


# --- the downsample itself: EVERY CHANNEL, ON ITS OWN -----------------------------------------
#
# The plate cell is (C, h, w) native dtype all the way to the widget, and that channel axis is what
# the channel toggle and the global-contrast recomposite are built on. Both producers of a cell
# resize each channel independently; a cell built from one channel's pixels, or one that lost the
# axis, renders as a plausible picture of the wrong thing.

_DOWNSAMPLE_CHANNELS = ["c0", "c1", "c2", "c3"]
_DOWNSAMPLE_FRAME = (64, 64)


def _downsample_meta(fovs=(0, 1)):
    """Two coordinate-placed FOVs, so the cell is a real mosaic rather than one square tile."""
    return {
        "channels": [{"name": c} for c in _DOWNSAMPLE_CHANNELS], "dtype": "uint16",
        "z_levels": [0, 1, 2], "regions": ["A1"], "fovs_per_region": {"A1": list(fovs)},
        "fov_positions_um": {("A1", f): (float(f) * 24.0, 0.0) for f in fovs},
        "frame_shape": _DOWNSAMPLE_FRAME, "pixel_size_um": 1.0, "n_z": 3,
    }


class _PerChannelReader:
    """Every plane is a constant that NAMES its channel, so a swapped axis is unmistakable."""

    def __init__(self, path):
        self._path = str(path)

    def read(self, region, fov, channel, z, t=0):
        return np.full(_DOWNSAMPLE_FRAME, (_DOWNSAMPLE_CHANNELS.index(str(channel)) + 1) * 1000,
                       dtype=np.uint16)


def _expected_levels():
    return [(i + 1) * 1000 for i in range(len(_DOWNSAMPLE_CHANNELS))]


def test_the_raw_preview_downsamples_every_channel_on_its_own(qapp, tmp_path):
    """``_PreviewWorker`` -- the pass that fills the plate before any operator runs."""
    meta = _downsample_meta()
    worker = V._PreviewWorker(_PerChannelReader(tmp_path), meta,
                              {"A1": {"rc": (0, 0), "well_id": "A1", "idx": 0}}, ["A1"], cache=None)
    got = []
    worker.tileReady.connect(lambda *a: got.append(a))
    worker.run()

    assert got, "the downsample pass produced no tile at all"
    for _ri, _ci, _wid, tile, box in got:
        tile = np.asarray(tile)
        assert tile.ndim == 3 and tile.shape[0] == len(_DOWNSAMPLE_CHANNELS), (
            f"the plate cell lost its channel axis: {tile.shape}")
        assert box is not None and tile.shape[1:] == (box[2], box[3]), (
            f"a {tile.shape[1:]} tile was emitted for a {box[2]}x{box[3]} box")
        assert [int(round(float(tile[c].mean()))) for c in range(tile.shape[0])] \
            == _expected_levels(), (
            "a channel of the plate cell was downsampled from another channel's pixels")


def test_an_operator_tile_downsamples_every_channel_on_its_own(qapp, tmp_path):
    """``_OperatorWorker._on_well`` -- the same cell, the other producer, the same rule.

    Driven with the shape ``project_well`` yields for a z-REDUCER, ``(T, C, 1, Y, X)``: the z axis
    is already collapsed by the time a tile is built, so what is left to get wrong is the channel
    axis.
    """
    meta = _downsample_meta()
    idx = {"A1": {"rc": (0, 0), "well_id": "A1", "idx": 0}}
    worker = V._OperatorWorker("mip", _PerChannelReader(tmp_path), meta, idx, "",
                               regions=["A1"], save=False, n_fovs=None)
    assert worker.mosaic_boxes, "this fixture must mosaic, or the per-FOV box path is untested"
    got = []
    worker.tileReady.connect(lambda *a: got.append(a))
    fh, fw = _DOWNSAMPLE_FRAME
    image = np.stack([np.full((1, fh, fw), (i + 1) * 1000, np.uint16)
                      for i in range(len(_DOWNSAMPLE_CHANNELS))])[None]      # (T, C, 1, Y, X)
    for fov in meta["fovs_per_region"]["A1"]:
        worker._on_well("A1", fov, image)

    assert len(got) == len(meta["fovs_per_region"]["A1"])
    for _ri, _ci, _wid, tile, box in got:
        tile = np.asarray(tile)
        assert tile.shape == (len(_DOWNSAMPLE_CHANNELS), box[2], box[3]), (
            f"the operator emitted {tile.shape} for a {box[2]}x{box[3]} box")
        assert [int(round(float(tile[c].mean()))) for c in range(tile.shape[0])] \
            == _expected_levels(), (
            "a channel of the operator's plate cell came from another channel's pixels")
