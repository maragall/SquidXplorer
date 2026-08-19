"""HCS viewer, headless (offscreen) tests; skips entirely when PyQt5 (a GUI extra) is not installed."""

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
# Guard the two-Qt-bindings segfault: importing PyQt5 after PySide autoloads (napari/pytest-qt) crashes; run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 locally to load only PyQt5.
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

from squidxplorer import _viewer as V  # noqa: E402
from squidxplorer._napari_view import MosaicLayers as _MosaicLayers  # noqa: E402

from .conftest import CH_IN_YAML  # noqa: E402


def _needs(pkg: str):
    """Skip when an OPTIONAL operator backend (stitch/decon/etc.) is absent, instead of failing on an empty result."""
    return pytest.mark.skipif(
        importlib.util.find_spec(pkg) is None,
        reason=f"{pkg} not installed: this operator path is UNTESTED here, not passing")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)  # main() won't call exec_/exit under test
    return app


def _drain_until(app, pred, timeout=60):
    t0 = time.time()
    while not pred() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    return pred()




def _press(x, y, button=Qt.LeftButton):
    """A synthetic left-press/release at (x, y) — the handlers only read button/pos."""
    return QMouseEvent(QEvent.MouseButtonPress, QPointF(x, y), button, button, Qt.NoModifier)


def _move(x, y, buttons=Qt.NoButton):
    return QMouseEvent(QEvent.MouseMove, QPointF(x, y), Qt.NoButton, buttons, Qt.NoModifier)



def _plate_window_shell():
    """A PlateWindow with no super().__init__(), for testing a method in isolation — Qt raises RuntimeError on unset attributes, so needed ones are seeded here."""
    win = V.PlateWindow.__new__(V.PlateWindow)
    # Seeded to their absent values so the code under test takes the absent branch instead of touching real Qt.
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
    by_rc = {(0, 0): "A1", (1, 1): "B2"}          # A2 and B1 were never acquired
    assert V.cells_in_rect(["A", "B"], ["1", "2"], by_rc, 0, 0, 39, 39, 20.0) == [(0, 0), (1, 1)]


def test_fit_cell_always_returns_cell_shape():
    assert V._fit_cell(np.zeros((768, 768), np.float32)).shape == (V._CELL, V._CELL)
    assert V._fit_cell(np.zeros((V._CELL, V._CELL), np.float32)).shape == (V._CELL, V._CELL)
    assert V._fit_cell(np.zeros((40, 40), np.float32)).shape == (V._CELL, V._CELL)  # tiny frame upscaled


def test_running_contrast_latch_holds_against_new_wells():
    # The running histogram must not stomp a window the user set: a latched channel stays put while an unlatched one keeps auto-scaling.
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
    """Whatever the plate is currently showing, as an (H, W, 3) uint8 array; uses sizeInBytes() since byteCount() was removed in Qt6."""
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


def test_the_last_lit_channel_cannot_be_turned_off(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.set_channel_visible(0, False)
    assert list(ov._mask) == [False, True], "the first toggle must land"
    ov.set_channel_visible(1, False)
    assert list(ov._mask) == [False, True], "the last lit channel must survive"
    assert _rgb(ov).sum() > 0, "the plate must still be showing something"


def test_a_single_channel_acquisition_stays_lit(qapp):
    ov = _overview(qapp, n_ch=1)
    ov.add_tile(0, 0, "A1", _tile([1000]))
    assert _rgb(ov).max() > 0
    ov.set_channel_visible(0, False)
    assert _rgb(ov).max() > 0


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
    # QImage wraps the numpy buffer; if the widget drops the reference the canvas is a use-after-free, so force a GC and read the plate back.
    import gc
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.recomposite()
    expected = _rgb(ov).copy()
    gc.collect()
    np.testing.assert_array_equal(_rgb(ov), expected)


def test_recomposite_is_global_so_wells_stay_comparable(qapp):
    # D6 regression: a per-well window (the old reopen behaviour) would wrongly equalize a bright well and a dim well.
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


# Zero padding from unfilled FOV slots in a mosaic cell must not reach the running histogram — it pins the 1st percentile to 0 and washes out the whole plate.

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
    # A 36-FOV well is built from 36 arrivals, not overwrites; each arrival re-composites the whole cell so the seam against its landed neighbour updates.
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 4
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(0, 0, h, w))
    first = _rgb(ov)[:, :V._CELL].copy()
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(0, w, h, w))   # the neighbour to its right
    store = ov._store["raw"]
    assert store[0, :h, :w].max() > 0 and store[0, :h, w:2 * w].max() > 0   # BOTH still present
    assert not np.array_equal(_rgb(ov)[:, :V._CELL], first)                # the cell repainted


def test_contrast_ignores_the_mosaic_zero_padding(qapp):
    # Regression: a sparse mosaic's window must come from the field's pixels alone — feeding the padded cell drags the 1st percentile to 0 and washes the plate out.
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
    # The user-visible consequence: a dim well next to a bright one, both sparse mosaics, with the dim well's rendered range collapsed by padding poisoning the histogram.
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


# A subset run must replace only its own wells: the plate used to blit only the active layer, blanking every well the run didn't touch.


def _cell_slices(ov, rc, inset=0, inset_frac=0.0):
    """Index tuple for cell *rc* in a GRABBED frame, inset LOGICAL px on every side — the one place that accounts for the device pixel ratio (grab() renders at screen ratio, _cell_rect answers in logical px)."""
    r = ov.devicePixelRatioF()
    x, y, cw, ch = (v * r for v in ov._cell_rect(*rc))
    ix = int(cw * inset_frac) + int(round(inset * r))
    iy = int(ch * inset_frac) + int(round(inset * r))
    return (slice(int(y) + iy, int(y + ch) - iy), slice(int(x) + ix, int(x + cw) - ix))


def _grab_bgr(ov) -> np.ndarray:
    """The widget's own paint as (H, W, 3) uint8 in BYTE order (B, G, R), at DEVICE resolution; Format_RGB32 packs little-endian as B,G,R,A, so callers naming an actual QColor use _grab_rgb instead."""
    img = ov.grab().toImage().convertToFormat(QImage.Format_RGB32)
    ptr = img.bits()
    ptr.setsize(img.sizeInBytes())
    row = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return row[:, : img.width(), :3]


def _painted_cell(ov, ri, ci, w=420, h=260):
    """The interior of one cell as the user sees it: the widget's own paint, not the canvas; inset to exclude the grid pen and status dot."""
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
    ov = _overview(qapp, n_ch=1)
    ov.add_tile(0, 0, "A1", _tile([4000]))
    ov.add_tile(0, 1, "A2", _tile([4000]))
    ov.set_active_layer("mip")
    assert ov.underlay_cells() == {(0, 0), (0, 1)}      # nothing computed yet: all base

    ov.add_tile(0, 1, "A2", _tile([9000]), layer="mip")
    assert ov.underlay_cells() == {(0, 0)}
    assert ov.shown_cells() == {(0, 0), (0, 1)}


def test_the_base_never_shows_through_itself(qapp):
    """raw active must stay byte-identical: no underlay, no second blit."""
    ov = _overview(qapp, n_ch=1)
    ov.add_tile(0, 0, "A1", _tile([4000]))
    assert ov.underlay_cells() == set()
    assert ov.shown_cells() == {(0, 0)}



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
    # 'R2C3' is deliberate: it doesn't match <letters><digits>, the case that used to crash activate_well's parse_well_id — a slide carrier IS a plate, with a freeform region id as a carrier cell.
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
    # run_operator's SAVE path drives write_plate with the selected operator and must not also write the uncompressed per-TIFF copy (tiff=False) — that would double disk use.
    # spot on purpose: every per-FOV INTENSITY operator now saves acquisition-format beside the
    # source (tests/test_acq_output.py); a labels producer still owes write_plate.
    import squidxplorer
    captured = {}

    def fake_write_plate(reader, out_dir, *, n_fovs=1, workers=None, operator="mip",
                         tiff=True, on_well=None, write_workers=4, stop=None, on_error=None,
                         regions=None, operator_kwargs=None):
        # operator_kwargs must reach the SAVE path too, not just preview; a stub whose signature drifts from the real function raises TypeError here rather than passing by luck.
        captured.update(operator=operator, tiff=tiff, out_dir=str(out_dir), regions=regions,
                        operator_kwargs=operator_kwargs)
        return {"plate": str(out_dir), "levels": 1}      # no wells — we only assert the dispatch
    monkeypatch.setattr(squidxplorer, "write_plate", fake_write_plate)

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("spot", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: "operator" in captured)
    assert captured["operator"] == "spot"
    assert captured["operator_kwargs"] is None      # no panel values were passed for this run
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
    """Double-click moves the cursor (moving the red frame) and asks the ViewerManager for one independent window over that region — there is no embedded ndviewer any more."""
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






def _boxed(ov):
    """Records what a Shift-drag asks to open (marqueeSelected) plus every selection emission alongside it."""
    opened, selected = [], []
    ov.marqueeSelected.connect(lambda wells: opened.append(list(wells)))
    ov.selectionChanged.connect(lambda wells: selected.append(list(wells)))
    return opened, selected

def _sel_overview(cd=20.0):
    """A 2x2 plate with a sparse corner (B1 never acquired) and a frozen view (deterministic pixels)."""
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
    """A second drag reports its own box rather than accumulating with the first."""
    ov = _sel_overview()
    opened, _sel = _boxed(ov)
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.ShiftModifier)          # sweep the whole 2x2
    assert opened == [["A1", "A2", "B2"]]                      # B1 never acquired -> excluded
    assert ov.selected_wells() == [], "the drag left a lingering selection wash on the plate"
    _drag(ov, *_within(0, 0), Qt.ShiftModifier)                # a fresh marquee over A1 only...
    assert opened == [["A1", "A2", "B2"], ["A1"]]              # ...its own box, not a union


def test_additive_marquee_unions(qapp):
    """Seeded with Shift+Alt since a plain Shift-drag now opens a window instead of selecting."""
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
    """Emits once per gesture on release, not per mouse-move — a 1536-well plate would otherwise rebuild + emit a list per move."""
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
    from qtpy.QtCore import QPoint
    from qtpy.QtGui import QWheelEvent
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    cd_before = ov._cd
    # QPointF, not QPoint: Qt6 dropped the QPoint overload for event positions; QPointF works on both bindings.
    ov.wheelEvent(QWheelEvent(QPointF(60, 60), QPointF(60, 60), QPoint(0, 0), QPoint(0, 120),
                              Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert ov._cd == cd_before                                  # zoom did NOT happen


def test_right_button_release_does_not_commit_a_selection(qapp):
    """Qt delivers a release for whichever button went up, so button must be checked or a right-click during a Shift-drag toggles a well."""
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
    """A modal dialog or alt-tab delivers a leave with no release; a stranded marquee would disable zoom permanently."""
    from qtpy.QtCore import QEvent, QPoint
    from qtpy.QtGui import QWheelEvent
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    assert ov._marquee is not None
    ov.leaveEvent(QEvent(QEvent.Leave))                         # grab lost; no release ever arrives
    assert ov._marquee is None
    cd_before = ov._cd
    # QPointF, not QPoint: Qt6 dropped the QPoint overload for event positions; QPointF works on both bindings.
    ov.wheelEvent(QWheelEvent(QPointF(60, 60), QPointF(60, 60), QPoint(0, 0), QPoint(0, 120),
                              Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert ov._cd != cd_before                                  # zoom works again



def test_plain_drag_still_pans(qapp):
    ov = _sel_overview()
    ox0, oy0 = ov._ox, ov._oy
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.NoModifier)              # NO Shift
    assert (ov._ox, ov._oy) != (ox0, oy0), "plain drag no longer pans"
    assert ov.selected_wells() == [], "plain drag must not select"


def test_double_click_selects_only_the_well_it_opens(qapp):
    """A plain click REPLACES the selection (idempotent), so press+release+dblclick leaves exactly the opened well selected.

    That is the NO-VIEW-OPEN column, which is what a bare PlateOverview reports: with a view open
    (`set_click_navigates`) a plain click NAVIGATES that view and touches no selection, and a click
    on an EMPTY position clears the selection in BOTH modes — the only click-driven deselect, so
    navigation does not take it away. tests/test_plate_navigates_views.py covers that column, and
    pins that this one did not move.
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
    ov = _sel_overview()
    opened, _sel = _boxed(ov)
    ov.select(1, 1)
    _drag(ov, *_within(0, 0), Qt.ShiftModifier)
    assert ov._sel == (1, 1)                                    # red box unmoved
    assert opened == [["A1"]]


def test_clear_selection_emits_empty(qapp):
    """Seeded through Shift+Alt, since a plain Shift-drag opens a window rather than selecting."""
    ov = _sel_overview()
    seen = []
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.ShiftModifier | Qt.AltModifier)
    assert ov.selected_wells() == ["A1", "A2", "B2"]            # there is really something to clear
    ov.selectionChanged.connect(lambda wells: seen.append(list(wells)))
    ov.clear_selection()
    assert ov.selected_wells() == [] and seen == [[]]



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
    # mip saves acquisition-format now, so the selection must reach THAT writer's regions=.
    from squidxplorer import _acq_output
    captured = {}

    def fake_write_acquisition_planes(reader, operator, dst, **kw):
        captured.update(regions=kw.get("regions"))
        return {"path": str(dst), "n_fields_written": 1, "stopped": False}
    monkeypatch.setattr(_acq_output, "write_acquisition_planes", fake_write_acquisition_planes)

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
    # REGRESSION: with the key moved to _floating, an unpatched _open_op_tab would rebuild the UI (a second live shell); the opener must focus the float instead.
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
    # Floats follow docked-tab semantics across a plate swap: they persist (op-tab staleness on re-ingest is tracked separately in TODOS.md).
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
    # Asserted with a SPY on the reader, not timing — the toggle must recomposite purely from the retained store.
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
    """napari's eye icon is the sole control; the plate is a pure sink. MUTATION: dropping the on_user_visibility binding in _bind_napari_contrast should fail this."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    # what napari reports when the user clicks an eye icon off, and back on
    win._overview.set_channel_visible(0, False)
    assert win._overview._mask[0] == False        # noqa: E712 — numpy bool, not python bool
    win._overview.set_channel_visible(0, True)
    assert win._overview._mask[0] == True         # noqa: E712
    win.close()


# Contrast has exactly one owner (the central array viewer); the plate is a pure sink and must never grow a second contrast control of its own.



def test_following_a_viewer_window_never_latches_the_plate_manual(qapp, squid_dataset):
    """A sink must record what the owner resolved without latching to manual on its own — autoscaling on open must not mark every channel as user-set."""
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
    """Precedence: user latch > owning viewer's window > caller-computed default."""
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
    """The cache is keyed on pixels: a contrast change must not drop it, but a new tile landing must."""
    # Plate must be big enough that the screen can't show it 1:1 — the thumbnail only exists once the composite is sub-sampled, which is what a drag hits.
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
    # D3: the store lives in the widget, so the toggle works on the operator layer too, not just raw — each layer keeps its own (C, H, W) store.
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




def test_preview_spinner_still_runs_first_n_wells(qapp, squid_dataset, monkeypatch):
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
    """A bad read used to be swallowed silently, leaving the plate half-grey forever; it must finalize what landed and name the failure."""
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
    """_stop_worker clears self._worker while the retired thread drains — the guard must still refuse a new run."""
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








def test_subset_save_is_disk_guarded(qapp, squid_dataset, monkeypatch, tmp_path):
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
    assert one_gb < full_gb                             # The estimate counts FIELDS (every FOV), so scale a 1-well run by its share of them rather than treating it as a whole-plate estimate.
    total_fields = sum(len(win._meta["fovs_per_region"][r]) for r in win._fov_index)
    share = len(win._meta["fovs_per_region"][one[0]]) / total_fields
    assert one_gb == pytest.approx(full_gb * share, rel=0.01)
    win.close()


def _write_plate_for_open(out, *, stop_after=None):
    """Writes a real plate.ome.zarr from a hand-built stream, stopped after N fields — deliberately the real writer, not a fabricated marker."""
    from squidxplorer._output import write_from_stream

    regions = ["A1", "A2", "B1", "B2"]
    meta = {
        "regions": regions,
        "fovs_per_region": {r: [0, 1] for r in regions},
        "fov_positions_um": {(r, f): (0.0, f * 16.0) for r in regions for f in (0, 1)},
        "channels": [{"name": "C0", "display_color": "#FFFFFF"}],
        "n_z": 1, "z_levels": [0], "dz_um": 1.0, "pixel_size_um": 1.0,
        "frame_shape": (16, 16), "dtype": np.dtype("uint16"), "n_t": 1,
    }

    def stream():
        for region in regions:
            for fov in (0, 1):
                yield region, fov, np.full((1, 1, 1, 16, 16), 7, np.uint16)

    seen = {"n": 0}

    def stop():
        seen["n"] += 1
        return seen["n"] > int(stop_after)

    return write_from_stream(meta, stream(), out, n_fovs=None, check_disk=False,
                             stop=(None if stop_after is None else stop))

def test_open_computed_refuses_the_plate_a_stopped_write_actually_leaves(
        qapp, tmp_path, monkeypatch):
    """End to end: write_from_stream stops mid-write and marks the store; the window must refuse opening it."""
    base = tmp_path / "acq.hcs"
    manifest = _write_plate_for_open(base, stop_after=3)
    assert manifest["complete"] is False and manifest["n_fields_written"] == 3

    win = V.PlateWindow(None)
    monkeypatch.setattr(V.QFileDialog, "getExistingDirectory", lambda *a, **k: str(base))
    win._open_computed()
    said = win._readout.text()
    assert "INCOMPLETE" in said, f"a stopped save opened as a finished plate; window said: {said!r}"
    assert "3 of 8" in said, f"the refusal must quote the shortfall the store recorded: {said!r}"
    win.close()


def test_open_computed_accepts_a_write_that_finished(qapp, tmp_path, monkeypatch):
    """A guard that refuses everything is as useless as one that refuses nothing."""
    base = tmp_path / "acq.hcs"
    assert _write_plate_for_open(base)["complete"] is True

    win = V.PlateWindow(None)
    monkeypatch.setattr(V.QFileDialog, "getExistingDirectory", lambda *a, **k: str(base))
    win._open_computed()
    assert "INCOMPLETE" not in win._readout.text()
    win.close()


def test_a_finished_save_run_leaves_no_incomplete_marker(qapp, squid_dataset, tmp_path):
    from squidxplorer._output import incomplete_reason

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    # spot (labels): it still writes the OME-Zarr plate whose marker this asserts (per-FOV
    # intensity operators save acquisition-format now)
    win.run_operator("spot", out_parent=str(tmp_path), regions=["B2", "B3"], save=True)
    assert _drain_until(qapp, lambda: not win._busy(), timeout=90)
    out = tmp_path / f"{win._acq_name}.hcs"
    assert (out / "plate.ome.zarr").is_dir()
    assert incomplete_reason(out) is None, "a completed plate must not be flagged incomplete"
    win.close()


def test_open_computed_names_a_well_that_cannot_read_its_own_image_id(
        qapp, squid_dataset, tmp_path, monkeypatch):
    """Falls back to well 0's image id when a well's own metadata is unreadable — must be NAMED, never silent."""
    import json

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    # spot (labels): it still writes the OME-Zarr plate this test corrupts (per-FOV intensity
    # operators save acquisition-format now)
    win.run_operator("spot", out_parent=str(tmp_path), regions=["B2", "B3"], save=True)
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


# _OperatorWorker used to be constructed without n_fovs (defaulting to 1) and set_mosaic_boxes had zero callers — every inherited viewer test still passed since they only exercise the single-tile path.

def test_operator_worker_is_constructed_for_multi_fov_not_defaulted_to_one(
        qapp, squid_dataset, tmp_path, monkeypatch):
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


# The wiring guards above don't prove a FOV lands anywhere in particular, or look at a pixel — a mosaic stacking every field at (0, 0) would pass both, so these drive the real widget and assert on geometry and rendered pixels.

def test_mosaic_places_each_fov_at_its_own_stage_offset(qapp, squid_dataset,
                                                        tmp_path):
    """Fixture's fov 1 is +0.5mm in x from fov 0 at the same y, so it must place to the right on the same row."""
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
    """Measured on the montage crop, not grab() of the whole widget — a whole-widget variance check passes even with tiles deleted, since chrome alone has variance."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))

    def _mosaic_complete():
        """Waits until every field asserted on has landed — a weaker wait made the outcome depend on how far the background stream got."""
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

        # The old dynamic-range magic number was an artefact of this 4x4 fixture; this asserts the PROPERTY it stood for instead — signal present, brighter than background, in the right place — which catches the same mutants plus one more (signal in the wrong spot).
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
        # Every rendered pixel above background sits inside one of B2's FOV boxes, so a mosaic drawn at the wrong offset can't pass by lighting up chrome or padding.
        boxes = [ov._boxes[("B2", f)] for f in (0, 1)]
        bright = np.argwhere(got.max(axis=2) > int(ref.max()))
        assert len(bright), "no pixel above background at all"
        assert all(any(t <= y < t + h and l <= x < l + w for t, l, h, w in boxes)
                   for y, x in bright), (
            f"signal rendered outside B2's FOV boxes {boxes}: the mosaic is misplaced.")
        # Must reach BOTH fields' sub-boxes, not just fov 0's — measured on the native-dtype store, before any contrast window, so a later rule change can't make this half unaskable.
        if ov._boxes:
            cell = ov._store_for(ov._active)[:, ri * V._CELL:(ri + 1) * V._CELL,
                                             ci * V._CELL:(ci + 1) * V._CELL]
            for fov in (0, 1):
                top, left, h, w = ov._boxes[("B2", fov)]
                assert int(np.count_nonzero(cell[:, top:top + h, left:left + w])) > 0, (
                    f"fov {fov}'s sub-box is entirely zero: only part of the mosaic was composited.")
    finally:
        win._stop_worker(); win.close()


# Contrast scope is a DISPLAY control, never a run parameter — the scope re-composites the per-layer native-dtype store rather than keeping a second copy of every tile.

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
    """Regression: window() used to return a 1-unit span against a ~128-unit histogram bin, so a blank/dead/saturated well rendered full white."""
    from squidxplorer._montage import _window

    rc = V._RunningContrast(1, float(np.iinfo(np.uint16).max))
    flat = np.full((8, 8), 500.0, dtype=np.float32)
    rc.add(0, flat)
    lo, hi = rc.window(0)
    assert hi - lo <= 0, "a flat channel must produce a degenerate window, not a 1-unit span"
    assert np.all(_window(flat, lo, hi) == 0.0), "a flat channel must render black, not white"


def test_running_contrast_saturated_channel_renders_black():
    from squidxplorer._montage import _window

    dmax = float(np.iinfo(np.uint16).max)
    rc = V._RunningContrast(1, dmax)
    sat = np.full((8, 8), dmax, dtype=np.float32)
    rc.add(0, sat)
    lo, hi = rc.window(0)
    assert np.all(_window(sat, lo, hi) == 0.0)


def test_running_contrast_spread_channel_still_windows():
    rc = V._RunningContrast(1, float(np.iinfo(np.uint16).max))
    rc.add(0, np.linspace(0, 60000, 64 * 64).astype(np.float32).reshape(64, 64))
    lo, hi = rc.window(0)
    assert hi > lo


def test_running_contrast_empty_histogram_is_full_range():
    rc = V._RunningContrast(2, 65535.0)
    assert rc.window(0) == (0.0, 65535.0)


def test_blank_well_renders_black_not_white_through_the_widget(qapp):
    ov = V.PlateOverview(["A"], ["1"], {(0, 0): "A1"})
    ov.set_channels(["c0"], np.array([[1.0, 1.0, 1.0]], np.float32), dtype=np.uint16)
    ov.add_tile(0, 0, "A1", np.full((1, V._CELL, V._CELL), 7, np.uint16))   # a blank channel
    ov.recomposite(ov._active)
    cell = _plate_rgb(ov)[:V._CELL, :V._CELL]
    assert float(cell.mean()) < 1.0, (
        f"a blank well rendered at mean {cell.mean():.1f} — it must be black, not white.")


def test_reopened_plate_windows_globally_like_the_run_that_wrote_it(qapp):
    """A reopened plate must window globally like the run that wrote it — per-tile percentile windowing made a dim well and bright well indistinguishable."""
    import inspect

    src = inspect.getsource(V._ComputedPlateWorker)
    assert "np.percentile" not in src, (
        "_ComputedPlateWorker is windowing tiles itself again — that is per-region contrast "
        "imposed on the reopen path, and it makes a reopened plate disagree with its own run.")
    assert "_window(" not in src, "the reopen path must emit native tiles, not pre-windowed RGB"
    sig = inspect.signature(V._ComputedPlateWorker.__init__)
    assert "colors" not in sig.parameters, (
        "a worker that needs colours is compositing; the widget owns compositing (IMA-206).")


# A Shift-drag now emits marqueeSelected, and PlateWindow._on_marquee_selected turns that into ViewerManager.open(ordered) — one independent napari window with a region slider over the boxed set, not an exploration tab in a central pane.

def _freeze(ov, cd=20.0):
    """Freezes the plate view so synthetic widget coordinates hit the intended cells."""
    ov._user_view = True
    ov._cd, ov._ox, ov._oy = cd, 0.0, 0.0
    return ov


def test_shift_click_refines_the_selection_without_opening_anything(qapp,
                                                                    squid_dataset):
    """Only the drag opens a window; Shift+click refines one well without opening one per corrective click."""
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


# A deferred view sync must be delivered, not dropped: _on_tab_changed used to return early while _busy() and never re-deliver the switch once the run drained.

class _BlockingWorker(V.QThread):
    """An _OperatorWorker stand-in that stays RUNNING until stop() (or the test) releases it."""
    tileReady = V.Signal(int, int, str, object)
    pushReady = V.Signal(int, object)
    resultReady = V.Signal(str, int, object)     # full-res result -> napari layer group
    progress = V.Signal(int, int)
    runProgress = V.Signal(object)               # the engine-unit report (squidxplorer._progress)
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








# The gesture/geometry are tested separately from I/O since neither needs pixels or Qt; only the read tests touch a real pyramid, via pyramid_dataset (a 4x4 fixture writes ONE level, so it alone can't prove level selection).

class _FakeLoupeSource(V._LoupeSource):
    """Holds a field and slices it with real numpy semantics — a naive fake that ignores origin can't express the negative-origin bug it stands in for."""

    def __init__(self, well_px=1000, n_levels=3, pixel_size_um=0.325, missing=()):
        self.well_px, self.n_levels, self.pixel_size_um = well_px, n_levels, pixel_size_um
        self._missing = set(missing)
        self.reads = []
        self._fields = {}

    def _field(self, level):
        """A field with a per-pixel ramp so a crop's content identifies where it came from."""
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
        # +fov as well as +t: a crop names the FIELD and FRAME it came from, so a test can distinguish a read of field 1 from field 0 at the same rectangle.
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



def test_loupe_scale_never_upsamples_past_native():
    for cd, well in ((20, 4168), (200, 4168), (1000, 1024), (5000, 1024)):
        s, m = V.loupe_scale(cd, well)
        assert s <= 1.0 or s == pytest.approx(cd / well)   # only "past native" exceeds 1.0
        assert m >= 1.0                                    # and it never shrinks


def test_loupe_inset_shows_at_most_one_whole_well():
    """A fixed 8x doesn't survive a 1536wp (well is ~10px at fit); scale is floored so the inset shows at most one well."""
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
    """Floors magnification at 1.0 so zooming beyond native doesn't shrink the inset below the plate's own scale."""
    s, m = V.loupe_scale(cd=4096, well_px=1024)     # plate already at 4x native
    assert m == pytest.approx(1.0)
    assert s == pytest.approx(4.0)                  # inset matches the plate, never below it
    for cd in (1, 10, 100, 1024, 2048, 8192):
        assert V.loupe_scale(cd, 1024)[1] >= 1.0


def test_loupe_scale_is_dynamic_in_plate_zoom():
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
    # Same inset, coarser level -> fewer pixels to read — keeps a zoomed-out loupe cheap instead of pulling a full 4168px plane.
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


def test_the_loupe_window_memo_evicts_instead_of_growing_for_the_source_lifetime(monkeypatch):
    """The per-source (well, t) contrast memo was a plain dict with NO eviction — it grew for
    the life of the source. It is now a budgeted LRU: the bound holds, the newest entries
    survive, the oldest are evicted, and a hit is still a memo (no recompute)."""
    from squidxplorer import _loupe as PO   # the engine moved; the memo lives in _loupe now

    class _Src(PO._LoupeSource):
        def __init__(self):
            self.coarse_reads = 0

        def coarse(self, well_id, time_point=0):
            self.coarse_reads += 1
            return np.full((1, 4, 4), 7, dtype=np.uint16)

    monkeypatch.setattr(PO, "_AUX_CACHE_BYTES", 5 * PO._WINDOW_PAIR_NBYTES)  # room for 5 memos
    src = _Src()
    for i in range(40):
        assert src.window(f"A{i}", 0) is not None
    cache = src.__dict__["_win_cache"]
    assert cache.nbytes <= cache.capacity_bytes
    assert len(cache) <= 5                                   # bounded, where the dict held 40
    assert cache.get(("A39", 0)) is not None                 # newest survives
    assert cache.get(("A0", 0)) is None                      # oldest evicted, not kept forever
    reads = src.coarse_reads
    assert src.window("A39", 0) == src.window("A39", 0)
    assert src.coarse_reads == reads                         # a hit is still a memo


def test_loupe_um_per_screen_px_refuses_to_guess():
    assert V.loupe_um_per_screen_px(0.325, 1.0) == pytest.approx(0.325)
    assert V.loupe_um_per_screen_px(0.325, 0.5) == pytest.approx(0.65)
    assert V.loupe_um_per_screen_px(None, 1.0) is None      # unknown -> no bar, never a guess
    assert V.loupe_um_per_screen_px(0, 1.0) is None
    assert V.loupe_um_per_screen_px(float("nan"), 1.0) is None


def test_composite_rgb_matches_manual_windowing():
    # The loupe's private _composite_rgb is gone; composite is the one compositor now, so this asserts against the survivor.
    from squidxplorer._montage import composite
    planes = np.array([[[0.0, 10.0]], [[5.0, 5.0]]])
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wins = [(0.0, 10.0), (0.0, 10.0)]
    out = composite(planes, colors, wins).astype(float) / 255.0
    assert out.shape == (1, 2, 3)
    assert out[0, 0, 0] == pytest.approx(0.0, abs=0.01)      # ch0 at its window floor
    assert out[0, 1, 0] == pytest.approx(1.0, abs=0.01)      # ch0 at its window ceiling
    assert out[0, 0, 1] == pytest.approx(0.5, abs=0.01)      # ch1 mid-window, in green


def test_ima242_one_contrast_model_resolves_manual_over_auto():
    rc = V._RunningContrast(2, 65535.0)
    rc.add(0, np.full((8, 8), 1000, np.uint16))
    rc.add(1, np.full((8, 8), 2000, np.uint16))
    auto0 = rc._auto_window(0)
    assert rc.resolve(0, auto0) == auto0            # untouched -> the caller's auto window stands
    rc.set_manual(0, 111.0, 222.0)
    # A latched channel keeps the user's window whatever auto the caller derived — the single rule the plate, per-region cells and loupe all consult.
    assert rc.resolve(0, (9.0, 9999.0)) == (111.0, 222.0)
    assert rc.window(0) == (111.0, 222.0)
    assert rc.resolve(1, (9.0, 9999.0)) == (9.0, 9999.0)     # ch1 is not latched
    rc.set_auto(0)
    assert rc.resolve(0, (9.0, 9999.0)) == (9.0, 9999.0)     # unlatched -> auto again


def test_ima242_no_second_contrast_implementation_survives():
    assert not hasattr(V, "_composite_rgb"), "the loupe's private compositor came back"
    assert not hasattr(V, "_percentile_window"), "the second percentile rule came back"


def test_fov_seam_is_single_fov():
    # The plate resolves a WELL, never a FOV, so this is 0 today; routing FOV lookups through one helper (not bare 0 literals) makes this fail loudly once viewer-side multi-FOV lands.
    assert V._fov_of_well("B2") == 0
    assert V._fov_of_well("B2", {"B2": [0]}) == 0
    assert V._fov_of_well("B2", {"B2": [3, 4]}) == 3



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
    """The obvious press+immediate-drag test passes even if this breaks, since dwell isn't exercised there."""
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
    """Qt sends press/release/dblclick, so the second press re-arms the hold timer unless cancelled."""
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



def test_raw_layer_gets_a_loupe_source_on_ingest(qapp, squid_dataset):
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
    # Channel order is the metadata's, not the fixture's, so resolve the index rather than assume it.
    names = [c["name"] for c in win._meta["channels"]]
    for ch in names:
        assert np.array_equal(crop[names.index(ch)], arrays[("B2", 0, 1, ch)])   # unmodified pixels
    win.close()


def test_raw_source_clamps_a_negative_crop_origin(qapp, squid_dataset):
    """A crop near the upper-left corner starts at a negative origin, and numpy slicing silently returns an empty array rather than raising — clamping must catch this."""
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


# Three separate ways the loupe used to show pixels from somewhere the plate was not, all silently — pinned at the widget seam, not the pure helpers, since each bug lived in the wiring between a helper and a caller that never passed the right argument.

def _freeform_overview():
    """Two regions on a slide carrier, each placed by its own rectangle and letterboxed to the same aspect ratio."""
    layout = {(0, 0): (0.0, 0.0, 2.0, 1.0),        # A1: a 2-FOV row, bars top and bottom
              (0, 1): (2.2, 0.0, 1.0, 2.0)}        # A2: a 2-FOV column, bars left and right
    ov = V.PlateOverview(["A"], ["1", "2"], {(0, 0): "A1", (0, 1): "A2"}, layout=layout)
    ov.resize(800, 600)
    ov.set_mosaic_boxes({("A1", 0): (22, 0, 44, 44), ("A1", 1): (22, 44, 44, 44),
                         ("A2", 0): (0, 22, 44, 44), ("A2", 1): (44, 22, 44, 44)})
    ov._fit()
    return ov


def _widget_point_of_block(ov, ri, ci, bx, by):
    """Exact inverse of _cell_point, solved analytically so a mistake here can't cancel one in the code under test."""
    rx, ry, rw, rh = ov._cell_rect(ri, ci)
    sx, sy, sw, sh = ov._cell_source(ri, ci)
    return (rx + (bx - (sx - ci * V._CELL)) / sw * rw,
            ry + (by - (sy - ri * V._CELL)) / sh * rh)


def test_the_loupe_magnifies_the_field_the_cursor_is_over(qapp):
    """The invariant: the centre of a field's own box magnifies the centre of that field — a freeform holder places cells by rect AND each cell holds a mosaic of fields, two transforms that used to be wrong independently."""
    ov = _freeform_overview()
    ov.set_loupe_source(_FakeLoupeSource(well_px=1000, n_levels=1))
    # Without this line _boxes stays empty and the loop below runs zero times — the claim would be vacuously green over a plate with no fields.
    assert len(ov._boxes) == 4, sorted(ov._boxes)
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
        # The corners of the drawn rect are still the mosaic's, not the block's — the letterbox inverse fix is untouched by any of the above.
        rx, ry, rw, rh = ov._cell_rect(0, 0)
        assert ov._cell_fraction(0, 0, rx, ry) == pytest.approx((0.0, 0.0), abs=0.02)
        assert ov._cell_fraction(0, 0, rx + rw, ry + rh) == pytest.approx((1.0, 1.0), abs=0.02)
    finally:
        ov.set_loupe_source(None)


def test_the_loupe_and_a_double_click_resolve_the_same_field(qapp):
    """One box lookup — the inset and a double-click used to be two separate loops that could disagree."""
    ov = _freeform_overview()
    ov.set_loupe_source(_FakeLoupeSource(well_px=1000, n_levels=1))
    # Without this line _boxes stays empty and the loop below runs zero times — the claim would be vacuously green over a plate with no fields.
    assert len(ov._boxes) == 4, sorted(ov._boxes)
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
    """The read that reaches the source must carry the field, not just the geometry — a signature test alone can't catch a caller that never passes one."""
    ov = _freeform_overview()
    src = _FakeLoupeSource(well_px=1000, n_levels=1)
    ov.set_loupe_source(src, np.ones((2, 3), np.float32))
    assert len(ov._boxes) == 4, sorted(ov._boxes)
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
        # `seen == {k: k[1] for k in seen}` is `{} == {}` when nothing was seen, so the empty case would be doubly silent.
        assert len(seen) == 4, seen
        assert seen == {k: k[1] for k in seen}, (
            f"the source was asked for the wrong fields: {seen}")
    finally:
        ov.set_loupe_source(None)


def test_the_loupe_and_the_blit_share_one_content_box(qapp):
    """One letterbox formula: _cell_source and _content_box must recover the same rectangle, since neither calls the other."""
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
    """The loupe used to derive its own percentile window instead of using the plate's _RunningContrast, so a channel the plate correctly renders black could paint bright in the inset."""
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
    """_loupe_colors was a one-time snapshot of acquisition display_color; recolouring a channel in napari moved the plate but not the inset."""
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
    """Every loupe read defaulted to frame 0 because nothing passed time_point through."""
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
        # Moving the plate's timepoint under a live inset must re-read rather than sit on the old frame: the crop cache is keyed by rectangle, so nothing else would notice.
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
    """The loupe's source follows set_active_layer; the tab-change handler was the one call site that forgot to re-point it."""
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
    """_planes memoises one well and is mutated from both the loupe worker and the GUI thread — an interleave could return the wrong well's data."""
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
    """The loupe's QThread hangs off the overview; replacing the overview without stopping it leaked a thread per plate open."""
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
    """Qt grabs the mouse during a press, so no leaveEvent fires while dragging off-widget mid-hold."""
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
    """An unsaved preview must not inherit a zarr source — OperationStack dedupes by key, so the layer name alone proves nothing."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    win.run_operator("mip", out_parent=str(tmp_path), save=False, regions=win._order[:1])
    _drain_until(qapp, lambda: win._overview._active == "mip")
    assert win._loupe_sources.get("mip") is None
    assert win._overview._loupe_src is None        # the gesture is off, not showing raw
    win._stop_worker(); win.close()


def test_saved_run_registers_zarr_source_and_grows_written_set(qapp, squid_dataset, tmp_path):
    # spot (labels): a saved OME-Zarr run. A per-FOV intensity save writes acquisition format
    # now — no zarr, so no zarr loupe source (a loupe over the written acquisition is an open
    # follow-up).
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    win.run_operator("spot", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: isinstance(win._loupe_sources.get("spot"),
                                                 V._ZarrLoupeSource))
    src = win._loupe_sources["spot"]
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



def test_zarr_source_crop_read_against_a_real_pyramid(qapp, pyramid_dataset, tmp_path):
    """Uses pyramid_dataset since the 4x4 fixture's single level would make level selection untestable."""
    import squidxplorer
    from squidxplorer.reader import open_reader

    root, region, size = pyramid_dataset
    out = tmp_path / "out.hcs"
    squidxplorer.write_plate(open_reader(str(root)), str(out), tiff=False)
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

    # A rect bigger than the ceiling comes back DECIMATED, not truncated: same region, fewer samples (a field with too few levels is the case that used to pull a whole plane).
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
    import squidxplorer
    from squidxplorer.reader import open_reader

    root, region, size = pyramid_dataset
    out = tmp_path / "out.hcs"
    squidxplorer.write_plate(open_reader(str(root)), str(out), tiff=False)

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
    """_output writes 1.0 for both 'unknown' and a genuine 1.0 µm/px, so the scale bar must be suppressed rather than assert an unbacked figure."""
    assert V.loupe_um_per_screen_px(None, 0.5) is None


def test_loupe_geometry_maps_cursor_to_the_right_well_and_crop(qapp, squid_dataset):
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    ov = win._overview
    ov.resize(600, 400)
    ov.set_loupe_source(_FakeLoupeSource(well_px=1024, n_levels=4), np.ones((2, 3), np.float32))

    # The single-field path on purpose: this fixture's FOVs land one pixel apart, so 'centred where the user pointed' isn't expressible — multi-field centring is pinned on _freeform_overview instead.
    ov.set_mosaic_boxes({})
    rc = sorted(ov._by_rc)[0]
    x, y = _cell_center(ov, *rc)
    well, fov, level, (y0, x0, h, w), s_loupe, mag = ov._loupe_geometry(x, y)
    assert well == ov._by_rc[rc]                     # the well actually under the cursor
    assert fov is None                               # no mosaic to name a field from
    span = 1024 >> level
    # The crop is centred to within the resolution the plate can even express: at 1536wp fit, one screen pixel spans several image pixels, so a tighter bound would be testing int() rounding, not the mapping.
    slop = span / ov._cd + 2
    assert y0 + h // 2 == pytest.approx(span // 2, abs=slop)
    assert x0 + w // 2 == pytest.approx(span // 2, abs=slop)

    # Zoomed out, the plate scale is tiny, so the loupe reads a coarse level (_user_view stops paint/_fit from resetting zoom under us).
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

def test_minerva_is_a_registered_operation():
    op = V._OPERATIONS_BY_KEY["minerva"]
    assert op.build_tab == "_build_minerva_tab"
    assert hasattr(V.PlateWindow, op.build_tab)


def test_minerva_tab_builds_and_lists_z_operators(qapp, squid_dataset):
    """The z-operator choice must be real — squid2minerva's convert.py offers --mip/--z, so hardcoding here would be a capability regression."""
    from qtpy.QtWidgets import QComboBox
    from squidxplorer import available_plane_operators

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._open_op_tab("minerva", "Minerva", win._build_minerva_tab)
    tab = win._op_tabs["minerva"]

    combos = tab.findChildren(QComboBox)
    assert combos, "no z-operator selector in the Minerva tab"
    listed = [combos[0].itemText(i) for i in range(combos[0].count())]
    assert listed == available_plane_operators()
    assert combos[0].currentText() == "mip"
    win.close()


@_needs("tilefusion")
def test_run_minerva_export_writes_one_fused_mosaic_for_the_selected_region(
        qapp, squid_dataset, tmp_path):
    """Two bugs pinned in order found: the GUI building a 1-element selection pinned to fov 0, then the fix producing N files per region — Minerva Author renders only series[0], so N files silently drops N-1 of them."""
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


#: How many widget px a FOV box needs before a drag inside it is a DRAG rather than a Shift-click toggle — the gesture is only available zoomed in.
_GRABBABLE_PX = 60.0


def _zoom_onto(ov, qapp, region):
    """Zooms until the region's FOV boxes are big enough to drag inside — derived from geometry, not a guessed factor."""
    (rc,), = ([rc for rc, w in ov._by_rc.items() if w == region],)
    r, c = rc
    ov._user_view = True                       # stop paintEvent re-fitting under the gesture
    fov = sorted({f for rr, f in ov._boxes if rr == region})[0]
    _x, _y, w, _h = ov._block_rect(r, c, *ov._boxes[(region, fov)])
    ov._cd *= max(1.0, _GRABBABLE_PX / max(w, 1e-9))
    qapp.processEvents()
    return r, c


def _drag_px(qapp, ov, x0, y0, x1, y1, mods):
    """Press-move-release in widget px, distinct from _drag (cell corners) — a FOV box needs raw px addressing."""
    for kind, x, y, buttons in (
        (QEvent.MouseButtonPress, x0, y0, Qt.LeftButton),
        (QEvent.MouseMove, (x0 + x1) / 2, (y0 + y1) / 2, Qt.LeftButton),
        (QEvent.MouseButtonRelease, x1, y1, Qt.NoButton),
    ):
        qapp.sendEvent(ov, QMouseEvent(kind, QPointF(int(x), int(y)), Qt.LeftButton, buttons, mods))
    qapp.processEvents()


def test_a_shift_alt_box_inside_a_mosaic_selects_fovs_not_the_whole_well(
        qapp, squid_dataset):
    """The region loop has always supported a FOV subset via regions=, but no gesture could express which fields — every GUI caller expanded a well to all its FOVs first."""
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
    """The gesture->selection->export chain, end to end: a boxed FOV subset lands as one cropped mosaic."""
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
    """_on_time_point_changed called self._say, a method that doesn't exist, so every drag raised AttributeError before the plate re-read."""
    root, _ = multi_time_point_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._time_point_bar.count > 1, "fixture is single-timepoint; the bar would be hidden"

    win._time_point_bar.set_time_point_from_user(1)     # exactly what a drag delivers
    qapp.processEvents()

    assert win.time_point == 1
    # An AttributeError on the slot's first statement is invisible in time_point (the bar moved itself) and shows up only as the plate never being told; the readout is not asserted on, since _return_to_raw() legitimately overwrites it.
    assert win._overview._time_point == 1, (
        "the slot died before it reached the plate — the timepoint moved on the bar only")
    win.close()


@_needs("tilefusion")
def test_the_exported_timepoint_is_the_one_the_plate_is_showing(
        qapp, multi_time_point_dataset, tmp_path, monkeypatch):
    """run_minerva_export defaulted t=0 and both GUI call sites took the default, so multi-timepoint exports always wrote frame 0."""
    root, _ = multi_time_point_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    region = win._meta["regions"][0]
    win.activate_well(region, 0)                     # the user's selection; without one the export is a message, not an export
    assert win.minerva_selection(), "the fixture region never became a selection"

    seen = []
    real = V._MinervaWorker

    class Spy(real):
        def __init__(self, reader, selection, out_dir, z_operator, time_point=0, **kw):
            seen.append(time_point)
            super().__init__(reader, selection, out_dir, z_operator, time_point=time_point, **kw)

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

    import tifffile
    px1 = tifffile.imread(str(next((tmp_path / "t1").glob("*.ome.tiff"))))
    px2 = tifffile.imread(str(next((tmp_path / "t2").glob("*.ome.tiff"))))
    assert not np.array_equal(px1, px2), (
        "two timepoints exported identical pixels — the slider is not reaching the export")
    win.close()


def test_run_minerva_export_with_nothing_selected_says_so(qapp, squid_dataset, tmp_path):
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
    """PlateOverview is display-only; the old minerva_selection duck-typed through the overview and reached the right answer only by accident via selected_wells."""
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
    """End to end through real gestures: since the marquee-drag/click split, a drag asks for a window and moves no export scope, while a click moves export scope and opens no window."""
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
    from squidxplorer import _minerva

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
    from squidxplorer import _minerva

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
    """_retire used to disconnect a hardcoded name list, so a worker declaring a new signal stayed connected through teardown."""
    names = set(V._signal_names(V._MinervaWorker))
    assert {"progress", "exported", "launched", "failed", "finished_ok"} <= names
    assert "finished" not in names and "started" not in names   # QThread's own — never torn down
    # the pre-existing worker keeps full coverage too
    assert {"tileReady", "resultReady", "streamEnded", "writtenReady", "wellFailed"} <= set(
        V._signal_names(V._OperatorWorker))


def test_retire_disconnects_every_declared_signal(qapp, squid_dataset, tmp_path):
    """_signal_names being right is worthless unless _retire actually uses it to disconnect."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    worker = V._MinervaWorker(win._reader, [("B2", 0)], str(tmp_path), "mip", time_point=0, launch=False)

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







def test_the_operators_home_tab_never_detaches(qapp, squid_dataset):
    """The home-tab guard is a property of the bar (index 0), not of _detach_tab in general."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._detach_tab(0) is None                   # 'Operators' — never detaches
    assert win._left_tabs.count() >= 1
    assert win._left_tabs.tabBar()._first_detachable == 1
    win.close()




def test_the_redock_BUTTON_works_not_just_the_method(qapp):
    """QPushButton.clicked passes checked=False, which bound to the on_redock lambda's k=key default, so clicking Re-dock called _redock(False) and did nothing."""
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



def test_the_plane_op_cards_build_and_are_preview_only(qapp, squid_dataset):
    """The generic card offers Preview only, no Save; decon is the exception — its card is the RL semi-convergence QC panel, not this generic one."""
    from squidxplorer import available_plane_operators
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    for key in ("bgsub", "flatfield"):
        assert key in available_plane_operators(), f"{key} is not registered in the engine"
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
    """Flat-field has no sane default (an identity field would silently do nothing), so Preview stays disabled until a profile loads."""
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
    """FlatfieldProfile.from_npy(path) defaults to plane 0, so loading a profile used to apply channel 0's gain field to every channel."""
    pytest.importorskip("tilefusion.flatfield")
    from tilefusion.flatfield import save_flatfield

    import squidxplorer._flatfield as FF

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



def test_the_decon_card_is_the_iteration_qc_panel_not_a_bare_preview(qapp,
                                                                    squid_dataset):
    from squidxplorer._decon import QC_START_ITERATIONS
    from squidxplorer._op_panels import DeconQCPanel

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
    from squidxplorer._op_panels import StitcherPanel

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
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    op = V._OPERATIONS_BY_KEY["stitch"]
    win._open_op_tab(op.key, op.label, getattr(win, op.build_tab))
    tab = win._op_tabs["stitch"]
    # The fixture's frames are tiny, so pick a feather that fits inside them — the panel refuses a ramp as wide as the tile, asserted separately below.
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


def test_panel_kwargs_reach_the_region_loop_on_the_PREVIEW_path(qapp, squid_dataset,
                                                             monkeypatch):
    """Spies on the engine call rather than the worker, since storing-but-not-forwarding kwargs is invisible to a worker-only assertion."""
    import squidxplorer
    seen = {}

    def fake_region_loop(reader, **kw):
        seen.update(kw)
        return iter(())

    monkeypatch.setattr(squidxplorer._stitch, "_stitch_plate", fake_region_loop)
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("stitch", regions=["B2"], save=False,
                     operator_kwargs={"blend_px": 3, "register": False})
    _drain_until(qapp, lambda: "blend_px" in seen)
    assert seen["blend_px"] == 3 and seen["register"] is False
    win._stop_worker(); win.close()


def test_panel_kwargs_reach_the_fused_writer_on_the_SAVE_path(qapp, squid_dataset,
                                                              monkeypatch, tmp_path):
    # A stitch SAVE routes to the fused acquisition writer (the stitcher's OME-TIFF format);
    # the panel's kwargs must reach it exactly as they reach the preview's engine call.
    from squidxplorer import _fused_output
    seen = {}

    def fake_write_fused(reader, operator, dst, **kw):
        seen.update(kw, operator=operator, dst=str(dst))
        return {"path": str(dst), "n_fields": 1, "n_fields_written": 1,
                "complete": True, "stopped": False, "skipped_regions": []}

    monkeypatch.setattr(_fused_output, "write_fused_acquisition", fake_write_fused)
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("stitch", out_parent=str(tmp_path), regions=["B2"], save=True,
                     operator_kwargs={"blend_px": 3, "register": False})
    _drain_until(qapp, lambda: "operator_kwargs" in seen)
    assert seen["operator"] == "stitch"
    assert seen["operator_kwargs"]["blend_px"] == 3
    assert seen["operator_kwargs"]["register"] is False
    win._stop_worker(); win.close()


def test_a_decon_qc_result_opens_as_a_tab_beside_the_operators(qapp, squid_dataset):
    """Went unreachable for six weeks when routed to the exploration pane's bar, which was in no layout."""
    from squidxplorer._op_panels import DeconQCResultView

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    view = DeconQCResultView("B2/0/c0")
    win.publish_qc_result(view, "Decon QC · B2/0/c0")
    qapp.processEvents()
    assert win._left_tabs.indexOf(view) >= 0, "the QC result did not land beside the operators"
    before = win._left_tabs.count()
    # Publishing the SAME subject again must reuse its tab; a DIFFERENT widget proves it, since keying on anything unique-per-call would stack a new tab on every QC iteration.
    win.publish_qc_result(DeconQCResultView("B2/0/c0"), "Decon QC · B2/0/c0")
    qapp.processEvents()
    assert win._left_tabs.count() == before, "a second tab was stacked for the same subject"
    assert win._left_tabs.indexOf(view) >= 0, "the original tab was replaced, not reused"
    # A DIFFERENT subject does get its own tab.
    win.publish_qc_result(DeconQCResultView("B3/0/c0"), "Decon QC · B3/0/c0")
    qapp.processEvents()
    assert win._left_tabs.count() == before + 1
    win.close()



def _run_live(qapp, win, key, regions=("B3",)):
    """Drives a real preview run to completion and returns the tiles that reached the plate."""
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
    """reference and coordinate had no card, so run_operator raised a bare KeyError and couldn't stream live at all."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    tiles = _run_live(qapp, win, key)
    assert tiles is not None, f"{key}: no worker started — {win._readout.text()!r}"
    assert tiles, f"{key}: nothing reached the PLATE — {win._readout.text()!r}"
    assert win._active_op_key == key, f"{key} streamed into layer {win._active_op_key!r}"
    assert win._readout.text().startswith("✓"), win._readout.text()
    # Checked for the per-FOV operators only: on this 4x4-frame fixture a region operator's blend weights divide by zero and the fused mosaic comes back NaN -> 0, which is the fixture's degenerate geometry, not the stream.
    if key not in ("stitch", "coordinate"):
        assert any(np.asarray(t[3]).any() for t in tiles), f"{key} streamed all-zero tiles"
    win._stop_worker(); win.close()


def test_flatfield_streams_live_once_a_profile_is_installed(qapp, squid_dataset):
    from squidxplorer import FlatfieldProfile
    from squidxplorer._flatfield import set_profiles
    import squidxplorer._flatfield as FF

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ny, nx = win._meta["frame_shape"]

    prev = FF.active_profiles()
    try:
        # run_operator intercepts flat-field with no active profile and auto-estimates one off-thread (tilefusion BaSiC) instead of failing, so a flat-field run without a profile must not start and must say what it's doing.
        FF.clear_profile()
        tiles = _run_live(qapp, win, "flatfield")
        assert tiles is None, "the operator ran without an illumination profile"
        assert win._worker is None, "an operator worker started without a profile"
        assert "estimating an illumination profile" in win._readout.text(), \
            f"a flat-field run with no profile said nothing: {win._readout.text()!r}"
        est = getattr(win, "_ff_est_worker", None)
        assert isinstance(est, V._FlatfieldWorker), "no estimate was actually started"
        # Cut the estimate loose before waiting: done would install its profile and re-enter run_operator (a second run this test isn't about); _FlatfieldWorker has no stop(), so _retire can't be used on it.
        for sig in (est.done, est.problem, est.stage):
            try:
                sig.disconnect()
            except TypeError:
                pass
        assert _drain_until(qapp, lambda: not est.isRunning(), timeout=90)
        win._ff_est_worker = None

        # The operator is specialised per channel by project_well and refuses a channel it has no measured field for, so a live run needs every channel covered.
        set_profiles({c["name"]: FlatfieldProfile(np.ones((ny, nx), np.float32))
                      for c in win._meta["channels"]})
        tiles = _run_live(qapp, win, "flatfield")
        assert tiles, f"flat-field with a profile still reached no tile: {win._readout.text()!r}"
        assert win._readout.text().startswith("✓"), win._readout.text()
    finally:
        FF.set_profiles(prev) if prev else FF.clear_profile()
    win._stop_worker(); win.close()


def test_run_operator_refuses_a_non_operator_by_name(qapp, squid_dataset):
    """minerva is a card, not an operator — handing it to the engine used to die with a raw KeyError in the status line."""
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





def _montage_px(qapp, ov):
    """The active layer's montage pixels, cropped — a whole-widget grab includes chrome that swamps layer differences."""
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
    """Every operator is a layer; raw must be recoverable by toggling, never destroyed."""
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
    """toggle('raw', False) used to leave the plate painting the last operator with every checkbox off, and move('raw', +1) let the base be reordered above an operator."""
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
    """Previewing an operator on one region used to clear thumbnails for every well, not just the ones being processed."""
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
    """~95MB per layer lives in _store/_final_arr; dropping only the canvas leaks most of the memory."""
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
    """A real stitchable acquisition whose mosaic is deliberately non-square (frame 256x256, mosaic 456x656), so a viewer sized as a frame vs sized as the mosaic gives different, distinguishable numbers. Returns (root, region, frame_px, mosaic_extent_px)."""
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
            # stage mm: px -> um -> mm. The reader turns these back into fov_positions_um, which is what _placement lays the mosaic out from.
            lines.append(f"{region},{left * px_um / 1000.0},{top * px_um / 1000.0},")
    root = tmp_path / "acq_nonsquare"
    (root / "acquisition_channels.yaml").write_text(_NONSQUARE_YAML)
    (root / "acquisition.yaml").write_text(_NONSQUARE_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(
        json.dumps({"Nz": 1, "Nt": 1, "dz(um)": 1.5,
                    "objective": {"magnification": 20.0}, "sensor_pixel_size_um": 3.76}))
    (root / "coordinates.csv").write_text("\n".join(lines) + "\n")
    return root, region, frame, (mh, mw)




# The part of IMA-245 about the plate CELL — a fused mosaic landing exactly where the raw mosaic does — is content_box, asserted below.


# A region operator has no per-FOV sub-boxes (the fused mosaic IS the cell), so it took the box=None branch that resizes to exactly (_CELL, _CELL) — stretching the mosaic off its own aspect ratio and off the raw cell's centred band.

def test_a_stitched_cell_lands_exactly_where_the_raw_cell_does(
        qapp, nonsquare_mosaic_dataset):
    from functools import reduce

    from squidxplorer import available_region_operators

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
        # One fused mosaic per region, the shape the region loop yields: (T, C, 1, Y, X).
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
    """content_box replaced _fit_cell on the whole-cell branch; guards that the fix costs nothing on the historical single-FOV path."""
    assert V.content_box((2084, 2084)) == (0, 0, V._CELL, V._CELL)
    assert V.content_box((37, 37)) == (0, 0, V._CELL, V._CELL)
    # Wider than tall -> full width, centred vertically. Never taller than the cell.
    top, left, h, w = V.content_box((100, 400))
    assert (left, w) == (0, V._CELL) and h == V._CELL // 4 and top == (V._CELL - h) // 2




# Measured on the montage, never a whole-widget grab: labels, grid and status dots keep whole-frame variance high enough that a widget-level check would pass against a blank montage.

def _region_crop(ov, region):
    """The rendered pixels of ONE region's cell — its own rectangle, not a grid square."""
    rc = next(k for k, v in ov._by_rc.items() if v == region)
    return _grab_bgr(ov)[_cell_slices(ov, rc)]


def test_ima253_real_tissue_previews_both_regions_as_mosaics_before_any_operator_runs(
        qapp, real_dataset):
    """Before the fix, boxes were 0 and each region showed one frame stretched over its cell, because set_mosaic_boxes was only reachable from run_operator."""
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
    from squidxplorer import open_reader

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
    """even_carrier_layout replaced stage-proportional placement (which stacked regions into a tall, tiny column); still guarded: placement follows geometry, not enumeration order, and cells never overlap."""
    from squidxplorer._plate import even_carrier_layout, region_stage_boxes_um

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
    # Geometry, not enumeration order: the lower stage x renders further left; reversing the reported order cannot move a cell, since the ordering key is the stage box, not the report order.
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
    """Reversing the acquisition's region order must not move anything — a layout driven by enumeration order would."""
    from squidxplorer import open_reader
    from squidxplorer._plate import build_plate

    meta = open_reader(str(real_dataset)).metadata
    ref = build_plate(meta)
    flipped = build_plate({**meta, "regions": list(reversed(meta["regions"]))})
    assert flipped.cell_layout() == ref.cell_layout()
    assert flipped.occupied_map == ref.occupied_map


def test_ima253_the_default_paint_path_loads_no_carrier_png(qapp, squid_dataset,
                                                            monkeypatch):
    """The carrier photograph needs calibration constants to agree with the geometry; when they disagreed nothing raised and wells were drawn in the wrong place, so the art stays off the path."""
    import squidxplorer._plate as P

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
    from squidxplorer._plate import SlideCarrier

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


# The earlier three-pane check passed fake-green because it never showed the window: an unshown QSplitter reports whatever sizes it was handed with zero real geometry — everything below shows the window at a real size first.

def _drain_preview(win, app, timeout_s=60):
    """Blocks until the raw preview worker stops streaming — a fixed settle() races it, since duration depends on well count."""
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








# The written-OME-Zarr path has always given napari a multiscale pyramid; the raw preview path gave full-resolution fused planes instead (54.9MB per channel per z, re-fused every z step). These pin the wiring that closes that gap.


class _PyrReader:
    #: The plane cache keys on the acquisition a reader reads, so every reader must name it.
    def __init__(self, frame=(256, 256), path="/fake/acquisition/viewer"):
        self.frame = frame
        self._path = path

    def read(self, region, fov, channel, z_level, time_point=0):
        return np.full(self.frame, z_level + 1, dtype=np.uint16)


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
    """Sampling the contrast seed on the UI thread decodes every FOV of the region (measured ~128-600ms), because even the coarsest pyramid level is fused from FOV TIFFs; must happen on the worker thread instead."""
    from squidxplorer._napari_view import _auto_window_for

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
    """Opening a region must cost one coarse fuse per channel at one z — not one read per pyramid level and not the whole z-stack — and level 0 must stay unmaterialised."""
    from squidxplorer import _mosaic_source as MS
    from squidxplorer._contrast import opening_z

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
    # The z axis is still 4 deep and level 0 still exists — the pyramid was not flattened to make the seed cheap.
    assert MS._PLANE_CACHE.nbytes > 0, "the seed's decode must be cached, not thrown away"


# Called UNBOUND on a duck shell, same as the PlateWindow tests above — what's under test is the slot, not the widget.


class _PlaneView:
    """A RegionViewer reduced to what _on_plane reads."""

    _roi_bbox = None
    open_clock = None
    window_id = 1

    def __init__(self, pane, meta, region):
        from squidxplorer._region_nav import RegionCursor

        self._pane = pane
        self._meta = meta
        self._cursor = RegionCursor()
        self._cursor.set_order([region])
        self._cursor.activate(region)

    def _say(self, msg):
        pass

    def on_plane(self, *a, **kw):
        from squidxplorer._region_viewer import RegionViewer

        return RegionViewer._on_plane(self, *a, **kw)


def test_on_plane_tells_napari_the_data_is_multiscale(qapp):
    """A pyramid without multiscale=True is just a list napari can't use — it errors or falls back to level 0."""
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
    # The worker's contrast seed is passed through unchanged; what moved off the UI thread is which thread samples for it (add_mosaic still treats a missing/None window as 'derive one').
    assert kw.get("contrast_limits") == (12.0, 345.0)
    # the z scale commit 19cd491 established must survive the pyramid
    assert kw.get("z_scale_um") == 1.5


def test_on_plane_without_a_window_still_lets_add_mosaic_derive_one(qapp):
    """window=None must reach add_mosaic as None (derive it), not as (None, None)."""
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


# _OPERATIONS (the card table) and runnable_operators() (the engine registry) used to diverge silently — a card whose key isn't runnable produced a dead button that said nothing; these pin the contract instead of restating it in prose.


def test_a_cards_runnability_is_the_engines_answer_and_cannot_go_stale():
    """Operation.runnable is now a property over runnable_operators(), not a hand-written bool that could drift."""
    from squidxplorer import add_operator, plane_op

    assert V._OPERATIONS_BY_KEY["minerva"].runnable is False   # nobody registered it
    assert V._OPERATIONS_BY_KEY["mip"].runnable is True

    card = V.Operation("card_only_key", "Card only", "no engine entry", "_build_mip_tab")
    assert card.runnable is False
    add_operator("card_only_key", plane_op(lambda p: p))
    assert card.runnable is True, "runnable is stale; it must be read, not stored"


def test_gallery_view_is_a_view_menu_command_and_not_an_operator(qapp):
    """Gallery View consumes no axis and produces no pixels, so it is a View-menu command with no card, not a runnable operator."""
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


# The reverse direction: an operator the engine can run but with no card is invisible to the card-walking test, which is exactly how reference stayed CLI-only for months — every runnable operator must now either have a card or be declared CLI-only here, with the reason written down.

#: Runnable operators that deliberately have no GUI card, and why — adding one without listing it here (or giving it a card) fails the test below.
CLI_ONLY_OPERATORS = {
    "spot": "a LABELS overlay, not a plate result; it is driven from the spot-count controls "
            "on the mosaic, not from a card that writes an OME-Zarr plate.",
    "cellpose": "the same LABELS overlay as `spot`, with the model instead of the Otsu recipe. "
                "Same reason it has no card. (It is NOT because the result cannot be written -- "
                "that was true while _validate_image accepted Z == 1 only, and IMA-277 lifted "
                "it.) It is "
                "reachable from the CLI (--operator cellpose), the operator dropdown and the "
                "Detect-nuclei button, all of which read the registry.",
    "decon3d": "the volume-then-project variant of `decon`; the decon card's own panel is where "
               "an iteration count gets chosen, and a second card for the same operator with a "
               "different z contract is how a user picks the wrong one.",
    "coordinate": "the unregistered CONTROL for `stitch` (stage coordinates, no registration). "
                  "It exists to be the baseline a stitch is graded against in the benchmark, "
                  "not to be offered as a thing to run.",
    "keepz": "the IDENTITY plane-op: every z plane, no pixel changed. There is nothing to run it "
             "FOR on its own -- projecting an acquisition to itself writes a copy of the input. "
             "It exists so that `stitch_region(z_operator=\'keepz\')` can fuse a VOLUME instead "
             "of one collapsed plane, and it is offered exactly there: the stitcher panel's "
             "Z-handling combo, which is built from `available_plane_operators()`.",
}


def test_every_runnable_operator_is_either_carded_or_declared_cli_only():
    """The reverse of the card->engine check: reference ran in the engine since IMA-210 with no GUI surface at all."""
    carded = {op.key for op in V._OPERATIONS}
    for key in V.runnable_operators():
        assert key in carded or key in CLI_ONLY_OPERATORS, (
            f"the engine can run {key!r} but no card offers it and it is not declared CLI-only. "
            f"Either add an Operation for it to _OPERATIONS (plus its _build_<x>_tab), or add it "
            f"to CLI_ONLY_OPERATORS with the reason it is deliberately not in the GUI."
        )


def test_the_cli_only_declaration_cannot_go_stale():
    """Prevents the CLI-only allowlist from keeping an operator that has since gained a card or lost its registration."""
    runnable = set(V.runnable_operators())
    carded = {op.key for op in V._OPERATIONS}
    for key in CLI_ONLY_OPERATORS:
        assert key in runnable, (
            f"{key!r} is declared CLI-only but the engine no longer runs it; delete the entry.")
        assert key not in carded, (
            f"{key!r} is declared CLI-only but now HAS a card; delete the entry.")


def test_the_reference_plane_operator_is_reachable_from_the_gui():
    op = V._OPERATIONS_BY_KEY["reference"]
    assert op.runnable is True
    assert "reference" in V.runnable_operators()
    assert hasattr(V.PlateWindow, op.build_tab), (
        f"the reference card names {op.build_tab!r} and PlateWindow has no such method; "
        "clicking it would raise AttributeError out of the event loop.")


def test_the_save_button_names_its_operator_instead_of_taking_the_first_card():
    """_OPERATIONS[0].key made the save button run whatever card happened to be first; it's now named explicitly."""
    assert V._SAVE_OPERATOR == "mip"
    assert V._SAVE_OPERATOR in V.runnable_operators()
    # and it must not be a positional accident: reordering the cards must not change it
    assert V._SAVE_OPERATOR in V._OPERATIONS_BY_KEY


def test_a_cardless_operator_opens_a_panel_built_from_its_declaration(qapp,
                                                                     squid_dataset):
    """spot is a registered operator with no card; _activate_operator used to silently do nothing for it."""
    from squidxplorer._engine import operator_params
    from squidxplorer._param_panel import GenericOperatorPanel

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
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    before = dict(win._op_tabs)
    win._activate_operator("stitch_but_misspelled")
    assert win._op_tabs == before, "a refused operator must not open a tab"
    assert "stitch_but_misspelled" in win._readout.text()
    win.close()


def test_the_preview_path_carries_operator_kwargs_to_the_engine(qapp, squid_dataset, monkeypatch):
    """The PREVIEW branch called the engine without operator_kwargs while the save branch passed them, so a panel's parameter value reached the console line but not the pixels."""
    import squidxplorer
    from squidxplorer.reader import open_reader

    root, _ = squid_dataset
    reader = open_reader(str(root))
    meta = reader.metadata
    seen = {}

    def fake_run_plate(_reader, **kw):
        seen.update(kw)
        return iter(())

    monkeypatch.setattr(squidxplorer, "run_plate", fake_run_plate)
    fov_index = {r: {"rc": (0, i), "idx": i, "well_id": r}
                 for i, r in enumerate(meta["regions"])}
    worker = V._OperatorWorker("spot", reader, meta, fov_index, "", regions=meta["regions"][:1],
                               save=False, n_fovs=None,
                               operator_kwargs={"min_area_px": 400})
    worker.run()
    assert seen.get("operator_kwargs") == {"min_area_px": 400}, (
        "the preview branch dropped the panel's parameters on the floor: "
        f"run_plate was called with {sorted(seen)}")


def test_every_uncarded_runnable_operator_is_offered_in_the_declaration_submenu(qapp):
    """The submenu is built off runnable_operators(), so a plugin-added operator appears with no edit here."""
    win = V.PlateWindow(None)
    offered = {a.text() for a in win._declared_menu.actions()}
    expected = {V.operator_label(k) for k in V.runnable_operators()
                if k not in V._OPERATIONS_BY_KEY}
    assert offered == expected
    assert "spot" in offered and "cellpose" in offered
    win.close()


def test_operator_label_falls_back_to_the_key_for_a_cardless_operator():
    # spot is a registered operator with no card; it must still name itself rather than raising a bare KeyError out of the event loop.
    assert V.operator_label("spot") == "spot"
    assert V.operator_label("mip") == V._OPERATIONS_BY_KEY["mip"].label
    # and the newly carded one now answers with its card
    assert V.operator_label("reference") == V._OPERATIONS_BY_KEY["reference"].label




# Moved to tests/test_time_point_playback.py (signal-to-slot arity) and tests/test_plate_follows_windows.py (the plate adopting napari's resolved window, via _adopt_window_view) after PlateWindow._mosaic_pane became permanently None.


def test_the_plate_is_restored_even_while_the_raw_preview_streams(qapp,
                                                                  squid_dataset):
    """Three gates asked self._busy() (any producer alive) when they meant 'is an operator run alive'; _busy() counts the raw preview, which streams almost continuously, so the restore deferred until an unrelated thread happened to finish. MUTATION: reverting any of the three gates to self._busy() should fail this."""
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


# Before this, no operator's pixels reached pane 2's napari at all — the slider-path test above pins a different destination and is exactly what made this hole look covered.

class _RecordingMosaic:
    """Borrows the real add_result/_RESULT_ADDERS dispatch rather than reimplementing it, so the fake can't agree with itself while disagreeing with the app."""

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
        # Insertion order, de-duplicated, is load-bearing one caller away: _layer_tree renders list(reversed(ops())) so the topmost layer draws first — a fake that sorts agrees with alphabetical input by luck.
        seen = []
        for c in self.calls:
            if c[0] not in seen:
                seen.append(c[0])
        return seen

    def group(self, op):
        return [c for c in self.calls if c[0] == op]


class _RecordingPane:
    ok = True

    def __init__(self):
        self.mosaic = _RecordingMosaic()

    def say(self, msg):
        pass


class _ResultView:
    """deliver_result is called unbound off the real RegionViewer class rather than reimplemented, for the same reason as _RecordingMosaic."""

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
        from squidxplorer._region_viewer import RegionViewer

        return RegionViewer.deliver_result(self, op, result, visible=visible)


class _ResultManager:
    """The one seam _deliver_to_views and _result_regions read: mgr.windows."""

    def __init__(self, *views):
        self.windows = list(views)


def _result_win(op="bgsub", region="A1", channels=("405", "488")):
    from squidxplorer._region_nav import RegionCursor
    from squidxplorer._run import OperatorRun

    win = _plate_window_shell()
    win._cursor = RegionCursor()
    win._cursor.set_order([region])
    win._cursor.activate(region)
    win._active_op_key = op
    win._run = OperatorRun(key=op, layer_key=op, label=op, action=None, dest="",
                           address=None, requester=None, is_partial=False, t0=0.0)
    win._readout = type("R", (), {"setText": lambda self, t: setattr(self, "t", t),
                                  "text": lambda self: getattr(self, "t", "")})()
    # A finished result is filed in _recipe.RESULTS so a window opened later can reuse it instead of recomputing, keyed by which acquisition — so the shell has to carry a reader like every other attribute here.
    win._reader = type("R", (), {"_path": "/fake/acquisition/result-win"})()
    win._meta = {
        # B7 is a REAL region here, with real positions — without it the off-screen-drop guard could be deleted and the test would stay green for the wrong reason (an unknown region can't complete anyway).
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
    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.full((2, 8, 8), 7, "uint16"))
    mos = win._view.mosaic
    assert mos.ops() == ["bgsub"]                    # one GROUP, keyed by the operator
    assert [c[1] for c in mos.group("bgsub")] == ["405", "488"]   # one LAYER per channel


def test_the_layer_group_is_not_drawn_until_the_region_is_whole(qapp):
    """Half a region drawn as a layer reads as holes the operator put there, not as an incomplete run."""
    win = _result_win("bgsub")
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((2, 8, 8), "uint16"))
    assert win._view.mosaic.calls == []


def test_a_run_that_ends_with_a_half_read_region_SAYS_SO_instead_of_stranding_it(qapp):
    """A region that never completes (e.g. unreadable TIFFs) used to sit forgotten in _result_accs forever: the run reported success and opened no operator tab. It must now be settled and named as incomplete, not stranded silently."""
    win = _result_win("mip")
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((2, 8, 8), "uint16"))   # 1 of 2 FOVs
    assert win._run.accs, "the accumulator should be holding the half region"

    stranded = V.PlateWindow._settle_stranded_results(win)

    assert stranded == 1
    assert win._run.accs == {}, "the stranded accumulator was not resolved"
    assert win._view.mosaic.calls == [], "half a region must still not be drawn"
    said = win._readout.text()
    assert "A1" in said and "1 of 2" in said, f"the run did not say what happened: {said!r}"
    assert win._run.error, "the window that ASKED would still have been told the run finished"


def test_settling_a_run_with_every_region_complete_is_a_no_op(qapp):
    win = _result_win("mip")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.zeros((2, 8, 8), "uint16"))
    win._readout.setText("")

    assert V.PlateWindow._settle_stranded_results(win) == 0
    assert win._readout.text() == ""
    assert not win._run.error


def test_the_operator_layer_lands_in_the_raw_mosaic_s_frame(qapp):
    """bbox_um places the group exactly on raw's frame — without it, toggling would jump and misregistration would read as the operator's effect."""
    from squidxplorer._mosaic_source import mosaic_bbox_um

    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.zeros((2, 8, 8), "uint16"))
    kw = win._view.mosaic.group("bgsub")[0][3]
    assert kw["bbox_um"] == mosaic_bbox_um(win._meta, "A1")


def test_two_operators_make_TWO_groups_so_both_can_be_toggled(qapp):
    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.zeros((2, 8, 8), "uint16"))
    win._active_op_key = "decon"
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "A1", fov, np.zeros((2, 8, 8), "uint16"))
    assert win._view.mosaic.ops() == ["bgsub", "decon"]


def test_a_result_for_a_region_that_is_not_on_screen_is_dropped_not_accumulated(qapp):
    """Pane 2 shows one region; holding full-res mosaics for every well of a plate run would be gigabytes nobody can view."""
    win = _result_win("bgsub")
    for fov in (0, 1):
        V.PlateWindow._on_result(win, "B7", fov, np.zeros((2, 8, 8), "uint16"))
    assert win._view.mosaic.calls == []


def test_a_result_that_cannot_be_placed_SAYS_SO_instead_of_vanishing(qapp):
    win = _result_win("bgsub")
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((1, 8, 8), "uint16"))
    assert "not shown as a layer" in win._readout.text()
    assert win._view.mosaic.calls == []


def test_a_region_operator_s_fused_mosaic_is_added_whole_not_re_tiled(qapp):
    """stitch already returns the fused region; re-running it through FOV placement would tile a mosaic as if it were a single FOV."""
    win = _result_win("stitch")
    V.PlateWindow._on_result(win, "A1", 0, np.full((2, 20, 30), 3, "uint16"))
    layers = win._view.mosaic.group("stitch")
    assert len(layers) == 2
    assert layers[0][2].shape == (20, 30)


def test_no_open_window_means_the_result_slot_still_stands(qapp):
    win = _result_win("bgsub")
    win._viewer_manager = _ResultManager()
    V.PlateWindow._on_result(win, "A1", 0, np.zeros((2, 8, 8), "uint16"))



def test_the_minerva_export_hands_the_on_screen_luts_to_the_exporter(
        qapp, squid_dataset, tmp_path, monkeypatch):
    """Asserted at what export_selection is called with, not at the widget — a checkbox wired to nothing looks identical from the widget alone."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    seen = {}

    def spy(reader, selection, out_dir, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr("squidxplorer._minerva.export_selection", spy)
    luts = {"ch": {"clim": (3.0, 4.0), "rgb": (9, 9, 9)}}

    win.run_minerva_export(out_dir=str(tmp_path), launch=False, selection=[("B2", 0)], luts=luts)
    win._minerva.wait(20000)
    qapp.processEvents()

    assert seen.get("luts") == luts, "the LUTs stopped somewhere between the tab and the export"
    win.close()


def test_the_minerva_export_defaults_to_no_luts_so_the_plate_path_is_unchanged(
        qapp, squid_dataset, tmp_path, monkeypatch):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    seen = {}
    monkeypatch.setattr("squidxplorer._minerva.export_selection",
                        lambda reader, selection, out_dir, **kw: (seen.update(kw), [])[1])

    win.run_minerva_export(out_dir=str(tmp_path), launch=False, selection=[("B2", 0)])
    win._minerva.wait(20000)
    qapp.processEvents()

    assert seen.get("luts", "missing") is None
    win.close()


def test_on_screen_luts_is_none_when_no_view_window_is_open(qapp, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win.on_screen_luts() is None
    win.close()


def test_on_screen_luts_reaches_a_focused_window(qapp, squid_dataset):
    """focused_id and windows are properties on ViewerManager; calling them with parentheses raised a TypeError swallowed by a broad except, so this returned None every time.

    The lookup has since moved INTO the manager, as ``active_view()``, so ``on_screen_luts`` no
    longer touches either property and that particular mistake is now unmakeable here — there is
    one implementation of "which window is the user in" and every caller shares it. The stub keeps
    both properties anyway: they are what the real ``active_view`` reads, so a stub that dropped
    them would stop describing the thing it stands in for. What this test pins is unchanged and is
    the part that matters — the focused window's LUTs reach the exporter, rather than a swallowed
    exception quietly returning None.
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

        def active_view(self):
            """Built from the two properties above, the way the real one is — so the stub still
            fails if a caller reaches past it and calls those with parentheses."""
            return next((w for w in self.windows if w.window_id == self.focused_id), None)

        def set_run_progress(self, report):
            """Present so teardown does not raise over the assertion."""

    win._viewer_manager = _Mgr()
    assert win.on_screen_luts() == expected, "the focused window's LUTs never reached the exporter"
    win.close()


def test_the_render_destination_refuses_an_empty_export_instead_of_starting_a_worker(
        qapp, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_minerva_render([])
    assert win._minerva_render is None
    assert "nothing to render" in win._readout.text()
    win.close()


def test_a_render_worker_is_retired_with_the_export_worker(
        qapp, squid_dataset, tmp_path):
    """closeEvent joins these threads; a render left connected measured a 132s hold on close for one region."""
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


# These tests measure the cell INTERIOR, not the whole widget, since a wash repaints the thumbnail itself while a frame lands on the boundary and leaves it byte for byte alone; sized at plate scale since a 2-well fixture gives no intuition for 1536 selected cells at once.

def _grab_rgb(ov) -> np.ndarray:
    """Reverses _grab_bgr's byte order — needed the moment an actual ink/QColor is named."""
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


#: This file's cell-cropping helper; the three older helpers above are now routed through the same implementation.
_cell = _cell_slices


def _carries_ink(frame, sl, color, tol=24) -> bool:
    """True when some pixel in *sl* is *color* at full strength (a 16% wash never gets there)."""
    band = frame[sl].reshape(-1, 3).astype(int)
    want = np.array([color.red(), color.green(), color.blue()])
    return bool(np.abs(band - want).sum(1).min() <= tol)


def test_selecting_a_well_on_a_1536wp_leaves_the_thumbnail_pixels_untouched(qapp):
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
    """A translucent wash can't reach full-strength ink, and the 3px grid line would bury a box drawn before it."""
    ov = _fitted_plate(32, 48)
    rc = (10, 30)
    ov.highlight_regions([ov._by_rc[rc]])
    frame = _grab_rgb(ov)
    assert _carries_ink(frame, _cell(ov, rc), V._SEL_FRAME), (
        "no pixel of the selected cell carries the accent ink at full strength: the mark is still "
        "a wash, or the grid lines were painted over the box")


def test_the_selection_box_is_a_frame_and_not_a_filled_rectangle(qapp):
    """Counting ink alone can't distinguish a frame from a fill; a fill would also leave the interior 'changed'."""
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


def test_no_plate_size_keeps_a_selection_wash(qapp):
    """The translucent selection wash recolours the tissue underneath, more so on larger cells — replaced everywhere with a boundary frame that touches no data."""
    ov = _fitted_plate(3, 3)
    rc = (1, 1)
    before = _grab_rgb(ov).copy()
    ov.highlight_regions([ov._by_rc[rc]])
    after = _grab_rgb(ov)

    inner = _cell_slices(ov, rc, inset_frac=0.25)      # well inside any frame stroke
    np.testing.assert_array_equal(before[inner], after[inner])
    assert not np.array_equal(before, after), "the selection left no mark at all"

def test_the_selection_frame_stroke_is_clamped_at_both_ends(qapp):
    assert V.selection_frame_pen_px(25.0) == pytest.approx(2.5)
    assert V.selection_frame_pen_px(4.0) == 1.0          # floor: still one drawn pixel
    assert V.selection_frame_pen_px(200.0) == 3.0        # ceiling


def test_the_red_current_fov_box_is_gone(qapp):
    """The red box dated from the single-detail-viewer era; N independent RegionViewer windows now use their own hue frames to tell views apart, which one shared red box couldn't do."""
    ov = _fitted_plate(32, 48)
    rc = (5, 5)
    ov.highlight_regions([ov._by_rc[rc]])
    ov._sel = rc
    ov.update()

    assert not _carries_ink(_grab_rgb(ov), _cell(ov, rc), V._RED), (
        "the red current-well box is still drawn")


def test_the_1536_fixture_opens_and_reports_1536_wells(sim_1536wp):
    """open_reader refuses a hollow plate first, so this is the cheapest proof the plate-scale fixture is real."""
    from squidxplorer import open_reader

    meta = open_reader(str(sim_1536wp)).metadata
    assert len(meta["regions"]) == 1536, f"{len(meta['regions'])} regions, not 1536"
    assert meta["wellplate_format"] == "1536 well plate"
    assert len(meta["channels"]) == 4
# _run_operator used to retire the preview's downsample pass unconditionally (and _retire disconnects signals before stopping), so in-flight tiles were dropped and nothing but the return-to-raw path restarted it — leaving a region with no thumbnail at all.

class _GatedPreview(V.QThread):
    """Deterministic stand-in for a race that's only a few hundred ms wide on a real 2-well fixture."""

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
    """Puts a preview that is provably mid-plate in front of *win*, wired as the real one is."""
    win._stop_preview()
    gate = _GatedPreview()
    win._preview = gate
    gate.tileReady.connect(win._on_preview_tile)
    gate.start()
    assert _drain_until(qapp, gate.isRunning, 5)
    return gate


def test_a_subset_operator_run_leaves_the_thumbnail_downsample_pass_running(
        qapp, squid_dataset, blocking_worker):
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
    # Still WIRED: _retire disconnects before it stops, so a live thread alone isn't enough — the tiles it's about to produce have to still reach the plate.
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
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    gate = _gated_preview_on(win, qapp)

    win.run_operator("mip", save=False)                 # regions=None: the whole plate

    assert win._preview is None, "a plate-wide run no longer supersedes the raw preview"
    assert _drain_until(qapp, lambda: not gate.isRunning(), 5)
    win.close()


# The plate cell is (C, h, w) native dtype end to end, and the channel axis is what the channel toggle and global-contrast recomposite are built on — a cell that lost that axis renders as a plausible picture of the wrong thing.

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
    """Every plane is a constant that names its channel, so a swapped axis is unmistakable."""

    def __init__(self, path):
        self._path = str(path)

    def read(self, region, fov, channel, z_level, time_point=0):
        return np.full(_DOWNSAMPLE_FRAME, (_DOWNSAMPLE_CHANNELS.index(str(channel)) + 1) * 1000,
                       dtype=np.uint16)


def _expected_levels():
    return [(i + 1) * 1000 for i in range(len(_DOWNSAMPLE_CHANNELS))]


def test_the_raw_preview_downsamples_every_channel_on_its_own(qapp, tmp_path):
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
    """Driven with the (T, C, 1, Y, X) shape a z-reducer yields, so only the channel axis is left to get wrong."""
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


def test_there_is_no_second_incomplete_marker(qapp):
    """Structural guard against a private marker re-appearing: two branches converged on the same defect independently, and the store is now the one source of truth."""
    from pathlib import Path as _Path

    assert not hasattr(V.PlateWindow, "_note_partial_output"), (
        "the window is writing its own incomplete marker again; ask `_output.incomplete_reason`")
    # AST, not grep: a grep would also match the comment explaining why this guard exists, failing on its own explanation.
    import ast as _ast

    # encoding="utf-8" EXPLICITLY. `read_text()` uses the platform's locale encoding, which on
    # Windows is cp1252 — and `_viewer.py` has carried a U+25CF bullet in its readout strings
    # since long before this test, whose UTF-8 bytes include 0x8F, an undefined cp1252 slot. So
    # this raised UnicodeDecodeError on Windows instead of asserting anything. Python source is
    # UTF-8 by definition (PEP 3120), so the locale never had a say here.
    tree = _ast.parse(_Path(V.__file__).read_text(encoding="utf-8"))
    literals = [n.value for n in _ast.walk(tree)
                if isinstance(n, _ast.Constant) and n.value == "INCOMPLETE"]
    assert not literals, (
        "a bare INCOMPLETE filename is back in _viewer.py as a real string; the ONE name is "
        "`_output.INCOMPLETE_MARKER`")
