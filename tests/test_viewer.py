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
from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QPushButton, QSlider, QSpinBox, QWidget,
)

from squidmip import _viewer as V  # noqa: E402


class _StubDetail(QWidget):
    """Stand-in for the embedded ndviewer_light detail viewer.

    Records the push API (start_acquisition / register_image / go_to_well_fov) so we can assert
    the seam WITHOUT constructing ndviewer's real vispy/GL widget — which segfaults offscreen
    under pytest's PySide6/napari-loaded environment (a Qt-binding conflict, not a code bug).
    """

    def __init__(self):
        super().__init__()
        self._fov_labels = []
        self._fov_slider = QSlider(Qt.Horizontal, self)
        self.registered = []
        self.arrays = []          # (t, idx, z, ch) of every register_array push (IMA-205)
        self.nav = []
        self.acquisitions = []    # one entry per start_acquisition — the slider's label list

    def start_acquisition(self, channels, nz, h, w, labels):
        self._fov_labels = list(labels)
        self._fov_slider.setMaximum(max(0, len(labels) - 1))
        self.acquisitions.append(list(labels))

    def register_image(self, t, idx, z, ch, path, page_idx=0):
        self.registered.append((t, idx, z, ch, path))

    def register_array(self, t, idx, z, ch, plane):
        """Record computed-well pushes. The real ndviewer indexes its slider by ``idx``, so a push
        whose idx exceeds the current label list would land out of range — recording it here is what
        makes the global->subset remap assertable at all (the push path was previously unobserved)."""
        self.arrays.append((t, idx, z, ch))

    def go_to_well_fov(self, well_id, fov):
        self.nav.append((well_id, fov))
        return True


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
                         regions=None):
        captured.update(projector=projector, tiff=tiff, out_dir=str(out_dir), regions=regions)
        return {"plate": str(out_dir), "levels": 1}      # no wells — we only assert the dispatch
    monkeypatch.setattr(squidmip, "write_plate", fake_write_plate)

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: "projector" in captured)
    assert captured["projector"] == "mip"
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


def test_channel_bar_drives_the_plate(qapp, stub_detail, squid_dataset):
    # The UI seam: one row per channel, checkbox -> mask, slider -> latched manual window, auto ->
    # back to the running one. The bar is built from the acquisition's RESOLVED display_color.
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    bar = win._channel_bar
    assert bar is not None and len(bar._rows) == len(win._meta["channels"])

    box, s_lo, s_hi = bar._rows[0]
    box.setChecked(False)
    assert win._overview._mask[0] == False        # noqa: E712 — numpy bool, not python bool
    box.setChecked(True)
    s_hi.setValue(s_hi.value() // 2)              # dragging a handle latches the channel manual
    assert win._overview._contrast.is_manual(0)
    bar._auto(0)
    assert not win._overview._contrast.is_manual(0)
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
    assert win._left_tabs.currentWidget() is tab
    win.close()


def test_open_exploration_tab_same_selection_focuses_not_duplicates(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n0 = win._left_tabs.count()
    k1 = win.open_exploration_tab(["B2", "B3"])
    k2 = win.open_exploration_tab(["B3", "B2"])       # same SET, different order
    assert k1 == k2
    assert win._left_tabs.count() == n0 + 1           # one tab, not two
    k3 = win.open_exploration_tab(["B3"])             # a different set DOES open another
    assert k3 != k1
    assert win._left_tabs.count() == n0 + 2
    win.close()


def test_open_exploration_tab_rejects_empty_and_unknown(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n0 = win._left_tabs.count()
    assert win.open_exploration_tab([]) is None
    assert "empty selection" in win._readout.text().lower()
    assert win.open_exploration_tab(["ZZ99"]) is None          # named, not a raw KeyError
    assert "not in this acquisition" in win._readout.text().lower()
    assert win._left_tabs.count() == n0
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
    idx = next(i for i in range(win._left_tabs.count())
               if win._left_tabs.widget(i) is win._op_tabs[key])
    win._close_op_tab(idx)                                       # close it, possibly mid-run
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
    n = win._left_tabs.count()
    idx = next(i for i in range(win._left_tabs.count())
               if win._left_tabs.widget(i) is win._op_tabs[key])
    win._close_op_tab(idx)
    assert win._left_tabs.count() == n - 1
    assert key not in win._op_tabs
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


def test_tab_switch_repoints_detail_and_home_restores_plate(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B3"])
    qapp.processEvents()
    assert win._detail._fov_labels == ["B3:0"]                  # follows the exploration tab
    assert win._active_exploration is win._op_tabs[key]
    win._left_tabs.setCurrentIndex(0)                           # back to "Process wells"
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
    assert not [i for i in range(win._left_tabs.count())
                if isinstance(win._left_tabs.widget(i), V._ExplorationTab)]
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
    idx = next(i for i in range(win._left_tabs.count())
               if win._left_tabs.widget(i) is win._op_tabs[key])
    win._close_op_tab(idx)                              # close AFTER it finished
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

    tabs = [win._left_tabs.widget(i) for i in range(win._left_tabs.count())
            if isinstance(win._left_tabs.widget(i), V._ExplorationTab)]
    assert len(tabs) == 1, "the shift-drag gesture did not open an exploration tab"
    tab = tabs[0]
    assert tab.regions == ["B3"]                       # scoped to EXACTLY the selected wells
    assert tab.listing.text() == "B3"
    assert win._left_tabs.currentWidget() is tab       # ...and it is brought to the front
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
    tab = win._left_tabs.currentWidget()
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
    first = win._left_tabs.currentWidget()
    _shift_drag_over(win, ["B3"])
    qapp.processEvents()
    tabs = [win._left_tabs.widget(i) for i in range(win._left_tabs.count())
            if isinstance(win._left_tabs.widget(i), V._ExplorationTab)]
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
    assert not [i for i in range(win._left_tabs.count())
                if isinstance(win._left_tabs.widget(i), V._ExplorationTab)]   # ...no tab
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
    assert not [i for i in range(win._left_tabs.count())
                if isinstance(win._left_tabs.widget(i), V._ExplorationTab)]
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

    idx = next(i for i in range(win._left_tabs.count())
               if win._left_tabs.widget(i) is win._op_tabs[key])
    win._close_op_tab(idx)                                    # close it MID-RUN
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
    assert win._left_tabs.currentWidget() is tab_b             # it is in front...
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
    win._left_tabs.setCurrentIndex(0)                          # switch home MID-RUN (deferred)
    qapp.processEvents()
    assert win._detail._fov_labels == ["B2:0"]
    blocking_worker[-1].failed.emit("boom")
    blocking_worker[-1].release()
    assert _drain_until(qapp, lambda: not win._busy())
    qapp.processEvents()
    assert win._detail._fov_labels == ["B2:0", "B3:0"]         # whole plate restored
    assert win._active_exploration is None
    win.close()
