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
from PyQt5.QtWidgets import QApplication, QSlider, QWidget  # noqa: E402

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
        self.nav = []

    def start_acquisition(self, channels, nz, h, w, labels):
        self._fov_labels = list(labels)
        self._fov_slider.setMaximum(max(0, len(labels) - 1))

    def register_image(self, t, idx, z, ch, path, page_idx=0):
        self.registered.append((t, idx, z, ch, path))

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


def test_resolve_plate_root(tmp_path):
    (tmp_path / "plate.ome.zarr").mkdir()
    _, is_plate = V.resolve_plate_root(tmp_path)
    assert is_plate
    acq = tmp_path / "acq"
    acq.mkdir()
    _, is_plate = V.resolve_plate_root(acq)
    assert not is_plate


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


def test_ingest_readable_non_wellplate_reports_not_crashes(qapp, stub_detail, tmp_path):
    # A readable Squid acquisition whose region is NOT a well id (glass slide / "R2C3" / manual).
    # It must report "not a well-plate", show the drop target, and leave no half-set state — NOT
    # crash out of ingest/__init__ (the HIGH bug the adversarial review found).
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
    assert "well-plate" in win._readout.text().lower()
    assert win._reader is None and win._overview is None
    assert win._drop.isVisible() or not win._drop.isHidden()
    # and the initial-path route through __init__ must not crash either
    win2 = V.PlateWindow(str(root))
    assert "well-plate" in win2._readout.text().lower()
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
    # bounded memory: the worker keeps one 88px tile per well, not the acquisition
    assert len(win._worker._raw) == 2
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


def _mouse(kind, pos, mods=Qt.NoModifier, buttons=Qt.LeftButton):
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent
    ev = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove,
          "release": QEvent.MouseButtonRelease, "dblclick": QEvent.MouseButtonDblClick}[kind]
    return QMouseEvent(ev, pos, Qt.LeftButton, buttons, mods)


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
