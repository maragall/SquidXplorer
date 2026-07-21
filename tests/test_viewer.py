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


# --- IMA-218: coordinate-placed mosaic ------------------------------------------------------

def test_mosaic_boxes_single_fov_fills_the_cell():
    """N=1 must fill the cell exactly — the no-hidden-special-case guarantee, by construction."""
    boxes = V.mosaic_boxes({0: (10.0, 20.0)}, (4, 4), 0.325)
    assert boxes[0] == (0, 0, V._CELL, V._CELL)


def test_mosaic_boxes_two_fovs_side_by_side():
    """Two FOVs one frame-width apart in X land side by side, each half the cell, same row."""
    pitch = 4 * 0.325 / 1000.0                      # one frame width in mm
    boxes = V.mosaic_boxes({0: (10.0, 20.0), 1: (10.0 + pitch, 20.0)}, (4, 4), 0.325)
    (x0, y0, w0, h0), (x1, y1, w1, h1) = boxes[0], boxes[1]
    assert y0 == y1                                  # same stage Y -> same row
    assert x0 == 0 and x1 == pytest.approx(V._CELL // 2, abs=1)
    assert w0 == pytest.approx(V._CELL // 2, abs=1)  # one scale for both axes, no stretch
    assert h0 == h1 == pytest.approx(V._CELL // 2, abs=1)   # square region letterboxed


def test_mosaic_boxes_flips_y_stage_up_to_rows_down():
    """Stage Y increases UP; image rows increase DOWN. The HIGHER stage Y must be the TOP row."""
    pitch = 4 * 0.325 / 1000.0
    boxes = V.mosaic_boxes({0: (10.0, 20.0), 1: (10.0, 20.0 + pitch)}, (4, 4), 0.325)
    assert boxes[1][1] < boxes[0][1], "higher stage Y should map to a smaller row index"


def test_compose_mosaic_later_fov_wins_on_overlap():
    """Overlap policy is documented as ascending-fov, later wins — pin it so it can't drift."""
    boxes = {0: (0, 0, 4, 4), 1: (2, 0, 4, 4)}       # fov 1 overlaps fov 0's right half
    out = V.compose_mosaic({0: np.full((4, 4), 10.0), 1: np.full((4, 4), 20.0)}, boxes, cell=8)
    assert out[0, 0] == 10 and out[0, 3] == 20       # the overlap column belongs to fov 1
    assert out[7, 7] == 0                            # uncovered stays background, never stretched


def test_compose_mosaic_uncovered_area_stays_zero():
    out = V.compose_mosaic({0: np.ones((4, 4))}, {0: (0, 0, 4, 4)}, cell=8)
    assert out[:4, :4].all() and not out[4:, 4:].any()


def test_run_operator_renders_a_two_fov_mosaic(qapp, stub_detail, squid_dataset, tmp_path):
    """End-to-end: the fixture's 2 coordinated FOVs per well render as one placed mosaic tile."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._plan_fovs()[0]["B2"] == [0, 1], "both FOVs should be planned (coords are complete)"
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: win._overview is not None
                        and len(win._overview._tiles) == 2 and win._overview._final is not None)
    # one tile per WELL (not per FOV) — the accumulator emitted exactly once per region
    assert len(win._worker._raw) == 2
    # the mosaic is 2 FOVs wide: both halves carry signal, and they differ (distinct pixel values)
    tile = win._worker._raw[win._fov_index["B2"]["rc"]][0]
    left, right = tile[:, :V._CELL // 2], tile[:, V._CELL // 2:]
    assert left.any() and right.any(), "both FOVs should be placed"
    assert not np.array_equal(left, right), "the two FOVs are distinct planes, not one duplicated"
    assert not win._worker._pending, "no region left half-accumulated"
    win._stop_worker()
    win.close()


def test_progress_counts_wells_not_fovs(qapp, stub_detail, squid_dataset, tmp_path):
    """Regression for the 144/4 bug: _done counts completed WELLS even though on_well is per FOV."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    seen = []
    win.run_operator("mip", out_parent=str(tmp_path))
    win._worker.progress.connect(lambda d, t: seen.append((d, t)))
    assert _drain_until(qapp, lambda: win._overview is not None and len(win._overview._tiles) == 2)
    assert win._worker._total == 2                     # 2 wells, NOT 4 (region, fov) pairs
    assert all(d <= t for d, t in seen), f"progress overshot its total: {seen}"
    win._stop_worker()
    win.close()


def test_single_fov_tile_is_byte_identical_to_the_pre_mosaic_path(squid_dataset):
    """CRITICAL (IMA-187 oracle): with one FOV the tile must be exactly _fit_cell(plane).

    Guards against a mosaic path that quietly re-samples the N=1 case — 'no hidden if n_fov==1'
    cuts both ways: no special case, and no silent change either.
    """
    plane = np.arange(16, dtype=np.float32).reshape(4, 4) * 7.0
    boxes = V.mosaic_boxes({0: (10.0, 20.0)}, (4, 4), 0.325)
    via_mosaic = V.compose_mosaic({0: plane}, boxes)
    np.testing.assert_array_equal(via_mosaic, V._fit_cell(plane))


def test_well_with_incomplete_coordinates_is_flagged_not_guessed(qapp, stub_detail,
                                                                 squid_dataset, tmp_path):
    """D5: a multi-FOV well missing a coordinate falls back to 1 FOV and is reported, not placed."""
    root, _ = squid_dataset
    # drop B3's second FOV row -> its coordinate cover is now incomplete
    lines = (root / "coordinates.csv").read_text().splitlines()
    kept = [ln for ln in lines if not ln.startswith("B3,1,")]
    (root / "coordinates.csv").write_text("\n".join(kept) + "\n")
    win = V.PlateWindow(None)
    win.ingest(str(root))
    plan, bad = win._plan_fovs()
    assert bad == ["B3"], f"B3 should be flagged, got {bad}"
    assert plan["B3"] == [0], "a well without full coordinates falls back to its first FOV"
    assert plan["B2"] == [0, 1], "the fully-coordinated well is unaffected"
    win.close()
