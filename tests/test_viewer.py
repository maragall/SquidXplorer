"""HCS viewer — headless (offscreen) tests.

Gates the viewer contract: pure hit-testing + fit-cell shape guard, ingest that LOADS a grey plate
without processing, ANY registered operator (MIP, reference) filling tiles + slider pushes + the hue
status through one operator-agnostic path, the loud gate on operator kinds the live path can't stream
yet, the raw-z-stack push into the embedded ndviewer on double-click (pointing at the acquisition's own
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

    Records the push API (start_acquisition / register_image / register_array / go_to_well_fov) so we
    can assert the seam WITHOUT constructing ndviewer's real vispy/GL widget — which segfaults
    offscreen under pytest's PySide6/napari-loaded environment (a Qt-binding conflict, not a code bug).
    ``register_array`` is the LIVE seam (a computed well streaming into the growing slider); without it
    here the whole pushReady path silently no-ops in every test.
    """

    push_raises = False   # flip to simulate a slider that rejects every push (failure-counter test)

    def __init__(self):
        super().__init__()
        self._fov_labels = []
        self._fov_slider = QSlider(Qt.Horizontal, self)
        self.registered = []
        self.arrays = []      # (t, fov_idx, z, channel) per live push
        self.nav = []
        self.nz = None        # nz the acquisition was started with (1 for a z-reducing operator)

    def start_acquisition(self, channels, nz, h, w, labels):
        self._fov_labels = list(labels)
        self.nz = nz
        self._fov_slider.setMaximum(max(0, len(labels) - 1))

    def register_image(self, t, idx, z, ch, path, page_idx=0):
        self.registered.append((t, idx, z, ch, path))

    def register_array(self, t, idx, z, ch, arr):
        if self.push_raises:
            raise RuntimeError("synthetic slider push failure")
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


def test_fit_cell_always_returns_cell_shape():
    assert V._fit_cell(np.zeros((768, 768), np.float32)).shape == (V._CELL, V._CELL)
    assert V._fit_cell(np.zeros((V._CELL, V._CELL), np.float32)).shape == (V._CELL, V._CELL)
    assert V._fit_cell(np.zeros((40, 40), np.float32)).shape == (V._CELL, V._CELL)  # tiny frame upscaled


def test_live_stream_shape_for_a_z_reducer():
    # the kind the live path streams today: one FOV per well, z collapsed to a single push plane
    assert V.live_stream_shape(frozenset({"z"})) == (1, 1)


@pytest.mark.parametrize("consumes, ticket", [(frozenset(), "IMA-223"), (frozenset({"fov"}), "IMA-222")])
def test_live_stream_shape_gates_unwired_kinds_loudly(consumes, ticket):
    # a plane operator / FOV reducer must fail NAMED, never render silently-wrong pixels
    with pytest.raises(NotImplementedError, match=ticket):
        V.live_stream_shape(consumes)


def test_disk_multiplier_per_operator_kind():
    meta = {"n_z": 5, "fovs_per_region": {"B2": [0, 1, 2], "B3": [0]}}
    assert V._disk_multiplier(frozenset({"z"}), meta) == 1.0        # one frame per region
    assert V._disk_multiplier(frozenset(), meta) == 5.0             # plane op: z preserved
    assert V._disk_multiplier(frozenset({"fov"}), meta) == 3.0      # stitch: zero-overlap upper bound


def test_compose_tile_applies_the_injected_contrast_strategy():
    # ONE compositor, two strategies: a running histogram (live/preview) vs a per-well percentile
    # (reopen). Same tiles + same window must give the same pixels either way.
    tiles = [np.linspace(0, 1000, V._CELL * V._CELL, dtype=np.float32).reshape(V._CELL, V._CELL)]
    colors = np.array([[1.0, 0.0, 0.0]])
    fixed = V._compose_tile(tiles, colors, lambda c_i, t: (0.0, 1000.0))
    assert fixed.shape == (V._CELL, V._CELL, 3) and fixed.dtype == np.uint8
    assert fixed[..., 1].max() == 0 and fixed[..., 0].max() == 255      # red channel only
    pct = V._compose_tile(tiles, colors, V._percentile_window)
    running = V._compose_tile(tiles, colors, V._running_window(V._RunningContrast(1, 1000.0)))
    assert not np.array_equal(pct, running)      # the strategies are genuinely distinct
    # the running strategy folds each tile it sees into the shared histogram
    contrast = V._RunningContrast(1, 1000.0)
    V._compose_tile(tiles, colors, V._running_window(contrast))
    assert contrast.window(0)[1] > 0


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


@pytest.mark.parametrize("key", ["mip", "reference"])
def test_any_registered_operator_streams_tiles_and_slider_pushes(qapp, stub_detail, squid_dataset,
                                                                 tmp_path, key):
    # ACCEPTANCE (IMA-226): a NON-MIP operator streams partial results to the plate tiles AND the
    # ndviewer growing slider exactly like MIP — same worker, same signals, nothing operator-specific.
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._detail.arrays.clear()
    win.run_operator(key, out_parent=str(tmp_path / key))
    assert _drain_until(qapp, lambda: len(win._overview._tiles) == 2 and len(win._detail.arrays) == 4)
    assert win._detail.nz == 1                                   # z-reducer -> ndv drops the z-slider
    assert win._detail._fov_labels == [f"{w}:0" for w in win._order]
    channels = [c["name"] for c in win._meta["channels"]]
    expected = {(0, win._fov_index[w]["idx"], 0, ch) for w in ("B2", "B3") for ch in channels}
    assert set(win._detail.arrays) == expected                   # one push per well per channel, z=0
    assert set(win._overview._status.values()) == {"done"}
    win._stop_worker(); win.close()


def test_run_operator_gates_an_unstreamable_operator_kind(qapp, stub_detail, squid_dataset,
                                                          monkeypatch, tmp_path):
    # A plane operator (Nz -> Nz) can't stream through the z-collapsed live path yet: it must raise
    # NAMED (with the ticket that lands the wiring) and leave the plate untouched — never paint
    # silently-wrong pixels.
    import squidmip._engine as engine
    saved = dict(engine._PROJECTORS)
    engine.add_projector("plane_noop", lambda planes: next(iter(planes)), consumes=frozenset())
    monkeypatch.setattr(V, "_OPERATIONS_BY_KEY", dict(
        V._OPERATIONS_BY_KEY, plane_noop=V.Operation("plane_noop", "Plane Op", "", "_build_mip_tab")))
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    try:
        with pytest.raises(NotImplementedError, match="IMA-223"):
            win.run_operator("plane_noop", out_parent=str(tmp_path))
        assert win._worker is None                              # no run started
        assert set(win._overview._status.values()) == {"empty"}  # plate untouched
    finally:
        engine._PROJECTORS.clear(); engine._PROJECTORS.update(saved)
        win.close()


def test_failing_slider_pushes_are_counted_and_surfaced(qapp, stub_detail, squid_dataset, tmp_path):
    # A slider that rejects every push used to be invisible (`except Exception: pass`) — the run
    # completed with a mysteriously empty detail view. Now the run still completes, and says so.
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._detail.push_raises = True
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: win._readout.text().startswith("✓"))
    assert "4 slider pushes failed" in win._readout.text()       # 2 wells x 2 channels
    assert win._detail.arrays == []
    win._stop_worker(); win.close()


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
