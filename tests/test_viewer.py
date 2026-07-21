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
from PyQt5.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QMouseEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication, QSlider, QWidget  # noqa: E402

from squidmip import _viewer as V  # noqa: E402

from .conftest import CH_IN_YAML  # noqa: E402


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


# --- loupe: press-and-hold magnifier (IMA-208) ----------------------------------------------
#
# The gesture and the geometry are tested separately from the I/O: the state machine needs no
# pixels, and the pure math needs no Qt. Only the read tests touch a real written pyramid —
# which is why `pyramid_dataset` exists (a 4x4 fixture writes ONE level, so it cannot prove
# level selection at all).

class _FakeLoupeSource(V._LoupeSource):
    """A source with known pixels, so gesture tests don't need zarr or TIFF decode."""

    def __init__(self, well_px=1000, n_levels=3, pixel_size_um=0.325, missing=()):
        self.well_px, self.n_levels, self.pixel_size_um = well_px, n_levels, pixel_size_um
        self._missing = set(missing)
        self.reads = []

    def available(self, well_id):
        if well_id in self._missing:
            return False, "not written yet"
        return True, ""

    def read_crop(self, well_id, level, y0, x0, h, w):
        self.reads.append((well_id, level, y0, x0, h, w))
        return np.full((2, max(1, h), max(1, w)), 500, np.uint16)

    def coarse(self, well_id):
        return np.full((2, 8, 8), 500, np.uint16)


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


def test_loupe_um_per_screen_px_refuses_to_guess():
    assert V.loupe_um_per_screen_px(0.325, 1.0) == pytest.approx(0.325)
    assert V.loupe_um_per_screen_px(0.325, 0.5) == pytest.approx(0.65)
    assert V.loupe_um_per_screen_px(None, 1.0) is None      # unknown -> no bar, never a guess
    assert V.loupe_um_per_screen_px(0, 1.0) is None
    assert V.loupe_um_per_screen_px(float("nan"), 1.0) is None


def test_composite_rgb_matches_manual_windowing():
    planes = [np.array([[0.0, 10.0]]), np.array([[5.0, 5.0]])]
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wins = [(0.0, 10.0), (0.0, 10.0)]
    out = V._composite_rgb(planes, colors, wins)
    assert out.shape == (1, 2, 3)
    assert out[0, 0, 0] == pytest.approx(0.0)      # ch0 at its window floor
    assert out[0, 1, 0] == pytest.approx(1.0)      # ch0 at its window ceiling
    assert out[0, 0, 1] == pytest.approx(0.5)      # ch1 mid-window, in green


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


def test_preview_run_gets_no_loupe_source(qapp, stub_detail, squid_dataset, tmp_path):
    """An unsaved preview writes nothing, so its layer must NOT inherit a zarr source — this is
    the stale-run trap: OperationStack dedupes by key, so the layer name alone proves nothing."""
    root, _ = squid_dataset
    win = _loupe_win(qapp, root)
    win.run_operator("mip", out_parent=str(tmp_path), save=False, preview_limit=1)
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
    full = src.read_crop(region, 0, 0, 0, size, size)
    assert np.array_equal(crop[0], full[0][100:132, 100:132])   # the crop is where we asked

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
