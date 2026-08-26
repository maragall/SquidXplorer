"""Channel checkboxes reach every brick of a channel, and only that channel (Julio, G7 set).

Two live reports, neither reproduced by the hand-built scenes (``build_flat_scene`` /
``build_volume_scene``), so both are driven through the REAL chain: a ``PlateWindow`` over a
G7-shaped acquisition, its ``ViewerManager`` and ``ViewDeck``, the headless
``model_pane_class`` pane (a real Qt-free ``ViewerModel`` under a real ``MosaicLayers``), an
ROI child, and the 3D tab ``_open_3d`` spawns.

(A) "For G7, when I 3D render and ROI, the 561_nm cannot be toggled on and off, even though
    it's visible."
(B) "When I turn off layer 561 for decon, the whole layer turns off."
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

import squidxplorer._viewer as V
from tests.conftest import shutdown_plate_window

pytestmark = pytest.mark.usefixtures("qapp")

REGION = "G7"
CHANNELS = ["BF_LED_matrix_full", "Fluorescence_488_nm_Ex", "Fluorescence_561_nm_Ex"]
FOVS = [0, 1, 2, 3]
NZ = 4
FRAME = 16
PX_UM = 0.75
FRAME_MM = FRAME * PX_UM / 1000.0
FOV_MM = {0: (10.0, 20.0), 1: (10.0 + FRAME_MM, 20.0),
          2: (10.0, 20.0 + FRAME_MM), 3: (10.0 + FRAME_MM, 20.0 + FRAME_MM)}

_CHANNEL_YAML = """\
version: 1
objective: 20x
channels:
- name: BF LED matrix full
  camera_settings:
    '1':
      display_color: '#FFFFFF'
      exposure_time_ms: 5.0
- name: Fluorescence 488 nm Ex
  camera_settings:
    '1':
      display_color: '#00FF00'
      exposure_time_ms: 50.0
- name: Fluorescence 561 nm Ex
  camera_settings:
    '1':
      display_color: '#FFA500'
      exposure_time_ms: 50.0
"""

_ACQ_YAML = f"""\
objective:
  pixel_size_um: {PX_UM}
  magnification: 20.0
  sensor_pixel_size_um: 3.76
sample:
  wellplate_format: 96 well plate
z_stack:
  nz: {NZ}
  delta_z_mm: 0.0015
time_series:
  nt: 1
"""

_PARAMS = {"Nz": NZ, "Nt": 1, "dz(um)": 1.5,
           "objective": {"magnification": 20.0, "NA": 0.8}, "sensor_pixel_size_um": 3.76}


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def build_g7_dataset(root: Path) -> Path:
    """G7's shape: one region, a 2x2 FOV grid, 4 z, three channels, 16 px frames."""
    folder = root / "0"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for fov in FOVS:
        for z in range(NZ):
            for c_i, ch in enumerate(CHANNELS):
                arr = (rng.integers(100, 3000, (FRAME, FRAME)) + 1000 * c_i).astype(np.uint16)
                tifffile.imwrite(folder / f"{REGION}_{fov}_{z}_{ch}.tiff", arr)
    lines = ["region,x (mm),y (mm),z (mm)"]
    for _z in range(NZ):
        for fov in FOVS:
            x, y = FOV_MM[fov]
            lines.append(f"{REGION},{x:.6f},{y:.6f},")
    (root / "coordinates.csv").write_text("\n".join(lines) + "\n")
    (root / "acquisition_channels.yaml").write_text(_CHANNEL_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    return root


@pytest.fixture
def g7_dataset(tmp_path) -> Path:
    return build_g7_dataset(tmp_path / "G7_acq")


def _drain_until(app, pred, timeout=30.0) -> bool:
    import time

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        if pred():
            return True
    return False


def _raw_landed(view) -> bool:
    mosaic = getattr(view._pane, "mosaic", None) if view._pane is not None else None
    return mosaic is not None and len(mosaic.channels("raw")) == len(CHANNELS)


def _open_plate(qapp, root):
    win = V.PlateWindow(None)
    win.ingest(str(root))
    mgr = win._viewer_manager
    mgr.tabbed_views = True
    view = mgr.open([REGION])
    assert view is not None
    assert _drain_until(qapp, lambda: _raw_landed(view)), "the parent view's raw never landed"
    return win, mgr, view


def _roi_bbox_um(meta) -> tuple:
    """An ROI over the middle of the 2x2 grid: part of every FOV."""
    from squidxplorer._napari3d import region_origin_um

    ox, oy = region_origin_um(meta, REGION)
    return (ox + 4.0, oy + 4.0, ox + 20.0, oy + 20.0)


def _open_3d_tab(qapp, mgr, parent):
    """`_open_3d` on *parent*: the NEW deck tab carrying the bricked volume, bricks landed."""
    deck = mgr.deck(create=False)
    before = deck.count()
    parent._open_3d()
    qapp.processEvents()
    assert deck.count() == before + 1, "3D did not open a new tab"
    tab = deck.current_page()
    assert tab is not parent and tab._render_mode == "3d"
    vol = tab._native3d
    assert vol is not None, f"the 3D tab holds no volume: {tab._pane.said}"
    want = vol.brick_count * len(CHANNELS)
    assert _drain_until(qapp, lambda: len(vol._layers) >= want, timeout=60), (
        f"{len(vol._layers)} of {want} bricks landed: {tab._pane.said}")
    # The 3D tab's own 2D mosaic lands AFTER the volume opens; let it arrive and settle.
    _drain_until(qapp, lambda: _raw_landed(tab) or tab._pane.mosaic.ours(), timeout=10)
    for _ in range(50):
        qapp.processEvents()
    return tab


def _row_for(model, op: str, channel: str):
    for op_row, (row_op, channels) in enumerate(model._rows):
        if row_op == op and channel in channels:
            parent = model.index(op_row, 0)
            return model.index(channels.index(channel), 0, parent)
    raise AssertionError(f"the tree has no row for ({op}, {channel}); rows: {model._rows}")


def _visible_by_channel(mosaic, op: str) -> dict:
    return {ch: [bool(ly.visible) for ly in mosaic.layers_for(op, ch)]
            for ch in mosaic.channels(op)}


# --- (A) the 3D ROI tab: untick one channel, every brick of it goes dark, the rest stay ---------

def test_a_channel_checkbox_in_the_3d_roi_tab_darkens_every_brick_of_that_channel_only(
        qapp, napari_pane_stub, g7_dataset):
    win, mgr, view = _open_plate(qapp, g7_dataset)
    try:
        child = mgr.open_child([REGION], roi_bbox=_roi_bbox_um(view._meta),
                               parent_id=view.window_id)
        assert child is not None
        assert _drain_until(qapp, lambda: _raw_landed(child)), "the ROI child's raw never landed"
        tab = _open_3d_tab(qapp, mgr, child)
        mosaic = tab._pane.mosaic
        model = tab._pane.layer_tree.model()
        assert model._mosaic is mosaic, "the 3D tab's tree is bound to another view's layers"

        ch_off = CHANNELS[2]                                   # the 561 of the report
        before = _visible_by_channel(mosaic, "raw")
        assert before and all(all(v) for v in before.values()), (
            f"the volume did not open with every brick lit: {before}")
        n_bricks = len(mosaic.layers_for("raw", ch_off))
        assert n_bricks == tab._native3d.brick_count, (
            f"layers_for finds {n_bricks} of {tab._native3d.brick_count} bricks")

        ok = model.setData(_row_for(model, "raw", ch_off), Qt.Unchecked, Qt.CheckStateRole)
        qapp.processEvents()
        assert ok, "setData refused the checkbox write"

        after = _visible_by_channel(mosaic, "raw")
        assert not any(after[ch_off]), f"{ch_off} still lit after unticking: {after}"
        for ch in CHANNELS[:2]:
            assert all(after[ch]), f"{ch} went dark with {ch_off}: {after}"
        assert mosaic.channel_visible(ch_off) is False

        ok = model.setData(_row_for(model, "raw", ch_off), Qt.Checked, Qt.CheckStateRole)
        qapp.processEvents()
        assert ok
        relit = _visible_by_channel(mosaic, "raw")
        assert all(relit[ch_off]), f"{ch_off} did not come back: {relit}"
        assert all(all(v) for v in relit.values()), relit
    finally:
        shutdown_plate_window(qapp, win)


# --- a REAL operator preview through the plate's own run_operator ------------------------------

def _run_preview(qapp, view, key: str, timeout: float = 180.0) -> None:
    """Select *key* in *view*'s dropdown, press Preview, and wait for the run to report back."""
    outcome: list = []
    done, failed = view.operator_done, view.operator_failed
    view.operator_done = lambda action, seconds: (outcome.append(("done", action)),
                                                  done(action, seconds))
    view.operator_failed = lambda action, reason: (outcome.append(("failed", reason)),
                                                   failed(action, reason))
    view.show_operator_controls_for(key)
    assert view._op_combo.currentData() == key, f"{key!r} is not in the view's dropdown"
    view._preview_view_operator()
    assert _drain_until(qapp, lambda: bool(outcome), timeout=timeout), (
        f"the {key} preview never reported back; said: {view._pane.said}")
    assert outcome[0][0] == "done", f"the {key} preview failed: {outcome[0][1]}"
    for _ in range(50):
        qapp.processEvents()


# --- (B) a 2D view: untick one decon channel, the other decon channels stay lit ----------------

@pytest.mark.parametrize("where", ["parent", "roi-child-after-a-replayed-parent-result"])
def test_unticking_one_decon_channel_leaves_the_other_decon_channels_lit(
        qapp, napari_pane_stub, g7_dataset, caplog, where):
    """*where*: the 2D view the preview runs in: the region view itself, or an ROI child that
    first received the parent's cached result dark (replayed on open) and then ran its own."""
    import logging

    win, mgr, view = _open_plate(qapp, g7_dataset)
    try:
        if where != "parent":
            _run_preview(qapp, view, "decon")
            view = mgr.open_child([REGION], roi_bbox=_roi_bbox_um(view._meta),
                                  parent_id=view.window_id)
            assert _drain_until(qapp, lambda: _raw_landed(view))
        _run_preview(qapp, view, "decon")
        mosaic = view._pane.mosaic
        assert set(mosaic.channels("decon")) == set(CHANNELS), (
            f"decon landed on {mosaic.channels('decon')}, not every channel")
        before = _visible_by_channel(mosaic, "decon")
        assert all(all(v) for v in before.values()), f"decon did not arrive lit: {before}"
        model = view._pane.layer_tree.model()
        assert model._mosaic is mosaic

        ch_off = CHANNELS[2]
        with caplog.at_level(logging.DEBUG, logger="squid.xplorer.layers"):
            ok = model.setData(_row_for(model, "decon", ch_off), Qt.Unchecked, Qt.CheckStateRole)
            qapp.processEvents()
        assert ok
        after = _visible_by_channel(mosaic, "decon")
        assert not any(after[ch_off]), f"decon/{ch_off} still lit: {after}"
        for ch in CHANNELS[:2]:
            assert all(after[ch]), (
                f"decon/{ch} went dark with {ch_off}: {after}; log: "
                f"{[r.getMessage() for r in caplog.records if 'checkbox' in r.getMessage()]}")
        assert mosaic.visible_op() == "decon"
    finally:
        shutdown_plate_window(qapp, win)


# --- (C) a decon preview from a 3D ROI tab lands as a full-depth volume of decon ---------------

@pytest.mark.parametrize("prior_2d_preview", [False, True],
                         ids=["fresh", "after-a-2d-decon-preview-in-the-parent"])
def test_a_decon_preview_from_a_3d_roi_tab_lands_as_a_full_depth_volume(
        qapp, napari_pane_stub, g7_dataset, prior_2d_preview):
    """*prior_2d_preview*: the parent already previewed decon in 2D (an unscoped run, so its
    one-plane result is CACHED and replayed into every later tab over the region)."""
    from squidxplorer._napari_view import full_res_level

    win, mgr, view = _open_plate(qapp, g7_dataset)
    try:
        if prior_2d_preview:
            _run_preview(qapp, view, "decon")
            assert int(full_res_level(view._pane.mosaic.find("decon", CHANNELS[0]).data).ndim) == 2
        child = mgr.open_child([REGION], roi_bbox=_roi_bbox_um(view._meta),
                               parent_id=view.window_id)
        assert _drain_until(qapp, lambda: _raw_landed(child))
        tab = _open_3d_tab(qapp, mgr, child)
        _run_preview(qapp, tab, "decon")
        mosaic = tab._pane.mosaic
        vol = tab._native3d
        assert vol is not None and vol._op == "decon", (
            f"the 3D tab's volume is {getattr(vol, '_op', None)!r}, not decon; "
            f"said: {tab._pane.said}")
        want = vol.brick_count * len(CHANNELS)
        assert _drain_until(qapp, lambda: len(vol._layers) >= want, timeout=60), (
            f"{len(vol._layers)} of {want} decon bricks landed: {tab._pane.said}")
        for ch in CHANNELS:
            bricks = mosaic.layers_for("decon", ch)
            assert bricks, f"no decon brick for {ch}: {[ly.name for ly in mosaic.model.layers]}"
            for ly in bricks:
                assert ly.data.ndim == 3 and int(ly.data.shape[0]) == NZ, (
                    f"decon/{ch} brick {ly.name} is {ly.data.shape}, not {NZ} planes deep")
        flat = [ly for ly in mosaic.model.layers if str(ly.name).startswith("decon ·")]
        assert len(flat) == len(CHANNELS), [ly.name for ly in mosaic.model.layers]
        for ly in flat:
            level0 = full_res_level(ly.data)
            assert level0.ndim == 3 and int(level0.shape[0]) == NZ, (
                f"{ly.name} landed {level0.shape}: flat, not {NZ} planes deep")
        for line in tab._pane.said:
            assert "no volume to render" not in line and "carries no z depth" not in line, line
        # ...and the decon volume's channel checkboxes drive its bricks, one channel each.
        model = tab._pane.layer_tree.model()
        ch_off = CHANNELS[2]
        assert model.setData(_row_for(model, "decon", ch_off), Qt.Unchecked, Qt.CheckStateRole)
        qapp.processEvents()
        after = _visible_by_channel(mosaic, "decon")
        assert not any(after[ch_off]), after
        for ch in CHANNELS[:2]:
            assert all(after[ch]), f"decon/{ch} went dark with {ch_off}: {after}"
    finally:
        shutdown_plate_window(qapp, win)
