"""Walking a region's FOVs with the camera — the boxes, the highlight, and the playback gate."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # headless Qt; must precede any Qt import

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from qtpy.QtWidgets import QApplication  # noqa: E402

from squidxplorer._mosaic_source import fov_at_point, mosaic_bbox_um, mosaic_fov_bboxes_um  # noqa: E402

napari = pytest.importorskip("napari")


@pytest.fixture(scope="module")
def qapp():
    """This module's QApplication, by the same convention every other GUI test module here uses."""
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)   # main() won't call exec_/exit under test
    return app


#: The real 40x AF-sweep geometry: 16 fields, 4x4, on a pitch of exactly 7x the field
#: (``overlap_percent: -600``), which is what makes 97% of the mosaic empty and the wheel-zoom
#: workflow this feature replaces so painful. Numbers taken off the acquisition, not invented.
PX_40X = 0.094
FRAME_40X = 4168
PITCH_UM = 2742.4064
X0_UM, Y0_UM = 10780.7125, 8021.6375


def _sweep_meta(n_side: int = 4) -> dict:
    positions = {}
    for i in range(n_side * n_side):
        row, col = divmod(i, n_side)
        positions[("A1", i)] = (X0_UM + col * PITCH_UM, Y0_UM + row * PITCH_UM)
    return {
        "pixel_size_um": PX_40X,
        "frame_shape": (FRAME_40X, FRAME_40X),
        "z_levels": [0],
        "n_t": 1,
        "channels": [{"name": "Fluorescence 405 nm Ex - penta"}],
        "regions": ["A1"],
        "fovs_per_region": {"A1": list(range(n_side * n_side))},
        "fov_positions_um": positions,
    }


def _walker(meta: dict, region: str = "A1", canvas_hw=(720, 860)):
    """A ``RegionViewer`` in FOVs mode, wired to a real ``MosaicLayers``, without a window."""
    from napari.components import ViewerModel

    from squidxplorer._fov_nav import FovSlider
    from squidxplorer._napari_view import MosaicLayers
    from squidxplorer._region_viewer import RegionViewer

    model = ViewerModel()
    model._canvas_size = tuple(canvas_hw)
    mosaic = MosaicLayers(model)
    mosaic.add_mosaic("raw", meta["channels"][0]["name"],
                      np.zeros((256, 256), np.uint16), contrast_limits=(0, 1),
                      colormap="gray", multiscale=False, bbox_um=mosaic_bbox_um(meta, region))

    class _Pane:
        ok = True

    pane = _Pane()
    pane.mosaic = mosaic
    pane._viewer = model    # every real pane's `_viewer` IS `mosaic.model`; cross that interface

    win = RegionViewer.__new__(RegionViewer)
    win._pane = pane
    win._meta = meta
    win._manager = None
    win.window_id = 1
    win._fov_mode = True
    win._fov_layer = None
    win._fov_boxes_cache = {}
    win._roi_layer = None
    win.said = []
    win._say = win.said.append
    win.current_region = lambda: region
    win._fov_slider = FovSlider(on_change=win._on_fov_changed)
    return win, model


def _open(win) -> None:
    """What ``_on_done`` does on ``first_look``: draw the boxes, then frame the current field."""
    win._draw_fov_boxes()
    fov = win._fov_slider.fov
    if fov is not None:
        win._on_fov_changed(win._fov_slider.index, int(fov))


def _fov_layer(model):
    return next(ly for ly in model.layers if ly.name == "FOVs")


# ══════════════════════════════════════════════════════════════════════════════════════════
# The boxes
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_every_fov_of_the_region_is_drawn_once_and_labelled(qapp):
    meta = _sweep_meta()
    win, model = _walker(meta)
    try:
        _open(win)
        layer = _fov_layer(model)
        boxes = mosaic_fov_bboxes_um(meta, "A1")

        assert len(layer.data) == 16 == len(boxes)
        assert list(layer.text.string.array) == [f"fov {f}" for f in boxes]

        for i, (fov, (x0, y0, x1, y1)) in enumerate(boxes.items()):
            got = np.asarray(layer.data[i])
            assert got[:, -2].min() == pytest.approx(y0)
            assert got[:, -2].max() == pytest.approx(y1)
            assert got[:, -1].min() == pytest.approx(x0)
            assert got[:, -1].max() == pytest.approx(x1)
    finally:
        win._fov_slider.shutdown()


def test_a_fov_box_is_never_clamped_to_the_texture_ceiling(qapp):
    """The FOV boxes must not go through the ROI layer's drawing-time clamp."""
    meta = _sweep_meta()
    win, model = _walker(meta)
    try:
        _open(win)
        layer = _fov_layer(model)
        last = np.asarray(layer.data[-1])
        field_um = FRAME_40X * PX_40X
        ulp = float(np.spacing(np.float32(X0_UM + 3 * PITCH_UM + FRAME_40X * PX_40X)))
        assert last[:, -1].max() - last[:, -1].min() == pytest.approx(field_um, abs=4 * ulp)
        assert last[:, -2].max() - last[:, -2].min() == pytest.approx(field_um, abs=4 * ulp)
        assert field_um > 2048 * PX_40X, "the fixture must be big enough for the clamp to bite"
        assert not any("ceiling" in s for s in win.said)
    finally:
        win._fov_slider.shutdown()


def test_the_fov_boxes_do_not_touch_the_users_roi_layer(qapp):
    """Opening a FOVs view must not create, rename, renumber or clear anything ROI-shaped."""
    meta = _sweep_meta()
    win, model = _walker(meta)
    try:
        _open(win)
        assert win._roi_layer is None
        assert [ly.name for ly in model.layers if ly.name == "ROIs"] == []
    finally:
        win._fov_slider.shutdown()


# ══════════════════════════════════════════════════════════════════════════════════════════
# The camera
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_stepping_a_fov_moves_the_camera_and_loads_nothing(qapp):
    """The whole design in one assertion: the camera moves, the pixels do not."""
    meta = _sweep_meta()
    win, model = _walker(meta)
    try:
        _open(win)
        images = [ly for ly in model.layers if hasattr(ly, "contrast_limits")]
        assert images, "the fixture must have a mosaic layer for this to mean anything"
        before = [(ly.name, id(ly.data)) for ly in images]
        n_layers = len(model.layers)
        start = tuple(model.camera.center)

        win._fov_slider.set_index_from_user(9)

        assert tuple(model.camera.center) != start
        assert len(model.layers) == n_layers, "a step must not add or drop a layer"
        after = [(ly.name, id(ly.data))
                 for ly in model.layers if hasattr(ly, "contrast_limits")]
        assert after == before, "a step must not re-point any mosaic's data — nothing is read"
    finally:
        win._fov_slider.shutdown()


def test_the_camera_lands_on_the_field_the_readout_names(qapp):
    """The falsifiable-by-eye check, as an assertion."""
    meta = _sweep_meta()
    win, model = _walker(meta)
    try:
        _open(win)
        boxes = mosaic_fov_bboxes_um(meta, "A1")
        for index, fov in enumerate(boxes):
            win._fov_slider.set_index_from_user(index)
            cy, cx = float(model.camera.center[-2]), float(model.camera.center[-1])
            assert fov_at_point(meta, "A1", cx, cy) == fov, f"camera is not on FOV {fov}"
            x0, y0, x1, y1 = boxes[fov]
            assert (cy, cx) == pytest.approx(((y0 + y1) / 2, (x0 + x1) / 2))
    finally:
        win._fov_slider.shutdown()


def test_the_framed_field_fills_the_canvas(qapp):
    """A field must be framed to FILL the canvas, not sit inside the region's zoom."""
    meta = _sweep_meta()
    win, model = _walker(meta, canvas_hw=(720, 860))
    try:
        _open(win)
        field_um = FRAME_40X * PX_40X
        on_canvas = float(model.camera.zoom) * field_um
        assert on_canvas == pytest.approx(0.95 * 720)      # the tighter axis, with napari's margin
        assert on_canvas < 720, "the framed field must not overflow the canvas"
    finally:
        win._fov_slider.shutdown()


# ══════════════════════════════════════════════════════════════════════════════════════════
# The highlight
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_the_current_fov_reads_as_current_and_follows_the_slider(qapp):
    meta = _sweep_meta()
    win, model = _walker(meta)
    try:
        _open(win)
        layer = _fov_layer(model)
        for index in (0, 7, 15):
            win._fov_slider.set_index_from_user(index)
            colours = [tuple(np.round(c, 4)) for c in layer.edge_color]
            idle = max(set(colours), key=colours.count)
            hot = [i for i, c in enumerate(colours) if c != idle]
            assert hot == [index], f"expected exactly index {index} highlighted, got {hot}"
    finally:
        win._fov_slider.shutdown()


def test_a_zoom_gesture_does_not_wipe_the_highlight(qapp):
    """``_sync_roi_width`` assigns a SCALAR ``edge_width`` on every ``camera.events.zoom``, and napari broadcasts a scalar across every shape."""
    meta = _sweep_meta()
    win, model = _walker(meta)
    try:
        _open(win)
        layer = _fov_layer(model)
        win._fov_slider.set_index_from_user(5)
        before = [tuple(np.round(c, 4)) for c in layer.edge_color]

        model.camera.zoom = float(model.camera.zoom) * 1.7    # fires camera.events.zoom

        after = [tuple(np.round(c, 4)) for c in layer.edge_color]
        assert after == before, "a zoom must not disturb which field reads as current"
    finally:
        win._fov_slider.shutdown()


# ══════════════════════════════════════════════════════════════════════════════════════════
# The playback gate
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_an_fov_step_opens_its_own_playback_gate(qapp):
    """napari closes the gate on every step and only a CANVAS DRAW reopens it."""
    meta = _sweep_meta()
    win, model = _walker(meta)
    try:
        _open(win)
        dims = win._fov_slider._dims
        dims._play_ready = False
        win._fov_slider.set_index_from_user(4)
        assert dims._play_ready is True
    finally:
        win._fov_slider.shutdown()


def test_a_mosaic_landing_does_not_open_the_fov_gate(qapp):
    """``_frame_done`` is about a mosaic arriving; a FOV step does not wait on one."""
    meta = _sweep_meta()
    win, model = _walker(meta)
    try:
        _open(win)
        win._slider = None
        win._time_point_bar = None
        dims = win._fov_slider._dims
        dims._play_ready = False
        win._frame_done()
        assert dims._play_ready is False
    finally:
        win._fov_slider.shutdown()


# ══════════════════════════════════════════════════════════════════════════════════════════
# Refusals — out loud, never a silent empty view
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_a_region_without_stage_positions_says_so_instead_of_drawing_nothing(qapp):
    meta = dict(_sweep_meta(), fov_positions_um={})
    win, model = _walker(meta)
    try:
        win._draw_fov_boxes()
        assert win.said and "no stage positions" in win.said[0]
        assert [ly.name for ly in model.layers if ly.name == "FOVs"] == []
        assert win._fov_slider.fovs == []
        assert win._fov_slider.fov is None
    finally:
        win._fov_slider.shutdown()


def test_a_single_fov_region_still_frames_that_field_and_refuses_to_play(qapp):
    meta = _sweep_meta(n_side=1)
    win, model = _walker(meta)
    try:
        _open(win)
        assert len(_fov_layer(model).data) == 1
        assert win._fov_slider.count == 1
        assert "one FOV" in win._fov_slider._refusal()
        assert fov_at_point(meta, "A1", float(model.camera.center[-1]),
                            float(model.camera.center[-2])) == 0
    finally:
        win._fov_slider.shutdown()


def test_a_view_cannot_be_both_an_roi_child_and_a_fov_walk():
    """No precedence rule and no fallback: one of the two would be silently discarded."""
    from squidxplorer._region_viewer import RegionViewer

    with pytest.raises(ValueError, match="cannot be both"):
        RegionViewer(None, _sweep_meta(), ["A1"], window_id=1,
                     roi_bbox=(0.0, 0.0, 10.0, 10.0), fovs=True)
