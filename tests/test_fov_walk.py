"""Walking a region's FOVs with the camera — the boxes, the highlight, and the playback gate.

WHY THESE NEED A REAL ``ViewerModel``. The suite's stub pane (``conftest.napari_pane_stub``) hands
out a viewer with ``dims`` and ``layers`` and nothing else: no ``camera``, no ``add_shapes``. So
none of what this feature actually DOES is visible through it — the camera never moves, the FOV
Shapes layer cannot be built, and the highlight has nothing to write to. That is the same blindness
``_region_viewer`` records for region transitions ("the pane STUB the rest of the suite uses
records ``add_mosaic`` and returns"), and the escape is the one ``test_time_point_playback`` uses:
a real ``napari.components.ViewerModel``, which is Qt-free, has a real evented camera, and reports
a real ``_canvas_size``.

WHAT IS ASSERTED, AND WHY IT IS THE CAMERA RATHER THAN THE PIXELS. On the time axis the right
assertion is the pixels, because stepping changes them. Here NOTHING about the pixels changes —
that is the entire design — so the observable is where the camera is pointing, and the instrument
that makes it falsifiable is ``fov_at_point``: the same readout the canvas prints under the user's
cursor must name the field the camera was just framed on. A camera that lands half a frame out
still shows a perfectly plausible picture, and that assertion is the only thing that catches it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # headless Qt; must precede any Qt import

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from qtpy.QtWidgets import QApplication  # noqa: E402

from squidmip._mosaic_source import fov_at_point, mosaic_bbox_um, mosaic_fov_bboxes_um  # noqa: E402

napari = pytest.importorskip("napari")


@pytest.fixture(scope="module")
def qapp():
    """This module's QApplication, by the same convention every other GUI test module here uses.

    NOT pytest-qt's fixture of the same name: ``tools/run_suite_chunked.py`` runs with
    ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``, so pytest-qt is not loaded and its ``qapp`` does not
    exist. A test that only ever ran under a bare ``pytest`` passes locally and errors at SETUP in
    the suite.
    """
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)   # main() won't call exec_/exit under test
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
    """A ``RegionViewer`` in FOVs mode, wired to a real ``MosaicLayers``, without a window.

    Built with ``__new__`` and the handful of attributes the FOV walk touches rather than through
    ``__init__``: constructing a real window needs a napari QMainWindow and a GL canvas, and every
    behaviour here is reachable without one. What is NOT stubbed is the part under test — the
    layers, the camera and the slider are all production objects.
    """
    from napari.components import ViewerModel

    from squidmip._fov_nav import FovSlider
    from squidmip._napari_view import MosaicLayers
    from squidmip._region_viewer import RegionViewer

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

        # Each rectangle is that field's own box, in napari's (y, x) world.
        for i, (fov, (x0, y0, x1, y1)) in enumerate(boxes.items()):
            got = np.asarray(layer.data[i])
            assert got[:, -2].min() == pytest.approx(y0)
            assert got[:, -2].max() == pytest.approx(y1)
            assert got[:, -1].min() == pytest.approx(x0)
            assert got[:, -1].max() == pytest.approx(x1)
    finally:
        win._fov_slider.shutdown()


def test_a_fov_box_is_never_clamped_to_the_texture_ceiling(qapp):
    """The FOV boxes must not go through the ROI layer's drawing-time clamp.

    ``_clamp_last_roi`` holds the last-drawn shape to ``GL_MAX_3D_TEXTURE_SIZE`` — 2048 px on the
    Apple floor, against a 4168 px 40x frame. That clamp is a promise about a box the USER dragged
    and it is the right promise; applied to a box the ACQUISITION drew it would silently shrink
    the last field by more than half and report "ROI held to the 3D ceiling" about something
    nobody drew. Separate layer, so no clamp — asserted on the LAST field, which is the one the
    clamp would have taken.
    """
    meta = _sweep_meta()
    win, model = _walker(meta)
    try:
        _open(win)
        layer = _fov_layer(model)
        last = np.asarray(layer.data[-1])
        field_um = FRAME_40X * PX_40X
        # napari stores Shapes data as FLOAT32. One ulp at this region's far corner (~19 400 um)
        # is 19400 * 1.19e-7 ~= 2.3e-3 um, and a box width is the difference of two such numbers,
        # so ~5e-3 um of slack is the STORAGE and nothing to do with the geometry. Derived rather
        # than tuned, so it cannot quietly grow to cover a real error: the clamp this test exists
        # to catch would take more than half the field, which is 5 orders of magnitude bigger.
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
        # The IMAGE layers are the pixels. (A Shapes layer's `.data` is rebuilt on each access, so
        # its id is not an identity — and the highlight legitimately writes that layer's colours.)
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
    """The falsifiable-by-eye check, as an assertion.

    ``fov_at_point`` is what the canvas prints under the cursor. If the camera centre does not
    report the field the slider says it is on, the user is looking at a plausible picture of a
    different field — which is exactly what using the plate's half-frame-offset box convention
    here would produce, with no error anywhere.
    """
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
    """A field must be framed to FILL the canvas, not sit inside the region's zoom.

    This is the entire point of the feature — the user's complaint is the wheel-zoom cycle — so
    the zoom is asserted against the field, not merely against "it changed".
    """
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
    """``_sync_roi_width`` assigns a SCALAR ``edge_width`` on every ``camera.events.zoom``, and
    napari broadcasts a scalar across every shape. A width-based highlight would therefore work
    until the user's first wheel click and then vanish — intermittently, which is worse than never
    having worked at all. Colour survives it; this is the assertion that keeps it colour.
    """
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
    """napari closes the gate on every step and only a CANVAS DRAW reopens it.

    ``AxisPlayback`` drives a ``Dims`` with no canvas behind it, so without an explicit open the
    axis advances exactly one frame and then sits until the 180 s stall watchdog fires. Asserted
    directly so that failure takes a millisecond instead of three minutes.
    """
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
    """``_frame_done`` is about a mosaic arriving; a FOV step does not wait on one.

    Routing the FOV axis through it would let a timepoint reload — or a RETIRED load, which
    ``_on_plane``'s generation check exists to drop — advance a FOV animation.
    """
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
        # The ORDER, not `AxisPlayback.count` — a napari dims range of (0, 0, 1) reports a count of
        # 1 whether it holds one position or none, so the widget's own axis cannot express "empty".
        # This is the same reason `FovSlider._refusal` counts from the order it was given.
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
        # It still framed it — asking for one field is a useful thing to have asked for.
        assert fov_at_point(meta, "A1", float(model.camera.center[-1]),
                            float(model.camera.center[-2])) == 0
    finally:
        win._fov_slider.shutdown()


def test_a_view_cannot_be_both_an_roi_child_and_a_fov_walk():
    """No precedence rule and no fallback: one of the two would be silently discarded."""
    from squidmip._region_viewer import RegionViewer

    with pytest.raises(ValueError, match="cannot be both"):
        RegionViewer(None, _sweep_meta(), ["A1"], window_id=1,
                     roi_bbox=(0.0, 0.0, 10.0, 10.0), fovs=True)
