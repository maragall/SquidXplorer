"""The canvas loupe: shift-left-click magnifies, the wheel changes the factor, the camera stays put.

DISPATCH IS NAPARI'S OWN. Every gesture below goes through
``napari.utils.interactions.mouse_press_callbacks`` / ``mouse_wheel_callbacks`` against a real
``napari.components.ViewerModel``, which is Qt-free. That is production dispatch with no GL, and it
exercises the generator protocol a drag callback actually runs under rather than a hand-rolled
stand-in that would agree with whatever the implementation happens to do.

THE ONE THAT MATTERS MOST is ``test_the_wheel_changes_magnification_and_never_zooms_the_camera``.
Everything about wheel-to-magnify rests on ``event.handled`` being honoured by napari's vispy
canvas, which is read off napari's source rather than promised anywhere. If a napari upgrade
changes it, that test is what says so — otherwise the symptom is a canvas that lurches every time
the user adjusts the loupe.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # headless Qt; must precede any Qt import

import threading  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from qtpy.QtWidgets import QApplication, QWidget  # noqa: E402

from squidxplorer._loupe import (  # noqa: E402
    _LOUPE_PX,
    canvas_scale,
    capped_at_native,
    loupe_inset_rect,
    loupe_label,
    loupe_scale,
    loupe_scale_at,
)

napari = pytest.importorskip("napari")

PX_40X = 0.094
FRAME = 4168
PITCH = 2742.4064
X0, Y0 = 10780.7125, 8021.6375


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


def _meta(n_side: int = 2) -> dict:
    positions = {}
    for i in range(n_side * n_side):
        row, col = divmod(i, n_side)
        positions[("A1", i)] = (X0 + col * PITCH, Y0 + row * PITCH)
    return {
        "pixel_size_um": PX_40X,
        "frame_shape": (FRAME, FRAME),
        "z_levels": [0],
        "n_t": 1,
        "channels": [{"name": "ch0", "display_color": "#FFFFFF"}],
        "regions": ["A1"],
        "fovs_per_region": {"A1": list(range(n_side * n_side))},
        "fov_positions_um": positions,
    }


class _Event:
    """A napari mouse event, shaped the way ``VispyCanvas`` builds one."""

    def __init__(self, **kw):
        self.handled = False
        self.button = 1
        self.modifiers = ()
        self.type = "mouse_press"
        self.pos = (400, 300)
        self.position = (0.0, 0.0)
        self.camera_zoom = 1.0
        self.delta = (0.0, 1.0)
        self.__dict__.update(kw)


class _Source:
    """A loupe source that answers instantly and records the thread it was read on."""

    n_levels = 1
    well_px = FRAME
    pixel_size_um = PX_40X

    def __init__(self):
        self.threads: list = []
        self.calls: list = []

    def available(self, well_id):
        return True, ""

    def read_crop(self, well_id, level, y0, x0, h, w, time_point=0, fov=None):
        self.threads.append(threading.get_ident())
        self.calls.append((well_id, level, y0, x0, h, w, time_point, fov))
        return np.zeros((1, max(1, h), max(1, w)), np.float32)

    def coarse(self, well_id, time_point=0):
        return np.zeros((1, 8, 8), np.float32)

    def window(self, well_id, time_point=0):
        return [(0.0, 1.0)]


def _loupe(qapp, meta=None, zoom=1.0, source=None):
    from napari.components import ViewerModel

    from squidxplorer._mosaic_source import mosaic_bbox_um
    from squidxplorer._napari_loupe import CanvasLoupe
    from squidxplorer._napari_view import MosaicLayers

    meta = meta or _meta()
    model = ViewerModel()
    model._canvas_size = (720, 860)
    mosaic = MosaicLayers(model)
    mosaic.add_mosaic("raw", "ch0", np.zeros((64, 64), np.uint16), contrast_limits=(0, 1),
                      colormap="gray", multiscale=False, bbox_um=mosaic_bbox_um(meta, "A1"))
    model.camera.zoom = zoom

    host = QWidget()
    host.resize(860, 720)
    src = source if source is not None else _Source()
    said: list = []
    loupe = CanvasLoupe(
        viewer=model, canvas_widget=host, meta=meta,
        source_for=lambda op: src, mosaic=mosaic,
        region_of=lambda: "A1", time_point_of=lambda: 0,
        look_of=lambda: (["ch0"], np.array([[1.0, 1.0, 1.0]], np.float32), [(0.0, 1.0)], [True]),
        say=said.append)
    return loupe, model, host, src, said


def _centre_of(meta, fov: int):
    from squidxplorer._mosaic_source import mosaic_fov_bboxes_um

    x0, y0, x1, y1 = mosaic_fov_bboxes_um(meta, "A1")[fov]
    return (y0 + y1) / 2, (x0 + x1) / 2      # world order is (y, x)


# ══════════════════════════════════════════════════════════════════════════════════════════
# The gesture
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_shift_left_click_raises_the_loupe_and_suppresses_the_pan(qapp):
    """``handled`` is the ONLY thing between this gesture and a camera pan."""
    from napari.utils.interactions import mouse_press_callbacks

    meta = _meta()
    loupe, model, _host, _src, _said = _loupe(qapp, meta)
    try:
        ev = _Event(modifiers=("Shift",), position=_centre_of(meta, 0))
        mouse_press_callbacks(model, ev)
        assert loupe._up is True
        assert ev.handled is True, "an unhandled shift-press pans the canvas out from under it"
    finally:
        loupe.shutdown()


def test_the_wheel_changes_magnification_and_never_zooms_the_camera(qapp):
    """THE load-bearing test. If a napari upgrade stops honouring ``handled``, this says so."""
    from napari.utils.interactions import mouse_press_callbacks, mouse_wheel_callbacks

    from squidxplorer._napari_loupe import _MAG_LADDER

    meta = _meta()
    loupe, model, _host, _src, _said = _loupe(qapp, meta)
    try:
        mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=_centre_of(meta, 0)))
        before = float(model.camera.zoom)
        start = loupe._mag_index

        ev = _Event(type="mouse_wheel", delta=(0.0, 1.0))
        mouse_wheel_callbacks(model, ev)

        assert ev.handled is True, "an unhandled wheel zooms the canvas instead of the loupe"
        assert loupe._mag_index == start + 1
        assert float(model.camera.zoom) == before, "the camera must not move while the loupe is up"

        down = _Event(type="mouse_wheel", delta=(0.0, -1.0))
        mouse_wheel_callbacks(model, down)
        assert loupe._mag_index == start
        assert _MAG_LADDER[loupe._mag_index] > 0
    finally:
        loupe.shutdown()


def test_the_wheel_reaches_the_camera_again_once_the_loupe_is_down(qapp):
    """The negative half. A gesture that suppresses zoom permanently is worse than no gesture."""
    from napari.utils.interactions import mouse_wheel_callbacks

    loupe, model, _host, _src, _said = _loupe(qapp)
    try:
        ev = _Event(type="mouse_wheel", delta=(0.0, 1.0))
        mouse_wheel_callbacks(model, ev)
        assert ev.handled is False
    finally:
        loupe.shutdown()


def test_the_ladder_clamps_and_never_wraps(qapp):
    """32x -> 2x on one more wheel click would be a magnifier lying about what you are seeing."""
    from napari.utils.interactions import mouse_press_callbacks, mouse_wheel_callbacks

    from squidxplorer._napari_loupe import _MAG_LADDER

    meta = _meta()
    loupe, model, _host, _src, said = _loupe(qapp, meta)
    try:
        mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=_centre_of(meta, 0)))
        for _ in range(20):
            mouse_wheel_callbacks(model, _Event(type="mouse_wheel", delta=(0.0, 1.0)))
        assert loupe._mag_index == len(_MAG_LADDER) - 1
        assert any("as far as it magnifies" in s for s in said), "the ceiling must be said, not felt"

        for _ in range(20):
            mouse_wheel_callbacks(model, _Event(type="mouse_wheel", delta=(0.0, -1.0)))
        assert loupe._mag_index == 0
    finally:
        loupe.shutdown()


def test_a_plain_click_dismisses_and_is_not_handled(qapp):
    """The click still pans and still selects an ROI — it only takes the panel out of the way."""
    from napari.utils.interactions import mouse_press_callbacks

    meta = _meta()
    loupe, model, _host, _src, _said = _loupe(qapp, meta)
    try:
        mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=_centre_of(meta, 0)))
        assert loupe._up
        ev = _Event(modifiers=(), position=_centre_of(meta, 0))
        mouse_press_callbacks(model, ev)
        assert loupe._up is False
        assert ev.handled is False, "a plain click must keep doing what it always did"
    finally:
        loupe.shutdown()


def test_shift_clicking_again_dismisses_it(qapp):
    from napari.utils.interactions import mouse_press_callbacks

    meta = _meta()
    loupe, model, _host, _src, _said = _loupe(qapp, meta)
    try:
        pt = _centre_of(meta, 0)
        mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=pt))
        assert loupe._up
        mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=pt))
        assert loupe._up is False
    finally:
        loupe.shutdown()


def test_it_refuses_in_3d_by_name_instead_of_reading_a_ray(qapp):
    from napari.utils.interactions import mouse_press_callbacks

    meta = _meta()
    loupe, model, _host, _src, said = _loupe(qapp, meta)
    try:
        model.dims.ndisplay = 3
        mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=(0.0, 0.0, 0.0)))
        assert loupe._up is False
        assert said and "2-D" in said[0]
    finally:
        loupe.shutdown()


# ══════════════════════════════════════════════════════════════════════════════════════════
# What it reads, and on which thread
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_the_loupe_reads_the_fov_under_the_cursor(qapp):
    """Sweeping every field must ask the source for THAT field, not for field 0 four times."""
    import time

    from napari.utils.interactions import mouse_press_callbacks

    meta = _meta(n_side=2)
    loupe, model, _host, src, _said = _loupe(qapp, meta)
    try:
        for fov in meta["fovs_per_region"]["A1"]:
            src.calls.clear()
            mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=_centre_of(meta, fov)))
            deadline = time.monotonic() + 10.0
            while not src.calls and time.monotonic() < deadline:
                qapp.processEvents()
                time.sleep(0.005)
            assert src.calls, f"no read was requested for FOV {fov}"
            assert src.calls[-1][-1] == fov, f"asked for the wrong field: {src.calls[-1]}"
            loupe.dismiss()
    finally:
        loupe.shutdown()


def test_a_point_in_a_gap_between_fields_says_so_rather_than_magnifying_a_neighbour(qapp):
    from napari.utils.interactions import mouse_press_callbacks

    meta = _meta(n_side=2)
    loupe, model, _host, src, _said = _loupe(qapp, meta)
    try:
        # The pitch is 7x the field, so the midpoint between two fields is empty stage.
        x_gap = X0 + PITCH / 2
        mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=(Y0, x_gap)))
        qapp.processEvents()
        assert src.calls == [], "there are no pixels there; nothing should have been read"
        assert loupe._inset._note == "no field here"
    finally:
        loupe.shutdown()


def test_the_loupe_never_reads_a_plane_on_the_qt_thread(qapp):
    """CLAUDE.md: nothing decodes on the Qt thread. Recorded, not assumed.

    The wait is on WALL CLOCK and not on a fixed number of ``processEvents`` turns: the read
    happens on the worker's own thread, so how many GUI turns pass before it is scheduled is up to
    the OS. A spin count makes this test fail on a busy machine and prove nothing on a fast one.
    """
    import time

    from napari.utils.interactions import mouse_press_callbacks

    meta = _meta()
    loupe, model, _host, src, _said = _loupe(qapp, meta)
    try:
        gui = threading.get_ident()
        mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=_centre_of(meta, 0)))
        deadline = time.monotonic() + 10.0
        while not src.threads and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert src.threads, "the read never happened, so this proves nothing"
        assert gui not in src.threads, "a crop was decoded on the GUI thread"
    finally:
        loupe.shutdown()


# ══════════════════════════════════════════════════════════════════════════════════════════
# The arithmetic: one rule, two vocabularies
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_the_two_entry_points_are_one_rule():
    """``loupe_scale`` must be ``loupe_scale_at`` with the ratio taken first, exactly."""
    for cd in (10.0, 88.0, 240.0, 900.0):
        for well_px in (256, 2084, 4168):
            assert loupe_scale(cd, well_px) == loupe_scale_at(cd / well_px, well_px)


def test_the_canvas_and_the_plate_agree_on_one_field():
    """Two surfaces drawing a field at the SAME scale must magnify it identically.

    This is what makes ``canvas_scale`` a translation rather than a second rule: the plate says
    "so many screen px per well", the canvas says "so many canvas px per micrometre", and once
    both are reduced to screen-px-per-image-px there is nothing left to disagree about.
    """
    field_px = FRAME
    zoom = 1.74582                                   # camera px per um, one field in a 720 canvas
    s_canvas = canvas_scale(zoom, PX_40X)
    on_screen = s_canvas * field_px                  # what the plate would call `cd`
    assert loupe_scale_at(s_canvas, field_px) == loupe_scale(on_screen, field_px)


def test_the_native_cap_is_reported_rather_than_absorbed():
    """Asking 32x on a surface that can only give 6.1x must SAY 6.1x, and say why.

    Measured on the real 40x sets: a 4168 px frame framed to fill an 860x720 window is a 6.1x
    downsample, so every rung from 8x up returns the same picture. Silence there is a control that
    moves and does nothing.
    """
    s_canvas = canvas_scale(1.74582, PX_40X)
    _s, m2 = loupe_scale_at(s_canvas, FRAME, mag=2.0)
    _s, m32 = loupe_scale_at(s_canvas, FRAME, mag=32.0)

    assert m2 == pytest.approx(2.0), "a factor the surface CAN give must be given exactly"
    assert not capped_at_native(m2, 2.0)
    assert m32 == pytest.approx(1.0 / s_canvas, rel=1e-6), "capped at 1:1, never upsampling"
    assert capped_at_native(m32, 32.0)

    assert "32× asked" in loupe_label("A1 fov 5", m32, requested=32.0)
    assert "native" in loupe_label("A1 fov 5", m32, requested=32.0)
    assert "native" not in loupe_label("A1 fov 5", m2, requested=2.0)


def test_the_inset_stays_whole_inside_its_host():
    """It flips sides at the edges rather than hanging off, at every corner."""
    for x, y in ((0, 0), (859, 0), (0, 719), (859, 719), (430, 360)):
        bx, by = loupe_inset_rect(x, y, 860, 720)
        assert 0 <= bx and bx + _LOUPE_PX <= 860
        assert 0 <= by and by + _LOUPE_PX <= 720


# ══════════════════════════════════════════════════════════════════════════════════════════
# Lifecycle
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_the_inset_never_steals_a_mouse_event(qapp):
    """It has no controls, so every event must pass through to the canvas underneath."""
    from qtpy.QtCore import Qt

    loupe, _model, _host, _src, _said = _loupe(qapp)
    try:
        assert loupe._inset.testAttribute(Qt.WA_TransparentForMouseEvents)
    finally:
        loupe.shutdown()


def test_shutdown_unhooks_from_napari_and_is_idempotent(qapp):
    """napari's callback lists hold a STRONG reference — a window that stayed registered would
    keep itself alive and keep handling events against a dead canvas."""
    loupe, model, _host, _src, _said = _loupe(qapp)
    n_drag = len(model.mouse_drag_callbacks)
    n_wheel = len(model.mouse_wheel_callbacks)
    loupe.shutdown()
    assert len(model.mouse_drag_callbacks) == n_drag - 1
    assert len(model.mouse_wheel_callbacks) == n_wheel - 1
    loupe.shutdown()          # a window can be disposed and then still receive a closeEvent
    assert len(model.mouse_drag_callbacks) == n_drag - 1


def test_shutdown_joins_the_worker(qapp):
    """A QThread destroyed while running aborts the process; closing a tab mid-read is ordinary."""
    from napari.utils.interactions import mouse_press_callbacks

    meta = _meta()
    loupe, model, _host, _src, _said = _loupe(qapp, meta)
    mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=_centre_of(meta, 0)))
    worker = loupe._worker
    assert worker is not None, "the gesture must have built a worker for this to mean anything"
    loupe.shutdown()
    assert loupe._worker is None
    assert not worker.isRunning()


def test_the_chosen_magnification_is_remembered_across_windows(qapp, tmp_path, monkeypatch):
    """A factor the user picked with the wheel must still be there for the NEXT window.

    Otherwise the wheel has to be re-taught to every window, which is the opposite of a setting.
    SESSION-scoped, not a prefs file: `_prefs` went with the 2026-08-13 kill list, so the memory
    is `_napari_loupe._SESSION_MAG` and it resets on the next launch — the same trade the
    close-all checkbox took.
    """
    from napari.utils.interactions import mouse_press_callbacks, mouse_wheel_callbacks

    from squidxplorer import _napari_loupe
    from squidxplorer._napari_loupe import _MAG_LADDER, _default_mag_index

    monkeypatch.setattr(_napari_loupe, "_SESSION_MAG", float(_napari_loupe._LOUPE_MAG))

    meta = _meta()
    loupe, model, _host, _src, _said = _loupe(qapp, meta)
    try:
        mouse_press_callbacks(model, _Event(modifiers=("Shift",), position=_centre_of(meta, 0)))
        start = loupe._mag_index
        mouse_wheel_callbacks(model, _Event(type="mouse_wheel", delta=(0.0, 1.0)))
        chosen = _MAG_LADDER[loupe._mag_index]
        assert loupe._mag_index == start + 1
    finally:
        loupe.shutdown()

    assert _napari_loupe._SESSION_MAG == chosen
    assert _MAG_LADDER[_default_mag_index()] == chosen, "the next window must open where this left"


def test_a_nonsense_remembered_factor_snaps_to_the_ladder(qapp, monkeypatch):
    """A stale session value must not leave the wheel somewhere it cannot reach."""
    from squidxplorer import _napari_loupe
    from squidxplorer._napari_loupe import _MAG_LADDER, _default_mag_index

    monkeypatch.setattr(_napari_loupe, "_SESSION_MAG", 9.3)
    assert _MAG_LADDER[_default_mag_index()] == 8.0, "nearest rung, not a value off the ladder"

    monkeypatch.setattr(_napari_loupe, "_SESSION_MAG", "not a number")
    assert _MAG_LADDER[_default_mag_index()] in _MAG_LADDER
