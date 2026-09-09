"""The Blender-style viewport camera gizmo: a compact overlay in the canvas's top-right
corner showing the live 3D orientation (Julio, 2026-09-09: "the camera controls shouldn't
be buttons. Maybe something in the top right corner like blender has").

Three colored axis handles, projected through the CURRENT ``camera.angles`` so the gizmo
turns with the volume (it repaints on ``camera.events.angles``, so a scripted orbit turns
it too). Click an axis handle to snap to the view looking down it (Z is the XY top view,
Y is XZ, X is YZ: the existing presets), click the center to refit, drag anywhere on the
disc to orbit live. Every camera write goes through :func:`_volume_view.snap_camera`, the
app's ONE writer of ``camera.angles`` (source-scan-pinned); a drag is a pure rotation, so
the release triggers ``refresh_bricks`` itself, exactly as ``snap_camera``'s docstring
requires.

The geometry (projection, hit-testing, the drag-to-angles mapping) is pure functions over
an angle triple; the widget only drives them, so offscreen tests pin the behavior and the
real GL look stays a hand check. The angle mapping is the one MEASURED in
``_camera_script`` (``ry`` yaws toward x, ``90 - rz`` tilts toward y, ``rx`` is roll);
the camera basis comes from napari's own ``Camera`` model, never a second Euler
derivation. Hosted as a child of the pane's GL canvas widget, the arrangement
``MosaicPane.canvas_widget`` documents as supported (napari parents its own welcome
widget there); the loupe inset is the in-repo precedent.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QColor, QFont, QPainter, QPen
from qtpy.QtWidgets import QWidget

from squidxplorer._logpane import get_logger

log = get_logger("camera_gizmo")

#: Widget edge, px (the disc fills it).
GIZMO_PX = 90
#: Gap between the gizmo and the canvas's top-right corner.
MARGIN_PX = 8
#: Center-to-handle distance and handle radius.
AXIS_LEN_PX = 32
HANDLE_R_PX = 9
#: The fit target in the middle.
CENTER_R_PX = 6
#: A handle projected closer to the center than this is edge-on (you are already looking
#: down it); it stops being clickable so the center's fit stays reachable under it.
CENTER_DEAD_PX = 14
#: Orbit gain, degrees of camera per pixel of drag (Blender's viewport is ~0.4).
DRAG_DEG_PER_PX = 0.4
#: A press that moves less than this is a click, not a drag.
DRAG_START_PX = 3

#: Blender-like axis colors, as (r, g, b).
AXIS_COLORS = {"x": (230, 80, 80), "y": (110, 180, 60), "z": (70, 130, 230)}

#: Click-to-snap: each handle lands the preset view looking DOWN its axis.
AXIS_SNAP = {"z": "xy", "y": "xz", "x": "yz"}

#: World unit vectors in napari's (z, y, x) order.
_AXIS_WORLD = {"x": (0.0, 0.0, 1.0), "y": (0.0, 1.0, 0.0), "z": (1.0, 0.0, 0.0)}


def camera_basis(angles) -> tuple:
    """``(right, up, forward)`` unit vectors in napari world (z, y, x) for a camera at
    *angles*, off napari's own ``Camera`` model (its Euler math, not a reimplementation).
    ``up`` points up on the canvas; ``forward`` points from the camera INTO the scene.
    """
    from napari.components import Camera

    cam = Camera(angles=tuple(float(v) for v in angles))
    forward = np.asarray(cam.view_direction, dtype=float)
    up = np.asarray(cam.up_direction, dtype=float)
    # Pinned at the XY top view: right must be world +x (screen x grows right).
    right = np.cross(forward, up)
    return right, up, forward


def project_axes(angles, length_px: float = AXIS_LEN_PX) -> dict:
    """Each axis handle's ``(dx, dy, depth)`` offset from the gizmo center, widget px.

    ``dy`` grows DOWN (widget coordinates, matching the canvas where +y is down at the
    top view); ``depth`` is the axis's component along the view direction, so a negative
    depth points at the viewer (near) and a positive one away (far, drawn dimmer).
    """
    right, up, forward = camera_basis(angles)
    out = {}
    for name, world in _AXIS_WORLD.items():
        a = np.asarray(world, dtype=float)
        out[name] = (float(np.dot(a, right)) * float(length_px),
                     -float(np.dot(a, up)) * float(length_px),
                     float(np.dot(a, forward)))
    return out


def hit_test(x: float, y: float, angles, size: int = GIZMO_PX) -> Optional[str]:
    """What a point of the widget hits: an axis name, ``"center"``, or ``None``.

    Near handles win an overlap (they are drawn on top); a handle inside the dead zone is
    edge-on and yields to the center.
    """
    cx = cy = size / 2.0
    offsets = project_axes(angles)
    for name, (dx, dy, depth) in sorted(offsets.items(), key=lambda kv: kv[1][2]):
        if math.hypot(dx, dy) < CENTER_DEAD_PX:
            continue
        if math.hypot(x - cx - dx, y - cy - dy) <= HANDLE_R_PX:
            return name
    if math.hypot(x - cx, y - cy) <= CENTER_R_PX + 3:
        return "center"
    return None


def drag_angles(start_angles, dx_px: float, dy_px: float,
                gain: float = DRAG_DEG_PER_PX) -> tuple:
    """The angle triple a drag from *start_angles* lands on: the ball rolls with the mouse.

    The measured ``_camera_script`` mapping: ``ry`` yaws toward x, ``90 - rz`` tilts
    toward y, ``rx`` is roll and a gizmo drag never rolls. A rightward drag brings the
    scene's -x face around (the view direction gains +x); a downward drag tips the top
    toward the viewer (the view direction gains +y).
    """
    rx, ry, rz = (float(v) for v in start_angles)
    return (rx, ry - float(dx_px) * gain, rz - float(dy_px) * gain)


class CameraGizmo(QWidget):
    """The overlay widget. Paints the projected axes; every camera write it makes goes
    through ``snap_camera`` (clicks settle; drag frames pass ``settle=False`` and the
    release refreshes the bricks itself). Hidden until ``set_active(True)`` (volume tabs
    only, exactly the rule the snap chips had).
    """

    def __init__(self, host: QWidget, win) -> None:
        super().__init__(host)
        self._win = win
        self._host = host
        self._press: Optional[tuple] = None      # ((x, y), start angles)
        self._dragging = False
        self._hover: Optional[str] = None
        self._cam_events = None
        self.setFixedSize(GIZMO_PX, GIZMO_PX)
        self.setFocusPolicy(Qt.NoFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click an axis to snap the camera to its view, click the center "
                        "to refit, drag to orbit.")
        host.installEventFilter(self)            # tracks the canvas's resizes
        self._connect_camera()
        self._reposition()
        self.hide()

    # -- wiring --------------------------------------------------------------------------

    def _camera(self):
        viewer = self._win._napari_viewer()
        return None if viewer is None else viewer.camera

    def _connect_camera(self) -> None:
        cam = self._camera()
        if cam is None:
            return
        try:
            cam.events.angles.connect(self._on_angles)
            self._cam_events = cam.events.angles
        except Exception as exc:                 # noqa: BLE001 - a repaint tap, never fatal
            log.debug("camera gizmo: no angles event to follow (%s)", exc)

    def _on_angles(self, _event=None) -> None:
        self.update()

    def shutdown(self) -> None:
        """Take the taps out: napari's event list and the canvas's filter both hold strong
        references, the same leak shape the loupe documents."""
        if self._cam_events is not None:
            try:
                self._cam_events.disconnect(self._on_angles)
            except Exception:                    # noqa: BLE001 - already torn down
                pass
            self._cam_events = None
        try:
            self._host.removeEventFilter(self)
        except (RuntimeError, TypeError):
            pass

    def set_active(self, on: bool) -> None:
        """Volume tabs only: ``note_volume_tab`` turns it on; a 2D tab never shows it."""
        self.setVisible(bool(on))
        if on:
            self._reposition()
            self.raise_()

    def eventFilter(self, obj, event):           # noqa: N802 - Qt naming
        if obj is self._host and event.type() == QEvent.Resize:
            self._reposition()
        return False

    def _reposition(self) -> None:
        host = self.parentWidget()
        if host is not None:
            self.move(max(0, host.width() - GIZMO_PX - MARGIN_PX), MARGIN_PX)

    def _angles(self) -> Optional[tuple]:
        cam = self._camera()
        return None if cam is None else tuple(float(v) for v in cam.angles)

    # -- interaction ---------------------------------------------------------------------

    def mousePressEvent(self, event):            # noqa: N802 - Qt naming
        x, y = event.pos().x(), event.pos().y()
        c = GIZMO_PX / 2.0
        angles = self._angles()
        if event.button() != Qt.LeftButton or angles is None or (
                math.hypot(x - c, y - c) > c):
            event.ignore()                       # the corners belong to the canvas
            return
        self._press = ((x, y), angles)
        self._dragging = False
        event.accept()

    def mouseMoveEvent(self, event):             # noqa: N802 - Qt naming
        x, y = event.pos().x(), event.pos().y()
        if self._press is None:
            angles = self._angles()
            hover = None if angles is None else hit_test(x, y, angles)
            if hover != self._hover:
                self._hover = hover
                self.update()
            return
        from squidxplorer import _volume_view

        (x0, y0), start = self._press
        dx, dy = x - x0, y - y0
        if not self._dragging and math.hypot(dx, dy) < DRAG_START_PX:
            return
        self._dragging = True
        _volume_view.snap_camera(self._win, drag_angles(start, dx, dy), settle=False)

    def mouseReleaseEvent(self, event):          # noqa: N802 - Qt naming
        from squidxplorer import _volume_view

        press, self._press = self._press, None
        if press is None:
            event.ignore()
            return
        if self._dragging:
            self._dragging = False
            # A drag is a pure rotation: zoom and center never moved, so the settle hook
            # never fires; the refinement is this release's own job (the snap-camera rule).
            _volume_view.refresh_bricks(self._win)
            return
        hit = hit_test(event.pos().x(), event.pos().y(), press[1])
        if hit == "center":
            _volume_view.snap_camera(self._win, "fit")
        elif hit in AXIS_SNAP:
            _volume_view.snap_camera(self._win, AXIS_SNAP[hit])

    def leaveEvent(self, event):                 # noqa: N802 - Qt naming
        if self._hover is not None:
            self._hover = None
            self.update()
        super().leaveEvent(event)

    # -- painting ------------------------------------------------------------------------

    def paintEvent(self, _event):                # noqa: N802 - Qt naming
        angles = self._angles()
        if angles is None:
            return
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            c = GIZMO_PX / 2.0
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(20, 20, 20, 110 if self._hover or self._press else 70))
            p.drawEllipse(1, 1, GIZMO_PX - 2, GIZMO_PX - 2)
            font = QFont(self.font())
            font.setPixelSize(10)
            font.setBold(True)
            p.setFont(font)
            offsets = project_axes(angles)
            # Far first, near last, so a near handle paints over a far one.
            for name, (dx, dy, depth) in sorted(
                    offsets.items(), key=lambda kv: -kv[1][2]):
                r, g, b = AXIS_COLORS[name]
                near = depth <= 0.0
                alpha = 235 if near else 110
                hx, hy = c + dx, c + dy
                pen = QPen(QColor(r, g, b, alpha))
                pen.setWidthF(2.0)
                p.setPen(pen)
                p.drawLine(int(c), int(c), int(hx), int(hy))
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(r, g, b, alpha))
                p.drawEllipse(int(hx - HANDLE_R_PX), int(hy - HANDLE_R_PX),
                              2 * HANDLE_R_PX, 2 * HANDLE_R_PX)
                if near:
                    p.setPen(QColor(20, 20, 20, 230))
                    p.drawText(int(hx - HANDLE_R_PX), int(hy - HANDLE_R_PX),
                               2 * HANDLE_R_PX, 2 * HANDLE_R_PX,
                               Qt.AlignCenter, name.upper())
            hover_center = self._hover == "center"
            p.setPen(QPen(QColor(255, 255, 255, 200 if hover_center else 120), 1.5))
            p.setBrush(QColor(255, 255, 255, 70 if hover_center else 35))
            p.drawEllipse(int(c - CENTER_R_PX), int(c - CENTER_R_PX),
                          2 * CENTER_R_PX, 2 * CENTER_R_PX)
        finally:
            p.end()
