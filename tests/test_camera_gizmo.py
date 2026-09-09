"""The Blender-style camera gizmo: projection, hit-testing, drag mapping, and the widget
driving them, all through ``snap_camera`` (the app's one writer of ``camera.angles``)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import math  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidxplorer import _camera_gizmo as G  # noqa: E402
from squidxplorer import _volume_view  # noqa: E402

from .conftest import build_volume_scene  # noqa: E402
from .test_viewer import qapp  # noqa: E402,F401  (fixture)

#: An oblique pose where all three handles are clear of the dead zone.
OBLIQUE = (0.0, 30.0, 60.0)


@pytest.fixture
def mosaic():
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    return MosaicLayers(ViewerModel())


class _Shell:
    """The duck the widget drives: ``_native3d`` for ``snap_camera``, ``_napari_viewer``
    for the camera the gizmo reads (on a real pane those share one viewer, as here)."""

    def __init__(self, vol):
        self._native3d = vol
        self.said = []

    def _say(self, text):
        self.said.append(text)

    def _napari_viewer(self):
        return self._native3d._viewer


def _volume_shell(mosaic):
    vol = build_volume_scene(mosaic, "raw", ("488",), bricks=1)
    vol._viewer.dims.ndisplay = 3
    return _Shell(vol), vol


def _mouse(kind, x, y):
    from qtpy.QtCore import QPointF, Qt
    from qtpy.QtGui import QMouseEvent

    try:
        return QMouseEvent(kind, QPointF(x, y), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    except TypeError:                            # Qt6 wants the global position too
        return QMouseEvent(kind, QPointF(x, y), QPointF(x, y),
                           Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)


def _handle_point(angles, axis):
    """The widget point over *axis*'s handle at *angles*."""
    dx, dy, _depth = G.project_axes(angles)[axis]
    c = G.GIZMO_PX / 2.0
    return c + dx, c + dy


def _click(giz, x, y):
    from qtpy.QtCore import QEvent

    giz.mousePressEvent(_mouse(QEvent.MouseButtonPress, x, y))
    giz.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, x, y))


# --- the pure geometry ----------------------------------------------------------------------


def test_the_projection_matches_the_canvas_orientation():
    """At the XY top view the gizmo shows what the canvas shows: x grows screen-right, y
    grows screen-down, and z is edge-on pointing AT the viewer (negative depth). At the
    oblique demo pose every handle is clear of the dead zone."""
    off = G.project_axes((0.0, 0.0, 90.0))
    assert off["x"][0] == pytest.approx(G.AXIS_LEN_PX) and abs(off["x"][1]) < 1e-6
    assert off["y"][1] == pytest.approx(G.AXIS_LEN_PX) and abs(off["y"][0]) < 1e-6
    assert math.hypot(off["z"][0], off["z"][1]) < 1e-6 and off["z"][2] < 0

    from squidxplorer._camera_script import OBLIQUE_HERO

    for name, (dx, dy, _d) in G.project_axes(OBLIQUE_HERO).items():
        assert math.hypot(dx, dy) >= G.CENTER_DEAD_PX, f"{name} is edge-on at the hero pose"


def test_hit_testing_names_handles_center_and_nothing():
    """Each handle hits its own axis; the middle hits the fit target even when an edge-on
    axis sits over it (the dead zone yields to the center); a corner hits nothing."""
    c = G.GIZMO_PX / 2.0
    for axis in ("x", "y", "z"):
        x, y = _handle_point(OBLIQUE, axis)
        assert G.hit_test(x, y, OBLIQUE) == axis
    assert G.hit_test(c, c, (0.0, 0.0, 90.0)) == "center", "the edge-on z stole the fit"
    assert G.hit_test(2.0, 2.0, OBLIQUE) is None


def test_a_drag_yaws_toward_x_tilts_toward_y_and_never_rolls():
    """The measured ``_camera_script`` mapping, pinned through ``camera.view_direction``
    ((z, y, x) world): a rightward drag gains +x, a downward drag gains +y, and ``rx``
    (roll) never moves."""
    from napari.components import Camera

    start = (0.0, 0.0, 90.0)
    right = G.drag_angles(start, 20.0, 0.0)
    down = G.drag_angles(start, 0.0, 20.0)
    assert right[0] == start[0] and down[0] == start[0], "a gizmo drag rolled the camera"
    vd_right = np.asarray(Camera(angles=right).view_direction, dtype=float)
    vd_down = np.asarray(Camera(angles=down).view_direction, dtype=float)
    assert vd_right[2] > 0 and abs(vd_right[1]) < 1e-6, f"drag right went {vd_right}"
    assert vd_down[1] > 0 and abs(vd_down[2]) < 1e-6, f"drag down went {vd_down}"


def test_the_gizmo_module_writes_camera_angles_nowhere_itself():
    """``snap_camera`` stays the app's ONE writer: the gizmo drives it for every write."""
    src = pathlib.Path(G.__file__).read_text()
    assert not re.search(r"camera\.angles\s*=", src)


# --- the widget -----------------------------------------------------------------------------


@pytest.fixture
def gizmo(qapp, mosaic):
    from qtpy.QtWidgets import QWidget

    shell, vol = _volume_shell(mosaic)
    refined = []
    vol.refresh = lambda *a, **k: refined.append(True)
    host = QWidget()
    host.resize(400, 300)
    giz = G.CameraGizmo(host, shell)
    yield giz, shell, vol, refined
    giz.shutdown()
    host.deleteLater()


def test_each_handle_click_lands_its_snap_view(gizmo):
    """Z lands the XY top view, Y lands XZ, X lands YZ: the existing presets, pinned by
    ``camera.view_direction`` in (z, y, x) world order, never the angle triple. A click
    settles (frames and refines), exactly what the chips did."""
    giz, shell, vol, refined = gizmo
    for axis, world_axis in (("z", 0), ("y", 1), ("x", 2)):
        _volume_view.snap_camera(shell, OBLIQUE, settle=False)
        angles = tuple(vol._viewer.camera.angles)
        x, y = _handle_point(angles, axis)
        del refined[:]
        _click(giz, x, y)
        vd = np.abs(np.asarray(vol._viewer.camera.view_direction, dtype=float))
        expected = np.zeros(3)
        expected[world_axis] = 1.0
        assert np.allclose(vd, expected, atol=1e-6), (
            f"the {axis} handle looks along {vd} (z, y, x), not the {'zyx'[world_axis]} axis")
        assert refined, f"the {axis} click left the bricks at the old frustum's stride"
    assert shell.said == [], shell.said


def test_the_center_click_refits_and_keeps_the_rotation(gizmo):
    """The middle is "fit": center and zoom come back to the framed values, the user's
    angles stay exactly where they were."""
    giz, shell, vol, _refined = gizmo
    _volume_view.snap_camera(shell, OBLIQUE)     # settle frames the box at this rotation
    cam = vol._viewer.camera
    framed_center, framed_zoom = tuple(cam.center), float(cam.zoom)
    angles = tuple(cam.angles)
    cam.center = (5.0, 111.0, 222.0)
    cam.zoom = framed_zoom * 0.01

    c = G.GIZMO_PX / 2.0
    _click(giz, c + G.CENTER_R_PX - 1, c)        # inside the fit target, off any handle

    assert tuple(cam.center) == pytest.approx(framed_center), "the center did not refit"
    assert float(cam.zoom) == pytest.approx(framed_zoom)
    assert tuple(cam.angles) == pytest.approx(angles), "fit spun the user's rotation back"
    assert shell.said == [], shell.said


def test_a_drag_orbits_through_snap_camera_only_and_refreshes_on_release(gizmo, monkeypatch):
    """Every drag frame is one ``snap_camera(..., settle=False)`` (the one writer, turning
    the camera alone); the release triggers the brick refinement itself, because a pure
    rotation fires no zoom or center event for the settle hook."""
    from qtpy.QtCore import QEvent

    giz, shell, vol, refined = gizmo
    _volume_view.snap_camera(shell, "xy", settle=False)
    calls = []
    real = _volume_view.snap_camera

    def spy(win, plane, *, settle=True):
        calls.append((plane, settle))
        return real(win, plane, settle=settle)

    monkeypatch.setattr(_volume_view, "snap_camera", spy)
    giz.mousePressEvent(_mouse(QEvent.MouseButtonPress, 30.0, 45.0))
    for x in (34.0, 44.0, 60.0):                 # rightward drag across the disc
        giz.mouseMoveEvent(_mouse(QEvent.MouseMove, x, 45.0))
    del refined[:]
    giz.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, 60.0, 45.0))

    assert calls and all(settle is False for _p, settle in calls)
    assert all(isinstance(p, tuple) and len(p) == 3 for p, _s in calls)
    assert all(p[0] == 0.0 for p, _s in calls), "a drag rolled the camera"
    rys = [p[1] for p, _s in calls]
    assert rys == sorted(rys, reverse=True) and rys[-1] < 0.0, (
        f"a rightward drag does not walk the yaw: {rys}")
    assert tuple(vol._viewer.camera.angles) == pytest.approx(calls[-1][0])
    assert refined, "the release left the bricks at the old frustum's stride"
    assert shell.said == [], shell.said


def test_the_gizmo_repaints_on_every_camera_turn_until_shutdown(gizmo):
    """It subscribes to ``camera.events.angles``, so a scripted orbit turns the drawn
    axes too; ``shutdown`` takes the tap out (napari's event list holds a strong ref)."""
    giz, shell, _vol, _refined = gizmo
    seen = []
    giz.update = lambda *a, **k: seen.append(1)  # instance attr shadows the Qt repaint
    _volume_view.snap_camera(shell, (5.0, 15.0, 25.0), settle=False)
    assert seen, "an angle write did not repaint the gizmo"
    giz.shutdown()
    del seen[:]
    _volume_view.snap_camera(shell, (6.0, 16.0, 26.0), settle=False)
    assert not seen, "a shut-down gizmo is still tapping the camera's events"
