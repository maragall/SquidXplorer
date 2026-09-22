"""The 3D camera snaps: in camera only, driven by the canvas gizmo on a volume tab."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

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

from squidxplorer import _viewer as V  # noqa: E402
from squidxplorer import _volume_view  # noqa: E402

from .conftest import build_volume_scene, shutdown_plate_window  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)


@pytest.fixture
def mosaic():
    """The app's layer model over a bare, Qt-free ``ViewerModel``."""
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    return MosaicLayers(ViewerModel())


class _Shell:
    """The duck shell the ``_volume_view`` docstring licenses: ``_native3d`` as an attribute."""

    def __init__(self, vol):
        self._native3d = vol
        self.said = []

    def _say(self, text):
        self.said.append(text)


def _volume_shell(mosaic):
    vol = build_volume_scene(mosaic, "raw", ("488",), bricks=1)
    vol._viewer.dims.ndisplay = 3
    return _Shell(vol), vol


def test_the_camera_poses_are_a_gizmo_not_buttons(qapp, napari_pane_stub, squid_dataset):
    """Julio, 2026-09-09: "the camera controls shouldn't be buttons." The XY/XZ/YZ/fit
    chips are gone whole; the canvas gizmo owns the poses, hidden on a 2D tab and shown
    by ``note_volume_tab`` (the one fact that makes a view the 3D tab). The orbit chip
    STAYS with the same visibility rule: it plays a scripted sequence, not a camera pose."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    v = win._viewer_manager.open([list(win._order)[0]])
    assert v is not None
    _drain_until(qapp, lambda: v._pane is not None, timeout=10)
    try:
        assert not hasattr(v, "_snap_chips"), "the snap-chip bookkeeping is back"
        giz = v._camera_gizmo
        assert giz is not None, "the pane came up without a camera gizmo"
        orbit = v._btn_orbit
        assert orbit.text() == "orbit" and orbit.toolTip()
        assert giz.isHidden() and orbit.isHidden(), "a 2D tab offers the 3D camera"
        v.note_volume_tab()
        assert not giz.isHidden(), "a volume tab hides its own camera gizmo"
        assert not orbit.isHidden() and orbit.isEnabled()
    finally:
        shutdown_plate_window(qapp, win)


def test_the_orbit_chip_plays_the_demo_live_and_disables_while_it_runs(
        qapp, napari_pane_stub, squid_dataset, monkeypatch):
    """One chip, volume tabs only, no recording: it PLAYS ``demo_orbit_steps()`` in place
    and cannot be double-started (disabled for the run's whole duration)."""
    from squidxplorer import _camera_script

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    v = win._viewer_manager.open([list(win._order)[0]])
    _drain_until(qapp, lambda: v._pane is not None, timeout=10)
    try:
        v.note_volume_tab()
        calls = []

        def fake_run(w, steps, **kw):
            calls.append((w, [s.pose for s in steps], v._btn_orbit.isEnabled()))
            return _camera_script.ScriptResult(None, 0, ())

        monkeypatch.setattr(_camera_script, "run_camera_script", fake_run)
        v._play_orbit()
        assert len(calls) == 1
        w, poses, enabled_during = calls[0]
        assert w is v and poses == [s.pose for s in _camera_script.demo_orbit_steps()]
        assert enabled_during is False, "the chip stayed clickable while the orbit ran"
        assert v._btn_orbit.isEnabled(), "the chip never came back after the run"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_movie_chip_on_a_volume_tab_records_the_camera_orbit(
        qapp, napari_pane_stub, squid_dataset, monkeypatch, tmp_path):
    """Julio, 2026-09-22: "we should add a button of exporting a video of the 3D
    rendering. Just like we had for the 2D png." On a volume tab the movie chip IS that
    button: a save dialog, then the camera-script recording of the demo orbit at the
    script's own fps. The 2D sweep path is untouched (tests/test_video_window.py pins
    it end to end on a 2D tab)."""
    from qtpy.QtWidgets import QFileDialog

    from squidxplorer import _camera_script, _video

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    v = win._viewer_manager.open([list(win._order)[0]])
    _drain_until(qapp, lambda: v._pane is not None, timeout=10)
    try:
        monkeypatch.setattr(_video, "encoder_problem", lambda: None)
        v.note_volume_tab()
        assert v._btn_record.isEnabled(), "the movie chip is dead on a volume tab"
        tip = v._btn_record.toolTip()
        assert "orbit" in tip and "zoom" in tip, tip
        v._native3d = object()               # a volume is up; the stub has no bricks yet
        out = tmp_path / "orbit.mp4"
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "Movie (*.mp4)")))
        calls = []

        def fake_run(w, steps, **kw):
            calls.append((w, [s.pose for s in steps], kw,
                          v._btn_record.isEnabled()))
            return _camera_script.ScriptResult(str(out), 7, ())

        monkeypatch.setattr(_camera_script, "run_camera_script", fake_run)
        v._record_movie()
        assert len(calls) == 1, "the volume tab's movie chip did not record the orbit"
        w, poses, kw, enabled_during = calls[0]
        assert w is v
        assert poses == [s.pose for s in _camera_script.demo_orbit_steps()], (
            "a brickless stub volume must record the chapterless demo orbit")
        assert kw["out_path"] == str(out) and kw["fps"] == _camera_script.DEFAULT_FPS
        assert enabled_during is False, "the chip stayed clickable while recording"
        assert v._btn_record.isEnabled(), "the chip never came back after the run"
        assert v._video_worker is None, "the orbit recording must not start the 2D sweep"
    finally:
        shutdown_plate_window(qapp, win)


def test_volume_zoom_targets_reads_the_bright_structure_and_the_demo_gains_the_dive():
    """The zoom chapter's target is DATA-chosen: the center of mass of the top 1%
    brightest z-MIP pixels of the resident bricks, in world um; home is the box center.
    With a target, ``demo_orbit_steps`` grows the glide chapter around it."""
    from squidxplorer import _camera_script

    class _Layer:
        def __init__(self):
            self.data = np.zeros((2, 8, 8), np.uint16)
            self.data[:, 6, 2] = 1000
            self.translate = (0.0, 10.0, 20.0)
            self.scale = (3.0, 1.0, 1.0)

    class _Vol:
        _layers = {"k": _Layer()}
        _origin_um = (0.0, 10.0, 20.0)
        _scale = (3.0, 1.0, 1.0)
        _window = (0, 8, 0, 8)
        _nz = 2

    target, home = _camera_script.volume_zoom_targets(_Vol())
    assert home == pytest.approx((3.0, 14.0, 24.0))
    assert target == pytest.approx((3.0, 16.5, 22.5)), (
        "the target must sit on the bright structure, not the box center")

    plain = _camera_script.demo_orbit_steps()
    dived = _camera_script.demo_orbit_steps(zoom_center=target, home_center=home)
    assert len(dived) == len(plain) + 3, "the zoom chapter is three glide steps"
    glides = [s for s in dived if s.glide]
    assert glides and glides[0].center == pytest.approx(target)
    assert glides[0].zoom == _camera_script.ZOOM_GLIDE_FACTOR
    assert glides[-1].center == pytest.approx(home)
    assert _camera_script.volume_zoom_targets(object()) == (None, None), (
        "an unreadable volume must refuse quietly, not guess a chapter")


def test_each_snap_points_the_camera_down_its_own_axis(mosaic):
    """The pin is ``camera.view_direction`` in napari's (z, y, x) world order, never the
    angle triple itself; and every snap refines the bricks (a pure rotation fires no
    zoom/center event, so the settle alone would never run)."""
    shell, vol = _volume_shell(mosaic)
    refined = []
    vol.refresh = lambda *a, **k: refined.append(True)

    for plane, axis in (("xy", 0), ("xz", 1), ("yz", 2)):
        _volume_view.snap_camera(shell, plane)
        vd = np.abs(np.asarray(vol._viewer.camera.view_direction, dtype=float))
        expected = np.zeros(3)
        expected[axis] = 1.0
        assert np.allclose(vd, expected, atol=1e-6), (
            f"{plane} looks along {vd} (z, y, x), not the {'zyx'[axis]} axis")

    assert shell.said == [], shell.said
    assert len(refined) == 3, "a snap left the bricks at the old frustum's stride"


def test_fit_reframes_and_leaves_the_rotation_alone(mosaic):
    """"fit" is the volume's own framing and nothing else: center and zoom come back to the
    framed values, the user's angles stay exactly where they were."""
    shell, vol = _volume_shell(mosaic)
    vol.refresh = lambda *a, **k: None
    _volume_view.snap_camera(shell, "yz")
    cam = vol._viewer.camera
    framed_center, framed_zoom = tuple(cam.center), float(cam.zoom)
    angles = tuple(cam.angles)

    cam.center = (5.0, 111.0, 222.0)
    cam.zoom = framed_zoom * 0.01
    _volume_view.snap_camera(shell, "fit")

    assert tuple(cam.center) == pytest.approx(framed_center), "fit did not reframe the box"
    assert float(cam.zoom) == pytest.approx(framed_zoom)
    assert tuple(cam.angles) == pytest.approx(angles), "fit spun the user's rotation back"
    assert shell.said == [], shell.said
