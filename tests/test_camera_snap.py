"""The 3D camera snap chips: XY/XZ/YZ/fit, in camera only, offered only on a volume tab."""

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


def test_the_snap_chips_exist_only_on_a_volume_tab(qapp, napari_pane_stub, squid_dataset):
    """A 2D tab has no 3D camera: the chips are HIDDEN there (not disabled clutter), and
    ``note_volume_tab`` — the one fact that makes a view the 3D tab — shows them."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    v = win._viewer_manager.open([list(win._order)[0]])
    assert v is not None
    _drain_until(qapp, lambda: v._pane is not None, timeout=10)
    try:
        chips = v._snap_chips
        assert [c.text() for c in chips] == ["XY", "XZ", "YZ", "fit"]
        assert all(c.isHidden() for c in chips), "a 2D tab offers the 3D snap chips"
        v.note_volume_tab()
        assert all(not c.isHidden() for c in chips), "a volume tab hides its own snap chips"
        assert all(c.isEnabled() for c in chips)
        assert all(c.toolTip() for c in chips), "a chip must say what it does"
    finally:
        shutdown_plate_window(qapp, win)


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
