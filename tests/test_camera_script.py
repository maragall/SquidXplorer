"""The scriptable 3D camera: declarative pose sequence, residency-gated dwell, recorded pans.

Sequencing and pose math run against a real ``ViewerModel``; the canvas capture is a stubbed
seam (offscreen has no GL), so the recorded LOOK is a hand check on a live window.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from squidxplorer import _camera_script as CS
from squidxplorer import _volume_view

from .conftest import build_volume_scene


@pytest.fixture
def mosaic():
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    return MosaicLayers(ViewerModel())


class _Shell:
    """The duck shell ``_volume_view``'s docstring licenses: ``_native3d`` as an attribute."""

    def __init__(self, vol):
        self._native3d = vol
        self.said = []

    def _say(self, text):
        self.said.append(text)


def _volume_shell(mosaic):
    vol = build_volume_scene(mosaic, "raw", ("488",), bricks=1)
    vol._viewer.dims.ndisplay = 3
    return _Shell(vol), vol


def test_the_script_walks_its_poses_in_order_and_dwell_counts_after_residency(mosaic):
    """Julio's sentence as steps: XY 3 s, 45 degrees across the diagonal, XZ 3 s, YZ.
    Pins the order, each preset by ``camera.view_direction``, the explicit pose by its own
    triple (an explicit pose's contract IS its angles), and that every dwell follows its
    step's residency wait - never the other way round."""
    shell, vol = _volume_shell(mosaic)
    vol.refresh = lambda *a, **k: None
    events = []

    def wait_ready(_win):
        vd = np.abs(np.asarray(vol._viewer.camera.view_direction, dtype=float))
        events.append(("ready", tuple(np.round(vd, 6)), tuple(vol._viewer.camera.angles)))
        return True

    def sleep(seconds):
        events.append(("dwell", float(seconds)))

    result = CS.run_camera_script(shell, [
        CS.Step("xy", dwell_s=3),
        CS.Step((45.0, 45.0, 45.0), dwell_s=3),
        CS.Step("xz", dwell_s=3),
        CS.Step("yz"),
    ], transition_s=0.0, sleep=sleep, wait_ready=wait_ready)

    kinds = [e[0] for e in events]
    assert kinds == ["ready", "dwell", "ready", "dwell", "ready", "dwell", "ready"]
    readies = [e for e in events if e[0] == "ready"]
    for i, axis in ((0, 0), (2, 1), (3, 2)):             # xy -> z, xz -> y, yz -> x (zyx order)
        expected = np.zeros(3)
        expected[axis] = 1.0
        assert np.allclose(readies[i][1], expected, atol=1e-6), (
            f"step {i} looks along {readies[i][1]}, not the {'zyx'[axis]} axis")
    assert readies[1][2] == pytest.approx((45.0, 45.0, 45.0))
    assert [e[1] for e in events if e[0] == "dwell"] == [3.0, 3.0, 3.0]
    assert shell.said == [], shell.said
    assert result.path is None and result.n_frames == 0
    assert result.poses == ("xy", (45.0, 45.0, 45.0), "xz", "yz")


def test_a_recording_pans_between_poses_and_latches_contrast_once(mosaic):
    """record=out_path: dwell is frames at the stated fps, the pan interpolates the gap's
    intermediate angles monotonically, and the volume's contrast latch runs exactly once."""
    shell, vol = _volume_shell(mosaic)
    vol.refresh = lambda *a, **k: None
    latched = []
    orig_latch = vol.latch_contrast
    vol.latch_contrast = lambda: latched.append(True) or orig_latch()
    captured = []

    def capture():
        captured.append(tuple(vol._viewer.camera.angles))
        return np.zeros((4, 4, 3), np.uint8)

    def writer(frames, out_path, fps):
        return str(out_path), sum(1 for _ in frames)

    result = CS.run_camera_script(
        shell, [CS.Step("xy", dwell_s=1), CS.Step("yz", dwell_s=1)],
        out_path="figure.mp4", fps=10, transition_s=0.5,
        capture=capture, wait_ready=lambda _w: True, writer=writer)

    # 10 dwell frames, 4 pan frames (5-frame pan, endpoints owned by the snaps), 10 dwell.
    assert result.n_frames == 24 and len(captured) == 24
    pan = captured[10:14]
    ry = [a[1] for a in pan]
    assert ry == sorted(ry) and 0.0 < ry[0] and ry[-1] < 90.0, (
        f"the pan does not walk ry from xy to yz monotonically: {ry}")
    assert len(latched) == 1, "contrast must latch ONCE for the whole recording"
    assert shell.said == [], shell.said


def test_an_unknown_pose_is_refused_by_name_before_the_camera_moves(mosaic):
    shell, vol = _volume_shell(mosaic)
    before = tuple(vol._viewer.camera.angles)
    with pytest.raises(ValueError, match="'zz'"):
        CS.run_camera_script(shell, [CS.Step("xy", 1), CS.Step("zz", 1)],
                             wait_ready=lambda _w: True, sleep=lambda _s: None)
    assert tuple(vol._viewer.camera.angles) == before, "a refused script moved the camera"


def test_the_script_module_writes_camera_angles_nowhere_itself():
    """``snap_camera`` stays the app's ONE writer: the script drives it for every pose."""
    import re

    src = pathlib.Path(CS.__file__).read_text()
    assert not re.search(r"camera\.angles\s*=", src)


def test_an_explicit_triple_reaches_the_camera_through_snap_camera(mosaic):
    """The extension the script rides on: ``snap_camera`` takes an (rx, ry, rz) triple, and
    ``settle=False`` turns the camera without touching zoom or center (a pan frame)."""
    shell, vol = _volume_shell(mosaic)
    vol.refresh = lambda *a, **k: None
    _volume_view.snap_camera(shell, "xy")
    cam = vol._viewer.camera
    center, zoom = tuple(cam.center), float(cam.zoom)

    _volume_view.snap_camera(shell, (10.0, 20.0, 30.0), settle=False)

    assert tuple(cam.angles) == pytest.approx((10.0, 20.0, 30.0))
    assert tuple(cam.center) == pytest.approx(center) and float(cam.zoom) == pytest.approx(zoom)
    assert shell.said == [], shell.said


def test_a_live_run_pans_between_poses_at_the_stated_pace(mosaic):
    """The orbit plays smoothly LIVE too: pan frames between poses, one 1/fps wait each."""
    shell, vol = _volume_shell(mosaic)
    vol.refresh = lambda *a, **k: None
    sleeps = []
    CS.run_camera_script(shell, [CS.Step("xy"), CS.Step("yz", dwell_s=1)],
                         fps=10, transition_s=0.5,
                         sleep=sleeps.append, wait_ready=lambda _w: True)
    assert sleeps == [0.1, 0.1, 0.1, 0.1, 1.0], sleeps


def test_a_zoom_step_multiplies_the_framed_zoom(mosaic):
    """``Step.zoom`` fills the frame after the snap's own fit; a zoom-less step stays at fit."""
    shell, vol = _volume_shell(mosaic)
    vol.refresh = lambda *a, **k: None
    CS.run_camera_script(shell, [CS.Step("xy")],
                         wait_ready=lambda _w: True, sleep=lambda _s: None)
    fitted = float(vol._viewer.camera.zoom)
    CS.run_camera_script(shell, [CS.Step("xy", zoom=1.15)],
                         wait_ready=lambda _w: True, sleep=lambda _s: None)
    assert float(vol._viewer.camera.zoom) == pytest.approx(fitted * 1.15)


def test_the_demo_orbit_storyboard_holds_its_35_degree_cone(mosaic):
    """The canned SOP demo: opens XY, visits XZ and YZ, ends back on the hero; every
    oblique pose lands on the 35 degree cone (tilt off z by ``camera.view_direction``)
    at azimuths 30 -> 80 -> 130, zoomed to fill; axial poses stay at fit."""
    steps = CS.demo_orbit_steps()
    assert steps[0].pose == "xy" and steps[0].zoom is None
    names = [s.pose for s in steps if isinstance(s.pose, str)]
    assert names == ["xy", "xz", "yz"]
    assert steps[-1].pose == CS.OBLIQUE_HERO

    shell, vol = _volume_shell(mosaic)
    vol.refresh = lambda *a, **k: None
    azimuths = []
    for step in steps:
        if isinstance(step.pose, str):
            continue
        assert step.zoom == pytest.approx(CS.DEMO_ZOOM)
        from squidxplorer import _volume_view

        _volume_view.snap_camera(shell, step.pose, settle=False)
        vd = np.asarray(vol._viewer.camera.view_direction, dtype=float)
        tilt = np.degrees(np.arccos(min(1.0, abs(vd[0]))))
        assert tilt == pytest.approx(35.0, abs=0.2), f"{step.pose} tilts {tilt:.1f}"
        azimuths.append(round(float(np.degrees(np.arctan2(-vd[2], vd[1])))))
    assert azimuths == [30, 80, 130, 30]


def _comparison_scene(mosaic, op="decon"):
    """Raw flat layers AND an operator volume in ONE viewer: both identities live."""
    from .conftest import build_flat_scene

    build_flat_scene(mosaic, "raw", ("488",))
    vol = build_volume_scene(mosaic, op, ("488",), bricks=1)
    vol._viewer.dims.ndisplay = 3
    return _Shell(vol), vol


def test_the_comparison_is_two_passes_composed_with_visibility_restored(mosaic):
    """Pass A raw-only, pass B result-only, same steps so the camera matches by
    construction; equal frame counts, composite width 2w + divider, and the user's
    visibility comes back exactly."""
    shell, vol = _comparison_scene(mosaic)
    vol.refresh = lambda *a, **k: None
    before = [(ly.name, bool(ly.visible)) for ly in mosaic.model.layers]
    seen = []

    def capture():
        raw = mosaic.find("raw", "488")
        seen.append((bool(raw.visible if raw is not None else False),
                     any(ly.visible for ly in mosaic.layers_for("decon", "488"))))
        return np.zeros((6, 8, 3), np.uint8)

    frames = []

    def writer(gen, out_path, fps):
        frames.extend(gen)
        return str(out_path), len(frames)

    result = CS.record_comparison(
        shell, "cmp.mp4", steps=[CS.Step("xy", dwell_s=0.5)], fps=4,
        capture=capture, wait_ready=lambda _w: True, sleep=lambda _s: None, writer=writer)

    half = len(seen) // 2
    assert seen[:half] and all(s == (True, False) for s in seen[:half]), seen
    assert all(s == (False, True) for s in seen[half:]), seen
    assert result.n_frames == half == 2
    assert all(f.shape == (6, 8 + CS.DIVIDER_PX + 8, 3) for f in frames)
    assert [(ly.name, bool(ly.visible)) for ly in mosaic.model.layers] == before
    assert shell.said == [], shell.said


def test_the_comparison_refuses_a_missing_side_by_name(mosaic):
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    shell, _vol = _volume_shell(mosaic)            # raw volume only: no operator side
    with pytest.raises(ValueError, match="no operator result"):
        CS.record_comparison(shell, "cmp.mp4", steps=[CS.Step("xy")])

    op_only = MosaicLayers(ViewerModel())
    vol2 = build_volume_scene(op_only, "decon", ("488",), bricks=1)
    with pytest.raises(ValueError, match="no raw layer"):
        CS.record_comparison(_Shell(vol2), "cmp.mp4", steps=[CS.Step("xy")])
