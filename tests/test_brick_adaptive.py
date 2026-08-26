"""Zoom-adaptive bricked 3D: the whole-region no-ROI branch, refine-only hysteresis, and the said-once budget note (2026-08-19)."""

from __future__ import annotations

import pytest
from qtpy.QtCore import QObject

from squidxplorer import _volume_view as VV
from squidxplorer._brick_view import BrickedVolume
from squidxplorer._napari_pane import SettleCoalescer


@pytest.fixture
def mosaic():
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    return MosaicLayers(ViewerModel())


def _vol(mosaic, *, budget=1 << 30, window=(0, 32, 0, 32), limit=16, channels=("c0",), say=None):
    """A real BrickedVolume whose loader RECORDS requests instead of running a QThread."""
    vol = BrickedVolume(
        mosaic, reader=None, meta={"z_levels": [0, 1, 2, 3]}, region="A1", window_px=window,
        channels=list(channels), scale=(1.5, 0.75, 0.75), origin_um=(0.0, 0.0, 0.0),
        limit=limit, budget_bytes=budget, say=say)
    requests: list = []
    vol._loader.start = lambda *a, **k: None
    vol._loader.stop = lambda *a, **k: None
    vol._loader.wait = lambda *a, **k: True
    vol._loader.request = lambda jobs, epoch: requests.append((list(jobs), int(epoch)))
    return vol, requests


def _mark_delivered(vol, jobs):
    """Book the requested bricks as resident at the stride they were read at."""
    for brick, ch, step in jobs:
        vol._steps[(ch, brick.key)] = int(step)


# -- zoom -> stride, through the camera the production callback reads ---------------------------


def test_zooming_in_refines_the_planned_stride_to_native(mosaic):
    vol, requests = _vol(mosaic)
    mosaic.model.camera.zoom = 0.25          # 4 um/screen px over 0.75 um voxels -> stride 4
    vol.refresh(force=True)
    assert requests, "the open queued no bricks"
    coarse_jobs = requests[-1][0]
    assert {s for _b, _c, s in coarse_jobs} == {4}
    _mark_delivered(vol, coarse_jobs)

    mosaic.model.camera.zoom = 2.0           # 0.5 um/screen px -> finer than native -> stride 1
    vol.refresh()
    fine_jobs = requests[-1][0]
    assert {s for _b, _c, s in fine_jobs} == {1}, "zooming in did not re-plan at native stride"
    assert len(fine_jobs) == len(coarse_jobs), "refinement missed part of the visible set"


def test_zoom_out_never_coarsens_a_resident_finer_brick(mosaic):
    """Hysteresis: a brick already resident at stride 1 renders at least as well as the coarser plan; re-reading it coarser spends a decode to show less."""
    vol, requests = _vol(mosaic)
    mosaic.model.camera.zoom = 2.0
    vol.refresh(force=True)
    _mark_delivered(vol, requests[-1][0])
    epoch_before, n_requests = vol._epoch, len(requests)

    mosaic.model.camera.zoom = 0.25          # zoom OUT: the plan says stride 4
    vol.refresh()
    assert len(requests) == n_requests, "zooming out re-read bricks already resident finer"
    assert vol._epoch == epoch_before, "a no-op refresh must not invalidate in-flight reads"


def test_finer_residents_are_coarsened_only_when_the_budget_breaks(mosaic):
    """The byte budget outranks hysteresis: finer residents that no longer fit beside the view are re-read at the planned stride, and that is the stated"""
    vol, requests = _vol(mosaic, budget=1 << 30)
    mosaic.model.camera.zoom = 2.0
    vol.refresh(force=True)
    native_jobs = requests[-1][0]
    _mark_delivered(vol, native_jobs)
    native_bytes = sum(b.nbytes(4, 2, s) for b, _c, s in native_jobs)

    vol._budget = native_bytes // 2          # the native residents can no longer all stay
    mosaic.model.camera.zoom = 0.25
    vol.refresh()
    assert len(requests) > 1, "the budget was broken and nothing was re-read"
    steps = {s for _b, _c, s in requests[-1][0]}
    assert 1 not in steps and steps, f"the re-read must be at the coarser planned stride: {steps}"


def test_the_budget_dropped_note_is_said_once_per_open(mosaic):
    said: list = []
    vol, _requests = _vol(mosaic, budget=8, say=said.append)   # too small even at stride 64
    mosaic.model.camera.zoom = 0.25
    vol.refresh(force=True)
    vol.refresh(force=True)
    vol.refresh(force=True)
    dropped_notes = [t for t in said if "left out" in t]
    assert len(dropped_notes) == 1, (
        f"the budget note must be said ONCE per open, not per camera settle: {said}")


def test_a_camera_settle_burst_replans_exactly_once(mosaic):
    """20 zoom events in one gesture are ONE re-plan, fired only after the quiet period."""
    vol, requests = _vol(mosaic)
    mosaic.model.camera.zoom = 2.0
    now = [0.0]
    settle = SettleCoalescer(0.12, vol.refresh, clock=lambda: now[0])
    for _ in range(20):                       # the burst: never quiet long enough
        settle.notify()
        now[0] += 0.01
        settle.poll()
    assert not requests, "a re-plan fired while the camera was still moving"
    now[0] += 0.2
    settle.poll()
    settle.poll()
    assert settle.fired == 1
    assert len(requests) == 1, f"{len(requests)} re-plans for one settled gesture"


# -- the no-ROI branch: the WHOLE region, bricked, in-window ------------------------------------


_PX = 0.75


def _region_meta(nz=3):
    """A 2x2 FOV region of 16x16 frames, 9 um stage step (12 px, 4 px overlap)."""
    fovs = [0, 1, 2, 3]
    return {
        "regions": ["A1"],
        "fovs_per_region": {"A1": fovs},
        "fov_positions_um": {("A1", 0): (0.0, 0.0), ("A1", 1): (9.0, 0.0),
                             ("A1", 2): (0.0, 9.0), ("A1", 3): (9.0, 9.0)},
        "frame_shape": (16, 16),
        "pixel_size_um": _PX,
        "dz_um": 1.5,
        "z_levels": list(range(nz)),
        "channels": [{"name": "c0"}],
    }


class _Pane:
    def __init__(self, mosaic):
        self.mosaic = mosaic
        self.ok = True
        self.settled: list = []

    def _live_max_3d_texture(self):
        return 16                             # small on purpose: the region must BRICK

    def on_camera_settled(self, cb):
        self.settled.append(cb)


class _Cursor:
    region = "A1"


class _Window(QObject):
    def __init__(self, meta, mosaic):
        super().__init__()
        self._meta = meta
        self._reader = object()
        self._pane = _Pane(mosaic) if mosaic is not None else None
        self._cursor = _Cursor()
        self._regions = ["A1"]
        self._roi_bbox = None
        self._native3d = None
        self.said: list = []

    def _say(self, text):
        self.said.append(text)

    def _selected_roi(self):
        return None, None

    def _roi_center_fov(self, region, bbox):
        return 0

    def _refresh_bricks(self):
        VV.refresh_bricks(self)

    def set_render_mode(self, mode):
        pass

    def current_region(self):
        return "A1"


@pytest.fixture
def quiet_loader(monkeypatch):
    """Keep the real BrickedVolume but never start its QThread."""
    started = []
    from squidxplorer._brick_view import _BrickLoader

    monkeypatch.setattr(_BrickLoader, "start", lambda self, *a, **k: started.append(self))
    monkeypatch.setattr(_BrickLoader, "stop", lambda self: None)
    monkeypatch.setattr(_BrickLoader, "wait", lambda self, *a, **k: True)
    return started


def test_the_no_roi_branch_bricks_the_whole_region_in_window(mosaic, quiet_loader, monkeypatch):
    """Julio's goal verbatim: full window 3D rendering w/ bricking."""
    import squidxplorer._napari3d as N3D

    popouts: list = []
    monkeypatch.setattr(N3D, "open_native_3d", lambda *a, **k: popouts.append(a) or object())
    win = _Window(_region_meta(), mosaic)

    VV.open_3d(win)

    assert not popouts, f"3D fell back to the single-FOV popout: {win.said}"
    vol = win._native3d
    assert isinstance(vol, BrickedVolume), f"no in-window volume opened: {win.said}"
    assert vol._window == (0, 28, 0, 28)
    assert vol.brick_count > 1, "a 28 px region over a 16 px texture limit must brick"
    assert win._pane.settled, "the camera-settle callback was never registered: zooming in " \
                              "would never refine the stride"
    vol.close()


def test_a_single_plane_acquisition_keeps_the_single_fov_popout(mosaic, monkeypatch):
    import squidxplorer._napari3d as N3D

    popouts: list = []
    monkeypatch.setattr(N3D, "open_native_3d",
                        lambda *a, **k: popouts.append((a, k)) or object())
    win = _Window(_region_meta(nz=1), mosaic)

    VV.open_3d(win)

    assert popouts, f"a single-plane acquisition lost its 3D popout: {win.said}"
    assert not isinstance(win._native3d, BrickedVolume)


def test_a_paneless_window_falls_back_to_the_popout(monkeypatch):
    import squidxplorer._napari3d as N3D

    popouts: list = []
    monkeypatch.setattr(N3D, "open_native_3d",
                        lambda *a, **k: popouts.append((a, k)) or object())
    win = _Window(_region_meta(), None)

    VV.open_3d(win)

    assert popouts, f"a window without a canvas lost its 3D popout: {win.said}"


def test_region_window_px_is_the_full_mosaic_extent():
    from squidxplorer._napari3d import region_window_px

    assert region_window_px(_region_meta(), "A1") == (0, 28, 0, 28)
    assert region_window_px({}, "A1") is None


def test_the_whole_region_open_states_its_read_cost_in_the_log(mosaic, quiet_loader, caplog):
    import logging

    win = _Window(_region_meta(), mosaic)
    with caplog.at_level(logging.INFO):
        VV.open_3d(win)
    assert isinstance(win._native3d, BrickedVolume)
    assert any("plane decode(s)" in r.message for r in caplog.records), (
        "the whole-region read cost must be CONSCIOUS, never silent")
    win._native3d.close()
