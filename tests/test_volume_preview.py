"""Ruling aa (Julio, 2026-08-25): a preview asked from a 3D tab delivers the FULL DEPTH to that
view as a bricked volume of the result under the channel's own LUT; 2D tabs keep one plane.

Julio's log: a 3D tab over a 1-FOV region (46 z, 2304^2, 3 channels) ran decon for 3 min 05 s
and then delivered ONE plane ("the layer shows z plane 22 of 46 ... only that plane is kept"),
so there was nothing to look at in 3D. And a region whose full-depth result exceeds the display
budget is REFUSED BY NAME BEFORE any plane is read, never solved for minutes and then flattened.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from squidxplorer import _volume_view
from squidxplorer._op_result import RegionResultAccumulator
from squidxplorer._workers import _OperatorWorker
from tests.conftest import FakeReader

pytestmark = pytest.mark.usefixtures("qapp")

REGION = "A1"
CHANNELS = ("488", "561")
N_C = len(CHANNELS)
N_Z = 4
FRAME = 16


def _meta(fovs=(0,), frame=FRAME, nz=N_Z, positions=None) -> dict:
    return {
        "regions": [REGION],
        "channels": [{"name": c} for c in CHANNELS],
        "z_levels": list(range(nz)), "n_z": nz, "n_t": 1, "dtype": "uint16",
        "frame_shape": (frame, frame), "pixel_size_um": 0.75, "dz_um": 1.5,
        "fovs_per_region": {REGION: list(fovs)},
        "fov_positions_um": positions or {(REGION, f): (f * frame * 0.75, 0.0) for f in fovs},
    }


def _plane_value(z: int, c: int) -> int:
    return 1000 + 100 * int(z) + int(c)


def _image_5d() -> np.ndarray:
    out = np.empty((1, N_C, N_Z, FRAME, FRAME), np.uint16)
    for c in range(N_C):
        for z in range(N_Z):
            out[0, c, z] = _plane_value(z, c)
    return out


def _worker(**kw) -> _OperatorWorker:
    return _OperatorWorker("decon", reader=None, meta=_meta(),
                           fov_index={REGION: {"rc": (0, 0), "well_id": "A1"}},
                           out_dir="", regions=[REGION], save=False, n_fovs=None, **kw)


# --- the worker: depth to a volume view, one plane to a plane view ----------------------------

def test_a_volume_views_preview_carries_every_plane_and_says_nothing_about_dropping(caplog):
    worker = _worker(deliver_depth=True, z_level=2)
    got: list = []
    worker.resultReady.connect(lambda region, fov, planes: got.append(planes))
    with caplog.at_level(logging.INFO):
        worker._on_well(REGION, 0, _image_5d())
    planes = np.asarray(got[-1])
    assert planes.shape == (N_C, N_Z, FRAME, FRAME), planes.shape
    for c in range(N_C):
        assert [int(planes[c, z].flat[0]) for z in range(N_Z)] == \
            [_plane_value(z, c) for z in range(N_Z)]
    assert "only that plane is kept" not in caplog.text


def test_a_plane_views_preview_still_gets_one_plane():
    worker = _worker(deliver_depth=False, z_level=2)
    got: list = []
    worker.resultReady.connect(lambda region, fov, planes: got.append(planes))
    worker._on_well(REGION, 0, _image_5d())
    assert np.asarray(got[-1]).shape == (N_C, FRAME, FRAME)


def test_the_accumulator_fuses_a_per_fov_stack_at_full_depth():
    meta = _meta(fovs=(0, 1))
    acc = RegionResultAccumulator("decon", REGION, meta, CHANNELS)
    for f in (0, 1):
        acc.add(f, _image_5d()[0])
    result = acc.result()
    assert result.z_depth == N_Z
    plane = result.plane("561")
    assert plane.shape == (N_Z, FRAME, 2 * FRAME)
    assert [int(plane[z, 0, 0]) for z in range(N_Z)] == [_plane_value(z, 1) for z in range(N_Z)]


# --- the view: the result becomes THIS view's bricked volume under its own LUT ----------------

@pytest.fixture
def stub_bricked(monkeypatch):
    import squidxplorer._brick_view as BV

    class _Stubbed(BV.BrickedVolume):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._loader.start = lambda *x, **k: None
            self._loader.stop = lambda *x, **k: None
            self._loader.wait = lambda *x, **k: True
            self._frame_camera = lambda *x, **k: None
            self.refresh = lambda *x, **k: None

    monkeypatch.setattr(BV, "BrickedVolume", _Stubbed)
    return _Stubbed


def _view(meta, reader, calls):
    from squidxplorer._region_viewer import RegionViewer

    return RegionViewer(reader, meta, [REGION], window_id=33,
                        operator_specs=[("decon", "Deconvolution"), ("mip", "MIP")],
                        run_operator=lambda key, **kw: calls.append((key, kw)))


def test_a_3d_tabs_result_lands_as_a_full_depth_volume_of_the_result(qapp, napari_pane_stub,
                                                                     stub_bricked):
    from squidxplorer._napari_view import key_of
    from squidxplorer._napari_pane import _colormap_for

    meta = _meta()
    reader = FakeReader(meta, planes=lambda r, f, c, z, t: np.full((FRAME, FRAME), 5, np.uint16))
    win = _view(meta, reader, [])
    try:
        mosaic = win._pane.mosaic
        for c in CHANNELS:
            mosaic.add_mosaic("raw", c, np.full((N_Z, FRAME, FRAME), 5, np.uint16),
                              bbox_um=(0.0, 0.0, 12.0, 12.0), z_scale_um=1.5)
        win.set_render_mode("3d")
        _volume_view.open_3d(win)                      # the 3D tab: raw bricks are up
        assert win._native3d is not None and win._native3d._op == "raw", win._pane.said

        acc = RegionResultAccumulator("decon", REGION, meta, CHANNELS)
        acc.add(0, _image_5d()[0])
        added = win.deliver_result("decon", acc.result(), visible=True)
        assert added == N_C

        vol = win._native3d
        assert vol is not None and vol._op == "decon", (
            f"the volume up is {getattr(vol, '_op', None)!r}; said {win._pane.said}")
        read = vol._loader._read
        assert read is not None, "3D fell back to the raw reader instead of the result layer"
        for ch_i, ch in enumerate(CHANNELS):
            for brick in vol._bricks:
                voxels = read(vol._offset_brick(brick), ch, 1, None)
                assert voxels is not None and voxels.shape[0] == N_Z
                assert [int(voxels[z].flat[0]) for z in range(N_Z)] == \
                    [_plane_value(z, ch_i) for z in range(N_Z)]
                vol._on_brick(brick, ch, voxels, 1, vol._epoch)   # land it, the real path
            layers = mosaic.layers_for("decon", ch)
            assert layers, f"no (decon, {ch}) volume layer"
            assert all(int(np.asarray(ly.data).shape[0]) == N_Z for ly in layers)
            assert all(key_of(ly).op == "decon" for ly in layers)
            want = getattr(_colormap_for(ch, meta["channels"]), "name", None)
            got = getattr(layers[0].colormap, "name", None)
            assert want is None or got == want, f"{ch}: LUT {got!r}, the channel's is {want!r}"
        assert mosaic.visible_op() == "decon"
    finally:
        win.dispose()


# --- the refusal: before any plane is read ----------------------------------------------------

def test_an_over_budget_3d_preview_is_refused_by_name_before_any_read(qapp, napari_pane_stub,
                                                                       monkeypatch):
    monkeypatch.setattr(_volume_view, "_brick_budget_bytes", lambda: 64 << 20)
    meta = _meta(fovs=(0, 1, 2, 3), frame=2304, nz=46)
    calls: list = []
    reader = FakeReader(meta, planes=lambda *k: np.zeros((2304, 2304), np.uint16))
    win = _view(meta, reader, calls)
    try:
        win.set_render_mode("3d")
        win._preview_view_operator()
        assert not calls, "the run was launched"
        assert reader.reads == [], "a plane was read before the refusal"
        said = " ".join(win._pane.said)
        assert "3D preview over 4 FOV(s) x 46 planes" in said and "draw an ROI" in said, said
        assert "GB" in said
        # the same scope from a 2D tab is not refused: the full solve runs, ONE plane lands
        win.set_render_mode("2d")
        win._preview_view_operator()
        assert calls and calls[-1][1]["deliver_depth"] is False
        assert "preview_z_level" not in calls[-1][1]
    finally:
        win.dispose()


def test_a_small_3d_preview_launches_asking_for_depth(qapp, napari_pane_stub):
    meta = _meta()
    calls: list = []
    win = _view(meta, FakeReader(meta), calls)
    try:
        win.set_render_mode("3d")
        win._preview_view_operator()
        assert calls, win._pane.said
        kw = calls[-1][1]
        assert kw["deliver_depth"] is True and "preview_z_level" not in kw
    finally:
        win.dispose()
