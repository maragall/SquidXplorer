"""A preview solves the z THE VIEW IS ON, and the result lands on that same z.

Julio (2026-08-25): "Looks like decon runs on a z-level that's not in view. Then when it
finishes the decon z-layer is different to the raw z-layer in view." The per-FOV display
path hardcoded plane 0 (`image[0, :, 0]`), so a depth-keeping operator's preview showed a
plane the user was not looking at, under the plane they were.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._workers import _OperatorWorker

pytestmark = pytest.mark.usefixtures("qapp")

CHANNELS = ("405", "488")
N_C = len(CHANNELS)
N_Z = 4
FRAME = 16
REGION = "A1"


def _plane_value(z: int, c: int) -> int:
    return 1000 + 100 * int(z) + int(c)


def _meta() -> dict:
    return {
        "regions": [REGION],
        "channels": [{"name": c} for c in CHANNELS],
        "z_levels": list(range(N_Z)),
        "n_z": N_Z,
        "n_t": 1,
        "dtype": "uint16",
        "frame_shape": (FRAME, FRAME),
        "pixel_size_um": 0.75,
        "dz_um": 1.5,
        "fovs_per_region": {REGION: [0]},
        "fov_positions_um": {(REGION, 0): (0.0, 0.0)},
    }


def _image_5d() -> np.ndarray:
    """A depth-keeping operator's yield: ``(T, C, Nz, Y, X)``, each plane self-numbering."""
    out = np.empty((1, N_C, N_Z, FRAME, FRAME), np.uint16)
    for c in range(N_C):
        for z in range(N_Z):
            out[0, c, z] = _plane_value(z, c)
    return out


def _worker(z_level: int) -> _OperatorWorker:
    return _OperatorWorker(
        "decon", reader=None, meta=_meta(),
        fov_index={REGION: {"rc": (0, 0), "well_id": "A1"}},
        out_dir="", regions=[REGION], save=False, n_fovs=None,
        z_level=z_level,
    )


def test_the_preview_plane_is_the_z_the_view_is_on(caplog):
    """z_level=2 in the view means the result layer gets THE SOLVED PLANE 2, said by name."""
    import logging

    worker = _worker(z_level=2)
    got: list = []
    worker.resultReady.connect(lambda region, fov, planes: got.append(planes))
    with caplog.at_level(logging.INFO):
        worker._on_well(REGION, 0, _image_5d())

    planes = np.asarray(got[-1])
    assert planes.shape == (N_C, FRAME, FRAME), "the per-FOV path grew a z axis"
    for c in range(N_C):
        assert int(planes[c].flat[0]) == _plane_value(2, c), (
            f"channel {CHANNELS[c]}: the layer got plane value {int(planes[c].flat[0])}, "
            f"not the in-view z=2's {_plane_value(2, c)}")
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "z plane 2 of 4" in said, f"the shown plane was not named. log: {said!r}"


def test_a_z_beyond_the_result_depth_is_clamped_not_an_index_error():
    """A z-reducing operator yields depth 1; a view sitting on z=3 still gets that one plane."""
    worker = _worker(z_level=3)
    got: list = []
    worker.resultReady.connect(lambda region, fov, planes: got.append(planes))
    flat = _image_5d()[:, :, :1]                  # (1, C, 1, Y, X): a mip-like yield
    worker._on_well(REGION, 0, flat)
    planes = np.asarray(got[-1])
    assert planes.shape == (N_C, FRAME, FRAME)
    for c in range(N_C):
        assert int(planes[c].flat[0]) == _plane_value(0, c)


def test_the_view_run_carries_its_z_and_the_dims_stay_put(qapp, napari_pane_stub):
    """The window's Run passes ITS z to the engine, and a landing result leaves dims alone."""
    from squidxplorer._op_result import RegionResultAccumulator
    from squidxplorer._region_viewer import RegionViewer

    calls: list = []

    def _capture_run(key, **kw):
        calls.append((key, kw))

    meta = _meta()
    win = RegionViewer(None, meta, [REGION], window_id=91,
                       operator_specs=[("decon", "Deconvolution")],
                       run_operator=_capture_run)
    try:
        viewer = win._napari_viewer()
        # a z-stack raw layer, so the model has a real z axis to sit on
        win._pane.mosaic.add_mosaic(
            "raw", CHANNELS[0],
            np.zeros((N_Z, FRAME, FRAME), np.uint16), bbox_um=(0.0, 0.0, 16.0, 16.0))
        viewer.dims.current_step = (2,) + tuple(viewer.dims.current_step)[1:]
        assert win._z_slider_index() == 2, "the scene has no z slider to test against"

        win._preview_view_operator()
        assert calls, "the Preview chip did not reach run_operator"
        _key, kw = calls[-1]
        assert kw.get("z_level") == 2, (
            f"the run was launched with z_level={kw.get('z_level')!r}; the view is on z=2")

        # the result lands: dims must not move off the user's plane
        acc = RegionResultAccumulator("decon", REGION, meta, list(CHANNELS),
                                      region_operator=False)
        acc.add(0, _image_5d()[0, :, 2])
        win.deliver_result("decon", acc.result(), visible=True)
        assert int(viewer.dims.current_step[0]) == 2, (
            f"the landing result moved the dims to z={viewer.dims.current_step[0]}")
    finally:
        win.dispose()
