"""A preview solves the z THE VIEW IS ON, and the result lands on that same z."""

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

        acc = RegionResultAccumulator("decon", REGION, meta, list(CHANNELS),
                                      region_operator=False)
        acc.add(0, _image_5d()[0, :, 2])
        win.deliver_result("decon", acc.result(), visible=True)
        assert int(viewer.dims.current_step[0]) == 2, (
            f"the landing result moved the dims to z={viewer.dims.current_step[0]}")
    finally:
        win.dispose()


# --- every preview is the FULL solve; the tab picks what is DISPLAYED (Julio, 2026-08-26) -----
# "Make the 2D preview show plane of the 3D solve. That's what a researcher should expect."

def _coupled_operator():
    """A core depth-keeping z-consumer whose planes COUPLE (decon's shape): each output plane
    is its input plus the stack's mean, so a 1-plane solve differs from plane z of the full
    solve, the way an in-plane decon differs from the volume solve."""
    from squidxplorer import add_operator

    def _coupled(planes):
        stack = np.asarray([np.asarray(p) for p in planes]).astype(np.uint32)
        return (stack + stack.sum(axis=0, keepdims=True) // len(stack)).astype(np.uint16)

    _coupled.keeps_depth = True
    add_operator("coupled_probe", _coupled, consumes={"z"})
    return "coupled_probe"


def _coupled_reader():
    from tests.conftest import FakeReader

    def _plane(region, fov, ch, z, t):
        return np.full((FRAME, FRAME), _plane_value(z, CHANNELS.index(ch)), np.uint16)

    return FakeReader(_meta(), _plane)


def _worker_from_launch(op, reader, kw, **extra) -> _OperatorWorker:
    """The worker the plate would build from a view's launch kwargs: every kwarg the
    constructor still accepts rides through, so a stale per-plane knob can only reach it
    while the constructor still declares one."""
    import inspect

    accepted = inspect.signature(_OperatorWorker.__init__).parameters
    forwarded = {k: v for k, v in kw.items()
                 if k in accepted and k not in ("regions", "save", "operator_kwargs")}
    return _OperatorWorker(op, reader, _meta(),
                           fov_index={REGION: {"rc": (0, 0), "well_id": "A1"}},
                           out_dir="", regions=kw.get("regions") or [REGION], save=False,
                           n_fovs=None, operator_kwargs=kw.get("operator_kwargs"),
                           **forwarded, **extra)


def test_a_2d_tab_preview_lands_plane_z_of_the_full_depth_solve(qapp, napari_pane_stub):
    """Pin 1 + 2: the plane a 2D tab shows is BIT-IDENTICAL to plane z of the full solve of
    the same FOV (reference through project_well at full depth), and the run read EVERY
    plane. Red before the per-plane arm went: the 2D preview solved a 1-plane stack."""
    from squidxplorer import bind_operator, operator_consumes
    from squidxplorer._region_viewer import RegionViewer
    from squidxplorer.projection import project_well

    op = _coupled_operator()
    meta = _meta()
    calls: list = []
    win = RegionViewer(None, meta, [REGION], window_id=92,
                       operator_specs=[(op, "Coupled probe")],
                       run_operator=lambda key, **kw: calls.append((key, kw)))
    try:
        viewer = win._napari_viewer()
        win._pane.mosaic.add_mosaic("raw", CHANNELS[0],
                                    np.zeros((N_Z, FRAME, FRAME), np.uint16),
                                    bbox_um=(0.0, 0.0, 16.0, 16.0))
        viewer.dims.current_step = (2,) + tuple(viewer.dims.current_step)[1:]
        assert win._z_slider_index() == 2 and win._render_mode != "3d"
        win._preview_view_operator()
        assert calls, "the Preview chip did not reach run_operator"
        _key, kw = calls[-1]
    finally:
        win.dispose()

    reader = _coupled_reader()
    worker = _worker_from_launch(op, reader, kw)
    got: list = []
    worker.resultReady.connect(lambda region, fov, planes: got.append(np.asarray(planes)))
    worker.run()
    assert got, "the preview delivered nothing"
    assert sorted({z for (_r, _f, _c, z, _t) in reader.reads}) == list(range(N_Z)), (
        f"a 2D tab's preview must read EVERY plane; it read z "
        f"{sorted({k[3] for k in reader.reads})}")

    reference = project_well(_coupled_reader(), REGION, 0, reduce=bind_operator(op, None),
                             consumes=operator_consumes(op))
    assert reference.shape[2] == N_Z
    landed = got[-1]
    assert landed.shape == (N_C, FRAME, FRAME), landed.shape
    assert np.array_equal(landed, reference[0, :, 2]), (
        "the 2D tab's plane is not plane 2 of the full-depth solve: "
        f"got {landed[0].flat[0]}, full solve {reference[0, 0, 2].flat[0]}, "
        f"in-plane solve would be {2 * _plane_value(2, 0)}")


def test_an_unscoped_depth_keeping_preview_says_draw_an_roi_exactly_once(caplog):
    """An UNSCOPED decon preview (whole fields) logs ONE INFO line at launch that names the
    cost and points at the ROI; an ROI preview keeps its own "ROI ...:" line and gets no
    hint; a reducer's preview says nothing."""
    import logging

    op = _coupled_operator()
    hint = "draw an ROI for a faster preview"

    def _lines(worker) -> list:
        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            worker.run()
        return [r for r in caplog.records if hint in r.getMessage()]

    whole = _lines(_worker_from_launch(op, _coupled_reader(), {"z_level": 1}))
    assert len(whole) == 1, [r.getMessage() for r in whole]
    assert whole[0].levelno == logging.INFO
    assert f"{op} preview over 1 whole field(s): {N_Z} planes x {N_C} channel(s)" in \
        whole[0].getMessage(), whole[0].getMessage()

    roi = _worker_from_launch(op, _coupled_reader(), {"z_level": 1},
                              windows={(REGION, 0): (2, 10, 3, 12)})
    with caplog.at_level(logging.INFO):
        caplog.clear()
        roi.run()
    said = [r.getMessage() for r in caplog.records]
    assert not [s for s in said if hint in s], said
    assert [s for s in said if s.startswith(f"ROI {op}:")], said

    assert _lines(_worker_from_launch("mip", _coupled_reader(), {"z_level": 1})) == []


def test_the_per_plane_preview_arm_is_gone_whole():
    """Absence pins: no surface carries a preview-only plane restriction any more; a
    depth-keeping z-consumer is refused a single plane like any z-consumer."""
    import inspect

    from squidxplorer import bind_operator, run_plate
    from squidxplorer._dispatch import run_operator_once
    from squidxplorer._region_viewer import RegionViewer
    from squidxplorer._runner import InProcessRunner, Runner
    from squidxplorer._viewer import PlateWindow
    from squidxplorer.projection import project_well

    for fn in (_OperatorWorker.__init__, run_operator_once, PlateWindow.run_operator):
        assert "preview_z_level" not in inspect.signature(fn).parameters, fn
    for fn in (run_plate, Runner.run_preview, InProcessRunner.run_preview):
        assert "z_level" not in inspect.signature(fn).parameters, fn
    assert not hasattr(_OperatorWorker, "_preview_z")
    src = inspect.getsource(RegionViewer)
    assert "preview_z" not in src and "just the z in view" not in src

    op = _coupled_operator()
    with pytest.raises(ValueError, match="plane-op"):
        project_well(_coupled_reader(), REGION, 0, reduce=bind_operator(op, None),
                     consumes={"z"}, z_level=1)
