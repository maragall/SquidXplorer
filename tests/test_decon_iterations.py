"""The iteration QC as a CHAINED OPERATION (Julio, 2026-09-09): tri-MIP captures ride
ONE solve through the ordinary dispatch, the result lands as ordinary layers in a
layers-only child whose depth axis is the iteration, napari's own slider steps k and the
DISPLAYED slice actually lands, an ordinary preview lands NOTHING, and teardown frees
the store."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer import _decon, _decon_gpu
from squidxplorer._decon import (
    DEFAULT_OPTICS,
    OpticsParams,
    arm_capture,
    clear_captures,
    deconvolve_stack,
    iteration_captures,
    set_session_ni,
    take_captures,
)
from squidxplorer.projection import cast_like


@pytest.fixture(autouse=True)
def _clean_capture_store():
    arm_capture(False)
    clear_captures()
    yield
    arm_capture(False)
    clear_captures()
    set_session_ni(None)


def test_the_tri_mip_capture_rides_the_one_solve(monkeypatch):
    """Each of the three projections at the final k IS the plain run's own max over that
    axis, bit for bit; the returned stack equals a plain run's; early iterations differ."""
    pytest.importorskip("petakit")
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    rng = np.random.default_rng(0)
    stack = (rng.random((3, 32, 40)) * 3000 + 100).astype(np.uint16)
    optics = OpticsParams(DEFAULT_OPTICS.na, DEFAULT_OPTICS.wavelength_um,
                          DEFAULT_OPTICS.dxy_um, DEFAULT_OPTICS.dz_um, 3)
    got: dict = {}
    with_sink = deconvolve_stack(stack, optics, 3, project=False, snapshot_sink=got.update)
    plain = deconvolve_stack(stack, optics, 3, project=False)
    assert sorted(got) == [0, 1, 2, 3], "k=0 is the RAW INPUT's own projections"
    assert sorted(got[3]) == ["xy", "xz", "yz"]
    assert got[3]["xy"].shape == (32, 40)
    assert got[3]["xz"].shape == (3, 40)
    assert got[3]["yz"].shape == (3, 32)
    assert np.array_equal(with_sink, plain)
    for axis, name in ((0, "xy"), (1, "xz"), (2, "yz")):
        assert np.array_equal(got[0][name], stack.max(axis=axis).astype(np.float32)), (
            f"the k=0 {name} capture must be the raw input's own max over axis {axis}")
        assert np.array_equal(cast_like(got[3][name], stack.dtype), plain.max(axis=axis)), (
            f"the final {name} capture must be the run's own max over axis {axis}")
    assert not np.array_equal(got[1]["xy"], got[3]["xy"])


class _Reader:
    metadata = {
        "regions": ["A1"], "fovs_per_region": {"A1": [0]},
        "channels": [{"name": "488"}], "z_levels": [0, 1],
        "n_t": 1, "frame_shape": (16, 16), "dtype": "uint16",
    }

    def read(self, region, fov, channel, z, t=0):
        plane = np.full((16, 16), 400, np.uint16)
        plane[6:10, 6:10] = 3000
        return plane


def _override_optics(nz=2):
    from squidxplorer._decon import set_optics

    set_optics(OpticsParams(DEFAULT_OPTICS.na, DEFAULT_OPTICS.wavelength_um,
                            DEFAULT_OPTICS.dxy_um, DEFAULT_OPTICS.dz_um, nz))


def test_an_ordinary_preview_lands_no_captures(monkeypatch):
    """The capture is the QC solve's alone: the same per-channel bind, run WITHOUT
    arming (every ordinary preview and save), leaves the store empty."""
    pytest.importorskip("petakit")
    from squidxplorer._decon import clear_optics, decon_op
    from squidxplorer.projection import project_well

    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    _override_optics()
    try:
        project_well(_Reader(), "A1", 0, decon_op(None, 2), time_point=0)
    finally:
        clear_optics()
    assert iteration_captures() == {}


def test_the_qc_solve_is_the_ordinary_dispatch_and_take_frees_the_store(monkeypatch):
    """THE fold pin (Julio: "it looks like logic is being duplicated"): run_qc_solve
    rides run_operator_once - the Preview's own dispatch - with capture armed; every
    channel lands all three projections; take_captures empties the store."""
    pytest.importorskip("petakit")
    from squidxplorer import _dispatch
    from squidxplorer._decon import clear_optics
    from squidxplorer._decon_qc import run_qc_solve

    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    rode_dispatch = {"n": 0}
    real = _dispatch.run_operator_once

    def counted(*a, **k):
        rode_dispatch["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_dispatch, "run_operator_once", counted)
    _override_optics()
    try:
        run_qc_solve(_Reader(), "A1", 0, None, {"iterations": 2})
    finally:
        clear_optics()
    assert rode_dispatch["n"] == 1, "the QC solve must ride the one dispatch path"
    caps = take_captures()
    assert sorted(caps) == ["488"]
    assert sorted(caps["488"]) == [0, 1, 2], "raw input rides as k=0"
    assert sorted(caps["488"][2]) == ["xy", "xz", "yz"]
    assert caps["488"][2]["xy"].shape == (16, 16)
    assert caps["488"][2]["xz"].shape == (2, 16)
    assert iteration_captures() == {}, "take_captures must empty the store"


def _tri(rng, k, h=24, w=30, z=4):
    # Intensity GROWS with k on purpose (RL concentrates flux; Julio's measured
    # brightening) so the display-normalization pin has real growth to flatten.
    return {"xy": ((rng.random((h, w)) * 1000 + 50) * k).astype(np.float32),
            "xz": ((rng.random((z, w)) * 1000 + 50) * k).astype(np.float32),
            "yz": ((rng.random((z, h)) * 1000 + 50) * k).astype(np.float32)}


def _drain(app, pred, timeout=10):
    import time

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        if pred():
            return True
    return False


def test_the_chained_qc_result_lands_as_layers_and_its_slices_actually_land(
        qapp, napari_pane_stub, squid_dataset, monkeypatch):
    """THE delivered-chain pin, carrying the measured defect of 2026-09-09: the shipped
    (N, 1, H, W) shape beside a z-ful raw shared world axis 1 and napari sliced the
    size-1 axis at raw's z point, out of range forever, so no QC slice ever landed and
    every slider read dead. Now: a LAYERS-ONLY child (no raw, no mosaic load), uniform
    (iteration, y, x) dims, ordinary delivery - and the pin asserts the DISPLAYED slice
    lands for the stepped k, which the old pins never checked."""
    import squidxplorer._viewer as V
    from squidxplorer._decon_qc import QC_XY, QC_XZ, QC_YZ, deliver_qc, qc_bbox_um
    from tests.conftest import shutdown_plate_window

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    view = win._viewer_manager.open(list(win._order)[:1])
    try:
        assert _drain(qapp, lambda: view._pane is not None)
        region = view.current_region()
        meta = view._meta
        fov = int(meta["fovs_per_region"][region][0])
        fh, fw = (int(v) for v in meta["frame_shape"])    # a real capture spans the field
        nz = 2
        rng = np.random.default_rng(1)
        caps = {"488": {k: _tri(rng, k, h=fh, w=fw, z=nz) for k in (1, 2, 3)}}
        raw_before = {k: {n: v.copy() for n, v in tri.items()}
                      for k, tri in caps["488"].items()}
        child = deliver_qc(view, caps, region, fov, None)
        assert child is not None and child is not view
        assert child._layers_only and child._worker is None, "no raw mosaic load, ever"
        mosaic = child._pane.mosaic
        model = mosaic.model
        assert "raw" not in mosaic.ops(), "the QC tab holds only the delivered layers"

        px = float(meta["pixel_size_um"])
        dz = float(meta.get("dz_um") or px)
        x0, y0, x1, y1 = qc_bbox_um(meta, region, fov, None)
        xy = mosaic.find(QC_XY, "488")
        xz = mosaic.find(QC_XZ, "488")
        yz = mosaic.find(QC_YZ, "488")
        assert xy.data.shape == (3, fh, fw)
        assert xz.data.shape == (3, nz, fw)
        assert yz.data.shape == (3, fh, nz), "yz is transposed so y aligns"
        assert np.allclose(xy.scale, (1.0, px, px)), "the iteration axis is unit scale"
        assert np.allclose(xz.scale, (1.0, dz, px)), "the band's z rides its own scale"
        assert np.allclose(yz.scale, (1.0, px, dz))
        assert np.allclose(xy.translate[-2:], (y0, x0))
        assert xz.translate[-2] > y1, "the XZ band sits below the MIP"
        assert yz.translate[-1] > x1, "the YZ band sits beside the MIP"
        assert all(ly.visible for ly in (xy, xz, yz)), "all three panels open lit"
        for band in (xz, yz):
            assert band.metadata.get("contrast_unlinked"), "a band owns its window"
            assert str(band.interpolation2d) == "linear", "a stretched band renders linear"
        assert band not in mosaic._link_set("488")
        assert model.dims.ndim == 3, "uniform (iteration, y, x): no axis mixing exists"
        assert model.dims.axis_labels[0] == "iteration"
        assert int(model.dims.current_step[0]) == 2, "opens on the final iteration"

        # THE display-normalization pin (Julio: "intesity grows with the iterations"):
        # every delivered iteration's XY p99.5 sits on the shared display target, so
        # stepping holds apparent brightness constant; the input caps stay RAW.
        p995 = [float(np.percentile(np.asarray(xy.data[i]), 99.5)) for i in range(3)]
        target = min(float(np.percentile(raw_before[k]["xy"], 99.5)) for k in (1, 2, 3))
        for i, p in enumerate(p995):
            assert abs(p - target) <= 0.02 * target, (
                f"delivered iteration {i} p99.5 {p:.0f} is off the shared display "
                f"target {target:.0f}; stepping would read as brightening")
        for k in (1, 2, 3):
            for name in ("xy", "xz", "yz"):
                assert np.array_equal(caps["488"][k][name], raw_before[k][name]), (
                    "delivery must not touch the raw captures (normalize at delivery, "
                    "not at capture)")

        def _no_solve(*_a, **_k):
            raise AssertionError("stepping the QC tab triggered a re-solve")

        monkeypatch.setattr(_decon, "_run", _no_solve)
        factor1 = target / float(np.percentile(raw_before[1]["xy"], 99.5))
        model.dims.set_current_step(0, 0)
        assert _drain(qapp, lambda: np.array_equal(
            np.asarray(xy._slice.image.view),
            cast_like(raw_before[1]["xy"] * factor1, np.dtype(np.uint16))), timeout=5), (
            "the DISPLAYED slice must land for the stepped k (the 2026-09-09 defect)")
        assert _drain(qapp, lambda: np.array_equal(
            np.asarray(xz._slice.image.view),
            cast_like(raw_before[1]["xz"] * factor1, np.dtype(np.uint16))), timeout=5)

        pane = child._pane
        child.dispose()
        assert pane.shutdowns >= 1, "the tab's ordinary dispose frees its viewer"
    finally:
        shutdown_plate_window(qapp, win)


