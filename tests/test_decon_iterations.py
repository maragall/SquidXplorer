"""The iteration QC (Julio, 2026-09-09: MIPs are the QC; the window steps them): tri-MIP
captures ride ONE solve, the QC window's slider swaps all three projections without a
re-solve, "use k" writes the panel, an ordinary preview lands NOTHING, and teardown frees
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
    assert sorted(got) == [1, 2, 3]
    assert sorted(got[3]) == ["xy", "xz", "yz"]
    assert got[3]["xy"].shape == (32, 40)
    assert got[3]["xz"].shape == (3, 40)
    assert got[3]["yz"].shape == (3, 32)
    assert np.array_equal(with_sink, plain)
    for axis, name in ((0, "xy"), (1, "xz"), (2, "yz")):
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


def test_the_qc_solve_lands_tri_captures_and_take_frees_the_store(monkeypatch):
    """The QC path: run_qc_solve arms around one project_well, every channel captured
    with all three projections; take_captures hands them over and empties the store."""
    pytest.importorskip("petakit")
    from squidxplorer._decon import clear_optics, decon_op
    from squidxplorer._decon_qc import run_qc_solve

    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    _override_optics()
    try:
        run_qc_solve(_Reader(), "A1", 0, None, decon_op(None, 2), 0)
    finally:
        clear_optics()
    caps = take_captures()
    assert sorted(caps) == ["488"]
    assert sorted(caps["488"]) == [1, 2]
    assert sorted(caps["488"][2]) == ["xy", "xz", "yz"]
    assert caps["488"][2]["xy"].shape == (16, 16)
    assert caps["488"][2]["xz"].shape == (2, 16)
    assert iteration_captures() == {}, "take_captures must empty the store"


def _tri(rng, k, h=24, w=30, z=4):
    return {"xy": (rng.random((h, w)) * 1000 + k).astype(np.float32),
            "xz": (rng.random((z, w)) * 1000 + k).astype(np.float32),
            "yz": (rng.random((z, h)) * 1000 + k).astype(np.float32)}


def test_the_qc_windows_slider_swaps_all_three_projections_without_a_new_solve(monkeypatch):
    """THE stepper pin, in the QC window: dragging to k shows iteration k's XY, XZ and YZ
    while any solve attempt raises; use-k hands the DISPLAYED count out; closing frees."""
    from squidxplorer._decon_qc import DeconQCWindow

    rng = np.random.default_rng(1)
    caps = {"488": {k: _tri(rng, k) for k in (1, 2, 3)}}
    adopted: list = []
    window = DeconQCWindow(caps, z_ratio=2.0, subject="A1 field 0",
                           on_use=adopted.append)
    try:
        assert window.shown_k == 3, "the position follows the final iteration"

        def _no_solve(*_a, **_k):
            raise AssertionError("the QC window triggered a re-solve")

        monkeypatch.setattr(_decon, "_run", _no_solve)
        window.slider.set_index_from_user(1)
        assert window.shown_k == 2
        for name in ("xy", "xz", "yz"):
            assert np.array_equal(window.projection(name), caps["488"][2][name])
        window.slider.set_index_from_user(0)
        assert window.shown_k == 1
        for name in ("xy", "xz", "yz"):
            assert np.array_equal(window.projection(name), caps["488"][1][name])
        assert not window.xy_label.pixmap().isNull()
        assert not window.xz_label.pixmap().isNull()
        assert not window.yz_label.pixmap().isNull()

        window.slider.turbo_check.setChecked(True)   # turbo rides the slider's own bar
        assert window._turbo
        window.use_btn.click()
        assert adopted == [1], "use k must hand out the DISPLAYED count"

        window.close()
        assert window._caps == {}, "closing the window frees the held projections"
    finally:
        window.slider.shutdown()
        window.deleteLater()


def test_use_k_writes_the_panels_iterations_spin():
    """The adoption seam: the window's on_use goes through DeconPanel.set_param, the
    run's single source of truth."""
    from squidxplorer._param_panel import DeconPanel

    class _Host:
        def say(self, text):
            self.said = text

    panel = DeconPanel(_Host())
    try:
        assert panel.set_param("iterations", 7) is None
        assert int(panel.widgets["iterations"].value()) == 7
        assert hasattr(panel, "inspect_btn"), "the QC button is the decon UI's entry"
    finally:
        panel.deleteLater()
