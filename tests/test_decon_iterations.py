"""The iteration stepper (Julio, 2026-09-05; capture redesign 2026-09-09): every run's
captures are MIP planes riding ONE solve, always on; stepping k repaints iteration k's
pixels without a new solve; turbo recolors and restores."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer import _decon, _decon_gpu
from squidxplorer._decon import (
    DEFAULT_OPTICS,
    OpticsParams,
    _land_captures,
    clear_captures,
    deconvolve_stack,
    iteration_captures,
    set_session_ni,
)
from squidxplorer.projection import cast_like


@pytest.fixture(autouse=True)
def _clean_capture_store():
    clear_captures()
    yield
    clear_captures()
    set_session_ni(None)


def test_the_capture_rides_the_one_solve_and_its_final_is_the_run(monkeypatch):
    """snapshot_sink receives every iteration's MIP of the SAME solve: the final capture
    IS the run's own MIP, bit for bit, the returned stack equals a plain run's, and the
    early iterations differ."""
    pytest.importorskip("petakit")
    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")
    rng = np.random.default_rng(0)
    stack = (rng.random((3, 32, 32)) * 3000 + 100).astype(np.uint16)
    optics = OpticsParams(DEFAULT_OPTICS.na, DEFAULT_OPTICS.wavelength_um,
                          DEFAULT_OPTICS.dxy_um, DEFAULT_OPTICS.dz_um, 3)
    got: dict = {}
    with_sink = deconvolve_stack(stack, optics, 3, project=False, snapshot_sink=got.update)
    plain = deconvolve_stack(stack, optics, 3, project=False)
    assert sorted(got) == [1, 2, 3]
    assert all(v.shape == (32, 32) for v in got.values()), "captures are MIP planes now"
    assert np.array_equal(with_sink, plain)
    mip = deconvolve_stack(stack, optics, 3, project=True)
    assert np.array_equal(cast_like(got[3], stack.dtype), mip)
    assert not np.array_equal(got[1], got[3])


def test_a_plain_run_lands_captures_with_no_panel_anywhere(monkeypatch):
    """THE always-on pin: project_well over the per-channel bind, no panel, no toggle,
    leaves {k: MIP plane} in the store for the solved channel."""
    pytest.importorskip("petakit")
    from squidxplorer._decon import clear_optics, decon_op, set_optics
    from squidxplorer.projection import project_well

    monkeypatch.setenv(_decon_gpu.ENV_VAR, "cpu")

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

    set_optics(OpticsParams(DEFAULT_OPTICS.na, DEFAULT_OPTICS.wavelength_um,
                            DEFAULT_OPTICS.dxy_um, DEFAULT_OPTICS.dz_um, 2))
    try:
        project_well(_Reader(), "A1", 0, decon_op(None, 2), time_point=0)
    finally:
        clear_optics()
    caps = iteration_captures()
    assert sorted(caps) == ["488"]
    assert sorted(caps["488"]) == [1, 2]
    assert all(v.shape == (16, 16) for v in caps["488"].values())


class _View:
    def __init__(self, pane):
        self._pane = pane


class _Manager:
    def __init__(self, view):
        self._view = view

    def active_view(self):
        return self._view


class _Host:
    def __init__(self, view):
        self._viewer_manager = _Manager(view)
        self.said: list = []

    def say(self, text):
        self.said.append(str(text))


def test_stepping_k_changes_the_displayed_pixels_without_a_new_solve(monkeypatch):
    """THE stepper pin: with captures held, stepping to k swaps the view's decon layer to
    iteration k's MIP while any solve attempt raises, and turbo recolors then restores.
    The panel carries NO capture checkbox: capture is always on."""
    from squidxplorer._napari_pane import model_pane_class
    from squidxplorer._param_panel import DeconPanel

    pane = model_pane_class()()
    pane.mosaic.add_result("intensity", "decon", "488", np.zeros((32, 40), np.uint16))
    layer = pane.mosaic.find("decon", "488")
    assert layer is not None
    panel = DeconPanel(_Host(_View(pane)))
    try:
        assert not hasattr(panel, "capture_check"), "the opt-in checkbox is deleted"
        rng = np.random.default_rng(1)
        planes = {k: (rng.random((32, 40)) * 1000 + k).astype(np.float32)
                  for k in (1, 2, 3)}
        _land_captures("488", planes)
        assert not panel._stepper_row.isHidden(), "captures exist, the row must appear"
        assert panel._shown_k == 3

        # From here, a re-solve is an assertion failure: stepping is a repaint of held bytes.
        def _no_solve(*_a, **_k):
            raise AssertionError("the stepper triggered a re-solve")

        monkeypatch.setattr(_decon, "_run", _no_solve)
        panel._step(-1)
        expect2 = cast_like(planes[2], np.dtype(np.uint16))
        assert np.array_equal(np.asarray(layer.data), expect2)
        panel._step(-1)
        expect1 = cast_like(planes[1], np.dtype(np.uint16))
        assert np.array_equal(np.asarray(layer.data), expect1)
        assert not np.array_equal(expect1, expect2)
        assert "iteration 1/3" in panel.iter_label.text()

        before = layer.colormap.name
        panel.turbo_check.setChecked(True)
        assert layer.colormap.name == "turbo"
        panel.turbo_check.setChecked(False)
        assert layer.colormap.name == before

        clear_captures()
        assert panel._stepper_row.isHidden()
    finally:
        panel.deleteLater()
        pane.deleteLater()
