"""The iteration stepper (Julio, 2026-09-05): captures ride ONE solve; stepping k repaints
iteration k's pixels without a new solve; turbo recolors and restores."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer import _decon, _decon_gpu
from squidxplorer._decon import (
    DEFAULT_OPTICS,
    OpticsParams,
    _land_captures,
    deconvolve_stack,
    set_capture_iterations,
    set_session_ni,
)
from squidxplorer.projection import cast_like


@pytest.fixture(autouse=True)
def _clean_capture_store():
    set_capture_iterations(False)
    yield
    set_capture_iterations(False)
    set_session_ni(None)


def test_the_capture_rides_the_one_solve_and_its_final_is_the_run(monkeypatch):
    """snapshot_sink receives every iteration of the SAME solve: the final capture IS the
    run's own result, bit for bit, and the early iterations differ from it."""
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
    assert np.array_equal(with_sink, plain)
    assert np.array_equal(cast_like(got[3], stack.dtype), plain)
    assert not np.array_equal(got[1], got[3])


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
    iteration k's MIP while any solve attempt raises, and turbo recolors then restores."""
    from squidxplorer._napari_pane import model_pane_class
    from squidxplorer._param_panel import DeconPanel

    pane = model_pane_class()()
    pane.mosaic.add_result("intensity", "decon", "488", np.zeros((4, 32, 40), np.uint16)[0])
    layer = pane.mosaic.find("decon", "488")
    assert layer is not None
    panel = DeconPanel(_Host(_View(pane)))
    try:
        rng = np.random.default_rng(1)
        vols = {k: (rng.random((4, 32, 40)) * 1000 + k).astype(np.float32)
                for k in (1, 2, 3)}
        set_capture_iterations(True)
        _land_captures("488", vols)
        assert not panel._stepper_row.isHidden(), "captures exist, the row must appear"
        assert panel._shown_k == 3

        # From here, a re-solve is an assertion failure: stepping is a repaint of held bytes.
        def _no_solve(*_a, **_k):
            raise AssertionError("the stepper triggered a re-solve")

        monkeypatch.setattr(_decon, "_run", _no_solve)
        panel._step(-1)
        expect2 = cast_like(vols[2].max(axis=0), np.dtype(np.uint16))
        assert np.array_equal(np.asarray(layer.data), expect2)
        panel._step(-1)
        expect1 = cast_like(vols[1].max(axis=0), np.dtype(np.uint16))
        assert np.array_equal(np.asarray(layer.data), expect1)
        assert not np.array_equal(expect1, expect2)
        assert "iteration 1/3" in panel.iter_label.text()

        before = layer.colormap.name
        panel.turbo_check.setChecked(True)
        assert layer.colormap.name == "turbo"
        panel.turbo_check.setChecked(False)
        assert layer.colormap.name == before

        # Disarming FREES the captures and the row goes with them.
        panel.capture_check.setChecked(True)
        panel.capture_check.setChecked(False)
        assert panel._stepper_row.isHidden()
    finally:
        panel.deleteLater()
        pane.deleteLater()
