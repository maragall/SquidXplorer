"""Run-vs-save is ONE FLOW (Julio, 2026-08-25): "They can preview on the window... After
they preview, they can say run on plate, and then it will save to disk. No body runs on
whole plate to preview."

The operators row carries exactly Preview (window scope, writes nothing) and Run on plate
(the one save path, which is also the bulk path now that the right-edge dock's cards are
retired). The save checkbox is gone.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("qapp")

REGION = "A1"


def _meta() -> dict:
    return {
        "regions": [REGION],
        "channels": [{"name": "405"}],
        "z_levels": [0],
        "n_z": 1,
        "n_t": 1,
        "dtype": "uint16",
        "frame_shape": (16, 16),
        "pixel_size_um": 0.75,
        "dz_um": 1.5,
        "fovs_per_region": {REGION: [0]},
        "fov_positions_um": {(REGION, 0): (0.0, 0.0)},
    }


@pytest.fixture
def view(qapp, napari_pane_stub):
    from squidxplorer._region_viewer import RegionViewer

    calls: list = []
    win = RegionViewer(None, _meta(), [REGION], window_id=93,
                       operator_specs=[("mip", "Maximum Intensity Projection")],
                       run_operator=lambda key, **kw: calls.append((key, kw)))
    win.operator_panel()
    yield win, calls
    win.dispose()


def test_preview_is_window_scoped_and_writes_nothing(view):
    win, calls = view
    win._btn_preview.click()
    assert calls, "Preview never reached run_operator"
    _key, kw = calls[-1]
    assert kw["save"] is False, "a preview must write nothing"
    assert list(kw["regions"]) == [REGION], "a preview is scoped to THIS view's regions"
    assert kw["requester"] is win


def test_run_on_plate_is_the_save_and_leaves_scope_to_the_plate(view):
    win, calls = view
    win._btn_run_plate.click()
    assert calls, "Run on plate never reached run_operator"
    _key, kw = calls[-1]
    assert kw["save"] is True, "Run on plate IS the save"
    assert kw["regions"] is None, (
        "the plate resolves the scope (selection, else whole plate); the view must not")


def test_the_save_checkbox_is_gone(view):
    win, _calls = view
    assert not hasattr(win, "_save_chk"), "the save checkbox is back; the flow is two buttons"


def test_every_operator_row_button_fits_its_text(view):
    """Julio: "Watchout, because some buttons are too small to fit text"."""
    win, _calls = view
    for btn in (win._btn_preview, win._btn_run_plate, win._btn_controls):
        needed = btn.fontMetrics().horizontalAdvance(btn.text())
        assert btn.sizeHint().width() >= needed, (
            f"{btn.text()!r}: sizeHint {btn.sizeHint().width()} px cannot fit its "
            f"{needed} px label")
        assert btn.minimumSizeHint().width() <= btn.sizeHint().width() + 1
