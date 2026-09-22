"""Bulk PNG export of the plate SELECTION under the focused view's latched look
(Nick, ValidTX: cross-comparable representative images, well to well, for a slide deck)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded — Qt binding conflict", allow_module_level=True)

from squidxplorer import _viewer as V  # noqa: E402

from .conftest import napari_pane_stub, shutdown_plate_window  # noqa: E402,F401
from .test_video import _make_5d  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)


@pytest.fixture
def plate_and_view(qapp, napari_pane_stub, tmp_path):
    """A real PlateWindow over 2 wells x 2 FOVs, with one view open (the look source)."""
    root = tmp_path / "acq5d"
    _make_5d().build(root, ["A1", "A2"], n_fovs=2, nz=2, nt=1, size=64)
    win = V.PlateWindow(None)
    win.ingest(str(root))
    view = win._viewer_manager.open(list(win._order))
    assert view is not None
    assert _drain_until(qapp, lambda: len(view._pane._viewer.layers) >= 2, timeout=20)
    yield win, view
    shutdown_plate_window(qapp, win)


@pytest.fixture
def dir_dialog(monkeypatch, tmp_path):
    """``getExistingDirectory`` answering with a folder, so nothing modal blocks the run."""
    from qtpy.QtWidgets import QFileDialog

    out = tmp_path / "pngs"
    out.mkdir()
    chosen = {"dir": str(out), "calls": 0}

    def _answer(*_a, **_k):
        chosen["calls"] += 1
        return chosen["dir"]

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(_answer))
    return chosen


def test_every_selected_well_lands_as_one_png_under_the_views_own_look(
        qapp, plate_and_view, dir_dialog):
    win, view = plate_and_view
    # Latch a deliberate look in the view: the export must carry it to EVERY well.
    mosaic = view._pane.mosaic
    ch0 = view._visible_channel_looks()[0][0]
    mosaic.set_contrast(ch0, 10.0, 900.0)

    win._selected_regions = ["A1", "A2"]
    win._export_selected_wells_pngs()
    worker = getattr(win, "_bulk_png", None)
    assert worker is not None, f"no worker launched; readout: {win._readout.text()!r}"
    assert _drain_until(qapp, lambda: not worker.isRunning())

    out = Path(dir_dialog["dir"])
    acq = win._acq_name
    files = {p.name for p in out.glob("*.png")}
    assert files == {f"{acq}_A1_raw.png", f"{acq}_A2_raw.png"}
    assert "2 of 2 file(s) written" in win._readout.text()

    # The files carry the view's latched look: each equals the one compositor's own answer
    # for that well under the SAME (clim, rgb) list, so any two wells compare like for like.
    from PIL import Image

    from squidxplorer._mosaic_source import fuse_region_mosaic
    from squidxplorer._png import PngChannel, render_view_png

    looks = [(n, c, r) for n, _l, c, r in view._visible_channel_looks()]
    for region in ("A1", "A2"):
        channels = [PngChannel(n, fuse_region_mosaic(win._reader, win._meta, region, n,
                                                     z_level=view._z_slider_index())[0],
                               c, r, z_index=None)
                    for n, c, r in looks]
        expected, _ = render_view_png(channels)
        got = np.asarray(Image.open(out / f"{acq}_{region}_raw.png"))
        np.testing.assert_array_equal(got, expected)


def test_a_strict_fov_subset_exports_exactly_those_fields(qapp, plate_and_view, dir_dialog,
                                                          monkeypatch):
    win, view = plate_and_view
    monkeypatch.setattr(win._overview, "fov_subsets", lambda: {"A1": [1]})
    win._selected_regions = ["A1", "A2"]
    win._export_selected_wells_pngs()
    worker = getattr(win, "_bulk_png", None)
    assert worker is not None
    assert _drain_until(qapp, lambda: not worker.isRunning())
    acq = win._acq_name
    files = {p.name for p in Path(dir_dialog["dir"]).glob("*.png")}
    assert files == {f"{acq}_A1_fov1_raw.png", f"{acq}_A2_raw.png"}


def test_the_refusals_are_named_and_nothing_modal_opens(qapp, plate_and_view, dir_dialog):
    win, _view = plate_and_view
    win._selected_regions = []
    win._export_selected_wells_pngs()
    assert "select wells on the plate first" in win._readout.text()
    assert dir_dialog["calls"] == 0

    win._selected_regions = ["A1"]
    win._viewer_manager._focused_id = None              # no focused view to take the look from
    win._export_selected_wells_pngs()
    assert "open a view first" in win._readout.text()
    assert dir_dialog["calls"] == 0
