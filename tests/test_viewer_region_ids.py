"""Freeform region ids must navigate verbatim, and window autofocus must rank a
representative FOV."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from qtpy.QtCore import QObject, Signal  # noqa: E402

from squidxplorer import _viewer as V  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import qapp  # noqa: E402,F401  (fixtures)

def _spy_focus_worker(monkeypatch, answer_z=1, note=""):
    """Replace `_FocusWorker` with a recorder; returns the list of ``(region, fov, channel)``."""
    calls = []

    class _SpyFocusWorker(QObject):
        ready = Signal(int, str)
        problem = Signal(str)

        def __init__(self, reader, meta, region, fov, channel, parent=None):
            super().__init__(parent)
            calls.append((region, int(fov), channel))

        def isRunning(self):
            return False

        def start(self):
            self.ready.emit(int(answer_z), note)

    monkeypatch.setattr(V, "_FocusWorker", _SpyFocusWorker)
    return calls


def _slide_acquisition(root, region: str):
    """A one-region slide-carrier acquisition whose region id is NOT <letters><digits>."""
    import tifffile

    (root / "0").mkdir(parents=True)
    for z in (0, 1):
        tifffile.imwrite(root / "0" / f"{region}_0_{z}_Fluorescence_638_nm_-_Penta.tiff",
                         np.zeros((4, 4), np.uint16))
    (root / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n- name: Fluorescence 638 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#FF0000'\n      exposure_time_ms: 50.0\n")
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.325\n  magnification: 20.0\n  sensor_pixel_size_um: 3.76\n"
        "sample:\n  wellplate_format: 1536 well plate\nz_stack:\n  nz: 2\n  delta_z_mm: 0.0015\n"
        "time_series:\n  nt: 1\n")
    return root


# "region_A" is excluded: an underscore is Squid's own filename field separator, so such
# ids never reach the viewer at all.
@pytest.mark.parametrize("region", ["R2C3", "tissue-1", "scan 3"])
def test_activate_well_opens_a_window_on_a_freeform_region_id_verbatim(
        qapp, napari_pane_stub, tmp_path, region):
    """The id must pass through unchanged; rebuilding it via parse_well_id raises on these."""
    root = _slide_acquisition(tmp_path / "slide_acq", region)
    win = V.PlateWindow(None)
    win.ingest(str(root))
    for w in list(win._viewer_manager.windows):        # ignore anything opened on ingest
        w.close()

    win.activate_well(region, 0)

    windows = win._viewer_manager.windows
    assert windows, f"{region!r}: double-click opened no view at all"
    assert windows[-1]._regions == [region], (
        f"{region!r} was not passed through verbatim: got {windows[-1]._regions!r}")
    assert windows[-1]._cursor.region == region
    assert win._cursor.region == region, "the plate's own cursor did not follow the double-click"
    shutdown_plate_window(qapp, win)


def test_window_autofocus_ranks_a_representative_fov_not_the_regions_first(
        qapp, napari_pane_stub, squid_dataset, monkeypatch):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._meta["fovs_per_region"]["B3"][:2] == [0, 1], "fixture needs 2 FOVs"

    w = win._viewer_manager.open(["B3"])
    # Three fields, clustered so the centroid sits nearest field 1: first != centre.
    w._meta = w._meta.model_copy(update={
        "fovs_per_region": {**dict(win._meta["fovs_per_region"]), "B3": [0, 1, 2]},
        "fov_positions_um": {("B3", 0): (0.0, 0.0), ("B3", 1): (100.0, 0.0),
                             ("B3", 2): (110.0, 0.0)},
    })
    calls = _spy_focus_worker(monkeypatch, answer_z=1)

    w._focus_reference_plane()

    assert calls, "the window's autofocus never started a scan"
    region, fov, channel = calls[-1]
    assert (region, fov) == ("B3", 1), (
        f"autofocus ranked {region}:{fov}; it ranked the wrong FOV")
    assert channel == w._meta["channels"][0]["name"], "autofocus ranked the wrong channel"
    assert w._napari_viewer().dims.current_step[0] == 1, "the window's z slider never moved"
    assert any("reference plane: z=1" in s for s in w._pane.said), w._pane.said
    shutdown_plate_window(qapp, win)


def test_window_autofocus_works_without_a_double_click(
        qapp, napari_pane_stub, squid_dataset, monkeypatch):
    """The button acts on the region the window is showing, from its own cursor."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    regions = list(win._order)
    w = win._viewer_manager.open(regions)
    assert win._current_well is None, "fixture invalid: something already activated a region"
    assert w._cursor.region == regions[0], "opening a window must put a region on screen"
    calls = _spy_focus_worker(monkeypatch, answer_z=1)

    w._focus_reference_plane()                        # no prior double-click, no prior activation

    assert calls, "the button did nothing without a prior double-click"
    assert calls[-1][0] == regions[0], "it focused a region the window is not showing"
    assert not any("double-click" in s for s in w._pane.said), w._pane.said
    assert not any("show a region in this view first" in s for s in w._pane.said), w._pane.said
    shutdown_plate_window(qapp, win)


def test_focus_reference_plane_on_a_single_plane_acquisition_says_so(
        qapp, napari_pane_stub, squid_dataset, monkeypatch):
    """A refusal must be a sentence, not a worker ranking a stack of one."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = win._viewer_manager.open(["B3"])
    w._meta = w._meta.model_copy(update={"z_levels": [w._meta["z_levels"][0]]})
    calls = _spy_focus_worker(monkeypatch, answer_z=0)

    w._focus_reference_plane()

    assert any("single z plane" in s for s in w._pane.said), w._pane.said
    assert calls == [], "a stack of one plane was still ranked for focus"
    shutdown_plate_window(qapp, win)
