"""Two bugs re-landed from the SUPERSEDED IMA-250 branch (c1c9063).

IMA-250's premise - that the "FOV" control should enumerate the FOVs of one region - was
reversed by the user: a region is the unit of navigation and a mosaic is always what loads
(IMA-265). None of that enumeration work is re-landed here. But two genuine defects were
found while writing it, and they are independent of the premise that was withdrawn:

(a) ``activate_well`` rebuilt the id it navigates to as ``f"{row}{col}"`` from
    ``parse_well_id(well_id)``. A region id is only sometimes a well id. On a slide carrier
    it is freeform - "R2C3", "region_A", "tissue-1", "scan 3" - and ``parse_well_id`` raises
    on those. The raise happened INSIDE a bare ``except Exception: pass``, so the double-click
    did not navigate and did not say why: the plate moved its red box onto the region and the
    detail viewer kept showing the previous one. "manual0" survived only by luck (it parses,
    and reassembles to the same string). Passing the id verbatim is the fix - there was never
    a reason to take the id apart, since it is only a label for the detail viewer.

(b) ``_focus_reference_plane`` ranked ``fovs_per_region[well][0]`` - the region's FIRST FOV -
    whatever FOV was actually on screen. It is a per-FOV autofocus, so focusing field 0 while
    the viewer shows field 12 reports a sharpest plane for pixels the user is not looking at.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt5")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from PyQt5.QtCore import QObject, pyqtSignal  # noqa: E402

from squidmip import _viewer as V  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import qapp, stub_detail  # noqa: E402,F401  (fixtures)

# WHERE THESE NOW POINT (decentralization, 2026-07-23).
#
# `PlateWindow._detail` is permanently None: there is no central viewer any more, and
# `_make_detail_viewer` (which the `stub_detail` fixture patches) has zero production call sites, so
# the stub records a viewer nobody builds. Both defects below outlived that change, they just moved:
#
#   (a) the region id is still passed through verbatim, now to `ViewerManager.open`, which spawns an
#       independent window over exactly that id;
#   (b) the autofocus is now `_region_viewer.RegionViewer._focus_reference_plane`, which picks the
#       region's CENTRE FOV (`_napari3d._center_fov`) rather than `fovs_per_region[region][0]`, and
#       moves THAT window's napari z slider.
#
# `stub_detail` is kept on the signatures because `PlateWindow` still builds against the seam and a
# real ndviewer under offscreen GL segfaults; nothing here reads the stub any more.


def _spy_focus_worker(monkeypatch, answer_z=1, note=""):
    """Replace `_FocusWorker` with a recorder. Returns the list of ``(region, fov, channel)``.

    Spying on the WORKER's arguments, not on reader.read, is deliberate: the plate's raw preview
    does its own reads of FOV 0 from another thread, so a read-spy pins the scheduler rather than
    the autofocus. Emitting `ready` from `start()` keeps the whole test on one thread.
    """
    calls = []

    class _SpyFocusWorker(QObject):
        ready = pyqtSignal(int, str)
        problem = pyqtSignal(str)

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


# "region_A" is deliberately NOT here. An underscore is Squid's own field separator in
# "<region>_<fov>_<z>_<channel>.tiff", so a region id containing one is ambiguous to the READER
# and never reaches the viewer at all (it ingests as zero regions). That is a real limitation,
# but it is a filename-grammar defect a long way upstream of this one - not something
# activate_well can fix, and not what this test is pinning.
@pytest.mark.parametrize("region", ["R2C3", "tissue-1", "scan 3"])
def test_activate_well_opens_a_window_on_a_freeform_region_id_verbatim(
        qapp, stub_detail, napari_pane_stub, tmp_path, region):
    """(a) A region id that is not <letters><digits> must still open its view.

    The id is passed through UNCHANGED. Rebuilding it from parse_well_id raised on every one
    of these, and the raise was swallowed, so the double-click silently did nothing. The
    destination moved from the central detail viewer to `ViewerManager.open`; the defect and its
    mutation-check did not, so this is the same test aimed at where the id now goes.

    MUTATION: rebuild the id as f"{row}{col}" via parse_well_id again -> a raise (swallowed) or a
    mangled id on the window -> red.
    """
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
        qapp, stub_detail, napari_pane_stub, squid_dataset, monkeypatch):
    """(b) Autofocus must rank a REPRESENTATIVE field, not `fovs_per_region[region][0]`.

    It is a per-FOV autofocus, so ranking field 0 while the user is looking elsewhere reports the
    sharpest plane of pixels they are not looking at. On the root plate "representative" meant the
    FOV on screen (`_current_fov`); a window shows a whole fused region and has no FOV cursor, so it
    means the field nearest the region's stage centroid (`_napari3d._center_fov`).

    The positions are rigged so the centre field is 1 and the region's first is 0 -- without that,
    "centre" and "first" are the same number on this fixture and nothing is being tested.

    MUTATION: rank `fovs_per_region[region][0]` -> fov 0 -> red.
    """
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
    # The answer must MOVE this window's z slider: a "focused" message over a slider that never
    # moved is the silent failure the reference-plane button exists to avoid.
    assert w._napari_viewer().dims.current_step[0] == 1, "the window's z slider never moved"
    assert any("reference plane: z=1" in s for s in w._pane.said), w._pane.said
    shutdown_plate_window(qapp, win)


def test_window_autofocus_works_without_a_double_click(
        qapp, stub_detail, napari_pane_stub, squid_dataset, monkeypatch):
    """The button must act on the region the window is SHOWING.

    It used to demand ``_current_well``, which is only set by a double-click, and under napari
    ``_detail`` was None so the handler returned before doing anything at all, a visible button
    with zero function, telling the user to double-click a well they had already opened. A window
    takes its region from its own cursor, which is seeded on open, so it never needs one.
    """
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
    assert not any("open a region first" in s for s in w._pane.said), w._pane.said
    shutdown_plate_window(qapp, win)


def test_focus_reference_plane_on_a_single_plane_acquisition_says_so(
        qapp, stub_detail, squid_dataset, monkeypatch):
    """A refusal must be a sentence. Silently doing nothing is the failure being removed."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    # _meta is a FROZEN Acquisition since the reader-boundary validation landed.
    win._meta = win._meta.model_copy(update={"z_levels": [win._meta["z_levels"][0]]})

    win._focus_reference_plane()

    assert "single z plane" in win._readout.text(), win._readout.text()
    win.close()
