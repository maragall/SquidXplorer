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

pytest.importorskip("qtpy.QtWidgets")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt6 GUI tests.",
        allow_module_level=True,
    )

from squidmip import _viewer as V  # noqa: E402

from .test_viewer import qapp, stub_detail  # noqa: E402,F401  (fixtures)


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
def test_activate_well_navigates_to_a_freeform_region_id_verbatim(
        qapp, stub_detail, tmp_path, region):
    """(a) A region id that is not <letters><digits> must still navigate the detail viewer.

    The id is passed through UNCHANGED. Rebuilding it from parse_well_id raised on every one
    of these, and the raise was swallowed, so the double-click silently did nothing.
    """
    root = _slide_acquisition(tmp_path / "slide_acq", region)
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._detail.nav.clear()          # ignore whatever the open-on-ingest navigated to

    win.activate_well(region, 0)

    assert win._detail.nav, f"{region!r}: double-click did not navigate at all"
    assert win._detail.nav[-1] == (region, 0), (
        f"{region!r} was not passed through verbatim: got {win._detail.nav[-1]!r}")
    win.close()


def test_focus_reference_plane_ranks_the_fov_in_view_not_the_regions_first(
        qapp, stub_detail, squid_dataset):
    """(b) Autofocus must rank the FOV ON SCREEN, not fovs_per_region[well][0]."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._meta["fovs_per_region"]["B3"][:2] == [0, 1], "fixture needs 2 FOVs to tell these apart"

    # The real detail viewer can move its z-slider; the recording stub deliberately cannot, and
    # _focus_reference_plane bails out early without it. Give it the one method it checks for.
    z_moves = []
    win._detail.set_current_index = lambda axis, i: z_moves.append((axis, i))

    win.activate_well("B3", 1)       # the user is looking at FOV 1
    assert win._current_fov == 1, "the FOV on screen was not recorded"

    win._focus_reference_plane()

    # The readout names the FOV that was actually ranked. Asserting it rather than spying on
    # reader.read is deliberate: a background worker does its own reads on FOV 0 from another
    # thread, so a read-spy pins the scheduler, not the autofocus.
    assert z_moves, "autofocus never moved the z-slider"
    assert win._readout.text().startswith("B3:1 "), (
        f"autofocus reported {win._readout.text()!r}; it ranked the wrong FOV")
    win.close()
