"""The navigation wiring inside PlateWindow: one cursor, three views, and the z slider.

``test_region_nav.py`` pins the cursor and the slider widget on their own. This pins that the
WINDOW actually joins them up — which is the half that has failed here before. Under napari
``_detail`` is None, and every one of these paths used to be guarded on ``_detail``, so they
silently did nothing in the configuration the user actually runs.

These run against the ndviewer_light stub because napari cannot build a GL context under
``QT_QPA_PLATFORM=offscreen``, which is what the whole suite runs under. The napari-specific
seams (``_napari_dims``, ``_napari_z_axis``, ``_set_z_index``, ``_restore_dims_step``) are
therefore driven through a FAKE dims model that has napari's shape. That is a real limitation
and it is stated rather than hidden: the napari path was additionally verified on a real GL
window, and the numbers and screenshots from that run are in the branch report.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("PyQt5")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidmip import _viewer as V  # noqa: E402

from .test_viewer import _drain_until, qapp, stub_detail  # noqa: E402,F401  (fixtures)


class _FakeDims:
    """napari's ``Dims`` shape, only the parts the window drives."""

    def __init__(self, nsteps):
        self.nsteps = tuple(nsteps)
        self.ndim = len(self.nsteps)
        self.current_step = tuple(0 for _ in self.nsteps)

    def set_current_step(self, axis, step):
        cur = list(self.current_step)
        cur[int(axis)] = int(step)
        self.current_step = tuple(cur)


# --------------------------------------------------------------------------------------
# One owner: the slider, the red frame and pane 2 cannot disagree
# --------------------------------------------------------------------------------------

def test_the_window_builds_a_region_slider_bound_to_its_cursor(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._region_slider is not None, (
        f"no region slider was built: {win._region_slider_failure}")
    assert win._region_slider.count == len(win._order), (
        "the slider is not the length of the plate")
    assert win._region_slider.index == win._cursor.index
    win.close()


def test_moving_the_region_slider_moves_the_red_frame(qapp, stub_detail, squid_dataset):
    """Requirement 2, end to end. The frame and the slider must never disagree."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    regions = win._cursor.regions
    assert len(regions) > 1, "fixture needs >1 region or this asserts nothing"

    # Spy on the LOAD, not on `_mosaic_region`. `_mosaic_region` is a property over the cursor,
    # so asserting it here would only prove the cursor moved — it would stay green with pane 2
    # never reloaded at all. The mutation run caught exactly that.
    loads = []
    real = win._load_mosaic
    win._load_mosaic = lambda region=None, op="raw": (loads.append(region), real(region, op))[1]

    first_frame = win._overview._sel
    win._region_slider.set_index_from_user(1)          # the user drags the slider

    assert win._cursor.region == regions[1]
    assert win._overview._sel != first_frame, "the red frame did not move with the slider"
    assert win._overview._sel == tuple(win._fov_index[regions[1]]["rc"]), (
        "the red frame is on a different region from the slider")
    assert loads == [regions[1]], (
        f"pane 2 was not reloaded for the region the slider moved to: {loads}")
    win.close()


def test_double_clicking_the_plate_moves_the_region_slider(qapp, stub_detail, squid_dataset):
    """The other direction. Two controls over one value must move together BOTH ways."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    regions = win._cursor.regions

    win.activate_well(regions[1], 0)

    assert win._region_slider.index == 1, "the slider did not follow the double-click"
    assert win._overview._sel == tuple(win._fov_index[regions[1]]["rc"])
    # `_current_well` must be the SAME value, not a field kept in step by hand.
    assert win._current_well == regions[1], "the opened region was not recorded"
    assert win._current_well == win._cursor.region, (
        "_current_well and the cursor are two copies of one fact again")
    win.close()


def test_there_is_no_second_copy_of_the_current_region(qapp, stub_detail, squid_dataset):
    """``_mosaic_region`` and ``_current_well`` must be VIEWS, not fields.

    A field can be assigned behind the cursor's back and then drift; that is precisely how the
    red frame and the FOV slider disagreed before. Assigning ``_mosaic_region`` must fail.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    with pytest.raises(AttributeError):
        win._mosaic_region = "not-a-region"
    assert win._mosaic_region == win._cursor.region
    win.close()


def test_opening_a_plate_shows_a_region_without_claiming_the_user_opened_it(
        qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._mosaic_region is not None, "nothing was put on screen on open"
    assert win._current_well is None, (
        "merely opening a plate counted as the user selecting a region; that would silently "
        "scope every operator run to region 0")
    win.close()


# --------------------------------------------------------------------------------------
# The z slider is GLOBAL: it is napari's, and it survives a region change
# --------------------------------------------------------------------------------------

def test_z_axis_is_derived_from_the_dims_rank_not_hard_coded(qapp, stub_detail, squid_dataset,
                                                             monkeypatch):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    monkeypatch.setattr(win, "_napari_dims", lambda: _FakeDims((10, 512, 512)))
    assert win._napari_z_axis() == 0, "(z, y, x) puts z at axis 0"
    monkeypatch.setattr(win, "_napari_dims", lambda: _FakeDims((3, 10, 512, 512)))
    assert win._napari_z_axis() == 1, "a leading t axis must shift z, not be ignored"
    monkeypatch.setattr(win, "_napari_dims", lambda: _FakeDims((512, 512)))
    assert win._napari_z_axis() is None, "a plain plane has no z axis to drive"
    win.close()


def test_the_z_position_survives_a_region_change(qapp, stub_detail, squid_dataset, monkeypatch):
    """Requirement 4's architecture note: the z slider is GLOBAL across the plate.

    Replacing the layers resets napari's dims to 0, so without this the z you were inspecting
    snapped back to the bottom of the stack every time you stepped to another region.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    dims = _FakeDims((10, 512, 512))
    monkeypatch.setattr(win, "_napari_dims", lambda: dims)

    dims.set_current_step(0, 6)
    win._pending_dims_step = win._napari_dims_step()   # what _load_mosaic records
    dims.set_current_step(0, 0)                        # what replacing the layers does
    win._restore_dims_step()                           # what _on_mosaic_done does

    assert dims.current_step[0] == 6, "z was not restored after the region change"
    win.close()


def test_restoring_z_refuses_a_step_the_new_region_does_not_have(qapp, stub_detail,
                                                                squid_dataset, monkeypatch):
    """A region with fewer planes must not be driven to a z that does not exist."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    monkeypatch.setattr(win, "_napari_dims", lambda: _FakeDims((10, 512, 512)))
    win._pending_dims_step = (9, 0, 0)
    shallow = _FakeDims((3, 512, 512))
    monkeypatch.setattr(win, "_napari_dims", lambda: shallow)

    win._restore_dims_step()

    assert shallow.current_step[0] == 0, "z was driven past the end of the new region's stack"
    win.close()


# --------------------------------------------------------------------------------------
# Focus reference plane: a BUTTON that moves THE z slider, and says so when it cannot
# --------------------------------------------------------------------------------------

def test_focus_moves_napari_s_z_slider_when_napari_is_the_viewer(qapp, stub_detail,
                                                                 squid_dataset, monkeypatch):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    dims = _FakeDims((10, 512, 512))
    monkeypatch.setattr(win, "_napari_dims", lambda: dims)

    assert win._set_z_index(4) is True
    assert dims.current_step[0] == 4, "napari's own z slider was not moved"
    win.close()


def test_focus_reports_when_no_z_slider_could_be_moved(qapp, stub_detail, squid_dataset,
                                                       monkeypatch):
    """A 'focused' message printed over a slider that never moved is the silent failure."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    monkeypatch.setattr(win, "_napari_dims", lambda: _FakeDims((512, 512)))
    win._detail = None                                 # napari mode: no ndv fallback either

    assert win._set_z_index(4) is False

    win._on_reference_plane("B3", 0, 4)
    assert "no z slider could be moved" in win._readout.text(), win._readout.text()
    win.close()


def test_focus_never_reports_a_plane_when_nothing_could_be_read(qapp, stub_detail,
                                                                squid_dataset):
    """Returning z=0 by default would report a 'sharpest plane' for pixels never examined."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    class _Unreadable:
        def read(self, *a, **k):
            raise OSError("disk gone")

    w = V._FocusWorker(_Unreadable(), win._meta, "B3", 0,
                       win._meta["channels"][0]["name"])
    got = []
    w.problem.connect(got.append)
    w.ready.connect(lambda *a: got.append(("READY", a)))
    w.run()                                            # run() directly: no thread, no race

    assert got and isinstance(got[0], str) and "not one z plane" in got[0], got
    assert not any(isinstance(g, tuple) for g in got), (
        "a sharpest plane was reported although nothing was read")
    win.close()


# --------------------------------------------------------------------------------------
# Copy
# --------------------------------------------------------------------------------------

def test_the_status_line_does_not_call_a_loaded_plate_live(qapp, stub_detail, squid_dataset):
    """This is POST-ACQUISITION review. "live" reads as a running scope."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert "live" not in win._readout.text().lower(), win._readout.text()
    win.close()
