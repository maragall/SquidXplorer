"""The navigation wiring inside PlateWindow: one cursor, three views, and the z slider.

Runs against the ndviewer_light stub, since napari cannot build a GL context under
``QT_QPA_PLATFORM=offscreen``, which is what this suite runs under.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidxplorer import _viewer as V  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)


def _shape_dims(viewer, nsteps):
    """Shape the REAL ``ViewerModel.dims`` to *nsteps* — the pane's model is real now, and its
    ``dims`` attribute is frozen, so the shape is CONFIGURED rather than replaced."""
    viewer.dims.ndim = len(nsteps)
    viewer.dims.range = tuple((0, max(0, n - 1), 1) for n in nsteps)
    viewer.dims.current_step = tuple(0 for _ in nsteps)
    return viewer.dims


def _open_window(win, regions):
    w = win._viewer_manager.open(list(regions))
    assert w is not None, "no window was opened"
    return w


def test_a_spawned_window_builds_a_region_slider_bound_to_its_own_cursor(
        qapp, napari_pane_stub, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    # ABSENCE of the attribute, not `is None` — a None value once passed this check for weeks
    # while the feature behind it was dead.
    assert not hasattr(win, "_region_slider"), (
        "the root plate grew a region slider again; navigation is per WINDOW now")
    assert not hasattr(win, "_make_region_slider"), (
        "the plate grew a region-slider BUILDER again; nothing on the plate calls one")
    assert not hasattr(win, "_region_slider_failure"), (
        "a failure string for a slider the plate does not build is a report nobody reads")

    w = _open_window(win, win._order)
    assert w._slider is not None, "the window built no region slider"
    assert w._cursor is not None
    assert w._slider.count == len(win._order), "the slider is not the length of what it navigates"
    assert w._slider.index == w._cursor.index == 0, "the slider and the cursor start disagreeing"
    assert w._cursor.regions == list(win._order)
    shutdown_plate_window(qapp, win)


def test_moving_a_windows_region_slider_reloads_that_windows_mosaic(
        qapp, napari_pane_stub, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    regions = list(win._order)
    assert len(regions) > 1, "fixture needs >1 region or this asserts nothing"

    w = _open_window(win, regions)
    pane = napari_pane_stub[-1]

    # Spy on the LOAD, not the cursor — asserting the cursor alone would stay green even if the
    # mosaic never reloaded. `_load_mosaic` is debounced by a QTimer, so drain for it.
    loads = []
    real = w._load_mosaic
    w._load_mosaic = lambda region=None: (loads.append(region), real(region))[1]

    w._slider.set_index_from_user(1)

    assert w._cursor.region == regions[1]
    assert w._slider.index == 1
    assert _drain_until(qapp, lambda: bool(loads), timeout=10), (
        "the window's mosaic was never reloaded for the region the slider moved to")
    assert loads == [regions[1]], f"the wrong region was loaded: {loads}"
    assert _drain_until(qapp, lambda: bool(len(pane._viewer.layers)), timeout=30), (
        "no layer reached the window's viewer")
    shutdown_plate_window(qapp, win)


def test_double_clicking_the_plate_opens_a_window_on_that_region_and_moves_the_red_frame(
        qapp, napari_pane_stub, squid_dataset):
    """The other direction: plate and navigation must move together both ways, now via an
    independent window rather than the root's own slider."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    regions = list(win._order)

    win.activate_well(regions[1], 0)

    windows = win._viewer_manager.windows
    assert len(windows) == 1, f"a double-click opened {len(windows)} windows"
    w = windows[0]
    assert w._regions == [regions[1]], "the window opened on the wrong region"
    assert w._cursor.region == regions[1], "the window's own cursor is not on that region"
    assert win._overview._sel == tuple(win._fov_index[regions[1]]["rc"]), (
        "the red frame did not follow the double-click")
    # `_current_well` must be the SAME value, not a field kept in step by hand.
    assert win._current_well == regions[1], "the opened region was not recorded"
    assert win._current_well == win._cursor.region, (
        "_current_well and the cursor are two copies of one fact again")
    shutdown_plate_window(qapp, win)


def test_there_is_no_second_copy_of_the_current_region(qapp, squid_dataset):
    """``_mosaic_region`` must be a view, not a field — a field can be assigned behind the
    cursor's back and drift, which is exactly how the red frame and FOV slider once disagreed."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    with pytest.raises(AttributeError):
        win._mosaic_region = "not-a-region"
    assert win._mosaic_region == win._cursor.region
    win.close()


def test_opening_a_plate_shows_a_region_without_claiming_the_user_opened_it(
        qapp, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._mosaic_region is not None, "nothing was put on screen on open"
    assert win._current_well is None, (
        "merely opening a plate counted as the user selecting a region; that would silently "
        "scope every operator run to region 0")
    win.close()


def test_focus_moves_the_windows_own_z_slider(qapp, napari_pane_stub, squid_dataset):
    """The answer must land on THIS window's napari dims, z being the leading axis of (z, y, x)."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = _open_window(win, ["B3"])
    _shape_dims(w._napari_viewer(), (10, 512, 512))

    w._on_reference_plane(4, "")

    assert w._napari_viewer().dims.current_step[0] == 4, "the window's z slider was not moved"
    assert any("reference plane: z=4" in s for s in w._pane.said), w._pane.said
    shutdown_plate_window(qapp, win)


def test_focus_reports_when_no_z_slider_could_be_moved(qapp, napari_pane_stub,
                                                       squid_dataset):
    """A 2D layer has no leading z axis; moving step[0] would silently drive Y instead and
    still announce success."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = _open_window(win, ["B3"])
    _shape_dims(w._napari_viewer(), (512, 512))     # (y, x): no z axis to move

    w._on_reference_plane(4, "")

    said = " ".join(w._pane.said)
    assert "no z slider could be moved" in said, w._pane.said
    assert w._napari_viewer().dims.current_step == (0, 0), "y was driven instead of z"
    shutdown_plate_window(qapp, win)


def test_a_single_plane_stack_is_also_no_z_slider(qapp, napari_pane_stub,
                                                  squid_dataset):
    """A single-step leading axis is a z axis that can't move — same refusal as the 2D case."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = _open_window(win, ["B3"])
    _shape_dims(w._napari_viewer(), (1, 512, 512))

    w._on_reference_plane(4, "")

    assert "no z slider could be moved" in " ".join(w._pane.said), w._pane.said
    shutdown_plate_window(qapp, win)


def test_the_answer_is_clamped_to_the_stack_this_window_is_showing(
        qapp, napari_pane_stub, squid_dataset):
    """A z index past the stack's end is a crash or a no-op depending on napari version; clamp
    instead."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = _open_window(win, ["B3"])
    _shape_dims(w._napari_viewer(), (5, 512, 512))

    w._on_reference_plane(99, "")

    assert w._napari_viewer().dims.current_step[0] == 4
    shutdown_plate_window(qapp, win)


def test_focus_never_reports_a_plane_when_nothing_could_be_read(qapp,
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


def test_the_status_line_does_not_call_a_loaded_plate_live(qapp, squid_dataset):
    """This is POST-ACQUISITION review. "live" reads as a running scope."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert "live" not in win._readout.text().lower(), win._readout.text()
    win.close()
