"""The navigation wiring inside PlateWindow: one cursor, three views, and the z slider.

``test_region_nav.py`` pins the cursor and the slider widget on their own. This pins that the
WINDOW actually joins them up — which is the half that has failed here before. Under napari
``_detail`` is None, and every one of these paths used to be guarded on ``_detail``, so they
silently did nothing in the configuration the user actually runs.

These run against the ndviewer_light stub because napari cannot build a GL context under
``QT_QPA_PLATFORM=offscreen``, which is what the whole suite runs under. The napari-specific
seams (``_napari_dims``, ``_napari_z_axis``, ``_restore_dims_step``, and the windows' own
``_on_reference_plane``) are
therefore driven through a FAKE dims model that has napari's shape. That is a real limitation
and it is stated rather than hidden: the napari path was additionally verified on a real GL
window, and the numbers and screenshots from that run are in the branch report.

The region slider itself has since moved OFF the root plate and onto each spawned window
(2026-07-23); the first three tests are re-pointed there and use ``napari_pane_stub``, which
replaces the one seam that needs a GL context so a real ``RegionViewer`` can be built headlessly.
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
# One owner: the slider, the red frame and the window's mosaic cannot disagree
# --------------------------------------------------------------------------------------
#
# THE REGION SLIDER MOVED OFF THE ROOT PLATE (decentralization, 2026-07-23). `_viewer.py`:
# "NO region slider on the root plate. The deck puts the region slider in each spawned WINDOW."
# The real slider is built in `squidxplorer/_region_viewer.py` (`RegionViewer._build`) from the same
# `RegionCursor` / `RegionSlider` pair, one per window.
#
# Until 2026-08-06 the plate still carried `self._region_slider = None`, a `_make_region_slider`
# with no call sites, a tooltip branch in `_on_region_changed` guarding on it, and a
# `_region_slider_failure` string written twice and read never. That is the `_mosaic_pane` shape
# exactly (see CLAUDE.md): a `None` that reads as "the feature is off" when the truth is "there is
# no feature", keeping a producer, a guard and an import alive behind it.
# `test_the_plate_has_no_region_slider_attribute_at_all` below pins the ABSENCE, not `is None`,
# for the same reason `tests/test_plate_follows_windows.py` pins `_mosaic_pane`'s.
#
# The three tests below are those three, re-pointed at the window. They were NOT deleted, because
# every property they pinned still exists: a slider the length of what it navigates, bound to a
# cursor; a user drag that reloads the mosaic (spying the LOAD, never the cursor, which is the
# mutation the original run caught); and the plate and the navigation agreeing in both directions.
# What changed is only which object owns them.


def _open_window(win, regions):
    """Open one RegionViewer over *regions* the way the plate does, and return it."""
    w = win._viewer_manager.open(list(regions))
    assert w is not None, "no window was opened"
    return w


def test_a_spawned_window_builds_a_region_slider_bound_to_its_own_cursor(
        qapp, napari_pane_stub, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    # The ABSENCE of the attribute, not `is None`. This assertion read `is None` until 2026-08-06
    # and passed for two weeks over a `_make_region_slider` nobody called, a guard nobody could
    # reach and an import only the dead method used: `None` is a value, and a value reads as a
    # feature that happens to be off.
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
    """Requirement 2, end to end, on the window that now owns the slider."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    regions = list(win._order)
    assert len(regions) > 1, "fixture needs >1 region or this asserts nothing"

    w = _open_window(win, regions)
    pane = napari_pane_stub[-1]

    # Spy on the LOAD, not on the cursor. The cursor is the thing the slider drives directly, so
    # asserting it would stay green with the mosaic never reloaded at all, the mutation the
    # original run of this test caught. `_load_mosaic` is debounced by a QTimer, so drain for it.
    loads = []
    real = w._load_mosaic
    w._load_mosaic = lambda region=None: (loads.append(region), real(region))[1]

    w._slider.set_index_from_user(1)                   # the user drags the window's slider

    assert w._cursor.region == regions[1]
    assert w._slider.index == 1
    assert _drain_until(qapp, lambda: bool(loads), timeout=10), (
        "the window's mosaic was never reloaded for the region the slider moved to")
    assert loads == [regions[1]], f"the wrong region was loaded: {loads}"
    assert _drain_until(qapp, lambda: bool(pane.mosaic.added), timeout=30), (
        "no layer reached the window's viewer")
    shutdown_plate_window(qapp, win)


def test_double_clicking_the_plate_opens_a_window_on_that_region_and_moves_the_red_frame(
        qapp, napari_pane_stub, squid_dataset):
    """The other direction. The plate and the navigation must move together BOTH ways.

    A double-click used to move the root's own region slider. It now opens (or is answered by) an
    independent window on that region, and the red frame on the plate follows the same cursor,
    so the two still cannot disagree, they are just in two windows.
    """
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
        qapp, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._mosaic_region is not None, "nothing was put on screen on open"
    assert win._current_well is None, (
        "merely opening a plate counted as the user selecting a region; that would silently "
        "scope every operator run to region 0")
    win.close()


# --------------------------------------------------------------------------------------
# THE GLOBAL Z SLIDER WAS THE CENTRAL PANE'S, AND IT WENT WITH IT (2026-08-06)
# --------------------------------------------------------------------------------------
#
# Three tests here pinned ``PlateWindow._napari_z_axis``, ``_napari_dims_step`` and
# ``_restore_dims_step``: the rule that stepping to another region must not snap napari's z back
# to the bottom of the stack. All three read ``self._mosaic_pane``, which has been unconditionally
# ``None`` since 2b8fbc5, and the only writer of ``_pending_dims_step`` was ``_load_mosaic``, which
# returned at its first line. So the plate has not restored a z position since 2026-07-23, and the
# tests passed by monkeypatching ``_napari_dims`` -- the exact seam that made the pane irrelevant.
# Deleted with the methods.
#
# STATED, NOT PAPERED OVER: no window restores z across a REGION change today.
# ``RegionViewer._load_mosaic`` calls ``remove_op(_RAW_OP)`` when the region changes, and it has to
# (a different region is a different mosaic shape, and reusing the layer raises out of napari --
# see tests/test_time_point_playback.py), so napari's dims reset with the layers. What IS preserved
# is a TIMEPOINT change, which keeps the layers and therefore keeps z, and that is pinned there.
# Restoring z across a region change would be a new feature on ``RegionViewer``, not a revival of
# these tests, and it is not part of this deletion.


# --------------------------------------------------------------------------------------
# Focus reference plane: a control that moves THE z slider, and says so when it cannot
# --------------------------------------------------------------------------------------
#
# THE REFERENCE PLANE MOVED OFF THE ROOT PLATE TOO (d07db43, and the orphan removed 2026-07-29).
# `PlateWindow._set_z_index` and `PlateWindow._on_reference_plane` are gone: their only entry point
# was a parentless `QPushButton` that Qt therefore treated as a top-level window, which is the stray
# window Julio kept seeing (tests/test_no_orphan_windows.py has the measurement). The control lives
# on each window's own z-slider now.
#
# The two tests below are the two that pinned those methods, re-pointed at the window. They were
# NOT deleted, because both properties they pinned are still properties of the product and the
# second one is a SILENT failure: a "focused" message printed over a slider that never moved. Both
# had to be built on `RegionViewer`, which had neither guard, so the guards moved with the button
# rather than being dropped along with the dead code.

def test_focus_moves_the_windows_own_z_slider(qapp, napari_pane_stub, squid_dataset):
    """The answer must land on THIS window's napari dims, z being the leading axis of (z, y, x)."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = _open_window(win, ["B3"])
    w._napari_viewer().dims = _FakeDims((10, 512, 512))

    w._on_reference_plane(4, "")

    assert w._napari_viewer().dims.current_step[0] == 4, "the window's z slider was not moved"
    assert any("reference plane: z=4" in s for s in w._pane.said), w._pane.said
    shutdown_plate_window(qapp, win)


def test_focus_reports_when_no_z_slider_could_be_moved(qapp, napari_pane_stub,
                                                       squid_dataset):
    """A 'focused' message printed over a slider that never moved is the silent failure.

    A 2D layer has no leading z axis at all, so the old body's ``step[0] = z_index`` moved Y and
    then announced a reference plane. MUTATION: drop the ndim guard -> the message becomes
    "reference plane: z=4" over a Y axis -> red.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = _open_window(win, ["B3"])
    w._napari_viewer().dims = _FakeDims((512, 512))     # (y, x): no z axis to move

    w._on_reference_plane(4, "")

    said = " ".join(w._pane.said)
    assert "no z slider could be moved" in said, w._pane.said
    assert w._napari_viewer().dims.current_step == (0, 0), "y was driven instead of z"
    shutdown_plate_window(qapp, win)


def test_a_single_plane_stack_is_also_no_z_slider(qapp, napari_pane_stub,
                                                  squid_dataset):
    """One step on the leading axis is a z axis that cannot move. Reporting a plane over it is the
    same lie as reporting one over a 2D layer, so it takes the same refusal."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = _open_window(win, ["B3"])
    w._napari_viewer().dims = _FakeDims((1, 512, 512))

    w._on_reference_plane(4, "")

    assert "no z slider could be moved" in " ".join(w._pane.said), w._pane.said
    shutdown_plate_window(qapp, win)


def test_the_answer_is_clamped_to_the_stack_this_window_is_showing(
        qapp, napari_pane_stub, squid_dataset):
    """A z index past the end of the layer is a crash or a no-op depending on the napari version.
    Clamp it, so the slider lands on the last plane and the window still says what it did."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = _open_window(win, ["B3"])
    w._napari_viewer().dims = _FakeDims((5, 512, 512))

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


# --------------------------------------------------------------------------------------
# Copy
# --------------------------------------------------------------------------------------

def test_the_status_line_does_not_call_a_loaded_plate_live(qapp, squid_dataset):
    """This is POST-ACQUISITION review. "live" reads as a running scope."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert "live" not in win._readout.text().lower(), win._readout.text()
    win.close()
