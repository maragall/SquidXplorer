"""The plate as a NAVIGATOR: left-click a well to move the active view onto that region."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # headless Qt; must precede any Qt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt GUI tests.",
        allow_module_level=True,
    )

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from squidxplorer import _viewer as V  # noqa: E402
from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import _drain_until, _mouse, _pt, qapp  # noqa: E402,F401  (fixtures + helpers)


CD = 20.0


def _freeze(ov, cd=CD):
    """Deterministic widget pixels — otherwise paintEvent's auto-fit moves the plate under the synthetic coordinates."""
    ov._user_view = True
    ov._cd, ov._ox, ov._oy = cd, 0.0, 0.0


def _click(ov, well_id, cd=CD):
    """A plain left-click on *well_id*'s cell: press then release, no modifier, no movement."""
    rc = next(rc for rc, wid in ov._by_rc.items() if wid == well_id)
    pos = _pt(rc[0], rc[1], cd)
    ov.mousePressEvent(_mouse("press", pos))
    ov.mouseReleaseEvent(_mouse("release", pos, buttons=Qt.NoButton))
    return pos


def _fire_pending_navigation(ov):
    """Run the deferred navigation now instead of waiting out the double-click interval."""
    if ov._nav.isActive():
        ov._nav.stop()
        ov._emit_navigation()


def _plate(qapp, root):
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _freeze(win._overview)
    return win


def _plate_with_view(qapp, root, regions):
    win = _plate(qapp, root)
    view = win._viewer_manager.open(list(regions))
    assert view is not None
    _drain_until(qapp, lambda: view._pane is not None, timeout=10)
    return win, view


# --- the headline -------------------------------------------------------------------------------

def test_a_plain_click_leaves_the_operator_selection_alone(qapp, napari_pane_stub, squid_dataset):
    """RULE ONE. The batch the user built must survive navigating away from it."""
    root, _ = squid_dataset
    win = _plate(qapp, root)
    ov = win._overview
    ov.select_all()                                   # a batch, as "Select all" would build it
    before = ov.selected_wells()
    assert len(before) >= 2, "fixture needs at least two wells to make this meaningful"
    view = win._viewer_manager.open(list(before))
    _drain_until(qapp, lambda: view._pane is not None, timeout=10)
    try:
        _click(ov, "B3")
        _fire_pending_navigation(ov)
        assert ov.selected_wells() == before, "navigating collapsed the operator selection"
        assert win._selected_regions == list(before), "the window's run scope changed"
    finally:
        shutdown_plate_window(qapp, win)


def test_clicking_a_well_the_view_never_held_still_navigates_there(qapp, napari_pane_stub,
                                                                    squid_dataset):
    """THE REQUEST: a view over B2 follows a click on B3, adopting it through the cursor."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, ["B2"])
    mgr = win._viewer_manager
    try:
        assert view._regions == ["B2"]
        _click(win._overview, "B3")
        _fire_pending_navigation(win._overview)
        _drain_until(qapp, lambda: view._cursor.region == "B3", timeout=10)
        assert view._cursor.region == "B3"
        assert "B3" in view._regions, "the window did not adopt the region"
        assert mgr.views()[0].regions == ("B2", "B3"), "views() did not follow the adoption"
    finally:
        shutdown_plate_window(qapp, win)


def test_adopting_does_not_reload_the_region_already_on_screen(qapp, napari_pane_stub,
                                                                squid_dataset):
    """A re-scope that snapped back to index 0 would reload the mosaic twice and land on the wrong region."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, ["B2"])
    loads = []
    real = view._load_mosaic
    view._load_mosaic = lambda region, *a, **k: (loads.append(str(region)), real(region, *a, **k))[1]
    try:
        _click(win._overview, "B3")
        _fire_pending_navigation(win._overview)
        _drain_until(qapp, lambda: "B3" in loads, timeout=10)
        assert loads == ["B3"], f"expected one load of B3, got {loads}"
    finally:
        view._load_mosaic = real
        shutdown_plate_window(qapp, win)


# --- the modes ----------------------------------------------------------------------------------

def test_a_plain_click_with_no_view_open_still_selects(qapp, napari_pane_stub, squid_dataset):
    """The untouched half. With nothing to navigate the plate keeps its original meaning."""
    root, _ = squid_dataset
    win = _plate(qapp, root)
    try:
        assert win._overview._click_navigates is False
        _click(win._overview, "B3")
        assert win._overview.selected_wells() == ["B3"], "a plain click stopped selecting"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_plate_stops_navigating_when_the_last_view_closes(qapp, napari_pane_stub,
                                                              squid_dataset):
    """The mode is derived, never latched: closing the last view must hand the plate back."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, ["B2"])
    try:
        assert win._overview._click_navigates is True
        view.close()
        qapp.processEvents()
        assert win._overview._click_navigates is False, "the plate kept navigating with no view"
        _click(win._overview, "B3")
        assert win._overview.selected_wells() == ["B3"], "the plate did not go back to selecting"
    finally:
        shutdown_plate_window(qapp, win)


def test_clicking_empty_space_still_clears_the_selection(qapp, napari_pane_stub, squid_dataset):
    """THE ESCAPE HATCH."""
    root, _ = squid_dataset
    win = _plate(qapp, root)
    ov = win._overview
    ov.select_all()
    view = win._viewer_manager.open(ov.selected_wells())
    _drain_until(qapp, lambda: view._pane is not None, timeout=10)
    try:
        assert ov._click_navigates is True
        far = _pt(50, 50, CD)                          # well off the 2-region fixture plate
        ov.mousePressEvent(_mouse("press", far))
        ov.mouseReleaseEvent(_mouse("release", far, buttons=Qt.NoButton))
        assert ov.selected_wells() == [], "clicking empty space no longer clears the selection"
    finally:
        shutdown_plate_window(qapp, win)


# --- the two gestures must not fight ------------------------------------------------------------

def test_a_double_click_opens_a_window_without_moving_the_first(qapp, napari_pane_stub,
                                                                squid_dataset):
    """Qt delivers press/release/dblclick, so the release already started a navigation."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, ["B2"])
    ov = win._overview
    try:
        rc = next(rc for rc, wid in ov._by_rc.items() if wid == "B3")
        pos = _pt(rc[0], rc[1], CD)
        ov.mousePressEvent(_mouse("press", pos))
        ov.mouseReleaseEvent(_mouse("release", pos, buttons=Qt.NoButton))
        assert ov._nav.isActive(), "the release did not defer a navigation"
        ov.mouseDoubleClickEvent(_mouse("dblclick", pos))
        assert not ov._nav.isActive(), "the double-click did not cancel the pending navigation"
        qapp.processEvents()
        assert len(win._viewer_manager.windows) == 2, "the double-click did not open a window"
        assert view._cursor.region == "B2", "the double-click moved the window it should not have"
    finally:
        shutdown_plate_window(qapp, win)


def test_a_new_press_supersedes_a_pending_navigation(qapp, napari_pane_stub, squid_dataset):
    """A queued navigation from a well the user has moved on from must not land later."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, ["B2"])
    ov = win._overview
    try:
        _click(ov, "B3")
        assert ov._nav.isActive()
        rc = next(rc for rc, wid in ov._by_rc.items() if wid == "B2")
        ov.mousePressEvent(_mouse("press", _pt(rc[0], rc[1], CD)))
        assert not ov._nav.isActive() and ov._nav_well is None, "the stale navigation survived"
    finally:
        shutdown_plate_window(qapp, win)


def test_no_navigation_timer_survives_the_plate_close(qapp, napari_pane_stub, squid_dataset):
    """A single-shot timer that fires into a widget its owner has dropped is the shape `test_window_lifetime` exists for — and this one reaches OUT of the"""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, ["B2"])
    ov = win._overview
    _click(ov, "B3")
    assert ov._nav.isActive()
    shutdown_plate_window(qapp, win)
    assert not ov._nav.isActive(), "a navigation timer outlived the plate"


# --- the red frame ------------------------------------------------------------------------------

def test_navigating_moves_the_red_frame_without_the_user_opening_a_region(
        qapp, napari_pane_stub, squid_dataset):
    """The frame follows what the view shows, but `_current_well` must stay None: `activate` means "the user explicitly opened this region" and SCOPES"""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, ["B2"])
    try:
        _click(win._overview, "B3")
        _fire_pending_navigation(win._overview)
        _drain_until(qapp, lambda: view._cursor.region == "B3", timeout=10)
        assert win._overview._sel == tuple(win._fov_index["B3"]["rc"]), "the red frame did not follow"
        assert win._current_well is None, "navigating counted as the user opening a region"
    finally:
        shutdown_plate_window(qapp, win)


def test_a_region_this_acquisition_does_not_have_is_refused_out_loud(qapp, napari_pane_stub,
                                                                     squid_dataset):
    """A navigator that silently does nothing is indistinguishable from one that is broken."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, ["B2"])
    try:
        said_before = len(view._pane.said)
        assert view.show_region("ZZ99") is False
        assert len(view._pane.said) > said_before, "the refusal was silent"
        assert view._cursor.region == "B2", "a refused navigation moved the window anyway"
    finally:
        shutdown_plate_window(qapp, win)
