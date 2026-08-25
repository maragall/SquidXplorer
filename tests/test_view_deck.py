"""ViewDeck — many views as tabs in one window, one of them current."""

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

from squidxplorer import _viewer as V  # noqa: E402
from squidxplorer._view_deck import _SOFT_TAB_CAP  # noqa: E402
from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)


def _tabbed_plate(qapp, root, n_views=2):
    """A plate whose views are TABS, with *n_views* of them open."""
    win = V.PlateWindow(None)
    win.ingest(str(root))
    mgr = win._viewer_manager
    mgr.tabbed_views = True
    order = list(win._order)
    views = []
    for i in range(n_views):
        v = mgr.open([order[i % len(order)]])
        assert v is not None
        views.append(v)
    _drain_until(qapp, lambda: all(v._pane is not None for v in views), timeout=10)
    return win, mgr, mgr.deck(create=False), views


# --- holding ------------------------------------------------------------------------------------

def test_a_spawned_view_lands_as_a_tab_and_stays_registered(qapp, napari_pane_stub, squid_dataset):
    """Becoming a tab must not change what a view IS: same id, same registry entry."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    try:
        assert deck is not None and deck.count() == 2
        assert sorted(w.window_id for w in mgr.windows) == sorted(v.window_id for v in views)
        for v in views:
            assert v.host is deck, "a tabbed view does not know its deck"
            assert not v.isWindow(), "a tabbed view is still a top-level"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_deck_is_made_only_when_a_view_is_opened(qapp, napari_pane_stub, squid_dataset):
    """Lazy, so a session that only looks at the plate builds no window nobody asked for — which is the premise `test_no_orphan_windows` is entitled to."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._viewer_manager.tabbed_views = True
    try:
        assert win._viewer_manager.deck(create=False) is None
        assert win._viewer_manager.decks() == []
    finally:
        shutdown_plate_window(qapp, win)


# --- the lifecycle obstacle: a tab removal fires no close event ----------------------------------

def test_closing_through_the_manager_leaves_no_tab_behind(qapp, napari_pane_stub, squid_dataset):
    """`ViewerManager.close` is what the navigator and the plate use. It must reach a tab."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    wid = views[0].window_id
    try:
        mgr.close(wid)
        qapp.processEvents()
        assert wid not in [w.window_id for w in mgr.windows]
        assert deck.count() == 1
        assert all(p.window_id != wid for p in deck.pages()), "a closed view is still a tab"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_deck_goes_away_once_its_last_view_does(qapp, napari_pane_stub, squid_dataset):
    """An empty deck must not stay ON SCREEN: it is a top-level, and a visible one with nothing in it keeps the process alive showing an empty frame — the"""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        mgr.close(views[0].window_id)
        qapp.processEvents()
        assert deck.count() == 0
        assert not deck.isVisible(), "an empty deck is still on screen"
        again = mgr.open([list(win._order)[0]])
        qapp.processEvents()
        assert mgr.deck(create=False) is deck, "a second deck was built"
        assert again.host is deck and deck.isVisible()
    finally:
        shutdown_plate_window(qapp, win)


# --- one at a time ------------------------------------------------------------------------------

def test_only_the_current_tab_is_active(qapp, napari_pane_stub, squid_dataset):
    """"Work with one tab at a time", stated as the render-halt rule the app already has: a view that is not the current page of an active deck is not being"""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    a, b = views
    calls = {}
    for v in (a, b):
        v.set_active = (lambda active, v=v: calls.__setitem__(v.window_id, active))
    try:
        deck.set_current(a)
        qapp.processEvents()
        assert calls.get(b.window_id) is False, "the outgoing tab was not stood down"
        assert a.isVisible() and not b.isVisible(), "more than one page has pixels"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_current_tab_is_the_focused_view(qapp, napari_pane_stub, squid_dataset):
    """THE CONTRACT WITH PLATE-AS-NAVIGATOR."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    a = views[0]
    try:
        deck.set_current(a)
        qapp.processEvents()
        assert mgr.focused_id == a.window_id, "switching tabs did not move the focus"
        assert mgr.active_view() is a
    finally:
        shutdown_plate_window(qapp, win)


# --- detach ---------------------------------------------------------------------------------

def test_detaching_gives_back_a_real_window_and_keeps_it_registered(
        qapp, napari_pane_stub, squid_dataset):
    """Detach is not a close."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    page = views[0]
    title = page.windowTitle()
    try:
        deck.undock_page(page)
        qapp.processEvents()
        assert page.host is None and page.isWindow(), "the page is not a window again"
        assert page.window_id in [w.window_id for w in mgr.windows], "detach deregistered it"
        assert page.windowTitle() == title, "the [id] name did not survive the reparent"
        assert deck.count() == 1
        assert not page._disposed, "detach disposed the view"
    finally:
        page.close()
        shutdown_plate_window(qapp, win)


def test_a_page_survives_many_dock_undock_cycles(qapp, napari_pane_stub, squid_dataset):
    """THE SEGFAULT CANARY, and phase 2's feasibility gate: dragging a tab between decks is this operation."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    page = views[0]
    pane = page._pane
    try:
        for _ in range(20):
            deck.undock_page(page)
            qapp.processEvents()
            deck.dock_page(page)
            qapp.processEvents()
        assert page._pane is pane, "a reparent rebuilt the pane"
        assert page.host is deck and not page.isWindow()
        assert page.window_id in [w.window_id for w in mgr.windows]
    finally:
        shutdown_plate_window(qapp, win)


# --- teardown -------------------------------------------------------------------------------

def test_closing_the_deck_disposes_every_page(qapp, napari_pane_stub, squid_dataset):
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=3)
    panes = [v._pane for v in views]
    try:
        deck.close()
        qapp.processEvents()
        assert all(v._disposed for v in views), "a page outlived its deck undisposed"
        assert all(p.shutdowns == 1 for p in panes), "a napari pane leaked with its deck"
        assert mgr.windows == [], "the registry still holds views from a closed deck"
    finally:
        shutdown_plate_window(qapp, win)


def test_closing_the_plate_takes_the_deck_with_it(qapp, napari_pane_stub, squid_dataset):
    """A deck is a top-level, so one left standing keeps the process alive with no plate — still holding the single-instance lock, which is the bug"""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    win.close()
    qapp.processEvents()
    assert not deck.isVisible(), "the deck outlived the plate"
    assert all(v._disposed for v in views)
    shutdown_plate_window(qapp, win)


def test_a_detach_that_fires_after_teardown_is_ignored(qapp, napari_pane_stub, squid_dataset):
    """`_DetachTabBar` defers its callback through `QTimer.singleShot(0, lambda: ...)` — a self-capturing lambda on a process-global timer."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    try:
        deck._closing = True
        deck._detach_page(0, deck._tabs)          # must be a no-op, not a reparent
        assert deck.count() == 2, "a detach ran during teardown"
        deck._closing = False
    finally:
        shutdown_plate_window(qapp, win)


# --- what the first use of it asked for --------------------------------------------------------

def test_every_tab_carries_its_own_close_button(qapp, napari_pane_stub, squid_dataset):
    """Spencer, on first use: "we could use a more obvious close tab button." A tab removal fires no close event, so the button must dispose."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    victim, keeper = views
    pane = victim._pane
    try:
        assert deck._tabs.tabsClosable(), "tabs have no close button"
        deck._tabs.tabCloseRequested.emit(deck._tabs.indexOf(victim))
        qapp.processEvents()
        assert victim._disposed, "the close button untabbed without disposing"
        assert pane.shutdowns == 1, "the close button leaked a napari pane"
        assert victim.window_id not in [w.window_id for w in mgr.windows]
        assert deck.count() == 1 and deck.current_page() is keeper
    finally:
        shutdown_plate_window(qapp, win)


def test_view_close_all_views_closes_every_tab(qapp, napari_pane_stub, squid_dataset):
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    try:
        assert win._close_views_act is not None
        win._close_all_views()
        qapp.processEvents()
        assert mgr.windows == [], "View > Close All Views left views open"
    finally:
        shutdown_plate_window(qapp, win)


# --- the right-edge operator dock is RETIRED (Julio, 2026-08-25: "I think that the operator
# --- right hand dock is obsolete"). The bulk path is a view's Run on plate; an operator's
# --- controls INSERT into the view's own left column. ------------------------------------------

def test_the_operator_dock_is_retired(qapp, napari_pane_stub, squid_dataset):
    import importlib

    import pytest as _pytest

    with _pytest.raises(ModuleNotFoundError):
        importlib.import_module("squidxplorer._operator_dock")
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        assert not hasattr(deck, "_operator_dock"), "the deck still grows an operator dock"
        assert not hasattr(mgr, "operator_dock_installer"), "the installer seam survived"
        assert not hasattr(win, "_op_docks") and not hasattr(win, "_op_cards")
    finally:
        shutdown_plate_window(qapp, win)


# --- the cost, said out loud ------------------------------------------------------------------

def test_the_deck_names_the_memory_once_there_are_many_views(qapp, napari_pane_stub,
                                                             squid_dataset):
    """MEASURED AT ~88 MB A VIEW."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    try:
        assert "MB" not in deck.statusBar().currentMessage(), "warned with only two views open"
        order = list(win._order)
        while deck.count() <= _SOFT_TAB_CAP:
            mgr.open([order[deck.count() % len(order)]])
        qapp.processEvents()
        msg = deck.statusBar().currentMessage()
        assert f"{deck.count()} views open" in msg
        assert "MB" in msg, f"past the soft cap the deck says nothing about cost: {msg!r}"
    finally:
        shutdown_plate_window(qapp, win)


# --- the FOV walk, as a child view ---------------------------------------------------------

def test_the_fovs_chip_opens_one_child_view_over_the_current_region(qapp, napari_pane_stub,
                                                                    squid_dataset):
    """One chip, one child, one tab — through the same `_spawn` every other opener arrives at."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        parent = views[0]
        before = deck.count()
        parent._open_fovs()
        qapp.processEvents()

        assert deck.count() == before + 1, "the FOV view must arrive as a tab, like every child"
        child = mgr.windows[-1]
        assert child._fov_mode is True
        assert child.parent_id == parent.window_id, "it must nest under the view it came from"
        assert child._roi_bbox is None, "a FOV walk is not cropped — it frames, it does not crop"
        assert child.display_name.startswith("FOVs · ")
        assert f"◂ view {parent.window_id}" in child.display_name

        view = mgr.view_for(child.window_id)
        assert view is not None and view.kind == "fovs"
        assert view.roi_bbox is None
    finally:
        shutdown_plate_window(qapp, win)


def test_the_fovs_chip_is_disabled_inside_a_fovs_view_and_says_why(qapp, napari_pane_stub,
                                                                   squid_dataset):
    """A dead-looking control must carry its reason, not just be grey."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        views[0]._open_fovs()
        qapp.processEvents()
        child = mgr.windows[-1]
        chip = child._btn_fovs
        assert not chip.isEnabled()
        assert "already steps through FOVs" in chip.toolTip()
    finally:
        shutdown_plate_window(qapp, win)


def test_a_fovs_view_keeps_its_region_slider_and_its_time_bar(qapp, napari_pane_stub,
                                                              squid_dataset):
    """Hidden, not absent — so every call site stays unconditional and the plate can still navigate this window to another region (`show_region`), which the"""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        views[0]._open_fovs()
        qapp.processEvents()
        child = mgr.windows[-1]
        assert child._slider is not None
        assert child._time_point_bar is not None
        assert child._fov_slider is not None, "a FOVs view is the one window that HAS this axis"
        assert views[0]._fov_slider is None, (
            "an ordinary window must not pay for an axis it cannot use")
    finally:
        shutdown_plate_window(qapp, win)
