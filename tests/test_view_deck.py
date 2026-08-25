"""ViewDeck — many views as tabs in one window, one of them current.

Spencer, 2026-08-10: "many selections open (as tabs) but only work with one tab at a time."

WHAT THESE CAN AND CANNOT SEE. The suite runs under a Qt platform with no OpenGL and every view
here is built on a stub pane, so nothing below proves that a napari canvas RENDERS in a tab. That
was checked by hand on the workstation before any of this was written — four canvases in one
QTabWidget, one page visible at a time, container minimum 925x725 against a 1920x1032 work area,
and a page surviving 20 dock/undock cycles with the same pane and the same napari Viewer. These
tests pin the parts that are not about pixels: the registry, the lifecycle, and who is active.

TABS ARE OPT-IN AND EACH TEST TURNS THEM ON. That is not a testing convenience — it is the state
the app ships in. Turned on for every spawn, reparenting on every open took four existing files
from passing to ABORTING (0xC0000005 / 0xC0000409), and an abort is not a failure you can read.
Until that is understood rather than routed around, a deck is something a caller asks for.

WHY GEOMETRY IS NEVER ASSERTED HERE. A non-current tab is 0x0 offscreen. `test_main_window_layout`
argues that a tab is the wrong home for something the user must read alongside something else —
that argument does not transfer to views, where the user asked for one at a time and a hidden
canvas that stops drawing is the memory policy landing for free. But the 0-px hazard is real, so
tab switching is asserted through BEHAVIOUR: visibility, `set_active`, and the focused id.
"""

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
from squidxplorer._region_viewer import RegionViewer  # noqa: E402
from squidxplorer._view_deck import ViewDeck, _SOFT_TAB_CAP  # noqa: E402
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
    """Lazy, so a session that only looks at the plate builds no window nobody asked for — which
    is the premise `test_no_orphan_windows` is entitled to."""
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

def test_closing_a_tab_deregisters_the_view(qapp, napari_pane_stub, squid_dataset):
    """THE OBSTACLE. `WA_DeleteOnClose` and the `closed` signal both hang off `closeEvent`, and a
    tab removal fires none — so without routing through `dispose` the registry would silently stop
    being told, and every worker join would silently stop happening."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    victim, keeper = views[0], views[1]
    pane = victim._pane
    try:
        deck.close_page(victim)
        qapp.processEvents()
        assert victim.window_id not in [w.window_id for w in mgr.windows], "the registry kept it"
        assert victim._disposed, "the page was untabbed but never disposed"
        assert pane.shutdowns == 1, "its napari pane was never shut down"
        assert deck.count() == 1 and deck.current_page() is keeper
    finally:
        shutdown_plate_window(qapp, win)


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
    """An empty deck must not stay ON SCREEN: it is a top-level, and a visible one with nothing in
    it keeps the process alive showing an empty frame — the plateless-remainder bug, one container
    out.

    It is HIDDEN rather than destroyed, and reused if another view opens. Qt's
    quit-on-last-window rule counts visible windows, so hiding settles the lifetime question; and
    tearing a container full of GL children down and rebuilding it is exactly the churn this
    codebase's segfault history is made of."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        mgr.close(views[0].window_id)
        qapp.processEvents()
        assert deck.count() == 0
        assert not deck.isVisible(), "an empty deck is still on screen"
        # ...and it comes back for the next view rather than a second one being built.
        again = mgr.open([list(win._order)[0]])
        qapp.processEvents()
        assert mgr.deck(create=False) is deck, "a second deck was built"
        assert again.host is deck and deck.isVisible()
    finally:
        shutdown_plate_window(qapp, win)


# --- one at a time ------------------------------------------------------------------------------

def test_only_the_current_tab_is_active(qapp, napari_pane_stub, squid_dataset):
    """"Work with one tab at a time", stated as the render-halt rule the app already has: a view
    that is not the current page of an active deck is not being watched and must not draw."""
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
    """THE CONTRACT WITH PLATE-AS-NAVIGATOR. The plate drives `manager.active_view()`, so a tab
    switch has to move it — otherwise clicking a well steers a view the user cannot see."""
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
    """Detach is not a close. The same object comes back as a top-level, still in the registry,
    still carrying its `[id] name` — the only visible join between a log line and a view."""
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
    """THE SEGFAULT CANARY, and phase 2's feasibility gate: dragging a tab between decks is this
    operation. A regression here does not fail an assertion, it kills the interpreter — which is
    the point, and why `test_window_lifetime` is written the same way."""
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
    """A deck is a top-level, so one left standing keeps the process alive with no plate — still
    holding the single-instance lock, which is the bug `PlateWindow.closeEvent` records."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    win.close()
    qapp.processEvents()
    assert not deck.isVisible(), "the deck outlived the plate"
    assert all(v._disposed for v in views)
    shutdown_plate_window(qapp, win)


def test_a_detach_that_fires_after_teardown_is_ignored(qapp, napari_pane_stub, squid_dataset):
    """`_DetachTabBar` defers its callback through `QTimer.singleShot(0, lambda: ...)` — a
    self-capturing lambda on a process-global timer. A press landing just before the deck closes
    would otherwise call into a deleted deck, which is `test_window_lifetime`'s BUG 3 exactly."""
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
    """Spencer, on first use: "we could use a more obvious close tab button."

    Closing a view was reachable from the navigator's "Close selected views" — in the OTHER window
    — or from the deck's title bar, which closes all of them. Asserted through the SIGNAL rather
    than a click, because a close button is a Qt-drawn sub-control with no widget of its own to
    press offscreen; what matters is that the request lands on the same `close_page` everything
    else uses, so a view closed here is disposed and deregistered rather than merely untabbed."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    victim = views[0]
    pane = victim._pane
    try:
        assert deck._tabs.tabsClosable(), "tabs have no close button"
        deck._tabs.tabCloseRequested.emit(deck._tabs.indexOf(victim))
        qapp.processEvents()
        assert victim._disposed, "the close button untabbed without disposing"
        assert pane.shutdowns == 1, "the close button leaked a napari pane"
        assert victim.window_id not in [w.window_id for w in mgr.windows]
        assert deck.count() == 1
    finally:
        shutdown_plate_window(qapp, win)


def test_the_window_navigator_is_gone_and_the_operator_dock_took_its_jobs(
        qapp, napari_pane_stub, squid_dataset):
    """2026-08-19: the ViewDeck's tabs superseded the navigator list. The ABSENCE is pinned the
    way this repo pins deleted features (`not hasattr`, never `is None`), and the two jobs that
    survived are checked where they landed: close-all in the View menu, the operator surface in
    the deck's collapsible right-edge dock."""
    import squidxplorer._region_viewer as RV

    assert not hasattr(RV, "OpenViewList"), "the navigator widget is back"
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    try:
        assert not hasattr(win, "_open_views"), "the plate still builds a navigator"
        assert not hasattr(mgr, "raise_views"), "a navigator-only method survived its widget"
        assert not hasattr(mgr, "collapse_all"), "a navigator-only method survived its widget"
        # Close-all stayed reachable: the View-menu action closes every view.
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


def test_controls_inserts_the_param_slot_under_the_operators_row(qapp, napari_pane_stub,
                                                                 squid_dataset):
    """Julio (2026-08-25): "see this as an insertion to a list." ⚙ controls re-hosts the
    plate's live panel into THIS view's param slot; a second click removes it; the plate's
    kwargs reader still sees the same widget (one source of truth)."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    v = views[0]
    try:
        v.operator_panel()
        combo = v._op_combo
        i = next(k for k in range(combo.count()) if combo.itemData(k) == "stitch")
        combo.setCurrentIndex(i)
        v._show_operator_controls()
        qapp.processEvents()
        panel = v._inserted_panel
        assert panel is not None and panel is win._op_tabs["stitch"], (
            "the inserted panel is not the plate's live widget")
        assert v._param_slot.indexOf(panel) >= 0, "the panel is not in the view's param slot"
        assert win._left_tabs.indexOf(panel) == -1, "the panel is still a plate tab"
        # The values set IN THE SLOT are what a run reads: one source of truth.
        panel.widgets["registration_channel"].setValue(2)
        assert win.operator_kwargs_for("stitch")["registration_channel"] == 2
        # The second click REMOVES the slot, and the panel survives (plate registry).
        v._show_operator_controls()
        assert v._inserted_panel is None
        assert win._op_tabs["stitch"] is panel and panel.widgets["registration_channel"].value() == 2
    finally:
        shutdown_plate_window(qapp, win)


def test_a_disposed_view_releases_the_inserted_panel_alive(qapp, napari_pane_stub,
                                                           squid_dataset):
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    v = views[0]
    try:
        v.operator_panel()
        combo = v._op_combo
        combo.setCurrentIndex(next(k for k in range(combo.count())
                                   if combo.itemData(k) == "stitch"))
        v._show_operator_controls()
        panel = v._inserted_panel
        panel.widgets["registration_channel"].setValue(7)
        v.dispose()
        qapp.processEvents()
        assert win._op_tabs["stitch"] is panel, "disposing the view lost the plate's panel"
        assert panel.widgets["registration_channel"].value() == 7, "the panel's state died with the view"
    finally:
        shutdown_plate_window(qapp, win)


# --- the cost, said out loud ------------------------------------------------------------------

def test_the_deck_names_the_memory_once_there_are_many_views(qapp, napari_pane_stub,
                                                             squid_dataset):
    """MEASURED AT ~88 MB A VIEW. A tab strip hides that each tab holds its own napari viewer and
    GL context, which is exactly when someone opens twelve. Nothing is refused — a cap that blocked
    work would be worse than the memory — but the number stops being invisible."""
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


def test_a_view_cannot_be_both_an_roi_child_and_a_fov_walk(qapp, napari_pane_stub, squid_dataset):
    """Refused by name at construction rather than resolved by a silent precedence rule."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        with pytest.raises(ValueError, match="cannot be both"):
            RegionViewer(None, {}, ["A1"], window_id=99,
                         roi_bbox=(0.0, 0.0, 1.0, 1.0), fovs=True)
    finally:
        shutdown_plate_window(qapp, win)


def test_a_fovs_view_keeps_its_region_slider_and_its_time_bar(qapp, napari_pane_stub,
                                                              squid_dataset):
    """Hidden, not absent — so every call site stays unconditional and the plate can still
    navigate this window to another region (`show_region`), which the walk must survive."""
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
