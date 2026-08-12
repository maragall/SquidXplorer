"""The main window's layout, measured on a really-shown window (2026-08-03 v2, plate-dominant).

Julio and Spencer, with a drawing
(AI-docs/SquidXplorer/assets/2026-08-03-main-window-layout-v2-plate-dominant.png). This REVERSES
the stacking half of the restack that landed four hours earlier the same day, and keeps the rest:

    1. THE PLATE GOES ON TOP, full width, and takes ROUGHLY HALF THE WINDOW. Spencer: it was at
       the bottom losing prominence to text-heavy panels. This is the request, and it is the one
       thing here that is a number rather than a shape: measured before the change, the plate was
       32.8% of a 596x850 window and 35.6% of 1280x900.
    2. Under it, a BAND: Window Navigator on the left, Operator over Log on the right. The
       Operator-over-Log stack and the log's "option to open in a new window" are KEPT from the
       13:14 restack — the drawing did not change them, only where the block sits.
    3. THE STATUS BLOCK IS GONE from under the navigator. Julio, verbatim: "the status bar and
       memory bar should be moved to inside the logger so that we save space." Point 3 of the
       previous version of this file said that block was "untouched: they are the internals of one
       widget, ``OpenViewList``, and the restack does not open it". That is exactly what this
       change reverses, and it is why that test is REPLACED here rather than deleted quietly.
    4. The "Filename ..." strip is absorbed into the plate pane as a thin caption. Spencer: "Test
       metadata label (e.g. '10x laser Z-stack') too large, taking up excess space."

EVERY GEOMETRY ASSERTION HERE IS ON A REALLY-SHOWN, REALLY-SIZED WINDOW under
``QT_QPA_PLATFORM=offscreen``. "All three visible at once" is not a statement about the widget
tree, it is a statement about pixels: widgets can all be ``isVisible()`` and one of them can be
0 px tall. NOTHING HERE WAS VERIFIED ON A REAL SCREEN.
"""

from __future__ import annotations

import pytest

from qtpy.QtWidgets import QApplication

import squidxplorer._viewer as V


DESIGN = (596, 850)     # PlateWindow._DESIGN_W, _DESIGN_H — the shape the layout was drawn against
DESKTOP = (1280, 900)   # a second realistic shape: the plate's share must not fall on a wide screen


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


def _window(qapp, monkeypatch, size):
    """A window the user could actually look at: really sized, really shown."""
    w = V.PlateWindow(None)
    w.resize(*size)
    w.show()
    qapp.processEvents()
    return w


@pytest.fixture
def shown(qapp, monkeypatch):
    w = _window(qapp, monkeypatch, DESIGN)
    yield w
    w.close()


def _h(widget) -> int:
    return widget.geometry().height()


def _is_inside(child, ancestor) -> bool:
    """Is *child* somewhere in *ancestor*'s widget subtree? Parentage, not coordinates: a widget
    can overlap another on screen and belong to neither."""
    p = child.parentWidget()
    while p is not None:
        if p is ancestor:
            return True
        p = p.parentWidget()
    return False


# --- the request: the plate on top, full width, about half the window ---------------------------

@pytest.mark.parametrize("size", [DESIGN, DESKTOP])
def test_the_plate_is_on_top_and_takes_about_half_the_window(qapp, monkeypatch, size):
    """THE REQUEST, in the units Spencer asked it in: a share of THE WINDOW, not of the central
    widget. The menu bar and the status bar are real pixels the plate does not get, so measuring
    against the central widget would flatter the number by about 6%.

    Measured with this change, offscreen: 50.5% at 596x850 and 52.2% at 1280x900. The floor is 48
    rather than 50 so a one-pixel style change is not a test failure, and it is still far above the
    32.8% / 35.6% this replaces — the mutation that matters (band and plate swapped back, or the
    band left uncapped at its size hint) lands at 33-36% and fails this by a mile.
    """
    win = _window(qapp, monkeypatch, size)
    try:
        plate_host = win._body.widget(0)
        share = 100.0 * _h(plate_host) / win.height()
        assert share >= 48.0, (
            f"the plate is {share:.1f}% of a {size[0]}x{size[1]} window; Spencer asked for ~50%")
        # ON TOP: above the band, in the same parent, so the comparison is meaningful.
        #
        # The band is asserted to be LAST, not to sit at a literal index. It stood at index 2 while
        # the exploration pane occupied index 1; that pane was deleted on 2026-08-05 and the
        # constant went stale the moment it was, which is exactly the failure mode a hard-coded
        # sibling count has. "Last" is the property the layout actually promises, and the mutation
        # the test is for -- band and plate swapped -- still fails it.
        assert win._body.indexOf(plate_host) == 0
        assert win._body.indexOf(win._band_host) == win._body.count() - 1
        assert plate_host.geometry().bottom() <= win._band_host.geometry().top()
    finally:
        win.close()


def test_the_plate_spans_the_full_width(shown):
    """"Full width" is the other half of "dominant". The plate must not be a child of the band's
    splitter, which is what would silently make it a column instead of a row."""
    win = shown
    plate_host = win._body.widget(0)
    assert plate_host.width() == win.centralWidget().width(), "the plate is no longer full width"
    assert win._band.indexOf(plate_host) == -1, "the plate got pulled into the band"
    assert win._right_col.indexOf(plate_host) == -1
    assert _h(plate_host) > 0, "the band ate the whole window"


def test_the_plate_keeps_the_growth_when_the_window_grows(qapp, monkeypatch):
    """The stretch factors, stated as behaviour. Qt hands a resize delta out by stretch factor, so
    a band with stretch 0 keeps its height and the plate takes the rest. Without this a taller
    window gives half the new space to the panels and the plate's share never improves."""
    win = _window(qapp, monkeypatch, DESIGN)
    try:
        before = _h(win._body.widget(0)), _h(win._band_host)
        win.resize(DESIGN[0], DESIGN[1] + 300)
        qapp.processEvents()
        after = _h(win._body.widget(0)), _h(win._band_host)
        assert after[1] == before[1], "the band grew with the window instead of the plate"
        assert after[0] >= before[0] + 250, "the plate did not take the new height"
    finally:
        win.close()


# --- three panels on screen at once, no tab switch ----------------------------------------------

def test_navigator_operator_and_log_are_all_on_screen_at_once(shown):
    """THE SHAPE OF THE BAND. Not "all three exist" — all three have real pixels, simultaneously,
    with no gesture. The mutation this is written to catch is the easiest one to make by accident:
    putting the log back in ``_left_tabs``. A tab is ``isVisible()`` only while it is the current
    one, so a tabbed log makes the log 0 px tall for the operator's whole lifetime and vice versa.
    """
    win = shown
    for name, w in (("navigator", win._open_views), ("operator", win._left_tabs),
                    ("log", win._log_panel)):
        assert w.isVisible(), f"the {name} is not on screen"
        assert _h(w) > 100, f"the {name} has no usable height ({_h(w)} px)"

    # neither is a tab of the other, and the order is the drawing's
    assert win._left_tabs.indexOf(win._log_panel) == -1
    assert win._band.indexOf(win._open_views) == 0, "the navigator is not on the left of the band"
    assert win._band.indexOf(win._right_col) == 1
    assert win._right_col.indexOf(win._left_tabs) == 0, "Operator is not above Log"
    assert win._right_col.indexOf(win._log_panel) == 1
    assert win._log_panel.geometry().top() > win._left_tabs.geometry().top()
    # left-of, in real pixels: siblings of the same splitter, so geometry() is comparable
    assert win._open_views.geometry().right() <= win._right_col.geometry().left()


def test_neither_the_console_nor_the_band_can_be_dragged_to_nothing(shown):
    """"A console you can lose is not a console" was what ``_FIXED_TABS = 2`` bought. A splitter
    will happily let you drag a child to zero. The band needs the same guard one level up: drag it
    away and the navigator and the operators go with it."""
    assert shown._right_col.childrenCollapsible() is False
    assert shown._body.childrenCollapsible() is False


def test_the_band_cap_is_enforced_on_a_plain_host_not_on_the_splitter(shown):
    """REGRESSION TEST FOR A QT DEFECT THIS LAYOUT WORK FOUND, kept in substance across the v2
    restack because the defect did not move — only the widget it applies to was renamed.

    ``splitter.setMaximumHeight(...)`` does not hold. QSplitterPrivate::recalc calls
    setMaximumSize() ON THE SPLITTER out of its children's maximums every time a child changes, so
    it overwrites the cap. Measured on 83c486c offscreen at 596x850: the splitter's
    ``maximumHeight()`` read 16777215 and the strip rendered 479 px against a 240 px cap. The cap
    therefore sits on a plain QWidget host, which does not rewrite its own maximum.
    """
    win = shown
    assert win._band_host.maximumHeight() == V._BAND_MAX_PX
    assert _h(win._band_host) <= V._BAND_MAX_PX, (
        f"the band is {_h(win._band_host)} px in a {V._BAND_MAX_PX} px cap")
    assert _h(win._band_host) == V._BAND_DEFAULT_PX, (
        "the band did not open at its default height; the plate's share is derived from it")


# --- the status bars moved into the log ---------------------------------------------------------

def test_the_memory_and_progress_indicators_are_inside_the_log_panel(shown):
    """Julio: "the status bar and memory bar should be moved to inside the logger so that we save
    space."

    THIS REPLACES ``test_the_left_column_is_untouched``, which asserted the opposite — that the
    navigator and "the two status bars" were internals of one widget the restack did not open.
    That was true of the 13:14 restack and is the thing the v2 drawing explicitly undoes, so the
    expectation is rewritten rather than dropped. What survives of it is its first line: the
    navigator is still the band's left-hand child and still has real pixels.
    """
    win = shown
    nav, log = win._open_views, win._log_panel

    assert win._band.indexOf(nav) == 0 and nav.isVisible() and _h(nav) > 100

    for name in ("_mem_label", "_mem_bar", "_work_label", "_work_bar"):
        w = getattr(nav, name)
        assert _is_inside(w, log), f"{name} is not inside the log panel"
        assert not _is_inside(w, nav), f"{name} is still in the navigator's own subtree"

    # ONE of each, still driven by the navigator's handlers: the move was a reparent, not a rebuild.
    nav._on_memory(0.42)
    assert nav._mem_bar.value() == 42, "the adopted memory bar stopped following the poller"


def test_the_adopted_status_strip_survives_collapsing_the_log(shown, qapp):
    """Collapsed hides the log BODY. A status readout that vanishes when you fold the log away
    cannot tell you the app is busy, which is the one thing it is for — and this is now the only
    place the memory bar exists."""
    win = shown
    win._log_panel.set_collapsed(True)
    qapp.processEvents()
    assert win._log_panel._status.isVisible(), "the status strip went with the log body"
    assert win._open_views._mem_bar.isVisible()
    win._log_panel.set_collapsed(False)


def test_the_progress_bar_is_still_absent_while_nothing_runs(shown):
    """The rule the work bar was built with ("absent means nothing is running") has to survive the
    move, or its new home costs a permanent empty bar in the panel that was meant to save space."""
    win = shown
    assert win._open_views._work_bar.isHidden()
    assert win._open_views._work_label.isHidden()


# --- the metadata label -------------------------------------------------------------------------

def test_the_acquisition_label_is_a_caption_not_a_headline(shown):
    """Spencer: "Test metadata label (e.g. '10x laser Z-stack') too large, taking up excess space."

    This is the acquisition-name strip, which is also the drawing's "Filename ..." box. It rendered
    a 38 px band across the design window (17 px type, weight 800, 9 px padding) and 59 px at 1280
    wide, because ``rescale_fonts`` scales it with the window. It is NOT deleted — it is the only
    thing on screen naming the open acquisition and the plate's mode. What is pinned is that it is
    small, and that it still says what it is for.
    """
    win = shown
    bar = win._plate_title.parentWidget()
    assert _h(bar) <= 26, f"the acquisition label is back to a {_h(bar)} px headline"
    css = win._plate_title.styleSheet()
    assert "font-size:12px" in css and "font-weight:600" in css
    win._on_hover("B7")
    assert "B7" in win._plate_title.text(), "the label stopped being the hover readout"


# --- the log in a window of its own (KEPT from the 13:14 restack) --------------------------------

def test_the_view_menu_reaches_the_log_in_every_state(shown, qapp):
    """THE INVARIANT that replaced "cannot detach": the panel exists for the life of the window and
    View > Log reaches it docked, collapsed or floated. If any state can strand it, the 2026-07-29
    decision to make it a fixed tab was right and this whole change is a regression."""
    win = shown
    acts = {a.text().replace("&", ""): a for a in win.menuBar().actions()
            if a.menu() is not None and a.text().replace("&", "") == "View"}
    view = acts["View"].menu()
    labels = [a.text().replace("&", "") for a in view.actions()]
    assert "Log" in labels, "View > Log is gone"
    assert "Log in a New Window" in labels, "the drawing's 'option to open in a new window' is gone"

    log_act = next(a for a in view.actions() if a.text().replace("&", "") == "Log")

    log_act.trigger()                                  # docked
    qapp.processEvents()
    assert win._log_panel.isVisible() and not win._log_panel.collapsed

    win._log_panel.set_collapsed(True)                 # collapsed
    log_act.trigger()
    assert not win._log_panel.collapsed, "View > Log left the console collapsed"

    float_act = next(a for a in view.actions()
                     if a.text().replace("&", "") == "Log in a New Window")
    float_act.trigger()                                # floated
    qapp.processEvents()
    fl = win._floating[win._LOG_FLOAT_KEY]
    log_act.trigger()
    assert win._floating.get(win._LOG_FLOAT_KEY) is fl, "View > Log lost the floated console"
    win._redock_log()


def test_detaching_and_redocking_preserves_the_console_and_its_scrollback(shown, qapp):
    """Re-dock returns the SAME object, so the lines already on screen are still there. A
    close-and-rebuild would silently empty the console, which is the failure a user would only
    notice when they went looking for the line that explained what went wrong.

    It now also carries the memory and progress bars, so a rebuild would strand the one memory bar
    in the process inside a deleted widget."""
    win = shown
    win._log_panel._append("INFO", "a line that must survive the round trip")
    qapp.processEvents()
    before = win._log_panel.text()
    assert "must survive" in before

    panel = win._log_panel
    fl = win._float_log()
    qapp.processEvents()
    assert fl.content() is panel, "the float does not hold the panel itself"
    assert win._right_col.indexOf(panel) == -1, "the panel is in two places at once"
    assert win._left_tabs.isVisible(), "the operator pane went with it"
    assert _is_inside(win._open_views._mem_bar, panel), "the memory bar was left behind"

    win._redock_log()
    qapp.processEvents()
    assert win._log_panel is panel
    assert win._right_col.indexOf(panel) == 1, "the console did not come back under the operators"
    assert panel.isVisible() and _h(panel) > 50, "it came back with no height"
    assert panel.text() == before, "the scrollback was rebuilt rather than returned"
    assert _is_inside(win._open_views._mem_bar, panel), "the memory bar did not come back"


def test_closing_the_float_gives_the_console_back_rather_than_deleting_it(shown, qapp):
    """The one outcome that would make "open in a new window" the wrong call. An operator float's
    close routes through ``_dispose_tab_widget``, which deletes; the console's must not."""
    win = shown
    panel = win._log_panel
    win._float_log()
    qapp.processEvents()
    win._floating[win._LOG_FLOAT_KEY].close()
    qapp.processEvents()
    assert win._LOG_FLOAT_KEY not in win._floating
    assert win._log_panel is panel
    assert win._right_col.indexOf(panel) == 1
    assert panel.isVisible()


def test_the_float_is_swept_by_the_windows_close(qapp, monkeypatch):
    """A floated console must not outlive the plate it reports on.

    ``to-do/2026-08-03-window-lifetime-design.md`` has NOT decided whether child windows outlive
    the plate. This test states where the log float sits today: ``closeEvent`` already sweeps
    ``_floating`` (unlike RegionViewers), so the log lands on the safe side by construction. It
    matters more than for an operator float, because the panel is a live sink on the process-wide
    root logger and ``closeEvent`` uninstalls that bus a few lines later — a surviving log window
    would be a console attached to nothing. If that document later chooses "windows are peers",
    this test is the thing that has to be argued with rather than quietly deleted.
    """
    win = V.PlateWindow(None)
    win.show()
    qapp.processEvents()
    panel = win._log_panel
    fl = win._float_log()
    qapp.processEvents()
    assert fl.isVisible()
    win.close()
    qapp.processEvents()
    assert not win._floating, "a console window survived the plate that was logging into it"
    # The window itself is gone: `_redock_log` deleteLater'd it and the event loop has run, so
    # touching it raises rather than answering. That IS the sweep; asserting `isVisible()` on a
    # deleted wrapper is asserting on a corpse.
    with pytest.raises(RuntimeError):
        fl.isVisible()
    assert panel.parent() is not None, "the panel was orphaned into a top-level of its own"
