"""The main window's layout, measured on a really-shown window (2026-08-03 restack).

Julio, with a drawing (AI-docs/SquidXplorer/assets/2026-08-03-main-window-layout-redesign.png):

    "Another thing, I think that we should modify the layout of our main window."

The drawing's before/after says four things. Three of them are changes and one of them is already
true, and saying which is which is half the value of this file:

    1. The right column was a TAB BAR alternating Operators and Log. It becomes a vertical STACK,
       Operator above, Log below, BOTH VISIBLE AT ONCE. This is the request.
    2. The Log gains "option to open in a new window".
    3. Window Navigator and the Status bars stay in the left column. Untouched: they are the
       internals of one widget, ``OpenViewList``, and the restack does not open it.
    4. "Plate view gains full width" — ALREADY TRUE before this change and after it. The plate has
       never been a child of the top strip's splitter; it is a sibling of it in the root layout.
       Pinned below anyway, because a claim that needs no work still needs to keep being true.

EVERY GEOMETRY ASSERTION HERE IS ON A REALLY-SHOWN, REALLY-SIZED WINDOW under
``QT_QPA_PLATFORM=offscreen``. "Both visible at once" is not a statement about the widget tree, it
is a statement about pixels: two widgets can both be ``isVisible()`` and one of them can be 0 px
tall. Nothing here was verified on a real screen.
"""

from __future__ import annotations

import pytest

from qtpy.QtWidgets import QApplication

import squidmip._viewer as V
from tests.test_viewer import _StubDetail   # the proven ndviewer stub (no offscreen-GL segfault)


DESIGN = (596, 850)     # PlateWindow._DESIGN_W, _DESIGN_H — the shape the layout was drawn against


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)
    return app


@pytest.fixture
def shown(qapp, monkeypatch):
    """A window the user could actually look at: really sized, really shown."""
    monkeypatch.setattr(V.PlateWindow, "_make_detail_viewer", lambda self: _StubDetail())
    w = V.PlateWindow(None)
    w.resize(*DESIGN)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def _h(widget) -> int:
    return widget.geometry().height()


# --- the request: both panels on screen at once, no tab switch ----------------------------------

def test_operator_and_log_are_both_on_screen_at_once(shown, qapp):
    """THE REQUEST. Not 'both exist' — both have real pixels, simultaneously, with no gesture.

    The mutation this is written to catch is the one that would be easiest to make by accident:
    putting the log back in ``_left_tabs``. A tab is ``isVisible()`` only while it is the current
    one, so a tabbed log makes the log's height 0 for the operator's whole lifetime and vice versa.
    """
    win = shown
    assert win._left_tabs.isVisible() and win._log_panel.isVisible()
    assert _h(win._left_tabs) > 100, "the operator pane has no usable height"
    assert _h(win._log_panel) > 100, "the log has no usable height — five lines is a status light"

    # neither is a tab of the other, and the order is the drawing's: Operator on top
    assert win._left_tabs.indexOf(win._log_panel) == -1
    assert win._right_col.indexOf(win._left_tabs) == 0
    assert win._right_col.indexOf(win._log_panel) == 1
    assert win._log_panel.geometry().top() > win._left_tabs.geometry().top()


def test_the_console_cannot_be_dragged_to_nothing(shown):
    """"A console you can lose is not a console" was what ``_FIXED_TABS = 2`` bought. A splitter
    will happily let you drag a child to zero, so the invariant is re-bought explicitly."""
    assert shown._right_col.childrenCollapsible() is False


def test_the_strip_cap_is_actually_enforced(shown):
    """REGRESSION TEST FOR A DEFECT THIS RESTACK FOUND, not for the restack itself.

    ``top_row.setMaximumHeight(_TOP_ROW_COMPACT_PX)`` did not work. QSplitterPrivate::recalc calls
    setMaximumSize() ON THE SPLITTER out of its children's maximums every time a child changes, so
    it overwrote the cap. Measured on 83c486c offscreen at 596x850: ``_top_row.maximumHeight()``
    read 16777215 and the strip rendered 479 px tall, not 240. The only thing re-applying the cap
    was ``_sync_top_row_height`` firing on ``currentChanged`` — which this change deletes. The cap
    now sits on a plain QWidget host, which does not rewrite its own maximum.
    """
    win = shown
    assert win._top_row_host.maximumHeight() == V._TOP_ROW_COMPACT_PX
    assert _h(win._top_row_host) <= V._TOP_ROW_COMPACT_PX, (
        f"the strip is {_h(win._top_row_host)} px in a {V._TOP_ROW_COMPACT_PX} px cap")


def test_the_plate_still_spans_the_whole_window(shown):
    """Point 4 of the drawing, which needed no work. The plate is a sibling of the top strip in the
    root layout, never a child of its splitter, so it has always been full width."""
    win = shown
    plate_host = win._body.widget(0)
    assert plate_host.width() == win.centralWidget().width(), "the plate is no longer full width"
    assert win._top_row.indexOf(plate_host) == -1, "the plate got pulled into the top strip"
    assert win._right_col.indexOf(plate_host) == -1
    # `geometry()` is parent-relative, so compare two SIBLINGS of the root layout, not a child of
    # `_body` against a child of `root` (which reads 0 > 519 and looks like a real failure).
    assert win._body.geometry().top() > win._top_row_host.geometry().bottom(), "plate not below"
    assert _h(plate_host) > 0, "the strip ate the whole window"


def test_the_left_column_is_untouched(shown):
    """Point 3. The Window Navigator and the two status bars are internals of ONE widget, and this
    change does not open it: they are still the top-left child of the top strip."""
    win = shown
    assert win._top_row.indexOf(win._open_views) == 0
    assert win._open_views.isVisible() and _h(win._open_views) > 100


# --- the log in a window of its own -------------------------------------------------------------

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
    notice when they went looking for the line that explained what went wrong."""
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

    win._redock_log()
    qapp.processEvents()
    assert win._log_panel is panel
    assert win._right_col.indexOf(panel) == 1, "the console did not come back under the operators"
    assert panel.isVisible() and _h(panel) > 50, "it came back with no height"
    assert panel.text() == before, "the scrollback was rebuilt rather than returned"


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
    monkeypatch.setattr(V.PlateWindow, "_make_detail_viewer", lambda self: _StubDetail())
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
