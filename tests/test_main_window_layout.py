"""The main window's layout (plate-dominant), measured on a really-shown window."""

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
    """Is *child* somewhere in *ancestor*'s widget subtree? Parentage, not coordinates."""
    p = child.parentWidget()
    while p is not None:
        if p is ancestor:
            return True
        p = p.parentWidget()
    return False


# --- the request: the plate on top, full width, about half the window ---------------------------

@pytest.mark.parametrize("size", [DESIGN, DESKTOP])
def test_the_plate_is_on_top_and_takes_about_half_the_window(qapp, monkeypatch, size):
    """The plate takes ~50% of the window; the floor is 48 so a one-pixel style change passes."""
    win = _window(qapp, monkeypatch, size)
    try:
        plate_host = win._body.widget(0)
        share = 100.0 * _h(plate_host) / win.height()
        assert share >= 48.0, (
            f"the plate is {share:.1f}% of a {size[0]}x{size[1]} window; Spencer asked for ~50%")
        # On top: above the band. The band is asserted to be LAST, not at a literal index.
        assert win._body.indexOf(plate_host) == 0
        assert win._body.indexOf(win._band_host) == win._body.count() - 1
        assert plate_host.geometry().bottom() <= win._band_host.geometry().top()
    finally:
        win.close()


def test_the_plate_spans_the_full_width(shown):
    """The plate must not be a child of the band's splitter, i.e. a column instead of a row."""
    win = shown
    plate_host = win._body.widget(0)
    assert plate_host.width() == win.centralWidget().width(), "the plate is no longer full width"
    assert win._right_col.indexOf(plate_host) == -1, "the plate got pulled into the band"
    assert _h(plate_host) > 0, "the band ate the whole window"


def test_the_plate_keeps_the_growth_when_the_window_grows(qapp, monkeypatch):
    """Stretch factors as behaviour: the band keeps its height, the plate takes the new space."""
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

def test_the_log_owns_the_band_and_the_navigator_is_gone(shown):
    """2026-08-19 (Julio's mock): the Window navigator is DELETED — its list was superseded by
    the deck's tabs — and the operator-tab bar hides while empty, so the LOG has the whole band.
    Absence is pinned with `not hasattr`, this repo's convention for deleted features."""
    win = shown
    assert not hasattr(win, "_open_views"), "the plate rebuilt a Window navigator"
    assert not hasattr(win, "_band"), "the horizontal band splitter survived its left child"

    assert win._log_panel.isVisible(), "the log is not on screen"
    assert _h(win._log_panel) > 100, f"the log has no usable height ({_h(win._log_panel)} px)"
    # The operator-tab bar exists but costs no pixels until an operator panel opens.
    assert win._left_tabs.count() == 0
    assert not win._left_tabs.isVisible(), "an empty operator-tab bar is taking the log's space"
    assert win._left_tabs.indexOf(win._log_panel) == -1
    assert win._right_col.indexOf(win._left_tabs) == 0, "Operator tabs are not above the Log"
    assert win._right_col.indexOf(win._log_panel) == 1

    # Opening an operator panel shows the bar; closing it hands the band back to the log.
    from qtpy.QtWidgets import QLabel
    win._open_op_tab("probe", "Probe", lambda: QLabel("probe"))
    assert win._left_tabs.isVisible() and win._left_tabs.count() == 1
    win._close_op_tab(0)
    assert win._left_tabs.count() == 0
    assert not win._left_tabs.isVisible(), "the bar kept the band after its last tab closed"


def test_neither_the_console_nor_the_band_can_be_dragged_to_nothing(shown):
    """A splitter will happily let you drag a child to zero."""
    assert shown._right_col.childrenCollapsible() is False
    assert shown._body.childrenCollapsible() is False


def test_the_band_cap_is_enforced_on_a_plain_host_not_on_the_splitter(shown):
    """QSplitterPrivate::recalc overwrites setMaximumHeight on the splitter, so the cap sits on a plain QWidget host."""
    win = shown
    assert win._band_host.maximumHeight() == V._BAND_MAX_PX
    assert _h(win._band_host) <= V._BAND_MAX_PX, (
        f"the band is {_h(win._band_host)} px in a {V._BAND_MAX_PX} px cap")
    assert _h(win._band_host) == V._BAND_DEFAULT_PX, (
        "the band did not open at its default height; the plate's share is derived from it")


# --- the status bars moved into the log ---------------------------------------------------------

def test_the_memory_and_progress_indicators_are_inside_the_log_panel(shown):
    """The status bars live inside the logger, driven by `StatusRow` — what survived the
    navigator's deletion (its tree went with the widget; these two bars still have a job)."""
    win = shown
    row, log = win._status_row, win._log_panel

    for name in ("_mem_label", "_mem_bar", "_work_label", "_work_bar"):
        w = getattr(row, name)
        assert _is_inside(w, log), f"{name} is not inside the log panel"

    # Still driven by the manager's signals through StatusRow's own handlers.
    row._on_memory(0.42)
    assert row._mem_bar.value() == 42, "the adopted memory bar stopped following the poller"


def test_the_adopted_status_strip_survives_collapsing_the_log(shown, qapp):
    """Collapsed hides the log body; the status strip stays visible."""
    win = shown
    win._log_panel.set_collapsed(True)
    qapp.processEvents()
    assert win._log_panel._status.isVisible(), "the status strip went with the log body"
    assert win._status_row._mem_bar.isVisible()
    win._log_panel.set_collapsed(False)


def test_the_progress_bar_is_still_absent_while_nothing_runs(shown):
    """Absent means nothing is running; the rule has to survive the move."""
    win = shown
    assert win._status_row._work_bar.isHidden()
    assert win._status_row._work_label.isHidden()


# --- the metadata label -------------------------------------------------------------------------

def test_the_acquisition_label_is_a_caption_not_a_headline(shown):
    """The acquisition-name strip is small and still says what it is for."""
    win = shown
    bar = win._plate_title.parentWidget()
    assert _h(bar) <= 26, f"the acquisition label is back to a {_h(bar)} px headline"
    css = win._plate_title.styleSheet()
    assert "font-size:12px" in css and "font-weight:600" in css
    win._on_hover("B7")
    assert "B7" in win._plate_title.text(), "the label stopped being the hover readout"


# --- the log in a window of its own --------------------------------------------------------------

def test_the_view_menu_reaches_the_log_in_every_state(shown, qapp):
    """View > Log reaches the panel docked, collapsed or floated."""
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
    """Re-dock returns the SAME object, so the lines already on screen are still there."""
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
    assert not win._left_tabs.isVisible(), "an EMPTY operator-tab bar showed itself on the float"
    assert _is_inside(win._status_row._mem_bar, panel), "the memory bar was left behind"

    win._redock_log()
    qapp.processEvents()
    assert win._log_panel is panel
    assert win._right_col.indexOf(panel) == 1, "the console did not come back under the operators"
    assert panel.isVisible() and _h(panel) > 50, "it came back with no height"
    assert panel.text() == before, "the scrollback was rebuilt rather than returned"
    assert _is_inside(win._status_row._mem_bar, panel), "the memory bar did not come back"


def test_closing_the_float_gives_the_console_back_rather_than_deleting_it(shown, qapp):
    """An operator float's close deletes; the console's must not."""
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
    """A floated console must not outlive the plate it reports on."""
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
    # The wrapper was deleteLater'd and the event loop has run, so touching it raises.
    with pytest.raises(RuntimeError):
        fl.isVisible()
    assert panel.parent() is not None, "the panel was orphaned into a top-level of its own"
