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

def test_the_log_owns_the_band_and_the_operator_tab_bar_hides_while_empty(shown):
    win = shown
    assert win._log_panel.isVisible(), "the log is not on screen"
    from squidxplorer._logpanel import LogPanel

    assert _h(win._log_panel) == win._log_panel.slot_px(), f"the log slot is {_h(win._log_panel)} px"
    # The operator-tab bar exists but costs no pixels until an operator panel opens.
    assert win._left_tabs.count() == 0
    assert not win._left_tabs.isVisible(), "an empty operator-tab bar is taking the log's space"
    assert win._left_tabs.indexOf(win._log_panel) == -1
    assert win._right_col.indexOf(win._left_tabs) == 0, "Operator tabs are not above the Log"
    assert win._right_col.indexOf(win._log_panel) == 1

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
    """The status bars live inside the logger, driven by `StatusRow` — what survived the navigator's deletion (its tree went with the widget; these two bars"""
    win = shown
    row, log = win._status_row, win._log_panel

    for name in ("_work_label", "_work_bar"):
        w = getattr(row, name)
        assert _is_inside(w, log), f"{name} is not inside the log panel"
    assert not hasattr(row, "_mem_bar"), "the memory bar is back"


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

