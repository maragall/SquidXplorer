"""The log panel WIDGET — mounted bottom-right, below the exploration pane.

The seam's rules (bounded, colour, third-party appears) are tested in ``test_logpane.py`` without
Qt. These tests are about the WIDGET: that it is bounded in practice, colours by level, shows the
RAM and activity readouts Squid shows continuously, collapses without stealing pane space, and —
the one role-level tests miss — actually PAINTS without raising (the technique that caught two
bugs in the layer tree).
"""

from __future__ import annotations

import logging

import pytest

from PyQt5.QtWidgets import QApplication

from squidmip._activity import ActivityLog
from squidmip._logpane import LogBus, color_for
from squidmip._logpanel import LogPanel, memory_line


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def bus():
    b = LogBus()
    yield b
    b.uninstall()


@pytest.fixture()
def panel(qapp, bus):
    p = LogPanel(bus, ActivityLog(), max_lines=50)
    yield p
    p.stop()
    p.deleteLater()


# --- lines arrive, attributed and coloured ------------------------------------------------------

def test_a_logged_line_reaches_the_panel(panel, bus):
    bus.install()
    logging.getLogger("squidmip.test").info("hello from a run")
    assert "hello from a run" in panel.text()


def test_a_third_party_library_appears_in_the_panel_without_being_wired(panel, bus):
    """The design property, end to end: a library that never heard of us logs, and the user sees
    it — because the bus attaches to the ROOT logger and the panel is a sink of the bus."""
    bus.install()
    logging.getLogger("tilefusion.optimization").warning("fusing region manual0")
    assert "tilefusion" in panel.text()
    assert "fusing region manual0" in panel.text()


def test_a_line_is_coloured_by_its_level(panel, bus):
    bus.install()
    logging.getLogger("x").error("it broke")
    # the QPlainTextEdit holds HTML with the level colour; read it back from the document
    html = panel._view.document().toHtml()
    assert color_for("ERROR").lstrip("#").lower() in html.lower()


def test_markup_in_a_log_line_is_shown_not_interpreted(panel, bus):
    """A log line 'shape <5, 4>' must appear verbatim, not vanish as a bogus tag."""
    bus.install()
    logging.getLogger("x").info("array shape <5, 4> ok")
    assert "<5, 4>" in panel.text()


# --- bounded, for free --------------------------------------------------------------------------

def test_the_view_is_bounded_no_matter_how_many_lines(qapp, bus):
    panel = LogPanel(bus, ActivityLog(), max_lines=20)
    bus.install()
    try:
        for i in range(200):
            logging.getLogger("run").info("well %d projected", i)
        assert panel.line_count() <= 20, "an unbounded log body is a leak with a nice UI"
        # the newest line survived; the oldest was evicted
        assert "well 199" in panel.text()
        assert "well 0 " not in panel.text()
    finally:
        panel.stop()
        panel.deleteLater()


# --- the two live readouts Squid shows continuously ---------------------------------------------

def test_the_memory_readout_is_a_real_sentence():
    line = memory_line()
    assert line.startswith("mem")
    # either a measured footprint or the honest dash, never a crash
    assert "GiB" in line or "MiB" in line or line == "mem —"


def test_the_activity_line_follows_the_activity_registry(qapp, bus):
    activity = ActivityLog()
    panel = LogPanel(bus, activity, max_lines=10)
    try:
        assert panel._activity_lbl.text() == "idle"
        activity.start("fuse", "fusing B2", total=None)
        assert "fusing B2" in panel._activity_lbl.text()
        activity.end("fuse")
        assert panel._activity_lbl.text() == "idle"
    finally:
        panel.stop()
        panel.deleteLater()


def test_warnings_and_errors_are_tallied_in_the_header(panel, bus):
    """Squid's warningErrorWidget auto-hides when nothing is pending; the tally starts empty and
    only fills when there is something to say."""
    bus.install()
    assert panel._tally_lbl.text() == ""
    logging.getLogger("x").warning("heads up")
    logging.getLogger("x").error("uh oh")
    tally = panel._tally_lbl.text()
    assert "1 warning" in tally and "1 error" in tally


def test_an_ordinary_info_run_leaves_the_error_tally_empty(panel, bus):
    bus.install()
    for i in range(5):
        logging.getLogger("run").info("well %d ok", i)
    assert panel._tally_lbl.text() == "", "an INFO-only run must not raise a false alarm"


# --- collapse must not steal pane space ---------------------------------------------------------

def test_collapsing_hides_the_body_and_caps_the_height(panel):
    assert not panel.collapsed
    panel.set_collapsed(True)
    assert panel.collapsed
    assert not panel._view.isVisibleTo(panel), "the body is still showing when collapsed"
    # the widget must not claim body-sized space it is not drawing
    assert panel.maximumHeight() <= panel.sizeHint().height() + 1
    panel.set_collapsed(False)
    assert panel._view.isVisibleTo(panel)
    assert panel.maximumHeight() > 1000, "expanding did not release the height cap"


def test_the_header_survives_collapse_so_the_status_is_never_hidden(panel):
    """A status bar that vanishes cannot tell you the app is busy — which is the one thing it is
    for. The RAM and activity labels stay visible when the body is gone."""
    panel.set_collapsed(True)
    assert panel._mem_lbl.isVisibleTo(panel)
    assert panel._activity_lbl.isVisibleTo(panel)


def test_the_toggle_text_reflects_the_state(panel):
    panel.set_collapsed(False)
    assert "▾" in panel._toggle.text()
    panel.set_collapsed(True)
    assert "▸" in panel._toggle.text()


# --- it actually PAINTS -------------------------------------------------------------------------

def test_the_panel_actually_PAINTS_without_raising(qapp, bus):
    """Render into a pixmap for real. Serving a widget's roles is not the same as surviving its
    paint — the layer tree shipped 54 tracebacks a launch that role-level tests never saw, because
    Qt swallows exceptions raised inside paint()."""
    import sys

    from PyQt5.QtGui import QPixmap

    panel = LogPanel(bus, ActivityLog(), max_lines=50)
    bus.install()
    logging.getLogger("run").info("well 0 projected")
    logging.getLogger("run").warning("well 1 skipped")
    panel.resize(600, 160)

    caught = []
    original = sys.excepthook
    sys.excepthook = lambda *a: caught.append(a)
    try:
        pm = QPixmap(panel.size())
        panel.render(pm)
        panel.set_collapsed(True)
        panel.render(pm)                    # collapsed must paint too
    finally:
        sys.excepthook = original
        panel.stop()
        panel.deleteLater()
    assert not caught, f"painting the panel raised: {caught[0][1] if caught else ''}"


def test_a_measured_run_line_flows_through_the_panel(panel, bus):
    """The measurement's one-line-per-run reaches the panel with no extra wiring, because
    ``measure_run`` logs at INFO to the root logger and the panel is a sink of the root logger."""
    from squidmip._measure import MetricsLog, measure_run

    bus.install()
    with measure_run("mip", "2 regions", metrics=MetricsLog()):
        pass
    text = panel.text()
    assert "mip" in text and "peak" in text
