"""The log panel widget: bounded, coloured, live readouts, collapse, and real painting."""

from __future__ import annotations

import logging

import pytest

from qtpy.QtWidgets import QApplication

from squidxplorer._activity import ActivityLog
from squidxplorer._logpane import LogBus, color_for
from squidxplorer._logpanel import LogPanel, memory_line


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

def test_a_third_party_library_appears_in_the_panel_without_being_wired(panel, bus):
    """The bus attaches to the root logger and the panel is a sink of the bus."""
    bus.install()
    logging.getLogger("tilefusion.optimization").warning("fusing region manual0")
    assert "tilefusion" in panel.text()
    assert "fusing region manual0" in panel.text()


def test_a_line_is_coloured_by_its_level(panel, bus):
    bus.install()
    logging.getLogger("x").error("it broke")
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
        assert "well 199" in panel.text()
        assert "well 0 " not in panel.text()
    finally:
        panel.stop()
        panel.deleteLater()


# --- the two live readouts Squid shows continuously ---------------------------------------------

def test_the_memory_readout_is_a_real_sentence():
    line = memory_line()
    assert line.startswith("mem")
    assert "GiB" in line or "MiB" in line or line == "mem -"


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
    """The tally stays empty through an INFO-only run and only fills when there is something to say."""
    bus.install()
    for i in range(5):
        logging.getLogger("run").info("well %d ok", i)
    assert panel._tally_lbl.text() == "", "an INFO-only run must not raise a false alarm"
    logging.getLogger("x").warning("heads up")
    logging.getLogger("x").error("uh oh")
    tally = panel._tally_lbl.text()
    assert "1 warning" in tally and "1 error" in tally


# --- collapse must not steal pane space ---------------------------------------------------------

# --- it actually PAINTS -------------------------------------------------------------------------

def test_the_panel_actually_PAINTS_without_raising(qapp, bus):
    """Render into a pixmap for real: Qt swallows exceptions raised inside paint()."""
    import sys

    from qtpy.QtGui import QPixmap

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
    finally:
        sys.excepthook = original
        panel.stop()
        panel.deleteLater()
    assert not caught, f"painting the panel raised: {caught[0][1] if caught else ''}"


def test_a_measured_run_line_flows_through_the_panel(panel, bus):
    """``measure_run`` logs at INFO to the root logger, so its line reaches the panel unwired."""
    from squidxplorer._measure import MetricsLog, measure_run

    bus.install()
    with measure_run("mip", "2 regions", metrics=MetricsLog()):
        pass
    text = panel.text()
    assert "mip" in text and "peak" in text
