"""The GUI half of the ONE command surface — a WindowExecutor over a live PlateWindow.

The point these tests hold down: the GUI answers the SAME commands the headless EngineExecutor
does (``tests/test_command.py``), so "does the button work" and "does the CLI work" stop being two
questions. They construct a real window (with the ndviewer stubbed to avoid the offscreen-GL
segfault) and drive it only through ``window.commands`` — never by calling private methods — which
is exactly how an agent would.
"""

from __future__ import annotations

import pytest

from qtpy.QtWidgets import QApplication

import squidmip._viewer as V
from squidmip import _run_scope
from tests.test_viewer import _StubDetail   # the proven ndviewer stub (no offscreen-GL segfault)
from squidmip._command import (
    BUSY,
    CommandBus,
    Describe,
    ListOperators,
    Metrics,
    NO_ACQUISITION,
    NO_RUN,
    OpenAcquisition,
    RunOperator,
    StopRun,
    UNKNOWN_OPERATOR,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)
    return app


@pytest.fixture
def stub_detail(monkeypatch):
    monkeypatch.setattr(V.PlateWindow, "_make_detail_viewer", lambda self: _StubDetail())


@pytest.fixture
def win(qapp, stub_detail):
    w = V.PlateWindow(None)
    yield w
    w.close()                    # stops + joins any run, uninstalls the log bus


@pytest.fixture
def open_win(win, squid_dataset):
    root, _ = squid_dataset
    assert win.commands.execute(OpenAcquisition(path=str(root))).ok
    return win


# --- the window HAS a command bus, and it is the shared type ------------------------------------

def test_the_window_exposes_a_command_bus(win):
    assert isinstance(win.commands, CommandBus)
    assert win.commands.surface == "gui"


def test_the_gui_surface_supports_the_run_and_control_commands(win):
    supported = win.commands.supported()
    for kind in ("open_acquisition", "describe", "list_operators", "run_operator", "stop_run",
                 "metrics"):
        assert kind in supported, f"the GUI cannot express {kind!r}"


def test_list_operators_is_answered_identically_to_the_engine(win):
    from squidmip._command import CommandBus as _Bus, EngineExecutor

    gui = win.commands.execute(ListOperators()).data["names"]
    engine = _Bus(EngineExecutor()).execute(ListOperators()).data["names"]
    assert gui == engine, "the two surfaces disagree on what can be run"


# --- opening and describing the window's live state ---------------------------------------------

def test_describe_refuses_by_name_before_anything_is_open(win):
    r = win.commands.execute(Describe())
    assert r.refusal == NO_ACQUISITION


def test_open_then_describe_reports_the_windows_regions_and_scope_state(open_win):
    d = open_win.commands.execute(Describe()).data
    assert d["surface"] == "gui"
    assert d["regions"] and d["n_regions"] == len(d["regions"])
    # the live pieces a run's scope is resolved from are all exposed
    assert "selection" in d and "current_region" in d
    assert list(_run_scope.RUN_SCOPES) == d["scopes"]


def test_opening_a_written_plate_is_refused_with_the_windows_own_sentence(win, tmp_path):
    # a path that is not a raw acquisition: ingest refuses in the readout, surfaced as a refusal
    r = win.commands.execute(OpenAcquisition(path=str(tmp_path)))
    assert r.status == "refused"


# --- running goes through the window's own run_operator -----------------------------------------

def test_an_unknown_operator_is_refused_by_name(open_win):
    r = open_win.commands.execute(RunOperator(operator="minerva"))
    assert r.refusal == UNKNOWN_OPERATOR and "minerva" in r.message


def test_running_with_nothing_open_is_refused(win):
    assert win.commands.execute(RunOperator(operator="mip")).refusal == NO_ACQUISITION


def test_a_run_STARTS_a_thread_rather_than_blocking(open_win, qapp):
    """A GUI run must not block the event loop, so the honest thing the command returns is that the
    run BEGAN — status 'started', not 'completed'."""
    r = open_win.commands.execute(RunOperator(operator="mip", scope=_run_scope.SCOPE_PLATE,
                                              save=False))
    assert r.ok and r.status == "started", r.message
    assert open_win._worker is not None, "no worker thread was started"
    # let it finish so teardown is clean
    for _ in range(500):
        qapp.processEvents()
        if not _run_scope.operator_busy(open_win._worker, open_win._retired):
            break


def test_a_second_run_while_one_is_in_flight_is_refused_as_busy(open_win, qapp, monkeypatch):
    """Two runs at once is a named refusal, not a silent overwrite."""
    # make the worker look perpetually alive for the duration of the check
    monkeypatch.setattr(_run_scope, "operator_busy", lambda *a, **k: True)
    r = open_win.commands.execute(RunOperator(operator="mip", scope=_run_scope.SCOPE_PLATE))
    assert r.refusal == BUSY


def test_stop_with_nothing_running_is_a_named_refusal_not_a_noop(open_win):
    r = open_win.commands.execute(StopRun())
    assert r.refusal == NO_RUN


def test_a_running_operator_is_measured_and_lands_in_the_shared_metrics(open_win, qapp):
    """The GUI's run path writes the SAME METRICS log the CLI does — one measurement, one table,
    both surfaces. After a run completes, the comparison table has a 'mip' row."""
    from squidmip._measure import METRICS

    before = len(METRICS)
    open_win.commands.execute(RunOperator(operator="mip", scope=_run_scope.SCOPE_PLATE, save=False))
    ok = False
    for _ in range(1000):
        qapp.processEvents()
        if len(METRICS) > before:
            ok = True
            break
    assert ok, "the GUI run recorded no metrics"
    table = open_win.commands.execute(Metrics(operator="mip")).data["table"]
    assert table and table[0]["operator"] == "mip"


def test_the_run_scope_is_resolved_by_the_shared_resolver_from_window_state(open_win, qapp,
                                                                            monkeypatch):
    """A run scoped to 'selected wells' resolves against the window's OWN selection, through the
    same _run_scope.resolve_run_scope the headless surface uses — not a second GUI-only resolver."""
    seen = {}
    orig = open_win.run_operator

    def spy(key, out_parent=None, regions=None, save=True, tab_key=None, operator_kwargs=None):
        seen["regions"] = regions
        # do not actually start a thread
        return None

    monkeypatch.setattr(open_win, "run_operator", spy)
    regions = open_win.commands.execute(Describe()).data["regions"][:1]
    open_win._selected_regions = list(regions)
    open_win.commands.execute(RunOperator(operator="mip", scope=_run_scope.SCOPE_SELECTION))
    assert seen["regions"] == regions, "the GUI did not resolve 'selected wells' from its selection"


# --- the log panel is THE one global console, and it is a fixed tab -----------------------------

def test_the_log_panel_is_stacked_under_the_operators_and_can_never_be_lost(win):
    """Rewritten THREE times, and the history is the point.

    2026-07-28: this asserted ``win._explore_col`` held the pane above the log panel, an
    exploration COLUMN that the decentralization removed; the attribute no longer existed, so the
    test could only ever AttributeError. It never reported that, because the QStyle lifetime bug
    (tests/test_window_lifetime.py) segfaulted this file five tests earlier and took the summary
    line with it, so a test that was broken and a test that never ran looked the same. It was then
    rewritten to pin the log as a separate top-level QMainWindow, with a note saying a future move
    to a tab should fail loudly instead of silently.

    2026-07-29, Task 1: the log became ONE GLOBAL CONSOLE and a FIXED TAB in the operators tab
    space, never closable and never detachable, and this test pinned exactly that:
    ``index < win._FIXED_TABS``, ``tabText == "Log"``, ``not hasattr(win, "_log_window")``,
    ``win._detach_tab(index) is None``.

    2026-08-03: THAT EXPECTATION IS CHANGED HERE, DELIBERATELY, and this docstring is the record.
    Julio, with a drawing: "I think that we should modify the layout of our main window" — Operator
    above, Log below, both visible at once, and "Log (option to open in a new window)". Two things
    move, and only one of them touches the 2026-07-29 decision:

    * Stacking does not reverse it at all. That decision was WINDOW vs NOT-WINDOW; tab vs stacked
      panel was never argued, and the tab bar was the only reason the two alternated. The panel was
      written as a stacked one ("The bottom-right log panel", _logpanel.py) and a console that is
      always on screen is MORE global than one behind a tab, not less.
    * "Open in a new window" does reverse the second half, and reintroduces the object the old
      version of this test asserted the absence of. It is justified on facts, not taste: the old
      `_log_window` was built and shown on EVERY launch, which is why Spencer saw it land over the
      main window every time, whereas this is a user gesture on an always-present panel; and the
      QStyle segfault the old float participated in is fixed at the seam a new float uses
      (_qt_tabs.py refuses the per-widget Fusion style for exactly that reason).

    WHAT IS PINNED INSTEAD is the invariant that survives the change, and it is stronger than
    "cannot detach": the panel exists for the life of the window and is reachable from View > Log
    in EVERY state — docked, collapsed, floated. What changes is where it is, never whether it is.
    If a floated log could be closed and not come back, the 2026-07-29 decision was right and this
    is a regression; that is the assertion at the end of this test.
    """
    panel = win._log_panel

    # NOT a tab any more: it is a sibling of the tab widget in the right column's vertical splitter
    assert win._left_tabs.indexOf(panel) == -1, "the log is back in the tab bar"
    assert win._right_col.indexOf(panel) >= 0, "the log is not in the right column at all"
    assert win._right_col.indexOf(win._left_tabs) == 0, "Operators is not above the log"
    assert win._FIXED_TABS == 1, "only the Operators home tab is fixed now"

    # ...and the tab bar it left still protects its own home tab
    assert win._close_op_tab(0) is None and win._left_tabs.count() >= 1
    assert win._detach_tab(0) is None, "the Operators home tab detached"

    # REACHABLE FROM THE VIEW MENU IN EVERY STATE. Docked and collapsed:
    panel.set_collapsed(True)
    win.show_log()
    assert not panel.collapsed, "View > Log did not expand the collapsed console"

    # ...and floated: show_log raises the float rather than losing track of it
    fl = win._float_log()
    assert fl is not None and win._floating[win._LOG_FLOAT_KEY] is fl
    assert win._float_log() is fl, "a second request built a SECOND console window"
    win.show_log()
    assert win._floating.get(win._LOG_FLOAT_KEY) is fl, "show_log dropped the float"

    # CLOSING THE FLOAT MUST NOT LOSE THE CONSOLE. An operator float's close disposes its widget;
    # this one re-docks. Same object, so the scrollback survives.
    fl.close()
    assert win._LOG_FLOAT_KEY not in win._floating
    assert win._log_panel is panel, "the console was replaced"
    assert win._right_col.indexOf(panel) >= 0, "the console did not come back to the window"


def test_a_plate_run_opens_AND_closes_a_started_done_pair_in_the_console(open_win, qapp, caplog):
    """The root plate is a window too, so its actions carry a view id and an address.

    The pair matters more than either line: an action that starts and then says nothing is
    indistinguishable from one still running, which is why the "done" is closed from the drain
    (fires on ok, failed and stopped alike) rather than from finished_ok.

    Asserted through the RECORDS. A run over exactly one region HAS an address; a plate-wide run
    is a set of extents that one Extent cannot say, so it carries the view id alone -- that gap is
    deliberate and is Task 2's to close."""
    import logging

    from squidmip._address import Extent
    from squidmip._logpane import ADDRESS_FIELD, VIEW_FIELD

    regions = open_win.commands.execute(Describe()).data["regions"][:1]
    with caplog.at_level(logging.INFO):
        open_win.run_operator("mip", regions=list(regions), save=False)
        for _ in range(2000):
            qapp.processEvents()
            if not _run_scope.operator_busy(open_win._worker, open_win._retired):
                break
        for _ in range(50):
            qapp.processEvents()          # let the queued finished slot land

    lines = [r for r in caplog.records if hasattr(r, VIEW_FIELD)]
    started = [r for r in lines if r.getMessage().endswith("started")]
    done = [r for r in lines if "done in" in r.getMessage()]
    assert started, f"the run never announced itself: {[r.getMessage() for r in lines]}"
    assert done, f"the run started and then went quiet: {[r.getMessage() for r in lines]}"
    assert getattr(started[-1], VIEW_FIELD) == 0, "the root plate is view 0"
    assert getattr(started[-1], ADDRESS_FIELD) == Extent(region_id=regions[0])
    assert started[-1].getMessage().startswith(f"[0] {regions[0]} ")


def test_a_run_shows_up_as_activity_in_the_log_panel_header(open_win, qapp, monkeypatch):
    """The activity registry the panel's header reads is fed by the run — this is what makes 'the
    GUI is doing something' visible. Freeze the run as busy and check the header lit up."""
    open_win.commands.execute(RunOperator(operator="mip", scope=_run_scope.SCOPE_PLATE, save=False))
    # the activity was started synchronously in run_operator, before the worker thread does anything
    assert open_win._activity.busy or open_win._activity.sentence() != "", \
        "the run did not register any activity"
    # drain to clean teardown
    for _ in range(1000):
        qapp.processEvents()
        if not _run_scope.operator_busy(open_win._worker, open_win._retired):
            break
