"""The GUI half of the ONE command surface — a WindowExecutor over a live PlateWindow.

The point these tests hold down: the GUI answers the SAME commands the headless EngineExecutor
does (``tests/test_command.py``), so "does the button work" and "does the CLI work" stop being two
questions. They construct a real window (with the ndviewer stubbed to avoid the offscreen-GL
segfault) and drive it only through ``window.commands`` — never by calling private methods — which
is exactly how an agent would.
"""

from __future__ import annotations

import pytest

from PyQt5.QtWidgets import QApplication

import squidmip._viewer as V
from squidmip import _explore
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
    # the three live pieces a run's scope is resolved from are all exposed
    assert "selection" in d and "current_region" in d and "parked_subset" in d
    assert list(_explore.RUN_SCOPES) == d["scopes"]


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
    r = open_win.commands.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE,
                                              save=False))
    assert r.ok and r.status == "started", r.message
    assert open_win._worker is not None, "no worker thread was started"
    # let it finish so teardown is clean
    for _ in range(500):
        qapp.processEvents()
        if not _explore.operator_busy(open_win._worker, open_win._retired):
            break


def test_a_second_run_while_one_is_in_flight_is_refused_as_busy(open_win, qapp, monkeypatch):
    """Two runs at once is a named refusal, not a silent overwrite."""
    # make the worker look perpetually alive for the duration of the check
    monkeypatch.setattr(_explore, "operator_busy", lambda *a, **k: True)
    r = open_win.commands.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE))
    assert r.refusal == BUSY


def test_stop_with_nothing_running_is_a_named_refusal_not_a_noop(open_win):
    r = open_win.commands.execute(StopRun())
    assert r.refusal == NO_RUN


def test_a_running_operator_is_measured_and_lands_in_the_shared_metrics(open_win, qapp):
    """The GUI's run path writes the SAME METRICS log the CLI does — one measurement, one table,
    both surfaces. After a run completes, the comparison table has a 'mip' row."""
    from squidmip._measure import METRICS

    before = len(METRICS)
    open_win.commands.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE, save=False))
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
    same _explore.resolve_run_scope the headless surface uses — not a second GUI-only resolver."""
    seen = {}
    orig = open_win.run_operator

    def spy(key, out_parent=None, regions=None, save=True, tab_key=None, operator_kwargs=None):
        seen["regions"] = regions
        # do not actually start a thread
        return None

    monkeypatch.setattr(open_win, "run_operator", spy)
    regions = open_win.commands.execute(Describe()).data["regions"][:1]
    open_win._selected_regions = list(regions)
    open_win.commands.execute(RunOperator(operator="mip", scope=_explore.SCOPE_SELECTION))
    assert seen["regions"] == regions, "the GUI did not resolve 'selected wells' from its selection"


# --- the log panel is mounted below the exploration pane ----------------------------------------

def test_the_log_panel_is_mounted_below_the_exploration_pane(win):
    col = win._explore_col
    assert col.widget(0) is win._explore_pane
    assert col.widget(1) is win._log_panel


def test_a_run_shows_up_as_activity_in_the_log_panel_header(open_win, qapp, monkeypatch):
    """The activity registry the panel's header reads is fed by the run — this is what makes 'the
    GUI is doing something' visible. Freeze the run as busy and check the header lit up."""
    open_win.commands.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE, save=False))
    # the activity was started synchronously in run_operator, before the worker thread does anything
    assert open_win._activity.busy or open_win._activity.sentence() != "", \
        "the run did not register any activity"
    # drain to clean teardown
    for _ in range(1000):
        qapp.processEvents()
        if not _explore.operator_busy(open_win._worker, open_win._retired):
            break
