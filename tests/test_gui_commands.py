"""The GUI half of the ONE command surface — a WindowExecutor over a live PlateWindow."""

from __future__ import annotations

import pytest

from qtpy.QtWidgets import QApplication

import squidxplorer._viewer as V
from squidxplorer import _run_scope
from squidxplorer._command import (
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
    app.setProperty("_squidxplorer_test", True)
    return app


@pytest.fixture
def win(qapp):
    w = V.PlateWindow(None)
    yield w
    w.close()                    # stops + joins any run, uninstalls the log bus


@pytest.fixture
def open_win(win, squid_dataset):
    root, _ = squid_dataset
    assert win.commands.execute(OpenAcquisition(path=str(root))).ok
    return win


def test_the_window_exposes_a_command_bus(win):
    assert isinstance(win.commands, CommandBus)
    assert win.commands.surface == "gui"


def test_the_gui_surface_supports_the_run_and_control_commands(win):
    supported = win.commands.supported()
    for kind in ("open_acquisition", "describe", "list_operators", "run_operator", "stop_run",
                 "metrics"):
        assert kind in supported, f"the GUI cannot express {kind!r}"


def test_list_operators_is_answered_identically_to_the_engine(win):
    from squidxplorer._command import CommandBus as _Bus, EngineExecutor

    gui = win.commands.execute(ListOperators()).data["names"]
    engine = _Bus(EngineExecutor()).execute(ListOperators()).data["names"]
    assert gui == engine, "the two surfaces disagree on what can be run"


def test_describe_refuses_by_name_before_anything_is_open(win):
    r = win.commands.execute(Describe())
    assert r.refusal == NO_ACQUISITION


def test_open_then_describe_reports_the_windows_regions_and_scope_state(open_win):
    d = open_win.commands.execute(Describe()).data
    assert d["surface"] == "gui"
    assert d["regions"] and d["n_regions"] == len(d["regions"])
    assert "selection" in d and "current_region" in d
    assert list(_run_scope.RUN_SCOPES) == d["scopes"]


def test_opening_a_written_plate_is_refused_with_the_windows_own_sentence(win, tmp_path):
    r = win.commands.execute(OpenAcquisition(path=str(tmp_path)))
    assert r.status == "refused"


def test_an_unknown_operator_is_refused_by_name(open_win):
    r = open_win.commands.execute(RunOperator(operator="not_an_operator"))
    assert r.refusal == UNKNOWN_OPERATOR and "not_an_operator" in r.message


def test_running_with_nothing_open_is_refused(win):
    assert win.commands.execute(RunOperator(operator="mip")).refusal == NO_ACQUISITION


def test_a_run_STARTS_a_thread_rather_than_blocking(open_win, qapp):
    r = open_win.commands.execute(RunOperator(operator="mip", scope=_run_scope.SCOPE_PLATE,
                                              save=False))
    assert r.ok and r.status == "started", r.message
    assert open_win._worker is not None, "no worker thread was started"
    for _ in range(500):
        qapp.processEvents()
        if not _run_scope.operator_busy(open_win._worker, open_win._retired):
            break


def test_a_second_run_while_one_is_in_flight_is_refused_as_busy(open_win, qapp, monkeypatch):
    monkeypatch.setattr(_run_scope, "operator_busy", lambda *a, **k: True)
    r = open_win.commands.execute(RunOperator(operator="mip", scope=_run_scope.SCOPE_PLATE))
    assert r.refusal == BUSY


def test_stop_with_nothing_running_is_a_named_refusal_not_a_noop(open_win):
    r = open_win.commands.execute(StopRun())
    assert r.refusal == NO_RUN


def test_a_running_operator_is_measured_and_lands_in_the_shared_metrics(open_win, qapp):
    from squidxplorer._measure import METRICS

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
    seen = {}
    orig = open_win.run_operator

    def spy(key, out_parent=None, regions=None, save=True, tab_key=None, operator_kwargs=None):
        seen["regions"] = regions
        return None

    monkeypatch.setattr(open_win, "run_operator", spy)
    regions = open_win.commands.execute(Describe()).data["regions"][:1]
    open_win._selected_regions = list(regions)
    open_win.commands.execute(RunOperator(operator="mip", scope=_run_scope.SCOPE_SELECTION))
    assert seen["regions"] == regions, "the GUI did not resolve 'selected wells' from its selection"


def test_the_log_panel_is_stacked_under_the_operators_and_can_never_be_lost(win):
    """The panel exists for the window's whole life and is reachable from View > Log in every
    state — docked, collapsed, floated."""
    panel = win._log_panel

    assert win._left_tabs.indexOf(panel) == -1, "the log is back in the tab bar"
    assert win._right_col.indexOf(panel) >= 0, "the log is not in the right column at all"
    assert win._right_col.indexOf(win._left_tabs) == 0, "Operator tabs are not above the log"
    assert win._FIXED_TABS == 0, "no fixed tab since the cards moved to the views window's dock"

    assert win._left_tabs.count() == 0, "a tab opened before anything was asked for"

    panel.set_collapsed(True)
    win.show_log()
    assert not panel.collapsed, "View > Log did not expand the collapsed console"

    fl = win._float_log()
    assert fl is not None and win._floating[win._LOG_FLOAT_KEY] is fl
    assert win._float_log() is fl, "a second request built a SECOND console window"
    win.show_log()
    assert win._floating.get(win._LOG_FLOAT_KEY) is fl, "show_log dropped the float"

    # closing the float re-docks the same object rather than disposing it, so scrollback survives
    fl.close()
    assert win._LOG_FLOAT_KEY not in win._floating
    assert win._log_panel is panel, "the console was replaced"
    assert win._right_col.indexOf(panel) >= 0, "the console did not come back to the window"


def test_a_plate_run_opens_AND_closes_a_started_done_pair_in_the_console(open_win, qapp, caplog):
    """A run over one region carries a view id and an address; a plate-wide run carries only the
    view id, since no single Extent can describe a set of regions."""
    import logging

    from squidxplorer._address import Extent
    from squidxplorer._logpane import ADDRESS_FIELD, VIEW_FIELD

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
    open_win.commands.execute(RunOperator(operator="mip", scope=_run_scope.SCOPE_PLATE, save=False))
    # activity is started synchronously in run_operator, before the worker thread does anything
    assert open_win._activity.busy or open_win._activity.sentence() != "", \
        "the run did not register any activity"
    for _ in range(1000):
        qapp.processEvents()
        if not _run_scope.operator_busy(open_win._worker, open_win._retired):
            break
