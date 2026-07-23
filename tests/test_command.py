"""The ONE command surface: named, declarative, serialisable, and never raising into a caller.

Headless — no Qt. The GUI's half of the same surface is tested in ``tests/test_gui_commands.py``,
and the fact that both are driven by the SAME commands is the point of the layer.
"""

from __future__ import annotations

import json

import pytest

from squidmip import _explore
from squidmip._command import (
    BAD_COMMAND,
    BAD_SCOPE,
    CommandBus,
    CommandResult,
    Describe,
    EMPTY_SCOPE,
    EngineExecutor,
    ListOperators,
    Metrics,
    NOT_SUPPORTED_HERE,
    NO_ACQUISITION,
    OpenAcquisition,
    RunOperator,
    StopRun,
    UNKNOWN_COMMAND,
    UNKNOWN_OPERATOR,
    UNKNOWN_REGION,
    parse_command,
)


@pytest.fixture()
def bus():
    return CommandBus(EngineExecutor())


@pytest.fixture()
def open_bus(squid_dataset):
    root, _arrays = squid_dataset
    b = CommandBus(EngineExecutor())
    assert b.execute(OpenAcquisition(path=str(root))).ok
    return b


# --- commands are DATA -------------------------------------------------------------------------

def test_a_command_survives_a_round_trip_through_plain_data():
    """The whole reason a command is a model and not a method call: something outside this process
    — an agent, a stored workflow, a replay — must be able to say it."""
    cmd = RunOperator(operator="mip", regions=["B2", "B3"], save=False)
    payload = json.loads(json.dumps(cmd.model_dump()))
    again = parse_command(payload)
    assert again == cmd


def test_a_command_can_be_written_the_way_a_human_writes_it():
    cmd = parse_command({"kind": "run_operator", "operator": "mip", "scope": "whole dataset"})
    assert isinstance(cmd, RunOperator) and cmd.operator == "mip"


def test_an_unknown_command_is_refused_by_name_and_lists_what_exists(bus):
    r = bus.execute({"kind": "frobnicate"})
    assert r.status == "refused" and r.refusal == UNKNOWN_COMMAND
    assert "run_operator" in r.message


def test_a_misspelled_field_is_refused_rather_than_silently_ignored(bus):
    """``region`` (singular) is the plausible guess an agent makes. Ignoring it would run the
    WHOLE PLATE instead of one well — hours of compute from a typo."""
    r = bus.execute({"kind": "run_operator", "operator": "mip", "region": "B2"})
    assert r.status == "refused" and r.refusal == BAD_COMMAND
    assert "region" in r.message


def test_a_command_describes_its_target_before_anything_runs():
    assert "mip" in RunOperator(operator="mip", regions=["B2"]).describe()
    assert "1 named region" in RunOperator(operator="mip", regions=["B2"]).describe()
    assert "not saved" in RunOperator(operator="mip").describe()


def test_every_command_kind_is_registered_and_carries_its_discriminator():
    from squidmip._command import COMMANDS

    samples = {
        "open_acquisition": OpenAcquisition(path="/x"),
        "list_operators": ListOperators(),
        "describe": Describe(),
        "run_operator": RunOperator(operator="mip"),
        "stop_run": StopRun(),
        "metrics": Metrics(),
    }
    for kind, model in COMMANDS.items():
        assert model.kind == kind
        # every command round-trips through plain data — that is what "serialisable" means
        assert parse_command(samples[kind].model_dump()).kind == kind


# --- every command returns a RESULT, and the bus never raises -----------------------------------

def test_the_bus_never_raises_even_when_the_executor_does():
    """The GUI calls this from a Qt slot, where a raised exception is SWALLOWED by Qt and the user
    sees a button that did nothing."""

    class Exploding:
        surface = "exploding"

        def do_describe(self, cmd):
            raise RuntimeError("kaboom")

    r = CommandBus(Exploding()).execute(Describe())
    assert r.status == "refused" and "kaboom" in r.message


def test_an_executor_that_forgets_to_return_a_result_is_refused_not_believed():
    class Sloppy:
        surface = "sloppy"

        def do_describe(self, cmd):
            return {"looks": "fine"}

    r = CommandBus(Sloppy()).execute(Describe())
    assert r.status == "refused" and "CommandResult" in r.message


def test_a_command_this_surface_cannot_express_is_a_named_refusal(bus):
    """The honest edge of the migration. The headless engine has no run to stop, and it says so
    by name rather than pretending."""
    r = bus.execute(StopRun())
    assert r.status == "refused" and r.refusal == NOT_SUPPORTED_HERE
    assert "engine" in r.message


def test_a_surface_reports_which_commands_it_supports(bus):
    supported = bus.supported()
    assert "run_operator" in supported and "describe" in supported
    assert "stop_run" not in supported


def test_a_result_is_falsy_when_it_refused(bus):
    assert not bus.execute(Describe())
    assert bool(bus.execute(ListOperators()))


def test_raise_for_refusal_is_opt_in_for_scripts(bus):
    with pytest.raises(RuntimeError, match="no_acquisition"):
        bus.execute(Describe()).raise_for_refusal()


# --- introspection: what an agent asks first ----------------------------------------------------

def test_list_operators_answers_off_the_engine_registry_not_a_card_table(bus):
    r = bus.execute(ListOperators())
    assert r.ok
    from squidmip import available_projectors, available_region_operators

    assert set(r.data["names"]) == set(available_projectors()) | set(available_region_operators())
    assert "mip" in r.data["names"] and "stitch" in r.data["names"]


def test_list_operators_reports_the_consumed_axis_so_a_caller_knows_the_output_shape(bus):
    rows = {row["name"]: row for row in bus.execute(ListOperators()).data["operators"]}
    assert rows["mip"]["kind"] == "z-reducer" and rows["mip"]["consumes"] == ["z"]
    assert rows["bgsub"]["kind"] == "plane-op" and rows["bgsub"]["consumes"] == []
    assert rows["stitch"]["kind"] == "region-operator"


def test_a_newly_registered_operator_appears_with_no_command_layer_edit(bus):
    """The registry scales to n algorithms; the command surface must scale with it for free."""
    from squidmip import add_projector
    from squidmip._engine import _PROJECTORS

    add_projector("test_only_op", lambda planes: next(iter(planes)))
    try:
        assert "test_only_op" in bus.execute(ListOperators()).data["names"]
    finally:
        _PROJECTORS.pop("test_only_op", None)


def test_describe_refuses_by_name_before_anything_is_open(bus):
    r = bus.execute(Describe())
    assert r.refusal == NO_ACQUISITION and "open_acquisition" in r.message


def test_describe_names_the_regions_channels_and_scopes_a_run_could_target(open_bus):
    d = open_bus.execute(Describe()).data
    assert d["regions"] and d["channels"]
    assert d["n_regions"] == len(d["regions"])
    assert list(_explore.RUN_SCOPES) == d["scopes"]


# --- the target, resolved by the ONE existing owner ----------------------------------------------

def test_nothing_selected_means_everything_the_established_convention(open_bus, monkeypatch):
    """``scope='selected wells'`` with nothing selected IS the whole dataset — byte-for-byte the
    behaviour that existed before a selector existed."""
    seen = {}
    import squidmip._command as mod

    def fake_project_plate(reader, **kw):
        seen["regions"] = kw.get("regions")
        return iter(())

    monkeypatch.setattr("squidmip.project_plate", fake_project_plate)
    open_bus.execute(RunOperator(operator="mip"))
    assert seen["regions"] is None, "None is the whole-plate path; a list would be a subset"


def test_an_explicit_region_list_wins_over_the_scope(open_bus, monkeypatch):
    seen = {}
    monkeypatch.setattr("squidmip.project_plate",
                        lambda reader, **kw: (seen.update(kw), iter(()))[1])
    regions = open_bus.execute(Describe()).data["regions"][:1]
    open_bus.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE, regions=regions))
    assert seen["regions"] == regions


def test_an_empty_region_list_is_refused_and_never_widened_to_everything(open_bus):
    """Running 1536 wells because a caller sent ``[]`` is hours of compute nobody asked for."""
    r = open_bus.execute(RunOperator(operator="mip", regions=[]))
    assert r.refusal == EMPTY_SCOPE


def test_a_region_that_is_not_in_this_acquisition_is_refused_by_name(open_bus):
    r = open_bus.execute(RunOperator(operator="mip", regions=["ZZ99"]))
    assert r.refusal == UNKNOWN_REGION and "ZZ99" in str(r.data["unknown"])


def test_an_invented_scope_is_refused_and_lists_the_real_ones(open_bus):
    r = open_bus.execute(RunOperator(operator="mip", scope="everything ever"))
    assert r.refusal == BAD_SCOPE
    for scope in _explore.RUN_SCOPES:
        assert scope in r.message


def test_the_selection_drives_the_selected_wells_scope(open_bus, monkeypatch):
    """The headless surface resolves 'selected wells' through the SAME
    ``_explore.resolve_run_scope`` the GUI does — there is not a second resolver."""
    seen = {}
    monkeypatch.setattr("squidmip.project_plate",
                        lambda reader, **kw: (seen.update(kw), iter(()))[1])
    regions = open_bus.execute(Describe()).data["regions"][:1]
    open_bus.executor.selection = list(regions)
    open_bus.execute(RunOperator(operator="mip", scope=_explore.SCOPE_SELECTION))
    assert seen["regions"] == regions


def test_an_unknown_operator_is_refused_before_any_work_and_lists_what_can_run(open_bus):
    r = open_bus.execute(RunOperator(operator="minerva"))
    assert r.refusal == UNKNOWN_OPERATOR
    assert "mip" in r.message and "minerva" in r.message


def test_running_with_nothing_open_is_refused_by_name(bus):
    assert bus.execute(RunOperator(operator="mip")).refusal == NO_ACQUISITION


def test_saving_headless_without_an_output_folder_is_refused_not_guessed(open_bus):
    r = open_bus.execute(RunOperator(operator="mip", save=True))
    assert r.refusal == BAD_COMMAND and "output_folder" in r.message


# --- the run actually runs, and is measured -----------------------------------------------------

def test_a_preview_run_computes_every_well_and_writes_nothing(open_bus, tmp_path):
    before = set(p.name for p in tmp_path.iterdir())
    r = open_bus.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE))
    assert r.ok and r.status == "completed"
    assert r.data["n_landed"] >= 1
    assert set(p.name for p in tmp_path.iterdir()) == before


def test_a_saved_run_returns_the_manifest(open_bus, tmp_path):
    r = open_bus.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE, save=True,
                                     output_folder=str(tmp_path), n_fovs=1))
    assert r.ok, r.message
    assert r.data["manifest"]["n_fields_written"] >= 1
    assert (tmp_path / "acq.hcs").is_dir()


def test_every_run_is_measured_and_the_result_carries_the_metrics(open_bus):
    r = open_bus.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE))
    m = r.data["metrics"]
    assert m["operator"] == "mip" and m["seconds"] >= 0
    assert m["outcome"] in ("ok", "partial")
    assert m["target"], "a duration with no target named is not comparable to anything"


def test_the_result_names_the_target_set_it_resolved(open_bus):
    r = open_bus.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE))
    assert "whole dataset" in r.data["target"]


def test_a_run_that_produced_nothing_is_partial_not_ok(open_bus, monkeypatch):
    """A run where every well raised still returns politely — the per-well fault isolation is what
    keeps one bad file from aborting a plate. It is not a success."""
    monkeypatch.setattr("squidmip.project_plate", lambda reader, **kw: iter(()))
    r = open_bus.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE))
    assert r.data["metrics"]["outcome"] == "partial"
    assert "produced nothing" in r.data["metrics"]["detail"]


def test_metrics_returns_the_comparison_table(open_bus):
    open_bus.execute(RunOperator(operator="mip", scope=_explore.SCOPE_PLATE))
    r = open_bus.execute(Metrics(operator="mip"))
    assert r.ok and r.data["table"]
    assert r.data["table"][0]["operator"] == "mip"
    assert all(run["operator"] == "mip" for run in r.data["runs"])


def test_a_result_is_serialisable_so_an_agent_can_read_it(open_bus):
    r = open_bus.execute(ListOperators())
    json.dumps(r.model_dump())


def test_a_refusal_always_carries_a_code_and_a_sentence(bus):
    from squidmip._command import REFUSALS

    for payload in ({"kind": "nope"}, Describe(), StopRun(), RunOperator(operator="mip")):
        r = bus.execute(payload)
        if r.status == "refused":
            assert r.refusal in REFUSALS, r
            assert r.message.strip(), "a refusal with no sentence is a button that did nothing"
            assert r.ok is False
