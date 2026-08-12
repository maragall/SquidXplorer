"""Operator composition: parsing, derived declarations, refusals, execution, writing, stitching."""

from __future__ import annotations

import numpy as np
import pytest

import squidxplorer
from squidxplorer._compose import compose_operator
from squidxplorer._engine import _OPERATORS, _resolve_operator, add_operator, bind_operator
from squidxplorer._recipe import Recipe, RecipeChain
from squidxplorer.projection import PLANE_OP, Z_REDUCER, plane_op

# Two plane-ops that do NOT commute: add-then-double is 2p+20, double-then-add is 2p+10.
ADD = "compose_add10"
DOUBLE = "compose_double"
COUNT = "compose_zcount"


def _add10(plane):
    return (plane.astype(np.int32) + 10).astype(plane.dtype)


def _double(plane):
    return (plane.astype(np.int32) * 2).astype(plane.dtype)


def _count(planes):
    """A z-reducer that returns HOW MANY planes it was handed."""
    planes = list(planes)
    return np.full_like(planes[0], len(planes))


_count.consumes = Z_REDUCER


@pytest.fixture(autouse=True)
def _register_test_operators():
    """Registered per test: ``add_operator`` writes to a process-global table."""
    add_operator(ADD, plane_op(_add10))
    add_operator(DOUBLE, plane_op(_double))
    add_operator(COUNT, _count)
    try:
        yield
    finally:
        for name in (ADD, DOUBLE, COUNT):
            _OPERATORS.pop(name, None)


def _planes(values, shape=(4, 4)):
    return [np.full(shape, v, np.uint16) for v in values]


# =================================================================================================
# 1. THE EXPRESSION IS THE LABEL
# =================================================================================================


def test_a_chain_is_written_the_way_it_is_printed():
    """One spelling for the label, the cache key, the paste script and the run."""
    chain = RecipeChain.of(Recipe.operator("flatfield"), Recipe.operator("decon"),
                           Recipe.operator("spot", min_area_px=80))

    assert chain.label() == "flatfield + decon + spot(min_area_px=80)"
    assert RecipeChain.parse(chain.label()) == chain
    assert _resolve_operator(chain.label()).name == chain.label()


def test_the_same_chain_resolves_whether_it_is_a_string_or_a_RecipeChain():
    chain = RecipeChain.parse(f"{ADD}+{DOUBLE}")

    assert _resolve_operator(chain).name == _resolve_operator(f"{ADD}+{DOUBLE}").name


# =================================================================================================
# 2. A COMPOSITION DECLARES WHAT ITS PARTS DECLARE
# =================================================================================================


def test_consumes_is_the_union_so_the_engine_loop_and_the_output_shape_follow():
    assert squidxplorer.operator_consumes(f"{ADD}+{DOUBLE}") == PLANE_OP
    assert squidxplorer.operator_consumes(f"{ADD}+{DOUBLE}+mip") == Z_REDUCER


def test_produces_is_the_last_steps_so_the_viewer_picks_the_right_layer_type():
    assert squidxplorer.operator_produces(f"{ADD}+{DOUBLE}") == "intensity"
    assert squidxplorer.operator_produces(f"{ADD}+spot") == "labels"


def test_params_are_namespaced_per_step_and_reach_that_step():
    declared = {p.name: p.default for p in squidxplorer.operator_params(f"{ADD}+spot")}
    assert declared["spot.min_area_px"] == 30

    # a value written INTO the expression becomes that parameter's default...
    assert {p.name: p.default for p in squidxplorer.operator_params(
        f"{ADD}+spot(min_area_px=99)")}["spot.min_area_px"] == 99
    # ...and operator_kwargs reaches the same place, by the same name.
    assert bind_operator(f"{ADD}+spot", {"spot.min_area_px": 77}).__name__ == \
        f"{ADD} + spot(min_area_px=77)"


def test_a_namespaced_parameter_changes_what_that_step_actually_does():
    """The name check above proves the chain was REBUILT; this proves the value ARRIVED."""
    plane = np.zeros((64, 64), np.uint16)
    plane[10:14, 10:14] = 5000        # 16 px
    plane[30:55, 30:55] = 5000        # 625 px

    big_only = bind_operator(f"{ADD}+spot", {"spot.min_area_px": 500})([plane])
    both = bind_operator(f"{ADD}+spot", {"spot.min_area_px": 4})([plane])

    assert sorted(np.unique(big_only).tolist()) == [0, 1]
    assert sorted(np.unique(both).tolist()) == [0, 1, 2]


def test_an_unknown_parameter_is_refused_naming_what_the_chain_does_accept():
    with pytest.raises(ValueError, match=r"has no parameter 'spot.nope'"):
        bind_operator(f"{ADD}+spot", {"spot.nope": 1})


def test_requires_is_the_union_so_one_refusal_names_every_missing_package():
    assert squidxplorer.operator_requires("flatfield+decon+mip") == ("tilefusion", "petakit")


def test_a_bare_name_is_still_the_registry_entry_itself():
    """Identity, not equality: nothing routes through composition unless a chain was asked for."""
    assert _resolve_operator("mip") is _OPERATORS["mip"]
    assert _resolve_operator("reference") is _OPERATORS["reference"]


# =================================================================================================
# 3. WHAT IS REFUSED, AND HOW LOUDLY
# =================================================================================================


def test_a_z_reducer_that_is_not_last_is_refused_naming_both_operators_and_the_fix():
    with pytest.raises(ValueError) as exc:
        _resolve_operator(f"mip+{ADD}")

    message = str(exc.value)
    assert "'mip'" in message and f"'{ADD}'" in message
    assert "consumes z" in message
    assert f"'{ADD} + mip'" in message, "the refusal must name the chain that WOULD work"


def test_two_z_reducers_are_refused_by_the_same_rule_and_not_by_name():
    with pytest.raises(ValueError, match="consumes z"):
        _resolve_operator("mip+mip")
    with pytest.raises(ValueError, match="consumes z"):
        _resolve_operator(f"{COUNT}+mip")


def test_a_labels_step_that_is_not_last_is_refused():
    with pytest.raises(ValueError) as exc:
        _resolve_operator(f"spot+{ADD}")

    assert "labels" in str(exc.value) and "does not exist" in str(exc.value)


def test_a_z_selecting_step_is_refused_inside_a_chain():
    with pytest.raises(ValueError) as exc:
        _resolve_operator(f"{ADD}+reference")

    assert "SELECTS one z" in str(exc.value)


def test_a_repeated_step_is_refused_because_its_parameters_would_be_ambiguous():
    with pytest.raises(ValueError, match="appears twice"):
        _resolve_operator(f"{ADD}+{ADD}")


def test_an_unknown_step_reads_exactly_like_an_unknown_name():
    with pytest.raises(KeyError, match="unknown operator 'noplease'"):
        _resolve_operator(f"{ADD}+noplease")


def test_a_region_operator_cannot_be_a_step_of_a_chain():
    with pytest.raises(ValueError, match="consumes fov"):
        _resolve_operator("stitch + mip")


def test_a_list_of_names_is_refused_by_type_naming_the_chain_spelling():
    with pytest.raises(TypeError) as exc:
        _resolve_operator(["flatfield", "mip"])

    assert "operator='flatfield + mip'" in str(exc.value)


def test_a_registered_name_may_not_contain_chain_punctuation():
    with pytest.raises(ValueError, match="chain punctuation"):
        add_operator("a+b", plane_op(_add10))


def test_an_empty_chain_names_no_operator():
    with pytest.raises(ValueError, match="names no operator"):
        compose_operator("raw", _resolve_operator)


# =================================================================================================
# 4. IT RUNS, IN ORDER, WITHOUT MATERIALISING THE STACK
# =================================================================================================


def test_the_steps_run_left_to_right():
    add_then_double = _resolve_operator(f"{ADD}+{DOUBLE}").fn
    double_then_add = _resolve_operator(f"{DOUBLE}+{ADD}").fn

    assert add_then_double(_planes([5]))[0, 0] == (5 + 10) * 2
    assert double_then_add(_planes([5]))[0, 0] == 5 * 2 + 10


def test_a_plane_op_prefix_feeds_the_z_reducer_the_whole_stack():
    assert _resolve_operator(f"{ADD}+{COUNT}").fn(_planes([1, 2, 3]))[0, 0] == 3
    assert _resolve_operator(f"{ADD}+{DOUBLE}+mip").fn(_planes([1, 5, 3]))[0, 0] == (5 + 10) * 2


def test_the_plane_ops_are_applied_lazily_as_the_reducer_pulls():
    """Events must interleave — read, transform, read, transform — never all reads then all transforms."""
    events: list = []

    def _traced(plane):
        events.append("op")
        return plane

    add_operator("compose_traced", plane_op(_traced))
    try:
        def _reading():
            for value in (1, 2, 3):
                events.append("read")
                yield np.full((4, 4), value, np.uint16)

        _resolve_operator("compose_traced+mip").fn(_reading())
    finally:
        _OPERATORS.pop("compose_traced", None)

    assert events == ["read", "op", "read", "op", "read", "op"]


def test_a_plane_op_chain_handed_a_stack_refuses_rather_than_dropping_planes():
    with pytest.raises(ValueError, match="more than one plane"):
        _resolve_operator(f"{ADD}+{DOUBLE}").fn(_planes([1, 2]))


def test_a_chain_runs_over_a_real_acquisition_through_run_plate(squid_dataset):
    root, _arrays = squid_dataset
    reader = squidxplorer.open_reader(root)
    n_z = len(reader.metadata["z_levels"])
    n_c = len(reader.metadata["channels"])

    planes = dict(((region, fov), image) for region, fov, image
                  in squidxplorer.run_plate(reader, n_fovs=1, workers=1,
                                            operator=f"{ADD}+{DOUBLE}"))
    reduced = dict(((region, fov), image) for region, fov, image
                   in squidxplorer.run_plate(reader, n_fovs=1, workers=1,
                                             operator=f"{ADD}+{DOUBLE}+mip"))

    assert planes and set(planes) == set(reduced)
    for key, image in planes.items():
        assert image.shape[1:3] == (n_c, n_z), image.shape
        assert reduced[key].shape[1:3] == (n_c, 1), reduced[key].shape
        # the reduction of the chain IS the chain's own maximum over z, pixel for pixel
        assert np.array_equal(reduced[key][:, :, 0], image.max(axis=2))


# =================================================================================================
# 5. IT WRITES, AND IT STITCHES
# =================================================================================================


def test_a_composed_chain_reaches_write_plate_and_the_store_keeps_its_depth(squid_dataset, tmp_path):
    root, _arrays = squid_dataset
    reader = squidxplorer.open_reader(root)
    n_z = len(reader.metadata["z_levels"])

    manifest = squidxplorer.write_plate(reader, tmp_path / "out", n_fovs=1, workers=1,
                                    operator=f"{ADD}+{DOUBLE}", check_disk=False)

    written = squidxplorer.open_reader(manifest["plate"])
    assert len(written.metadata["z_levels"]) == n_z
    # and the pixels are the chain's, not one step's
    plain = squidxplorer.open_reader(root)
    region, fov = written.metadata["regions"][0], 0
    raw = plain.read(region, fov, plain.metadata["channels"][0]["name"], 0, 0).astype(np.int32)
    got = written.read(region, fov, written.metadata["channels"][0]["name"], 0, 0)
    assert np.array_equal(got, ((raw + 10) * 2).astype(got.dtype))


def _stitch_fixtures():
    pytest.importorskip("tilefusion", reason="tilefusion (maragall/stitcher) not installed: the "
                                             "stitch adapter is UNTESTED here, not passing")
    from tests.test_stitch_zplanes import N_Z, _ZReader
    from tests.test_stitch import GRID, _master

    return _ZReader(_master()), list(range(GRID * GRID)), N_Z


def test_a_composed_plane_op_chain_stitches_every_z_plane():
    reader, fovs, n_z = _stitch_fixtures()
    from squidxplorer._stitch import stitch_region

    out = np.asarray(stitch_region(reader, "A1", fovs, z_operator=f"{ADD}+{DOUBLE}",
                                   register=False, correct_illumination=False))
    reduced = np.asarray(stitch_region(reader, "A1", fovs, z_operator=f"{ADD}+{DOUBLE}+mip",
                                       register=False, correct_illumination=False))

    assert out.shape[2] == n_z, out.shape
    assert reduced.shape[2] == 1, reduced.shape
    assert np.array_equal(reduced[:, :, 0], out.max(axis=2))


def test_a_chain_containing_flatfield_does_not_defeat_the_double_apply_guard():
    """The guard reads ``corrects_illumination`` off the callable, so a composition must declare it."""
    reader, fovs, _n_z = _stitch_fixtures()
    from squidxplorer._stitch import stitch_region

    for chain in ("flatfield+mip", f"{ADD}+flatfield", f"flatfield+{ADD}+mip"):
        assert getattr(_resolve_operator(chain).fn, "corrects_illumination", False), chain
        with pytest.raises(ValueError, match="flat-field corrects its input"):
            stitch_region(reader, "A1", fovs, z_operator=chain, register=False)

    # ...and a chain of operators that do NOT correct is untouched by the guard.
    assert not getattr(_resolve_operator(f"{ADD}+mip").fn, "corrects_illumination", False)


def test_a_chain_ending_in_labels_is_refused_by_stitching_for_the_reason_labels_are():
    reader, fovs, _n_z = _stitch_fixtures()
    from squidxplorer._stitch import stitch_region

    with pytest.raises(ValueError, match="label images"):
        stitch_region(reader, "A1", fovs, z_operator=f"{ADD}+spot", register=False,
                      correct_illumination=False)


# =================================================================================================
# 6. THE OUTSIDE SURFACES TAKE A CHAIN WHEREVER THEY TAKE A NAME
# =================================================================================================


def test_the_command_surface_runs_a_chain_and_explains_an_impossible_one(squid_dataset):
    from squidxplorer._command import CommandBus, EngineExecutor, OpenAcquisition, RunOperator

    root, _arrays = squid_dataset
    executor = CommandBus(EngineExecutor())
    assert executor.execute(OpenAcquisition(path=str(root))).ok

    ok = executor.execute(RunOperator(operator=f"{ADD}+{DOUBLE}+mip", n_fovs=1, workers=1))
    assert ok.status == "completed", ok.message

    impossible = executor.execute(RunOperator(operator=f"mip+{ADD}", n_fovs=1, workers=1))
    assert impossible.status == "refused"
    assert "consumes z" in impossible.message

    unknown = executor.execute(RunOperator(operator="noplease", n_fovs=1, workers=1))
    assert unknown.status == "refused" and "is not a runnable operator" in unknown.message


def test_the_cli_validates_a_chain_before_it_writes_anything(tmp_path):
    import pydantic

    from squidxplorer._cli import ProcessParameters

    (tmp_path / "acq").mkdir()
    common = dict(input_folder=str(tmp_path / "acq"), output_folder=str(tmp_path / "out"))

    assert ProcessParameters(operator=f"{ADD}+mip", **common).operator == f"{ADD}+mip"
    with pytest.raises(pydantic.ValidationError, match="consumes z"):
        ProcessParameters(operator=f"mip+{ADD}", **common)
    with pytest.raises(pydantic.ValidationError, match="unknown operator"):
        ProcessParameters(operator="noplease", **common)

    # ...and --param reaches ONE step of a chain, by the namespaced name the chain declares.
    assert ProcessParameters(operator=f"{ADD}+spot", param=["spot.min_area_px=80"],
                             **common).parameters() == {"spot.min_area_px": 80}
    with pytest.raises(pydantic.ValidationError, match="does not take spot.nope"):
        ProcessParameters(operator=f"{ADD}+spot", param=["spot.nope=1"], **common)


def test_operator_available_answers_for_a_chain_rather_than_calling_it_unknown():
    ok, why = squidxplorer.operator_available(f"{ADD}+{DOUBLE}")
    assert ok and why == ""

    ok, why = squidxplorer.operator_available(f"mip+{ADD}")
    assert not ok and "consumes z" in why
