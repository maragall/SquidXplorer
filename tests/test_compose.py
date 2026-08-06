"""Operator COMPOSITION: ``flatfield + decon + mip`` runs, writes and stitches.

What this file is for, stated as the failures it would catch:

  * "the chain parsed and nothing ran it"      — the state this feature was in for two weeks;
  * "the steps ran in the wrong order"         — a chain is an ordered thing or it is nothing;
  * "the composition materialised the stack"   — the memory bound is the whole reason the shipped
                                                 reductions stream, and a chain must not spend it;
  * "a chain defeated the flat-field guard"    — measured at 88.6% of pixels wrong, silently;
  * "an impossible chain ran anyway"           — a z-reducer that is not last, a labels step that
                                                 is not last, a z-SELECTING step inside a chain;
  * "the composition is a special case"        — it must reach `write_plate` and `stitch_region`
                                                 through the same `projector=` every name uses.

The refusals are asserted on their MESSAGES, not merely on the exception type: a chain is typed by
a human into a CLI flag or a command, so "which two operators, and what to write instead" is the
part that has to survive.
"""

from __future__ import annotations

import numpy as np
import pytest

import squidmip
from squidmip._compose import compose_operator
from squidmip._engine import _PROJECTORS, _resolve_projector, add_projector, bind_projector
from squidmip._recipe import Recipe, RecipeChain
from squidmip.projection import PLANE_OP, Z_REDUCER, plane_op

# --- two plane-ops that do NOT commute, so order is a numeric fact and not a comment -------------
#
# `bgsub` (the obvious real plane-op) subtracts a flat fixture away to zeros, and a test comparing
# all-zero planes proves nothing. These two do: add-then-double is 2p+20 and double-then-add is
# 2p+10, so a silently reordered chain is a wrong NUMBER rather than a wrong shape.
ADD = "compose_add10"
DOUBLE = "compose_double"
COUNT = "compose_zcount"


def _add10(plane):
    return (plane.astype(np.int32) + 10).astype(plane.dtype)


def _double(plane):
    return (plane.astype(np.int32) * 2).astype(plane.dtype)


def _count(planes):
    """A z-reducer that returns HOW MANY planes it was handed, so a chain's grouping is testable."""
    planes = list(planes)
    return np.full_like(planes[0], len(planes))


_count.consumes = Z_REDUCER


@pytest.fixture(autouse=True)
def _register_test_operators():
    """Registered per test. ``add_projector`` writes to a PROCESS-GLOBAL table and
    ``tests/test_operator_integration.py`` asserts its exact contents, so an import-time
    registration here would fail a different file than the one that caused it."""
    add_projector(ADD, plane_op(_add10))
    add_projector(DOUBLE, plane_op(_double))
    add_projector(COUNT, _count)
    try:
        yield
    finally:
        for name in (ADD, DOUBLE, COUNT):
            _PROJECTORS.pop(name, None)


def _planes(values, shape=(4, 4)):
    return [np.full(shape, v, np.uint16) for v in values]


# =================================================================================================
# 1. THE EXPRESSION IS THE LABEL
# =================================================================================================


def test_a_chain_is_written_the_way_it_is_printed():
    """One spelling for the label, the cache key, the paste script and the run.

    A second syntax for "the same chain, but runnable" would put a legend row and a working command
    one transcription error apart, which is exactly the drift the naming law exists to stop.
    """
    chain = RecipeChain.of(Recipe.operator("flatfield"), Recipe.operator("decon"),
                           Recipe.operator("spot", min_area_px=80))

    assert chain.label() == "flatfield + decon + spot(min_area_px=80)"
    assert RecipeChain.parse(chain.label()) == chain
    assert _resolve_projector(chain.label()).name == chain.label()


def test_the_same_chain_resolves_whether_it_is_a_string_or_a_RecipeChain():
    """A caller holding the STRUCTURE (a pasted recipe script, ``paste_chain()``) must not have to
    render it to a string and have the engine parse it straight back."""
    chain = RecipeChain.parse(f"{ADD}+{DOUBLE}")

    assert _resolve_projector(chain).name == _resolve_projector(f"{ADD}+{DOUBLE}").name


# =================================================================================================
# 2. A COMPOSITION DECLARES WHAT ITS PARTS DECLARE
# =================================================================================================


def test_consumes_is_the_union_so_the_engine_loop_and_the_output_shape_follow():
    """``consumes`` decides the loop and the output shape. A chain that did not derive it would
    make the engine group over the wrong axis, which is a silently reshaped result."""
    assert squidmip.projector_consumes(f"{ADD}+{DOUBLE}") == PLANE_OP
    assert squidmip.projector_consumes(f"{ADD}+{DOUBLE}+mip") == Z_REDUCER


def test_produces_is_the_last_steps_so_the_viewer_picks_the_right_layer_type():
    """``produces`` decides the napari layer type. Only the LAST step can be non-intensity (see the
    refusal below), so the last step's answer is the chain's answer."""
    assert squidmip.projector_produces(f"{ADD}+{DOUBLE}") == "intensity"
    assert squidmip.projector_produces(f"{ADD}+spot") == "labels"


def test_params_are_namespaced_per_step_and_reach_that_step():
    """One entry, several steps, so a parameter has to say WHICH step it is for.

    ``spot.min_area_px`` rather than ``min_area_px``: two steps of one chain can declare the same
    parameter name, and a flat namespace would set both from one value with nothing saying so.
    """
    declared = {p.name: p.default for p in squidmip.projector_params(f"{ADD}+spot")}
    assert declared["spot.min_area_px"] == 30

    # a value written INTO the expression becomes that parameter's default...
    assert {p.name: p.default for p in squidmip.projector_params(
        f"{ADD}+spot(min_area_px=99)")}["spot.min_area_px"] == 99
    # ...and operator_kwargs reaches the same place, by the same name.
    assert bind_projector(f"{ADD}+spot", {"spot.min_area_px": 77}).__name__ == \
        f"{ADD} + spot(min_area_px=77)"


def test_an_unknown_parameter_is_refused_naming_what_the_chain_does_accept():
    with pytest.raises(ValueError, match=r"has no parameter 'spot.nope'"):
        bind_projector(f"{ADD}+spot", {"spot.nope": 1})


def test_requires_is_the_union_so_one_refusal_names_every_missing_package():
    """``requires`` is checked at BIND time, before a single well is read. A chain that dropped a
    step's requirement would reproduce the measured silent success it exists to end: an ImportError
    one call deep, filed by ``on_error`` as a per-well skip, and a green run that wrote nothing."""
    assert squidmip.projector_requires("flatfield+decon+mip") == ("tilefusion", "petakit")


def test_a_bare_name_is_still_the_registry_entry_itself():
    """Nothing routes through composition unless a chain was asked for.

    Identity and not equality: ``projector="mip"`` must be the exact object the table has held since
    IMA-188, so no existing run changes by a pixel — and so ``reference``'s ``select_index``, which
    a composition refuses to carry, is still reachable on its own.
    """
    assert _resolve_projector("mip") is _PROJECTORS["mip"]
    assert _resolve_projector("reference") is _PROJECTORS["reference"]


# =================================================================================================
# 3. WHAT IS REFUSED, AND HOW LOUDLY
# =================================================================================================


def test_a_z_reducer_that_is_not_last_is_refused_naming_both_operators_and_the_fix():
    """``consumes`` decides it: after a reducer there is one plane and no stack, so the next step is
    not "mapped over z" in any sense the declaration can express. Never silently reordered — a run
    that quietly ran ``decon+mip`` when the user typed ``mip+decon`` is a wrong result that looks
    right."""
    with pytest.raises(ValueError) as exc:
        _resolve_projector(f"mip+{ADD}")

    message = str(exc.value)
    assert "'mip'" in message and f"'{ADD}'" in message
    assert "consumes z" in message
    assert f"'{ADD} + mip'" in message, "the refusal must name the chain that WOULD work"


def test_two_z_reducers_are_refused_by_the_same_rule_and_not_by_name():
    with pytest.raises(ValueError, match="consumes z"):
        _resolve_projector("mip+mip")
    with pytest.raises(ValueError, match="consumes z"):
        _resolve_projector(f"{COUNT}+mip")


def test_a_labels_step_that_is_not_last_is_refused():
    """``produces="labels"`` means the pixels are integer OBJECT IDS. A following step would do
    arithmetic on names: the mean of label 12 and label 37 is label 24, an object that does not
    exist. Same argument ``stitch_region`` already makes when it refuses to feather labels."""
    with pytest.raises(ValueError) as exc:
        _resolve_projector(f"spot+{ADD}")

    assert "labels" in str(exc.value) and "does not exist" in str(exc.value)


def test_a_z_selecting_step_is_refused_inside_a_chain():
    """Not a shape problem. ``project_well`` solves ``reference``'s focus ONCE per (t, fov) on RAW
    planes and shares that z across channels, OUTSIDE the operator — so a chain around it would
    never touch the planes it picks, and every other step would be silently dropped."""
    with pytest.raises(ValueError) as exc:
        _resolve_projector(f"{ADD}+reference")

    assert "SELECTS one z" in str(exc.value)


def test_a_repeated_step_is_refused_because_its_parameters_would_be_ambiguous():
    with pytest.raises(ValueError, match="appears twice"):
        _resolve_projector(f"{ADD}+{ADD}")


def test_an_unknown_step_reads_exactly_like_an_unknown_name():
    """A typo inside a chain is a typo. It must not turn into a parser error about punctuation."""
    with pytest.raises(KeyError, match="unknown projector 'noplease'"):
        _resolve_projector(f"{ADD}+noplease")


def test_a_list_of_names_is_refused_by_type_naming_the_chain_spelling():
    """The shape people actually reached for. It used to surface as ``TypeError: unhashable type:
    'list'`` from a dict lookup — an error about a dict, raised at a caller asking about
    operators."""
    with pytest.raises(TypeError) as exc:
        _resolve_projector(["flatfield", "mip"])

    assert "projector='flatfield + mip'" in str(exc.value)


def test_a_registered_name_may_not_contain_chain_punctuation():
    with pytest.raises(ValueError, match="chain punctuation"):
        add_projector("a+b", plane_op(_add10))


def test_an_empty_chain_names_no_operator():
    with pytest.raises(ValueError, match="names no operator"):
        compose_operator("raw", _resolve_projector)


# =================================================================================================
# 4. IT RUNS, IN ORDER, WITHOUT MATERIALISING THE STACK
# =================================================================================================


def test_the_steps_run_left_to_right():
    """2p+20, not 2p+10. Order is the whole content of a chain."""
    add_then_double = _resolve_projector(f"{ADD}+{DOUBLE}").fn
    double_then_add = _resolve_projector(f"{DOUBLE}+{ADD}").fn

    assert add_then_double(_planes([5]))[0, 0] == (5 + 10) * 2
    assert double_then_add(_planes([5]))[0, 0] == 5 * 2 + 10


def test_a_plane_op_prefix_feeds_the_z_reducer_the_whole_stack():
    """plane-op -> z-reducer: the reducer consumes what the plane-ops produced, at full depth. The
    counting reducer makes "it only saw one plane" a numeric failure rather than an invisible one."""
    assert _resolve_projector(f"{ADD}+{COUNT}").fn(_planes([1, 2, 3]))[0, 0] == 3
    assert _resolve_projector(f"{ADD}+{DOUBLE}+mip").fn(_planes([1, 5, 3]))[0, 0] == (5 + 10) * 2


def test_the_plane_ops_are_applied_lazily_as_the_reducer_pulls():
    """The memory bound, asserted rather than assumed.

    ``project`` streams z one plane at a time and keeps one accumulator; a composition that mapped
    its plane-ops eagerly would hold the whole transformed stack and spend exactly the property the
    shipped reductions were written to have. So the events must INTERLEAVE — read, transform, read,
    transform — never every read and then every transform.
    """
    events: list = []

    def _traced(plane):
        events.append("op")
        return plane

    add_projector("compose_traced", plane_op(_traced))
    try:
        def _reading():
            for value in (1, 2, 3):
                events.append("read")
                yield np.full((4, 4), value, np.uint16)

        _resolve_projector("compose_traced+mip").fn(_reading())
    finally:
        _PROJECTORS.pop("compose_traced", None)

    assert events == ["read", "op", "read", "op", "read", "op"]


def test_a_plane_op_chain_handed_a_stack_refuses_rather_than_dropping_planes():
    """A seam bug, not a data fault: it can only happen if the chain was run with the wrong
    ``consumes``. Silently using the first plane is how "my background subtraction dropped all but
    one z" happens."""
    with pytest.raises(ValueError, match="more than one plane"):
        _resolve_projector(f"{ADD}+{DOUBLE}").fn(_planes([1, 2]))


def test_a_chain_runs_over_a_real_acquisition_through_project_plate(squid_dataset):
    """End to end through the engine's own loop, with no argument the engine did not already have.

    The plane-op chain keeps z at full depth and the z-reducing chain collapses it to 1, both from
    the chain's own derived ``consumes`` — which is the same dispatch a single operator gets.
    """
    root, _arrays = squid_dataset
    reader = squidmip.open_reader(root)
    n_z = len(reader.metadata["z_levels"])
    n_c = len(reader.metadata["channels"])

    planes = dict(((region, fov), image) for region, fov, image
                  in squidmip.project_plate(reader, n_fovs=1, workers=1,
                                            projector=f"{ADD}+{DOUBLE}"))
    reduced = dict(((region, fov), image) for region, fov, image
                   in squidmip.project_plate(reader, n_fovs=1, workers=1,
                                             projector=f"{ADD}+{DOUBLE}+mip"))

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
    """A composition is only real if it can be SAVED. Per-plane fusion (IMA-277) taught
    ``_validate_image`` to accept Nz>1; this asserts a composed plane-op chain gets through it and
    lands with the acquisition's full depth rather than a silently flattened one."""
    root, _arrays = squid_dataset
    reader = squidmip.open_reader(root)
    n_z = len(reader.metadata["z_levels"])

    manifest = squidmip.write_plate(reader, tmp_path / "out", n_fovs=1, workers=1,
                                    projector=f"{ADD}+{DOUBLE}", check_disk=False)

    written = squidmip.open_reader(manifest["plate"])
    assert len(written.metadata["z_levels"]) == n_z
    # and the pixels are the chain's, not one step's
    plain = squidmip.open_reader(root)
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
    """``stitch_region`` reads the SAME ``consumes`` declaration to size its z-outer loop, so a
    composition needs no edit there — and a chain that got its ``consumes`` wrong would show up as
    a mosaic with the wrong depth."""
    reader, fovs, n_z = _stitch_fixtures()
    from squidmip._stitch import stitch_region

    out = np.asarray(stitch_region(reader, "A1", fovs, projector=f"{ADD}+{DOUBLE}",
                                   register=False, correct_illumination=False))
    reduced = np.asarray(stitch_region(reader, "A1", fovs, projector=f"{ADD}+{DOUBLE}+mip",
                                       register=False, correct_illumination=False))

    assert out.shape[2] == n_z, out.shape
    assert reduced.shape[2] == 1, reduced.shape
    assert np.array_equal(reduced[:, :, 0], out.max(axis=2))


def test_a_chain_containing_flatfield_does_not_defeat_the_double_apply_guard():
    """THE regression this feature could have introduced.

    ``stitch_region`` refuses "the read path corrects AND the operator corrects" by reading
    ``corrects_illumination`` off the operator's CALLABLE — a declaration, deliberately not a
    ``== "flatfield"`` test, because this package hands out flat-field operators under names it does
    not choose. A composition is one more such name. Correcting twice changes 88.6% of pixels by up
    to 23 counts and nothing downstream can tell, so the composed callable declares what its parts
    declare, and this asserts it on the guard itself.
    """
    reader, fovs, _n_z = _stitch_fixtures()
    from squidmip._stitch import stitch_region

    for chain in ("flatfield+mip", f"{ADD}+flatfield", f"flatfield+{ADD}+mip"):
        assert getattr(_resolve_projector(chain).fn, "corrects_illumination", False), chain
        with pytest.raises(ValueError, match="flat-field corrects its input"):
            stitch_region(reader, "A1", fovs, projector=chain, register=False)

    # ...and a chain of operators that do NOT correct is untouched by the guard.
    assert not getattr(_resolve_projector(f"{ADD}+mip").fn, "corrects_illumination", False)


def test_a_chain_ending_in_labels_is_refused_by_stitching_for_the_reason_labels_are():
    """The composition inherits the last step's ``produces``, so the fusion refusal that already
    exists finds it — without ``_stitch`` learning that compositions are a thing."""
    reader, fovs, _n_z = _stitch_fixtures()
    from squidmip._stitch import stitch_region

    with pytest.raises(ValueError, match="label images"):
        stitch_region(reader, "A1", fovs, projector=f"{ADD}+spot", register=False,
                      correct_illumination=False)


# =================================================================================================
# 6. THE OUTSIDE SURFACES TAKE A CHAIN WHEREVER THEY TAKE A NAME
# =================================================================================================


def test_the_command_surface_runs_a_chain_and_explains_an_impossible_one(squid_dataset):
    """``run_operator`` refused everything not in the registry, which is every chain. It now asks
    the engine to resolve, so the two refusals stay DISTINCT: a name nobody registered lists what
    exists, an unrunnable chain of real operators carries ``_compose``'s own reason."""
    from squidmip._command import CommandBus, EngineExecutor, OpenAcquisition, RunOperator

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
    """The CLI resolved ``--projector`` against ``available_projectors()``, so every chain was
    rejected at the flag while the same string ran fine from Python. Validation stays UP FRONT: the
    alternative is a plate skeleton on disk and then a traceback."""
    import pydantic

    from squidmip._cli import ProcessParameters

    (tmp_path / "acq").mkdir()
    common = dict(input_folder=str(tmp_path / "acq"), output_folder=str(tmp_path / "out"))

    assert ProcessParameters(projector=f"{ADD}+mip", **common).projector == f"{ADD}+mip"
    with pytest.raises(pydantic.ValidationError, match="consumes z"):
        ProcessParameters(projector=f"mip+{ADD}", **common)
    with pytest.raises(pydantic.ValidationError, match="unknown projector"):
        ProcessParameters(projector="noplease", **common)


def test_operator_available_answers_for_a_chain_rather_than_calling_it_unknown():
    """A caller offering a row (``list_operators``, the viewer) has ONE place to put the reason. A
    membership test against the table reported every chain as unknown, which is the same answer it
    gives a typo."""
    ok, why = squidmip.operator_available(f"{ADD}+{DOUBLE}")
    assert ok and why == ""

    ok, why = squidmip.operator_available(f"mip+{ADD}")
    assert not ok and "consumes z" in why
