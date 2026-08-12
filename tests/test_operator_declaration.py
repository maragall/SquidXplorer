"""What an operator DECLARES about itself, and whether the app reads the declaration or the name.

Two properties are pinned here, and both are properties of the SYSTEM rather than of any one
operator — which is the point. Cellpose was the operator that made them visible, and neither test
mentions it by name.

1. **Every registered operator delivers the result kind it declares.** Parametrized over
   ``available_projectors()``, so an operator added tomorrow is validated by a test written today.
   That is the difference between a conformance test and a test per feature.

2. **No module branches on an operator's name.** ``consumes`` was the entire dispatch and there is
   no ``if op == "mip"`` anywhere in the engine; that is why a new z-reducer costs one registry
   entry and zero engine edits, and it is the property that has to survive dozens more operators.
   It is checkable over the AST, so it is checked rather than asserted in prose.

WHY THE CONFORMANCE TEST RUNS THE OPERATOR instead of reading the registry twice. A test that
asserts ``operator_produces("spot") == "labels"`` asserts that a line of code says what it says.
The question worth asking is whether the pixels that come out satisfy the promise the declaration
makes about them, and whether the delivery path then builds the layer type that promise names. So
each operator is RUN on a fixture plane and the result is DELIVERED through the real
``MosaicLayers.add_result`` onto a real (Qt-free) napari ``ViewerModel``.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

import squidxplorer
from squidxplorer import (
    available_projectors,
    available_region_operators,
    project_well,
    operator_consumes,
    operator_params,
    operator_produces,
)
from squidxplorer._engine import Param, _resolve_operator, add_projector, bind_operator
from squidxplorer._spots import LAYER_KEY as SPOT_KEY, available_segmenters, segmenter_available

napari = pytest.importorskip("napari")

_REPO = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------

def _four_nuclei(shape=(64, 64)) -> np.ndarray:
    """A plane with STRUCTURE: four bright discs on a dim, noisy background.

    Structure is load-bearing for the intensity half of the conformance test below, which asserts
    that an operator declaring ``"intensity"`` does not in fact emit a label image. On a BLANK
    plane every operator returns all-zeros, which is a perfectly legal (empty) label image, and the
    negative would pass for every operator whatever it declared.
    """
    plane = np.zeros(shape, dtype=np.uint16)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        plane[(yy - cy) ** 2 + (xx - cx) ** 2 <= 36] = 800
    plane += np.random.RandomState(0).randint(0, 20, shape).astype(np.uint16)
    return plane


@pytest.fixture
def mosaic():
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    return MosaicLayers(ViewerModel())


@pytest.fixture(autouse=True)
def _flatfield_profile():
    """``flatfield`` refuses to run with no profile installed, on purpose (an identity field would
    silently do nothing). Install an identity one so the operator can be RUN here — the point of
    this file is the declaration, not the correction."""
    from squidxplorer import _flatfield

    before = _flatfield.active_profiles()
    _flatfield.set_profiles(
        {"405": _flatfield.FlatfieldProfile(np.ones((64, 64), dtype=np.float32))})
    yield
    if not before:
        _flatfield.clear_profile()
    else:
        _flatfield.set_profiles(before)


def _run(name: str, plane: np.ndarray) -> np.ndarray:
    """Run one registered operator over *plane*, grouping exactly as the engine would.

    The grouping is derived from ``consumes`` — the engine's own rule — rather than from a table
    in this file, so a new axis in ``CONSUMABLE_AXES`` does not silently make this helper wrong.
    """
    group = [plane, plane] if "z" in operator_consumes(name) else [plane]
    return _resolve_operator(name).fn(group)


def _is_a_label_image(arr: np.ndarray) -> bool:
    """Are these pixels a LABEL IMAGE — integer object ids, sequentially numbered from 1?

    That is what every segmenter in this package returns (``_spots.result_from_labels`` runs
    ``relabel_sequential``), and it is what makes a label image countable: the number of objects is
    ``max()``, with no gaps to make ``max()`` and ``len(unique())`` disagree.
    """
    if not (np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool):
        return False
    values = np.unique(arr)
    non_zero = values[values != 0]
    return bool(np.array_equal(non_zero, np.arange(1, len(non_zero) + 1)))


def _slow(name: str) -> bool:
    """Is this operator a model that has to be downloaded and run on a GPU-less test box?"""
    return name in available_segmenters() and name != "otsu-watershed"


# ==============================================================================================
# 1. THE CONFORMANCE TEST — parametrized over the registry, not over a list of operators
# ==============================================================================================

@pytest.mark.parametrize("name", available_projectors())
def test_every_operator_delivers_the_result_kind_it_declares(name, mosaic):
    """Run each registered operator and check its output against its own ``produces`` declaration.

    Three things are asserted, and each catches a different way a declaration can be a lie:

    * the kind is one the DELIVERY PATH knows — an operator declaring something
      ``MosaicLayers.add_result`` cannot draw is unreachable, and must not silently fall back to
      being drawn as an image;
    * the layer that lands is the TYPE that kind names — ``Labels`` for labels, ``Image`` for
      intensity. This is the defect that was live: ``spot`` emitted a label image and got an
      ``Image`` layer, auto-windowed by the fluorescence contrast rule as if label 37 were 37
      photons;
    * the PIXELS match the promise. ``labels`` must actually be a sequential label image;
      ``intensity`` must not be one.

    This test is why a future operator does not need a test of its own: registering it adds a
    parameter case here.
    """
    if _slow(name) and not segmenter_available(name)[0]:
        pytest.skip(f"{name} needs an optional package that is not installed")

    kind = operator_produces(name)
    assert kind in mosaic._RESULT_ADDERS, (
        f"{name!r} declares produces={kind!r}, which no delivery adapter serves; "
        f"MosaicLayers can draw {sorted(mosaic._RESULT_ADDERS)}")

    out = np.asarray(_run(name, _four_nuclei()))
    layer = mosaic.add_result(kind, name, "405", out, bbox_um=(0.0, 0.0, 64.0, 64.0))

    from napari.layers import Image, Labels

    if kind == "labels":
        assert isinstance(layer, Labels), (
            f"{name!r} declares produces='labels' but landed as {type(layer).__name__}")
        assert _is_a_label_image(out), (
            f"{name!r} declares produces='labels' but its output is not a sequential label "
            f"image: {len(np.unique(out))} distinct values, max {out.max()}")
    else:
        assert isinstance(layer, Image), (
            f"{name!r} declares produces={kind!r} but landed as {type(layer).__name__}")
        assert not _is_a_label_image(out), (
            f"{name!r} declares produces={kind!r} but its output IS a sequential label image; "
            f"delivered as an Image it will be auto-windowed as if its object ids were photons")


def test_the_conformance_fixture_can_tell_the_two_kinds_apart():
    """The conformance test above is only worth its green if its discriminator discriminates.

    A blank plane makes every operator return all-zeros, which is a legal EMPTY label image, and
    the intensity half of that test would then pass for an operator that really did emit labels.
    So: on this fixture, the shipped intensity operator is NOT a label image and the shipped
    segmenter IS one.
    """
    plane = _four_nuclei()
    assert not _is_a_label_image(_run("mip", plane))
    assert _is_a_label_image(_run(SPOT_KEY, plane))
    # ...and the discriminator is not simply "is it integer": both are uint16.
    assert _run("mip", plane).dtype == _run(SPOT_KEY, plane).dtype


def test_a_labels_result_is_not_handed_a_contrast_window(mosaic):
    """The actual damage the declaration prevents, on a real napari layer.

    An ``Image`` layer gets ``contrast_limits`` seeded from the pixels by ``add_mosaic``. Label 1
    and label 400 are not dim and bright, they are two different cells, so a window over them is
    meaningless — and the fluorescence rule's "background peak to black" puts most of the mask
    below black. A ``Labels`` layer has no ``contrast_limits`` attribute at all, which is the
    strongest possible statement that nothing windowed it.
    """
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[4:8, 4:8] = 1
    labels[20:24, 20:24] = 2
    layer = mosaic.add_result("labels", "spot", "405", labels)
    assert not hasattr(layer, "contrast_limits")
    assert np.array_equal(np.asarray(layer.data), labels), "the delivery path altered the labels"


def test_a_declared_kind_the_viewer_cannot_draw_is_refused_by_name(mosaic):
    """No silent fallback to add_image. An operator declaring a kind nothing implements must fail
    loud, naming the operator and what the viewer does know — the alternative is a segmentation
    quietly rendered as fluorescence, which is the whole defect."""
    with pytest.raises(ValueError, match="mesh"):
        mosaic.add_result("mesh", "some_future_op", "405", np.zeros((8, 8), dtype=np.uint16))


def test_a_labels_declaration_over_float_pixels_is_refused_naming_the_operator(mosaic):
    """napari's Labels layer rejects floats. Caught here so the message names the operator whose
    declaration was wrong, instead of surfacing from inside napari with no provenance."""
    with pytest.raises(ValueError, match="float_op"):
        mosaic.add_result("labels", "float_op", "405", np.zeros((8, 8), dtype=np.float32))


# ==============================================================================================
# 2. NOBODY BRANCHES ON AN OPERATOR'S NAME
# ==============================================================================================

#: The name comparisons that exist TODAY, with the reason each survives. Both are the same one:
#: ``flatfield`` needs an illumination profile installed before it can run, and until this branch
#: the registry could not carry a parameter, so "which operator needs a profile" had to be spelled
#: as a name. They are the measured residue of Gap 2 and they go away when ``flatfield``'s profile
#: becomes a declared ``Param`` instead of a module-level global behind a lock.
#:
#: An entry here is a debt, not a licence. Adding one requires a reason in this dict, which is the
#: point: the decision has to be made out loud.
KNOWN_NAME_BRANCHES = {
    ("squidxplorer/_viewer.py", "flatfield"):
        "run_operator auto-estimates an illumination profile before running flatfield. It is a "
        "PRECONDITION expressed as a name because the registry could not carry the profile as a "
        "parameter; _flatfield.set_profile is a module-level global behind a lock for the same "
        "reason.",
    ("squidxplorer/_benchmark.py", "flatfield"):
        "_prepare installs the same profile outside the timed window. Its own docstring says why: "
        "'it is selected by name, so it cannot take a profile argument'.",
}


def _name_branches() -> list:
    """Every ``<something> == "<a registered operator>"`` comparison in the package, by AST.

    Over the AST and not over the text, so a comment, a docstring or a log message mentioning an
    operator's name cannot fail this, and a real comparison cannot hide inside a helper or a
    multi-line expression. Same technique ``tests/test_result.py`` uses to assert the absence of a
    result-to-result comparison.
    """
    known = set(available_projectors()) | set(available_region_operators())
    found = []
    for path in sorted((_REPO / "squidxplorer").rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            sides = [node.left, *node.comparators]
            for side in sides:
                if (isinstance(side, ast.Constant) and isinstance(side.value, str)
                        and side.value in known):
                    found.append((str(path.relative_to(_REPO)), side.value, side.lineno,
                                  ast.unparse(node)))
    return found


def test_no_module_branches_on_an_operator_name():
    """``consumes`` is the entire dispatch, and ``produces`` and ``params`` join it on those terms.

    The property: adding an operator is a registry entry, never an edit to a module that has to
    learn its name. An ``if op == "cellpose"`` anywhere is that property being spent, so it fails
    here unless it is written down in ``KNOWN_NAME_BRANCHES`` with its reason.
    """
    unexpected = [f"{path}:{line}  {src}"
                  for path, name, line, src in _name_branches()
                  if (path, name) not in KNOWN_NAME_BRANCHES]
    assert not unexpected, (
        "a module now branches on an operator's NAME:\n  " + "\n  ".join(unexpected) +
        "\n\nExtend the DECLARATION instead (Operator.consumes / .produces / .params) and let the "
        "generic path read it. If the branch is genuinely unavoidable, add it to "
        "KNOWN_NAME_BRANCHES with the reason.")


def test_the_known_name_branch_list_cannot_go_stale():
    """A declared branch that has since been removed must be removed from the list too.

    Without this the list becomes a place where names go to be forgotten, and the test above would
    keep passing over a codebase that had already been cleaned up — so the debt would never be
    visibly paid off.
    """
    live = {(path, name) for path, name, _line, _src in _name_branches()}
    stale = sorted(set(KNOWN_NAME_BRANCHES) - live)
    assert not stale, (
        f"{stale} no longer branch on an operator name; delete the entry from "
        "KNOWN_NAME_BRANCHES rather than leaving a debt recorded against code that paid it.")


# ==============================================================================================
# 3. PARAMETERS ON THE REGISTRY ENTRY
# ==============================================================================================

def test_one_entry_runs_at_a_value_the_registration_never_named():
    """The gap: a parameterised operator used to mean one registry entry per parameter set.

    ``_spots``' own docstring recommended ``add_projector("spot_tight", spots_op(SpotParams(
    min_area_px=80)))`` — a second name for the same algorithm. Now the ONE entry takes the value
    at run time, and it changes the answer, which is what makes this a behaviour test rather than a
    plumbing test.
    """
    plane = _four_nuclei()
    loose = bind_operator(SPOT_KEY)([plane])
    strict = bind_operator(SPOT_KEY, {"min_area_px": 5000})([plane])
    assert loose.max() == 4, f"the fixture should hold four nuclei, got {loose.max()}"
    assert strict.max() == 0, "min_area_px did not reach the segmenter"


def test_the_default_binding_is_the_object_the_table_holds():
    """No kwargs must be byte-identical to the un-parameterised registration it replaced.

    Identity, not equality: if ``bind`` rebuilt the callable for every default run, every operator
    in the table would become a fresh closure per plate run, and ``project_plate``'s ``reduce=``
    would no longer be the thing ``available_projectors`` described.
    """
    op = _resolve_operator(SPOT_KEY)
    assert bind_operator(SPOT_KEY) is op.fn
    assert bind_operator(SPOT_KEY, {}) is op.fn
    assert bind_operator(SPOT_KEY, None) is op.fn


def test_an_operator_that_declares_no_parameters_refuses_them_by_name():
    """Accept-and-drop would run the operator at its defaults while the console line, the recipe
    and the user all said otherwise."""
    assert operator_params("mip") == ()
    with pytest.raises(ValueError, match="declares no parameters"):
        bind_operator("mip", {"radius_px": 3})


def test_an_undeclared_parameter_is_refused_naming_what_is_accepted():
    with pytest.raises(ValueError, match="no parameter 'diameter'"):
        bind_operator(SPOT_KEY, {"diameter": 30})


def test_a_declared_parameter_defaults_to_the_dataclass_it_came_from():
    """``SPOT_PARAMS`` is derived from ``SpotParams``' fields, so the dataclass stays the one place
    the knobs and their defaults are written down. A hand-copied list is how the two drift."""
    from squidxplorer._spots import DEFAULT_PARAMS

    defaults = _resolve_operator(SPOT_KEY).defaults()
    assert defaults["min_area_px"] == DEFAULT_PARAMS.min_area_px
    assert defaults["sigma_px"] == DEFAULT_PARAMS.sigma_px
    assert set(defaults) == {"sigma_px", "min_area_px", "min_distance_px", "split_touching"}


def test_a_registered_factory_is_called_at_its_declared_defaults():
    """``params=`` is what makes the registered object a FACTORY. One registrar, one rule."""
    seen = {}

    def _factory(*, scale=3):
        seen["scale"] = scale
        return lambda planes: next(iter(planes)) * scale

    add_projector("_decl_test_scaled", _factory, params=(Param("scale", 3),),
                  consumes=frozenset())
    assert seen["scale"] == 3, "the factory was not called with the declared default"
    plane = np.ones((4, 4), dtype=np.uint16)
    assert bind_operator("_decl_test_scaled")([plane]).max() == 3
    assert bind_operator("_decl_test_scaled", {"scale": 7})([plane]).max() == 7


def test_a_duplicate_declared_parameter_is_refused():
    with pytest.raises(ValueError, match="declares a parameter twice"):
        add_projector("_decl_test_dup", lambda **kw: (lambda planes: next(iter(planes))),
                      params=(Param("a", 1), Param("a", 2)))


def test_an_unknown_result_kind_is_refused_at_registration():
    """The vocabulary is enforced at the boundary where the declaration is made. A typo that fell
    back to 'intensity' would reproduce the exact defect this field exists to end, with no
    symptom."""
    with pytest.raises(ValueError, match="unknown result kind"):
        add_projector("_decl_test_kind", lambda planes: next(iter(planes)), produces="lables")


# ==============================================================================================
# 4. CELLPOSE, AS AN OPERATOR
# ==============================================================================================

def test_cellpose_is_in_the_engine_registry_not_only_the_segmenter_table():
    """The gap this branch closes, stated as one assertion.

    Cellpose was a registered SEGMENTER — the table behind the GUI's Detect-nuclei button — and
    nothing else. ``available_projectors()`` had never heard of it, so no CLI ``--projector``, no
    operator dropdown, no benchmark row, no plate-scale run: the analysis feature bolted on BESIDE
    the operator system instead of into it.
    """
    from squidxplorer._cellpose import OPERATOR_NAME

    from squidxplorer._operations import runnable_operators

    assert OPERATOR_NAME in available_projectors()
    assert OPERATOR_NAME in available_segmenters(), "the two tables disagree about the spelling"
    # ...and therefore in every surface that reads the registry rather than a hardcoded list.
    assert OPERATOR_NAME in runnable_operators()


def test_cellpose_declares_the_same_three_things_the_generic_path_reads():
    from squidxplorer._cellpose import OPERATOR_NAME

    assert operator_consumes(OPERATOR_NAME) == frozenset(), "z must survive a segmentation"
    assert operator_produces(OPERATOR_NAME) == "labels"
    # ONE parameter, not the four ``SpotParams`` has. ``cellpose_nuclei`` reads
    # ``min_distance_px`` (as the diameter) and nothing else; this used to declare all four, so
    # ``_param_panel`` drew four spin boxes for it and three could not change the answer.
    # Measured on synthetic_1536_wellplate A1 / 405 nm, 1024 px crop: ``min_area_px`` 30 and 4000
    # both returned the SAME 42 masks, byte for byte, where the parameter's documented meaning
    # would have left 2.
    assert [p.name for p in operator_params(OPERATOR_NAME)] == ["min_distance_px"]


def test_cellpose_refuses_the_parameters_it_cannot_honour_instead_of_ignoring_them():
    """The user-visible half of the fix, and the reason a narrower declaration is the right one.

    ``Operator.bind`` refuses an undeclared parameter BY NAME, so ``--param min_area_px=80`` (or
    the same key from a saved recipe, or a script) is now an error the caller reads rather than a
    number the run drops. Before this, that command completed, reported ``min_area_px=80`` in the
    console line and the recipe, and segmented at Cellpose's own defaults.
    """
    from squidxplorer._cellpose import OPERATOR_NAME
    from squidxplorer._engine import bind_operator

    for dead in ("sigma_px", "min_area_px", "split_touching"):
        with pytest.raises(ValueError) as excinfo:
            bind_operator(OPERATOR_NAME, {dead: 4000})
        assert dead in str(excinfo.value) and "min_distance_px" in str(excinfo.value), (
            f"{dead!r} must be refused by name, and the refusal must say what CAN be set; "
            f"got {excinfo.value}"
        )
    bind_operator(OPERATOR_NAME, {"min_distance_px": 20})       # the honoured one still binds


def test_every_parameter_a_segmentation_operator_DECLARES_changes_its_pixels():
    """A declared parameter that cannot change the answer is a control that does nothing.

    This is the test whose absence let ``cellpose`` ship four widgets for one working knob. It is
    run over ``spot``, whose segmenter is fast enough to evaluate once per parameter; the same
    property for ``cellpose`` is held by construction — its ``params`` are FILTERED by
    ``_spots.segmenter_honours``, the one declaration ``cellpose_nuclei`` is written against —
    and by the refusal test above.
    """
    import numpy as np

    from squidxplorer._engine import bind_operator

    rng = np.random.default_rng(0)
    plane = np.zeros((256, 256), np.uint16)
    yy, xx = np.mgrid[0:256, 0:256]
    for cy, cx in ((60, 60), (66, 70), (170, 60), (60, 175), (180, 180), (186, 190)):
        plane[(yy - cy) ** 2 + (xx - cx) ** 2 <= 64] = 4000        # touching and isolated blobs
    plane = np.clip(plane + rng.integers(0, 200, plane.shape), 0, 65535).astype(np.uint16)

    base = np.asarray(bind_operator("spot", {})(plane[None, ...]))
    probes = {"sigma_px": 9.0, "min_area_px": 400, "min_distance_px": 40, "split_touching": False}
    declared = [p.name for p in operator_params("spot")]
    assert sorted(probes) == sorted(declared), (
        f"this test must probe every declared parameter; spot declares {declared}"
    )
    for name, value in probes.items():
        got = np.asarray(bind_operator("spot", {name: value})(plane[None, ...]))
        assert not np.array_equal(got, base), (
            f"'spot' declares {name!r}, so a run at {name}={value!r} must not return the label "
            f"image the defaults return — it returned a byte-identical one, which is a control "
            f"the panel offers and the pixels never see"
        )


def test_registering_cellpose_does_not_import_torch():
    """The strongest argument for lazy REGISTRATION would be that registering costs an import.

    It does not, and this pins that. Building the entry's default binding is a closure over a
    dataclass and a string; the ``from cellpose import models`` lives one call deeper, inside the
    segmenter, and runs when a plane is actually segmented. Measured: ``import squidxplorer`` 0.146 s,
    ``import cellpose`` 0.52 s (torch alone 0.498 s), so eager importing here would multiply app
    start-up by roughly four for a feature most sessions never touch.

    Run in a SUBPROCESS: this pytest process has already imported cellpose in other test files, so
    checking ``sys.modules`` in-process would assert nothing.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import squidxplorer, sys; "
         "assert 'cellpose' in squidxplorer.available_projectors(), 'not registered'; "
         "print('torch' in sys.modules, 'cellpose' in sys.modules)"],
        cwd=str(_REPO), capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False", out.stdout


# ==============================================================================================
# 5. THE PERSISTENCE ANSWER, PINNED
# ==============================================================================================

def test_a_labels_operator_is_written_to_a_plate_as_a_real_z_stack(squid_dataset):
    """A segmentation is a plane-op, so z survives at full depth — and that is now WRITTEN.

    This test asserted the opposite until IMA-277: ``write_plate`` refused any Z>1 result ("z
    collapsed to 1"), so a label result — like every other plane-op result — could be computed
    and displayed but never persisted. The refusal was not protecting an invariant; the store has
    always been 5-D with a real z axis. What it cost was five of the eight registered operators
    having nowhere to go.

    The label result is written PER FOV, which is where a label image is meaningful. Fusing it
    across a well is separately refused (``_stitch``: averaging object ids invents objects), and
    that refusal is pinned in tests/test_stitch_zplanes.py.
    """
    import numpy as np
    import tensorstore as ts
    from squidxplorer import open_reader, write_plate

    root, _ = squid_dataset
    reader = open_reader(str(root))
    n_z = reader.metadata["n_z"]
    assert n_z >= 2, (
        "this fixture has one z plane, so a plane-op's output would be Z==1 and this test could "
        "not tell a written stack from a written plane")
    out = root.parent / "labels_out"
    manifest = write_plate(reader, str(out), projector=SPOT_KEY, n_fovs=1)
    assert manifest["complete"] and manifest["n_fields_written"] >= 1

    region = reader.metadata["regions"][0]
    fov = reader.metadata["fovs_per_region"][region][0]
    row = "".join(c for c in region if c.isalpha())
    col = "".join(c for c in region if not c.isalpha())
    field = out / "plate.ome.zarr" / row / col / str(fov) / "0"
    arr = ts.open({"driver": "zarr3",
                   "kvstore": {"driver": "file", "path": str(field)}}).result()[...].read().result()
    assert arr.shape[2] == n_z, f"wrote {arr.shape[2]} z planes for an {n_z}-plane acquisition"
    assert np.issubdtype(arr.dtype, np.integer), "label ids must stay integers on disk"


def test_write_plate_refuses_kwargs_only_for_an_operator_that_declares_none(squid_dataset):
    """The refusal moved from a rule about WHICH TABLE to the operator's own declaration.

    It used to be "``operator_kwargs`` is for region operators; a projector is parameterised when
    it is REGISTERED". Now a projector that declares parameters takes them, and one that does not
    still refuses — from its own entry, before any output directory is made.
    """
    from squidxplorer import open_reader, write_plate

    root, _ = squid_dataset
    reader = open_reader(str(root))
    out = root.parent / "kwargs_out"
    with pytest.raises(ValueError, match="declares no parameters"):
        write_plate(reader, str(out), projector="mip", operator_kwargs={"nope": 1})
    assert not out.exists(), "the run made its output tree before refusing"


# ==============================================================================================
# 6. THE ONE SINK READS THE DECLARATION
# ==============================================================================================
#
# There is exactly ONE place an operator result becomes a layer: an open region window
# (`RegionViewer.deliver_result`). It used to call `add_mosaic` unconditionally, and so did a
# SECOND sink -- `PlateWindow._add_result_layers`, which painted the plate window's own napari
# pane. That pane (`_mosaic_pane`) was pinned to None on 2026-07-23 and the only branch calling the
# method read `if self._mosaic_pane is not None`, so the second sink could never run; both were
# deleted on 2026-08-06. "A fix applied to one of two sinks" is the shape of half this codebase's
# reported defects, and the cleanest way to keep two sinks in agreement turned out to be to stop
# having two.

class _RecordingMosaic:
    """Records the TERMINAL layer calls and borrows the REAL kind dispatch off ``MosaicLayers``.

    Only the napari boundary is faked. The table that turns a declared kind into a layer type is
    the app's own, so a fake cannot agree with itself while disagreeing with what ships.
    """

    def __init__(self):
        from squidxplorer._napari_view import MosaicLayers

        self._RESULT_ADDERS = MosaicLayers._RESULT_ADDERS
        self.add_result = MosaicLayers.add_result.__get__(self)
        self.images: list = []
        self.labels: list = []

    def add_mosaic(self, op, channel, data, **kw):
        self.images.append((op, channel, data, kw))

    def add_labels(self, op, channel, data, **kw):
        self.labels.append((op, channel, data, kw))


def _label_result(op: str, channels=("405", "488")):
    """A finished ``Result`` whose substance declares labels — what a segmentation run produces."""
    from squidxplorer._address import Extent
    from squidxplorer._result import Result

    plane = np.zeros((8, 8), dtype=np.uint16)
    plane[1:3, 1:3] = 1
    return Result.of(Extent(region_id="A1"), [plane] * len(channels), channels=channels,
                     z_depth=1, pixel_size_um=1.0, dtype="uint16", kind="labels")


def _meta_for(region="A1", channels=("405", "488")):
    return {
        "fovs_per_region": {region: [0, 1]},
        "fov_positions_um": {(region, 0): (0.0, 0.0), (region, 1): (6.0, 0.0)},
        "pixel_size_um": 1.0, "frame_shape": (8, 8), "dtype": "uint16",
        "channels": [{"name": c} for c in channels], "dz_um": 1.0,
    }


def test_a_region_window_draws_a_labels_result_as_labels():
    """``RegionViewer.deliver_result``, the sink, driven with a labels result."""
    from squidxplorer import _region_viewer as RV

    pane = type("P", (), {"ok": True})()
    pane.mosaic = _RecordingMosaic()
    win = type("W", (), {"window_id": "w1", "_roi_bbox": None,
                         "_meta": _meta_for(), "_result_region": None,
                         "current_region": lambda self: "A1",
                         "_say": lambda self, m: None})()
    win._pane = pane

    added = RV.RegionViewer.deliver_result(win, "spot", _label_result("spot"), visible=True)
    assert added == 2
    assert [c[1] for c in pane.mosaic.labels] == ["405", "488"]
    assert pane.mosaic.images == [], "a label image was drawn as an Image layer"


def test_an_intensity_result_still_goes_to_the_image_path():
    """The other direction, so the sink is not simply drawing everything as labels now."""
    from squidxplorer._address import Extent
    from squidxplorer._region_viewer import RegionViewer
    from squidxplorer._result import Result

    plane = np.full((8, 8), 500, dtype=np.uint16)
    result = Result.of(Extent(region_id="A1"), [plane, plane], channels=("405", "488"),
                       z_depth=1, pixel_size_um=1.0, dtype="uint16")
    pane = type("P", (), {"ok": True})()
    pane.mosaic = _RecordingMosaic()
    win = type("W", (), {"window_id": "w1", "_roi_bbox": None, "_meta": _meta_for(),
                         "_result_region": None,
                         "current_region": lambda self: "A1",
                         "_say": lambda self, m: None})()
    win._pane = pane
    RegionViewer.deliver_result(win, "mip", result, visible=True)
    assert [c[1] for c in pane.mosaic.images] == ["405", "488"]
    assert pane.mosaic.labels == []


# ==============================================================================================
# 7. THE DECLARATION SURVIVES THE TRIP
# ==============================================================================================

def test_a_result_carries_the_kind_its_operator_declared():
    """``_as_result`` is the ONE place the registry is consulted on the display side. If the kind
    did not ride on the ``Result`` from there, every sink would have to take a possibly-scoped
    layer key apart and ask again, and two sinks asking twice is two chances to disagree."""
    from squidxplorer._operations import operator_name, result_kind

    assert operator_name("spot@tab2") == "spot"
    assert operator_name("mip") == "mip"
    assert result_kind("spot@tab2") == "labels", "a tab-scoped run lost its declaration"
    assert result_kind("mip") == "intensity"
    assert result_kind("stitch") == "intensity", "a region operator has no produces column"
    assert result_kind("computed") == "intensity", "the reopened-plate pseudo-key"


def test_the_kind_round_trips_and_an_old_declaration_still_reads():
    """The on-disk cache holds declarations written before this field existed.

    Refusing them would reject the installed base to enforce a field invented after it — the same
    judgement the plate contract makes about an unstamped ``plate_contract_version``. Those results
    are intensities, because intensity is all this path could produce.
    """
    from squidxplorer._result import Substance

    sub = Substance(channels=("405",), z_depth=1, dtype="uint16", pixel_size_um=0.3, kind="labels")
    assert Substance.from_dict(sub.to_dict()).kind == "labels"
    assert "labels" in sub.label(), "a legend cannot say what it is looking at"

    old = {"channels": ["405"], "z_depth": 1, "dtype": "uint16", "pixel_size_um": 0.3}
    assert Substance.from_dict(old).kind == "intensity"
    # ...and an intensity label is UNCHANGED, so no existing legend line moves.
    assert Substance.from_dict(old).label() == "405  z_depth 1  uint16  0.3 um/px"


# ==============================================================================================
# 8. THE PARAMETER SURVIVES THE WHOLE PIPELINE
# ==============================================================================================
#
# ``bind`` returning the right callable is not the claim worth testing. The claim is that the value
# a user typed reaches the PIXELS, through project_plate's thread pool and through write_plate's
# writer. Both legs below are driven with a registered test operator whose parameter has an
# unmissable effect, because the shipped segmentation operators find nothing on a 4x4 fixture frame
# and an assertion of "zero nuclei" would then hold whether the parameter arrived or not. That is
# the shape of test this repo has already merged once and had to unpick: green, mutation-checked,
# and asserting a vacuous truth.

def _halving_factory(*, divisor=1):
    """A z-reducer whose one parameter visibly changes every output pixel."""
    def _op(planes):
        it = iter(planes)
        acc = np.array(next(it), copy=True)
        for plane in it:
            np.maximum(acc, plane, out=acc)
        return (acc // int(divisor)).astype(acc.dtype)

    return _op


def _register_halving(name: str) -> None:
    add_projector(name, _halving_factory, params=(Param("divisor", 1),), consumes=frozenset({"z"}))


def test_a_parameterised_projector_reaches_the_pixels_through_project_plate(squid_dataset):
    """End to end through the ENGINE: ``operator_kwargs`` has to survive the thread pool."""
    from squidxplorer import open_reader, project_plate

    _register_halving("_decl_test_halve")
    root, _ = squid_dataset
    reader = open_reader(str(root))
    region = reader.metadata["regions"][0]

    def _run_plate(**kw):
        return {(r, f): np.asarray(img) for r, f, img in project_plate(
            reader, n_fovs=1, workers=1, projector="_decl_test_halve", regions=[region], **kw)}

    plain = _run_plate()
    assert plain and max(int(v.max()) for v in plain.values()) > 1, (
        "the fixture is too dim for a division to show; the assertion below would be vacuous")
    halved = _run_plate(operator_kwargs={"divisor": 2})
    assert set(plain) == set(halved)
    for key, img in halved.items():
        assert np.array_equal(img, plain[key] // 2), (
            "the parameter did not reach project_well through project_plate")


def test_a_parameterised_projector_reaches_the_pixels_through_write_plate(squid_dataset, tmp_path):
    """...and through the WRITER, which is where ``operator_kwargs`` used to be refused outright.

    Reads the written OME-Zarr back rather than trusting the manifest: the refusal that was removed
    here sat between the caller and ``project_plate``, so the only honest proof that removing it
    connected something is that the bytes on disk changed.
    """
    import zarr

    from squidxplorer import open_reader, write_plate
    from squidxplorer.contract import field_path

    _register_halving("_decl_test_halve_w")
    root, _ = squid_dataset
    reader = open_reader(str(root))

    def _write(out, **kw):
        write_plate(reader, str(out), projector="_decl_test_halve_w", n_fovs=1,
                    check_disk=False, **kw)
        store = out / "plate.ome.zarr"
        grp = zarr.open_group(str(store), mode="r")
        plate = grp.attrs["ome"]["plate"]
        well = plate["wells"][0]["path"]
        fov = zarr.open_group(str(store / well), mode="r").attrs["ome"]["well"]["images"][0]["path"]
        return np.asarray(zarr.open_array(field_path(str(store), well, fov, 0), mode="r")[:])

    plain = _write(tmp_path / "plain")
    assert int(plain.max()) > 1, "the fixture is too dim for a division to show"
    halved = _write(tmp_path / "halved", operator_kwargs={"divisor": 2})
    assert np.array_equal(halved, plain // 2), (
        "operator_kwargs did not reach the projector through write_plate")
