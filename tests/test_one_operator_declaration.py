"""ONE declaration, every consumer: each test registers an operator in exactly one call and then
asks a different consumer a question it can only answer if it read the declaration."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import squidxplorer as s
from squidxplorer._engine import Param

# The GUI half needs a Qt binding and the same PySide guard the other GUI test modules use.
if "PySide6" in sys.modules or "PySide2" in sys.modules:   # pragma: no cover
    pytest.skip("a PySide binding is already loaded", allow_module_level=True)

from qtpy.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


PLANE_OP_NAME = "declared_plane_op"
REGION_OP_NAME = "declared_region_op"

# Non-zero and non-default-looking so "the parameter reached the pixels" is decidable from them.
PLANE_DEFAULT_OFFSET = 11
REGION_DEFAULT_FILL = 7


def _plane_op_factory(*, offset: int = PLANE_DEFAULT_OFFSET, gain: float = 1.0):
    def _operator(planes):
        out = None
        for plane in planes:                      # streams; never materialises the stack
            scaled = np.asarray(plane).astype(np.float64) * gain + offset
            out = scaled if out is None else np.maximum(out, scaled)
        return out.astype(np.asarray(plane).dtype)
    _operator.consumes = frozenset({"z"})         # a z-reducer, so it can be written as a plate
    return _operator


def _region_op_factory(*, fill: int = REGION_DEFAULT_FILL):
    """A whole-well operator: ``(reader, region, fovs, **kwargs) -> (T, C, Nz, Y, X)``."""
    def _operator(reader, region, fovs, **_kwargs):
        meta = reader.metadata
        y, x = meta["frame_shape"]
        shape = (int(meta["n_t"]), len(meta["channels"]), 1, y, x)
        return np.full(shape, fill, dtype=np.dtype(meta.get("dtype", "uint16")))
    return _operator


@pytest.fixture
def declared_operators():
    s.add_projector(
        PLANE_OP_NAME, _plane_op_factory,
        params=(Param("offset", PLANE_DEFAULT_OFFSET, "counts added to every plane"),
                Param("gain", 1.0, "multiplied in before the offset")),
    )
    s.add_region_operator(
        REGION_OP_NAME, _region_op_factory,
        params=(Param("fill", REGION_DEFAULT_FILL, "the value every pixel of the mosaic gets"),),
    )
    return PLANE_OP_NAME, REGION_OP_NAME


@pytest.fixture
def reader(squid_dataset):
    root, _arrays = squid_dataset
    return s.open_reader(str(root))


# CONSUMER 1: the engine — it runs, plate-scope and region-scope, at declared parameters

def test_it_runs_plate_scope_and_the_declared_default_is_in_the_pixels(declared_operators, reader):
    out = dict(((region, fov), image)
               for region, fov, image in s.project_plate(reader, n_fovs=1,
                                                         projector=PLANE_OP_NAME))

    assert {region for region, _fov in out} == set(reader.metadata["regions"])
    assert out, "the operator ran over no region at all"
    for image in out.values():
        assert image.shape[2] == 1, "a z-reducer must collapse z, per its own declaration"
        assert int(image.min()) >= PLANE_DEFAULT_OFFSET


def test_it_runs_region_scope_and_touches_only_that_region(declared_operators, reader):
    one = reader.metadata["regions"][0]
    seen = {region for region, _fov, _img in
            s.project_plate(reader, n_fovs=1, projector=PLANE_OP_NAME, regions=[one])}

    assert seen == {one}


def test_a_parameter_the_run_names_reaches_the_pixels(declared_operators, reader):
    """A value must reach the pixels, not just the console line."""
    one = reader.metadata["regions"][0]

    def _run(**kwargs):
        return next(iter(s.project_plate(reader, n_fovs=1, projector=PLANE_OP_NAME,
                                         regions=[one], operator_kwargs=kwargs or None)))[2]

    assert int(_run(offset=200).min()) - int(_run().min()) == 200 - PLANE_DEFAULT_OFFSET


def test_the_region_operator_runs_plate_scope_and_region_scope(declared_operators, reader):
    """The whole-well loop, reached with no edit to `stitch_plate` and no second table."""
    regions = list(reader.metadata["regions"])
    whole = [(region, image) for region, _fov, image in
             s.stitch_plate(reader, operator=REGION_OP_NAME)]

    assert {region for region, _img in whole} == set(regions)
    assert all(int(image.min()) == REGION_DEFAULT_FILL for _r, image in whole)

    scoped = [region for region, _fov, _img in
              s.stitch_plate(reader, operator=REGION_OP_NAME, regions=[regions[0]])]
    assert scoped == [regions[0]]


def test_the_region_operators_declared_parameter_is_applied(declared_operators, reader):
    """Same `params=` seam as any other operator; a region operator's kwargs used to be unchecked."""
    one = reader.metadata["regions"][0]
    _region, _fov, image = next(iter(
        s.stitch_plate(reader, operator=REGION_OP_NAME, regions=[one], fill=42)))

    assert int(image.min()) == int(image.max()) == 42


def test_it_is_saved_to_a_plate_with_no_edit_to_the_writer(declared_operators, reader, tmp_path):
    """The SAVE path, not just preview: `write_plate` dispatches on the declaration."""
    manifest = s.write_plate(reader, tmp_path / "out.hcs", projector=PLANE_OP_NAME, n_fovs=1,
                             operator_kwargs={"offset": 3})

    assert int(manifest.get("n_fields_written") or 0) > 0


# CONSUMER 2: the CLI

def test_the_cli_accepts_the_name_and_checks_the_declared_parameters(declared_operators):
    """`--projector <name> --param <declared>=v` validates; an undeclared one is refused by name."""
    from squidxplorer._cli import ProcessParameters

    params = ProcessParameters(input_folder=".", projector=PLANE_OP_NAME, param=["offset=5"])
    assert params.projector == PLANE_OP_NAME

    with pytest.raises(ValueError, match="offsett|declares"):
        ProcessParameters(input_folder=".", projector=PLANE_OP_NAME, param=["offsett=5"])


def test_the_cli_help_lists_the_operator_with_its_declared_defaults(declared_operators):
    from squidxplorer._cli import _operator_catalogue

    catalogue = _operator_catalogue()
    assert f"{PLANE_OP_NAME}(offset=11, gain=1.0)" in catalogue
    assert f"{REGION_OP_NAME}(fill=7" in catalogue


def test_the_cli_names_a_region_operator_too(declared_operators):
    from squidxplorer._cli import ProcessParameters

    assert ProcessParameters(input_folder=".",
                             projector=REGION_OP_NAME).projector == REGION_OP_NAME


# CONSUMER 3: the command surface (what an agent or script drives the app with)

def test_list_operators_describes_it_from_the_declaration(declared_operators):
    """Every column of the row is read off the declaration, so a new operator arrives fully
    described in `ops list`."""
    from squidxplorer._command import CommandBus, EngineExecutor, ListOperators

    bus = CommandBus(EngineExecutor())
    rows = {row["name"]: row for row in bus.execute(ListOperators()).data["operators"]}

    assert rows[PLANE_OP_NAME]["kind"] == "z-reducer"
    assert rows[PLANE_OP_NAME]["params"] == {"offset": 11, "gain": 1.0}
    assert rows[PLANE_OP_NAME]["available"] is True
    assert rows[REGION_OP_NAME]["kind"] == "region-operator"
    assert rows[REGION_OP_NAME]["params"] == {"fill": 7}
    assert rows[REGION_OP_NAME]["consumes"] == ["fov"]


# CONSUMER 4: the desktop GUI — the list, and the widgets

def test_the_gui_offers_it_in_the_operator_menu_with_no_card(declared_operators, qapp):
    """The 'From their declaration' submenu is built off the registry, so an operator added
    anywhere — including in an installed plugin package — appears in it."""
    import squidxplorer._viewer as V

    win = V.PlateWindow(None)
    try:
        offered = {a.text() for a in win._declared_menu.actions()}
        assert PLANE_OP_NAME in offered
        assert REGION_OP_NAME in offered
    finally:
        win.close()


def test_its_params_become_widgets_seeded_at_the_declared_defaults(declared_operators, qapp):
    """One widget per declared `Param`, chosen from the type of its default, seeded at it."""
    from squidxplorer._param_panel import GenericOperatorPanel, panel_refusal
    from tests.test_op_panels import _Host

    assert panel_refusal(PLANE_OP_NAME) is None
    panel = GenericOperatorPanel(_Host(), PLANE_OP_NAME)

    assert sorted(panel.widgets) == ["gain", "offset"]
    assert panel.kwargs() == {"offset": PLANE_DEFAULT_OFFSET, "gain": 1.0}
    assert panel.widgets["offset"].toolTip() == "counts added to every plane"


def test_a_region_operator_that_declares_params_gets_a_panel_too(declared_operators, qapp):
    """`stitch` still has no generic panel because it declares no params — a fact about `stitch`,
    not about region operators."""
    from squidxplorer._param_panel import GenericOperatorPanel, panel_refusal
    from tests.test_op_panels import _Host

    assert panel_refusal(REGION_OP_NAME) is None
    panel = GenericOperatorPanel(_Host(), REGION_OP_NAME)

    assert sorted(panel.widgets) == ["fill"]
    assert panel.kwargs() == {"fill": REGION_DEFAULT_FILL}

    why = panel_refusal("stitch")
    assert why and "declares no params" in why


def test_the_gui_run_path_dispatches_it_to_the_right_engine_loop(declared_operators, reader):
    """`_OperatorWorker` picks `project_plate` or `stitch_plate` off `is_region_operator`."""
    from squidxplorer._workers import _OperatorWorker

    meta = reader.metadata
    fov_index = {r: {"rc": (0, i), "idx": i, "well_id": r}
                 for i, r in enumerate(meta["regions"])}

    plane_worker = _OperatorWorker(PLANE_OP_NAME, reader, meta, fov_index, "",
                                   regions=meta["regions"][:1], save=False, n_fovs=1)
    region_worker = _OperatorWorker(REGION_OP_NAME, reader, meta, fov_index, "",
                                    regions=meta["regions"][:1], save=False, n_fovs=1)

    assert plane_worker._region_op is False
    assert region_worker._region_op is True
