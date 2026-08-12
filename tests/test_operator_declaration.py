"""What an operator DECLARES about itself, and whether the app reads the declaration or the name."""

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


# fixtures

def _four_nuclei(shape=(64, 64)) -> np.ndarray:
    """A plane with STRUCTURE: four bright discs on a dim, noisy background."""
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
    """Install an identity profile so ``flatfield`` can be RUN here (it refuses with none)."""
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
    """Run one registered operator over *plane*, grouping exactly as the engine would."""
    group = [plane, plane] if "z" in operator_consumes(name) else [plane]
    return _resolve_operator(name).fn(group)


def _is_a_label_image(arr: np.ndarray) -> bool:
    """Are these pixels a LABEL IMAGE — integer object ids, sequentially numbered from 1?"""
    if not (np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool):
        return False
    values = np.unique(arr)
    non_zero = values[values != 0]
    return bool(np.array_equal(non_zero, np.arange(1, len(non_zero) + 1)))


def _slow(name: str) -> bool:
    """Is this operator a model that has to be downloaded and run on a GPU-less test box?"""
    return name in available_segmenters() and name != "otsu-watershed"


# 1. the conformance test — parametrized over the registry

@pytest.mark.parametrize("name", available_projectors())
def test_every_operator_delivers_the_result_kind_it_declares(name, mosaic):
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
    plane = _four_nuclei()
    assert not _is_a_label_image(_run("mip", plane))
    assert _is_a_label_image(_run(SPOT_KEY, plane))
    # ...and the discriminator is not simply "is it integer": both are uint16.
    assert _run("mip", plane).dtype == _run(SPOT_KEY, plane).dtype


def test_a_labels_result_is_not_handed_a_contrast_window(mosaic):
    """A Labels layer has no ``contrast_limits`` attribute at all — nothing windowed it."""
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[4:8, 4:8] = 1
    labels[20:24, 20:24] = 2
    layer = mosaic.add_result("labels", "spot", "405", labels)
    assert not hasattr(layer, "contrast_limits")
    assert np.array_equal(np.asarray(layer.data), labels), "the delivery path altered the labels"


def test_a_declared_kind_the_viewer_cannot_draw_is_refused_by_name(mosaic):
    with pytest.raises(ValueError, match="mesh"):
        mosaic.add_result("mesh", "some_future_op", "405", np.zeros((8, 8), dtype=np.uint16))


def test_a_labels_declaration_over_float_pixels_is_refused_naming_the_operator(mosaic):
    """napari's Labels layer rejects floats; the refusal must name the operator."""
    with pytest.raises(ValueError, match="float_op"):
        mosaic.add_result("labels", "float_op", "405", np.zeros((8, 8), dtype=np.float32))


# 2. nobody branches on an operator's name

#: The name comparisons that exist TODAY, with the reason each survives.
#: An entry here is a debt, not a licence: adding one requires a reason in this dict.
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
    """Every ``<something> == "<a registered operator>"`` comparison in the package, by AST."""
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
    unexpected = [f"{path}:{line}  {src}"
                  for path, name, line, src in _name_branches()
                  if (path, name) not in KNOWN_NAME_BRANCHES]
    assert not unexpected, (
        "a module now branches on an operator's NAME:\n  " + "\n  ".join(unexpected) +
        "\n\nExtend the DECLARATION instead (Operator.consumes / .produces / .params) and let the "
        "generic path read it. If the branch is genuinely unavoidable, add it to "
        "KNOWN_NAME_BRANCHES with the reason.")


def test_the_known_name_branch_list_cannot_go_stale():
    live = {(path, name) for path, name, _line, _src in _name_branches()}
    stale = sorted(set(KNOWN_NAME_BRANCHES) - live)
    assert not stale, (
        f"{stale} no longer branch on an operator name; delete the entry from "
        "KNOWN_NAME_BRANCHES rather than leaving a debt recorded against code that paid it.")


# 3. parameters on the registry entry

def test_one_entry_runs_at_a_value_the_registration_never_named():
    plane = _four_nuclei()
    loose = bind_operator(SPOT_KEY)([plane])
    strict = bind_operator(SPOT_KEY, {"min_area_px": 5000})([plane])
    assert loose.max() == 4, f"the fixture should hold four nuclei, got {loose.max()}"
    assert strict.max() == 0, "min_area_px did not reach the segmenter"


def test_the_default_binding_is_the_object_the_table_holds():
    """Identity, not equality: no kwargs must return the exact registered callable."""
    op = _resolve_operator(SPOT_KEY)
    assert bind_operator(SPOT_KEY) is op.fn
    assert bind_operator(SPOT_KEY, {}) is op.fn
    assert bind_operator(SPOT_KEY, None) is op.fn


def test_an_operator_that_declares_no_parameters_refuses_them_by_name():
    assert operator_params("mip") == ()
    with pytest.raises(ValueError, match="declares no parameters"):
        bind_operator("mip", {"radius_px": 3})


def test_an_undeclared_parameter_is_refused_naming_what_is_accepted():
    with pytest.raises(ValueError, match="no parameter 'diameter'"):
        bind_operator(SPOT_KEY, {"diameter": 30})


def test_a_declared_parameter_defaults_to_the_dataclass_it_came_from():
    from squidxplorer._spots import DEFAULT_PARAMS

    defaults = _resolve_operator(SPOT_KEY).defaults()
    assert defaults["min_area_px"] == DEFAULT_PARAMS.min_area_px
    assert defaults["sigma_px"] == DEFAULT_PARAMS.sigma_px
    assert set(defaults) == {"sigma_px", "min_area_px", "min_distance_px", "split_touching"}


def test_a_registered_factory_is_called_at_its_declared_defaults():
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
    with pytest.raises(ValueError, match="unknown result kind"):
        add_projector("_decl_test_kind", lambda planes: next(iter(planes)), produces="lables")


# 4. cellpose, as an operator

def test_cellpose_is_in_the_engine_registry_not_only_the_segmenter_table():
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
    # One parameter, not SpotParams' four: cellpose_nuclei reads only min_distance_px.
    assert [p.name for p in operator_params(OPERATOR_NAME)] == ["min_distance_px"]


def test_cellpose_refuses_the_parameters_it_cannot_honour_instead_of_ignoring_them():
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
    """A declared parameter that cannot change the label image is a control that does nothing."""
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
    """Run in a subprocess: this pytest process has already imported cellpose elsewhere."""
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


# 5. the persistence answer, pinned

def test_a_labels_operator_is_written_to_a_plate_as_a_real_z_stack(squid_dataset):
    """A segmentation is a plane-op, so z survives at full depth — and that is written per FOV."""
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
    from squidxplorer import open_reader, write_plate

    root, _ = squid_dataset
    reader = open_reader(str(root))
    out = root.parent / "kwargs_out"
    with pytest.raises(ValueError, match="declares no parameters"):
        write_plate(reader, str(out), projector="mip", operator_kwargs={"nope": 1})
    assert not out.exists(), "the run made its output tree before refusing"


# 6. the one sink (RegionViewer.deliver_result) reads the declaration

class _RecordingMosaic:
    """Records the TERMINAL layer calls and borrows the REAL kind dispatch off ``MosaicLayers``."""

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


# 7. the declaration survives the trip

def test_a_result_carries_the_kind_its_operator_declared():
    from squidxplorer._operations import operator_name, result_kind

    assert operator_name("spot@tab2") == "spot"
    assert operator_name("mip") == "mip"
    assert result_kind("spot@tab2") == "labels", "a tab-scoped run lost its declaration"
    assert result_kind("mip") == "intensity"
    assert result_kind("stitch") == "intensity", "a region operator has no produces column"
    assert result_kind("computed") == "intensity", "the reopened-plate pseudo-key"


def test_the_kind_round_trips_and_an_old_declaration_still_reads():
    """A pre-``kind`` on-disk declaration still reads, as intensity."""
    from squidxplorer._result import Substance

    sub = Substance(channels=("405",), z_depth=1, dtype="uint16", pixel_size_um=0.3, kind="labels")
    assert Substance.from_dict(sub.to_dict()).kind == "labels"
    assert "labels" in sub.label(), "a legend cannot say what it is looking at"

    old = {"channels": ["405"], "z_depth": 1, "dtype": "uint16", "pixel_size_um": 0.3}
    assert Substance.from_dict(old).kind == "intensity"
    # ...and an intensity label is UNCHANGED, so no existing legend line moves.
    assert Substance.from_dict(old).label() == "405  z_depth 1  uint16  0.3 um/px"


# 8. the parameter survives the whole pipeline, driven with a test operator whose
# parameter has an unmissable effect on every pixel.

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
    """...and through the WRITER: reads the written OME-Zarr back rather than trusting the manifest."""
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
