"""An ROI run computes only the fields the box touches, end to end."""

from __future__ import annotations

import pytest

from squidxplorer._mosaic_source import fovs_overlapping_bbox, mosaic_bbox_um
from squidxplorer.projection import scope_wells


def _meta(n=2, frame=64, pitch=1.0):
    """A grid of ``n x n`` FOVs, ``frame`` px each at ``pitch`` um/px, laid out edge to edge."""
    span = frame * pitch
    positions = {}
    fovs = []
    for i in range(n * n):
        row, col = divmod(i, n)
        positions[("A1", i)] = (col * span, row * span)
        fovs.append(i)
    return {
        "regions": ["A1"],
        "fovs_per_region": {"A1": fovs},
        "fov_positions_um": positions,
        "pixel_size_um": pitch,
        "frame_shape": (frame, frame),
    }


# ------------------------------------------------------------------ link 1: box -> whole fields


def test_a_box_in_the_middle_of_four_fields_picks_all_four():
    meta = _meta(2)
    x0, y0, x1, y1 = mosaic_bbox_um(meta, "A1")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    assert fovs_overlapping_bbox(meta, "A1", (cx - 1, cy - 1, cx + 1, cy + 1)) == [0, 1, 2, 3]


def test_the_fields_are_WHOLE_never_cropped():
    """A FOV is the unit the reader decodes and registration solves on; the crop happens on the fused result."""
    meta = _meta(2, frame=64, pitch=1.0)
    x0, y0, *_ = mosaic_bbox_um(meta, "A1")
    picked = fovs_overlapping_bbox(meta, "A1", (x0 + 30.0, y0 + 30.0, x0 + 32.0, y0 + 32.0))
    assert picked == [0], "a tiny box inside field 0 must select field 0, whole"


def test_a_box_that_touches_nothing_falls_back_rather_than_running_on_nothing():
    """``None`` means "I cannot answer", and the caller then runs the whole region."""
    meta = _meta(2)
    _x0, _y0, x1, y1 = mosaic_bbox_um(meta, "A1")
    assert fovs_overlapping_bbox(meta, "A1", (x1 + 100, y1 + 100, x1 + 200, y1 + 200)) is None
    assert fovs_overlapping_bbox(meta, "A1", None) is None


def test_a_selection_that_outlived_its_acquisition_is_intersected_not_trusted():
    meta = _meta(2)
    assert scope_wells(meta, None, {"A1": [0, 99]}) == {"A1": [0]}
    assert scope_wells(meta, None, {"nosuchwell": [0]}) == {}


# ------------------------------------------------------- link 2: the resolver both engines share


def test_the_three_regions_shapes_resolve_the_way_each_engine_needs():
    meta = _meta(2)
    assert scope_wells(meta, None, None) == {"A1": [0, 1, 2, 3]}
    assert scope_wells(meta, None, ["A1"]) == {"A1": [0, 1, 2, 3]}
    assert scope_wells(meta, None, {"A1": [1, 2]}) == {"A1": [1, 2]}


def test_n_fovs_does_not_apply_to_an_explicit_field_list():
    """The caller has already decided which fields; a per-well cap would silently drop some."""
    meta = _meta(2)
    assert scope_wells(meta, 1, {"A1": [0, 1, 2, 3]}) == {"A1": [0, 1, 2, 3]}
    assert scope_wells(meta, 1, ["A1"]) == {"A1": [0]}


def test_ALL_THREE_consumers_call_the_one_resolver():
    """A private copy of the resolution in either engine OR THE WRITER goes red here."""
    import inspect

    from squidxplorer import _engine, _output, _stitch

    for module, spelling in ((_engine, "scope_wells(meta, n_fovs, regions)"),
                             (_stitch, "scope_wells(meta, n_fovs, regions)"),
                             (_output, "scope_wells(metadata, n_fovs, regions)")):
        src = inspect.getsource(module)
        assert spelling in src, (
            f"{module.__name__} no longer resolves its scope through the shared resolver")


# ------------------------------------------------- link 3: the writer owes what the stream yields


def test_an_ROI_save_is_COMPLETE_when_every_mapped_field_lands(tmp_path):
    """``write_from_stream`` with a ``{region: [fov, ...]}`` mapping owes exactly those fields."""
    import numpy as np

    from squidxplorer._output import is_incomplete, write_from_stream

    meta = _meta(2)
    meta.update({"channels": [{"name": "405", "display_color": "#8000FF"}], "dtype": "uint16"})
    img = np.zeros((1, 1, 1, 64, 64), dtype=np.uint16)
    manifest = write_from_stream(meta, iter([("A1", 1, img), ("A1", 2, img)]), tmp_path,
                                 n_fovs=None, regions={"A1": [1, 2]}, tiff=False)

    assert manifest["n_fields"] == 2, "the writer owed fields the mapped stream will never yield"
    assert manifest["n_fields_written"] == 2
    assert manifest["complete"] is True
    assert not is_incomplete(tmp_path / "plate.ome.zarr")


# ------------------------------------------------------------- link 3: the whole chain, measured


def test_run_plate_really_runs_only_the_requested_fields(squid_dataset):
    from squidxplorer import open_reader, run_plate

    root, _ = squid_dataset
    reader = open_reader(str(root))
    region = list(reader.metadata["regions"])[0]
    every = list(reader.metadata["fovs_per_region"][region])
    if len(every) < 2:
        pytest.skip("this fixture has one FOV per well; there is no subset to take")

    ran = [(r, f) for r, f, _ in
           run_plate(reader, regions={region: [every[0]]}, n_fovs=None, operator="mip")]
    assert ran == [(region, every[0])], (
        f"asked for one field, ran {len(ran)}: the FOV subset was dropped between the caller "
        f"and the engine")


def test_run_operator_does_not_widen_a_mapping_back_to_whole_wells():
    """Pinned on the source: an unconditional `regions = list(regions)` flattens the request."""
    import inspect

    from squidxplorer import _viewer

    src = inspect.getsource(_viewer.PlateWindow.run_operator)
    assert "names = list(regions)" in src, (
        "run_operator no longer takes the region NAMES separately, so its checks are flattening "
        "the request they are checking")
    assert "if not isinstance(regions, dict):" in src, (
        "run_operator flattens `regions` unconditionally again; a `{region: [fov, ...]}` request "
        "degrades to whole wells one call before the worker")
