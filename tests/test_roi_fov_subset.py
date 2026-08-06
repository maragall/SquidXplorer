"""AN ROI RUN COMPUTES ONLY THE FIELDS THE BOX TOUCHES -- end to end.

Julio, 2026-08-06: *"If I draw an ROI in the middle of 4 FOVs ... if I click stitching on that ROI
window, then it will just get those 4 full fovs, stitch them and then reflect the result on the sub
fov render."* And, on why it matters: *"we're accelerating the compute because we are getting a
subset of the dataset just for the current window, that will make decon, stitching, etc very very
fast."*

THE SUBSET IS A CHAIN, AND IT BROKE AT TWO SEPARATE LINKS, both silently and both by the same
mechanism -- `list(dict)` yields a dict's KEYS, so a `{region: [fov, ...]}` request degraded into
"every field of these wells" with no error anywhere:

* ``project_plate`` did it at ``list(dict.fromkeys(regions))``. Measured on ``sim_5d_2x2_t3``:
  ``regions={"A1": [0]}`` yielded four ``(region, fov)`` pairs, not one. So the acceleration
  existed for ``stitch`` (whose engine had the rule) and for nothing else -- not decon, not bgsub,
  not spot, not cellpose, which are the expensive ones.
* ``PlateWindow.run_operator`` did it at ``regions = list(regions)``, one call before the worker,
  widening the request back to whole wells after the window had correctly narrowed it.

A chain that degrades to "more work, same answer" cannot be caught by looking at the pixels: every
one of these bugs produced the CORRECT image, just N times slower. So the tests below assert on
WHICH UNITS RAN, at each link and then through the whole chain.
"""

from __future__ import annotations

import pytest

from squidmip._mosaic_source import fovs_overlapping_bbox, mosaic_bbox_um
from squidmip.projection import scope_wells


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
    """Julio's own case, exactly as he described it."""
    meta = _meta(2)
    x0, y0, x1, y1 = mosaic_bbox_um(meta, "A1")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    assert fovs_overlapping_bbox(meta, "A1", (cx - 1, cy - 1, cx + 1, cy + 1)) == [0, 1, 2, 3]


def test_a_box_inside_one_field_picks_only_that_field():
    meta = _meta(2)
    x0, y0, _x1, _y1 = mosaic_bbox_um(meta, "A1")
    assert fovs_overlapping_bbox(meta, "A1", (x0 + 1, y0 + 1, x0 + 3, y0 + 3)) == [0]


def test_the_fields_are_WHOLE_never_cropped():
    """A FOV is the unit the reader decodes and the unit registration solves on: half a field has
    half the overlap to phase-correlate against its neighbour. So a 2 um box selects a 64 um field,
    and the crop happens afterwards on the fused result."""
    meta = _meta(2, frame=64, pitch=1.0)
    x0, y0, *_ = mosaic_bbox_um(meta, "A1")
    picked = fovs_overlapping_bbox(meta, "A1", (x0 + 30.0, y0 + 30.0, x0 + 32.0, y0 + 32.0))
    assert picked == [0], "a tiny box inside field 0 must select field 0, whole"


def test_a_box_that_touches_nothing_falls_back_rather_than_running_on_nothing():
    """``None`` = "I cannot answer", and the caller then runs the whole region. An empty set would
    be a run that finds no signal and looks exactly like a failure."""
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
    """The caller has already decided which fields; a per-well cap would silently drop some of
    them, which is the failure this whole file is about."""
    meta = _meta(2)
    assert scope_wells(meta, 1, {"A1": [0, 1, 2, 3]}) == {"A1": [0, 1, 2, 3]}
    assert scope_wells(meta, 1, ["A1"]) == {"A1": [0]}


def test_BOTH_engines_call_the_one_resolver():
    """The rule existed only in ``stitch_plate`` and that is exactly why ``project_plate`` drifted.

    MUTATION: give either engine its own copy of the resolution and this goes red -- which is the
    point, because the two copies agreeing today is not the property that matters.
    """
    import inspect

    from squidmip import _engine, _stitch

    for module in (_engine, _stitch):
        src = inspect.getsource(module)
        assert "scope_wells(meta, n_fovs, regions)" in src, (
            f"{module.__name__} no longer resolves its scope through the shared resolver")


# ------------------------------------------------------------- link 3: the whole chain, measured


def test_project_plate_really_runs_only_the_requested_fields(squid_dataset):
    """The measurement that named the bug. A per-FOV operator is where the time goes, so this is
    the assertion that says an ROI run is CHEAP rather than merely correct."""
    from squidmip import open_reader, project_plate

    root, _ = squid_dataset
    reader = open_reader(str(root))
    region = list(reader.metadata["regions"])[0]
    every = list(reader.metadata["fovs_per_region"][region])
    if len(every) < 2:
        pytest.skip("this fixture has one FOV per well; there is no subset to take")

    ran = [(r, f) for r, f, _ in
           project_plate(reader, regions={region: [every[0]]}, n_fovs=None, projector="mip")]
    assert ran == [(region, every[0])], (
        f"asked for one field, ran {len(ran)}: the FOV subset was dropped between the caller "
        f"and the engine")


def test_run_operator_does_not_widen_a_mapping_back_to_whole_wells():
    """``PlateWindow.run_operator``'s own `regions = list(regions)` was the second place the field
    lists died. Pinned on the SOURCE, because reaching this line needs a live plate, a reader and a
    QThread, and the property is about one statement.

    MUTATION: restore the unconditional `regions = list(regions)` and this goes red.
    """
    import inspect

    from squidmip import _viewer

    src = inspect.getsource(_viewer.PlateWindow.run_operator)
    assert "names = list(regions)" in src, (
        "run_operator no longer takes the region NAMES separately, so its checks are flattening "
        "the request they are checking")
    assert "if not isinstance(regions, dict):" in src, (
        "run_operator flattens `regions` unconditionally again; a `{region: [fov, ...]}` request "
        "degrades to whole wells one call before the worker")
