"""IMA-187: n_fovs=None ("all FOVs") through selection, engine and writer.

Two things are load-bearing here:

1. ``n_fovs=None`` must survive the whole pipeline. OME-NGFF's ``field_count`` is a plate-level
   scalar that gets ``int()``-ed, so an unresolved ``None`` raises TypeError deep inside the
   writer — far from the caller that passed it.
2. ``n_fovs=1`` must behave EXACTLY as before. The mosaic work is worthless if it perturbs the
   single-FOV path every existing acquisition uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._output import write_from_stream
from squidxplorer.projection import project_well, resolve_n_fovs, select_fovs
from squidxplorer.reader import open_reader


def _meta(fovs_per_region):
    return {
        "regions": sorted(fovs_per_region),
        "fovs_per_region": fovs_per_region,
        "channels": [{"name": "C0", "display_color": "#FF0000"}],
        "n_z": 1, "z_levels": [0], "dz_um": 1.0, "pixel_size_um": 0.5,
        "wellplate_format": "384 well plate", "frame_shape": (4, 4),
        "dtype": np.dtype("uint16"), "n_t": 1, "fov_positions_um": {},
    }


# --- select_fovs ----------------------------------------------------------------------------

def test_none_selects_every_fov():
    meta = _meta({"A1": [0, 1, 2], "A2": [0, 1, 2]})
    assert select_fovs(meta, n_fovs=None) == {"A1": [0, 1, 2], "A2": [0, 1, 2]}


def test_none_tolerates_ragged_wells():
    """One short well must not abort the plate — that is the whole point of None."""
    meta = _meta({"A1": [0, 1, 2, 3], "A2": [0, 1]})
    assert select_fovs(meta, n_fovs=None) == {"A1": [0, 1, 2, 3], "A2": [0, 1]}


def test_explicit_count_still_raises_on_a_short_well():
    """The explicit-count contract is unchanged: asking for more than a well has is bad input."""
    meta = _meta({"A1": [0, 1, 2, 3], "A2": [0, 1]})
    with pytest.raises(ValueError, match="only 2 FOV"):
        select_fovs(meta, n_fovs=4)


def test_explicit_count_error_points_at_the_none_escape_hatch():
    meta = _meta({"A1": [0]})
    with pytest.raises(ValueError, match="n_fovs=None"):
        select_fovs(meta, n_fovs=2)


def test_default_is_still_one_fov_per_well():
    meta = _meta({"A1": [0, 1, 2], "A2": [0, 1, 2]})
    assert select_fovs(meta) == {"A1": [0], "A2": [0]}


def test_zero_and_negative_counts_rejected():
    meta = _meta({"A1": [0, 1]})
    for bad in (0, -1):
        with pytest.raises(ValueError, match=">= 1 or None"):
            select_fovs(meta, n_fovs=bad)


# --- resolve_n_fovs (the field_count guard) -------------------------------------------------

def test_resolve_none_takes_the_max_across_ragged_wells():
    meta = _meta({"A1": [0, 1, 2, 3], "A2": [0, 1]})
    assert resolve_n_fovs(meta, None) == 4


def test_resolve_passes_an_explicit_count_through():
    assert resolve_n_fovs(_meta({"A1": [0, 1]}), 2) == 2


def test_resolve_never_returns_none():
    """int(None) inside plate_metadata is the exact TypeError this function exists to prevent."""
    assert isinstance(resolve_n_fovs(_meta({"A1": [0, 1]}), None), int)


# --- writer: n_fovs=None must not TypeError -------------------------------------------------

def _stream(meta, wells):
    for region, fovs in wells.items():
        for fov in fovs:
            yield region, fov, np.full((1, 1, 1, 4, 4), fov + 1, np.uint16)


def test_write_from_stream_accepts_n_fovs_none(tmp_path):
    meta = _meta({"A1": [0, 1], "A2": [0, 1]})
    wells = select_fovs(meta, n_fovs=None)
    manifest = write_from_stream(meta, _stream(meta, wells), tmp_path, n_fovs=None)
    assert manifest["n_wells"] == 2
    assert manifest["n_fields_written"] == 4       # 2 wells x 2 FOVs


def test_plate_field_count_reflects_the_max_on_a_ragged_plate(tmp_path):
    import json

    meta = _meta({"A1": [0, 1, 2], "A2": [0]})
    wells = select_fovs(meta, n_fovs=None)
    write_from_stream(meta, _stream(meta, wells), tmp_path, n_fovs=None)
    plate = json.loads((tmp_path / "plate.ome.zarr" / "zarr.json").read_text())
    node = plate["attributes"]["ome"] if "attributes" in plate else plate
    assert node["plate"]["field_count"] == 3


def test_every_fov_gets_its_own_field_directory(tmp_path):
    meta = _meta({"A1": [0, 1, 2]})
    wells = select_fovs(meta, n_fovs=None)
    write_from_stream(meta, _stream(meta, wells), tmp_path, n_fovs=None)
    well_dir = tmp_path / "plate.ome.zarr" / "A" / "1"
    assert sorted(d.name for d in well_dir.iterdir() if d.is_dir()) == ["0", "1", "2"]


# --- N=1 regression -------------------------------------------------------------------------

def test_n1_projection_is_byte_identical_to_the_single_fov_path(squid_dataset):
    """The keystone regression guard: multi-FOV support must not perturb one-FOV output."""
    root, _ = squid_dataset
    reader = open_reader(root)
    meta = reader.metadata
    region = meta["regions"][0]
    fov = meta["fovs_per_region"][region][0]

    direct = project_well(reader, region, fov)
    via_all = project_well(reader, region, select_fovs(meta, n_fovs=None)[region][0])
    assert np.array_equal(direct, via_all)
    assert direct.dtype == via_all.dtype


def test_n1_selection_unchanged_by_the_positions_work(squid_dataset):
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert select_fovs(meta, n_fovs=1) == {r: [0] for r in meta["regions"]}
