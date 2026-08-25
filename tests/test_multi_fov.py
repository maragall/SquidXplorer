"""n_fovs=None ("all FOVs") through selection, engine and writer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from squidxplorer._output import write_from_stream
from squidxplorer.projection import select_fovs
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


def test_none_selects_every_fov_and_tolerates_ragged_wells_while_the_default_is_one():
    meta = _meta({"A1": [0, 1, 2, 3], "A2": [0, 1]})
    assert select_fovs(meta, n_fovs=None) == {"A1": [0, 1, 2, 3], "A2": [0, 1]}
    assert select_fovs(meta) == {"A1": [0], "A2": [0]}


def test_an_explicit_count_refuses_a_short_well_by_name_and_points_at_none():
    meta = _meta({"A1": [0, 1, 2, 3], "A2": [0, 1]})
    with pytest.raises(ValueError, match="only 2 FOV"):
        select_fovs(meta, n_fovs=4)
    with pytest.raises(ValueError, match="n_fovs=None"):
        select_fovs(_meta({"A1": [0]}), n_fovs=2)
    for bad in (0, -1):
        with pytest.raises(ValueError, match=">= 1 or None"):
            select_fovs(meta, n_fovs=bad)


def _stream(meta, wells):
    for region, fovs in wells.items():
        for fov in fovs:
            yield region, fov, np.full((1, 1, 1, 4, 4), fov + 1, np.uint16)


def test_the_writer_takes_n_fovs_none_and_a_ragged_plate_gets_one_field_dir_per_fov(tmp_path):
    meta = _meta({"A1": [0, 1, 2], "A2": [0]})
    wells = select_fovs(meta, n_fovs=None)
    manifest = write_from_stream(meta, _stream(meta, wells), tmp_path, n_fovs=None)
    assert manifest["n_wells"] == 2 and manifest["n_fields_written"] == 4
    plate = json.loads((tmp_path / "plate.ome.zarr" / "zarr.json").read_text())
    node = plate["attributes"]["ome"] if "attributes" in plate else plate
    assert node["plate"]["field_count"] == 3
    well_dir = tmp_path / "plate.ome.zarr" / "A" / "1"
    assert sorted(d.name for d in well_dir.iterdir() if d.is_dir()) == ["0", "1", "2"]


def test_n1_selection_is_unchanged_on_a_real_acquisition(squid_dataset):
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert select_fovs(meta, n_fovs=1) == {r: [0] for r in meta["regions"]}
    region = meta["regions"][0]
    assert select_fovs(meta, n_fovs=None)[region][0] == meta["fovs_per_region"][region][0]
