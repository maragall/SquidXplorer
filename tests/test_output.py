"""IMA-184 output writer — clean-room unit tests (no reader, no data on disk).

Drives ``write_from_stream`` with a fabricated metadata dict + a hand-built
``(region, fov, image)`` stream, then reads the written store back with tensorstore + json
(the same v3 store ndviewer_light reads). The real-seam cross commit (``project_plate`` on
``sim_1536wp`` + hongquan) lives in tests/test_integration.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorstore as ts

from squidmip._output import (
    plate_metadata,
    pyramid_levels,
    split_well,
    write_from_stream,
)

CH = [
    {"name": "Fluorescence_638_nm_-_Penta", "display_name": "Fluorescence 638 nm - Penta", "display_color": "#FF0000"},
    {"name": "Fluorescence_405_nm_-_Penta", "display_name": "Fluorescence 405 nm - Penta", "display_color": "#20ADF8"},
]
# B10 present so column sort is natural (2,3,10 not 10,2,3) and no zero-padding is exercised.
REGIONS = ["B2", "B3", "B10"]


def _meta():
    return {
        "regions": REGIONS,
        "fovs_per_region": {r: [0] for r in REGIONS},
        "channels": CH,
        "pixel_size_um": 0.325,
    }


def _image(seed: int, t=1, c=2, y=8, x=8, dtype=np.uint16):
    # deterministic, unique-ish per (seed, t, c) plane, kept small so it also fits uint8
    base = np.arange(y * x).reshape(y, x)
    out = np.empty((t, c, 1, y, x), dtype=dtype)
    for ti in range(t):
        for ci in range(c):
            out[ti, ci, 0] = ((base + seed * 20 + ti * 7 + ci * 3) % 200).astype(dtype)
    return out


def _stream(images: dict):
    # completion order is arbitrary in reality; yield out of plate order on purpose
    for region in ("B3", "B10", "B2"):
        yield region, 0, images[region]


def _read_array(path: Path) -> np.ndarray:
    store = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}}, open=True
    ).result()
    return np.asarray(store[...].read().result())


# --- pure helpers ---------------------------------------------------------------------------

def test_split_well_no_zero_padding_roundtrips():
    assert split_well("B2") == ("B", "2")
    assert split_well("AA12") == ("AA", "12")
    for region in ("B2", "H12", "AA1", "AF48"):
        row, col = split_well(region)
        assert row + col == region  # ndviewer reconstructs well_id = row + col


def test_split_well_rejects_non_plate_region():
    import pytest

    with pytest.raises(ValueError):
        split_well("region_1")


def test_pyramid_level_dims_are_floor_division():
    arr = _image(0, y=8, x=8)
    levels = pyramid_levels(arr)
    assert len(levels) == 3
    assert [lv.shape[-2:] for lv in levels] == [(8, 8), (4, 4), (2, 2)]  # 8, 8//2, 8//4
    assert all(lv.dtype == np.uint16 for lv in levels)


def test_pyramid_stops_when_too_small():
    # a 3x3 frame: factor 4 would give 0 -> only levels 0 (3x3) and 1 (1x1)
    levels = pyramid_levels(_image(0, y=3, x=3))
    assert [lv.shape[-2:] for lv in levels] == [(3, 3), (1, 1)]


def test_plate_metadata_natural_column_sort_and_well_paths():
    ome = plate_metadata(REGIONS, field_count=1)["plate"]
    assert [c["name"] for c in ome["columns"]] == ["2", "3", "10"]  # int sort, not lexicographic
    assert [c["name"] for c in ome["rows"]] == ["B"]
    paths = {w["path"] for w in ome["wells"]}
    assert paths == {"B/2", "B/3", "B/10"}  # no zero-padding


# --- full write via the stream --------------------------------------------------------------

def test_write_from_stream_layout_and_pixels(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=True)

    plate = Path(manifest["plate"])
    assert plate.name == "plate.ome.zarr"
    assert manifest["n_wells"] == 3 and manifest["n_fields_written"] == 3
    assert manifest["pyramid_levels"] == 3

    # plate group metadata
    plate_doc = json.loads((plate / "zarr.json").read_text())
    assert plate_doc["node_type"] == "group"
    assert plate_doc["attributes"]["ome"]["plate"]["field_count"] == 1

    # each well: group metadata + field pyramid + omero + pixel-exact full-res array
    for region in REGIONS:
        row, col = split_well(region)
        well_doc = json.loads((plate / row / col / "zarr.json").read_text())
        assert well_doc["attributes"]["ome"]["well"]["images"] == [{"path": "0"}]

        field = plate / row / col / "0"
        field_doc = json.loads((field / "zarr.json").read_text())["attributes"]["ome"]
        ds_paths = [d["path"] for d in field_doc["multiscales"][0]["datasets"]]
        assert ds_paths == ["0", "1", "2"]  # NGFF array naming, NOT scale{N}/image
        colors = [c["color"] for c in field_doc["omero"]["channels"]]
        assert colors == ["FF0000", "20ADF8"]  # hex without '#', in channel order

        assert np.array_equal(_read_array(field / "0"), images[region])  # full-res pixel-exact
        assert _read_array(field / "1").shape[-2:] == (4, 4)


def test_write_from_stream_individual_tiffs(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=True)

    import tifffile

    tiff_dir = tmp_path / "tiff" / "0"
    for region in REGIONS:
        for c_i, ch in enumerate(CH):
            f = tiff_dir / f"{region}_0_0_{ch['name']}.tiff"
            assert f.exists(), f
            plane = tifffile.imread(f)
            assert plane.dtype == np.uint16  # native dtype preserved
            assert np.array_equal(plane, images[region][0, c_i, 0])  # pixel-exact, z collapsed


def test_tiff_disabled(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=False)
    assert manifest["tiff"] is None
    assert not (tmp_path / "tiff").exists()


def test_uint8_dtype_preserved(tmp_path):
    images = {r: _image(i, dtype=np.uint8) for i, r in enumerate(REGIONS)}
    write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=False)
    arr = _read_array(tmp_path / "plate.ome.zarr" / "B" / "2" / "0" / "0")
    assert arr.dtype == np.uint8
    assert np.array_equal(arr, images["B2"])
