"""Tests for open_reader + SquidReader (AC1, AC4, AC5, AC6 + edge cases + decisions 4/5/6)."""

import json

import numpy as np
import pytest
import tifffile

from squidmip import open_reader
from squidmip.reader import SquidReader
from tests.conftest import CH_IN_YAML, CH_NOT_IN_YAML, _write_timepoint


# --- AC1 / AC5: metadata discovery ------------------------------------------
def test_metadata_discovery(squid_dataset):
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert meta["regions"] == ["B2", "B3"]
    assert meta["fovs_per_region"] == {"B2": [0, 1], "B3": [0, 1]}
    assert meta["n_z"] == 2
    assert meta["z_levels"] == [0, 1]
    assert meta["frame_shape"] == (4, 4)
    assert meta["dtype"] == np.uint16
    assert meta["n_t"] == 1
    assert meta["dz_um"] == 1.5
    assert meta["pixel_size_um"] == pytest.approx(3.76 / 20.0)


def test_channel_count_independent_of_nz(squid_dataset):
    # AC5: 2 channels, NOT 2 * Nz(=2)
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert len(meta["channels"]) == 2
    names = {c["name"] for c in meta["channels"]}
    assert names == {CH_IN_YAML, CH_NOT_IN_YAML}


def test_channel_colors_yaml_and_fallback(squid_dataset):
    # AC2: 638 from YAML nested camera_settings; 561 absent from YAML -> wavelength fallback
    root, _ = squid_dataset
    by_name = {c["name"]: c for c in open_reader(root).metadata["channels"]}
    assert by_name[CH_IN_YAML]["display_color"] == "#FF0000"
    assert by_name[CH_IN_YAML]["display_name"] == "Fluorescence 638 nm - Penta"
    assert by_name[CH_NOT_IN_YAML]["display_color"] == "#FFCF00"  # 561 from CHANNEL_COLORS_MAP


def test_positions_from_legacy_coordinates(squid_dataset):
    root, _ = squid_dataset
    pos = open_reader(root).metadata["positions"]
    assert pos[("B2", 0)] == (1.0, 2.0)
    assert pos[("B3", 1)] == (3.0, 5.0)


# --- AC4: exact-pixel read ---------------------------------------------------
def test_read_exact_pixels(squid_dataset):
    root, arrays = squid_dataset
    reader = open_reader(root)
    for key, expected in arrays.items():
        region, fov, z, ch = key
        got = reader.read(region, fov, ch, z)
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)


def test_read_matches_tifffile_directly(squid_dataset):
    root, _ = squid_dataset
    reader = open_reader(root)
    got = reader.read("B3", 1, CH_IN_YAML, 0)
    direct = tifffile.imread(root / "0" / f"B3_1_0_{CH_IN_YAML}.tiff")
    np.testing.assert_array_equal(got, direct)


# --- AC6: laziness -----------------------------------------------------------
def test_read_is_lazy_one_file(squid_dataset, monkeypatch):
    root, _ = squid_dataset
    reader = open_reader(root)
    reader.metadata  # warm metadata first (its own single-frame read is separate)

    calls = {"n": 0}
    real = tifffile.imread

    def counting_imread(path, *a, **k):
        calls["n"] += 1
        return real(path, *a, **k)

    monkeypatch.setattr("squidmip.reader.tifffile.imread", counting_imread)
    reader.read("B2", 0, CH_IN_YAML, 0)
    assert calls["n"] == 1


# --- decision 5: non-2D refusal ---------------------------------------------
def test_read_rejects_non_2d(squid_dataset):
    root, _ = squid_dataset
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    tifffile.imwrite(root / "0" / f"B2_0_0_{CH_IN_YAML}.tiff", rgb)  # overwrite with RGB
    reader = open_reader(root)
    with pytest.raises(ValueError, match="not a 2D grayscale plane|not supported"):
        reader.read("B2", 0, CH_IN_YAML, 0)


# --- dtype contract: uint8/uint16 only (Squid's real grayscale set) ----------
def test_read_rejects_uint32(squid_dataset):
    root, _ = squid_dataset
    tifffile.imwrite(
        root / "0" / f"B2_0_0_{CH_IN_YAML}.tiff", np.arange(16, dtype=np.uint32).reshape(4, 4)
    )
    with pytest.raises(ValueError, match="dtype"):
        open_reader(root).read("B2", 0, CH_IN_YAML, 0)


def test_read_accepts_uint8_native(squid_dataset):
    # MONO8 is a valid (if contrast-poor) Squid format; accept it, preserve native dtype
    root, _ = squid_dataset
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    tifffile.imwrite(root / "0" / f"B2_0_0_{CH_IN_YAML}.tiff", arr)
    got = open_reader(root).read("B2", 0, CH_IN_YAML, 0)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, arr)


# --- decision 6: time dimension ---------------------------------------------
def test_multi_timepoint(squid_dataset):
    root, arrays = squid_dataset
    t1_arrays: dict = {}
    _write_timepoint(root / "1", t1_arrays, tag=1)
    reader = open_reader(root)
    assert reader.metadata["n_t"] == 2
    got = reader.read("B2", 0, CH_IN_YAML, 0, t=1)
    np.testing.assert_array_equal(got, t1_arrays[("B2", 0, 0, CH_IN_YAML)])
    # t=0 and t=1 differ (tag offset), proving t routes to the right folder
    assert not np.array_equal(got, arrays[("B2", 0, 0, CH_IN_YAML)])


def test_read_t_out_of_range(squid_dataset):
    root, _ = squid_dataset
    with pytest.raises(IndexError, match="out of range"):
        open_reader(root).read("B2", 0, CH_IN_YAML, 0, t=5)


# --- validation + edges ------------------------------------------------------
@pytest.mark.parametrize(
    "args",
    [
        ("ZZ", 0, CH_IN_YAML, 0),   # bad region
        ("B2", 99, CH_IN_YAML, 0),  # bad fov
        ("B2", 0, "Nope", 0),       # bad channel
        ("B2", 0, CH_IN_YAML, 9),   # bad z
    ],
)
def test_read_invalid_args_raise(squid_dataset, args):
    root, _ = squid_dataset
    with pytest.raises(KeyError):
        open_reader(root).read(*args)


def test_tif_suffix_fallback(squid_dataset):
    # a plane stored as .tif (not .tiff) is still discovered and read
    root, _ = squid_dataset
    arr = np.full((4, 4), 7, dtype=np.uint16)
    tifffile.imwrite(root / "0" / f"B2_0_5_{CH_IN_YAML}.tif", arr)
    reader = open_reader(root)
    got = reader.read("B2", 0, CH_IN_YAML, 5)
    np.testing.assert_array_equal(got, arr)


def test_nz_mismatch_warns(squid_dataset):
    # params say Nz=2 but we only wrote z in {0,1}=2, so no warn here; force mismatch
    root, _ = squid_dataset
    (root / "acquisition parameters.json").write_text(
        json.dumps({"Nz": 5, "Nt": 1, "dz(um)": 1.5, "objective": {"magnification": 20.0}, "sensor_pixel_size_um": 3.76})
    )
    with pytest.warns(UserWarning, match="Nz"):
        open_reader(root).metadata


# --- format dispatch ---------------------------------------------------------
def test_open_reader_rejects_ome_tiff(tmp_path):
    (tmp_path / "ome_tiff").mkdir()
    with pytest.raises(NotImplementedError, match="OME-TIFF"):
        open_reader(tmp_path)


def test_open_reader_rejects_non_directory(tmp_path):
    f = tmp_path / "x.tiff"
    f.write_bytes(b"")
    with pytest.raises(NotImplementedError, match="not a directory"):
        open_reader(f)


def test_empty_dir_raises(tmp_path):
    (tmp_path / "0").mkdir()
    with pytest.raises(ValueError, match="No Squid individual-TIFF"):
        open_reader(tmp_path).metadata


# --- integration: the real hongquan dataset (AC1 + AC4 on real data) ---------
@pytest.mark.integration
def test_real_dataset(real_dataset):
    reader = open_reader(real_dataset)
    meta = reader.metadata
    assert set(meta["regions"]) == {"B2", "B3", "B4"}
    total_fovs = sum(len(v) for v in meta["fovs_per_region"].values())
    assert total_fovs == 48
    assert len(meta["channels"]) == 4
    assert meta["n_z"] == 3
    assert meta["frame_shape"] == (4168, 4168)
    assert meta["dtype"] == np.uint16
    # AC4 exact read against tifffile
    got = reader.read("B3", 15, "Fluorescence_638_nm_-_Penta", 0)
    direct = tifffile.imread(real_dataset / "0" / "B3_15_0_Fluorescence_638_nm_-_Penta.tiff")
    np.testing.assert_array_equal(got, direct)
