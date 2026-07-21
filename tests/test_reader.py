"""Tests for open_reader + SquidReader (AC1, AC4, AC5, AC6 + edge cases + decisions 4/5/6)."""

import numpy as np
import pytest
import tifffile

from squidmip import open_reader
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
    # 0.325 is the stored acquisition.yaml value, NOT the recomputed 3.76/20=0.188 -> proves
    # we read the authoritative pixel size rather than recomputing it.
    assert meta["pixel_size_um"] == 0.325
    assert meta["wellplate_format"] == "1536 well plate"


def test_metadata_no_dead_attributes(squid_dataset):
    # every metadata key must be present AND functionally derived (no dead/None scalars)
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert set(meta) == {
        "regions",
        "fovs_per_region",
        "channels",
        "n_z",
        "z_levels",
        "dz_um",
        "pixel_size_um",
        "wellplate_format",
        "frame_shape",
        "dtype",
        "n_t",
        "fov_positions",
    }
    for key, value in meta.items():
        assert value is not None, f"metadata[{key!r}] is None — dead attribute"
    assert all(meta.values()) or meta["n_z"] >= 1  # no empty containers on a real dataset


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
    # keep the dataset self-consistent (nt=2) so the Nt cross-check stays quiet
    (root / "acquisition.yaml").write_text(
        "z_stack:\n  nz: 2\n  delta_z_mm: 0.0015\ntime_series:\n  nt: 2\n"
    )
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
    # acquisition.yaml is authoritative: declare nz=5 while filenames only have z in {0,1}
    root, _ = squid_dataset
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.325\nz_stack:\n  nz: 5\n  delta_z_mm: 0.0015\n"
        "time_series:\n  nt: 1\n"
    )
    with pytest.warns(UserWarning, match="Nz"):
        open_reader(root).metadata


# --- format dispatch ---------------------------------------------------------
def test_open_reader_uses_ome_reader_when_ome_files_present(tmp_path):
    # ome_tiff/ that CONTAINS .ome.tiff files -> the OME-TIFF reader (5-D TZCYX per well-FOV).
    import numpy as np
    import tifffile

    from squidmip.reader import SquidOMEReader

    ome = tmp_path / "ome_tiff"
    ome.mkdir()
    tifffile.imwrite(ome / "A1_0.ome.tiff", np.zeros((2, 2, 2, 16, 16), np.uint16),   # T,Z,C,Y,X
                     metadata={"axes": "TZCYX"}, compression="lzw")
    (tmp_path / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n- name: Fluorescence 405 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#20ADF8'\n      exposure_time_ms: 1.0\n"
        "- name: Fluorescence 488 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#00FF00'\n      exposure_time_ms: 1.0\n")
    (tmp_path / "acquisition.yaml").write_text(
        "sample:\n  wellplate_format: 384 well plate\nz_stack:\n  nz: 2\n  delta_z_mm: 0.0\n"
        "time_series:\n  nt: 2\n")
    r = open_reader(tmp_path)
    assert isinstance(r, SquidOMEReader)
    assert r.metadata["regions"] == ["A1"]
    assert r.metadata["n_z"] == 2 and r.metadata["n_t"] == 2 and r.metadata["frame_shape"] == (16, 16)
    assert len(r.metadata["channels"]) == 2
    assert r.read("A1", 0, r.metadata["channels"][1]["name"], 1, 1).shape == (16, 16)


def test_open_reader_ignores_empty_ome_tiff_placeholder(tmp_path):
    # Squid leaves an EMPTY ome_tiff/ beside an individual-TIFF acquisition; it must NOT block the
    # individual-TIFF reader. With individual TIFFs present, open_reader should succeed.
    import numpy as np
    import tifffile

    (tmp_path / "ome_tiff").mkdir()                        # empty placeholder
    (tmp_path / "0").mkdir()
    tifffile.imwrite(tmp_path / "0" / "A1_0_0_Fluorescence_488_nm_-_Penta.tiff",
                     np.zeros((4, 4), np.uint16))
    (tmp_path / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n- name: Fluorescence 488 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#00FF00'\n      exposure_time_ms: 1.0\n")
    (tmp_path / "acquisition.yaml").write_text(
        "sample:\n  wellplate_format: 384 well plate\nz_stack:\n  nz: 1\n  delta_z_mm: 0.0\n"
        "time_series:\n  nt: 1\n")
    r = open_reader(tmp_path)                              # must NOT raise
    assert r.metadata["regions"] == ["A1"]


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


# --- IMA-215: coordinates.csv -> fov_positions ----------------------------------------------

def test_fov_positions_monkey_layout(squid_dataset):
    """Explicit fov column: positions are keyed straight off it."""
    root, _ = squid_dataset
    pos = open_reader(root).metadata["fov_positions"]
    assert set(pos) == {("B2", 0), ("B2", 1), ("B3", 0), ("B3", 1)}
    x0, y0, _ = pos[("B2", 0)]
    x1, y1, _ = pos[("B2", 1)]
    assert x1 > x0 and y1 == y0            # fov 1 sits to the RIGHT of fov 0, same row


def test_fov_positions_absent_file_is_not_an_error(squid_dataset):
    """An older acquisition with no coordinates.csv yields {} — absence is normal, not a failure."""
    root, _ = squid_dataset
    (root / "coordinates.csv").unlink()
    assert open_reader(root).metadata["fov_positions"] == {}


def test_fov_positions_20x_layout_without_fov_column(squid_dataset):
    """20x layout (region,x,y,z — NO fov column): per-region ROW ORDER supplies the fov identity."""
    root, _ = squid_dataset
    (root / "coordinates.csv").write_text(
        "region,x (mm),y (mm),z (mm)\n"
        "B2,10.0,20.0,0.05\n"
        "B2,10.5,20.0,0.05\n"
        "B3,15.0,25.0,0.05\n"
        "B3,15.5,25.0,0.05\n"
    )
    pos = open_reader(root).metadata["fov_positions"]
    assert set(pos) == {("B2", 0), ("B2", 1), ("B3", 0), ("B3", 1)}
    assert pos[("B2", 0)][0] == 10.0 and pos[("B2", 1)][0] == 10.5
    assert pos[("B2", 0)][2] == 50.0       # z (mm) normalised to um


def test_fov_positions_row_count_mismatch_omits_the_region(squid_dataset):
    """The silent-misplacement guard: fewer CSV rows than filename FOVs -> omit, never mis-assign.

    Without a fov column the row order is only an ASSUMPTION about identity. When the counts
    disagree that assumption is unsafe, so the region is dropped (and the caller fails the well
    loudly) rather than placing real images at the wrong physical position.
    """
    root, _ = squid_dataset
    (root / "coordinates.csv").write_text(
        "region,x (mm),y (mm),z (mm)\n"
        "B2,10.0,20.0,0.05\n"          # only ONE row for a region whose filenames declare TWO FOVs
        "B3,15.0,25.0,0.05\n"
        "B3,15.5,25.0,0.05\n"
    )
    with pytest.warns(UserWarning, match="omitting the region"):
        pos = open_reader(root).metadata["fov_positions"]
    assert not any(r == "B2" for r, _ in pos), "B2 must be omitted, not half-assigned"
    assert set(pos) == {("B3", 0), ("B3", 1)}


def test_fov_positions_malformed_rows_are_skipped_not_guessed(squid_dataset):
    root, _ = squid_dataset
    (root / "coordinates.csv").write_text(
        "region,fov,z_level,x (mm),y (mm),z (um),time\n"
        "B2,0,0,10.0,20.0,0.0,0.0\n"
        "B2,1,0,NOT_A_NUMBER,20.0,0.0,0.0\n"
    )
    pos = open_reader(root).metadata["fov_positions"]
    assert set(pos) == {("B2", 0)}
