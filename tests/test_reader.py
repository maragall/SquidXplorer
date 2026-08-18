"""Tests for open_reader and SquidReader."""

import numpy as np
import pytest
import tifffile

from squidxplorer import open_reader
from tests.conftest import CH_IN_YAML, CH_NOT_IN_YAML, _write_timepoint


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
    # 0.325 is the stored acquisition.yaml value, not the recomputed 3.76/20=0.188 — proves we
    # read the authoritative pixel size rather than recomputing it.
    assert meta["pixel_size_um"] == 0.325
    assert meta["wellplate_format"] == "1536 well plate"


def test_metadata_no_dead_attributes(squid_dataset):
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert set(meta) == {
        "regions",
        "fovs_per_region",
        "fov_positions_um",   # {(region, fov): (x_um, y_um)}; {} when no coordinates.csv
        "channels",
        "n_z",
        "z_levels",
        "dz_um",
        "pixel_size_um",
        "wellplate_format",
        "frame_shape",
        "dtype",
        "n_t",
    }
    for key, value in meta.items():
        assert value is not None, f"metadata[{key!r}] is None — dead attribute"
    # Guards against a past tautology (`... or n_z >= 1`, always true) letting empty containers pass.
    empty = [k for k, v in meta.items() if not v]
    assert not empty, f"empty metadata container(s) on a real dataset: {empty}"


def test_channel_count_independent_of_nz(squid_dataset):
    # 2 channels, not 2 * Nz(=2)
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert len(meta["channels"]) == 2
    names = {c["name"] for c in meta["channels"]}
    assert names == {CH_IN_YAML, CH_NOT_IN_YAML}


def test_channel_colors_yaml_and_fallback(squid_dataset):
    # 638 comes from YAML's nested camera_settings; 561 is absent from YAML -> wavelength fallback
    root, _ = squid_dataset
    by_name = {c["name"]: c for c in open_reader(root).metadata["channels"]}
    assert by_name[CH_IN_YAML]["display_color"] == "#FF0000"
    assert by_name[CH_IN_YAML]["display_name"] == "Fluorescence 638 nm - Penta"
    assert by_name[CH_NOT_IN_YAML]["display_color"] == "#FFCF00"  # from CHANNEL_COLORS_MAP


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


def test_read_is_lazy_one_file(squid_dataset, monkeypatch):
    root, _ = squid_dataset
    reader = open_reader(root)
    reader.metadata  # warm metadata first (its own single-frame read is separate)

    calls = {"n": 0}
    real = tifffile.imread

    def counting_imread(path, *a, **k):
        calls["n"] += 1
        return real(path, *a, **k)

    monkeypatch.setattr("squidxplorer.reader.tifffile.imread", counting_imread)
    reader.read("B2", 0, CH_IN_YAML, 0)
    assert calls["n"] == 1


def test_read_rejects_non_2d(squid_dataset):
    root, _ = squid_dataset
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    tifffile.imwrite(root / "0" / f"B2_0_0_{CH_IN_YAML}.tiff", rgb)
    reader = open_reader(root)
    with pytest.raises(ValueError, match="not a 2D grayscale plane|not supported"):
        reader.read("B2", 0, CH_IN_YAML, 0)


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
    got = reader.read("B2", 0, CH_IN_YAML, 0, time_point=1)
    np.testing.assert_array_equal(got, t1_arrays[("B2", 0, 0, CH_IN_YAML)])
    # t=0 and t=1 differ (tag offset), proving t routes to the right folder
    assert not np.array_equal(got, arrays[("B2", 0, 0, CH_IN_YAML)])


def test_read_t_out_of_range(squid_dataset):
    root, _ = squid_dataset
    with pytest.raises(IndexError, match="out of range"):
        open_reader(root).read("B2", 0, CH_IN_YAML, 0, time_point=5)


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
    root, _ = squid_dataset
    arr = np.full((4, 4), 7, dtype=np.uint16)
    tifffile.imwrite(root / "0" / f"B2_0_5_{CH_IN_YAML}.tif", arr)
    reader = open_reader(root)
    got = reader.read("B2", 0, CH_IN_YAML, 5)
    np.testing.assert_array_equal(got, arr)


def test_a_declared_nz_larger_than_the_files_PADS_as_a_stopped_run(squid_dataset):
    # Was `test_nz_mismatch_warns`: a declared nz above the files used to be a mismatch warning.
    # Under pad-partial-acquisitions it MEANS a stopped run — the declared plan wins and the
    # missing planes read black, said out loud.
    root, _ = squid_dataset
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.325\nz_stack:\n  nz: 5\n  delta_z_mm: 0.0015\n"
        "time_series:\n  nt: 1\n"
    )
    with pytest.warns(UserWarning, match="partial acquisition"):
        m = open_reader(root, pad_partial=True).metadata
    assert m["n_z"] == 5
    # Un-padded (the CLI/engine default) the same dataset keeps the honest mismatch warning.
    with pytest.warns(UserWarning, match="Nz"):
        assert open_reader(root).metadata["n_z"] == 2


def test_a_declared_nz_SMALLER_than_the_files_still_warns(squid_dataset):
    # The other direction is not a stopped run — the files outnumber the plan, so one of the two
    # is wrong and the cross-check must say so.
    root, _ = squid_dataset
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.325\nz_stack:\n  nz: 1\n  delta_z_mm: 0.0015\n"
        "time_series:\n  nt: 1\n"
    )
    with pytest.warns(UserWarning, match="Nz"):
        open_reader(root).metadata


def test_open_reader_uses_ome_reader_when_ome_files_present(tmp_path):
    # ome_tiff/ that CONTAINS .ome.tiff files -> the OME-TIFF reader (5-D TZCYX per well-FOV).
    import numpy as np
    import tifffile

    from squidxplorer.reader import SquidOMEReader

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
    # Squid leaves an empty ome_tiff/ beside an individual-TIFF acquisition; it must not block
    # the individual-TIFF reader.
    import numpy as np
    import tifffile

    (tmp_path / "ome_tiff").mkdir()
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
    # The refusal comes from open_reader's dispatch and names every writer it looked for, not
    # just the individual-TIFF one.
    (tmp_path / "0").mkdir()
    with pytest.raises(ValueError, match="not a readable Squid acquisition"):
        open_reader(tmp_path).metadata


def test_opening_an_acquisition_lists_the_timepoint_folder_exactly_once(squid_dataset,
                                                                       monkeypatch):
    """Counts directory listings rather than timing them, so a reintroduced second scan fails
    this on any machine, unlike a timing-based benchmark."""
    from pathlib import Path

    root, _arrays = squid_dataset
    folder = (root / "0").resolve()
    calls = []
    real_iterdir = Path.iterdir

    def counting_iterdir(self):
        if self.resolve() == folder:
            calls.append(self)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", counting_iterdir)
    reader = open_reader(root)
    reader.metadata
    assert len(calls) == 1, f"the timepoint folder was listed {len(calls)} times, expected 1"


def test_the_index_seed_is_used_once_and_a_later_rebuild_re_reads_disk(squid_dataset):
    """A reader built with a seeded listing drops the seed after one use — keeping it would
    serve a stale listing after the folder changes underneath it."""
    root, _arrays = squid_dataset
    reader = open_reader(root)
    assert reader._scanned is not None, "open_reader should hand its listing to the reader"
    first = reader._build_index()
    assert reader._scanned is None, "the seed must be consumed by the first index build"

    # A plane written after the seed was taken is found by a rebuilt index, not hidden by it.
    new = f"{list(first)[0][0]}_9_0_{CH_IN_YAML}.tiff"
    tifffile.imwrite(root / "0" / new, np.zeros((4, 4), np.uint16))
    reader._index = None
    assert any(k[1] == 9 for k in reader._build_index()), \
        "a rebuild after the seed was consumed must re-read the directory"


# A glass slide with freeform regions — the harder, more representative case than a regular grid.
@pytest.mark.integration
def test_real_dataset(real_dataset):
    reader = open_reader(real_dataset)
    meta = reader.metadata
    assert set(meta["regions"]) == {"manual0", "manual1"}
    assert len(meta["fovs_per_region"]["manual0"]) == 27
    assert len(meta["fovs_per_region"]["manual1"]) == 28
    assert len(meta["channels"]) == 4
    assert meta["n_z"] == 10
    assert meta["frame_shape"] == (2084, 2084)
    assert meta["dtype"] == np.uint16
    assert meta["pixel_size_um"] == pytest.approx(0.752, abs=1e-3)
    # the units contract: positions are MICROMETRES and the key says so
    assert "fov_positions" not in meta
    xs = [v[0] for v in meta["fov_positions_um"].values()]
    assert max(xs) - min(xs) > 1000        # a mm value here would be ~1000x too small
