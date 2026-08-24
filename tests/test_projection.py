"""Unit tests for select_fovs, project (primitive), and project_well."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from squidxplorer import open_reader, plane_op, project, project_well, select_fovs
from squidxplorer.projection import project_reference, select_reference_z


def _write_plane(folder: Path, region, fov, z, channel, arr, t=0):
    tp = folder / str(t)
    tp.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(tp / f"{region}_{fov}_{z}_{channel}.tiff", arr)


def _plane(val, dtype=np.uint16, shape=(4, 4)):
    return (np.arange(np.prod(shape), dtype=dtype).reshape(shape) + val).astype(dtype)


def _write_min_yaml(root: Path, nz: int, nt: int = 1):
    root.mkdir(parents=True, exist_ok=True)
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.325\n"
        f"z_stack:\n  nz: {nz}\n  delta_z_mm: 0.001\n"
        f"time_series:\n  nt: {nt}\n"
    )


def test_project_equals_np_max_reference():
    planes = [_plane(0), _plane(50), _plane(20)]
    out = project(iter(planes))
    np.testing.assert_array_equal(out, np.max(np.stack(planes), axis=0))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_project_preserves_native_dtype(dtype):
    planes = [_plane(1, dtype=dtype), _plane(9, dtype=dtype)]
    out = project(planes)
    assert out.dtype == dtype


def test_project_single_plane_returns_equal_but_own_buffer():
    p = _plane(7)
    out = project([p])
    np.testing.assert_array_equal(out, p)
    assert out is not p


def test_project_does_not_mutate_caller_planes():
    first = _plane(3)
    before = first.copy()
    project([first, _plane(99)])
    np.testing.assert_array_equal(first, before)


def test_project_empty_raises():
    with pytest.raises(ValueError, match="at least one plane"):
        project(iter([]))


def test_project_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        project([_plane(0, shape=(4, 4)), _plane(0, shape=(4, 5))])


def test_project_dtype_mismatch_raises():
    with pytest.raises(ValueError, match="dtype"):
        project([_plane(0, dtype=np.uint16), _plane(0, dtype=np.uint8)])


def test_project_streams_single_pass():
    pulled = []

    def gen():
        for i in range(5):
            pulled.append(i)
            yield _plane(i * 10)

    out = project(gen())
    assert pulled == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(out, _plane(40))


def test_project_well_shape_and_dtype(squid_dataset):
    root, _ = squid_dataset
    reader = open_reader(root)
    out = project_well(reader, "B2", 0)
    assert out.shape == (1, 2, 1, 4, 4)
    assert out.dtype == np.uint16


def test_a_keeps_depth_z_consumer_returns_every_processed_plane(squid_dataset):
    """decon3d's shape: one call over the whole stack, the WHOLE processed stack back
    (Julio 2026-08-21: decon output the same size as the input — the planes get examined)."""
    root, arrays = squid_dataset
    reader = open_reader(root)
    meta = reader.metadata
    nz = int(meta["n_z"])

    def flipz(planes):
        return np.asarray(list(planes))[::-1]

    flipz.consumes = frozenset({"z"})
    flipz.keeps_depth = True
    out = project_well(reader, "B2", 0, reduce=flipz)
    assert out.shape[2] == nz, "depth must survive a keeps_depth z-consumer"
    for c_i, ch in enumerate(c["name"] for c in meta["channels"]):
        for k, z in enumerate(reversed(meta["z_levels"])):
            np.testing.assert_array_equal(out[0, c_i, k], arrays[("B2", 0, z, ch)])


def test_a_keeps_depth_operator_owes_the_full_stack_shape(squid_dataset):
    root, _ = squid_dataset
    reader = open_reader(root)

    def liar(planes):
        return np.asarray(list(planes))[0]      # one plane where a stack is owed

    liar.consumes = frozenset({"z"})
    liar.keeps_depth = True
    with pytest.raises(ValueError, match="keeps_depth"):
        project_well(reader, "B2", 0, reduce=liar)


def test_project_well_matches_np_max_per_channel(squid_dataset):
    root, arrays = squid_dataset
    reader = open_reader(root)
    meta = reader.metadata
    out = project_well(reader, "B3", 1)
    for c_i, ch in enumerate(c["name"] for c in meta["channels"]):
        ref = np.max(np.stack([arrays[("B3", 1, z, ch)] for z in meta["z_levels"]]), axis=0)
        np.testing.assert_array_equal(out[0, c_i, 0], ref)


def test_project_well_channels_distinct_and_ordered(squid_dataset):
    root, _ = squid_dataset
    reader = open_reader(root)
    meta = reader.metadata
    out = project_well(reader, "B2", 0)
    assert out.shape[1] == len(meta["channels"])
    assert not np.array_equal(out[0, 0, 0], out[0, 1, 0])


def test_project_well_iterates_z_levels_not_range(tmp_path):
    root = tmp_path / "acq"
    ch = "Fluorescence_638_nm_-_Penta"
    vals = {0: _plane(0), 1: _plane(10), 3: _plane(30)}
    for z, arr in vals.items():
        _write_plane(root, "A1", 0, z, ch, arr)
    _write_min_yaml(root, nz=3)
    reader = open_reader(root)
    assert reader.metadata["z_levels"] == [0, 1, 3]
    assert reader.metadata["n_z"] == 3

    read_zs = []
    orig_read = reader.read
    reader.read = lambda region, fov, channel, z, t=0: (
        read_zs.append(z) or orig_read(region, fov, channel, z, t)
    )
    out = project_well(reader, "A1", 0)

    assert sorted(set(read_zs)) == [0, 1, 3]
    np.testing.assert_array_equal(out[0, 0, 0], np.max(np.stack(list(vals.values())), axis=0))


def test_project_well_multi_timepoint(tmp_path):
    root = tmp_path / "acq"
    ch = "Fluorescence_638_nm_-_Penta"
    t0 = {0: _plane(0), 1: _plane(5)}
    t1 = {0: _plane(100), 1: _plane(105)}
    for z, arr in t0.items():
        _write_plane(root, "A1", 0, z, ch, arr, t=0)
    for z, arr in t1.items():
        _write_plane(root, "A1", 0, z, ch, arr, t=1)
    _write_min_yaml(root, nz=2, nt=2)
    reader = open_reader(root)
    assert reader.metadata["n_t"] == 2
    out = project_well(reader, "A1", 0)
    assert out.shape == (2, 1, 1, 4, 4)
    np.testing.assert_array_equal(out[0, 0, 0], np.max(np.stack(list(t0.values())), axis=0))
    np.testing.assert_array_equal(out[1, 0, 0], np.max(np.stack(list(t1.values())), axis=0))


def _two_timepoint_reader(tmp_path):
    """A 2-timepoint, 2-z, single-channel acquisition; returns (reader, t0_planes, t1_planes)."""
    root = tmp_path / "acq"
    ch = "Fluorescence_638_nm_-_Penta"
    t0 = {0: _plane(0), 1: _plane(5)}
    t1 = {0: _plane(100), 1: _plane(105)}
    for z, arr in t0.items():
        _write_plane(root, "A1", 0, z, ch, arr, t=0)
    for z, arr in t1.items():
        _write_plane(root, "A1", 0, z, ch, arr, t=1)
    _write_min_yaml(root, nz=2, nt=2)
    return open_reader(root), t0, t1


def test_project_well_t_selects_one_timepoint(tmp_path):
    reader, t0, t1 = _two_timepoint_reader(tmp_path)

    out = project_well(reader, "A1", 0, time_point=1)
    assert out.shape == (1, 1, 1, 4, 4)
    np.testing.assert_array_equal(out[0, 0, 0], np.max(np.stack(list(t1.values())), axis=0))

    out0 = project_well(reader, "A1", 0, time_point=0)
    np.testing.assert_array_equal(out0[0, 0, 0], np.max(np.stack(list(t0.values())), axis=0))


def test_project_well_t_reads_only_that_timepoint(tmp_path):
    reader, _, _ = _two_timepoint_reader(tmp_path)
    seen = []
    real_read = type(reader).read

    def spy(self, region, fov, channel, z_level, time_point=0):
        seen.append(time_point)
        return real_read(self, region, fov, channel, z_level, time_point)

    type(reader).read = spy
    try:
        project_well(reader, "A1", 0, time_point=1)
    finally:
        type(reader).read = real_read
    assert set(seen) == {1}
    assert len(seen) == 2, "one read per z level, for the single requested timepoint only"


def test_project_well_t_none_keeps_every_timepoint(tmp_path):
    reader, _, _ = _two_timepoint_reader(tmp_path)
    assert project_well(reader, "A1", 0).shape == (2, 1, 1, 4, 4)
    np.testing.assert_array_equal(
        project_well(reader, "A1", 0), project_well(reader, "A1", 0, time_point=None)
    )


@pytest.mark.parametrize("bad", [2, -1, 99])
def test_project_well_t_out_of_range_raises_named(tmp_path, bad):
    reader, _, _ = _two_timepoint_reader(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        project_well(reader, "A1", 0, time_point=bad)


def test_project_accepts_a_legacy_acquisition(tmp_path):
    """Was ``test_project_requires_acquisition_yaml`` — the refusal it pinned was overridden on
    2026-08-16 (Julio: "We should be able to support old acquisitions too"): a dataset carrying
    only the legacy 'acquisition parameters.json' now loads through the one metadata loader's
    fallback, with a warning, and projects like any other."""
    ch = "Fluorescence_638_nm_-_Penta"
    root = tmp_path / "no_yaml"
    for z, arr in {0: _plane(0), 1: _plane(30)}.items():
        _write_plane(root, "A1", 0, z, ch, arr)
    (root / "acquisition parameters.json").write_text('{"Nz": 2}')
    with pytest.warns(UserWarning, match="legacy"):
        result = project_well(open_reader(root), "A1", 0)
    assert result is not None


def test_project_well_single_z(tmp_path):
    root = tmp_path / "acq"
    ch = "Fluorescence_638_nm_-_Penta"
    only = _plane(42)
    _write_plane(root, "A1", 0, 0, ch, only)
    _write_min_yaml(root, nz=1)
    reader = open_reader(root)
    assert reader.metadata["z_levels"] == [0]
    out = project_well(reader, "A1", 0)
    np.testing.assert_array_equal(out[0, 0, 0], only)


def _meta(fovs_per_region):
    return {"regions": sorted(fovs_per_region), "fovs_per_region": fovs_per_region}


def test_select_fovs_default_one_per_well():
    meta = _meta({"B2": [0, 1], "B3": [0, 1]})
    assert select_fovs(meta) == {"B2": [0], "B3": [0]}


def test_select_fovs_keys_are_regions():
    meta = _meta({"B2": [0], "B3": [0], "B4": [0]})
    assert set(select_fovs(meta)) == {"B2", "B3", "B4"}


def test_select_fovs_n_fovs_two():
    meta = _meta({"B2": [0, 1, 2], "B3": [0, 1, 2]})
    assert select_fovs(meta, n_fovs=2) == {"B2": [0, 1], "B3": [0, 1]}


def test_select_fovs_over_count_raises_named():
    meta = _meta({"B2": [0, 1], "B3": [0]})
    with pytest.raises(ValueError, match="B3.*only 1 FOV"):
        select_fovs(meta, n_fovs=2)


def test_select_fovs_bad_n_fovs_raises():
    with pytest.raises(ValueError, match="n_fovs must be"):
        select_fovs(_meta({"B2": [0]}), n_fovs=0)


def test_select_fovs_from_real_reader_metadata(squid_dataset):
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert select_fovs(meta, n_fovs=1) == {"B2": [0], "B3": [0]}
    assert select_fovs(meta, n_fovs=2) == {"B2": [0, 1], "B3": [0, 1]}


def test_project_reference_picks_sharpest_plane():
    import numpy as np
    from squidxplorer.projection import project_reference
    rng = np.random.default_rng(1)
    flat = (np.ones((48, 48)) * 800).astype(np.uint16)
    sharp = rng.integers(0, 4000, (48, 48)).astype(np.uint16)
    dim = (sharp.astype(np.float32) * 0.25).astype(np.uint16)
    out = project_reference(iter([flat, dim, sharp]))
    assert np.array_equal(out, sharp)
    import squidxplorer
    assert "reference" in squidxplorer.available_plane_operators()


CH_A = "Fluorescence_405_nm_-_Penta"
CH_B = "Fluorescence_638_nm_-_Penta"


def _sharp(shape=(8, 8), dtype=np.uint16):
    """A high-gradient plane: Tenengrad scores it far above a flat one."""
    a = np.zeros(shape, dtype=dtype)
    a[::2, :] = np.iinfo(dtype).max // 4
    return a


def _flat(val=3, shape=(8, 8), dtype=np.uint16):
    return np.full(shape, val, dtype=dtype)


def _per_channel_sharpest(root: Path, sharp_z: dict, nz=4, shape=(8, 8)):
    """Build a 1-fov acquisition where EACH channel is sharpest at a DIFFERENT z."""
    _write_min_yaml(root, nz=nz)
    for channel, zc in sharp_z.items():
        for z in range(nz):
            _write_plane(root, "A1", 0, z, channel,
                         _sharp(shape) if z == zc else _flat(shape=shape))
    return root


def test_select_reference_z_returns_position_of_sharpest():
    assert select_reference_z([_flat(), _sharp(), _flat()]) == 1


def test_select_reference_z_ties_keep_earliest():
    assert select_reference_z([_sharp(), _sharp()]) == 0


def test_select_reference_z_empty_raises():
    with pytest.raises(ValueError, match="at least one plane"):
        select_reference_z(iter([]))


def test_project_reference_advertises_that_it_selects_an_index():
    assert getattr(project_reference, "select_index", None) is select_reference_z
    assert getattr(project, "select_index", None) is None


def test_the_fixture_really_does_split_channels_per_channel(tmp_path):
    root = _per_channel_sharpest(tmp_path / "split", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    per_channel = {
        ch: reader.metadata["z_levels"][
            select_reference_z(reader.read("A1", 0, ch, z, 0)
                               for z in reader.metadata["z_levels"])
        ]
        for ch in [c["name"] for c in reader.metadata["channels"]]
    }
    assert len(set(per_channel.values())) > 1, (
        f"fixture is useless -- channels already agree: {per_channel}")


def test_reference_projection_lands_every_channel_on_one_z(tmp_path):
    root = _per_channel_sharpest(tmp_path / "aligned", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    channels = [c["name"] for c in reader.metadata["channels"]]
    picked: dict = {}
    project_well(reader, "A1", 0, reduce=project_reference, picked_z=picked)
    assert len({picked[(0, c)] for c in channels}) == 1, picked


def test_reference_projection_defaults_to_the_first_channel(tmp_path):
    root = _per_channel_sharpest(tmp_path / "first", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    picked: dict = {}
    project_well(reader, "A1", 0, reduce=project_reference, picked_z=picked)
    assert picked[(0, CH_A)] == 0
    assert picked[(0, CH_B)] == 0


def test_reference_channel_override_moves_every_channel(tmp_path):
    root = _per_channel_sharpest(tmp_path / "override", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    picked: dict = {}
    project_well(reader, "A1", 0, reduce=project_reference,
                 reference_channel=CH_B, picked_z=picked)
    assert picked[(0, CH_A)] == picked[(0, CH_B)] == 3


def test_unknown_reference_channel_is_loud(tmp_path):
    root = _per_channel_sharpest(tmp_path / "bad", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    with pytest.raises(ValueError, match="is not a channel"):
        project_well(reader, "A1", 0, reduce=project_reference, reference_channel="Fluorescence_999_nm_-_Penta")


def test_a_combining_reduction_records_no_picked_z(tmp_path):
    root = _per_channel_sharpest(tmp_path / "mip", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    picked: dict = {}
    project_well(reader, "A1", 0, reduce=project, picked_z=picked)
    assert picked == {}


def _z_stack_acq(root: Path, nz=3, channels=(CH_A, CH_B), nt=1):
    """A tiny real acquisition: value == z*10 + channel index, so every plane is identifiable."""
    _write_min_yaml(root, nz=nz, nt=nt)
    for t in range(nt):
        for c_i, ch in enumerate(channels):
            for z in range(nz):
                _write_plane(root, "A1", 0, z, ch, _plane(z * 10 + c_i), t=t)
    return root


def test_plane_op_keeps_every_z_plane(tmp_path):
    reader = open_reader(_z_stack_acq(tmp_path / "planeop", nz=3))
    out = project_well(reader, "A1", 0, reduce=plane_op(lambda p: p), consumes=frozenset())
    assert out.shape == (1, 2, 3, 4, 4)
    for c_i, ch in enumerate([c["name"] for c in reader.metadata["channels"]]):
        for k, z in enumerate(reader.metadata["z_levels"]):
            np.testing.assert_array_equal(out[0, c_i, k], reader.read("A1", 0, ch, z, 0))


def test_plane_op_output_is_the_op_applied_per_plane(tmp_path):
    reader = open_reader(_z_stack_acq(tmp_path / "shift", nz=3))
    out = project_well(reader, "A1", 0, reduce=plane_op(lambda p: p + 1), consumes=frozenset())
    for c_i, ch in enumerate([c["name"] for c in reader.metadata["channels"]]):
        for k, z in enumerate(reader.metadata["z_levels"]):
            np.testing.assert_array_equal(out[0, c_i, k], reader.read("A1", 0, ch, z, 0) + 1)


def test_plane_op_sees_exactly_one_plane_per_call(tmp_path):
    reader = open_reader(_z_stack_acq(tmp_path / "one", nz=4))
    seen = []

    def spy(planes):
        planes = list(planes)
        seen.append(len(planes))
        return planes[0]

    project_well(reader, "A1", 0, reduce=spy, consumes=frozenset())
    assert set(seen) == {1}, f"plane-op handed stacks of {sorted(set(seen))} planes"
    assert len(seen) == 4 * 2


def test_plane_op_records_no_picked_z(tmp_path):
    reader = open_reader(_z_stack_acq(tmp_path / "prov", nz=2))
    picked: dict = {}
    project_well(reader, "A1", 0, reduce=plane_op(lambda p: p),
                 consumes=frozenset(), picked_z=picked)
    assert picked == {}


def test_plane_op_preserves_dtype_and_timepoints(tmp_path):
    reader = open_reader(_z_stack_acq(tmp_path / "t", nz=2, nt=2))
    out = project_well(reader, "A1", 0, reduce=plane_op(lambda p: p), consumes=frozenset())
    assert out.shape == (2, 2, 2, 4, 4)
    assert out.dtype == reader.metadata["dtype"]


def test_default_consumes_is_the_z_reducer_contract(tmp_path):
    reader = open_reader(_z_stack_acq(tmp_path / "default", nz=3))
    out = project_well(reader, "A1", 0)
    assert out.shape == (1, 2, 1, 4, 4)
    for c_i, ch in enumerate([c["name"] for c in reader.metadata["channels"]]):
        stack = [reader.read("A1", 0, ch, z, 0) for z in reader.metadata["z_levels"]]
        np.testing.assert_array_equal(out[0, c_i, 0], np.max(np.stack(stack), axis=0))


def test_z_selecting_reduction_is_unaffected_by_the_consumes_seam(tmp_path):
    root = _per_channel_sharpest(tmp_path / "still_aligned", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    channels = [c["name"] for c in reader.metadata["channels"]]
    picked: dict = {}
    out = project_well(reader, "A1", 0, reduce=project_reference,
                       consumes=frozenset({"z"}), picked_z=picked)
    assert out.shape[2] == 1
    assert len({picked[(0, c)] for c in channels}) == 1, picked


def test_plane_op_adapter_rejects_a_multi_plane_group(tmp_path):
    with pytest.raises(ValueError, match="plane-op"):
        plane_op(lambda p: p)([_plane(0), _plane(1)])


def test_n_equals_1_mip_is_byte_identical(tmp_path):
    reader = open_reader(_z_stack_acq(tmp_path / "n1", nz=1))
    out = project_well(reader, "A1", 0)
    for c_i, ch in enumerate([c["name"] for c in reader.metadata["channels"]]):
        np.testing.assert_array_equal(out[0, c_i, 0], reader.read("A1", 0, ch, 0, 0))


def test_cast_like_rounds_and_clips_instead_of_truncating_and_wrapping():
    from squidxplorer.projection import cast_like

    got = cast_like(np.array([-3.0, 10.5, 11.5, 12.7, 70000.0], dtype=np.float32), np.uint16)
    np.testing.assert_array_equal(got, np.array([0, 10, 12, 13, 65535], dtype=np.uint16))
    assert got.dtype == np.uint16


def test_cast_like_in_place_gives_the_same_answer_as_the_copying_form():
    from squidxplorer.projection import cast_like

    values = np.array([-3.0, 10.5, 11.5, 12.7, 70000.0], dtype=np.float32)
    np.testing.assert_array_equal(cast_like(values.copy(), np.uint16),
                                  cast_like(values.copy(), np.uint16, copy=False))


def test_cast_like_in_place_refuses_an_integer_buffer_by_name():
    from squidxplorer.projection import cast_like

    with pytest.raises(ValueError, match="floating-point buffer"):
        cast_like(np.array([1, 2, 3], dtype=np.uint16), np.uint16, copy=False)


def test_no_module_carries_a_second_dtype_cast():
    import pathlib

    import squidxplorer

    pkg = pathlib.Path(squidxplorer.__file__).parent
    offenders = [str(p.relative_to(pkg)) for p in sorted(pkg.rglob("*.py"))
                 if "def _cast_like" in p.read_text()]
    assert not offenders, (
        f"{offenders} define a private dtype cast; there is one, `projection.cast_like`, and it "
        "is what `_background`, `_decon`, `_flatfield`, `_stitch`, `_output` and `_tilesource` "
        "all call"
    )
