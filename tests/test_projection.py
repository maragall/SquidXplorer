"""Unit tests for IMA-183: select_fovs, project (primitive), project_well.

Covers the a-priori design contracts:
  * project() — pure, dtype-preserving, single-pass (bounded-memory) reduction.
  * project_well() — (T, C, 1, Y, X) TCZYX Z=1, native dtype, channels distinct,
    iterates z_levels (NOT range(n_z)) so non-contiguous z is correct.
  * select_fovs() — IMA-187 fold: n_fovs param, list-per-well, positional, loud over-count.

The tiny standard fixture (2 regions x 2 fov x 2 z x 2 ch) comes from conftest;
non-contiguous-z and multi-timepoint cases build their own minimal datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from squidmip import open_reader, plane_op, project, project_well, select_fovs
from squidmip.projection import project_reference, select_reference_z


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _write_plane(folder: Path, region, fov, z, channel, arr, t=0):
    tp = folder / str(t)
    tp.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(tp / f"{region}_{fov}_{z}_{channel}.tiff", arr)


def _plane(val, dtype=np.uint16, shape=(4, 4)):
    return (np.arange(np.prod(shape), dtype=dtype).reshape(shape) + val).astype(dtype)


def _write_min_yaml(root: Path, nz: int, nt: int = 1):
    # acquisition.yaml is the single required metadata format (JSON support removed).
    root.mkdir(parents=True, exist_ok=True)
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.325\n"
        f"z_stack:\n  nz: {nz}\n  delta_z_mm: 0.001\n"
        f"time_series:\n  nt: {nt}\n"
    )


# ======================================================================================
# A. project() — the pure MIP primitive
# ======================================================================================
def test_project_equals_np_max_reference():
    planes = [_plane(0), _plane(50), _plane(20)]
    out = project(iter(planes))
    np.testing.assert_array_equal(out, np.max(np.stack(planes), axis=0))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_project_preserves_native_dtype(dtype):
    planes = [_plane(1, dtype=dtype), _plane(9, dtype=dtype)]
    out = project(planes)
    assert out.dtype == dtype  # no upcast, no overflow


def test_project_single_plane_returns_equal_but_own_buffer():
    p = _plane(7)
    out = project([p])
    np.testing.assert_array_equal(out, p)
    assert out is not p  # pure: does not hand back / mutate the caller's array


def test_project_does_not_mutate_caller_planes():
    first = _plane(3)
    before = first.copy()
    project([first, _plane(99)])
    np.testing.assert_array_equal(first, before)  # first plane untouched


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
    # Bounded memory: project must consume the iterable exactly once, one plane at a time,
    # never stacking the whole run. A one-shot generator that records its pulls proves it.
    pulled = []

    def gen():
        for i in range(5):
            pulled.append(i)
            yield _plane(i * 10)

    out = project(gen())
    assert pulled == [0, 1, 2, 3, 4]  # single pass, all planes, in order
    np.testing.assert_array_equal(out, _plane(40))  # max is the last (largest) plane


# ======================================================================================
# B. project_well() — (T, C, 1, Y, X), native dtype, z_levels iteration
# ======================================================================================
def test_project_well_shape_and_dtype(squid_dataset):
    root, _ = squid_dataset
    reader = open_reader(root)
    out = project_well(reader, "B2", 0)
    # n_t=1, n_channels=2, Z=1, frame 4x4
    assert out.shape == (1, 2, 1, 4, 4)
    assert out.dtype == np.uint16


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
    assert out.shape[1] == len(meta["channels"])  # C, not C*Nz (no z-as-channel)
    # distinct channels -> distinct projected planes (fixture values differ per channel)
    assert not np.array_equal(out[0, 0, 0], out[0, 1, 0])


def test_project_well_iterates_z_levels_not_range(tmp_path):
    # Non-contiguous z: files at z in {0,1,3} (plane 2 missing, e.g. partial acquisition).
    # range(n_z=3) would read z=2 (KeyError) and skip z=3; z_levels=[0,1,3] is correct.
    root = tmp_path / "acq"
    ch = "Fluorescence_638_nm_-_Penta"
    vals = {0: _plane(0), 1: _plane(10), 3: _plane(30)}
    for z, arr in vals.items():
        _write_plane(root, "A1", 0, z, ch, arr)
    _write_min_yaml(root, nz=3)
    reader = open_reader(root)
    assert reader.metadata["z_levels"] == [0, 1, 3]
    assert reader.metadata["n_z"] == 3

    # spy on read() — project_well calls it positionally as read(region, fov, channel, z, t)
    read_zs = []
    orig_read = reader.read
    reader.read = lambda region, fov, channel, z, t=0: (
        read_zs.append(z) or orig_read(region, fov, channel, z, t)
    )
    out = project_well(reader, "A1", 0)

    assert sorted(set(read_zs)) == [0, 1, 3]  # used z_levels, never attempted the missing z=2
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
    """IMA-228: single-frame consumers need one timepoint, not all of them."""
    reader, t0, t1 = _two_timepoint_reader(tmp_path)

    out = project_well(reader, "A1", 0, t=1)
    assert out.shape == (1, 1, 1, 4, 4)
    np.testing.assert_array_equal(out[0, 0, 0], np.max(np.stack(list(t1.values())), axis=0))

    out0 = project_well(reader, "A1", 0, t=0)
    np.testing.assert_array_equal(out0[0, 0, 0], np.max(np.stack(list(t0.values())), axis=0))


def test_project_well_t_reads_only_that_timepoint(tmp_path):
    """The reason the parameter exists: without it a caller wanting one frame paid an
    n_t-fold read of the whole z-stack and then discarded n_t-1 of the results."""
    reader, _, _ = _two_timepoint_reader(tmp_path)
    seen = []
    real_read = type(reader).read

    def spy(self, region, fov, channel, z, t=0):
        seen.append(t)
        return real_read(self, region, fov, channel, z, t)

    type(reader).read = spy
    try:
        project_well(reader, "A1", 0, t=1)
    finally:
        type(reader).read = real_read
    assert set(seen) == {1}
    assert len(seen) == 2, "one read per z level, for the single requested timepoint only"


def test_project_well_t_none_keeps_every_timepoint(tmp_path):
    """Backward compatibility: the default must remain 'project everything'."""
    reader, _, _ = _two_timepoint_reader(tmp_path)
    assert project_well(reader, "A1", 0).shape == (2, 1, 1, 4, 4)
    np.testing.assert_array_equal(
        project_well(reader, "A1", 0), project_well(reader, "A1", 0, t=None)
    )


@pytest.mark.parametrize("bad", [2, -1, 99])
def test_project_well_t_out_of_range_raises_named(tmp_path, bad):
    reader, _, _ = _two_timepoint_reader(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        project_well(reader, "A1", 0, t=bad)


def test_project_requires_acquisition_yaml(tmp_path):
    # Single metadata format: acquisition.yaml is required. A dataset with valid frames but
    # no acquisition.yaml (or only a legacy JSON) must fail loud, not silently degrade.
    ch = "Fluorescence_638_nm_-_Penta"
    root = tmp_path / "no_yaml"
    for z, arr in {0: _plane(0), 1: _plane(30)}.items():
        _write_plane(root, "A1", 0, z, ch, arr)
    (root / "acquisition parameters.json").write_text('{"Nz": 2}')  # legacy JSON ignored
    with pytest.raises(FileNotFoundError, match="acquisition.yaml"):
        project_well(open_reader(root), "A1", 0)


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


# ======================================================================================
# C. select_fovs() — IMA-187 fold
# ======================================================================================
def _meta(fovs_per_region):
    return {"regions": sorted(fovs_per_region), "fovs_per_region": fovs_per_region}


def test_select_fovs_default_one_per_well():
    meta = _meta({"B2": [0, 1], "B3": [0, 1]})
    assert select_fovs(meta) == {"B2": [0], "B3": [0]}  # first FOV positionally, list-shaped


def test_select_fovs_keys_are_regions():
    meta = _meta({"B2": [0], "B3": [0], "B4": [0]})
    assert set(select_fovs(meta)) == {"B2", "B3", "B4"}


def test_select_fovs_n_fovs_two():
    meta = _meta({"B2": [0, 1, 2], "B3": [0, 1, 2]})
    assert select_fovs(meta, n_fovs=2) == {"B2": [0, 1], "B3": [0, 1]}  # first 2, sorted


def test_select_fovs_over_count_raises_named():
    meta = _meta({"B2": [0, 1], "B3": [0]})
    with pytest.raises(ValueError, match="B3.*only 1 FOV"):
        select_fovs(meta, n_fovs=2)


def test_select_fovs_bad_n_fovs_raises():
    with pytest.raises(ValueError, match="n_fovs must be"):
        select_fovs(_meta({"B2": [0]}), n_fovs=0)


def test_select_fovs_from_real_reader_metadata(squid_dataset):
    # cross-check against actual reader output (fov [0,1] per region in the fixture)
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert select_fovs(meta, n_fovs=1) == {"B2": [0], "B3": [0]}
    assert select_fovs(meta, n_fovs=2) == {"B2": [0, 1], "B3": [0, 1]}


def test_project_reference_picks_sharpest_plane():
    # Reference-plane reduction returns the single sharpest z-plane by Tenengrad focus (streaming,
    # bounded). A high-gradient plane beats flat/dim ones; the exact plane is returned unchanged.
    import numpy as np
    from squidmip.projection import project_reference
    rng = np.random.default_rng(1)
    flat = (np.ones((48, 48)) * 800).astype(np.uint16)
    sharp = rng.integers(0, 4000, (48, 48)).astype(np.uint16)
    dim = (sharp.astype(np.float32) * 0.25).astype(np.uint16)
    out = project_reference(iter([flat, dim, sharp]))
    assert np.array_equal(out, sharp)
    # registered as a pluggable projector, so the engine/CLI can select it by name
    import squidmip
    assert "reference" in squidmip.available_projectors()


# ======================================================================================
# D. c-alignment: a z-SELECTING reduction must not re-solve the focus per channel
#
# The bug this section exists to prevent, measured on the real 10x tissue z-stack before
# the fix: 23 of 55 (t, fov) units had their channels land on DIFFERENT z planes, worst
# case spanning four ({405:3, 488:0, 561:8, 638:9}). Channels sampled at different z do
# not overlay. project_reference is z-selecting, so the focus is solved ONCE per (t, fov)
# on a reference channel and that z is read for every channel.
# ======================================================================================
# Real Squid channel names: _channels.py refuses an unrecognised channel rather than
# handing back a placeholder colour, so the fixture must use resolvable names.
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
    """Build a 1-fov acquisition where EACH channel is sharpest at a DIFFERENT z.

    This is the fixture that makes the bug reproducible: a per-channel focus solve picks
    sharp_z[channel] for each channel, so the channels disagree. A c-aligned solve picks
    the reference channel's z for all of them.
    """
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
    """The marker attribute IS the contract; project_well dispatches on it."""
    assert getattr(project_reference, "select_index", None) is select_reference_z
    assert getattr(project, "select_index", None) is None   # MIP combines, it does not select


def test_the_fixture_really_does_split_channels_per_channel(tmp_path):
    """Guard the guard: prove this fixture WOULD misregister under a per-channel solve.

    Without this, the invariant test below could pass against a fixture where every
    channel happens to be sharpest at the same z -- i.e. it would prove nothing.
    """
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
    """THE INVARIANT. len({picked_z[c] for c in channels}) == 1, checked on data."""
    root = _per_channel_sharpest(tmp_path / "aligned", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    channels = [c["name"] for c in reader.metadata["channels"]]
    picked: dict = {}
    project_well(reader, "A1", 0, reduce=project_reference, picked_z=picked)
    assert len({picked[(0, c)] for c in channels}) == 1, picked


def test_reference_projection_defaults_to_the_first_channel(tmp_path):
    """The default is deterministic: the acquisition's first channel drives focus."""
    root = _per_channel_sharpest(tmp_path / "first", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    picked: dict = {}
    project_well(reader, "A1", 0, reduce=project_reference, picked_z=picked)
    assert picked[(0, CH_A)] == 0        # CH_A is sharpest at z 0, and CH_A leads
    assert picked[(0, CH_B)] == 0        # CH_B follows rather than picking its own z 3


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
    """A MIP consumes every z, so no single index describes it; picked_z stays empty."""
    root = _per_channel_sharpest(tmp_path / "mip", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    picked: dict = {}
    project_well(reader, "A1", 0, reduce=project, picked_z=picked)
    assert picked == {}


# ======================================================================================
# E. IMA-210 — project_well's consumes= seam (plane-op vs z-reducer), on real files
# ======================================================================================
def _z_stack_acq(root: Path, nz=3, channels=(CH_A, CH_B), nt=1):
    """A tiny real acquisition: value == z*10 + channel index, so every plane is identifiable."""
    _write_min_yaml(root, nz=nz, nt=nt)
    for t in range(nt):
        for c_i, ch in enumerate(channels):
            for z in range(nz):
                _write_plane(root, "A1", 0, z, ch, _plane(z * 10 + c_i), t=t)
    return root


def test_plane_op_keeps_every_z_plane(tmp_path):
    """consumes={} maps plane->plane: Z survives at full depth, in z_levels order."""
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
    assert len(seen) == 4 * 2      # nz x channels calls, one per output plane


def test_plane_op_records_no_picked_z(tmp_path):
    """A plane-op makes no geometric CHOICE, so there is no provenance to record."""
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
    """No consumes= → the shipped behaviour, byte for byte: Z collapses to 1 and it is a MIP."""
    reader = open_reader(_z_stack_acq(tmp_path / "default", nz=3))
    out = project_well(reader, "A1", 0)
    assert out.shape == (1, 2, 1, 4, 4)
    for c_i, ch in enumerate([c["name"] for c in reader.metadata["channels"]]):
        stack = [reader.read("A1", 0, ch, z, 0) for z in reader.metadata["z_levels"]]
        np.testing.assert_array_equal(out[0, c_i, 0], np.max(np.stack(stack), axis=0))


def test_z_selecting_reduction_is_unaffected_by_the_consumes_seam(tmp_path):
    """reference is consumes={"z"} AND select_index: the c-alignment invariant still holds."""
    root = _per_channel_sharpest(tmp_path / "still_aligned", {CH_A: 0, CH_B: 3})
    reader = open_reader(str(root))
    channels = [c["name"] for c in reader.metadata["channels"]]
    picked: dict = {}
    out = project_well(reader, "A1", 0, reduce=project_reference,
                       consumes=frozenset({"z"}), picked_z=picked)
    assert out.shape[2] == 1
    assert len({picked[(0, c)] for c in channels}) == 1, picked


def test_plane_op_adapter_rejects_a_multi_plane_group(tmp_path):
    """plane_op() lifts a plane->plane function; handing it a stack is a seam bug, not a silent take-first."""
    with pytest.raises(ValueError, match="plane-op"):
        plane_op(lambda p: p)([_plane(0), _plane(1)])


def test_n_equals_1_mip_is_byte_identical(tmp_path):
    """Regression guard: a single-z MIP returns that plane's bytes untouched."""
    reader = open_reader(_z_stack_acq(tmp_path / "n1", nz=1))
    out = project_well(reader, "A1", 0)
    for c_i, ch in enumerate([c["name"] for c in reader.metadata["channels"]]):
        np.testing.assert_array_equal(out[0, c_i, 0], reader.read("A1", 0, ch, 0, 0))
