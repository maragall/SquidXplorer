"""Fused mosaics — the unit displayed is a MOSAIC, never a single FOV. Qt-free."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from squidxplorer import _mosaic_source
from squidxplorer._mosaic_source import (
    fuse_region_mosaic,
    level_paths,
    mosaic_bbox_um,
    open_pyramid,
)


class _Reader:
    """Reads a distinct constant per FOV so placement is checkable by value."""

    # ``_path`` names the acquisition; the plane cache keys on it, and a reader without one
    # is refused rather than silently sharing a cache key with every other fake.
    _counter = itertools.count()

    def __init__(self, frame=(4, 6), values=None, fail=()):
        self.frame = frame
        self.values = values or {}
        self.fail = set(fail)
        self._path = f"/fake/acquisition/{next(_Reader._counter)}"

    def read(self, region, fov, channel, z_level, time_point=0):
        if fov in self.fail:
            raise OSError("simulated unreadable FOV")
        return np.full(self.frame, self.values.get(fov, fov + 1), dtype=np.uint16)


def _meta(positions, fovs, frame=(4, 6), px=2.0):
    return {
        "regions": ["A1"],
        "fovs_per_region": {"A1": fovs},
        "fov_positions_um": positions,
        "pixel_size_um": px,
        "frame_shape": frame,
        "dtype": "uint16",
        "channels": [{"name": "488"}],
    }


def test_a_region_absent_from_a_partial_positions_map_degrades_to_none():
    """A PARTIAL positions map (one region present, one absent) must degrade per region, not raise."""
    meta = {
        "regions": ["A1", "C3"],
        "fovs_per_region": {"A1": [0, 1], "C3": [0, 1]},
        "fov_positions_um": {("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)},
        "pixel_size_um": 2.0,
        "frame_shape": (4, 6),
        "dtype": "uint16",
        "channels": [{"name": "488"}],
    }

    assert _mosaic_source._planned_plane(meta, "A1", 2048) is not None, \
        "the well that cross-checked must still build"
    assert _mosaic_source._planned_plane(meta, "C3", 2048) is None, \
        "the well with no positions must degrade to None, not raise"
    assert _mosaic_source.mosaic_bbox_um(meta, "C3") is None


def test_two_fovs_are_placed_side_by_side_by_stage_position():
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    out = fuse_region_mosaic(_Reader(), meta, "A1", "488")
    assert out is not None
    mosaic, step = out
    assert step == 1.0
    assert mosaic.shape == (4, 12)
    assert np.all(mosaic[:, :6] == 1)
    assert np.all(mosaic[:, 6:] == 2)


def test_overlap_is_covered_not_left_as_a_hole():
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (6.0, 0.0)}, [0, 1])
    mosaic, _ = fuse_region_mosaic(_Reader(), meta, "A1", "488")
    assert mosaic.shape == (4, 9)
    assert not (mosaic == 0).any(), "a placed mosaic must have no unwritten pixels"


def test_a_mosaic_is_not_derivable_without_positions_and_says_so_by_returning_none():
    """A guessed layout would be a WRONG picture, not a rough one."""
    meta = _meta({}, [0, 1])
    assert fuse_region_mosaic(_Reader(), meta, "A1", "488") is None

    meta_no_px = _meta({("A1", 0): (0.0, 0.0)}, [0])
    meta_no_px["pixel_size_um"] = 0
    assert fuse_region_mosaic(_Reader(), meta_no_px, "A1", "488") is None


def test_an_unreadable_fov_leaves_a_hole_rather_than_shifting_its_neighbours():
    """A silent skip would slide every later FOV over by one frame and look fine while wrong."""
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    mosaic, _ = fuse_region_mosaic(_Reader(fail=[0]), meta, "A1", "488")
    assert mosaic.shape == (4, 12)
    assert np.all(mosaic[:, :6] == 0)
    assert np.all(mosaic[:, 6:] == 2)


def test_an_unreadable_fov_is_ANNOUNCED_and_not_only_left_as_a_hole(caplog):
    """The hole stays put, but a black rectangle in a fluorescence mosaic reads as "no signal
    there" (a fact about the sample) unless something announces it as a read failure."""
    import logging

    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    with caplog.at_level(logging.WARNING):
        mosaic, _ = fuse_region_mosaic(_Reader(fail=[0]), meta, "A1", "488")
    assert np.all(mosaic[:, :6] == 0)
    said = [r.getMessage() for r in caplog.records if "FOV 0" in r.getMessage()]
    assert said, (
        "an unreadable FOV left a black rectangle in region A1 / channel 488 and produced no log "
        f"record naming it; records were {[r.getMessage() for r in caplog.records]}"
    )
    assert "A1" in said[0] and "488" in said[0], (
        f"the warning must name the region and channel so the gap can be located; got {said[0]!r}"
    )


def test_a_large_region_is_decimated_rather_than_truncated():
    """Bounding RAM must not change which region the mosaic covers."""
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (2000.0, 0.0)}, [0, 1],
                 frame=(1000, 1000), px=1.0)
    mosaic, step = fuse_region_mosaic(_Reader(frame=(1000, 1000)), meta, "A1", "488",
                                      max_px=1000)
    assert step > 1
    assert max(mosaic.shape) <= 1000
    assert mosaic.shape[1] == pytest.approx(mosaic.shape[0] * 3, rel=0.02)


def test_bbox_um_is_the_regions_stage_extent():
    meta = _meta({("A1", 0): (100.0, 50.0), ("A1", 1): (112.0, 50.0)}, [0, 1])
    x0, y0, x1, y1 = mosaic_bbox_um(meta, "A1")
    assert (x0, y0) == (100.0, 50.0)
    assert (x1 - x0, y1 - y0) == (24.0, 8.0)


def test_bbox_um_is_none_when_placement_is_not_derivable():
    assert mosaic_bbox_um(_meta({}, [0]), "A1") is None


def _write_pyramid(root: Path, shapes):
    """Minimal OME-NGFF v0.5 image group with several levels."""
    import zarr

    root.mkdir(parents=True, exist_ok=True)
    for i, shape in enumerate(shapes):
        z = zarr.create_array(store=str(root / str(i)), shape=shape, dtype="uint16",
                              chunks=tuple(min(8, s) for s in shape), overwrite=True)
        z[:] = i + 1
    doc = {
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"multiscales": [{
            "axes": [{"name": n} for n in "tczyx"],
            "datasets": [{"path": str(i)} for i in range(len(shapes))],
        }]}},
    }
    (root / "zarr.json").write_text(json.dumps(doc))


def test_level_paths_follow_the_datasets_list_not_directory_sort(tmp_path):
    """Directory names sort '10' before '2'; the datasets list is the authority on level order."""
    root = tmp_path / "img"
    _write_pyramid(root, [(1, 1, 1, 32, 32), (1, 1, 1, 16, 16), (1, 1, 1, 8, 8)])
    assert [p.name for p in level_paths(root)] == ["0", "1", "2"]


def test_open_pyramid_returns_lazy_decreasing_levels(tmp_path):
    root = tmp_path / "img"
    _write_pyramid(root, [(1, 1, 1, 32, 32), (1, 1, 1, 16, 16), (1, 1, 1, 8, 8)])
    pyr = open_pyramid(root)

    assert [tuple(d.shape) for d in pyr] == [(32, 32), (16, 16), (8, 8)]
    # lazy: dask, not materialised
    assert all(hasattr(d, "compute") for d in pyr)
    assert int(np.asarray(pyr[0][0, 0])) == 1


def test_open_pyramid_drops_a_level_that_does_not_shrink(tmp_path):
    """napari needs strictly decreasing levels; a duplicate would make it pick nonsense."""
    root = tmp_path / "img"
    _write_pyramid(root, [(1, 1, 1, 32, 32), (1, 1, 1, 32, 32), (1, 1, 1, 8, 8)])
    pyr = open_pyramid(root)
    assert [tuple(d.shape) for d in pyr] == [(32, 32), (8, 8)]


def test_a_group_without_multiscales_is_a_loud_error(tmp_path):
    root = tmp_path / "img"
    root.mkdir()
    (root / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {}}))
    with pytest.raises(ValueError, match="multiscales"):
        level_paths(root)


class _StepReader(_Reader):
    """Records each read so the fusion strategy (per-level vs coarsen-over-level-0) is provable."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reads: list = []

    def read(self, region, fov, channel, z_level, time_point=0):
        self.reads.append((region, fov, channel, z_level, time_point))
        return np.full(self.frame, z_level + 1, dtype=np.uint16)


@pytest.fixture
def _record_fuse_calls(monkeypatch):
    """Record which pyramid levels each fuse pass actually produced, by their ``max_px``."""
    from squidxplorer import _mosaic_source as ms

    calls: list = []
    real = ms._fuse_levels

    def spy(reader, meta, region, channel, z, t, plans):
        calls.append([p[0] for p in plans])
        return real(reader, meta, region, channel, z, t, plans)

    monkeypatch.setattr(ms, "_fuse_levels", spy)
    return calls


def _pyr_meta(nz=6, frame=(256, 256), px=1.0, n=16):
    """16 FOVs in a row (256x4096 px) — wide enough that a pyramid has somewhere to go."""
    positions = {("A1", i): (i * frame[1] * px, 0.0) for i in range(n)}
    meta = _meta(positions, list(range(n)), frame=frame, px=px)
    meta["n_z"] = nz
    meta["dz_um"] = 1.5
    return meta


def test_deep_zoom_gets_native_pixels_on_demand():
    """Fine rungs below the cap go down to NATIVE, and a window materialises only the chunk
    under it — a slice reads the FOVs of that chunk, never the region. The fix for 'raw stays
    pixelated past _MAX_FUSED_PX'."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    reader = _StepReader(frame=(256, 256))    # frames as big as the meta declares: no holes
    meta = _pyr_meta(nz=1)
    levels, step0, _nz = fuse_region_pyramid(reader, meta, "A1", "488", max_px=1024)
    assert step0 == 4
    assert levels[0].shape == (256, 4096), "native resolution must lead the pyramid"
    assert levels[1].shape == (128, 2048)

    n_before = len(reader.reads)
    win = np.asarray(levels[0][0:100, 0:100])
    touched = {f for (_r, f, _c, _z, _t) in reader.reads[n_before:]}
    # dask pushes the exact window into the getter, so only the FOV under it decodes.
    assert touched == {0}, f"a 100x100 window sits on FOV 0 alone; read {sorted(touched)}"
    assert (win == 1).all()                       # z0 reads as 1: native pixels, no holes


def test_fine_levels_respect_the_plane_budget(monkeypatch):
    """A rung whose WHOLE plane would blow the budget is not offered: the 3-D full-res swap
    and _full_res_mip still take level 0 whole."""
    from squidxplorer import _mosaic_source as ms

    monkeypatch.setattr(ms, "_PLANE_BUDGET_BYTES", 300_000)
    levels, _step0, _nz = ms.fuse_region_pyramid(_StepReader(), _pyr_meta(nz=1), "A1", "488",
                                                 max_px=1024)
    assert levels[0].shape == (64, 1024), "over budget: the capped level must still lead"


def test_the_raw_preview_returns_a_pyramid_of_strictly_decreasing_levels():
    """napari's ``multiscale=True`` contract: a LIST, highest resolution first, each level
    strictly smaller than the one above it in both displayed axes."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta()
    levels, step, nz = fuse_region_pyramid(_Reader(), meta, "A1", "488")

    assert isinstance(levels, list) and len(levels) > 1, "a pyramid needs more than one level"
    assert nz == 6
    for above, below in zip(levels, levels[1:]):
        assert below.shape[-2] < above.shape[-2] and below.shape[-1] < above.shape[-1], (
            f"levels must strictly decrease: {above.shape} -> {below.shape}")


def test_every_pyramid_level_keeps_the_z_axis_and_its_length():
    """A pyramid must not silently flatten z. Levels downsample y and x ONLY, so napari's
    z slider (and the dz_um scale commit 19cd491 established) survive the change."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta(nz=6)
    levels, _step, nz = fuse_region_pyramid(_Reader(), meta, "A1", "488")

    for i, lv in enumerate(levels):
        assert lv.ndim == 3, f"level {i} lost the z axis: {lv.shape}"
        assert lv.shape[0] == nz == 6, f"level {i} changed the z length: {lv.shape}"


def test_every_pyramid_level_is_lazy():
    """Building the pyramid must not materialise anything, or opening a region costs GBs upfront."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    reader = _StepReader()
    levels, _step, _nz = fuse_region_pyramid(reader, _pyr_meta(), "A1", "488")

    assert all(hasattr(lv, "compute") for lv in levels)
    assert reader.reads == [], "building the pyramid must read NOTHING"


def test_a_coarse_level_is_fused_directly_and_never_materialises_level_zero(_record_fuse_calls):
    """Coarsening over the lazy level-0 graph would force the full plane to be pasted just to
    throw most of it away; fusing the level directly from the FOV tiles strides on read instead."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta()
    levels, _step, _nz = fuse_region_pyramid(_Reader(), meta, "A1", "488")
    assert len(levels) >= 3

    _record_fuse_calls.clear()
    coarse = np.asarray(levels[2][1])

    assert _record_fuse_calls, "computing a level must fuse something"
    produced = [px for plan in _record_fuse_calls for px in plan]
    level0_px = max(levels[0].shape[-2:])
    assert max(produced) < level0_px, (
        f"the fuse produced a level at max_px={max(produced)}, i.e. level 0 resolution "
        f"({level0_px} px) — level 0 was materialised to build a coarse level")
    assert coarse.shape == levels[2].shape[1:]


def test_fusing_a_level_also_yields_the_coarser_levels_from_the_same_decode():
    """napari pulls TWO levels per channel per z (the visible one, and the coarsest one for the
    thumbnail); TIFF decode is whole-frame, so the thumbnail must come from the decode already
    in hand rather than triggering a second read pass."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    reader = _StepReader()
    levels, _step, _nz = fuse_region_pyramid(reader, _pyr_meta(), "A1", "488")
    assert len(levels) >= 3

    np.asarray(levels[1][2])
    after_visible = len(reader.reads)
    assert after_visible > 0

    np.asarray(levels[-1][2])
    assert len(reader.reads) == after_visible, (
        "the coarsest level re-decoded every FOV; it must come from the pass that already "
        "read them for the visible level")


def test_a_level_finer_than_the_one_requested_is_never_produced(_record_fuse_calls):
    """Yielding the COARSER levels is free; yielding finer ones would defeat the whole point."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    levels, _step, _nz = fuse_region_pyramid(_Reader(frame=(256, 256)), _pyr_meta(), "A1", "488")
    _record_fuse_calls.clear()
    np.asarray(levels[-1][0])

    produced = [px for plan in _record_fuse_calls for px in plan]
    coarsest_px = max(levels[-1].shape[-2:])
    assert max(produced) <= max(coarsest_px, 128), (
        f"asking for the coarsest level produced a finer one (max_px={max(produced)})")


def test_a_region_whose_every_fov_is_unreadable_is_an_error_not_a_blank_mosaic():
    """A mosaic where EVERY FOV failed is not a picture at all; a black plane would report a
    read failure as empty tissue."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta(n=4)
    reader = _Reader(frame=(256, 256), fail=range(4))
    levels, _step, _nz = fuse_region_pyramid(reader, meta, "A1", "488")

    with pytest.raises(ValueError, match="no FOV.*could be read|unreadable"):
        np.asarray(levels[0][0])


def test_revisiting_a_z_plane_does_not_re_fuse_it():
    """Without a cache, stepping z back and forth re-reads every FOV of every channel each time."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    reader = _StepReader()
    levels, _step, _nz = fuse_region_pyramid(reader, _pyr_meta(), "A1", "488")

    np.asarray(levels[1][2])
    first = len(reader.reads)
    assert first > 0

    np.asarray(levels[1][2])
    assert len(reader.reads) == first, "a revisited (level, z) must come from the cache"

    np.asarray(levels[1][3])
    assert len(reader.reads) > first


def test_the_cache_is_bounded_in_bytes_and_evicts_rather_than_growing():
    """An unbounded cache is a slow memory leak wearing a performance costume."""
    from squidxplorer._mosaic_source import PYRAMID_CACHE_BYTES, fuse_region_pyramid

    assert isinstance(PYRAMID_CACHE_BYTES, int) and PYRAMID_CACHE_BYTES > 0

    reader = _StepReader()
    meta = _pyr_meta(nz=6)
    # budget holds roughly one level-0 plane, so a second z must push the first out
    levels, _step, _nz = fuse_region_pyramid(reader, meta, "A1", "488",
                                             cache_bytes=int(2.2 * 256 * 4096))

    np.asarray(levels[0][0])
    n1 = len(reader.reads)
    np.asarray(levels[0][1])
    np.asarray(levels[0][0])
    assert len(reader.reads) > 2 * n1, "a full cache must evict, not grow without bound"


def test_a_plane_larger_than_the_whole_cache_is_a_loud_error_not_a_silent_no_op():
    """A cache that logs and silently no-ops on an oversized item leaves slowness as the only
    symptom; this must raise instead."""
    from squidxplorer._mosaic_source import MemoryBoundedLRUCache

    cache = MemoryBoundedLRUCache(1024)
    with pytest.raises(ValueError, match="larger than the whole"):
        cache.put(("k",), np.zeros(4096, dtype=np.uint16))


def test_the_plane_cache_serialises_concurrent_writers():
    """dask may compute planes concurrently, so two threads can land in put() at once. Byte-drift
    under concurrency could not be made to fail reliably as a behavioural assertion, so the lock
    is checked structurally; the hammer below is only a smoke test for exceptions/bound violations."""
    import threading

    from squidxplorer._mosaic_source import MemoryBoundedLRUCache

    cache = MemoryBoundedLRUCache(64 * 1024)
    assert isinstance(cache._lock, type(threading.Lock())), (
        "the cache must hold a real lock; dask workers call put() concurrently")

    plane = np.zeros(512, dtype=np.uint16)
    errors = []

    def hammer(base):
        try:
            for i in range(200):
                cache.put((base, i), plane.copy())
                cache.get((base, i % 7))
        except Exception as exc:                    # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(b,)) for b in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert cache.nbytes <= cache.capacity_bytes, (
        f"byte accounting drifted under concurrency: {cache.nbytes} > {cache.capacity_bytes}")
    assert cache.nbytes == len(cache) * plane.nbytes


def test_a_mosaic_too_small_to_shrink_gets_one_level_not_a_degenerate_pyramid():
    """napari needs strictly decreasing levels, so a level that does not shrink is dropped."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    meta = _meta({("A1", 0): (0.0, 0.0)}, [0], frame=(4, 6), px=2.0)
    meta["n_z"] = 3
    levels, _step, _nz = fuse_region_pyramid(_Reader(), meta, "A1", "488")

    assert len(levels) == 1, f"a 4x6 mosaic has no room for a second level: {[l.shape for l in levels]}"


def test_the_pyramid_levels_agree_with_the_full_resolution_picture():
    """A misregistered level looks fine in a shape assertion and wrong on screen: a feature at a
    given fraction of the mosaic must land at that same fraction on every level."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta(nz=2)
    values = {i: (i + 1) * 100 for i in range(16)}
    levels, _step, _nz = fuse_region_pyramid(_Reader(frame=(256, 256), values=values),
                                             meta, "A1", "488")
    assert len(levels) > 1

    full = np.asarray(levels[0][0])
    for i, lv in enumerate(levels[1:], start=1):
        coarse = np.asarray(lv[0])
        for frac in (0.1, 0.35, 0.6, 0.85):
            fx = int(frac * full.shape[1])
            cx = int(frac * coarse.shape[1])
            assert coarse[0, cx] == full[0, fx], (
                f"level {i} is misregistered at x={frac:.0%}: "
                f"{coarse[0, cx]} != {full[0, fx]}")


def test_a_pyramid_with_no_positions_is_not_derivable_and_says_so():
    from squidxplorer._mosaic_source import fuse_region_pyramid

    meta = _meta({}, [0, 1])
    meta["n_z"] = 4
    assert fuse_region_pyramid(_Reader(), meta, "A1", "488") is None


def test_a_single_plane_acquisition_pyramid_has_no_singleton_z_axis():
    """Same rule as the flat stack: a one-position slider is clutter."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta(nz=1)
    levels, _step, nz = fuse_region_pyramid(_Reader(), meta, "A1", "488")

    assert nz == 1
    assert all(lv.ndim == 2 for lv in levels)
    for above, below in zip(levels, levels[1:]):
        assert below.shape[0] < above.shape[0] and below.shape[1] < above.shape[1]


def test_an_oversized_level_zero_is_still_refused_loudly():
    """A pyramid is not a licence to skip the plane budget guard."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (40000.0, 0.0)}, [0, 1],
                 frame=(40000, 40000), px=1.0)
    meta["n_z"] = 4
    with pytest.raises(MemoryError, match="plane budget"):
        fuse_region_pyramid(_Reader(), meta, "A1", "488", max_px=10_000_000)


def test_a_reader_that_cannot_identify_its_acquisition_is_refused():
    """A reader that cannot say which acquisition it reads would share a cache key with every
    other such reader."""
    from squidxplorer._mosaic_source import _source_token

    with pytest.raises(ValueError, match="_path"):
        _source_token(object())


def test_the_cache_is_keyed_by_the_acquisition_not_by_the_reader_object():
    """Keying on ``id(reader)`` would miss the cache for a second reader over the same
    acquisition, and (since CPython recycles ids) could serve another dataset's pixels."""
    from squidxplorer._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta()
    first = _StepReader(frame=(256, 256))
    second = _StepReader(frame=(256, 256))
    second._path = first._path              # same acquisition, different reader object

    a, _s, _n = fuse_region_pyramid(first, meta, "A1", "488")
    np.asarray(a[1][2])
    assert len(first.reads) > 0

    b, _s, _n = fuse_region_pyramid(second, meta, "A1", "488")
    np.asarray(b[1][2])
    assert second.reads == [], (
        "a second reader over the SAME acquisition re-read every FOV; the cache is keyed by "
        "object identity rather than by the dataset")
