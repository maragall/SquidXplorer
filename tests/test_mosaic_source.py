"""Fused mosaics for pane 2 — the unit displayed is a MOSAIC, never a single FOV.

Qt-free: these exercise the loader and the geometry, which is where the wrongness would be.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from squidmip._mosaic_source import (
    fuse_region_mosaic,
    level_paths,
    mosaic_bbox_um,
    open_pyramid,
)


class _Reader:
    """Reads a distinct constant per FOV so placement is checkable by value."""

    #: Every real reader exposes the acquisition it reads; the plane cache keys on it so a stale
    #: entry can never be served to a different dataset. A fake without one is refused, loudly.
    _counter = itertools.count()

    def __init__(self, frame=(4, 6), values=None, fail=()):
        self.frame = frame
        self.values = values or {}
        self.fail = set(fail)
        self._path = f"/fake/acquisition/{next(_Reader._counter)}"

    def read(self, region, fov, channel, z, t=0):
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


# ------------------------------------------------------------------ fusing


def test_two_fovs_are_placed_side_by_side_by_stage_position():
    # 6 px wide frame, 2 µm/px -> 12 µm; put FOV 1 exactly one frame to the right.
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    out = fuse_region_mosaic(_Reader(), meta, "A1", "488")
    assert out is not None
    mosaic, step = out
    assert step == 1.0
    assert mosaic.shape == (4, 12)
    assert np.all(mosaic[:, :6] == 1)     # FOV 0
    assert np.all(mosaic[:, 6:] == 2)     # FOV 1


def test_overlap_is_covered_not_left_as_a_hole():
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (6.0, 0.0)}, [0, 1])
    mosaic, _ = fuse_region_mosaic(_Reader(), meta, "A1", "488")
    assert mosaic.shape == (4, 9)
    assert not (mosaic == 0).any(), "a placed mosaic must have no unwritten pixels"


def test_a_mosaic_is_not_derivable_without_positions_and_says_so_by_returning_none():
    """The same 'not derivable, do not guess' signal _mosaic_boxes returns {} for. A guessed
    layout is a WRONG picture, not a rough one."""
    meta = _meta({}, [0, 1])
    assert fuse_region_mosaic(_Reader(), meta, "A1", "488") is None

    meta_no_px = _meta({("A1", 0): (0.0, 0.0)}, [0])
    meta_no_px["pixel_size_um"] = 0
    assert fuse_region_mosaic(_Reader(), meta_no_px, "A1", "488") is None


def test_an_unreadable_fov_leaves_a_hole_rather_than_shifting_its_neighbours():
    """A silent skip would slide every later FOV over by one frame and produce a mosaic that
    looks fine and is wrong."""
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    mosaic, _ = fuse_region_mosaic(_Reader(fail=[0]), meta, "A1", "488")
    assert mosaic.shape == (4, 12)
    assert np.all(mosaic[:, :6] == 0)     # the hole stays where the unreadable FOV was
    assert np.all(mosaic[:, 6:] == 2)     # FOV 1 did NOT move left


def test_a_large_region_is_decimated_rather_than_truncated():
    """Bounding RAM must not change which region the mosaic covers."""
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (2000.0, 0.0)}, [0, 1],
                 frame=(1000, 1000), px=1.0)
    mosaic, step = fuse_region_mosaic(_Reader(frame=(1000, 1000)), meta, "A1", "488",
                                      max_px=1000)
    assert step > 1
    assert max(mosaic.shape) <= 1000
    # aspect preserved: the full extent is 1000 x 3000, so the mosaic stays 1:3
    assert mosaic.shape[1] == pytest.approx(mosaic.shape[0] * 3, rel=0.02)


# ------------------------------------------------------------------ world placement


def test_bbox_um_is_the_regions_stage_extent():
    meta = _meta({("A1", 0): (100.0, 50.0), ("A1", 1): (112.0, 50.0)}, [0, 1])
    x0, y0, x1, y1 = mosaic_bbox_um(meta, "A1")
    assert (x0, y0) == (100.0, 50.0)
    # extent is 4 px tall, 12 px wide at 2 µm/px -> 8 x 24 µm
    assert (x1 - x0, y1 - y0) == (24.0, 8.0)


def test_bbox_um_is_none_when_placement_is_not_derivable():
    assert mosaic_bbox_um(_meta({}, [0]), "A1") is None


# ------------------------------------------------------------------ pyramid loading


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


# ------------------------------------------------------------ lazy z stacks


class _CountingReader(_Reader):
    """Counts reads so laziness is provable rather than asserted."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reads = 0

    def read(self, region, fov, channel, z, t=0):
        self.reads += 1
        return np.full(self.frame, z + 1, dtype=np.uint16)


def test_a_zstack_becomes_a_lazy_z_y_x_array():
    from squidmip._mosaic_source import fuse_region_stack

    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    meta["n_z"] = 5
    data, step, nz = fuse_region_stack(_Reader(), meta, "A1", "488")

    assert nz == 5
    assert data.shape == (5, 4, 12)
    assert hasattr(data, "compute"), "the stack must stay lazy"


def test_only_the_requested_plane_is_ever_materialised():
    """The architecture taken from record-zstack-viewer: requests cost only what is visible.
    Eagerly fusing every z would turn a 10-plane 28-FOV region into ~10x the reads on open."""
    from squidmip._mosaic_source import fuse_region_stack

    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    meta["n_z"] = 8
    reader = _CountingReader()
    data, _step, _nz = fuse_region_stack(reader, meta, "A1", "488")

    after_probe = reader.reads          # one plane is probed for shape/dtype
    plane = np.asarray(data[3])         # ask for exactly one z

    assert plane.shape == (4, 12)
    assert np.all(plane == 4)           # z=3 -> value 4, so the right plane was fused
    assert reader.reads - after_probe == 2, "materialising one z must read one z, once per FOV"


def test_a_single_plane_acquisition_gets_no_singleton_z_axis():
    """A one-position slider is clutter; record-zstack-viewer relies on the viewer hiding
    singleton sliders and napari does the same when the axis simply is not there."""
    from squidmip._mosaic_source import fuse_region_stack

    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    meta["n_z"] = 1
    data, _step, nz = fuse_region_stack(_Reader(), meta, "A1", "488")

    assert nz == 1
    assert data.ndim == 2


def test_an_oversized_plane_is_refused_loudly_rather_than_paging_the_machine():
    from squidmip._mosaic_source import fuse_region_stack

    # A genuinely huge extent, with decimation effectively disabled.
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (40000.0, 0.0)}, [0, 1],
                 frame=(40000, 40000), px=1.0)
    meta["n_z"] = 4
    with pytest.raises(MemoryError, match="plane budget"):
        fuse_region_stack(_Reader(), meta, "A1", "488", max_px=10_000_000)


def test_a_stack_with_no_positions_is_still_not_derivable():
    from squidmip._mosaic_source import fuse_region_stack

    meta = _meta({}, [0, 1])
    meta["n_z"] = 4
    assert fuse_region_stack(_Reader(), meta, "A1", "488") is None


# ------------------------------------------------------- lazy RAW mosaic pyramids
#
# The written-OME-Zarr path has always handed napari a pyramid (``open_pyramid``). The RAW
# preview path did not: it handed over full-resolution fused planes, so napari uploaded a
# 5731x4793 uint16 plane per channel (54.9 MB) and re-fused all four on every z step. These
# tests pin the preview path to the SAME shape as the path that was already right.


class _StepReader(_Reader):
    """Records the decimation each read is asked to serve, so the fusion strategy is provable.

    ``fuse_region_mosaic`` strides the frame AFTER reading, so the reader cannot see the step
    directly; what it can see is HOW MANY times it was asked. Combined with the recorded
    ``max_px`` of each fuse (see ``_record_fuse_calls``), that is enough to tell a per-level
    fuse from a coarsen-over-level-0.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reads: list = []

    def read(self, region, fov, channel, z, t=0):
        self.reads.append((region, fov, channel, z, t))
        return np.full(self.frame, z + 1, dtype=np.uint16)


@pytest.fixture
def _record_fuse_calls(monkeypatch):
    """Record which pyramid levels each fuse pass actually produced, by their ``max_px``.

    This is the one internal worth spying on: "each level is fused DIRECTLY from the FOV tiles"
    versus "level 0 is fused and then reduced" is invisible in the output shapes and is the whole
    performance claim. Measured on the real 10x set: direct per-level is 3.2x cheaper (19 ms vs
    60 ms for a 956x799 level), because the coarsen path must allocate and paste the 54.9 MB
    level-0 plane first.
    """
    from squidmip import _mosaic_source as ms

    calls: list = []
    real = ms._fuse_levels

    def spy(reader, meta, region, channel, z, t, plans):
        calls.append([p[0] for p in plans])
        return real(reader, meta, region, channel, z, t, plans)

    monkeypatch.setattr(ms, "_fuse_levels", spy)
    return calls


def _pyr_meta(nz=6, frame=(256, 256), px=1.0, n=16):
    """A region wide enough that a pyramid has somewhere to go: 16 FOVs in a row, 256x4096 px.

    Deliberately not the 4x6 toy the flat-stack tests use. A mosaic smaller than the coarsest
    rung has NO pyramid to build — every level would be a duplicate of level 0 and the guard
    drops them all, which is correct behaviour and useless for testing pyramids.
    """
    positions = {("A1", i): (i * frame[1] * px, 0.0) for i in range(n)}
    meta = _meta(positions, list(range(n)), frame=frame, px=px)
    meta["n_z"] = nz
    meta["dz_um"] = 1.5
    return meta


def test_the_raw_preview_returns_a_pyramid_of_strictly_decreasing_levels():
    """napari's ``multiscale=True`` contract: a LIST, highest resolution first, each level
    strictly smaller than the one above it in both displayed axes."""
    from squidmip._mosaic_source import fuse_region_pyramid

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
    from squidmip._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta(nz=6)
    levels, _step, nz = fuse_region_pyramid(_Reader(), meta, "A1", "488")

    for i, lv in enumerate(levels):
        assert lv.ndim == 3, f"level {i} lost the z axis: {lv.shape}"
        assert lv.shape[0] == nz == 6, f"level {i} changed the z length: {lv.shape}"


def test_every_pyramid_level_is_lazy():
    """Level 0 of the real region is 54.9 MB and there are 10 z and 4 channels. If building the
    pyramid materialised anything, opening the region would cost 2.2 GB before the first paint."""
    from squidmip._mosaic_source import fuse_region_pyramid

    reader = _StepReader()
    levels, _step, _nz = fuse_region_pyramid(reader, _pyr_meta(), "A1", "488")

    assert all(hasattr(lv, "compute") for lv in levels)
    assert reader.reads == [], "building the pyramid must read NOTHING"


def test_a_coarse_level_is_fused_directly_and_never_materialises_level_zero(_record_fuse_calls):
    """THE performance property. Coarsening over the lazy level-0 graph would force the full
    54.9 MB plane to be pasted just to throw 63/64 of it away; fusing the level directly from
    the FOV tiles strides each frame on read instead. Measured 3.2x cheaper on the real set."""
    from squidmip._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta()
    levels, _step, _nz = fuse_region_pyramid(_Reader(), meta, "A1", "488")
    assert len(levels) >= 3

    _record_fuse_calls.clear()
    coarse = np.asarray(levels[2][1])          # one z of the third level

    assert _record_fuse_calls, "computing a level must fuse something"
    produced = [px for plan in _record_fuse_calls for px in plan]
    # The FINEST thing produced must be the level asked for. Anything finer means a
    # higher-resolution plane was materialised only to be thrown away.
    level0_px = max(levels[0].shape[-2:])
    assert max(produced) < level0_px, (
        f"the fuse produced a level at max_px={max(produced)}, i.e. level 0 resolution "
        f"({level0_px} px) — level 0 was materialised to build a coarse level")
    assert coarse.shape == levels[2].shape[1:]


def test_fusing_a_level_also_yields_the_coarser_levels_from_the_same_decode():
    """napari pulls TWO levels per channel per z: the visible one, and the COARSEST one for the
    layer thumbnail. Measured on the real region — 216 frame decodes per z step instead of 108,
    which made the pyramid SLOWER per z than the flat stack it replaced.

    TIFF decode is whole-frame, so a coarse level is cheaper to paste but not cheaper to read.
    The fix is to yield every coarser level from the decode already in hand: level 3..6 of one
    (z, channel) together are ~2 MB, so the thumbnail costs a stride, not a second read pass.
    """
    from squidmip._mosaic_source import fuse_region_pyramid

    reader = _StepReader()
    levels, _step, _nz = fuse_region_pyramid(reader, _pyr_meta(), "A1", "488")
    assert len(levels) >= 3

    np.asarray(levels[1][2])                    # the "visible" level
    after_visible = len(reader.reads)
    assert after_visible > 0

    np.asarray(levels[-1][2])                   # the thumbnail level, SAME z
    assert len(reader.reads) == after_visible, (
        "the coarsest level re-decoded every FOV; it must come from the pass that already "
        "read them for the visible level")


def test_a_level_finer_than_the_one_requested_is_never_produced(_record_fuse_calls):
    """Yielding the COARSER levels is free; yielding finer ones would defeat the whole point."""
    from squidmip._mosaic_source import fuse_region_pyramid

    levels, _step, _nz = fuse_region_pyramid(_Reader(frame=(256, 256)), _pyr_meta(), "A1", "488")
    _record_fuse_calls.clear()
    np.asarray(levels[-1][0])                   # ask for the COARSEST level only

    produced = [px for plan in _record_fuse_calls for px in plan]
    coarsest_px = max(levels[-1].shape[-2:])
    assert max(produced) <= max(coarsest_px, 128), (
        f"asking for the coarsest level produced a finer one (max_px={max(produced)})")


def test_a_region_whose_every_fov_is_unreadable_is_an_error_not_a_blank_mosaic():
    """A hole where one FOV failed is a visible hole. A mosaic where EVERY FOV failed is not a
    picture at all, and handing napari a black plane would report a read failure as empty tissue.
    Six confirmed silent failures in this project say this must be loud."""
    from squidmip._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta(n=4)
    reader = _Reader(frame=(256, 256), fail=range(4))
    levels, _step, _nz = fuse_region_pyramid(reader, meta, "A1", "488")

    with pytest.raises(ValueError, match="no FOV.*could be read|unreadable"):
        np.asarray(levels[0][0])


def test_revisiting_a_z_plane_does_not_re_fuse_it():
    """Julio steps z back and forth. Without a cache every step re-reads every FOV of every
    channel — measured at 428 ms per step on the real region."""
    from squidmip._mosaic_source import fuse_region_pyramid

    reader = _StepReader()
    levels, _step, _nz = fuse_region_pyramid(reader, _pyr_meta(), "A1", "488")

    np.asarray(levels[1][2])
    first = len(reader.reads)
    assert first > 0

    np.asarray(levels[1][2])               # same z, same level
    assert len(reader.reads) == first, "a revisited (level, z) must come from the cache"

    np.asarray(levels[1][3])               # a DIFFERENT z must still be fused
    assert len(reader.reads) > first


def test_the_cache_is_bounded_in_bytes_and_evicts_rather_than_growing():
    """An unbounded cache is a slow memory leak wearing a performance costume. The bound is
    explicit and reported, and the oldest entry goes when it is reached."""
    from squidmip._mosaic_source import PYRAMID_CACHE_BYTES, fuse_region_pyramid

    assert isinstance(PYRAMID_CACHE_BYTES, int) and PYRAMID_CACHE_BYTES > 0

    reader = _StepReader()
    meta = _pyr_meta(nz=6)
    # A budget that holds roughly ONE level-0 plane (256x4096 uint16 = 2 MiB) and no more, so a
    # second z has to push the first out.
    levels, _step, _nz = fuse_region_pyramid(reader, meta, "A1", "488",
                                             cache_bytes=int(2.2 * 256 * 4096))

    np.asarray(levels[0][0])
    n1 = len(reader.reads)
    np.asarray(levels[0][1])               # evicts z=0
    np.asarray(levels[0][0])               # must be re-fused, not served stale
    assert len(reader.reads) > 2 * n1, "a full cache must evict, not grow without bound"


def test_a_plane_larger_than_the_whole_cache_is_a_loud_error_not_a_silent_no_op():
    """The ported ndviewer_light cache logs at DEBUG and returns when an item exceeds the budget.
    That is log-and-continue: the cache silently does nothing and the only symptom is that
    everything is slow. This repo has six confirmed silent failures already; make it audible."""
    from squidmip._mosaic_source import MemoryBoundedLRUCache

    cache = MemoryBoundedLRUCache(1024)
    with pytest.raises(ValueError, match="larger than the whole"):
        cache.put(("k",), np.zeros(4096, dtype=np.uint16))


def test_the_plane_cache_serialises_concurrent_writers():
    """_MosaicWorker is a QThread and dask may compute planes concurrently, so two threads can
    land in put() at once. Without a lock the byte accounting drifts and the bound stops being a
    bound — which is the whole point of the class.

    WHAT THIS CAN AND CANNOT PROVE. I could not make the drift fail reliably: 12 threads x 3000
    puts at a 1 microsecond switch interval stayed byte-consistent with the lock removed. A
    purely behavioural assertion here would therefore read green forever and be a comment rather
    than a test — this repo already shipped one of those and it was dead its whole life. So the
    mutual exclusion is asserted STRUCTURALLY (that does go red when the lock is removed), and
    the concurrent hammer below stays as a smoke test for the things it CAN catch: exceptions
    escaping a worker, and the byte bound being blown.
    """
    import threading

    from squidmip._mosaic_source import MemoryBoundedLRUCache

    cache = MemoryBoundedLRUCache(64 * 1024)
    assert isinstance(cache._lock, type(threading.Lock())), (
        "the cache must hold a real lock; dask workers call put() concurrently")

    plane = np.zeros(512, dtype=np.uint16)          # 1 KiB each; 64 fit
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
    # and the accounting must still MATCH what is actually held
    assert cache.nbytes == len(cache) * plane.nbytes


def test_a_mosaic_too_small_to_shrink_gets_one_level_not_a_degenerate_pyramid():
    """The guard ``open_pyramid`` already applies: napari needs strictly decreasing levels, so a
    level that does not shrink is dropped rather than handed over as a duplicate."""
    from squidmip._mosaic_source import fuse_region_pyramid

    meta = _meta({("A1", 0): (0.0, 0.0)}, [0], frame=(4, 6), px=2.0)
    meta["n_z"] = 3
    levels, _step, _nz = fuse_region_pyramid(_Reader(), meta, "A1", "488")

    assert len(levels) == 1, f"a 4x6 mosaic has no room for a second level: {[l.shape for l in levels]}"


def test_the_pyramid_levels_agree_with_the_full_resolution_picture():
    """A pyramid that renders a misregistered level looks fine in a shape assertion and wrong on
    screen. Each level must show the SAME picture, coarser — so a feature at a given fraction of
    the mosaic must land at that same fraction on every level."""
    from squidmip._mosaic_source import fuse_region_pyramid

    # 16 FOVs left-to-right, each a distinct constant: the picture is a ramp along x.
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
    from squidmip._mosaic_source import fuse_region_pyramid

    meta = _meta({}, [0, 1])
    meta["n_z"] = 4
    assert fuse_region_pyramid(_Reader(), meta, "A1", "488") is None


def test_a_single_plane_acquisition_pyramid_has_no_singleton_z_axis():
    """Same rule as the flat stack: a one-position slider is clutter."""
    from squidmip._mosaic_source import fuse_region_pyramid

    meta = _pyr_meta(nz=1)
    levels, _step, nz = fuse_region_pyramid(_Reader(), meta, "A1", "488")

    assert nz == 1
    assert all(lv.ndim == 2 for lv in levels)
    for above, below in zip(levels, levels[1:]):
        assert below.shape[0] < above.shape[0] and below.shape[1] < above.shape[1]


def test_an_oversized_level_zero_is_still_refused_loudly():
    """The plane budget guards the pyramid too — a pyramid is not a licence to skip it."""
    from squidmip._mosaic_source import fuse_region_pyramid

    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (40000.0, 0.0)}, [0, 1],
                 frame=(40000, 40000), px=1.0)
    meta["n_z"] = 4
    with pytest.raises(MemoryError, match="plane budget"):
        fuse_region_pyramid(_Reader(), meta, "A1", "488", max_px=10_000_000)


def test_a_reader_that_cannot_identify_its_acquisition_is_refused():
    """The cache key must name the DATASET. A reader that cannot say which acquisition it reads
    would share a key with every other such reader."""
    from squidmip._mosaic_source import _source_token

    with pytest.raises(ValueError, match="_path"):
        _source_token(object())


def test_the_cache_is_keyed_by_the_acquisition_not_by_the_reader_object():
    """Keying on ``id(reader)`` has two faults, and this pins both.

    A second reader over the same acquisition would miss the cache entirely — and worse, a cache
    entry can outlive the reader that produced it, CPython recycles ids, so a later reader over a
    DIFFERENT dataset could collide with a stale key and be served another acquisition's pixels.
    """
    from squidmip._mosaic_source import fuse_region_pyramid

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
