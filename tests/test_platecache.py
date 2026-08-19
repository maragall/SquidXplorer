"""The plate cells survive a restart, and they are never written into somebody's data."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from squidxplorer import _platecache
from squidxplorer._budget import cache_budget
from squidxplorer._mosaic_source import MemoryBoundedLRUCache
from squidxplorer._platecache import CellTile, PlateCellCache


def _acquisition(tmp_path: Path, name: str = "acq") -> Path:
    """A folder shaped enough like an acquisition for the token to have something to stat."""
    root = tmp_path / name
    (root / "0").mkdir(parents=True)
    (root / "acquisition.yaml").write_text("objective:\n  pixel_size_um: 0.325\n")
    (root / "coordinates.csv").write_text("region,x (mm),y (mm),z (mm)\nA1,1.0,1.0,\n")
    return root


def _cache(tmp_path: Path, experiment: Path, **kw) -> PlateCellCache:
    return PlateCellCache(experiment, cell_px=88, channels=["c0", "c1"], dtype=np.uint16,
                          root=tmp_path / "cachedir", **kw)


def _cell(value: int = 7, shape=(2, 88, 88)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint16)


def _tree(root: Path) -> set:
    """Every path under *root*, for the "nothing was written here" assertion."""
    out = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in list(dirnames) + list(filenames):
            out.add(str(Path(dirpath) / name))
    return out


def test_a_published_cell_comes_back_with_its_pixels_and_its_box(tmp_path):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    assert cache.put("A1", _cell(11), (4, 6, 88 - 4, 88 - 6))
    got = cache.get("A1")
    assert got is not None
    assert np.array_equal(np.asarray(got), _cell(11))
    assert got.box == (4, 6, 84, 82), "the content box must travel WITH the pixels"


def test_the_ram_tier_answers_after_the_file_is_gone(tmp_path):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(3), (0, 0, 88, 88))
    cache.path_for("A1").unlink()
    assert cache.get("A1") is not None, "the RAM tier is not in front of the disk tier"


def test_the_disk_tier_answers_a_brand_new_process(tmp_path):
    exp = _acquisition(tmp_path)
    _cache(tmp_path, exp).put("A1", _cell(5), (0, 0, 88, 88))
    _platecache.clear_memory_tier()
    got = _cache(tmp_path, exp).get("A1")
    assert got is not None and np.array_equal(np.asarray(got), _cell(5))


def test_a_changed_store_yields_a_new_token_so_the_stale_cell_is_never_LOOKED_UP(tmp_path):
    exp = _acquisition(tmp_path)
    old = _cache(tmp_path, exp)
    old.put("A1", _cell(9), (0, 0, 88, 88))
    stale_file = old.path_for("A1")
    assert stale_file.exists()

    os.utime(exp / "coordinates.csv", (1_000_000, 1_000_000))
    _platecache.clear_memory_tier()
    fresh = _cache(tmp_path, exp)
    assert fresh.token != old.token, "a changed store produced the same token"
    assert fresh.get("A1") is None, "a stale cell was served after the store changed"
    assert stale_file.exists(), "the stale entry should be unreachable, not swept"


def test_the_token_covers_the_channel_list_and_the_cell_size_too(tmp_path):
    exp = _acquisition(tmp_path)
    a = PlateCellCache(exp, cell_px=88, channels=["c0", "c1"], dtype=np.uint16,
                       root=tmp_path / "c")
    b = PlateCellCache(exp, cell_px=88, channels=["c1", "c0"], dtype=np.uint16,
                       root=tmp_path / "c")
    c = PlateCellCache(exp, cell_px=44, channels=["c0", "c1"], dtype=np.uint16,
                       root=tmp_path / "c")
    assert len({a.token, b.token, c.token}) == 3


def test_a_growing_timepoint_folder_changes_the_token(tmp_path):
    exp = _acquisition(tmp_path)
    before = _platecache.plate_token(exp)
    (exp / "0" / "A1_0_0_ch.tiff").write_bytes(b"x")
    assert _platecache.plate_token(exp) != before


def test_the_token_costs_a_bounded_number_of_stats_whatever_the_plate_holds(tmp_path, monkeypatch):
    exp = _acquisition(tmp_path)
    for i in range(200):
        (exp / "0" / f"A{i}_0_0_ch.tiff").write_bytes(b"x")

    calls = []
    real_stat = os.stat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: (calls.append(p), real_stat(p, *a, **k))[1])
    _platecache.plate_token(exp)
    assert len(calls) <= _platecache.MAX_TOKEN_STATS, (
        f"the token cost {len(calls)} stats; it must stay bounded regardless of plate size")


def test_the_publish_is_a_temp_file_and_an_os_replace(tmp_path, monkeypatch):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    seen = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append((str(a), str(b))),
                                                     real_replace(a, b))[1])
    cache.put("A1", _cell(1), (0, 0, 88, 88))
    assert seen, "the cell was not published through os.replace"
    src, dst = seen[0]
    assert src.endswith(".tmp") and dst == str(cache.path_for("A1"))
    assert Path(src).parent == Path(dst).parent, (
        "the temp must sit beside the destination: os.replace is only atomic within one filesystem")


def test_a_failed_write_leaves_the_previous_cell_intact_and_no_debris(tmp_path, monkeypatch):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(2), (0, 0, 88, 88))

    def boom(*a, **k):
        raise OSError("no space left on device")
    monkeypatch.setattr(np, "savez", boom)
    assert cache.put("A1", _cell(99), (0, 0, 88, 88)) is False

    _platecache.clear_memory_tier()
    got = cache.get("A1")
    assert got is not None and int(np.asarray(got)[0, 0, 0]) == 2, "the published cell was damaged"
    assert not [p for p in cache.dir.iterdir() if p.name.endswith(".tmp")], "a temp was stranded"


def test_a_damaged_entry_is_a_miss_and_not_an_exception(tmp_path):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(4), (0, 0, 88, 88))
    _platecache.clear_memory_tier()
    cache.path_for("A1").write_bytes(b"not an npz")
    assert cache.get("A1") is None
    assert not cache.path_for("A1").exists(), "the damaged entry should be dropped, not re-read"


def test_a_stale_generation_is_pruned_on_the_next_publish(tmp_path):
    exp = _acquisition(tmp_path)
    old = _cache(tmp_path, exp)
    old.put("A1", _cell(1), (0, 0, 88, 88))
    old.put("A2", _cell(1), (0, 0, 88, 88))
    generations = old.dir.parent

    os.utime(exp / "acquisition.yaml", (2_000_000, 2_000_000))
    new = _cache(tmp_path, exp)
    assert len(list(generations.iterdir())) == 1
    new.put("A1", _cell(2), (0, 0, 88, 88))
    assert [p.name for p in generations.iterdir()] == [new.token], "the stale generation survived"


def test_a_finished_pass_compacts_into_ONE_memory_mapped_page(tmp_path):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    regions = [f"A{i}" for i in range(8)]
    for i, r in enumerate(regions):
        cache.put(r, _cell(i + 1)[:, :40, :50], (2, 3, 40, 50))
    assert len(list(cache.dir.glob("*.npz"))) == 8

    assert cache.pack(regions) is True
    assert cache.pack_array_path.exists() and cache.pack_index_path.exists()
    assert not list(cache.dir.glob("*.npz")), "the per-well files were not compacted away"

    _platecache.clear_memory_tier()
    fresh = _cache(tmp_path, exp)
    for i, r in enumerate(regions):
        hit = fresh.get(r)
        assert hit is not None and hit.box == (2, 3, 40, 50)
        assert np.array_equal(np.asarray(hit), _cell(i + 1)[:, :40, :50])


def test_the_page_is_MAPPED_and_not_read_into_the_heap(tmp_path):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(1), (0, 0, 88, 88))
    cache.pack(["A1"])
    _platecache.clear_memory_tier()
    fresh = _cache(tmp_path, exp)
    hit = fresh.get("A1")
    base = np.asarray(hit)
    while base is not None and not isinstance(base, np.memmap):
        base = getattr(base, "base", None)
    assert isinstance(base, np.memmap), "the page was loaded into memory instead of mapped"


def test_an_incomplete_pass_is_NEVER_compacted(tmp_path):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(1), (0, 0, 88, 88))
    assert cache.pack(["A1", "A2"]) is False
    assert not cache.pack_index_path.exists()
    assert cache.get("A1") is not None, "the per-well cells must survive a refused compaction"


def test_a_page_from_another_generation_is_not_read(tmp_path):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(1), (0, 0, 88, 88))
    cache.pack(["A1"])
    forged = _cache(tmp_path, exp)
    forged.dir.mkdir(parents=True, exist_ok=True)
    forged.pack_array_path.write_bytes(cache.pack_array_path.read_bytes())
    index = json.loads(cache.pack_index_path.read_text())
    index["token"] = "0000000000000000"
    forged.pack_index_path.write_text(json.dumps(index))
    _platecache.clear_memory_tier()
    assert _cache(tmp_path, exp).get("A1") is None


def test_the_sidecar_is_JSON_and_never_pickle(tmp_path):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(1), (0, 0, 88, 88))
    cache.pack(["A1"])
    index = json.loads(cache.pack_index_path.read_text(encoding="utf-8"))
    assert index["regions"] == ["A1"] and index["t"] == 0, \
        "the sidecar records the timepoint this page is of; see PlateCellCache.pack"
    import inspect

    src = inspect.getsource(_platecache)
    assert "import pickle" not in src and "pickle.load" not in src, "the cache started unpickling"


def test_NOTHING_is_ever_written_under_the_experiment_root(tmp_path):
    exp = _acquisition(tmp_path)
    before = _tree(exp)
    cache = _cache(tmp_path, exp)
    for i in range(64):
        cache.put(f"A{i}", _cell(i % 7), (0, 0, 88, 88))
        cache.get(f"A{i}")
    assert cache.pack([f"A{i}" for i in range(64)]), "the compaction path must be covered too"
    assert _tree(exp) == before, "the plate cache wrote into the acquisition folder"
    assert cache.dir.exists() and str(exp) not in str(cache.dir.resolve())


def test_a_cache_root_pointed_inside_the_experiment_is_REFUSED(tmp_path):
    exp = _acquisition(tmp_path)
    with pytest.raises(RuntimeError, match="never writes into your data"):
        PlateCellCache(exp, cell_px=88, channels=["c0"], dtype=np.uint16, root=exp / "cache")


def test_the_default_root_is_the_platform_user_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(_platecache.ENV_DIR, raising=False)
    import platformdirs

    assert str(_platecache.cache_root()) == platformdirs.user_cache_dir("squidxplorer", "cephla")


def test_the_ram_tier_is_bounded_by_BYTES_from_the_measured_budget(tmp_path):
    from squidxplorer import _budget

    assert isinstance(_platecache._CELLS, MemoryBoundedLRUCache)
    assert _platecache.cache_budget is cache_budget, "the cache stopped using the measured budget"
    assert _budget.FLOOR_BYTES <= _platecache._CELLS.capacity_bytes <= _budget.CEILING_BYTES
    assert _platecache._CELLS.capacity_bytes != 192 << 20, \
        "that is record-zstack-viewer's hardcoded 192 MB; _budget argues why we do not ship it"


def test_the_ram_tier_evicts_by_bytes_and_the_disk_tier_still_answers(tmp_path, monkeypatch):
    exp = _acquisition(tmp_path)
    one = _cell(1).nbytes
    monkeypatch.setattr(_platecache, "_CELLS", MemoryBoundedLRUCache(4 * one))
    cache = _cache(tmp_path, exp)
    for i in range(20):
        cache.put(f"A{i}", _cell(i + 1), (0, 0, 88, 88))
    assert _platecache._CELLS.nbytes <= 4 * one, "the RAM tier grew past its byte bound"
    assert len(_platecache._CELLS) == 4
    first = cache.get("A0")
    assert first is not None and int(np.asarray(first)[0, 0, 0]) == 1


def test_the_cell_carries_its_box_through_every_numpy_view(tmp_path):
    t = CellTile(_cell(1), (2, 3, 40, 50))
    assert t[:, :10, :10].box == (2, 3, 40, 50)
    assert np.asarray(t).shape == (2, 88, 88)


def test_the_cache_can_be_turned_off(monkeypatch):
    monkeypatch.setenv(_platecache.ENV_ENABLED, "0")
    assert _platecache.enabled() is False
    assert PlateCellCache.for_reader(object(), {}, cell_px=88) is None


def test_a_reader_with_no_path_degrades_to_uncached_rather_than_raising():
    assert PlateCellCache.for_reader(object(), {"channels": [], "dtype": "uint16"},
                                     cell_px=88) is None


pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication            # noqa: E402

import squidxplorer._viewer as V                        # noqa: E402
import squidxplorer._workers as W                       # noqa: E402

FRAME = (8, 8)
CHANNELS = ["c0", "c1"]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _CountingReader:
    """Counts every plane the preview pass pulls out of the acquisition. That count is the cost."""

    def __init__(self, path, boom_after: int = 0):
        self._path = str(path)
        self.reads = 0
        self._boom_after = int(boom_after)

    def read(self, region, fov, channel, z_level, time_point=0):
        self.reads += 1
        if self._boom_after and self.reads > self._boom_after:
            raise OSError("disk gone")
        return np.full(FRAME, 100 + 10 * int(fov) + CHANNELS.index(str(channel)), dtype=np.uint16)


def _meta(positions=None) -> dict:
    return {"channels": [{"name": c} for c in CHANNELS], "dtype": "uint16", "z_levels": [0, 1, 2],
            "regions": ["A1", "A2"], "fovs_per_region": {"A1": [0], "A2": [0]},
            "frame_shape": FRAME, "pixel_size_um": 1.0,
            "fov_positions_um": dict(positions or {})}


def _run(worker) -> list:
    """Run the worker in this thread and collect (ri, ci, region, tile, box)."""
    got: list = []
    worker.tileReady.connect(lambda *a: got.append(a))
    worker.run()
    return got


def test_reopening_a_plate_reads_NOTHING_from_the_acquisition(qapp, tmp_path):
    exp = _acquisition(tmp_path)
    meta, idx = _meta(), {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}}

    cold = _CountingReader(exp)
    first = _run(W._PreviewWorker(cold, meta, idx, ["A1", "A2"], cache=_cache(tmp_path, exp)))
    assert cold.reads == len(CHANNELS) * 2, "the cold open must read one plane per channel per well"

    _platecache.clear_memory_tier()
    warm = _CountingReader(exp)
    worker = W._PreviewWorker(warm, meta, idx, ["A1", "A2"], cache=_cache(tmp_path, exp))
    second = _run(worker)
    assert warm.reads == 0, "the reopen re-read the acquisition"
    assert worker.cache_hits == 2 and worker.cache_reads == 0
    assert [a[2] for a in second] == [a[2] for a in first]
    for a, b in zip(first, second):
        assert np.array_equal(np.asarray(a[3]), np.asarray(b[3])), "the replayed cell differs"


def test_a_cached_mosaic_replays_with_its_CONTENT_BOX_not_the_whole_cell(qapp, tmp_path):
    exp = _acquisition(tmp_path)
    positions = {("A1", 0): (0.0, 0.0), ("A1", 1): (4.0, 0.0),
                 ("A2", 0): (100.0, 0.0), ("A2", 1): (104.0, 0.0)}
    meta = _meta(positions)
    meta["fovs_per_region"] = {"A1": [0, 1], "A2": [0, 1]}
    idx = {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}}

    first = _run(W._PreviewWorker(_CountingReader(exp), meta, idx, ["A1", "A2"],
                                  cache=_cache(tmp_path, exp)))
    assert len(first) == 4 and all(a[4] is not None for a in first), "this fixture must mosaic"
    union = first[0][4]
    for _ri, _ci, region, _tile, box in first:
        if region == "A1":
            union = V._box_union(union, box)

    _platecache.clear_memory_tier()
    replay = _run(W._PreviewWorker(_CountingReader(exp), meta, idx, ["A1", "A2"],
                                   cache=_cache(tmp_path, exp)))
    assert len(replay) == 2, "a cached mosaic replays as ONE tile per well, not one per FOV"
    box = replay[0][4]
    assert box == union, f"the replayed box {box} is not the mosaic's own extent {union}"
    assert box[2] < 88, "the replay covers the whole cell, padding included"
    assert np.asarray(replay[0][3]).shape == (len(CHANNELS), box[2], box[3])


def test_a_preview_that_FAILS_caches_nothing(qapp, tmp_path):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    reader = _CountingReader(exp, boom_after=1)
    worker = W._PreviewWorker(reader, _meta(), {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}},
                              ["A1", "A2"], cache=cache)
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert failures, "the failure must still be named"
    assert not cache.dir.exists() or not list(cache.dir.glob("*.npz")), \
        "a preview that could not finish published a cell anyway"


def test_the_plate_preview_actually_goes_through_the_cache():
    import inspect

    src = (inspect.getsource(W._PreviewWorker.run)
           + inspect.getsource(W._PreviewWorker._run_body))
    assert "_replay_cached" in src, "the preview stopped consulting the cache"
    assert "_remember" in src, "the preview stopped filling the cache"
    assert "capture_stdout_to_log" in src, "the preview stopped capturing print() into the log"


class _TimeReader:
    """A reader whose pixels NAME their timepoint, so a stuck frame is visible, not inferred."""

    def __init__(self, path):
        self._path = str(path)
        self.reads = 0
        self.reads_at: dict = {}

    def read(self, region, fov, channel, z_level, time_point=0):
        self.reads += 1
        self.reads_at[int(time_point)] = self.reads_at.get(int(time_point), 0) + 1
        return np.full(FRAME, 1000 * (int(time_point) + 1) + CHANNELS.index(str(channel)), dtype=np.uint16)


def _preview_cells(worker, cell_px: int = 88) -> dict:
    """``{region: cell}`` from one preview pass -- the PIXELS the plate would paint."""
    cells: dict = {}
    for _ri, _ci, region, tile, box in _run(worker):
        arr = np.asarray(tile)
        canvas = cells.get(region)
        if canvas is None:
            canvas = cells[region] = np.zeros((arr.shape[0], cell_px, cell_px), dtype=arr.dtype)
        top, left = (0, 0) if box is None else (int(box[0]), int(box[1]))
        canvas[:, top:top + arr.shape[1], left:left + arr.shape[2]] = arr
    return cells


def test_a_cell_is_identified_by_its_TIMEPOINT_as_well_as_its_region(tmp_path):
    exp = _acquisition(tmp_path)
    at0 = _cache(tmp_path, exp)
    at1 = _cache(tmp_path, exp, time_point=1)
    at0.put("A1", _cell(10), (0, 0, 88, 88))

    assert at1.get("A1") is None, "timepoint 1 was served timepoint 0's cell"
    at1.put("A1", _cell(20), (0, 0, 88, 88))
    assert np.asarray(at0.get("A1")).max() == 10, "publishing t=1 overwrote t=0's cell"
    assert np.asarray(at1.get("A1")).max() == 20

    _platecache.clear_memory_tier()
    assert np.asarray(_cache(tmp_path, exp).get("A1")).max() == 10
    assert np.asarray(_cache(tmp_path, exp, time_point=1).get("A1")).max() == 20
    assert at0.path_for("A1") != at1.path_for("A1"), "two timepoints share one file"


def test_the_timepoint_is_in_the_KEY_and_not_in_the_TOKEN(tmp_path):
    exp = _acquisition(tmp_path)
    at0, at1 = _cache(tmp_path, exp), _cache(tmp_path, exp, time_point=1)
    assert at0.token == at1.token and at0.dir == at1.dir

    at0.put("A1", _cell(1), (0, 0, 88, 88))
    at1.put("A1", _cell(2), (0, 0, 88, 88))
    _platecache.clear_memory_tier()
    assert _cache(tmp_path, exp).get("A1") is not None, \
        "visiting timepoint 1 pruned timepoint 0's cells"

    os.utime(exp / "coordinates.csv", (1_000_000, 1_000_000))
    _platecache.clear_memory_tier()
    assert (_cache(tmp_path, exp).get("A1") is None
            and _cache(tmp_path, exp, time_point=1).get("A1") is None), \
        "a changed acquisition must invalidate EVERY timepoint, not just the live one"


def test_cells_written_before_the_re_key_are_unreachable_and_then_DELETED(tmp_path, monkeypatch):
    exp = _acquisition(tmp_path)
    monkeypatch.setattr(_platecache, "FORMAT_VERSION", 1)
    old = _cache(tmp_path, exp)
    old.put("A1", _cell(42), (0, 0, 88, 88))
    old_dir = old.dir
    assert old_dir.exists()

    monkeypatch.undo()
    _platecache.clear_memory_tier()
    new = _cache(tmp_path, exp)
    assert new.dir != old_dir, "the re-keyed cache reads the pre-timepoint generation's directory"
    assert new.get("A1") is None, "a cell with no timepoint was served under one"

    new.put("A1", _cell(7), (0, 0, 88, 88))
    assert not old_dir.exists(), "the pre-timepoint generation was left under $HOME forever"


def test_the_packed_page_is_per_timepoint(tmp_path):
    exp = _acquisition(tmp_path)
    at0, at1 = _cache(tmp_path, exp), _cache(tmp_path, exp, time_point=1)
    at0.put("A1", _cell(3), (0, 0, 88, 88))
    at1.put("A1", _cell(4), (0, 0, 88, 88))
    assert at0.pack(["A1"]) and at1.pack(["A1"])
    assert at0.pack_array_path != at1.pack_array_path
    assert json.loads(at1.pack_index_path.read_text())["t"] == 1

    _platecache.clear_memory_tier()
    assert np.asarray(_cache(tmp_path, exp).get("A1")).max() == 3
    assert np.asarray(_cache(tmp_path, exp, time_point=1).get("A1")).max() == 4


def test_a_page_whose_sidecar_names_ANOTHER_timepoint_is_not_read(tmp_path):
    exp = _acquisition(tmp_path)
    at1 = _cache(tmp_path, exp, time_point=1)
    at1.put("A1", _cell(4), (0, 0, 88, 88))
    at1.pack(["A1"])
    index = json.loads(at1.pack_index_path.read_text())
    index["t"] = 0
    at1.pack_index_path.write_text(json.dumps(index))
    _platecache.clear_memory_tier()
    assert _cache(tmp_path, exp, time_point=1).get("A1") is None


def test_the_plate_CELL_at_t1_differs_from_the_cell_at_t0(qapp, tmp_path):
    exp = _acquisition(tmp_path)
    meta, idx = _meta(), {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}}
    reader = _TimeReader(exp)

    at0 = _preview_cells(W._PreviewWorker(reader, meta, idx, ["A1", "A2"],
                                          cache=_cache(tmp_path, exp), time_point=0))
    at1 = _preview_cells(W._PreviewWorker(reader, meta, idx, ["A1", "A2"],
                                          cache=_cache(tmp_path, exp, time_point=1), time_point=1))

    assert set(at0) == set(at1) == {"A1", "A2"}
    for region in ("A1", "A2"):
        assert not np.array_equal(at0[region], at1[region]), \
            f"{region}'s plate cell did not move with the timepoint"
        assert at0[region].max() == 1001 and at1[region].max() == 2001, \
            "the cell is not the frame the reader was asked for"

    _platecache.clear_memory_tier()
    cold = _TimeReader(exp)
    for t, expected in ((1, at1), (0, at0)):
        replayed = _preview_cells(W._PreviewWorker(cold, meta, idx, ["A1", "A2"],
                                                   cache=_cache(tmp_path, exp, time_point=t), time_point=t))
        assert cold.reads == 0, "the replay re-read the acquisition"
        for region in ("A1", "A2"):
            assert np.array_equal(replayed[region], expected[region]), \
                f"{region}: the reopened plate showed another frame at t={t}"


def test_the_preview_READS_the_timepoint_it_was_asked_for(qapp, tmp_path):
    exp = _acquisition(tmp_path)
    reader = _TimeReader(exp)
    _run(W._PreviewWorker(reader, _meta(), {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}},
                          ["A1", "A2"], cache=None, time_point=2))
    assert set(reader.reads_at) == {2}, f"the preview read timepoints {sorted(reader.reads_at)}"


def test_stepping_BACK_to_a_visited_timepoint_reads_NOTHING(qapp, tmp_path):
    exp = _acquisition(tmp_path)
    meta, idx = _meta(), {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}}
    reader = _TimeReader(exp)
    per_pass = len(CHANNELS) * len(meta["regions"])

    _run(W._PreviewWorker(reader, meta, idx, ["A1", "A2"], cache=_cache(tmp_path, exp), time_point=0))
    assert reader.reads == per_pass, "the first visit must read the plate"
    _run(W._PreviewWorker(reader, meta, idx, ["A1", "A2"],
                          cache=_cache(tmp_path, exp, time_point=1), time_point=1))
    assert reader.reads == 2 * per_pass, "a NEW timepoint must read the plate"

    back = W._PreviewWorker(reader, meta, idx, ["A1", "A2"], cache=_cache(tmp_path, exp), time_point=0)
    _run(back)
    assert reader.reads == 2 * per_pass, "stepping back to t=0 re-read the acquisition"
    assert back.cache_hits == 2 and back.cache_reads == 0


def test_a_preview_handed_a_cache_for_ANOTHER_timepoint_refuses_to_start(qapp, tmp_path):
    exp = _acquisition(tmp_path)
    with pytest.raises(ValueError, match="wrong frame"):
        W._PreviewWorker(_TimeReader(exp), _meta(), {"A1": {"rc": (0, 0)}}, ["A1"],
                         cache=_cache(tmp_path, exp), time_point=1)


def test_the_plate_previews_the_timepoint_the_BAR_says():
    import inspect

    assert "t=self.time_point" in inspect.getsource(V.PlateWindow._start_preview), \
        "the plate's preview stopped carrying the bar's timepoint"
    assert "_start_preview" in inspect.getsource(V.PlateWindow._return_to_raw), \
        "returning to raw stopped restarting the preview, so a timepoint change repaints nothing"
    assert "_return_to_raw" in inspect.getsource(V.PlateWindow._on_time_point_changed), \
        "a timepoint change stopped asking the plate to re-read"


FIXTURE_5D = Path("~/Downloads/sim_5d_2x2_t3").expanduser()


@pytest.mark.skipif(not FIXTURE_5D.exists(),
                    reason=f"{FIXTURE_5D} is absent (build it: tools/make_5d_fixture.py)")
def test_the_plate_cell_follows_t_on_the_REAL_5D_acquisition(qapp, tmp_path):
    from squidxplorer.reader import open_reader

    reader = open_reader(FIXTURE_5D)
    meta = reader.metadata
    assert meta["n_t"] == 3, "this fixture is supposed to be the multi-timepoint one"
    order = list(meta["regions"])
    idx = {r: {"rc": (i // 2, i % 2), "idx": i} for i, r in enumerate(order)}

    def _cache_at(t):
        return PlateCellCache.for_reader(reader, meta, cell_px=88, time_point=t,
                                         root=tmp_path / "cachedir")

    cells = {}
    for t in range(3):
        cells[t] = _preview_cells(
            W._PreviewWorker(reader, meta, idx, order, cache=_cache_at(t), time_point=t))
        assert set(cells[t]) == set(order)

    for region in order:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            assert not np.array_equal(cells[a][region], cells[b][region]), \
                f"{region}: the plate cell at t={a} is identical to the one at t={b}"

    _platecache.clear_memory_tier()
    for t in range(3):
        worker = W._PreviewWorker(reader, meta, idx, order, cache=_cache_at(t), time_point=t)
        replayed = _preview_cells(worker)
        assert worker.cache_hits == len(order) and worker.cache_reads == 0
        for region in order:
            assert np.array_equal(replayed[region], cells[t][region]), \
                f"{region}: the cached cell for t={t} came back as another frame"
