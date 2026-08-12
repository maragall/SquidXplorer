"""The plate cells survive a restart, and they are never written into somebody's data.

Gap 1 of the three-viewers review (Hongquan, 2026-07-28), which is also the blocker
``NEXT_STEPS.md`` records against the deep-zoom work. At HEAD ``platformdirs`` appeared nowhere in
``squidxplorer/`` and both plate producers re-derived every well on every open.

These tests pin the five properties the design is made of, and they are deliberately separate
because each one, missing, produces a different and quiet failure:

* **tiering** -- RAM first, then disk. A disk-only cache re-reads a file per well per repaint; a
  RAM-only cache is ``_recipe.ResultCache``, which dies with the process and is the thing gap 1
  says is not the answer.
* **the mtime token** -- a changed store yields a NEW token, so the stale entry is not deleted,
  it is unreachable. That is the ported design's whole trick: no invalidation pass to get wrong,
  and no lock.
* **the atomic publish** -- temp file plus ``os.replace``. Two windows on one plate, or a crash
  mid-write, must never leave a reader looking at half a file.
* **nothing under the experiment root, ever** -- asserted hard, and asserted twice. Squid
  experiments live on Dropbox, NAS and read-only mounts, and the README promises this tool never
  writes into your acquisition folder.
* **the byte bound** -- via ``MemoryBoundedLRUCache`` and ``_budget.cache_budget()``, NOT
  record-zstack-viewer's item count and not its hardcoded 192 MB. A cell is 62 KB on a 1536-well
  plate and 62 KB on a 4-well plate while the COUNT differs by 384x, so any item count is either
  a leak on the big plate or a no-op on the small one.

Plus the property the whole thing exists for: reopening a plate reads NOTHING from the
acquisition, and a coarse tile that measured 25 s becomes a lookup.
"""
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


# --- helpers ---------------------------------------------------------------------------------

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


# --- tiering ---------------------------------------------------------------------------------

def test_a_published_cell_comes_back_with_its_pixels_and_its_box(tmp_path):
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    assert cache.put("A1", _cell(11), (4, 6, 88 - 4, 88 - 6))
    got = cache.get("A1")
    assert got is not None
    assert np.array_equal(np.asarray(got), _cell(11))
    assert got.box == (4, 6, 84, 82), "the content box must travel WITH the pixels"


def test_the_ram_tier_answers_after_the_file_is_gone(tmp_path):
    """Tier 1 exists so a repaint is not a file read per well. Prove it is really in front."""
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(3), (0, 0, 88, 88))
    cache.path_for("A1").unlink()
    assert cache.get("A1") is not None, "the RAM tier is not in front of the disk tier"


def test_the_disk_tier_answers_a_brand_new_process(tmp_path):
    """Tier 2 is the point of the whole module: the cells must outlive the process."""
    exp = _acquisition(tmp_path)
    _cache(tmp_path, exp).put("A1", _cell(5), (0, 0, 88, 88))
    _platecache.clear_memory_tier()                       # i.e. a restart
    got = _cache(tmp_path, exp).get("A1")
    assert got is not None and np.array_equal(np.asarray(got), _cell(5))


# --- the mtime token ---------------------------------------------------------------------------

def test_a_changed_store_yields_a_new_token_so_the_stale_cell_is_never_LOOKED_UP(tmp_path):
    """Ported verbatim, and the reason there is no invalidation pass and no lock.

    The stale entry is not deleted by this mechanism -- it is UNREACHABLE, because the token that
    would address it is no longer the token that gets computed. That distinction is the design:
    deleting requires knowing what is stale, and knowing that requires a sweep and a lock.
    """
    exp = _acquisition(tmp_path)
    old = _cache(tmp_path, exp)
    old.put("A1", _cell(9), (0, 0, 88, 88))
    stale_file = old.path_for("A1")
    assert stale_file.exists()

    os.utime(exp / "coordinates.csv", (1_000_000, 1_000_000))       # the store changed
    _platecache.clear_memory_tier()
    fresh = _cache(tmp_path, exp)
    assert fresh.token != old.token, "a changed store produced the same token"
    assert fresh.get("A1") is None, "a stale cell was served after the store changed"
    assert stale_file.exists(), "the stale entry should be unreachable, not swept"


def test_the_token_covers_the_channel_list_and_the_cell_size_too(tmp_path):
    """A window with a different channel order must not read another window's cells."""
    exp = _acquisition(tmp_path)
    a = PlateCellCache(exp, cell_px=88, channels=["c0", "c1"], dtype=np.uint16,
                       root=tmp_path / "c")
    b = PlateCellCache(exp, cell_px=88, channels=["c1", "c0"], dtype=np.uint16,
                       root=tmp_path / "c")
    c = PlateCellCache(exp, cell_px=44, channels=["c0", "c1"], dtype=np.uint16,
                       root=tmp_path / "c")
    assert len({a.token, b.token, c.token}) == 3


def test_a_growing_timepoint_folder_changes_the_token(tmp_path):
    """Appending a plane moves the CONTAINING directory's mtime, not the root's.

    A token built from the root alone would keep serving a plate that has since grown, which is
    the exact failure a post-acquisition tool hits when the acquisition was not finished after all.
    """
    exp = _acquisition(tmp_path)
    before = _platecache.plate_token(exp)
    (exp / "0" / "A1_0_0_ch.tiff").write_bytes(b"x")
    assert _platecache.plate_token(exp) != before


def test_the_token_costs_a_bounded_number_of_stats_whatever_the_plate_holds(tmp_path, monkeypatch):
    """MEASURED, and the reason this token is plate-level rather than per-FOV.

    record-zstack-viewer stats 2 to 4 paths per FOV. On sim_1536wp that is 6144 stats: 12.8 ms on
    local APFS, but 14x that per stat on a Dropbox FileProvider mount (measured 28.8 us) and ~1 ms
    per stat on a cold network share, i.e. SIX SECONDS before the first pixel, on the very path
    this cache exists to make fast. So the token is bounded, and this test is what stops it
    quietly becoming O(wells) again.
    """
    exp = _acquisition(tmp_path)
    for i in range(200):                                   # a plate with a lot of everything
        (exp / "0" / f"A{i}_0_0_ch.tiff").write_bytes(b"x")

    calls = []
    real_stat = os.stat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: (calls.append(p), real_stat(p, *a, **k))[1])
    _platecache.plate_token(exp)
    assert len(calls) <= _platecache.MAX_TOKEN_STATS, (
        f"the token cost {len(calls)} stats; it must stay bounded regardless of plate size")


# --- the atomic publish -------------------------------------------------------------------------

def test_the_publish_is_a_temp_file_and_an_os_replace(tmp_path, monkeypatch):
    """A reader in another process sees the old file or the whole new one, never half of one."""
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
    """A full disk must degrade to "uncached", never to a truncated cell or a stranded temp."""
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
    """A cache must never be the reason pixels cannot be shown."""
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(4), (0, 0, 88, 88))
    _platecache.clear_memory_tier()
    cache.path_for("A1").write_bytes(b"not an npz")
    assert cache.get("A1") is None
    assert not cache.path_for("A1").exists(), "the damaged entry should be dropped, not re-read"


def test_a_stale_generation_is_pruned_on_the_next_publish(tmp_path):
    """The token-in-the-key design frees nothing by itself: without the prune, a store that
    changes ten times leaves ten full generations of every cell under $HOME."""
    exp = _acquisition(tmp_path)
    old = _cache(tmp_path, exp)
    old.put("A1", _cell(1), (0, 0, 88, 88))
    old.put("A2", _cell(1), (0, 0, 88, 88))
    generations = old.dir.parent

    os.utime(exp / "acquisition.yaml", (2_000_000, 2_000_000))
    new = _cache(tmp_path, exp)
    assert len(list(generations.iterdir())) == 1            # nothing published yet
    new.put("A1", _cell(2), (0, 0, 88, 88))
    assert [p.name for p in generations.iterdir()] == [new.token], "the stale generation survived"


# --- the compacted, memory-mapped page ---------------------------------------------------------

def test_a_finished_pass_compacts_into_ONE_memory_mapped_page(tmp_path):
    """Adopted from ``ndviewer_hcs/plate_stack.py``: the reopen wants every well at once.

    1536 wells is 1536 file opens in the per-well form, which measured 0.261 s to replay and
    1.59 s to seed the coarse rungs. The page turns both into one open plus slices.
    """
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
    """The property that lets a page exceed RAM. A read must be a view, not a copy of 96 MB."""
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
    """A partial plate must not become a page that claims to be the plate."""
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(1), (0, 0, 88, 88))
    assert cache.pack(["A1", "A2"]) is False
    assert not cache.pack_index_path.exists()
    assert cache.get("A1") is not None, "the per-well cells must survive a refused compaction"


def test_a_page_from_another_generation_is_not_read(tmp_path):
    """The token is in the page too, so a stale page is unreachable exactly like a stale cell."""
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
    """``ndviewer_hcs`` uses ``pickle.dump`` here. A cache under $HOME that is unpickled on open
    is an arbitrary-code-execution surface for anything that can write there."""
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    cache.put("A1", _cell(1), (0, 0, 88, 88))
    cache.pack(["A1"])
    index = json.loads(cache.pack_index_path.read_text(encoding="utf-8"))
    assert index["regions"] == ["A1"] and index["t"] == 0, \
        "the sidecar records the timepoint this page is of; see PlateCellCache.pack"
    import inspect

    src = inspect.getsource(_platecache)          # the docstring names pickle to reject it
    assert "import pickle" not in src and "pickle.load" not in src, "the cache started unpickling"


# --- never under the experiment root -------------------------------------------------------------

def test_NOTHING_is_ever_written_under_the_experiment_root(tmp_path):
    """The hard one. Squid experiments live on Dropbox, NAS and read-only mounts.

    A hidden sidecar written into somebody's data folder syncs to their whole lab, and the README
    promises this tool never does it. The whole tree is snapshotted, a full plate's worth of cells
    is published, and the tree must be byte-for-byte the same set of paths.
    """
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
    """The env override can name any directory, including the experiment. Refuse it loudly."""
    exp = _acquisition(tmp_path)
    with pytest.raises(RuntimeError, match="never writes into your data"):
        PlateCellCache(exp, cell_px=88, channels=["c0"], dtype=np.uint16, root=exp / "cache")


def test_the_default_root_is_the_platform_user_cache_dir(monkeypatch, tmp_path):
    """The choice ``user_cache_dir`` makes, stated as a test so it cannot drift to a sidecar.

    If compute or storage ever moves off this workstation the CONSTRAINT is workstation shaped
    and its reasoning, not its answer, ports: never write into a store you do not own. See the
    module docstring.
    """
    monkeypatch.delenv(_platecache.ENV_DIR, raising=False)
    import platformdirs

    assert str(_platecache.cache_root()) == platformdirs.user_cache_dir("squidxplorer", "cephla")


# --- the byte bound ------------------------------------------------------------------------------

def test_the_ram_tier_is_bounded_by_BYTES_from_the_measured_budget(tmp_path):
    """The deliberate divergence from the ported design. Do not simplify it back to a count.

    record-zstack-viewer bounds its thumbnail tier by item count and its byte pool by a hardcoded
    192 MB. ``_budget`` argues a constant "encodes an assumption about a machine it has never
    seen", and ``_tsctx`` already refused the same literal.
    """
    from squidxplorer import _budget

    assert isinstance(_platecache._CELLS, MemoryBoundedLRUCache)
    assert _platecache.cache_budget is cache_budget, "the cache stopped using the measured budget"
    # NOT `== cache_budget()`: the budget is derived from AVAILABLE memory, which moves between
    # the import that sized this cache and the call in this assertion. A test that compares two
    # measurements of a moving quantity is a flake, and a flake here would teach people to
    # replace the measurement with a constant -- the exact regression this test guards.
    assert _budget.FLOOR_BYTES <= _platecache._CELLS.capacity_bytes <= _budget.CEILING_BYTES
    assert _platecache._CELLS.capacity_bytes != 192 << 20, \
        "that is record-zstack-viewer's hardcoded 192 MB; _budget argues why we do not ship it"


def test_the_ram_tier_evicts_by_bytes_and_the_disk_tier_still_answers(tmp_path, monkeypatch):
    """Eviction is what makes the bound real, and the disk tier is what makes it harmless."""
    exp = _acquisition(tmp_path)
    one = _cell(1).nbytes
    monkeypatch.setattr(_platecache, "_CELLS", MemoryBoundedLRUCache(4 * one))
    cache = _cache(tmp_path, exp)
    for i in range(20):
        cache.put(f"A{i}", _cell(i + 1), (0, 0, 88, 88))
    assert _platecache._CELLS.nbytes <= 4 * one, "the RAM tier grew past its byte bound"
    assert len(_platecache._CELLS) == 4
    first = cache.get("A0")                       # evicted from RAM, still on disk
    assert first is not None and int(np.asarray(first)[0, 0, 0]) == 1


def test_the_cell_carries_its_box_through_every_numpy_view(tmp_path):
    """``CellTile`` is an ndarray subclass for the reason ``PlacedArray`` is one: the geometry
    must not be able to arrive separately from the pixels."""
    t = CellTile(_cell(1), (2, 3, 40, 50))
    assert t[:, :10, :10].box == (2, 3, 40, 50)
    assert np.asarray(t).shape == (2, 88, 88)


# --- the switch ----------------------------------------------------------------------------------

def test_the_cache_can_be_turned_off(monkeypatch):
    """Off is a supported state. A user who wants a cold read must be able to have one."""
    monkeypatch.setenv(_platecache.ENV_ENABLED, "0")
    assert _platecache.enabled() is False
    assert PlateCellCache.for_reader(object(), {}, cell_px=88) is None


def test_a_reader_with_no_path_degrades_to_uncached_rather_than_raising():
    """Identity is the acquisition PATH, never ``id(reader)`` (``_mosaic_source._source_token``).

    A reader that cannot say where it reads from gets no cache at all, because the alternative is
    a key that could collide with another acquisition and serve the wrong pixels.
    """
    assert PlateCellCache.for_reader(object(), {"channels": [], "dtype": "uint16"},
                                     cell_px=88) is None


# --- the property the whole module exists for: a reopen reads nothing ------------------------

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication            # noqa: E402

import squidxplorer._viewer as V                        # noqa: E402

FRAME = (8, 8)
CHANNELS = ["c0", "c1"]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _CountingReader:
    """Counts every plane the preview pass pulls out of the acquisition. That count is the cost."""

    def __init__(self, path, boom_after: int = 0):
        self._path = str(path)               # the identity _source_token asks for
        self.reads = 0
        self._boom_after = int(boom_after)

    def read(self, region, fov, channel, z, t=0):
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
    worker.run()                              # in-thread: signal delivery is synchronous here
    return got


def test_reopening_a_plate_reads_NOTHING_from_the_acquisition(qapp, tmp_path):
    """Gap 1 in one assertion. The second open must not touch a single plane."""
    exp = _acquisition(tmp_path)
    meta, idx = _meta(), {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}}

    cold = _CountingReader(exp)
    first = _run(V._PreviewWorker(cold, meta, idx, ["A1", "A2"], cache=_cache(tmp_path, exp)))
    assert cold.reads == len(CHANNELS) * 2, "the cold open must read one plane per channel per well"

    _platecache.clear_memory_tier()                       # i.e. the app was restarted
    warm = _CountingReader(exp)
    worker = V._PreviewWorker(warm, meta, idx, ["A1", "A2"], cache=_cache(tmp_path, exp))
    second = _run(worker)
    assert warm.reads == 0, "the reopen re-read the acquisition"
    assert worker.cache_hits == 2 and worker.cache_reads == 0
    assert [a[2] for a in second] == [a[2] for a in first]
    for a, b in zip(first, second):
        assert np.array_equal(np.asarray(a[3]), np.asarray(b[3])), "the replayed cell differs"


def test_a_cached_mosaic_replays_with_its_CONTENT_BOX_not_the_whole_cell(qapp, tmp_path):
    """The contrast rule, preserved across a restart.

    ``add_tile`` feeds the running histogram whatever tile it is handed, and a mosaic cell is
    zero-padded wherever no FOV lands. Replaying the padding would pin the 1st percentile at 0 and
    wash the whole plate out on every reopen, while the first open looked right -- a difference
    nobody would attribute to a cache.
    """
    exp = _acquisition(tmp_path)
    positions = {("A1", 0): (0.0, 0.0), ("A1", 1): (4.0, 0.0),
                 ("A2", 0): (100.0, 0.0), ("A2", 1): (104.0, 0.0)}
    meta = _meta(positions)
    meta["fovs_per_region"] = {"A1": [0, 1], "A2": [0, 1]}
    idx = {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}}

    first = _run(V._PreviewWorker(_CountingReader(exp), meta, idx, ["A1", "A2"],
                                  cache=_cache(tmp_path, exp)))
    assert len(first) == 4 and all(a[4] is not None for a in first), "this fixture must mosaic"
    union = first[0][4]
    for _ri, _ci, region, _tile, box in first:
        if region == "A1":
            union = V._box_union(union, box)

    _platecache.clear_memory_tier()
    replay = _run(V._PreviewWorker(_CountingReader(exp), meta, idx, ["A1", "A2"],
                                   cache=_cache(tmp_path, exp)))
    assert len(replay) == 2, "a cached mosaic replays as ONE tile per well, not one per FOV"
    box = replay[0][4]
    assert box == union, f"the replayed box {box} is not the mosaic's own extent {union}"
    assert box[2] < 88, "the replay covers the whole cell, padding included"
    assert np.asarray(replay[0][3]).shape == (len(CHANNELS), box[2], box[3])


def test_a_preview_that_FAILS_caches_nothing(qapp, tmp_path):
    """A half-read cell must never be persisted as though it were the well."""
    exp = _acquisition(tmp_path)
    cache = _cache(tmp_path, exp)
    reader = _CountingReader(exp, boom_after=1)
    worker = V._PreviewWorker(reader, _meta(), {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}},
                              ["A1", "A2"], cache=cache)
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert failures, "the failure must still be named"
    assert not cache.dir.exists() or not list(cache.dir.glob("*.npz")), \
        "a preview that could not finish published a cell anyway"


def test_the_plate_preview_actually_goes_through_the_cache():
    """The regression guard, in the shape ``test_tsctx`` uses: the wiring, not just the module."""
    import inspect

    # `run` is now a two-line wrapper (it opens the stdout capture and calls `_run_body`), the same
    # split `_OperatorWorker` uses, so read BOTH — pointing this at `run` alone would go quietly
    # green forever the moment anything else moves out of it.
    src = (inspect.getsource(V._PreviewWorker.run)
           + inspect.getsource(V._PreviewWorker._run_body))
    assert "_replay_cached" in src, "the preview stopped consulting the cache"
    assert "_remember" in src, "the preview stopped filling the cache"
    # ...and the capture itself, which is the wiring Julio's "it doesn't show may standalone
    # stitchers log messages" report was actually about: the preview printed into a terminal
    # nobody is watching because the capture was on the operator worker only.
    assert "capture_stdout_to_log" in src, "the preview stopped capturing print() into the log"


# --- the timepoint is part of a cell's identity (2026-08-05) -------------------------------------
#
# The plate's timepoint bar moved and the plate did not, and the cache was half the reason. The
# other half was `_PreviewWorker` never passing a `t` to `reader.read`; fixing only that would have
# been WORSE than the bug, because the first frame's cells were cached under a key with no
# timepoint in it and would then have been replayed under a label saying t=1.
#
# So these tests are about PIXELS wherever they can be. A test that asserts the slider's value, or
# that the worker was constructed with t=1, passes against a cache that serves frame 0 -- which is
# exactly the failure being prevented. Where a test can only reach state (the on-disk generation,
# the read count) it says which property it is standing in for.

class _TimeReader:
    """A reader whose pixels NAME their timepoint, so a stuck frame is visible, not inferred.

    The same trick ``tools/make_5d_fixture.py`` plays and for the same reason: every fixture in
    this suite was Nt=1, so the timepoint bugs were invisible by construction rather than by
    oversight (``docs/plate-contract.md``). One value per (t, channel) is enough here -- the plate
    cell is an area downsample of the frame, so a frame filled with a value reduces to a cell
    filled with it.
    """

    def __init__(self, path):
        self._path = str(path)               # the identity _source_token asks for
        self.reads = 0
        self.reads_at: dict = {}             # t -> planes read at that timepoint

    def read(self, region, fov, channel, z, t=0):
        self.reads += 1
        self.reads_at[int(t)] = self.reads_at.get(int(t), 0) + 1
        return np.full(FRAME, 1000 * (int(t) + 1) + CHANNELS.index(str(channel)), dtype=np.uint16)


def _preview_cells(worker, cell_px: int = 88) -> dict:
    """``{region: cell}`` from one preview pass -- the PIXELS the plate would paint.

    The tiles are composited into the cell at their boxes, exactly as ``PlateOverview.add_tile``
    does, because a cold pass emits one tile per FOV and a cache replay emits ONE tile per region
    (that is the reopen win). Comparing the raw emissions would compare a mosaic against a field.
    """
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
    """The re-key, at the cache's own boundary: t=0's cell is not an answer to t=1's question."""
    exp = _acquisition(tmp_path)
    at0 = _cache(tmp_path, exp)
    at1 = _cache(tmp_path, exp, time_point=1)
    at0.put("A1", _cell(10), (0, 0, 88, 88))

    assert at1.get("A1") is None, "timepoint 1 was served timepoint 0's cell"
    at1.put("A1", _cell(20), (0, 0, 88, 88))
    assert np.asarray(at0.get("A1")).max() == 10, "publishing t=1 overwrote t=0's cell"
    assert np.asarray(at1.get("A1")).max() == 20

    _platecache.clear_memory_tier()                       # i.e. a restart: now the DISK tier
    assert np.asarray(_cache(tmp_path, exp).get("A1")).max() == 10
    assert np.asarray(_cache(tmp_path, exp, time_point=1).get("A1")).max() == 20
    assert at0.path_for("A1") != at1.path_for("A1"), "two timepoints share one file"


def test_the_timepoint_is_in_the_KEY_and_not_in_the_TOKEN(tmp_path):
    """Both would be correct; only one is useful, and the difference is a whole plate re-read.

    The token directory is what ``prune_stale`` deletes. Put the timepoint in it and stepping
    t=0 -> t=1 deletes t=0's cells, so stepping back re-reads the acquisition: a cache that is
    correct and empty. Timepoints coexist under one generation; a changed ACQUISITION still
    invalidates every one of them at once.
    """
    exp = _acquisition(tmp_path)
    at0, at1 = _cache(tmp_path, exp), _cache(tmp_path, exp, time_point=1)
    assert at0.token == at1.token and at0.dir == at1.dir

    at0.put("A1", _cell(1), (0, 0, 88, 88))
    at1.put("A1", _cell(2), (0, 0, 88, 88))               # publishes, and prunes stale generations
    _platecache.clear_memory_tier()
    assert _cache(tmp_path, exp).get("A1") is not None, \
        "visiting timepoint 1 pruned timepoint 0's cells"

    os.utime(exp / "coordinates.csv", (1_000_000, 1_000_000))       # the store changed
    _platecache.clear_memory_tier()
    assert (_cache(tmp_path, exp).get("A1") is None
            and _cache(tmp_path, exp, time_point=1).get("A1") is None), \
        "a changed acquisition must invalidate EVERY timepoint, not just the live one"


def test_cells_written_before_the_re_key_are_unreachable_and_then_DELETED(tmp_path, monkeypatch):
    """The on-disk tier, migrated by a version bump rather than by a guess.

    A ``FORMAT_VERSION`` 1 cell carries no timepoint. It was written by a producer that always
    read frame 0, but the RECORD does not say so, and the point of this change is that a cell
    whose timepoint is unknown is never served under a timepoint that is. ``FORMAT_VERSION`` is
    hashed into the token, so every v1 entry sits under a token nothing computes any more -- and
    then ``prune_stale`` removes it on the first publish of the new generation.
    """
    exp = _acquisition(tmp_path)
    monkeypatch.setattr(_platecache, "FORMAT_VERSION", 1)
    old = _cache(tmp_path, exp)
    old.put("A1", _cell(42), (0, 0, 88, 88))
    old_dir = old.dir
    assert old_dir.exists()

    monkeypatch.undo()                                    # today's build, FORMAT_VERSION 2
    _platecache.clear_memory_tier()
    new = _cache(tmp_path, exp)
    assert new.dir != old_dir, "the re-keyed cache reads the pre-timepoint generation's directory"
    assert new.get("A1") is None, "a cell with no timepoint was served under one"

    new.put("A1", _cell(7), (0, 0, 88, 88))               # the first publish prunes
    assert not old_dir.exists(), "the pre-timepoint generation was left under $HOME forever"


def test_the_packed_page_is_per_timepoint(tmp_path):
    """Compaction is where a shared page would quietly merge two frames into one picture."""
    exp = _acquisition(tmp_path)
    at0, at1 = _cache(tmp_path, exp), _cache(tmp_path, exp, time_point=1)
    at0.put("A1", _cell(3), (0, 0, 88, 88))
    at1.put("A1", _cell(4), (0, 0, 88, 88))
    assert at0.pack(["A1"]) and at1.pack(["A1"])
    assert at0.pack_array_path != at1.pack_array_path
    assert json.loads(at1.pack_index_path.read_text())["t"] == 1

    _platecache.clear_memory_tier()                       # force the read through the PAGE
    assert np.asarray(_cache(tmp_path, exp).get("A1")).max() == 3
    assert np.asarray(_cache(tmp_path, exp, time_point=1).get("A1")).max() == 4


def test_a_page_whose_sidecar_names_ANOTHER_timepoint_is_not_read(tmp_path):
    """The file name is a convention; the sidecar is the record, and it is checked too."""
    exp = _acquisition(tmp_path)
    at1 = _cache(tmp_path, exp, time_point=1)
    at1.put("A1", _cell(4), (0, 0, 88, 88))
    at1.pack(["A1"])
    index = json.loads(at1.pack_index_path.read_text())
    index["t"] = 0                                        # the page now lies about its frame
    at1.pack_index_path.write_text(json.dumps(index))
    _platecache.clear_memory_tier()
    assert _cache(tmp_path, exp, time_point=1).get("A1") is None


# --- the PLATE's pixels follow the bar -----------------------------------------------------------

def test_the_plate_CELL_at_t1_differs_from_the_cell_at_t0(qapp, tmp_path):
    """The whole bug, in pixels: not "the worker took a t", but what the plate would PAINT.

    Mutation-verified: make ``PlateCellCache._ram_key`` ignore ``self.time_point`` (the key this
    change introduced) and this goes red, because t=1's pass is served t=0's cached cells.
    """
    exp = _acquisition(tmp_path)
    meta, idx = _meta(), {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}}
    reader = _TimeReader(exp)

    at0 = _preview_cells(V._PreviewWorker(reader, meta, idx, ["A1", "A2"],
                                          cache=_cache(tmp_path, exp), t=0))
    at1 = _preview_cells(V._PreviewWorker(reader, meta, idx, ["A1", "A2"],
                                          cache=_cache(tmp_path, exp, time_point=1), t=1))

    assert set(at0) == set(at1) == {"A1", "A2"}
    for region in ("A1", "A2"):
        assert not np.array_equal(at0[region], at1[region]), \
            f"{region}'s plate cell did not move with the timepoint"
        assert at0[region].max() == 1001 and at1[region].max() == 2001, \
            "the cell is not the frame the reader was asked for"

    # ...and it survives the restart, which is where a cache lies most convincingly. BOTH
    # timepoints are replayed, in the order that catches a disk tier keyed without t: t=1 was
    # written last, so replaying only t=1 would look right while t=0 came back as t=1's picture.
    _platecache.clear_memory_tier()
    cold = _TimeReader(exp)
    for t, expected in ((1, at1), (0, at0)):
        replayed = _preview_cells(V._PreviewWorker(cold, meta, idx, ["A1", "A2"],
                                                   cache=_cache(tmp_path, exp, time_point=t), t=t))
        assert cold.reads == 0, "the replay re-read the acquisition"
        for region in ("A1", "A2"):
            assert np.array_equal(replayed[region], expected[region]), \
                f"{region}: the reopened plate showed another frame at t={t}"


def test_the_preview_READS_the_timepoint_it_was_asked_for(qapp, tmp_path):
    """``reader.read`` takes a ``t`` and defaulted to 0 forever because nothing passed one.

    ``docs/plate-contract.md`` records that exact shape for the loupe: a signature is not a call.
    The test above proves the pixels differ; this one names WHY, so a regression reads as "the
    preview stopped passing t" rather than as a cache mystery.
    """
    exp = _acquisition(tmp_path)
    reader = _TimeReader(exp)
    _run(V._PreviewWorker(reader, _meta(), {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}},
                          ["A1", "A2"], cache=None, t=2))
    assert set(reader.reads_at) == {2}, f"the preview read timepoints {sorted(reader.reads_at)}"


def test_stepping_BACK_to_a_visited_timepoint_reads_NOTHING(qapp, tmp_path):
    """The performance half. ``_on_time_point_changed`` re-reads the whole plate on every tick,
    so a revisited timepoint has to HIT or scrubbing t costs a full plate read per step.

    This is the property the re-key buys that a ``(token, region)`` key could not, and it is also
    the one a timepoint-in-the-token would have thrown away.
    """
    exp = _acquisition(tmp_path)
    meta, idx = _meta(), {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}}
    reader = _TimeReader(exp)
    per_pass = len(CHANNELS) * len(meta["regions"])

    _run(V._PreviewWorker(reader, meta, idx, ["A1", "A2"], cache=_cache(tmp_path, exp), t=0))
    assert reader.reads == per_pass, "the first visit must read the plate"
    _run(V._PreviewWorker(reader, meta, idx, ["A1", "A2"],
                          cache=_cache(tmp_path, exp, time_point=1), t=1))
    assert reader.reads == 2 * per_pass, "a NEW timepoint must read the plate"

    back = V._PreviewWorker(reader, meta, idx, ["A1", "A2"], cache=_cache(tmp_path, exp), t=0)
    _run(back)
    assert reader.reads == 2 * per_pass, "stepping back to t=0 re-read the acquisition"
    assert back.cache_hits == 2 and back.cache_reads == 0


def test_a_preview_handed_a_cache_for_ANOTHER_timepoint_refuses_to_start(qapp, tmp_path):
    """Two objects, each correct alone, whose disagreement publishes a cell under the wrong frame.
    Reconciling that silently is how this class of bug survives, so it raises instead."""
    exp = _acquisition(tmp_path)
    with pytest.raises(ValueError, match="wrong frame"):
        V._PreviewWorker(_TimeReader(exp), _meta(), {"A1": {"rc": (0, 0)}}, ["A1"],
                         cache=_cache(tmp_path, exp), t=1)


def test_the_plate_previews_the_timepoint_the_BAR_says():
    """The wiring at the window, which no unit test above can reach.

    ``_start_preview`` is the ONE place a preview is built, so reading ``self.time_point`` there
    is what makes the first ingest, the tab re-scope and ``_return_to_raw`` (which is what a
    timepoint change calls) agree without three call sites having to remember.
    """
    import inspect

    assert "t=self.time_point" in inspect.getsource(V.PlateWindow._start_preview), \
        "the plate's preview stopped carrying the bar's timepoint"
    assert "_start_preview" in inspect.getsource(V.PlateWindow._return_to_raw), \
        "returning to raw stopped restarting the preview, so a timepoint change repaints nothing"
    assert "_return_to_raw" in inspect.getsource(V.PlateWindow._on_time_point_changed), \
        "a timepoint change stopped asking the plate to re-read"


# --- the same property, on the real 3-timepoint acquisition --------------------------------------

FIXTURE_5D = Path("~/Downloads/sim_5d_2x2_t3").expanduser()


@pytest.mark.skipif(not FIXTURE_5D.exists(),
                    reason=f"{FIXTURE_5D} is absent (build it: tools/make_5d_fixture.py)")
def test_the_plate_cell_follows_t_on_the_REAL_5D_acquisition(qapp, tmp_path):
    """The only test in this file that could have failed for the ORIGINAL reason.

    Every other acquisition on this machine is n_t=1, so a stuck timepoint is invisible in them by
    construction. ``sim_5d_2x2_t3`` moves a bright blob across the frame WITH t (4 regions, 4 FOVs,
    3 z, 2 channels, 3 timepoints, 256 px), precisely so the difference is in the pixels rather
    than in a call. Written by ``tools/make_5d_fixture.py``.

    The cache root is ``tmp_path``: a test must never write into the developer's real cache, and
    ``_platecache`` has ``SQUIDXPLORER_CACHE_DIR`` for exactly this.
    """
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
            V._PreviewWorker(reader, meta, idx, order, cache=_cache_at(t), t=t))
        assert set(cells[t]) == set(order)

    for region in order:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            assert not np.array_equal(cells[a][region], cells[b][region]), \
                f"{region}: the plate cell at t={a} is identical to the one at t={b}"

    _platecache.clear_memory_tier()                          # the reopen, served from disk
    for t in range(3):
        worker = V._PreviewWorker(reader, meta, idx, order, cache=_cache_at(t), t=t)
        replayed = _preview_cells(worker)
        assert worker.cache_hits == len(order) and worker.cache_reads == 0
        for region in order:
            assert np.array_equal(replayed[region], cells[t][region]), \
                f"{region}: the cached cell for t={t} came back as another frame"
