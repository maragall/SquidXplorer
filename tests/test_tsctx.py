"""The TensorStore reads share one bounded context and one bounded pool of handles.

`_ComputedPlateWorker._read` used to open a brand new store per well per pyramid level, called
twice per well, so a 1536-well plate did 3072 fresh opens, each allocating its own private cache.
These tests pin the two properties that fix it, deliberately orthogonal: a byte cap with unbounded
handles still exhausts descriptors on a scrub, and a handle cap with no byte cap still lets one
big read blow the footprint.

The byte limit comes from `_budget.cache_budget()`, not a hardcoded constant — asserted below so
nobody "simplifies" it back to one.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("tensorstore")

import numpy as np  # noqa: E402
import tensorstore as ts  # noqa: E402

from squidxplorer import _tsctx  # noqa: E402
from squidxplorer._budget import cache_budget  # noqa: E402


def _write_store(path, shape=(1, 1, 1, 4, 4)):
    """A minimal real zarr3 store, so these tests exercise TensorStore rather than a mock."""
    store = ts.open(
        {"driver": "zarr3",
         "kvstore": {"driver": "file", "path": str(path)},
         "metadata": {"shape": list(shape), "data_type": "uint16",
                      "chunk_grid": {"name": "regular",
                                     "configuration": {"chunk_shape": list(shape)}}}},
        create=True, delete_existing=True).result()
    store[...] = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    return store


def test_the_context_is_one_object_for_the_whole_process():
    """Two call sites must share a cache_pool, or the byte limit is per-site and means nothing."""
    assert _tsctx.ts_context() is _tsctx.ts_context()


def test_the_byte_limit_comes_from_the_measured_budget_not_a_constant():
    limit = cache_budget()
    assert limit > 0
    assert _tsctx.cache_budget is cache_budget, "the context stopped using the measured budget"


def test_handles_are_reused_rather_than_reopened(tmp_path):
    p = tmp_path / "a.zarr"
    _write_store(p)
    cache = _tsctx.HandleCache()
    first = cache.get(p)
    for _ in range(50):
        assert cache.get(p) is first, "the same path opened more than once"
    assert len(cache) == 1


def test_the_handle_pool_is_bounded(tmp_path):
    """Bounded by CONSTRUCTION, not by a caller remembering to evict."""
    cache = _tsctx.HandleCache(max_open=4)
    for i in range(12):
        p = tmp_path / f"s{i}.zarr"
        _write_store(p)
        cache.get(p)
    assert len(cache) == 4, "the pool grew past its bound"


def test_the_pool_is_least_recently_used_not_arbitrary(tmp_path):
    """A scrub revisits the same wells; LRU is what makes reuse survive a sweep."""
    paths = []
    for i in range(3):
        p = tmp_path / f"l{i}.zarr"
        _write_store(p)
        paths.append(p)
    cache = _tsctx.HandleCache(max_open=2)
    a = cache.get(paths[0])
    cache.get(paths[1])
    assert cache.get(paths[0]) is a          # touch 0, so 1 is now the oldest
    cache.get(paths[2])                      # evicts 1, not 0
    assert cache.get(paths[0]) is a, "the recently used handle was evicted"


def test_concurrent_first_opens_yield_one_handle(tmp_path):
    """The open happens inside the lock so a race on a cold miss opens once, not once per thread."""
    p = tmp_path / "race.zarr"
    _write_store(p)
    cache = _tsctx.HandleCache()
    seen, barrier = [], threading.Barrier(8)

    def grab():
        barrier.wait()
        seen.append(cache.get(p))

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 8
    assert len({id(s) for s in seen}) == 1, "a race opened the same store more than once"
    assert len(cache) == 1


def test_the_plate_scrub_goes_through_the_pool():
    """The regression that matters: _ComputedPlateWorker must not call ts.open directly again."""
    import inspect

    from squidxplorer import _viewer

    src = inspect.getsource(_viewer._ComputedPlateWorker._read)
    assert "HANDLES.get" in src, "the plate scrub stopped using the shared pool"
    assert "ts.open(" not in src, "the plate scrub opens stores directly again (3072 on a 1536wp)"
