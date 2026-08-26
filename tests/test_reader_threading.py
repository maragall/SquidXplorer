"""`reader._TiffHandles` is the mutual exclusion around a decode."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from squidxplorer import reader as R


class _FakePage:
    def __init__(self, on_decode, index):
        self._on_decode, self._index = on_decode, index

    def asarray(self):
        self._on_decode(self._index)
        return np.zeros((4, 4), dtype=np.uint16)


class _FakeTiff:
    """Stands in for a cached `TiffFile`. `pages[i].asarray()` calls *on_decode*."""

    def __init__(self, on_decode):
        self.pages = _FakePages(on_decode)


class _FakePages:
    def __init__(self, on_decode):
        self._on_decode = on_decode

    def __getitem__(self, index):
        return _FakePage(self._on_decode, index)


def _seed(handles: R._TiffHandles, path: Path, on_decode) -> None:
    """Put a fake handle in the cache, exactly as `_entry` would have."""
    with handles._guard:
        handles._handles[path] = _FakeTiff(on_decode)
        handles._locks[path] = threading.Lock()


def test_two_threads_never_decode_one_file_at_the_same_time():
    handles = R._TiffHandles()
    path = Path("one.ome.tiff")
    inside = threading.Semaphore(0)
    overlapped = threading.Event()
    depth = [0]
    depth_lock = threading.Lock()

    def on_decode(_index):
        with depth_lock:
            depth[0] += 1
            if depth[0] > 1:
                overlapped.set()      # a second decoder got in: the lock is not doing its job
        inside.acquire(timeout=1.0)
        with depth_lock:
            depth[0] -= 1

    _seed(handles, path, on_decode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(handles.page, path, i) for i in range(2)]
        inside.release()
        inside.release()
        for f in futures:
            f.result(timeout=10)

    assert not overlapped.is_set(), (
        "two threads were inside one file's decode at once — a TiffFile is a file object and "
        "pages[p].asarray() seeks, so this is the silent-wrong-pixels bug, not a slow read")


def test_two_threads_DO_decode_two_different_files_at_the_same_time():
    """The lock is per FILE. A global one would pass the test above and serialise every FOV."""
    handles = R._TiffHandles()
    paths = [Path("a.ome.tiff"), Path("b.ome.tiff")]
    barrier = threading.Barrier(2, timeout=5.0)

    def on_decode(_index):
        barrier.wait()

    for p in paths:
        _seed(handles, p, on_decode)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(handles.page, p, 0) for p in paths]
        for f in futures:
            f.result(timeout=10)      # a BrokenBarrierError here means the reads were serialised


@pytest.mark.parametrize("fixture_name", ["ome_tiff_dataset", "multipage_dataset"])
def test_every_tiff_read_goes_through_the_guarded_handle_cache(request, fixture_name):
    """Asserts the class-level fact instead of the timing one: the only route to a page is `_TiffHandles.page`, so every decode is inside the lock by construction."""
    root, _ = request.getfixturevalue(fixture_name)
    rdr = R.open_reader(str(root))
    handles = getattr(rdr, "_handles", None)
    assert isinstance(handles, R._TiffHandles), (
        f"{type(rdr).__name__} does not own a _TiffHandles; a raw cached TiffFile is a seek "
        "position shared between threads")

    decodes: list = []
    original = R._TiffHandles.page

    def counting_page(self, path, index):
        decodes.append((path, index))
        return original(self, path, index)

    meta = rdr.metadata
    region, fovs = next(iter(sorted(meta["fovs_per_region"].items())))
    channel = getattr(meta["channels"][0], "name", meta["channels"][0])

    R._TiffHandles.page = counting_page
    try:
        plane = rdr.read(region, sorted(fovs)[0], channel, 0, 0)
    finally:
        R._TiffHandles.page = original

    assert plane.ndim == 2
    assert decodes, "reader.read() decoded a plane without going through _TiffHandles.page"
