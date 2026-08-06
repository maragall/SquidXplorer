"""``reader._TiffHandles`` is the mutual exclusion around a decode. These pin the mechanism.

Why the mechanism and not the symptom
-------------------------------------
A cached ``tifffile.TiffFile`` is a FILE OBJECT: ``pages[p].asarray()`` seeks, so two threads
decoding two pages of one file move one seek position under each other. Measured on the real 10x
acquisition (``manual0``) with ``tools/thread_stress.py``, the lock removed:

    40 threads: 151 of 400 reads WRONG (42 raised, 109 returned WRONG PIXELS) in 1.6 s
    40 threads:   0 of 400 wrong, with the lock

The 109 is the number that decides how these tests are written. The original report of this bug
counted only the exceptions ("10 of 40"), because exceptions are what a stress run notices —
but MOST of the damage is a read that decodes cleanly and returns another plane's bytes. A test
that hammers a real file and asserts "no exceptions" would therefore pass on a build that is
silently corrupting two reads in three.

So these do not race and hope. They install a fake handle whose decode BLOCKS on a barrier, which
turns "is there mutual exclusion?" into a question with a yes/no answer and no timing luck:

* two threads on ONE file must not overlap — the second must be made to wait;
* two threads on DIFFERENT files must overlap — the lock is per FILE, because the parallelism this
  package needs is across FOVs and each FOV is its own file. A global lock would pass the first
  test and destroy the concurrency the reader exists to allow, so it is pinned too.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from squidmip import reader as R


class _FakePage:
    def __init__(self, on_decode, index):
        self._on_decode, self._index = on_decode, index

    def asarray(self):
        self._on_decode(self._index)
        return np.zeros((4, 4), dtype=np.uint16)


class _FakeTiff:
    """Stands in for a cached ``TiffFile``. ``pages[i].asarray()`` calls *on_decode*."""

    def __init__(self, on_decode):
        self.pages = _FakePages(on_decode)


class _FakePages:
    def __init__(self, on_decode):
        self._on_decode = on_decode

    def __getitem__(self, index):
        return _FakePage(self._on_decode, index)


def _seed(handles: R._TiffHandles, path: Path, on_decode) -> None:
    """Put a fake handle in the cache, exactly as ``_entry`` would have. Bypasses no locking."""
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
        # Hold the file "open" long enough that an unguarded second thread is certain to enter.
        # With the lock it cannot, so this wait is what makes the negative result deterministic:
        # a broken build sets the event immediately, a correct one blocks the other thread here.
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
    # Both decoders must arrive before either may leave. Under a global lock the first holds it
    # while waiting for a second that can never start, and the barrier times out — so this is a
    # deadlock detector, not a race.
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
    """The lock protects nothing if a reader still holds the raw ``TiffFile`` itself.

    Both TIFF readers used to reach a shared ``_tif(path)`` that handed the object out unguarded.
    This asserts the class-level fact instead of the timing one: the only route to a page is
    ``_TiffHandles.page``, so every decode is inside the lock by construction.
    """
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
