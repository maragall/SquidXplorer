"""One bounded TensorStore context, and one bounded pool of open stores, for the whole process.

Gap 2 of the three-viewers review (Hongquan, 2026-07-28), verified against HEAD and found WORSE
than reported: the note counted three ``ts.open`` call sites, there were seven, and ``ts.Context``,
``cache_pool`` and ``total_bytes_limit`` appeared ZERO times in the repo. Every ``ts.open`` got
TensorStore's own default resources, so each site had a private, unshared, undeclared cache, and
nothing bounded how many stores were open at once.

The cost was measured on the plate-scrub path. ``_ComputedPlateWorker._read`` opened a brand new
store per well per pyramid level, twice per well, with no reuse: on a 1536-well plate that is
**3072 fresh opens**, each allocating its own pool. The same shape sat in four other places, and
the four dicts that did cache handles (``reader``, ``_tilesource``, the loupe source) were
UNBOUNDED and mutated from QThread workers with no lock.

The design is taken from ``record-zstack-viewer``'s ``cache/memory.py``, with one deliberate
change. It hardcodes 192 MB. We do not: ``_budget.cache_budget()`` derives the limit from
``psutil``'s AVAILABLE memory with a floor, a ceiling and an env override, and its docstring
argues at length that a constant "encodes an assumption about a machine it has never seen".
Shipping their literal would have contradicted a decision this repo already made and documented.

Two bounds, and they are orthogonal, which is the actual point of the design:

* the shared ``cache_pool`` caps decoded BYTES across every reader that binds to it;
* the LRU caps the number of OPEN HANDLES.

Neither alone is sufficient. A byte cap with unbounded handles still exhausts file descriptors on
a plate scrub; a handle cap with no byte cap still lets one big read blow the footprint.

Why this is worth doing at all, in one sentence borrowed from the review: our engine bounds memory
by a DISCIPLINE it must maintain, while binding every reader to one shared resource bounds it BY
CONSTRUCTION, and construction survives a call site added by someone who never read the docstring.
``_viewer.py``'s per-well open was exactly that call site.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional

from squidmip._budget import cache_budget

#: Open stores kept alive at once. From record-zstack-viewer. It bounds file descriptors and
#: TensorStore's per-store bookkeeping, NOT decoded pixels: the context below bounds those.
DEFAULT_MAX_OPEN = 32

_CTX = None
_CTX_LOCK = threading.Lock()


def ts_context(byte_limit: Optional[int] = None):
    """The process-wide TensorStore context, created on first use.

    Every ``ts.open`` in this package should pass ``context=ts_context()``. Two stores opened with
    the same context share one ``cache_pool``, so the byte limit is a property of the PROCESS and
    not of whichever call site happened to run first.

    ``byte_limit`` defaults to :func:`squidmip._budget.cache_budget`. Note the multiplier caveat in
    that module: each consumer takes its fraction independently, so adding a fourth consumer moves
    the total from 3x to 4x of the configured fraction. This context is one such consumer.
    """
    global _CTX
    with _CTX_LOCK:
        if _CTX is None:
            import tensorstore as ts
            limit = int(byte_limit if byte_limit is not None else cache_budget())
            _CTX = ts.Context({"cache_pool": {"total_bytes_limit": limit}})
        return _CTX


class HandleCache:
    """Bounded LRU of open TensorStore handles, safe to share across QThreads.

    Thread safety is not optional here and it is the thing the four existing handle dicts got
    wrong. ``_ComputedPlateWorker``, ``_LoupeWorker`` and Spencer's ``_TileFetcher`` all read
    concurrently with the GUI thread, and all three reach the same stores.

    The open happens INSIDE the lock, deliberately: two threads racing the same missing path open
    it once rather than twice. That costs a little concurrency on a cold miss and buys idempotence.

    Eviction is a plain reference drop. TensorStore has no explicit close, so dropping the last
    reference is the close.
    """

    def __init__(self, max_open: int = DEFAULT_MAX_OPEN, byte_limit: Optional[int] = None) -> None:
        self._d: "OrderedDict[str, Any]" = OrderedDict()
        self._max = int(max_open)
        self._byte_limit = byte_limit
        self._lock = threading.Lock()

    def get(self, path, *, driver: str = "zarr3", recheck: bool = False, open_only: bool = False):
        """The store at ``path``, opened once and reused.

        ``driver`` is a parameter rather than a constant because the reader supports BOTH zarr v2
        and v3 on disk: it picks by probing for a ``.zarray``. Hardcoding v3 here would have
        quietly excluded every v2 acquisition from the pool, which is the sort of narrowing that
        looks like a shared resource and is not one.

        ``recheck=True`` sets ``recheck_cached_data="open"``, which revalidates at open rather than
        per read. That is for stores that GROW while we read them (a live acquisition, a plate
        being written). A finished acquisition does not need it and pays for nothing.

        ``open_only=True`` passes ``open=True``, i.e. refuse to create. Callers reading an existing
        acquisition want that: creating one silently would write into somebody's data.
        """
        key = f"{path}\x00{driver}\x00{int(recheck)}\x00{int(open_only)}"
        with self._lock:
            hit = self._d.get(key)
            if hit is not None:
                self._d.move_to_end(key)
                return hit
            import tensorstore as ts
            spec = {"driver": driver, "kvstore": {"driver": "file", "path": str(path)}}
            if recheck:
                spec["recheck_cached_data"] = "open"
            kw = {"open": True} if open_only else {}
            store = ts.open(spec, context=ts_context(self._byte_limit), **kw).result()
            self._d[key] = store
            while len(self._d) > self._max:
                self._d.popitem(last=False)
            return store

    def clear(self) -> None:
        """Drop every handle. Pair with ``recheck`` for a store that has been rewritten."""
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)


#: The process-wide pool. A module global for the same reason `_fusion_style()` and `qt_app()` are:
#: an object whose lifetime must not belong to whichever window, worker or fixture reached it first.
HANDLES = HandleCache()
