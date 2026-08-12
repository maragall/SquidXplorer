"""One bounded TensorStore context, and one bounded pool of open stores, for the whole process."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional

from squidxplorer._budget import cache_budget

#: Open stores kept alive at once. Bounds file descriptors and per-store bookkeeping,
#: not decoded pixels: the shared context bounds those.
DEFAULT_MAX_OPEN = 32

_CTX = None
_CTX_LOCK = threading.Lock()


def ts_context(byte_limit: Optional[int] = None):
    """The process-wide TensorStore context, created on first use."""
    global _CTX
    with _CTX_LOCK:
        if _CTX is None:
            import tensorstore as ts
            limit = int(byte_limit if byte_limit is not None else cache_budget())
            _CTX = ts.Context({"cache_pool": {"total_bytes_limit": limit}})
        return _CTX


class HandleCache:
    """Bounded LRU of open TensorStore handles, safe to share across QThreads."""

    def __init__(self, max_open: int = DEFAULT_MAX_OPEN, byte_limit: Optional[int] = None) -> None:
        self._d: "OrderedDict[str, Any]" = OrderedDict()
        self._max = int(max_open)
        self._byte_limit = byte_limit
        self._lock = threading.Lock()

    def get(self, path, *, driver: str = "zarr3", recheck: bool = False, open_only: bool = False):
        """The store at ``path``, opened once and reused."""
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
            # Open inside the lock so two threads racing the same missing path open it once.
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


#: The process-wide pool.
HANDLES = HandleCache()
