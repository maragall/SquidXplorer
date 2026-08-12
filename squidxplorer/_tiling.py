"""Viewport -> tiles: LOD pick, frustum cull, and the tile cache.

Pure python/numpy — no Qt, no reader, no I/O. World space is stage micrometres;
``select_tiles`` returns O(viewport) descriptors, never O(all FOVs).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Protocol, Sequence

import numpy as np

# Zoom deadband past a level boundary before the LOD pick switches.
_DEFAULT_HYSTERESIS = 0.25

# Fraction of the byte budget that pinned (parent-of-pending) tiles may hold.
_PIN_BUDGET_FRACTION = 0.5


@dataclass(frozen=True)
class TileDescriptor:
    """One cacheable tile: level, key, channel, world bbox, timepoint.

    Frozen because it is the cache key. ``t`` has no default on purpose: the timepoint is
    part of the identity a producer must state.
    """

    level: int
    key: Hashable
    channel: str
    bbox_um: tuple[float, float, float, float]
    t: int


class Level:
    """One rung of the ladder: a resolution plus the (N, 4) world boxes of its tiles."""

    def __init__(self, scale_um_per_px: float, bboxes, keys: Sequence[Hashable]) -> None:
        scale = float(scale_um_per_px)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale_um_per_px must be finite and > 0, got {scale_um_per_px!r}")
        arr = np.ascontiguousarray(bboxes, dtype=np.float64).reshape(-1, 4)
        if arr.size and not np.isfinite(arr).all():
            # A NaN coord never compares true in the cull: the tile would silently never render.
            raise ValueError("level bboxes must all be finite (NaN/inf stage coordinate?)")
        if arr.size and not (np.all(arr[:, 2] > arr[:, 0]) and np.all(arr[:, 3] > arr[:, 1])):
            raise ValueError("level bboxes must satisfy x1 > x0 and y1 > y0")
        keys = tuple(keys)
        if len(keys) != len(arr):
            raise ValueError(f"got {len(keys)} keys for {len(arr)} bboxes")
        self.scale_um_per_px = scale
        self.bboxes = arr
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)


class Geometry:
    """The whole ladder, finest first: ``levels[0]`` is level 0 (full resolution)."""

    def __init__(self, levels: Sequence[Level]) -> None:
        levels = tuple(levels)
        if not levels:
            raise ValueError("Geometry needs at least one level")
        scales = [lv.scale_um_per_px for lv in levels]
        if any(b <= a for a, b in zip(scales, scales[1:])):
            raise ValueError(f"levels must be ordered finest-first with strictly increasing scale, got {scales}")
        counts = [len(lv) for lv in levels]
        # A coarser rung with MORE tiles than the one below is a construction bug; equal is legal.
        bad = [(i, counts[i], counts[i + 1]) for i in range(len(counts) - 1) if counts[i + 1] > counts[i]]
        if bad:
            raise ValueError(
                "a coarser level cannot hold more tiles than the level below it; "
                f"offending (level, n, next_n): {bad}")
        self.levels = levels
        self._scales = np.asarray(scales, dtype=np.float64)

    @property
    def worst_case_tiles(self) -> int:
        """Upper bound on tiles any single view can request: the coarsest level's tile count."""
        return len(self.levels[-1])

    def __len__(self) -> int:
        return len(self.levels)

    def pick_level(self, um_per_px: float, current_level: int | None = None,
                   hysteresis: float = _DEFAULT_HYSTERESIS) -> int:
        """The coarsest level still at least as fine as the screen; ``current_level`` adds a deadband."""
        req = float(um_per_px)
        if not np.isfinite(req) or req <= 0:
            raise ValueError(f"um_per_px must be finite and > 0, got {um_per_px!r}")
        finer = np.flatnonzero(self._scales <= req)
        ideal = int(finer[-1]) if finer.size else 0
        if current_level is None or not (0 <= current_level < len(self.levels)) or current_level == ideal:
            return ideal
        # Current level's band, widened both ways by the hysteresis.
        lo = self._scales[current_level] / (1.0 + hysteresis)
        hi = (self._scales[current_level + 1] * (1.0 + hysteresis)
              if current_level + 1 < len(self.levels) else np.inf)
        return current_level if lo <= req < hi else ideal


class TileSource(Protocol):
    """Turn a descriptor into pixels. Synchronous; the caller threads."""

    def read_tile(self, desc: TileDescriptor) -> np.ndarray:  # pragma: no cover - protocol
        ...


def select_tiles(bbox_um: tuple[float, float, float, float], um_per_px: float, geometry: Geometry, *,
                 channels: Sequence[str] = ("0",), t: int = 0, current_level: int | None = None,
                 hysteresis: float = _DEFAULT_HYSTERESIS) -> list[TileDescriptor]:
    """The ideal tile set for a viewport: LOD pick, then frustum cull. Pure, stateless.

    Returns tiles in deterministic order (channel-major, then level index order).
    """
    x0, y0, x1, y1 = (float(v) for v in bbox_um)
    if not all(np.isfinite(v) for v in (x0, y0, x1, y1)):
        raise ValueError(f"bbox_um must be finite, got {bbox_um!r}")
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"bbox_um must satisfy x1 > x0 and y1 > y0, got {bbox_um!r}")

    level_idx = geometry.pick_level(um_per_px, current_level, hysteresis)
    level = geometry.levels[level_idx]
    b = level.bboxes
    if len(b) == 0:
        return []
    # One vectorized overlap test; touching edges do not count as overlap.
    hit = (b[:, 0] < x1) & (b[:, 2] > x0) & (b[:, 1] < y1) & (b[:, 3] > y0)
    idx = np.flatnonzero(hit)

    out: list[TileDescriptor] = []
    for ch in channels:
        for i in idx:
            i = int(i)
            out.append(TileDescriptor(level_idx, level.keys[i], ch,
                                      (b[i, 0], b[i, 1], b[i, 2], b[i, 3]), int(t)))
    return out


def viewport(bbox_world: tuple[float, float, float, float], zoom: float, geometry: Geometry, *,
             channels: Sequence[str] = ("0",), t: int = 0, current_level: int | None = None,
             hysteresis: float = _DEFAULT_HYSTERESIS) -> list[TileDescriptor]:
    """``select_tiles`` in the renderer's units: ``zoom`` is screen pixels per world unit."""
    z = float(zoom)
    if not np.isfinite(z) or z <= 0:
        raise ValueError(f"zoom must be finite and > 0, got {zoom!r}")
    return select_tiles(bbox_world, 1.0 / z, geometry, channels=channels, t=t,
                        current_level=current_level, hysteresis=hysteresis)


def _contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    """Is ``inner`` inside ``outer`` (closed)? Used to find a coarse tile covering a fine one."""
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


class TileCache:
    """Byte-budget LRU of decoded tiles + keep-parent-until-child-ready.

    The fetch lifecycle belongs to the caller: ``mark_pending`` before the read, then
    ``insert`` or ``fetch_failed``. A pending tile pins its nearest cached ancestor.
    """

    def __init__(self, budget_bytes: int) -> None:
        if budget_bytes < 0:
            raise ValueError(f"budget_bytes must be >= 0, got {budget_bytes}")
        self._budget = int(budget_bytes)
        self._cached: "OrderedDict[TileDescriptor, np.ndarray]" = OrderedDict()  # LRU: oldest first
        self._pending: "OrderedDict[TileDescriptor, TileDescriptor | None]" = OrderedDict()  # child -> pinned parent
        self._bytes = 0

    @property
    def nbytes(self) -> int:
        return self._bytes

    @property
    def budget_bytes(self) -> int:
        return self._budget

    def __len__(self) -> int:
        return len(self._cached)

    def __contains__(self, desc: object) -> bool:
        return desc in self._cached

    def cached_descriptors(self) -> list[TileDescriptor]:
        """LRU order, oldest first (the eviction order)."""
        return list(self._cached)

    def pending_descriptors(self) -> list[TileDescriptor]:
        return list(self._pending)

    def pinned_descriptors(self) -> list[TileDescriptor]:
        return [p for p in self._pending.values() if p is not None]

    def get(self, desc: TileDescriptor) -> np.ndarray | None:
        """Fetch a cached tile and promote it to most-recently-used, or None if absent."""
        arr = self._cached.get(desc)
        if arr is None:
            return None
        self._cached.move_to_end(desc)
        return arr

    def mark_pending(self, desc: TileDescriptor) -> None:
        """Declare a fetch in flight: pins the nearest cached ancestor so it survives eviction."""
        if desc in self._cached or desc in self._pending:
            return
        self._pending[desc] = self._nearest_ancestor(desc)
        self._enforce_pin_cap()

    def insert(self, desc: TileDescriptor, arr: np.ndarray) -> None:
        """The fetch landed: cache the tile (MRU), release its parent's pin, trim to budget."""
        self._pending.pop(desc, None)
        if desc in self._cached:
            self._bytes -= self._cached[desc].nbytes
            del self._cached[desc]
        self._cached[desc] = arr
        self._bytes += arr.nbytes
        self._evict_to_budget()

    def fetch_failed(self, desc: TileDescriptor) -> None:
        """The fetch died: drop the pending entry so a failed child never leaks an immortal pin."""
        self._pending.pop(desc, None)

    def invalidate(self, predicate: Callable[[TileDescriptor], bool]) -> int:
        """Drop every cached/pending tile matching ``predicate``; returns how many were cached."""
        doomed = [d for d in self._cached if predicate(d)]
        for d in doomed:
            self._bytes -= self._cached.pop(d).nbytes
        for d in [p for p in self._pending if predicate(p)]:
            del self._pending[d]
        for child, parent in list(self._pending.items()):
            if parent is not None and parent not in self._cached:
                self._pending[child] = self._nearest_ancestor(child)   # re-pin, parent is gone
        return len(doomed)

    def resolve(self, ideal: Iterable[TileDescriptor]) -> list[tuple[TileDescriptor, np.ndarray]]:
        """Ideal set -> what can be drawn right now, substituting cached ancestors."""
        out: list[tuple[TileDescriptor, np.ndarray]] = []
        seen: set[TileDescriptor] = set()
        for desc in ideal:
            arr = self.get(desc)
            if arr is None:
                parent = self._nearest_ancestor(desc)
                if parent is None:
                    continue
                desc, arr = parent, self.get(parent)
            if desc in seen:
                continue                                 # one coarse parent covers many children
            seen.add(desc)
            out.append((desc, arr))
        return out

    def _nearest_ancestor(self, desc: TileDescriptor) -> TileDescriptor | None:
        """The finest cached tile of the same channel AND timepoint, coarser, whose bbox covers it."""
        best: TileDescriptor | None = None
        for other in self._cached:
            if other.channel != desc.channel or other.t != desc.t or other.level <= desc.level:
                continue
            if not _contains(other.bbox_um, desc.bbox_um):
                continue
            if best is None or other.level < best.level:
                best = other
        return best

    def _enforce_pin_cap(self) -> None:
        """Keep pinned bytes under the cap by dropping the oldest pending descriptors."""
        cap = self._budget * _PIN_BUDGET_FRACTION
        while self._pending:
            pinned = {p for p in self._pending.values() if p is not None}
            if sum(self._cached[p].nbytes for p in pinned if p in self._cached) <= cap:
                return
            self._pending.popitem(last=False)            # oldest pending request loses its pin

    def _evict_to_budget(self) -> None:
        """Evict LRU-first until under budget, skipping pinned parents."""
        if self._bytes <= self._budget:
            return
        pinned = {p for p in self._pending.values() if p is not None}
        for desc in list(self._cached):
            if self._bytes <= self._budget:
                return
            if desc in pinned or len(self._cached) == 1:
                continue                                 # never evict a pinned parent or the last tile
            self._bytes -= self._cached.pop(desc).nbytes
