"""Viewport -> tiles: the LOD pick + frustum cull + tile cache core (IMA-216).

Pure python/numpy. **No Qt, no reader, no I/O** (precedent: ``_layers.py``) so the same
algorithm backs the desktop plate view (IMA-218) and a future web renderer. Callers hand in
the FOV/tile geometry; this module never discovers it.

World space is **stage micrometres**; a viewport is ``bbox_um = (x0, y0, x1, y1)`` and the
zoom is ``um_per_px`` (micrometres per *screen* pixel — bigger = zoomed further out). The
viewer converts at its edge; ``viewport()`` accepts the screen-pixels-per-micrometre form.

Two separated pieces, so the pure part stays trivially testable:

    caller (viewer/web)              _tiling.py                       IMA-217
    ───────────────────              ──────────                       ───────
    (bbox_um, um_per_px) ─► select_tiles ─ideal─► TileCache.resolve ─misses─► TileSource
                                │                      │                        │
                          pure geometry          LRU + pins          cache.insert(desc, arr)
                          no state, no I/O       renderable ─► caller draws

The invariant that makes the viewer fast: ``select_tiles`` returns O(viewport) descriptors,
never O(all FOVs) — it culls against the *level* the screen actually resolves, so fit-to-plate
reads a handful of coarse tiles instead of every level-0 FOV plane.

    TileCache tile state machine
                        insert(desc, arr)
       ABSENT ──mark_pending──► PENDING ──insert──► CACHED (LRU-ordered)
         ▲                        │  │                 │
         │        fetch_failed────┘  │                 │ evicted (unpinned only)
         └───────────◄───────────────┴────────◄────────┘
       PINNED = CACHED ancestor of a PENDING desc; skipped by eviction; released on the
       child's insert/fetch_failed; pinned bytes capped at budget/2 (blur, never deadlock).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Protocol, Sequence

import numpy as np

# Zoom deadband: how far past a level boundary the zoom must travel before the LOD pick
# actually switches. 0.25 = 25% of the boundary scale; enough that wheel jitter parked on a
# boundary does not thrash the fetch queue, small enough to stay imperceptible.
_DEFAULT_HYSTERESIS = 0.25

# Fraction of the byte budget that pinned (parent-of-pending) tiles may hold. Past it the
# oldest pending descriptor is dropped: the screen degrades to blur instead of deadlocking.
_PIN_BUDGET_FRACTION = 0.5


@dataclass(frozen=True)
class TileDescriptor:
    """One cacheable tile: a level, its key in that level, a channel and its world bbox.

    Tiles are cached **pre-composite, per channel** — contrast/LUT changes are the renderer's
    job and invalidate nothing here. Frozen (and hashable) because it is the cache key.
    """

    level: int
    key: Hashable
    channel: str
    bbox_um: tuple[float, float, float, float]


class Level:
    """One rung of the ladder: a resolution plus the world boxes of every tile at it.

    ``scale_um_per_px`` is the tile's native resolution. ``bboxes`` is an ``(N, 4)`` float64
    array of ``(x0, y0, x1, y1)`` µm — one vectorized array so the cull is a single numpy op,
    not a per-FOV python loop. ``keys[i]`` names ``bboxes[i]`` for the tile source.
    """

    def __init__(self, scale_um_per_px: float, bboxes, keys: Sequence[Hashable]) -> None:
        scale = float(scale_um_per_px)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale_um_per_px must be finite and > 0, got {scale_um_per_px!r}")
        arr = np.ascontiguousarray(bboxes, dtype=np.float64).reshape(-1, 4)
        if arr.size and not np.isfinite(arr).all():
            # A bad coordinates.csv row (NaN stage coord) would never compare true in the cull
            # and the tile would silently never render. Fail loud and early instead.
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
    """The whole ladder, finest first: ``levels[0]`` is level 0 (full resolution).

    Levels need NOT align power-of-two: plate-level tiles (coarse) and per-FOV pyramid levels
    (fine) coexist as independent rungs — the regridding at the seam is IMA-217's construction
    problem and is invisible here. Only the ordering matters (strictly increasing scale).
    """

    def __init__(self, levels: Sequence[Level]) -> None:
        levels = tuple(levels)
        if not levels:
            raise ValueError("Geometry needs at least one level")
        scales = [lv.scale_um_per_px for lv in levels]
        if any(b <= a for a, b in zip(scales, scales[1:])):
            raise ValueError(f"levels must be ordered finest-first with strictly increasing scale, got {scales}")
        counts = [len(lv) for lv in levels]
        # A coarser rung holding MORE tiles than the one below it is always a construction bug
        # (regridding cannot invent tiles). Equal counts are legal: a per-FOV pyramid keeps one
        # tile per FOV at every level. See `worst_case_tiles` for the cost that actually bites.
        bad = [(i, counts[i], counts[i + 1]) for i in range(len(counts) - 1) if counts[i + 1] > counts[i]]
        if bad:
            raise ValueError(
                "a coarser level cannot hold more tiles than the level below it; "
                f"offending (level, n, next_n): {bad}")
        self.levels = levels
        self._scales = np.asarray(scales, dtype=np.float64)

    @property
    def worst_case_tiles(self) -> int:
        """Upper bound on tiles any single view can request: the coarsest level's tile count.

        Fit-to-plate clamps to the coarsest rung, so this IS the fit-to-plate cost. The
        O(viewport) promise is a promise about *this number*, not about the algorithm: if the
        ladder has no plate-level rung, the coarsest level is still per-FOV and this equals the
        FOV count — culling then returns every FOV and the plate view crawls, silently.
        IMA-217 builds the ladder and should assert this stays small (tens, not thousands).
        """
        return len(self.levels[-1])

    def __len__(self) -> int:
        return len(self.levels)

    def pick_level(self, um_per_px: float, current_level: int | None = None,
                   hysteresis: float = _DEFAULT_HYSTERESIS) -> int:
        """The coarsest level still at least as fine as the screen — "just finer than screen res".

        Formally: the largest index with ``scale_um_per_px <= um_per_px`` (a tie picks that level
        exactly). Both ends clamp: finer than level 0 -> level 0; coarser than the coarsest ->
        the coarsest, which is why a plate-level rung above the per-FOV pyramids is what keeps
        fit-to-plate O(viewport).

        ``current_level`` adds a deadband: while the zoom stays inside the current level's band
        widened by ``hysteresis``, the pick sticks — crossing a boundary back and forth must not
        re-fetch the world.
        """
        req = float(um_per_px)
        if not np.isfinite(req) or req <= 0:
            raise ValueError(f"um_per_px must be finite and > 0, got {um_per_px!r}")
        finer = np.flatnonzero(self._scales <= req)
        ideal = int(finer[-1]) if finer.size else 0
        if current_level is None or not (0 <= current_level < len(self.levels)) or current_level == ideal:
            return ideal
        # Band of the current level: [its own scale, the next level's scale), widened both ways.
        lo = self._scales[current_level] / (1.0 + hysteresis)
        hi = (self._scales[current_level + 1] * (1.0 + hysteresis)
              if current_level + 1 < len(self.levels) else np.inf)
        return current_level if lo <= req < hi else ideal


class TileSource(Protocol):
    """What IMA-217 implements: turn a descriptor into pixels. Synchronous; the caller threads."""

    def read_tile(self, desc: TileDescriptor) -> np.ndarray:  # pragma: no cover - protocol
        ...


def select_tiles(bbox_um: tuple[float, float, float, float], um_per_px: float, geometry: Geometry, *,
                 channels: Sequence[str] = ("0",), current_level: int | None = None,
                 hysteresis: float = _DEFAULT_HYSTERESIS) -> list[TileDescriptor]:
    """The **ideal** tile set for a viewport: LOD pick, then frustum cull. Pure, stateless.

    ``bbox_um`` is ``(x0, y0, x1, y1)`` stage µm and must be non-degenerate (explicit > clever:
    an inverted or zero-area box raises rather than being silently normalized). Returns tiles
    in a deterministic order — channel-major, then level index order — so callers can diff two
    consecutive viewports cheaply.
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
    # The cull: ONE vectorized overlap test over the level's (N, 4) array. Touching edges do
    # not count as overlap (strict), so an FOV flush against the viewport border is not fetched.
    hit = (b[:, 0] < x1) & (b[:, 2] > x0) & (b[:, 1] < y1) & (b[:, 3] > y0)
    idx = np.flatnonzero(hit)

    out: list[TileDescriptor] = []
    for ch in channels:
        for i in idx:
            i = int(i)
            out.append(TileDescriptor(level_idx, level.keys[i], ch,
                                      (b[i, 0], b[i, 1], b[i, 2], b[i, 3])))
    return out


def viewport(bbox_world: tuple[float, float, float, float], zoom: float, geometry: Geometry, *,
             channels: Sequence[str] = ("0",), current_level: int | None = None,
             hysteresis: float = _DEFAULT_HYSTERESIS) -> list[TileDescriptor]:
    """``select_tiles`` in the renderer's units: ``zoom`` is **screen pixels per world unit**.

    Convenience for call sites that already carry a zoom factor (the viewer's ``_cd``-style
    state); ``um_per_px = 1 / zoom``. The canonical entry point is ``select_tiles``.
    """
    z = float(zoom)
    if not np.isfinite(z) or z <= 0:
        raise ValueError(f"zoom must be finite and > 0, got {zoom!r}")
    return select_tiles(bbox_world, 1.0 / z, geometry, channels=channels,
                        current_level=current_level, hysteresis=hysteresis)


def _contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    """Is ``inner`` inside ``outer`` (closed)? Used to find a coarse tile covering a fine one."""
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


class TileCache:
    """Byte-budget LRU of decoded tiles + keep-parent-until-child-ready.

    Budget is **bytes** (``arr.nbytes``), not tile count: a level-0 FOV plane is ~8 MB while a
    coarse plate tile is ~128 KB, so counting tiles budgets nothing. A single tile bigger than
    the whole budget is admitted alone — refusing it would blank the screen.

    The fetch lifecycle belongs to the CALLER (no threads, no callbacks, no Qt in here):
    ``mark_pending(desc)`` before the read, then ``insert(desc, arr)`` or ``fetch_failed(desc)``.
    While a child is pending its nearest cached ancestor is *pinned* and skipped by eviction,
    so panning/zooming never punches a blank hole in the mosaic.
    """

    def __init__(self, budget_bytes: int) -> None:
        if budget_bytes < 0:
            raise ValueError(f"budget_bytes must be >= 0, got {budget_bytes}")
        self._budget = int(budget_bytes)
        self._cached: "OrderedDict[TileDescriptor, np.ndarray]" = OrderedDict()  # LRU: oldest first
        self._pending: "OrderedDict[TileDescriptor, TileDescriptor | None]" = OrderedDict()  # child -> pinned parent
        self._bytes = 0

    # ----- inspection -------------------------------------------------------------------
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

    # ----- fetch lifecycle --------------------------------------------------------------
    def mark_pending(self, desc: TileDescriptor) -> None:
        """Declare a fetch in flight: pins the nearest cached ancestor so it survives eviction."""
        if desc in self._cached or desc in self._pending:
            return
        self._pending[desc] = self._nearest_ancestor(desc)
        self._enforce_pin_cap()

    def insert(self, desc: TileDescriptor, arr: np.ndarray) -> None:
        """The fetch landed: cache the tile (MRU), release its parent's pin, trim to budget."""
        self._pending.pop(desc, None)                   # child ready -> parent unpinned
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
        """Drop every cached/pending tile matching ``predicate``; returns how many were cached.

        Streaming acquisition uses this: a newly written FOV invalidates the coarse plate-level
        tiles that cover it. (Contrast/LUT changes invalidate nothing — caching is per channel.)
        """
        doomed = [d for d in self._cached if predicate(d)]
        for d in doomed:
            self._bytes -= self._cached.pop(d).nbytes
        for d in [p for p in self._pending if predicate(p)]:
            del self._pending[d]
        for child, parent in list(self._pending.items()):
            if parent is not None and parent not in self._cached:
                self._pending[child] = self._nearest_ancestor(child)   # re-pin, parent is gone
        return len(doomed)

    # ----- the renderable set -----------------------------------------------------------
    def resolve(self, ideal: Iterable[TileDescriptor]) -> list[tuple[TileDescriptor, np.ndarray]]:
        """Ideal set -> what can actually be drawn right now, substituting cached ancestors.

        A missing tile falls back to its nearest cached ancestor (coarser, blurrier, but never a
        hole); with no ancestor the slot is simply absent and the caller draws nothing there.
        Once the child lands it replaces the parent, which drops out of the renderable set.
        """
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

    # ----- internals --------------------------------------------------------------------
    def _nearest_ancestor(self, desc: TileDescriptor) -> TileDescriptor | None:
        """The finest cached tile of the same channel, coarser level, whose bbox covers ``desc``."""
        best: TileDescriptor | None = None
        for other in self._cached:
            if other.channel != desc.channel or other.level <= desc.level:
                continue
            if not _contains(other.bbox_um, desc.bbox_um):
                continue
            if best is None or other.level < best.level:
                best = other
        return best

    def _enforce_pin_cap(self) -> None:
        """Keep pinned bytes under half the budget by dropping the OLDEST pending descriptors.

        Pins are unbounded otherwise: a fast pan can queue more parents than the cache holds and
        wedge eviction. Dropping the oldest pending request degrades to blur, never to deadlock.
        """
        cap = self._budget * _PIN_BUDGET_FRACTION
        while self._pending:
            pinned = {p for p in self._pending.values() if p is not None}
            if sum(self._cached[p].nbytes for p in pinned if p in self._cached) <= cap:
                return
            self._pending.popitem(last=False)            # oldest pending request loses its pin

    def _evict_to_budget(self) -> None:
        """Evict LRU-first until under budget, skipping pinned parents.

        If everything unpinned is gone and we are still over (one huge tile, or pins holding the
        rest), we stop: admitting an oversized tile alone beats blanking the screen.
        """
        if self._bytes <= self._budget:
            return
        pinned = {p for p in self._pending.values() if p is not None}
        for desc in list(self._cached):
            if self._bytes <= self._budget:
                return
            if desc in pinned or len(self._cached) == 1:
                continue                                 # never evict a pinned parent or the last tile
            self._bytes -= self._cached.pop(desc).nbytes
