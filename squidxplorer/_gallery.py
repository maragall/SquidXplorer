"""Gallery View's PRODUCER: one region's mosaic per channel, at gallery resolution, off-thread.

Qt-free on purpose; the Qt side (:mod:`squidxplorer._gallery_window`) holds nothing but layout.
Cells fuse at PREVIEW placement (later FOV overwrites earlier), never through the stitcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from squidxplorer._logpane import get_logger

log = get_logger("gallery")

__all__ = [
    "GalleryScope",
    "GalleryCell",
    "MAX_GALLERY_CELLS",
    "TARGET_CELL_PX",
    "cell_cache_key",
    "fuse_gallery_cell",
    "cell_window",
    "shared_windows",
]

#: The longest edge a gallery cell is fused to.
TARGET_CELL_PX = 512

#: How many (Region, channel) cells one gallery will build. Over the cap the gallery shows the
#: first N and says so, naming the selection as the control.
MAX_GALLERY_CELLS = 256

def _every_z(levels: list[int]) -> list[int]:
    """gallery-view's region view: a per-region stitched MIP over the whole depth."""
    return levels


def _opening_z_only(levels: list[int]) -> list[int]:
    """One z — the opening z, which is the plane the viewer is parked on and the plate previews."""
    from squidxplorer._contrast import opening_z

    return [levels[opening_z(len(levels))]]


#: How each projection picks the z values one FOV contributes — a lookup, never a name branch.
Z_SELECTORS = {"mip": _every_z, "plane": _opening_z_only}

#: Projections a cell can be, in the order a control should offer them.
PROJECTIONS = tuple(Z_SELECTORS)


@dataclass(frozen=True)
class GalleryScope:
    """WHICH pixels a gallery is over: regions, the FOVs of each, channels, timepoint, projection.

    The FOV mapping is the same shape ``run_plate(regions=...)`` takes. Frozen (mapping stored
    as a tuple-of-tuples) because a scope is passed to a worker thread.
    """

    regions: tuple[str, ...]
    fovs: tuple[tuple[str, tuple[int, ...]], ...]
    channels: tuple[str, ...]
    t: int = 0
    projection: str = "mip"
    #: True when the scope came from a plate SELECTION rather than the whole acquisition.
    from_selection: bool = False

    def __post_init__(self) -> None:
        if self.projection not in PROJECTIONS:
            raise ValueError(
                f"projection must be one of {PROJECTIONS}, got {self.projection!r}."
            )

    # -- construction ---------------------------------------------------------------------------

    @classmethod
    def whole(cls, meta: Mapping, *, channels: Optional[Sequence[str]] = None,
              t: int = 0, projection: str = "mip") -> "GalleryScope":
        """Every region of the acquisition, every FOV of each — the no-selection case."""
        per = dict((meta.get("fovs_per_region") or {}))
        regions = [str(r) for r in (meta.get("regions") or list(per)) if str(r) in per]
        return cls._build(regions, {r: per[r] for r in regions},
                          _channel_names(meta, channels), t, projection, from_selection=False)

    @classmethod
    def from_region_fovs(cls, meta: Mapping, pairs: Iterable[tuple],
                         *, channels: Optional[Sequence[str]] = None,
                         t: int = 0, projection: str = "mip") -> "GalleryScope":
        """A plate selection — ``[(region, fov), ...]`` — as a scope, in plate order.

        Pairs naming a region or FOV the acquisition does not have are dropped, as
        ``run_plate`` drops an unknown region.
        """
        per = dict((meta.get("fovs_per_region") or {}))
        order = [str(r) for r in (meta.get("regions") or list(per))]
        rank = {r: i for i, r in enumerate(order)}
        picked: dict[str, list[int]] = {}
        for region, fov in pairs:
            region = str(region)
            known = per.get(region)
            if known is None or int(fov) not in set(int(f) for f in known):
                continue
            picked.setdefault(region, [])
            if int(fov) not in picked[region]:
                picked[region].append(int(fov))
        regions = sorted(picked, key=lambda r: (rank.get(r, len(rank)), r))
        return cls._build(regions, picked, _channel_names(meta, channels), t, projection,
                          from_selection=True)

    @classmethod
    def _build(cls, regions, picked, channels, t, projection, *, from_selection):
        regions = tuple(str(r) for r in regions if picked.get(str(r)))
        return cls(
            regions=regions,
            fovs=tuple((r, tuple(int(f) for f in picked[r])) for r in regions),
            channels=tuple(str(c) for c in channels),
            t=int(t),
            projection=str(projection),
            from_selection=bool(from_selection),
        )

    # -- reading it -----------------------------------------------------------------------------

    def fovs_of(self, region: str) -> tuple[int, ...]:
        for name, fovs in self.fovs:
            if name == region:
                return fovs
        return ()

    @property
    def cell_count(self) -> int:
        return len(self.regions) * len(self.channels)

    def is_empty(self) -> bool:
        return not self.regions or not self.channels

    def crops(self, meta: Mapping) -> tuple[str, ...]:
        """The regions whose scope is a PROPER SUBSET of their FOVs — i.e. genuinely cropped."""
        per = meta.get("fovs_per_region") or {}
        return tuple(r for r, fovs in self.fovs
                     if len(fovs) < len(per.get(r) or ()))

    def capped(self, max_cells: int = MAX_GALLERY_CELLS) -> tuple["GalleryScope", int]:
        """``(scope, dropped_regions)`` — the first whole regions that fit under *max_cells*."""
        if not self.channels or self.cell_count <= max_cells:
            return self, 0
        keep = max(1, int(max_cells) // len(self.channels))
        if keep >= len(self.regions):
            return self, 0
        kept = self.regions[:keep]
        return (
            GalleryScope(
                regions=kept,
                fovs=tuple((r, f) for r, f in self.fovs if r in set(kept)),
                channels=self.channels,
                t=self.t,
                projection=self.projection,
                from_selection=self.from_selection,
            ),
            len(self.regions) - keep,
        )

    def cells(self) -> list[tuple[str, str]]:
        """Every ``(region, channel)`` to build, region-major — the order they will be queued in."""
        return [(r, c) for r in self.regions for c in self.channels]

    def describe(self, meta: Optional[Mapping] = None) -> str:
        """One line naming the scope, for the console and the window's status bar."""
        what = "selection" if self.from_selection else "whole acquisition"
        n_fov = sum(len(f) for _r, f in self.fovs)
        line = (f"{len(self.regions)} region(s), {n_fov} FOV(s), {len(self.channels)} channel(s), "
                f"t={self.t}, {self.projection} — {what}")
        if meta is not None:
            cropped = self.crops(meta)
            if cropped:
                line += f"; {len(cropped)} region(s) cropped to a FOV subset"
        return line


@dataclass(frozen=True)
class GalleryCell:
    """One finished cell: a Region's mosaic for ONE channel, plus everything needed to draw it.

    ``window`` is ``None`` when auto-contrast refused (blank/flat channel) — carried through,
    never replaced by a guess.
    """

    region: str
    channel: str
    image: np.ndarray
    covered: np.ndarray
    window: Optional[tuple[float, float]]
    step: float
    full_shape: tuple[int, int]
    n_fovs: int
    unreadable: tuple[int, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.image.shape[0]), int(self.image.shape[1]))

    @property
    def has_holes(self) -> bool:
        return bool(self.unreadable)


# --- the producer -------------------------------------------------------------------------------


def channel_field(channel: Any, field: str, default=None):
    """One field of a channel record, duck-typed: readers hand back dicts OR pydantic Channels."""
    try:
        return channel[field]
    except Exception:                                   # noqa: BLE001 - fall through to attributes
        return getattr(channel, field, default)


def _channel_names(meta: Mapping, channels: Optional[Sequence[str]]) -> list[str]:
    if channels is not None:
        return [str(c) for c in channels]
    return [str(channel_field(c, "name", c)) for c in (meta.get("channels") or [])]


def cell_cache_key(token: str, region: str, fovs: Sequence[int], channel: str,
                   t: int, projection: str, step: float, what: str) -> tuple:
    """The key a gallery cell occupies in ``_mosaic_source.plane_cache()``; the FOV tuple is IN it."""
    return ("gallery", what, str(token), str(region), tuple(int(f) for f in fovs),
            str(channel), int(t), str(projection), float(step))


def plan_cell(meta: Mapping, region: str, fovs: Sequence[int],
              *, target_px: int = TARGET_CELL_PX):
    """``(offsets, out_h, out_w, step, full_shape, dtype)`` a cell WOULD have. Reads nothing.

    Returns ``None``, never a guess, when the acquisition carries no stage positions or pixel size.
    """
    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    pixel_size = meta.get("pixel_size_um")
    if not positions or pixel_size in (None, 0):
        return None
    fovs = [int(f) for f in fovs]
    if not fovs:
        return None
    frame_h, frame_w = (int(v) for v in meta["frame_shape"])
    try:
        offsets = fov_offsets_px(positions, str(region), fovs, pixel_size)
        full_h, full_w = mosaic_extent_px(offsets, (frame_h, frame_w))
    except (KeyError, ValueError):
        return None
    step = max(1, int(np.ceil(max(full_h, full_w) / float(max(1, int(target_px))))))
    out_h = int(np.ceil(full_h / step))
    out_w = int(np.ceil(full_w / step))
    dtype = np.dtype(meta.get("dtype", "uint16"))
    return offsets, out_h, out_w, float(step), (int(full_h), int(full_w)), dtype


def _z_indices(meta: Mapping, projection: str) -> list[int]:
    """The z values one FOV contributes, via :data:`Z_SELECTORS` — iterates the on-disk levels."""
    levels = [int(z) for z in (meta.get("z_levels") or [0])]
    if len(levels) <= 1:
        return levels
    return Z_SELECTORS[projection](levels)


def fuse_gallery_cell(
    reader: Any,
    meta: Mapping,
    region: str,
    fovs: Sequence[int],
    channel: str,
    *,
    t: int = 0,
    projection: str = "mip",
    target_px: int = TARGET_CELL_PX,
    cache: Any = None,
    token: Optional[str] = None,
    should_stop=None,
) -> Optional[GalleryCell]:
    """Fuse ONE gallery cell: this region's FOVs, this channel, placed by stage position.

    Returns ``None`` when the geometry is not derivable or *should_stop* asked to stop —
    never a partial cell dressed as a whole one.
    """
    plan = plan_cell(meta, region, fovs, target_px=target_px)
    if plan is None:
        return None
    offsets, out_h, out_w, step, full_shape, dtype = plan
    fovs = [int(f) for f in fovs]
    istep = int(step)

    hit = _cached_cell(cache, token, region, fovs, channel, t, projection, step,
                       (out_h, out_w))
    if hit is not None:
        image, covered = hit
        return GalleryCell(str(region), str(channel), image, covered,
                           cell_window(image, covered), step, full_shape, len(fovs))

    zs = _z_indices(meta, projection)
    image = np.zeros((out_h, out_w), dtype=dtype)
    covered = np.zeros((out_h, out_w), dtype=bool)
    unreadable: list[int] = []

    for fov in fovs:
        if should_stop is not None and should_stop():
            return None
        tile = _fov_tile(reader, region, fov, channel, zs, t, istep)
        if tile is None:
            # A hole, in the right place, counted: dropping the FOV would shift its neighbours.
            unreadable.append(int(fov))
            continue
        row, col = offsets[fov]
        r0, c0 = row // istep, col // istep
        r1 = min(r0 + tile.shape[0], out_h)
        c1 = min(c0 + tile.shape[1], out_w)
        if r1 > r0 and c1 > c0:
            # Later FOVs overwrite earlier ones in the overlap — preview placement, no blending.
            image[r0:r1, c0:c1] = tile[: r1 - r0, : c1 - c0]
            covered[r0:r1, c0:c1] = True

    if unreadable and len(unreadable) == len(fovs):
        # Every FOV bad is not a picture at all; a black cell would report a read failure
        # as empty tissue.
        raise ValueError(
            f"{region}/{channel} t={t}: not one of the {len(fovs)} FOV(s) in scope could be read, "
            f"so there is no mosaic. First failures: {unreadable[:3]}"
        )

    if not unreadable:
        # A degraded cell is not cached: it would outlive the transient, and a hit reconstructs
        # with unreadable=() so the hole would stop being reported.
        _cache_cell(cache, token, region, fovs, channel, t, projection, step, image, covered)
    return GalleryCell(str(region), str(channel), image, covered,
                       cell_window(image, covered), step, full_shape, len(fovs),
                       tuple(unreadable))


def _fov_tile(reader, region, fov, channel, zs: Sequence[int], t: int, istep: int):
    """One FOV's contribution, decimated on read. ``None`` when nothing of it could be read.

    Maxes into an already-strided accumulator, so peak memory per FOV is one decimated plane.
    Ragged z is cropped to the common shape rather than raising.
    """
    acc = None
    for z in zs:
        try:
            frame = reader.read(region, int(fov), channel, int(z), int(t))
        except Exception:                       # noqa: BLE001 - counted by the caller, never hidden
            continue
        if frame is None:
            continue
        frame = np.asarray(frame)
        if frame.ndim != 2:
            frame = frame.reshape(frame.shape[-2:])
        sub = frame[::istep, ::istep]
        if acc is None:
            # A COPY, not the strided view: `sub` aliases the decoded frame's buffer.
            acc = np.ascontiguousarray(sub)
            continue
        h = min(acc.shape[0], sub.shape[0])
        w = min(acc.shape[1], sub.shape[1])
        acc = np.maximum(acc[:h, :w], sub[:h, :w])
    return acc


def _cached_cell(cache, token, region, fovs, channel, t, projection, step, shape):
    if cache is None or token is None:
        return None
    img = cache.get(cell_cache_key(token, region, fovs, channel, t, projection, step, "img"))
    cov = cache.get(cell_cache_key(token, region, fovs, channel, t, projection, step, "cov"))
    if img is None or cov is None:
        return None
    if img.shape != tuple(shape) or cov.shape != tuple(shape):
        return None                          # geometry moved under the key; recompute rather than draw it
    return img, cov


def _cache_cell(cache, token, region, fovs, channel, t, projection, step, image, covered):
    if cache is None or token is None:
        return
    try:
        cache.put(cell_cache_key(token, region, fovs, channel, t, projection, step, "img"), image)
        cache.put(cell_cache_key(token, region, fovs, channel, t, projection, step, "cov"), covered)
    except ValueError as exc:                # a cell over the whole budget: say so, keep drawing
        log.warning("gallery cell %s/%s not cached: %s", region, channel, exc)


# --- contrast -----------------------------------------------------------------------------------


def cell_window(image: np.ndarray, covered: Optional[np.ndarray] = None):
    """``(lo, hi)`` for one cell, over the COVERED pixels only, or ``None`` if there is no window."""
    from squidxplorer._contrast import auto_contrast

    data = image if covered is None else image[covered]
    if data.size == 0:
        return None
    return auto_contrast(data)


def shared_windows(cells: Iterable[GalleryCell]) -> dict[str, tuple[float, float]]:
    """One window PER CHANNEL across every cell — the union of the per-cell windows.

    Channels whose every cell refused a window are absent, never fabricated.
    """
    out: dict[str, tuple[float, float]] = {}
    for cell in cells:
        if cell.window is None:
            continue
        lo, hi = cell.window
        have = out.get(cell.channel)
        out[cell.channel] = (lo, hi) if have is None else (min(have[0], lo), max(have[1], hi))
    return out
