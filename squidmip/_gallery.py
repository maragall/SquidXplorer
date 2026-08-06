"""Gallery View's PRODUCER: one region's mosaic per channel, at gallery resolution, off-thread.

Qt-free on purpose. Everything here is arithmetic over the reader, so the whole gallery can be
tested — scope, cropping, contrast, incremental arrival — without a window, and so the Qt side
(:mod:`squidmip._gallery_window`) holds nothing but layout.

ADOPTING hongquanli/gallery-view's RECIPE, NOT IMPORTING IT
-----------------------------------------------------------
Same precedent as :mod:`squidmip._napari3d` (read its docstring): gallery-view pins napari <0.6
and we run 0.6.6, so it cannot be a dependency. Its newest commit, "Add Region view: stitched
per-region MIPs (#7)", is the behaviour asked for, and these four decisions are taken from it
verbatim because they are the ones that are easy to get wrong:

1. **Rows are the unit, columns are the channel.** gallery-view's gallery is not a reflowing grid
   of thumbnails: it is a table, one row per sample, one column per wavelength, so a channel reads
   DOWN the page across samples. Here the row is a Region and the column is a channel. That is
   what "tile the selected regions side by side for comparison" means when there is more than one
   channel, and it is why a cell is per (Region, channel) rather than a composite: a composite
   cannot be compared channel by channel, which is the whole point of the view.
2. **The canvas comes from ALL the coordinates in scope; only the FOVs you have are painted.** A
   FOV that fails to read leaves a hole in the right PLACE rather than shifting its neighbours.
3. **Contrast is computed over the COVERED pixels only.** gallery-view's ``stitch_region`` takes
   its percentiles over ``mosaic[covered]``, and it says why: black gaps between FOVs otherwise
   drag the low end to zero and the whole region washes out.
4. **Integer block decimation, planned before anything is read.** The cell is ~512 px; reading a
   region at full resolution to shrink it afterwards is the cost the pyramid exists to avoid.

DIVERGED, twice, and both times toward code this repo already owns:

* **The contrast function is ours, not gallery-view's 0.5/99.5 percentiles.**
  :func:`squidmip._contrast.auto_contrast` is a port of ``maragall/stitcher``'s window (background
  histogram mode + 2 sigma to black, 99.9th percentile on top) and ``_contrast``'s own docstring
  records the measurement that a percentile low end lands INSIDE the fluorescence background and
  blows the image out. Using gallery-view's percentiles here would re-introduce, in a new window,
  the exact defect that module exists to fix.
* **The windowing is :func:`squidmip._montage.composite`, not a private ``mip_to_rgba``.**
  ``_montage.composite`` is declared the single home of the window-multiply-sum so that what is
  on screen and what is exported cannot drift. A gallery with its own ramp would be a second
  answer to "what does this channel look like at this window".

WHY IT DOES NOT GO THROUGH ``stitch_plate``
-------------------------------------------
``stitch_plate(reader, regions={region: [fov, ...]})`` genuinely crops and is the operator of
record, and the gallery's SCOPE is exactly its ``regions=`` mapping — that is deliberate, so the
two speak the same subset language. But it is a registered stitch: registration and fusion at
native resolution, ~0.9 GB per 27-FOV well. A gallery is a look, not a result; paying a stitch per
cell would put minutes in front of first paint. So the gallery fuses at PREVIEW placement, the
same "later FOV overwrites earlier" rule ``_mosaic_source.fuse_region_mosaic`` uses, through the
same ``_placement`` helpers — one geometry, two resolutions, never two implementations.

NOTHING HERE MAY RUN ON THE Qt THREAD, AND THE CONTRAST SEED IS PART OF "HERE"
------------------------------------------------------------------------------
This is the same defect ``_MosaicWorker`` was fixed for (``_workers.py``, merged at 400c63f), one
path over. ``_contrast.sample_plane`` picks the COARSEST pyramid level and therefore looks free,
but every level of a raw-preview pyramid is fused from the FOV TIFFs at its own decimation, so
materialising even the smallest rung decodes every FOV of the region: 128 ms of frozen UI per
region measured there, 493-604 ms on the machine it was reported from. A gallery is N regions, so
the same mistake here is N freezes rather than one.

So the window never calls anything in this module directly. :class:`GalleryCell` carries its
``window`` as DATA, computed in :class:`squidmip._gallery_window.GalleryWorker` beside the pixels
it describes, and the Qt thread only ever windows an array that is already in RAM at cell
resolution. ``test_gallery.py::test_the_gallery_never_reads_a_plane_on_the_qt_thread`` records the
thread ident of every ``reader.read`` and fails on the main one, so this cannot decay quietly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from squidmip._logpane import get_logger

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

#: The longest edge a gallery cell is fused to. gallery-view's ``target_longest_px`` is 1024 for a
#: cell it displays at <= 320; ours is half that because we display at <= 320 too and the cell is
#: re-scaled by Qt on every size change, so the only thing the extra 2x buys is memory. At 512 a
#: 4-channel 96-region gallery holds ~400 MB of uint16 mosaics, which the shared plane cache bounds.
TARGET_CELL_PX = 512

#: How many (Region, channel) cells one gallery will build. NOT a limit gallery-view has, and it is
#: here for a reason it does not face: this product opens 1536-well plates. 1536 regions x 4
#: channels is 6144 cells, ~630 MB of QPixmap alone before a single mosaic is fused, and no human
#: compares 6144 pictures. Over the cap the gallery shows the first N **and says so, naming the
#: selection as the control** — which is the point of the subset requirement, not a workaround for it.
MAX_GALLERY_CELLS = 256

def _every_z(levels: list[int]) -> list[int]:
    """gallery-view's region view: a per-region stitched MIP over the whole depth."""
    return levels


def _opening_z_only(levels: list[int]) -> list[int]:
    """One z — the opening z, which is the plane the viewer is parked on and the plate previews."""
    from squidmip._contrast import opening_z

    return [levels[opening_z(len(levels))]]


#: How each projection picks the z values one FOV contributes. A TABLE, and deliberately so.
#:
#: ``tests/test_operator_declaration.py::test_no_module_branches_on_an_operator_name`` fails the
#: build on any ``x == "mip"`` in the package, because "mip" is a registered projector and the
#: standing property is that adding an operator is a registry entry, never an edit to a module that
#: had to learn its name. This gallery setting is not that registry — it is a display choice that
#: happens to share a word — but a string comparison here has the same decay, and the same fix:
#: adding a projection is an entry in this dict, not a branch in two files. The comparison the
#: window used to make is a ``findData`` lookup now.
Z_SELECTORS = {"mip": _every_z, "plane": _opening_z_only}

#: Projections a cell can be, in the order a control should offer them.
PROJECTIONS = tuple(Z_SELECTORS)


@dataclass(frozen=True)
class GalleryScope:
    """WHICH pixels a gallery is over: regions, the FOVs of each, channels, timepoint, projection.

    The FOV mapping is the load-bearing field and is deliberately the SAME shape
    ``stitch_plate(regions=...)`` takes — ``{region: [fov, ...]}``. A gallery is therefore
    subset-native by construction rather than by a second selection mechanism: the plate's marquee
    and its selected wells already produce ``(region, fov)`` pairs
    (``PlateWindow.selected_region_fovs``), and :meth:`from_region_fovs` is the only adapter needed.

    Frozen, and the mapping is stored as a tuple-of-tuples, because a scope is passed to a worker
    thread. A dict that the UI could still mutate would be a data race with no error message.
    """

    regions: tuple[str, ...]
    fovs: tuple[tuple[str, tuple[int, ...]], ...]
    channels: tuple[str, ...]
    t: int = 0
    projection: str = "mip"
    #: True when the scope came from a plate SELECTION rather than the whole acquisition. Carried
    #: rather than re-derived so the window can say which it is without re-reading the plate.
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

        This is ``PlateWindow.selected_region_fovs()``'s output unchanged. Pairs naming a region or
        a FOV the acquisition does not have are DROPPED, exactly as ``stitch_plate`` drops an
        unknown region: a selection can outlive a re-ingest, and refusing the whole gallery over
        one stale well would be worse than showing the wells that are still there.
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
        """The regions whose scope is a PROPER SUBSET of their FOVs — i.e. genuinely cropped.

        Named so the window can say "3 of 4 regions are cropped to a FOV subset" rather than
        leaving the user to infer from the picture whether the marquee was honoured.
        """
        per = meta.get("fovs_per_region") or {}
        return tuple(r for r, fovs in self.fovs
                     if len(fovs) < len(per.get(r) or ()))

    def capped(self, max_cells: int = MAX_GALLERY_CELLS) -> tuple["GalleryScope", int]:
        """``(scope, dropped_regions)`` — the first whole regions that fit under *max_cells*.

        Truncation is by REGION, never by cell: half a region's channels is a row with holes that
        look like read failures. Returning the count instead of logging it means the window has to
        decide what to say about it, which is the point.
        """
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
        """Every ``(region, channel)`` to build, region-major — the order they will be queued in.

        Region-major and not channel-major so a row completes before the next one starts: a user
        comparing two wells gets one whole well early rather than every well's first channel.
        """
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

    ``image`` is native dtype (uint8/uint16) because :func:`squidmip._montage.composite` windows
    through a lookup table on integer stores and falls back to elementwise arithmetic otherwise —
    handing it float32 would silently take the slow path for every cell.

    ``window`` is ``None`` when :func:`squidmip._contrast.auto_contrast` refused, which it does for
    a blank or flat channel. That is carried through rather than replaced by a guess: a blank
    channel given a fabricated window renders as full-intensity noise, i.e. it reads as signal.
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
    """One field of a channel record, whatever a reader hands back for ``meta["channels"]``.

    Squid readers return plain dicts; the Acquisition model returns a pydantic ``Channel`` that
    supports ``[]`` and ``.get`` but is NOT a ``dict``. An ``isinstance(c, dict)`` test therefore
    silently skips every channel on the acquisition model and the gallery fuses ``str(<repr>)`` as
    a channel name — which is not a KeyError, it is a read of a plane that does not exist, for
    every FOV, reported as "no FOV could be read". Measured on ``sim_5d_2x2_t3``. So: duck-type.
    """
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
    """The key a gallery cell occupies in ``_mosaic_source.plane_cache()``.

    ``"gallery"`` leads so the namespace can never collide with the pyramid's own
    ``(token, region, channel, t, step, z)`` keys, and the FOV tuple is IN the key because that is
    what a subset changes: the same region at the same step with two different FOV selections is
    two different pictures, and serving one for the other is the class of bug this project calls
    "a plausible image".
    """
    return ("gallery", what, str(token), str(region), tuple(int(f) for f in fovs),
            str(channel), int(t), str(projection), float(step))


def plan_cell(meta: Mapping, region: str, fovs: Sequence[int],
              *, target_px: int = TARGET_CELL_PX):
    """``(offsets, out_h, out_w, step, full_shape, dtype)`` a cell WOULD have. Reads nothing.

    Pure geometry over :func:`squidmip._placement.fov_offsets_px`, which normalises the top-left
    FOV of whatever set it is handed to ``(0, 0)`` — that normalisation IS the crop, and it is why
    a FOV subset needs no special case anywhere below. Returns ``None``, never a guess, when the
    acquisition carries no stage positions or no pixel size (the same signal ``fuse_region_mosaic``
    and ``mosaic_bbox_um`` use).
    """
    from squidmip._placement import fov_offsets_px, mosaic_extent_px

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
    """The z values one FOV contributes, via :data:`Z_SELECTORS` — a lookup, never a name branch.

    ``meta["z_levels"]`` holds the z values as they appear on disk, and ``reader.read`` keys on
    that value — so this iterates the LEVELS, like ``_FocusWorker`` does, rather than
    ``range(n_z)``. On every acquisition seen so far the two agree; when they do not, ``range``
    is the one that raises KeyError halfway through a gallery.

    A single-plane acquisition short-circuits before the lookup: every projection is the same
    plane there, and reading the table would only be a slower way to say so.
    """
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

    The port of gallery-view's ``stitch_region``, with its four decisions intact (see the module
    docstring) and its two divergences. Returns ``None`` when the geometry is not derivable, or
    when *should_stop* asked to stop mid-way — never a partial cell dressed as a whole one.

    *cache* and *token*: pass ``_mosaic_source.plane_cache()`` and
    ``_mosaic_source.source_token(reader)`` to share the process-wide byte budget. Both or
    neither; a token without a cache caches nothing, which is a no-op rather than an error, so a
    stub reader with no identity simply runs uncached.
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
            # A HOLE, in the right place, counted. gallery-view's canvas is built from every
            # coordinate for exactly this reason: dropping the FOV instead would shift its
            # neighbours and the region would be misregistered rather than incomplete.
            unreadable.append(int(fov))
            continue
        row, col = offsets[fov]
        r0, c0 = row // istep, col // istep
        r1 = min(r0 + tile.shape[0], out_h)
        c1 = min(c0 + tile.shape[1], out_w)
        if r1 > r0 and c1 > c0:
            # Later FOVs overwrite earlier ones in the overlap — PREVIEW placement, the same rule
            # `_mosaic_source.fuse_region_mosaic` states. No blending, no registration; the
            # stitch operator is what produces a mosaic of record.
            image[r0:r1, c0:c1] = tile[: r1 - r0, : c1 - c0]
            covered[r0:r1, c0:c1] = True

    if unreadable and len(unreadable) == len(fovs):
        # Every FOV bad is not a picture at all. A black cell here would report a read failure as
        # empty tissue, which is the silent failure this codebase has six confirmed instances of.
        raise ValueError(
            f"{region}/{channel} t={t}: not one of the {len(fovs)} FOV(s) in scope could be read, "
            f"so there is no mosaic. First failures: {unreadable[:3]}"
        )

    if not unreadable:
        # A DEGRADED CELL IS NOT CACHED, for two reasons that point the same way. It would outlive
        # the transient that caused it — a disk hiccup or a file being written would become this
        # session's permanent picture of that region, unrefreshable. And the cache stores pixels
        # only, so a hit reconstructs the cell with `unreadable=()`: the hole would still be on
        # screen while the caption stopped saying so, which is worse than the hole. A clean cell
        # is deterministic and safe to keep; a holed one is a read to retry.
        _cache_cell(cache, token, region, fovs, channel, t, projection, step, image, covered)
    return GalleryCell(str(region), str(channel), image, covered,
                       cell_window(image, covered), step, full_shape, len(fovs),
                       tuple(unreadable))


def _fov_tile(reader, region, fov, channel, zs: Sequence[int], t: int, istep: int):
    """One FOV's contribution, decimated on read. ``None`` when nothing of it could be read.

    ``frame[::step, ::step]`` strides at read, so a coarse cell allocates and pastes a fraction of
    a frame — the decode itself is whole-frame either way, which is why the MIP loop maxes into an
    already-strided accumulator instead of stacking full frames and reducing at the end. Peak memory
    per FOV is therefore ONE decimated plane, not the z-stack.

    Planes of a FOV are assumed to be the same shape and are CROPPED to the common one when they
    are not, rather than raising: a ragged z is a broken acquisition, but losing the whole cell over
    it would hide the FOVs that are fine. ``fuse_region_pyramid`` raises on a ragged z because a
    misaligned pyramid LEVEL misregisters the stack; here the z axis is being collapsed away, so
    there is nothing left to misregister.
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
            # A COPY, not the strided view: `sub` aliases the decoded frame, and returning it would
            # hand a cache a window onto a buffer whose lifetime is this loop.
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
    """``(lo, hi)`` for one cell, over the COVERED pixels only, or ``None`` if there is no window.

    gallery-view's lesson, kept: percentiles taken over the whole canvas include the black gaps
    between FOVs, which drags the low end to zero and washes the region out. The *function* is
    ours (``_contrast.auto_contrast``: background histogram mode + 2 sigma to black, 99.9th
    percentile on top), because a percentile low end is the thing that module exists to not do.
    """
    from squidmip._contrast import auto_contrast

    data = image if covered is None else image[covered]
    if data.size == 0:
        return None
    return auto_contrast(data)


def shared_windows(cells: Iterable[GalleryCell]) -> dict[str, tuple[float, float]]:
    """One window PER CHANNEL across every cell — the union of the per-cell windows.

    gallery-view is per-cell and only per-cell, and for its use (one sample at a time, judged on
    its own) that is right. A gallery of regions is for COMPARING them, and per-cell contrast makes
    a dim well and a bright well look the same, which is the one question the view exists to
    answer. So both are offered and this is the default; the widest window that contains every
    cell's is the honest join, because narrowing it would clip a region the user chose to look at.

    Channels whose every cell refused a window are ABSENT from the result rather than present with
    a fabricated one — see :attr:`GalleryCell.window`.
    """
    out: dict[str, tuple[float, float]] = {}
    for cell in cells:
        if cell.window is None:
            continue
        lo, hi = cell.window
        have = out.get(cell.channel)
        out[cell.channel] = (lo, hi) if have is None else (min(have[0], lo), max(have[1], hi))
    return out
