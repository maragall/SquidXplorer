"""Fused mosaics for the napari pane — the unit displayed is a MOSAIC, never a single FOV.

Two sources, because acquisitions arrive in two states:

* **Written OME-Zarr** (what IMA-217 writes, what ``stitch_plate`` produces). Read the
  multiscale pyramid LAZILY, one dask array per level. ``_zarr_store`` chunks full-res planes
  at 512 px, which is ``_tiling.DEFAULT_TILE_PX``, so one tile read is exactly one chunk read —
  keep those two equal or read amplification comes back.
* **Raw acquisition** (the tissue set: TIFFs, no pyramid on disk). Fuse the region's FOVs into
  one plane by pasting each frame at the pixel offset ``_placement.fov_offsets_px`` computes
  from stage positions. This is what makes a mosaic appear ON OPEN, before any operator runs.

Why not ``viewer.open(path, plugin="napari-ome-zarr")``, which is what ian-stitcher does:

1. ``napari-ome-zarr`` is **not installed** in this environment, and disk is too tight to add
   it plus its ``ome-zarr``/npe2 dependency chain right now.
2. It would pull in the npe2 plugin surface we deliberately avoided by embedding a bare
   ``QtViewer`` with no napari ``Window`` — the whole point of the "watch out for feature bloat"
   constraint.
3. **It names layers and nothing else.** Recovering which channel a layer is would then mean
   parsing ``layer.name``, which is precisely the bug class this design refuses:
   ian-stitcher does ``extractWavelength(layer.name)``, petakit's reader emits channel names its
   own regex cannot parse, and 3f1bf3f fixed ``Fluorescence_488_nm_Ex`` failing a parser that
   wanted ``\\s*nm``.

Reading the multiscale directly is what the plugin does internally anyway (dask over zarr), and
it lets identity come from OUR metadata. The tradeoff is that we do not inherit the plugin's
handling of exotic NGFF layouts; ours are written by us, and ``reader._Multiscale`` already
parses the spec.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# Level 0 of a full plate mosaic can be far larger than RAM, so the zarr path stays lazy and the
# raw path is bounded by _MAX_FUSED_PX below.


def level_paths(group: Path) -> list[Path]:
    """Every resolution level of an OME-NGFF image group, highest resolution first.

    ``reader._Multiscale`` parses ``multiscales[0]`` but exposes only ``datasets[0]`` — it was
    written for readers that only ever want full resolution. A pyramid renderer needs all of
    them, so the same ``datasets`` list is read here rather than re-deriving level order from
    directory names (which sorts "10" before "2").
    """
    group = Path(group)
    doc = json.loads((group / "zarr.json").read_text())
    attrs = doc.get("attributes", {})
    ome = attrs.get("ome", attrs)
    multiscales = ome.get("multiscales") or []
    if not multiscales:
        raise ValueError(f"{group}: no 'multiscales' metadata; not an OME-NGFF image group.")
    datasets = multiscales[0].get("datasets") or []
    if not datasets:
        raise ValueError(f"{group}: multiscales carries no 'datasets' (no resolution levels).")
    return [group / str(d["path"]) for d in datasets]


def open_pyramid(group, *, t: int = 0, c: int = 0, z: int = 0) -> list:
    """Lazy 2-D dask pyramid for one (t, c, z) of a written field/mosaic group.

    Returns the list napari wants for ``multiscale=True``: one array per level, highest
    resolution first. Nothing is read here — only opened — so this is cheap even on a plate
    whose level 0 does not fit in RAM.

    Levels whose shape does not match the level-0 axis layout are dropped rather than guessed at.
    """
    import dask.array as da
    import zarr

    out = []
    for path in level_paths(group):
        arr = zarr.open_array(str(path), mode="r")
        d = da.from_array(arr, chunks=arr.chunks)
        # Squid canonical order is (t, c, z, y, x); 2-D/3-D stores are legal NGFF too.
        if d.ndim == 5:
            d = d[t, c, z]
        elif d.ndim == 4:
            d = d[t, c]
        elif d.ndim == 3:
            d = d[z]
        elif d.ndim != 2:
            raise ValueError(f"{path}: unsupported rank {d.ndim} for a mosaic plane.")
        out.append(d)

    return strictly_decreasing_levels(out)


def strictly_decreasing_levels(levels: list) -> list:
    """Drop any level that does not shrink in BOTH displayed axes. The one pyramid guard.

    napari requires strictly decreasing level sizes; a source that emitted a duplicate or an
    out-of-order level would otherwise make it pick nonsense. Drop, don't reorder: an unexpected
    order means the source is wrong and guessing hides that.

    Compares the TRAILING two axes so it serves both pyramid producers in this module — the
    written-OME-Zarr one (2-D levels) and the raw preview one (``(z, y, x)`` levels, which
    downsample y and x only and must keep z intact). Two copies of this rule that disagreed about
    rank is exactly the defect shape this project keeps paying for.
    """
    if not levels:
        raise ValueError("a pyramid needs at least one level; got none.")
    kept = [levels[0]]
    for d in levels[1:]:
        if d.shape[-2] < kept[-1].shape[-2] and d.shape[-1] < kept[-1].shape[-1]:
            kept.append(d)
    return kept


# A fused RAW mosaic is materialised in RAM (there is no pyramid on disk to be lazy about), so
# it is bounded. 8192 px on the long side is ~134 MB at uint16 — enough to see a 28-FOV strip
# whole, small enough not to evict the plate. Beyond that the frames are decimated on read.
_MAX_FUSED_PX = 8192


def fuse_region_mosaic(
    reader: Any,
    meta: dict,
    region: str,
    channel: str,
    *,
    z: int = 0,
    t: int = 0,
    max_px: int = _MAX_FUSED_PX,
) -> Optional[tuple[np.ndarray, float]]:
    """Paste a region's FOVs into ONE plane, placed by stage position.

    Returns ``(mosaic, step)`` where ``step`` is the decimation factor applied (1 = native), or
    ``None`` when the acquisition carries no stage positions / no pixel size. That ``None`` is
    the same "not derivable, do not guess" signal ``_mosaic_boxes`` returns ``{}`` for — a
    mosaic without positions would be a wrong picture, not a rough one.

    A FOV that fails to read is left as zeros and counted, never silently skipped: the caller
    reports the count. Six confirmed silent failures in this project say a hole must be visible.
    """
    from squidmip._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    pixel_size = meta.get("pixel_size_um")
    if not positions or pixel_size in (None, 0):
        return None

    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        return None

    frame_h, frame_w = (int(v) for v in meta["frame_shape"])
    try:
        offsets = fov_offsets_px(positions, region, fovs, pixel_size)
        full_h, full_w = mosaic_extent_px(offsets, (frame_h, frame_w))
    except (KeyError, ValueError):
        return None

    step = max(1, int(np.ceil(max(full_h, full_w) / float(max_px))))
    out_h = int(np.ceil(full_h / step))
    out_w = int(np.ceil(full_w / step))

    dtype = np.dtype(meta.get("dtype", "uint16"))
    mosaic = np.zeros((out_h, out_w), dtype=dtype)

    for fov in fovs:
        row, col = offsets[fov]
        try:
            frame = reader.read(region, fov, channel, z, t)
        except Exception:
            continue          # leave zeros; the caller counts and reports the gap
        if frame is None:
            continue
        frame = np.asarray(frame)
        if frame.ndim != 2:
            frame = frame.reshape(frame.shape[-2:])
        sub = frame[::step, ::step]
        r0, c0 = row // step, col // step
        r1, c1 = min(r0 + sub.shape[0], out_h), min(c0 + sub.shape[1], out_w)
        if r1 > r0 and c1 > c0:
            # Later FOVs overwrite earlier ones in the overlap. This is a PREVIEW placement, not
            # a stitch: no blending, no registration. stitch_plate() is what produces a fused
            # mosaic of record, and its output comes back through open_pyramid() instead.
            mosaic[r0:r1, c0:c1] = sub[: r1 - r0, : c1 - c0]

    return mosaic, float(step)


#: Refuse to materialise more than this in one slice request. Mirrors the plane budget in
#: hongquanli/record-zstack-viewer, which raises rather than quietly dragging a machine into swap.
_PLANE_BUDGET_BYTES = 2 * 1024 ** 3


def _planned_plane(meta: dict, region: str, max_px: int):
    """``(out_h, out_w, step, dtype)`` a fused plane WOULD have. Pure geometry, reads nothing.

    Exists so the plane budget can be enforced before any allocation, and so the lazy stack can
    declare its shape without fusing a plane to find out.
    """
    from squidmip._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    pixel_size = meta.get("pixel_size_um")
    if not positions or pixel_size in (None, 0):
        return None
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        return None
    try:
        offsets = fov_offsets_px(positions, region, fovs, pixel_size)
        full_h, full_w = mosaic_extent_px(offsets, tuple(int(v) for v in meta["frame_shape"]))
    except (KeyError, ValueError):
        return None

    step = max(1, int(np.ceil(max(full_h, full_w) / float(max_px))))
    return (int(np.ceil(full_h / step)), int(np.ceil(full_w / step)),
            float(step), np.dtype(meta.get("dtype", "uint16")))


def fuse_region_stack(
    reader: Any,
    meta: dict,
    region: str,
    channel: str,
    *,
    t: int = 0,
    max_px: int = _MAX_FUSED_PX,
):
    """A LAZY ``(z, y, x)`` mosaic stack — one fused plane materialised per visible z.

    This is what makes z navigable. napari puts a native dimension slider on every axis it is
    not displaying, so handing it a 3-D array is all that is required; handing it a 2-D array
    (which is what fusing at a fixed ``z`` produces) leaves no axis to put a slider on, which is
    exactly why z was not controllable.

    ARCHITECTURE TAKEN FROM ``hongquanli/record-zstack-viewer`` (Squid's own author), which
    solves this problem for the same data. Its ``FovDataWrapper`` exposes the source's NAMED
    axes to the viewer and materialises ONE plane per visible slice via ``read_plane`` — "requests
    only cost what is visible" — with an explicit budget that raises rather than swapping, and it
    relies on the viewer hiding singleton sliders so the z selector appears exactly when
    ``n_planes > 1``.

    WHERE WE DIVERGE, and why: that viewer is built on **ndv**, whose sliders are driven by a
    ``DataWrapper``. napari has no such hook — its dimension sliders are driven by the ARRAY's
    shape. So the same architecture is expressed as a dask array whose per-z blocks are computed
    on demand: one fused plane per slider position, nothing eager, and napari's own slider is the
    control surface rather than one we build. Choosing napari's native slider over porting the
    DataWrapper is deliberate: a ported wrapper would be a second control surface to keep in sync,
    which is the defect shape this whole effort is trying to remove.

    NOT taken from it: ``_auto_clims`` widens a collapsed window "minimally so the LUT" works.
    ``_pct_window`` deliberately REFUSES that widening — IMA-269 found the loupe's separate
    compositor doing exactly this and rendering a sparse channel WHITE where the plate renders it
    BLACK. Julio's repos are prior art to read critically, not to copy wholesale.
    """
    import dask.array as da

    nz = int(meta.get("n_z") or 1)

    # Size the plane from GEOMETRY, before anything is allocated. The budget check has to come
    # before the allocation it guards; probing first would allocate the very plane we are trying
    # to refuse.
    planned = _planned_plane(meta, region, max_px)
    if planned is None:
        return None
    h, w, step, dtype = planned

    # The budget is PER PLANE, not for the stack: the stack is lazy, so only the visible z is
    # ever in RAM. A single plane over budget is refused loudly — that is a real condition
    # (max_px raised, or a huge region) and discovering it as a swap storm is worse.
    per_plane = h * w * dtype.itemsize
    if per_plane > _PLANE_BUDGET_BYTES:
        raise MemoryError(
            f"{region}/{channel}: one fused plane is {per_plane / 1e9:.1f} GB "
            f"({h}x{w} {dtype}), over the {_PLANE_BUDGET_BYTES / 1e9:.1f} GB plane budget. "
            "Lower max_px rather than letting this page the machine."
        )

    if nz <= 1:
        # A singleton z axis would give napari a slider with one position. Return the plane so
        # the axis does not exist at all, matching the "hide singleton sliders" behaviour.
        probe = fuse_region_mosaic(reader, meta, region, channel, z=0, t=t, max_px=max_px)
        if probe is None:
            return None
        return probe[0], probe[1], 1

    # ONE-PLANE CACHE, and only one. Fusing a plane of the 10x tissue set costs ~150 ms, and it
    # was being paid TWICE for the same z on every region change: once by whoever samples the
    # stack to decide a contrast window, and again by napari when it slices the same z to draw
    # it. The cache holds the LAST plane only, so it is bounded at one plane per channel stack
    # (~55 MB on that set) and dies with the stack when the region changes. It is not a general
    # z cache: scrubbing z still costs one fuse per plane, which is the "requests only cost what
    # is visible" rule this design is built on.
    _cache: dict = {"z": None, "plane": None}
    _cache_lock = __import__("threading").Lock()

    def _plane(z: int):
        z = int(z)
        with _cache_lock:
            if _cache["z"] == z and _cache["plane"] is not None:
                return _cache["plane"]
        got = fuse_region_mosaic(reader, meta, region, channel, z=int(z), t=t, max_px=max_px)
        if got is None:
            out = np.zeros((h, w), dtype=dtype)
        else:
            arr = got[0]
            if arr.shape != (h, w):    # a ragged z would silently misalign the stack
                out = np.zeros((h, w), dtype=dtype)
                out[: min(h, arr.shape[0]), : min(w, arr.shape[1])] = \
                    arr[: min(h, arr.shape[0]), : min(w, arr.shape[1])]
            else:
                out = arr
        with _cache_lock:
            _cache["z"], _cache["plane"] = z, out
        return out

    from dask import delayed

    blocks = [
        da.from_delayed(delayed(_plane)(z), shape=(h, w), dtype=dtype)[None, ...]
        for z in range(nz)
    ]
    return da.concatenate(blocks, axis=0), step, nz


#: Byte bound on the fused-plane cache, shared across every region/channel/level/z.
#:
#: 256 MiB, the same number and the same reasoning as ``_tilesource.DEFAULT_PREVIEW_BUDGET_BYTES``
#: ("~3% of an 8 GB workstation"). It is EXPLICIT, not unbounded: a preview cache that grows with
#: navigation is a memory leak wearing a performance costume, and this path already had a user
#: complaining it loads too much of the computer.
#:
#: What it buys on the real 10x region (manual0, 5731x4793 level 0): every coarse level of every
#: z of all four channels fits at once — level 2 is 3.4 MB, so 10 z x 4 ch = 137 MB — which is
#: precisely the working set napari touches at fit-to-window zoom. Level-0 planes are 54.9 MB and
#: evict each other, which is correct: at full zoom napari only ever wants the current one.
PYRAMID_CACHE_BYTES = 256 << 20

#: Runaway guard on level count, not a tuning: each level halves ``max_px``, so 12 spans a 4096x
#: zoom range. Levels stop earlier anyway once a level stops shrinking.
_MAX_PREVIEW_LEVELS = 12

#: Below this the level is smaller than a thumbnail and buys nothing.
_MIN_LEVEL_PX = 128


class MemoryBoundedLRUCache:
    """Thread-safe LRU cache bounded by BYTES, not by entry count.

    PORTED from ``ndviewer_light/data.py`` (``MemoryBoundedLRUCache``), which solves exactly this
    problem for exactly this kind of value: large numpy image planes of varying size. Copied
    rather than imported, deliberately — the owner is removing ndviewer_light from this product
    ("I don't want to have an ndviewer light fall back"), so taking a runtime dependency on it
    would work against that. It is ~75 lines and depends on nothing but the stdlib.

    Bounding by bytes is the point. The same repo also has an ``@lru_cache(maxsize=128)`` over
    ``load_plane``; over variable-size mosaic planes that is unbounded memory wearing a number
    that looks like a limit — 128 entries is 3 GB at level 0 and 100 MB at level 4.

    The LOCK is not decoration: ``_MosaicWorker`` is a QThread and dask may compute planes
    concurrently, so two threads can land in ``put`` at once and the byte accounting drifts.

    ONE DELIBERATE DIVERGENCE from the original: an item bigger than the whole budget raises
    instead of logging at DEBUG and returning. Log-and-continue there means the cache silently
    does nothing and the only symptom is that everything is slow — a misconfigured budget must be
    visible, not merely slow.
    """

    def __init__(self, max_memory_bytes: int):
        max_memory_bytes = int(max_memory_bytes)
        if max_memory_bytes <= 0:
            raise ValueError(f"cache capacity must be positive, got {max_memory_bytes} bytes.")
        self._max_memory = max_memory_bytes
        self._current_memory = 0
        self._cache: "OrderedDict[tuple, Any]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def capacity_bytes(self) -> int:
        return self._max_memory

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._current_memory

    def get(self, key: tuple):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: tuple, value) -> None:
        item_size = int(value.nbytes)
        if item_size > self._max_memory:
            raise ValueError(
                f"cannot cache a {item_size / 1e6:.1f} MB plane: that is larger than the whole "
                f"{self._max_memory / 1e6:.1f} MB cache budget. Raise the budget or lower "
                "max_px — a cache that silently stores nothing is just a slow viewer."
            )
        with self._lock:
            if key in self._cache:
                self._current_memory -= self._cache.pop(key).nbytes
            while self._current_memory + item_size > self._max_memory and self._cache:
                _oldest_key, oldest = self._cache.popitem(last=False)
                self._current_memory -= oldest.nbytes
            self._cache[key] = value
            self._current_memory += item_size

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

    def invalidate(self, key: tuple) -> bool:
        with self._lock:
            if key in self._cache:
                self._current_memory -= self._cache.pop(key).nbytes
                return True
            return False

    def __contains__(self, key: tuple) -> bool:
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


#: The process-wide preview cache. One instance, so the bound above is a bound on the WHOLE
#: preview path rather than per channel (four channels x a private cache each would be 1 GB).
_PLANE_CACHE = MemoryBoundedLRUCache(PYRAMID_CACHE_BYTES)


def _source_token(reader: Any) -> str:
    """Stable identity of the acquisition a reader reads, for cache keys.

    NOT ``id(reader)``: a cache entry can outlive the reader that produced it, and CPython
    recycles ids, so a new reader over a DIFFERENT dataset could collide with a stale key and be
    served another acquisition's pixels. The path is stable, meaningful, and makes two readers
    over the same dataset share cache entries, which is what you want.
    """
    path = getattr(reader, "_path", None)
    if path is None:
        raise ValueError(
            f"{type(reader).__name__} exposes no '_path', so its cache entries cannot be told "
            "apart from another acquisition's. Refusing to risk serving the wrong pixels."
        )
    return str(path)


def _level_max_px(max_px: int, k: int) -> int:
    return max(_MIN_LEVEL_PX, int(max_px) >> k)


def _fuse_levels(reader: Any, meta: dict, region: str, channel: str, z: int, t: int, plans: list):
    """Fuse ONE z into SEVERAL pyramid levels in a single pass over the FOV frames.

    ``plans`` is ``[(level_px, h, w, step, dtype), ...]`` for the requested level and every
    COARSER one. Returns ``{level_px: ndarray}``.

    WHY ONE PASS. TIFF decode is whole-frame: striding on read (``frame[::step, ::step]``) makes
    a coarse level cheaper to ALLOCATE and PASTE, but not cheaper to READ. napari pulls two
    levels per channel per z — the visible one, and the coarsest one for the layer thumbnail — so
    fusing each level independently decoded every FOV twice. Measured on the real 10x region:
    216 frame reads per z step instead of 108, which made the pyramid slower per z than the flat
    stack it replaced even while using a fraction of the memory.

    So a fuse yields every level coarser than the one asked for, off the frames already in hand.
    Levels 3..6 of one (z, channel) come to ~2 MB together against level 3's 1.5 MB alone — the
    thumbnail costs a stride rather than a second read pass. Levels FINER than the one requested
    are never produced; that would rebuild the 54.9 MB plane the pyramid exists to avoid.
    """
    from squidmip._placement import fov_offsets_px

    positions = meta.get("fov_positions_um") or {}
    pixel_size = meta.get("pixel_size_um")
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    offsets = fov_offsets_px(positions, region, fovs, pixel_size)

    outs = {px: np.zeros((h, w), dtype=dt) for px, h, w, _st, dt in plans}
    unreadable = []
    for fov in fovs:
        try:
            frame = reader.read(region, fov, channel, int(z), int(t))
        except Exception as exc:                 # noqa: BLE001 - collected, then reported
            unreadable.append((fov, f"{type(exc).__name__}: {exc}"))
            continue
        if frame is None:
            unreadable.append((fov, "reader returned None"))
            continue
        frame = np.asarray(frame)
        if frame.ndim != 2:
            frame = frame.reshape(frame.shape[-2:])
        row, col = offsets[fov]
        for px, h, w, st, _dt in plans:
            step = int(st)
            sub = frame[::step, ::step]
            r0, c0 = row // step, col // step
            r1, c1 = min(r0 + sub.shape[0], h), min(c0 + sub.shape[1], w)
            if r1 > r0 and c1 > c0:
                # Later FOVs overwrite earlier ones in the overlap, exactly as
                # fuse_region_mosaic does. This is a PREVIEW placement, not a stitch.
                outs[px][r0:r1, c0:c1] = sub[: r1 - r0, : c1 - c0]

    if unreadable and len(unreadable) == len(fovs):
        # One bad FOV is a visible hole. EVERY FOV bad is not a picture at all, and a black plane
        # would report a read failure as empty tissue — the exact silent failure this codebase has
        # six confirmed instances of.
        why = "; ".join(f"fov {f}: {m}" for f, m in unreadable[:3])
        raise ValueError(
            f"{region}/{channel} z={z}: no FOV in the region could be read "
            f"({len(unreadable)} of {len(fovs)} failed) — {why}"
        )
    return outs


def fuse_region_pyramid(
    reader: Any,
    meta: dict,
    region: str,
    channel: str,
    *,
    t: int = 0,
    max_px: int = _MAX_FUSED_PX,
    cache_bytes: Optional[int] = None,
):
    """A LAZY MULTISCALE PYRAMID over the fused region mosaic — what napari actually wants.

    Returns ``(levels, step, nz)``: ``levels`` is the list napari takes for ``multiscale=True``,
    highest resolution first, each entry a lazy ``(z, y, x)`` dask array (2-D when ``n_z == 1``,
    so no singleton slider appears). ``step`` is level 0's decimation, unchanged from
    ``fuse_region_stack`` so callers that report it keep reporting the same number.

    WHY THIS EXISTS. The written-OME-Zarr path has handed napari a pyramid since day one
    (``open_pyramid`` above). The RAW preview path did not — it handed over full-resolution fused
    planes. On the real 10x tissue set that is a 5731x4793 uint16 plane, 54.9 MB, per channel, and
    four channels composite additively, re-fused on every z step. Julio: "I think that you're
    loading too much of the computer to visualize the mosaic." He was right, and the fix is to
    make this path match the path that was already correct rather than to invent anything:
    napari sits on vispy, fetches only the clipped visible region of the level matching the
    current zoom, and ``docs/napari-gate.md`` already measured that it does so (16.7 ms/warm tile,
    and issue #1942 does NOT reproduce). Hand-rolled viewport culling is the wheel this project
    migrated to napari to stop reinventing.

    EACH LEVEL IS FUSED DIRECTLY FROM THE FOV TILES, at its own decimation, rather than by
    coarsening the level-0 dask graph. Both were measured on manual0: producing a 956x799 level
    costs 19 ms direct versus 60 ms via ``da.coarsen`` over level 0 — 3.2x — because the coarsen
    route must allocate and paste the whole 54.9 MB level-0 plane before throwing 63/64 of it
    away. ``fuse_region_mosaic`` already strides each frame on read (``frame[::step, ::step]``),
    so a coarse level never materialises a full-resolution intermediate at all. That is the
    difference between a pyramid that helps and a pyramid that only moves the cost around.

    Levels downsample y and x ONLY. z is never coarsened: napari puts its dimension slider on it,
    and ``layer.scale`` stays ``(dz_um, py, px)`` so the 3-D toggle renders anisotropic data
    anisotropically (IMA-255, commit 19cd491). A pyramid that quietly halved z would look right in
    a shape assertion and be wrong on screen.

    Returns ``None`` — never a guess — when the acquisition carries no stage positions or no pixel
    size, the same signal ``fuse_region_stack`` and ``mosaic_bbox_um`` use.
    """
    import dask.array as da
    from dask import delayed

    base = _planned_plane(meta, region, max_px)
    if base is None:
        return None
    _h0, _w0, step0, dtype = base

    # The plane budget guards level 0 exactly as before. A pyramid is not a licence to skip it:
    # level 0 is still what napari materialises when the user zooms all the way in.
    per_plane = _h0 * _w0 * dtype.itemsize
    if per_plane > _PLANE_BUDGET_BYTES:
        raise MemoryError(
            f"{region}/{channel}: one fused plane is {per_plane / 1e9:.1f} GB "
            f"({_h0}x{_w0} {dtype}), over the {_PLANE_BUDGET_BYTES / 1e9:.1f} GB plane budget. "
            "Lower max_px rather than letting this page the machine."
        )

    cache = _PLANE_CACHE if cache_bytes is None else MemoryBoundedLRUCache(cache_bytes)
    token = _source_token(reader)
    nz = int(meta.get("n_z") or 1)

    # Plan every rung up front. Pure geometry, reads nothing — and it is what lets a fuse know
    # which COARSER levels it can hand back off the same decode.
    plans: list = []
    for k in range(_MAX_PREVIEW_LEVELS):
        level_px = _level_max_px(max_px, k)
        plan = _planned_plane(meta, region, level_px)
        if plan is None:
            # Level 0's geometry resolved above, so a later level failing means the geometry is
            # not self-consistent. Say so rather than hand napari a partial pyramid.
            raise ValueError(
                f"{region}/{channel}: level {k} (max_px={level_px}) has no derivable geometry "
                f"although level 0 does. Refusing to hand napari a partial pyramid."
            )
        h, w, step, dt = plan
        plans.append((level_px, h, w, step, dt))
        if level_px <= _MIN_LEVEL_PX:
            break

    def _plane(i: int, z: int):
        """Level ``i`` at ``z``, from the cache or from one decode pass that also fills the
        coarser levels. ``i`` indexes ``plans``, which is finest-first."""
        level_px, h, w, step, dt = plans[i]
        key = (token, region, channel, int(t), float(step), int(z))
        hit = cache.get(key)
        if hit is not None:
            return hit

        # This level and every coarser one: the frames have to be decoded either way, and the
        # extra pastes are a stride each. Nothing FINER is built.
        wanted = plans[i:]
        outs = _fuse_levels(reader, meta, region, channel, int(z), int(t), wanted)
        for px, ph, pw, pstep, _pdt in wanted:
            arr = outs[px]
            if arr.shape != (ph, pw):
                raise ValueError(
                    f"{region}/{channel}: z={z} fused to {arr.shape}, but this pyramid level is "
                    f"{(ph, pw)}. A ragged z would misalign the stack and misregister the level."
                )
            cache.put((token, region, channel, int(t), float(pstep), int(z)), arr)
        return outs[level_px]

    levels = []
    for i, (_px, h, w, _step, dt) in enumerate(plans):
        if nz <= 1:
            lv = da.from_delayed(delayed(_plane)(i, 0), shape=(h, w), dtype=dt)
        else:
            lv = da.concatenate(
                [da.from_delayed(delayed(_plane)(i, z), shape=(h, w), dtype=dt)[None, ...]
                 for z in range(nz)],
                axis=0,
            )
        levels.append(lv)

    # ONE guard, shared with open_pyramid: a level that did not shrink is dropped, not handed to
    # napari as a duplicate. Halving max_px does not always halve the mosaic (the decimation step
    # is a ceiling), and a small region runs out of room after one rung.
    return strictly_decreasing_levels(levels), step0, nz


def mosaic_bbox_um(meta: dict, region: str) -> Optional[tuple[float, float, float, float]]:
    """``(x0, y0, x1, y1)`` stage micrometres covered by a region's mosaic.

    This is what places the layer in napari's world, and it is why the tiling layer's
    ``bbox_um`` maps across with no unit conversion: both already speak stage micrometres.
    """
    from squidmip._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    pixel_size = meta.get("pixel_size_um")
    if not positions or pixel_size in (None, 0):
        return None
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        return None
    try:
        offsets = fov_offsets_px(positions, region, fovs, pixel_size)
        h, w = mosaic_extent_px(offsets, tuple(int(v) for v in meta["frame_shape"]))
    except (KeyError, ValueError):
        return None

    xs = [float(positions[(region, f)][0]) for f in fovs]
    ys = [float(positions[(region, f)][1]) for f in fovs]
    x0, y0 = min(xs), min(ys)
    return (x0, y0, x0 + w * float(pixel_size), y0 + h * float(pixel_size))
