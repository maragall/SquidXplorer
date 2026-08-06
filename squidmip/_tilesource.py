"""IMA-217: the tile ladder + the two ``TileSource`` implementations IMA-216 asks for.

``_tiling.py`` (IMA-216) owns the *algorithm* — LOD pick, frustum cull, byte-budget LRU — and
declares one hole: a :class:`~squidmip._tiling.TileSource` that turns a
:class:`~squidmip._tiling.TileDescriptor` into pixels, plus the
:class:`~squidmip._tiling.Geometry` describing what tiles exist. This module fills both, twice::

    plate.ome.zarr on disk (IMA-184/185)  ─►  ZarrPyramidSource   (persistent, pixel-exact)
    a live acquisition stream (on_well)   ─►  InMemoryMultiscale  (preview, byte-budgeted)

Both hand out the SAME ``Geometry``, so the viewer can start on the RAM preview mid-run and
switch to the zarr source when the write finishes without re-deriving a single coordinate.

Two more have been added since, for the case the viewer actually spends its time in — a raw
acquisition folder with no written plate::

    a raw acquisition, via the reader    ─►  ReaderTileSource       (projected, cached by bytes)
    raw + the persisted preview cells    ─►  CompositePlateSource   (plate rungs from _platecache,
                                                                     FOV rungs from the reader)

``CompositePlateSource`` is what closes the coarse-rung gap ``NEXT_STEPS.md`` measured at 25 s;
see its docstring.

World space is stage MICROMETRES throughout; every key ends ``_um``. Positions come from
``metadata["fov_positions_um"]`` (the reader already converted coordinates.csv's mm), and are
FOV **centres** — :func:`fov_bboxes_um` expands each to the frame's extent. Feeding millimetres
in is caught, not tolerated: :func:`plate_ladder` refuses a grid whose FOV pitch is absurdly
small relative to the frame.

The ladder (why fit-to-plate is O(viewport) and not O(plate))
-------------------------------------------------------------
Two kinds of rung, stacked::

    scale (µm/px)   rung          tiles                        read path
    ─────────────   ────────────  ───────────────────────────  ─────────────────────────────
    p, 2p, 4p …     per-FOV       one per FOV, keyed (region,  the written pyramid level,
                                  fov); the field IS the tile  pixel-exact, one array read
    ─── crossover: fov_extent_um == tile_px * scale ───────────────────────────────────────
    … 8p, 16p …     plate grid    a fixed tile_px grid over     composited from the coarse
                                  the world, EMPTY CELLS       per-FOV levels of whatever
                                  DROPPED, keyed (gy, gx)      FOVs fall in the cell

The crossover is the whole trick. A per-FOV rung's tile count is fixed at N_fov, so a view that
sees the whole plate at a per-FOV rung fetches N_fov tiles — the O(plate) failure IMA-216 warns
about in ``Geometry.worst_case_tiles``. Above the crossover a plate tile covers more world than
an FOV does, so tiles-per-view is bounded by (screen area / tile area) and each coarser rung
holds ~1/4 the tiles of the one below. Measured: fit-to-plate returns 25 tiles on the 144-FOV
dataset and 16 on a 14,400-FOV plate — smaller on the bigger plate, because tile count follows
the SCREEN, not the sample.

Prior art
---------
* **OME-NGFF multiscales v0.4/v0.5** — the on-disk layout is unchanged canonical NGFF: a
  ``datasets`` list, per-dataset ``coordinateTransformations`` of ``scale`` then ``translation``,
  one entry per axis, ``tczyx``. IMA-217's only addition to the writer is the ``translation``
  (the field's top-left corner in stage µm), which is the spec's own mechanism for placing images
  in a shared world frame — so ``ZarrPyramidSource`` derives the whole plate layout from the store
  and a stock reader (ome-zarr-py, napari-ome-zarr) still opens it. No private layout, no sidecar.
* **ome-zarr-py's ``Scaler``/ngio** — factor-2 halving per level, stop at a small-enough coarsest
  level; ``_output._pyramid`` already did exactly this, so the per-FOV rungs reuse the written
  levels rather than inventing a second downsample chain.
* **Tanner/Migdal/Jones, "The Clipmap: A Virtual Mipmap" (SIGGRAPH 1998)** — the clipmap keeps a
  fixed-size window per level and *blends across level boundaries* precisely because a hard
  switch pops. The 2-D tile analogue of that blend is a hysteresis deadband on the level pick, so
  a zoom parked on a boundary does not thrash the fetch queue; ``_tiling.pick_level`` already
  implements it (``_DEFAULT_HYSTERESIS = 0.25``). This module deliberately does NOT add a second
  level-selection policy — one is enough, and it lives with the algorithm.

Memory
------
``ZarrPyramidSource`` holds no pixels: every read is one tile, and the caller's ``TileCache``
owns the byte budget. ``InMemoryMultiscale`` holds pixels by definition, so it takes an
**explicit** ``budget_bytes``, admits plate rungs coarsest-first while the FULLY-FILLED capacity
of the admitted set stays inside it, and refuses to start if even the coarsest rung does not fit.
``add_field`` never materialises more than one resampled channel at a time, matching the
streaming discipline of the projection engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Hashable, Mapping, Optional, Sequence

import numpy as np

from squidmip._budget import cache_budget

from squidmip._montage import _area_downsample
from squidmip._mosaic_source import MemoryBoundedLRUCache
from squidmip._output import _PYRAMID_MAX_LEVELS, _PYRAMID_MIN_YX, pyramid_shapes
from squidmip._tiling import Geometry, Level, TileDescriptor

# Plate-rung tile size in pixels. 512 matches the store's 1024 px chunking closely enough that a
# coarse composite touches few chunks, and keeps one uint16 tile at 512 KB — small enough that a
# 25-tile fit-to-plate view is ~13 MB, i.e. one TileCache's worth.
DEFAULT_TILE_PX = 512

# Default byte budget for the in-RAM preview multiscale. 256 MiB is ~3% of a 8 GB workstation and
# holds the top five rungs of a 1536wp plate at 2 channels; it is a DEFAULT, never a silent cap —
# the constructor reports ``capacity_bytes`` and raises when the coarsest rung alone overflows.
# MEASURED, not hardcoded -- see squidmip._budget. The old comment said "256 MiB is ~3% of an
# 8 GB workstation", which is exactly the problem: it encodes an assumption about a machine it
# has never seen. Derived from AVAILABLE memory, floored so the cache cannot thrash and capped so
# it stays bounded. Still a DEFAULT, never a silent cap: the constructor reports capacity_bytes
# and raises when the coarsest rung alone overflows.
DEFAULT_PREVIEW_BUDGET_BYTES = cache_budget()

# How many plate rungs to stack above the per-FOV ones. Each is 2x coarser and ~1/4 the tiles, so
# 12 spans a 4096x zoom range — far more than any plate needs; it is a runaway guard, not a tuning.
_MAX_PLATE_LEVELS = 12

# A recorded FOV pitch below frame_extent_um / _MM_PITCH_RATIO is not a stage pattern, it is
# millimetres that leaked into a ``_um`` key (a 705 µm pitch becomes 0.7 µm — two pixels).
_MM_PITCH_RATIO = 100.0


# --- world geometry from acquisition metadata -------------------------------------------------

def fov_bboxes_um(positions_um: Mapping[tuple, tuple], frame_shape, pixel_size_um) -> dict:
    """``{(region, fov): (x0, y0, x1, y1)}`` in stage µm, from FOV **centre** positions.

    ``metadata["fov_positions_um"]`` records where the stage was — the middle of the frame — so a
    box is the frame's physical extent centred there. Using the position as a corner instead shifts
    the whole mosaic by half an FOV (388 µm on a 2084 px 20x field): a plausible-looking, uniformly
    wrong picture, which is the failure mode ``_placement.py``'s docstring is written against.
    """
    p = float(pixel_size_um)
    if not p > 0:
        raise ValueError(f"pixel_size_um must be > 0 to size an FOV in µm, got {pixel_size_um!r}")
    h, w = int(frame_shape[0]), int(frame_shape[1])
    half_w, half_h = w * p / 2.0, h * p / 2.0
    out = {}
    for key, (x, y) in positions_um.items():
        x, y = float(x), float(y)
        out[key] = (x - half_w, y - half_h, x + half_w, y + half_h)
    return out


def _min_axis_pitch(values: np.ndarray) -> float:
    """Smallest positive gap between distinct coordinates along one axis (inf if there is none).

    O(n log n) — a pairwise nearest-neighbour scan would be O(n²) and a 14,400-FOV plate would
    spend 200 M comparisons proving a units invariant.
    """
    u = np.unique(np.round(np.asarray(values, dtype=np.float64), 6))
    if u.size < 2:
        return float("inf")
    return float(np.min(np.diff(u)))


def _check_micrometres(bboxes: dict, frame_extent_um: float) -> None:
    """Fail loud when the positions look like millimetres wearing a ``_um`` key."""
    if len(bboxes) < 2:
        return
    centres = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in bboxes.values()])
    pitch = min(_min_axis_pitch(centres[:, 0]), _min_axis_pitch(centres[:, 1]))
    if pitch < frame_extent_um / _MM_PITCH_RATIO:
        raise ValueError(
            f"FOV pitch is {pitch:.4g} µm for a {frame_extent_um:.4g} µm frame — a stage does not "
            "step 1/100th of a field. These positions are almost certainly MILLIMETRES stored under "
            "a `_um` key; the reader converts coordinates.csv at the producer "
            "(squidmip.reader.load_fov_positions_um). Refusing to build a 1000x-too-small plate."
        )


@dataclass(frozen=True)
class PlateLadder:
    """The world layout + the :class:`~squidmip._tiling.Geometry` built from it.

    ``geometry.levels[i]`` is a per-FOV rung for ``i < n_fov_levels`` (keys ``(region, fov)``) and a
    plate-grid rung above (keys ``(grid_y, grid_x)``). ``fov_level_shapes`` lists EVERY written
    pyramid level, including the ones too coarse to be a rung — those are still read, as the source
    pixels a plate tile is composited from.
    """

    geometry: Geometry
    fov_bboxes: dict
    fov_level_shapes: list
    n_fov_levels: int
    tile_px: int
    world_bbox_um: tuple
    pixel_size_um: float
    frame_shape: tuple
    _plate_grids: dict = _dc_field(default_factory=dict, repr=False)

    # ---- rung introspection ------------------------------------------------------------
    def is_fov_level(self, level: int) -> bool:
        return 0 <= int(level) < self.n_fov_levels

    def plate_grid_shape(self, level: int) -> tuple[int, int]:
        """``(n_rows, n_cols)`` of the DENSE grid a plate rung is carved from (empty cells dropped)."""
        return self._plate_grids[int(level)][1]

    def cell_bbox_um(self, level: int, key: Hashable) -> tuple:
        """World bbox of one tile — an FOV's frame extent, or a plate grid cell."""
        level = int(level)
        if self.is_fov_level(level):
            return self.fov_bboxes[key]
        tile_um, _ = self._plate_grids[level]
        gy, gx = key
        x0 = self.world_bbox_um[0] + gx * tile_um
        y0 = self.world_bbox_um[1] + gy * tile_um
        return (x0, y0, x0 + tile_um, y0 + tile_um)

    def fov_source_level(self, scale_um_per_px: float) -> int:
        """Which WRITTEN per-FOV pyramid level to composite a tile of *scale* from.

        The coarsest level still at least as fine as the target — the same "just finer than the
        screen" rule ``pick_level`` applies to the ladder, applied one layer down to the pixels.
        Reading level 0 for a 50 µm/px plate tile would move 130x more bytes for the same result.
        """
        s = float(scale_um_per_px)
        best = 0
        for i in range(len(self.fov_level_shapes)):
            if self.fov_level_scale(i) <= s:
                best = i
        return best

    def fov_level_scale(self, level: int) -> float:
        """µm/px of a written per-FOV pyramid level (the coarser of its Y and X factors)."""
        y0, x0 = self.fov_level_shapes[0]
        y, x = self.fov_level_shapes[int(level)]
        return self.pixel_size_um * max(y0 / y, x0 / x)

    def fovs_overlapping(self, bbox_um: tuple) -> list:
        """FOV keys whose frame overlaps *bbox_um* (strict, matching ``select_tiles``' cull)."""
        x0, y0, x1, y1 = bbox_um
        return [k for k, b in self.fov_bboxes.items()
                if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0]


def plate_ladder(metadata: Mapping, *, tile_px: int = DEFAULT_TILE_PX,
                 min_yx: int = _PYRAMID_MIN_YX, max_levels: int = _PYRAMID_MAX_LEVELS,
                 max_plate_levels: int = _MAX_PLATE_LEVELS) -> PlateLadder:
    """Build the whole tile ladder from acquisition metadata alone — pure, no I/O, no pixels.

    Needs ``fov_positions_um``, ``pixel_size_um`` and ``frame_shape``; the pyramid rungs come from
    :func:`squidmip._output.pyramid_shapes`, i.e. from exactly the levels the writer writes, so the
    ladder cannot drift from the store.

    Raises ``ValueError`` on missing positions/pixel size, or on positions that look like mm.
    """
    positions = metadata.get("fov_positions_um") or {}
    if not positions:
        raise ValueError(
            "no fov_positions_um in the metadata: without stage coordinates every FOV would sit at "
            "the same spot and the plate view would be a single stacked pile. (coordinates.csv "
            "missing or unusable — see squidmip.reader._fov_positions_um_or_empty.)")
    pixel_size_um = metadata.get("pixel_size_um")
    if not pixel_size_um:
        raise ValueError("pixel_size_um is required to size an FOV in µm; acquisition.yaml has none.")
    frame_shape = metadata.get("frame_shape")
    if frame_shape is None:
        raise ValueError("frame_shape is required to size an FOV in µm.")
    tile_px = int(tile_px)
    if tile_px < 1:
        raise ValueError(f"tile_px must be >= 1, got {tile_px}")

    p = float(pixel_size_um)
    boxes = fov_bboxes_um(positions, frame_shape, p)
    fov_extent_um = max(int(frame_shape[0]), int(frame_shape[1])) * p
    _check_micrometres(boxes, fov_extent_um)

    keys = sorted(boxes)                      # deterministic tile order across runs
    arr = np.array([boxes[k] for k in keys], dtype=np.float64)
    world = (float(arr[:, 0].min()), float(arr[:, 1].min()),
             float(arr[:, 2].max()), float(arr[:, 3].max()))

    shapes = pyramid_shapes(frame_shape, min_yx=min_yx, max_levels=max_levels)
    y0, x0 = shapes[0]
    fov_scales = [p * max(y0 / y, x0 / x) for (y, x) in shapes]

    # --- per-FOV rungs, up to the crossover ---------------------------------------------
    # Keep a per-FOV rung only while its tile still covers at least as much world as a plate tile
    # would (fov_extent_um >= tile_px * scale). Past that, a per-FOV rung is strictly worse: same
    # tile COUNT (one per FOV, forever) for less world per tile — that is the O(plate) fit-to-plate.
    # Level 0 is always kept: it is the only pixel-exact, no-resampling read path.
    n_fov_levels = 1
    for i in range(1, len(fov_scales)):
        if fov_scales[i] * tile_px <= fov_extent_um:
            n_fov_levels = i + 1
        else:
            break

    levels: list[Level] = [Level(fov_scales[i], arr, keys) for i in range(n_fov_levels)]

    # --- plate-grid rungs above it -------------------------------------------------------
    grids: dict[int, tuple[float, tuple[int, int]]] = {}
    width, height = world[2] - world[0], world[3] - world[1]
    prev_count = len(keys)
    scale = p * (2.0 ** n_fov_levels)
    for _ in range(int(max_plate_levels)):
        if scale <= levels[-1].scale_um_per_px:       # ladder must strictly increase
            scale *= 2.0
            continue
        tile_um = tile_px * scale
        n_cols = max(1, int(np.ceil(width / tile_um)))
        n_rows = max(1, int(np.ceil(height / tile_um)))
        cells = _occupied_cells(arr, world, tile_um, n_rows, n_cols)
        # A coarser rung holding >= the tiles of the one below buys nothing (and > would make
        # Geometry raise). Skip it and try the next doubling.
        if len(cells) < prev_count:
            idx = len(levels)
            grids[idx] = (tile_um, (n_rows, n_cols))
            levels.append(Level(scale, _cell_bboxes(cells, world, tile_um), cells))
            prev_count = len(cells)
            if len(cells) == 1:
                break
        scale *= 2.0

    return PlateLadder(
        geometry=Geometry(levels),
        fov_bboxes=boxes,
        fov_level_shapes=[tuple(s) for s in shapes],
        n_fov_levels=n_fov_levels,
        tile_px=tile_px,
        world_bbox_um=world,
        pixel_size_um=p,
        frame_shape=(int(frame_shape[0]), int(frame_shape[1])),
        _plate_grids=grids,
    )


def _occupied_cells(fov_bboxes: np.ndarray, world: tuple, tile_um: float,
                    n_rows: int, n_cols: int) -> list:
    """Grid cells that at least one FOV touches, in row-major order — EMPTY CELLS DROPPED.

    A sparse plate (four wells 30 mm apart on a 1536wp) is mostly empty space; a dense grid would
    charge the viewer for tiles that can only ever be black, and would inflate ``worst_case_tiles``
    into a number that no longer means anything.
    """
    gx0 = np.clip(np.floor((fov_bboxes[:, 0] - world[0]) / tile_um).astype(np.int64), 0, n_cols - 1)
    gx1 = np.clip(np.ceil((fov_bboxes[:, 2] - world[0]) / tile_um).astype(np.int64) - 1, 0, n_cols - 1)
    gy0 = np.clip(np.floor((fov_bboxes[:, 1] - world[1]) / tile_um).astype(np.int64), 0, n_rows - 1)
    gy1 = np.clip(np.ceil((fov_bboxes[:, 3] - world[1]) / tile_um).astype(np.int64) - 1, 0, n_rows - 1)
    seen = set()
    for a, b, c, d in zip(gy0, gy1, gx0, gx1):
        for gy in range(int(a), int(b) + 1):
            for gx in range(int(c), int(d) + 1):
                seen.add((gy, gx))
    return sorted(seen)


def _cell_bboxes(cells: Sequence[tuple], world: tuple, tile_um: float) -> np.ndarray:
    g = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
    x0 = world[0] + g[:, 1] * tile_um
    y0 = world[1] + g[:, 0] * tile_um
    return np.stack([x0, y0, x0 + tile_um, y0 + tile_um], axis=1)


# --- resampling: one field's pixels into one tile ----------------------------------------------

def _resample(plane: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """*plane* -> ``(out_h, out_w)`` float32. Area-average when shrinking, nearest when growing.

    Area-averaging (``_montage._area_downsample``, already load-bearing for the plate montage) so a
    coarse tile reflects the whole field rather than one sampled pixel; nearest on the rare grow
    path (a plate tile finer than the coarsest available pyramid level) because inventing detail
    with interpolation would be a lie about resolution.
    """
    h, w = plane.shape
    if out_h <= h and out_w <= w:
        return _area_downsample(plane, out_h, out_w)
    yi = np.minimum((np.arange(out_h) * h) // max(out_h, 1), h - 1)
    xi = np.minimum((np.arange(out_w) * w) // max(out_w, 1), w - 1)
    return plane[yi][:, xi].astype(np.float32, copy=False)


def _paste_field(dst: np.ndarray, dst_bbox_um: tuple, scale_um_per_px: float,
                 plane: np.ndarray, fov_bbox_um: tuple) -> bool:
    """Resample the part of *plane* inside *dst_bbox_um* into *dst*. True if anything landed.

    Both rectangles are in world µm, so this is the ONLY place FOV pixels meet tile pixels and the
    only place a placement bug could hide. It works in world coordinates end to end: intersect, map
    the intersection into destination pixels AND into source pixels, crop, resample, assign.
    """
    cx0, cy0, cx1, cy1 = dst_bbox_um
    fx0, fy0, fx1, fy1 = fov_bbox_um
    ix0, iy0 = max(cx0, fx0), max(cy0, fy0)
    ix1, iy1 = min(cx1, fx1), min(cy1, fy1)
    if not (ix1 > ix0 and iy1 > iy0):
        return False

    th, tw = dst.shape
    dx0 = int(np.clip(round((ix0 - cx0) / scale_um_per_px), 0, tw - 1))
    dx1 = int(np.clip(round((ix1 - cx0) / scale_um_per_px), dx0 + 1, tw))
    dy0 = int(np.clip(round((iy0 - cy0) / scale_um_per_px), 0, th - 1))
    dy1 = int(np.clip(round((iy1 - cy0) / scale_um_per_px), dy0 + 1, th))

    sh, sw = plane.shape
    px_um_x, px_um_y = (fx1 - fx0) / sw, (fy1 - fy0) / sh
    sx0 = int(np.clip(round((ix0 - fx0) / px_um_x), 0, sw - 1))
    sx1 = int(np.clip(round((ix1 - fx0) / px_um_x), sx0 + 1, sw))
    sy0 = int(np.clip(round((iy0 - fy0) / px_um_y), 0, sh - 1))
    sy1 = int(np.clip(round((iy1 - fy0) / px_um_y), sy0 + 1, sh))

    resampled = _resample(plane[sy0:sy1, sx0:sx1], dy1 - dy0, dx1 - dx0)
    if np.issubdtype(dst.dtype, np.integer):
        info = np.iinfo(dst.dtype)
        np.rint(resampled, out=resampled)
        np.clip(resampled, info.min, info.max, out=resampled)
    dst[dy0:dy1, dx0:dx1] = resampled.astype(dst.dtype, copy=False)
    return True


# --- source 1: the written OME-Zarr plate ------------------------------------------------------

def _read_ome(group_dir: Path) -> dict:
    return json.loads((group_dir / "zarr.json").read_text()).get("attributes", {}).get("ome", {})


class ZarrPyramidSource:
    """``TileSource`` over a written ``plate.ome.zarr`` — the persistent, pixel-exact path.

    Self-describing: the plate's own NGFF metadata (per-dataset ``scale`` + ``translation``, plus
    each level-0 array's shape) is enough to rebuild the world layout, so this never re-reads
    coordinates.csv and cannot disagree with the store about where a field is.

    Reads are per tile and nothing is retained — the caller's ``TileCache`` owns the byte budget:

    * **per-FOV rung** — one array read of the matching written pyramid level. Pixel-exact.
    * **plate rung** — composite: every FOV in the cell, read at the coarsest pyramid level still
      finer than the cell, area-resampled into the cell's grid. Bounded by the tile, not the plate.

    The honest cost note: a *fit-to-plate* view is O(tiles-on-screen) tile reads, but between them
    those tiles touch every FOV once — that is inherent to deriving a plate overview from per-FOV
    data, and it is read from the ~130 px coarse levels, not from full res. :class:`InMemoryMultiscale`
    is the O(1)-per-view answer for a live run, because it is built incrementally as fields arrive.
    """

    def __init__(self, plate_path, *, tile_px: int = DEFAULT_TILE_PX, t: int = 0,
                 min_yx: int = _PYRAMID_MIN_YX, max_levels: int = _PYRAMID_MAX_LEVELS) -> None:
        self.plate_dir = _resolve_plate_dir(plate_path)
        self.t = int(t)
        self._stores: dict = {}
        layout = _read_plate_layout(self.plate_dir)
        self.channels: list[str] = layout["channels"]
        self._field_dirs: dict = layout["field_dirs"]
        meta = {
            "fov_positions_um": layout["centres_um"],
            "pixel_size_um": layout["pixel_size_um"],
            "frame_shape": layout["frame_shape"],
        }
        self.ladder = plate_ladder(meta, tile_px=tile_px, min_yx=min_yx, max_levels=max_levels)

    # ---- TileSource --------------------------------------------------------------------
    def read_tile(self, desc: TileDescriptor) -> np.ndarray:
        """One tile as a 2-D native-dtype array. Satisfies ``_tiling.TileSource``."""
        c = self._channel_index(desc.channel)
        if self.ladder.is_fov_level(desc.level):
            return self._read_fov_plane(desc.key, desc.level, c)
        return self._composite_cell(desc.level, desc.key, c)

    # ---- internals ---------------------------------------------------------------------
    def _channel_index(self, channel: str) -> int:
        s = str(channel)
        if s in self.channels:
            return self.channels.index(s)
        if s.isdigit() and int(s) < len(self.channels):
            return int(s)          # positional channel ids, as _tiling's default ("0", "1", ...)
        raise KeyError(f"unknown channel {channel!r}; plate has {self.channels}")

    def _store(self, fov_key, level: int):
        # Through the process-wide pool (_tsctx): this used to be a per-instance dict with no
        # bound and no lock, mutated from the tile fetcher's thread while the GUI read the same
        # stores. The pool bounds live handles at 32 and binds every reader to one cache_pool, so
        # a deep-zoom scrub over a big plate cannot grow the footprint without limit.
        from squidmip._tsctx import HANDLES
        return HANDLES.get(self._field_dirs[fov_key] / str(level))

    def _read_fov_plane(self, fov_key, level: int, c: int) -> np.ndarray:
        store = self._store(fov_key, level)
        t = min(self.t, store.shape[0] - 1)
        return np.asarray(store[t, c, 0].read().result())

    def _composite_cell(self, level: int, key, c: int) -> np.ndarray:
        bbox = self.ladder.cell_bbox_um(level, key)
        scale = self.ladder.geometry.levels[level].scale_um_per_px
        src_level = self.ladder.fov_source_level(scale)
        tile = np.zeros((self.ladder.tile_px, self.ladder.tile_px), dtype=self._dtype())
        for fov_key in self.ladder.fovs_overlapping(bbox):
            plane = self._read_fov_plane(fov_key, src_level, c)
            _paste_field(tile, bbox, scale, plane, self.ladder.fov_bboxes[fov_key])
        return tile

    def _dtype(self):
        first = next(iter(self._field_dirs))
        return np.dtype(self._store(first, 0).dtype.numpy_dtype)


def _resolve_plate_dir(plate_path) -> Path:
    p = Path(plate_path)
    if (p / "zarr.json").exists() and "plate" in _read_ome(p):
        return p
    if (p / "plate.ome.zarr").is_dir():
        return p / "plate.ome.zarr"
    raise ValueError(f"{plate_path!s} is not an OME-NGFF HCS plate (no plate group metadata).")


def _read_plate_layout(plate_dir: Path) -> dict:
    """Walk the plate's NGFF metadata into ``{centres_um, pixel_size_um, frame_shape, ...}``.

    The ``translation`` on dataset 0 is the field's top-left corner in stage µm; the ladder wants
    centres, so half a frame is added back here. A plate written before IMA-217 (no translation)
    is refused with a message that names the fix rather than silently stacking every field at the
    origin — which is exactly what a missing translation would draw.
    """
    plate = _read_ome(plate_dir).get("plate")
    if not plate:
        raise ValueError(f"{plate_dir!s} has no OME plate metadata (attributes.ome.plate).")
    centres: dict = {}
    field_dirs: dict = {}
    channels: list[str] = []
    pixel_size_um: Optional[float] = None
    frame_shape: Optional[tuple] = None

    for well in plate["wells"]:
        row_name, col_name = well["path"].split("/")
        region = row_name + col_name
        well_dir = plate_dir / row_name / col_name
        for image in _read_ome(well_dir).get("well", {}).get("images", []):
            fov = int(image["path"])
            field_dir = well_dir / str(image["path"])
            ome = _read_ome(field_dir)
            ms = ome["multiscales"][0]
            ds0 = ms["datasets"][0]
            xforms = {x["type"]: x for x in ds0["coordinateTransformations"]}
            if "translation" not in xforms:
                raise ValueError(
                    f"field {field_dir!s} has no NGFF `translation` transform, so the plate does "
                    "not say where this field sits in stage µm. Rewrite the plate with the "
                    "IMA-217 writer (squidmip._output.write_plate), which emits it.")
            shape = json.loads((field_dir / "0" / "zarr.json").read_text())["shape"]
            fy, fx = int(shape[-2]), int(shape[-1])
            sy, sx = float(xforms["scale"]["scale"][-2]), float(xforms["scale"]["scale"][-1])
            ty, tx = (float(xforms["translation"]["translation"][-2]),
                      float(xforms["translation"]["translation"][-1]))
            centres[(region, fov)] = (tx + fx * sx / 2.0, ty + fy * sy / 2.0)
            field_dirs[(region, fov)] = field_dir
            if pixel_size_um is None:
                pixel_size_um, frame_shape = sx, (fy, fx)
                channels = [str(ch.get("label") or i)
                            for i, ch in enumerate(ome.get("omero", {}).get("channels", []))]
    if not centres:
        raise ValueError(f"{plate_dir!s} lists no fields.")
    return {"centres_um": centres, "pixel_size_um": pixel_size_um, "frame_shape": frame_shape,
            "field_dirs": field_dirs, "channels": channels or ["0"]}


# --- source 2: the in-RAM preview multiscale ---------------------------------------------------

class InMemoryMultiscale:
    """``TileSource`` holding the coarse plate rungs in RAM, under an EXPLICIT byte budget.

    The live-acquisition path: ``write_from_stream``'s ``on_well`` hands each projected field here
    as it lands, and :meth:`add_field` folds it into every resident rung. A fit-to-plate view is
    then O(tiles-on-screen) dict lookups — no disk, no recomposite, no dependence on plate size —
    which is what makes a 1536wp run scrub smoothly while it is still being written.

    Budget, not vibes. Only PLATE rungs are resident (a per-FOV rung in RAM would be the whole
    acquisition). Rungs are admitted coarsest-first while the fully-filled capacity of the admitted
    set stays inside ``budget_bytes``; ``capacity_bytes`` is that worst case and is guaranteed, not
    hoped for, because tiles are fixed-size. If the coarsest rung alone does not fit, the
    constructor raises rather than quietly rendering nothing.

    Thread safety: ``add_field`` runs on ``write_from_stream``'s writer threads, which are already
    required to be thread-safe by ``on_well``'s contract; per-tile arrays are allocated under a
    lock and pixel writes go to disjoint sub-rectangles of distinct tiles per field.
    """

    def __init__(self, ladder: PlateLadder, channels: Sequence[str], dtype=np.uint16, *,
                 budget_bytes: int = DEFAULT_PREVIEW_BUDGET_BYTES, t: int = 0) -> None:
        import threading

        self.ladder = ladder
        self.channels = [str(c) for c in channels]
        self.dtype = np.dtype(dtype)
        self.budget_bytes = int(budget_bytes)
        self.t = int(t)
        if self.budget_bytes < 0:
            raise ValueError(f"budget_bytes must be >= 0, got {budget_bytes}")
        if not self.channels:
            raise ValueError("InMemoryMultiscale needs at least one channel")

        per_tile = ladder.tile_px * ladder.tile_px * len(self.channels) * self.dtype.itemsize
        plate_levels = list(range(ladder.n_fov_levels, len(ladder.geometry)))
        if not plate_levels:
            raise ValueError(
                "this ladder has no plate rungs (a single-FOV acquisition, or a tile_px larger than "
                "the plate); there is nothing for an in-RAM preview to hold.")

        self.levels: list[int] = []
        self.capacity_bytes = 0
        for lvl in reversed(plate_levels):                 # coarsest first
            cost = len(ladder.geometry.levels[lvl]) * per_tile
            if self.capacity_bytes + cost > self.budget_bytes:
                break
            self.levels.append(lvl)
            self.capacity_bytes += cost
        if not self.levels:
            need = len(ladder.geometry.levels[plate_levels[-1]]) * per_tile
            raise ValueError(
                f"budget_bytes={self.budget_bytes} cannot hold even the coarsest plate rung "
                f"({need} bytes: {len(ladder.geometry.levels[plate_levels[-1]])} tiles x "
                f"{ladder.tile_px}² px x {len(self.channels)} channels x {self.dtype.itemsize} B). "
                "Raise the budget or lower tile_px.")

        self._tiles: dict = {}                             # (level, key) -> (C, tile_px, tile_px)
        self._lock = threading.Lock()

    # ---- inspection --------------------------------------------------------------------
    @property
    def nbytes(self) -> int:
        """Bytes actually allocated so far — always <= ``capacity_bytes``."""
        return sum(a.nbytes for a in self._tiles.values())

    def __len__(self) -> int:
        return len(self._tiles)

    # ---- TileSource --------------------------------------------------------------------
    def read_tile(self, desc: TileDescriptor) -> np.ndarray:
        """One channel of one resident tile. An untouched tile reads as ZEROS, never an error.

        A plate is half-acquired for most of a run; raising on the cells that have not arrived yet
        would make the viewer's fetch path throw once per empty tile per frame. Black is the honest
        rendering of "nothing here yet", and ``add_field`` names the tiles to invalidate when it
        stops being true.
        """
        if desc.level not in self.levels:
            raise KeyError(
                f"level {desc.level} is not resident in this preview (resident: {self.levels}); "
                "the fine rungs come from ZarrPyramidSource once the field is written.")
        c = self.channels.index(str(desc.channel))
        arr = self._tiles.get((desc.level, desc.key))
        if arr is None:
            return np.zeros((self.ladder.tile_px, self.ladder.tile_px), dtype=self.dtype)
        return arr[c]

    # ---- accumulation ------------------------------------------------------------------
    def add_field(self, region: str, fov: int, image: np.ndarray) -> list[TileDescriptor]:
        """Fold one projected field into every resident rung; returns the tiles it dirtied.

        *image* is the writer's ``(T, C, 1, Y, X)`` projection (a ``(C, Y, X)`` stack is accepted
        too). Bounded memory: one channel's resampled patch exists at a time, and nothing about the
        field is retained — the tiles are fixed-size and shared between fields.

        Hand the returned descriptors to ``TileCache.invalidate`` so the viewer re-reads exactly the
        coarse tiles this field changed, which is the seam ``_tiling.invalidate`` documents.
        """
        key = (str(region), int(fov))
        bbox = self.ladder.fov_bboxes.get(key)
        if bbox is None:
            raise KeyError(f"{key} has no recorded stage position; it is not on this ladder.")
        return self.add_patch(bbox, image)

    def add_patch(self, bbox_um: tuple, image: np.ndarray) -> list[TileDescriptor]:
        """Fold ONE world-placed patch of pixels into every resident rung.

        The general form of :meth:`add_field`, which is now the special case "this patch is one
        FOV, at that FOV's recorded frame extent". Splitting them costs nothing and buys the other
        producer this class needs: :class:`CompositePlateSource` folds in whole PLATE CELLS from
        ``_platecache`` — one per well, already composited from every FOV of that well by the
        preview pass — and a cell is a patch with no fov id to look up.

        *bbox_um* is where these pixels are in stage micrometres. *image* is ``(C, h, w)`` or the
        writer's ``(T, C, 1, Y, X)``; ``h`` and ``w`` need bear no relation to ``tile_px`` or to a
        frame, because :func:`_paste_field` maps world to pixels at both ends.
        """
        planes = self._planes(image)
        dirty: list[TileDescriptor] = []
        for lvl in self.levels:
            level = self.ladder.geometry.levels[lvl]
            scale = level.scale_um_per_px
            for cell in self._cells_for(lvl, bbox_um):
                cell_bbox = self.ladder.cell_bbox_um(lvl, cell)
                tile = self._tile(lvl, cell)
                touched = False
                for c in range(len(self.channels)):
                    touched |= _paste_field(tile[c], cell_bbox, scale, planes[c], bbox_um)
                if touched:
                    dirty.extend(TileDescriptor(lvl, cell, ch, cell_bbox) for ch in self.channels)
        return dirty

    # ---- internals ---------------------------------------------------------------------
    def _planes(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 5:
            t = min(self.t, arr.shape[0] - 1)
            arr = arr[t, :, 0]
        if arr.ndim != 3 or arr.shape[0] != len(self.channels):
            raise ValueError(
                f"expected a (T, C, 1, Y, X) or (C, Y, X) field with C={len(self.channels)}, "
                f"got shape {np.asarray(image).shape}")
        return arr

    def _cells_for(self, level: int, bbox: tuple) -> list:
        lv = self.ladder.geometry.levels[level]
        b = lv.bboxes
        hit = (b[:, 0] < bbox[2]) & (b[:, 2] > bbox[0]) & (b[:, 1] < bbox[3]) & (b[:, 3] > bbox[1])
        return [lv.keys[int(i)] for i in np.flatnonzero(hit)]

    def _tile(self, level: int, cell) -> np.ndarray:
        with self._lock:
            arr = self._tiles.get((level, cell))
            if arr is None:
                arr = np.zeros((len(self.channels), self.ladder.tile_px, self.ladder.tile_px),
                               dtype=self.dtype)
                self._tiles[(level, cell)] = arr
            return arr


# --- source 3: a RAW acquisition, straight off the microscope ----------------------------------

class ReaderTileSource:
    """``TileSource`` over a RAW acquisition, via the reader — no written plate required.

    The two sources above both need a plate that has already been WRITTEN: ``ZarrPyramidSource``
    reads ``plate.ome.zarr`` off disk, and ``InMemoryMultiscale`` is fed by the writer as fields
    land. That leaves the case the viewer spends most of its time in — an acquisition folder
    straight off the microscope, opened for a look — with no tile source at all, and therefore no
    deep zoom: the plate overview falls back to smooth-scaling one 88 px-per-well montage, so
    zooming in blurs instead of resolving.

    This closes that. It is deliberately the *simplest* thing that satisfies the protocol: every
    tile, at every rung, is composited the same way — take the FOVs whose frame overlaps the
    tile's world box and paste each one in. :func:`_paste_field` does the world-to-pixel mapping
    for both, so a per-FOV rung and a plate rung differ only in how many FOVs the loop visits
    (usually one, sometimes many). There is no second code path to keep in step.

    Tiles are **maximum-intensity projections** by default. The projection reuses the registered
    ``mip`` operator (``_engine._OPERATORS`` → ``projection.project``) rather than folding a max
    here, so ``reference`` — the Tenengrad best-focus plane — is a one-word change, and anything
    registered later with ``add_projector`` works with no edit to this class.

    **Why no pyramid level selection here.** ``PlateLadder.fov_source_level`` exists to pick a
    written pyramid level to composite from, which is what makes a coarse plate tile cheap on the
    zarr path. A raw acquisition has no written pyramid — there is exactly one resolution on disk
    — so a coarse tile necessarily decodes a full frame and area-averages it down. That cost is
    real and is why ``planes`` is cached by bytes: the same frame is reused across every tile and
    every rung that touches it. It is NOT worked around by inventing a pyramid, because writing
    one is the writer's job (IMA-184) and doing it here would duplicate it badly.

    **The montage disagrees, for now.** ``_PreviewWorker`` fills the 88 px plate montage from a
    single MID-STACK plane, because a projection there would multiply the first-paint cost by the
    stack depth. So the coarse montage and these tiles are not the same image on a raw
    acquisition. That is a real seam and it is recorded in ``NEXT_STEPS.md``; the projected tile
    is the one the product wants, so the montage is what should move.

    Empty world reads as ZEROS, never an error — the same convention
    :meth:`InMemoryMultiscale.read_tile` documents. A plate is part-acquired for most of a run and
    a viewport routinely covers stage area no FOV was ever placed on; raising there would make the
    fetch path throw once per empty tile per frame.
    """

    def __init__(self, reader, metadata: Mapping, ladder: PlateLadder, *,
                 projector: Optional[str] = "mip", z: Optional[int] = None, t: int = 0,
                 cache_bytes: Optional[int] = None) -> None:
        self.reader = reader
        self.meta = dict(metadata)
        self.ladder = ladder
        self.t = int(t)
        self.dtype = np.dtype(self.meta.get("dtype") or np.uint16)
        self.z_levels = list(self.meta.get("z_levels") or [0])

        # PROJECTED by default (Spencer: "I do want an MIP for this application"), reusing the
        # registered operator rather than folding a max here -- so `reference` (the Tenengrad
        # best-focus plane) is the same one-word change, and a projector added with
        # ``add_projector`` works with no edit to this class.
        #
        # Refuse a projector that does NOT consume z: this collapses a stack to one plane, and a
        # plane-op (consumes=frozenset(), e.g. decon or bgsub) has no z to collapse. Silently
        # running one per z and keeping the last would be a picture that looks plausible and is
        # not what was asked for.
        self.projector = projector
        self.z = None if z is None else int(z)
        if projector is not None and self.z is None:
            from squidmip._engine import operator_consumes

            if "z" not in operator_consumes(projector):
                raise ValueError(
                    f"projector {projector!r} does not consume z, so it cannot reduce a stack to "
                    "a tile. Pass a z-reducer (mip, reference) or an explicit z=.")
        elif projector is None and self.z is None:
            # Neither a projector nor a plane: fall back to the mid-stack plane _PreviewWorker
            # uses, so this degrades to the montage's own image rather than to z=0.
            self.z = int(self.z_levels[len(self.z_levels) // 2])

        self._planes = MemoryBoundedLRUCache(
            DEFAULT_PREVIEW_BUDGET_BYTES if cache_bytes is None else int(cache_bytes))

    # ---- TileSource --------------------------------------------------------------------
    def read_tile(self, desc: TileDescriptor) -> np.ndarray:
        """One channel of one tile, ``(tile_px, tile_px)`` in the acquisition's native dtype."""
        tile_px = self.ladder.tile_px
        bbox = tuple(float(v) for v in desc.bbox_um)
        # The tile's own resolution, derived from its box rather than from the level's nominal
        # scale: a plate rung's cells are square and uniform, but an FOV rung's tile IS the frame,
        # whose aspect need not match tile_px. Deriving it here keeps _paste_field exact for both.
        scale = (bbox[2] - bbox[0]) / float(tile_px)
        out = np.zeros((tile_px, tile_px), dtype=self.dtype)

        for key in self.ladder.fovs_overlapping(bbox):
            plane = self._plane(key, desc.channel)
            if plane is None:
                continue            # an unreadable field is a hole, not a dead viewport
            _paste_field(out, bbox, scale, plane, self.ladder.fov_bboxes[key])
        return out

    # ---- pixels ------------------------------------------------------------------------
    def _plane(self, key, channel: str):
        """The FOV's image for one channel — projected over z, or one plane if ``z`` was given.

        Cached by BYTES, and this is the whole performance story. A coarse plate tile touches many
        FOVs, adjacent tiles touch the same FOVs again, and every rung above revisits them, so
        without the cache a pan would re-project continuously. Note what is cached: the RESULT,
        one plane per FOV, not the stack — so a 10-deep projection costs 10 reads ONCE and then
        occupies exactly what a single-plane preview would.
        """
        region, fov = key
        ck = (str(region), int(fov), str(channel),
              "z%d" % self.z if self.z is not None else "p:%s" % self.projector, int(self.t))
        hit = self._planes.get(ck)
        if hit is not None:
            return hit
        try:
            if self.z is not None:
                plane = np.asarray(self._read(region, fov, channel, self.z))
            else:
                from squidmip._engine import _resolve_operator

                reduce = _resolve_operator(self.projector).fn
                plane = np.asarray(reduce(
                    np.asarray(self._read(region, fov, channel, z)) for z in self.z_levels))
        except Exception:
            return None             # decode failure: leave the hole, keep the viewport alive
        self._planes.put(ck, plane)
        return plane

    def _read(self, region, fov, channel: str, z: int):
        """One plane from the reader, tolerating readers whose ``read`` has no ``t``."""
        try:
            return self.reader.read(region, int(fov), str(channel), int(z), t=int(self.t))
        except TypeError:
            return self.reader.read(region, int(fov), str(channel), int(z))


# --- source 4: the composite. Plate rungs from the cache, FOV rungs from the reader ------------

def region_bbox_um(ladder: PlateLadder, region: str) -> Optional[tuple]:
    """World bbox of a whole REGION: the union of its FOV frames, or ``None`` if it has none.

    This is the world extent a plate CELL covers, and it is the same rectangle
    ``_placement.mosaic_extent_px`` scales into the 88 px cell — both are the bounding box of the
    region's placed frames, one in micrometres and one in pixels. That equality is what lets a
    cached cell be pasted back into world space without a second geometry to keep in step.
    """
    boxes = [b for k, b in ladder.fov_bboxes.items() if str(k[0]) == str(region)]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


class CompositePlateSource:
    """``TileSource`` that serves PLATE rungs from the persisted preview cells and FOV rungs from
    the reader. This is the composite source ``NEXT_STEPS.md`` scoped and did not build.

    The blocker, in Spencer's words:

        Coarse rungs cannot be served by ``ReaderTileSource`` as it stands: a fit-to-plate tile
        overlaps all 72 FOVs and measured **25 s** to build.

    The 25 s is not a slow loop, it is the arithmetic of the read: a fit-to-plate tile covers the
    whole sample, so every FOV in the acquisition overlaps it, and a raw acquisition has no
    written pyramid to read a coarse version from — each of those FOVs decodes a full frame and
    is area-averaged down to a handful of pixels. On this repo's 1536-well fixture the same tile
    would touch 1536 fields, not 72.

    The fix is to stop deriving that picture at read time, because we ALREADY DERIVED IT: the
    preview pass composites every well into an 88 px cell on open, and ``_platecache`` now keeps
    those cells across restarts. A plate rung is those cells pasted into world micrometres, which
    is what :meth:`InMemoryMultiscale.add_patch` does, so a fit-to-plate tile becomes a dict
    lookup in RAM.

    **What the plate rungs are, honestly.** They are montage resolution: 88 px per well, the same
    picture the plate overview already draws, now placed in stage micrometres and addressable as
    tiles. They are not a second, finer downsample chain, because building one would duplicate the
    writer's job (IMA-184) and would put the 25 s straight back. The FOV rungs below the crossover
    are where real resolution comes from, and they are unchanged: ``ReaderTileSource``, pixel for
    pixel.

    **A cell that is not cached is not silently black.** That tile is composited by
    ``ReaderTileSource``, at its real cost, and counted in :attr:`coarse_from_reader`. Serving
    zeros would be a picture that looks acquired and is not, and falling back quietly would hide
    the one number that says whether this is working.

    MEASURED, one fit-to-plate tile, this machine, page cache warm (so these are the conservative
    numbers)::

        dataset                          ReaderTileSource   composite, first   composite, steady
        real 10x tissue, 55 FOVs             2.387 s            65 ms             < 0.01 ms
        sim_1536wp, 1536 FOVs                8.448 s          1 439 ms               0.14 ms

    "first" includes seeding: reading the cells and pasting every well into every resident rung.
    "steady" is the lookup a pan or a zoom actually pays. The FOV counts are the point: the
    reader's cost grows with the SAMPLE, and the composite's steady cost does not.

    Note where the seeding cost went. After ``_platecache`` compacted its cells into one
    memory-mapped page the seed only improved from 1 591 ms to 1 439 ms, because seeding is not
    I/O bound: it is 1536 ``_paste_field`` calls into the resident rungs. Making it cheaper means
    pasting fewer, larger patches (the page IS one array; a rung could be resampled from it whole),
    and that is worth doing when someone measures a coarse-rung stall, not before.
    """

    def __init__(self, reader, metadata: Mapping, ladder: PlateLadder, *,
                 cache=None, cells: Optional[Mapping] = None,
                 budget_bytes: Optional[int] = None, **fov_kwargs) -> None:
        self.reader = reader
        self.meta = dict(metadata)
        self.ladder = ladder
        self.cache = cache
        self.fov_source = ReaderTileSource(reader, metadata, ladder, **fov_kwargs)
        self.channels = [str(c["name"]) for c in (self.meta.get("channels") or [])] or ["0"]
        self.dtype = np.dtype(self.meta.get("dtype") or np.uint16)
        self.seeded: set = set()
        self.coarse_from_cells = 0
        self.coarse_from_reader = 0
        try:
            self.plate_source = InMemoryMultiscale(
                ladder, self.channels, self.dtype,
                budget_bytes=(DEFAULT_PREVIEW_BUDGET_BYTES if budget_bytes is None
                              else int(budget_bytes)),
                t=int(fov_kwargs.get("t", 0)))
        except ValueError:
            # No plate rungs on this ladder (a single-FOV acquisition, or a tile_px larger than
            # the whole sample). Then every rung is an FOV rung and there is nothing to compose;
            # this degrades to exactly ReaderTileSource rather than pretending otherwise.
            self.plate_source = None
        self._seed_pending = cells is not None or cache is not None
        self._cells = dict(cells) if cells else None

    # ---- seeding ------------------------------------------------------------------------
    def seed(self, cells: Mapping) -> int:
        """Fold ``{region: (C, h, w) cell}`` into the plate rungs. Returns regions folded in.

        A cell may be a ``_platecache.CellTile``, in which case only the sub-rectangle it says it
        covers is used — the letterbox padding around a mosaic is not acquired pixels and must not
        be pasted into the world as though it were.
        """
        if self.plate_source is None:
            return 0
        n = 0
        for region, cell in cells.items():
            if self._add_cell(str(region), cell):
                n += 1
        return n

    def _add_cell(self, region: str, cell) -> bool:
        bbox = region_bbox_um(self.ladder, region)
        if bbox is None:
            return False                    # a well with no stage position is not on this ladder
        arr = np.asarray(cell)
        box = getattr(cell, "box", None)
        if box is not None:
            top, left, h, w = (int(v) for v in box)
            arr = arr[:, top:top + h, left:left + w]
        if arr.ndim != 3 or arr.shape[0] != len(self.channels) or min(arr.shape[1:]) < 1:
            return False
        self.plate_source.add_patch(bbox, arr)
        self.seeded.add(region)
        return True

    def _ensure_seeded(self) -> None:
        """Load the cells the first time a coarse tile is actually asked for, never at open.

        Lazy on purpose. Seeding reads one small file per well, and a session that never zooms out
        past the crossover should not pay for a rung it will not look at. The plate overview's own
        preview pass is what populates the cache in the first place, so this costs nothing on a
        first open and is a warm read on every one after.
        """
        if not self._seed_pending:
            return
        self._seed_pending = False
        if self._cells:
            self.seed(self._cells)
            self._cells = None
        if self.cache is not None:
            regions = sorted({str(k[0]) for k in self.ladder.fov_bboxes})
            self.seed(self.cache.load_all(regions))

    # ---- TileSource ---------------------------------------------------------------------
    def read_tile(self, desc: TileDescriptor) -> np.ndarray:
        """One channel of one tile. Satisfies ``_tiling.TileSource``, like the three above."""
        if self.ladder.is_fov_level(desc.level) or self.plate_source is None:
            return self.fov_source.read_tile(desc)
        self._ensure_seeded()
        if desc.level not in self.plate_source.levels or not self._covered(desc):
            self.coarse_from_reader += 1
            return self.fov_source.read_tile(desc)
        self.coarse_from_cells += 1
        return self.plate_source.read_tile(desc)

    def _covered(self, desc: TileDescriptor) -> bool:
        """Whether every region overlapping this tile was seeded from a cell.

        Partial coverage goes to the reader whole rather than being drawn half from cells and
        half from black. A tile assembled from two sources at two resolutions is the seam this
        source exists to remove.
        """
        bbox = tuple(float(v) for v in desc.bbox_um)
        regions = {str(k[0]) for k in self.ladder.fovs_overlapping(bbox)}
        return bool(regions) and regions <= self.seeded
