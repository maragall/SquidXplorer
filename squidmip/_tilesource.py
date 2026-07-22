"""IMA-217: the tile ladder + the two ``TileSource`` implementations IMA-216 asks for.

``_tiling.py`` (IMA-216) owns the *algorithm* — LOD pick, frustum cull, byte-budget LRU — and
declares one hole: a :class:`~squidmip._tiling.TileSource` that turns a
:class:`~squidmip._tiling.TileDescriptor` into pixels, plus the
:class:`~squidmip._tiling.Geometry` describing what tiles exist. This module fills both, twice::

    plate.ome.zarr on disk (IMA-184/185)  ─►  ZarrPyramidSource   (persistent, pixel-exact)
    a live acquisition stream (on_well)   ─►  InMemoryMultiscale  (preview, byte-budgeted)

Both hand out the SAME ``Geometry``, so the viewer can start on the RAM preview mid-run and
switch to the zarr source when the write finishes without re-deriving a single coordinate.

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
        cached = self._stores.get((fov_key, level))
        if cached is None:
            import tensorstore as ts
            path = self._field_dirs[fov_key] / str(level)
            cached = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}},
                             open=True).result()
            self._stores[(fov_key, level)] = cached
        return cached

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
        planes = self._planes(image)
        dirty: list[TileDescriptor] = []
        for lvl in self.levels:
            level = self.ladder.geometry.levels[lvl]
            scale = level.scale_um_per_px
            for cell in self._cells_for(lvl, bbox):
                cell_bbox = self.ladder.cell_bbox_um(lvl, cell)
                tile = self._tile(lvl, cell)
                touched = False
                for c in range(len(self.channels)):
                    touched |= _paste_field(tile[c], cell_bbox, scale, planes[c], bbox)
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
