"""The tile ladder and the ``TileSource`` implementations over it.

World space is stage micrometres throughout; per-FOV rungs sit below a crossover, plate-grid
rungs above it, so tile count follows the screen rather than the sample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Hashable, Mapping, Optional, Sequence

import numpy as np

from squidxplorer._budget import cache_budget

from squidxplorer._montage import _area_downsample
from squidxplorer._mosaic_source import MemoryBoundedLRUCache
from squidxplorer._output import _PYRAMID_MAX_LEVELS, _PYRAMID_MIN_YX, pyramid_shapes
from squidxplorer._tiling import Geometry, Level, TileDescriptor
from squidxplorer.projection import cast_like

# Plate-rung tile size in pixels; 512 keeps one uint16 tile at 512 KB.
DEFAULT_TILE_PX = 512

# Default byte budget for the in-RAM preview multiscale, derived from available memory.
DEFAULT_PREVIEW_BUDGET_BYTES = cache_budget()

# How many plate rungs to stack above the per-FOV ones; a runaway guard, not a tuning.
_MAX_PLATE_LEVELS = 12

# An FOV pitch below frame_extent_um / _MM_PITCH_RATIO is millimetres leaked into a ``_um`` key.
_MM_PITCH_RATIO = 100.0


def fov_bboxes_um(positions_um: Mapping[tuple, tuple], frame_shape, pixel_size_um) -> dict:
    """``{(region, fov): (x0, y0, x1, y1)}`` in stage µm, from FOV **centre** positions."""
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
    """Smallest positive gap between distinct coordinates along one axis (inf if there is none)."""
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
            "(squidxplorer.reader.load_fov_positions_um). Refusing to build a 1000x-too-small plate."
        )


@dataclass(frozen=True)
class PlateLadder:
    """The world layout + the :class:`~squidxplorer._tiling.Geometry` built from it."""

    geometry: Geometry
    fov_bboxes: dict
    fov_level_shapes: list
    n_fov_levels: int
    tile_px: int
    world_bbox_um: tuple
    pixel_size_um: float
    frame_shape: tuple
    _plate_grids: dict = _dc_field(default_factory=dict, repr=False)

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
        """Which WRITTEN per-FOV pyramid level to composite a tile of *scale* from."""
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
    """Build the whole tile ladder from acquisition metadata alone — pure, no I/O, no pixels."""
    positions = metadata.get("fov_positions_um") or {}
    if not positions:
        raise ValueError(
            "no fov_positions_um in the metadata: without stage coordinates every FOV would sit at "
            "the same spot and the plate view would be a single stacked pile. (coordinates.csv "
            "missing or unusable — see squidxplorer.reader._fov_positions_um_or_empty.)")
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

    # Keep a per-FOV rung only while its tile covers at least as much world as a plate tile would;
    # level 0 is always kept as the only pixel-exact read path.
    n_fov_levels = 1
    for i in range(1, len(fov_scales)):
        if fov_scales[i] * tile_px <= fov_extent_um:
            n_fov_levels = i + 1
        else:
            break

    levels: list[Level] = [Level(fov_scales[i], arr, keys) for i in range(n_fov_levels)]

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
        # A coarser rung holding >= the tiles of the one below buys nothing; try the next doubling.
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
    """Grid cells that at least one FOV touches, in row-major order — empty cells dropped."""
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


def _resample(plane: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """*plane* -> ``(out_h, out_w)`` float32. Area-average when shrinking, nearest when growing."""
    h, w = plane.shape
    if out_h <= h and out_w <= w:
        return _area_downsample(plane, out_h, out_w)
    yi = np.minimum((np.arange(out_h) * h) // max(out_h, 1), h - 1)
    xi = np.minimum((np.arange(out_w) * w) // max(out_w, 1), w - 1)
    return plane[yi][:, xi].astype(np.float32, copy=False)


def _paste_field(dst: np.ndarray, dst_bbox_um: tuple, scale_um_per_px: float,
                 plane: np.ndarray, fov_bbox_um: tuple) -> bool:
    """Resample the part of *plane* inside *dst_bbox_um* into *dst*. True if anything landed."""
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
    # in place: `_resample` always hands back a fresh float32 array, and this runs per tile
    dst[dy0:dy1, dx0:dx1] = cast_like(resampled, dst.dtype, copy=False)
    return True


# THE one store walk (contract.store): v0.4/v0.5-normalising attrs and plate-dir resolution.
from squidxplorer.contract.store import ome_attrs as _read_ome
from squidxplorer.contract.store import resolve_plate_dir as _resolve_plate_dir


class ZarrPyramidSource:
    """``TileSource`` over a written ``plate.ome.zarr`` — the persistent, pixel-exact path."""

    def __init__(self, plate_path, *, tile_px: int = DEFAULT_TILE_PX,
                 min_yx: int = _PYRAMID_MIN_YX, max_levels: int = _PYRAMID_MAX_LEVELS) -> None:
        self.plate_dir = _resolve_plate_dir(plate_path)
        self._stores: dict = {}
        layout = plate_layout_from_store(self.plate_dir)
        self.channels: list[str] = layout["channels"]
        self._field_dirs: dict = layout["field_dirs"]
        meta = {
            "fov_positions_um": layout["centres_um"],
            "pixel_size_um": layout["pixel_size_um"],
            "frame_shape": layout["frame_shape"],
        }
        self.ladder = plate_ladder(meta, tile_px=tile_px, min_yx=min_yx, max_levels=max_levels)

    def read_tile(self, desc: TileDescriptor) -> np.ndarray:
        """One tile as a 2-D native-dtype array; the timepoint comes off the descriptor."""
        c = self._channel_index(desc.channel)
        if self.ladder.is_fov_level(desc.level):
            return self._read_fov_plane(desc.key, desc.level, c, int(desc.time_point))
        return self._composite_cell(desc.level, desc.key, c, int(desc.time_point))

    def _channel_index(self, channel: str) -> int:
        s = str(channel)
        if s in self.channels:
            return self.channels.index(s)
        if s.isdigit() and int(s) < len(self.channels):
            return int(s)          # positional channel ids, as _tiling's default ("0", "1", ...)
        raise KeyError(f"unknown channel {channel!r}; plate has {self.channels}")

    def _store(self, fov_key, level: int):
        # Through the process-wide pool: bounds live handles and binds readers to one cache_pool.
        from squidxplorer._tsctx import HANDLES
        return HANDLES.get(self._field_dirs[fov_key] / str(level))

    def _read_fov_plane(self, fov_key, level: int, c: int, time_point: int) -> np.ndarray:
        store = self._store(fov_key, level)
        time_point = min(max(0, int(time_point)), store.shape[0] - 1)     # clamped to what this store actually holds
        return np.asarray(store[time_point, c, 0].read().result())

    def _composite_cell(self, level: int, key, c: int, time_point: int) -> np.ndarray:
        bbox = self.ladder.cell_bbox_um(level, key)
        scale = self.ladder.geometry.levels[level].scale_um_per_px
        src_level = self.ladder.fov_source_level(scale)
        tile = np.zeros((self.ladder.tile_px, self.ladder.tile_px), dtype=self._dtype())
        for fov_key in self.ladder.fovs_overlapping(bbox):
            plane = self._read_fov_plane(fov_key, src_level, c, time_point)
            _paste_field(tile, bbox, scale, plane, self.ladder.fov_bboxes[fov_key])
        return tile

    def _dtype(self):
        first = next(iter(self._field_dirs))
        return np.dtype(self._store(first, 0).dtype.numpy_dtype)


def plate_layout_from_store(plate_dir: Path) -> dict:
    """Walk the plate's NGFF metadata into ``{centres_um, pixel_size_um, frame_shape, ...}``."""
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
                    "IMA-217 writer (squidxplorer._output.write_plate), which emits it.")
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


class InMemoryMultiscale:
    """``TileSource`` holding the coarse plate rungs in RAM, under an explicit byte budget.

    Rungs are admitted coarsest-first while their fully-filled capacity fits ``budget_bytes``;
    the constructor raises if the coarsest rung alone does not fit.
    """

    def __init__(self, ladder: PlateLadder, channels: Sequence[str], dtype=np.uint16, *,
                 budget_bytes: int = DEFAULT_PREVIEW_BUDGET_BYTES, time_point: int = 0) -> None:
        import threading

        self.ladder = ladder
        self.channels = [str(c) for c in channels]
        self.dtype = np.dtype(dtype)
        self.budget_bytes = int(budget_bytes)
        self.time_point = int(time_point)
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

    @property
    def nbytes(self) -> int:
        """Bytes actually allocated so far — always <= ``capacity_bytes``."""
        return sum(a.nbytes for a in self._tiles.values())

    def __len__(self) -> int:
        return len(self._tiles)

    def read_tile(self, desc: TileDescriptor) -> np.ndarray:
        """One channel of one resident tile; untouched tiles read as zeros, another timepoint raises."""
        if int(desc.time_point) != self.time_point:
            raise KeyError(
                f"this preview holds timepoint {self.time_point}; tile {desc.key!r} was asked "
                f"for at timepoint {desc.time_point}. Its pixels are not that frame's.")
        if desc.level not in self.levels:
            raise KeyError(
                f"level {desc.level} is not resident in this preview (resident: {self.levels}); "
                "the fine rungs come from ZarrPyramidSource once the field is written.")
        c = self.channels.index(str(desc.channel))
        arr = self._tiles.get((desc.level, desc.key))
        if arr is None:
            return np.zeros((self.ladder.tile_px, self.ladder.tile_px), dtype=self.dtype)
        return arr[c]

    def add_field(self, region: str, fov: int, image: np.ndarray) -> list[TileDescriptor]:
        """Fold one projected field into every resident rung; returns the tiles it dirtied."""
        key = (str(region), int(fov))
        bbox = self.ladder.fov_bboxes.get(key)
        if bbox is None:
            raise KeyError(f"{key} has no recorded stage position; it is not on this ladder.")
        return self.add_patch(bbox, image)

    def add_patch(self, bbox_um: tuple, image: np.ndarray) -> list[TileDescriptor]:
        """Fold ONE world-placed patch of pixels into every resident rung."""
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
                    dirty.extend(TileDescriptor(lvl, cell, ch, cell_bbox, self.time_point)
                                 for ch in self.channels)
        return dirty

    def _planes(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 5:
            t = min(self.time_point, arr.shape[0] - 1)
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


class ReaderTileSource:
    """``TileSource`` over a RAW acquisition, via the reader — no written plate required.

    Every tile at every rung is composited the same way: the overlapping FOVs are pasted in,
    projected over z (or a single plane), with the result plane cached by bytes.
    """

    def __init__(self, reader, metadata: Mapping, ladder: PlateLadder, *,
                 operator: Optional[str] = "mip", z_level: Optional[int] = None,
                 cache_bytes: Optional[int] = None) -> None:
        self.reader = reader
        self.meta = dict(metadata)
        self.ladder = ladder
        self.dtype = np.dtype(self.meta.get("dtype") or np.uint16)
        self.z_levels = list(self.meta.get("z_levels") or [0])

        # Refuse an operator that does not consume z: it cannot reduce a stack to a tile.
        self.operator = operator
        self.z_level = None if z_level is None else int(z_level)
        if operator is not None and self.z_level is None:
            from squidxplorer._engine import operator_consumes

            if "z" not in operator_consumes(operator):
                raise ValueError(
                    f"operator {operator!r} does not consume z, so it cannot reduce a stack to "
                    "a tile. Pass a z-reducer (mip, reference) or an explicit z_level=.")
        elif operator is None and self.z_level is None:
            # Neither an operator nor a plane: fall back to the montage's mid-stack plane.
            self.z_level = int(self.z_levels[len(self.z_levels) // 2])

        self._planes = MemoryBoundedLRUCache(
            DEFAULT_PREVIEW_BUDGET_BYTES if cache_bytes is None else int(cache_bytes))

    def read_tile(self, desc: TileDescriptor) -> np.ndarray:
        """One channel of one tile, ``(tile_px, tile_px)`` in the acquisition's native dtype."""
        tile_px = self.ladder.tile_px
        bbox = tuple(float(v) for v in desc.bbox_um)
        # Derive the tile's resolution from its own box: an FOV rung's tile IS the frame, whose
        # aspect need not match tile_px.
        scale = (bbox[2] - bbox[0]) / float(tile_px)
        out = np.zeros((tile_px, tile_px), dtype=self.dtype)

        for key in self.ladder.fovs_overlapping(bbox):
            plane = self._plane(key, desc.channel, int(desc.time_point))
            if plane is None:
                continue            # an unreadable field is a hole, not a dead viewport
            _paste_field(out, bbox, scale, plane, self.ladder.fov_bboxes[key])
        return out

    def _plane(self, key, channel: str, time_point: int):
        """The FOV's image for one channel at one timepoint — projected over z, or one plane."""
        region, fov = key
        ck = (str(region), int(fov), str(channel),
              "z%d" % self.z_level if self.z_level is not None else "op:%s" % self.operator,
              int(time_point))
        hit = self._planes.get(ck)
        if hit is not None:
            return hit
        try:
            if self.z_level is not None:
                plane = np.asarray(self._read(region, fov, channel, self.z_level, time_point))
            else:
                from squidxplorer._engine import _resolve_operator

                reduce = _resolve_operator(self.operator).fn
                plane = np.asarray(reduce(
                    np.asarray(self._read(region, fov, channel, z, time_point)) for z in self.z_levels))
        except Exception:
            return None             # decode failure: leave the hole, keep the viewport alive
        self._planes.put(ck, plane)
        return plane

    def _read(self, region, fov, channel: str, z_level: int, time_point: int):
        """One plane from the reader. ``time_point`` is part of the contract's ``read`` — the
        old TypeError retry existed for readers without it, of which there are none, so its
        only live effect was to swallow a genuine TypeError from INSIDE a decode and silently
        re-read frame 0."""
        return self.reader.read(region, int(fov), str(channel), int(z_level),
                                time_point=int(time_point))


def region_bbox_um(ladder: PlateLadder, region: str) -> Optional[tuple]:
    """World bbox of a whole REGION: the union of its FOV frames, or ``None`` if it has none."""
    boxes = [b for k, b in ladder.fov_bboxes.items() if str(k[0]) == str(region)]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


class CompositePlateSource:
    """``TileSource`` serving plate rungs from the persisted preview cells, FOV rungs from the reader.

    An unseeded coarse tile is composited by ``ReaderTileSource`` at its real cost and counted in
    :attr:`coarse_from_reader`, never served as silent black.
    """

    def __init__(self, reader, metadata: Mapping, ladder: PlateLadder, *,
                 cache=None, cells: Optional[Mapping] = None, time_point: int = 0,
                 budget_bytes: Optional[int] = None, **fov_kwargs) -> None:
        self.reader = reader
        self.meta = dict(metadata)
        self.ladder = ladder
        self.cache = cache
        # `time_point` is which frame the plate rungs are; a cache for a different frame is
        # refused, not reconciled.
        self.time_point = max(0, int(time_point))
        cache_t = getattr(cache, "time_point", self.time_point) if cache is not None else self.time_point
        if int(cache_t) != self.time_point:
            raise ValueError(
                f"CompositePlateSource(time_point={self.time_point}) was handed a plate cell cache "
                f"for timepoint {cache_t}: its cells would be pasted into the world under the wrong "
                "frame.")
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
                time_point=self.time_point)
        except ValueError:
            # No plate rungs on this ladder: degrade to exactly ReaderTileSource.
            self.plate_source = None
        self._seed_pending = cells is not None or cache is not None
        self._cells = dict(cells) if cells else None

    def seed(self, cells: Mapping) -> int:
        """Fold ``{region: (C, h, w) cell}`` into the plate rungs. Returns regions folded in."""
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
        """Load the cells the first time a coarse tile is actually asked for, never at open."""
        if not self._seed_pending:
            return
        self._seed_pending = False
        if self._cells:
            self.seed(self._cells)
            self._cells = None
        if self.cache is not None:
            regions = sorted({str(k[0]) for k in self.ladder.fov_bboxes})
            self.seed(self.cache.load_all(regions))

    def read_tile(self, desc: TileDescriptor) -> np.ndarray:
        """One channel of one tile; a coarse tile at another timepoint goes to the reader."""
        if self.ladder.is_fov_level(desc.level) or self.plate_source is None:
            return self.fov_source.read_tile(desc)
        if int(desc.time_point) != self.time_point:
            self.coarse_from_reader += 1
            return self.fov_source.read_tile(desc)
        self._ensure_seeded()
        if desc.level not in self.plate_source.levels or not self._covered(desc):
            self.coarse_from_reader += 1
            return self.fov_source.read_tile(desc)
        self.coarse_from_cells += 1
        return self.plate_source.read_tile(desc)

    def _covered(self, desc: TileDescriptor) -> bool:
        """Whether every region overlapping this tile was seeded from a cell."""
        bbox = tuple(float(v) for v in desc.bbox_um)
        regions = {str(k[0]) for k in self.ladder.fovs_overlapping(bbox)}
        return bool(regions) and regions <= self.seeded
