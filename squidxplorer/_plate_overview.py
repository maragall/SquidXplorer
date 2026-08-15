"""The plate overview: the low-resolution, one-cell-per-well navigator, and the geometry under it."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, QRectF, QThread, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QRegion
from qtpy.QtWidgets import QWidget

from squidxplorer import _qtstyle
from squidxplorer._budget import cache_budget
from squidxplorer._logpane import get_logger
from squidxplorer._mosaic_source import MemoryBoundedLRUCache
from squidxplorer._montage import _area_downsample, composite
from squidxplorer._plate import display_well_id
from squidxplorer._tiling import TileDescriptor
from squidxplorer.contract import field_levels, field_path

log = get_logger("viewer")

_BG = _qtstyle.BG
_GRID, _RED, _MUTED, _ACCENT = _qtstyle.GRID, _qtstyle.RED, _qtstyle.MUTED, _qtstyle.ACCENT
_STATUS = _qtstyle.STATUS

_CELL = 88
_PUSH_PX = 512
_HDR, _COLH = 46, 30
_PAD = 16

# Labels are set in PIXELS, not points: a point size resolves against logicalDpiY, which varies
# per screen/OS, so labels would not hold still when the window changed screens.
_LABEL_PX = 11
_SCALE_PX = 10


def _plate_font(px: int, weight=None) -> QFont:
    """A plate label at a fixed PIXEL size, so it is the same size on every display."""
    f = QFont("Helvetica Neue")
    f.setPixelSize(int(px))
    if weight is not None:
        f.setWeight(weight)
    return f

_VIEW_WASH = QColor(88, 166, 255, 40)

# Full-alpha bounding box instead of a wash: a wash changes the pixels the user is judging.
_SEL_FRAME = QColor(88, 166, 255)

# Below this grid size the wash is unambiguous (cells are huge, few selected at once).
_FRAME_MIN_GRID = 3


def frames_for_grid(nrows: int, ncols: int) -> bool:
    """True when the selection is drawn as a bounding box rather than an alpha wash."""
    return nrows > _FRAME_MIN_GRID or ncols > _FRAME_MIN_GRID


def selection_frame_pen_px(cell_disp: float) -> float:
    """Frame stroke width for a cell *cell_disp* px across, clamped so a huge cell gets no slab."""
    return max(1.0, min(cell_disp * 0.10, 3.0))

# A quarter of the cache budget: the deep-zoom tile cache is one of several consumers.
_TILE_CACHE_BYTES = max(64 << 20, cache_budget() // 4)

# The same quarter-share discipline for this file's other consumers — the loupe's crop and
# coarse-plane caches, the contrast-window memo and the tile QImage cache. Each is an
# independent LRU bounded at this cap; the cap is a ceiling, not a reservation.
_AUX_CACHE_BYTES = max(64 << 20, cache_budget() // 4)

#: Nominal bytes of one per-channel (lo, hi) contrast pair, for sizing window-memo entries.
_WINDOW_PAIR_NBYTES = 16


class _SizedValue:
    """Adapts a value without an ``nbytes`` (a QImage, a list of contrast pairs) to the
    byte-bounded LRU, which sizes every entry — on put and on eviction — by ``value.nbytes``."""

    __slots__ = ("value", "nbytes")

    def __init__(self, value, nbytes: int):
        self.value = value
        self.nbytes = int(nbytes)

_CLICK_SLOP = 3   # px of travel below which a Shift-drag counts as a click


def well_at(rows, cols, by_rc, px: float, py: float, cell_disp: float) -> Optional[dict]:
    """Map a plate pixel (px, py) at *cell_disp* px/well to a cell, or None if out of bounds."""
    if px < 0 or py < 0:
        return None
    ci, ri = int(px // cell_disp), int(py // cell_disp)
    if ci >= len(cols) or ri >= len(rows):
        return None
    return {"row_index": ri, "col_index": ci, "row": rows[ri], "col": cols[ci],
            "well_id": by_rc.get((ri, ci))}


def cells_in_rect(rows, cols, by_rc, x0: float, y0: float, x1: float, y1: float,
                  cell_disp: float) -> list:
    """Every acquired cell whose square meets the drag rect (x0,y0)-(x1,y1), row-major sorted."""
    if cell_disp <= 0:
        return []
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    if hi_x < 0 or hi_y < 0:
        return []
    c0, c1 = int(max(0.0, lo_x) // cell_disp), int(max(0.0, hi_x) // cell_disp)
    r0, r1 = int(max(0.0, lo_y) // cell_disp), int(max(0.0, hi_y) // cell_disp)
    c1, r1 = min(c1, len(cols) - 1), min(r1, len(rows) - 1)
    return [(ri, ci) for ri in range(r0, r1 + 1) for ci in range(c0, c1 + 1) if (ri, ci) in by_rc]


def _fit_cell(a: np.ndarray) -> np.ndarray:
    """Resize a 2D plane to exactly (_CELL, _CELL) for the montage tile."""
    if a.shape == (_CELL, _CELL):
        return a
    if a.shape[0] >= _CELL and a.shape[1] >= _CELL:
        return _area_downsample(a, _CELL, _CELL)
    yi = (np.arange(_CELL) * a.shape[0]) // _CELL
    xi = (np.arange(_CELL) * a.shape[1]) // _CELL
    return a[yi][:, xi].astype(np.float32)


def _fit_box(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a 2D plane to exactly (h, w), the arbitrary-target sibling of :func:`_fit_cell`."""
    h, w = max(1, int(h)), max(1, int(w))
    if a.shape == (h, w):
        return a
    if a.shape[0] >= h and a.shape[1] >= w:
        return _area_downsample(a, h, w)
    yi = (np.arange(h) * a.shape[0]) // h
    xi = (np.arange(w) * a.shape[1]) // w
    return a[yi][:, xi].astype(np.float32)
def _box_union(a, b):
    """Union of two ``(top, left, h, w)`` boxes; ``a`` may be None (nothing accumulated yet)."""
    if a is None:
        return tuple(int(v) for v in b)
    top = min(a[0], b[0])
    left = min(a[1], b[1])
    bottom = max(a[0] + a[2], b[0] + b[2])
    right = max(a[1] + a[3], b[1] + b[3])
    return (int(top), int(left), int(bottom - top), int(right - left))


def resolve_plate_root(path) -> tuple[Path, bool]:
    """(path, is_plate): is_plate True when *path* already holds an OME-zarr plate."""
    p = Path(path)
    if (p / "plate.ome.zarr").is_dir() or (p.name.endswith(".zarr") and (p / "zarr.json").exists()):
        return p, True
    return p, False
class _RunningContrast:
    """Per-channel global contrast that updates as wells stream in (histogram over tiles so far).

    Each channel also carries an auto/manual latch: once the user drags a channel's contrast it
    latches manual and the next well to land can no longer stomp the window they just set.
    """

    def __init__(self, n_ch: int, dmax: float, pct=(1.0, 99.8), bins=512):
        self._bins, self._dmax, self._pct = bins, max(1.0, float(dmax)), pct
        self._hist = [np.zeros(bins, dtype=np.int64) for _ in range(n_ch)]
        self._manual: dict[int, tuple[float, float]] = {}
        # The window the owning viewer resolved and is rendering with; not the same as _manual.
        self._followed: dict[int, tuple[float, float]] = {}

    @property
    def dmax(self) -> float:
        return self._dmax

    def add(self, ch: int, tile: np.ndarray):
        idx = np.clip((tile.ravel() / self._dmax * self._bins).astype(int), 0, self._bins - 1)
        self._hist[ch] += np.bincount(idx, minlength=self._bins)

    def set_manual(self, ch: int, lo: float, hi: float):
        """Latch *ch* to a user-set window (hi kept above lo so _window never divides by zero)."""
        self._manual[ch] = (float(lo), float(max(hi, lo + 1)))

    def set_auto(self, ch: int):
        """Unlatch *ch* — it goes back to following the running histogram."""
        self._manual.pop(ch, None)

    def is_manual(self, ch: int) -> bool:
        return ch in self._manual

    def set_followed(self, ch: int, lo: float, hi: float):
        """Record the window the owning viewer resolved for *ch*.

        Not a latch: a followed window is a sink recording what the owner is showing, while a
        manual latch is a policy decision only the user makes.
        """
        self._followed[ch] = (float(lo), float(max(hi, lo + 1)))

    def is_followed(self, ch: int) -> bool:
        return ch in self._followed

    def resolve(self, ch: int, auto: tuple[float, float],
                follow: bool = True) -> tuple[float, float]:
        """Precedence: user latch > the owning viewer's window > whatever the caller computed.

        *follow* is False only for SCOPE_PER_REGION, meaning "derive this cell's window from this
        cell's pixels" — a user's explicit latch still wins even then.
        """
        if ch in self._manual:
            return self._manual[ch]
        if follow and ch in self._followed:
            return self._followed[ch]
        return auto

    def window(self, ch: int) -> tuple[float, float]:
        """(lo, hi) for this channel: user latch, else the viewer's, else the running histogram."""
        return self.resolve(ch, self._auto_window(ch))

    def _auto_window(self, ch: int) -> tuple[float, float]:
        """Background peak to black, 99.9th percentile on top; ignores any latch.

        A degenerate window (hi <= lo) is returned deliberately for a blank/flat channel so it
        renders black rather than reading as signal.
        """
        h = self._hist[ch].astype(np.float64)
        tot = h.sum()
        if tot == 0:
            return 0.0, self._dmax
        centers = (np.arange(self._bins) + 0.5) / self._bins * self._dmax
        cdf = np.cumsum(h) / tot
        mode_val = float(centers[int(np.argmax(h))])
        med_bin = min(int(np.searchsorted(cdf, 0.5)), self._bins - 1)
        bg = h[: med_bin + 1]
        bg_tot = float(bg.sum())
        if bg_tot > 0:
            bc = centers[: med_bin + 1]
            bg_mean = float((bc * bg).sum() / bg_tot)
            bg_std = float(np.sqrt(max(0.0, ((bc - bg_mean) ** 2 * bg).sum() / bg_tot)))
        else:
            bg_std = abs(mode_val) * 0.1
        lo = mode_val + 2.0 * bg_std                                 # push background to black
        hi = float(centers[min(int(np.searchsorted(cdf, 0.999)), self._bins - 1)])   # 99.9th pct
        if hi <= lo:
            return lo, lo                                            # degenerate -> black
        return float(lo), float(hi)

    def windows(self) -> list[tuple[float, float]]:
        return [self.window(ch) for ch in range(len(self._hist))]


# Contrast scope: GLOBAL is one window per channel across the whole plate (wells stay
# comparable); PER_REGION is one window per cell (each region fills its own range, at the cost
# of comparability). It is a display control, never a run parameter — flipping it re-composites
# from retained tiles rather than re-running the plate.

_PCT = (1.0, 99.8)


def _pct_window(a: np.ndarray, pct=_PCT) -> tuple[float, float]:
    """Exact percentile window over *a* (the running histogram only quantizes to bins)."""
    if a.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(a, pct)
    return float(lo), float(hi)


# The loupe reads the real data behind whatever layer is on screen (raw TIFFs, or a windowed
# read of the written pyramid) — the plate montage tile is far too downsampled to magnify.
# Magnification is derived from the current plate zoom and capped at native resolution:
#   s_plate = cd / well_px; s_loupe = min(1.0, MAG * s_plate); level = coarsest still >= s_loupe.

_LOUPE_PX = 240
_LOUPE_MAG = 8.0
_LOUPE_HOLD_MS = 350
_LOUPE_SLOP = 3
_LOUPE_MAX_CROP = 2 * _LOUPE_PX   # a source can run out of pyramid levels, so cap the read too


def _fov_of_well(well_id, fovs_per_region=None) -> int:
    """The FOV index to address for *well_id* when nothing has resolved one from the hit-test."""
    if fovs_per_region:
        fovs = fovs_per_region.get(well_id)
        if fovs:
            return int(fovs[0])
    return 0


def loupe_scale(cd: float, well_px: int, mag: float = _LOUPE_MAG,
                inset_px: int = _LOUPE_PX) -> tuple[float, float]:
    """(s_loupe, M) for a plate showing ``cd`` screen px per well of ``well_px`` image px.

    ``s_loupe`` is clamped to 1.0 (native), floored at the plate's own scale (so a
    past-native zoom doesn't demagnify), and floored again at ``inset_px / well_px`` so the
    inset shows at most one whole well.
    """
    well_px = max(1, int(well_px))
    s_plate = max(1e-9, float(cd) / well_px)
    fill_well = float(inset_px) / well_px
    # Cap at native first, then floor at the plate's scale — capping last would demagnify.
    s_loupe = max(s_plate, min(1.0, max(mag * s_plate, fill_well)))
    return s_loupe, s_loupe / s_plate


def loupe_level(s_loupe: float, n_levels: int) -> int:
    """Coarsest pyramid level whose native resolution still satisfies ``s_loupe``."""
    s = min(1.0, max(1e-9, float(s_loupe)))
    return int(max(0, min(int(np.floor(np.log2(1.0 / s))), max(0, int(n_levels) - 1))))


def loupe_crop_px(s_loupe: float, level: int, inset_px: int = _LOUPE_PX) -> int:
    """Image pixels to read at ``level`` to fill an ``inset_px`` square inset."""
    eff = max(1e-9, float(s_loupe) * (2 ** int(level)))
    return int(max(1, np.ceil(inset_px / eff)))


def loupe_decimation(crop_px: int, max_px: int = _LOUPE_MAX_CROP) -> int:
    """Power-of-two stride that brings a ``crop_px``-wide read down to <= ``max_px`` samples."""
    step = 1
    while crop_px // step > max(1, int(max_px)):
        step *= 2
    return step


def loupe_clamp_crop(y0: int, x0: int, h: int, w: int, ny: int, nx: int):
    """Fit a crop rect inside a ``ny`` x ``nx`` field by shifting the origin in, not the extent."""
    ny, nx = max(1, int(ny)), max(1, int(nx))
    h, w = max(1, min(int(h), ny)), max(1, min(int(w), nx))
    return max(0, min(int(y0), ny - h)), max(0, min(int(x0), nx - w)), h, w


def loupe_um_per_screen_px(pixel_size_um, s_loupe: float):
    """µm per screen pixel inside the inset, or None when the pixel size isn't trustworthy."""
    if pixel_size_um is None:
        return None
    p = float(pixel_size_um)
    if not np.isfinite(p) or p <= 0:
        return None
    return p / max(1e-9, float(s_loupe))


def _nice_scale_um(rough: float) -> float:
    """Round a scale-bar length to a 1/2/5 x 10^n figure, the way a microscope overlay would."""
    rough = max(1e-6, float(rough))
    decade = 10.0 ** np.floor(np.log10(rough))
    for step in (1.0, 2.0, 5.0, 10.0):
        if rough <= step * decade:
            return step * decade
    return 10.0 * decade


def _fmt_um(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:g} mm"
    return f"{v:g} µm" if v >= 1 else f"{v * 1000:g} nm"


_LOUPE_WIN_LOCK = threading.Lock()   # guards lazy creation of the per-source window memo


class _LoupeSource:
    """Where the loupe's real pixels come from for the layer currently on the plate.

    Availability is per (source, well), never per layer key: a layer key can point at a stale
    save while a newer unsaved preview shares the same key.
    """

    n_levels = 1
    well_px = 1
    pixel_size_um = None

    def available(self, well_id) -> tuple[bool, str]:
        """(ok, reason-if-not). ``reason`` is shown to the user verbatim."""
        return False, "no pixel source"

    def read_crop(self, well_id, level, y0, x0, h, w, time_point: int = 0, fov=None):
        """(C, y, x) crop at ``level``, clamped into the field and decimated to at most
        _LOUPE_MAX_CROP samples per side. Runs on the worker thread."""
        raise NotImplementedError

    def coarse(self, well_id, time_point: int = 0):
        """A small whole-field (C, y, x) plane used only to derive the contrast window."""
        raise NotImplementedError

    def window(self, well_id, time_point: int = 0):
        """Per-channel contrast window for a well, computed here (worker thread) and memoised."""
        key = (well_id, int(time_point))
        with _LOUPE_WIN_LOCK:
            cache = self.__dict__.get("_win_cache")
            if cache is None:
                cache = self.__dict__["_win_cache"] = MemoryBoundedLRUCache(_AUX_CACHE_BYTES)
        hit = cache.get(key)
        if hit is not None:
            return hit.value
        coarse = self.coarse(well_id, time_point)
        win = [_pct_window(coarse[c]) for c in range(coarse.shape[0])]
        cache.put(key, _SizedValue(win, _WINDOW_PAIR_NBYTES * len(win)))
        return win


class _RawLoupeSource(_LoupeSource):
    """Raw-acquisition source: the loupe works the moment a folder is open, before any operator."""

    def __init__(self, reader, meta, fov_of):
        self._reader, self._meta, self._fov_of = reader, meta, fov_of
        ny, nx = meta["frame_shape"]
        self.well_px = int(min(ny, nx))
        self.n_levels = 1
        self.pixel_size_um = meta.get("pixel_size_um")
        self._channels = [c["name"] for c in meta["channels"]]
        zs = meta["z_levels"]
        self._z = zs[len(zs) // 2]
        self._lock = threading.RLock()
        self._cache_key = None
        self._cache = None
        self._coarse = MemoryBoundedLRUCache(_AUX_CACHE_BYTES)

    def available(self, well_id) -> tuple[bool, str]:
        if well_id in self._meta["regions"]:
            return True, ""
        return False, "no image for this well"

    def _planes(self, well_id, time_point: int = 0, fov=None):
        """The field's (C, y, x) planes at ``time_point``, decoded once and cached under a lock."""
        f = self._fov_of(well_id) if fov is None else int(fov)
        key = (well_id, f, int(time_point))
        with self._lock:
            if self._cache_key != key:
                planes = np.stack([
                    np.asarray(self._reader.read(well_id, f, ch, self._z, int(time_point)))
                    for ch in self._channels])
                self._cache, self._cache_key = planes, key
            return self._cache

    def read_crop(self, well_id, level, y0, x0, h, w, time_point: int = 0, fov=None):
        p = self._planes(well_id, time_point, fov)
        ny, nx = p.shape[-2], p.shape[-1]
        y0, x0, h, w = loupe_clamp_crop(y0, x0, h, w, ny, nx)
        crop = p[:, y0:y0 + h, x0:x0 + w]
        step = loupe_decimation(max(h, w))
        if step == 1:
            return crop
        oh, ow = max(1, h // step), max(1, w // step)
        return np.stack([_area_downsample(crop[c], oh, ow).astype(np.float32, copy=False)
                         for c in range(crop.shape[0])])

    def coarse(self, well_id, time_point: int = 0):
        # First field, not the one under the cursor: the window is per well, not per FOV.
        key = (well_id, int(time_point))
        hit = self._coarse.get(key)
        if hit is None:
            p = self._planes(well_id, time_point)
            hit = np.stack([_area_downsample(p[c], _CELL, _CELL) for c in range(p.shape[0])])
            self._coarse.put(key, hit)
        return hit


class _ZarrLoupeSource(_LoupeSource):
    """Written-plate source: a windowed tensorstore read of one pyramid level.

    ``written`` is the set of wells this run has actually persisted, so the loupe works on
    completed wells during a long run rather than magnifying an unfinished well's pixels.
    """

    def __init__(self, base, path_of, fov_of, levels, well_px, pixel_size_um, written=None):
        self._base = str(base)
        self._path_of, self._fov_of = path_of, fov_of
        self._levels = list(levels) if levels is not None else None
        self.n_levels = max(1, len(self._levels)) if self._levels else 1
        self.well_px = int(well_px)
        self.pixel_size_um = pixel_size_um
        self._written = written
        self._coarse = MemoryBoundedLRUCache(_AUX_CACHE_BYTES)   # internally locked
        # Guards the _levels/n_levels publish below: two loupe workers can be alive at once
        # (see _detach) and must not race that publish.
        self._lock = threading.RLock()

    def mark_written(self, well_id):
        if self._written is not None:
            self._written.add(well_id)

    def available(self, well_id) -> tuple[bool, str]:
        if self._written is not None and well_id not in self._written:
            return False, "not written yet"
        if self._path_of(well_id) is None:
            return False, "no image for this well"
        return True, ""

    def _resolve_levels(self, well_id):
        """Read the field's multiscales once, to learn how many pyramid levels exist."""
        with self._lock:
            if self._levels is not None:
                return self._levels
            self._levels = field_levels(
                field_path(self._base, self._path_of(well_id), self._fov_of(well_id)))
            self.n_levels = max(1, len(self._levels))
            return self._levels

    def _open(self, well_id, level, fov=None):
        levels = self._resolve_levels(well_id)
        level = max(0, min(int(level), len(levels) - 1))
        f = self._fov_of(well_id) if fov is None else int(fov)
        from squidxplorer._tsctx import HANDLES

        return HANDLES.get(field_path(self._base, self._path_of(well_id), f, levels[level]),
                           open_only=True)

    def read_crop(self, well_id, level, y0, x0, h, w, time_point: int = 0, fov=None):
        arr = self._open(well_id, level, fov)
        ny, nx = arr.shape[-2], arr.shape[-1]
        y0, x0, h, w = loupe_clamp_crop(y0, x0, h, w, ny, nx)
        step = loupe_decimation(max(h, w))
        t_idx = max(0, min(int(time_point), arr.shape[0] - 1))
        return np.asarray(
            arr[t_idx, :, 0, y0:y0 + h:step, x0:x0 + w:step].read().result())

    def coarse(self, well_id, time_point: int = 0):
        key = (well_id, int(time_point))
        hit = self._coarse.get(key)
        if hit is not None:
            return hit
        # Read outside the cache's lock: it is a whole coarse plane, and only the store must
        # not race.
        arr = self._open(well_id, self.n_levels - 1)
        t_idx = max(0, min(int(time_point), arr.shape[0] - 1))
        plane = np.asarray(arr[t_idx, :, 0].read().result())
        prior = self._coarse.get(key)     # setdefault semantics: a racing worker's plane wins
        if prior is not None:
            return prior
        self._coarse.put(key, plane)
        return plane


_DETACHED: "set" = set()


def _detach(worker) -> None:
    """Cut a still-running worker loose instead of destroying it.

    A QThread whose C++ half is destroyed while ``isRunning()`` aborts the process (qFatal, no
    traceback). A worker that will not stop in time is reparented to nobody and kept referenced
    until it finishes on its own, at which point it removes itself — the cost of a straggler is
    one idle thread, the cost of the alternative is SIGABRT.
    """
    if worker is None:
        return
    try:
        if not worker.isRunning():
            return
        worker.setParent(None)
    except RuntimeError:
        return
    _DETACHED.add(worker)
    log.warning("%s did not stop in time; detached rather than destroyed (it is still reading)",
                type(worker).__name__)
    worker.finished.connect(lambda w=worker: _DETACHED.discard(w))


class _LoupeWorker(QThread):
    """Serves loupe crops off the GUI thread, coalescing to the newest request.

    One pending slot, overwritten by each new request, is the coalescing. Results carry the
    generation they were asked for, so a late arrival for a stale position is dropped rather
    than flashing.
    """

    ready = Signal(int, str, object, object, object)  # (gen, well, crop|None, window|None, err)

    def __init__(self, source: _LoupeSource):
        super().__init__()
        self._source = source
        self._cv = threading.Condition()
        self._pending = None
        self._stop = False
        self._cache = MemoryBoundedLRUCache(_AUX_CACHE_BYTES)

    def request(self, gen, well_id, level, y0, x0, h, w, time_point: int = 0, fov=None):
        with self._cv:
            self._pending = (gen, well_id, level, y0, x0, h, w, int(time_point),
                             None if fov is None else int(fov))
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                gen, well_id, level, y0, x0, h, w, time_point, fov = self._pending
                self._pending = None
            # The LRU key carries the timepoint and FOV, else a stale crop answers for a
            # different frame or field at the same rectangle.
            key = (well_id, fov, level, y0, x0, h, w, time_point)
            try:
                crop = self._cache.get(key)
                if crop is None:
                    crop = self._source.read_crop(well_id, level, y0, x0, h, w, time_point,
                                                  fov=fov)
                    self._cache.put(key, crop)
                try:
                    win = self._source.window(well_id, time_point)
                except Exception:
                    win = None
                self.ready.emit(gen, well_id, crop, win, None)
            except Exception as e:
                self.ready.emit(gen, well_id, None, None, f"{type(e).__name__}: {e}")


def _deep_zoom_enabled() -> bool:
    return os.environ.get("SQUIDXPLORER_DEEP_ZOOM", "1") != "0"


_TILE_QUEUE_MAX = 24


class _TileFetcher(QThread):
    """Decode tiles off the GUI thread and hand them back one at a time.

    Newest-first (LIFO): a pan generates requests faster than they can be served, and the tiles
    worth decoding are the ones under the cursor now, not the ones passed over a moment ago.
    """

    ready = Signal(object, object)        # (TileDescriptor, np.ndarray)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self._source = source
        self._pending: list = []              # treated as a stack
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()

    def request(self, descs) -> None:
        """Queue *descs*, newest last. Already-queued descriptors are not duplicated."""
        with self._lock:
            have = set(self._pending)
            for d in descs:
                if d not in have:
                    self._pending.append(d)
            del self._pending[:-_TILE_QUEUE_MAX]     # drop the stalest, keep the cap honest
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                desc = self._pending.pop() if self._pending else None
            if desc is None:
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            try:
                arr = self._source.read_tile(desc)
            except Exception as exc:
                log.warning("tile %s/%s failed to decode: %s: %s",
                            desc.level, desc.key, type(exc).__name__, exc)
                continue
            if not self._stop.is_set():
                self.ready.emit(desc, arr)


class PlateOverview(QWidget):
    """The low-res plate: an RGB canvas of MIP tiles, a per-well status hue, a red box, and a
    press-and-hold loupe that overlays real acquisition pixels for the well under the cursor.

    The RGB canvas is only what is currently shown. What the widget actually owns is a per-layer
    ``(C, nr*_CELL, nc*_CELL)`` native-dtype store, plus a channel mask and per-channel contrast
    window, so toggling a channel or dragging its contrast re-composites from retained pixels —
    no reader I/O, no re-projection.
    """

    hovered = Signal(str)
    wellActivated = Signal(str, int)
    selectionChanged = Signal(list)
    # Shift-drag specifically: opens an exploration tab. Shift-click refines the selection one
    # well at a time and deliberately does not fire this, or every corrective click spawns a tab.
    marqueeSelected = Signal(list)
    activeLayerChanged = Signal(str)

    def __init__(self, rows, cols, wells: dict, layout: Optional[dict] = None):
        """``wells``: (row_index, col_index) -> well_id for every acquired well.

        ``layout`` is ``{(row_index, col_index): (x, y, w, h)}`` in grid units for a holder whose
        cells are placed by real geometry rather than a uniform pitch (a freeform tissue slide);
        ``None`` is the uniform grid a well plate is. Cells absent from the map fall back to
        their nominal ``(c, r, 1, 1)`` square.
        """
        super().__init__()
        self._rows, self._cols = list(rows), list(cols)
        self._layout: Optional[dict] = ({tuple(k): tuple(float(v) for v in val)
                                         for k, val in layout.items()} if layout else None)
        self._nr, self._nc = len(self._rows), len(self._cols)
        self._by_rc: dict[tuple, str] = dict(wells)
        self._status: dict[tuple, str] = {rc: "empty" for rc in wells}
        self._tiles: set[tuple] = set()
        self._tiles_by_layer: dict[str, set] = {}
        self._canvas = QImage(self._nc * _CELL, self._nr * _CELL, QImage.Format_RGB888)
        self._canvas.fill(QColor(_BG))
        self._final = None
        # Layer stack: base ("raw") is self._canvas; each operator draws into its own
        # canvas/final. self._active is the layer currently shown (set_active_layer).
        self._op_canvas: dict[str, QImage] = {}
        self._op_final: dict[str, QImage] = {}
        self._final_arr: dict[str, np.ndarray] = {}   # keeps each RGB alive: QImage wraps it, no copy
        self._active = "raw"
        self._labels: list[str] = []
        self._colors = None
        self._dtype = np.uint16
        self._store: dict[str, np.ndarray] = {}
        # A contrast window is a point transform, so it commutes with subsampling: only the
        # display-sized composite needs re-deriving per tick. These two caches hold everything
        # upstream of that, so a drag re-reads no store and re-percentiles nothing.
        self._disp: dict[str, tuple] = {}
        self._cell_auto: dict[str, dict] = {}
        self._mask = None
        self._contrast = None
        self._full = QTimer(self)
        self._full.setSingleShot(True)
        self._full.timeout.connect(self._on_full_timeout)
        self._scaled = None
        self._scaled_cd = None
        self._scaled_base = None
        self._scaled_base_key = None
        self._base_gen = 0   # the base canvas is painted in place, so only a counter can say it moved
        self._cd = float(_CELL)
        self._ox = self._oy = _PAD
        self._hover = None
        self._sel = None
        # _sel is the one well the detail viewer shows (red box); _selection is the set the
        # operator picked (blue box). Never merge them: each must survive the other's gesture.
        self._selection: set = set()
        self._view_hues: list = []
        self._marquee = None
        self._marquee_add = False
        self._ctrl_click = None
        self._press = None
        self._panning = False
        self._user_view = False
        self._boxes: dict = {}
        self._boxed_regions: set = set()
        self._fov_selection: dict = {}
        self._ladder = None
        self._tile_src = None
        self._tile_cache = None
        self._tile_fetch = None
        self._tile_level = None
        self._carrier = None
        self._carrier_slide = False
        self._slides = None
        self._tile_rgn = None
        self._loupe_src = None
        self._loupe_worker = None
        self._loupe = None
        self._loupe_gen = 0
        self._loupe_img = None
        self._loupe_note = ""
        self._loupe_win = {}
        self._loupe_colors = None
        self._time_point = 0
        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.setInterval(_LOUPE_HOLD_MS)
        self._hold.timeout.connect(self._arm_loupe)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMinimumSize(240, 200)

    def set_tile_source(self, reader, meta) -> bool:
        """Arm deep zoom for this acquisition. Returns whether it armed.

        Fail-quiet: an acquisition with no usable stage positions cannot be placed in world µm,
        so the honest response is to keep the montage rather than draw FOVs at one spot.
        """
        self.clear_tile_source()
        if not _deep_zoom_enabled() or reader is None or not meta:
            return False
        try:
            from squidxplorer._platecache import PlateCellCache
            from squidxplorer._tiling import TileCache
            from squidxplorer._tilesource import CompositePlateSource, plate_ladder

            self._ladder = plate_ladder(meta)
            # Composite, not a bare ReaderTileSource: plate rungs come from the persisted preview
            # cells (cheap), FOV rungs from the reader. time_point must match on both sides or
            # CompositePlateSource refuses the pair.
            self._tile_src = CompositePlateSource(
                reader, meta, self._ladder, time_point=self._time_point,
                cache=PlateCellCache.for_reader(reader, meta, cell_px=_CELL,
                                                time_point=self._time_point))
        except Exception:
            self._ladder = self._tile_src = None
            return False
        self._tile_cache = TileCache(budget_bytes=_TILE_CACHE_BYTES)
        # Unparented on purpose: a QThread parented to this widget is deleted by Qt on widget
        # destruction whether or not it is running, which aborts. See _detach.
        self._tile_fetch = _TileFetcher(self._tile_src)
        self._tile_fetch.ready.connect(self._on_tile_ready)
        self._tile_fetch.start()
        return True

    def clear_tile_source(self) -> None:
        """Stop and forget the tile machinery. Idempotent; safe on a half-built state."""
        if self._tile_fetch is not None:
            self._tile_fetch.stop()
            if not self._tile_fetch.wait(1500):
                _detach(self._tile_fetch)
            self._tile_fetch = None
        self._ladder = self._tile_src = self._tile_cache = None
        self._tile_level = None

    def shutdown(self) -> None:
        """Stop both threads this widget owns. Idempotent; the one call a destroyer must make."""
        self.clear_tile_source()
        self.set_loupe_source(None)

    def _on_tile_ready(self, desc, arr) -> None:
        if self._tile_cache is None:
            return
        self._tile_cache.insert(desc, arr)
        self.update()

    def _visible_fov_tiles(self) -> list:
        """``[(TileDescriptor, QRectF), ...]`` for the FOVs on screen, or ``[]`` to stay coarse.

        Engages only above native cell size (``cd > _CELL``): below it the montage already shows
        the right image and serving tiles would cost a full-plate decode. Tiles are placed with
        ``_placement.cell_boxes`` at the current cell size, the same call the montage uses, so the
        overlay lands pixel-aligned on the thumbnail underneath.
        """
        if (self._tile_src is None or self._ladder is None or self._cd <= _CELL
                or self._layout is not None):
            return []
        meta = getattr(self._tile_src, "meta", {})
        positions = meta.get("fov_positions_um") or {}
        px = meta.get("pixel_size_um")
        frame = meta.get("frame_shape")
        if not positions or not px or frame is None:
            return []

        from squidxplorer._placement import cell_boxes, fov_offsets_px

        cd = int(round(self._cd))
        vis = self.rect().adjusted(-cd, -cd, cd, cd)         # one cell of slack, so panning
        out: list = []                                       # pre-warms the edge
        for rc, region in self._by_rc.items():
            x0, y0, cw, chh = self._cell_rect(*rc)
            if not vis.intersects(QRectF(x0, y0, cw, chh).toRect()):
                continue
            fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
            if not fovs:
                continue
            try:
                boxes = cell_boxes(fov_offsets_px(positions, region, fovs, px), frame, cd)
            except (KeyError, ValueError):
                continue                 # this region has no derivable mosaic; montage stands
            for fov, (top, left, bh, bw) in boxes.items():
                key = (region, fov)
                if key not in self._ladder.fov_bboxes or bh < 2 or bw < 2:
                    continue
                rect = QRectF(x0 + left, y0 + top, bw, bh)
                if not vis.intersects(rect.toRect()):
                    continue
                # Rung chosen from what this FOV occupies ON SCREEN, not the plate zoom: a
                # letterboxed mosaic gives each FOV a fraction of the cell.
                um_per_px = (float(frame[1]) * float(px)) / max(float(bw), 1.0)
                lvl = self._ladder.geometry.pick_level(um_per_px, self._tile_level)
                if not self._ladder.is_fov_level(lvl):
                    continue
                self._tile_level = lvl
                out.append((TileDescriptor(level=lvl, key=key, channel=self._tile_channel(),
                                           bbox_um=self._ladder.fov_bboxes[key],
                                           time_point=self._time_point), rect))
        return out

    def _tile_channel(self) -> str:
        """Which channel the overlay reads. One channel for now — the montage stays the composite."""
        chans = (getattr(self._tile_src, "meta", {}) or {}).get("channels") or []
        return str(chans[0]["name"]) if chans else "0"

    def _paint_tiles(self, p) -> None:
        """Draw whatever the cache can serve, and queue the rest. Never blocks on a decode."""
        wanted = self._visible_fov_tiles()
        if not wanted:
            return
        by_desc = {d: r for d, r in wanted}
        for desc, arr in self._tile_cache.resolve(list(by_desc)):
            rect = by_desc.get(desc)
            if rect is None:
                continue                 # resolve() substituted an ancestor we did not ask to place
            img = self._tile_qimage(desc, arr)
            if img is not None:
                p.drawImage(rect, img)
        missing = [d for d in by_desc if self._tile_cache.get(d) is None]
        if missing and self._tile_fetch is not None:
            self._tile_fetch.request(missing)

    def _tile_qimage(self, desc, arr):
        """One cached tile as an 8-bit greyscale QImage, windowed by the plate's own contrast."""
        cache = self.__dict__.get("_tile_qimages")
        if cache is None:
            cache = self.__dict__["_tile_qimages"] = MemoryBoundedLRUCache(_AUX_CACHE_BYTES)
        hit = cache.get(desc)
        if hit is not None:
            return hit.value
        lo, hi = self._tile_window()
        a = np.clip((arr.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255)
        a = np.ascontiguousarray(a.astype(np.uint8))
        img = QImage(a.data, a.shape[1], a.shape[0], a.shape[1], QImage.Format_Grayscale8).copy()
        cache.put(desc, _SizedValue(img, img.sizeInBytes()))
        return img

    def _tile_window(self) -> tuple:
        """The active channel's display window, from the plate's running contrast when it has one."""
        c = self._contrast
        if c is not None:
            try:
                lo, hi = c.window(0)
                if hi > lo:
                    return float(lo), float(hi)
            except Exception:
                pass
        return 0.0, 65535.0

    def set_loupe_source(self, source, colors=None):
        """Point the loupe at the data behind the active layer. ``None`` disables the gesture."""
        self._dismiss_loupe()
        if self._loupe_worker is not None:
            self._loupe_worker.stop()
            if not self._loupe_worker.wait(2000):
                _detach(self._loupe_worker)
            self._loupe_worker = None
        self._loupe_src = source
        self._loupe_colors = colors
        self._loupe_win.clear()
        if source is not None:
            self._loupe_worker = _LoupeWorker(source)
            self._loupe_worker.ready.connect(self._on_loupe_crop)
            self._loupe_worker.start()

    def set_time_point(self, time_point: int):
        """Tell the plate which timepoint it is showing, so the loupe reads the same frame.

        Deep zoom needs nothing rebuilt here: the timepoint is part of TileDescriptor, so
        _visible_fov_tiles stamps it on every tile request and every source reads the frame off
        the request.
        """
        tp = max(0, int(time_point))
        if tp == self._time_point:
            return
        self._time_point = tp
        self._loupe_win.clear()
        if self._loupe is not None:          # a live inset re-reads at the new frame
            self._request_loupe(self._loupe["x"], self._loupe["y"])
        if self._tile_src is not None:
            self.update()                    # repaint -> new descriptors -> the new frame's tiles

    def _arm_loupe(self):
        """Hold timer fired: the press became a loupe. Only reachable while still armed."""
        self._hold.stop()   # a pending fire must never re-arm and blank a live loupe
        if self._press is None or self._panning:
            return
        x, y = self._press[0], self._press[1]
        c = self._cell(x, y)
        if not c or not c["well_id"]:
            return
        self._loupe = {"well": c["well_id"], "x": x, "y": y}
        self._loupe_img, self._loupe_note = None, ""
        self._request_loupe(x, y)
        self.update()

    def _dismiss_loupe(self):
        if self._loupe is not None or self._loupe_img is not None or self._loupe_note:
            self._loupe = self._loupe_img = None
            self._loupe_note = ""
            self.update()

    def _loupe_geometry(self, x, y):
        """Map a widget point to (well_id, level, crop rect, s_loupe, M) — or None if off-plate."""
        src = self._loupe_src
        c = self._cell(x, y)
        if src is None or not c or not c["well_id"]:
            return None
        s_loupe, mag = loupe_scale(self._cd, src.well_px)
        level = loupe_level(s_loupe, src.n_levels)
        crop = loupe_crop_px(s_loupe, level)
        tgt = self._loupe_target(c["row_index"], c["col_index"], c["well_id"], x, y)
        if tgt is None:
            return None
        fov, fx, fy = tgt
        span = max(1, src.well_px >> level)
        cy, cx = int(fy * span), int(fx * span)
        # Clamp here too: a hold near a field edge then produces the same key as the cursor
        # drifts, so the LRU hits instead of decoding a fresh crop per pixel of motion.
        y0, x0, h, w = loupe_clamp_crop(cy - crop // 2, cx - crop // 2, crop, crop, span, span)
        return c["well_id"], fov, level, (y0, x0, h, w), s_loupe, mag

    def _loupe_target(self, ri: int, ci: int, region, x, y) -> Optional[tuple]:
        """``(fov, fx, fy)``: which field the cursor is over, and where in it, 0..1.

        ``fov`` is ``None`` when the cell holds no mosaic (one field fills its block), and the
        fraction is then across the cell's whole content, the single-FOV path.
        """
        pt = self._cell_point(ri, ci, x, y)
        if pt is None:
            return None
        bx, by = pt
        hit = self._fov_box_at(region, bx, by)
        if hit is None:
            frac = self._cell_fraction(ri, ci, x, y)
            return None if frac is None else (None, frac[0], frac[1])
        fov, (top, left, bh, bw) = hit
        return fov, (bx - left) / max(float(bw), 1e-9), (by - top) / max(float(bh), 1e-9)

    def _fov_box_at(self, region, bx, by) -> Optional[tuple]:
        """``(fov, (top, left, h, w))`` for the mosaic box under block point ``(bx, by)``.

        Boxes overlap by ~9% at the seams, so the last match wins, matching the draw order that
        paints later FOVs over earlier ones.
        """
        if not region or not self._boxes:
            return None
        hit = None
        for (r, fov), (top, left, h, w) in self._boxes.items():
            if r == region and top <= by < top + h and left <= bx < left + w:
                hit = (fov, (top, left, h, w))
        return hit

    def _request_loupe(self, x, y):
        geo = self._loupe_geometry(x, y)
        if geo is None:                    # dragged onto the margin / an un-acquired cell
            if self._loupe_img is not None or not self._loupe_note:
                self._loupe_img, self._loupe_note = None, "no well here"
                self.update()
            return
        if self._loupe_worker is None:
            return
        well, fov, level, (y0, x0, h, w), _s, _m = geo
        ok, why = self._loupe_src.available(well)
        if not ok:
            self._loupe_img, self._loupe_note = None, why
            self.update()
            return
        self._loupe_gen += 1
        self._loupe_worker.request(self._loupe_gen, well, level, y0, x0, h, w,
                                   self._time_point, fov)

    def _on_loupe_crop(self, gen, well_id, crop, window, error):
        """A crop landed. Drop it unless it is the newest request and the loupe is still up."""
        if gen != self._loupe_gen or self._loupe is None:
            return
        if error is not None or crop is None or crop.size == 0:
            self._loupe_img, self._loupe_note = None, error or "no pixels here"
            self.update()
            return
        win, colors = self._loupe_lut(well_id, crop.shape[0], window)
        # The same compositor the plate uses, with the same channel mask: unticking a channel must
        # remove it from the inset as well, or the loupe contradicts the plate it sits on top of.
        planes = np.stack([crop[c].astype(np.float32) for c in range(crop.shape[0])])
        mask = self._mask if (self._mask is not None
                              and len(self._mask) == crop.shape[0]) else None
        rgb = composite(planes, colors, win, mask)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()   # copy: rgb is transient
        self._loupe_img, self._loupe_note = img, ""
        self.update()

    def _loupe_lut(self, well_id, n_ch: int, window):
        """``(windows, colours)`` the inset paints with: the plate's own, whenever it has them.

        The loupe is a magnifier of the plate, so the only defensible window and colour are the
        ones the plate is already painting with. The source-derived window survives only as the
        fallback for a plate with no contrast model at all.
        """
        win = self.channel_windows()
        if len(win) != n_ch:
            # No plate contrast model (or a source whose channel count is not the plate's): fall
            # back to the source's own window, still resolved through the one precedence rule.
            auto = window if window is not None else self._loupe_win.get(well_id)
            if auto is None:
                auto = [(0.0, 1.0)] * n_ch
            self._loupe_win[well_id] = auto          # memo the AUTO window, never the resolved one
            win = ([self._contrast.resolve(c, auto[c]) for c in range(len(auto))]
                   if self._contrast is not None else list(auto))
        colors = self._colors
        if colors is None or len(colors) != n_ch:
            colors = self._loupe_colors
        if colors is None or len(colors) != n_ch:
            colors = np.ones((n_ch, 3), np.float32)
        return win, colors

    def _fit(self):
        """Reset the view: the whole plate fits the widget, centered (zoom = 1)."""
        if self._nr == 0 or self._nc == 0:
            return
        w, h = self.width(), self.height()
        self._cd = max(2.0, min((w - _HDR - 2 * _PAD) / self._nc, (h - _COLH - 2 * _PAD) / self._nr))
        self._ox = max(_PAD, (w - _HDR - self._nc * self._cd) / 2)
        self._oy = max(_PAD, (h - _COLH - self._nr * self._cd) / 2)

    def _fit_cd(self) -> float:
        w, h = self.width(), self.height()
        return max(2.0, min((w - _HDR - 2 * _PAD) / self._nc, (h - _COLH - 2 * _PAD) / self._nr))

    def _canvas_for(self, layer: str) -> QImage:
        if layer == "raw":
            return self._canvas
        cv = self._op_canvas.get(layer)
        if cv is None:
            cv = QImage(self._nc * _CELL, self._nr * _CELL, QImage.Format_RGB888)
            cv.fill(QColor(_BG))
            self._op_canvas[layer] = cv
        return cv

    def _active_source(self) -> QImage:
        return self._final or self._canvas_for(self._active)

    def _base_source(self) -> QImage:
        """The base ("raw") layer's montage: the picture showing through wherever the active
        layer has nothing (see ``underlay_cells``)."""
        return self._op_final.get("raw") or self._canvas

    # A layer sits OVER the base, not in place of it: a cell the active layer has no pixels for
    # shows the base layer's cell instead of going blank.

    def underlay_cells(self) -> set:
        """Cells the active layer has no pixels for, but the base layer does."""
        if self._active == "raw":
            return set()
        return (self._tiles_by_layer.get("raw", set())
                - self._tiles_by_layer.get(self._active, set()))

    def shown_cells(self) -> set:
        """Every ``(ri, ci)`` showing a thumbnail right now, from whichever layer supplies it."""
        return set(self._tiles_by_layer.get(self._active, set())) | self.underlay_cells()

    def set_channels(self, labels, colors: np.ndarray, dtype=np.uint16, pct=(1.0, 99.8)):
        """Declare the acquisition's channels — the per-channel store/mask/contrast start here."""
        self._labels = [str(x) for x in labels]
        self._colors = np.asarray(colors, dtype=np.float32)
        self._dtype = np.dtype(dtype)
        self._mask = np.ones(len(self._labels), dtype=bool)     # every channel on by default (OV8)
        dmax = float(np.iinfo(self._dtype).max) if self._dtype.kind in "ui" else 1.0
        self._contrast = _RunningContrast(len(self._labels), dmax, pct=pct)
        self._store.clear()

    def _store_for(self, layer: str) -> Optional[np.ndarray]:
        """The layer's (C, H, W) plate store, allocated lazily on first tile."""
        if self._colors is None:
            return None
        st = self._store.get(layer)
        if st is None:
            st = np.zeros((len(self._colors), self._nr * _CELL, self._nc * _CELL), self._dtype)
            self._store[layer] = st
        return st

    def channel_windows(self) -> list:
        """The effective (lo, hi) per channel — the latched manual window, else the running one."""
        return self._contrast.windows() if self._contrast is not None else []

    def _invalidate_pixels(self, layer: str, rc=None):
        """The layer's STORE changed. Drop everything derived from its pixels.

        Called from the one place that writes pixels (``add_tile``) and from ``reset_layer``.
        These caches are keyed on PIXELS alone, so a contrast change must never come through
        here — keeping the two invalidations apart is what lets a drag re-read nothing.

        *rc* narrows the per-cell percentile drop to the one cell that was written; the display
        thumbnail is a whole-plate copy, so it always goes.
        """
        self._disp.pop(layer, None)
        if rc is None:
            self._cell_auto.pop(layer, None)
        else:
            self._cell_auto.get(layer, {}).pop(rc, None)

    def _disp_store(self, layer: str, store: np.ndarray, step: int) -> np.ndarray:
        """A contiguous (C, h, w) thumbnail of *store* at subsampling *step*, cached.

        Contiguity is not a detail: a strided view would make both the LUT and the BLAS reduce
        in ``composite`` silently materialise their own copy on every tick.
        """
        if step <= 1:
            return store
        hit = self._disp.get(layer)
        if hit is not None and hit[0] == step:
            return hit[1]
        thumb = np.ascontiguousarray(store[:, ::step, ::step])
        self._disp[layer] = (step, thumb)
        return thumb

    # Per-region contrast is gone deliberately: it made a dim well readable next to a bright one
    # at the cost of comparability, which is what a plate view is for. One window per channel,
    # owned by napari, and the plate follows it.

    def set_channel_color(self, ch: int, rgb) -> bool:
        """Re-tint one channel to the colour the centre viewer is using, and repaint."""
        if self._colors is None or not (0 <= ch < len(self._colors)):
            return False
        new_rgb = np.asarray(rgb, dtype=np.float32)
        if np.allclose(self._colors[ch], new_rgb):
            return False
        self._colors[ch] = new_rgb
        self._refresh()
        return True

    def channel_rgb(self, ch: int):
        """The (r, g, b) this channel is composited with, or None — the reader beside
        :meth:`set_channel_color`, so the LUT clipboard can carry the plate's colour out."""
        if self._colors is None or not (0 <= ch < len(self._colors)):
            return None
        return tuple(float(v) for v in self._colors[ch])

    def channel_visible(self, ch: int):
        """Whether this channel is in the composite, or None — the reader beside
        :meth:`set_channel_visible`."""
        if self._mask is None or not (0 <= ch < len(self._mask)):
            return None
        return bool(self._mask[ch])

    def set_channel_visible(self, ch: int, on: bool):
        """Toggle a channel in/out of the plate composite. Recomposites from the retained store.

        The last channel cannot be turned off: the plate is a navigator with no controls of its
        own, so a black grid with every channel masked off has no way to be filled back in.
        """
        if self._mask is None or not (0 <= ch < len(self._mask)):
            return
        on = bool(on)
        if not on and not any(bool(v) for i, v in enumerate(self._mask) if i != ch):
            log.debug("plate: keeping channel %d on — it is the last one lit, and a plate with "
                      "no channels is a black navigator", ch)
            return
        self._mask[ch] = on
        self._refresh()

    def set_channel_window(self, ch: int, lo: float, hi: float):
        """Re-window one channel and repaint. LATCHES that channel manual (D4) so the wells still
        streaming in can't stomp the window the user just set."""
        if self._contrast is None or not (0 <= ch < len(self._mask)):
            return
        self._contrast.set_manual(ch, lo, hi)
        self._refresh()

    def follow_channel_window(self, ch: int, lo: float, hi: float):
        """Render *ch* with the window the owning array viewer resolved, and repaint.

        Does not latch the channel manual: the owning viewer autoscales on its own, so recording
        that as a user gesture would kill the plate's own auto-contrast. See
        ``_RunningContrast.set_followed``.
        """
        if self._contrast is None or not (0 <= ch < len(self._mask)):
            return
        self._contrast.set_followed(ch, lo, hi)
        self._refresh()

    def set_channel_auto(self, ch: int):
        """Unlatch a channel: it goes back to auto-scaling off the running histogram."""
        if self._contrast is None or not (0 <= ch < len(self._mask)):
            return
        self._contrast.set_auto(ch)
        self._refresh()

    def _refresh(self):
        """A user gesture: repaint now at display resolution, then land full-res once it settles."""
        self.recomposite(quick=True)
        self._full.start(150)
        self._refresh_loupe()

    def _refresh_loupe(self):
        """Re-render the loupe inset under the contrast that just changed.

        Re-issuing the request is cheap: the worker memoises crops, so this re-colours bytes it
        already has and re-reads nothing.
        """
        if self._loupe is None or self._loupe_worker is None:
            return
        try:
            self._request_loupe(self._loupe["x"], self._loupe["y"])
        except Exception:
            pass

    def _on_full_timeout(self):
        """The coalescing timer fired. Guarded: the plate may have been torn down while queued."""
        try:
            self.recomposite(quick=False)
        except RuntimeError:
            pass   # the C++ widget went away while the full-res pass was still pending

    def recomposite(self, layer: Optional[str] = None, *, quick: bool = False):
        """Rebuild a layer's plate image from its store, at the current mask + windows.

        ``quick=True`` composites a strided view at roughly the on-screen resolution (cheap enough
        to run on every slider tick); the default walks the whole store — the end-of-stream pass.
        """
        layer = layer or self._active
        store = self._store.get(layer)
        if store is None or self._colors is None:
            return
        step = max(1, int(round(_CELL / max(1.0, self._cd)))) if quick else 1
        rgb = composite(self._disp_store(layer, store, step), self._colors,
                        self.channel_windows(), self._mask)
        self._final_arr[layer] = rgb          # hold the buffer: the QImage below only wraps it
        h, w, _ = rgb.shape
        self.set_final(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888), layer)

    def hideEvent(self, e):
        self._full.stop()   # a plate on its way out must not repaint from a queued timer
        super().hideEvent(e)

    def reset_layer(self, layer: str):
        """Forget a layer's retained pixels — its store, its recomposite and its painted canvas."""
        self._store.pop(layer, None)
        self._invalidate_pixels(layer)
        self._final_arr.pop(layer, None)
        self._op_final.pop(layer, None)
        self._tiles_by_layer.pop(layer, None)
        if layer == "raw":
            self._canvas.fill(QColor(_BG))
            self._base_gen += 1        # the underlay's pixels moved; its scaled cache is stale
        else:
            self._op_canvas.pop(layer, None)
        if layer == self._active:
            self._final = None
            self._scaled = None
            self.update()

    # -- data in --
    def add_tile(self, ri: int, ci: int, well_id: str, tile: np.ndarray, layer: str = "raw",
                 box=None):
        """Take one per-channel tile ``(C, h, w)``, retain it in the layer's store, feed the
        running contrast, and re-composite that whole cell.

        ``box`` is ``(top, left, h, w)`` in cell px for a multi-FOV mosaic; ``box=None`` is the
        single-tile path (one field fills the cell at (0, 0)).

        Contrast is fed the tile, never the store slice: a mosaic cell is zero-padded wherever no
        FOV lands, and those zeros would pin the 1st percentile at 0 for the whole plate.
        """
        if (ri, ci) not in self._by_rc:
            return
        store = self._store_for(layer)
        if store is None:
            return
        tile = np.asarray(tile)
        y0, x0 = ri * _CELL, ci * _CELL
        top, left = (int(box[0]), int(box[1])) if box is not None else (0, 0)
        th, tw = tile.shape[1], tile.shape[2]
        store[:, y0 + top:y0 + top + th,
              x0 + left:x0 + left + tw] = tile
        self._invalidate_pixels(layer, (ri, ci))
        for c_i in range(tile.shape[0]):
            self._contrast.add(c_i, tile[c_i])
        wins = self.channel_windows()
        cell = composite(store[:, y0:y0 + _CELL, x0:x0 + _CELL], self._colors, wins, self._mask)
        img = QImage(cell.data, _CELL, _CELL, 3 * _CELL, QImage.Format_RGB888)
        p = QPainter(self._canvas_for(layer))
        p.drawImage(x0, y0, img)   # drawImage copies, so `cell` may die after p.end()
        p.end()
        if self._op_final.pop(layer, None) is not None:
            self._final_arr.pop(layer, None)
            if layer == self._active:
                self._final = None
        self._tiles.add((ri, ci))
        self._tiles_by_layer.setdefault(layer, set()).add((ri, ci))
        if layer == "raw":
            self._base_gen += 1
        if layer == self._active:
            self._scaled = None
            self.update()
        elif layer == "raw":
            self.update()

    def set_status(self, ri: int, ci: int, state: str):
        if (ri, ci) not in self._status:   # never let a foreign/stale key leak into the status map
            return
        self._status[(ri, ci)] = state
        self.update()

    def set_all_status(self, state: str):
        for rc in self._status:
            self._status[rc] = state
        self.update()

    def set_final(self, img: QImage, layer: str = "raw"):
        self._op_final[layer] = img
        if layer == "raw":
            self._base_gen += 1
        if layer == self._active:
            self._final = img
            self._scaled = None
            self.update()
        elif layer == "raw":
            self.update()

    def set_active_layer(self, layer: str):
        """Show a layer. Swaps in its montage + streamed canvas, and announces the change since
        ``_active`` is what the loupe's source is chosen by."""
        self._active = layer
        self._final = self._op_final.get(layer)
        self._scaled = None
        self.update()
        if layer in self._store:
            self.recomposite(layer)
        self.activeLayerChanged.emit(str(layer))

    def drop_layer(self, layer: str):
        """Forget a layer entirely and free its canvas and per-channel store. Falls back to the
        base layer if the dropped one was showing."""
        if layer == "raw":
            return
        self._op_canvas.pop(layer, None)
        self._op_final.pop(layer, None)
        self._store.pop(layer, None)
        self._invalidate_pixels(layer)
        self._final_arr.pop(layer, None)
        self._tiles_by_layer.pop(layer, None)
        self._tiles = set().union(*self._tiles_by_layer.values()) if self._tiles_by_layer else set()
        if self._active == layer:
            self.set_active_layer("raw")
        else:
            self.update()

    def select(self, ri: int, ci: int):
        """Move the red box to a well (driven by the ndviewer FOV slider)."""
        self._sel = (ri, ci)
        self.update()

    def resizeEvent(self, e):
        self._user_view = False       # a resize re-fits (drop any zoom/pan)
        self._fit()
        self.update()

    # Mouse: wheel-zoom anchored at cursor, left-drag pan, press-and-hold loupe. The left button
    # means pan or loupe depending on timing: press arms a hold timer; moving past _LOUPE_SLOP
    # before it fires cancels to a pan, letting it fire raises the loupe (pan dead while up).
    # A double-click must cancel the timer, or the second press would re-arm a loupe alongside
    # the detail-viewer open.
    def _cell(self, x, y):
        if self._layout is not None:
            return self._freeform_cell(x, y)
        px, py = x - (self._ox + _HDR), y - (self._oy + _COLH)
        return well_at(self._rows, self._cols, self._by_rc, px, py, self._cd)

    def _freeform_cell(self, x, y):
        """Hit-test a geometrically placed holder: the first cell whose own rect contains (x, y).

        Placed cells are tested first and in reverse paint order, so an overlap resolves to the
        one drawn on top. Nominal (empty-slot) rects are only consulted when no real region
        claims the point.
        """
        placed = [rc for rc in self._by_rc if rc in self._layout]
        for rc in list(reversed(placed)) + [rc for rc in self._by_rc if rc not in self._layout]:
            rx, ry, rw, rh = self._cell_rect(*rc)
            if rx <= x < rx + rw and ry <= y < ry + rh:
                ri, ci = rc
                return {"row_index": ri, "col_index": ci, "row": self._rows[ri],
                        "col": self._cols[ci], "well_id": self._by_rc.get(rc)}
        return None

    def _cells_in(self, x0, y0, x1, y1) -> list:
        """Widget px -> acquired cells, via the pure helper (same margin removal as _cell)."""
        if self._layout is not None:
            lo_x, hi_x = min(x0, x1), max(x0, x1)
            lo_y, hi_y = min(y0, y1), max(y0, y1)
            hits = []
            for rc in self._by_rc:
                rx, ry, rw, rh = self._cell_rect(*rc)
                if rx < hi_x and rx + rw > lo_x and ry < hi_y and ry + rh > lo_y:
                    hits.append(rc)
            return sorted(hits)
        ox, oy = self._ox + _HDR, self._oy + _COLH
        return cells_in_rect(self._rows, self._cols, self._by_rc,
                             x0 - ox, y0 - oy, x1 - ox, y1 - oy, self._cd)

    def selected_wells(self) -> list:
        """The selection as acquired well ids, in plate row-major order."""
        return [self._by_rc[rc] for rc in sorted(self._selection)]

    def fov_subsets(self) -> dict:
        """``{region: [fov, ...]}`` for the selected regions a marquee picked only part of.

        A region absent from this dict is selected whole. Pruned on read against the live
        selection, and filtered to strict subsets, so a box that completes a region collapses
        back to "whole region" with no special case in the gesture.
        """
        live = set(self.selected_wells())
        out = {}
        for region, fovs in self._fov_selection.items():
            if region not in live or not fovs:
                continue
            if len(fovs) < len(self._region_fovs(region)):
                out[region] = list(fovs)
        return out

    def _region_fovs(self, region) -> list:
        """Every field this region has a mosaic box for, sorted. ``[]`` when it has no mosaic."""
        return sorted({f for r, f in self._boxes if r == region})

    def _fovs_in(self, x0, y0, x1, y1, cells) -> dict:
        """``{region: [fov, ...]}`` for the fields of *cells* the widget-px box actually touches.

        A region with fewer than two mosaic boxes is skipped: one field fills its cell, nothing
        to subset. Full coverage is not filtered here — that collapse happens in
        :meth:`fov_subsets`, the one place that does it.
        """
        if not self._boxes:
            return {}
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        out: dict = {}
        for ri, ci in cells:
            region = self._by_rc.get((ri, ci))
            if not region:
                continue
            if len(self._region_fovs(region)) < 2:
                continue                       # one field fills the cell: nothing to subset
            a = self._cell_point(ri, ci, lo_x, lo_y)
            b = self._cell_point(ri, ci, hi_x, hi_y)
            if a is None or b is None:
                continue
            bx0, by0 = min(a[0], b[0]), min(a[1], b[1])
            bx1, by1 = max(a[0], b[0]), max(a[1], b[1])
            hit = sorted(
                fov for (r, fov), (top, left, h, w) in self._boxes.items()
                if r == region and left < bx1 and left + w > bx0 and top < by1 and top + h > by0
            )
            if hit:
                out[region] = hit
        return out

    def clear_selection(self):
        """Drop the whole selection and tell listeners (used on re-ingest)."""
        self._fov_selection = {}
        if self._selection:
            self._selection = set()
            self.selectionChanged.emit([])
            self.update()

    def select_all(self):
        """Select every occupied well (the Select all button and Cmd/Ctrl-A)."""
        self._selection = set(self._by_rc.keys())
        self._fov_selection = {}       # "all wells" means all of every well, boxes included
        self.selectionChanged.emit(self.selected_wells())
        self.update()

    def highlight_regions(self, region_ids):
        """Move the blue highlight onto *region_ids*, used when the user clicks an open view."""
        want = set(region_ids or [])
        self._selection = {rc for rc, rid in self._by_rc.items() if rid in want}
        self._fov_selection = {}     # a window holds whole regions; it cannot mean a FOV box
        self.selectionChanged.emit(self.selected_wells())
        self.update()

    def set_view_hues(self, entries):
        """Colour-code the open views on the plate: *entries* is ``[(region_ids, QColor), ...]``,
        one per open window, painted under the blue focus/selection mark."""
        hues = []
        for region_ids, color in (entries or []):
            rcs = {rc for rc, rid in self._by_rc.items() if rid in set(region_ids or [])}
            if rcs:
                hues.append((rcs, color))
        self._view_hues = hues
        self.update()

    def wheelEvent(self, e):
        if self._marquee is not None:
            return          # a marquee owns the drag; zooming would slide the plate under the rect
        if self._loupe is not None:      # zooming the plate under a live loupe would fight it
            return
        mx, my = e.x() - (self._ox + _HDR), e.y() - (self._oy + _COLH)    # cursor in plate px
        new_cd = self._cd * (1.0015 ** e.angleDelta().y())
        new_cd = max(self._fit_cd(), min(self._fit_cd() * 40, new_cd))    # never zoom out past fit
        scale = new_cd / self._cd
        self._ox = e.x() - _HDR - mx * scale         # keep the point under the cursor fixed
        self._oy = e.y() - _COLH - my * scale
        self._cd = new_cd
        self._user_view = True
        self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        # Shift owns multi-well selection: Shift-drag opens the wells you box, Shift+Alt unions,
        # Cmd/Ctrl-click toggles one. A plain click replaces rather than toggles, which is what
        # keeps double-click safe (Qt delivers press+release before mouseDoubleClickEvent).
        if e.modifiers() & Qt.ShiftModifier:
            self._marquee = (e.x(), e.y(), e.x(), e.y())
            self._marquee_add = bool(e.modifiers() & Qt.AltModifier)
            self._press = None
            self._panning = False
            self.update()
            return
        # Cmd/Ctrl-click toggles this well in the batch selection, committed on release so a
        # cmd-drag can still not-select if the user changes their mind.
        if e.modifiers() & Qt.ControlModifier:
            self._ctrl_click = (e.x(), e.y())
            self._press = None
            self._panning = False
            return
        self._press = (e.x(), e.y(), self._ox, self._oy)
        self._panning = False
        c = self._cell(e.x(), e.y())
        if self._loupe_src is not None and c and c["well_id"]:
            self._hold.start()

    def mouseMoveEvent(self, e):
        if self._loupe is not None:
            # Qt grabs the mouse during a press, so no leaveEvent fires until release; a move
            # with coordinates outside rect() is the only signal the cursor left the widget.
            if not self.rect().contains(e.x(), e.y()):
                self._hold.stop()
                self._dismiss_loupe()
                return
            self._loupe["x"], self._loupe["y"] = e.x(), e.y()
            self._request_loupe(e.x(), e.y())
            self.update()
            return
        if self._marquee is not None and (e.buttons() & Qt.LeftButton):
            x0, y0, _, _ = self._marquee          # grow the rubber band; emit NOTHING until release
            self._marquee = (x0, y0, e.x(), e.y())
            self.update()
            return
        if self._press is not None and (e.buttons() & Qt.LeftButton):
            dx, dy = e.x() - self._press[0], e.y() - self._press[1]
            if abs(dx) + abs(dy) > 3:
                self._panning = True
                self._hold.stop()                    # moved -> this press is a pan, not a hold
            if self._panning:
                self._ox, self._oy = self._press[2] + dx, self._press[3] + dy
                self._user_view = True
                self.update()
                return
        c = self._cell(e.x(), e.y())
        new_hover = (c["row_index"], c["col_index"]) if c else None
        if new_hover == self._hover:   # same cell: skip the repaint, or every pixel of motion repaints
            return
        self._hover = new_hover
        if c and c["well_id"]:
            enc = display_well_id(c["well_id"])
            text = c["well_id"] if enc == c["well_id"] else f'{c["well_id"]} ({enc})'
        elif c:
            text = c["row"] + c["col"] + "  ·  empty"
        else:
            text = ""
        self.hovered.emit(text)
        self.update()

    def mouseReleaseEvent(self, e):
        self._hold.stop()
        had_loupe = self._loupe is not None
        # Only the left release commits: Qt delivers a release for whichever button went up.
        if self._marquee is not None and e.button() == Qt.LeftButton:
            x0, y0, x1, y1 = self._marquee
            add, self._marquee, self._marquee_add = self._marquee_add, None, False
            dragged = abs(x1 - x0) + abs(y1 - y0) > _CLICK_SLOP
            if not dragged:   # Shift+click: toggle one well
                hit = self._cell(x1, y1)
                if hit and hit["well_id"]:
                    self._selection ^= {(hit["row_index"], hit["col_index"])}
                    # A whole-well gesture means the whole well, even if a marquee had cropped it.
                    self._fov_selection.pop(hit["well_id"], None)
                self.selectionChanged.emit(self.selected_wells())
            else:
                # Shift-drag opens a window over the boxed regions and leaves no persistent wash.
                boxed = [self._by_rc[rc] for rc in sorted(set(self._cells_in(x0, y0, x1, y1)))]
                if add:
                    # Shift+Alt-drag unions into the batch selection, and — zoomed in far enough
                    # that the box lands inside a mosaic — unions the fields it covers rather than
                    # the whole well.
                    cells = set(self._cells_in(x0, y0, x1, y1))
                    self._selection |= cells
                    for region, fovs in self._fovs_in(x0, y0, x1, y1, cells).items():
                        prev = self._fov_selection.get(region)
                        self._fov_selection[region] = (
                            sorted(set(prev) | set(fovs)) if prev else list(fovs))
                    self.selectionChanged.emit(self.selected_wells())
                else:
                    self.marqueeSelected.emit(boxed)
                    if self._selection:
                        self._selection = set()
                        self.selectionChanged.emit([])
            self.update()
            self._press = None
            self._panning = False
            self._dismiss_loupe()
            return
        # Cmd/Ctrl-click toggle (add/remove to the batch selection).
        if self._ctrl_click is not None and e.button() == Qt.LeftButton:
            px, py, self._ctrl_click = *self._ctrl_click, None
            hit = self._cell(px, py)
            if hit and hit["well_id"]:
                self._selection ^= {(hit["row_index"], hit["col_index"])}
                self.selectionChanged.emit(self.selected_wells())
                self.update()
            self._press = None
            self._panning = False
            self._dismiss_loupe()
            return
        # Plain click (no modifier, no pan, no loupe): select only this well, or clear on empty.
        if (self._press is not None and not self._panning and not had_loupe
                and e.button() == Qt.LeftButton):
            hit = self._cell(e.x(), e.y())
            new_sel = {(hit["row_index"], hit["col_index"])} if hit and hit["well_id"] else set()
            if new_sel != self._selection:
                self._selection = new_sel
                self.selectionChanged.emit(self.selected_wells())
                self.update()
        self._press = None
        self._panning = False
        self._dismiss_loupe()

    def leaveEvent(self, e):
        self._hold.stop()
        self._dismiss_loupe()
        self._hover = None
        # Drop any in-flight marquee: if the grab is lost mid-drag, no release ever arrives.
        self._marquee = None
        self._marquee_add = False
        self.hovered.emit("")
        self.update()

    def keyPressEvent(self, e):
        """Cmd/Ctrl-A selects every well; Escape clears."""
        if (e.modifiers() & Qt.ControlModifier) and e.key() == Qt.Key_A:
            self.select_all()
            return
        if e.key() == Qt.Key_Escape:
            self.clear_selection()
            return
        super().keyPressEvent(e)

    def set_mosaic_boxes(self, boxes: dict):
        """Adopt the per-FOV cell boxes so a double-click can resolve which FOV was hit.

        Also tells the paint path which cells hold a letterboxed mosaic rather than a
        cell-filling single tile, so the two read one dict and can never disagree.
        """
        self._boxes = dict(boxes or {})
        self._boxed_regions = {r for r, _f in self._boxes}
        self.update()

    def set_carrier(self, plate, images_dir=None):
        """Adopt *plate*'s geometry so the holder can be drawn behind the cells.

        Drawn from :class:`~squidxplorer._plate.PlateGeometry` rather than blitting a carrier
        photograph, which lives in its own pixel space and can silently misregister against the
        cell grid. *images_dir* is accepted and ignored, so callers that passed one still work.
        """
        self._carrier = getattr(plate, "geometry", None) if plate is not None else None
        try:
            from squidxplorer._plate import SlideCarrier
            self._carrier_slide = isinstance(plate, SlideCarrier)
        except Exception:
            self._carrier_slide = False
        self._slides = None
        self.update()

    def _cell_rect(self, ri: int, ci: int) -> tuple:
        """Widget-pixel ``(x, y, w, h)`` of cell (ri, ci) at the current zoom/pan.

        A freeform holder returns the region's own rectangle (its mosaic's bounding box) scaled
        by one transform shared across every region.
        """
        cd = self._cd
        ax, ay = self._ox + _HDR, self._oy + _COLH
        if self._layout is not None:
            r = self._layout.get((ri, ci))
            if r is not None:
                return (ax + r[0] * cd, ay + r[1] * cd, r[2] * cd, r[3] * cd)
        return (ax + ci * cd, ay + ri * cd, cd, cd)

    def _cell_source(self, ri: int, ci: int) -> tuple:
        """The sub-rectangle of the montage canvas that ``_cell_rect(ri, ci)`` shows.

        A mosaic is letterboxed into the cell's ``_CELL`` x ``_CELL`` square, so the bars must be
        excluded or it would be stretched back into them; a single tile fills the whole block.
        Which of the two a cell holds is read from ``self._boxes``.
        """
        full = (ci * _CELL, ri * _CELL, _CELL, _CELL)
        if self._layout is None or self._by_rc.get((ri, ci)) not in self._boxed_regions:
            return full
        r = self._layout.get((ri, ci))
        if r is None or not (r[2] > 0 and r[3] > 0):
            return full
        a = r[2] / r[3]                                    # target aspect == mosaic aspect
        iw = _CELL * min(1.0, a)
        ih = _CELL * min(1.0, 1.0 / a)
        return (ci * _CELL + (_CELL - iw) / 2.0, ri * _CELL + (_CELL - ih) / 2.0, iw, ih)

    def _content_box(self, ri: int, ci: int) -> tuple:
        """``(x, y, w, h)`` in the cell's own ``_CELL`` px block: where the pixels actually are.

        The letterbox bars are background, not acquired data, so "half way across the cell" and
        "half way across the image" differ whenever a cell is letterboxed. An unboxed cell fills
        its block, which is the fallback.
        """
        region = self._by_rc.get((ri, ci))
        box = None
        if region is not None and region in self._boxed_regions:
            for (r, _fov), b in self._boxes.items():
                if r == region:
                    box = _box_union(box, b)
        if box is None:
            return (0.0, 0.0, float(_CELL), float(_CELL))
        top, left, h, w = box
        return (float(left), float(top), float(max(w, 1)), float(max(h, 1)))

    def _cell_point(self, ri: int, ci: int, x, y) -> Optional[tuple]:
        """A widget point as ``(bx, by)`` in cell (ri, ci)'s own ``_CELL`` px block.

        The widget-to-cell inverse: ``_fov_at`` and ``_loupe_geometry`` both go through it, so a
        click and a press-and-hold at the same pixel can never resolve to different places.
        """
        rx, ry, rw, rh = self._cell_rect(ri, ci)
        sx, sy, sw, sh = self._cell_source(ri, ci)
        if not (rw > 0 and rh > 0):
            return None
        return ((x - rx) / rw * sw + (sx - ci * _CELL),
                (y - ry) / rh * sh + (sy - ri * _CELL))

    def _block_rect(self, ri: int, ci: int, top, left, h, w) -> Optional[tuple]:
        """A ``_CELL``-block rectangle back out to widget px, the inverse of :meth:`_cell_point`."""
        rx, ry, rw, rh = self._cell_rect(ri, ci)
        sx, sy, sw, sh = self._cell_source(ri, ci)
        if not (sw > 0 and sh > 0):
            return None
        ox, oy = sx - ci * _CELL, sy - ri * _CELL
        return (rx + (left - ox) * rw / sw, ry + (top - oy) * rh / sh,
                w * rw / sw, h * rh / sh)

    def _cell_fraction(self, ri: int, ci: int, x, y) -> Optional[tuple]:
        """A widget point as ``(fx, fy)`` in 0..1 across the cell's content, or ``None``."""
        pt = self._cell_point(ri, ci, x, y)
        if pt is None:
            return None
        bx, by = pt
        ix, iy, iw, ih = self._content_box(ri, ci)
        return ((bx - ix) / iw, (by - iy) / ih)

    def _tiled_region(self) -> "QRegion":
        """The cells that have an image on the active layer, as a QRegion at pan origin (0, 0).

        Clipping the montage blit to this region is what lets empty wells show the carrier art
        rather than opaque background. Cached and only translated on pan.
        """
        cells = self._tiles_by_layer.get(self._active, set())
        return self._cell_region("active", cells, len(cells))

    def _underlay_region(self) -> "QRegion":
        """The cells the base layer shows through, as a QRegion at pan origin (0, 0)."""
        return self._cell_region(
            "under", self.underlay_cells(),
            (len(self._tiles_by_layer.get("raw", ())),
             len(self._tiles_by_layer.get(self._active, ()))))

    def _cell_region(self, tag: str, cells: set, size) -> "QRegion":
        if self._tile_rgn is None:
            self._tile_rgn = {}
        key = (self._cd, self._active, size)
        hit = self._tile_rgn.get(tag)
        if hit is None or hit[0] != key:
            cd = self._cd
            rgn = QRegion()
            for ri, ci in cells:
                rgn = rgn.united(QRegion(int(ci * cd), int(ri * cd),
                                         int(cd) + 1, int(cd) + 1))   # +1: no hairline seams
            hit = self._tile_rgn[tag] = (key, rgn)
        return hit[1]

    def _fov_at(self, c: dict, e) -> int:
        """FOV index under the cursor within cell *c*, or 0 when there is no mosaic to resolve.

        Boxes overlap by ~9% at the seams, so the last match wins, matching the draw order that
        paints later FOVs over earlier ones.
        """
        region = c["well_id"]
        if not region or not self._boxes:
            return 0
        pt = self._cell_point(c["row_index"], c["col_index"], e.x(), e.y())
        if pt is None:
            return 0
        hit = self._fov_box_at(region, pt[0], pt[1])
        return 0 if hit is None else hit[0]

    def focusOutEvent(self, e):
        self._hold.stop()
        self._dismiss_loupe()
        super().focusOutEvent(e)

    def mouseDoubleClickEvent(self, e):
        # Qt sends press/release/dblclick: the second press already re-armed the hold timer.
        self._hold.stop()
        self._dismiss_loupe()
        c = self._cell(e.x(), e.y())
        if c and c["well_id"]:
            self.wellActivated.emit(c["well_id"], self._fov_at(c, e))

    def paintEvent(self, _):
        if not self._user_view:
            self._fit()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(_BG))
        cd, nr, nc = self._cd, self._nr, self._nc
        ax, ay = self._ox + _HDR, self._oy + _COLH
        W, H = nc * cd, nr * cd
        tiled = self._tiles_by_layer.get(self._active, set())
        under = self.underlay_cells()
        shown = tiled | under
        # The holder is drawn from the plate's own geometry, so nothing to calibrate or drift.
        self._paint_carrier(p, shown)
        if self._layout is not None:
            # Freeform: each region's cell is its own rectangle, blitted individually.
            src = self._active_source()
            base = self._base_source()
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            for rc in sorted(shown):
                if rc not in self._by_rc:
                    continue
                img = src if rc in tiled else base
                p.drawImage(QRectF(*self._cell_rect(*rc)), img, QRectF(*self._cell_source(*rc)))
        else:
            # Blit the montage from a cached pixmap scaled to the current zoom; the smooth
            # resample runs once per zoom/source-change, not every repaint.
            w, h = max(1, int(W)), max(1, int(H))
            if (self._scaled is None or self._scaled_cd != cd
                    or self._scaled.width() != w or self._scaled.height() != h):
                self._scaled = QPixmap.fromImage(self._active_source()).scaled(
                    w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self._scaled_cd = cd
            if under:
                # The base under a partial layer, clipped to the cells the active layer lacks.
                base_key = (cd, w, h, self._base_gen)
                if self._scaled_base is None or self._scaled_base_key != base_key:
                    self._scaled_base = QPixmap.fromImage(self._base_source()).scaled(
                        w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                    self._scaled_base_key = base_key
                p.save()
                p.setClipRegion(self._underlay_region().translated(int(ax), int(ay)))
                p.drawPixmap(int(ax), int(ay), self._scaled_base)
                p.restore()
            if len(tiled) < nr * nc:
                # The montage canvas is opaque wherever no tile landed; clip it to real pixels.
                p.save()
                p.setClipRegion(self._tiled_region().translated(int(ax), int(ay)))
                p.drawPixmap(int(ax), int(ay), self._scaled)
                p.restore()
            else:
                p.drawPixmap(int(ax), int(ay), self._scaled)

        # Deep zoom, on top of the montage and under every annotation: a tile that has not
        # arrived leaves the coarse pixels showing rather than a hole.
        if self._tile_cache is not None:
            self._paint_tiles(p)

        # Per-cell dot over the whole plate grid: amber = processing, red x = failed, grey = no
        # image on the active layer, no dot once a cell has an image.
        d = min(max(3.0, cd * 0.36), 15.0)
        for ri in range(nr):
            for ci in range(nc):
                state = self._status.get((ri, ci), "empty")
                has_img = (ri, ci) in shown
                x0, y0, cw, ch = self._cell_rect(ri, ci)
                ex, ey = int(x0 + (cw - d) / 2), int(y0 + (ch - d) / 2)
                if state == "processing":
                    p.setPen(Qt.NoPen)
                    p.setBrush(_STATUS["processing"])
                    p.drawEllipse(ex, ey, int(d), int(d))
                elif state == "failed":
                    p.setPen(QPen(_STATUS["failed"], max(1.5, min(cd * 0.09, 3.0))))
                    p.drawLine(ex, ey, ex + int(d), ey + int(d))
                    p.drawLine(ex + int(d), ey, ex, ey + int(d))
                elif not has_img:
                    p.setPen(Qt.NoPen)
                    p.setBrush(_STATUS["empty"])
                    p.drawEllipse(ex, ey, int(d), int(d))
        p.setBrush(Qt.NoBrush)

        if self._view_hues:
            # A frame, never a wash: a translucent fill would tint the tissue the user is
            # reading. Opaque at full alpha, since a wash-alpha colour is nearly invisible as
            # a 3 px stroke.
            w = selection_frame_pen_px(cd)
            for rcs, color in self._view_hues:
                pen_colour = QColor(color)
                pen_colour.setAlpha(255)
                p.setPen(QPen(pen_colour, w))
                p.setBrush(Qt.NoBrush)
                for ri, ci in rcs:
                    rx, ry, rw, rh = self._cell_rect(ri, ci)
                    p.drawRect(QRectF(rx + w / 2, ry + w / 2,
                                      max(rw - w, 1.0), max(rh - w, 1.0)))
            p.setBrush(Qt.NoBrush)

        # No selection wash at any plate size: a translucent fill would recolour the tissue.

        if self._layout is None:
            p.setPen(QPen(_GRID, 3))
            for c in range(nc + 1):
                p.drawLine(int(ax + c * cd), int(ay), int(ax + c * cd), int(ay + H))
            for r in range(nr + 1):
                p.drawLine(int(ax), int(ay + r * cd), int(ax + W), int(ay + r * cd))
        # A freeform holder has no shared grid; _paint_carrier already outlined each cell.
        p.setFont(_plate_font(_LABEL_PX, QFont.DemiBold))
        if self._layout is not None:
            # A freeform region is named, not numbered, and the gutter is sized for "A".."AF" —
            # "manual0" gets sliced in half there. Its own cell is the only place wide enough and
            # the only place that is unambiguous when cells are individually positioned.
            for rc, region in self._by_rc.items():
                rx, ry, rw, _rh = self._cell_rect(*rc)
                p.setPen(_ACCENT if self._hover == rc else _MUTED)
                p.drawText(QRectF(rx, ry - _COLH, max(rw, 60.0), _COLH),
                           int(Qt.AlignCenter), str(region))
        # Column/row labels THIN OUT as cells shrink so they never overlap (a 48-col 1536wp would
        # otherwise cram "1..48" into a few px). Always draw the hovered row/col so hover still
        # reads. Skipped entirely for a freeform holder: its rows and columns are an internal
        # bookkeeping key, not something on the glass, and the names are already on the cells.
        cstep = max(1, int(np.ceil(22.0 / cd)))
        rstep = max(1, int(np.ceil(18.0 / cd)))
        for c in range(nc if self._layout is None else 0):
            hov = bool(self._hover and self._hover[1] == c)
            if c % cstep and not hov:
                continue
            p.setPen(_ACCENT if hov else _MUTED)
            p.drawText(int(ax + c * cd), int(self._oy), int(cd), _COLH, Qt.AlignCenter, str(self._cols[c]))
        for r in range(nr if self._layout is None else 0):
            hov = bool(self._hover and self._hover[0] == r)
            if r % rstep and not hov:
                continue
            p.setPen(_ACCENT if hov else _MUTED)
            p.drawText(int(self._ox), int(ay + r * cd), _HDR, int(cd), Qt.AlignCenter, str(self._rows[r]))
        if self._selection:
            # A bounding box, not a wash, so the thumbnail keeps its own pixels/LUT/contrast.
            # Inset by half the stroke so the box lands inside its own cell.
            w = selection_frame_pen_px(cd)
            p.setPen(QPen(_SEL_FRAME, w))
            p.setBrush(Qt.NoBrush)
            for ri, ci in self._selection:
                rx, ry, rw, rh = self._cell_rect(ri, ci)
                p.drawRect(QRectF(rx + w / 2, ry + w / 2, max(rw - w, 1.0), max(rh - w, 1.0)))
        # `_fov_selection` truth-tested first: a plate with no field box must pay one dict test,
        # not a fov_subsets() scan, since this runs on every hover repaint.
        subsets = self.fov_subsets() if self._fov_selection else {}
        if subsets:
            # A partly selected well: outline the fields the box actually picked, thinner than
            # the whole-well frame so it reads as a refinement of it.
            pen = QPen(_SEL_FRAME, max(1.0, selection_frame_pen_px(cd) * 0.6))
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            for ri, ci in self._selection:
                for fov in subsets.get(self._by_rc.get((ri, ci)), ()):
                    box = self._boxes.get((self._by_rc[(ri, ci)], fov))
                    if box is None:
                        continue
                    r = self._block_rect(ri, ci, *box)
                    if r is not None:
                        p.drawRect(QRectF(*r))
            p.setBrush(Qt.NoBrush)
        if self._hover is not None:
            ri, ci = self._hover
            x0, y0, hw, hh = self._cell_rect(ri, ci)
            ex, ey = int(x0 + (hw - d) / 2), int(y0 + (hh - d) / 2)
            p.setPen(Qt.NoPen)
            p.setBrush(_RED)
            p.drawEllipse(ex, ey, int(d), int(d))
        if self._marquee is not None:
            mx0, my0, mx1, my1 = self._marquee
            p.setPen(QPen(_ACCENT, 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(int(min(mx0, mx1)), int(min(my0, my1)),
                       int(abs(mx1 - mx0)), int(abs(my1 - my0)))
        if self._loupe is not None:
            self._paint_loupe(p)
        p.setPen(QPen(QColor("#c9d1d9"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.end()

    def _paint_carrier(self, p: QPainter, tiled: set):
        """Draw the sample holder: body outline, per-cell boundary, empty vs occupied.

        Drawn from the cells' own geometry, so the holder cannot drift out of register with the
        wells, and an acquisition with no artwork renders identically to one with artwork.
        Three states: occupied+imaged (pixels speak, drawn over this), occupied+waiting (solid
        accent boundary + fill), empty slot (dashed dim boundary, no fill). Skipped below a few
        px per cell, where the boundaries are smaller than the status dots.
        """
        if self._carrier is None:
            return
        cd = self._cd
        ax, ay = self._ox + _HDR, self._oy + _COLH
        if self._slides is not None:
            # Real glass slides at true size, side by side; the slides ARE the holder.
            from squidxplorer._slide_art import paint_slides
            slide_rects_px = [(ax + s[0] * cd, ay + s[1] * cd, s[2] * cd, s[3] * cd)
                              for s in self._slides]
            paint_slides(p, slide_rects_px)
            if cd < 6.0:
                return
            self._paint_carrier_cells(p, tiled)
            return
        # The holder body: the union of every cell rectangle, padded by the implied margin.
        rects = [self._cell_rect(r, c) for r in range(self._nr) for c in range(self._nc)]
        if not rects:
            return
        bx0 = min(r[0] for r in rects)
        by0 = min(r[1] for r in rects)
        bx1 = max(r[0] + r[2] for r in rects)
        by1 = max(r[1] + r[3] for r in rects)
        pad = max(4.0, cd * 0.18)
        body = QRectF(bx0 - pad, by0 - pad, (bx1 - bx0) + 2 * pad, (by1 - by0) + 2 * pad)
        p.setBrush(QColor(28, 32, 40))
        p.setPen(QPen(QColor(90, 100, 116), 2))
        p.drawRoundedRect(body, min(10.0, pad), min(10.0, pad))
        # Orientation cue: the A1 / first-slot corner is chamfered, like a real notched plate.
        p.setPen(QPen(QColor(120, 132, 150), 2))
        ch = min(14.0, pad * 2.0)
        p.drawLine(int(body.left()), int(body.top() + ch), int(body.left() + ch), int(body.top()))
        if cd < 6.0:
            return
        self._paint_carrier_cells(p, tiled)

    def _paint_carrier_cells(self, p: QPainter, tiled: set):
        """The per-cell occupied/empty boundaries, shared by the well plate and the slide holder."""
        cd = self._cd
        for ri in range(self._nr):
            for ci in range(self._nc):
                rx, ry, rw, rh = self._cell_rect(ri, ci)
                occupied = (ri, ci) in self._by_rc
                if occupied and (ri, ci) in tiled:
                    continue
                if occupied:
                    p.setPen(QPen(_ACCENT, max(1.0, min(cd * 0.03, 2.0))))
                    p.setBrush(QColor(56, 139, 253, 40))
                else:
                    p.setPen(QPen(QColor(74, 84, 100), 1, Qt.DashLine))
                    p.setBrush(Qt.NoBrush)
                if self._carrier_slide:
                    p.drawRect(QRectF(rx, ry, rw, rh))
                else:
                    # Well diameter relative to pitch, so a 96wp reads as fat wells and a
                    # 1536wp as pinpricks.
                    g = self._carrier
                    f = (g.cell_size_um / g.pitch_x_um) if g.pitch_x_um else 0.8
                    f = float(min(max(f, 0.15), 1.0))
                    p.drawEllipse(QRectF(rx + rw * (1 - f) / 2, ry + rh * (1 - f) / 2, rw * f, rh * f))
        p.setBrush(Qt.NoBrush)

    def _paint_loupe(self, p: QPainter):
        """The inset: real pixels, a µm scale bar when the pixel size is known, or the reason
        there are no pixels. Offset from the cursor so the hand never covers what it points at,
        and clamped inside the widget so it stays whole at the plate's edges."""
        x, y = self._loupe["x"], self._loupe["y"]
        s = _LOUPE_PX
        bx = x + 18 if x + 18 + s < self.width() else x - 18 - s
        by = y + 18 if y + 18 + s < self.height() else y - 18 - s
        bx = int(max(2, min(bx, self.width() - s - 2)))
        by = int(max(2, min(by, self.height() - s - 2)))
        p.fillRect(bx, by, s, s, QColor("#05070b"))
        if self._loupe_img is not None:
            p.save()
            p.setClipRect(bx, by, s, s)
            p.drawPixmap(bx, by, QPixmap.fromImage(self._loupe_img).scaled(
                s, s, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation))
            p.restore()
        else:
            p.setPen(_MUTED)
            p.setFont(_plate_font(_LABEL_PX))
            p.drawText(bx, by, s, s, Qt.AlignCenter | Qt.TextWordWrap,
                       self._loupe_note or "reading …")
        geo = self._loupe_geometry(x, y)
        if geo is not None and self._loupe_img is not None:
            _w, _f, _l, _r, s_loupe, mag = geo
            um_px = loupe_um_per_screen_px(getattr(self._loupe_src, "pixel_size_um", None), s_loupe)
            p.setFont(_plate_font(_SCALE_PX, QFont.DemiBold))
            if um_px is None:
                p.setPen(_MUTED)
                p.drawText(bx + 8, by + s - 10, "scale unknown")
            else:
                target = _nice_scale_um(um_px * (s * 0.4))
                bar = int(round(target / um_px))
                p.setPen(QPen(QColor("#e6edf3"), 2))
                p.drawLine(bx + 10, by + s - 14, bx + 10 + bar, by + s - 14)
                p.setPen(QColor("#e6edf3"))
                p.drawText(bx + 10, by + s - 18, f"{_fmt_um(target)}")
            p.setPen(_ACCENT)
            label = f"{self._loupe['well']}  ·  {mag:.1f}×" if mag >= 1.05 else \
                    f"{self._loupe['well']}  ·  native"
            p.drawText(bx + 8, by + 16, label)
        p.setPen(QPen(QColor("#c9d1d9"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(bx, by, s, s)


def _mosaic_boxes(meta: dict) -> dict:
    """``{(region, fov): (top, left, h, w)}`` — every FOV's box inside its _CELL thumbnail.

    Pure geometry, delegated to :mod:`squidxplorer._placement`. Returns ``{}`` when the
    acquisition has no stage positions or no pixel size, the signal to keep the single-tile
    path. Placement failures for one region are contained to that region.
    """
    from squidxplorer._placement import cell_boxes, fov_offsets_px

    positions = meta.get("fov_positions_um") or {}
    if not positions or meta.get("pixel_size_um") in (None, 0):
        return {}
    frame_shape = meta["frame_shape"]
    out: dict = {}
    for region in meta["regions"]:
        fovs = meta["fovs_per_region"][region]
        if len(fovs) < 2:
            continue
        try:
            offsets = fov_offsets_px(positions, region, fovs, meta.get("pixel_size_um"))
            for fov, box in cell_boxes(offsets, frame_shape, _CELL).items():
                out[(region, fov)] = box
        except (KeyError, ValueError):
            continue
    return out


def content_box(shape, h: int = _CELL, w: int = _CELL) -> tuple[int, int, int, int]:
    """``(top, left, height, width)``: where a *shape*-shaped plane lands in an ``h`` x ``w`` box.

    Applies ``_placement.cell_boxes``' rule (``s = min(box/mh, box/mw)``, then centre) to the
    whole mosaic rather than to its individual FOVs, so a fused mosaic and the raw mosaic of the
    same region land in the same place, at the same size, in the same cell.
    """
    h, w = max(1, int(h)), max(1, int(w))
    mh, mw = max(1, int(shape[0])), max(1, int(shape[1]))
    s = min(h / mh, w / mw)
    ih = max(1, min(h, int(round(mh * s))))
    iw = max(1, min(w, int(round(mw * s))))
    return (h - ih) // 2, (w - iw) // 2, ih, iw


