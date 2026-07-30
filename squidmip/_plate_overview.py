"""The PLATE OVERVIEW: the low-resolution, one-cell-per-well navigator, and the geometry under it.

Gap 6 of the GUI backlog plan (2026-07-29), step 1 of the split of ``squidmip/_viewer.py``.

WHY THIS WAS CUT, AND WHY HERE
------------------------------
``_viewer.py`` grew from 2,081 lines to 8,388 in 199 commits, and the bill has already been paid
once: ``origin/ima-qt6`` was abandoned because its payload sat inside this file, which main had
meanwhile moved by +2706/-1184. A whole planned Qt6 migration died of a merge conflict in one
module. Nothing about that risk is specific to Qt6; it applies to every future change that has to
touch the plate.

The seam was not invented here. ``_viewer.py`` already carried the literal comment

    # --- pure geometry (unit-testable, no Qt display) ---

and everything below it was already Qt-free and already unit-tested as pure functions. This module
is that comment made structural: the cut follows the line the file itself had drawn.

WHAT IS IN HERE, IN THE ORDER THE FILE HAD IT
---------------------------------------------
* **plate geometry**, Qt-free and unit-testable: :func:`well_at`, :func:`cells_in_rect`,
  :func:`_fit_cell`, :func:`_fit_box`, :func:`_box_union`, :func:`_row_letter`,
  :func:`_plate_grid`, :func:`resolve_plate_root`, and the mosaic-box geometry
  (:func:`_mosaic_boxes`, :func:`region_mosaic_extent_px`, :func:`push_shape_for`,
  :func:`_fit_letterboxed`).
* **contrast over a streaming plate**: :class:`_RunningContrast` and :func:`_pct_window`, the
  during-run histogram approximation and the exact percentile window the final render uses.
* **the loupe** (IMA-208): the magnification math, and the three sources it can read real pixels
  from (:class:`_RawLoupeSource` over the acquisition's TIFFs, :class:`_ZarrLoupeSource` over a
  written pyramid) plus :class:`_LoupeWorker`, which serves crops off the GUI thread.
* **deep zoom's tile fetcher** (:class:`_TileFetcher`).
* **the widget itself**, :class:`PlateOverview`.

WHAT IS DELIBERATELY NOT IN HERE
--------------------------------
The two QThreads that DID stay, :class:`_LoupeWorker` and :class:`_TileFetcher`, are here rather
than in :mod:`squidmip._workers` with the other eight for a structural reason, not a stylistic one.
They are private to :class:`PlateOverview` and it is their only caller, while ``_workers`` imports
the plate geometry ABOVE (``_fit_cell``, ``_CELL``, ``push_shape_for``) to fill its tiles. Putting
them in ``_workers`` would make ``_workers`` and this module import each other, and a cycle is a
worse outcome than two threads filed next to their owner.

``_ChannelBar`` stays in ``_viewer.py``: it is a sibling widget under the plate, not part of it,
and ``PlateWindow`` is its only constructor.

THE DIRECTION OF THE ARROWS
---------------------------
``_viewer`` -> ``_workers`` -> ``_plate_overview`` -> (``_montage``, ``_plate``, ``_tiling``,
``contract``, ``_budget``, ``_qtstyle``). One direction, no cycles. This module imports nothing
from ``_viewer``, which is the property that makes it possible to change the plate without
reopening the god object, and the property a re-export would silently destroy.

Behaviour is unchanged by the move: every name below is byte-identical to the ``_viewer.py`` it
came from, and ``_viewer.py`` re-exports all of them so existing importers and the ~40 tests that
reach in through ``from squidmip import _viewer as V`` are untouched. This removed 2,459 lines from
``_viewer.py``, which went from 8,388 lines to 5,940.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, QRectF, QThread, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QRegion
from qtpy.QtWidgets import QWidget

from squidmip import _qtstyle
from squidmip._budget import cache_budget
from squidmip._logpane import get_logger
from squidmip._montage import _area_downsample, composite
from squidmip._plate import display_well_id
from squidmip._tiling import TileDescriptor
from squidmip.contract import field_levels, field_path

#: Same logger name the plate's code logged under before the move, so a log line reads identically.
log = get_logger("viewer")

# Chrome (colours, stylesheets, palette) is defined ONCE in `squidmip._qtstyle` and aliased here
# so existing call sites keep their short private names. These are NOT second definitions: change
# a colour in _qtstyle and every widget in the window moves with it.
_BG = _qtstyle.BG
_GRID, _RED, _MUTED, _ACCENT = _qtstyle.GRID, _qtstyle.RED, _qtstyle.MUTED, _qtstyle.ACCENT
_STATUS = _qtstyle.STATUS   # processing-status hue coding; see squidmip/_qtstyle.py

_CELL = 88                # per-well px in the low-res overview (1536wp -> ~4224x2816)
_PUSH_PX = 512             # per-well px pushed to the ndviewer scan-slider (downsampled -> bounded RAM)
_HDR, _COLH = 46, 30       # left / top label margins (px)
_PAD = 16                  # breathing room around the plate

#: The plate's region highlight — a MORE TRANSPARENT light-blue wash than _SEL_FILL (Julio). Shown
#: on the manually-picked wells AND on the regions of the open view you click (highlight_regions).
_VIEW_WASH = QColor(88, 166, 255, 40)   # ~16% alpha light blue

# Byte budget for the deep-zoom tile cache. A quarter of the measured cache budget: the plate
# overlay is one of several consumers (the mosaic pyramid and the reader's own plane cache are the
# others), so it must not claim the whole allowance. TileCache enforces it by eviction.
_TILE_CACHE_BYTES = max(64 << 20, cache_budget() // 4)

_CLICK_SLOP = 3                       # px of travel below which a Shift-drag counts as a click
#                                        (matches the pan threshold, so the two gestures agree)


# --- pure geometry (unit-testable, no Qt display) -------------------------------------------

def well_at(rows, cols, by_rc, px: float, py: float, cell_disp: float) -> Optional[dict]:
    """Map a plate pixel (px, py) at *cell_disp* px/well to a cell, or None if out of bounds.

    ``by_rc`` maps (row_index, col_index) -> well_id for acquired wells (else the cell is 'empty').
    Pixels are relative to the plate's top-left (label margins already removed by the caller).
    """
    if px < 0 or py < 0:
        return None
    ci, ri = int(px // cell_disp), int(py // cell_disp)
    if ci >= len(cols) or ri >= len(rows):
        return None
    return {"row_index": ri, "col_index": ci, "row": rows[ri], "col": cols[ci],
            "well_id": by_rc.get((ri, ci))}


def cells_in_rect(rows, cols, by_rc, x0: float, y0: float, x1: float, y1: float,
                  cell_disp: float) -> list:
    """Every ACQUIRED cell whose square meets the drag rect (x0,y0)-(x1,y1), row-major sorted.

    Same plate-pixel space as ``well_at`` (label margins already removed by the caller). The rect
    is NORMALIZED first, so an up-left drag selects exactly what the equivalent down-right drag
    does. Out-of-grid edges clamp instead of inventing cells, and a cell is returned only when
    ``by_rc`` holds a well there — a marquee over a sparse plate never selects the un-acquired
    positions the grey dots mark.

        by_rc = {(0,0):A1, (1,1):B2}          drag (0,0)->(39,39) at 20px/cell
        +-------+-------+
        |  A1   |  A2   |   -> [(0,0), (1,1)]   A2/B1 are plate positions, not acquisitions
        |  (B1) |  B2   |
        +-------+-------+
    """
    if cell_disp <= 0:
        return []
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)      # normalize: any drag direction is equal
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    if hi_x < 0 or hi_y < 0:                             # entirely above/left of the plate
        return []
    c0, c1 = int(max(0.0, lo_x) // cell_disp), int(max(0.0, hi_x) // cell_disp)
    r0, r1 = int(max(0.0, lo_y) // cell_disp), int(max(0.0, hi_y) // cell_disp)
    c1, r1 = min(c1, len(cols) - 1), min(r1, len(rows) - 1)   # clamp at the far edge
    return [(ri, ci) for ri in range(r0, r1 + 1) for ci in range(c0, c1 + 1) if (ri, ci) in by_rc]


def _fit_cell(a: np.ndarray) -> np.ndarray:
    """Resize a 2D plane to EXACTLY (_CELL, _CELL) for the montage tile.

    Area-downsample when larger (the common case: a ~768px tile -> 88); nearest-upscale a tiny
    frame so the tile shape is always (_CELL, _CELL) (guards the <88px-frame crash the review found).
    """
    if a.shape == (_CELL, _CELL):
        return a
    if a.shape[0] >= _CELL and a.shape[1] >= _CELL:
        return _area_downsample(a, _CELL, _CELL)
    yi = (np.arange(_CELL) * a.shape[0]) // _CELL
    xi = (np.arange(_CELL) * a.shape[1]) // _CELL
    return a[yi][:, xi].astype(np.float32)


def _fit_box(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a 2D plane to EXACTLY (h, w) — the arbitrary-target sibling of :func:`_fit_cell`.

    Used to place one FOV into its box inside a multi-FOV mosaic cell (IMA-187), where each box
    is a fraction of _CELL and generally not square. Same policy as _fit_cell: area-downsample
    when shrinking (the normal case — a 2084px frame into a ~15px box), nearest-sample when
    upscaling, so a tiny synthetic frame in a test can never crash the render path.
    """
    h, w = max(1, int(h)), max(1, int(w))
    if a.shape == (h, w):
        return a
    if a.shape[0] >= h and a.shape[1] >= w:
        return _area_downsample(a, h, w)
    yi = (np.arange(h) * a.shape[0]) // h
    xi = (np.arange(w) * a.shape[1]) // w
    return a[yi][:, xi].astype(np.float32)
def _box_union(a, b):
    """Union of two ``(top, left, h, w)`` boxes; ``a`` may be None (nothing accumulated yet).

    The union of a region's FOV boxes is the rectangle the mosaic actually occupies inside its
    cell, and that is what gets cached and replayed. It is the same rectangle
    ``_placement.cell_boxes`` centred there in the first place.
    """
    if a is None:
        return tuple(int(v) for v in b)
    top = min(a[0], b[0])
    left = min(a[1], b[1])
    bottom = max(a[0] + a[2], b[0] + b[2])
    right = max(a[1] + a[3], b[1] + b[3])
    return (int(top), int(left), int(bottom - top), int(right - left))


# The Squid well-plate formats we fit a plate to (well count -> (rows, cols)). An acquisition whose
# format isn't one of these falls back to a present-only grid (see _plate_grid).
_PLATE_DIMS = {4: (2, 2), 6: (2, 3), 12: (3, 4), 24: (4, 6), 96: (8, 12),
               384: (16, 24), 1536: (32, 48)}


def _row_letter(i: int) -> str:
    """0->A, 25->Z, 26->AA, ... (plate row labels)."""
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _plate_grid(wellplate_format) -> Optional[tuple[list, list]]:
    """Full (rows, cols) label grid for a Squid wellplate format, so the plate view shows every
    position evenly spaced (present wells fill; absent stay blank) rather than collapsing gaps.
    Returns None for an unknown/absent format (caller falls back to present-only)."""
    import re
    m = re.search(r"(\d+)", str(wellplate_format or ""))
    dims = _PLATE_DIMS.get(int(m.group(1))) if m else None
    if not dims:
        return None
    nr, nc = dims
    return [_row_letter(i) for i in range(nr)], [str(c) for c in range(1, nc + 1)]


def resolve_plate_root(path) -> tuple[Path, bool]:
    """(path, is_plate): is_plate True when *path* already holds an OME-zarr plate (not a raw
    acquisition); False for a raw acquisition (the case this viewer opens)."""
    p = Path(path)
    if (p / "plate.ome.zarr").is_dir() or (p.name.endswith(".zarr") and (p / "zarr.json").exists()):
        return p, True
    return p, False
class _RunningContrast:
    """Per-channel global contrast that updates as wells stream in (histogram over tiles so far).

    Each channel also carries an auto/manual LATCH (IMA-206). The histogram keeps growing while a
    run streams, so an untouched channel keeps auto-scaling; the first time the user drags that
    channel's contrast it latches MANUAL and the next well to land can no longer stomp the window
    they just set. ``set_auto`` unlatches it back onto the running histogram.
    """

    def __init__(self, n_ch: int, dmax: float, pct=(1.0, 99.8), bins=512):
        self._bins, self._dmax, self._pct = bins, max(1.0, float(dmax)), pct
        self._hist = [np.zeros(bins, dtype=np.int64) for _ in range(n_ch)]
        self._manual: dict[int, tuple[float, float]] = {}   # ch -> the window the USER latched
        # ch -> the window the OWNING VIEWER (ndviewer_light) resolved and is rendering with.
        # Deliberately NOT the same dict as _manual: see set_followed.
        self._followed: dict[int, tuple[float, float]] = {}

    @property
    def dmax(self) -> float:
        return self._dmax

    def add(self, ch: int, tile: np.ndarray):
        idx = np.clip((tile.ravel() / self._dmax * self._bins).astype(int), 0, self._bins - 1)
        self._hist[ch] += np.bincount(idx, minlength=self._bins)

    def set_manual(self, ch: int, lo: float, hi: float):
        """Latch *ch* to a user-set window (hi is kept above lo so _window never divides by zero)."""
        self._manual[ch] = (float(lo), float(max(hi, lo + 1)))

    def set_auto(self, ch: int):
        """Unlatch *ch* — it goes back to following the running histogram."""
        self._manual.pop(ch, None)

    def is_manual(self, ch: int) -> bool:
        """Did the USER latch this channel? Never true merely because the viewer autoscaled."""
        return ch in self._manual

    def set_followed(self, ch: int, lo: float, hi: float):
        """Record the window the OWNING VIEWER resolved for *ch* (IMA-261).

        THIS IS NOT A LATCH, AND THE DISTINCTION IS THE WHOLE POINT
        ------------------------------------------------------------
        The first version of this recorded ndv's window by calling ``set_manual``, which read as
        "the user has taken manual control of this channel". It was wrong twice over, and both
        showed on screen:

          * ndv autoscales on its own, at open, before the user has touched anything — so every
            channel came up latched MANUAL and the plate's running histogram was permanently
            overridden. Auto-contrast was dead from the first frame.
          * ``resolve`` puts a manual latch above everything, so under SCOPE_PER_REGION every cell
            resolved to ndv's one global window. All 1536 wells were painted identically while the
            plate still drew the amber "wells NOT comparable" badge over the top. The control did
            nothing and the caveat was a lie.

        A followed window is a SINK recording what the owner is showing. A manual latch is a
        POLICY decision, and only the user makes it — the sink never writes policy back into the
        model. Same numbers, different authority, and the authority is what ``resolve`` reads.
        """
        self._followed[ch] = (float(lo), float(max(hi, lo + 1)))

    def clear_followed(self, ch: int):
        self._followed.pop(ch, None)

    def is_followed(self, ch: int) -> bool:
        return ch in self._followed

    def resolve(self, ch: int, auto: tuple[float, float],
                follow: bool = True) -> tuple[float, float]:
        """THE precedence rule, in one place (IMA-242, extended by IMA-261).

            user latch  >  the owning viewer's window  >  whatever the caller computed

        Every renderer derives its own *auto* window legitimately — the plate from the running
        histogram, a per-region cell from exact percentiles over that cell, the loupe from a
        well's coarse plane. What they must NOT each decide for themselves is whether the user's
        gesture outranks that, because a renderer that forgets to ask is a renderer where the
        control silently does nothing. One rule, one place, three callers.

        *follow* is how a caller says "I am not rendering the viewer's global view". Only
        SCOPE_PER_REGION passes False, and it means exactly what the user asked for by choosing
        per-region: derive this cell's window from this cell's pixels. A user's explicit latch
        still wins even then — that is a decision about the channel, not about the scope.
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
        """The FLUORESCENCE window for *ch* from the running histogram, ignoring any latch.

        This is the maragall/stitcher rule (``_contrast.auto_contrast``): background peak to BLACK,
        99.9th percentile on top — the SAME rule the viewer windows use. It replaces a plain
        (1st, 99.8th) percentile low end, which lands INSIDE the fluorescence background so the
        whole field lifts off black and saturates (``_contrast`` module docstring: "a percentile
        window washes fluorescence out"). The plate used to get the good window only by FOLLOWING
        the central pane; with the pane gone (decentralized root) the plate must carry the rule
        itself, and it already keeps the per-channel histogram the rule needs.

        A DEGENERATE window (hi <= lo) is returned DELIBERATELY for a blank/flat channel —
        ``_window``'s ``span <= 0`` guard renders that black, the honest answer when there is no
        contrast. Blank wells are normal on a partially acquired plate and must not read as signal.
        """
        h = self._hist[ch].astype(np.float64)
        tot = h.sum()
        if tot == 0:
            return 0.0, self._dmax
        centers = (np.arange(self._bins) + 0.5) / self._bins * self._dmax
        cdf = np.cumsum(h) / tot
        mode_val = float(centers[int(np.argmax(h))])                 # background peak = the mode
        # std of the BACKGROUND (bins at or below the median), computed from the histogram.
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


# --- contrast scope (IMA-207): how wide a net each contrast window is computed over ------------
#
#   GLOBAL      one window per channel across the whole plate. Wells stay COMPARABLE — a dim well
#               looks dim — but a dim region can be crushed to black beside a bright one.
#   PER_REGION  one window per channel PER CELL. Every region fills its own range, so a dim and a
#               bright region are both readable at once — at the cost of comparability: two wells
#               that look identical may differ by orders of magnitude. That is why the active
#               scope is drawn INTO the plate (see paintEvent) rather than living only in a
#               dropdown that a screenshot would crop out.
#
# Scope is a DISPLAY control, NOT a run parameter. Flipping it re-composites from the native-dtype
# tiles PlateOverview already retains (IMA-206's per-layer store); it never re-runs the plate,
# because a 1536wp run is minutes and that would make the control unusable.
#
# PER_FOV is deliberately absent. It slots into `_scoped_windows`' bucketing when someone wants
# per-field windows inside a mosaic cell — no other change.

_PCT = (1.0, 99.8)   # clip the darkest 1% / brightest 0.2% so hot pixels don't crush the window


def _pct_window(a: np.ndarray, pct=_PCT) -> tuple[float, float]:
    """EXACT percentile window over *a*.

    Exactness is the point. ``_RunningContrast`` quantizes to a bin ~dmax/bins wide (~128 counts on
    uint16), so a dim region spanning a few hundred counts collapses into two or three bins and its
    window comes out garbage — precisely the region PER_REGION exists to rescue. The histogram
    stays the live during-run approximation; this is what the final render uses.

    A degenerate result (hi <= lo) is returned as-is on purpose: ``_window`` renders it black.
    """
    if a.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(a, pct)
    return float(lo), float(hi)


# --- loupe (IMA-208): press-and-hold magnifier over the plate ------------------------------
#
# The plate montage CANNOT be the loupe's source: a tile is _CELL (88) px per well, a ~47x
# downsample of a ~4168px field (see _fit_cell), so magnifying it yields interpolation, not
# pixels. The loupe therefore reads the real data behind whatever layer is on screen — the
# acquisition's TIFFs in raw mode, a windowed read of the written pyramid otherwise.
#
# Magnification is derived from the CURRENT plate zoom (so it is dynamic, per the spec) and
# capped at native resolution (so it never invents detail):
#
#     s_plate = cd / well_px            screen px per image px, at the current plate zoom
#     s_loupe = min(1.0, MAG * s_plate) screen px per image px inside the inset (cap = native)
#     M       = s_loupe / s_plate       actual magnification, in (1, MAG]
#     L       = coarsest pyramid level whose own pixels are still >= s_loupe
#
# Reading level L instead of level 0 is what keeps this cheap: L is chosen so the level's
# pixels land ~1:1 on the inset's screen pixels, so the crop is a few hundred px per side no
# matter how far out the plate is zoomed.

_LOUPE_PX = 240            # inset size on screen (px)
_LOUPE_MAG = 8.0           # target magnification over the plate's current scale
_LOUPE_HOLD_MS = 350       # press-and-hold dwell before the loupe arms
_LOUPE_SLOP = 3            # cursor may drift this many px while arming (matches the pan threshold)
_LOUPE_CACHE = 8           # decoded crops kept (small: a crop is a few MB, not a whole well)
_LOUPE_MAX_CROP = 2 * _LOUPE_PX   # ceiling on the RETURNED array's side, in px
# Why a ceiling at all, when level selection is supposed to bound the read: a source can run OUT
# of levels. Raw TIFFs have no pyramid on disk (n_levels == 1), and a written field below
# _PYRAMID_MIN_YX collapses to level 0 alone, so loupe_level clamps to 0 and the crop becomes
# inset/s_loupe — the WHOLE field. Measured on the 2084 px synthetic plate at fit: a 1826 px
# crop, 4 channels, 26.7 MB, composited ON THE GUI THREAD at ~118 ms per cursor move, and the
# worker's LRU — keyed on (well, level, y0, x0, h, w), i.e. a new key for every pixel of motion
# — held eight of them (213 MB). A 4168 px field is 4x worse on both counts. The fix is decimation, not truncation: the requested RECTANGLE
# still defines the region the inset covers (truncating it would silently change the
# magnification), but a source returns it at no more than this many samples per side. The inset
# is 240 px on screen; beyond 2x that, nobody can see the difference.


def _fov_of_well(well_id, fovs_per_region=None) -> int:
    """The FOV index the plate addresses for ``well_id`` — the single seam for multi-FOV.

    The plate hit-test resolves a WELL, never a FOV, so today this is always 0: the viewer is
    one-FOV-per-well (the library folded IMA-187, the viewer has not). Everything that needs a
    FOV goes through here rather than writing a bare ``0``, so when the plate grows FOV
    sub-cells there is one place to change and ``test_fov_seam_is_single_fov`` fails loudly
    instead of the loupe silently magnifying FOV 0 of every position."""
    if fovs_per_region:
        fovs = fovs_per_region.get(well_id)
        if fovs:
            return int(fovs[0])
    return 0


def loupe_scale(cd: float, well_px: int, mag: float = _LOUPE_MAG,
                inset_px: int = _LOUPE_PX) -> tuple[float, float]:
    """(s_loupe, M) for a plate showing ``cd`` screen px per well of ``well_px`` image px.

    ``s_loupe`` is clamped to 1.0 — one screen pixel per level-0 image pixel is as far as
    honest magnification goes; past that we would be upsampling, which is the very thing
    the montage already does badly. ``M`` is what the user actually gains, in [1, mag].

    Two lower clamps, both learned the hard way:

    * Once the user has wheel-zoomed the plate PAST native (a well drawn bigger than its own
      pixel count), the 1.0 cap alone would put the inset BELOW the plate's own scale — a
      loupe that shrinks what it points at. Floor at the plate's scale; the caller labels
      that case "native", since there is no detail left to reveal.
    * A fixed target magnification does not survive a 1536-well plate. At fit, a well is ~10
      screen px, so 8x fills only ~85 px of a 240 px inset and the rest would have to come
      from neighbouring wells. Floor at ``inset_px / well_px`` so the inset shows AT MOST one
      whole well — which is also what the gesture means: look closely at *this* well. On a
      1536wp that yields ~22x rather than 8x, still derived entirely from the plate's zoom."""
    well_px = max(1, int(well_px))
    s_plate = max(1e-9, float(cd) / well_px)
    fill_well = float(inset_px) / well_px             # scale at which one well fills the inset
    # Order matters: cap at native FIRST, then floor at the plate's own scale. Capping last
    # would drag a plate that is already past native back down to 1.0 and demagnify.
    s_loupe = max(s_plate, min(1.0, max(mag * s_plate, fill_well)))
    return s_loupe, s_loupe / s_plate


def loupe_level(s_loupe: float, n_levels: int) -> int:
    """Coarsest pyramid level whose native resolution still satisfies ``s_loupe``.

    Level L is downsampled by 2**L, so its pixels carry scale 1/2**L relative to level 0. We
    want the largest L with 2**-L >= s_loupe, i.e. L <= log2(1/s_loupe). Clamped into the
    levels that actually exist (a small field writes a single level — see _PYRAMID_MIN_YX)."""
    s = min(1.0, max(1e-9, float(s_loupe)))
    return int(max(0, min(int(np.floor(np.log2(1.0 / s))), max(0, int(n_levels) - 1))))


def loupe_crop_px(s_loupe: float, level: int, inset_px: int = _LOUPE_PX) -> int:
    """Image pixels to read AT ``level`` to fill an ``inset_px`` square inset."""
    eff = max(1e-9, float(s_loupe) * (2 ** int(level)))   # screen px per level-``level`` px
    return int(max(1, np.ceil(inset_px / eff)))


def loupe_decimation(crop_px: int, max_px: int = _LOUPE_MAX_CROP) -> int:
    """Power-of-two stride that brings a ``crop_px``-wide read down to <= ``max_px`` samples.

    Applied by the SOURCE, after the rectangle is fixed: the region the inset covers is set by
    the crop rect and must not change, only the sample count within it."""
    step = 1
    while crop_px // step > max(1, int(max_px)):
        step *= 2
    return step


def loupe_clamp_crop(y0: int, x0: int, h: int, w: int, ny: int, nx: int):
    """Fit a crop rect inside a ``ny`` x ``nx`` field: shift the ORIGIN in, keep the extent.

    Every source must do this, and the reason it is a free function rather than four lines
    repeated per source is IMA-208's primary bug: ``_ZarrLoupeSource`` clamped and
    ``_RawLoupeSource`` did not, so raw mode — the DEFAULT on every folder open — passed a
    negative origin straight into a numpy slice. ``a[-427:1399]`` is not an error, it is an
    EMPTY array, so the inset said "no pixels here" over the ~75% of every well whose crop
    starts left of or above the field. Clamping the origin (rather than truncating the extent,
    which would return a 1 px sliver at an edge) keeps the inset full near the field border."""
    ny, nx = max(1, int(ny)), max(1, int(nx))
    h, w = max(1, min(int(h), ny)), max(1, min(int(w), nx))
    return max(0, min(int(y0), ny - h)), max(0, min(int(x0), nx - w)), h, w


def loupe_um_per_screen_px(pixel_size_um, s_loupe: float):
    """µm per SCREEN pixel inside the inset, or None when the pixel size isn't trustworthy.

    Returns None rather than a guess. ``_output.py`` writes 1.0 into the multiscales scale for
    BOTH "unknown" and a genuine 1.0 µm/px, so a computed plate cannot distinguish them (see
    TODOS.md) — callers pass None for that case. A microscopy tool that displays a confidently
    wrong micron figure is worse than one that admits it doesn't know."""
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


# IMA-242: `_composite_rgb` and `_percentile_window` used to live here — a second compositor and a
# second percentile rule, each a hand-synced twin of `composite` and `_pct_window`. They had drifted
# apart in exactly the way that shape always drifts:
#
#   * `_percentile_window` widened a degenerate window to (lo, lo + 1); `_pct_window` deliberately
#     does NOT, because +1 is one DATA unit and (v - lo)/1 clips to 1.0 — a blank or saturated
#     channel rendered FULL WHITE and read as signal. The loupe had the bug the plate had fixed.
#   * `_composite_rgb` took no channel mask, so unticking a channel removed it from the plate and
#     left it in the loupe.
#   * Neither consulted the manual latch, so dragging a contrast slider moved the plate and left
#     the loupe showing the old window forever.
#
# Both are now gone. `composite` is the one compositor, `_pct_window` the one percentile rule, and
# `_RunningContrast.resolve` the one place the manual-outranks-auto precedence is decided.


_LOUPE_WIN_LOCK = threading.Lock()   # guards the per-source window memo (worker thread writes)


class _LoupeSource:
    """Where the loupe's real pixels come from for the layer currently on the plate.

    Availability is per (source, WELL) — never per layer key. A layer key cannot express what
    is actually on disk: ``OperationStack.add`` dedupes by key, so a saved run and a later
    unsaved preview collapse into one "mip" layer while ``_processed_plate`` still points at
    the older save. Ask the source about the specific well instead."""

    n_levels = 1
    well_px = 1
    pixel_size_um = None

    def available(self, well_id) -> tuple[bool, str]:
        """(ok, reason-if-not). ``reason`` is shown to the user verbatim."""
        return False, "no pixel source"

    def read_crop(self, well_id, level, y0, x0, h, w):
        """(C, y, x) crop at ``level``, CLAMPED into the field (see loupe_clamp_crop) and
        decimated to at most _LOUPE_MAX_CROP samples per side. Runs on the worker thread."""
        raise NotImplementedError

    def coarse(self, well_id):
        """A small whole-field (C, y, x) plane used ONLY to derive the contrast window."""
        raise NotImplementedError

    def window(self, well_id):
        """Per-channel contrast window for a well, mirroring the tile's rule.

        Computed HERE, on the loupe worker thread, and memoised per well — never on the GUI
        thread. It used to be derived in ``_on_loupe_crop`` by calling ``coarse()``, which for
        raw meant decoding a whole TIFF plane inside a paint-driven slot AND touching the same
        plane cache the worker was writing (two threads, no lock, one well's pixels labelled as
        another's). One owner, one thread."""
        with _LOUPE_WIN_LOCK:
            cache = self.__dict__.setdefault("_win_cache", {})
            hit = cache.get(well_id)
        if hit is not None:
            return hit
        coarse = self.coarse(well_id)
        win = [_pct_window(coarse[c]) for c in range(coarse.shape[0])]
        with _LOUPE_WIN_LOCK:
            cache[well_id] = win
        return win


class _RawLoupeSource(_LoupeSource):
    """Raw-acquisition source: the loupe works the moment a folder is open, before any operator.

    Reads the same representative plane per channel that _PreviewWorker already reads, so the
    inset shows exactly the data the raw plate tile was built from. Individual TIFFs hold one
    plane per file and aren't tiled, so a crop means decoding that plane — hence the one-well
    plane cache. Bounded to a single well's channels (~C x frame bytes)."""

    def __init__(self, reader, meta, fov_of):
        self._reader, self._meta, self._fov_of = reader, meta, fov_of
        ny, nx = meta["frame_shape"]
        self.well_px = int(min(ny, nx))
        self.n_levels = 1                      # raw TIFFs have no pyramid ON DISK
        self.pixel_size_um = meta.get("pixel_size_um")
        self._channels = [c["name"] for c in meta["channels"]]
        zs = meta["z_levels"]
        self._z = zs[len(zs) // 2]             # mid plane, as the preview does
        self._lock = threading.RLock()         # _planes is touched by the worker AND the GUI thread
        self._cache_key = None
        self._cache = None
        self._coarse: dict[str, np.ndarray] = {}

    def available(self, well_id) -> tuple[bool, str]:
        if well_id in self._meta["regions"]:
            return True, ""
        return False, "no image for this well"

    def _planes(self, well_id):
        """The well's (C, y, x) planes, decoded once and cached.

        Held under a lock for the whole check-decode-publish sequence. Unsynchronised, the two
        callers (worker thread reading a crop, GUI thread deriving a window) could interleave
        between the key test and the store and hand back ANOTHER well's pixels labelled as the
        well under the cursor — a wrong-image bug in a microscopy tool, not a glitch. The GUI
        thread no longer calls in at all (see _LoupeSource.window), but the lock stays: the class
        must be correct for its callers, not for today's call sites."""
        with self._lock:
            if self._cache_key != well_id:
                fov = self._fov_of(well_id)
                planes = np.stack([
                    np.asarray(self._reader.read(well_id, fov, ch, self._z))
                    for ch in self._channels])
                self._cache, self._cache_key = planes, well_id
            return self._cache

    def read_crop(self, well_id, level, y0, x0, h, w):
        """Level is always 0 here — raw has no pyramid — so the whole burden of bounding the
        work falls on decimation. At plate fit the rect IS most of the field (2084 px on the
        synthetic plate); area-averaging it down to <= _LOUPE_MAX_CROP happens HERE, on the
        worker thread, so what crosses to the GUI thread to be composited is a 456 px square
        (3.3 MB, 11 ms) instead of a 1826 px one (26.7 MB, 118 ms) — which is also what the
        worker's LRU then caches."""
        p = self._planes(well_id)
        ny, nx = p.shape[-2], p.shape[-1]
        y0, x0, h, w = loupe_clamp_crop(y0, x0, h, w, ny, nx)   # NEGATIVE origin -> empty slice
        crop = p[:, y0:y0 + h, x0:x0 + w]
        step = loupe_decimation(max(h, w))
        if step == 1:
            return crop
        oh, ow = max(1, h // step), max(1, w // step)
        # float32, not _area_downsample's float64 default: the compositor casts to float32 anyway,
        # and this array crosses a thread boundary and sits in the worker's LRU.
        return np.stack([_area_downsample(crop[c], oh, ow).astype(np.float32, copy=False)
                         for c in range(crop.shape[0])])

    def coarse(self, well_id):
        if well_id not in self._coarse:
            p = self._planes(well_id)
            self._coarse[well_id] = np.stack(
                [_area_downsample(p[c], _CELL, _CELL) for c in range(p.shape[0])])
        return self._coarse[well_id]


class _ZarrLoupeSource(_LoupeSource):
    """Written-plate source: a WINDOWED tensorstore read of one pyramid level.

    Deliberately NOT _ComputedPlateWorker._read, which pulls a whole plane (~139 MB per well at
    level 0 on a 1536wp) — right for its one-pass streaming job, ruinous for a gesture that
    re-reads as the cursor moves. Arrays are chunked (1, 1, 1, <=1024, <=1024) (_zarr_store)
    precisely so a viewer can read a region, so a loupe crop touches a handful of chunks.

    ``written`` is the set of wells this run has actually persisted. It grows as wells land, so
    the loupe works on completed wells DURING a long run, and a subset save / failed well is
    reported as "not written yet" instead of magnifying some other well's pixels."""

    def __init__(self, base, path_of, fov_of, levels, well_px, pixel_size_um, written=None):
        self._base = str(base)
        self._path_of, self._fov_of = path_of, fov_of
        self._levels = list(levels) if levels is not None else None   # None -> discover on first use
        self.n_levels = max(1, len(self._levels)) if self._levels else 1
        self.well_px = int(well_px)
        self.pixel_size_um = pixel_size_um
        self._written = written                # None = every well (a plate opened from disk)
        self._handles: dict[tuple, object] = {}
        self._coarse: dict[str, np.ndarray] = {}

    def mark_written(self, well_id):
        """A well just landed on disk. Availability grows DURING a run — which is exactly when
        someone is watching the plate fill and wants to glance at what already finished."""
        if self._written is not None:
            self._written.add(well_id)

    def available(self, well_id) -> tuple[bool, str]:
        if self._written is not None and well_id not in self._written:
            return False, "not written yet"
        if self._path_of(well_id) is None:
            return False, "no image for this well"
        return True, ""

    def _resolve_levels(self, well_id):
        """Read the field's multiscales once, to learn how many pyramid levels exist.

        Deferred because a run that is still writing hasn't declared its levels yet — and how
        many there are depends on the field size (_PYRAMID_MIN_YX collapses small fields to a
        single level, which is exactly what the test fixtures hit)."""
        if self._levels is not None:
            return self._levels
        # Through the contract, not a hand-parse. This block used to reconstruct the field path by
        # f-string and read multiscales -> datasets[*].path itself behind a bare `except
        # Exception`, i.e. a private copy of the layout plus an unwritten fallback. Both now have
        # one home: squidmip/contract, and docs/plate-contract.md says the pyramid is OPTIONAL and
        # that level "0" is what its absence falls back to.
        self._levels = field_levels(
            field_path(self._base, self._path_of(well_id), self._fov_of(well_id)))
        self.n_levels = max(1, len(self._levels))
        return self._levels

    def _open(self, well_id, level):
        levels = self._resolve_levels(well_id)
        level = max(0, min(int(level), len(levels) - 1))
        key = (well_id, level)
        if key not in self._handles:
            import tensorstore as ts
            path = field_path(self._base, self._path_of(well_id), self._fov_of(well_id),
                              levels[level])
            self._handles[key] = ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}}).result()
        return self._handles[key]

    def read_crop(self, well_id, level, y0, x0, h, w, time_point: int = 0):
        arr = self._open(well_id, level)
        ny, nx = arr.shape[-2], arr.shape[-1]
        # Clamp the ORIGIN so the window stays whole near an edge (shift it in), rather than
        # truncating the extent — clamping y0 to ny-1 first would return a 1px sliver.
        y0, x0, h, w = loupe_clamp_crop(y0, x0, h, w, ny, nx)
        # A field below _PYRAMID_MIN_YX writes level 0 alone, so level selection cannot bound
        # this read; stride it in tensorstore so the I/O itself shrinks, not just the result.
        step = loupe_decimation(max(h, w))
        # Was hardcoded to timepoint 0, which made a 40-timepoint plate look identical to a
        # 1-timepoint one with no error anywhere. Clamped rather than trusted: a caller holding a
        # stale slider position must not index off the end of a shorter re-ingest.
        t_idx = max(0, min(int(time_point), arr.shape[0] - 1))
        return np.asarray(
            arr[t_idx, :, 0, y0:y0 + h:step, x0:x0 + w:step].read().result())

    def coarse(self, well_id, time_point: int = 0):
        # Keyed by (well, timepoint): the old cache was keyed by well alone, so once a well had been
        # read at one timepoint every later timepoint got that same picture back. A cache that
        # answers the wrong question quickly is worse than no cache.
        key = (well_id, int(time_point))
        if key not in self._coarse:
            arr = self._open(well_id, self.n_levels - 1)          # coarsest level = cheapest
            t_idx = max(0, min(int(time_point), arr.shape[0] - 1))
            self._coarse[key] = np.asarray(arr[t_idx, :, 0].read().result())
        return self._coarse[key]


class _LoupeWorker(QThread):
    """Serves loupe crops off the GUI thread, coalescing to the NEWEST request.

    Only the latest cursor position matters: if the user sweeps across three wells while a read
    is in flight, the two intermediate reads are worthless. One pending slot (overwritten by
    each new request) IS the coalescing. Results carry the generation they were asked for, so a
    late arrival for a stale position is dropped by the widget rather than flashing."""

    ready = Signal(int, str, object, object, object)  # (gen, well, crop|None, window|None, err)

    def __init__(self, source: _LoupeSource):
        super().__init__()
        self._source = source
        self._cv = threading.Condition()
        self._pending = None
        self._stop = False
        self._cache: dict[tuple, np.ndarray] = {}
        self._order: list[tuple] = []

    def request(self, gen, well_id, level, y0, x0, h, w):
        with self._cv:
            self._pending = (gen, well_id, level, y0, x0, h, w)
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _cached(self, key):
        hit = self._cache.get(key)
        if hit is not None:
            self._order.remove(key)
            self._order.append(key)
        return hit

    def _store(self, key, val):
        self._cache[key] = val
        self._order.append(key)
        while len(self._order) > _LOUPE_CACHE:
            self._cache.pop(self._order.pop(0), None)

    def run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                gen, well_id, level, y0, x0, h, w = self._pending
                self._pending = None
            key = (well_id, level, y0, x0, h, w)
            try:
                crop = self._cached(key)
                if crop is None:
                    crop = self._source.read_crop(well_id, level, y0, x0, h, w)
                    self._store(key, crop)
                # The contrast window belongs on this side too: deriving it on the GUI thread
                # meant a paint-driven slot could decode a whole TIFF plane (IMA-208).
                try:
                    win = self._source.window(well_id)
                except Exception:
                    win = None                        # the widget falls back to a flat window
                self.ready.emit(gen, well_id, crop, win, None)
            except Exception as e:                    # a racing writer / deleted plate / bad path
                self.ready.emit(gen, well_id, None, None, f"{type(e).__name__}: {e}")


# --- plate overview widget (one cell per well; hue-coded status; fit-to-view) ---------------

#: Deep zoom off-switch. On by default; ``SQUIDMIP_DEEP_ZOOM=0`` restores the pure-montage plate
#: without a revert, which is what makes this safe to ship before it has run on many datasets.
def _deep_zoom_enabled() -> bool:
    return os.environ.get("SQUIDMIP_DEEP_ZOOM", "1") != "0"


#: Most tiles to have in flight at once. The queue is drained newest-first, so a pan that outruns
#: the disk discards stale requests rather than rendering the route the cursor took.
_TILE_QUEUE_MAX = 24


class _TileFetcher(QThread):
    """Decode tiles OFF the GUI thread and hand them back one at a time.

    The piece TODOS.md records as unowned ("Tile render loop + async fetch executor are unowned").
    It has to be a thread: a single per-FOV tile is a full-frame decode per z-plane -- measured at
    ~350 ms on the 10-deep WELLPLATE dataset -- so doing this in ``paintEvent`` would freeze the
    window for seconds per repaint.

    Newest-first (a LIFO) on purpose. A user who pans across the plate generates requests faster
    than they can be served, and the tiles worth decoding are the ones under the cursor NOW, not
    the ones it passed over a second ago. FIFO would render the journey; LIFO renders the
    destination.
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

    def drop_all(self) -> None:
        with self._lock:
            self._pending.clear()

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
                # A tile that will not decode is a hole and the montage still shows — but a hole
                # that says nothing is how "deep zoom quietly does nothing" ships. Name it once
                # per descriptor; the viewport stays alive either way.
                log.warning("tile %s/%s failed to decode: %s: %s",
                            desc.level, desc.key, type(exc).__name__, exc)
                continue
            if not self._stop.is_set():
                self.ready.emit(desc, arr)


class PlateOverview(QWidget):
    """The low-res plate: an RGB canvas of MIP tiles, a per-well status hue, a red box, and a
    press-and-hold LOUPE that overlays real acquisition pixels for the well under the cursor
    (IMA-208 — the montage itself is far too coarse to magnify; see the loupe block above).

    The RGB canvas is only what is CURRENTLY shown. What the widget actually owns (IMA-206) is a
    per-layer ``(C, nr*_CELL, nc*_CELL)`` native-dtype store — the plate with its channel axis
    still intact — plus a channel mask and a per-channel contrast window. Producers hand over
    per-channel tiles and this widget does all compositing, so toggling a channel or dragging its
    contrast re-composites from the retained pixels: no reader I/O, no re-projection, ever.
    """

    hovered = Signal(str)              # region id (or "" off-plate), for the window's readout
    wellActivated = Signal(str, int)   # (well_id, fov_index) double-clicked -> load in ndviewer
    selectionChanged = Signal(list)    # acquired well ids the operator picked (row-major)
    marqueeSelected = Signal(list)     # ...and specifically by a Shift-DRAG: opens an exploration
                                           # tab (IMA-205). Shift+CLICK refines the selection one well
                                           # at a time and deliberately does NOT fire it — otherwise
                                           # every corrective click spawns another tab.

    def __init__(self, rows, cols, wells: dict, layout: Optional[dict] = None):
        """``wells``: (row_index, col_index) -> well_id for every acquired well (drawn grey until
        processed). Tiles/status arrive as an operator runs.

        ``layout`` (IMA-253) is ``{(row_index, col_index): (x, y, w, h)}`` in GRID UNITS for a
        holder whose cells are placed by real geometry rather than by a uniform pitch -- a freeform
        tissue slide, where each region's cell is its own mosaic's bounding box. ``None`` is the
        uniform grid every well plate is, and keeps the single-blit fast path a 1536-well plate
        needs. Cells absent from the map fall back to their nominal ``(c, r, 1, 1)`` square, which
        is what an EMPTY slot (no stage coordinates to place it by) can honestly be drawn as.
        """
        super().__init__()
        self._rows, self._cols = list(rows), list(cols)
        self._layout: Optional[dict] = ({tuple(k): tuple(float(v) for v in val)
                                         for k, val in layout.items()} if layout else None)
        self._nr, self._nc = len(self._rows), len(self._cols)
        self._by_rc: dict[tuple, str] = dict(wells)            # every acquired well (for status + hit-test)
        self._status: dict[tuple, str] = {rc: "empty" for rc in wells}
        self._tiles: set[tuple] = set()                        # cells that have a tile painted (any layer)
        self._tiles_by_layer: dict[str, set] = {}              # layer -> cells with an image there
        self._canvas = QImage(self._nc * _CELL, self._nr * _CELL, QImage.Format_RGB888)
        self._canvas.fill(QColor(_BG))
        self._final = None            # global-contrast recomposite of the ACTIVE layer (or None)
        # Layer stack render: the base ("raw") is self._canvas; each operator draws into its own
        # per-layer canvas/final. self._active is the layer the plate currently shows (LayersTab picks
        # it via set_active_layer). Keeps memory to one small montage-canvas per layer used.
        self._op_canvas: dict[str, QImage] = {}
        self._op_final: dict[str, QImage] = {}
        self._final_arr: dict[str, np.ndarray] = {}   # keeps each recomposited RGB alive: QImage
        #                                               WRAPS the numpy buffer, it does not copy it
        self._active = "raw"
        # --- the channel axis (IMA-206) — set_channels declares it; empty until then -----------
        self._labels: list[str] = []      # channel display names, for the channel bar
        self._colors = None               # (C, 3) float RGB, the RESOLVED display_color per channel
        self._dtype = np.uint16           # store dtype: the acquisition's native dtype (half the RAM)
        self._store: dict[str, np.ndarray] = {}   # layer -> (C, nr*_CELL, nc*_CELL), allocated lazily
        # --- what a contrast change is allowed to touch (IMA-261) ------------------------------
        # A contrast window is a POINT transform, so it commutes with subsampling: windowing the
        # display-sized thumbnail is bit-identical to windowing the whole plate and subsampling
        # afterwards (see squidmip._montage._window_lut). The only thing that must be re-derived
        # per tick is therefore the composite of the DISPLAY-SIZED buffer. These two caches hold
        # everything upstream of that, so a drag re-reads no store and re-percentiles nothing:
        self._disp: dict[str, tuple] = {}      # layer -> (step, contiguous (C, h, w) thumbnail)
        self._cell_auto: dict[str, dict] = {}  # layer -> {(ri, ci): [per-channel AUTO window]}
        #                                        SCOPE_PER_REGION's exact percentiles, which depend
        #                                        on the PIXELS only — never on the contrast.
        self._mask = None                 # (C,) bool: which channels composite into the plate
        self._contrast = None             # _RunningContrast: global per-channel window + auto/manual
        #                                   re-composites from the store above — it never re-runs.
        self._full = QTimer(self)         # coalesces the full-res recomposite behind a gesture
        self._full.setSingleShot(True)    # (a drag repaints at DISPLAY res; full-res lands once)
        self._full.timeout.connect(self._on_full_timeout)
        self._scaled = None           # cached pixmap of (final|canvas) scaled to the current zoom;
        self._scaled_cd = None        # rebuilt only when zoom (cd) or the source image changes — so
        #                               a hover/pan repaint blits 1:1 instead of re-resampling 12 MP.
        self._cd = float(_CELL)       # displayed px/well (fit baseline, then wheel-zoomed)
        self._ox = self._oy = _PAD    # top-left of the plate within the widget (pan-able)
        self._hover = None
        self._sel = None              # well selected from the ndviewer FOV slider
        # SELECTION (IMA-221) is a DIFFERENT concept from _sel above: _sel is "the one well the
        # detail viewer is showing" (red box, driven by the FOV slider); _selection is "the set the
        # operator picked" (tint, driven by Shift-gestures). Never merge them — the red box must
        # survive selecting, and selecting must survive scrubbing.
        self._selection: set = set()  # acquired (row_index, col_index) the user picked. A SET:
        #                               paintEvent membership-tests it once per cell, 1536x on a 1536wp.
        self._view_hues: list = []    # [(rc_set, QColor)] per open view — plate colour-codes threads
        self._marquee = None          # (x0, y0, x1, y1) widget px while a Shift-drag is in flight
        self._marquee_add = False     # this drag unions (Shift+Alt) rather than replaces
        self._ctrl_click = None       # (x, y) of a Cmd/Ctrl-press, committed as a TOGGLE on release
        self._press = None            # (x, y, ox, oy) at left-press, for drag-to-pan
        self._panning = False
        self._user_view = False       # True once the user wheel-zooms/pans (stop auto-fitting)
        self._boxes: dict = {}        # (region, fov) -> (top, left, h, w) in cell px; {} = single-FOV
        self._boxed_regions: set = set()   # regions whose cell holds a LETTERBOXED mosaic, not one tile
        # DEEP ZOOM (below). All None until set_tile_source() succeeds; every path checks _tile_src
        # so an acquisition without stage positions simply keeps the montage and costs nothing.
        self._ladder = None
        self._tile_src = None
        self._tile_cache = None
        self._tile_fetch = None
        self._tile_level = None       # last rung picked, for pick_level's hysteresis
        # -- carrier geometry (IMA-220, redrawn for IMA-253: geometry, not a photograph) --
        self._carrier = None          # the _plate.PlateGeometry to draw the holder outline from
        self._carrier_slide = False   # slot-shaped cells (a slide carrier) vs round wells
        self._slides = None           # [(x, y, w, h), ...] in GRID UNITS: real glass slides drawn
        #                               behind a tissue acquisition (IMA-265, _slide_art). None on
        #                               a well plate and on a carrier with no stage coordinates.
        self._tile_rgn = None         # cached QRegion of cells that HAVE an image, at pan origin
        self._tile_rgn_key = None     # (cd, active layer, n tiled cells) the cached region was built for
        # -- loupe (IMA-208) --
        self._loupe_src = None        # _LoupeSource for the ACTIVE layer, or None (loupe disabled)
        self._loupe_worker = None
        self._loupe = None            # armed/live state dict, or None when idle
        self._loupe_gen = 0           # bumped per request; late results for older gens are dropped
        self._loupe_img = None        # QImage currently shown in the inset
        self._loupe_note = ""         # user-visible reason when the loupe can't show pixels
        self._loupe_win = {}          # well_id -> per-channel window, mirroring the tile's rule
        self._loupe_colors = None     # (C, 3) float RGB, set with the source
        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.setInterval(_LOUPE_HOLD_MS)
        self._hold.timeout.connect(self._arm_loupe)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)   # so focusOutEvent can actually fire (see below)
        self.setMinimumSize(240, 200)

    # -- deep zoom (tile overlay) ---------------------------------------------------------------
    def set_tile_source(self, reader, meta) -> bool:
        """Arm deep zoom for this acquisition. Returns whether it armed.

        Deliberately fail-quiet: an acquisition with no usable stage positions cannot be placed in
        world µm, ``plate_ladder`` says so by raising, and the honest response is to keep the
        montage rather than draw a pile of FOVs at one spot. That is the SAME condition the
        per-region coordinates salvage leaves behind, so a plate with one truncated well arms
        normally and only that well stays coarse.
        """
        self.clear_tile_source()
        if not _deep_zoom_enabled() or reader is None or not meta:
            return False
        try:
            from squidmip._platecache import PlateCellCache
            from squidmip._tiling import TileCache
            from squidmip._tilesource import CompositePlateSource, plate_ladder

            self._ladder = plate_ladder(meta)
            # COMPOSITE, not a bare ReaderTileSource: plate rungs come from the persisted preview
            # cells and FOV rungs from the reader, which is the source NEXT_STEPS.md scoped and
            # did not build. FOV-rung behaviour is byte-identical (it delegates); what changes is
            # that a coarse rung can be served at all, at a dict lookup rather than the 25 s
            # full-plate decode Spencer measured. Seeding is lazy: nothing is read until a coarse
            # tile is actually asked for.
            self._tile_src = CompositePlateSource(
                reader, meta, self._ladder,
                cache=PlateCellCache.for_reader(reader, meta, cell_px=_CELL))
        except Exception:
            self._ladder = self._tile_src = None
            return False
        self._tile_cache = TileCache(budget_bytes=_TILE_CACHE_BYTES)
        self._tile_fetch = _TileFetcher(self._tile_src, self)
        self._tile_fetch.ready.connect(self._on_tile_ready)
        self._tile_fetch.start()
        return True

    def clear_tile_source(self) -> None:
        """Stop and forget the tile machinery. Idempotent; safe on a half-built state."""
        if self._tile_fetch is not None:
            self._tile_fetch.stop()
            self._tile_fetch.wait(1500)
            self._tile_fetch = None
        self._ladder = self._tile_src = self._tile_cache = None
        self._tile_level = None

    def _on_tile_ready(self, desc, arr) -> None:
        if self._tile_cache is None:
            return                      # the source was swapped while this tile was in flight
        self._tile_cache.insert(desc, arr)
        self.update()

    def _visible_fov_tiles(self) -> list:
        """``[(TileDescriptor, QRectF), ...]`` for the FOVs on screen, or ``[]`` to stay coarse.

        The engage rule is one comparison: ``cd > _CELL``. Below it the montage is being shown at
        or under its native 88 px per cell and is exactly the right image — serving tiles there
        would cost a full-plate decode (measured: 25 s) to reproduce a picture that is already on
        screen. Above it the montage is an upscale, which is the blur this feature exists to fix.

        Tiles are placed by reusing ``_placement.cell_boxes`` at the CURRENT cell size, the same
        function the montage itself is composited with. That is what makes the overlay land
        pixel-aligned on the thumbnail underneath instead of merely near it.
        """
        if (self._tile_src is None or self._ladder is None or self._cd <= _CELL
                or self._layout is not None):     # freeform cells are not a uniform grid
            return []
        meta = getattr(self._tile_src, "meta", {})
        positions = meta.get("fov_positions_um") or {}
        px = meta.get("pixel_size_um")
        frame = meta.get("frame_shape")
        if not positions or not px or frame is None:
            return []

        from squidmip._placement import cell_boxes, fov_offsets_px

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
                # The rung is chosen from what this FOV occupies ON SCREEN, not from the plate
                # zoom: a letterboxed mosaic gives each FOV a fraction of the cell, so the plate's
                # µm/px would over-fetch every one of them.
                um_per_px = (float(frame[1]) * float(px)) / max(float(bw), 1.0)
                lvl = self._ladder.geometry.pick_level(um_per_px, self._tile_level)
                if not self._ladder.is_fov_level(lvl):
                    # Coarser than the crossover. The montage already wins HERE because this
                    # enumerator is per FOV and a plate rung is keyed by a world grid cell, not by
                    # (region, fov) -- the montage's uniform cell grid and the ladder's stage
                    # micrometres agree only inside a cell. The rung itself is no longer
                    # unservable: CompositePlateSource answers it from the cached preview cells.
                    # What is still missing is a world-space enumerator to place those tiles, and
                    # that is the continuous zoom-out in NEXT_STEPS.md's MIP-on-plateview item.
                    continue
                self._tile_level = lvl
                out.append((TileDescriptor(level=lvl, key=key, channel=self._tile_channel(),
                                           bbox_um=self._ladder.fov_bboxes[key]), rect))
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
        """One cached tile as an 8-bit greyscale QImage, windowed by the plate's own contrast.

        Reusing ``_RunningContrast`` is the point: a tile that windowed itself would jump in
        brightness the instant it replaced the thumbnail under it, which reads as a rendering bug
        even though every pixel is right.
        """
        cache = self.__dict__.setdefault("_tile_qimages", {})
        hit = cache.get(desc)
        if hit is not None:
            return hit
        lo, hi = self._tile_window()
        a = np.clip((arr.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255)
        a = np.ascontiguousarray(a.astype(np.uint8))
        img = QImage(a.data, a.shape[1], a.shape[0], a.shape[1], QImage.Format_Grayscale8).copy()
        if len(cache) > 256:
            cache.clear()                # bounded; the pixel cache underneath is the real one
        cache[desc] = img
        return img

    def _tile_window(self) -> tuple:
        """The active channel's display window, from the plate's running contrast when it has one.

        ``_RunningContrast.window(ch)`` already resolves the user latch, the followed pane and the
        running histogram in that order — so a tile lands with exactly the contrast the thumbnail
        under it was drawn with, and replacing one with the other is invisible.
        """
        c = self._contrast
        if c is not None:
            try:
                lo, hi = c.window(0)
                if hi > lo:
                    return float(lo), float(hi)
            except Exception:
                pass                # no histogram yet (nothing streamed): fall through to full range
        return 0.0, 65535.0

    # -- loupe wiring --
    def set_loupe_source(self, source, colors=None):
        """Point the loupe at the data behind the ACTIVE layer. ``None`` disables the gesture.

        Called whenever what the plate is showing changes identity — a new acquisition, an
        operator run persisting, a layer switch, a preview superseding a saved run. Re-pointing
        is what stops a stale run's pixels appearing under a newer run's tiles."""
        self._dismiss_loupe()
        if self._loupe_worker is not None:
            self._loupe_worker.stop()
            self._loupe_worker.wait(2000)
            self._loupe_worker = None
        self._loupe_src = source
        self._loupe_colors = colors
        self._loupe_win.clear()
        if source is not None:
            self._loupe_worker = _LoupeWorker(source)
            self._loupe_worker.ready.connect(self._on_loupe_crop)
            self._loupe_worker.start()

    def _arm_loupe(self):
        """Hold timer fired: the press became a loupe. Only reachable while still ARMED."""
        self._hold.stop()             # a pending fire must never re-arm and blank a LIVE loupe:
                                      # _arm_loupe clears _loupe_img, so a second arm 350 ms after
                                      # the first shows an empty inset until the re-read lands.
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
        # cursor -> position within the cell -> image px at level 0 -> image px at ``level``
        ax, ay = self._ox + _HDR, self._oy + _COLH
        fy = (y - (ay + c["row_index"] * self._cd)) / max(1e-9, self._cd)
        fx = (x - (ax + c["col_index"] * self._cd)) / max(1e-9, self._cd)
        span = max(1, src.well_px >> level)
        cy, cx = int(fy * span), int(fx * span)
        # Clamp HERE as well as in the source. Two reasons beyond belt-and-braces: the request
        # that reaches the worker is then always a rectangle that exists, and a hold near a field
        # edge produces the SAME key as the cursor drifts, so the LRU hits instead of decoding a
        # fresh full-field crop per pixel of motion.
        y0, x0, h, w = loupe_clamp_crop(cy - crop // 2, cx - crop // 2, crop, crop, span, span)
        return c["well_id"], level, (y0, x0, h, w), s_loupe, mag

    def _request_loupe(self, x, y):
        geo = self._loupe_geometry(x, y)
        if geo is None:                    # dragged onto the margin / an un-acquired cell
            if self._loupe_img is not None or not self._loupe_note:
                self._loupe_img, self._loupe_note = None, "no well here"
                self.update()
            return
        if self._loupe_worker is None:
            return
        well, level, (y0, x0, h, w), _s, _m = geo
        ok, why = self._loupe_src.available(well)
        if not ok:
            self._loupe_img, self._loupe_note = None, why
            self.update()
            return
        self._loupe_gen += 1
        self._loupe_worker.request(self._loupe_gen, well, level, y0, x0, h, w)

    def _on_loupe_crop(self, gen, well_id, crop, window, error):
        """A crop landed. Drop it unless it is the newest request and the loupe is still up.

        Everything expensive already happened on the worker thread: this slot only windows and
        colours a <= _LOUPE_MAX_CROP square. It must stay that way — it runs inside the paint
        loop of a widget the user is dragging across."""
        if gen != self._loupe_gen or self._loupe is None:
            return
        if error is not None or crop is None or crop.size == 0:
            self._loupe_img, self._loupe_note = None, error or "no pixels here"
            self.update()
            return
        # Mirror the TILE's contrast rule on the WELL's pixels (computed by the source, per well)
        # — never percentiles of the crop under the cursor, which would make brightness lurch as
        # the cursor moves and make the inset look like different data.
        #
        # That AUTO window is then resolved through the plate's one contrast model (IMA-242), so a
        # channel the user latched with the slider shows the user's window here too. Before, the
        # loupe kept its own memo and the inset went on displaying the pre-drag contrast — two
        # representations of one truth, never synced.
        auto = window if window is not None else self._loupe_win.get(well_id)
        if auto is None:
            auto = [(0.0, 1.0)] * crop.shape[0]
        self._loupe_win[well_id] = auto              # memo the AUTO window, never the resolved one
        win = ([self._contrast.resolve(c, auto[c]) for c in range(len(auto))]
               if self._contrast is not None else list(auto))
        colors = self._loupe_colors
        if colors is None:
            colors = np.ones((crop.shape[0], 3), np.float32)
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

    # -- the channel axis: store, mask, per-channel contrast (IMA-206) --
    def set_channels(self, labels, colors: np.ndarray, dtype=np.uint16, pct=(1.0, 99.8)):
        """Declare the acquisition's channels — the per-channel store/mask/contrast start here.

        *colors* is the (C, 3) float RGB of the RESOLVED ``display_color`` (the acquisition's YAML
        first, the wavelength fallback map second — resolve_channels already settled that), so the
        plate is tinted exactly the way every other compositing site tints it.
        """
        self._labels = [str(x) for x in labels]
        self._colors = np.asarray(colors, dtype=np.float32)
        self._dtype = np.dtype(dtype)
        self._mask = np.ones(len(self._labels), dtype=bool)     # every channel on by default (OV8)
        dmax = float(np.iinfo(self._dtype).max) if self._dtype.kind in "ui" else 1.0
        self._contrast = _RunningContrast(len(self._labels), dmax, pct=pct)
        self._store.clear()

    def _store_for(self, layer: str) -> Optional[np.ndarray]:
        """The layer's (C, H, W) plate store, allocated on first tile (one per layer, lazily —
        each layer that supports toggling costs its own ~95 MB at 1536wp x 4ch uint16)."""
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
        """A C-CONTIGUOUS (C, h, w) thumbnail of *store* at subsampling *step*, cached.

        This is the "precompute once at ingest, remap cheaply per tick" half of IMA-261. Building
        it costs one strided copy of the whole store (~2.6 ms on a 1536-well plate); it is
        rebuilt only when the pixels change or the zoom changes, so a continuous contrast drag
        pays for it ZERO times and every tick composites straight out of it.

        Contiguity is not a detail: ``store[:, ::4, ::4]`` is a strided VIEW, and both the table
        lookup and the BLAS reduce in ``composite`` would silently materialise their own copy of
        it on every single tick.
        """
        if step <= 1:
            return store          # already the thing itself; caching it would just double the RAM
        hit = self._disp.get(layer)
        if hit is not None and hit[0] == step:
            return hit[1]
        thumb = np.ascontiguousarray(store[:, ::step, ::step])
        self._disp[layer] = (step, thumb)
        return thumb

    # PER-REGION CONTRAST IS GONE, AND THAT IS THE POINT.
    #
    # `_cell_auto_windows`, `_cell_windows` and `_composite_per_region` lived here: one contrast
    # window per WELL, each cell stretched to its own percentiles. Julio: "the contrast should be
    # only global, I don't understand why there's a per region contrast... I don't think that
    # there's any scientific basis." He is right. It makes a dim well readable next to a bright
    # one, which is a presentation trick, and it costs the one thing a plate view exists for:
    # two wells that look identical may differ by orders of magnitude, so the plate can no longer
    # be read as relative signal. The amber "wells NOT comparable" badge was an admission that
    # the picture was misleading, printed on top of the misleading picture.
    #
    # Deleting it also removes the `follow=False` branch, which is why napari's contrast did not
    # reach the plate: per-region DELIBERATELY ignored the owning viewer's window. There is now
    # one window per channel, owned by napari, and the plate follows it. One owner, one value.

    def set_channel_color(self, ch: int, rgb) -> bool:
        """Re-tint one channel to the colour the CENTRE VIEWER is using, and repaint.

        The plate keeps a ``(C, 3)`` LUT table resolved once from the acquisition's
        ``display_color``. Left alone it is a second, stale answer to "what colour is this
        channel", and recolouring a layer in napari made the two panes disagree about the same
        channel. This is the sink half: napari decides, the plate follows, and nothing is re-read
        -- the composite is rebuilt from the native-dtype tiles already retained.
        """
        if self._colors is None or not (0 <= ch < len(self._colors)):
            return False
        new_rgb = np.asarray(rgb, dtype=np.float32)
        if np.allclose(self._colors[ch], new_rgb):
            return False
        self._colors[ch] = new_rgb
        self._refresh()
        return True

    def set_channel_visible(self, ch: int, on: bool):
        """Toggle a channel in/out of the plate composite. Recomposites from the RETAINED store —
        no reader I/O, no re-projection (that is the whole point of keeping the channel axis)."""
        if self._mask is None or not (0 <= ch < len(self._mask)):
            return
        self._mask[ch] = bool(on)
        self._refresh()

    def set_channel_window(self, ch: int, lo: float, hi: float):
        """Re-window one channel and repaint. LATCHES that channel manual (D4) so the wells still
        streaming in can't stomp the window the user just set."""
        if self._contrast is None or not (0 <= ch < len(self._mask)):
            return
        self._contrast.set_manual(ch, lo, hi)
        self._refresh()

    def follow_channel_window(self, ch: int, lo: float, hi: float):
        """Render *ch* with the window the OWNING ARRAY VIEWER resolved, and repaint (IMA-261).

        The sink half of the one-owner contract. It does NOT latch the channel manual: ndv
        autoscales on its own, at open and on every data change, so recording that as a user
        gesture would kill the plate's own auto-contrast before the user had touched anything,
        and would outrank the per-region scope the user explicitly selected. See
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
        """A user gesture: repaint NOW at display resolution, then land full-res once it settles.

        The invariant is that no gesture ever touches the full (C, 2816, 4224) canvas — a slider
        drag composites the sub-sampled view the screen can actually show (a few thousand px),
        and the single full-res pass is coalesced behind the last event.
        """
        self.recomposite(quick=True)
        self._full.start(150)
        self._refresh_loupe()

    def _refresh_loupe(self):
        """Re-render the loupe inset under the contrast that just changed (IMA-242).

        The inset holds a rendered QImage, so repainting the plate alone would leave it showing
        the PRE-drag contrast until the cursor happened to move — the plate and the magnifier of
        the plate disagreeing about the same pixels. Re-issuing the request is cheap: the worker
        memoises crops, so this re-colours the bytes it already has and re-reads nothing.
        """
        if self._loupe is None or self._loupe_worker is None:
            return
        try:
            self._request_loupe(self._loupe["x"], self._loupe["y"])
        except Exception:
            pass          # a contrast drag must never fail because the loupe could not re-render

    def _on_full_timeout(self):
        """The coalescing timer fired. Guarded: a pending recomposite must not outlive the widget
        (the plate is torn down and rebuilt on every open, and the timer is queued, not immediate)."""
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
        """Forget a layer's retained pixels — its store, its recomposite and its painted canvas.

        Called before a run streams into a layer that already has one. Without it a mosaic re-run
        that lands FEWER fields (a smaller selection, a failed well) would leave the previous run's
        FOVs standing inside the same cell, blended into the new ones with no way to tell which is
        which — and the ~95 MB store would never be freed either.
        """
        self._store.pop(layer, None)
        self._invalidate_pixels(layer)
        self._final_arr.pop(layer, None)
        self._op_final.pop(layer, None)
        self._tiles_by_layer.pop(layer, None)
        if layer == "raw":
            self._canvas.fill(QColor(_BG))
        else:
            self._op_canvas.pop(layer, None)
        if layer == self._active:
            self._final = None
            self._scaled = None
            self.update()

    # -- data in --
    def add_tile(self, ri: int, ci: int, well_id: str, tile: np.ndarray, layer: str = "raw",
                 box=None):
        """Take one PER-CHANNEL tile ``(C, h, w)`` (native dtype), retain it in the layer's store,
        feed the running contrast, and re-composite that whole cell.

        ``box`` is ``(top, left, h, w)`` in cell px for a multi-FOV mosaic (IMA-187): the tile is
        one FIELD inside the cell, so it is written at that offset and the cell is re-composited
        around it, which is what makes the seams between neighbouring FOVs update as they land.
        ``box=None`` is the historical single-tile path — one field fills the cell at (0, 0).

        CONTRAST IS FED THE TILE, NEVER THE STORE SLICE. A mosaic cell is zero-padded wherever no
        FOV lands (margins, gaps between fields); feeding those zeros to the running histogram
        would pin the 1st percentile at 0 for the whole plate and silently wash every well out.
        Only real acquired pixels get a vote.
        """
        if (ri, ci) not in self._by_rc:    # ignore a stale tile from a retired run / foreign cell
            return
        store = self._store_for(layer)
        if store is None:                  # no channel axis declared yet -> nothing to composite
            return
        tile = np.asarray(tile)
        y0, x0 = ri * _CELL, ci * _CELL
        top, left = (int(box[0]), int(box[1])) if box is not None else (0, 0)
        th, tw = tile.shape[1], tile.shape[2]      # place by ACTUAL shape: a field smaller than
        store[:, y0 + top:y0 + top + th,           # the cell must not broadcast-crash
              x0 + left:x0 + left + tw] = tile
        self._invalidate_pixels(layer, (ri, ci))   # these pixels are new: nothing derived survives
        for c_i in range(tile.shape[0]):
            self._contrast.add(c_i, tile[c_i])     # real FOV pixels only — see the docstring
        wins = self.channel_windows()     # ONE window per channel, owned by napari (see above)
        cell = composite(store[:, y0:y0 + _CELL, x0:x0 + _CELL], self._colors, wins, self._mask)
        img = QImage(cell.data, _CELL, _CELL, 3 * _CELL, QImage.Format_RGB888)
        p = QPainter(self._canvas_for(layer))
        p.drawImage(x0, y0, img)           # drawImage COPIES, so `cell` may die after p.end()
        p.end()
        if self._op_final.pop(layer, None) is not None:   # a new tile supersedes the old recomposite
            self._final_arr.pop(layer, None)              # -> back to the streamed canvas
            if layer == self._active:
                self._final = None
        self._tiles.add((ri, ci))
        self._tiles_by_layer.setdefault(layer, set()).add((ri, ci))   # per-layer: drives the grey dots
        if layer == self._active:         # only the shown layer needs a repaint / cache rebuild
            self._scaled = None
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
        if layer == self._active:
            self._final = img
            self._scaled = None       # source changed -> the scaled cache is stale
            self.update()

    def set_active_layer(self, layer: str):
        """Show a layer (LayersTab toggle/reorder). Swaps in its montage + streamed canvas."""
        self._active = layer
        self._final = self._op_final.get(layer)   # None for "raw" -> falls back to the base canvas
        self._scaled = None
        self.update()
        if layer in self._store:     # bring it up to the CURRENT mask/windows: its canvas was blitted
            self.recomposite(layer)  # cell-by-cell, with whatever mask was set when each tile landed

    def drop_layer(self, layer: str):
        """Forget a layer entirely and FREE its canvas (IMA-205: an exploration tab's layers die
        with the tab). ``_canvas_for`` lazily allocates a full plate-sized RGB888 image per layer
        (nc*_CELL x nr*_CELL — tens of MB on a 1536wp), so without this a closed tab's montage
        stays resident forever. Falls back to the base layer if the dropped one was showing.

        The per-channel STORE (IMA-206) is the bigger half — ~95 MB of retained (C, H, W) pixels
        per layer — so it goes too. Dropping the canvas while the store survived would look like a
        fix and leak the majority of the memory."""
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

    def status_snapshot(self) -> dict:
        """Copy of the per-well status map — the window saves one per exploration tab so a tab's
        amber/failed dots follow its own run, not whatever ran last (IMA-205)."""
        return dict(self._status)

    def set_status_map(self, status: dict):
        """Restore a snapshot. Foreign keys are ignored, same as ``set_status``."""
        for rc, state in status.items():
            if rc in self._status:
                self._status[rc] = state
        self.update()

    def select(self, ri: int, ci: int):
        """Move the red box to a well (driven by the ndviewer FOV slider)."""
        self._sel = (ri, ci)
        self.update()

    def resizeEvent(self, e):
        self._user_view = False       # a resize re-fits (drop any zoom/pan)
        self._fit()
        self.update()

    # -- mouse: wheel-zoom anchored at cursor, left-drag pan (Hongquan's navigator gestures),
    #    and press-and-hold loupe (IMA-208). The left button now means three different things
    #    depending on TIMING, so the rules live here as a diagram rather than as scattered flags:
    #
    #                        ┌───────────────────────────────────────────┐
    #                        │                  IDLE                     │
    #                        └──────────────────┬────────────────────────┘
    #          left-press on an acquired cell   │   (off-plate / empty: never arms)
    #                                           ▼
    #                        ┌───────────────────────────────────────────┐
    #                        │  ARMED   _hold running (_LOUPE_HOLD_MS)   │
    #                        │  cursor must stay within _LOUPE_SLOP px   │
    #                        └───┬───────────────────────┬───────────────┘
    #          move > slop       │                       │  timer fires
    #          (kill the timer)  │                       │
    #                            ▼                       ▼
    #                  ┌──────────────────┐   ┌──────────────────────────┐
    #                  │       PAN        │   │          LOUPE           │
    #                  │  drag the plate  │   │  inset follows cursor;   │
    #                  │  (unchanged)     │   │  pan is DEAD while up;   │
    #                  └────────┬─────────┘   │  hover + wheel suppressed│
    #                           │             └────────────┬─────────────┘
    #                           │ release                  │ release / dragged off the widget
    #                           │                          │ / leave / focus-out
    #                           ▼                          ▼
    #                        ┌───────────────────────────────────────────┐
    #                        │                  IDLE                     │
    #                        └───────────────────────────────────────────┘
    #
    #    Two edges worth stating because they are easy to regress:
    #      * SLOW PAN stays a pan. Press, dwell past the timer, then drag — the timer only runs
    #        while the cursor is still, and any move past the slop kills it. A press that has
    #        already become a loupe is dismissed on release, so the next drag pans normally.
    #      * DOUBLE-CLICK must cancel the timer. Qt delivers press/release/dblclick/release, and
    #        the second press re-arms; without the cancel you would open the detail viewer AND
    #        raise a loupe from one gesture.
    def _cell(self, x, y):
        if self._layout is not None:
            return self._freeform_cell(x, y)
        px, py = x - (self._ox + _HDR), y - (self._oy + _COLH)
        return well_at(self._rows, self._cols, self._by_rc, px, py, self._cd)

    def _freeform_cell(self, x, y):
        """Hit-test a geometrically placed holder: the first cell whose own rect contains (x, y).

        Placed cells are tested FIRST and in reverse paint order, so a click in the small area
        where two regions' bounding boxes overlap resolves to the one drawn on top — the same
        last-one-wins rule ``_fov_at`` uses inside a mosaic. Nominal (empty-slot) rects are only
        consulted when no real region claims the point, so an empty slot can never shadow a region
        that overlaps it. Freeform holders have a handful of cells, so a linear scan is free.
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

    # -- selection API (IMA-221) --
    def selected_wells(self) -> list:
        """The selection as acquired well ids, in plate row-major order."""
        return [self._by_rc[rc] for rc in sorted(self._selection)]

    def clear_selection(self):
        """Drop the whole selection and tell listeners (used on re-ingest)."""
        if self._selection:
            self._selection = set()
            self.selectionChanged.emit([])
            self.update()

    def select_all(self):
        """Select every occupied well (the Select all button and Cmd/Ctrl-A)."""
        self._selection = set(self._by_rc.keys())
        self.selectionChanged.emit(self.selected_wells())
        self.update()

    def highlight_regions(self, region_ids):
        """Move the blue wash onto *region_ids* — used when the user clicks an OPEN VIEW so the
        plate shows which regions that window holds. Same wash the manual selection uses."""
        want = set(region_ids or [])
        self._selection = {rc for rc, rid in self._by_rc.items() if rid in want}
        self.selectionChanged.emit(self.selected_wells())
        self.update()

    def set_view_hues(self, entries):
        """Colour-code the OPEN VIEWS on the plate: *entries* is a list of ``(region_ids, QColor)``,
        one per open window/thread. Each view's wells get that view's hue, so overlapping/adjacent
        views are told apart at a glance (Julio's "hue the different view threads"). Painted UNDER
        the blue focus/selection wash, which still marks the one active view. Empty list clears it."""
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
        # Shift owns MULTI-well selection (IMA-221): Shift-drag opens the wells you box, Shift+Alt
        # unions, Cmd/Ctrl-click toggles one. A plain click also selects, but it REPLACES rather
        # than toggles (file-manager semantics, added in 2b8fbc5), which is what keeps double-click
        # safe: Qt delivers press+release BEFORE mouseDoubleClickEvent, so a plain-click TOGGLE
        # would silently flip a well every time you opened one. Replace is idempotent, toggle is
        # not, and that difference is the whole reason this is safe.
        # Corrected 2026-07-28: this comment still said "keeping selection off the plain click",
        # which the plain-click replace at the bottom of mouseReleaseEvent had already contradicted.
        # (Ctrl is out: on macOS Ctrl+click is right-click and Qt maps Cmd -> ControlModifier.)
        if e.modifiers() & Qt.ShiftModifier:
            self._marquee = (e.x(), e.y(), e.x(), e.y())
            self._marquee_add = bool(e.modifiers() & Qt.AltModifier)   # Shift+Alt = union
            self._press = None                                          # ...so this drag never pans
            self._panning = False
            self.update()
            return          # a Shift-drag is a selection, never a pan and never a loupe
        # Cmd/Ctrl-click = TOGGLE this well in the batch selection (Linux-file-manager add/remove).
        # On macOS Cmd maps to ControlModifier, and a real Ctrl+click is a right-click (not
        # LeftButton), so this only ever fires for the intended gesture. Committed on RELEASE so a
        # cmd-drag can still not-select if the user changes their mind, and so it never pans.
        if e.modifiers() & Qt.ControlModifier:
            self._ctrl_click = (e.x(), e.y())
            self._press = None
            self._panning = False
            return
        self._press = (e.x(), e.y(), self._ox, self._oy)
        self._panning = False
        c = self._cell(e.x(), e.y())
        if self._loupe_src is not None and c and c["well_id"]:   # ARM (never off-plate/empty)
            self._hold.start()

    def mouseMoveEvent(self, e):
        if self._loupe is not None:                  # LOUPE: the inset tracks; panning is dead
            # Drag off the widget and the loupe must go. leaveEvent CANNOT do this: Qt grabs the
            # mouse for the duration of a press, so no leave is delivered until the button comes
            # up — the inset used to stay pinned over the neighbouring pane showing stale pixels.
            # The grab is also why this works: move events keep arriving, with coordinates
            # outside rect(), which is the signal.
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
        c = self._cell(e.x(), e.y())                 # hover (only when not dragging)
        new_hover = (c["row_index"], c["col_index"]) if c else None
        if new_hover == self._hover:                 # still the same cell -> no repaint (kills the
            return                                   # per-pixel repaint storm; only cross-cell moves repaint)
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
        # Only the LEFT release commits a selection. The gesture is opened by a left press, but Qt
        # delivers a release for whichever button went up — so a right-click while a Shift-drag is
        # in flight would otherwise silently toggle/replace the selection.
        if self._marquee is not None and e.button() == Qt.LeftButton:
            x0, y0, x1, y1 = self._marquee
            add, self._marquee, self._marquee_add = self._marquee_add, None, False
            dragged = abs(x1 - x0) + abs(y1 - y0) > _CLICK_SLOP
            if not dragged:                                     # Shift+CLICK -> toggle ONE well
                hit = self._cell(x1, y1)
                if hit and hit["well_id"]:
                    self._selection ^= {(hit["row_index"], hit["col_index"])}
                self.selectionChanged.emit(self.selected_wells())
            else:
                # Shift-DRAG opens a WINDOW over the boxed regions (the meeting's "shift-drag a box
                # -> a floating view"). It does NOT leave a persistent wash on the plate: you see
                # that set in the new window's region slider, so a lingering highlight is just the
                # "stays selected forever" clutter Julio flagged. Emit the window request, then
                # clear the wash. Shift+Alt still UNIONS into the batch selection instead of opening.
                boxed = [self._by_rc[rc] for rc in sorted(set(self._cells_in(x0, y0, x1, y1)))]
                if add:
                    self._selection |= set(self._cells_in(x0, y0, x1, y1))
                    self.selectionChanged.emit(self.selected_wells())
                else:
                    self.marqueeSelected.emit(boxed)            # open a window over the box
                    if self._selection:                         # drop any lingering batch wash
                        self._selection = set()
                        self.selectionChanged.emit([])
            self.update()
            self._press = None
            self._panning = False
            self._dismiss_loupe()
            return
        # Cmd/Ctrl-click TOGGLE (Linux-style add/remove to the batch selection).
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
        # Plain CLICK (no modifier, no pan, no loupe) = select ONLY this well, or clear on empty.
        # This is the deselect path that was missing: without it a batch selection could never be
        # dropped by clicking, so it "stayed selected forever". A plain DRAG still pans (guarded by
        # _panning), and a hold that raised the loupe does not select (had_loupe).
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
        self._dismiss_loupe()                        # release always dismisses

    def leaveEvent(self, e):
        self._hold.stop()                            # cursor left mid-hold: release may never come
        self._dismiss_loupe()
        self._hover = None
        # Drop any in-flight marquee too. If the grab is lost mid-drag (modal dialog, alt-tab) no
        # release ever arrives, and a stranded _marquee both paints a dashed rect forever and makes
        # wheelEvent's mid-marquee guard disable zoom permanently.
        self._marquee = None
        self._marquee_add = False
        self.hovered.emit("")
        self.update()

    def keyPressEvent(self, e):
        """Keyboard selection, Linux-file-manager style. Cmd/Ctrl-A selects every well; Escape
        clears. Focus is ClickFocus, so these arrive once the user has clicked the plate."""
        if (e.modifiers() & Qt.ControlModifier) and e.key() == Qt.Key_A:
            self.select_all()
            return
        if e.key() == Qt.Key_Escape:
            self.clear_selection()
            return
        super().keyPressEvent(e)

    def set_mosaic_boxes(self, boxes: dict):
        """Adopt the per-FOV cell boxes so a double-click can resolve WHICH FOV was hit.

        Also tells the paint path WHICH cells hold a letterboxed mosaic rather than a
        cell-filling single tile (see :meth:`_cell_source`), so the two can never disagree about
        the same cell: they read one dict.
        """
        self._boxes = dict(boxes or {})
        self._boxed_regions = {r for r, _f in self._boxes}
        self.update()

    # -- carrier geometry (IMA-220 -> IMA-253: DRAWN, not blitted) --
    def set_carrier(self, plate, images_dir=None):
        """Adopt *plate*'s geometry so the holder can be DRAWN behind the cells.

        This used to blit Squid's carrier PHOTOGRAPH. It no longer does, and the reason is
        registration, not taste. A PNG lives in its own pixel space and has to be brought into the
        cell grid's space through three calibration constants (``a1_x_pixel``, ``a1_x_mm``,
        ``mm_per_pixel``) that must agree with the geometry the cells are laid out from. When they
        disagree, nothing raises — you get a plausible picture with the wells in the wrong places,
        which is exactly what shipped, and it is unfixable in general for a FREEFORM holder because
        there is no photograph of "two tissues wherever the operator happened to put them".

        Drawing the outline, the slot/well boundaries and the empty-vs-occupied state from
        :class:`~squidmip._plate.PlateGeometry` puts the holder in the SAME coordinate system as
        the cells, so it cannot misalign, it cannot vanish on pan or zoom (there is no separately
        positioned blit to drift), and an acquisition with no artwork on disk renders identically
        to one with artwork. ``plate.art()`` and the whole PNG registry stay in ``_plate.py`` for
        an optional skin; they are simply not on this path.

        *images_dir* is accepted and ignored, so callers that passed one still work.
        """
        self._carrier = getattr(plate, "geometry", None) if plate is not None else None
        try:
            from squidmip._plate import SlideCarrier
            self._carrier_slide = isinstance(plate, SlideCarrier)
        except Exception:
            self._carrier_slide = False
        # NO true-scale SLIDE ART (Julio, 2026-07-23). The slide-art layout drew glass slides at
        # true micron scale (a 25 mm slide dwarfing an 8 mm tissue) and placed the mosaics at their
        # real stage positions — which stacked two tissues into a tall, tiny, uneven column and
        # "looked like shite". The plate now keeps its EVEN carrier layout (``even_carrier_layout``,
        # equal cells side by side) set at construction, so this no longer overrides ``self._layout``
        # and draws no slide bodies. Even, horizontal, non-overlapping cells beat true-scale slides
        # for a browse view, whatever each tissue's geometry.
        self._slides = None
        self.update()

    # -- cell rectangles: the ONE place a (row, col) becomes widget pixels (IMA-253) --
    def _cell_rect(self, ri: int, ci: int) -> tuple:
        """Widget-pixel ``(x, y, w, h)`` of cell (ri, ci) at the current zoom/pan.

        Uniform plates return the historical ``(ax + ci*cd, ay + ri*cd, cd, cd)`` exactly. A
        freeform holder returns the region's own rectangle: its mosaic's bounding box, scaled by
        the same single transform for every region, so relative offset and relative scale are
        preserved and two regions of different size get different-sized cells.
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

        The store keeps every cell as one ``_CELL`` x ``_CELL`` square. A MOSAIC is LETTERBOXED into
        it (``_placement.cell_boxes`` centres it, preserving aspect), so the bars must be excluded
        or the mosaic would be stretched back into them. A single tile — one FOV, or a region
        operator's already-fused result — FILLS the block, so the whole block is the source. Which
        of the two a cell holds is read from ``self._boxes``, the same dict the tiles were placed
        by, so the blit and the pixels cannot disagree.

        Since the cell rect and the letterbox come from the SAME aspect ratio, the inner box is
        recoverable from the rect alone: no second bookkeeping table to fall out of sync.
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

    def _tiled_region(self) -> "QRegion":
        """The cells that HAVE an image on the active layer, as a QRegion at pan origin (0, 0).

        The montage canvas is opaque _BG wherever no tile landed, so blitting it whole would paint
        the carrier art out. Clipping the blit to this region is what lets the background show
        through the empty wells. Cached and only translated on pan: a full 1536wp is 1536 rects,
        and rebuilding that union on every hover repaint would be the one thing that makes the
        plate feel slow.
        """
        cells = self._tiles_by_layer.get(self._active, set())
        key = (self._cd, self._active, len(cells))
        if self._tile_rgn is None or self._tile_rgn_key != key:
            cd = self._cd
            rgn = QRegion()
            for ri, ci in cells:
                rgn = rgn.united(QRegion(int(ci * cd), int(ri * cd),
                                         int(cd) + 1, int(cd) + 1))   # +1: no hairline seams
            self._tile_rgn, self._tile_rgn_key = rgn, key
        return self._tile_rgn

    def _fov_at(self, c: dict, e) -> int:
        """FOV index under the cursor within cell *c*, or 0 when there is no mosaic to resolve.

        Inverts the placement transform: find where the click landed inside the cell (in _CELL
        units), then pick the FOV whose box contains it. Boxes overlap by ~9% at the seams, so
        the LAST match wins — matching the draw order in ``_OperatorWorker._on_well``, where
        later FOVs paint over earlier ones. Without that agreement a click in a seam would open
        a different FOV than the one visibly on top.
        """
        region = c["well_id"]
        if not region or not self._boxes:
            return 0
        ri, ci = c["row_index"], c["col_index"]
        rx, ry, rw, rh = self._cell_rect(ri, ci)
        sx, sy, sw, sh = self._cell_source(ri, ci)
        if not (rw > 0 and rh > 0):
            return 0
        # position within the cell, normalised to the _CELL-px space the boxes live in. Going via
        # the cell's SOURCE rect is what keeps the hit-test agreeing with the blit on a freeform
        # holder, where the drawn rect is the mosaic's box and not the whole square block.
        fx = (e.x() - rx) / rw * sw + (sx - ci * _CELL)
        fy = (e.y() - ry) / rh * sh + (sy - ri * _CELL)
        hit = 0
        for (r, fov), (top, left, h, w) in self._boxes.items():
            if r == region and top <= fy < top + h and left <= fx < left + w:
                hit = fov
        return hit

    def focusOutEvent(self, e):
        # Only reachable because __init__ sets ClickFocus: with the default NoFocus this widget
        # never held focus, so this handler was dead code pretending to cover "window
        # deactivated mid-hold". A press now focuses the plate, so losing focus is a real signal.
        self._hold.stop()
        self._dismiss_loupe()
        super().focusOutEvent(e)

    def mouseDoubleClickEvent(self, e):
        # Qt sends press/release/dblclick — the second press already re-armed the hold timer, so
        # kill it here or one double-click both opens the well AND raises a loupe.
        self._hold.stop()
        self._dismiss_loupe()
        c = self._cell(e.x(), e.y())
        if c and c["well_id"]:
            self.wellActivated.emit(c["well_id"], self._fov_at(c, e))

    # -- paint --
    def paintEvent(self, _):
        if not self._user_view:          # auto-fit until the user first zooms/pans
            self._fit()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(_BG))
        cd, nr, nc = self._cd, self._nr, self._nc
        ax, ay = self._ox + _HDR, self._oy + _COLH   # plate top-left (after label margins)
        W, H = nc * cd, nr * cd
        tiled = self._tiles_by_layer.get(self._active, set())
        # THE HOLDER (IMA-253), behind everything: drawn from the plate's own geometry, in the
        # cells' own coordinate system. No photograph, so nothing to calibrate and nothing to
        # drift on pan/zoom; see set_carrier.
        self._paint_carrier(p, tiled)
        if self._layout is not None:
            # FREEFORM: each region's cell is its own rectangle, so the montage is blitted per
            # cell rather than as one grid-aligned image. A handful of regions, one drawImage each.
            src = self._active_source()
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            for rc in sorted(tiled):
                if rc not in self._by_rc:
                    continue
                p.drawImage(QRectF(*self._cell_rect(*rc)), src, QRectF(*self._cell_source(*rc)))
        else:
            # Blit the montage from a cached pixmap scaled to the current zoom. The expensive
            # smooth resample runs ONCE per zoom/source-change (not every repaint) — pan/hover
            # just re-blit.
            w, h = max(1, int(W)), max(1, int(H))
            if (self._scaled is None or self._scaled_cd != cd
                    or self._scaled.width() != w or self._scaled.height() != h):
                self._scaled = QPixmap.fromImage(self._active_source()).scaled(
                    w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self._scaled_cd = cd
            if len(tiled) < nr * nc:
                # The montage canvas is opaque _BG wherever no tile landed, so let it cover only
                # the cells that actually have pixels — otherwise it paints the holder out.
                p.save()
                p.setClipRegion(self._tiled_region().translated(int(ax), int(ay)))
                p.drawPixmap(int(ax), int(ay), self._scaled)
                p.restore()
            else:
                p.drawPixmap(int(ax), int(ay), self._scaled)

        # DEEP ZOOM, on top of the montage and under every annotation. Ordering is the whole
        # design: the thumbnail has already painted, so a tile that has not arrived yet leaves the
        # coarse pixels showing rather than a hole, and the view sharpens in place as tiles land.
        # Nothing here blocks -- misses are queued for the fetcher and the next repaint draws them.
        if self._tile_cache is not None:
            self._paint_tiles(p)

        # per-cell DOT over the WHOLE plate grid (so a sparse acquisition still shows the full plate
        # shape — e.g. 32x48 for 1536, 16x24 for 384 — with grey dots on the un-acquired wells):
        # amber = processing, red x = failed, GREY = no image on the active layer, no dot once a cell
        # HAS an image (the image speaks for itself). Dot size is a capped absolute size.
        d = min(max(3.0, cd * 0.36), 15.0)
        active_tiles = self._tiles_by_layer.get(self._active, set())
        for ri in range(nr):
            for ci in range(nc):
                state = self._status.get((ri, ci), "empty")
                has_img = (ri, ci) in active_tiles
                x0, y0, cw, ch = self._cell_rect(ri, ci)
                ex, ey = int(x0 + (cw - d) / 2), int(y0 + (ch - d) / 2)
                if state == "processing":                   # amber dot
                    p.setPen(Qt.NoPen)
                    p.setBrush(_STATUS["processing"])
                    p.drawEllipse(ex, ey, int(d), int(d))
                elif state == "failed":                     # red x within the dot box
                    p.setPen(QPen(_STATUS["failed"], max(1.5, min(cd * 0.09, 3.0))))
                    p.drawLine(ex, ey, ex + int(d), ey + int(d))
                    p.drawLine(ex + int(d), ey, ex, ey + int(d))
                elif not has_img:                           # grey dot: an empty plate position
                    p.setPen(Qt.NoPen)
                    p.setBrush(_STATUS["empty"])
                    p.drawEllipse(ex, ey, int(d), int(d))
                # else: has an image on the active layer -> no dot
        p.setBrush(Qt.NoBrush)

        if self._view_hues:           # PER-VIEW HUES (under the focus wash): each open window/thread
            p.setPen(Qt.NoPen)         # tints its wells in its own colour, so views are told apart.
            for rcs, color in self._view_hues:
                p.setBrush(color)
                for ri, ci in rcs:
                    rx, ry, rw, rh = self._cell_rect(ri, ci)
                    p.drawRect(int(rx), int(ry), int(rw), int(rh))
            p.setBrush(Qt.NoBrush)

        if self._selection:            # SELECTED / focused-view wells = a light blue wash. More
            p.setPen(Qt.NoPen)         # transparent than before (Julio), and it FOLLOWS the open
            p.setBrush(_VIEW_WASH)     # view you click (highlight_regions), as well as manual picks.
            for ri, ci in self._selection:
                rx, ry, rw, rh = self._cell_rect(ri, ci)
                p.drawRect(int(rx), int(ry), int(rw), int(rh))
            p.setBrush(Qt.NoBrush)

        if self._layout is None:
            p.setPen(QPen(_GRID, 3))   # black grid lines between wells (multi-FOV mosaics sit INSIDE a cell)
            for c in range(nc + 1):
                p.drawLine(int(ax + c * cd), int(ay), int(ax + c * cd), int(ay + H))
            for r in range(nr + 1):
                p.drawLine(int(ax), int(ay + r * cd), int(ax + W), int(ay + r * cd))
        # (a freeform holder has no shared grid lines to draw — its cells are individually placed
        #  rectangles, and _paint_carrier already outlined each one.)
        p.setFont(QFont("Helvetica Neue", 11, QFont.DemiBold))
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
        if self._sel is not None:          # the CURRENT well in the detail viewer = a red BOX
            p.setPen(QPen(_RED, 2))
            p.setBrush(Qt.NoBrush)
            sx, sy, sw, sh = self._cell_rect(*self._sel)
            p.drawRect(int(sx), int(sy), int(sw), int(sh))
        if self._hover is not None:        # where the cursor is, moving around the plate = a red DOT
            ri, ci = self._hover           # SAME geometry as the status dots -> overlays them exactly
            x0, y0, hw, hh = self._cell_rect(ri, ci)
            ex, ey = int(x0 + (hw - d) / 2), int(y0 + (hh - d) / 2)
            p.setPen(Qt.NoPen)
            p.setBrush(_RED)
            p.drawEllipse(ex, ey, int(d), int(d))
        if self._marquee is not None:      # live drag rectangle while Shift-dragging
            mx0, my0, mx1, my1 = self._marquee
            p.setPen(QPen(_ACCENT, 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(int(min(mx0, mx1)), int(min(my0, my1)),
                       int(abs(mx1 - mx0)), int(abs(my1 - my0)))
        if self._loupe is not None:        # press-and-hold magnifier, over everything else
            self._paint_loupe(p)
        # a fine outer white frame around the whole plate view
        p.setPen(QPen(QColor("#c9d1d9"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.end()

    def _paint_carrier(self, p: QPainter, tiled: set):
        """Draw the sample holder: body outline, per-cell boundary, empty vs occupied (IMA-253).

        Everything here comes out of the geometry the cells themselves are placed from, so there is
        exactly one coordinate system and the holder cannot drift out of register with the wells —
        which is the failure a separately-positioned photograph kept producing, and could not
        avoid. It also means an acquisition with no artwork on disk renders IDENTICALLY to one with
        artwork, because artwork is no longer consulted.

        Three states, deliberately distinct, because "which slots are empty" was the exact question
        the photograph answered badly:

            occupied, imaged   the pixels themselves (drawn over this)
            occupied, waiting  a solid accent-tinted boundary + fill
            empty slot         a DASHED, dim boundary and no fill

        Skipped entirely below a few px per cell: at 1536-well zoom the boundaries are smaller than
        the status dots, so drawing 1536 of them would cost a repaint and show nothing.
        """
        if self._carrier is None:
            return
        cd = self._cd
        ax, ay = self._ox + _HDR, self._oy + _COLH
        if self._slides is not None:
            # SLIDE ACQUISITION (IMA-265): real glass slides at true size, side by side, drawn by
            # _slide_art from the same grid units the tissue cells are placed in. No generic
            # carrier body -- the slides ARE the holder, and the tissue mosaics paint on top of
            # them through the ordinary cell path (so every gesture is untouched).
            from squidmip._slide_art import paint_slides
            slide_rects_px = [(ax + s[0] * cd, ay + s[1] * cd, s[2] * cd, s[3] * cd)
                              for s in self._slides]
            paint_slides(p, slide_rects_px)
            if cd < 6.0:
                return
            self._paint_carrier_cells(p, tiled)
            return
        # The holder BODY: the union of every cell rectangle, padded by the margin the geometry
        # implies (half a pitch beyond the outer cell centres on a well plate).
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
        # An orientation cue instead of a picture of one: the A1 / first-slot corner is chamfered,
        # the way a real plate's notched corner reads.
        p.setPen(QPen(QColor(120, 132, 150), 2))
        ch = min(14.0, pad * 2.0)
        p.drawLine(int(body.left()), int(body.top() + ch), int(body.left() + ch), int(body.top()))
        if cd < 6.0:                     # boundaries smaller than the status dots: not worth it
            return
        self._paint_carrier_cells(p, tiled)

    def _paint_carrier_cells(self, p: QPainter, tiled: set):
        """The per-cell occupied/empty boundaries, shared by the well plate and the slide holder.

        A cell that already has imaged pixels is left alone (the pixels speak); an occupied but
        un-imaged cell gets a solid accent-tinted boundary; an empty slot gets a dashed dim one.
        The SHAPE differs: a well is round, a slide slot / tissue region is rectangular.
        """
        cd = self._cd
        for ri in range(self._nr):
            for ci in range(self._nc):
                rx, ry, rw, rh = self._cell_rect(ri, ci)
                occupied = (ri, ci) in self._by_rc
                if occupied and (ri, ci) in tiled:
                    continue             # the acquired pixels will cover it; do not tint them
                if occupied:
                    p.setPen(QPen(_ACCENT, max(1.0, min(cd * 0.03, 2.0))))
                    p.setBrush(QColor(56, 139, 253, 40))
                else:
                    p.setPen(QPen(QColor(74, 84, 100), 1, Qt.DashLine))
                    p.setBrush(Qt.NoBrush)
                if self._carrier_slide:  # a slide slot is a rectangle; a well is round
                    p.drawRect(QRectF(rx, ry, rw, rh))
                else:
                    # Well diameter relative to pitch, straight from sample_formats.csv, so a 96wp
                    # reads as fat wells and a 1536wp as pinpricks — the real difference between them.
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
                s, s, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation))   # 1:1-ish: no smoothing
            p.restore()
        else:
            p.setPen(_MUTED)
            p.setFont(QFont("Helvetica Neue", 11))
            p.drawText(bx, by, s, s, Qt.AlignCenter | Qt.TextWordWrap,
                       self._loupe_note or "reading …")
        geo = self._loupe_geometry(x, y)
        if geo is not None and self._loupe_img is not None:
            _w, _l, _r, s_loupe, mag = geo
            um_px = loupe_um_per_screen_px(getattr(self._loupe_src, "pixel_size_um", None), s_loupe)
            p.setFont(QFont("Helvetica Neue", 10, QFont.DemiBold))
            if um_px is None:
                # No trustworthy pixel size: say so rather than draw a bar that would be fiction.
                p.setPen(_MUTED)
                p.drawText(bx + 8, by + s - 10, "scale unknown")
            else:
                target = _nice_scale_um(um_px * (s * 0.4))     # ~40% of the inset, rounded to 1/2/5
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

    Pure geometry, delegated to :mod:`squidmip._placement` (which is Qt-free and unit-tested
    against exact pixel offsets). Returns ``{}`` when the acquisition has no stage positions or
    no pixel size, which is the signal for the caller to keep the historical single-tile path —
    a mosaic is simply not derivable without both, and guessing would draw a wrong picture.

    Placement failures for ONE region are contained: that region falls back to single-tile
    rendering rather than aborting a whole-plate run. The reader has already fail-loud checked
    the CSV/filename agreement, so anything reaching here is a genuine per-region oddity
    (e.g. a region with images but no coordinate rows).
    """
    from squidmip._placement import cell_boxes, fov_offsets_px

    positions = meta.get("fov_positions_um") or {}
    if not positions or meta.get("pixel_size_um") in (None, 0):
        return {}
    frame_shape = meta["frame_shape"]
    out: dict = {}
    for region in meta["regions"]:
        fovs = meta["fovs_per_region"][region]
        if len(fovs) < 2:
            continue                     # a single-FOV well fills its cell; no mosaic needed
        try:
            offsets = fov_offsets_px(positions, region, fovs, meta.get("pixel_size_um"))
            for fov, box in cell_boxes(offsets, frame_shape, _CELL).items():
                out[(region, fov)] = box
        except (KeyError, ValueError):
            continue                     # this region renders single-tile; the rest still mosaic
    return out


def region_mosaic_extent_px(meta: dict, regions: Optional[list] = None) -> Optional[tuple]:
    """Full-resolution ``(height, width)`` bounding box of the mosaics a REGION operator will fuse.

    IMA-245. A region operator (``available_region_operators()``) yields ONE fused mosaic per
    region, whose extent is the bounding box of the region's coordinate-placed frames — NOT the
    frame shape. Anything that has to size a surface for that result (the array viewer's canvas,
    the push planes fed into it) must ask this, or it sizes a mosaic as a frame.

    Returns the max extent over *regions* (``None`` = every region in the acquisition), because
    one array viewer serves the whole run and its canvas is declared once. Returns ``None`` when
    the acquisition carries no stage positions / no pixel size — the same "not derivable, do not
    guess" signal :func:`_mosaic_boxes` returns ``{}`` for.
    """
    from squidmip._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    if not positions or meta.get("pixel_size_um") in (None, 0):
        return None
    frame_shape = meta["frame_shape"]
    scoped = list(meta["regions"]) if regions is None else list(regions)
    best = None
    for region in scoped:
        fovs = meta["fovs_per_region"].get(region) or []
        if not fovs:
            continue
        try:
            offsets = fov_offsets_px(positions, region, fovs, meta.get("pixel_size_um"))
            h, w = mosaic_extent_px(offsets, frame_shape)
        except (KeyError, ValueError):
            continue                     # this region contributes nothing; the rest still size it
        best = (h, w) if best is None else (max(best[0], h), max(best[1], w))
    return best


def push_shape_for(meta: dict, region_op: bool, regions: Optional[list] = None) -> tuple:
    """The ``(height, width)`` of every plane pushed into the array viewer for this run.

    A per-FOV operator pushes a FRAME, so the surface is the frame shape. A REGION operator pushes
    a whole-region MOSAIC, so the surface is the mosaic EXTENT. Either is scaled into a ``_PUSH_PX``
    box PRESERVING ASPECT — a freeform 27-FOV strip is not a square, and squashing it into one
    (which is what a fixed ``(_PUSH_PX, _PUSH_PX)`` surface did) is how it arrives unrecognisable.

    Aspect is preserved the same way :func:`squidmip._placement.cell_boxes` preserves it for the
    plate thumbnail, so the plate and the array viewer describe one geometry rather than two.
    Falls back to the square when the extent is not derivable (no stage positions / pixel size);
    the caller reports that fallback rather than showing a silently squashed mosaic.
    """
    extent = region_mosaic_extent_px(meta, regions) if region_op else None
    if extent is None:                              # per-FOV op, or a region op with no geometry
        extent = tuple(meta["frame_shape"])
    mh, mw = int(extent[0]), int(extent[1])
    s = min(_PUSH_PX / mh, _PUSH_PX / mw, 1.0)     # never UPSCALE: a push is a bounded thumbnail
    return (max(1, int(round(mh * s))), max(1, int(round(mw * s))))


def _fit_letterboxed(a: np.ndarray, h: int, w: int, dtype) -> np.ndarray:
    """Scale a 2D plane into an exactly ``(h, w)`` canvas, aspect preserved and centred.

    The array viewer's canvas is declared ONCE per run (``start_acquisition``), so every push has
    to be that exact shape — while two regions of one acquisition can have differently shaped
    mosaics. Letterboxing is the only way to satisfy both without stretching one of them, and it
    is the policy the plate cell already uses (``cell_boxes`` centres a mosaic in its cell).
    """
    h, w = max(1, int(h)), max(1, int(w))
    s = min(h / a.shape[0], w / a.shape[1])
    ih = max(1, min(h, int(round(a.shape[0] * s))))
    iw = max(1, min(w, int(round(a.shape[1] * s))))
    out = np.zeros((h, w), dtype)
    out[(h - ih) // 2:(h - ih) // 2 + ih,
        (w - iw) // 2:(w - iw) // 2 + iw] = _fit_box(a, ih, iw)
    return out
