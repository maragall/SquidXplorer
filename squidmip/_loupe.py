"""The LOUPE ENGINE: real pixels around a point, magnified, off the GUI thread.

Extracted from ``_plate_overview`` on 2026-08-13, verbatim, when a SECOND surface needed a loupe:
the napari canvas in a view window (``squidmip._napari_loupe``), raised with shift-left-click.

WHY THE EXTRACTION AND NOT A SECOND LOUPE
------------------------------------------
Because this module's own IMA-242 note, further down, is the record of what a second one costs.
``_composite_rgb`` and ``_percentile_window`` were private twins of ``composite`` and
``_pct_window``, and they drifted in three separate ways -- a degenerate window rendered full
white, an unticked channel stayed visible, a contrast drag moved the plate and not the inset. All
three were invisible until someone looked at the two surfaces side by side.

The plate loupe and the canvas loupe differ in exactly two things -- what raises them (a
press-and-hold on a widget vs a shift-click on a napari canvas) and where the inset is painted
(inside the plate's own ``paintEvent`` vs a floating overlay). Everything else -- what
magnification MEANS, which pyramid level answers it, how the crop is bounded, where the pixels
come from, how the bar is drawn -- is the same question, so it is answered once, here.

WHAT IS AND IS NOT IN THIS MODULE
----------------------------------
In: the magnification arithmetic (pure, unit-tested), the SOURCES (raw TIFFs / a written pyramid),
the coalescing worker, and the inset PAINTING that both surfaces share.

Out, deliberately: who raises it, who dismisses it, and where the inset goes on screen. Those are
gestures, and the two surfaces genuinely disagree about them.

NOTHING HERE MAY DECODE ON THE Qt THREAD. That is not this module's rule to bend -- see
``CLAUDE.md``, "Nothing decodes on the Qt thread". The sources are called by :class:`_LoupeWorker`
and by nothing else, and ``_LoupeSource.window`` exists precisely because deriving the contrast
window on the GUI thread meant a paint-driven slot decoding a whole TIFF plane.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np
from qtpy.QtCore import QThread, Signal

from squidmip._logpane import get_logger
from squidmip._montage import _area_downsample, _pct_window
# THE LAYOUT CONTRACT, not a hand-parse. `_ZarrLoupeSource` asks `squidmip/contract` where a
# field's pyramid lives rather than reconstructing the path by f-string; see the comment in its
# `_pyramid_levels`, and `docs/plate-contract.md` for why the pyramid is optional.
from squidmip.contract import field_levels, field_path

log = get_logger("loupe")

#: How small a whole-field plane may be shrunk to and still give a trustworthy 1/99.8 percentile.
#: This used to be spelled ``_CELL`` -- the plate's per-well thumbnail size -- which was a true
#: number for the wrong reason: the coarse plane is about having enough pixels for a percentile,
#: not about fitting a plate cell. Same value, named for what it is, so a change to the plate's
#: cell size cannot silently change what contrast the loupe computes.
_LOUPE_COARSE_PX = 88


# --- loupe (IMA-208): press-and-hold magnifier over the plate ------------------------------
#
# The plate montage CANNOT be the loupe's source: a tile is _LOUPE_COARSE_PX (88) px per well, a ~47x
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
    """The FOV index the plate addresses for ``well_id`` when nothing has named one — the
    FALLBACK half of the multi-FOV seam.

    It used to be the whole story, and that is what put the loupe on FOV 0 of every multi-FOV
    region while ``_cell_fraction`` was handing it a position across the whole mosaic. The plate
    hit-test DOES resolve a field now, from the mosaic boxes it already draws by
    (``PlateOverview._fov_box_at``, used by both ``_fov_at`` and ``_loupe_target``), and that
    field is passed down to the sources. This remains the answer for a cell that holds a single
    field, for a point in a gap between fields, and for any caller that has not resolved one --
    a stated default rather than a bare ``0`` scattered across four read paths."""
    if fovs_per_region:
        fovs = fovs_per_region.get(well_id)
        if fovs:
            return int(fovs[0])
    return 0


def loupe_scale_at(s_screen: float, well_px: int, mag: float = _LOUPE_MAG,
                   inset_px: int = _LOUPE_PX) -> tuple[float, float]:
    """(s_loupe, M) for a surface currently drawing ``s_screen`` SCREEN px per level-0 IMAGE px.

    THE RULE, and the only copy of it. Two surfaces state their current scale in two different
    vocabularies -- the plate in screen-px-per-well (:func:`loupe_scale`), the napari canvas in
    camera zoom (:func:`canvas_scale`) -- and neither of them gets to re-decide what to DO with
    it. Everything below is unchanged from when this was the plate's alone.

    ``s_loupe`` is clamped to 1.0 — one screen pixel per level-0 image pixel is as far as
    honest magnification goes; past that we would be upsampling, which is the very thing
    the montage already does badly. ``M`` is what the user actually gains, in [1, mag].

    Two lower clamps, both learned the hard way:

    * Once the user has zoomed the surface PAST native (a field drawn bigger than its own
      pixel count), the 1.0 cap alone would put the inset BELOW the surface's own scale — a
      loupe that shrinks what it points at. Floor at the surface's scale; the caller labels
      that case "native", since there is no detail left to reveal.
    * A fixed target magnification does not survive a 1536-well plate. At fit, a well is ~10
      screen px, so 8x fills only ~85 px of a 240 px inset and the rest would have to come
      from neighbouring wells. Floor at ``inset_px / well_px`` so the inset shows AT MOST one
      whole field — which is also what the gesture means: look closely at *this* one. On a
      1536wp that yields ~22x rather than 8x, still derived entirely from the surface's zoom."""
    well_px = max(1, int(well_px))
    s_plate = max(1e-9, float(s_screen))
    fill_well = float(inset_px) / well_px             # scale at which one field fills the inset
    # Order matters: cap at native FIRST, then floor at the surface's own scale. Capping last
    # would drag a surface that is already past native back down to 1.0 and demagnify.
    s_loupe = max(s_plate, min(1.0, max(mag * s_plate, fill_well)))
    return s_loupe, s_loupe / s_plate


def loupe_scale(cd: float, well_px: int, mag: float = _LOUPE_MAG,
                inset_px: int = _LOUPE_PX) -> tuple[float, float]:
    """THE PLATE'S vocabulary: ``cd`` screen px per well of ``well_px`` image px.

    A thin spelling of :func:`loupe_scale_at`, kept because the plate genuinely knows its scale
    as a ratio of two things it draws, and because every existing caller and test says it this
    way. ``cd / well_px`` IS screen px per image px.
    """
    return loupe_scale_at(float(cd) / max(1, int(well_px)), well_px, mag, inset_px)


def canvas_scale(camera_zoom: float, pixel_size_um: float) -> float:
    """THE NAPARI CANVAS'S vocabulary -> screen px per level-0 ACQUISITION px.

    ``camera.zoom`` is documented by napari as "scale from canvas pixels to world pixels", and
    this app's world units ARE stage micrometres (``_napari_view.scale_translate_from_bbox_um``).
    ``pixel_size_um`` is micrometres per ACQUISITION pixel. So the product is canvas px per
    acquisition px, which is exactly what :func:`loupe_scale_at` wants.

    DELIBERATELY NOT ``layer.scale``. The mosaic on the canvas is a pyramid and what is drawn is
    usually a decimated level, so ``layer.scale`` answers "micrometres per pixel OF THE LEVEL
    BEING DRAWN". The loupe does not read that level — it reads the acquisition, through the
    sources below — so the pitch that matters is the acquisition's. Using the layer's would make
    the magnification silently depend on which pyramid rung napari happened to pick.
    """
    return float(camera_zoom) * float(pixel_size_um)


#: Below this, "magnified" is a lie worth not telling: the surface is already at native
#: resolution and the inset can only re-show the same pixels. Was an inline literal.
_LOUPE_NATIVE_M = 1.05


#: How close the achieved magnification must come to the asked-for one before the ask counts as
#: honoured. Below this the native cap has taken the difference and the label must say so.
_LOUPE_ASK_MET = 0.98


def capped_at_native(mag: float, requested: "Optional[float]" = None) -> bool:
    """Has the 1:1 cap swallowed the factor the user asked for?

    ``loupe_scale_at`` clamps ``s_loupe`` at 1.0 because past that it would be upsampling. The
    consequence is that on a surface ALREADY NEAR NATIVE every rung of the ladder returns the same
    picture -- measured on the 40x sets, a field framed to fill an 860x720 window is a 6.1x
    downsample, so 8x, 16x and 32x all yield 6.09x. That is correct behaviour and an invisible
    one: the control moves and nothing happens.

    So it is a question with a name, asked by both the label and the spoken note. Deliberately NOT
    ``mag < 1.05``, which is the older test for "no magnification at all" -- that is true only at
    the extreme, and the interesting case is the whole range above it where the ask is partly met.
    """
    if requested is None:
        return float(mag) < _LOUPE_NATIVE_M
    return float(mag) < float(requested) * _LOUPE_ASK_MET


def loupe_label(subject: str, mag: float, requested: "Optional[float]" = None) -> str:
    """``"A1 fov 7  ·  8.0×"``, or ``"A1 fov 7  ·  6.1× · native (32× asked)"`` when capped.

    A wheel step that changes nothing must SAY it changed nothing, instead of reading as a dead
    control. That matters more on a canvas than on the plate: a view window is routinely already
    near native, where every rung of the ladder yields the same picture.
    """
    if float(mag) < _LOUPE_NATIVE_M:
        return f"{subject}  ·  native"
    shown = f"{subject}  ·  {float(mag):.1f}×"
    if capped_at_native(mag, requested):
        return f"{shown} · native ({float(requested):g}× asked)"
    return shown


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

    def read_crop(self, well_id, level, y0, x0, h, w, time_point: int = 0, fov=None):
        """(C, y, x) crop at ``level``, CLAMPED into the field (see loupe_clamp_crop) and
        decimated to at most _LOUPE_MAX_CROP samples per side. Runs on the worker thread.

        ``fov`` is the FIELD the crop is in, resolved from the plate's mosaic boxes by
        ``PlateOverview._loupe_target``. ``None`` means "the plate could not name one" (a cell
        holding a single field, or a point in a gap) and falls back to ``_fov_of_well`` — which
        is the seam, not a guess."""
        raise NotImplementedError

    def coarse(self, well_id, time_point: int = 0):
        """A small whole-field (C, y, x) plane used ONLY to derive the contrast window."""
        raise NotImplementedError

    def window(self, well_id, time_point: int = 0):
        """Per-channel contrast window for a well, mirroring the tile's rule.

        Computed HERE, on the loupe worker thread, and memoised per well — never on the GUI
        thread. It used to be derived in ``_on_loupe_crop`` by calling ``coarse()``, which for
        raw meant decoding a whole TIFF plane inside a paint-driven slot AND touching the same
        plane cache the worker was writing (two threads, no lock, one well's pixels labelled as
        another's). One owner, one thread.

        Keyed by ``(well, timepoint)``, for the reason the coarse cache already is: a window
        memoised at one timepoint would go on contrast-stretching every later timepoint by the
        first frame's percentiles, which is a cache answering the wrong question quickly."""
        key = (well_id, int(time_point))
        with _LOUPE_WIN_LOCK:
            cache = self.__dict__.setdefault("_win_cache", {})
            hit = cache.get(key)
        if hit is not None:
            return hit
        coarse = self.coarse(well_id, time_point)
        win = [_pct_window(coarse[c]) for c in range(coarse.shape[0])]
        with _LOUPE_WIN_LOCK:
            cache[key] = win
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
        self._coarse: dict[tuple, np.ndarray] = {}

    def available(self, well_id) -> tuple[bool, str]:
        if well_id in self._meta["regions"]:
            return True, ""
        return False, "no image for this well"

    def _planes(self, well_id, time_point: int = 0, fov=None):
        """The FIELD's (C, y, x) planes at ``time_point``, decoded once and cached.

        Held under a lock for the whole check-decode-publish sequence. Unsynchronised, the two
        callers (worker thread reading a crop, GUI thread deriving a window) could interleave
        between the key test and the store and hand back ANOTHER well's pixels labelled as the
        well under the cursor — a wrong-image bug in a microscopy tool, not a glitch. The GUI
        thread no longer calls in at all (see _LoupeSource.window), but the lock stays: the class
        must be correct for its callers, not for today's call sites.

        The key is ``(well, fov, timepoint)`` for the same reason: a key missing either of the
        last two hands back one field's / one timepoint's pixels under another's label, which is
        the identical wrong-image failure with the axis changed."""
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
        """Level is always 0 here — raw has no pyramid — so the whole burden of bounding the
        work falls on decimation. At plate fit the rect IS most of the field (2084 px on the
        synthetic plate); area-averaging it down to <= _LOUPE_MAX_CROP happens HERE, on the
        worker thread, so what crosses to the GUI thread to be composited is a 456 px square
        (3.3 MB, 11 ms) instead of a 1826 px one (26.7 MB, 118 ms) — which is also what the
        worker's LRU then caches."""
        p = self._planes(well_id, time_point, fov)
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

    def coarse(self, well_id, time_point: int = 0):
        # Deliberately the REGION's first field (fov=None), not the field under the cursor: the
        # contrast window is per WELL, so that brightness does not lurch as the cursor crosses a
        # mosaic seam and make one region look like two different acquisitions.
        key = (well_id, int(time_point))
        if key not in self._coarse:
            p = self._planes(well_id, time_point)
            self._coarse[key] = np.stack(
                [_area_downsample(p[c], _LOUPE_COARSE_PX, _LOUPE_COARSE_PX) for c in range(p.shape[0])])
        return self._coarse[key]


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
        self._coarse: dict[tuple, np.ndarray] = {}
        # Guards `_coarse` AND the `_levels`/`n_levels` publish in `_resolve_levels`. Both are
        # touched from a `_LoupeWorker` QThread, and there can be two of them at once: when
        # `set_loupe_source`'s `wait(2000)` times out the outgoing worker is detached (see
        # `_detach`) and keeps reading from this same source object while the new one starts.
        # Its sibling `_RawLoupeSource._planes` has always been locked; this one was not.
        self._lock = threading.RLock()

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
        single level, which is exactly what the test fixtures hit).

        Under the lock because it publishes TWO attributes, ``_levels`` and ``n_levels``, and
        ``coarse()`` reads ``n_levels`` to pick its level. Two loupe workers (see ``_detach``) can
        both arrive here; unguarded, the second could read a ``n_levels`` that does not yet match
        the ``_levels`` the first is about to store, and magnify the wrong rung."""
        with self._lock:
            if self._levels is not None:
                return self._levels
            # Through the contract, not a hand-parse. This block used to reconstruct the field path
            # by f-string and read multiscales -> datasets[*].path itself behind a bare `except
            # Exception`, i.e. a private copy of the layout plus an unwritten fallback. Both now
            # have one home: squidmip/contract, and docs/plate-contract.md says the pyramid is
            # OPTIONAL and that level "0" is what its absence falls back to.
            self._levels = field_levels(
                field_path(self._base, self._path_of(well_id), self._fov_of(well_id)))
            self.n_levels = max(1, len(self._levels))
            return self._levels

    def _open(self, well_id, level, fov=None):
        levels = self._resolve_levels(well_id)
        level = max(0, min(int(level), len(levels) - 1))
        f = self._fov_of(well_id) if fov is None else int(fov)
        # Through the shared, LOCKED, bounded handle cache — not a private dict.
        # This was the last raw `ts.open` on a read path in the package, and `_tsctx`'s own
        # docstring names "the loupe source" as one of the four unbounded unlocked handle dicts it
        # was written to replace; it was the one that never got converted. Unbounded mattered: a
        # 1536-well plate browsed at three levels is 4608 open stores that nothing ever evicted.
        from squidmip._tsctx import HANDLES

        return HANDLES.get(field_path(self._base, self._path_of(well_id), f, levels[level]),
                           open_only=True)

    def read_crop(self, well_id, level, y0, x0, h, w, time_point: int = 0, fov=None):
        arr = self._open(well_id, level, fov)
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
        with self._lock:
            hit = self._coarse.get(key)
        if hit is not None:
            return hit
        # The READ is outside the lock on purpose: it is a whole coarse plane and holding the lock
        # across it would serialise two loupe workers reading two different wells. Two threads
        # racing the same key both compute and the second store wins — identical bytes, one wasted
        # read. What must not race is the dict itself.
        arr = self._open(well_id, self.n_levels - 1)          # coarsest level = cheapest
        t_idx = max(0, min(int(time_point), arr.shape[0] - 1))
        plane = np.asarray(arr[t_idx, :, 0].read().result())
        with self._lock:
            return self._coarse.setdefault(key, plane)


# THE OWNERSHIP RULE FOR THE WORKER BELOW is `squidmip._qthread_life.detach`, not anything in
# this module: a `QThread` whose C++ half is destroyed while it is still running calls `qFatal`
# and the process aborts. This worker can always be asked to stop in the middle of a decode that
# cannot be interrupted, so a caller that cannot join it in time must DETACH it -- park it,
# referenced and unparented, until it finishes on its own -- rather than dropping the last
# reference. `PlateOverview._stop_loupe` and `_napari_loupe.CanvasLoupe._stop_worker` both do
# exactly that; see `_qthread_life` for the measurement behind it (exit code 134).


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

    def request(self, gen, well_id, level, y0, x0, h, w, time_point: int = 0, fov=None):
        with self._cv:
            self._pending = (gen, well_id, level, y0, x0, h, w, int(time_point),
                             None if fov is None else int(fov))
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

    def clear_cache(self) -> None:
        """Drop the decoded crops WITHOUT stopping the thread.

        For a view that has gone background: the crops are the memory (the deck budgets ~88 MB a
        view and eight of these is a few MB of that), while the thread is cheap and idle. Stopping
        and restarting it on every tab click would pay a thread teardown for nothing.
        """
        self._cache.clear()
        self._order.clear()

    def run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                gen, well_id, level, y0, x0, h, w, time_point, fov = self._pending
                self._pending = None
            # The LRU key carries the TIMEPOINT and the FOV. Without the timepoint the first
            # frame's crop answers for every later one at the same rectangle; without the FOV the
            # first FIELD's crop answers for every other field of the same region at the same
            # rectangle, which on a mosaic is every field the cursor crosses. The plate moves, the
            # inset does not, and nothing errors. Same rule as the sources' own coarse caches.
            key = (well_id, fov, level, y0, x0, h, w, time_point)
            try:
                crop = self._cached(key)
                if crop is None:
                    crop = self._source.read_crop(well_id, level, y0, x0, h, w, time_point,
                                                  fov=fov)
                    self._store(key, crop)
                # The contrast window belongs on this side too: deriving it on the GUI thread
                # meant a paint-driven slot could decode a whole TIFF plane (IMA-208).
                try:
                    win = self._source.window(well_id, time_point)
                except Exception:
                    win = None                        # the widget falls back to a flat window
                self.ready.emit(gen, well_id, crop, win, None)
            except Exception as e:                    # a racing writer / deleted plate / bad path
                self.ready.emit(gen, well_id, None, None, f"{type(e).__name__}: {e}")


# --- the inset, painted. Shared by the plate and by the napari canvas. ----------------------
#
# `PlateOverview._paint_loupe` and `_napari_loupe.LoupeInset.paintEvent` differ in exactly one
# thing -- WHERE the square goes. The plate offsets it from the cursor inside its own paintEvent;
# the canvas overlay is a widget that already IS the square. Everything inside the square is the
# same question, so it is answered once, here, by a function that takes a rect.
#
# The chrome colours are the LOUPE's, not the plate's, which is why they moved with it.

_INSET_BG = "#05070b"
_INSET_BORDER = "#c9d1d9"
_INSET_BAR = "#e6edf3"
_INSET_LABEL = "#58a6ff"
_INSET_MUTED = "#8b949e"

#: Type sizes for the inset's own chrome. Same values `_plate_overview` used, carried over so the
#: two insets cannot drift apart in a way nobody would think to look for.
_INSET_LABEL_PX = 11
_INSET_SCALE_PX = 10


def loupe_inset_rect(x: int, y: int, host_w: int, host_h: int,
                     inset_px: int = _LOUPE_PX, gap: int = 18):
    """Where an ``inset_px`` square goes for an anchor at ``(x, y)`` inside a host widget.

    Offset from the anchor so the hand never covers what it points at, flipped to the other side
    at the far edges, and clamped whole inside the host so it never hangs off. Lifted verbatim
    from ``PlateOverview._paint_loupe``; returns ``(bx, by)``.
    """
    s = int(inset_px)
    bx = x + gap if x + gap + s < host_w else x - gap - s
    by = y + gap if y + gap + s < host_h else y - gap - s
    return (int(max(2, min(bx, host_w - s - 2))),
            int(max(2, min(by, host_h - s - 2))))


def paint_loupe_inset(p, bx: int, by: int, *, image=None, note: str = "",
                      label: str = "", um_per_screen_px=None,
                      inset_px: int = _LOUPE_PX, font=None) -> None:
    """Paint ONE loupe inset at ``(bx, by)`` into whatever *p* is painting.

    The pixmap blit, the µm scale bar, the corner label and the border — and nothing about WHO is
    painting or WHERE. That split is the whole reason this takes a position rather than being a
    method on a widget: the plate calls it from its own ``paintEvent`` with a cursor-avoiding
    rect, the canvas overlay calls it from its own ``paintEvent`` with ``(0, 0)``.

    ``um_per_screen_px`` of ``None`` DRAWS NO BAR and says "scale unknown" instead. A microscopy
    tool that displays a confidently wrong micron figure is worse than one that admits it does not
    know — see :func:`loupe_um_per_screen_px` for when that happens and why.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QColor, QFont, QPen, QPixmap

    s = int(inset_px)
    if font is None:
        def font(px, weight=None):
            f = QFont("Segoe UI", px)
            if weight is not None:
                f.setWeight(weight)
            return f

    p.fillRect(bx, by, s, s, QColor(_INSET_BG))
    if image is not None:
        p.save()
        p.setClipRect(bx, by, s, s)
        # FastTransformation, i.e. no smoothing: the whole promise of a loupe is that these are
        # the real pixels. Interpolating them would be the montage's defect, one level down.
        p.drawPixmap(bx, by, QPixmap.fromImage(image).scaled(
            s, s, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation))
        p.restore()
    else:
        p.setPen(QColor(_INSET_MUTED))
        p.setFont(font(_INSET_LABEL_PX))
        p.drawText(bx, by, s, s, Qt.AlignCenter | Qt.TextWordWrap, note or "reading …")

    if image is not None:
        p.setFont(font(_INSET_SCALE_PX, QFont.Weight.DemiBold))
        if um_per_screen_px is None:
            p.setPen(QColor(_INSET_MUTED))
            p.drawText(bx + 8, by + s - 10, "scale unknown")
        else:
            target = _nice_scale_um(um_per_screen_px * (s * 0.4))   # ~40% of the inset, 1/2/5
            bar = int(round(target / um_per_screen_px))
            p.setPen(QPen(QColor(_INSET_BAR), 2))
            p.drawLine(bx + 10, by + s - 14, bx + 10 + bar, by + s - 14)
            p.setPen(QColor(_INSET_BAR))
            p.drawText(bx + 10, by + s - 18, _fmt_um(target))
        if label:
            p.setPen(QColor(_INSET_LABEL))
            p.drawText(bx + 8, by + 16, label)

    p.setPen(QPen(QColor(_INSET_BORDER), 1))
    p.setBrush(Qt.NoBrush)
    p.drawRect(bx, by, s, s)
