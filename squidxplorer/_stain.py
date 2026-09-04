"""Colorimetric channels saved gray are DETECTED and displayed in the stain's own color.

Squid's mosaic_view yaml records ``rgb_channel_names`` — the instrument's own statement that a
channel was RGB live. A channel listed there whose files are 2-D on disk was color recorded
gray (``MULTIPOINT_BF_SAVING_OPTION`` = RGB2GRAY / Green Channel Only). The recorded chroma
survives only in the colored mosaic PNG, so the stain's absorbance vector is measured from it
once (Beer-Lambert, OD ratios to green) and becomes the channel's display LUT:
``t -> (t^kR(od), t, t^kB(od))``, with (k_R, k_B) fit per green-OD quantile bin so the hue can
follow density (Masson's trichrome: dense collagen is blue where light cytoplasm is red; one
global vector reads as H&E). A colormap, never synthesized pixels: the files are untouched and
the look is exactly the stain physics the PNG recorded.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
import os
from pathlib import Path
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)

#: The white point percentile shared by the LUT fit and the display seed window. The LUT's t is
#: transmittance, so white IS this percentile of the data and the faithful window is [0, white].
STAIN_WHITE_PERCENTILE = 99

#: Set to 1/true/yes/on to disable derived color: gray channels then stay gray, yaml color only.
NO_RECONSTRUCTED_COLOR_ENV = "SQUIDXPLORER_NO_RECONSTRUCTED_COLOR"

#: The View menu's live override; ``None`` defers to the environment variable.
_reconstruct_override: Optional[bool] = None


def reconstruction_enabled() -> bool:
    """Whether derived color (chroma reconstruction / the estimated stain LUT) may attach.

    THE one switch both attach paths consult, so the reader stays deterministic per flag: a
    metadata build under one value expands and colors the same way everywhere.
    """
    if _reconstruct_override is not None:
        return _reconstruct_override
    return os.environ.get(NO_RECONSTRUCTED_COLOR_ENV, "").strip().lower() not in {
        "1", "true", "yes", "on"}


def set_reconstruction(on: Optional[bool]) -> None:
    """Override the flag for subsequent metadata builds (``None`` returns to the env var).

    Callers re-ingest to apply: a cached ``reader.metadata`` was built under the old value.
    """
    global _reconstruct_override
    _reconstruct_override = None if on is None else bool(on)

#: Minimum stained pixels for a trustworthy absorbance fit; below it, stay gray.
_MIN_STAINED_PX = 5_000
#: Green optical density above which a pixel counts as stained (background is ~0).
_OD_FLOOR = 0.08
_LUT_N = 256
#: Quantile bins over green OD (quantile, not linear: density histograms are heavy-tailed).
_N_OD_BINS = 16
#: A bin below this count is dropped; its median is noise, not a hue.
_MIN_BIN_PX = 200


def _norm(name: str) -> str:
    return str(name).lower().replace("_", " ").strip()


def _mosaic_rgb_pngs(root: Path) -> list[tuple[str, Path, dict]]:
    """``(rgb_channel_name, png_path, yaml_doc)`` triples from every mosaic_view yaml under *root*."""
    import yaml

    out = []
    for y in sorted(root.glob("*/mosaic_view/*.yaml")):
        try:
            doc = yaml.safe_load(y.read_text()) or {}
        except Exception:                       # noqa: BLE001 - a broken overview is no overview
            continue
        names = list(doc.get("rgb_channel_names") or [])
        files = list(doc.get("rgb_view_files") or [])
        for name, fn in zip(names, files):
            png = y.parent / str(fn)
            if png.is_file():
                out.append((str(name), png, doc))
    return out


def _matches_rgb_name(entry, rgb_name: str) -> bool:
    """Whether a channel entry is the one the mosaic yaml recorded as RGB live."""
    want = (_norm(entry.get("display_name") or ""), _norm(entry.get("name") or ""))
    return any(w and _norm(rgb_name).endswith(w) for w in want)


def _smooth3(v) -> np.ndarray:
    """3-tap moving average with edge replication; kills hue banding between bins."""
    v = np.asarray(v, dtype=np.float64)
    if v.size < 3:
        return v
    p = np.concatenate([v[:1], v, v[-1:]])
    return (p[:-2] + p[1:-1] + p[2:]) / 3.0


def _binned_stain_ks(od_g, ratio_r, ratio_b) -> tuple:
    """``(bin_center_od, k_R, k_B)`` arrays: median OD ratios per green-OD quantile bin."""
    edges = np.unique(np.quantile(od_g, np.linspace(0.0, 1.0, _N_OD_BINS + 1)))
    idx = np.clip(np.searchsorted(edges, od_g, side="right") - 1, 0, max(len(edges) - 2, 0))
    centers, k_r, k_b = [], [], []
    for b in range(max(len(edges) - 1, 0)):
        sel = idx == b
        if int(sel.sum()) < _MIN_BIN_PX:
            continue
        centers.append(float(np.median(od_g[sel])))
        k_r.append(float(np.median(ratio_r[sel])))
        k_b.append(float(np.median(ratio_b[sel])))
    if not centers:                             # too sparse to bin: the global fit is the answer
        centers = [float(np.median(od_g))]
        k_r = [float(np.median(ratio_r))]
        k_b = [float(np.median(ratio_b))]
    return np.asarray(centers), _smooth3(k_r), _smooth3(k_b)


def stain_lut(png_path: Path) -> Optional[tuple]:
    """A 256-stop RGB LUT from the overview's measured stain vector, or None when unfittable.

    Each stop's hue takes the (k_R, k_B) of its own density: the stop's OD is ``-ln(t)``,
    interpolated between bin centers and clamped at the ends. Constant-ratio data (one stain)
    reduces to the old global fit; density-varying data (trichrome) keeps its hue curve.
    """
    from PIL import Image

    png = np.asarray(Image.open(png_path)).astype(np.float64)
    if png.ndim != 3 or png.shape[2] < 3:
        return None
    white = np.percentile(png.reshape(-1, png.shape[2])[:, :3], STAIN_WHITE_PERCENTILE, axis=0)
    if not (white > 0).all():
        return None
    od = -np.log(np.clip(png[..., :3] / white, 1e-3, 1.0))
    mask = od[..., 1] > _OD_FLOOR
    if int(mask.sum()) < _MIN_STAINED_PX:
        return None
    od_g = od[..., 1][mask]
    centers, k_r, k_b = _binned_stain_ks(
        od_g, od[..., 0][mask] / od_g, od[..., 2][mask] / od_g)
    t = np.linspace(0.0, 1.0, _LUT_N)
    od_t = -np.log(np.clip(t, 1e-3, None))
    kr_t = np.maximum(np.interp(od_t, centers, k_r), 1e-6)   # k floor keeps t=0 black
    kb_t = np.maximum(np.interp(od_t, centers, k_b), 1e-6)
    lut = np.stack([t ** kr_t, t, t ** kb_t], axis=1)        # t=1 is exactly (1, 1, 1)
    return tuple(tuple(float(v) for v in row) for row in lut)


def attach_stain_luts(root, channels: list, rgb_bases: set) -> None:
    """Give each color-recorded-gray channel entry its measured ``display_lut``, in place.

    ``rgb_bases`` — channels whose files really are 3-component AND channels expanded into
    chroma components (see :func:`chroma_sources`) — are skipped: their components carry real
    color, so a LUT on top would double-tint. The LUT is the FALLBACK when no chroma exists.
    """
    pairs = _mosaic_rgb_pngs(Path(root))
    if not pairs:
        return
    if not reconstruction_enabled():
        _log.info(
            "reconstructed color is OFF (%s or View > Reconstructed Color): color-recorded-gray "
            "channels stay gray with their yaml color.", NO_RECONSTRUCTED_COLOR_ENV)
        return
    for entry in channels:
        if entry.get("name") in rgb_bases or entry.get("display_lut"):
            continue
        for rgb_name, png, _doc in pairs:
            if not _matches_rgb_name(entry, rgb_name):
                continue
            lut = stain_lut(png)
            if lut is None:
                continue
            entry["display_lut"] = lut
            entry["color_source"] = "estimated"   # a density fit, never the file's own color
            _log.info(
                "channel %s was RGB live but its files are grayscale (Squid's "
                "MULTIPOINT_BF_SAVING_OPTION); displaying through the stain colormap measured "
                "from %s. Set the option to \"Raw\" to record true color.",
                entry.get("name"), png.name)
            break


# --- Virtual chroma components: real local color from the overview PNG ------------------------
#
# The LUT above is one global hue curve; a trichrome section carries DIFFERENT hues side by side
# (pink cytoplasm, blue collagen). Where the overview PNG's geometry is declared, the recorded
# chroma can be put back per pixel instead: the reader expands the channel into (R)/(G)/(B)
# components, (G) is the file's own plane, and (R)/(B) scale it by the PNG's local R/G and B/G
# ratios over that FOV. Additive blending of the three then reconstructs the mosaic's own color
# at the file's full resolution and luminance.

#: Ratio denominator floor in PNG counts; a dim denominator rolls the ratio off (bounded at
#: 255/16) instead of flat-capping dense stain, whose true R/G runs past 10 (measured p99 10.5
#: on the 20x trichrome set); division glow on near-black PNG is _luminance_weight's job.
_RATIO_DENOM_FLOOR = 16.0
#: PNG green at/below this is unwritten mosaic area, not tissue: ratio stays neutral there.
_CHROMA_G_FLOOR = 2.0
#: Per-FOV ratio windows kept in memory (each ~2.6 MB at 2 um over a 1900 px frame).
_CHROMA_CACHE_MAX = 32


def _luminance_weight(png_g: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """[0, 1] confidence that each PNG pixel's ratio describes THIS plane's content.

    The overview's overlap zones hold NEIGHBOR frames' pixels (later-overwrites-earlier), and
    its 2 um blur smears tissue over lumen edges — so a tissue ratio (R/G ~ 3, honestly
    measured in DARK png content) can land on a BRIGHT file pixel, clip at the dtype ceiling
    and glow hot magenta (measured: FOV 72 of the 20x trichrome set, 27k px at the uint8
    ceiling, 96% in the FOV-overlap band; damping removed all but 2 and RAISED the
    overview-window correlation, R 0.39 -> 0.56, B 0.51 -> 0.67).

    Weight = clip(G_png / (plane * gain), 0, 1) on the PNG grid, gain the median G_png/plane
    over covered pixels: a PNG darker than the file's own luminance is distrust, scaled; the
    reverse mismatch (PNG brighter) only desaturates and is left alone. A window with no
    covered pixels, or a degenerate gain, damps nothing.
    """
    h, w = png_g.shape
    ys = np.linspace(0.0, plane.shape[0] - 1.0, h).astype(np.intp)
    xs = np.linspace(0.0, plane.shape[1] - 1.0, w).astype(np.intp)
    coarse = plane[np.ix_(ys, xs)].astype(np.float32)
    covered = (png_g > _CHROMA_G_FLOOR) & (coarse > 0)
    if not covered.any():
        return np.ones_like(png_g, dtype=np.float32)
    gain = float(np.median(png_g[covered] / coarse[covered]))
    if not np.isfinite(gain) or gain <= 0:
        return np.ones_like(png_g, dtype=np.float32)
    return np.clip(png_g / np.maximum(coarse * gain, 1e-6), 0.0, 1.0).astype(np.float32)


def _box3(a: np.ndarray) -> np.ndarray:
    """3x3 box sum with edge replication (numpy only)."""
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])


#: Fill growth cap in pixels; at 2 um that is ~0.5 mm, far beyond any real overlap band.
_FILL_MAX_STEPS = 256


def _extend_valid(values: list, valid: np.ndarray, targets: np.ndarray) -> tuple:
    """Extend *valid* pixels' values across *targets*: iterative 3x3 neighborhood-mean growth
    (nearest-valid in spirit, inherently smooth) plus one 3x3 box smoothing over the filled
    pixels. Chroma is low-frequency, so a band a few dozen pixels wide fills faithfully.

    Returns ``(filled_values, reached)``: copies of *values* where reached targets carry the
    extended values, valid pixels keep their exact measured values, and targets the growth
    never reached (no valid source, or past the cap) keep their originals.
    """
    filled = valid.copy()
    acc = [np.where(valid, v, 0.0).astype(np.float32) for v in values]
    remaining = targets & ~filled
    for _ in range(_FILL_MAX_STEPS):
        if not remaining.any():
            break
        cnt = _box3(filled.astype(np.float32))
        grow = remaining & (cnt > 0)
        if not grow.any():
            break
        for a in acc:
            s = _box3(a)
            a[grow] = s[grow] / cnt[grow]
        filled |= grow
        remaining &= ~grow
    reached = targets & filled & ~valid
    if reached.any():
        n = np.maximum(_box3(filled.astype(np.float32)), 1e-6)
        smooth = [_box3(a) / n for a in acc]
        for a, s in zip(acc, smooth):
            a[reached] = s[reached]
    out = []
    for v, a in zip(values, acc):
        r = np.asarray(v, dtype=np.float32).copy()
        r[reached] = a[reached]
        out.append(r)
    return out, reached


def _acquisition_order_positions(png_path: Path) -> Optional[list]:
    """``[(region, x_um, y_um)]`` in coordinates.csv row order (the blit order), or None.

    The overview PNG is written tile over tile as the run acquires, so the csv's row order IS
    the overwrite order. The executed record beside the PNG's timepoint folder is preferred;
    the root (planned) csv is the fallback. Repeated positions (one row per z or per t) keep
    their first occurrence: a re-blit of the same footprint changes no ownership. Anything
    unreadable returns None: ownership then simply never engages.
    """
    if png_path.parent.name != "mosaic_view":
        return None
    t_dir = png_path.parent.parent
    for csv_path in (t_dir / "coordinates.csv", t_dir.parent / "coordinates.csv"):
        if csv_path.is_file():
            break
    else:
        return None
    import csv
    import io

    from squidxplorer.reader import _coord_columns, _parse_mm_pair

    try:
        rows = csv.DictReader(io.StringIO(csv_path.read_text()))
        x_col, y_col = _coord_columns(rows.fieldnames)
        out: list = []
        seen: set = set()
        for line_no, row in enumerate(rows, start=2):
            region = (row.get("region") or "").strip()
            if not region:
                continue
            pair = _parse_mm_pair((row.get(x_col) or "").strip(),
                                  (row.get(y_col) or "").strip(), region, line_no)
            if pair is None:
                continue
            key = (region, round(pair[0], 6), round(pair[1], 6))
            if key in seen:
                continue
            seen.add(key)
            out.append((region, pair[0] * 1000.0, pair[1] * 1000.0))
    except Exception:                              # noqa: BLE001 - no order record, no ownership
        _log.debug("chroma ownership: %s is unreadable; ratios stay damped only.", csv_path)
        return None
    return out or None


def _upsample_bilinear(a: np.ndarray, shape: tuple) -> np.ndarray:
    """Bilinear upsample of a small 2-D array to *shape* (numpy only; chroma is low-frequency)."""
    h, w = a.shape
    y = np.linspace(0.0, h - 1.0, shape[0])
    x = np.linspace(0.0, w - 1.0, shape[1])
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (y - y0).astype(np.float32)[:, None]
    wx = (x - x0).astype(np.float32)[None, :]
    return (a[np.ix_(y0, x0)] * (1 - wy) * (1 - wx) + a[np.ix_(y1, x0)] * wy * (1 - wx)
            + a[np.ix_(y0, x1)] * (1 - wy) * wx + a[np.ix_(y1, x1)] * wy * wx)


def _yaml_geometry(doc: dict, png_path: Path) -> Optional[tuple]:
    """``(top_left_mm_yx, resolution_um)`` from a mosaic_view yaml, or None when unfittable.

    ``full.top_left_mm`` is ``(y, x)`` and ``full.extents_mm`` is ``(y0, y1, x0, x1)``; when
    extents are declared the order is cross-checked against the PNG's own size (rows = y span /
    resolution) rather than trusted, and a swapped yaml is read swapped. No extents, no check:
    the verified (y, x) convention is used as declared.
    """
    from PIL import Image

    res_um = doc.get("resolution_um")
    full = doc.get("full") or {}
    top = full.get("top_left_mm")
    try:
        res_um = float(res_um)
        top = (float(top[0]), float(top[1]))
    except (TypeError, ValueError, IndexError):
        return None
    if res_um <= 0:
        return None
    ext = full.get("extents_mm")
    if ext is not None and len(ext) >= 4:
        try:
            with Image.open(png_path) as im:
                width, height = im.size            # header only, no decode
        except Exception:                          # noqa: BLE001 - unreadable overview: no chroma
            return None
        res_mm = res_um / 1000.0
        span0 = (float(ext[1]) - float(ext[0])) / res_mm
        span1 = (float(ext[3]) - float(ext[2])) / res_mm
        tol = 2.0
        if abs(span0 - height) <= tol and abs(span1 - width) <= tol:
            return top, res_um                     # declared (y, x): the verified convention
        if abs(span0 - width) <= tol and abs(span1 - height) <= tol:
            return (top[1], top[0]), res_um        # swapped yaml, read swapped
        return None                                # extents disagree with the PNG: refuse
    return top, res_um


class ChromaSource:
    """Per-FOV local R/G and B/G chroma ratios from one overview PNG, addressed in stage um.

    Geometry: PNG rows map stage-y and cols map stage-x from ``top_left_mm`` (y, x) at
    ``resolution_um``, and a FOV's stage position is its CENTER — verified by template matching
    on the real 20x trichrome set (corr 0.87-0.89 at exactly zero offset centered; 0.01-0.06
    with the corner convention). Uncovered or unwritten (near-black) PNG area is neutral 1.0.

    Ownership: the overview is written tile over tile in acquisition order
    (later-overwrites-earlier), so in an FOV's overlap band the PNG holds the NEIGHBOR frame's
    pixels (measured: FOV 72's right band correlates 0.894 with neighbor 73's frame vs 0.515
    with its own). Where the neighbor's rendering is MISALIGNED, the luminance damping can
    only pull the foreign ratio toward NEUTRAL — an under-colored band, the customer's
    second-round complaint. Ownership decides what mistrust FALLS BACK TO: on unowned pixels
    the fallback is the frame's own owned hue extended across the band (:func:`_extend_valid`),
    itself damped against the extended own luminance. Where the PNG's luminance AGREES with
    the plane the measured ratio still stands in full — on the well aligned ground-truth set
    the band's measured hue is the same tissue seen through the neighbor's frame, and
    replacing it wholesale measured WORSE (window corr 0.977 -> 0.775).
    """

    def __init__(self, png_path, top_left_mm_yx, resolution_um) -> None:
        self._png_path = Path(png_path)
        self._top_mm = (float(top_left_mm_yx[0]), float(top_left_mm_yx[1]))
        self._res_um = float(resolution_um)
        self._png: Optional[np.ndarray] = None
        self._size: Optional[tuple] = None
        #: (region, fov) -> (6, h, w) float32 measured+fallback chroma window; bounded LRU.
        self._windows: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        #: Lazily parsed blit-order positions; False = not read yet, None = no record.
        self._blit_order: object = False

    @property
    def png_path(self) -> Path:
        return self._png_path

    def _image(self) -> np.ndarray:
        if self._png is None:
            from PIL import Image

            self._png = np.asarray(Image.open(self._png_path))[..., :3]
        return self._png

    def _png_size(self) -> tuple:
        """(rows, cols) without decoding the pixels."""
        if self._size is None:
            if self._png is not None:
                self._size = self._png.shape[:2]
            else:
                from PIL import Image

                with Image.open(self._png_path) as im:
                    self._size = (im.size[1], im.size[0])
        return self._size

    def _window_box(self, x_um: float, y_um: float, frame_shape: tuple,
                    pixel_size_um: float) -> tuple:
        """``(r0, c0, h, w)`` of the FOV's window in PNG pixels (CENTER convention)."""
        h = max(int(round(frame_shape[0] * pixel_size_um / self._res_um)), 1)
        w = max(int(round(frame_shape[1] * pixel_size_um / self._res_um)), 1)
        r0 = int(round((y_um - self._top_mm[0] * 1000.0) / self._res_um - h / 2.0))
        c0 = int(round((x_um - self._top_mm[1] * 1000.0) / self._res_um - w / 2.0))
        return r0, c0, h, w

    def _positions(self) -> Optional[list]:
        """The acquisition-order ``(region, x_um, y_um)`` list, parsed once; None without one."""
        if self._blit_order is False:
            self._blit_order = _acquisition_order_positions(self._png_path)
        return self._blit_order  # type: ignore[return-value]

    def _ownership_mask(self, region: str, x_um: float, y_um: float, frame_shape: tuple,
                        pixel_size_um: float) -> np.ndarray:
        """(h, w) bool over the FOV's PNG window: True where the overview holds THIS frame.

        Squid blits the overview tile over tile in acquisition order, so a window pixel is
        owned by the LAST FOV whose footprint covers it: everything under a LATER footprint is
        the neighbor frame's pixels, not this one's. Without a blit-order record (or when this
        FOV cannot be located in it) everything reads as owned, the damped-only behavior.
        """
        r0, c0, h, w = self._window_box(x_um, y_um, frame_shape, pixel_size_um)
        owned = np.ones((h, w), dtype=bool)
        positions = self._positions()
        if not positions:
            return owned
        # Locate this FOV's own row by stage position (the reader's position came from the
        # same csv family); a mismatch beyond half a frame is "not found", never a guess.
        best, best_d = None, max(frame_shape) * pixel_size_um / 2.0
        for i, (reg, px, py) in enumerate(positions):
            if reg != str(region):
                continue
            d = max(abs(px - x_um), abs(py - y_um))
            if d < best_d:
                best, best_d = i, d
        if best is None:
            _log.debug("chroma ownership: no coordinates row matches %s at (%.1f, %.1f) um; "
                       "ratios stay damped only.", region, x_um, y_um)
            return owned
        for _reg, px, py in positions[best + 1:]:
            rr, cc, hh, ww = self._window_box(px, py, frame_shape, pixel_size_um)
            ra, rb = max(rr - r0, 0), min(rr + hh - r0, h)
            ca, cb = max(cc - c0, 0), min(cc + ww - c0, w)
            if ra < rb and ca < cb:
                owned[ra:rb, ca:cb] = False
        return owned

    def coverage(self, fov_centers_um: dict, frame_shape: tuple, pixel_size_um: float) -> tuple:
        """``(outside, partial, total)`` FOV counts against the PNG's bounds (geometry only)."""
        rows, cols = self._png_size()
        outside = partial = 0
        for (x_um, y_um) in fov_centers_um.values():
            r0, c0, h, w = self._window_box(x_um, y_um, frame_shape, pixel_size_um)
            if r0 + h <= 0 or c0 + w <= 0 or r0 >= rows or c0 >= cols:
                outside += 1
            elif r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
                partial += 1
        return outside, partial, len(fov_centers_um)

    def _ratios(self, region: str, fov: int, x_um: float, y_um: float, frame_shape: tuple,
                pixel_size_um: float) -> np.ndarray:
        """The FOV's (6, h, w) float32 chroma window at PNG resolution, cached (LRU).

        Rows 0..2 are the MEASURED [R/G, B/G, PNG G]: row 2 is the luminance each ratio was
        measured at, what :func:`_luminance_weight` weighs the ratio's trust by. Zero outside
        coverage: an uncovered cell's ratio is neutral 1.0 and its weight is moot.

        Rows 3..5 are the FALLBACK [R/G, B/G, G]: neutral 1.0 (with row 5 = row 2) on pixels
        the overview still holds from this frame (see :meth:`_ownership_mask`), and the OWNED
        ratios extended across the unowned overlap band elsewhere — with the owned luminance
        extended alongside, so the fallback is weighed against this frame's OWN luminance,
        never the neighbor's. A neutral fallback makes the display formula reduce exactly to
        the plain damping, so owned pixels are untouched by construction.
        """
        key = (str(region), int(fov))
        cached = self._windows.get(key)
        if cached is not None:
            self._windows.move_to_end(key)
            return cached
        r0, c0, h, w = self._window_box(x_um, y_um, frame_shape, pixel_size_um)
        out = np.ones((6, h, w), dtype=np.float32)
        out[2] = 0.0
        out[5] = 0.0
        rows, cols = self._png_size()
        ra, rb = max(r0, 0), min(r0 + h, rows)
        ca, cb = max(c0, 0), min(c0 + w, cols)
        if ra < rb and ca < cb:
            crop = self._image()[ra:rb, ca:cb].astype(np.float32)
            g = crop[..., 1]
            usable = g > _CHROMA_G_FLOOR           # near-black PNG is unwritten, not tissue
            denom = np.maximum(g, _RATIO_DENOM_FLOOR)
            ratio_r = np.where(usable, crop[..., 0] / denom, 1.0)
            ratio_b = np.where(usable, crop[..., 2] / denom, 1.0)
            fb_r = np.ones_like(ratio_r)
            fb_b = np.ones_like(ratio_b)
            fb_g = g.astype(np.float32).copy()
            owned = self._ownership_mask(str(region), x_um, y_um, frame_shape, pixel_size_um)
            own_c = owned[ra - r0:rb - r0, ca - c0:cb - c0]
            valid = usable & own_c
            targets = ~own_c
            if targets.any() and valid.any():
                (ext_r, ext_b, ext_g), reached = _extend_valid(
                    [ratio_r, ratio_b, g], valid, targets)
                fb_r[reached] = ext_r[reached]
                fb_b[reached] = ext_b[reached]
                fb_g[reached] = ext_g[reached]
                _log.info(
                    "chroma for %s fov %s (overview %s): %.0f%% of the window is held by "
                    "later FOVs' pixels; its mistrust fallback is this frame's own owned "
                    "hue on the %.0f%% reached by the fill.",
                    region, fov, self._png_path.name, 100.0 * targets.mean(),
                    100.0 * reached.mean())
            out[0, ra - r0:rb - r0, ca - c0:cb - c0] = ratio_r
            out[1, ra - r0:rb - r0, ca - c0:cb - c0] = ratio_b
            out[2, ra - r0:rb - r0, ca - c0:cb - c0] = g
            out[3, ra - r0:rb - r0, ca - c0:cb - c0] = fb_r
            out[4, ra - r0:rb - r0, ca - c0:cb - c0] = fb_b
            out[5, ra - r0:rb - r0, ca - c0:cb - c0] = fb_g
        if len(self._windows) >= _CHROMA_CACHE_MAX:
            self._windows.popitem(last=False)
        self._windows[key] = out
        return out

    def component_plane(self, plane: np.ndarray, component: int, region: str, fov: int,
                        x_um: float, y_um: float, pixel_size_um: float) -> np.ndarray:
        """One chroma component of a color-recorded-gray plane, in the plane's own dtype.

        Component 1 (G) is the file's own pixels untouched; 0 (R) and 2 (B) scale them by the
        upsampled local ratio, damped where the PNG's luminance disagrees with the plane's
        own (see :func:`_luminance_weight` — the overlap-band hot-magenta fix). What mistrust
        falls back to is the fallback rows of :meth:`_ratios`: neutral on owned pixels, the
        frame's own extended hue on the unowned overlap band — itself damped against the
        extended own luminance, so a bright lumen pixel under a tissue fallback still cannot
        glow. Where the fallback is neutral the formula reduces exactly to the plain damping.
        The result is clipped to the dtype's range and cast back.
        """
        if component == 1:
            return plane
        ratios = self._ratios(region, fov, x_um, y_um, plane.shape, pixel_size_um)
        trust = _luminance_weight(ratios[2], plane)
        fb_trust = _luminance_weight(ratios[5], plane)
        fallback = 1.0 + (ratios[3 if component == 0 else 4] - 1.0) * fb_trust
        damped = fallback + (ratios[0 if component == 0 else 1] - fallback) * trust
        ratio = _upsample_bilinear(damped, plane.shape)
        out = plane.astype(np.float32) * ratio
        if plane.dtype.kind in "iu":
            # ROUND, never truncate: bilinear weights in float32 leave 1.0 as 0.99999994, and a
            # truncating cast would make even a NEUTRAL ratio change pixels (measured: 9.5% of a
            # real uncovered FOV off by one).
            info = np.iinfo(plane.dtype)
            out = np.clip(np.rint(out), info.min, info.max)
        return out.astype(plane.dtype)


def chroma_sources(root, channels: list, rgb_bases: set) -> dict:
    """``{channel_name: ChromaSource}`` for color-recorded-gray channels with usable geometry.

    Detection is :func:`attach_stain_luts`'s own matching plus geometry: the mosaic_view yaml
    must declare ``resolution_um`` and ``full.top_left_mm`` that fit the PNG (see
    :func:`_yaml_geometry`). A channel this returns is expanded into (R)/(G)/(B) components by
    the reader and must then be in ``attach_stain_luts``'s skip set — the components carry real
    color, so the LUT stays the fallback for overviews without geometry.
    """
    if not reconstruction_enabled():
        _log.info("derived color is OFF (%s / the View menu); gray channels stay gray.",
                  NO_RECONSTRUCTED_COLOR_ENV)
        return {}
    pairs = _mosaic_rgb_pngs(Path(root))
    out: dict = {}
    for entry in channels:
        name = entry.get("name")
        if name in rgb_bases or name in out:
            continue
        for rgb_name, png, doc in pairs:
            if not _matches_rgb_name(entry, rgb_name):
                continue
            geometry = _yaml_geometry(doc, png)
            if geometry is None:
                continue
            out[name] = ChromaSource(png, *geometry)
            break
    return out
