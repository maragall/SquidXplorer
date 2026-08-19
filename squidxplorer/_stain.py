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
from pathlib import Path
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)

#: The white point percentile shared by the LUT fit and the display seed window. The LUT's t is
#: transmittance, so white IS this percentile of the data and the faithful window is [0, white].
STAIN_WHITE_PERCENTILE = 99

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

#: Chroma ratio ceiling; stain hues live well inside it, and a division artifact must not glow.
_RATIO_MAX = 4.0
#: PNG green at/below this is unwritten mosaic area, not tissue: ratio stays neutral there.
_CHROMA_G_FLOOR = 2.0
#: Per-FOV ratio windows kept in memory (each ~1.3 MB at 2 um over a 1900 px frame).
_CHROMA_CACHE_MAX = 32


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
    """

    def __init__(self, png_path, top_left_mm_yx, resolution_um) -> None:
        self._png_path = Path(png_path)
        self._top_mm = (float(top_left_mm_yx[0]), float(top_left_mm_yx[1]))
        self._res_um = float(resolution_um)
        self._png: Optional[np.ndarray] = None
        self._size: Optional[tuple] = None
        #: (region, fov) -> (2, h, w) float32 [R/G, B/G] window; bounded LRU.
        self._windows: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

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
        """The FOV's (2, h, w) float32 [R/G, B/G] window at PNG resolution, cached (LRU)."""
        key = (str(region), int(fov))
        cached = self._windows.get(key)
        if cached is not None:
            self._windows.move_to_end(key)
            return cached
        r0, c0, h, w = self._window_box(x_um, y_um, frame_shape, pixel_size_um)
        out = np.ones((2, h, w), dtype=np.float32)
        rows, cols = self._png_size()
        ra, rb = max(r0, 0), min(r0 + h, rows)
        ca, cb = max(c0, 0), min(c0 + w, cols)
        if ra < rb and ca < cb:
            crop = self._image()[ra:rb, ca:cb].astype(np.float32)
            g = crop[..., 1]
            usable = g > _CHROMA_G_FLOOR           # near-black PNG is unwritten, not tissue
            with np.errstate(divide="ignore", invalid="ignore"):
                for i, comp in enumerate((0, 2)):
                    ratio = np.where(usable, crop[..., comp] / np.maximum(g, 1.0), 1.0)
                    out[i, ra - r0:rb - r0, ca - c0:cb - c0] = np.clip(ratio, 0.0, _RATIO_MAX)
        if len(self._windows) >= _CHROMA_CACHE_MAX:
            self._windows.popitem(last=False)
        self._windows[key] = out
        return out

    def component_plane(self, plane: np.ndarray, component: int, region: str, fov: int,
                        x_um: float, y_um: float, pixel_size_um: float) -> np.ndarray:
        """One chroma component of a color-recorded-gray plane, in the plane's own dtype.

        Component 1 (G) is the file's own pixels untouched; 0 (R) and 2 (B) scale them by the
        upsampled local ratio. The result is clipped to the dtype's range and cast back.
        """
        if component == 1:
            return plane
        ratios = self._ratios(region, fov, x_um, y_um, plane.shape, pixel_size_um)
        ratio = _upsample_bilinear(ratios[0 if component == 0 else 1], plane.shape)
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
