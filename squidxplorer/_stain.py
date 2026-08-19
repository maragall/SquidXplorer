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
from pathlib import Path
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)

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


def _mosaic_rgb_pngs(root: Path) -> list[tuple[str, Path]]:
    """``(rgb_channel_name, png_path)`` pairs from every mosaic_view yaml under *root*."""
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
                out.append((str(name), png))
    return out


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
    white = np.percentile(png.reshape(-1, png.shape[2])[:, :3], 99, axis=0)
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

    ``rgb_bases`` (channels whose files really are 3-component) are skipped: they expand into
    true primaries and need no reconstruction.
    """
    pairs = _mosaic_rgb_pngs(Path(root))
    if not pairs:
        return
    for entry in channels:
        if entry.get("name") in rgb_bases or entry.get("display_lut"):
            continue
        want = (_norm(entry.get("display_name") or ""), _norm(entry.get("name") or ""))
        for rgb_name, png in pairs:
            if not any(w and _norm(rgb_name).endswith(w) for w in want):
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
