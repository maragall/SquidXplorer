"""Side-by-side PNG of coordinate-placed vs stitched mosaics over one seam.

    python tools/stitch_demo.py [--dataset PATH] [--region manual0] [--out docs/....png]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# This checkout wins over an editable install pointing somewhere else.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squidxplorer import open_reader, stitch_region  # noqa: E402

DATASET = (
    "/Users/julioamaragall/Downloads/"
    "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy"
)
# A 2x2 seam neighbourhood of manual0: the smallest set with a four-way junction.
FOVS = [10, 11, 15, 16]
CHANNEL = 1          # Fluorescence_488_nm_Ex
CROP = 700           # px, 1:1 crop size

_LABEL_PX = 60       # drawn cap height on canvas -> >= 30 px at 50% display
_GUTTER = 24
_BAND = 96           # label band height


def _font(target_px: int) -> ImageFont.FreeTypeFont:
    """A bold face scaled so a capital letter is *target_px* tall."""
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(path).exists():
            size = target_px
            for _ in range(24):
                f = ImageFont.truetype(path, size)
                bbox = f.getbbox("H")
                cap = bbox[3] - bbox[1]
                if cap >= target_px:
                    return f
                size += 2
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _window(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Apply one contrast window to a plane -> uint8; both panes must share it."""
    x = (a.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


# The seam metric lives in squidxplorer._benchmark so demo and benchmark share one implementation.
from squidxplorer._benchmark import overlap_ncc as _overlap_ncc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--region", default="manual0")
    ap.add_argument("--fovs", default=",".join(str(f) for f in FOVS))
    ap.add_argument("--channel", type=int, default=CHANNEL)
    ap.add_argument("--out", default="docs/ima-222-stitch-vs-coordinate.png")
    args = ap.parse_args()

    fovs = [int(f) for f in args.fovs.split(",")]

    # Optional profiler; the demo must still run without it.
    try:
        from profiling.stages import StageTimer

        timer = StageTimer(time.perf_counter())
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"[warn] profiling.stages unavailable ({exc}); running untimed")
        timer = None

    reader = open_reader(args.dataset)
    meta = reader.metadata
    ch_name = meta["channels"][args.channel]["name"]
    px = float(meta["pixel_size_um"])
    tile_px = int(meta["frame_shape"][0])

    pos = meta["fov_positions_um"]
    steps = sorted({abs(pos[(args.region, a)][0] - pos[(args.region, b)][0])
                    for a in fovs for b in fovs} - {0.0})
    print(f"dataset : {args.dataset}")
    print(f"region  : {args.region}  fovs {fovs}  channel {args.channel} ({ch_name})")
    print(f"geometry: {px} um/px, tile {tile_px} px = {tile_px * px / 1000:.3f} mm, "
          f"measured x-step {steps[0]:.1f} um = {steps[0] / px:.1f} px "
          f"-> overlap {tile_px - steps[0] / px:.0f} px "
          f"({(tile_px - steps[0] / px) / tile_px:.1%})")

    kw = dict(channels=[args.channel], registration_channel=args.channel)
    geo_s: dict = {}
    geo_p: dict = {}
    t0 = time.perf_counter()
    stitched = stitch_region(reader, args.region, fovs, register=True, timer=timer,
                             geometry=geo_s, **kw)
    t_stitch = time.perf_counter() - t0
    t0 = time.perf_counter()
    placed = stitch_region(reader, args.region, fovs, register=False, geometry=geo_p, **kw)
    t_place = time.perf_counter() - t0

    if timer is not None:
        print("\nper-stage (profiling.stages.StageTimer), stitched run:")
        for name, start, end in timer.spans:
            print(f"  {name:<10} {end - start:8.1f} ms")
    print(f"  {'TOTAL':<10} {t_stitch * 1000:8.1f} ms   (coordinate-only: "
          f"{t_place * 1000:.1f} ms)")

    a_s = stitched[0, 0, 0]
    a_p = placed[0, 0, 0]
    print(f"\nmosaic  : stitched {a_s.shape}  coordinate {a_p.shape}")
    print("solved correction (px, dy dx):")
    for f, o in zip(geo_s["fovs"], geo_s["offsets_px"]):
        print(f"  fov {f:>3}  {o[0]:+7.2f} {o[1]:+7.2f}")

    # Crop in a common physical frame: registration moves the mosaic origin, so anchor
    # on FOV 0's top-left as each mosaic itself reports it.
    half = CROP // 2

    def crop(a, geo, rel_y, rel_x):
        """Crop around a point given RELATIVE TO FOV 0's top-left corner."""
        oy, ox = geo["origins_px"][0]
        y0 = max(0, min(int(round(oy + rel_y)) - half, a.shape[0] - CROP))
        x0 = max(0, min(int(round(ox + rel_x)) - half, a.shape[1] - CROP))
        return a[y0:y0 + CROP, x0:x0 + CROP]

    # Show the adjacent pair whose relative correction is largest.
    o = geo_p["origins_px"]
    ty, tx = geo_p["tile_shape"]
    off = geo_s["offsets_px"]
    best = None
    for i in range(len(fovs)):
        for j in range(i + 1, len(fovs)):
            dy = abs(o[j][0] - o[i][0])
            dx = abs(o[j][1] - o[i][1])
            # Same row or column and actually overlapping; diagonal tiles share no seam.
            same_row_or_col = dy < ty / 2 or dx < tx / 2
            if not (same_row_or_col and dy < ty and dx < tx):
                continue
            rel = float(np.hypot(*(off[j] - off[i])))
            if best is None or rel > best[0]:
                best = (rel, i, j)
    rel, i, j = best
    # Seam centre in FOV 0's frame: the midpoint of the two tiles' overlap on each axis.
    seam_y = (max(o[i][0], o[j][0]) + min(o[i][0], o[j][0]) + ty) / 2.0 - o[0][0]
    seam_x = (max(o[i][1], o[j][1]) + min(o[i][1], o[j][1]) + tx) / 2.0 - o[0][1]
    # Slide along the seam to the most-structured window, measured on the coordinate pane.
    horizontal = abs(o[j][0] - o[i][0]) > abs(o[j][1] - o[i][1])   # tiles stacked in y
    ax = 1 if horizontal else 0                                    # the axis to slide along
    span_lo = max(o[i][ax], o[j][ax])                              # the seam only EXISTS over
    span_hi = min(o[i][ax], o[j][ax]) + (tx if horizontal else ty)  # the two tiles' shared span
    if horizontal:
        strip = a_p[int(o[0][0] + seam_y) - 12:int(o[0][0] + seam_y) + 12, :]
        energy = np.abs(np.diff(strip.astype(np.float32), axis=1)).mean(axis=0)
    else:
        strip = a_p[:, int(o[0][1] + seam_x) - 12:int(o[0][1] + seam_x) + 12]
        energy = np.abs(np.diff(strip.astype(np.float32), axis=0)).mean(axis=1)
    smooth = np.convolve(energy, np.ones(CROP) / CROP, "same")
    # Only centres whose whole crop stays on the shared span are candidates.
    lo_i = int(span_lo) + half
    hi_i = max(lo_i + 1, int(span_hi) - half)
    centre = float(lo_i + int(np.argmax(smooth[lo_i:hi_i])))
    if horizontal:
        seam_x = centre - o[0][1]
    else:
        seam_y = centre - o[0][0]
    print(f"crop    : seam FOV {fovs[i]}|{fovs[j]} — the largest relative correction "
          f"({rel:.1f} px) — centred at +{seam_y:.0f}, +{seam_x:.0f} from FOV "
          f"{fovs[0]}'s top-left")

    c_s, c_p = crop(a_s, geo_s, seam_y, seam_x), crop(a_p, geo_p, seam_y, seam_x)

    # ONE window, from the stitched pane, applied to both.
    lo, hi = np.percentile(c_s[c_s > 0], (1.0, 99.5))
    img_s, img_p = _window(c_s, lo, hi), _window(c_p, lo, hi)

    # Score the seam on the source FOVs at the stage-reported vs solved offsets.
    from squidxplorer.projection import project

    z_levels = meta["z_levels"]
    mip_i, mip_j = (
        project(reader.read(args.region, fovs[k], ch_name, z, 0) for z in z_levels)
        for k in (i, j)
    )
    d_stage = (o[j][0] - o[i][0], o[j][1] - o[i][1])
    os_ = geo_s["origins_px"]
    d_reg = (os_[j][0] - os_[i][0], os_[j][1] - os_[i][1])
    ncc_p = _overlap_ncc(mip_i, mip_j, *d_stage)
    ncc_s = _overlap_ncc(mip_i, mip_j, *d_reg)
    print(f"seam FOV {fovs[i]}|{fovs[j]} overlap NCC: coordinate {ncc_p:.3f}  "
          f"-> stitched {ncc_s:.3f}   (offset {d_stage[0]:.1f},{d_stage[1]:.1f} px "
          f"-> {d_reg[0]:.1f},{d_reg[1]:.1f} px)")

    # compose: panes side by side
    w = CROP * 2 + _GUTTER * 3
    h = CROP + _BAND + _GUTTER * 2
    canvas = Image.new("RGB", (w, h), (16, 16, 18))
    canvas.paste(Image.fromarray(img_p).convert("RGB"), (_GUTTER, _BAND + _GUTTER))
    canvas.paste(Image.fromarray(img_s).convert("RGB"), (_GUTTER * 2 + CROP, _BAND + _GUTTER))

    draw = ImageDraw.Draw(canvas)
    font = _font(_LABEL_PX)
    cap = font.getbbox("H")
    cap_px = cap[3] - cap[1]
    for x, text, color in (
        (_GUTTER, "COORDINATE", (255, 120, 110)),
        (_GUTTER * 2 + CROP, "STITCHED", (120, 235, 160)),
    ):
        draw.text((x, _GUTTER - cap[1]), text, font=font, fill=color)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"\nwrote {out.resolve()}  ({w}x{h}, {out.stat().st_size / 1024:.0f} KB)")
    print(f"label cap height {cap_px} px on a {w} px canvas -> "
          f"{cap_px / 2:.0f} px at 50% display (>= 29 px required)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
