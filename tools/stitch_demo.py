"""IMA-222 evidence: coordinate-placed vs stitched, same region, same channel, same contrast.

Runs the region operator BOTH ways over one 2x2 seam neighbourhood of the 10x tissue
acquisition and writes a single side-by-side PNG, cropped at 1:1 into the seam the solve
corrected MOST and centred on the most-structured stretch of it. Both choices are made on
the coordinate-placed pane, so neither favours the stitched one. A whole-region thumbnail
would show nothing: a 15 px seam error is a fifth of a thumbnail pixel.

Nothing but the PNG touches disk: the fused mosaics live in RAM and are freed on exit.

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

# THIS checkout wins over an editable install pointing somewhere else — same bootstrap as
# tools/acceptance.py, and the reason is not hypothetical: on the build machine `squidmip`
# resolved to a different worktree entirely.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squidmip import open_reader, stitch_region  # noqa: E402

DATASET = (
    "/Users/julioamaragall/Downloads/"
    "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy"
)
# A 2x2 seam neighbourhood of manual0 (grid cols 1-2, rows 2-3). Four tiles is the smallest
# set with a four-way junction, which is where placement error is least deniable.
FOVS = [10, 11, 15, 16]
CHANNEL = 1          # Fluorescence_488_nm_Ex — measured 8x the gradient energy of 405/561
CROP = 700           # px, 1:1 (no resampling: a resampled seam is not evidence). Sized to the
#                      MEASURED 208 px overlap: the whole seam band plus ~150 px of each
#                      tile's exclusive area, which is where a doubled feature is undeniable.

# Legibility (the 20-arcmin optimal-legibility standard at 1 m => 29 px in DELIVERED pixels).
# The canvas is ~1.9k wide and will typically be viewed at ~50% in a chat/PR pane, so the
# on-canvas requirement is doubled. See the arithmetic printed at the end of a run.
_LABEL_PX = 60       # drawn cap height on canvas -> >= 30 px at 50% display
_GUTTER = 24
_BAND = 96           # label band height


def _font(target_px: int) -> ImageFont.FreeTypeFont:
    """A bold face scaled so a capital letter is *target_px* tall, measured not guessed."""
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
    """Apply ONE contrast window to a plane -> uint8. Both panes must share it or the
    comparison measures the stretch, not the stitch."""
    x = (a.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


# The seam metric moved to squidmip._benchmark at IMA-233 so the demo and the benchmark
# harness score seams with ONE implementation. Two copies would eventually disagree, and
# then the demo's claim and the benchmark table's claim would be about different things.
from squidmip._benchmark import overlap_ncc as _overlap_ncc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--region", default="manual0")
    ap.add_argument("--fovs", default=",".join(str(f) for f in FOVS))
    ap.add_argument("--channel", type=int, default=CHANNEL)
    ap.add_argument("--out", default="docs/ima-222-stitch-vs-coordinate.png")
    args = ap.parse_args()

    fovs = [int(f) for f in args.fovs.split(",")]

    # Julio's own profiler (profiling/stages.py in the stitcher repo). Optional: the demo
    # must still run for someone who only has squidmip.
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

    # Crop the four-tile junction in a COMMON PHYSICAL FRAME. Registration moves the mosaic
    # origin (the stitched mosaic here is 20 px taller), so the same pixel coordinate is NOT
    # the same place in the two mosaics -- cropping by raw index would compare two different
    # bits of tissue and prove nothing. Anchor instead on the junction of the first and last
    # FOV as each mosaic itself reports it.
    half = CROP // 2

    def crop(a, geo, rel_y, rel_x):
        """Crop around a point given RELATIVE TO FOV 0's top-left corner."""
        oy, ox = geo["origins_px"][0]
        y0 = max(0, min(int(round(oy + rel_y)) - half, a.shape[0] - CROP))
        x0 = max(0, min(int(round(ox + rel_x)) - half, a.shape[1] - CROP))
        return a[y0:y0 + CROP, x0:x0 + CROP]

    # WHICH seam to show: the adjacent pair whose RELATIVE correction is largest. That is
    # where coordinate placement is most wrong, so it is where the two panes must differ; a
    # seam the solve barely touched would make a comparison that proves nothing either way.
    o = geo_p["origins_px"]
    ty, tx = geo_p["tile_shape"]
    off = geo_s["offsets_px"]
    best = None
    for i in range(len(fovs)):
        for j in range(i + 1, len(fovs)):
            dy = abs(o[j][0] - o[i][0])
            dx = abs(o[j][1] - o[i][1])
            # A shared seam means same row or same column AND actually overlapping. DIAGONAL
            # tiles fail this: they touch only at a corner, so "the seam between them" is not
            # a place you can crop.
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
    # WHERE along that seam: a seam through blank background proves nothing, so slide along it
    # to the most-structured window. Measured on the COORDINATE pane, so the choice cannot be
    # accused of having been made to flatter the stitched one.
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
    # Only centres whose whole crop stays on the seam are candidates; sliding off the shared
    # span would centre the "seam" crop on a spot where there is no seam.
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

    # The seam, measured on the SOURCE FOVs rather than eyeballed on the render: how well the
    # two tiles agree in their overlap at the stage-reported offset vs the solved one.
    from squidmip.projection import project

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

    # ---- compose: PANES SIDE BY SIDE, never stacked --------------------------------
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
