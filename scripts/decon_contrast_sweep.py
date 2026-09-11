#!/usr/bin/env python
"""Decon-only ceiling sweep: ONE GL boot of decon46_i2, hero pose, six 561 windows
recaptured without rebooting - contrast is a display map, the bricks never change.

Ceilings come from the i2 561 z-MIP's own tail (p99, p99.5, p99.9, p99.99, p99.999,
max), floor fixed at Julio's settled 95. Output: one labeled 2x3 contact sheet (also
copied to the Desktop and opened in Preview for immediate QC) plus the six PNGs.

Usage: python scripts/decon_contrast_sweep.py --out-dir DIR   # replaces a running app
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

FLOOR = 95.0
PERCENTILES = (99.0, 99.5, 99.9, 99.99, 99.999, 100.0)
DESKTOP = Path("/Users/julioamaragall/Desktop")
TILE_LABEL_PX = 96


def candidates() -> list:
    """[(percentile, ceiling, clip_pct)] off the i2 561 z-MIP's own tail."""
    from compare_figures import DECON_SETS, channel_mip

    mip = channel_mip(DECON_SETS["i2"])
    out = []
    for p in PERCENTILES:
        c = float(mip.max()) if p == 100.0 else float(np.percentile(mip, p))
        out.append((p, c, float(np.mean(mip > c)) * 100.0))
    return out


def _label_tile(frame: np.ndarray, text: str) -> np.ndarray:
    """The tile with a black strip above it carrying *text* in large type."""
    from PIL import Image, ImageDraw, ImageFont

    strip = Image.new("RGB", (frame.shape[1], TILE_LABEL_PX), (0, 0, 0))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.load_default(size=64)
    except TypeError:                        # older Pillow: the small bitmap font
        font = ImageFont.load_default()
    draw.text((24, 12), text, fill=(235, 235, 235), font=font)
    return np.concatenate([np.asarray(strip), frame], axis=0)


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cands = candidates()
    for p, c, clip in cands:
        name = "max" if p == 100.0 else f"p{p:g}"
        print(f"{name}: ceiling {c:.0f}, clips {clip:.3f}% of the MIP", flush=True)

    import imageio.v3 as iio
    import volume_figure as vf

    from compare_figures import DECON_SETS
    from record_volume_comparison import _pin_canvas
    from squidxplorer import _volume_view
    from squidxplorer._camera_script import wait_bricks_resident

    # Julio's GUI holds the single app slot; he has been told the render takes it.
    subprocess.run(["pkill", "-f", "squidxplorer._viewer import main"], check=False)
    time.sleep(2)

    booted = vf.boot_561_volume(DECON_SETS["i2"], "i2_sweep",
                                contrast=(FLOOR, cands[0][1]))
    if booted is None:
        return 1
    app, win, view, names, target, fallback = booted
    vol = view._native3d
    _pin_canvas(app, view, vol)
    hero = next(p for p in vf.POSES if p[0] == "hero")
    _volume_view.snap_camera(view, hero[1])
    if hero[2] is not None:
        cam = vol._viewer.camera
        cam.zoom = float(cam.zoom) * float(hero[2])
        _volume_view.refresh_bricks(view)
    wait_bricks_resident(view, timeout_s=300.0)
    vf._pump(app, 0.5)

    mosaic = view._pane.mosaic
    tiles = []
    for p, c, clip in cands:
        mosaic.set_contrast(target, FLOOR, c)   # display map only: no read, no wait
        vf._pump(app, 0.4)
        shot = np.asarray(vol._viewer.screenshot(canvas_only=True, flash=False))[..., :3]
        name = "pmax" if p == 100.0 else f"p{p:g}"
        # Self-describing, Julio's QC flow: the six files ARE the deliverable.
        png = args.out_dir / (f"decon_i2_w{int(FLOOR)}-{int(round(c))}"
                              f"_{name}_clip{clip:.3f}.png")
        iio.imwrite(png, shot)
        print(f"captured {png}", flush=True)
        tiles.append(_label_tile(shot, f"(95, {c:.0f}) - {name} - clips {clip:.3f}%"))

    pngs = sorted(args.out_dir.glob("decon_i2_w*_clip*.png"))
    for png in pngs:
        shutil.copy2(png, DESKTOP / png.name)
    subprocess.run(["open", "-a", "Preview",
                    *(str(DESKTOP / p.name) for p in pngs)], check=False)
    print(f"copied {len(pngs)} PNG(s) to {DESKTOP}, opened together in Preview",
          flush=True)
    # The contact sheet stays as a bonus beside the six files.
    rows = [np.concatenate(tiles[i:i + 3], axis=1) for i in (0, 3)]
    sheet_path = args.out_dir / "i2_ceiling_sweep.png"
    iio.imwrite(sheet_path, np.concatenate(rows, axis=0))
    print(f"contact sheet {sheet_path}", flush=True)

    vf.teardown(app, win, view, names)       # a teardown crash cannot cost the sheet
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
