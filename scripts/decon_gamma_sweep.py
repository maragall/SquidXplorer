#!/usr/bin/env python
"""Decon gamma sweep: ONE GL boot of decon46_i2, hero pose, windows (95, 7123) and
(95, 12395) crossed with gamma {0.9, 0.7, 0.55, 0.4} - 8 recaptures, no reboot.

Julio on the linear ramp: "I like the peaks, but now the low intensity is too low."
A peak-preserving ceiling with gamma below 1 lifts the low end without clipping;
napari renders gamma in the shader, so the captures carry it. Deliverable: eight
self-describing PNGs on the Desktop, opened together in one Preview window.

Usage: python scripts/decon_gamma_sweep.py --out-dir DIR   # replaces a running app
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
CEILINGS = (7123, 12395)                     # p99.999 and the MIP max: peak-preserving
GAMMAS = (0.9, 0.7, 0.55, 0.4)
DESKTOP = Path("/Users/julioamaragall/Desktop")


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import imageio.v3 as iio
    import volume_figure as vf

    from compare_figures import DECON_SETS
    from record_volume_comparison import _pin_canvas
    from squidxplorer import _volume_view
    from squidxplorer._camera_script import wait_bricks_resident

    # Julio's GUI holds the single app slot; the sweep takes it, as coordinated.
    subprocess.run(["pkill", "-f", "squidxplorer._viewer import main"], check=False)
    time.sleep(2)

    booted = vf.boot_561_volume(DECON_SETS["i2"], "i2_gamma",
                                contrast=(FLOOR, float(CEILINGS[0])))
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
    pngs = []
    for c in CEILINGS:
        mosaic.set_contrast(target, FLOOR, float(c))
        for g in GAMMAS:
            # Gamma is an IDENTITY_PROP: write every layer of the identity so the
            # mirror and the bricks agree; the shader renders it into the capture.
            with mosaic.programmatic():
                for op in mosaic.ops():
                    for ly in mosaic.layers_for(op, target):
                        ly.gamma = float(g)
            vf._pump(app, 0.4)
            shot = np.asarray(
                vol._viewer.screenshot(canvas_only=True, flash=False))[..., :3]
            png = args.out_dir / f"decon_i2_w95-{c}_g{g:g}.png"
            iio.imwrite(png, shot)
            pngs.append(png)
            print(f"captured {png}", flush=True)

    for png in pngs:
        shutil.copy2(png, DESKTOP / png.name)
    subprocess.run(["open", "-a", "Preview",
                    *(str(DESKTOP / p.name) for p in pngs)], check=False)
    print(f"copied {len(pngs)} PNG(s) to {DESKTOP}, opened together in Preview",
          flush=True)

    vf.teardown(app, win, view, names)       # a teardown crash cannot cost the PNGs
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
