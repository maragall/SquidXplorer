#!/usr/bin/env python
"""Wrong-PSF vs corrected-PSF decon proof figures, 561 only, real GL, one boot per side.

The 25x_C4 acquisition was recorded under the wrong rig profile (20x, NA 0.8, pixel
0.325 um, NI air); the physical truth is 25x, NA 0.85, water (NI 1.333), pixel 0.26 um.
decon46_i2_* was solved with the recorded (wrong) PSF; decon46_i2_25x_* with the
corrected one (decon_whole.py --na 0.85 --dxy-um 0.26 --ni 1.333, PSF only - display
geometry everywhere stays the recorded 0.325).

Three composites, each side at its OWN SOP matched window (own rule floor, own p99.999
561 z-MIP ceiling, decon_demo_sop.matched_window), gray at gamma 0.7, turbo at gamma 1:
  compare_psf_i2.png            wrong-PSF i2 | corrected i2_25x, hero pose
  compare_zoom_i2_25x_g0.7.png  raw | corrected i2_25x, 5x zoom
  mirror_seam_i2_25x_turbo.png  raw | fliplr(corrected i2_25x), 5x zoom, turbo, no divider

Usage: python scripts/compare_psf_figures.py --out-dir DIR   # replaces a running app
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
sys.path.insert(0, str(HERE))            # volume_figure's shared boot/teardown

RAW_SET = Path("/Users/julioamaragall/Downloads/25x_C4_dz=3_2026-08-14_16-51-15.744692")
SETS = {
    "raw": RAW_SET,
    "wrong_i2": RAW_SET.parent / f"decon46_i2_{RAW_SET.name}",
    "corr_i2": RAW_SET.parent / f"decon46_i2_25x_{RAW_SET.name}",
}
GAMMA = 0.7
ZOOM5X = 5.0                             # vs the hero's 1.15, both over the fit zoom
LOOKS = ("hero_g0.7", "zoom5x_g0.7", "zoom5x_turbo")
DIVIDER_PX = 4
DESKTOP = Path("/Users/julioamaragall/Desktop")


def _set_look(view, target: str, gamma: float, colormap) -> None:
    """Gamma and colormap on every layer of the 561 identity (IDENTITY_PROPs, so the
    mirror and the bricks agree); the shader renders both into the capture."""
    mosaic = view._pane.mosaic
    with mosaic.programmatic():
        for op in mosaic.ops():
            for ly in mosaic.layers_for(op, target):
                ly.gamma = float(gamma)
                if colormap is not None:
                    ly.colormap = colormap


def figure_pass(source: Path, prefix: str, window: tuple, out_dir: Path) -> int:
    """One booted app: hero and 5x-zoom captures at gamma 0.7, plus 5x-zoom turbo."""
    import imageio.v3 as iio
    import volume_figure as vf

    from record_volume_comparison import _pin_canvas
    from squidxplorer import _volume_view

    booted = vf.boot_561_volume(source, prefix, contrast=window)
    if booted is None:
        return 1
    app, win, view, names, target, fallback = booted
    vol = view._native3d
    _pin_canvas(app, view, vol)
    cam = vol._viewer.camera
    base_zoom = float(cam.zoom)          # the fit zoom; snap_camera never touches zoom
    hero = next(p for p in vf.POSES if p[0] == "hero")
    _volume_view.snap_camera(view, hero[1])

    def capture(name: str) -> None:
        vf._wait_resident(view, f"{prefix} {name}")
        vf._pump(app, 0.5)
        shot = np.asarray(vol._viewer.screenshot(canvas_only=True, flash=False))[..., :3]
        out = out_dir / f"{prefix}_{name}.png"
        iio.imwrite(out, shot)
        print(f"saved {out} canvas {shot.shape[1]}x{shot.shape[0]}"
              f"{' (FALLBACK ROI)' if fallback else ' (full FOV)'}", flush=True)

    _set_look(view, target, GAMMA, None)
    cam.zoom = base_zoom * float(hero[2])
    _volume_view.refresh_bricks(view)
    capture("hero_g0.7")

    cam.zoom = base_zoom * ZOOM5X
    _volume_view.refresh_bricks(view)
    capture("zoom5x_g0.7")

    _set_look(view, target, 1.0, "turbo")
    vf._pump(app, 0.4)
    capture("zoom5x_turbo")

    vf.teardown(app, win, view, names)
    return 0


def _pass_pngs(out_dir: Path, prefix: str) -> list:
    return [out_dir / f"{prefix}_{look}.png" for look in LOOKS]


def run_pass_subprocess(source: Path, prefix: str, window: tuple, out_dir: Path) -> bool:
    """Spawn the pass; a teardown crash after every PNG landed is tolerated as before."""
    subprocess.run(["pkill", "-f", "squidxplorer._viewer import main"], check=False)
    time.sleep(2)
    rc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--out-dir", str(out_dir),
         "--pass", str(source), prefix, str(window[0]), str(window[1])],
        check=False).returncode
    done = all(p.exists() for p in _pass_pngs(out_dir, prefix))
    if rc != 0 and done:
        print(f"WARNING: the {prefix} pass exited {rc} at teardown; "
              f"its stills are whole, continuing", flush=True)
    return done


def compose(out_dir: Path) -> list:
    import imageio.v3 as iio

    def beside(a, b, divider_px: int):
        h = min(a.shape[0], b.shape[0])
        parts = [a[:h]]
        if divider_px:
            parts.append(np.full((h, divider_px, 3), 64, np.uint8))
        parts.append(b[:h])
        return np.concatenate(parts, axis=1)

    plan = [
        ("compare_psf_i2.png", "wrong_i2_hero_g0.7", "corr_i2_hero_g0.7",
         DIVIDER_PX, False),
        ("compare_zoom_i2_25x_g0.7.png", "raw_zoom5x_g0.7", "corr_i2_zoom5x_g0.7",
         DIVIDER_PX, False),
        ("mirror_seam_i2_25x_turbo.png", "raw_zoom5x_turbo", "corr_i2_zoom5x_turbo",
         0, True),
    ]
    paths = []
    for name, left, right, div, mirror in plan:
        a = iio.imread(out_dir / f"{left}.png")
        b = iio.imread(out_dir / f"{right}.png")
        if mirror:
            b = b[:, ::-1]
        out = out_dir / name
        iio.imwrite(out, beside(a, b, div))
        paths.append(out)
        print(f"composed {out}", flush=True)
    return paths


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pass", dest="pass_args", nargs=4,
                    metavar=("SOURCE", "PREFIX", "FLOOR", "CEIL"),
                    help="internal: run one stills pass in this process")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.pass_args:
        src, prefix, lo, hi = args.pass_args
        return figure_pass(Path(src), prefix, (float(lo), float(hi)), args.out_dir)

    from decon_demo_sop import matched_window

    for prefix, source in SETS.items():
        if not source.exists():
            print(f"FAIL: {source} does not exist", flush=True)
            return 1
        if all(p.exists() for p in _pass_pngs(args.out_dir, prefix)):
            print(f"reusing existing {prefix} stills", flush=True)
            continue
        window = matched_window(source)
        print(f"{prefix} matched window ({window[0]:.1f}, {window[1]:.1f})", flush=True)
        if not run_pass_subprocess(source, prefix, window, args.out_dir):
            print(f"FAIL: the {prefix} pass left no complete stills", flush=True)
            return 1

    finals = compose(args.out_dir)
    for png in finals:
        shutil.copy2(png, DESKTOP / png.name)
    subprocess.run(["open", "-a", "Preview",
                    *(str(DESKTOP / p.name) for p in finals)], check=False)
    print(f"copied {len(finals)} figure(s) to {DESKTOP}, opened together in Preview",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
