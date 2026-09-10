#!/usr/bin/env python
"""Raw-beside-decon comparison stills at the QC'd poses (hero, XZ), 561 only, real GL.

The decon floor is DERIVED, not tasted: the decon drained background flux into puncta, so
under raw's own (135, 1109) window the decon side renders near-binary. Per decon set, the
floor is the 561 z-MIP quantile at raw's below-135 pixel fraction, so both sides black out
the same fraction of the field; the ceiling stays raw's 1109 (clipping is measured and
reported, never silently rewindowed). One boot per source through volume_figure's shared
machinery; composites are raw | decon with a 4 px divider.

Usage: python scripts/compare_figures.py --out-dir DIR   # replaces a running instance
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))            # volume_figure's shared boot/teardown

RAW_SET = Path("/Users/julioamaragall/Downloads/25x_C4_dz=3_2026-08-14_16-51-15.744692")
DECON_SETS = {
    "i1": RAW_SET.parent / f"decon46_i1_{RAW_SET.name}",
    "i2": RAW_SET.parent / f"decon46_i2_{RAW_SET.name}",
}
RAW_WINDOW = (135.0, 1109.0)
POSE_NAMES = ("hero", "xz")              # the QC'd poses; xy skipped on Julio's word
DIVIDER_PX = 4


def channel_mip(source: Path) -> np.ndarray:
    """The 561 z-MIP over every field of *source*, one (Y, X) array per field, stacked."""
    from volume_figure import ONLY_CHANNEL, _norm

    from squidxplorer.reader import open_reader

    reader = open_reader(source)
    meta = reader.metadata
    name = next(c["name"] for c in meta["channels"] if _norm(c["name"]) == ONLY_CHANNEL)
    mips = []
    for region, fovs in meta["fovs_per_region"].items():
        for fov in fovs:
            stack = np.stack([reader.read(region, fov, name, z)
                              for z in range(meta["n_z"])])
            mips.append(stack.max(axis=0))
    return np.stack(mips)


def derived_floor(raw_mip: np.ndarray, decon_mip: np.ndarray, raw_floor: float) -> tuple:
    """(floor, raw_fraction, decon_fraction_at_floor): match below-floor pixel fractions."""
    frac = float(np.mean(raw_mip < raw_floor))
    floor = float(np.quantile(decon_mip, frac))
    return floor, frac, float(np.mean(decon_mip < floor))


def figure_pass(source: Path, prefix: str, window: tuple, out_dir: Path) -> int:
    """One booted app, hero + XZ screenshots; runs alone in its own process."""
    import imageio.v3 as iio

    import volume_figure as vf

    from squidxplorer import _volume_view

    booted = vf.boot_561_volume(source, prefix, contrast=window)
    if booted is None:
        return 1
    app, win, view, names, target, fallback = booted
    vol = view._native3d
    for pose_name, pose, zoom in vf.POSES:
        if pose_name not in POSE_NAMES:
            continue
        _volume_view.snap_camera(view, pose)
        if zoom is not None:
            cam = vol._viewer.camera
            cam.zoom = float(cam.zoom) * float(zoom)
            _volume_view.refresh_bricks(view)
        vf._wait_resident(view, f"{prefix} {pose_name}")
        vf._pump(app, 0.5)               # let the settled bricks actually draw
        shot = vol._viewer.screenshot(canvas_only=True, flash=False)
        out = out_dir / f"{prefix}_{pose_name}.png"
        iio.imwrite(out, shot[..., :3])
        print(f"saved {out} canvas {shot.shape[1]}x{shot.shape[0]}"
              f"{' (FALLBACK ROI)' if fallback else ' (full FOV)'}", flush=True)
    vf.teardown(app, win, view, names)
    return 0


def _pass_pngs(out_dir: Path, prefix: str) -> list:
    return [out_dir / f"{prefix}_{p}.png" for p in POSE_NAMES]


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

    paths = []
    for key in DECON_SETS:
        for pose in POSE_NAMES:
            a = iio.imread(out_dir / f"raw_{pose}.png")
            b = iio.imread(out_dir / f"{key}_{pose}.png")
            h = min(a.shape[0], b.shape[0])
            divider = np.full((h, DIVIDER_PX, 3), 64, np.uint8)
            out = out_dir / f"compare_{key}_{pose}.png"
            iio.imwrite(out, np.concatenate([a[:h], divider, b[:h]], axis=1))
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

    raw_mip = channel_mip(RAW_SET)
    ceil = RAW_WINDOW[1]
    print(f"raw 561 z-MIP: {np.mean(raw_mip < RAW_WINDOW[0]) * 100:.2f}% below "
          f"{RAW_WINDOW[0]:.0f}, {np.mean(raw_mip > ceil) * 100:.2f}% above {ceil:.0f}",
          flush=True)
    windows = {"raw": RAW_WINDOW}
    for key, path in DECON_SETS.items():
        mip = channel_mip(path)
        floor, frac, got = derived_floor(raw_mip, mip, RAW_WINDOW[0])
        print(f"{key} floor {floor:.1f}: raw fraction below {RAW_WINDOW[0]:.0f} is "
              f"{frac * 100:.2f}%, {key} fraction below {floor:.1f} is {got * 100:.2f}%; "
              f"{np.mean(mip > ceil) * 100:.2f}% above the shared ceiling {ceil:.0f}",
              flush=True)
        windows[key] = (floor, ceil)

    for prefix, source in (("raw", RAW_SET), *DECON_SETS.items()):
        if not run_pass_subprocess(source, prefix, windows[prefix], args.out_dir):
            print(f"FAIL: the {prefix} pass left no complete stills", flush=True)
            return 1
    compose(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
