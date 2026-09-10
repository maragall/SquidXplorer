#!/usr/bin/env python
"""Scripted stills of the RAW whole-FOV volume, 561 only: oblique hero, XZ, XY, real GL.

Launch the real GUI (cocoa GL), let the working layout open its view, open the whole-region
bricked 3D (2304 px bricks past the 2048 GL cap - the drawn-ROI clamp guards interaction,
not a scripted still), apply fixed raw-derived contrast windows, wait full brick residency
per pose, screenshot each pose at the canvas's native size.

Usage: python scripts/volume_figure.py   # replaces a running instance (single-window cap)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# This checkout's package (SQUIDXPLORER_REPO overrides when the script runs from elsewhere).
REPO = Path(os.environ.get("SQUIDXPLORER_REPO") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(REPO))

# Stride-1 residency needs ~1.47 GB of bricks; the measured 10%-of-available budget is less.
os.environ.setdefault("SQUIDXPLORER_CACHE_MB", "2200")

RAW_SET = Path("/Users/julioamaragall/Downloads/25x_C4_dz=3_2026-08-14_16-51-15.744692")
OUT_DIR = Path("/private/tmp/claude-501/-Users-julioamaragall-CEPHLA-projects-SquidXplorer"
               "--claude-worktrees-plate-navigator-port/3cd1b779-3898-4c7a-87cd-4e0ebbb2b711"
               "/scratchpad")

# Julio, mid-render: "We are only going to use 561." The one visible channel, windowed at
# the raw z-MIP 1/99.8 percentiles; the others hide through the layer model.
ONLY_CHANNEL = "Fluorescence_561_nm_Ex"
CONTRAST_561 = (135.0, 1109.0)

# (name, pose, zoom-after-frame); hero matches _camera_script's OBLIQUE_HERO / DEMO_ZOOM.
POSES = [("hero", (0.0, 16.7, 58.8), 1.15), ("xz", "xz", None), ("xy", "xy", None)]

RESIDENCY_TIMEOUT_S = 300.0
FALLBACK_ROI_PX = 2048


def _norm(name: str) -> str:
    return str(name).replace(" ", "_")


def _pump(app, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _wait_resident(view, label: str) -> float:
    """Residency wait for the current epoch; returns seconds, raises on timeout."""
    from squidxplorer._camera_script import wait_bricks_resident

    t0 = time.monotonic()
    ok = wait_bricks_resident(view, timeout_s=RESIDENCY_TIMEOUT_S)
    dt = time.monotonic() - t0
    print(f"residency [{label}]: {'settled' if ok else 'TIMEOUT'} in {dt:.1f} s", flush=True)
    if not ok:
        raise TimeoutError(f"bricks not resident after {RESIDENCY_TIMEOUT_S:.0f} s ({label})")
    return dt


def _centered_roi_um(meta: dict, region: str, side_px: int) -> tuple:
    """A side_px x side_px stage-um box centred on the region's mosaic; deterministic."""
    from squidxplorer._mosaic_source import mosaic_fov_bboxes_um

    boxes = mosaic_fov_bboxes_um(meta, region)
    x0 = min(b[0] for b in boxes.values())
    y0 = min(b[1] for b in boxes.values())
    x1 = max(b[2] for b in boxes.values())
    y1 = max(b[3] for b in boxes.values())
    px = float(meta["pixel_size_um"])
    half = side_px * px / 2.0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def run_pass(source: Path, prefix: str) -> int:
    import imageio.v3 as iio
    import psutil

    from squidxplorer import _viewer, _volume_view

    avail0 = psutil.virtual_memory().available
    _viewer.enable_hidpi()
    app = _viewer.qt_app([])
    win = _viewer.PlateWindow(str(source), default_layout=True, tabbed_views=True)
    win.show()

    # The working layout spawns the one view on showEvent; wait for it, then for raw layers.
    view = None
    end = time.monotonic() + 60
    while view is None and time.monotonic() < end:
        app.processEvents()
        wins = list(win._viewer_manager.windows)
        view = wins[0] if wins else None
        time.sleep(0.02)
    if view is None:
        print("FAIL: the default view never opened", flush=True)
        return 1

    mosaic = view._pane.mosaic
    names = [c["name"] for c in (view._meta or {}).get("channels", [])]
    end = time.monotonic() + 180
    while len(mosaic.channels("raw")) < len(names) and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)
    got = mosaic.channels("raw")
    if len(got) < len(names):
        print(f"FAIL: raw mosaic loaded {got} of {names}", flush=True)
        return 1

    # 561 visible and windowed on the 2D layers BEFORE open_3d: the bricks seed from this
    # scene, taking each hidden identity's visibility with them (loads still cover every
    # declared channel; the render and the GPU textures are 561-only).
    target = next((n for n in names if _norm(n) == ONLY_CHANNEL), None)
    if target is None:
        print(f"FAIL: channel {ONLY_CHANNEL} not among {names}", flush=True)
        return 1
    for name in names:
        mosaic.set_channel_visible(name, name == target)
    mosaic.set_contrast(target, *CONTRAST_561)
    _pump(app, 0.5)

    # Whole region = the whole 2304 px FOV; no drawn ROI, so no interactive clamp applies.
    view._roi_bbox = None
    _volume_view.open_3d(view)
    fallback = False
    try:
        if view._native3d is None:
            raise TimeoutError("open refused")
        _wait_resident(view, f"{prefix} open (full FOV)")
    except TimeoutError as exc:
        # ONE fallback: a centred 2048 px ROI, said out loud.
        print(f"FALLBACK: full-FOV volume failed ({exc}); "
              f"retrying a centred {FALLBACK_ROI_PX} px ROI", flush=True)
        fallback = True
        _volume_view.close_native3d(view)
        region = view.current_region()
        view._roi_bbox = _centered_roi_um(view._meta, region, FALLBACK_ROI_PX)
        _volume_view.open_3d(view)
        if view._native3d is None:
            print("FAIL: the fallback ROI volume did not open either", flush=True)
            return 1
        _wait_resident(view, f"{prefix} open (fallback ROI)")

    # Re-assert 561-only AFTER the open: the identity moved onto the bricks at open, and a
    # brick's first layer arrives visible, so the pre-open hide does not reach them. The
    # channel toggle resolves over the live brick layers (the G7 channel-toggle path).
    for name in names:
        mosaic.set_channel_visible(name, name == target)
    mosaic.set_contrast(target, *CONTRAST_561)
    _pump(app, 0.3)

    vol = view._native3d
    for pose_name, pose, zoom in POSES:
        _volume_view.snap_camera(view, pose)
        if zoom is not None:
            cam = vol._viewer.camera
            cam.zoom = float(cam.zoom) * float(zoom)
            _volume_view.refresh_bricks(view)
        _wait_resident(view, f"{prefix} {pose_name}")
        _pump(app, 0.5)                       # let the settled bricks actually draw
        shot = vol._viewer.screenshot(canvas_only=True, flash=False)
        out = OUT_DIR / f"{prefix}_{pose_name}.png"
        iio.imwrite(out, shot[..., :3])
        print(f"saved {out} canvas {shot.shape[1]}x{shot.shape[0]}"
              f"{' (FALLBACK ROI)' if fallback else ' (full FOV)'}", flush=True)

    print(f"memory: available {avail0 / 1e9:.1f} -> "
          f"{psutil.virtual_memory().available / 1e9:.1f} GB", flush=True)
    # Restore every channel before teardown: closing over hidden 2D layers segfaulted at
    # exit (measured, exit 139 after the captures); the all-visible closes exited 0.
    for name in names:
        mosaic.set_channel_visible(name, True)
    _pump(app, 0.3)
    _volume_view.close_native3d(view)
    win.close()
    _pump(app, 0.5)
    return 0


def main(argv: list) -> int:
    """No args: the raw pass. Or: SOURCE PREFIX (the decon pass reuses the same pipeline)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(argv[0]) if argv else RAW_SET
    prefix = argv[1] if len(argv) > 1 else "raw_volume"
    # The app caps real windows at one instance; Julio approved replacing his.
    subprocess.run(["pkill", "-f", "squidxplorer._viewer import main"], check=False)
    time.sleep(2)
    return run_pass(source, prefix)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
