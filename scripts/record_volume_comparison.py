#!/usr/bin/env python
"""Seeded-orbit raw-vs-decon comparison video, 561 only, whole-FOV bricked 3D, real GL.

Two sequential headful passes over volume_figure's boot (raw set, then the decon46 set),
each recording THE SAME random_orbit_steps(seed) list through run_camera_script, then the
two .mp4s composed frame k beside frame k with a 4 px divider. Sync is structural: one
steps list, two passes; the seed buys reproducibility and variety across videos. Both
passes window 561 at volume_figure's CONTRAST_561, one intensity scale.

Usage: python scripts/record_volume_comparison.py [--seed N]   # replaces a running instance
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))            # volume_figure's shared boot/teardown

RAW_SET = Path("/Users/julioamaragall/Downloads/25x_C4_dz=3_2026-08-14_16-51-15.744692")
DECON_SET = Path("/Users/julioamaragall/Downloads/"
                 "decon46_25x_C4_dz=3_2026-08-14_16-51-15.744692")
DESKTOP = Path("/Users/julioamaragall/Desktop")
DEFAULT_SEED = 4242
FPS = 12
RESIDENCY_TIMEOUT_S = 300.0


def record_pass(source: Path, out_mp4: Path, seed: int) -> int:
    """One booted app, one recorded orbit; runs alone in its own process (one GL app)."""
    import volume_figure as vf

    from squidxplorer._camera_script import (random_orbit_steps, run_camera_script,
                                             wait_bricks_resident)

    booted = vf.boot_561_volume(source, source.name)
    if booted is None:
        return 1
    app, win, view, names, target, fallback = booted
    if fallback:
        print(f"WARNING: {source.name} rendered the FALLBACK ROI, not the full FOV",
              flush=True)
    steps = random_orbit_steps(seed)
    result = run_camera_script(
        view, steps, out_path=str(out_mp4), fps=FPS,
        wait_ready=lambda w: wait_bricks_resident(w, timeout_s=RESIDENCY_TIMEOUT_S))
    print(f"pass {source.name}: {result.n_frames} frame(s) -> {result.path}", flush=True)
    # Let the last snap's brick refresh settle before closing: closing over a busy loader
    # segfaulted after the recording (measured, exit -11 with the mp4 already whole).
    wait_bricks_resident(view, timeout_s=60.0)
    vf._pump(app, 1.0)
    vf.teardown(app, win, view, names)
    return 0


def _readable_frames(path: Path) -> int:
    """Frames imageio can count in *path*; 0 for a missing or unreadable file."""
    if not path.exists():
        return 0
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        n = int(reader.count_frames())
        reader.close()
        return n
    except Exception:                        # noqa: BLE001 - unreadable = not reusable
        return 0


def compose(a_path: Path, b_path: Path, out_path: Path) -> tuple[int, tuple]:
    """Frame k beside frame k, raw left, decon right; counts must match by construction."""
    import imageio.v2 as imageio
    import numpy as np

    from squidxplorer._camera_script import DIVIDER_PX, _side_by_side
    from squidxplorer._video import write_mp4

    ra, rb = imageio.get_reader(str(a_path)), imageio.get_reader(str(b_path))
    na, nb = ra.count_frames(), rb.count_frames()
    assert na == nb, f"pass frame counts diverged: raw {na}, decon {nb}"
    frames = (_side_by_side(np.asarray(fa)[..., :3], np.asarray(fb)[..., :3], DIVIDER_PX)
              for fa, fb in zip(ra, rb))
    _path, n = write_mp4(frames, out_path, fps=FPS)
    ra.close(); rb.close()

    check = imageio.get_reader(str(out_path))
    n_out = check.count_frames()
    shape = np.asarray(check.get_data(0)).shape
    check.close()
    assert n_out == n, f"composed file reads {n_out} frame(s), wrote {n}"
    return n_out, shape


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pass", dest="pass_args", nargs=2, metavar=("SOURCE", "OUT_MP4"),
                    help="internal: run one recording pass in this process")
    args = ap.parse_args(argv)

    if args.pass_args:
        return record_pass(Path(args.pass_args[0]), Path(args.pass_args[1]), args.seed)

    print(f"seed {args.seed}", flush=True)
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "squidxplorer_comparison"
    tmp.mkdir(exist_ok=True)
    outs = {}
    for name, source in (("raw", RAW_SET), ("decon", DECON_SET)):
        out = tmp / f"{name}_seed{args.seed}.mp4"
        if _readable_frames(out) > 0:
            print(f"reusing existing {out}", flush=True)
            outs[name] = out
            continue
        # The app caps real windows at one instance; Julio approved replacing his.
        subprocess.run(["pkill", "-f", "squidxplorer._viewer import main"], check=False)
        time.sleep(2)
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             "--seed", str(args.seed), "--pass", str(source), str(out)],
                            check=False).returncode
        if rc != 0:
            # A teardown crash AFTER the recording still leaves a whole, readable mp4.
            if _readable_frames(out) > 0:
                print(f"WARNING: the {name} pass exited {rc} at teardown; "
                      f"its recording is whole, continuing", flush=True)
            else:
                print(f"FAIL: the {name} pass exited {rc}", flush=True)
                return rc
        outs[name] = out

    final = DESKTOP / f"raw_vs_decon_561_seed{args.seed}.mp4"
    n, shape = compose(outs["raw"], outs["decon"], final)
    size_mb = final.stat().st_size / 1e6
    print(f"composed {final}: {n} frame(s), {n / FPS:.1f} s at {FPS} fps, "
          f"{shape[1]}x{shape[0]} px, {size_mb:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
