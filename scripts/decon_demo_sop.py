#!/usr/bin/env python
"""The decon demo SOP: solve, figures, seeded-orbit comparison video, one entry point.

Pure composition of the three existing scripts, in order:
  1. scripts/decon_whole.py       solves the acquisition at the chosen iterations
                                  into a sibling decon46_iN_<name> set (skipped when
                                  that set already exists);
  2. scripts/compare_figures.py   renders the pose stills and the eight side-by-side
                                  figures, floor-95 and peak-matched schemes (its own
                                  reuse skips finished passes);
  3. scripts/record_volume_comparison.py
                                  records the seeded-orbit raw-vs-decon video, the
                                  decon side windowed by the chosen scheme.

Usage: python scripts/decon_demo_sop.py [--iterations 2] [--seed 4242] [--scheme ...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

DEFAULT_ACQ = Path("/Users/julioamaragall/Downloads/25x_C4_dz=3_2026-08-14_16-51-15.744692")
DEFAULT_CHANNEL = "Fluorescence_561_nm_Ex"
FLOOR_WINDOW = (95.0, 1109.0)
# Julio's pick, then the symmetry rule: floor, (95, 1109) on BOTH sides. Asymmetric
# windows bias apparent sharpness (verified by pixel-matching the halves when Julio
# read the sides as swapped); a symmetric floor-down keeps his preference unbiased.
DEFAULT_SCHEME = "floor"


def run_stage(name: str, cmd: list) -> None:
    print(f"== {name}: {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.run([sys.executable, *cmd], check=False).returncode
    if rc != 0:
        raise SystemExit(f"SOP: the {name} stage exited {rc}")


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acquisition", nargs="?", type=Path, default=DEFAULT_ACQ)
    ap.add_argument("--channel", default=DEFAULT_CHANNEL)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--scheme", choices=("shared", "floor"), default=DEFAULT_SCHEME,
                    help="the video's ONE window, both sides: shared (135, 1109) or "
                         "floor (95, 1109). Per-side windows are figures-only: the "
                         "comparison video never windows its sides differently")
    ap.add_argument("--out-dir", type=Path,
                    default=Path.home() / "Desktop" / "decon_demo_figures")
    args = ap.parse_args(argv)

    # The figure and video scripts are pinned to the 25x_C4 set and 561 today; a
    # different input is a named refusal, never a silently wrong render.
    if args.acquisition.resolve() != DEFAULT_ACQ:
        raise SystemExit(f"SOP: the figure and video stages are pinned to "
                         f"{DEFAULT_ACQ.name}; parameterizing them is future work.")
    if args.channel != DEFAULT_CHANNEL:
        raise SystemExit(f"SOP: the shared contrast window is measured for "
                         f"{DEFAULT_CHANNEL} only; other channels are future work.")

    decon_set = args.acquisition.parent / (
        f"decon46_i{args.iterations}_{args.acquisition.name}")
    if decon_set.exists():
        print(f"== solve: reusing existing {decon_set}", flush=True)
    else:
        run_stage("solve", [HERE / "decon_whole.py", args.acquisition,
                            "--iterations", str(args.iterations),
                            "--suffix", f"_i{args.iterations}"])

    run_stage("figures", [HERE / "compare_figures.py", "--out-dir", args.out_dir])

    video_cmd = [HERE / "record_volume_comparison.py", "--seed", str(args.seed),
                 "--decon-source", decon_set]
    if args.scheme == "floor":
        video_cmd += ["--window", str(FLOOR_WINDOW[0]), str(FLOOR_WINDOW[1])]
    run_stage("video", video_cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
