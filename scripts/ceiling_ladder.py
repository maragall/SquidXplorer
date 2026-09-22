#!/usr/bin/env python
"""Ceiling ladder: raw | decon-i2 hero stills at shared windows (95, C), one composite
per candidate ceiling, so the ceiling is picked from real renders. The floors are
settled at 95; the shared 1109 ceiling clipped the decon side's peaks (its MIP p99.9 is
2659 against raw's 1917), so the ladder climbs raw's peak, decon's peak, and i2's p99.9
from the peak-matched measurement.

Usage: python scripts/ceiling_ladder.py --out-dir DIR   # replaces a running instance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from compare_figures import (DECON_SETS, DIVIDER_PX, RAW_SET, _pass_pngs,  # noqa: E402
                             channel_mip, run_pass_subprocess)

CEILINGS = (1900, 2659, 3219)
FLOOR = 95.0


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_mip = channel_mip(RAW_SET)
    i2_mip = channel_mip(DECON_SETS["i2"])
    for c in CEILINGS:
        print(f"ceiling {c}: raw {np.mean(raw_mip > c) * 100:.3f}% of MIP above, "
              f"i2 {np.mean(i2_mip > c) * 100:.3f}% above", flush=True)

    for c in CEILINGS:
        for prefix, source in ((f"raw_c{c}", RAW_SET), (f"i2_c{c}", DECON_SETS["i2"])):
            if all(p.exists() for p in _pass_pngs(args.out_dir, prefix)):
                print(f"reusing existing {prefix} stills", flush=True)
                continue
            if not run_pass_subprocess(source, prefix, (FLOOR, float(c)), args.out_dir):
                print(f"FAIL: the {prefix} pass left no complete stills", flush=True)
                return 1

    import imageio.v3 as iio

    for c in CEILINGS:
        a = iio.imread(args.out_dir / f"raw_c{c}_hero.png")
        b = iio.imread(args.out_dir / f"i2_c{c}_hero.png")
        h = min(a.shape[0], b.shape[0])
        divider = np.full((h, DIVIDER_PX, 3), 64, np.uint8)
        out = args.out_dir / f"ladder_c{c}.png"
        iio.imwrite(out, np.concatenate([a[:h], divider, b[:h]], axis=1))
        print(f"composed {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
