"""Subprocess driver: register one region with ASHLAR, emit positions.json.

ASHLAR is a Python library rather than a position-dumping CLI, so this driver drives
its API and writes the same ``positions.json`` contract the tilefusion driver writes.
That uniformity is the point of IMA-233 D3: the runner spawns and measures one kind of
thing, no matter how different the tools are underneath.

Squid's ``{region}_{fov}_{z}_{channel}.tiff`` layout is not a format ASHLAR reads
natively, so the driver builds an in-memory reader from the tile list and the stage
positions the harness already parsed.

Exit codes: 0 ok, 3 ashlar not importable, 4 registration failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ASHLAR benchmark driver")
    ap.add_argument("--input", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--channel", required=True)
    ap.add_argument("--z", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-shift-um", type=float, default=30.0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import ashlar
        from ashlar import reg
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 3

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from bench.dataset import load_acquisition

        acq = load_acquisition(args.input)
        positions_px = acq.positions_px(args.region)
        fovs = sorted(positions_px)
        if len(fovs) < 2:
            print(f"region {args.region} has {len(fovs)} fov(s); nothing to register", file=sys.stderr)
            return 4

        reader = _SquidReader(acq, args.region, args.channel, args.z, fovs, reg)
        aligner = reg.EdgeAligner(
            reader,
            channel=0,
            max_shift=args.max_shift_um,
            verbose=False,
        )
        aligner.run()

        solved = {int(fovs[i]): (float(p[0]), float(p[1])) for i, p in enumerate(aligner.positions)}
        (out / "positions.json").write_text(
            json.dumps(
                {
                    "tool": "ashlar",
                    "version": getattr(ashlar, "__version__", "unknown"),
                    "region": args.region,
                    "channel": args.channel,
                    "positions_px": {str(k): [v[0], v[1]] for k, v in solved.items()},
                },
                indent=2,
            )
        )
        return 0
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 4


def _SquidReader(acq, region, channel, z, fovs, reg):
    """Build an ASHLAR reader over one region's tiles.

    Constructed lazily inside the driver so importing this module never requires
    ashlar to be installed.
    """

    class _Metadata(reg.PlateMetadata):
        def __init__(self):
            super().__init__()
            self._fovs = fovs
            self.pixel_size = acq.pixel_size_um

        @property
        def _num_images(self):
            return len(self._fovs)

        @property
        def num_channels(self):
            return 1

        @property
        def pixel_dtype(self):
            return np.dtype(acq.dtype)

        def tile_position(self, i):
            return np.array(acq.positions_px(region)[self._fovs[i]], dtype=float)

        def tile_size(self, i):
            return np.array(acq.frame_shape, dtype=float)

    class _Reader(reg.Reader):
        def __init__(self):
            self.path = str(acq.root)
            self.metadata = _Metadata()

        def read(self, series, c):
            return acq.read(region, self.metadata._fovs[series], z, channel)

    return _Reader()


if __name__ == "__main__":
    raise SystemExit(main())
