"""Subprocess driver: stitch one region with tilefusion, emit positions.json.

Runs as its own process so the harness can measure it from outside like every other
tool, and so tilefusion's heavy ``__init__`` (numba, GPU probe, basicpy) never enters
the harness process.

Contract with the adapter -- writes into ``--out``:
    positions.json   {"positions_px": {"<fov>": [row_px, col_px]}, "tool": ..., "version": ...}
    plus whatever the tool itself writes

Exit codes: 0 ok, 3 tilefusion not importable, 4 stitch failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="tilefusion benchmark driver")
    ap.add_argument("--input", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--channel", required=True)
    ap.add_argument("--z", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from tilefusion import TileFusion  # noqa: F401
        import tilefusion
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 3

    try:
        tf = TileFusion(
            input_folder=args.input,
            output_folder=str(out),
            registration_channel=args.channel,
        )
        # Registration only: we need the SOLVED POSITIONS, not a fused mosaic. Fusing
        # would burn time and disk producing an artifact the metric cannot use anyway
        # (a fused mosaic has no independent overlap views left to correlate).
        if hasattr(tf, "register"):
            tf.register()
        elif hasattr(tf, "run"):
            tf.run()

        positions = _extract_positions(tf)
        if not positions:
            print("tilefusion exposed no per-tile positions", file=sys.stderr)
            return 4

        (out / "positions.json").write_text(
            json.dumps(
                {
                    "tool": "tilefusion",
                    "version": getattr(tilefusion, "__version__", "unknown"),
                    "region": args.region,
                    "channel": args.channel,
                    "positions_px": {str(k): [float(v[0]), float(v[1])] for k, v in positions.items()},
                },
                indent=2,
            )
        )
        return 0
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 4


def _extract_positions(tf) -> dict[int, tuple[float, float]]:
    """Pull per-tile pixel positions off a TileFusion instance.

    tilefusion has moved this attribute around between versions, so try the known
    names rather than pinning one and breaking on upgrade.
    """
    for attr in ("positions", "tile_positions", "final_positions", "_positions"):
        raw = getattr(tf, attr, None)
        if raw is None:
            continue
        try:
            if isinstance(raw, dict):
                return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}
            return {int(i): (float(p[0]), float(p[1])) for i, p in enumerate(raw)}
        except (TypeError, ValueError, IndexError):
            continue
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
