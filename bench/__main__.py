"""CLI: ``python -m bench <acquisition> [options]``.

    python -m bench ~/CEPHLA/Data/20x_scan_2025-09-05_17-57-50 \\
        --region C5 --channel Fluorescence_405_nm_Ex \\
        --out-csv benchmark.csv --report benchmark.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.adapters import build_adapters
from bench.dataset import AcquisitionError, load_acquisition
from bench.report import markdown_table
from bench.runner import DEFAULT_TIMEOUT_S, RunConfig, run_benchmark
from bench.sampler import DEFAULT_INTERVAL_S, DEFAULT_MIN_FREE_BYTES


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m bench",
        description="IMA-233 stitcher benchmark: speed, footprint and seam quality.",
    )
    ap.add_argument("acquisition", help="Squid acquisition directory")
    ap.add_argument("--region", help="region/well to benchmark (default: first region)")
    ap.add_argument("--channel", help="channel to register on (default: first channel)")
    ap.add_argument("--z", type=int, default=0)
    ap.add_argument("--tools", nargs="*", default=None, help="adapters to run (default: all)")
    ap.add_argument("--out-dir", default="./bench-out", help="scratch output root")
    ap.add_argument("--out-csv", default="benchmark.csv")
    ap.add_argument("--report", default=None, help="write a markdown table here")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--compression", default="none")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--warm-runs", type=int, default=1, help="0 = report the cold time only")
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_BYTES / 1024**3,
        help="abort a tool when free space would fall below this",
    )
    ap.add_argument("--sampler-interval", type=float, default=DEFAULT_INTERVAL_S)
    ap.add_argument("--n-blocks", type=int, default=8, help="blocks per seam")
    ap.add_argument("--keep-output", action="store_true", help="do not reclaim tool output")
    ap.add_argument("--list-tools", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_tools:
        for a in build_adapters(None):
            print(f"{a.name:14s} available={a.is_available()}")
        return 0

    try:
        acq = load_acquisition(args.acquisition)
    except AcquisitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    region = args.region or (acq.regions[0] if acq.regions else "")
    channel = args.channel or (acq.channels[0] if acq.channels else "")
    if region not in acq.regions:
        print(f"error: region {region!r} not in {acq.regions}", file=sys.stderr)
        return 2
    if channel not in acq.channels:
        print(f"error: channel {channel!r} not in {acq.channels}", file=sys.stderr)
        return 2

    cfg = RunConfig(
        region=region,
        channel=channel,
        z=args.z,
        threads=args.threads,
        compression=args.compression,
        timeout_s=args.timeout,
        min_free_bytes=int(args.min_free_gb * 1024**3),
        sampler_interval_s=args.sampler_interval,
        warm_runs=args.warm_runs,
        keep_output=args.keep_output,
        n_blocks=args.n_blocks,
    )

    print(
        f"{acq.root.name}: region {region}, {len(acq.fovs(region))} FOVs, "
        f"{acq.frame_shape[0]}x{acq.frame_shape[1]} {acq.dtype}, "
        f"{acq.pixel_size_um:.4f} um/px, channel {channel}"
    )

    try:
        adapters = build_adapters(args.tools)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = run_benchmark(adapters, acq, cfg, args.out_dir, csv_path=args.out_csv)

    table = markdown_table(rows)
    print("\n" + table)
    if args.report:
        Path(args.report).write_text(f"# Stitcher benchmark — {acq.root.name}\n\n{table}")
        print(f"report: {args.report}")
    print(f"csv: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
