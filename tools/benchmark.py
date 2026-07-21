"""IMA-233 evidence: run the per-operator benchmark on a real acquisition, print the table.

    python tools/benchmark.py --dataset "/path/to/acquisition"
    python tools/benchmark.py --dataset ... --operators mip,stitch --regions manual0
    python tools/benchmark.py --dataset ... --csv out.csv --json out.json

Speed, footprint and quality for every operator in the registry, measured with Julio's
own profiling suite (``profiling/`` in the stitcher repo — StageTimer, assign_stages,
RSSSampler, AllocationSampler, compute_ranking, harness._collect). See
``squidmip/_benchmark.py`` for exactly which functions are adapted and why.

Nothing is written unless you ask for --csv/--json, and neither is large. The run itself
is storage- and memory-guarded before it starts: see ``_benchmark.guard_memory`` and
``_benchmark.persist_estimate``, which route through the writer's OWN
``estimate_write_bytes`` / ``check_disk_space`` (overlap-aware for region operators)
rather than a second opinion.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squidmip._benchmark import (  # noqa: E402
    DEFAULT_OPERATORS,
    benchmark_dataset,
    format_allocations,
    format_stages,
    format_table,
    write_csv,
    write_json,
)

DATASET = (
    "/Users/julioamaragall/Downloads/"
    "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--operators", default=",".join(DEFAULT_OPERATORS),
                    help=f"comma-separated (default: {','.join(DEFAULT_OPERATORS)})")
    ap.add_argument("--regions", default=None, help="comma-separated well subset")
    ap.add_argument("--n-fovs", type=int, default=1,
                    help="FOVs per well for FOV operators (default 1)")
    ap.add_argument("--region-n-fovs", type=int, default=None,
                    help="FOVs per well for REGION operators (default: all — an arbitrary "
                         "prefix of a serpentine scan is not a connected tile graph, and the "
                         "solve falls back to affine placement)")
    ap.add_argument("--region-channels", default="1",
                    help="channel indices for region operators (default 1; a 27-FOV 4-channel "
                         "mosaic is ~0.9 GB resident and measures memory, not stitching)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    operators = [o.strip() for o in args.operators.split(",") if o.strip()]
    regions = [r.strip() for r in args.regions.split(",")] if args.regions else None
    channels = [int(c) for c in args.region_channels.split(",") if c.strip() != ""] or None

    print(f"dataset   : {args.dataset}")
    print(f"operators : {', '.join(operators)}")
    print(f"scope     : n_fovs={args.n_fovs} (fov ops) / {args.region_n_fovs} (region ops), "
          f"regions={regions or 'all'}, workers={args.workers}")
    print("", flush=True)

    def _on_error(op, exc):
        print(f"  ! {op}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    results = benchmark_dataset(
        args.dataset, operators,
        regions=regions, n_fovs=args.n_fovs,
        region_n_fovs=args.region_n_fovs, region_channels=channels,
        workers=args.workers, quality=not args.no_quality,
        on_error=_on_error,
    )
    elapsed = time.perf_counter() - t0

    print(format_table(results))
    print("\nper-stage (profiling.stages.StageTimer + assign_stages):")
    print(format_stages(results) or "  (none)")
    print("\nallocation ranking (profiling.ranking.compute_ranking):")
    print(format_allocations(results) or "  (none)")
    print(f"\nsuite wall time: {elapsed:.1f}s")

    if args.csv:
        print(f"wrote {write_csv(results, args.csv)}")
    if args.json:
        print(f"wrote {write_json(results, args.json, meta={'dataset': args.dataset})}")
    return 0 if all(r.error is None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
