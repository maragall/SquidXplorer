"""IMA-234 evidence: local vs http-served OME-Zarr, for Odon.

    python tools/odon_benchmark.py --dataset "/path/to/acquisition" --clean
    python tools/odon_benchmark.py --hcs-dir /path/to/existing/output    # reuse a plate

Writes a SMALL plate (one well, one FOV by default), serves it over ``http.server``,
and measures both sides of the local/remote seam. ``--clean`` deletes the plate again;
without a plate already on disk there is nothing to serve, so this is the one thing here
that costs bytes, and ``write_plate``'s own ``check_disk_space`` gate stays on.

The honest part is in the output, not this docstring: whatever cannot actually be
executed on this machine is printed under "NOT MEASURED" with the reason, and no number
is produced for it. See ``squidxplorer/_odon_bench.py``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squidxplorer._odon_bench import format_report, run, write_json  # noqa: E402

DATASET = "/Users/julioamaragall/Downloads/synthetic_2x2_wellplate"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET, help="raw acquisition to MIP into a plate")
    ap.add_argument("--hcs-dir", default=None,
                    help="an EXISTING squidxplorer output dir (skips writing a plate)")
    ap.add_argument("--out", default=None, help="where to write the plate (default: a temp dir)")
    ap.add_argument("--regions", default=None, help="comma-separated wells to write")
    ap.add_argument("--n-fovs", type=int, default=1)
    ap.add_argument("--level", default="0", help="pyramid level whose chunks are timed")
    ap.add_argument("--limit", type=int, default=64, help="chunks per transport row")
    ap.add_argument("--workers", type=int, default=8, help="parallel fan-out width")
    ap.add_argument("--repeats", type=int, default=5, help="odon --check runs")
    ap.add_argument("--odon-bin", default=None, help="overrides $ODON_BIN / PATH discovery")
    ap.add_argument("--json", default=None)
    ap.add_argument("--clean", action="store_true", help="delete the written plate afterwards")
    args = ap.parse_args()

    hcs_dir = args.hcs_dir
    written = None
    if hcs_dir is None:
        import tempfile

        from squidxplorer import open_reader, write_plate

        out = args.out or tempfile.mkdtemp(prefix="odon-bench-")
        written = out
        regions = [r.strip() for r in args.regions.split(",")] if args.regions else None
        print(f"writing a plate from {args.dataset} -> {out}", flush=True)
        manifest = write_plate(open_reader(args.dataset), out,
                               n_fovs=args.n_fovs, regions=regions)
        print(f"  {manifest['n_fields_written']} field(s), "
              f"{manifest['levels']} pyramid levels\n", flush=True)
        hcs_dir = out

    try:
        report = run(hcs_dir, level=args.level, limit=args.limit, workers=args.workers,
                     repeats=args.repeats, odon_bin=args.odon_bin)
        print(format_report(report))
        if args.json:
            print(f"\nwrote {write_json(report, args.json)}")
        # Exit 0 even when odon could not be measured: "the measurement did not run, and
        # here is why" is a successful outcome for this harness. A non-zero code would say
        # the harness failed, which is a different and untrue statement.
        return 0
    finally:
        if args.clean and written:
            shutil.rmtree(written, ignore_errors=True)
            print(f"\ncleaned {written}")


if __name__ == "__main__":
    sys.exit(main())
