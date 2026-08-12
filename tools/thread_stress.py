#!/usr/bin/env python
"""Hammer one acquisition from N threads and count the reads that come back wrong.

    python tools/thread_stress.py <acquisition> --threads 40 --reads 400
    python tools/thread_stress.py <acquisition> --serial
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from squidxplorer.reader import open_reader


def _digest(arr) -> str:
    return hashlib.blake2b(memoryview(arr).tobytes(), digest_size=8).hexdigest()


def _planes(reader, meta, limit: int) -> list:
    """``[(region, fov, channel, z, t), ...]`` — a deterministic sample of real planes."""
    out = []
    # meta["channels"] may be Channel models or strings; the canonical NAME is the hashable key.
    channels = [getattr(c, "name", c) for c in (meta.get("channels") or [])]
    n_z = int(meta.get("n_z") or 1)
    n_t = int(meta.get("n_t") or 1)
    for region, fovs in sorted((meta.get("fovs_per_region") or {}).items()):
        for fov, channel, z, t in itertools.product(
                sorted(fovs), channels, range(n_z), range(n_t)):
            out.append((region, int(fov), channel, int(z), int(t)))
            if len(out) >= limit:
                return out
    return out


def _truth(reader, planes: list) -> dict:
    """Serial pass: the answer every threaded read must reproduce."""
    truth, failed = {}, []
    for key in planes:
        try:
            truth[key] = _digest(reader.read(*key))
        except Exception as exc:                      # genuinely unreadable, not a race
            failed.append((key, f"{type(exc).__name__}: {exc}"))
    return truth, failed


def _run(reader, truth: dict, keys: list, threads: int, reads: int) -> tuple:
    """Return ``(raised, corrupt, elapsed)``."""
    raised: list = []
    corrupt: list = []
    lock = threading.Lock()
    plan = [random.choice(keys) for _ in range(reads)]

    def one(key):
        try:
            got = _digest(reader.read(*key))
        except Exception as exc:
            with lock:
                raised.append((key, f"{type(exc).__name__}: {exc}"))
            return
        if got != truth[key]:
            with lock:
                corrupt.append((key, got, truth[key]))

    t0 = time.perf_counter()
    if threads <= 1:
        for key in plan:
            one(key)
    else:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(one, plan))
    return raised, corrupt, time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="an acquisition directory open_reader() understands")
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--reads", type=int, default=400, help="total reads across all threads")
    ap.add_argument("--planes", type=int, default=64, help="distinct planes to sample")
    ap.add_argument("--serial", action="store_true", help="also run the 1-thread control")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    random.seed(args.seed)

    reader = open_reader(args.path)
    meta = reader.metadata
    print(f"reader: {type(reader).__name__}  regions={len(meta.get('fovs_per_region') or {})} "
          f"channels={len(meta.get('channels') or [])} n_z={meta.get('n_z')} n_t={meta.get('n_t')}")

    keys = _planes(reader, meta, args.planes)
    if not keys:
        print("no planes to read", file=sys.stderr)
        return 2
    truth, unreadable = _truth(reader, keys)
    keys = [k for k in keys if k in truth]
    print(f"serial baseline: {len(truth)} planes checksummed, {len(unreadable)} unreadable "
          f"(genuinely, not a race)")
    for key, why in unreadable[:5]:
        print(f"    unreadable {key}: {why}")
    if not keys:
        return 2

    for n in ([1] if args.serial else []) + [args.threads]:
        raised, corrupt, dt = _run(reader, truth, keys, n, args.reads)
        bad = len(raised) + len(corrupt)
        print(f"\n{n:>3} thread(s): {bad} of {args.reads} reads WRONG "
              f"({len(raised)} raised, {len(corrupt)} returned wrong pixels) in {dt:.1f}s")
        for key, why in raised[:5]:
            print(f"    raised {key}: {why}")
        for key, got, want in corrupt[:5]:
            print(f"    CORRUPT {key}: {got} != {want}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
