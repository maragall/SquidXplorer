#!/usr/bin/env python
"""Hammer one acquisition from N threads and count the reads that come back WRONG.

Why this exists
---------------
A serial run proves nothing about this class of bug. ``reader._TiffHandles`` was fixed on
2026-08-06 because a cached ``tifffile.TiffFile`` is a FILE OBJECT and ``pages[p].asarray()``
seeks: two threads decoding two pages of one file move one seek position under each other. The
measurement that found it was **0 errors in 8 serial reads, 10 of 40 threaded** — the serial number
is the whole point, because it is the number a normal test suite produces.

So this is the instrument, kept as a tool rather than a test on purpose: it needs a real
multi-gigabyte acquisition, and its failure counts are inherently probabilistic (an unlocked read
that happens to be scheduled serially returns correct pixels). A test that asserts "0 of 40" would
pass on a broken build whenever the machine was idle. What IS pinned deterministically lives in
``tests/test_reader_threading.py``; this tool is for the numbers you quote in a report.

What it checks
--------------
Correctness, not just absence of exceptions. Every plane is read once serially to build a
per-plane checksum, then read again from ``--threads`` threads; a read is counted WRONG if it
raises OR if its bytes differ from the serial answer. Silent corruption is the failure mode that
matters: a torn read that happens to decode without raising is a plausible-looking image.

    python tools/thread_stress.py ~/Downloads/<acquisition> --threads 40 --reads 400
    python tools/thread_stress.py <acq> --serial      # the control number, always report both
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

from squidmip.reader import open_reader


def _digest(arr) -> str:
    return hashlib.blake2b(memoryview(arr).tobytes(), digest_size=8).hexdigest()


def _planes(reader, meta, limit: int) -> list:
    """``[(region, fov, channel, z, t), ...]`` — a deterministic sample of real planes."""
    out = []
    # ``meta["channels"]`` may be `Channel` models (unhashable pydantic) or plain strings; the key
    # `read()` accepts is the canonical NAME either way, and a plane key has to be hashable here.
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
    """Serial pass: the answer every threaded read must reproduce. Also the control number."""
    truth, failed = {}, []
    for key in planes:
        try:
            truth[key] = _digest(reader.read(*key))
        except Exception as exc:                      # a genuinely unreadable plane, not a race
            failed.append((key, f"{type(exc).__name__}: {exc}"))
    return truth, failed


def _run(reader, truth: dict, keys: list, threads: int, reads: int) -> tuple:
    """Return ``(wrong, raised, corrupt, elapsed)``. *corrupt* is the frightening one."""
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
