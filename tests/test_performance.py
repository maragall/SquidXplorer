"""IMA-183 performance baselines — single-thread MIP speed + memory footprint.

These are the numbers IMA-188's parallel/streaming engine must BEAT. Marked ``integration``
(need `sim_1536wp` on disk); deselected in clean-room CI.

Run:  pytest tests/test_performance.py -m integration -s   (‑s to see the printed baseline)

IMA-188 can import ``benchmark_single_well`` to measure its per-worker cost against this
single-threaded baseline (same metric, apples-to-apples).
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import require_usable_dataset

from squidmip import open_reader, project_well

SIM_1536WP = Path("/Users/julioamaragall/CEPHLA/Data/sim_1536wp")


@pytest.fixture
def sim_1536wp():
    return require_usable_dataset(SIM_1536WP, "sim_1536wp")


def benchmark_single_well(reader, region, fov) -> dict:
    """Measure one well's projection: speed (full / read / compute) and memory footprint.

    Returns a dict of:
      full_ms      - project_well wall time (read + MIP)
      read_ms      - tifffile decode + I/O for all planes
      compute_ms   - np.maximum reduction with all planes already in RAM
      read_MB      - bytes read for the well
      peak_bytes   - tracemalloc peak during project_well (numpy domain is tracked)
      result_bytes - size of the (T,C,1,Y,X) output (legitimate, not overhead)
      plane_bytes  - one Y*X plane
      full_stack_bytes - what a naive np.stack of all planes would cost (the anti-pattern)

    Reused by IMA-188 to compare its per-worker cost to the IMA-183 single-thread baseline.
    """
    m = reader.metadata
    chans = [c["name"] for c in m["channels"]]
    z_levels = m["z_levels"]
    y, x = m["frame_shape"]
    itemsize = np.dtype(m["dtype"]).itemsize
    plane_bytes = y * x * itemsize

    # full project_well + peak memory (streaming must stay far below a full stack)
    tracemalloc.start()
    t0 = time.perf_counter()
    out = project_well(reader, region, fov)
    full = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # read-only (decode + I/O)
    t0 = time.perf_counter()
    for c in chans:
        for z in z_levels:
            reader.read(region, fov, c, z)
    read = time.perf_counter() - t0

    # compute-only: planes preloaded, time just the np.maximum fold
    planes = {c: [reader.read(region, fov, c, z) for z in z_levels] for c in chans}
    t0 = time.perf_counter()
    for c in chans:
        acc = planes[c][0].copy()
        for p in planes[c][1:]:
            np.maximum(acc, p, out=acc)
    compute = time.perf_counter() - t0

    return {
        "full_ms": full * 1000,
        "read_ms": read * 1000,
        "compute_ms": compute * 1000,
        "read_MB": len(chans) * len(z_levels) * plane_bytes / 1e6,
        "peak_bytes": peak,
        "result_bytes": out.nbytes,
        "plane_bytes": plane_bytes,
        "full_stack_bytes": len(z_levels) * len(chans) * plane_bytes,
    }


@pytest.mark.filterwarnings("ignore:Recorded Nz")
@pytest.mark.integration
def test_single_well_speed_baseline(sim_1536wp, capsys):
    reader = open_reader(sim_1536wp)
    regions = reader.metadata["regions"]
    project_well(reader, regions[100], 0)              # warm page cache / steady state
    b = benchmark_single_well(reader, regions[0], 0)
    n = len(regions)
    with capsys.disabled():
        print(
            f"\n[IMA-183 baseline] per well: full={b['full_ms']:.0f}ms "
            f"(read={b['read_ms']:.0f}ms, compute={b['compute_ms']:.0f}ms) | "
            f"{b['read_MB']:.0f}MB read | peak={b['peak_bytes']/1e6:.0f}MB"
        )
        print(
            f"[IMA-183 baseline] {n} wells single-thread ~= {b['full_ms']*n/1000:.0f}s "
            f"({b['full_ms']*n/60000:.1f} min, cache-warm) -> IMA-188 parallelizes this"
        )
    # Loose regression ceiling only (timing is machine/cache dependent; the value is the
    # printed baseline, not a tight bound). Fails only on catastrophic slowdown.
    assert b["full_ms"] < 30_000
    # the algorithm is memory-bandwidth bound: compute must be a meaningful, non-zero share
    assert b["compute_ms"] > 0


@pytest.mark.filterwarnings("ignore:Recorded Nz")
@pytest.mark.integration
def test_single_well_memory_footprint(sim_1536wp):
    # streaming MIP must NOT materialise the whole z-stack: peak ≈ the (T,C,1,Y,X) result
    # plus a couple of in-flight planes, far below stacking all Nz*C planes.
    reader = open_reader(sim_1536wp)
    b = benchmark_single_well(reader, reader.metadata["regions"][0], 0)
    assert b["peak_bytes"] < b["result_bytes"] + 6 * b["plane_bytes"]
    assert b["peak_bytes"] < b["full_stack_bytes"]      # never the naive all-planes footprint
