"""IMA-233 — stitcher benchmark harness.

Runs each stitcher as an isolated subprocess against one Squid acquisition, measures
it from the outside, and emits one comparison row per (tool, dataset, run).

Deliberately OUTSIDE the ``squidmip`` package: ``pyproject.toml`` packages only
``["squidmip"]``, so nothing here ships in Nick's wheel, and nothing here may be
imported by the runtime MIP pipeline.

Why measurement happens from the outside (IMA-233 D3)
-----------------------------------------------------
``tracemalloc`` (used by ``tests/test_performance.py``) sees only allocations made by
*this* interpreter. Three of the four target stitchers are not this interpreter:

    ASHLAR       Python      in-proc measurable, but run as a subprocess anyway
    MCmicro      Nextflow/Docker
    BigStitcher  Fiji/Java
    PetaKit5D    MATLAB

Pointing tracemalloc at a JVM subprocess reports ~0 and makes the heaviest tool look
like the leanest. So every tool -- including the Python ones -- runs as a subprocess
and is measured uniformly: wall clock, peak RSS of the process tree, and bytes on disk.

Why quality is measured PRE-fusion (IMA-233 R1)
-----------------------------------------------
Seam misalignment needs TWO INDEPENDENT views of the same physical region::

    tile_i --+                                  fused mosaic
             +-- overlap -> 2 strips              +-------------+
    tile_j --+   (correlatable)                   | blended px  | -> 1 strip
                                                  +-------------+   (nothing to
                                                                     correlate)

A fused mosaic has already blended the overlap into one pixel set. You cannot phase-
correlate an image against itself. So the metric re-reads the INPUT tiles and places
them at the positions the stitcher solved for. That makes parsing each tool's position
output the real work -- and it leaves the metric *undefined* (not merely unimplemented)
for BigStitcher's non-rigid model, which has no scalar shift to report.
"""

from __future__ import annotations

__all__ = [
    "Acquisition",
    "Tile",
    "load_acquisition",
    "adjacent_pairs",
    "seam_residual",
    "BenchmarkRow",
    "STATUS_OK",
]

from bench.dataset import Acquisition, Tile, load_acquisition
from bench.metrics import adjacent_pairs, seam_residual
from bench.report import STATUS_OK, BenchmarkRow
