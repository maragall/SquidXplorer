"""Per-operator benchmark harness: speed, footprint, quality — on real data.

An **adapter**, not a profiler: every measurement primitive (StageTimer, RSSSampler,
AllocationSampler, compute_ranking, the CSV writers) is imported from Julio's profiling
suite (``profiling/`` in the stitcher repo) rather than reimplemented. What is new here is
the per-operator driver over squidxplorer's registry, the quality metrics, and the storage
guard.

Storage/memory guards reuse ``squidxplorer._output.estimate_write_bytes`` /
``check_disk_space``: a run reports what persisting its output would cost, and refuses to
start if it does not fit with headroom.

``read_ms`` is measured by accumulation on ``reader.read``, not a StageTimer span, because
``project_well`` hands the reducer a generator: the reads happen lazily inside the operator
call, and a span there would nest and double-count. Only region operators get the four-phase
project/register/optimize/fuse breakdown; FOV operators report open/stream.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from squidxplorer._engine import available_plane_operators, operator_consumes
from squidxplorer._output import check_disk_space, estimate_write_bytes, free_bytes
from squidxplorer._engine import available_region_operators, is_region_operator

_MB = 1024.0 ** 2
_GB = 1024.0 ** 3

# Fraction of free RAM a run may occupy before it is refused rather than started.
_RAM_BUDGET = 0.5


class BenchmarkGuardError(RuntimeError):
    """A run was refused before it started because it would not fit in RAM or on disk."""


def _profiling():
    """Import Julio's profiling package (dev tool, not a squidxplorer runtime dependency)."""
    try:
        from profiling import attribution, harness, ranking, record, sampler, stages
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "the benchmark harness ADAPTS Julio's profiling suite and cannot run without "
            "it. It lives in the stitcher repo (profiling/: stages.py, sampler.py, "
            "attribution.py, ranking.py, harness.py, record.py) — put that repo on "
            f"PYTHONPATH. Original error: {exc}"
        ) from exc
    return stages, sampler, attribution, ranking, harness, record


# --------------------------------------------------------------------------------------
# Quality metrics
# --------------------------------------------------------------------------------------

def relative_gradient_energy(plane: np.ndarray) -> float:
    """Mean absolute gradient / mean intensity — "structure per photon", comparable across
    operators whose mean shifts (e.g. bgsub). Absolute diff, not squared Sobel: squaring is
    dominated by the brightest speck rather than tissue structure."""
    a = np.asarray(plane, dtype=np.float32)
    if a.size == 0:
        return float("nan")
    mean = float(a.mean())
    if mean <= 1e-9:
        return float("nan")
    gy = np.abs(np.diff(a, axis=0)).mean() if a.shape[0] > 1 else 0.0
    gx = np.abs(np.diff(a, axis=1)).mean() if a.shape[1] > 1 else 0.0
    return float((gy + gx) / 2.0 / mean)


def _abs_gradient(plane: np.ndarray) -> float:
    """Mean absolute gradient, NOT divided by the mean — companion to
    :func:`relative_gradient_energy` (``sharp_abs`` in :func:`_quality`)."""
    a = np.asarray(plane, dtype=np.float32)
    if a.size == 0:
        return float("nan")
    gy = np.abs(np.diff(a, axis=0)).mean() if a.shape[0] > 1 else 0.0
    gx = np.abs(np.diff(a, axis=1)).mean() if a.shape[1] > 1 else 0.0
    return float((gy + gx) / 2.0)


def block_uniformity(plane: np.ndarray, blocks: int = 8) -> float:
    """Illumination flatness in [0, 1]: ``1 - std/mean`` of a ``blocks x blocks`` mean field.
    1.0 = perfectly flat."""
    a = np.asarray(plane, dtype=np.float32)
    if a.ndim != 2 or min(a.shape) < blocks:
        return float("nan")
    ys = np.array_split(a, blocks, axis=0)
    grid = np.array([[b.mean() for b in np.array_split(row, blocks, axis=1)] for row in ys])
    mean = float(grid.mean())
    if mean <= 1e-9:
        return float("nan")
    return float(max(0.0, 1.0 - grid.std() / mean))


def overlap_ncc(tile_i: np.ndarray, tile_j: np.ndarray, dy: float, dx: float) -> float:
    """How well two FOVs agree in their overlap when placed *dy, dx* apart. NCC in [-1, 1].

    THE seam metric: gradient energy in the seam band is misleading here because sub-pixel
    placement bilinearly resamples the tiles, smoothing noise and lowering gradient energy
    even as the seam gets better (measured: 137.3 -> 116.5 while NCC rose 0.666 -> 0.759).

    Bounds come from ``tilefusion.registration.compute_pair_bounds``, imported lazily since
    ``tilefusion``'s package init is heavy and squidxplorer's reader path must not pay for it.
    """
    from tilefusion.registration import compute_pair_bounds

    Y, X = tile_i.shape
    idy, idx = int(round(dy)), int(round(dx))
    bounds = compute_pair_bounds([(0, 1, idy, idx, Y - abs(idy), X - abs(idx))], (Y, X))
    if not bounds:
        return float("nan")
    _i, _j, biy, bix, bjy, bjx = bounds[0]
    a = tile_i[biy[0]:biy[1], bix[0]:bix[1]].astype(np.float64).ravel()
    b = tile_j[bjy[0]:bjy[1], bjx[0]:bjx[1]].astype(np.float64).ravel()
    n = min(a.size, b.size)
    if n < 16 or a[:n].std() < 1e-6 or b[:n].std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


# What each operator's quality columns mean and which direction is good.
QUALITY_NOTES = {
    "mip": "read sharp_abs vs the middle z-plane — and see the caveat: it is NOT "
           "expected to exceed 1 on a laser-AF stack",
    "reference": "read sharp_abs vs the middle z-plane; ~1.00 means the focus solve "
                 "picked (near) the middle plane, which on an AF'd stack is correct",
    "bgsub": "read flat_gain — but see the note below; clipped is the cost, pixels driven "
             "to 0, the only place this transform loses information",
    "decon": "read sharp_abs > 1; that is the entire claim of a deconvolution. It does "
             "not move the mean, so sharp_gain agrees",
    "flatfield": "read flat_gain > 1 (vignetting removed) with sharp_abs ~ 1: a gain "
                 "field must correct illumination, not invent structure",
    "stitch": "seam_ncc against the 'coordinate' row — SAME pair, chosen from the stage "
              "coordinates before either operator runs. The delta is registration's value",
    "coordinate": "the unregistered control for 'stitch'; its seam_ncc is the baseline",
}

# Where a number does not mean what its name suggests.
QUALITY_CAVEATS = {
    "mip": "sharp_abs < 1 is EXPECTED here, not a defect. The 10x acquisition is "
           "laser-autofocused, so its middle z-plane is already in focus; max-projection "
           "then adds out-of-focus haze and gradient energy falls (measured 0.68). MIP's "
           "value is coverage, and on an AF'd stack sharp_abs is the price of that.",
    "bgsub": "flat_gain < 1 here is expected, not a regression: block_uniformity is "
             "1 - std/mean of a coarse mean field, and subtracting the background "
             "collapses the mean, making the same residual variation a larger fraction "
             "of it. The operator's success shows up as sharp_gain and a small clipped "
             "fraction.",
}


# --------------------------------------------------------------------------------------
# Result record
# --------------------------------------------------------------------------------------

@dataclass
class OperatorResult:
    """One operator, on one acquisition: speed, footprint, quality, and what it would cost."""

    operator: str
    kind: str                       # "fov" (one array per FOV) | "region" (one mosaic per well)
    dataset: str
    regions: tuple = ()
    n_fovs: Optional[int] = None
    channels: Optional[tuple] = None

    wall_ms: float = 0.0
    stage_ms: dict = field(default_factory=dict)
    read_ms: float = 0.0
    read_calls: int = 0
    read_mb: float = 0.0

    baseline_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    stage_peak_rss_mb: dict = field(default_factory=dict)
    integrated_mb_s: float = 0.0
    top_functions: tuple = ()

    wells: int = 0
    out_shape: tuple = ()
    out_megapixels: float = 0.0
    persist_bytes: int = 0
    persist_fits: bool = True

    quality: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def mpix_per_s(self) -> float:
        return self.out_megapixels / (self.wall_ms / 1000.0) if self.wall_ms else float("nan")

    @property
    def compute_ms(self) -> float:
        """Wall time not spent inside ``reader.read``; clamped at 0."""
        return max(0.0, self.wall_ms - self.read_ms)

    def as_row(self) -> dict:
        return {
            "operator": self.operator,
            "kind": self.kind,
            "wells": self.wells,
            "out_Mpix": round(self.out_megapixels, 1),
            "wall_ms": round(self.wall_ms, 1),
            "read_ms": round(self.read_ms, 1),
            "compute_ms": round(self.compute_ms, 1),
            "Mpix_s": round(self.mpix_per_s, 1),
            "peak_RSS_MB": round(self.peak_rss_mb, 1),
            "dRSS_MB": round(self.peak_rss_mb - self.baseline_rss_mb, 1),
            "MB_s": round(self.integrated_mb_s, 1),
            "persist_GB": round(self.persist_bytes / _GB, 3),
            "quality": "  ".join(f"{k}={_fmt(v)}" for k, v in self.quality.items()),
            "error": self.error or "",
        }


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.3f}"
    return str(v)


# --------------------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------------------

class _ReadRecorder:
    """Accumulate time and bytes spent inside ``reader.read`` (not a StageTimer span — the
    reads happen lazily inside the reducer, see module docstring). Locked because
    the per-FOV loop fans out over a thread pool."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ms = 0.0
        self.calls = 0
        self.nbytes = 0

    @contextlib.contextmanager
    def wrap(self, reader):
        original = reader.read

        def read(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                out = original(*args, **kwargs)
            finally:
                dt = (time.perf_counter() - t0) * 1000.0
            with self.lock:
                self.ms += dt
                self.calls += 1
                self.nbytes += getattr(out, "nbytes", 0)
            return out

        reader.read = read       # instance attribute, shadowing the bound method
        try:
            yield self
        finally:
            try:
                del reader.read
            except AttributeError:      # pragma: no cover - reader without __dict__
                reader.read = original


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------

def expected_output_bytes(meta: dict, *, kind: str, regions: Sequence[str],
                          n_fovs: Optional[int], consumes_z: bool,
                          channels: Optional[Sequence] = None) -> int:
    """Bytes the harness expects to hold RESIDENT for one run's outputs.

    This is the RAM question, distinct from the disk question: a plane-op keeps z at full
    depth (bgsub on Nz=10 is ten times a MIP), while the disk estimate is written
    post-projection.
    """
    frame = meta.get("frame_shape")
    if not frame:
        return 0
    ny, nx = int(frame[0]), int(frame[1])
    itemsize = np.dtype(meta.get("dtype", "uint16")).itemsize
    n_c = len(channels) if channels else len(meta.get("channels") or [1])
    n_t = int(meta.get("n_t", 1) or 1)
    n_z = 1 if consumes_z else max(1, len(meta.get("z_levels") or [1]))
    fovs_per_region = meta.get("fovs_per_region") or {}

    if kind == "region":
        from squidxplorer._output import _region_mosaic_pixels
        px = _region_mosaic_pixels(meta, list(regions), (ny, nx))
        # A region run holds ONE well's mosaic at a time (workers=1 by contract).
        px = px // max(1, len(regions))
    else:
        n_fields = sum(min(int(n_fovs), len(fovs_per_region.get(r, ())))
                       if n_fovs is not None else len(fovs_per_region.get(r, ()))
                       for r in regions)
        px = n_fields * ny * nx
    return int(px * n_c * n_t * n_z * itemsize)


def guard_memory(expected_bytes: int, *, what: str, budget: float = _RAM_BUDGET) -> dict:
    """Refuse a run whose expected resident output exceeds *budget* of free RAM."""
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except Exception:                          # pragma: no cover - psutil always present with profiling
        return {"available_gb": None, "checked": False}
    if expected_bytes > available * budget:
        raise BenchmarkGuardError(
            f"refusing to run {what}: its outputs alone are ~{expected_bytes / _GB:.2f} GB "
            f"but only {available / _GB:.2f} GB of RAM is free (budget {budget:.0%}). "
            "Lower --n-fovs, restrict --regions, or subset --channels."
        )
    return {"available_gb": round(available / _GB, 2), "checked": True}


def persist_estimate(meta: dict, *, kind: str, regions: Sequence[str],
                     n_fovs: Optional[int]) -> int:
    """What writing this operator's output as an OME-Zarr plate would cost, in bytes —
    the same estimator the writer itself gates on."""
    return estimate_write_bytes(meta, n_fovs=n_fovs, regions=list(regions),
                                region_operator=(kind == "region"))


# --------------------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------------------

def _fov_runner(reader, operator: str, regions, n_fovs, workers, timer):
    """Build the callable that streams one FOV-operator run, and its result sink."""
    from squidxplorer._engine import _project_plate

    sink: list = []

    def run() -> None:
        with timer.stage("open"):
            _ = reader.metadata          # warm the lazy index single-threaded
        with timer.stage("stream"):
            for region, fov, image in _project_plate(
                reader, operator=operator, n_fovs=n_fovs, regions=list(regions),
                workers=workers,
            ):
                sink.append((region, fov, image, None))

    return run, sink


def _region_runner(reader, operator: str, regions, n_fovs, channels, timer):
    """Build the callable for a REGION operator run: ``stitch_region`` accepts Julio's
    ``StageTimer`` directly, which is why this path gets a real four-phase breakdown."""
    from squidxplorer._stitch import _stitch_plate

    sink: list = []
    geometry: dict = {}
    kwargs = {"geometry": geometry}
    if channels is not None:
        kwargs["channels"] = list(channels)
        kwargs["registration_channel"] = list(channels)[0]

    def run() -> None:
        _ = reader.metadata
        for region, fov, image in _stitch_plate(
            reader, operator=operator, n_fovs=n_fovs, regions=list(regions),
            workers=1, timer=timer, **kwargs,
        ):
            # `geometry` is reused across wells; snapshot it at the yield (workers=1, so
            # that is the moment it describes THIS well).
            sink.append((region, fov, image, dict(geometry)))

    return run, sink


def _aggregate_stage_ms(spans) -> dict:
    """Total ms per stage name, in first-seen order. Spans repeat (one per call)."""
    out: dict = {}
    for name, start, end in spans:
        out[name] = out.get(name, 0.0) + (end - start)
    return out


def _stage_peaks(samples, spans, assign_stages) -> dict:
    """Peak RSS within each stage — Julio's ``assign_stages``, used for what it is for."""
    peaks: dict = {}
    for _t_ms, rss_mb, stage in assign_stages(samples, spans):
        peaks[stage] = max(peaks.get(stage, 0.0), rss_mb)
    return peaks


def benchmark_operator(
    reader,
    operator: str,
    *,
    regions: Optional[Sequence[str]] = None,
    n_fovs: Optional[int] = 1,
    workers: int = 1,
    channels: Optional[Sequence[int]] = None,
    quality: bool = True,
    rss_interval: float = 0.05,
    alloc_interval: float = 0.25,
) -> OperatorResult:
    """Measure ONE operator on a real acquisition: speed, footprint, quality, disk cost.

    Runs at ``workers=1`` by default: peak RSS scales with workers by construction, so a
    many-worker number belongs in its own row, not mixed into an operator comparison.
    """
    stages_mod, sampler_mod, attribution_mod, ranking_mod, harness_mod, _record = _profiling()

    meta = reader.metadata
    all_regions = list(meta.get("fovs_per_region") or {})
    regions = list(regions) if regions else all_regions
    kind = "region" if is_region_operator(operator) else "fov"
    if kind == "fov" and operator not in available_plane_operators():
        raise KeyError(f"unknown operator {operator!r}; known: "
                       f"{available_plane_operators() + available_region_operators()}")
    consumes_z = (kind == "region") or ("z" in operator_consumes(operator))

    result = OperatorResult(
        operator=operator, kind=kind, dataset=str(getattr(reader, "_path", "")),
        regions=tuple(regions), n_fovs=n_fovs,
        channels=tuple(channels) if channels else None,
    )

    # guards, before anything is allocated
    result.persist_bytes = persist_estimate(meta, kind=kind, regions=regions, n_fovs=n_fovs)
    result.persist_fits = result.persist_bytes < max(0, free_bytes(".")) * (1 - 0.10)
    guard_memory(
        expected_output_bytes(meta, kind=kind, regions=regions, n_fovs=n_fovs,
                              consumes_z=consumes_z, channels=channels),
        what=f"operator {operator!r} on {len(regions)} region(s)",
    )

    _prepare(operator, reader, meta, regions, n_fovs)

    t0 = time.perf_counter()
    timer = stages_mod.StageTimer(t0)
    recorder = _ReadRecorder()
    if kind == "region":
        run, sink = _region_runner(reader, operator, regions, n_fovs, channels, timer)
    else:
        run, sink = _fov_runner(reader, operator, regions, n_fovs, workers, timer)

    import psutil
    result.baseline_rss_mb = psutil.Process().memory_info().rss / 1e6

    wall0 = time.perf_counter()
    with recorder.wrap(reader):
        # Julio's harness._collect: RSSSampler + AllocationSampler, guaranteed teardown.
        profile = harness_mod._collect(run, t0, timer, rss_interval, alloc_interval)
    result.wall_ms = (time.perf_counter() - wall0) * 1000.0
    result.error = profile.error

    result.read_ms, result.read_calls = recorder.ms, recorder.calls
    result.read_mb = recorder.nbytes / _MB
    result.stage_ms = _aggregate_stage_ms(profile.stage_spans)
    result.stage_peak_rss_mb = _stage_peaks(profile.samples, profile.stage_spans,
                                            stages_mod.assign_stages)
    result.peak_rss_mb = max((s.rss_mb for s in profile.samples),
                             default=result.baseline_rss_mb)

    ranked = ranking_mod.compute_ranking(profile.alloc_records)
    result.integrated_mb_s = sum(r["integrated_mb_s"] for r in ranked)
    result.top_functions = tuple(
        (r["function"], round(r["integrated_mb_s"], 2), round(r["peak_mb"], 2))
        for r in ranked[:5]
    )

    result.wells = len(sink)
    if sink:
        result.out_shape = tuple(sink[0][2].shape)
        result.out_megapixels = sum(
            int(np.prod(img.shape)) for _r, _f, img, _g in sink) / 1e6

    if quality and sink and not result.error:
        result.quality = _quality(operator, kind, reader, meta, sink, channels)

    del sink        # a fused mosaic is ~0.9 GB; do not hold it across operators
    return result


def _prepare(operator: str, reader, meta: dict, regions, n_fovs) -> None:
    """Anything an operator needs installed before it can run, outside the timed window.

    Only ``flatfield`` has one: it is selected by name, cannot take a profile argument, and
    fails loud when none is installed.
    """
    if operator != "flatfield":
        return
    from squidxplorer._flatfield import active_profiles, set_profiles
    from squidxplorer._stitch import estimate_region_flatfield

    if active_profiles():
        return
    region = regions[0]
    fovs = list((meta.get("fovs_per_region") or {}).get(region, ()))[: (n_fovs or 3) or 3]
    if not fovs:
        return
    # One profile per channel: the operator refuses a channel it has no measured field for.
    set_profiles(estimate_region_flatfield(reader, region, fovs, max_tiles=len(fovs)))


def _quality(operator: str, kind: str, reader, meta: dict, sink, channels) -> dict:
    """Operator-appropriate quality numbers, measured against a REAL baseline: for a FOV
    operator, the same FOV's middle z-plane, read raw."""
    region, fov, image, geometry = sink[0]
    out = {}
    try:
        if kind == "region":
            return _seam_quality(reader, meta, region, geometry, channels)

        c_i = 0
        plane = np.asarray(image[0, c_i, image.shape[2] // 2])
        z_levels = meta["z_levels"]
        channel = meta["channels"][c_i]["name"]
        raw = reader.read(region, fov, channel, z_levels[len(z_levels) // 2], 0)

        s_out, s_raw = relative_gradient_energy(plane), relative_gradient_energy(raw)
        f_out, f_raw = block_uniformity(plane), block_uniformity(raw)
        out["sharp_gain"] = s_out / s_raw if s_raw else float("nan")
        # Also the UNNORMALISED ratio: normalisation is not neutral for every operator (a
        # MIP lifts the mean more than the edges, so sharp_gain alone reads 0.65 on a
        # plainly sharper image). Both are printed; QUALITY_NOTES says which is operative.
        out["sharp_abs"] = _abs_gradient(plane) / (_abs_gradient(raw) or float("nan"))
        out["flat_gain"] = f_out / f_raw if f_raw else float("nan")
        if operator in ("bgsub", "decon"):
            out["clipped"] = float((plane == 0).mean())
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _seam_quality(reader, meta: dict, region: str, geometry: Optional[dict],
                  channels) -> dict:
    """Overlap NCC on the strongest overlapping FOV pair, at the placement this run solved.

    Offsets come from the operator's own ``geometry["origins_px"]``, not the stage
    coordinates — that is what makes ``stitch`` and ``coordinate`` rows comparable, their
    difference being registration's value.
    """
    if not geometry or "origins_px" not in geometry:
        return {"seam_ncc": float("nan"), "note": "operator reported no geometry"}
    fovs = list(geometry["fovs"])
    origins = list(geometry["origins_px"])
    ty, tx = geometry["tile_shape"]
    if len(fovs) < 2:
        return {"seam_ncc": float("nan"), "note": "region has < 2 FOVs"}

    # WHICH pair is chosen from the stage coordinates (the shared input), never from either
    # operator's own solved origins — else stitch and coordinate could score different seams.
    from squidxplorer._placement import fov_offsets_px

    stage = fov_offsets_px(meta["fov_positions_um"], region, fovs,
                           float(meta["pixel_size_um"]))
    best = None
    for i in range(len(fovs)):
        for j in range(i + 1, len(fovs)):
            sy = float(stage[fovs[j]][0] - stage[fovs[i]][0])
            sx = float(stage[fovs[j]][1] - stage[fovs[i]][1])
            # Diagonal neighbours touch at a corner only, not a scoreable rectangle.
            if abs(sy) >= ty or abs(sx) >= tx:
                continue
            area = (ty - abs(sy)) * (tx - abs(sx))
            if best is None or area > best[0]:
                best = (area, i, j)
    if best is None:
        return {"seam_ncc": float("nan"), "note": "no overlapping FOV pair"}
    _area, i, j = best
    # ...but the OFFSETS are the operator's own solved placement.
    dy = float(origins[j][0] - origins[i][0])
    dx = float(origins[j][1] - origins[i][1])

    from squidxplorer.projection import project

    c_i = list(channels)[0] if channels else 0
    channel = meta["channels"][c_i]["name"]
    z_levels = meta["z_levels"]
    tiles = [project(reader.read(region, fovs[k], channel, z, 0) for z in z_levels)
             for k in (i, j)]
    return {"seam_ncc": overlap_ncc(tiles[0], tiles[1], dy, dx),
            "seam_pair": f"{fovs[i]}|{fovs[j]}",
            "seam_dy_px": round(dy, 2), "seam_dx_px": round(dx, 2)}


# --------------------------------------------------------------------------------------
# Suite + reporting
# --------------------------------------------------------------------------------------

DEFAULT_OPERATORS = ("mip", "reference", "flatfield", "bgsub", "decon", "coordinate", "stitch")


def benchmark_dataset(
    dataset: str,
    operators: Sequence[str] = DEFAULT_OPERATORS,
    *,
    regions: Optional[Sequence[str]] = None,
    n_fovs: Optional[int] = 1,
    region_n_fovs: Optional[int] = None,
    region_channels: Optional[Sequence[int]] = (1,),
    workers: int = 1,
    quality: bool = True,
    on_error: Optional[Callable[[str, Exception], None]] = None,
) -> list:
    """Run the suite over one acquisition and return one :class:`OperatorResult` each.

    Region operators get ALL FOVs but a SINGLE channel by default: fusing at 4 channels
    would measure memory rather than stitching, and truncating to the first few FOVs is
    worse — ``select_fovs`` takes an index-order prefix of a serpentine scan, which is not
    a connected region (measured: disconnected tile groups, placed by the affine fallback).
    """
    from squidxplorer import open_reader

    reader = open_reader(dataset)
    region_ops = set(available_region_operators())
    results = []
    for op in operators:
        is_region = op in region_ops
        try:
            results.append(benchmark_operator(
                reader, op, regions=regions,
                n_fovs=region_n_fovs if is_region else n_fovs,
                workers=workers,
                channels=region_channels if is_region else None,
                quality=quality,
            ))
        except Exception as exc:
            if on_error is not None:
                on_error(op, exc)
            results.append(OperatorResult(
                operator=op, kind="region" if is_region else "fov", dataset=dataset,
                error=f"{type(exc).__name__}: {exc}",
            ))
    return results


_COLUMNS = ("operator", "kind", "wells", "out_Mpix", "wall_ms", "read_ms", "compute_ms",
            "Mpix_s", "peak_RSS_MB", "dRSS_MB", "MB_s", "persist_GB", "quality")


def format_table(results: Sequence[OperatorResult], *, notes: bool = True) -> str:
    """A fixed-width table, one row per operator. Errors get their own line, never a blank."""
    rows = [r.as_row() for r in results]
    widths = {c: max(len(c), *(len(str(row[c])) for row in rows)) for c in _COLUMNS} if rows \
        else {c: len(c) for c in _COLUMNS}
    lines = ["  ".join(c.rjust(widths[c]) if c != "quality" else c for c in _COLUMNS)]
    lines.append("  ".join("-" * widths[c] for c in _COLUMNS))
    for row in rows:
        lines.append("  ".join(
            str(row[c]).rjust(widths[c]) if c != "quality" else str(row[c]) for c in _COLUMNS))
    for r in results:
        if r.error:
            lines.append(f"  ! {r.operator}: {r.error}")
    if notes:
        lines.append("")
        lines.append("quality, and which number to read:")
        for r in results:
            note = QUALITY_NOTES.get(r.operator)
            if note:
                lines.append(f"  {r.operator:<11} {note}")
        caveats = [(r.operator, QUALITY_CAVEATS[r.operator])
                   for r in results if r.operator in QUALITY_CAVEATS]
        if caveats:
            lines.append("")
            lines.append("caveats:")
            for op, text in caveats:
                lines.append(f"  {op}: {text}")
    return "\n".join(lines)


def format_stages(results: Sequence[OperatorResult]) -> str:
    """Per-stage ms and peak RSS — the ``assign_stages`` half of Julio's suite."""
    lines = []
    for r in results:
        if not r.stage_ms:
            continue
        lines.append(f"{r.operator} ({r.kind}):")
        for name, ms in r.stage_ms.items():
            peak = r.stage_peak_rss_mb.get(name)
            peak_s = f"{peak:8.1f} MB peak RSS" if peak else " " * 20
            lines.append(f"    {name:<12} {ms:9.1f} ms   {peak_s}")
        # Stages never sum to the wall clock: this is the work outside any span (setup,
        # sampler overhead, teardown), which `assign_stages` calls "(other)".
        other = r.wall_ms - sum(r.stage_ms.values())
        peak = r.stage_peak_rss_mb.get("(other)")
        peak_s = f"{peak:8.1f} MB peak RSS" if peak else " " * 20
        lines.append(f"    {'(other)':<12} {other:9.1f} ms   {peak_s}")
    return "\n".join(lines)


def format_allocations(results: Sequence[OperatorResult], top: int = 3) -> str:
    """Top allocating functions per operator, by integrated MB·seconds (``compute_ranking``)."""
    lines = []
    for r in results:
        if not r.top_functions:
            continue
        lines.append(f"{r.operator}: {r.integrated_mb_s:.1f} MB·s total")
        for func, mb_s, peak in r.top_functions[:top]:
            lines.append(f"    {func:<44} {mb_s:9.2f} MB·s   {peak:8.2f} MB peak")
    return "\n".join(lines)


def write_csv(results: Sequence[OperatorResult], path) -> Path:
    """The table as CSV, in the shape of ``profiling.record``'s writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(_COLUMNS) + ["error"])
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in r.as_row().items() if k in w.fieldnames})
    return path


def write_json(results: Sequence[OperatorResult], path, *, meta: Optional[dict] = None) -> Path:
    """Everything, including the machine it was measured on — for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "meta": meta or {},
        "results": [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(r).items()}
            | {"mpix_per_s": r.mpix_per_s, "compute_ms": r.compute_ms}
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
