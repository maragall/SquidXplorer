"""The benchmark runner: spawn, measure, score, reclaim, repeat.

One tool at a time, and each tool's output is deleted once its metrics are extracted.
That per-tool reclaim is not tidiness -- it is what makes the run possible at all. Peak
disk becomes one mosaic instead of N, which is the difference between a run that
completes on Nick's hardware and one that dies halfway through the acquisition
everyone waited months for.

Per tool::

    is_available? ---- no ---> MISSING_TOOL row, move on
         | yes
    preflight free space ---- insufficient ---> DISK_ABORT row, move on
         | ok
    spawn subprocess  <---- sampler thread: peak RSS + disk low-water watchdog
         |                                       |
         |                                  breach -> kill tree -> DISK_ABORT
         |
    wait(timeout) ---- expired ---> kill tree -> TIMEOUT row
         | exit 0                             ---- non-zero ---> CRASH row
    collect solved positions ---- none ---> quality QUALITY_NA(no_positions)
         |
    seam residual on INPUT tiles at those positions
         |
    row -> CSV, delete output, next tool
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from bench.adapters.base import StitcherAdapter, StitchRequest
from bench.dataset import Acquisition
from bench.metrics import adjacent_pairs, seam_residual
from bench.report import (
    STATUS_CRASH,
    STATUS_DISK_ABORT,
    STATUS_MISSING_TOOL,
    STATUS_OK,
    STATUS_QUALITY_NA,
    STATUS_TIMEOUT,
    BenchmarkRow,
)
from bench.sampler import (
    DEFAULT_INTERVAL_S,
    DEFAULT_MIN_FREE_BYTES,
    ResourceSampler,
    dir_bytes,
    kill_tree,
    preflight_disk,
)

DEFAULT_TIMEOUT_S = 3600.0


@dataclass
class RunConfig:
    region: str = ""
    channel: str = ""
    z: int = 0
    threads: int = 1
    compression: str = "none"
    timeout_s: float = DEFAULT_TIMEOUT_S
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    sampler_interval_s: float = DEFAULT_INTERVAL_S
    warm_runs: int = 1
    keep_output: bool = False
    n_blocks: int = 8


@dataclass
class ProcOutcome:
    returncode: int
    wall_s: float
    peak_rss_mb: float
    output_bytes: int
    disk_abort: bool
    timed_out: bool
    stderr: str


def _spawn_and_measure(cmd: list[str], out_dir: Path, cfg: RunConfig, cwd: Path) -> ProcOutcome:
    """Run one subprocess to completion under the sampler. Never raises on tool failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    sampler = ResourceSampler(
        pid=proc.pid,
        watch_dir=out_dir,
        min_free_bytes=cfg.min_free_bytes,
        interval_s=cfg.sampler_interval_s,
        on_abort=lambda: kill_tree(proc.pid),
    )
    sampler.start()

    timed_out = False
    try:
        _, stderr = proc.communicate(timeout=cfg.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc.pid)
        try:
            _, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - the kill failed
            stderr = ""
    wall = time.perf_counter() - t0
    res = sampler.stop()

    return ProcOutcome(
        returncode=proc.returncode if proc.returncode is not None else -1,
        wall_s=wall,
        peak_rss_mb=res.peak_rss_mb,
        output_bytes=max(res.peak_output_bytes, dir_bytes(out_dir)),
        disk_abort=res.disk_abort,
        timed_out=timed_out,
        stderr=(stderr or "")[-4000:],
    )


def _base_row(adapter: StitcherAdapter, acq: Acquisition, cfg: RunConfig) -> BenchmarkRow:
    return BenchmarkRow(
        tool=adapter.name,
        dataset=acq.root.name,
        path=str(acq.root),
        region=cfg.region,
        n_tiles=len(acq.fovs(cfg.region)),
        tile_y=acq.frame_shape[0],
        tile_x=acq.frame_shape[1],
        pixel_size_um=acq.pixel_size_um,
        n_channels=len(acq.channels),
        rss_measurable=adapter.measurable_rss,
        output_bytes_measurable=adapter.measurable_output_bytes,
        threads=cfg.threads,
        compression=cfg.compression,
        sampler_interval_s=cfg.sampler_interval_s,
    )


def _score_quality(
    adapter: StitcherAdapter,
    acq: Acquisition,
    cfg: RunConfig,
    row: BenchmarkRow,
    positions_px: dict[int, tuple[float, float]] | None,
) -> None:
    """Fill the quality columns, or say precisely why they cannot be filled."""
    if not adapter.supports_quality:
        row.quality_status = STATUS_QUALITY_NA
        row.quality_na_reason = adapter.quality_na_reason or "non_rigid_model"
        return
    if not positions_px:
        row.quality_status = STATUS_QUALITY_NA
        row.quality_na_reason = "no_positions"
        return

    pairs = adjacent_pairs(positions_px, acq.frame_shape)
    if not pairs:
        row.quality_status = STATUS_QUALITY_NA
        row.quality_na_reason = "no_overlap"
        return

    cache: dict[int, object] = {}

    def read_tile(fov: int):
        if fov not in cache:
            cache[fov] = acq.read(cfg.region, fov, cfg.z, cfg.channel)
        return cache[fov]

    stats = seam_residual(
        read_tile,
        positions_px,
        pairs=pairs,
        n_blocks=cfg.n_blocks,
    )
    row.resid_median_px = float(stats["resid_median_px"])
    row.resid_mean_px = float(stats["resid_mean_px"])
    row.resid_p90_px = float(stats["resid_p90_px"])
    row.n_seams_measured = int(stats["n_seams_measured"])
    row.n_blocks_locked = int(stats["n_blocks_locked"])
    row.n_pairs_candidate = int(stats["n_pairs_candidate"])
    if row.n_seams_measured == 0:
        row.quality_status = STATUS_QUALITY_NA
        row.quality_na_reason = "no_texture"


def run_one(
    adapter: StitcherAdapter,
    acq: Acquisition,
    cfg: RunConfig,
    out_root: str | Path,
    repo_root: str | Path | None = None,
) -> BenchmarkRow:
    """Benchmark a single stitcher. Always returns a row -- failures are data."""
    out_root = Path(out_root)
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    row = _base_row(adapter, acq, cfg)

    if not adapter.is_available():
        row.status = STATUS_MISSING_TOOL
        row.quality_status = STATUS_QUALITY_NA
        row.quality_na_reason = "not_run"
        row.detail = f"{adapter.name} is not installed or not importable"
        return row

    row.tool_version = adapter.version()
    out_dir = out_root / adapter.name
    req = StitchRequest(
        acquisition=acq,
        region=cfg.region,
        channel=cfg.channel,
        z=cfg.z,
        out_dir=out_dir,
        threads=cfg.threads,
        compression=cfg.compression,
    )

    # A stitcher's peak disk is dominated by the fused mosaic; require headroom for one
    # plus the low-water reserve before we even start.
    required = acq.frame_shape[0] * acq.frame_shape[1] * 2 * max(1, len(acq.fovs(cfg.region)))
    ok, free = preflight_disk(out_root, required + cfg.min_free_bytes)
    if not ok:
        row.status = STATUS_DISK_ABORT
        row.quality_status = STATUS_QUALITY_NA
        row.quality_na_reason = "not_run"
        row.detail = f"preflight: need ~{required} B + reserve, only {free} B free"
        return row

    try:
        adapter.prepare(req)
        cmd = adapter.build_command(req)

        # Cold pass: includes interpreter/JVM/container startup, which for a live
        # stitcher is often the number that actually decides adoption.
        cold = _spawn_and_measure(cmd, out_dir, cfg, cwd=repo_root)
        row.t_wall_cold_s = cold.wall_s
        row.peak_rss_mb = cold.peak_rss_mb
        row.output_bytes = cold.output_bytes

        if cold.disk_abort:
            row.status = STATUS_DISK_ABORT
            row.quality_status = STATUS_QUALITY_NA
            row.quality_na_reason = "not_run"
            row.detail = "free space crossed the low-water mark; subprocess killed"
            return row
        if cold.timed_out:
            row.status = STATUS_TIMEOUT
            row.quality_status = STATUS_QUALITY_NA
            row.quality_na_reason = "not_run"
            row.detail = f"exceeded {cfg.timeout_s}s; subprocess killed"
            return row
        if cold.returncode != 0:
            row.status = STATUS_CRASH
            row.quality_status = STATUS_QUALITY_NA
            row.quality_na_reason = "not_run"
            row.detail = f"exit {cold.returncode}: {cold.stderr.strip().splitlines()[-1] if cold.stderr.strip() else ''}"
            return row

        row.t_wall_s = cold.wall_s
        for _ in range(max(0, cfg.warm_runs)):
            warm = _spawn_and_measure(cmd, out_dir, cfg, cwd=repo_root)
            if warm.returncode == 0 and not warm.timed_out and not warm.disk_abort:
                row.t_wall_s = warm.wall_s
                row.peak_rss_mb = max(row.peak_rss_mb, warm.peak_rss_mb)
                row.output_bytes = max(row.output_bytes, warm.output_bytes)

        outcome = adapter.collect(req)
        _score_quality(adapter, acq, cfg, row, outcome.positions_px)
        row.status = STATUS_OK
        return row
    finally:
        if not cfg.keep_output:
            _reclaim(out_dir)


def _reclaim(out_dir: Path) -> None:
    """Delete one tool's output so peak disk stays at one mosaic, not N."""
    import shutil

    try:
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
    except OSError:
        pass


def run_benchmark(
    adapters: list[StitcherAdapter],
    acq: Acquisition,
    cfg: RunConfig,
    out_root: str | Path,
    csv_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> list[BenchmarkRow]:
    """Run every adapter in turn, appending each row as it completes.

    Rows are written incrementally so a run killed at tool 3 still leaves the results
    for tools 1 and 2 on disk.
    """
    from bench.report import append_csv

    rows: list[BenchmarkRow] = []
    for adapter in adapters:
        row = run_one(adapter, acq, cfg, out_root, repo_root=repo_root)
        rows.append(row)
        if csv_path:
            append_csv(row, csv_path)
    return rows
