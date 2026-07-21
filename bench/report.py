"""Benchmark row schema, CSV append, and the markdown comparison table.

Schema decisions (IMA-233 D4)
-----------------------------
Provenance travels with every row, because a benchmark whose rows cannot be told apart
is a benchmark nobody trusts: timestamp, host, platform, dataset, tile geometry.

``git_sha`` is deliberately NOT the provenance key. The lost ``residual_benchmark.py``
keyed on it because it benchmarked *one* codebase against itself; here four of the five
tools are external programs whose behaviour has nothing to do with this repo's commit.
Each row therefore carries ``tool_version`` -- a container digest, a Fiji update-site
version, an ``ashlar.__version__`` -- captured by the adapter.

Every unmeasurable cell is an explicit status, never blank and never zero. A blank cell
reads as "the tool failed"; a zero reads as "perfect". Both are lies when the truth is
"this metric is undefined for this tool".
"""

from __future__ import annotations

import csv
import math
import platform
import socket
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

STATUS_OK = "OK"
STATUS_DISK_ABORT = "DISK_ABORT"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_CRASH = "CRASH"
STATUS_MISSING_TOOL = "MISSING_TOOL"
STATUS_QUALITY_NA = "QUALITY_NA"

ALL_STATUSES = (
    STATUS_OK,
    STATUS_DISK_ABORT,
    STATUS_TIMEOUT,
    STATUS_CRASH,
    STATUS_MISSING_TOOL,
    STATUS_QUALITY_NA,
)

# Why a metric could not be produced, so an empty column is never ambiguous.
QUALITY_NA_REASONS = {
    "non_rigid_model": "tool solves a non-rigid/affine warp; no scalar shift exists",
    "no_positions": "tool did not emit parseable tile positions",
    "no_overlap": "tiles do not overlap; nothing to measure",
    "no_texture": "overlap regions failed the NCC gate (blank or saturated)",
    "not_run": "tool did not complete, so there is no output to measure",
}


@dataclass
class BenchmarkRow:
    """One (tool, dataset, run) result. Field order IS the CSV column order."""

    # identity / provenance
    timestamp: str = ""
    host: str = ""
    platform: str = ""
    tool: str = ""
    tool_version: str = ""
    dataset: str = ""
    path: str = ""

    # dataset geometry
    region: str = ""
    n_tiles: int = 0
    tile_y: int = 0
    tile_x: int = 0
    pixel_size_um: float = float("nan")
    n_channels: int = 0

    # speed -- cold includes runtime/container startup, warm is steady state.
    # Reported separately because for a LIVE stitcher, per-invocation startup is
    # often the number that decides adoption.
    t_wall_cold_s: float = float("nan")
    t_wall_s: float = float("nan")

    # footprint
    peak_rss_mb: float = float("nan")
    rss_measurable: bool = True
    output_bytes: int = 0
    output_bytes_measurable: bool = True

    # quality
    resid_median_px: float = float("nan")
    resid_mean_px: float = float("nan")
    resid_p90_px: float = float("nan")
    n_seams_measured: int = 0
    n_blocks_locked: int = 0
    n_pairs_candidate: int = 0

    # outcome
    status: str = STATUS_OK
    quality_status: str = STATUS_OK
    quality_na_reason: str = ""
    detail: str = ""

    # run config, so a row is reproducible
    threads: int = 0
    compression: str = ""
    sampler_interval_s: float = float("nan")

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not self.host:
            self.host = socket.gethostname()
        if not self.platform:
            self.platform = f"{platform.system()}-{platform.machine()}"


CSV_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(BenchmarkRow))


def append_csv(row: BenchmarkRow, out_path: str | Path) -> Path:
    """Append one row, writing the header if the file is new.

    Refuses to append to a file with a different schema rather than silently producing
    a CSV whose columns shift halfway down -- that corrupts every downstream read.
    """
    out_path = Path(out_path)
    exists = out_path.exists() and out_path.stat().st_size > 0
    if exists:
        with out_path.open(newline="") as fh:
            header = next(csv.reader(fh), [])
        if tuple(header) != CSV_COLUMNS:
            raise ValueError(
                f"{out_path} has a different schema ({len(header)} cols, expected "
                f"{len(CSV_COLUMNS)}). Refusing to append and corrupt it; write to a "
                "new file or migrate the old one."
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow(asdict(row))
    return out_path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _fmt(value, spec: str = ".2f") -> str:
    if value is None or value == "":
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "n/a"
    return format(f, spec)


def _fmt_bytes(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "n/a"
    if n <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"  # pragma: no cover


def markdown_table(rows: list[BenchmarkRow] | list[dict]) -> str:
    """Human-readable comparison table -- the actual deliverable of IMA-233.

    Unmeasurable cells render as ``n/a`` with the reason in a footnote, so a reader can
    never mistake "we could not measure this" for "this tool scored zero".
    """
    dicts = [r if isinstance(r, dict) else asdict(r) for r in rows]
    if not dicts:
        return "_No benchmark rows._\n"

    head = (
        "| Tool | Version | Status | Wall (s) | Cold (s) | Peak RSS (MB) | "
        "Output | Seam resid median (px) | p90 | Seams |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    notes: list[str] = []

    for d in dicts:
        rss = (
            _fmt(d.get("peak_rss_mb"))
            if str(d.get("rss_measurable", True)).lower() not in ("false", "0")
            else "n/a"
        )
        out = (
            _fmt_bytes(d.get("output_bytes"))
            if str(d.get("output_bytes_measurable", True)).lower() not in ("false", "0")
            else "n/a"
        )
        qstat = d.get("quality_status", STATUS_OK)
        if qstat == STATUS_OK:
            med, p90 = _fmt(d.get("resid_median_px")), _fmt(d.get("resid_p90_px"))
        else:
            med = p90 = "n/a"
            reason = d.get("quality_na_reason") or ""
            note = QUALITY_NA_REASONS.get(reason, reason)
            if note:
                notes.append(f"- **{d.get('tool')}** quality n/a: {note}")
        lines.append(
            f"| {d.get('tool','')} | {d.get('tool_version','') or 'n/a'} | "
            f"{d.get('status','')} | {_fmt(d.get('t_wall_s'))} | "
            f"{_fmt(d.get('t_wall_cold_s'))} | {rss} | {out} | {med} | {p90} | "
            f"{d.get('n_seams_measured', 0)} |"
        )

    body = "\n".join(lines)
    if notes:
        body += "\n\n" + "\n".join(dict.fromkeys(notes))
    body += (
        "\n\n_Seam residual is measured on the INPUT tiles placed at each tool's own "
        "solved positions; lower is better. Peak RSS is sampled, so spikes shorter "
        "than the sampling interval are missed._\n"
    )
    return body
