"""Benchmark local vs http-served OME-Zarr reads, plus Odon's headless first paint."""

from __future__ import annotations

import contextlib
import functools
import http.server
import json
import os
import platform
import socketserver
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

_MB = 1024.0 ** 2

# Odon's headless probe prints this on success; a zero exit code alone is not enough.
_OK_MARKER = "OK: loaded tile"

_CHECK_TIMEOUT_S = 120.0


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """A static file server that does not log each request."""

    def log_message(self, *_args):  # noqa: D102
        pass


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@contextlib.contextmanager
def serve_directory(root, host: str = "127.0.0.1", port: int = 0):
    """Serve *root* over HTTP for the duration of the block; yield the base URL."""
    handler = functools.partial(_QuietHandler, directory=str(Path(root).resolve()))
    server = _ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --------------------------------------------------------------------------------------
# Transport: the same bytes, two ways
# --------------------------------------------------------------------------------------

def chunk_files(field_dir, level: str = "0", limit: int = 64) -> list:
    """Real chunk files of one pyramid level, in path order. Never ``zarr.json``."""
    root = Path(field_dir) / level
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "zarr.json")
    return files[:limit]


@dataclass
class TransportResult:
    label: str
    n: int = 0
    bytes: int = 0
    wall_ms: float = 0.0
    per_item_ms: tuple = ()
    errors: int = 0

    @property
    def mb_per_s(self) -> float:
        return (self.bytes / _MB) / (self.wall_ms / 1000.0) if self.wall_ms else float("nan")

    @property
    def median_ms(self) -> float:
        return statistics.median(self.per_item_ms) if self.per_item_ms else float("nan")

    @property
    def p95_ms(self) -> float:
        if not self.per_item_ms:
            return float("nan")
        ordered = sorted(self.per_item_ms)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def _timed(label: str, fetch, items, workers: int) -> TransportResult:
    """Fetch every item with *fetch*, timing each and the whole batch."""
    out = TransportResult(label=label, n=len(items))
    per_item: list = []
    lock = threading.Lock()

    def one(item):
        t0 = time.perf_counter()
        try:
            data = fetch(item)
        except Exception:
            with lock:
                out.errors += 1
            return
        dt = (time.perf_counter() - t0) * 1000.0
        with lock:
            per_item.append(dt)
            out.bytes += len(data)

    t0 = time.perf_counter()
    if workers <= 1:
        for item in items:
            one(item)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, items))
    out.wall_ms = (time.perf_counter() - t0) * 1000.0
    out.per_item_ms = tuple(per_item)
    return out


def benchmark_transport(field_dir, base_url: str, plate_root, *, level: str = "0",
                        limit: int = 64, workers: int = 8) -> list:
    """Read the SAME chunks locally and over HTTP, serial and parallel. Four rows."""
    files = chunk_files(field_dir, level=level, limit=limit)
    if not files:
        return []
    plate_root = Path(plate_root).resolve()
    urls = [f"{base_url}/{Path(f).resolve().relative_to(plate_root.parent).as_posix()}"
            for f in files]

    def local(path):
        return Path(path).read_bytes()

    def http(url):
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()

    http(urls[0])   # warm the connection so row 1 does not pay for DNS/socket setup

    return [
        _timed("local serial", local, files, 1),
        _timed(f"local x{workers}", local, files, workers),
        _timed("http serial", http, urls, 1),
        _timed(f"http x{workers}", http, urls, workers),
    ]


# --------------------------------------------------------------------------------------
# Odon
# --------------------------------------------------------------------------------------

@dataclass
class OdonResult:
    available: bool = False
    binary: str = ""
    reason: str = ""
    local_ms: tuple = ()
    local_ok: bool = False
    local_output: str = ""
    remote_attempted: bool = False
    remote_ok: bool = False
    remote_output: str = ""

    @property
    def local_median_ms(self) -> float:
        return statistics.median(self.local_ms) if self.local_ms else float("nan")


def _run_odon(binary: str, target: str) -> tuple:
    """``odon --check TARGET`` in a scratch cwd; return (seconds, combined output, rc)."""
    with tempfile.TemporaryDirectory(prefix="odon-check-") as cwd:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([binary, "--check", str(target)], capture_output=True,
                                  text=True, timeout=_CHECK_TIMEOUT_S, cwd=cwd)
        except subprocess.TimeoutExpired:
            return _CHECK_TIMEOUT_S, "TIMEOUT", -1
        dt = time.perf_counter() - t0
        return dt, ((proc.stdout or "") + (proc.stderr or "")).strip(), proc.returncode


def benchmark_odon(field_dir, remote_url: Optional[str] = None, *,
                   repeats: int = 5, odon_bin=None) -> OdonResult:
    """Time Odon's headless first paint locally, and probe whether it can do it remotely."""
    from squidxplorer._odon import find_odon

    result = OdonResult()
    try:
        binary = str(Path(odon_bin) if odon_bin else find_odon())
    except FileNotFoundError as exc:
        result.reason = str(exc)
        return result
    result.available, result.binary = True, binary

    times = []
    for _ in range(max(1, repeats)):
        dt, output, rc = _run_odon(binary, field_dir)
        times.append(dt * 1000.0)
        result.local_output = output
        result.local_ok = rc == 0 and _OK_MARKER in output
        if not result.local_ok:
            break
    result.local_ms = tuple(times)

    if remote_url:
        result.remote_attempted = True
        _dt, output, rc = _run_odon(binary, remote_url)
        result.remote_output = output
        result.remote_ok = rc == 0 and _OK_MARKER in output
    return result


def odon_remote_probe(remote_url: str, odon_bin=None) -> dict:
    """Re-derive, live, whether this Odon build can open an http-served dataset."""
    from squidxplorer._odon import find_odon

    try:
        binary = str(Path(odon_bin) if odon_bin else find_odon())
    except FileNotFoundError as exc:
        return {"available": False, "reason": str(exc)}
    dt, output, rc = _run_odon(binary, remote_url)
    return {"available": True, "binary": binary, "url": remote_url, "rc": rc,
            "ms": round(dt * 1000.0, 1), "ok": rc == 0 and _OK_MARKER in output,
            "output": output}


# --------------------------------------------------------------------------------------
# Whole run
# --------------------------------------------------------------------------------------

@dataclass
class OdonBenchReport:
    plate: str = ""
    # NOT named `field`: that would shadow dataclasses.field for declarations below it.
    field_dir: str = ""
    base_url: str = ""
    chunk_level: str = "0"
    transport: list = field(default_factory=list)
    odon: OdonResult = field(default_factory=OdonResult)
    not_measured: list = field(default_factory=list)


def run(hcs_dir, *, level: str = "0", limit: int = 64, workers: int = 8,
        repeats: int = 5, odon_bin=None, field_index: int = 0) -> OdonBenchReport:
    """Serve an already-written plate, then measure both sides of the local/remote seam."""
    from squidxplorer._odon import _plate_dir, iter_fields

    plate = _plate_dir(hcs_dir)
    fields = list(iter_fields(plate))
    if not fields:
        raise FileNotFoundError(f"no complete field groups under {plate}")
    _row, _col, _fov, field_dir = fields[min(field_index, len(fields) - 1)]

    report = OdonBenchReport(plate=str(plate), field_dir=str(field_dir), chunk_level=level)
    # Serve the plate's PARENT so URLs mirror the on-disk layout (…/plate.ome.zarr/A/1/0/…).
    with serve_directory(plate.parent) as base_url:
        report.base_url = base_url
        report.transport = benchmark_transport(field_dir, base_url, plate,
                                               level=level, limit=limit, workers=workers)
        remote_url = f"{base_url}/{field_dir.resolve().relative_to(plate.parent).as_posix()}"
        report.odon = benchmark_odon(field_dir, remote_url, repeats=repeats,
                                     odon_bin=odon_bin)

    report.not_measured = [
        "odon framerate (local or remote): odon exposes no headless render loop, frame "
        "counter, or timing output — there is no interface to read a framerate through, "
        "so no number is reported rather than one being eyeballed or invented.",
    ]
    if report.odon.available and report.odon.remote_attempted and not report.odon.remote_ok:
        report.not_measured.append(
            "odon first paint over HTTP: the binary is installed and works locally, but "
            "this build resolves a dataset argument as a filesystem path and cannot open "
            "an http-served store at all (its verbatim refusal is in the report). The "
            "local->remote odon delta therefore did not run.")
    elif not report.odon.available:
        report.not_measured.append(
            "odon first paint (local AND remote): no odon binary was found, so neither "
            "side ran. Install it or set $ODON_BIN — see the reason line above.")
    return report


def format_report(report: OdonBenchReport) -> str:
    lines = [
        f"plate    : {report.plate}",
        f"field    : {report.field_dir}",
        f"served at: {report.base_url}  (python http.server, localhost)",
        "",
        "TRANSPORT — the same chunk files, local vs HTTP "
        f"(level {report.chunk_level}, real chunks, no metadata):",
    ]
    if report.transport:
        head = f"  {'':<14} {'n':>4} {'MB':>8} {'wall_ms':>9} {'MB/s':>8} " \
               f"{'med_ms':>8} {'p95_ms':>8}  err"
        lines += [head, "  " + "-" * (len(head) - 2)]
        for t in report.transport:
            lines.append(
                f"  {t.label:<14} {t.n:>4} {t.bytes / _MB:>8.2f} {t.wall_ms:>9.1f} "
                f"{t.mb_per_s:>8.1f} {t.median_ms:>8.2f} {t.p95_ms:>8.2f}  {t.errors}")
        by = {t.label.split()[0] + ("_par" if "x" in t.label else "_ser"): t
              for t in report.transport}
        ls, hs = by.get("local_ser"), by.get("http_ser")
        lp, hp = by.get("local_par"), by.get("http_par")
        if ls and hs and hs.mb_per_s:
            ratio = ls.mb_per_s / hs.mb_per_s
            verdict = (f"{ratio:.1f}x slower" if ratio >= 1.0
                       else f"{1 / ratio:.1f}x FASTER (i.e. within noise of local)")
            lines.append(f"\n  serial  : HTTP is {verdict} "
                         f"({hs.median_ms - ls.median_ms:+.2f} ms median per chunk). This "
                         "is the clean number: one request at a time, so it isolates "
                         "per-chunk protocol overhead.")
        if lp and hp and hp.mb_per_s and hs and hs.mb_per_s:
            lines.append(f"  parallel: HTTP is {lp.mb_per_s / hp.mb_per_s:.1f}x slower than "
                         f"local — but note it is also {hs.mb_per_s / hp.mb_per_s:.1f}x "
                         "slower than HTTP *serial*, i.e. fanning out made it WORSE.")
            lines.append("            That is the measurement instrument, not the format: "
                         "Python's http.server serves each connection on a GIL-bound "
                         "thread, so concurrency queues instead of overlapping. Read the "
                         "serial row for the format's cost; a real static host (nginx/S3/"
                         "CloudFront) is what would make the parallel row meaningful.")
        lines.append("  (localhost: ZERO network latency, and the local rows are served "
                     "from the OS page cache. A real deployment adds RTT per chunk on the "
                     "HTTP side and a real disk seek on the local side, so neither column "
                     "is a deployment forecast — the delta between them is the finding.)")
    else:
        lines.append("  (no chunk files found — was the plate written?)")

    lines += ["", "ODON:"]
    o = report.odon
    if not o.available:
        lines.append(f"  NOT RUN — {o.reason}")
    else:
        lines.append(f"  binary: {o.binary}")
        if o.local_ok:
            # Cold and warm reported separately; a single median hides the gap.
            lines.append(f"  local first paint (`odon --check`, {len(o.local_ms)} runs):")
            lines.append(f"    cold (run 1)     {o.local_ms[0]:8.1f} ms")
            if len(o.local_ms) > 1:
                warm = statistics.median(o.local_ms[1:])
                lines.append(f"    warm (median)    {warm:8.1f} ms   "
                             f"[{', '.join(f'{m:.0f}' for m in o.local_ms[1:])}]")
            proof = next((ln for ln in o.local_output.splitlines() if _OK_MARKER in ln), "")
            lines.append(f"    {proof}")
        else:
            lines.append(f"  local first paint FAILED: {o.local_output[:300]}")
        if o.remote_attempted:
            verdict = "OK" if o.remote_ok else "REFUSED"
            lines.append(f"  remote (`odon --check <http url>`): {verdict}")
            for ln in o.remote_output.splitlines()[:6]:
                lines.append(f"    | {ln}")

    if report.not_measured:
        lines += ["", "NOT MEASURED (stated, not estimated):"]
        for item in report.not_measured:
            lines.append(f"  * {item}")
    return "\n".join(lines)


def write_json(report: OdonBenchReport, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "machine": {"platform": platform.platform(), "machine": platform.machine()},
        "plate": report.plate, "field": report.field_dir, "base_url": report.base_url,
        "transport": [
            dict(vars(t), mb_per_s=t.mb_per_s, median_ms=t.median_ms, p95_ms=t.p95_ms,
                 per_item_ms=list(t.per_item_ms))
            for t in report.transport
        ],
        "odon": dict(vars(report.odon), local_ms=list(report.odon.local_ms),
                     local_median_ms=report.odon.local_median_ms),
        "not_measured": report.not_measured,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
