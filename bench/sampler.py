"""External resource measurement: process-tree peak RSS + a disk low-water watchdog.

Two jobs, one thread (IMA-233 D3 + D5). They share a thread because they share a
cadence, and because the disk check must keep running for exactly as long as the
subprocess it is guarding.

Peak RSS is sampled with ``ps`` rather than ``resource.getrusage`` or ``psutil``:

  * ``getrusage(RUSAGE_CHILDREN).ru_maxrss`` is a high-water mark over *all* children
    ever reaped, so consecutive tools contaminate each other -- and its unit differs by
    platform (bytes on macOS, kilobytes on Linux), which silently corrupts any table
    that merges rows from both.
  * ``psutil`` is not a SquidMIP dependency and this harness must not add runtime deps.

``ps -o rss=`` reports kilobytes on both macOS and Linux, so one code path is correct
on both.

KNOWN BLIND SPOT -- containers. A tool that runs its real work inside Docker (MCmicro)
executes in a cgroup that is not a descendant of our subprocess. Process-tree RSS
cannot see it and will report a small, wrong number. :func:`sample_tree_rss_mb` returns
what it can see; the adapter declares ``measurable_rss = False`` so the runner records
``NaN`` instead of a lie. Same for ``du`` when a tool writes to a container volume.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Poll interval. Fast enough to catch a plausible allocation ramp, slow enough that the
# `ps` fork does not perturb what it is measuring. Short spikes between polls are missed
# by construction -- that is inherent to sampling, and is stated in the report.
DEFAULT_INTERVAL_S = 0.25

# Abort a run when free space would drop below this. Sized so a killed run still leaves
# room to write the CSV row explaining why it died.
DEFAULT_MIN_FREE_BYTES = 2 * 1024**3  # 2 GiB


def _ps_snapshot() -> list[tuple[int, int, int]]:
    """``[(pid, ppid, rss_kb)]`` for every visible process. Empty if ``ps`` is unusable."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[tuple[int, int, int]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return rows


def sample_tree_rss_mb(root_pid: int) -> float:
    """Summed RSS in MB of ``root_pid`` and every descendant alive at this instant.

    Summed, not maxed: a stitcher that forks four workers really is using the total.
    """
    rows = _ps_snapshot()
    if not rows:
        return 0.0
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for pid, ppid, kb in rows:
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb
    if root_pid not in rss:
        return 0.0

    total_kb = 0
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total_kb += rss.get(pid, 0)
        stack.extend(children.get(pid, ()))
    return total_kb / 1024.0


def dir_bytes(path: str | Path) -> int:
    """Total bytes of a directory tree. Follows no symlinks; missing tree is 0."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def free_bytes(path: str | Path) -> int:
    """Free bytes on the filesystem holding ``path`` (nearest existing ancestor)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        # Fail OPEN on a stat error: refusing to run because we could not read the
        # disk would be worse than running without the guard.
        return 1 << 62


@dataclass
class SampleResult:
    peak_rss_mb: float = 0.0
    samples: int = 0
    disk_abort: bool = False
    free_bytes_at_abort: int = 0
    peak_output_bytes: int = 0
    interval_s: float = DEFAULT_INTERVAL_S
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class ResourceSampler:
    """Polls a subprocess tree for peak RSS while watching free disk.

    On breaching the low-water mark it invokes ``on_abort`` (the runner kills the
    subprocess) and records ``disk_abort``. The run then becomes a ``DISK_ABORT`` row
    rather than a full disk -- which is the acceptance criterion IMA-233 is named for.

    Usage::

        with ResourceSampler(proc.pid, out_dir) as s:
            proc.wait()
        s.result.peak_rss_mb
    """

    def __init__(
        self,
        pid: int,
        watch_dir: str | Path | None = None,
        min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
        interval_s: float = DEFAULT_INTERVAL_S,
        on_abort=None,
    ) -> None:
        self.pid = pid
        self.watch_dir = Path(watch_dir) if watch_dir else None
        self.min_free_bytes = min_free_bytes
        self.interval_s = interval_s
        self.on_abort = on_abort
        self.result = SampleResult(interval_s=interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ResourceSampler:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="bench-sampler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> SampleResult:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        # Final output-size reading after the process has finished writing.
        if self.watch_dir is not None:
            self.result.peak_output_bytes = max(
                self.result.peak_output_bytes, dir_bytes(self.watch_dir)
            )
        return self.result

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = sample_tree_rss_mb(self.pid)
            if rss > self.result.peak_rss_mb:
                self.result.peak_rss_mb = rss
            self.result.samples += 1

            if self.watch_dir is not None:
                grown = dir_bytes(self.watch_dir)
                if grown > self.result.peak_output_bytes:
                    self.result.peak_output_bytes = grown
                free = free_bytes(self.watch_dir)
                if free < self.min_free_bytes and not self.result.disk_abort:
                    self.result.disk_abort = True
                    self.result.free_bytes_at_abort = free
                    if self.on_abort is not None:
                        try:
                            self.on_abort()
                        except Exception:  # pragma: no cover - abort must not raise
                            pass
                    return
            self._stop.wait(self.interval_s)


def preflight_disk(out_dir: str | Path, required_bytes: int) -> tuple[bool, int]:
    """Check free space before launching. Returns ``(ok, free_bytes)``.

    Unlike ``squidmip/_viewer.py:_check_disk`` this is never bypassed for subset runs
    and is applied per-tool, because each stitcher writes its own output independently.
    """
    free = free_bytes(out_dir)
    return free >= required_bytes, free


def kill_tree(pid: int) -> None:
    """Best-effort SIGKILL of a process and its descendants, deepest first."""
    rows = _ps_snapshot()
    children: dict[int, list[int]] = {}
    for p, ppid, _ in rows:
        children.setdefault(ppid, []).append(p)

    order: list[int] = []
    stack = [pid]
    seen: set[int] = set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        order.append(p)
        stack.extend(children.get(p, ()))

    for p in reversed(order):  # children before parents, so nothing gets re-parented
        try:
            os.kill(p, 9)
        except (OSError, ProcessLookupError):
            continue


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Poll ``predicate`` until true or timeout. Used by tests to avoid sleep races."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
