"""Current and peak memory footprint of this process, on every platform we ship to."""
from __future__ import annotations

import sys
from typing import Any, Optional

_MB = 1024 * 1024

#: Sentinel: "not passed, find the real source". Distinct from an explicit ``None``.
_AUTO = object()


def _memory_info(process: Any) -> Any:
    """``process.memory_info()``, or None. Never raises."""
    if process is None:
        return None
    try:
        return process.memory_info()
    except Exception:                       # noqa: BLE001 - access denied, dead process, no psutil
        return None


def _peak_from_working_set(info: Any, platform: str) -> Optional[float]:
    """The Windows peak working set in MB, or None where there is no such number."""
    if not platform.startswith("win"):
        return None
    value = getattr(info, "peak_wset", None)
    if not value:
        return None
    return float(value) / _MB


def _peak_from_rusage(usage: Any, platform: str) -> Optional[float]:
    """The POSIX high-water mark in MB, or None on Windows."""
    if platform.startswith("win"):
        return None
    value = getattr(usage, "ru_maxrss", None)
    if not value:
        return None
    m = float(value)
    return m / _MB if platform == "darwin" else m / 1024.0     # darwin: bytes, linux: kilobytes


def _default_process() -> Any:
    try:
        import os

        import psutil

        return psutil.Process(os.getpid())
    except Exception:                       # noqa: BLE001 - psutil is declared but keep it optional
        return None


def _default_usage(platform: str) -> Any:
    if platform.startswith("win"):
        return None                         # resource is POSIX-only
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF)
    except Exception:                       # noqa: BLE001
        return None


def rss_mb(platform: Optional[str] = None, process: Any = _AUTO,
           usage: Any = _AUTO) -> "tuple[float, Optional[float]]":
    """``(peak_MB, current_MB_or_None)`` for this process; parameters exist for test injection."""
    plat = platform or sys.platform
    if process is _AUTO:
        process = _default_process()
    if usage is _AUTO:
        usage = _default_usage(plat)

    info = _memory_info(process)
    cur: Optional[float] = None
    rss = getattr(info, "rss", None)
    if rss is not None:
        cur = float(rss) / _MB

    peak = _peak_from_rusage(usage, plat)
    if peak is None:
        peak = _peak_from_working_set(info, plat)
    if peak is None:
        peak = cur
    return (float(peak) if peak is not None else 0.0), cur
