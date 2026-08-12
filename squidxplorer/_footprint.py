"""How much memory this process has used, and the most it ever used, on every platform we ship to.

The peak is the interesting half and it cannot be recovered after the fact, so it is taken from the
OPERATING SYSTEM's high-water mark rather than by sampling: a 5 second poll can miss a whole fuse.
Both major platforms keep such a mark, under different names and in different units, and reading
only one of them is how the footprint line came to print ``peak 0 MB`` for the life of the product
on Windows -- which is the platform v1 ships to.

    POSIX    ``resource.getrusage(RUSAGE_SELF).ru_maxrss``   bytes on darwin, kilobytes on linux
    Windows  ``PeakWorkingSetSize``, via ``psutil`` as ``memory_info().peak_wset``, bytes

Lives in its own module rather than in ``_viewer`` because it is process arithmetic with no Qt in
it, and because the platform branch is only trustworthy if it can be tested from the OTHER platform.
Both sources and the platform string are therefore parameters with real defaults, and
``tests/test_footprint.py`` exercises the Windows branch on macOS.
"""
from __future__ import annotations

import sys
from typing import Any, Optional

_MB = 1024 * 1024

#: "you did not pass this, go and find the real one". Distinct from ``None``, which is a caller
#: saying "there is NO such source" -- the case the tests need in order to prove the fallbacks.
_AUTO = object()


def _memory_info(process: Any) -> Any:
    """``process.memory_info()``, or None. Never raises: the caller can be an excepthook running
    while the app is already dying, and it must not be the thing that raises there."""
    if process is None:
        return None
    try:
        return process.memory_info()
    except Exception:                       # noqa: BLE001 - access denied, dead process, no psutil
        return None


def _peak_from_working_set(info: Any, platform: str) -> Optional[float]:
    """The Windows peak working set in MB, or None where there is no such number.

    Gated on the platform as well as on the attribute so the branch is explicit and injectable:
    ``peak_wset`` simply does not exist in a macOS or Linux psutil build, and a silent
    ``getattr(..., None)`` would leave the Windows-only path untested on the machines we develop on.

    A ``peak_wset`` of 0 is read as NO ANSWER, not as zero bytes. A live process cannot have peaked
    at nothing, so 0 here means the field was unavailable, and passing it on as a measurement is the
    exact lie this module exists to stop.
    """
    if not platform.startswith("win"):
        return None
    value = getattr(info, "peak_wset", None)
    if not value:
        return None
    return float(value) / _MB


def _peak_from_rusage(usage: Any, platform: str) -> Optional[float]:
    """The POSIX high-water mark in MB, or None on Windows, where ``resource`` does not exist."""
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
        return None                         # POSIX-only module; importing it here is the old bug
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF)
    except Exception:                       # noqa: BLE001
        return None


def rss_mb(platform: Optional[str] = None, process: Any = _AUTO,
           usage: Any = _AUTO) -> "tuple[float, Optional[float]]":
    """``(peak_MB, current_MB_or_None)`` for this process.

    *platform*, *process* and *usage* exist to be injected by the tests; unpassed, each is found for
    real. Passing an explicit ``None`` means "there is no such source", which is how the tests reach
    the fallbacks; that is why the defaults are a sentinel and not ``None``.

    The peak resolves in this order: the OS high-water mark for the platform, then the CURRENT rss
    as a floor, then 0.0. The middle step matters. When no high-water mark is reachable, the current
    rss is a true lower bound on the peak, so it is a weaker statement rather than a false one; 0.0
    is reserved for "nothing at all could be measured", which is the only case where it is true.
    """
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
