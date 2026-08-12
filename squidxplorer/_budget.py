"""How much memory a cache may use — derived from what is actually available, not hardcoded.

A fixed constant cannot be right on both a laptop and a 96 GB demo machine: too timid on one,
too blunt to protect the other. So the budget is a fraction of AVAILABLE memory (not total —
this process should not win the allocation and push the user's other windows into swap), with a
floor (below which a cache stops being a cache and starts thrashing) and a ceiling (bounded
memory is a project principle, not a consequence of small hardware).
"""

from __future__ import annotations

import os
from typing import Optional

#: Share of available memory one cache may hold. Several caches exist and each takes this
#: share independently, so the total is a small multiple of it.
DEFAULT_FRACTION = 0.10

#: Below ~64 MiB a fused-plane cache holds fewer than two 5731x4793 planes and evicts the one
#: it is about to need.
FLOOR_BYTES = 64 << 20

CEILING_BYTES = 4 << 30

#: Override, in MiB, for a human who knows something the measurement cannot.
ENV_VAR = "SQUIDXPLORER_CACHE_MB"


def available_bytes() -> Optional[int]:
    """Memory this process can take without pushing anything into swap, or None if unknowable."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:                       # noqa: BLE001 - psutil missing or refused
        return None


def cache_budget(fraction: float = DEFAULT_FRACTION,
                 floor: int = FLOOR_BYTES,
                 ceiling: int = CEILING_BYTES,
                 env: Optional[dict] = None,
                 available: Optional[int] = None) -> int:
    """Bytes one cache may hold on this machine, right now.

    Precedence: an explicit override beats a measurement, and a measurement beats the floor.
    """
    src = os.environ if env is None else env
    raw = str(src.get(ENV_VAR, "")).strip()
    if raw:
        try:
            override = int(float(raw) * (1 << 20))
        except ValueError:
            override = 0
        if override > 0:
            # not clamped to the ceiling or floor: an override is a human decision
            return override

    avail = available_bytes() if available is None else available
    if avail is None or avail <= 0:
        return floor
    return int(max(floor, min(ceiling, avail * float(fraction))))


def describe(budget: int) -> str:
    """One line for the log panel."""
    avail = available_bytes()
    mib = budget / (1 << 20)
    if avail is None:
        return f"cache budget {mib:.0f} MiB (memory not measurable here; using the floor)"
    return (f"cache budget {mib:.0f} MiB "
            f"({100.0 * budget / avail:.0f}% of {avail / (1 << 30):.1f} GiB available)")
