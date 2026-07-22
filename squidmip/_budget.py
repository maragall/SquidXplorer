"""How much memory a cache may use — MEASURED, not hardcoded.

Julio: "the guard for memory is interesting because we have to maximize performance and that
doesn't happen by hardcoding guards." And: "There are so many variations of desktop
configuration."

Both are right, and the numbers make it concrete. The caches were fixed at 256 MiB each. On the
development machine that is ~5% of available RAM; on the CSO's 96 GB demo machine it is ~0.5%.
The same constant is simultaneously too timid to use the machine and too blunt to protect a small
one. A constant cannot be right on both, because it encodes an assumption about a machine it has
never seen.

So the budget is DERIVED at runtime from what is actually available.

WHY *AVAILABLE* AND NOT *TOTAL*
-------------------------------
``psutil.virtual_memory().total`` says what the machine has; ``.available`` says what this process
can take without pushing something else into swap. A viewer that sizes itself off total RAM on a
laptop already running a browser and an IDE will win the allocation and lose the machine — and
swapping a 550 MB fused plane is far worse than re-reading it. The scientist's other windows are
not our memory to spend.

WHY A FRACTION, A FLOOR AND A CEILING
-------------------------------------
* **Fraction** — scales with the machine, which is the whole point.
* **Floor** — below a certain size a cache stops being a cache: it evicts the plane it is about
  to be asked for again, and every z step pays a full re-fuse. Better to keep a small cache and be
  honest about it than to thrash.
* **Ceiling** — an unbounded fraction on a 512 GB workstation would let one cache hold more fused
  planes than any user will revisit, and this project's first principle is bounded memory. The
  ceiling is what keeps "derived" from meaning "unlimited".

The environment variable exists because a measurement cannot know that the user is about to open
something else. It is an override for a human, not a default.
"""

from __future__ import annotations

import os
from typing import Optional

#: Share of AVAILABLE memory one cache may hold. Deliberately modest: several caches exist (the
#: fused-plane pyramid cache, the plate preview), and each takes this share independently, so the
#: total is a small multiple of it rather than this number.
DEFAULT_FRACTION = 0.10

#: Never smaller than this. Below ~64 MiB a fused-plane cache holds fewer than two 5731x4793
#: planes (54.9 MB each) and evicts the one it is about to need.
FLOOR_BYTES = 64 << 20

#: Never larger than this, however big the machine. Bounded memory is a project principle, not a
#: consequence of small hardware.
CEILING_BYTES = 4 << 30

#: Override, in MiB. For a human who knows something the measurement cannot.
ENV_VAR = "SQUIDMIP_CACHE_MB"


def available_bytes() -> Optional[int]:
    """Memory this process can take without pushing anything into swap, or None if unknowable.

    None rather than a guess: a caller that cannot measure should fall back to a stated default,
    not to a number invented here that would look measured.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:                       # noqa: BLE001 - psutil missing or refused; say so by None
        return None


def cache_budget(fraction: float = DEFAULT_FRACTION,
                 floor: int = FLOOR_BYTES,
                 ceiling: int = CEILING_BYTES,
                 env: Optional[dict] = None,
                 available: Optional[int] = None) -> int:
    """Bytes one cache may hold on THIS machine, right now.

    Precedence, and it matters: an explicit override beats a measurement, and a measurement beats
    a default. That is the same rule the contrast model uses (user latch > owner's window >
    computed), and for the same reason — the person who typed a number knows something the
    process does not.
    """
    src = os.environ if env is None else env
    raw = str(src.get(ENV_VAR, "")).strip()
    if raw:
        try:
            override = int(float(raw) * (1 << 20))
        except ValueError:
            override = 0
        if override > 0:
            # NOT clamped to the ceiling. An override is a human decision about their own machine;
            # silently capping it would make the control a lie. The floor is not applied either.
            return override

    avail = available_bytes() if available is None else available
    if avail is None or avail <= 0:
        return floor
    return int(max(floor, min(ceiling, avail * float(fraction))))


def describe(budget: int) -> str:
    """One line a human can read, for the log panel. A budget nobody can see is a magic number."""
    avail = available_bytes()
    mib = budget / (1 << 20)
    if avail is None:
        return f"cache budget {mib:.0f} MiB (memory not measurable here; using the floor)"
    return (f"cache budget {mib:.0f} MiB "
            f"({100.0 * budget / avail:.0f}% of {avail / (1 << 30):.1f} GiB available)")
