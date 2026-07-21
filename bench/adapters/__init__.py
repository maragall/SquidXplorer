"""Stitcher adapters.

Registry order is the order rows appear in the report. tilefusion is first because it
is the incumbent -- the thing the other tools are being measured against.
"""

from __future__ import annotations

from bench.adapters.ashlar import AshlarAdapter
from bench.adapters.base import (
    AdapterOutcome,
    StitcherAdapter,
    StitchRequest,
    read_positions_json,
)
from bench.adapters.tilefusion import TileFusionAdapter

#: Adapters shipped today. IMA-233 deliberately starts narrow: both of these are
#: Python with translation-only position models, so both are fully measurable.
#: MCmicro, BigStitcher and PetaKit5D are deferred -- see docs/ima-233-eng-review.md
#: ("Start narrow"). BigStitcher's non-rigid model makes the quality metric undefined,
#: and PetaKit5D needs an unverified MATLAB/MCR licence.
BUILTIN_ADAPTERS: tuple[type[StitcherAdapter], ...] = (
    TileFusionAdapter,
    AshlarAdapter,
)


def build_adapters(names: list[str] | None = None) -> list[StitcherAdapter]:
    by_name = {cls.name: cls for cls in BUILTIN_ADAPTERS}
    if names is None:
        return [cls() for cls in BUILTIN_ADAPTERS]
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown adapter(s): {unknown}; known: {sorted(by_name)}")
    return [by_name[n]() for n in names]


__all__ = [
    "AdapterOutcome",
    "AshlarAdapter",
    "BUILTIN_ADAPTERS",
    "StitcherAdapter",
    "StitchRequest",
    "TileFusionAdapter",
    "build_adapters",
    "read_positions_json",
]
