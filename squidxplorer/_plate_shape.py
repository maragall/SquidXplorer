"""Infer the wellplate format from the region ids, with a manual override.

Span + snap: bound the observed rows/columns, then pick the smallest standard Squid
format that contains them. A declared format always wins over inference.
"""

from __future__ import annotations

import os
import re
from typing import Optional

# Stricter than _output.parse_well_id on purpose: "manual0" must not read as a well.
_WELL_RE = re.compile(r"^([A-Za-z]{1,2})(\d+)$")

# Squid's standard formats, smallest first so the first containing entry is the snap target.
GLASS_SLIDE = "glass slide"
_STANDARD_FORMATS = (
    (GLASS_SLIDE, 1, 1),
    ("6 well plate", 2, 3),
    ("12 well plate", 3, 4),
    ("24 well plate", 4, 6),
    ("96 well plate", 8, 12),
    ("384 well plate", 16, 24),
    ("1536 well plate", 32, 48),
)
_DIMS = {name: (rows, cols) for name, rows, cols in _STANDARD_FORMATS}

# Manual override for headless / CLI runs: SQUIDXPLORER_WELLPLATE_FORMAT="96 well plate" (or "96").
_OVERRIDE_ENV = "SQUIDXPLORER_WELLPLATE_FORMAT"


class PlateShapeError(ValueError):
    """The observed well ids fit no known Squid format (or an override names no known format)."""


def plate_dims(wellplate_format) -> Optional[tuple[int, int]]:
    """(rows, cols) for a Squid format string, or None when it names no standard format."""
    name = normalize_plate_format(wellplate_format, strict=False)
    return _DIMS.get(name) if name else None


def normalize_plate_format(wellplate_format, strict: bool = True) -> Optional[str]:
    """Canonicalize a user/yaml format to one of ``_STANDARD_FORMATS``' names."""
    if wellplate_format is None:
        return None
    s = str(wellplate_format).strip().lower()
    if not s:
        return None
    if "slide" in s:
        return GLASS_SLIDE
    m = re.search(r"(\d+)", s)
    name = f"{m.group(1)} well plate" if m else None
    if name in _DIMS:
        return name
    if strict:
        raise PlateShapeError(
            f"{wellplate_format!r} is not a Squid wellplate format; known formats are "
            f"{', '.join(_DIMS)}."
        )
    return None


def well_span(well_ids) -> Optional[tuple[int, int]]:
    """(n_rows, n_cols) bounding box of the well ids from A1, or None if any id is not a well."""
    max_row = max_col = 0
    for region in well_ids:
        m = _WELL_RE.match(str(region))
        if not m or int(m.group(2)) < 1:
            return None                     # freeform/tissue id — not a well plate at all
        max_row = max(max_row, _row_index(m.group(1)) + 1)
        max_col = max(max_col, int(m.group(2)))
    if not max_row:
        return None                         # empty id set
    return max_row, max_col


def infer_plate_format(well_ids, override=None) -> str:
    """Infer the Squid wellplate format from the observed well ids; override/env wins."""
    override = override if override is not None else os.environ.get(_OVERRIDE_ENV)
    forced = normalize_plate_format(override)
    if forced:
        return forced

    span = well_span(well_ids)
    if span is None:
        return GLASS_SLIDE                  # freeform / tissue regions: a slide, not a plate
    n_rows, n_cols = span
    for name, rows, cols in _STANDARD_FORMATS:
        if n_rows <= rows and n_cols <= cols:
            return name
    raise PlateShapeError(
        f"wells span {n_rows} rows x {n_cols} columns, which exceeds every Squid format "
        f"(largest is 1536 well plate at 32x48). Check the region ids, or force a format with "
        f"the override / {_OVERRIDE_ENV}."
    )


def resolve_plate_format(metadata, override=None) -> str:
    """The format the viewer/CLI should lay out: override -> declared -> inferred."""
    forced = normalize_plate_format(
        override if override is not None else os.environ.get(_OVERRIDE_ENV)
    )
    if forced:
        return forced
    declared = normalize_plate_format(metadata.get("wellplate_format"), strict=False)
    if declared:
        return declared
    return infer_plate_format(metadata.get("regions") or [])


def _row_index(letters: str) -> int:
    """"A"->0, "Z"->25, "AA"->26, ...; raises KeyError on anything that is not letters."""
    n = 0
    for ch in str(letters).upper():
        if not ch.isalpha():
            raise KeyError(letters)
        n = n * 26 + (ord(ch) - 64)
    return n - 1
