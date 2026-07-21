"""Infer the wellplate format from the region ids alone, with a manual override (IMA-219).

Two real acquisitions carry no ``sample.wellplate_format`` at all, so the viewer has nothing to
lay a plate out with. This module recovers a format from the only evidence that is always
present — the well ids parsed out of the filenames::

    infer_plate_format(["A1", "A2", "B1", "B2"])   -> "6 well plate"     (2x2 fits 2x3)
    infer_plate_format(["A1", ..., "H12"])         -> "96 well plate"
    infer_plate_format(["manual0", "manual1"])     -> "glass slide"      (not wells at all)
    infer_plate_format(["A1", "AZ99"])             -> PlateShapeError    (fits nothing)

The rule is SPAN + SNAP: take the bounding box of the observed row letters and column numbers,
then pick the SMALLEST standard Squid format that contains it. Snapping matters — a 2x2 box is
not a Squid plate, and rendering one as a literal 2x2 grid would put every well at the wrong
physical position. Squid's standard geometries are vendored in ``_STANDARD_FORMATS`` below
(they mirror Squid's own ``sample_formats.csv``, which lives outside this repo).

Span is a LOWER BOUND, not a measurement: four wells at the top-left corner of a 96wp span the
same 2x2 box as a (hypothetical) 6wp acquisition, so a sparse plate infers too small. The
physical fix is to match the per-well pitch from ``coordinates.csv`` against each format's well
spacing (IMA-219 D5), which is a separate, larger change; until then the manual override is the
remedy, and a declared format always wins over inference (D1).
"""

from __future__ import annotations

import os
import re
from typing import Optional

# A WELL id, stricter than _output.parse_well_id on purpose: that parser only asserts
# <letters><digits>, which "manual0" satisfies (row "MANUAL", column 0) — exactly the freeform
# tissue id this module must recognise as NOT a well. A plate row is 1-2 letters (max is AF on a
# 1536wp) and a plate column is 1-based, so both bounds are real, not cosmetic.
_WELL_RE = re.compile(r"^([A-Za-z]{1,2})(\d+)$")

# Squid's standard sample formats, vendored (well count -> rows x cols), SMALLEST FIRST so the
# first containing entry is the snap target. "glass slide" is the degenerate 1x1 sample and is
# also what a non-wellplate (freeform / tissue) acquisition reports as.
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

# Manual override for headless / CLI runs: SQUIDMIP_WELLPLATE_FORMAT="96 well plate" (or "96").
_OVERRIDE_ENV = "SQUIDMIP_WELLPLATE_FORMAT"


class PlateShapeError(ValueError):
    """The observed well ids fit no known Squid format (or an override names no known format)."""


def plate_dims(wellplate_format) -> Optional[tuple[int, int]]:
    """(rows, cols) for a Squid format string, or None when it names no standard format."""
    name = normalize_plate_format(wellplate_format, strict=False)
    return _DIMS.get(name) if name else None


def normalize_plate_format(wellplate_format, strict: bool = True) -> Optional[str]:
    """Canonicalize a user/yaml format to one of ``_STANDARD_FORMATS``' names.

    Accepts what people actually type or what Squid writes: ``96``, ``"96"``, ``"96wp"``,
    ``"96 well plate"``, ``"glass slide"``, ``"slide"``. Returns None (strict=False) or raises
    PlateShapeError (strict=True) for anything else, so a typo can't silently lay out a plate.
    """
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
    """(n_rows, n_cols) bounding box of the well ids, 1-based from A1 — or None if any id is not
    a well id. Rows/cols are counted from the plate ORIGIN, not from the first observed well:
    wells C3/C4 span 3x4, because a plate always starts at A1."""
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
    """Infer the Squid wellplate format from the observed well ids. See the module docstring.

    *override*, when given, short-circuits inference entirely (after normalization) — that is the
    manual path for a sparse acquisition the span rule under-reads. It falls back to the
    ``SQUIDMIP_WELLPLATE_FORMAT`` environment variable so headless/CLI runs have an override too.

    Returns a format name from ``_STANDARD_FORMATS``; ``"glass slide"`` when the regions are not
    well ids (freeform/tissue names like ``manual0`` — reported, never raised, so a non-plate
    acquisition still opens).

    Raises
    ------
    PlateShapeError
        If the wells exceed every known format (e.g. a row past AF or a column past 48), or if
        *override* names no known format. Refusing beats laying out a plate we cannot draw.
    """
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
    """The format the viewer/CLI should lay out: declared -> override -> inferred.

    A declared ``wellplate_format`` is authoritative (IMA-219 D1) — inference is a FALLBACK, run
    only when the acquisition carries no format. An explicit *override* beats both.
    """
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
    """"A"->0, "Z"->25, "AA"->26, ... — the inverse of the viewer's ``_row_letter``."""
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1
