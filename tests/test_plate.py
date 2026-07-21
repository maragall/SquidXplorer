"""Tests for squidmip._plate — the single source of truth for plate geometry + well ids.

Most of these pin invariants that previously had NO test at all. Before this module,
plate geometry lived in four places (_viewer._PLATE_DIMS/_row_letter/_plate_grid,
reader._plate_key, _output._row_sort_key/parse_well_id) and only parse_well_id was
directly covered — so the row-major ordering rule, which is the whole reason
reader._plate_key existed, could have been silently reverted by any refactor.
"""

from __future__ import annotations

import pytest

from squidmip._plate import (
    NotAWellPlateError,
    Plate,
    is_well_id,
    parse_well_id,
    row_index,
    row_letter,
    sort_key,
)


# --- R1: the row-major invariant (previously pinned by NOTHING) -----------------------

def test_row_major_order_single_before_double_letter():
    """A..Z must precede AA.., and columns sort numerically — NOT lexicographically.

    This is the IMA-189 fix. A naive natural sort puts "AA" before "B", so a 1536-well
    plate fills row A, then AA-AF, then B..Z — visually wrong. test_reader.py only ever
    asserted ["B2","B3"] (same row), which cannot detect a regression here.
    """
    regions = ["AA1", "B10", "A1", "B2", "AF48", "B3", "Z1"]
    assert sorted(regions, key=sort_key) == ["A1", "B2", "B3", "B10", "Z1", "AA1", "AF48"]


def test_non_well_ids_sort_after_wells_stably():
    regions = ["B2", "zzz", "A1"]
    ordered = sorted(regions, key=sort_key)
    assert ordered[:2] == ["A1", "B2"] and ordered[-1] == "zzz"


def test_row_letter_and_index_roundtrip():
    assert [row_letter(i) for i in (0, 25, 26, 31)] == ["A", "Z", "AA", "AF"]
    for i in (0, 1, 25, 26, 27, 31, 100):
        assert row_index(row_letter(i)) == i


# --- R2b: B2 — Squid's real flexible-region naming must be refused --------------------

def test_column_zero_refused_squid_flexible_region_naming():
    """Squid's flexible mode names regions R0, R1, ... (control/widgets.py:6584).

    The old parser accepted "R0" as row "R", column "0" and wrote it to
    plate.ome.zarr/R/0/ — a silent mislabel on Squid's DEFAULT naming path. No Squid
    plate format has a column 0; columns are 1-based everywhere.
    """
    for bad in ("R0", "A0", "AA0"):
        with pytest.raises(NotAWellPlateError):
            parse_well_id(bad)
    assert not is_well_id("R0")


def test_canonical_well_ids_still_accepted():
    assert parse_well_id("B2") == ("B", "2")
    assert parse_well_id("aa3") == ("AA", "3")     # upper-cased
    assert parse_well_id("B10") == ("B", "10")     # never zero-padded


def test_non_canonical_region_ids_refused():
    # "R2C3" and "region_1" were already refused; "0" and "region" have no row/col at all.
    for bad in ("R2C3", "region_1", "region", "0", "1A", "B2C", ""):
        with pytest.raises(NotAWellPlateError):
            parse_well_id(bad)


def test_not_a_well_plate_error_is_a_valueerror():
    """Existing `except ValueError` handlers must keep working."""
    assert issubclass(NotAWellPlateError, ValueError)


# --- plate construction ---------------------------------------------------------------

@pytest.mark.parametrize("fmt,rows,cols", [
    ("1536 well plate", 32, 48),
    ("384 well plate", 16, 24),
    ("96 well plate", 8, 12),
    ("24 well plate", 4, 6),
    ("12 well plate", 3, 4),
    ("6 well plate", 2, 3),
    ("glass slide", 1, 1),
    ("0", 1, 1),          # Squid's hardcoded no-plate special case (_def.py:1196)
])
def test_from_format_known_formats(fmt, rows, cols):
    p = Plate.from_format(fmt)
    assert (p.rows, p.cols) == (rows, cols)


def test_from_format_tolerant_fallback():
    """Preserves the viewer's old re.search(r"(\\d+)") behaviour for format-string drift."""
    for fmt in ("384wp", "384", "a 384 well plate"):
        assert Plate.from_format(fmt).rows == 16


def test_from_format_unknown_returns_none_not_raises():
    """The viewer falls back to a present-only grid on None; raising would break ingest."""
    assert Plate.from_format("banana") is None
    assert Plate.from_format(None) is None


def test_1536_labels_span_the_double_letter_boundary():
    p = Plate.from_format("1536 well plate")
    assert p.row_labels[0] == "A" and p.row_labels[25] == "Z"
    assert p.row_labels[26] == "AA" and p.row_labels[-1] == "AF"
    assert p.col_labels[0] == "1" and p.col_labels[-1] == "48"
    assert len(p.row_labels) == 32 and len(p.col_labels) == 48


def test_index_of_and_well_id_roundtrip():
    p = Plate.from_format("1536 well plate")
    assert p.index_of("A1") == (0, 0)
    assert p.index_of("AA3") == (26, 2)
    assert p.well_id(26, 2) == "AA3"


def test_well_outside_the_plate_is_refused():
    """Canonical but off-plate — must not be placed out of grid silently."""
    p = Plate.from_format("96 well plate")          # 8 rows x 12 cols
    assert p.contains("H12")
    for bad in ("B99", "Z1"):
        assert not p.contains(bad)
        with pytest.raises(NotAWellPlateError):
            p.index_of(bad)


def test_plate_is_frozen_and_labels_precomputed():
    """Labels are built once per instance — sort_key/index_of run once per region inside
    sorted(), so rebuilding a label list per call would make a 1536-well sort quadratic."""
    p = Plate.from_format("96 well plate")
    with pytest.raises(Exception):
        p.rows = 99                                  # frozen
    assert p.row_labels is p.row_labels              # same object, not rebuilt
    assert p._row_of["B"] == 1 and p._col_of["12"] == 11
