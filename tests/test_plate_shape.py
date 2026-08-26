"""Tests for wellplate-format inference from well ids."""

import pytest

from squidxplorer._plate import _row_letter
from squidxplorer._plate_shape import (
    GLASS_SLIDE,
    PlateShapeError,
    _row_index,
    infer_plate_format,
    normalize_plate_format,
    resolve_plate_format,
    well_span,
)


def _full_plate(rows, cols):
    """Every well id of an r x c plate: A1..{row}{col}."""
    return [f"{_row_letter(r)}{c}" for r in range(rows) for c in range(1, cols + 1)]


def test_the_span_from_the_plate_origin_snaps_to_the_smallest_containing_format():
    assert infer_plate_format(["A1", "A2", "B1", "B2"]) == "6 well plate"
    assert infer_plate_format(_full_plate(8, 12)) == "96 well plate"
    assert infer_plate_format(_full_plate(16, 24)) == "384 well plate"
    assert well_span(["C3", "C4"]) == (3, 4)
    assert infer_plate_format(["C3", "C4"]) == "12 well plate"
    assert infer_plate_format(["A1"]) == GLASS_SLIDE
    assert infer_plate_format(["manual0", "manual1", "manual2"]) == GLASS_SLIDE
    assert well_span(["manual0"]) is None
    assert infer_plate_format(["A1", "B2", "manual0"]) == GLASS_SLIDE
    with pytest.raises(PlateShapeError, match="exceeds every Squid format"):
        infer_plate_format(["A1", "BZ99"])


def test_an_override_beats_inference_and_a_bad_one_is_loud(monkeypatch):
    wells = ["A1", "A2", "B1", "B2"]              # would infer 6wp from span alone
    assert infer_plate_format(wells, override="96 well plate") == "96 well plate"
    assert infer_plate_format(wells, override=96) == "96 well plate"
    assert infer_plate_format(wells, override="1536wp") == "1536 well plate"
    assert infer_plate_format(["A1", "BZ99"], override="384") == "384 well plate"
    monkeypatch.setenv("SQUIDXPLORER_WELLPLATE_FORMAT", "384 well plate")
    assert infer_plate_format(wells) == "384 well plate"
    with pytest.raises(PlateShapeError, match="not a Squid wellplate format"):
        infer_plate_format(["A1"], override="7 well plate")


def test_normalize_and_resolve_prefer_declared_then_infer():
    assert normalize_plate_format("1536 well plate") == "1536 well plate"
    assert normalize_plate_format("glass slide") == GLASS_SLIDE
    assert normalize_plate_format("nonsense", strict=False) is None
    declared = {"wellplate_format": "1536 well plate", "regions": ["A1", "A2"]}
    assert resolve_plate_format(declared) == "1536 well plate"
    absent = {"wellplate_format": None, "regions": ["A1", "A2", "B1", "B2"]}
    assert resolve_plate_format(absent) == "6 well plate"
    assert resolve_plate_format(declared, override="96") == "96 well plate"


def test_row_letter_and_row_index_are_inverses_and_refuse_non_letters():
    for i in range(0, 703):              # 0..ZZ, well past a 1536wp's 32 rows (AF)
        assert _row_index(_row_letter(i)) == i
    for bad in ("A1", "manual0", "1"):
        with pytest.raises(KeyError):
            _row_index(bad)
