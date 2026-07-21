"""Tests for wellplate-format inference from well ids (IMA-219).

Deliberately NON-Qt: tests/test_viewer.py is behind ``pytest.importorskip("PyQt5")`` and would
silently not run headless, so the inference contract is pinned here instead.
"""

import pytest

from squidmip._plate_shape import (
    GLASS_SLIDE,
    PlateShapeError,
    infer_plate_format,
    normalize_plate_format,
    plate_dims,
    resolve_plate_format,
    well_span,
)


def _row_letter(i: int) -> str:
    """0->A, 25->Z, 26->AA (a local copy — _viewer's is behind a PyQt5 import)."""
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _full_plate(rows, cols):
    """Every well id of an r x c plate: A1..{row}{col}."""
    return [f"{_row_letter(r)}{c}" for r in range(rows) for c in range(1, cols + 1)]


def test_2x2_snaps_to_smallest_containing_format():
    # ~/Downloads/synthetic_2x2_wellplate: A1/A2/B1/B2. A literal 2x2 is not a Squid format, so
    # the 2x2 box snaps UP to the smallest one that contains it (6wp = 2x3).
    assert infer_plate_format(["A1", "A2", "B1", "B2"]) == "6 well plate"


def test_full_96_plate():
    assert infer_plate_format(_full_plate(8, 12)) == "96 well plate"


def test_full_384_plate():
    assert infer_plate_format(_full_plate(16, 24)) == "384 well plate"


def test_span_is_measured_from_the_plate_origin():
    # A plate always starts at A1: C3/C4 span 3 rows x 4 cols (-> 12wp), not 1x2.
    assert well_span(["C3", "C4"]) == (3, 4)
    assert infer_plate_format(["C3", "C4"]) == "12 well plate"


def test_one_well_is_a_glass_slide():
    # The degenerate 1x1 sample — smallest containing format for a single A1.
    assert infer_plate_format(["A1"]) == GLASS_SLIDE


def test_exceeding_every_format_raises():
    # A row past AF (32) / a column past 48 fits nothing; refuse rather than draw a wrong plate.
    with pytest.raises(PlateShapeError, match="exceeds every Squid format"):
        infer_plate_format(["A1", "BZ99"])


def test_freeform_regions_report_a_slide_not_a_crash():
    # Tissue / manual acquisitions: not wells at all -> non-wellplate layout, never an exception.
    assert infer_plate_format(["manual0", "manual1", "manual2"]) == GLASS_SLIDE
    assert well_span(["manual0"]) is None
    # A single freeform id mixed into real wells still means "not a well plate".
    assert infer_plate_format(["A1", "B2", "manual0"]) == GLASS_SLIDE


def test_manual_override_beats_inference():
    wells = ["A1", "A2", "B1", "B2"]              # would infer 6wp from span alone
    assert infer_plate_format(wells, override="96 well plate") == "96 well plate"
    assert infer_plate_format(wells, override=96) == "96 well plate"
    assert infer_plate_format(wells, override="1536wp") == "1536 well plate"
    # The override also rescues a set that fits no format at all.
    assert infer_plate_format(["A1", "BZ99"], override="384") == "384 well plate"


def test_manual_override_via_environment(monkeypatch):
    monkeypatch.setenv("SQUIDMIP_WELLPLATE_FORMAT", "384 well plate")
    assert infer_plate_format(["A1", "A2", "B1", "B2"]) == "384 well plate"


def test_bad_override_is_loud():
    with pytest.raises(PlateShapeError, match="not a Squid wellplate format"):
        infer_plate_format(["A1"], override="7 well plate")


def test_normalize_and_dims():
    assert normalize_plate_format("1536 well plate") == "1536 well plate"
    assert normalize_plate_format("glass slide") == GLASS_SLIDE
    assert normalize_plate_format("nonsense", strict=False) is None
    assert plate_dims("96") == (8, 12)
    assert plate_dims(GLASS_SLIDE) == (1, 1)
    assert plate_dims("4 well plate") is None      # not a Squid standard format


def test_resolve_prefers_declared_then_infers():
    declared = {"wellplate_format": "1536 well plate", "regions": ["A1", "A2"]}
    assert resolve_plate_format(declared) == "1536 well plate"      # D1: declared is authoritative
    absent = {"wellplate_format": None, "regions": ["A1", "A2", "B1", "B2"]}
    assert resolve_plate_format(absent) == "6 well plate"           # fallback: inference
    assert resolve_plate_format(declared, override="96") == "96 well plate"   # override beats both
