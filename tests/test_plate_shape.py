"""Tests for wellplate-format inference from well ids (IMA-219).

Deliberately NON-Qt: tests/test_viewer.py is behind ``pytest.importorskip("qtpy")`` and would
silently not run headless, so the inference contract is pinned here instead.
"""

import pytest

from squidmip._plate import _row_letter
from squidmip._plate_shape import (
    GLASS_SLIDE,
    PlateShapeError,
    _row_index,
    infer_plate_format,
    normalize_plate_format,
    plate_dims,
    resolve_plate_format,
    well_span,
)

# The row alphabet is IMPORTED, not restated. This file used to carry a third copy of it,
# justified by "_viewer's is behind a PyQt5 import" — a claim about a function that no longer
# exists, guarding a test whose whole subject is which row a letter names.


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




# --- ONE row alphabet, wherever it lives ---------------------------------------------------------
#
# `_row_letter` was two byte-identical copies (`_plate`, `_plate_overview`) and `_row_index` was
# two copies that were NOT identical: `_plate`'s refused a non-letter, `_plate_shape`'s did not,
# and `ord(ch) - 64` is a number for every character. Both collapsed on 2026-08-06. These pin the
# COUNT, not the module, so the two independent collapses of the same duplication agree.

def _defs(name: str) -> list:
    import pathlib

    import squidmip

    pkg = pathlib.Path(squidmip.__file__).parent
    return sorted(str(p.relative_to(pkg)) for p in pkg.rglob("*.py")
                  if f"def {name}(" in p.read_text())


def test_row_index_refuses_anything_that_is_not_a_row_letter():
    """The un-guarded copy answered these, silently, and the answers are in the message."""
    for bad, was in (("A1", 10), ("manual0", 4034554195), ("1", -16)):
        with pytest.raises(KeyError):
            _row_index(bad)
        assert isinstance(was, int)      # documents the wrong answer beside the refusal


def test_row_letter_and_row_index_are_inverses_over_every_plate_row():
    for i in range(0, 703):              # 0..ZZ, well past a 1536wp's 32 rows (AF)
        assert _row_index(_row_letter(i)) == i


@pytest.mark.parametrize("name", ["_row_letter", "_row_index"])
def test_the_row_alphabet_is_defined_exactly_once(name):
    """Structural: a second definition anywhere is the copy coming back.

    Text-level on purpose -- the deleted `_row_letter`s were byte-identical bodies, so only a
    grep would ever have caught a third being added, and nothing did for the second.
    """
    found = _defs(name)
    assert len(found) == 1, f"{name} is defined in {found}; there must be exactly one"
