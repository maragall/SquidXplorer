"""Fixed-width integer ids: the flat-cache scope and the logger's numeric id."""

from squidxplorer._plate import (
    cache_scope, decode_code, roi_code, well_code,
)


def test_codes_are_fixed_width_zero_based_and_round_trip_into_the_cache_scope():
    assert well_code("A1") == 0
    assert well_code("A2") == 10_000
    assert well_code("B1") == 1_000_000
    assert well_code("C18") == 2 * 1_000_000 + 17 * 10_000
    code = roi_code("C18", 3)
    assert code == well_code("C18") + 3
    assert decode_code(code) == (2, 17, 3)          # (row, col, roi), 0-based
    assert cache_scope("C18") == str(well_code("C18"))
    assert cache_scope("C18", 3) == str(code)


def test_freeform_region_has_no_code():
    assert well_code("manual0") is None
    assert cache_scope("manual0") == "manual0"      # falls back to the raw region key
