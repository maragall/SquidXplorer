"""Fixed-width integer ids: the flat-cache scope and the logger's numeric id."""

from squidxplorer._plate import (
    cache_scope, decode_code, format_code, roi_code, well_code,
)


def test_well_code_is_fixed_width_and_zero_based():
    assert well_code("A1") == 0
    assert well_code("A2") == 10_000
    assert well_code("B1") == 1_000_000
    assert well_code("C18") == 2 * 1_000_000 + 17 * 10_000


def test_roi_code_and_decode_round_trip():
    code = roi_code("C18", 3)
    assert code == well_code("C18") + 3
    assert decode_code(code) == (2, 17, 3)          # (row, col, roi), 0-based
    assert format_code(code) == "02 17 0003"        # Row Column ROI


def test_freeform_region_has_no_code():
    assert well_code("manual0") is None
    assert cache_scope("manual0") == "manual0"      # falls back to the raw region key


def test_cache_scope_is_the_integer_id():
    assert cache_scope("C18") == str(well_code("C18"))
    assert cache_scope("C18", 3) == str(roi_code("C18", 3))
