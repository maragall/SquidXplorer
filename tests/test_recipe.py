"""The transform-recipe + content-addressed result cache (squidxplorer._recipe)."""

from squidxplorer._address import Extent
from squidxplorer._recipe import LUT, OPERATOR, Recipe, RecipeChain, ResultCache
from squidxplorer._result import Result, Substance


def _result(region: str, channel: str = "DAPI") -> Result:
    """A minimal self-describing result. No pixels: these tests are about the key, not the data."""
    return Result(extent=Extent(region),
                  substance=Substance(channels=(channel,), z_depth=1, dtype="uint16",
                                      pixel_size_um=0.325))


def test_recipe_key_is_content_addressed():
    a = Recipe.operator("decon", iters=20)
    b = Recipe.operator("decon", iters=20)
    c = Recipe.operator("decon", iters=5)
    assert a.key() == b.key()
    assert a.key() != c.key()


def test_recipe_kinds():
    assert Recipe.operator("mip").kind == OPERATOR
    lut = Recipe.contrast({"c0": {"clim": (10, 200), "cmap": "green"}})
    assert lut.kind == LUT and lut.name == "contrast"


def test_chain_order_matters():
    stitch, decon = Recipe.operator("stitch"), Recipe.operator("decon")
    assert RecipeChain.of(stitch, decon).key() != RecipeChain.of(decon, stitch).key()


def test_chain_script_round_trips():
    ch = RecipeChain.of(Recipe.operator("stitch"), Recipe.operator("decon", iters=15))
    assert RecipeChain.from_script(ch.to_script()).key() == ch.key()


def test_a_chain_round_trips_through_its_own_label():
    for text in ("mip",
                 "flatfield + decon + mip",
                 "blob(min_area_px=80)",
                 "demo + blob(min_area_px=80, split_touching=False)",
                 "raw"):
        assert RecipeChain.parse(text).label() == text, text
    assert RecipeChain.parse("raw") == RecipeChain()
    assert RecipeChain.parse("   ") == RecipeChain()


def test_parse_reads_a_parameter_as_the_literal_it_is_written_as():
    params = RecipeChain.parse("decon(iters=15, gpu=False, sigma=2.0)").recipes[0].params
    assert params == {"iters": 15, "gpu": False, "sigma": 2.0}
    for value in params.values():
        assert not isinstance(value, str)


def test_parse_reads_an_unquoted_string_back_as_a_string():
    chain = RecipeChain.of(Recipe.operator("seg", segmenter="watershed"))
    assert chain.label() == "seg(segmenter=watershed)"
    assert RecipeChain.parse(chain.label()) == chain


def test_a_separator_inside_an_argument_list_is_not_a_separator():
    assert len(RecipeChain.parse("demo(scale=1e+5)").recipes) == 1
    assert len(RecipeChain.parse("crop(shape=(4, 4)) + mip").recipes) == 2


def test_a_malformed_chain_is_refused_naming_what_was_typed():
    import re

    import pytest

    for text, expected in (("mip + + decon", "empty step"),
                           ("mip(", "unbalanced '('"),
                           ("mip)", "unbalanced ')'"),
                           ("decon(15)", "not name=value")):
        with pytest.raises(ValueError, match=re.escape(expected)):
            RecipeChain.parse(text)
        with pytest.raises(ValueError, match=re.escape(repr(text))):
            RecipeChain.parse(text)


def test_cache_shares_by_content_not_window():
    ch = RecipeChain.of(Recipe.operator("decon"))
    cache = ResultCache()
    vol = _result("B7")
    cache.put("B7", ch, vol)
    assert cache.get("B7", ch) is vol
    assert cache.has("B7", ch)
    assert cache.get("B7", RecipeChain.of(Recipe.operator("mip"))) is None


def test_cache_version_separates_stale_from_fresh():
    ch = RecipeChain.of(Recipe.operator("decon"))
    cache = ResultCache()
    old, fresh = _result("B7"), _result("B7")
    cache.put("B7", ch, old, version=0)
    assert cache.get("B7", ch, version=0) is old
    assert cache.get("B7", ch, version=1) is None
    cache.put("B7", ch, fresh, version=1)
    assert cache.get("B7", ch, version=1) is fresh
    assert cache.get("B7", ch) is old


def test_cache_is_bounded_lru():
    ch = RecipeChain.of(Recipe.operator("decon"))
    cache = ResultCache(max_entries=2)
    cache.put("B7", ch, _result("B7"))
    cache.put("A1", ch, _result("A1"))
    cache.put("A2", ch, _result("A2"))
    assert len(cache) == 2
    assert not cache.has("B7", ch)
    assert cache.has("A1", ch) and cache.has("A2", ch)


def test_the_cache_refuses_a_bare_array():
    import pytest

    cache = ResultCache()
    with pytest.raises(TypeError, match="Result"):
        cache.put("B7", RecipeChain.of(Recipe.operator("mip")), [[1, 2], [3, 4]])


def test_the_lookup_is_scoped_to_its_region_its_operator_and_its_acquisition():
    from squidxplorer import _recipe

    _recipe.RESULTS.clear()
    mine = _result("B7", "DAPI")
    _recipe.cache_operator_result("mip", mine, version="/acq/one")
    _recipe.cache_operator_result("stitch", _result("B7", "GFP"), version="/acq/one")
    _recipe.cache_operator_result("mip", _result("B8", "DAPI"), version="/acq/one")
    _recipe.cache_operator_result("mip", _result("B7", "DAPI"), version="/acq/TWO")

    got = dict(_recipe.cached_operator_results("B7", "/acq/one"))
    assert set(got) == {"mip", "stitch"}, "another region's or acquisition's entry leaked in"
    assert got["mip"] is mine
    assert _recipe.cached_operator_results("B7", "/acq/three") == []
    assert _recipe.cached_operator_results("ZZ99", "/acq/one") == []


def test_a_tab_scoped_run_is_a_different_entry_from_the_plate_wide_one():
    from squidxplorer import _recipe

    _recipe.RESULTS.clear()
    whole = _result("B7", "DAPI")
    subset = _result("B7", "DAPI")
    _recipe.cache_operator_result("mip", whole, version="/acq/one")
    _recipe.cache_operator_result("mip@preview:1", subset, version="/acq/one")

    got = dict(_recipe.cached_operator_results("B7", "/acq/one"))
    assert got["mip"] is whole and got["mip@preview:1"] is subset
