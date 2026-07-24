"""The transform-recipe + content-addressed result cache (squidmip._recipe).

The window tree is navigation; results live in a flat cache keyed by (scope, op-chain). A recipe is
the copy/paste unit (an operator or a LUT). These tests pin the three properties the design relies
on: content-addressing (same transform -> same key), order sensitivity of a chain, and the cache's
sharing + LRU bound.
"""

from squidmip._recipe import LUT, OPERATOR, Recipe, RecipeChain, ResultCache


def test_recipe_key_is_content_addressed():
    a = Recipe.operator("decon", iters=20)
    b = Recipe.operator("decon", iters=20)
    c = Recipe.operator("decon", iters=5)
    assert a.key() == b.key()          # same transform -> same key
    assert a.key() != c.key()          # different params -> different key


def test_recipe_kinds():
    assert Recipe.operator("mip").kind == OPERATOR
    lut = Recipe.contrast({"c0": {"clim": (10, 200), "cmap": "green"}})
    assert lut.kind == LUT and lut.name == "contrast"


def test_chain_order_matters():
    stitch, decon = Recipe.operator("stitch"), Recipe.operator("decon3d")
    assert RecipeChain.of(stitch, decon).key() != RecipeChain.of(decon, stitch).key()


def test_chain_script_round_trips():
    ch = RecipeChain.of(Recipe.operator("stitch"), Recipe.operator("decon3d", iters=15))
    assert RecipeChain.from_script(ch.to_script()).key() == ch.key()


def test_cache_shares_by_content_not_window():
    ch = RecipeChain.of(Recipe.operator("decon3d"))
    cache = ResultCache()
    cache.put("B7", ch, "VOL")
    # A different window asking for the SAME (scope, chain) gets the same entry -- no window id in key.
    assert cache.get("B7", ch) == "VOL"
    assert cache.has("B7", ch)
    # A different chain on the same scope is a miss.
    assert cache.get("B7", RecipeChain.of(Recipe.operator("mip"))) is None


def test_cache_is_bounded_lru():
    ch = RecipeChain.of(Recipe.operator("decon3d"))
    cache = ResultCache(max_entries=2)
    cache.put("B7", ch, "x")
    cache.put("A1", ch, "y")
    cache.put("A2", ch, "z")          # evicts the least-recently-used (B7)
    assert len(cache) == 2
    assert not cache.has("B7", ch)
    assert cache.has("A1", ch) and cache.has("A2", ch)
