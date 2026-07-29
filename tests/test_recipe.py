"""The transform-recipe + content-addressed result cache (squidmip._recipe).

The window tree is navigation; results live in a flat cache keyed by (scope, op-chain). A recipe is
the copy/paste unit (an operator or a LUT). These tests pin the three properties the design relies
on: content-addressing (same transform -> same key), order sensitivity of a chain, and the cache's
sharing + LRU bound.

Task 2 (2026-07-29) changed what the cache STORES: a :class:`squidmip._result.Result` rather than a
bare array. The three cache tests below therefore store results, and what they pin is unchanged --
sharing, versioning and the LRU bound are properties of the KEY, and the key did not move. What a
result is and why it has to describe itself lives in ``tests/test_result.py``.
"""

from squidmip._address import Extent
from squidmip._recipe import LUT, OPERATOR, Recipe, RecipeChain, ResultCache
from squidmip._result import Result, Substance


def _result(region: str, channel: str = "DAPI") -> Result:
    """A minimal self-describing result. No pixels: these tests are about the key, not the data."""
    return Result(extent=Extent(region),
                  substance=Substance(channels=(channel,), z_depth=1, dtype="uint16",
                                      pixel_size_um=0.325))


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
    vol = _result("B7")
    cache.put("B7", ch, vol)
    # A different window asking for the SAME (scope, chain) gets the same entry -- no window id in key.
    assert cache.get("B7", ch) is vol
    assert cache.has("B7", ch)
    # A different chain on the same scope is a miss.
    assert cache.get("B7", RecipeChain.of(Recipe.operator("mip"))) is None


def test_cache_version_separates_stale_from_fresh():
    # Time baked into the key: a new acquisition version is a MISS, so a live scope recomputes
    # instead of serving a stale result, with no explicit invalidation pass. Static folders stay v0.
    ch = RecipeChain.of(Recipe.operator("decon3d"))
    cache = ResultCache()
    old, fresh = _result("B7"), _result("B7")
    cache.put("B7", ch, old, version=0)
    assert cache.get("B7", ch, version=0) is old
    assert cache.get("B7", ch, version=1) is None      # new frames -> miss -> recompute
    cache.put("B7", ch, fresh, version=1)
    assert cache.get("B7", ch, version=1) is fresh
    assert cache.get("B7", ch) is old                  # default version 0 still resolves


def test_cache_is_bounded_lru():
    ch = RecipeChain.of(Recipe.operator("decon3d"))
    cache = ResultCache(max_entries=2)
    cache.put("B7", ch, _result("B7"))
    cache.put("A1", ch, _result("A1"))
    cache.put("A2", ch, _result("A2"))    # evicts the least-recently-used (B7)
    assert len(cache) == 2
    assert not cache.has("B7", ch)
    assert cache.has("A1", ch) and cache.has("A2", ch)


def test_the_cache_refuses_a_bare_array():
    """The store held bare arrays, and a bare array cannot say what it is. Accepting both kinds is
    the interim state: the next reader would have to branch on which kind it got, and a plate built
    from a mix could only find out what a cell is by comparing it with its neighbours.

    MUTATION: relax ``put`` to accept anything and this goes red."""
    import pytest

    cache = ResultCache()
    with pytest.raises(TypeError, match="Result"):
        cache.put("B7", RecipeChain.of(Recipe.operator("mip")), [[1, 2], [3, 4]])
