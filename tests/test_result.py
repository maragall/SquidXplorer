"""A cached result says what it is, so nothing ever compares two of them (see
``squidxplorer/_result.py``). Pins: round-trip through JSON, two runs with different channel sets
coexisting under one plate, and the absence of any comparison between two results.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib

import numpy as np
import pytest

from squidxplorer._address import Address, Extent
from squidxplorer._recipe import Recipe, RecipeChain, ResultCache
from squidxplorer._result import (
    Result,
    Substance,
    composite_channels,
    composite_plate,
)


def _pixels(n_channels: int, value: int = 7) -> np.ndarray:
    """A channel-major ``(C, Y, X)`` stack whose planes are distinguishable by their value."""
    return np.stack([np.full((4, 4), value + i, dtype=np.uint16) for i in range(n_channels)])


def test_a_result_round_trips_its_extent_and_its_substance_through_json():
    extent = Extent("A1", fovs=(0, 3), z_levels=range(0, 12), time_points=range(0, 3),
                    channels=("DAPI", "GFP"), bbox_um=(10.0, 20.0, 110.0, 220.0))
    substance = Substance(channels=("GFP", "DAPI"), z_depth=1, dtype="uint16",
                          pixel_size_um=0.325)
    result = Result(extent=extent, substance=substance, data=_pixels(2))

    back = Result.from_dict(json.loads(json.dumps(result.to_dict())))
    assert back.extent == extent
    assert back.substance == substance
    assert back.extent.key() == extent.key()
    assert back.data is None, "the declaration must not smuggle pixels through JSON"


def test_a_substance_covers_the_channel_set_the_z_depth_the_dtype_and_the_pixel_size():
    s = Substance(channels=("DAPI", "GFP"), z_depth=12, dtype=np.uint16, pixel_size_um=0.325)
    assert s.channels == ("DAPI", "GFP")
    assert s.z_depth == 12
    assert s.dtype == "uint16", "a dtype has to be a stable string to survive JSON"
    assert s.pixel_size_um == 0.325

    r = Result(extent=Extent("A1"), substance=s)
    assert (r.channels, r.z_depth, r.dtype, r.pixel_size_um) == \
           (("DAPI", "GFP"), 12, "uint16", 0.325)
    assert r.region_id == "A1"


def test_a_substance_is_never_vague_where_an_extent_is_allowed_to_be():
    assert Extent("A1").channels is None                # a request may say "all of it"
    assert Extent("A1", channels=()).channels is None   # and the empty set means the same

    with pytest.raises(ValueError, match="at least one channel"):
        Substance(channels=(), z_depth=1, dtype="uint16", pixel_size_um=0.325)


def test_a_result_refuses_a_scale_it_does_not_have():
    for bad in (0.0, -0.325):
        with pytest.raises(ValueError, match="pixel_size_um"):
            Substance(channels=("DAPI",), z_depth=1, dtype="uint16", pixel_size_um=bad)
    with pytest.raises(ValueError, match="z_depth"):
        Substance(channels=("DAPI",), z_depth=0, dtype="uint16", pixel_size_um=0.325)


def test_the_channel_order_of_a_substance_is_the_PIXEL_order_and_not_sorted():
    assert Extent("A1", channels=("GFP", "DAPI")).channels == ("DAPI", "GFP")   # sorted: a key

    s = Substance(channels=("GFP", "DAPI"), z_depth=1, dtype="uint16", pixel_size_um=0.325)
    assert s.channels == ("GFP", "DAPI"), "a substance sorted its channels: planes now misnamed"

    r = Result(extent=Extent("A1"), substance=s, data=_pixels(2))
    assert int(r.plane("GFP")[0, 0]) == 7
    assert int(r.plane("DAPI")[0, 0]) == 8


def test_a_plane_is_found_by_NAME_and_a_miss_says_what_the_result_DOES_carry():
    r = Result(extent=Extent("A1"),
               substance=Substance(channels=("DAPI",), z_depth=1, dtype="uint16",
                                   pixel_size_um=0.325),
               data=_pixels(1))
    with pytest.raises(KeyError) as excinfo:
        r.plane("GFP")
    assert "DAPI" in str(excinfo.value)


def test_of_takes_the_dtype_from_the_pixels_and_refuses_a_channel_count_that_disagrees():
    r = Result.of(Extent("A1"), _pixels(2), channels=("DAPI", "GFP"), z_depth=1,
                  pixel_size_um=0.325)
    assert r.dtype == "uint16"

    with pytest.raises(ValueError, match="refusing to guess"):
        Result.of(Extent("A1"), _pixels(2), channels=("DAPI",), z_depth=1, pixel_size_um=0.325)


def _run(region: str, channels, chain: RecipeChain, cache: ResultCache) -> Result:
    """One run's result for one cell, put in the cache under its own chain."""
    r = Result.of(Extent(region), _pixels(len(channels)), channels=channels, z_depth=1,
                  pixel_size_um=0.325)
    cache.put(region, chain, r)
    return r


def test_two_results_with_different_channel_sets_coexist_and_each_reports_its_own():
    cache = ResultCache()
    two_colour = RecipeChain.of(Recipe.operator("mip"))
    one_colour = RecipeChain.of(Recipe.operator("mip"), Recipe.operator("decon", sigma=2.0))

    _run("A1", ("DAPI", "GFP"), two_colour, cache)
    _run("B2", ("DAPI",), one_colour, cache)

    assert cache.get("A1", two_colour).channels == ("DAPI", "GFP")
    assert cache.get("B2", one_colour).channels == ("DAPI",)
    assert len(cache) == 2, "one cell's declaration overwrote the other's"


def test_two_runs_over_the_SAME_cell_are_two_entries_each_with_its_own_declaration():
    cache = ResultCache()
    plain = RecipeChain.of(Recipe.operator("mip"))
    deconned = RecipeChain.of(Recipe.operator("mip"), Recipe.operator("decon", sigma=2.0))

    _run("A1", ("DAPI", "GFP"), plain, cache)
    _run("A1", ("DAPI",), deconned, cache)

    assert cache.get("A1", plain).channels == ("DAPI", "GFP")
    assert cache.get("A1", deconned).channels == ("DAPI",)


def test_a_plate_built_from_BOTH_renders_BOTH():
    """Union of channels shown; a cell missing a channel is absent from it, not drawn black."""
    cache = ResultCache()
    two_colour = RecipeChain.of(Recipe.operator("mip"))
    one_colour = RecipeChain.of(Recipe.operator("mip"), Recipe.operator("decon", sigma=2.0))
    _run("A1", ("DAPI", "GFP"), two_colour, cache)
    _run("B2", ("DAPI",), one_colour, cache)

    cells = {Address("A1"): cache.get("A1", two_colour),
             Address("B2"): cache.get("B2", one_colour)}

    assert composite_channels(cells.values()) == ("DAPI", "GFP")

    dapi = composite_plate(cells, "DAPI")
    assert set(dapi) == {Address("A1"), Address("B2")}, "a cell went missing from a channel it has"
    assert all(p.shape == (4, 4) for p in dapi.values())

    gfp = composite_plate(cells, "GFP")
    assert set(gfp) == {Address("A1")}, "a cell was drawn in a channel it never produced"


def test_a_divergence_in_z_depth_or_pixel_size_needs_NO_new_code():
    a = Result.of(Extent("A1"), _pixels(1), channels=("DAPI",), z_depth=1, pixel_size_um=0.325)
    b = Result.of(Extent("B2"), _pixels(1), channels=("DAPI",), z_depth=21, pixel_size_um=0.65)

    cells = {Address("A1"): a, Address("B2"): b}
    assert composite_channels(cells.values()) == ("DAPI",)
    assert set(composite_plate(cells, "DAPI")) == {Address("A1"), Address("B2")}
    assert (a.z_depth, a.pixel_size_um) == (1, 0.325)
    assert (b.z_depth, b.pixel_size_um) == (21, 0.65)


#: The fields a result declares; a comparison of any of these between two objects is the banned
#: shape this suite checks for.
DECLARED_FIELDS = {"channels", "dtype", "z_depth", "pixel_size_um", "substance", "extent",
                   "region_id"}

#: The modules that know what a Result is. Add a new one here when it learns about Results, or the
#: no-comparison check below stops covering it.
KNOWS_ABOUT_RESULTS = ("squidxplorer._result", "squidxplorer._recipe")


def _equality_on_a_declaration(module_name: str) -> "list[str]":
    """Every ``==``/``!=`` in *module_name* with a declared field on either side, found over the
    AST (not the text) so this module's own prose can't affect the result. ``in`` is not flagged:
    membership asks one object what it holds, equality needs a second to hold it against.

    Known limit: catches the direct spelling only, not one laundered through a call
    (``set(a.channels) != set(b.channels)``); ``test_no_function_takes_TWO_results`` covers that.
    """
    module = __import__(module_name, fromlist=["_"])
    tree = ast.parse(pathlib.Path(inspect.getfile(module)).read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            continue
        for operand in [node.left, *node.comparators]:
            if isinstance(operand, ast.Attribute) and operand.attr in DECLARED_FIELDS:
                hits.append(f"{module_name}: {ast.unparse(node)}")
                break                       # one report per comparison, not one per operand
    return hits


def test_NO_code_path_compares_two_results():
    """Mutation check: add ``if a.channels != b.channels`` anywhere in ``_result.py`` or
    ``_recipe.py`` and this goes red."""
    hits = [h for name in KNOWS_ABOUT_RESULTS for h in _equality_on_a_declaration(name)]
    assert hits == [], (
        "a result's declaration is being compared against something. Two runs with different "
        "channel sets are not a mismatch to detect; each cell declares what it is and the plate "
        "draws that. See the module docstring of squidxplorer/_result.py.\n  " + "\n  ".join(hits))


def test_two_results_are_not_comparable_BY_CONSTRUCTION():
    """``Result`` is ``eq=False``, so ``==`` falls back to identity; ``Substance`` stays comparable
    since comparing two descriptions (not two results) is fine."""
    assert Result.__eq__ is object.__eq__, "Result grew an __eq__: two results became comparable"

    s = Substance(channels=("DAPI",), z_depth=1, dtype="uint16", pixel_size_um=0.325)
    a, b = Result(Extent("A1"), s), Result(Extent("A1"), s)
    assert a != b and a == a                       # identity, not field-by-field
    assert Substance(("DAPI",), 1, "uint16", 0.325) == s      # descriptions do compare


def test_no_function_takes_TWO_results():
    """Complements the AST check above: a comparison laundered through a call still needs both
    results in one frame, so nothing in these modules should accept two."""
    for module_name in KNOWS_ABOUT_RESULTS:
        module = __import__(module_name, fromlist=["_"])
        tree = ast.parse(pathlib.Path(inspect.getfile(module)).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            named = [a.arg for a in args
                     if a.annotation is not None and "Result" in ast.unparse(a.annotation)]
            assert len(named) < 2, (
                f"{module_name}.{node.name} takes two results ({named}); the only reason to hold "
                "two at once is to compare them")


def test_a_cache_entry_carries_the_CHAIN_OBJECT_and_not_only_its_hash():
    """The key holds ``chain.key()``, a sha1 prefix that can't be un-hashed, so a legend needs the
    chain object itself to name what produced a result."""
    cache = ResultCache()
    chain = RecipeChain.of(Recipe.operator("mip"), Recipe.operator("decon", sigma=2.0))
    _run("A1", ("DAPI",), chain, cache)

    entry = cache.entries()[0]
    assert entry.scope == "A1" and entry.version == "0"
    assert entry.chain.key() == chain.key()
    assert [r.name for r in entry.chain.recipes] == ["mip", "decon"]
    assert entry.chain.recipes[1].params == {"sigma": 2.0}
    assert entry.result.channels == ("DAPI",)


def test_a_census_over_the_cache_needs_nothing_but_the_entries():
    """Groups entries by chain; grouping asks each entry which bucket it belongs in, never whether
    two entries agree."""
    cache = ResultCache()
    plain = RecipeChain.of(Recipe.operator("mip"))
    deconned = RecipeChain.of(Recipe.operator("mip"), Recipe.operator("decon", sigma=2.0))
    _run("A1", ("DAPI", "GFP"), plain, cache)
    _run("A2", ("DAPI", "GFP"), plain, cache)
    _run("B2", ("DAPI",), deconned, cache)

    census: "dict[str, list]" = {}
    for entry in cache.entries():
        census.setdefault(entry.chain.key(), []).append(Address(entry.result.region_id))

    assert len(census) == 2
    assert census[plain.key()] == [Address("A1"), Address("A2")]
    assert census[deconned.key()] == [Address("B2")]
