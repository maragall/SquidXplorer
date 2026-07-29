"""A cached result says what it is, so nothing ever compares two of them.

WHAT WAS WRONG
--------------
``squidmip._recipe.ResultCache`` stored BARE ARRAYS. An entry could not say which channels it
carried, how deep in z it was, what dtype it was, or what one pixel measured. Everything a consumer
needed in order to draw it had to come from somewhere else, and the only "somewhere else" available
was the acquisition the whole plate was assumed to share.

That assumption is what fails the moment two runs land under one plate, which is exactly Task 3.
Two runs whose recipes used different channel sets produce cells that are not alike, and with bare
arrays the only remaining move is to COMPARE cells and special-case the disagreement.

THE BANNED INTERIM, NAMED SO IT CANNOT COME BACK
------------------------------------------------
An earlier draft of Task 3 proposed exactly that: detect mixed-recipe plates and warn about them.
Julio removed it. It is disclosure bolted onto a painter, and the bolt does not generalise: the
next divergence is z depth, then pixel size, then dtype, and each needs its own comparison, its own
warning and its own test. **Nothing interim ships.**

So the replacement is not a better comparison. It is the removal of the question. A result carries
its own ``Extent`` (WHERE) and its own ``Substance`` (WHAT it is made of), and the plate composites
what each cell DECLARES. Two runs with different channel sets are not a mismatch to detect; they
are two results that each say what they are.

WHAT IS PINNED HERE
-------------------
* A result round-trips its extent and its substance, through JSON, with nothing lost.
* A substance covers all four: channel set, z depth, dtype, and pixel size IN MICROMETRES.
* A substance is never vague. ``None``/"all of it" is legal in an extent, which describes a
  REQUEST, and illegal in a substance, which describes a PRODUCT.
* Two results with different channel sets coexist in one cache and each reports its own.
* A plate built from both renders both: the union of channels is shown, each cell contributes only
  what it declares, and a cell that lacks a channel is ABSENT from it rather than drawn black.
* **The absence of the banned thing, asserted structurally.** ``tests/test_tsctx.py`` greps
  ``_ComputedPlateWorker._read`` for a ``ts.open(`` that must not come back; this does the same for
  a comparison, but over the AST rather than the text, so a docstring that mentions the word cannot
  make it pass. Plus the stronger form: ``Result`` has no ``__eq__`` at all.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib

import numpy as np
import pytest

from squidmip._address import Address, Extent
from squidmip._recipe import Recipe, RecipeChain, ResultCache
from squidmip._result import (
    Result,
    Substance,
    composite_channels,
    composite_plate,
)


def _pixels(n_channels: int, value: int = 7) -> np.ndarray:
    """A channel-major ``(C, Y, X)`` stack whose planes are distinguishable by their value."""
    return np.stack([np.full((4, 4), value + i, dtype=np.uint16) for i in range(n_channels)])


# --- a result round-trips its extent AND its substance -------------------------------------------

def test_a_result_round_trips_its_extent_and_its_substance_through_json():
    """The declaration is what has to survive to disk, so it has to survive JSON. The pixels do
    not: they go to disk as pixels (see ``_platecache``), and splitting them is what lets Task 3's
    census answer "what is on this plate" without paging a single array in."""
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
    """All four, because all four are what a later divergence will be about. z depth and pixel size
    are in here NOW, before anything diverges on them, which is the difference between this and a
    warning bolted on per axis."""
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
    """The reason the channel set is not simply read off the extent.

    ``Extent.channels`` may be ``None``, meaning "all of them", because an extent describes what was
    ASKED FOR. That is not an answer a painter can use: resolving it means going back to the
    acquisition, which is precisely the outside knowledge a self-describing result exists to remove.
    A substance describes what the result IS, so it is concrete or it does not construct.
    """
    assert Extent("A1").channels is None                # a request may say "all of it"
    assert Extent("A1", channels=()).channels is None   # and the empty set means the same

    with pytest.raises(ValueError, match="at least one channel"):
        Substance(channels=(), z_depth=1, dtype="uint16", pixel_size_um=0.325)


def test_a_result_refuses_a_scale_it_does_not_have():
    """``reader.py`` raises rather than placing FOVs "at positions that would look plausible but be
    wrong". A pixel size of zero, or a negative one, is that same wrongness one layer down: it
    renders as a plausible picture at the wrong magnification."""
    for bad in (0.0, -0.325):
        with pytest.raises(ValueError, match="pixel_size_um"):
            Substance(channels=("DAPI",), z_depth=1, dtype="uint16", pixel_size_um=bad)
    with pytest.raises(ValueError, match="z_depth"):
        Substance(channels=("DAPI",), z_depth=0, dtype="uint16", pixel_size_um=0.325)


def test_the_channel_order_of_a_substance_is_the_PIXEL_order_and_not_sorted():
    """Deliberately the opposite rule to ``Extent``, and the contrast IS the test.

    An extent sorts its channels because it is a KEY and one slab must not have two spellings. A
    substance must not sort, because its order is the axis order of the pixels: sorting it would
    leave every plane under the wrong name, which renders as a plausible picture in the wrong
    colour."""
    assert Extent("A1", channels=("GFP", "DAPI")).channels == ("DAPI", "GFP")   # sorted: a key

    s = Substance(channels=("GFP", "DAPI"), z_depth=1, dtype="uint16", pixel_size_um=0.325)
    assert s.channels == ("GFP", "DAPI"), "a substance sorted its channels: planes now misnamed"

    r = Result(extent=Extent("A1"), substance=s, data=_pixels(2))
    assert int(r.plane("GFP")[0, 0]) == 7       # plane 0
    assert int(r.plane("DAPI")[0, 0]) == 8      # plane 1


def test_a_plane_is_found_by_NAME_and_a_miss_says_what_the_result_DOES_carry():
    """By name and not by index, because the channel order at the producer is not the channel order
    at the display and an index resolves silently to the wrong colour. The error names what is
    carried, since a self-describing result is exactly the thing that can say so."""
    r = Result(extent=Extent("A1"),
               substance=Substance(channels=("DAPI",), z_depth=1, dtype="uint16",
                                   pixel_size_um=0.325),
               data=_pixels(1))
    with pytest.raises(KeyError) as excinfo:
        r.plane("GFP")
    assert "DAPI" in str(excinfo.value)


def test_of_takes_the_dtype_from_the_pixels_and_refuses_a_channel_count_that_disagrees():
    """A producer cannot mislabel its own output's dtype, because it does not get to state it.

    ``z_depth`` IS stated, on purpose: a ``(C, Z, Y, X)`` stack and a ``(C, Y, X)`` plane set are
    told apart only by knowing which the operator produced, and inferring it from ``ndim`` is the
    plausible-and-wrong guess this codebase refuses."""
    r = Result.of(Extent("A1"), _pixels(2), channels=("DAPI", "GFP"), z_depth=1,
                  pixel_size_um=0.325)
    assert r.dtype == "uint16"

    with pytest.raises(ValueError, match="refusing to guess"):
        Result.of(Extent("A1"), _pixels(2), channels=("DAPI",), z_depth=1, pixel_size_um=0.325)


# --- two channel sets under one plate ------------------------------------------------------------

def _run(region: str, channels, chain: RecipeChain, cache: ResultCache) -> Result:
    """One run's result for one cell, put in the cache under its own chain."""
    r = Result.of(Extent(region), _pixels(len(channels)), channels=channels, z_depth=1,
                  pixel_size_um=0.325)
    cache.put(region, chain, r)
    return r


def test_two_results_with_different_channel_sets_coexist_and_each_reports_its_own():
    """The situation the banned warning existed for. Nothing here notices that the two differ, and
    nothing needs to: each is asked what it is and each answers."""
    cache = ResultCache()
    two_colour = RecipeChain.of(Recipe.operator("mip"))
    one_colour = RecipeChain.of(Recipe.operator("mip"), Recipe.operator("decon", sigma=2.0))

    _run("A1", ("DAPI", "GFP"), two_colour, cache)
    _run("B2", ("DAPI",), one_colour, cache)

    assert cache.get("A1", two_colour).channels == ("DAPI", "GFP")
    assert cache.get("B2", one_colour).channels == ("DAPI",)
    assert len(cache) == 2, "one cell's declaration overwrote the other's"


def test_two_runs_over_the_SAME_cell_are_two_entries_each_with_its_own_declaration():
    """Task 3's per-cell identity decision, seen from the cache: two runs are already two keys under
    one node, so the read path is a lookup and never a reconciliation."""
    cache = ResultCache()
    plain = RecipeChain.of(Recipe.operator("mip"))
    deconned = RecipeChain.of(Recipe.operator("mip"), Recipe.operator("decon", sigma=2.0))

    _run("A1", ("DAPI", "GFP"), plain, cache)
    _run("A1", ("DAPI",), deconned, cache)

    assert cache.get("A1", plain).channels == ("DAPI", "GFP")
    assert cache.get("A1", deconned).channels == ("DAPI",)


def test_a_plate_built_from_BOTH_renders_BOTH():
    """The whole point. Two cells with different channel sets, one plate, both drawn.

    * The plate shows the UNION of what its cells declare. An intersection would silently hide GFP,
      which was computed; taking the first cell's set would silently invent GFP for a cell that
      never produced it.
    * Each cell contributes only what it declares, and a cell that lacks a channel is ABSENT from
      that channel rather than drawn black. Black is a measurement.
    """
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
    """The generalisation claim, made checkable. The banned warning was about channel sets; the next
    divergence would have been z depth, then pixel size, then dtype, each needing its own bolt.

    Nothing below is a new code path. The two cells simply declare different things and each still
    answers for itself, using the same functions the channel case used."""
    a = Result.of(Extent("A1"), _pixels(1), channels=("DAPI",), z_depth=1, pixel_size_um=0.325)
    b = Result.of(Extent("B2"), _pixels(1), channels=("DAPI",), z_depth=21, pixel_size_um=0.65)

    cells = {Address("A1"): a, Address("B2"): b}
    assert composite_channels(cells.values()) == ("DAPI",)
    assert set(composite_plate(cells, "DAPI")) == {Address("A1"), Address("B2")}
    assert (a.z_depth, a.pixel_size_um) == (1, 0.325)
    assert (b.z_depth, b.pixel_size_um) == (21, 0.65)


# --- the absence, asserted ----------------------------------------------------------------------

#: The fields a result declares. A comparison of any of these BETWEEN two objects is the banned
#: shape, whatever it is called.
DECLARED_FIELDS = {"channels", "dtype", "z_depth", "pixel_size_um", "substance", "extent",
                   "region_id"}

#: The modules that know what a Result is. Task 3 adds its census and its legend: ADD THEM HERE.
KNOWS_ABOUT_RESULTS = ("squidmip._result", "squidmip._recipe")


def _equality_on_a_declaration(module_name: str) -> "list[str]":
    """Every ``==``/``!=`` in *module_name* with a declared field on either side.

    Over the AST and not the text, so the prose in these modules -- which has to name the thing it
    refuses in order to explain it -- cannot make the check pass or fail. ``in`` is deliberately NOT
    flagged: membership asks ONE object what it holds, where equality needs a second object to hold
    it against, and ``Result.declares`` is built on exactly that distinction.

    Known limit, stated rather than papered over: this catches the direct spelling
    (``a.channels != b.channels``), not one laundered through a call (``set(a.channels) !=
    set(b.channels)``). The complementary checks below close the gap from the other side.
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
    """The test the whole task is judged by.

    Modelled on ``test_the_plate_scrub_goes_through_the_pool`` in ``tests/test_tsctx.py``, which
    asserts that a ``ts.open(`` did not come back. This asserts that a comparison never arrived.

    MUTATION: add ``if a.channels != b.channels`` anywhere in ``_result.py`` or ``_recipe.py`` and
    this goes red."""
    hits = [h for name in KNOWS_ABOUT_RESULTS for h in _equality_on_a_declaration(name)]
    assert hits == [], (
        "a result's declaration is being compared against something. Two runs with different "
        "channel sets are not a mismatch to detect; each cell declares what it is and the plate "
        "draws that. See the module docstring of squidmip/_result.py.\n  " + "\n  ".join(hits))


def test_two_results_are_not_comparable_BY_CONSTRUCTION():
    """The stronger form of the same rule: the equality a caller would reach for does not exist.

    ``Result`` is declared ``eq=False``, so ``==`` falls back to identity. Rebuilding the banned
    feature therefore takes a deliberate act rather than an ``==`` that reads innocently. Contrast
    ``Substance``, which IS comparable: comparing two DESCRIPTIONS is how Task 3's legend lists what
    is present, and it is comparing two RESULTS that is refused."""
    assert Result.__eq__ is object.__eq__, "Result grew an __eq__: two results became comparable"

    s = Substance(channels=("DAPI",), z_depth=1, dtype="uint16", pixel_size_um=0.325)
    a, b = Result(Extent("A1"), s), Result(Extent("A1"), s)
    assert a != b and a == a                       # identity, not field-by-field
    assert Substance(("DAPI",), 1, "uint16", 0.325) == s      # descriptions do compare


def test_no_function_takes_TWO_results():
    """The other side of the AST gap. A comparison laundered through a call still needs both
    results in one frame, and nothing in these modules accepts two."""
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


# --- what Task 3 will read ----------------------------------------------------------------------

def test_a_cache_entry_carries_the_CHAIN_OBJECT_and_not_only_its_hash():
    """Task 3's legend must read ``mip + decon sigma 2.0`` and never a hash, and the plan says so in
    those words. The key holds ``chain.key()``, which is a sha1 prefix and cannot be un-hashed, so
    the chain itself is kept beside the result. Without this the census could count cells and could
    not name what produced them."""
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
    """A dry run of the shape Task 3 specifies, ``{chain: [address]}``, written here only to prove
    the entries carry enough. It groups; it does not compare. Grouping asks each entry which bucket
    it belongs in; comparing asks two entries whether they agree."""
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
