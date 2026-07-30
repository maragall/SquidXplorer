"""Recipes, content-addressed results, and copy/paste of transforms.

The design we drew (2026-07-24): the window TREE is navigation only. RESULTS live in a flat,
CONTENT-ADDRESSED cache keyed by the data scope plus the op-chain, so two windows over the same well
with the same chain resolve to the SAME entry. Cross-propagation is then free and lazy, with no
window-to-window messaging, no signal recursion, and no need to wake halted windows.

A RECIPE is the serializable unit you copy/paste: an OPERATOR (a data transform, key + params) or a
LUT (a contrast transform). Same mechanism, which is exactly why "copy LUTs" and "copy an operator"
are one system rather than two. A CHAIN of recipes, e.g. [stitch, decon3d] or [contrast], is BOTH
the content-address of a result AND the script you paste onto another view or the plate.

What the cache STORES is :class:`squidmip._result.Result` (Task 2, 2026-07-29): a result that
carries its own extent and its own substance, so a plate built from several runs draws each cell
from that cell's declaration instead of comparing cells and special-casing the disagreement. Bare
arrays are refused. See :mod:`squidmip._result` for why comparison is the wrong shape.

:func:`census` (Task 3, 2026-07-29) is the read half: it groups cache entries by chain and returns
``{chain: [address]}``, the plurality chain and the divergent cells. It is HERE and not in the
viewer because "what is on this plate" is a question about the cache. :meth:`Recipe.label` and
:meth:`RecipeChain.label` are the one renderer that turns a chain into words a legend can show, so
a legend never has to show a hash.

This module is pure Python, no Qt, no numpy: the model, testable in isolation. The GUI layer builds
recipes from what a window shows and applies them by registering keys the cache computes lazily.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from squidmip._address import Address
from squidmip._result import Result, composite_channels

#: Recipe kinds. OPERATOR is a data transform (mip, stitch, decon, ...); LUT is a contrast transform
#: (per-channel contrast_limits + colormap). Both are transforms, so both flow through one path.
OPERATOR = "operator"
LUT = "lut"


@dataclass(frozen=True)
class Recipe:
    """One transform. ``kind`` is OPERATOR or LUT; ``name`` is the op key (``"decon"``) or
    ``"contrast"``; ``params`` is the transform's arguments (op kwargs, or per-channel LUTs).

    Its ``key`` is a stable content hash: two recipes with the same kind/name/params hash the same,
    so the cache can tell "the same transform" from "a different one" without comparing pixels."""

    kind: str
    name: str
    params: dict = field(default_factory=dict)

    def key(self) -> str:
        blob = json.dumps(
            {"kind": self.kind, "name": self.name, "params": self.params},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def label(self) -> str:
        """What a HUMAN reads: ``mip``, ``decon(sigma=2.0)``, ``contrast(DAPI,GFP)``.

        Never the key. Task 3's legend has to name what produced a cell, and a sha1 prefix cannot
        be un-hashed, so the label is derived from the recipe itself.

        This is the ONE renderer for a recipe in this application: ``_viewer._action_label``, which
        writes the console's ``[3] A1 decon(sigma=2.0) started``, delegates here. Two spellings for
        one transform is the drift the naming law (see :mod:`squidmip._address`) exists to stop, and
        a legend and a console disagreeing about the same run is exactly where it would start.

        Parameters are SORTED, so the same call always renders the same string and two runs can be
        told apart by eye. A LUT names its channels rather than dumping their windows: a legend row
        has to stay one line, and the per-channel limits are the contrast panel's business.
        """
        if self.kind == LUT:
            per = (self.params or {}).get("per_channel") or {}
            return f"{self.name}({','.join(sorted(str(c) for c in per))})" if per else str(self.name)
        if not self.params:
            return str(self.name)
        args = ", ".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"{self.name}({args})"

    def __str__(self) -> str:
        return self.label()

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "params": dict(self.params)}

    @staticmethod
    def from_dict(d: dict) -> "Recipe":
        return Recipe(str(d["kind"]), str(d["name"]), dict(d.get("params") or {}))

    # Convenience builders so callers do not hardcode the kind strings.
    @staticmethod
    def operator(key: str, **params: Any) -> "Recipe":
        return Recipe(OPERATOR, str(key), dict(params))

    @staticmethod
    def contrast(per_channel: dict) -> "Recipe":
        """A LUT recipe: ``per_channel`` maps channel name -> {"clim": (lo, hi), "cmap": <name>}."""
        return Recipe(LUT, "contrast", {"per_channel": dict(per_channel)})


@dataclass(frozen=True)
class RecipeChain:
    """An ORDERED list of recipes. Order matters (stitch then decon != decon then stitch), so the
    chain key folds the recipe keys in sequence. The chain is the cache key and the paste script."""

    recipes: tuple = ()

    def key(self) -> str:
        h = hashlib.sha1()
        for r in self.recipes:
            h.update(r.key().encode("utf-8"))
        return h.hexdigest()[:16]

    def add(self, recipe: Recipe) -> "RecipeChain":
        return RecipeChain(self.recipes + (recipe,))

    def is_empty(self) -> bool:
        return not self.recipes

    def label(self) -> str:
        """``mip + decon(sigma=2.0)``. The chain, in order, in words.

        ``+`` and not ``->`` because the legend row is read left to right as "this, then that", and
        an arrow in a 596 px pane reads as a control rather than as prose.

        An EMPTY chain is ``raw``, which is this application's existing word for the untransformed
        layer (``PlateOverview._active`` starts at ``"raw"``, the plate title says ``· raw``). A
        cell that had nothing applied to it is a real thing to say in a legend, and saying it in the
        word already on screen beats inventing a second one for the same state.
        """
        return " + ".join(r.label() for r in self.recipes) if self.recipes else "raw"

    def __str__(self) -> str:
        return self.label()

    def to_script(self) -> str:
        """A tiny, human-readable, re-loadable JSON script. "Copy an operator" yields this string;
        Julio: copying an operator "in reality generates a script"."""
        return json.dumps([r.to_dict() for r in self.recipes], indent=2)

    @staticmethod
    def from_script(text: str) -> "RecipeChain":
        data = json.loads(text)
        return RecipeChain(tuple(Recipe.from_dict(d) for d in data))

    @staticmethod
    def of(*recipes: Recipe) -> "RecipeChain":
        return RecipeChain(tuple(recipes))


@dataclass(frozen=True)
class Entry:
    """One cache entry, with everything needed to say what it is WITHOUT a second lookup.

    Task 3's plate census walks these. It needs the ``chain`` OBJECT and not just the hash the key
    carries, because the legend it builds must read ``mip + decon sigma 2.0`` and never a hash, and
    a hash cannot be un-hashed back into recipes. So the chain is kept beside the result rather
    than only folded into the key.
    """

    scope: str
    version: str
    chain: RecipeChain
    result: Result


class ResultCache:
    """Flat, content-addressed result store: the key is ``(scope, version, chain.key())``.

    ``scope`` is the data node's identity (the integer RRCCOOOO id, see ``_plate.cache_scope``);
    ``chain`` is the op-chain; ``version`` is the ACQUISITION VERSION, baked in now so re-parenting
    to the live Squid source is trivial later. For a static folder ``version`` stays ``0`` and the
    key is a forever-key. For a live scope the reader bumps ``version`` as a node's frames arrive, so
    a stale result (yesterday's decon of a well still being imaged) simply misses and recomputes,
    without any explicit invalidation pass. The temporal dimension is thus part of identity, not a
    side channel.

    Because the key is content, two windows over the same node running the same chain at the same
    version hit the SAME entry: results cross-propagate for free, lazily, with no window-to-window
    signalling. Bounded LRU so a long session never blows memory.

    THE VALUE IS A :class:`squidmip._result.Result`, AND ONLY THAT (Task 2, 2026-07-29). This store
    held bare arrays, so an entry could not say which channels it carried, how deep in z it was,
    what dtype it was or what a pixel measured. A plate built from such entries can only be drawn
    by assuming every cell is like every other one, and the moment two runs differ the only
    remaining move is to COMPARE cells and special-case the disagreement, which is the thing Julio
    banned. A result that knows itself removes the question: the plate composites what each cell
    declares. A bare array is refused rather than accepted-and-wrapped, because a store that takes
    both is the interim state, and the next reader of that store would have to branch on which kind
    it got.

    The KEY is deliberately untouched here: still the packed ``row*1e6 + col*1e4 + roi`` scope
    string. Moving it onto :class:`squidmip._address.Extent` is Task 3.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._d: "OrderedDict[tuple, Entry]" = OrderedDict()
        self._max = max(1, int(max_entries))

    @staticmethod
    def _k(scope: str, chain: RecipeChain, version: Any) -> tuple:
        return (str(scope), str(version), chain.key())

    def get(self, scope: str, chain: RecipeChain, version: Any = 0) -> Optional[Result]:
        k = self._k(scope, chain, version)
        if k in self._d:
            self._d.move_to_end(k)          # most-recently used
            return self._d[k].result
        return None

    def put(self, scope: str, chain: RecipeChain, result: Result, version: Any = 0) -> None:
        if not isinstance(result, Result):
            raise TypeError(
                "ResultCache stores squidmip._result.Result, not "
                f"{type(result).__name__}. A cached result has to carry its own extent and its own "
                "substance (channels, z depth, dtype, pixel size), or the plate that draws it has "
                "no way to know what it is drawing and ends up comparing cells to find out.")
        k = self._k(scope, chain, version)
        self._d[k] = Entry(scope=str(scope), version=str(version), chain=chain, result=result)
        self._d.move_to_end(k)
        while len(self._d) > self._max:
            self._d.popitem(last=False)      # evict least-recently used

    def has(self, scope: str, chain: RecipeChain, version: Any = 0) -> bool:
        return self._k(scope, chain, version) in self._d

    def entries(self) -> "list[Entry]":
        """Every entry, least-recently-used first. What a plate census reads.

        A snapshot list rather than a live view: a census that walked the OrderedDict directly
        would see it reorder under it on the first ``get``, since ``get`` is what marks an entry
        most-recently-used.
        """
        return list(self._d.values())

    def clear(self) -> None:
        self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


#: The process-wide result cache. One store for the whole app, so any window/plate that renders a
#: (scope, chain) it has computed before, or that ANOTHER window computed, reuses the result.
RESULTS = ResultCache()

#: The copy/paste buffer for a recipe chain (generalises the contrast-only _LUT_CLIPBOARD). "Copy"
#: puts a chain here (and its script); "Paste" applies it to a view / the plate / everything.
CLIPBOARD: "dict[str, RecipeChain]" = {"chain": RecipeChain()}


def copy_chain(chain: RecipeChain) -> str:
    """Put *chain* on the clipboard and return its script (what a Copy action shows / stores)."""
    CLIPBOARD["chain"] = chain
    return chain.to_script()


def paste_chain() -> RecipeChain:
    """The chain currently on the clipboard (empty chain if nothing was copied)."""
    return CLIPBOARD.get("chain") or RecipeChain()


# --- the plate census: which recipes made this plate, and which cells diverge --------------------
#
# Task 3 (2026-07-29): combine two runs in one plate view. The workflow it serves is real and
# ordinary -- dial parameters on A1 alone, run the other 95 with what you learned, look at the whole
# plate -- and Julio's decision was PER-CELL IDENTITY: a mixed-recipe plate is legal.
#
# THIS LIVES IN THE MODEL, NOT IN THE PAINTER. "What is on this plate" is a question about the
# CACHE: the entries are already keyed (scope, version, chain), so two runs are already two keys
# under one node and the answer is a GROUPING of what is there. A painter asking it would have to
# hold plate state to answer, and holding it is how the disclosure ends up bolted to the paint.
#
# IT GROUPS. IT DOES NOT COMPARE. Grouping asks each entry which bucket it belongs in; comparing
# asks two entries whether they agree. An earlier draft of Task 3 proposed the second thing --
# detect a mixed plate, warn about it -- and Julio banned it, because the next divergence (z depth,
# pixel size, dtype) would need its own comparison, its own warning and its own test. Every channel
# set below is read from a cell's own Substance via composite_channels; nothing here holds two
# results at once, and tests/test_result.py asserts that over this module's AST.

@dataclass(frozen=True)
class ChainCensus:
    """One chain's share of a plate: what it is, which cells it made, what those cells carry.

    Exactly the three things the plan says a legend row must show, and no more: a HUMAN label
    (:meth:`label`, from the recipes, never a hash), a cell count, and a channel set read from the
    cells' declarations.
    """

    chain: RecipeChain
    addresses: "tuple" = ()      # the cells this chain produced, first-seen order, deduplicated
    channels: "tuple" = ()       # the UNION of what those cells declare (composite_channels)
    diverges: bool = False       # this is NOT the plurality chain, so its cells carry the mark

    @property
    def key(self) -> str:
        """The chain's content hash. For identity only. It must never reach a legend."""
        return self.chain.key()

    @property
    def count(self) -> int:
        """Cells, not entries. Two runs of one chain over one cell are one cell."""
        return len(self.addresses)

    def label(self) -> str:
        """``mip + decon(sigma=2.0)``. See :meth:`RecipeChain.label`."""
        return self.chain.label()

    def __str__(self) -> str:
        return f"{self.label()}  {self.count} cell(s)  {','.join(self.channels)}"


@dataclass(frozen=True)
class PlateCensus:
    """What a plate is made of. The three things Task 3 asks for, in one object.

    ``{chain: [address]}`` is :attr:`by_chain`, keyed by the chain's KEY rather than by the chain
    object, for a blunt reason: ``Recipe.params`` is a dict, so a ``Recipe`` (and therefore a
    ``RecipeChain``) is unhashable and cannot be a dict key. The chain OBJECT is not lost -- it is
    on every :class:`ChainCensus` in :attr:`groups`, which is what lets a legend read words instead
    of a hash.
    """

    groups: "tuple" = ()                          # ChainCensus per chain, first-seen order
    plurality: "Optional[ChainCensus]" = None     # the chain that made the most cells
    divergent: "tuple" = ()                       # cells a NON-plurality chain made

    @property
    def by_chain(self) -> dict:
        """``{chain key: (address, ...)}``, the shape the plan specifies."""
        return {g.key: g.addresses for g in self.groups}

    @property
    def is_mixed(self) -> bool:
        """More than one chain is present, so the plate must SAY SO ON ITS FACE.

        The legend's visibility is this, and nothing else. Earned expensively on 2026-07-28, when a
        tooltip promised a "3D view (AGAVE)..." button the app did not have and a passing test held
        that phantom in place for weeks: a plate that is showing two recipes discloses it in the
        window, not in a hover.
        """
        return len(self.groups) > 1

    def __str__(self) -> str:
        return " | ".join(str(g) for g in self.groups) if self.groups else "empty"


def census(source: "Any" = None) -> PlateCensus:
    """Group *source*'s entries by chain: ``{chain: [address]}``, the plurality, the divergent set.

    *source* is a :class:`ResultCache` or any iterable of :class:`Entry` -- a window censuses the
    process-wide cache filtered to ITS OWN cells (``e.result.region_id in self._fov_index``), which
    is a membership test on one entry and never a comparison of two.

    An address is ``Address(region_id)``: on a plate a CELL IS A REGION, so that is the granularity
    the legend counts and the granularity a border can be drawn around. An ROI inside a cell is a
    ``bbox_um`` on an extent and is not a cell (see :mod:`squidmip._address`).

    PLURALITY is the chain with the most cells, ties broken by which was seen first. A tie is
    stable rather than meaningful, and that is stated rather than hidden: on a 50/50 plate the
    marked half is the second-seen one, and the legend still lists both chains with both counts, so
    nothing is being concealed by the choice.

    DIVERGENT is every cell some non-plurality chain produced. A cell that carries BOTH the
    plurality chain's result and another one IS divergent: it holds something the rest of the plate
    does not, which is the whole reason a mark exists.
    """
    entries = source.entries() if hasattr(source, "entries") else list(source or ())

    grouped: "OrderedDict[str, dict]" = OrderedDict()
    for entry in entries:
        bucket = grouped.setdefault(
            entry.chain.key(), {"chain": entry.chain, "cells": OrderedDict(), "results": []})
        bucket["cells"].setdefault(Address(entry.result.region_id), None)
        bucket["results"].append(entry.result)

    counted = tuple(
        ChainCensus(chain=b["chain"], addresses=tuple(b["cells"]),
                    channels=composite_channels(b["results"]))
        for b in grouped.values())

    # max() returns the FIRST maximal element, which is what makes the tie rule "first seen wins".
    biggest = max(counted, key=lambda g: g.count, default=None)
    plurality_key = None if biggest is None else biggest.key

    # ``diverges`` is decided HERE, once, so that the legend and the plate's border read the same
    # fact rather than each deciding what "divergent" means. A painter that re-derived it is a
    # second definition, and two definitions of one word is how the two disagree by one cell.
    groups = tuple(
        ChainCensus(chain=g.chain, addresses=g.addresses, channels=g.channels,
                    diverges=(plurality_key is not None and g.key != plurality_key))
        for g in counted)
    plurality = next((g for g in groups if not g.diverges), None) if groups else None

    divergent: "OrderedDict[Address, None]" = OrderedDict()
    for group in groups:
        if not group.diverges:
            continue
        for address in group.addresses:
            divergent.setdefault(address, None)

    return PlateCensus(groups=groups, plurality=plurality, divergent=tuple(divergent))
