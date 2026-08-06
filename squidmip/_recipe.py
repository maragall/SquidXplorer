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

THE CHAIN NOW RUNS (2026-08-05)
-------------------------------
:class:`RecipeChain` documented ``mip + decon(sigma=2.0)`` and the rule that order matters, and for
two weeks **nothing executed it**: it was a cache key and a label renderer, and every reader of this
module concluded the application could chain operators when it could not. That is the worse half of
dead code -- not unused, but actively misleading -- and it is closed by making the label the
EXPRESSION rather than by building a second syntax beside it:

    RecipeChain.label()  ->  "flatfield + decon + mip"   the words a human reads
    RecipeChain.parse()  <-  the exact inverse, so a legend line can be pasted back into a run

:func:`squidmip._compose.compose_operator` turns a parsed chain into ONE
:class:`squidmip._engine.Operator` whose ``consumes``/``produces``/``params``/``requires`` are
derived from its parts, so ``project_plate(projector="flatfield+mip")`` and
``stitch_region(projector=...)`` take a chain wherever they take a name, with no new call shape.
One spelling for the label, the cache key, the paste script and the run.

What was REMOVED in the same commit, for the same reason: ``census`` / ``PlateCensus`` /
``ChainCensus`` grouped cache entries by chain for a plate legend (``squidmip/_legend.py``) that was
never constructed by anything and whose docstring cited a test file that does not exist. Two dead
halves of one feature, documented as if live. If the mixed-recipe plate legend is built, this is the
grouping it wants and ``git log`` has it; a module that describes a feature nobody can reach is not
an asset.

This module is pure Python, no Qt, no numpy: the model, testable in isolation. The GUI layer builds
recipes from what a window shows and applies them by registering keys the cache computes lazily.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from squidmip._result import Result

#: Recipe kinds. OPERATOR is a data transform (mip, stitch, decon, ...); LUT is a contrast transform
#: (per-channel contrast_limits + colormap). Both are transforms, so both flow through one path.
OPERATOR = "operator"
LUT = "lut"

#: What separates two steps of a chain, in the ONE spelling that is both read and written. It is
#: ``+`` and not ``->`` for the reason :meth:`RecipeChain.label` gives (an arrow in a 596 px legend
#: pane reads as a control rather than as prose), and now that the label is also the expression the
#: engine accepts, that choice binds both directions.
CHAIN_SEPARATOR = "+"

#: What an EMPTY chain is called. Already this application's word for the untransformed layer
#: (``PlateOverview._active`` starts at ``"raw"``), so :meth:`RecipeChain.parse` accepts it and
#: label -> parse -> label round-trips for every chain including the empty one.
RAW = "raw"


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


# --- reading a chain back out of its own label ---------------------------------------------------
#
# A hand-rolled splitter rather than a grammar, because the language is two characters wide and a
# parser generator would be more machinery than the thing it parses. It tracks parenthesis DEPTH so
# a separator inside an argument list is not a separator: ``decon(shape=(4, 4)) + mip`` splits into
# two steps and not four, and ``bgsub(scale=1e+5)`` is one step and not two.


def _split_top_level(text: str, separator: str, whole: str) -> "list[str]":
    """Split *text* on *separator*, ignoring separators nested inside parentheses.

    *whole* is the full expression, quoted in any refusal: a caller who mistyped one step needs to
    see what they typed, not the fragment this function happened to be looking at.
    """
    parts: "list[str]" = []
    depth, buf = 0, []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced ')' in operator chain {whole!r}")
        if char == separator and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    if depth:
        raise ValueError(f"unbalanced '(' in operator chain {whole!r}")
    parts.append("".join(buf))
    return parts


def _literal(text: str):
    """A parameter value: a Python literal when it is one, otherwise the bare string it looks like.

    The fallback is not laxness, it is what makes the round trip hold. :meth:`Recipe.label` renders
    a value with ``str``, so a string parameter comes out UNQUOTED (``segmenter=cellpose``) and a
    parser that insisted on quotes could not read back the label it is the inverse of.
    """
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _parse_step(step: str, whole: str) -> Recipe:
    """One step of a chain: ``mip``, or ``spot(min_area_px=80, split_touching=False)``."""
    step = step.strip()
    if not step:
        raise ValueError(
            f"empty step in operator chain {whole!r}; every '{CHAIN_SEPARATOR}' separates two "
            "operators, so a trailing or doubled separator has nothing to run")
    if not step.endswith(")"):
        if "(" in step:
            raise ValueError(f"unbalanced '(' in operator chain {whole!r}")
        return Recipe.operator(step)

    name, _, arguments = step.partition("(")
    name = name.strip()
    if not name:
        raise ValueError(f"a step of operator chain {whole!r} has arguments but no operator name")

    params: dict = {}
    for argument in _split_top_level(arguments[:-1], ",", whole):
        argument = argument.strip()
        if not argument:
            continue                      # a trailing comma is a typo, not a refusal
        key, assigned, value = argument.partition("=")
        if not assigned:
            raise ValueError(
                f"argument {argument!r} of {name!r} in operator chain {whole!r} is not "
                "name=value; an operator's parameters are named, never positional")
        params[key.strip()] = _literal(value.strip())
    return Recipe.operator(name, **params)


@dataclass(frozen=True)
class RecipeChain:
    """An ORDERED list of recipes. Order matters (stitch then decon != decon then stitch), so the
    chain key folds the recipe keys in sequence. The chain is the cache key and the paste script.

    Since 2026-08-05 it is also the RUN: :meth:`parse` reads a chain back out of :meth:`label`, and
    :func:`squidmip._compose.compose_operator` turns the result into one registry-shaped operator.
    See this module's docstring for why the label and the expression are deliberately one string."""

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
        if not self.recipes:
            return RAW
        return f" {CHAIN_SEPARATOR} ".join(r.label() for r in self.recipes)

    def __str__(self) -> str:
        return self.label()

    @staticmethod
    def parse(text: str) -> "RecipeChain":
        """``"flatfield + decon + mip"`` -> the chain. The inverse of :meth:`label`.

        THE EXPRESSION AND THE LABEL ARE ONE STRING. A second syntax for "the same chain, but for
        the engine" would be two spellings of one fact, which is the drift the naming law (see
        :mod:`squidmip._address`) exists to stop -- and it would put a legend row and a runnable
        command one transcription error apart. So the words a legend shows are exactly the words
        ``project_plate(projector=...)`` accepts, and ``parse(chain.label()) == chain`` for every
        chain this application builds, the empty one (:data:`RAW`) included.

        Whitespace around the separator and inside the argument list is free. Parameter VALUES go
        through :func:`ast.literal_eval`, so ``iterations=15`` is an int, ``sigma=2.0`` a float and
        ``gpu=False`` a bool; a value that is not a Python literal is taken as the bare string it
        looks like (``segmenter=cellpose``), which is what :meth:`Recipe.label` writes for a string
        parameter and therefore what round-tripping requires.

        Every step is an OPERATOR recipe. A LUT is not an engine operator and has no place in a
        run's projector expression; contrast travels with the view, not with the pixels.

        Raises
        ------
        ValueError
            On unbalanced parentheses, an empty step (``"mip + + decon"``), a step with no name, or
            an argument that is not ``name=value``. Every one of them names the offending text: a
            chain is typed by a human into a CLI flag or a command, so the refusal has to say which
            character was wrong rather than that "the chain" was.
        """
        raw = str(text).strip()
        if not raw or raw == RAW:
            return RecipeChain()
        return RecipeChain(tuple(
            _parse_step(step, raw) for step in _split_top_level(raw, CHAIN_SEPARATOR, raw)))

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

    The ``chain`` OBJECT is kept beside the result, not just folded into the key, because a hash
    cannot be un-hashed back into recipes and anything reporting what produced a cell must read
    ``flatfield + decon + mip`` rather than ``a3f9c1``. :func:`cached_operator_results` is the
    reader that depends on it today; it is also what would let a chain be re-RUN from a cache entry,
    now that :meth:`RecipeChain.label` is an expression the engine accepts.
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
        """Every entry, least-recently-used first. What :func:`cached_operator_results` reads.

        A snapshot list rather than a live view: a caller that walked the OrderedDict directly
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


# --- the two production doors onto RESULTS ------------------------------------------------------
#
# Both live HERE, next to the store, because they are the same fact read twice: what a run puts in
# and what a newly-opened window takes out have to agree about the key or the cache is a write-only
# log. Until 2026-08-05 `RESULTS` had a writer nowhere and a reader nowhere -- exercised only by
# tests -- so a second window opening a region another window had already computed showed nothing,
# and the only way to see the result was to run the operator again.
#
# `cache_scope` is imported in the body rather than at module scope on purpose: this module is pure
# Python (no Qt, no numpy) and importing it must stay free, which `squidmip._plate` is not.

def acquisition_version(reader: Any) -> str:
    """What ``version`` means for a static acquisition: WHICH acquisition.

    ``ResultCache``'s key has always carried a version and every caller has always passed the
    default ``0``. A region id is not unique across datasets -- every plate has a ``B2`` -- so
    with one constant version, ingesting a second acquisition into the same process would replay
    the first one's ``B2`` result into a window showing the second one's ``B2``. Reusing the field
    for the acquisition's identity is what its own docstring reserves it for ("baked in now so
    re-parenting to the live Squid source is trivial later"), and it is the same token
    ``_mosaic_source`` keys its plane cache on, so the two caches agree about what "the same
    acquisition" is.
    """
    from squidmip._mosaic_source import _source_token

    return _source_token(reader)


def cache_operator_result(op: str, result, version: Any = 0) -> None:
    """File one finished operator result under ``(its region, that operator, this acquisition)``.

    ``op`` is the LAYER KEY, not the bare operator name, so a run scoped to an exploration tab
    (``mip@preview:…``) is a different entry from the plate-wide ``mip``. That is the same
    distinction the layer stack makes, and folding them together here would let a tab's subset
    result be replayed into a window as if it were the whole-plate run.
    """
    from squidmip._plate import cache_scope

    RESULTS.put(cache_scope(str(result.region_id)),
                RecipeChain.of(Recipe.operator(str(op))), result, version=version)


def cached_operator_results(region: str, version: Any = 0) -> "list[tuple[str, Result]]":
    """``[(op, result)]`` already computed for *region* of this acquisition, LRU first.

    Order is :meth:`ResultCache.entries`'s, which is a snapshot: a caller delivering these into a
    window must not have the store reorder under it while it iterates.
    """
    from squidmip._plate import cache_scope

    scope, version = cache_scope(str(region)), str(version)
    return [(str(e.chain.recipes[0].name), e.result) for e in RESULTS.entries()
            if e.scope == scope and e.version == version and e.chain.recipes]

# There is no chain clipboard here. `CLIPBOARD` / `copy_chain` / `paste_chain` lived at the bottom
# of this module until 2026-08-06 with ZERO callers in the package, the tests, the tools or the
# CLI, described as generalising `_region_viewer._LUT_CLIPBOARD` -- which is still the module-level
# dict that both Copy LUTs and Paste LUTs actually use. A second, unwired copy/paste buffer sitting
# in the recipe module is the "what the user last chose, owned twice" shape wearing a plausible
# name: the next person to add Copy Recipe would reasonably wire it here and end up with two
# clipboards. Copy/paste of a chain belongs wherever the LUT clipboard is when it is built.

