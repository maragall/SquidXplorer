"""Recipes, content-addressed results, and copy/paste of transforms.

The design we drew (2026-07-24): the window TREE is navigation only. RESULTS live in a flat,
CONTENT-ADDRESSED cache keyed by the data scope plus the op-chain, so two windows over the same well
with the same chain resolve to the SAME entry. Cross-propagation is then free and lazy, with no
window-to-window messaging, no signal recursion, and no need to wake halted windows.

A RECIPE is the serializable unit you copy/paste: an OPERATOR (a data transform, key + params) or a
LUT (a contrast transform). Same mechanism, which is exactly why "copy LUTs" and "copy an operator"
are one system rather than two. A CHAIN of recipes, e.g. [stitch, decon3d] or [contrast], is BOTH
the content-address of a result AND the script you paste onto another view or the plate.

This module is pure Python, no Qt, no numpy: the model, testable in isolation. The GUI layer builds
recipes from what a window shows and applies them by registering keys the cache computes lazily.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

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
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._d: "OrderedDict[tuple, Any]" = OrderedDict()
        self._max = max(1, int(max_entries))

    @staticmethod
    def _k(scope: str, chain: RecipeChain, version: Any) -> tuple:
        return (str(scope), str(version), chain.key())

    def get(self, scope: str, chain: RecipeChain, version: Any = 0) -> Optional[Any]:
        k = self._k(scope, chain, version)
        if k in self._d:
            self._d.move_to_end(k)          # most-recently used
            return self._d[k]
        return None

    def put(self, scope: str, chain: RecipeChain, value: Any, version: Any = 0) -> None:
        k = self._k(scope, chain, version)
        self._d[k] = value
        self._d.move_to_end(k)
        while len(self._d) > self._max:
            self._d.popitem(last=False)      # evict least-recently used

    def has(self, scope: str, chain: RecipeChain, version: Any = 0) -> bool:
        return self._k(scope, chain, version) in self._d

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
