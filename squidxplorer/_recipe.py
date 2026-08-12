"""Recipes, content-addressed results, and copy/paste of transforms.

A Recipe is one transform (operator or LUT); a RecipeChain is both the cache key and the
runnable expression (``label()`` and ``parse()`` are inverses). Pure Python: no Qt, no numpy.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from squidxplorer._result import Result

#: Recipe kinds: OPERATOR is a data transform, LUT a contrast transform.
OPERATOR = "operator"
LUT = "lut"

#: What separates two steps of a chain, in the one spelling that is both read and written.
CHAIN_SEPARATOR = "+"

#: What an EMPTY chain is called — this application's word for the untransformed layer.
RAW = "raw"


@dataclass(frozen=True)
class Recipe:
    """One transform; ``key()`` is a stable content hash of kind/name/params."""

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
        """What a human reads: ``mip``, ``decon(sigma=2.0)``, ``contrast(DAPI,GFP)``."""
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

    @staticmethod
    def operator(key: str, **params: Any) -> "Recipe":
        return Recipe(OPERATOR, str(key), dict(params))

    @staticmethod
    def contrast(per_channel: dict) -> "Recipe":
        """A LUT recipe: ``per_channel`` maps channel name -> {"clim": (lo, hi), "cmap": <name>}."""
        return Recipe(LUT, "contrast", {"per_channel": dict(per_channel)})


def _split_top_level(text: str, separator: str, whole: str) -> "list[str]":
    """Split *text* on *separator*, ignoring separators nested inside parentheses."""
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
    """A Python literal when it is one, else the bare string — what label() writes for strings."""
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
    """An ORDERED list of recipes: the cache key, the paste script, and the run expression."""

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
        """``mip + decon(sigma=2.0)`` — the chain, in order, in words; empty is :data:`RAW`."""
        if not self.recipes:
            return RAW
        return f" {CHAIN_SEPARATOR} ".join(r.label() for r in self.recipes)

    def __str__(self) -> str:
        return self.label()

    @staticmethod
    def parse(text: str) -> "RecipeChain":
        """``"flatfield + decon + mip"`` -> the chain. The exact inverse of :meth:`label`."""
        raw = str(text).strip()
        if not raw or raw == RAW:
            return RecipeChain()
        return RecipeChain(tuple(
            _parse_step(step, raw) for step in _split_top_level(raw, CHAIN_SEPARATOR, raw)))

    def to_script(self) -> str:
        """A tiny, human-readable, re-loadable JSON script."""
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
    """One cache entry; the chain OBJECT rides along because a hash cannot be un-hashed."""

    scope: str
    version: str
    chain: RecipeChain
    result: Result


class ResultCache:
    """Flat, content-addressed result store keyed ``(scope, version, chain.key())``, bounded LRU.

    Values are :class:`squidxplorer._result.Result` only — a cached result must carry its own
    extent and substance, so the plate draws what each cell declares.
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
                "ResultCache stores squidxplorer._result.Result, not "
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
        """Every entry, least-recently-used first, as a snapshot (``get`` reorders the dict)."""
        return list(self._d.values())

    def clear(self) -> None:
        self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


#: The process-wide result cache: one store for the whole app.
RESULTS = ResultCache()


def acquisition_version(reader: Any) -> str:
    """What ``version`` means for a static acquisition: WHICH acquisition."""
    from squidxplorer._mosaic_source import _source_token

    return _source_token(reader)


def cache_operator_result(op: str, result, version: Any = 0) -> None:
    """File one finished operator result under ``(its region, that operator, this acquisition)``.

    ``op`` is the LAYER KEY, not the bare operator name, so a tab-scoped run stays distinct.
    """
    from squidxplorer._plate import cache_scope

    RESULTS.put(cache_scope(str(result.region_id)),
                RecipeChain.of(Recipe.operator(str(op))), result, version=version)


def cached_operator_results(region: str, version: Any = 0) -> "list[tuple[str, Result]]":
    """``[(op, result)]`` already computed for *region* of this acquisition, LRU first."""
    from squidxplorer._plate import cache_scope

    scope, version = cache_scope(str(region)), str(version)
    return [(str(e.chain.recipes[0].name), e.result) for e in RESULTS.entries()
            if e.scope == scope and e.version == version and e.chain.recipes]
