"""Operator composition: a chain expression ("flatfield + decon + mip") resolved into ONE
registry-shaped Operator whose consumes/produces/params/requires are derived from its parts.
Impossible combinations are refused by declaration, never reordered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from squidxplorer._recipe import Recipe, RecipeChain
from squidxplorer.projection import INTENSITY, bind_channel

__all__ = ["CHAIN_CHARS", "compose_operator", "is_chain_expression"]

#: The characters that make an ``operator=`` string an EXPRESSION rather than a table key.
#: A registered name may contain neither, so this test can never shadow a real entry.
CHAIN_CHARS = "+()"


def is_chain_expression(name: Any) -> bool:
    """Is *name* written as a chain (``"a+b"``, ``"spot(min_area_px=80)"``) rather than a bare key?"""
    return isinstance(name, str) and any(char in name for char in CHAIN_CHARS)


@dataclass(frozen=True)
class _Step:
    """One resolved step: the part's name, the callable to run, and whether it consumes the stack."""

    name: str
    fn: Callable[[Iterable[Any]], Any]
    reduces: bool


def compose_operator(expression: Any, lookup: Callable[[str], Any]) -> Any:
    """Resolve *expression* (a chain string or RecipeChain) into ONE Operator via *lookup*."""
    from squidxplorer._engine import Operator, Param

    chain = expression if isinstance(expression, RecipeChain) else RecipeChain.parse(str(expression))
    recipes = chain.recipes
    if not recipes:
        raise ValueError(
            f"{str(expression)!r} names no operator. An operator chain is one or more registered "
            "operators separated by '+', e.g. 'flatfield + decon + mip'.")

    parts = [(recipe, lookup(recipe.name)) for recipe in recipes]

    # A bare name is the table entry ITSELF, not a one-step composition of it — nothing
    # routes through this module unless a chain was asked for.
    if len(parts) == 1 and not recipes[0].params:
        return parts[0][1]

    label = chain.label()
    _refuse_impossible(parts, label)

    steps = [_Step(recipe.name, operator.with_params(recipe.params), "z" in operator.consumes)
             for recipe, operator in parts]
    consumes = frozenset().union(*(operator.consumes for _r, operator in parts))
    return Operator(
        name=label,
        fn=_composed_callable(steps, label, consumes),
        consumes=consumes,
        produces=parts[-1][1].produces,
        params=tuple(Param(f"{recipe.name}.{p.name}", recipe.params.get(p.name, p.default), p.blurb)
                     for recipe, operator in parts for p in operator.params),
        factory=_rebinder(chain, lookup),
        requires=tuple(dict.fromkeys(
            module for _r, operator in parts for module in operator.requires)),
    )


def _refuse_impossible(parts: Sequence[tuple], label: str) -> None:
    """Refuse combinations the declarations say cannot mean anything — never by operator name."""
    seen: dict = {}
    for position, (recipe, operator) in enumerate(parts):
        if recipe.name in seen:
            raise ValueError(
                f"cannot compose {label!r}: {recipe.name!r} appears twice (step {seen[recipe.name]} "
                f"and step {position + 1}), so a parameter named '{recipe.name}.<x>' would address "
                "both and neither. Run it once with the parameters you want.")
        seen[recipe.name] = position + 1

        if "fov" in operator.consumes:
            raise ValueError(
                f"cannot compose {label!r}: {recipe.name!r} consumes fov — it takes a whole well "
                "(reader, region, fovs) and returns its fused mosaic, so there are no planes to "
                "hand the other steps and no plane stream to join it to. Run it on its own with "
                "run_plate, then run the chain over the result.")

        if getattr(operator.fn, "select_index", None) is not None:
            raise ValueError(
                f"cannot compose {label!r}: {recipe.name!r} SELECTS one z per (t, fov) and reads "
                "that same z for every channel, so the channels of one FOV overlay. That solve "
                "happens in project_well, on RAW planes, OUTSIDE the operator — so the other steps "
                "of this chain would never touch the planes it picks, and composing it would drop "
                "them silently. Run it on its own.")

        if position == len(parts) - 1:
            continue
        following = parts[position + 1][0].name

        if "z" in operator.consumes:
            raise ValueError(
                f"cannot compose {label!r}: {recipe.name!r} consumes z (it reduces the stack to one "
                f"plane), so by the time {following!r} runs there is no z left to map it over. A "
                f"z-reducer is the LAST step of a chain or the only one — write "
                f"'{following} + {recipe.name}' if you meant to {following} the planes and then "
                f"{recipe.name} them.")

        if operator.produces != INTENSITY:
            raise ValueError(
                f"cannot compose {label!r}: {recipe.name!r} produces {operator.produces!r} — "
                f"integer OBJECT IDS, not a measurement of light — and {following!r} would read "
                "them as pixel values, so the mean of label 12 and label 37 becomes label 24, an "
                "object that does not exist. An operator that produces labels is the LAST step of a "
                "chain.")


def _map_planes(fn: Callable, stream: Iterable) -> Iterable:
    """Apply a plane-op to every plane of *stream*, LAZILY.

    A named function, not an inline genexpr: an inline one would close over the loop
    variable in _composed_callable and run every step with the last callable.
    """
    return (fn((plane,)) for plane in stream)


def _composed_callable(steps: Sequence[_Step], label: str, consumes: frozenset):
    """Build the ``Iterable[plane] -> plane`` callable that runs *steps* in order."""
    def _run(planes: Iterable) -> Any:
        stream: Iterable = iter(planes)
        for step in steps:
            stream = iter((step.fn(stream),)) if step.reduces else _map_planes(step.fn, stream)
        stream = iter(stream)
        result = next(stream, None)
        if result is None:
            raise ValueError(f"operator chain {label!r} requires at least one plane; got an empty "
                             "iterable.")
        if next(stream, None) is not None:
            raise ValueError(
                f"operator chain {label!r} consumes no axis (every step is a plane-op) and was "
                "handed more than one plane. project_well groups by the chain's own `consumes`, so "
                "this is a seam bug, not a data fault — it would silently discard every plane but "
                "the first.")
        return result

    _run.__name__ = label
    _run.consumes = consumes

    # Carry the flat-field double-apply guard through: a chain containing `flatfield`
    # corrects its input just as much as `flatfield` does.
    if any(getattr(step.fn, "corrects_illumination", False) for step in steps):
        _run.corrects_illumination = True

    # Carry channel specialisation through: rebuild with each part specialised per channel.
    if any(hasattr(step.fn, "for_channel") for step in steps):
        def _for_channel(path, channel):
            return _composed_callable(
                [_Step(step.name, bind_channel(step.fn, path, channel), step.reduces)
                 for step in steps], label, consumes)

        _run.for_channel = _for_channel
    return _run


def _rebinder(chain: RecipeChain, lookup: Callable[[str], Any]):
    """The composed entry's ``factory``: apply namespaced ``operator_kwargs`` and rebuild the chain."""
    def _rebuild(**operator_kwargs):
        merged = {recipe.name: dict(recipe.params) for recipe in chain.recipes}
        for key, value in operator_kwargs.items():
            step, _, parameter = key.partition(".")
            merged[step][parameter] = value
        return compose_operator(
            RecipeChain(tuple(Recipe.operator(recipe.name, **merged[recipe.name])
                              for recipe in chain.recipes)), lookup).fn

    return _rebuild
