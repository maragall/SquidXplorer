"""Operator COMPOSITION: a chain of registered operators, resolved into one registry-shaped operator.

``squidmip._recipe.RecipeChain`` has documented ``mip + decon(sigma=2.0)`` and the rule that order
matters since 2026-07-24, and until now **nothing executed it**. Every call shape people reached for
was refused: ``projector=["flatfield", "mip"]`` raised ``TypeError: unhashable type: 'list'``,
``"flatfield+mip"`` raised ``KeyError: unknown projector``, and ``bind_operator("mip", {"then":
"decon"})`` raised ``ValueError: operator 'mip' declares no parameters``. The blocker was never a
missing convenience API: ``_OPERATORS`` is ``dict[str, Operator]`` and ``project_well`` applies
exactly ONE ``reduce`` per ``(t, c, z_group)``, so there was no seam a second operator could sit in.

WHAT THIS MODULE DOES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------------
It adds no seam to the loop. It builds ONE :class:`squidmip._engine.Operator` out of several, whose
four declarations are DERIVED from its parts, and hands that to the loop that already exists::

    project_plate(reader, projector="flatfield + decon + mip")
    stitch_region(reader, "B2", fovs, projector="bgsub+mip", correct_illumination=False)
    write_plate(reader, out, projector="bgsub+mip")

So a chain is accepted anywhere a name is, by every caller, with no edit to any of them: the engine,
the writer, the stitcher, the CLI's ``--projector``, the ``run_operator`` command. That is the whole
reason the expression is a STRING and not a new argument -- ``projector=`` is already threaded
through six call paths, and a parallel ``projectors=`` list would have to be threaded through all
six again and then reconciled with the one that was already there.

The expression is exactly :meth:`squidmip._recipe.RecipeChain.label` -- the words the console and a
legend already print for a chain -- and :meth:`~squidmip._recipe.RecipeChain.parse` is its inverse.
One spelling for the label, the cache key, the paste script and the run.

THE THREE COMBINATIONS THAT ARE REFUSED, AND WHY EACH IS REFUSED RATHER THAN REORDERED
--------------------------------------------------------------------------------------
``consumes`` decides the engine's loop and the output shape, so it also decides what can follow
what. This is arithmetic on the declarations, not a table of operator names (which
``tests/test_operator_declaration.py`` fails the build over):

* **plane-op -> plane-op** composes. Both keep ``Nz``, so the second maps over what the first
  produced: ``flatfield + decon`` is ``(T, C, Nz, Y, X)``.
* **plane-op -> z-reducer** composes. The reducer consumes the stack the plane-ops produced:
  ``flatfield + decon + mip`` is ``(T, C, 1, Y, X)``, and the plane-ops are applied LAZILY as the
  reducer pulls, so the whole stack is still never resident.
* **z-reducer -> anything** is REFUSED. After a reducer there is one plane and no stack, so the
  step after it is not "mapped over z" in any sense the declaration can express; ``mip + mip`` is
  not a second projection and ``mip + decon`` is deconvolution of a projection, which is a
  different scientific claim from the deconvolution the chain appears to promise. A z-reducer is
  the LAST step or the only one.
* **labels -> anything** is REFUSED. ``produces="labels"`` means the pixels are integer OBJECT IDS,
  so a following operator would do arithmetic on names: the mean of label 12 and label 37 is label
  24, an object that does not exist. Same argument ``_stitch`` already makes when it refuses to
  feather labels. An operator that produces labels is the LAST step.
* a **z-SELECTING** step (``select_index``: ``reference``) is REFUSED inside a chain. It is not a
  shape problem: ``project_well`` solves the focus ONCE per ``(t, fov)`` on RAW planes and shares
  that z across channels, outside the operator, so a chain around it would never touch the planes
  it picks. Composing it would silently drop every other step.

Refused BY NAME, with the reason and the fix in the message, and never silently reordered: a run
that quietly ran ``decon + mip`` when the user typed ``mip + decon`` is a wrong result that looks
right, which is the failure mode this whole registry is shaped to avoid.

WHAT A COMPOSED OPERATOR DECLARES
---------------------------------
============ =================================================================================
declaration  derived as
============ =================================================================================
``consumes`` the UNION of the parts'. Because only the last step may consume z, that is ``{"z"}``
             iff the last step is a z-reducer -- so the engine's loop and the output shape follow
             from the chain with no special case.
``produces`` the LAST step's. Nothing may follow a non-intensity step, so the intermediate ones
             are all ``"intensity"`` by construction.
``params``   every part's, NAMESPACED ``<step>.<param>`` (``spot.min_area_px``), defaulted to what
             the expression already fixed. So ``operator_kwargs`` reaches one step of a chain
             unambiguously, and ``list_operators``-shaped readers see a chain like any operator.
``requires`` the union, in first-declared order. One refusal at bind time naming every missing
             package, before a single well is read.
============ =================================================================================

Two ATTRIBUTES are carried too, both because a generic reader elsewhere depends on them:

* ``corrects_illumination`` -- true if ANY part corrects. ``_stitch.stitch_region`` reads it off
  the callable to refuse the flat-field double-apply (measured: correcting twice changes 88.6% of
  pixels by up to 23 counts, silently). A chain CONTAINING ``flatfield`` must not defeat that
  guard, so the composed callable declares what its parts declare.
* ``for_channel`` -- present if ANY part is channel-specialisable. ``project_well`` calls it once
  per channel; the composition rebuilds itself with each part specialised, so ``bgsub + decon``
  still deconvolves 638 with the 638 PSF.

A one-step chain with no parameters resolves to the registry entry ITSELF, the same object the
table has always held. ``projector="mip"`` is therefore byte-identical to what it was, including
``reference``'s ``select_index``: nothing routes through this module unless a chain was asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from squidmip._recipe import Recipe, RecipeChain
from squidmip.projection import INTENSITY, bind_channel

__all__ = ["CHAIN_CHARS", "compose_operator", "is_chain_expression"]

#: The characters that make a ``projector=`` string an EXPRESSION rather than a table key. A
#: registered name may contain neither (``add_projector`` refuses one that does), so this test can
#: never shadow a real entry, and an unknown plain name still gets the registry's own KeyError
#: rather than a parser's.
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
    """Resolve *expression* into ONE :class:`squidmip._engine.Operator`.

    *expression* is a chain string (``"flatfield + decon + mip"``) or an already parsed
    :class:`~squidmip._recipe.RecipeChain`.

    *lookup* resolves one bare operator NAME to its registry record. Passed in rather than reached
    for, so the resolution RULE ("a registered name wins, then a chain") lives in exactly one place
    -- :func:`squidmip._engine._resolve_operator`, which is the door every ``projector=`` arrives
    at. A second copy of that rule here is how the CLI and the engine would come to disagree about
    what a string means.

    Raises
    ------
    KeyError
        From *lookup*, when a step names an operator that is not registered. Unchanged wording, so
        a typo inside a chain reads exactly like a typo outside one.
    ValueError
        On an empty chain, a repeated step, or one of the combinations this module's docstring
        refuses. Every message names the two operators involved and what to do instead.
    """
    from squidmip._engine import Operator, Param

    chain = expression if isinstance(expression, RecipeChain) else RecipeChain.parse(str(expression))
    recipes = chain.recipes
    if not recipes:
        raise ValueError(
            f"{str(expression)!r} names no operator. An operator chain is one or more registered "
            "operators separated by '+', e.g. 'flatfield + decon + mip'.")

    parts = [(recipe, lookup(recipe.name)) for recipe in recipes]

    # A BARE NAME IS THE TABLE ENTRY, not a one-step composition of it. The identity matters: it is
    # what keeps `projector="mip"` the exact object the registry has always held, `reference`'s
    # `select_index` reachable, and this module out of every existing run's path.
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
    """Refuse the combinations the declarations say cannot mean anything. See the module docstring.

    Every test here reads a DECLARATION (``consumes``, ``produces``, ``select_index``) and never an
    operator's name, which is the property ``tests/test_operator_declaration.py`` asserts over the
    package's AST. A chain of operators nobody here has heard of is refused on the same grounds as
    a chain of the shipped ones.
    """
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
                "stitch_plate, then run the chain over the result.")

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

    A named function and not an inline generator expression, because an inline one would close over
    the loop variable in :func:`_composed_callable` and, being lazy, would run EVERY step with
    whichever callable the loop happened to finish on. Binding both arguments in this frame is what
    makes the composition correct; laziness is what keeps it bounded, so the reducer at the end of
    ``flatfield + decon + mip`` still folds one plane at a time and the stack is never resident.
    """
    return (fn((plane,)) for plane in stream)


def _composed_callable(steps: Sequence[_Step], label: str, consumes: frozenset):
    """Build the ``Iterable[plane] -> plane`` callable that runs *steps* in order.

    ONE callable shape out, exactly as in: nothing downstream can tell a composition from a shipped
    operator, which is what lets ``project_well``, ``write_plate`` and ``stitch_region`` take one
    with no edit. A z-reducing step swallows the stream and yields one plane; a plane-op step is
    mapped over it. Both cases end with exactly one plane, and anything else is refused loud rather
    than silently indexed into.
    """
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

    # THE FLAT-FIELD DOUBLE-APPLY GUARD, carried through the composition. `_stitch.stitch_region`
    # reads this attribute off the callable to refuse "the read path corrects AND the operator
    # corrects" — measured at 88.6% of pixels changed by up to 23 counts, with nothing downstream
    # able to tell. A chain containing `flatfield` corrects its input just as much as `flatfield`
    # does, so it says so, and the guard holds without knowing that compositions exist.
    if any(getattr(step.fn, "corrects_illumination", False) for step in steps):
        _run.corrects_illumination = True

    # THE CHANNEL SPECIALISATION, carried through the same way. `project_well` calls `for_channel`
    # once per channel; a composition rebuilds itself with each part specialised, so `bgsub + decon`
    # still gets the 638 PSF for 638 rather than one module-level default for all four.
    if any(hasattr(step.fn, "for_channel") for step in steps):
        def _for_channel(path, channel):
            return _composed_callable(
                [_Step(step.name, bind_channel(step.fn, path, channel), step.reduces)
                 for step in steps], label, consumes)

        _run.for_channel = _for_channel
    return _run


def _rebinder(chain: RecipeChain, lookup: Callable[[str], Any]):
    """The composed entry's ``factory``: apply NAMESPACED ``operator_kwargs`` and rebuild the chain.

    ``Operator.bind`` validates the names against the declared ``params`` first, so every key that
    arrives here is ``<step>.<param>`` for a step of this chain. Rebuilding through
    :func:`compose_operator` rather than patching the built callable means a parameterised run and
    a run whose parameters were written into the expression are the SAME object — there is one way
    a chain becomes a callable, so the two cannot drift.
    """
    def _rebuild(**operator_kwargs):
        merged = {recipe.name: dict(recipe.params) for recipe in chain.recipes}
        for key, value in operator_kwargs.items():
            step, _, parameter = key.partition(".")
            merged[step][parameter] = value
        return compose_operator(
            RecipeChain(tuple(Recipe.operator(recipe.name, **merged[recipe.name])
                              for recipe in chain.recipes)), lookup).fn

    return _rebuild
