"""Parallel, streaming plate engine and the single pluggable operator registry."""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from squidxplorer._compose import CHAIN_CHARS, compose_operator, is_chain_expression
from squidxplorer._recipe import CHAIN_SEPARATOR, RecipeChain
from squidxplorer.projection import (
    INTENSITY,
    PLANE_OP,
    REGION_OP,
    Z_REDUCER,
    MissingDependency,
    missing_requirements,
    normalise_consumes,
    normalise_produces,
    normalise_requires,
    plane_op,
    scope_wells,
    project,
    project_reference,
    project_well,
    requirement_refusal,
    select_fovs,
)

if TYPE_CHECKING:  # avoid import cost / cycle at runtime
    from squidxplorer.reader import SquidReader

Projector = Callable[[Iterable[np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class Param:
    """One parameter a registry entry declares it can be run with."""

    name: str
    default: Any
    blurb: str = ""


class MissingOperatorDependency(MissingDependency):
    """A registered operator's optional package is not importable."""


@dataclass(frozen=True)
class Operator:
    """A registry entry: name, callable, and the four declarations the engine dispatches on."""
    name: str
    fn: Projector
    consumes: frozenset[str]
    produces: str = INTENSITY
    params: tuple[Param, ...] = ()
    factory: Optional[Callable[..., Projector]] = None
    #: Importable module names this operator needs, e.g. ``("petakit",)``.
    requires: tuple[str, ...] = ()

    def available(self) -> tuple[bool, str]:
        """``(ok, reason_if_not)`` — are this operator's declared packages importable right now?"""
        missing = missing_requirements(self.requires)
        if missing:
            return False, requirement_refusal("operator", self.name, missing)
        return True, ""

    def bind(self, operator_kwargs: Optional[dict] = None) -> Projector:
        """The callable to run, with *operator_kwargs* applied; refuses unknown params by name."""
        ok, why = self.available()
        if not ok:
            raise MissingOperatorDependency(why)
        return self.with_params(operator_kwargs)

    def with_params(self, operator_kwargs: Optional[dict] = None) -> Projector:
        """The callable *operator_kwargs* names, without the availability check :meth:`bind` makes."""
        if not operator_kwargs:
            return self.fn
        if self.factory is None:
            raise ValueError(
                f"operator {self.name!r} declares no parameters, so it cannot be run with "
                f"{sorted(operator_kwargs)}. Its behaviour is fixed at registration; register a "
                f"named variant, or give the entry params= and a factory."
            )
        known = {p.name for p in self.params}
        unknown = sorted(set(operator_kwargs) - known)
        if unknown:
            raise ValueError(
                f"operator {self.name!r} has no parameter {unknown[0]!r}; it declares "
                f"{sorted(known)}."
            )
        return self.factory(**operator_kwargs)

    def defaults(self) -> dict:
        """``{name: default}`` for every declared parameter."""
        return {p.name: p.default for p in self.params}


# Environment faults on_error must not absorb: they raise identically for every well.
_NOT_A_WELL_FAULT = (ImportError, MissingDependency)

# THE operator table — name -> Operator. `consumes` is the dispatch; nothing branches on a name.
_OPERATORS: dict[str, Operator] = {
    "mip": Operator("mip", project, Z_REDUCER),
    "reference": Operator("reference", project_reference, Z_REDUCER),
    # Identity plane-op: keeps every z plane unchanged, so stitching can run per z-level.
    "keepz": Operator("keepz", plane_op(lambda plane: plane), PLANE_OP),
}


def _default_workers() -> int:
    """Thread count when the caller doesn't specify — affinity/cgroup aware, never hardcoded."""
    n = os.process_cpu_count() if hasattr(os, "process_cpu_count") else None
    if not n and hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    if not n:
        n = os.cpu_count()
    return n or 1


def add_projector(name: str, projector: Projector, *, consumes=None, produces=None,
                  params: Sequence[Param] = (), requires=()) -> None:
    """Add a named operator so it can be selected by name in :func:`project_plate`."""
    _declare(name, projector, consumes=consumes, produces=produces, params=params,
             requires=requires, region=False)


def add_region_operator(name: str, operator, *, produces=None, params: Sequence[Param] = (),
                        requires=()) -> None:
    """Add a named REGION operator — one that eats a whole well's FOVs — to the same table."""
    _declare(name, operator, consumes=REGION_OP, produces=produces, params=params,
             requires=requires, region=True)


def _declare(name: str, fn, *, consumes, produces, params, requires, region: bool) -> None:
    """Validate ONE operator and file it in :data:`_OPERATORS`. The only writer of that table."""
    kind = "region operator" if region else "projector"
    if not name:
        raise ValueError(f"{kind} name must be a non-empty string")
    reserved = sorted(set(name) & set(CHAIN_CHARS))
    if reserved:
        raise ValueError(
            f"{kind} name {name!r} contains {reserved[0]!r}, which is chain punctuation: "
            f"'{CHAIN_CHARS}' spell a COMPOSITION ('flatfield + decon + mip', "
            "see squidxplorer._compose). A name carrying one would read as an expression everywhere a "
            "chain is written down — in a console line, a CLI flag, a pasted recipe script — so it "
            "is refused here rather than left to be ambiguous there.")
    if not callable(fn):
        raise ValueError(f"{kind} for {name!r} is not callable: {fn!r}")
    if name in _OPERATORS:
        raise ValueError(
            f"{kind} {name!r} is already defined; pick a distinct name "
            f"(defined: {runnable_operators()})."
        )
    declared = tuple(params)
    for p in declared:
        if not isinstance(p, Param) or not p.name:
            raise ValueError(
                f"{kind} {name!r}: params must be Param(name, default) records with a "
                f"non-empty name; got {p!r}")
    names = [p.name for p in declared]
    if len(set(names)) != len(names):
        raise ValueError(
            f"{kind} {name!r} declares a parameter twice: {sorted(names)}; a duplicate makes "
            "operator_kwargs ambiguous")

    factory: Optional[Callable[..., Projector]] = None
    if declared:
        factory = fn
        fn = factory(**{p.name: p.default for p in declared})
        if not callable(fn):
            raise ValueError(
                f"{kind} factory for {name!r} returned {fn!r}, which is not callable. "
                "With params=, the registered object is a FACTORY: it is called with the declared "
                "defaults and must return the operator callable.")
    if region:
        axes = REGION_OP                       # stamped, never inferred: see add_region_operator
    else:
        if consumes is None:
            consumes = getattr(fn, "consumes", Z_REDUCER)
        axes = normalise_consumes(consumes)
    if produces is None:
        produces = getattr(fn, "produces", INTENSITY)
    _OPERATORS[name] = Operator(
        name, fn, axes, normalise_produces(produces),
        declared, factory, normalise_requires(requires),
    )


def runnable_operators() -> list[str]:
    """EVERY operator this application can run, sorted."""
    return sorted(_OPERATORS)


def available_projectors() -> list[str]:
    """Registered operators with the ``Iterable[plane] -> plane`` shape — what ``project_plate`` runs."""
    return sorted(n for n, op in _OPERATORS.items() if "fov" not in op.consumes)


def is_region_operator(name) -> bool:
    """Does *name* eat a whole well's FOVs — i.e. must it be run by ``stitch_plate``?"""
    try:
        return "fov" in _resolve_operator(name).consumes
    except (KeyError, TypeError, ValueError):
        return False


def available_region_operators() -> list[str]:
    """Registered operators that eat a whole well's FOVs — what :func:`squidxplorer.stitch_plate` runs."""
    return sorted(n for n, op in _OPERATORS.items() if "fov" in op.consumes)


def operator_available(name: str) -> tuple[bool, str]:
    """``(ok, reason_if_not)`` — can this operator actually run right now?"""
    try:
        op = _resolve_operator(name)
    except (KeyError, TypeError, ValueError) as exc:
        return False, str(exc).strip('"')
    return op.available()


def operator_requires(name: str) -> tuple[str, ...]:
    """The modules a registered operator declares it needs — ``()`` when it needs nothing extra."""
    return _resolve_operator(name).requires


def operator_consumes(name: str) -> frozenset[str]:
    """Return the axis a registered operator consumes — ``frozenset()`` (plane-op) or ``{"z"}``."""
    return _resolve_operator(name).consumes


def operator_produces(name: str) -> str:
    """Return what a registered operator's pixels MEAN — ``"intensity"`` or ``"labels"``."""
    return _resolve_operator(name).produces


def operator_params(name: str) -> tuple[Param, ...]:
    """The parameters a registered operator declares — ``()`` when its behaviour is fixed."""
    return _resolve_operator(name).params


def bind_operator(name: str, operator_kwargs: Optional[dict] = None) -> Projector:
    """Resolve *name* and apply *operator_kwargs*, raising on an unknown name or parameter."""
    return _resolve_operator(name).bind(operator_kwargs)


def _resolve_operator(name) -> Operator:
    """Look up an operator by name OR by chain expression, failing loud on an unknown key."""
    if isinstance(name, RecipeChain):
        return compose_operator(name, _resolve_operator)
    if isinstance(name, (list, tuple)):
        spelled = f" {CHAIN_SEPARATOR} ".join(str(part) for part in name)
        raise TypeError(
            f"a projector is named by one string, got {type(name).__name__} {name!r}. A chain is "
            f"written as a string too: projector={spelled!r}. See squidxplorer._compose.")
    if not isinstance(name, str):
        raise TypeError(
            f"a projector is named by a string or a RecipeChain, got {type(name).__name__}: "
            f"{name!r}.")
    operator = _OPERATORS.get(name)
    if operator is not None:
        return operator
    if is_chain_expression(name):
        return compose_operator(name, _resolve_operator)
    raise KeyError(
        f"unknown operator {name!r}; available: {runnable_operators()}. "
        "Add new modes with squidxplorer.add_projector(name, fn) — or "
        "squidxplorer.add_region_operator(name, fn) for one that fuses a whole well — or chain "
        "registered ones with '+' (e.g. 'flatfield+decon+mip')."
    )


def project_plate(
    reader: "SquidReader",
    *,
    n_fovs: Optional[int] = 1,
    workers: int | None = None,
    projector: str = "mip",
    on_error=None,
    regions=None,
    operator_kwargs: Optional[dict] = None,
) -> Iterator[tuple[str, int, np.ndarray]]:
    """Project every selected well in parallel, streaming ``(region, fov, image)`` results.

    The in-flight window is bounded at *workers* wells, so peak memory is flat in plate size.
    ``on_error(region, fov, exc)`` opts in to per-well fault isolation for data faults only.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    n_workers = workers if workers is not None else _default_workers()

    op = _resolve_operator(projector)
    # A region operator is a whole-well callable; this loop hands out planes.
    if "fov" in op.consumes:
        raise ValueError(
            f"{projector!r} consumes fov — it fuses a whole well's FOVs and takes "
            "(reader, region, fovs), which is not what project_plate hands an operator. Run it "
            f"with squidxplorer.stitch_plate(reader, operator={projector!r}).")
    fn = bind_operator(projector, operator_kwargs)

    # Warm the reader's lazy state single-threaded before fan-out.
    meta = reader.metadata
    wells = scope_wells(meta, n_fovs, regions)
    tasks: Iterator[tuple[str, int]] = (
        (region, fov) for region, fovs in wells.items() for fov in fovs
    )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        in_flight: dict = {}

        def _submit_next() -> bool:
            """Submit the next well, if any; return False when the task stream is exhausted."""
            try:
                region, fov = next(tasks)
            except StopIteration:
                return False
            future = pool.submit(project_well, reader, region, fov,
                                 reduce=fn, consumes=op.consumes)
            in_flight[future] = (region, fov)
            return True

        for _ in range(n_workers):  # prime the window
            if not _submit_next():
                break

        while in_flight:
            done, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                region, fov = in_flight.pop(future)
                _submit_next()  # slide the window forward first, so a SKIPPED well still refills it
                try:
                    image = future.result()
                except _NOT_A_WELL_FAULT:
                    # A missing package is not a corrupt well: it fails every well identically.
                    raise
                except Exception as exc:
                    if on_error is None:
                        raise
                    on_error(region, fov, exc)      # record + SKIP this well, keep going
                    continue
                yield region, fov, image
