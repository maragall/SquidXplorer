"""Parallel, streaming plate engine and the single pluggable operator registry."""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from squidxplorer.projection import (
    INTENSITY,
    REGION_OP,
    Z_REDUCER,
    MissingDependency,
    missing_requirements,
    normalise_consumes,
    normalise_produces,
    normalise_requires,
    scope_wells,
    project,
    project_well,
    requirement_refusal,
)

if TYPE_CHECKING:  # avoid import cost / cycle at runtime
    from squidxplorer.reader import SquidReader

OperatorFn = Callable[[Iterable[np.ndarray]], np.ndarray]

# '+()' are expression punctuation in a recipe label ('stitch(register=False)'), so a registered
# name may carry none of them.
_CHAIN_CHARS = "+()"


@dataclass(frozen=True)
class Param:
    """One parameter a registry entry declares it can be run with."""

    name: str
    default: Any
    blurb: str = ""
    #: A knob the GUI hides behind the collapsed "advanced" slot (Julio, 2026-08-25: "the
    #: user should only tweak what can't be deduced from acquisition filenames"). Headline
    #: (False) is the exception, not the rule; the panels branch on THIS flag, never on a
    #: parameter's name.
    advanced: bool = False


class MissingOperatorDependency(MissingDependency):
    """A registered operator's optional package is not importable."""


@dataclass(frozen=True)
class Operator:
    """A registry entry: name, callable, and the four declarations the engine dispatches on."""
    name: str
    fn: OperatorFn
    consumes: frozenset[str]
    produces: str = INTENSITY
    params: tuple[Param, ...] = ()
    factory: Optional[Callable[..., OperatorFn]] = None
    #: Importable module names this operator needs, e.g. ``("petakit",)``.
    requires: tuple[str, ...] = ()
    #: The pyproject optional-dependency group that installs those modules; ``None`` means core.
    extra: Optional[str] = None
    #: Keyword arguments the CALLABLE takes beyond the declared params — the region loop passes
    #: them through verbatim, and anything outside declared ∪ accepts is refused BY NAME. This is
    #: the explicit passthrough declaration: it exists so a knob that cannot be a Param (a
    #: measured-from-the-data None default, an injected object) is still part of the record
    #: instead of an unvalidated splat.
    accepts: tuple[str, ...] = ()
    #: The declared param naming an INNER operator whose own declarations shape this operator's
    #: OUTPUT (stitch's ``z_operator``). :func:`operator_output` reads it; the writer asks that
    #: query instead of reconstructing the declaration from a parameter name.
    inner_param: Optional[str] = None

    def available(self) -> tuple[bool, str]:
        """``(ok, reason_if_not)`` — are this operator's declared packages importable right now?"""
        missing = missing_requirements(self.requires)
        if missing:
            return False, requirement_refusal("operator", self.name, missing)
        return True, ""

    def bind(self, operator_kwargs: Optional[dict] = None) -> OperatorFn:
        """The callable to run, with *operator_kwargs* applied; refuses unknown params by name."""
        ok, why = self.available()
        if not ok:
            raise MissingOperatorDependency(why)
        return self.with_params(operator_kwargs)

    def with_params(self, operator_kwargs: Optional[dict] = None) -> OperatorFn:
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
# (Shelved 2026-08-24: `keepz` — keeping every z plane is spelled `z_operator=None` on stitch
# now — and `reference`, the Tenengrad z-selecting reducer. Git history reinstates.)
_OPERATORS: dict[str, Operator] = {
    "mip": Operator("mip", project, Z_REDUCER),
}


def _default_workers() -> int:
    """Thread count when the caller doesn't specify — affinity/cgroup aware, never hardcoded."""
    n = os.process_cpu_count() if hasattr(os, "process_cpu_count") else None
    if not n and hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    if not n:
        n = os.cpu_count()
    return n or 1


def add_operator(name: str, operator: OperatorFn, *, consumes=None, produces=None,
                 params: Sequence[Param] = (), requires=(),
                 extra: Optional[str] = None) -> None:
    """Add a named operator so it can be selected by name in :func:`run_plate`."""
    _declare(name, operator, consumes=consumes, produces=produces, params=params,
             requires=requires, extra=extra, region=False)


def add_region_operator(name: str, operator, *, produces=None, params: Sequence[Param] = (),
                        requires=(), extra: Optional[str] = None,
                        accepts: Sequence[str] = (),
                        inner_param: Optional[str] = None) -> None:
    """Add a named REGION operator — one that eats a whole well's FOVs — to the same table.

    ``accepts`` names the extra keyword arguments the callable takes beyond ``params`` (the
    region loop passes them through; anything else is refused by name). ``inner_param`` names
    the declared param holding an INNER operator whose declarations shape the output —
    :func:`operator_output` resolves it, so the writer never reconstructs it.
    """
    _declare(name, operator, consumes=REGION_OP, produces=produces, params=params,
             requires=requires, extra=extra, region=True, accepts=accepts,
             inner_param=inner_param)


def _declare(name: str, fn, *, consumes, produces, params, requires, extra,
             region: bool, accepts: Sequence[str] = (),
             inner_param: Optional[str] = None) -> None:
    """Validate ONE operator and file it in :data:`_OPERATORS`. The only writer of that table."""
    kind = "region operator" if region else "operator"
    if not name:
        raise ValueError(f"{kind} name must be a non-empty string")
    reserved = sorted(set(name) & set(_CHAIN_CHARS))
    if reserved:
        raise ValueError(
            f"{kind} name {name!r} contains {reserved[0]!r}: '{_CHAIN_CHARS}' are expression "
            "punctuation in a recipe label ('spot(min_area_px=80)'), so a name carrying one "
            "would not round-trip through RecipeChain.parse - it is refused here rather than "
            "left to be ambiguous everywhere a recipe is written down.")
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
    if extra is not None and (not isinstance(extra, str) or not extra):
        raise ValueError(
            f"{kind} {name!r}: extra must be None (core) or the non-empty name of a "
            f"[project.optional-dependencies] group; got {extra!r}")
    accepted = tuple(accepts)
    for a in accepted:
        if not isinstance(a, str) or not a:
            raise ValueError(
                f"{kind} {name!r}: accepts must name keyword arguments as non-empty strings; "
                f"got {a!r}")
    overlap = sorted(set(accepted) & set(names))
    if overlap:
        raise ValueError(
            f"{kind} {name!r} lists {overlap[0]!r} in both params and accepts; a knob is "
            "either DECLARED (a Param, described and probed) or passed through, never both.")
    if inner_param is not None and inner_param not in names:
        raise ValueError(
            f"{kind} {name!r}: inner_param {inner_param!r} must name one of its own declared "
            f"params ({sorted(names)}); the output query resolves the inner operator off the "
            "declared default, so an undeclared name would leave it nothing to read.")

    factory: Optional[Callable[..., OperatorFn]] = None
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
        declared, factory, normalise_requires(requires), extra,
        accepted, inner_param,
    )


def runnable_operators() -> list[str]:
    """EVERY operator this application can run, sorted."""
    return sorted(_OPERATORS)


def available_plane_operators() -> list[str]:
    """Registered operators with the ``Iterable[plane] -> plane`` shape — what the per-FOV loop runs."""
    return sorted(n for n, op in _OPERATORS.items() if "fov" not in op.consumes)


def is_region_operator(name) -> bool:
    """Does *name* eat a whole well's FOVs — i.e. must it be run by the region loop?"""
    try:
        return "fov" in _resolve_operator(name).consumes
    except (KeyError, TypeError, ValueError):
        return False


def available_region_operators() -> list[str]:
    """Registered operators that eat a whole well's FOVs — what the region loop runs."""
    return sorted(n for n, op in _OPERATORS.items() if "fov" in op.consumes)


def operator_saves_copy(name) -> bool:
    """Whether *name*'s SAVE artifact is a registered copy of the acquisition, not a plate.

    Declared, never name-matched: a region operator accepting ``copy`` writes its copy itself
    when the run passes ``copy=True`` (register's ``stitched_<folder>``), so a save routes
    through the engine with that flag instead of the OME-Zarr plate writer.
    """
    try:
        op = _resolve_operator(name)
    except (KeyError, TypeError, ValueError):
        return False
    return "fov" in op.consumes and "copy" in op.accepts


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


def operator_extra(name: str) -> Optional[str]:
    """The optional-dependency group that makes this operator runnable — ``None`` means core."""
    return _resolve_operator(name).extra


def operator_consumes(name: str) -> frozenset[str]:
    """Return the axis a registered operator consumes — ``frozenset()`` (plane-op) or ``{"z"}``."""
    return _resolve_operator(name).consumes


def operator_produces(name: str) -> str:
    """Return what a registered operator's pixels MEAN — ``"intensity"`` or ``"labels"``."""
    return _resolve_operator(name).produces


def operator_params(name: str) -> tuple[Param, ...]:
    """The parameters a registered operator declares — ``()`` when its behaviour is fixed."""
    return _resolve_operator(name).params


def operator_inner_param(name: str) -> "Optional[str]":
    """The declared param naming *name*'s INNER operator (stitch's ``z_operator``), or None.

    What lets a generated panel draw that one param as a combo over the plane operators
    plus the keep-every-plane label, instead of a bare text field."""
    return _resolve_operator(name).inner_param


def operator_accepts(name: str) -> tuple[str, ...]:
    """Extra keyword arguments the operator's callable takes beyond its declared params."""
    return _resolve_operator(name).accepts


def split_operator_kwargs(name: str, operator_kwargs: Optional[dict] = None
                          ) -> tuple[dict, dict]:
    """``(declared, passthrough)`` for one run's kwargs — refusing anything else BY NAME.

    THE one validator for both engine arms: declared keys bind through the factory, keys the
    record ``accepts`` pass through to the callable, and an unknown key raises here — before a
    directory is made, on the region arm exactly as :meth:`Operator.with_params` always did on
    the plane arm.
    """
    op = _resolve_operator(name)
    kwargs = dict(operator_kwargs or {})
    declared_names = {p.name for p in op.params}
    declared = {k: kwargs.pop(k) for k in list(kwargs) if k in declared_names}
    unknown = sorted(set(kwargs) - set(op.accepts))
    if unknown:
        raise ValueError(
            f"operator {name!r} has no parameter {unknown[0]!r}; it declares "
            f"{sorted(declared_names)}"
            + (f" and accepts {sorted(op.accepts)} as keyword arguments." if op.accepts
               else "."))
    return declared, kwargs


def operator_output(name: str, operator_kwargs: Optional[dict] = None) -> tuple[bool, str]:
    """``(collapses_z, produces)`` for what a run of *name* with *operator_kwargs* EMITS.

    An operator with ``inner_param`` defers to the inner operator that param names — resolved
    from the run's kwargs, defaulting to the DECLARED default, so the writer's depth and
    pyramid reducer come off the record instead of a writer-side reconstruction.
    """
    op = _resolve_operator(name)
    if op.inner_param is not None:
        default = next(p.default for p in op.params if p.name == op.inner_param)
        inner_name = (operator_kwargs or {}).get(op.inner_param, default)
        if inner_name is None:
            # z_operator=None: keep every acquired plane, fused unchanged — full z, intensity.
            return False, INTENSITY
        op = _resolve_operator(inner_name)
    # A z-consumer declaring ``keeps_depth`` on its callable (decon: the whole deconvolved
    # stack, same size as the input) eats the z AXIS without collapsing the OUTPUT — the same
    # declaration project_well and the acquisition writer honour. Reading consumes alone here
    # made write_plate declare n_z=1 against a full-depth stream and made the display side
    # flatten a decon volume on its own visibility toggle (2026-08-24).
    keeps_depth = bool(getattr(op.fn, "keeps_depth", False))
    return "z" in op.consumes and not keeps_depth, op.produces


def operator_reduces_depth(name: str) -> bool:
    """Does *name*'s OUTPUT collapse z to one plane, whatever a run's kwargs?

    The NAME-ONLY half of :func:`operator_output`, for callers that hold no kwargs (the
    display side's layer keys). An operator with ``inner_param`` answers False here — its
    depth depends on the run's inner choice, which a name cannot know, so the layer's own
    data decides. A z-consumer declaring ``keeps_depth`` (decon) keeps its planes.
    """
    op = _resolve_operator(name)
    return "z" in op.consumes and not bool(getattr(op.fn, "keeps_depth", False))


def run_halo_px(reader, name: str, operator_kwargs: Optional[dict], nz: int) -> int:
    """The halo an ROI run of *name* pads its windows with: the max over the acquisition's
    channels of each channel-bound operator's declared ``halo_px`` (0 when none declares)."""
    from squidxplorer.projection import acquisition_path, bind_channel, operator_halo_px

    fn = bind_operator(name, operator_kwargs)
    channels = [c["name"] for c in reader.metadata["channels"]]
    path = acquisition_path(reader) if hasattr(fn, "for_channel") else None
    return max((operator_halo_px(bind_channel(fn, path, c), nz) for c in channels), default=0)


def bind_operator(name: str, operator_kwargs: Optional[dict] = None) -> OperatorFn:
    """Resolve *name* and apply *operator_kwargs*, raising on an unknown name or parameter."""
    return _resolve_operator(name).bind(operator_kwargs)


def _resolve_operator(name) -> Operator:
    """Look up an operator by name, failing loud on an unknown key."""
    if not isinstance(name, str):
        raise TypeError(
            f"an operator is named by one string, got {type(name).__name__}: {name!r}.")
    operator = _OPERATORS.get(name)
    if operator is not None:
        return operator
    if name == "decon3d":
        raise KeyError(
            "operator 'decon3d' was renamed to 'decon' (2026-08-24): decon deconvolves the "
            "whole z stack (every plane kept; on an n_z=1 acquisition it equals the old "
            "per-plane result). Run operator='decon'.")
    if any(char in name for char in _CHAIN_CHARS):
        raise ValueError(
            f"{name!r} is a chain expression, and operator chaining was removed: an operator is "
            "ONE registered name. Compose in Python instead - wrap the steps in one callable and "
            "register it (squidxplorer.projection.plane_op + squidxplorer.add_operator, a few "
            "lines).")
    raise KeyError(
        f"unknown operator {name!r}; available: {runnable_operators()}. "
        "Add new modes with squidxplorer.add_operator(name, fn) - or "
        "squidxplorer.add_region_operator(name, fn) for one that fuses a whole well."
    )


class _LoopDefault:
    """Sentinel type for :data:`N_FOVS_LOOP_DEFAULT` — repr'd for signatures and refusals."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "N_FOVS_LOOP_DEFAULT"


#: ``run_plate``'s ``n_fovs`` default: "the loop's own default" — 1 for the per-FOV loop (its
#: historical preview default), every FOV for the region loop. A sentinel, because the two
#: loops' honest defaults differ and a shared literal was silently DISCARDED on the region arm
#: (``run_plate(operator="stitch", n_fovs=1)`` ran every FOV with no refusal).
N_FOVS_LOOP_DEFAULT = _LoopDefault()


def run_plate(
    reader: "SquidReader",
    *,
    operator: str = "mip",
    regions=None,
    n_fovs=N_FOVS_LOOP_DEFAULT,
    workers: int | None = None,
    on_error=None,
    operator_kwargs: Optional[dict] = None,
    z_level: Optional[int] = None,
    windows: Optional[dict] = None,
) -> Iterator[tuple[str, int, np.ndarray]]:
    """Run *operator* over every selected well, streaming ``(region, fov, image)`` results.

    Dispatches off the operator's ``consumes``: ``{"fov"}`` runs the region loop (one fused
    result per well, every FOV, one well in flight unless *workers* says otherwise), anything
    else the per-FOV loop. ``n_fovs`` defaults to the LOOP's own default (1 per-FOV; every FOV
    for a region operator); an explicit int is the per-FOV loop's knob and is REFUSED on the
    region arm — a FOV subset of a region is spelled ``regions={region: [fov, ...]}``.
    ``z_level=`` restricts the per-FOV loop to one acquisition plane (``project_well``'s own
    knob: plane-ops and depth-keeping z-consumers only) and is refused on the region arm.
    ``windows={(region, fov): (r0, r1, c0, c1)}`` runs each named field on that frame-pixel
    window plus the operator's declared halo (ruling z, sub-FOV decon); refused on the
    region arm, whose fusion needs whole frames.
    """
    if is_region_operator(operator):
        if z_level is not None:
            raise ValueError(
                f"a region operator's z handling is its z_operator: z_level={z_level!r} would "
                "silently crop the fusion. Pass z_operator= in operator_kwargs instead.")
        if windows:
            raise ValueError(
                f"a region operator fuses whole frames: a window on {len(windows)} field(s) "
                "would register and blend cropped tiles. Select FOVs with "
                "regions={region: [fov, ...]} instead.")
        from squidxplorer._stitch import _stitch_plate

        if n_fovs is not N_FOVS_LOOP_DEFAULT and n_fovs is not None:
            raise ValueError(
                f"a region operator fuses whole wells: n_fovs={n_fovs!r} would silently crop "
                f"each well to its first {n_fovs} FOV(s) in row-major order, which is not a "
                "thing anyone draws. Select FOVs with regions={region: [fov, ...]} - the one "
                "spelling of a FOV subset - or pass n_fovs=None for every FOV.")
        return _stitch_plate(reader, n_fovs=None, workers=1 if workers is None else workers,
                             operator=operator, on_error=on_error, regions=regions,
                             **(operator_kwargs or {}))
    return _project_plate(reader,
                          n_fovs=1 if n_fovs is N_FOVS_LOOP_DEFAULT else n_fovs,
                          workers=workers, operator=operator,
                          on_error=on_error, regions=regions, operator_kwargs=operator_kwargs,
                          z_level=z_level, windows=windows)


def _project_plate(
    reader: "SquidReader",
    *,
    n_fovs: Optional[int] = 1,
    workers: int | None = None,
    operator: str = "mip",
    on_error=None,
    regions=None,
    operator_kwargs: Optional[dict] = None,
    z_level: Optional[int] = None,
    windows: Optional[dict] = None,
) -> Iterator[tuple[str, int, np.ndarray]]:
    """Project every selected well in parallel, streaming ``(region, fov, image)`` results.

    The in-flight window is bounded at *workers* wells, so peak memory is flat in plate size.
    ``on_error(region, fov, exc)`` opts in to per-well fault isolation for data faults only.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    n_workers = workers if workers is not None else _default_workers()

    op = _resolve_operator(operator)
    # A region operator is a whole-well callable; this loop hands out planes.
    if "fov" in op.consumes:
        raise ValueError(
            f"{operator!r} consumes fov - it fuses a whole well's FOVs and takes "
            "(reader, region, fovs), which is not what the per-FOV loop hands an operator. Run "
            f"it with squidxplorer.run_plate(reader, operator={operator!r}).")
    fn = bind_operator(operator, operator_kwargs)
    by_field = {(str(r), int(f)): w for (r, f), w in (windows or {}).items()}

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
                                 reduce=fn, consumes=op.consumes, z_level=z_level,
                                 window=by_field.get((str(region), int(fov))))
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
