"""Parallel/streaming plate engine + pluggable projector table (IMA-188).

IMA-183 made per-well projection *correct and optimal* (single-thread ~0.44 s/well,
memory-bounded via a streaming running-max). IMA-188 makes it *fast across the whole
plate* without changing a single pixel: run ``project_well`` across wells on a thread
pool, stream results well-by-well so the whole plate never sits in RAM, and let the
z-reduction be swapped by name (MIP now, EDF/mean later) through the ``reduce=`` seam
183 already built.

Why threads, not processes: the per-well cost is I/O + ``tifffile`` decode + a single
``np.maximum`` fold — decode and the ufunc both release the GIL, so threads scale on the
bound work. A process pool would pay ~139 MB (one ``(T,C,1,Y,X)`` result) of pickling per
well crossing the boundary, for nothing.

Data flow::

    project_plate(reader, n_fovs=1, workers=N, projector="mip")   # n_fovs=None -> ALL FOVs
        │
        ▼  reader.metadata            (warm ONCE, single-threaded → populates the reader's
        │                              lazy index/time-folders/meta so concurrent read() only
        │                              touches immutable state; no locks needed downstream)
        ▼  select_fovs(meta, n_fovs)  → {region: [fov, ...]}  → flat [(region, fov), ...]
        │                              (n_fovs=None → every FOV; a 36-FOV well emits 36 tasks)
        ▼  _PROJECTORS[projector]     → Operator(fn, consumes) passed as project_well(reduce=,
        │                              consumes=); `consumes` alone decides the grouping (IMA-210)
        │
        ▼  ThreadPoolExecutor(max_workers=N)          bounded window: ≤ N wells in flight
        │     prime N tasks ─┐                        so completed ~139 MB results can NOT
        │        ┌───────────┘                        accumulate → peak RSS ≈ N × one-well
        │        ▼   wait(FIRST_COMPLETED)            footprint, FLAT in plate size
        │     as each future completes:
        │        result = fut.result()  ── raises ──► propagate LOUD (fail-fast; per-well
        │        submit one refill (slide the window)  resilience/manifest is IMA-186's job)
        │        yield (region, fov, result)
        ▼
    Iterator[(region, fov, ndarray(T, C, 1, Y, X))]   ← the stream IMA-184 serializes

The projector table is the IMA-188 half of the pluggable-projector contract: 183 ships
``project`` (MIP); a future EDF/EMF/mean projector is added by name here and runs through
``project_plate(..., projector="<name>")`` with **zero engine edits**.

IMA-210 widens that table from "z-reductions" to "operators" by making each entry declare the
axis it CONSUMES (see :class:`Operator`), and deriving the loop from that declaration:

    consumes=frozenset({"z"})   z-reducer   stack -> plane, output (T, C,  1, Y, X)   mip, reference
    consumes=frozenset()        plane-op    plane -> plane, output (T, C, Nz, Y, X)   decon, bgsub…

One group-by-then-reduce loop (in ``project_well``) serves both, so a new plane-op is ONE
``add_projector`` call and no engine edit. ``{"fov"}`` is deliberately not a member — inter-FOV
work needs stage geometry a ``Iterable[plane] -> plane`` callable never sees (IMA-222's seam).

Two more declarations were added when Cellpose was made a real operator (2026-08-03), and both
follow the same rule: **extend the declaration, never the loop.** Nothing in this module branches
on an operator's name, and nothing added here does either.

    produces="intensity"        the pixels measure light -> a napari Image layer, windowed
    produces="labels"           the pixels are integer OBJECT IDS -> a napari Labels layer

    params=(Param("min_area_px", 30), ...)   the entry is ONE operator that can be RUN with
                                             arguments, not N entries frozen at N parameter sets

``produces`` closes a defect that was live rather than theoretical: ``spot`` emitted a label image
down the intensity delivery path, so a segmentation arrived auto-windowed by the FLUORESCENCE
contrast rule, as if label 37 were 37 photons. ``params`` closes the asymmetry ``write_plate``
documented in a hardcoded refusal — a region operator took ``operator_kwargs`` and a projector did
not, so the only way to change a projector's number was a second registry entry (``decon`` /
``decon3d``, ``"bgsub_tight"``, ``"flatfield_638"``) or a module-level global behind a lock
(``_flatfield.set_profile``; ``_decon.set_optics`` was the other one, and it is now an OVERRIDE
rather than the source of truth — see the next paragraph).

WHAT ``params`` CANNOT EXPRESS, and what was done about it (2026-08-04). "Detect on
Fluorescence_405_nm_Ex" cannot be a ``Param``: ``project_well`` calls the operator once per
``(t, c, z_group)`` and hands it PLANES ONLY, so a value bound at registration never learns which
channel it is looking at. This module used to record that as permanent, and it cost a scientific
defect — ``decon`` deconvolved 405, 561 and 638 with the 488 line's PSF (525 nm), because its
optics came from a module-level default nothing ever set.

The answer was the same rule again, **extend the declaration, never the loop**:

    for_channel(acquisition_path, channel) -> operator    an attribute ON the callable

``project_well`` calls it once per channel and runs what it returns; the callable shape is still
``Iterable[plane] -> plane`` and nothing here changed. So an operator can now be specialised to a
channel, while a ``Param`` still cannot BE one — a parameter is a value fixed for the run and a
channel is a thing the loop knows. See ``squidmip.projection.bind_channel`` and
``_decon.optics_for_channel``. ``consumes={"fov"}`` remains genuinely refused: stage geometry is
not carried by any declaration on the callable.

NOTE for plane-ops: ``write_plate``/IMA-184 currently accept only ``Z == 1`` frames and reject a
Z>1 frame LOUD (``_validate_image``). So a plane-op streams correctly out of ``project_plate``
today, and gains a persistence path when the writer learns Z>1 — it is not silently wrong.

Prior art (what established pipelines declare, and what IMA-210 took from each)
------------------------------------------------------------------------------
* **ITK** — ``itk::ProjectionImageFilter`` declares ``m_ProjectionDimension``: the ONE axis it
  accumulates over, and it "reduces the size of the accumulated dimension to 1". A plain
  ``ImageToImageFilter``/``InPlaceImageFilter`` declares nothing and is shape-preserving.
  TAKEN: the whole model. ``consumes={"z"}`` → that axis becomes size 1; ``consumes={}`` →
  shape-preserving map. Also taken: keep the collapsed axis at size 1 instead of dropping it
  (ITK's ExtractImageFilter shows that removing an axis forces the caller to invent a
  direction/geometry for what is left; our writer would need the same special-casing).
* **Fractal** (fractal-tasks-core ``__FRACTAL_MANIFEST__.json``) — each task declares
  ``input_types``/``output_types``: "Project Image (HCS Plate)" is ``{"is_3D": true}`` →
  ``{"is_3D": false}``, while "Illumination Correction" is
  ``{"illumination_corrected": false}`` → ``{"illumination_corrected": true}`` and leaves
  ``is_3D`` alone. That is EXACTLY the z-reducer / plane-op split, declared as data.
  TAKEN: one declarative record per operator that the runner dispatches on, so the runner has
  no per-task branch. NOT taken: the general input/output type-filter machinery (arbitrary
  provenance flags, task chaining by filter) — that is workflow bookkeeping, not this ticket.
* **dask / scikit-image** — ``da.map_blocks(func, drop_axis=…, new_axis=…)`` vs ``da.reduction``,
  and ``skimage.util.apply_parallel(function, array, depth=…, channel_axis=…)``: the axis
  arguments are the declaration; the SAME map machinery serves both, and per-block funcs stay
  plain array->array. TAKEN: one callable shape (``Iterable[plane] -> plane``) plus an axis
  declaration, rather than two call conventions — which is why one loop covers both cases.
  NOT taken: ``depth``/halo (overlap is meaningless when the group is a single plane).
* **CellProfiler** — ``Module.volumetric()`` returns ``False`` by default, i.e. "run me per
  plane"; a module opts IN to seeing a volume. Modules declare I/O through settings
  (``ImageSubscriber`` consumes a named image, ``ImageName`` provides one).
  TAKEN: the *default* is the conservative one. Ours is inverted only because this table's
  history is z-reductions: ``add_projector`` with no ``consumes=`` still means ``{"z"}``, so no
  pre-IMA-210 registration changes meaning. NOT taken: named-image wiring — a projector's input
  is positional (the FOV's planes), so a name-based dataflow graph would be ceremony.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from squidmip.projection import (
    INTENSITY,
    PLANE_OP,
    Z_REDUCER,
    normalise_consumes,
    normalise_produces,
    project,
    project_reference,
    project_well,
    select_fovs,
)

if TYPE_CHECKING:  # avoid import cost / cycle at runtime
    from squidmip.reader import SquidReader

# An operator maps a GROUP of planes to one plane (the ``reduce=`` argument of project_well).
# One callable shape for every operator — what differs is which axis the engine groups over.
Projector = Callable[[Iterable[np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class Param:
    """One parameter a registry entry declares it can be run with.

    A NAME and a DEFAULT, nothing clever. It exists so a parameterised operator is one entry that
    says what it takes, rather than one entry per parameter set — which is what the table held
    before: ``decon``/``decon3d`` are two entries for one algorithm, and the docstrings of
    ``decon_op``/``bgsub_op``/``flatfield_op`` all tell the reader to register ANOTHER name
    (``"decon_sharp"``, ``"bgsub_tight"``, ``"flatfield_638"``) to change a number.

    ``blurb`` is the one line a UI shows. There is deliberately no type, no range and no widget
    hint: this is a declaration the engine validates NAMES against, not a form builder. The moment
    it grows a widget hint it has become the UI's schema and two places own the same fact.
    """

    name: str
    default: Any
    blurb: str = ""


@dataclass(frozen=True)
class Operator:
    """A registry entry: a name, the callable, and the axis it consumes (IMA-210).

    ``consumes`` is the whole dispatch. The engine derives its loop from it instead of asking what
    kind of operator this is:

      * ``frozenset()``       — **plane-op**: plane -> plane. z is not consumed, so z SURVIVES at
        full depth and the operator is MAPPED over the planes (deconvolution, background
        subtraction, flat-field — IMA-223/224/225).
      * ``frozenset({"z"})``  — **z-reducer**: a (t, c)'s whole stack -> one plane, z collapses to
        size 1 (``mip``, ``reference``).

    It is a declaration of WHICH AXIS, not of HOW: ``reference`` picks one z and ``mip`` combines
    all of them, and both are ``{"z"}``. The "how" already has its own marker — the ``select_index``
    attribute that makes ``project_well`` solve focus once per (t, fov) and share it across
    channels. Encoding "selects" as a distinct consumed axis would re-open the bug where the
    channels of one FOV were sampled at different z and stopped overlaying.

    ``produces`` is the SECOND declaration and the same kind of thing: ``consumes`` decides the
    engine's loop and the output SHAPE, ``produces`` decides what the output pixels MEAN and
    therefore which napari layer type they become. See
    :data:`squidmip.projection.INTENSITY` / :data:`squidmip.projection.LABELS`. Default
    ``"intensity"``, so every pre-existing registration keeps its exact meaning.

    ``params`` and ``factory`` are how ONE entry carries a parameter set. ``factory(**kwargs)``
    builds a projector; ``fn`` is that factory called at the declared defaults, so a run that names
    no parameters is byte-identical to the un-parameterised registration it replaced. An entry with
    no ``factory`` is a fixed operator and refuses parameters by name.
    """
    name: str
    fn: Projector
    consumes: frozenset[str]
    produces: str = INTENSITY
    params: tuple[Param, ...] = ()
    factory: Optional[Callable[..., Projector]] = None

    def bind(self, operator_kwargs: Optional[dict] = None) -> Projector:
        """The callable to run, with *operator_kwargs* applied. THE parameter seam.

        No kwargs -> ``fn``, the default binding built at registration. That identity is what
        makes this change invisible to every existing caller: ``project_plate(projector="mip")``
        runs the exact object the table has always held.

        Refuses LOUD on an unknown parameter name, naming what this operator does accept. Accepting
        and dropping it would run the operator at its defaults while the console line and the
        recipe both said otherwise, which is a wrong result that looks right.
        """
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
        """``{name: default}`` for every declared parameter. What a UI seeds its fields from."""
        return {p.name: p.default for p in self.params}


# name -> Operator. Selected by name in project_plate; extended via add_projector.
_PROJECTORS: dict[str, Operator] = {
    "mip": Operator("mip", project, Z_REDUCER),
    "reference": Operator("reference", project_reference, Z_REDUCER),
}


def _default_workers() -> int:
    """Thread count when the caller doesn't specify — adapt to the machine, never hardcode.

    Prefers the number of CPUs actually usable by *this process* (respects CPU-affinity and
    cgroup/container limits), then falls back across Python versions and platforms:
      1. ``os.process_cpu_count()``      — Python 3.13+, affinity/cgroup aware
      2. ``len(os.sched_getaffinity(0))``— Linux, the CPUs this process may run on
      3. ``os.cpu_count()``              — total logical cores
      4. ``1``                           — last-resort floor
    """
    n = os.process_cpu_count() if hasattr(os, "process_cpu_count") else None
    if not n and hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    if not n:
        n = os.cpu_count()
    return n or 1


def add_projector(name: str, projector: Projector, *, consumes=None, produces=None,
                  params: Sequence[Param] = ()) -> None:
    """Add a named operator so it can be selected by name in :func:`project_plate`.

    This is how a new operator plugs in **without touching the engine**: add a name and its
    consumed axis, then call ``project_plate(..., projector="<name>")``. (Named ``add_``, not
    ``register_``, to avoid confusion with image *registration* / alignment.)::

        add_projector("mean", lambda planes: ...)                       # z-reducer (the default)
        add_projector("bgsub", plane_op(subtract_background))           # plane-op, consumes inferred
        add_projector("decon", my_decon, consumes=frozenset())          # plane-op, declared

    Parameters
    ----------
    name:
        The projector's table key (e.g. ``"mip"``, ``"mean"``). Non-empty.
    projector:
        A callable with the :func:`squidmip.project` signature — takes an iterable of
        equal-shape planes and returns one plane. It SHOULD stream (bounded memory) to keep
        the plate engine's per-worker footprint flat; a projector that materialises the whole
        z-stack (e.g. EDF) is allowed but owns its own, documented, memory profile. A natural
        ``plane -> plane`` function is lifted into this shape by :func:`squidmip.plane_op`.
    consumes:
        Which axis the operator eats — see :class:`Operator`. Any iterable of axis names
        (``frozenset()``, ``{"z"}``, ``"z"``). ``None`` (default) reads the callable's own
        ``consumes`` attribute (:func:`squidmip.plane_op` stamps one) and otherwise falls back to
        ``frozenset({"z"})``, the shipped z-reduction contract — so every pre-IMA-210 registration
        keeps its exact meaning. ``{"fov"}`` is refused by name: see :class:`Operator`.
    produces:
        What this operator's pixels MEAN — ``"intensity"`` or ``"labels"``, see
        :data:`squidmip.projection.LABELS`. ``None`` (default) reads the callable's own
        ``produces`` attribute (:func:`squidmip.projection.labels_op` stamps one) and otherwise
        falls back to ``"intensity"``, so every pre-existing registration keeps its exact meaning.
    params:
        The parameters this ONE entry can be run with, as :class:`Param` records. When non-empty,
        *projector* is read as a **factory**: it is called with the declared defaults to build the
        entry's default binding, and called again with a caller's ``operator_kwargs`` per run
        (:meth:`Operator.bind`)::

            add_projector("cellpose", _build, params=(Param("min_area_px", 30),))
            project_plate(reader, projector="cellpose", operator_kwargs={"min_area_px": 80})

        The presence of *params* is what makes *projector* a factory — there is one registrar and
        one rule, not two entry points that mean nearly the same thing. Empty (default) keeps the
        original contract exactly: *projector* is the callable, and the entry refuses parameters.

    Raises
    ------
    ValueError
        If *name* is empty, *projector* is not callable, *consumes* names an axis this engine
        cannot group over, *produces* is not a known result kind, a declared parameter is unnamed
        or duplicated, or *name* is already defined (a silent clobber of an existing projector
        would be a quiet correctness bug).
    """
    if not name:
        raise ValueError("projector name must be a non-empty string")
    if not callable(projector):
        raise ValueError(f"projector for {name!r} is not callable: {projector!r}")
    if name in _PROJECTORS:
        raise ValueError(
            f"projector {name!r} is already defined; pick a distinct name "
            f"(defined: {available_projectors()})."
        )
    declared = tuple(params)
    for p in declared:
        if not isinstance(p, Param) or not p.name:
            raise ValueError(
                f"projector {name!r}: params must be Param(name, default) records with a "
                f"non-empty name; got {p!r}")
    names = [p.name for p in declared]
    if len(set(names)) != len(names):
        raise ValueError(
            f"projector {name!r} declares a parameter twice: {sorted(names)}; a duplicate makes "
            "operator_kwargs ambiguous")

    factory: Optional[Callable[..., Projector]] = None
    if declared:
        factory = projector
        projector = factory(**{p.name: p.default for p in declared})
        if not callable(projector):
            raise ValueError(
                f"projector factory for {name!r} returned {projector!r}, which is not callable. "
                "With params=, the registered object is a FACTORY: it is called with the declared "
                "defaults and must return the operator callable.")
    if consumes is None:
        consumes = getattr(projector, "consumes", Z_REDUCER)
    if produces is None:
        produces = getattr(projector, "produces", INTENSITY)
    _PROJECTORS[name] = Operator(
        name, projector, normalise_consumes(consumes), normalise_produces(produces),
        declared, factory,
    )


def available_projectors() -> list[str]:
    """Return the available projector names, sorted (``["mip", ...]``)."""
    return sorted(_PROJECTORS)


def projector_consumes(name: str) -> frozenset[str]:
    """Return the axis a registered operator consumes — ``frozenset()`` (plane-op) or ``{"z"}``.

    The registry's declaration, for callers that must branch on output shape (a plane-op keeps z at
    full depth, a z-reducer collapses it to 1) rather than re-deriving it from the callable.
    """
    return _resolve_projector(name).consumes


def projector_produces(name: str) -> str:
    """Return what a registered operator's pixels MEAN — ``"intensity"`` or ``"labels"``.

    The registry's declaration, for the delivery path that must pick a napari layer type. Read
    this rather than testing the operator's NAME: a sink that knows ``"spot"`` and ``"cellpose"``
    by name needs editing for every segmenter after them, and that is precisely the property the
    ``consumes`` table was built to avoid.
    """
    return _resolve_projector(name).produces


def projector_params(name: str) -> tuple[Param, ...]:
    """The parameters a registered operator declares — ``()`` when its behaviour is fixed."""
    return _resolve_projector(name).params


def bind_projector(name: str, operator_kwargs: Optional[dict] = None) -> Projector:
    """Resolve *name* and apply *operator_kwargs*, raising on an unknown name or parameter.

    Exists so a caller that must refuse EARLY can. ``project_plate`` is a generator, so anything it
    validates is validated at the first ``next()``; ``write_plate`` creates a directory before it
    gets there, and "the run failed after making the output tree" is a worse answer than "the run
    did not start". Same object, same errors, asked sooner.
    """
    return _resolve_projector(name).bind(operator_kwargs)


def _resolve_projector(name: str) -> Operator:
    """Look up an operator by name, failing loud (named) on an unknown key."""
    try:
        return _PROJECTORS[name]
    except KeyError:
        raise KeyError(
            f"unknown projector {name!r}; available: {available_projectors()}. "
            "Add new modes with squidmip.add_projector(name, fn)."
        ) from None


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
    """Project every selected well of a plate in parallel, streaming results well-by-well.

    The throughput entry point IMA-184 consumes. Runs IMA-183's ``project_well`` across wells
    on a thread pool with a **bounded in-flight window** (≤ *workers* wells at once), so the
    whole plate is never resident: peak memory ≈ *workers* × one well's footprint, flat in
    plate size. Concurrency changes no pixel — each well's output is byte-for-byte identical
    to the single-thread projection.

    Parameters
    ----------
    reader:
        An IMA-189 ``SquidReader`` (from ``open_reader``). Its ``metadata`` is accessed once
        up front (single-threaded) so the reader's lazy state is populated before any worker
        calls ``read()`` — concurrent reads then touch only immutable state.
    n_fovs:
        FOVs per well to project (default 1). ``None`` = every FOV in every well (the
        IMA-187 mosaic path). Passed straight to :func:`squidmip.select_fovs`.
    workers:
        Thread-pool size. ``None`` (default) → :func:`_default_workers` (CPUs usable by this
        process — affinity/cgroup aware, not a hardcoded constant). Peak RSS scales with this,
        so pin it on many-core machines.
    projector:
        A projector name from the table (default ``"mip"``). See :func:`add_projector`.

    Yields
    ------
    tuple[str, int, np.ndarray]
        ``(region, fov, image)`` per selected well, in completion order (not plate order —
        downstream keys by ``(region, fov)``). ``image`` is ``(T, C, Nz, Y, X)`` native dtype;
        ``Nz`` is 1 for a z-reducer and the acquisition's depth for a plane-op.

    Raises
    ------
    ValueError
        If *workers* < 1, or (via ``select_fovs``) *n_fovs* is invalid for the plate.
    KeyError
        If *projector* names a projector that is not in the table.
    Exception
        Any error from a well (e.g. a corrupt/missing plane raised by ``reader.read``) is
        propagated LOUD, aborting the stream — UNLESS *on_error* is given (see below).

    Other Parameters
    ----------------
    on_error:
        Opt-in per-well fault isolation for high-throughput/unattended runs (IMA-186). When set to a
        callable ``on_error(region, fov, exc)``, a well whose projection raises is passed to it and
        then SKIPPED — the stream keeps going instead of aborting the whole plate on one corrupt
        file. ``None`` (default) keeps the fail-fast contract exactly. Peak-memory bound is unchanged.
    operator_kwargs:
        Per-run parameters for *projector*, applied through :meth:`Operator.bind`. Only an operator
        that DECLARES parameters accepts them; anything else raises, naming what it does accept.
        ``None``/``{}`` (default) runs the entry's default binding — the same object the table has
        always held, so no existing run changes by a pixel. This is the projector half of the seam
        ``stitch_plate`` has had since IMA-222; the two tables were asymmetric for no reason other
        than history, and ``write_plate`` had a hardcoded refusal saying so.

    Notes
    -----
    Bounded window: exactly *workers* tasks are primed, then one refill is submitted for each
    completion (the window slides forward one well at a time). At most ``workers`` results are
    in flight plus the one being yielded, so ~139 MB per-well results cannot accumulate into an
    unbounded backlog if the consumer is slow. This is what keeps peak RSS independent of the
    number of wells.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    n_workers = workers if workers is not None else _default_workers()

    op = _resolve_projector(projector)
    fn = bind_projector(projector, operator_kwargs)  # the declaration decides, never the name

    # Warm the reader's lazy index/time-folders/metadata single-threaded BEFORE fan-out.
    meta = reader.metadata
    wells = select_fovs(meta, n_fovs=n_fovs)
    if regions is not None:   # subset preview: keep only the requested wells (in their given order)
        keep = list(dict.fromkeys(regions))
        wells = {r: wells[r] for r in keep if r in wells}
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
                except Exception as exc:
                    if on_error is None:
                        raise                       # default: fail-fast (unchanged contract)
                    on_error(region, fov, exc)      # opt-in: record + SKIP this well, keep going
                    continue
                yield region, fov, image
