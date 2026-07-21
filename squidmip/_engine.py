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

    project_plate(reader, n_fovs=1, workers=N, projector="mip")
        │
        ▼  reader.metadata            (warm ONCE, single-threaded → populates the reader's
        │                              lazy index/time-folders/meta so concurrent read() only
        │                              touches immutable state; no locks needed downstream)
        ▼  select_fovs(meta, n_fovs)  → {region: [fov, ...]}  → flat [(region, fov), ...]
        ▼  _PROJECTORS[projector]     → the z-reduce callable passed as project_well(reduce=)
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
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Optional

import numpy as np

from squidmip.projection import project, project_reference, project_well, select_fovs

if TYPE_CHECKING:  # avoid import cost / cycle at runtime
    from squidmip.reader import SquidReader

# A projector reduces one channel's z-planes to a single plane (the ``reduce=`` argument of
# project_well). MIP is the only one 183 ships; the projector table is the seam for the rest.
Projector = Callable[[Iterable[np.ndarray]], np.ndarray]

# name -> z-reduction callable. Selected by name in project_plate; extended via add_projector.
_PROJECTORS: dict[str, Projector] = {"mip": project, "reference": project_reference}


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


def add_projector(name: str, projector: Projector) -> None:
    """Add a named z-reduction so it can be selected by name in :func:`project_plate`.

    This is how a future projector (EDF/EMF/mean) plugs in **without touching the engine**:
    add a name, then call ``project_plate(..., projector="<name>")``. (Named ``add_``, not
    ``register_``, to avoid confusion with image *registration* / alignment.)

    Parameters
    ----------
    name:
        The projector's table key (e.g. ``"mip"``, ``"mean"``). Non-empty.
    projector:
        A callable with the :func:`squidmip.project` signature — takes an iterable of
        equal-shape planes and returns one plane. It SHOULD stream (bounded memory) to keep
        the plate engine's per-worker footprint flat; a projector that materialises the whole
        z-stack (e.g. EDF) is allowed but owns its own, documented, memory profile.

    Raises
    ------
    ValueError
        If *name* is empty, *projector* is not callable, or *name* is already defined
        (a silent clobber of an existing projector would be a quiet correctness bug).
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
    _PROJECTORS[name] = projector


def available_projectors() -> list[str]:
    """Return the available projector names, sorted (``["mip", ...]``)."""
    return sorted(_PROJECTORS)


def _resolve_projector(name: str) -> Projector:
    """Look up a projector by name, failing loud (named) on an unknown key."""
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
        FOVs per well to project (default 1). Passed to :func:`squidmip.select_fovs`.
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
        downstream keys by ``(region, fov)``). ``image`` is ``(T, C, 1, Y, X)`` native dtype.

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

    reduce = _resolve_projector(projector)

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
            future = pool.submit(project_well, reader, region, fov, reduce=reduce)
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
