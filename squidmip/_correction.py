"""Per-plane corrections that decorate the ``reduce=`` seam (IMA-224 background subtraction).

WHY THIS IS NOT A PROJECTOR
---------------------------
``projection.py:162`` writes ``out[t, c, 0] = reduce(planes)`` — Z is pinned to size 1, and
``_output.write_plate``, ``_montage`` and ``_viewer`` all depend on that. A *correction* is
1 plane in, 1 plane out; registering it in ``_PROJECTORS`` as a peer of ``mip`` would produce
a shape the writer cannot serialize. So a correction **decorates** a z-reducer instead:

    with_correction(project, corr, side) -> a plain Projector

BEFORE vs AFTER — and why AFTER is legal for MIP
------------------------------------------------
For a background scalar ``b`` that does not vary with z, subtraction commutes exactly with a
maximum::

    max_z( p_z - b )  ==  max_z( p_z ) - b

so correcting AFTER the reduction is bit-identical at 1/N_z the cost (on a 1536wp x 20z x 4c
run that is ~123k plane operations avoided). This does NOT generalise to every reducer —
``project_reference`` picks a plane by a focus metric — so ``BEFORE`` is the safe default and
only callers who know their reducer commutes opt into ``AFTER``.

    planes ──► [corr] ──► reduce ──► plane          side=BEFORE  (always correct, N_z work)
    planes ──►          reduce ──► [corr] ──► plane  side=AFTER   (max only, 1x work)

THE TWO TRAPS
-------------
1. **Unsigned underflow.** Planes are typically uint16. ``100 - 200`` does not go negative,
   it wraps to 65436 — and since the reducer is a *maximum*, wrapped background pixels win
   every comparison and the whole plate renders inverted. Hence clamp-then-subtract.
2. **float64 promotion.** ``np.percentile`` returns a float64 scalar, and
   ``np.maximum(uint16_array, float64_scalar)`` promotes the *result* to float64 — which
   trips ``projection.py``'s dtype check or silently widens the write. Hence the explicit
   cast to the plane's own dtype before any arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal

import numpy as np

Plane = np.ndarray
Correction = Callable[[Plane], Plane]
Side = Literal["before", "after"]

BEFORE: Side = "before"
AFTER: Side = "after"

#: Planes sampled per channel when estimating the background. Bounded so the estimate costs
#: O(1) in plate size rather than growing with the number of wells.
BACKGROUND_SAMPLE_PLANES = 8

#: Stride used when sampling pixels *within* a plane for the percentile. A low percentile of
#: a strided subsample is statistically indistinguishable from the full-frame value at a
#: fraction of the cost. Implementation constant, deliberately not a second user-facing knob.
BACKGROUND_PIXEL_STRIDE = 4

#: Default percentile treated as "background". Low enough to sit in true background for a
#: typical field, high enough to be robust to hot pixels and read noise.
DEFAULT_BACKGROUND_PERCENTILE = 10.0


def subtract_background(plane: Plane, background) -> Plane:
    """Subtract a scalar *background* from *plane*, clipped at zero, dtype preserved.

    Clamp-then-subtract: raise everything below the background *up* to it first, so the
    subtraction cannot underflow by construction (no element is ever smaller than the value
    being taken away). Equivalent to ``clip(plane - background, 0, None)`` but without
    widening to a signed type, which keeps the bounded-memory contract of the reduce path.

    Parameters
    ----------
    plane:
        A single 2-D plane in the reader's native dtype.
    background:
        Scalar background level. Cast to *plane*'s dtype before use — passing the raw float64
        from ``np.percentile`` would otherwise promote the result to float64.

    Returns
    -------
    np.ndarray
        Same shape and dtype as *plane*, with values ``<= plane.max()``. Never mutates the
        caller's array.
    """
    if background is None:
        return plane
    bg = _as_dtype(background, plane.dtype)
    if bg == 0:
        return plane
    # np.maximum() allocates the result, so the caller's plane is untouched; the in-place
    # subtract then reuses that buffer (one temporary, same width as the input).
    out = np.maximum(plane, bg)
    out -= bg
    return out


def _as_dtype(value, dtype: np.dtype):
    """Cast a scalar to *dtype*, rounding and clamping so integer dtypes stay in range.

    This is the guard for the float64-promotion trap: ``np.percentile`` returns float64, and
    ``np.maximum(uint16_plane, float64_scalar)`` yields a float64 array.
    """
    dt = np.dtype(dtype)
    if dt.kind in "ui":
        info = np.iinfo(dt)
        v = int(np.rint(float(value)))
        return dt.type(min(max(v, info.min), info.max))
    return dt.type(value)


def with_correction(reduce: Callable[[Iterable[Plane]], Plane],
                    corr: Correction | None,
                    side: Side = BEFORE) -> Callable[[Iterable[Plane]], Plane]:
    """Decorate a z-reducer with a per-plane *corr*, returning a plain projector callable.

    This is the composition seam. The result has the exact ``Projector`` signature
    (``Iterable[plane] -> plane``), so it drops into ``project_well(reduce=...)`` and the
    plate engine with no other change.

    Parameters
    ----------
    reduce:
        The z-reduction to wrap (e.g. :func:`squidmip.project`).
    corr:
        A 1-plane-in/1-plane-out callable, or ``None`` for an identity pass-through (so
        callers can wire this unconditionally and let "no correction" be the null case).
    side:
        ``BEFORE`` applies *corr* to every plane, then reduces — always correct, N_z work.
        ``AFTER`` reduces first, then corrects once — only valid when the correction commutes
        with the reducer (see the module docstring), at 1/N_z the cost.

    Raises
    ------
    ValueError
        If *side* is not ``"before"`` or ``"after"``.
    """
    if corr is None:
        return reduce
    if side == BEFORE:
        def _reduce_before(planes: Iterable[Plane]) -> Plane:
            return reduce(corr(p) for p in planes)   # generator: still one streaming pass
        return _reduce_before
    if side == AFTER:
        def _reduce_after(planes: Iterable[Plane]) -> Plane:
            return corr(reduce(planes))
        return _reduce_after
    raise ValueError(f"side must be {BEFORE!r} or {AFTER!r}, got {side!r}")


def background_corrector(background) -> Correction | None:
    """Build a :data:`Correction` that subtracts *background*, or ``None`` if there is none."""
    if background is None:
        return None
    return lambda plane: subtract_background(plane, background)


def estimate_background(reader,
                        percentile: float = DEFAULT_BACKGROUND_PERCENTILE,
                        *,
                        sample_planes: int = BACKGROUND_SAMPLE_PLANES,
                        channel=None) -> float:
    """Estimate one background level for the plate from a bounded sample of raw planes.

    ONE scalar for the whole run, not one per well and not one per plane. That is deliberate:
    a per-well or per-plane estimate varies with cell density and focus, so a confluent field
    would get a larger subtraction than a sparse one. Under a max-projection that changes
    *which z wins* per pixel and makes intensities non-comparable across wells — which is the
    entire point of a high-content screen.

    Sampling is bounded (``sample_planes``) and deterministic (first wells/z in reader order),
    so cost does not grow with plate size and two runs of the same acquisition agree.

    Parameters
    ----------
    reader:
        An IMA-189 ``SquidReader``.
    percentile:
        Percentile of sampled pixels treated as background, in [0, 100].
    sample_planes:
        Maximum number of planes to read for the estimate.
    channel:
        Channel *name* to sample (reader planes are keyed by name, not index), or ``None``
        for the first channel in ``metadata["channels"]``.

    Returns
    -------
    float
        The estimated background level. ``0.0`` if nothing could be sampled (an empty or
        unreadable acquisition degrades to "no correction" rather than raising — the caller
        is running a plate, not a diagnostic).

    Raises
    ------
    ValueError
        If *percentile* is outside [0, 100] or *sample_planes* < 1.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    if sample_planes < 1:
        raise ValueError(f"sample_planes must be >= 1, got {sample_planes}")

    values = [
        np.percentile(plane[::BACKGROUND_PIXEL_STRIDE, ::BACKGROUND_PIXEL_STRIDE], percentile)
        for plane in _sample_planes(reader, sample_planes, channel)
    ]
    if not values:
        return 0.0
    return float(np.median(values))


def _sample_planes(reader, limit: int, channel) -> Iterator[Plane]:
    """Yield up to *limit* raw planes in a deterministic order, tolerating unreadable ones.

    ``reader.metadata`` is a **dict** (reader.py:157) and planes are keyed by channel *name*
    (reader.py:215 builds its key with ``str(channel)``), so both are looked up, not guessed.
    """
    from squidmip.projection import select_fovs

    meta = reader.metadata
    channels = list(meta.get("channels") or [])
    if channel is not None:
        c = channel
    elif channels:
        c = channels[0]
    else:
        return
    c = c.get("name", c) if isinstance(c, dict) else c
    z_levels = list(meta.get("z_levels") or range(int(meta.get("n_z", 1) or 1)))
    n = 0
    for region, fov in _iter_wells(meta, select_fovs):
        for z in z_levels:
            if n >= limit:
                return
            try:
                plane = reader.read(region, fov, c, z, 0)
            except Exception:
                continue      # a corrupt plane must not abort a whole plate's estimate
            if plane is None or plane.ndim != 2 or plane.size == 0:
                continue
            n += 1
            yield plane


def _iter_wells(meta, select_fovs) -> Iterator[tuple[str, int]]:
    """Yield ``(region, fov)`` pairs in a stable order, whatever shape select_fovs returns."""
    try:
        chosen = select_fovs(meta, n_fovs=1)
    except Exception:
        return
    for region, fovs in sorted(chosen.items()):
        for fov in (fovs if isinstance(fovs, (list, tuple, set)) else [fovs]):
            yield region, int(fov)


def write_provenance(out_dir, info: dict) -> Path:
    """Write a JSON sidecar recording how an output was produced.

    Background subtraction is destructive and irreversible: the corrected plate does not carry
    the level that was removed, and two runs at different percentiles otherwise land on the
    same output path. The sidecar makes an archived plate self-describing.

    Returns the path written.
    """
    path = Path(out_dir) / "squidmip-provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, sort_keys=True, default=str) + "\n")
    return path
