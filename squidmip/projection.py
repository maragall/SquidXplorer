"""One-FOV-per-well maximum-intensity projection (IMA-183, folds IMA-187).

Consumes the IMA-189 reader (``open_reader``) and produces one projected image per
selected FOV. The reductions in this module reduce the **z-axis only** (they are the
``consumes={"z"}`` entries of the IMA-226 projector registry); timepoint (t) and channel (c)
are preserved. Per-FOV output is 5-D ``(T, C, 1, Y, X)`` in Squid's canonical Zarr axis
order (TCZYX, verified in ``Squid/.../job_processing.py``) with **Z kept size-1** — the
MIP is an in-place z-reduction (Nz -> 1), not an axis removal, so the IMA-184 OME-zarr
writer needs no special-casing. Operators that consume a different axis (a plane operator
preserving z, an FOV reducer) declare it on the registry and do NOT flow through this
``reduce=`` seam unchanged — see ``_engine.ProjectorSpec``.

Data flow::

    open_reader(path).metadata ──► select_fovs(meta, n_fovs=1) ──► {well: [fov, ...]}
                                                                          │
                                              for each (well, fov):       ▼
        project_well(reader, well, fov)
            for t in range(n_t), for channel c:
                planes = (reader.read(well, fov, c, z, t) for z in z_levels)   # streamed
                out[t, c, 0] = project(planes)        # running np.maximum, bounded memory
            └──► (T, C, 1, Y, X) native dtype

Design contracts:
  * ``project`` is a pure, dtype-preserving, bounded-memory reduction — it streams planes
    and never materialises the whole z-stack. It is the primitive IMA-188 wraps in its
    parallel/streaming engine and registers as the projector (MIP now, EDF later).
  * The z iterator is ``metadata["z_levels"]`` (the real, filename-derived z indices),
    NOT ``range(n_z)`` — ``n_z`` is a *count*, so ``range`` would be wrong the moment z is
    non-contiguous (a partial acquisition: files {0,1,3} -> z_levels [0,1,3], n_z 3).
  * Native dtype (uint8/uint16) is preserved end to end; no cast, no upcast.
  * IMA-183 depends only on metadata fields that are complete for BOTH the acquisition.yaml
    and the legacy pre-yaml generations (regions/fovs/z_levels/channels/frame_shape/dtype).
    The yaml-only scalars (pixel_size_um, wellplate_format) are never required here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

if TYPE_CHECKING:  # avoid import cost / cycle at runtime
    from squidmip.reader import SquidReader


def project(planes: Iterable[np.ndarray]) -> np.ndarray:
    """Maximum-intensity project an iterable of planes into one plane.

    A pure, dtype-preserving, bounded-memory reduction: it consumes *planes* one at a
    time and keeps a single running-max accumulator, so it never holds the whole stack.
    This is the projector primitive IMA-188 wraps (parallel/streaming) and swaps (EDF).

    Parameters
    ----------
    planes:
        Iterable of equal-shape, equal-dtype arrays (typically one FOV's z-planes for a
        single channel/timepoint). Must yield at least one plane.

    Returns
    -------
    np.ndarray
        The element-wise maximum, same shape and dtype as the input planes.

    Raises
    ------
    ValueError
        If *planes* is empty, or a later plane's shape/dtype differs from the first.
    """
    it = iter(planes)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("project() requires at least one plane; got an empty iterable.")

    acc = np.array(first, copy=True)  # own buffer; never mutate the caller's plane
    for plane in it:
        if plane.shape != acc.shape:
            raise ValueError(f"plane shape {plane.shape} != first plane {acc.shape}")
        if plane.dtype != acc.dtype:
            raise ValueError(f"plane dtype {plane.dtype} != first plane {acc.dtype}")
        np.maximum(acc, plane, out=acc)  # in place -> dtype preserved, no extra buffer
    return acc


def _tenengrad(plane: np.ndarray) -> float:
    """Tenengrad focus measure — sum of squared gradient magnitude. Higher = sharper (more in focus).

    The standard, cheap autofocus metric: a sharp plane has strong edges (large gradients). Computed
    in float32 on a transient gradient pair (freed each call), so it adds no lasting memory.
    """
    gy, gx = np.gradient(plane.astype(np.float32, copy=False))
    return float(np.square(gx).sum() + np.square(gy).sum())


def project_reference(planes: Iterable[np.ndarray]) -> np.ndarray:
    """Reference-plane reduction: return the single sharpest z-plane by Tenengrad focus.

    In HCS drug discovery you don't scrub Z — you MIP or pick one best-focus reference plane, then
    work in T. This is that pick, as a z-reduction Strategy (drops in via ``add_projector``). Like
    :func:`project` it is **streaming + bounded**: it keeps only the best plane seen so far (a copy)
    plus the current one, never the whole stack. Ties keep the earliest (lowest z).
    """
    it = iter(planes)
    try:
        best = next(it)
    except StopIteration:
        raise ValueError("project_reference requires at least one plane; got an empty iterable.")
    best = np.array(best, copy=True)          # own buffer; never alias the caller's plane
    best_score = _tenengrad(best)
    for plane in it:
        if plane.shape != best.shape:
            raise ValueError(f"plane shape {plane.shape} != first plane {best.shape}")
        if plane.dtype != best.dtype:
            raise ValueError(f"plane dtype {plane.dtype} != first plane {best.dtype}")
        score = _tenengrad(plane)
        if score > best_score:
            best_score, best = score, np.array(plane, copy=True)
    return best


def project_well(
    reader: "SquidReader",
    region: str,
    fov: int,
    reduce: Callable[[Iterable[np.ndarray]], np.ndarray] = project,
) -> np.ndarray:
    """Project one FOV's z-stack for every channel and timepoint.

    Reduces z only; t and c are preserved. Output is ``(T, C, 1, Y, X)`` (TCZYX, Z=1) in
    the reader's native dtype.

    Parameters
    ----------
    reader:
        An IMA-189 ``SquidReader`` (from ``open_reader``).
    region, fov:
        The well and field-of-view to project (a valid ``(region, fov)`` per the reader).
    reduce:
        The z-reduction primitive. Defaults to :func:`project` (MIP). IMA-188 passes its
        own projector here (EDF/EMF/…) — this is the pluggable seam; 183 ships MIP only.

    Returns
    -------
    np.ndarray
        Shape ``(n_t, n_channels, 1, Y, X)``, dtype ``reader.metadata["dtype"]``.
        Channels are ordered as ``reader.metadata["channels"]`` (kept distinct — no
        z-as-channel collapse).

    Notes
    -----
    A corrupt/unreadable plane surfaces as the reader's own error from ``read()`` and is
    propagated loud — never swallowed into a silent partial projection.
    """
    meta = reader.metadata
    channels = [c["name"] for c in meta["channels"]]
    z_levels = meta["z_levels"]
    n_t = meta["n_t"]
    y, x = meta["frame_shape"]

    out = np.empty((n_t, len(channels), 1, y, x), dtype=meta["dtype"])
    for t in range(n_t):
        for c_i, channel in enumerate(channels):
            planes = (reader.read(region, fov, channel, z, t) for z in z_levels)
            out[t, c_i, 0] = reduce(planes)  # streamed z; bounded memory
    return out


def select_fovs(metadata: dict, n_fovs: int = 1) -> dict[str, list[int]]:
    """Pick the FOV(s) to project for each well.

    Folds IMA-187: the FOV *count* is a data-model parameter and the return is a list per
    well, so up-to-4-FOV support needs no data-model change. v1 uses ``n_fovs=1`` (one FOV
    per well). Selection is positional — the first ``n_fovs`` FOVs of each well, in the
    reader's sorted ``fovs_per_region`` order (so it never depends on a literal, possibly
    1-based, filename FOV label).

    Parameters
    ----------
    metadata:
        ``reader.metadata`` from IMA-189.
    n_fovs:
        FOVs to select per well (default 1).

    Returns
    -------
    dict[str, list[int]]
        ``{region: [fov, ...]}`` for every region, each list of length ``n_fovs``.

    Raises
    ------
    ValueError
        If ``n_fovs < 1``, or a well has fewer than ``n_fovs`` FOVs (named loud, never a
        silent short slice).
    """
    if n_fovs < 1:
        raise ValueError(f"n_fovs must be >= 1, got {n_fovs}")

    fovs_per_region = metadata["fovs_per_region"]
    selected: dict[str, list[int]] = {}
    for region in metadata["regions"]:
        available = fovs_per_region[region]
        if n_fovs > len(available):
            raise ValueError(
                f"n_fovs={n_fovs} requested but region {region!r} has only "
                f"{len(available)} FOV(s): {available}"
            )
        selected[region] = list(available[:n_fovs])
    return selected
