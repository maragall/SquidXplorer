"""Per-FOV maximum-intensity projection (IMA-183; multi-FOV selection IMA-187).

Consumes the IMA-189 reader (``open_reader``) and produces one projected image per
selected FOV. The projection reduces the **z-axis only**; timepoint (t) and channel (c)
are preserved. Per-FOV output is 5-D ``(T, C, 1, Y, X)`` in Squid's canonical Zarr axis
order (TCZYX, verified in ``Squid/.../job_processing.py``) with **Z kept size-1** — the
MIP is an in-place z-reduction (Nz -> 1), not an axis removal, so the IMA-184 OME-zarr
writer needs no special-casing.

Data flow::

    open_reader(path).metadata ──► select_fovs(meta, n_fovs) ──► {well: [fov, ...]}
                                   (n_fovs=None ──► ALL FOVs per well, ragged ok)
                                                                          │
                                              for each (well, fov):       ▼
        project_well(reader, well, fov)
            for t in range(n_t), for channel c:
                planes = (reader.read(well, fov, c, z, t) for z in z_levels)   # streamed
                out[t, c, 0] = project(planes)        # running np.maximum, bounded memory
            └──► (T, C, 1, Y, X) native dtype

Design contracts:
  * A **z-selecting** reduction (``project_reference``) is the exception to that per-channel loop:
    it chooses one z, and a geometric choice must not be re-solved per channel or the channels of
    one FOV land on different planes and stop overlaying. ``project_well`` solves it once per
    ``(t, fov)`` on the reference channel and reads that z for every channel, recording the z
    actually consumed in ``picked_z`` and asserting the channels agree.
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

from typing import TYPE_CHECKING, Callable, Iterable, Optional

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


def select_reference_z(planes: Iterable[np.ndarray]) -> int:
    """Return the POSITION (0-based, into the iterable) of the sharpest plane by Tenengrad.

    The *selection* half of :func:`project_reference`, split out so a caller can solve the focus
    once and then apply the same z to other channels — see :func:`project_well`. Streaming and
    bounded: only a score is carried between planes, never a plane. Ties keep the earliest.
    """
    best_i, best_score = None, -np.inf
    shape = dtype = None
    for i, plane in enumerate(planes):
        if shape is None:
            shape, dtype = plane.shape, plane.dtype
        elif plane.shape != shape:
            raise ValueError(f"plane shape {plane.shape} != first plane {shape}")
        elif plane.dtype != dtype:
            raise ValueError(f"plane dtype {plane.dtype} != first plane {dtype}")
        score = _tenengrad(plane)
        if score > best_score:
            best_i, best_score = i, score
    if best_i is None:
        raise ValueError("select_reference_z requires at least one plane; got an empty iterable.")
    return best_i


def project_reference(planes: Iterable[np.ndarray]) -> np.ndarray:
    """Reference-plane reduction: return the single sharpest z-plane by Tenengrad focus.

    In HCS drug discovery you don't scrub Z — you MIP or pick one best-focus reference plane, then
    work in T. This is that pick, as a z-reduction Strategy (drops in via ``add_projector``). Like
    :func:`project` it is **streaming + bounded**: it keeps only the best plane seen so far (a copy)
    plus the current one, never the whole stack. Ties keep the earliest (lowest z).

    WARNING — this is a *z-selecting* reduction, and a z-selection must not be re-solved per
    channel: channels of one FOV sampled at different z do not overlay (misregistration). Do not
    call it once per channel. :func:`project_well` recognises it via the ``select_index`` attribute
    below and instead solves the focus ONCE per ``(t, fov)`` on the reference channel, then reads
    that one z for every channel. The attribute is the contract any future z-selecting projector
    (EDF pick, brightest-plane, …) implements to get the same c-aligned treatment for free.
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


# The z-selecting marker: a projector carrying ``.select_index`` tells project_well "I choose one z
# rather than combining all of them", so project_well solves that z once per (t, fov) and shares it
# across channels instead of calling the projector per channel.
project_reference.select_index = select_reference_z


def project_well(
    reader: "SquidReader",
    region: str,
    fov: int,
    reduce: Callable[[Iterable[np.ndarray]], np.ndarray] = project,
    reference_channel: Optional[str] = None,
    picked_z: Optional[dict] = None,
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
    reference_channel:
        Which channel drives focus selection when *reduce* is z-selecting (carries
        ``select_index``). Defaults to the acquisition's FIRST channel — see the c-alignment
        note below. Ignored by combining reductions such as MIP. Must name a real channel.
    picked_z:
        Optional out-dict for **provenance**: filled with ``{(t, channel): z}`` — the z index
        actually consumed for every (t, channel) written. Left empty by combining reductions
        (a MIP consumes every z, so no single index describes it). Callers/tests use it to
        assert the c-alignment invariant on data rather than trusting a comment.

    c-alignment (the invariant)
    ---------------------------
    A reference projector is **z-selecting but c-aligned and t-aligned**: it may choose *which*
    z to sample, but all channels of one (t, fov) must be sampled at the SAME z or they do not
    overlay. Rule of thumb: do not re-solve a geometric selection on an axis you do not consume.
    So the focus is estimated ONCE per ``(t, fov)`` on *reference_channel* and that single z is
    read for every channel; the invariant ``len({picked_z[(t, c)] for c in channels}) == 1`` is
    asserted here, per timepoint, not merely documented.

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

    if picked_z is None:
        picked_z = {}

    # A z-SELECTING projector advertises how to pick the index (see project_reference).
    select_index = getattr(reduce, "select_index", None)

    if select_index is None:
        # Combining reduction (MIP, mean, …): every z is consumed in every channel, so there is
        # no shared geometric choice to align and nothing to record in picked_z.
        out = np.empty((n_t, len(channels), 1, y, x), dtype=meta["dtype"])
        for t in range(n_t):
            for c_i, channel in enumerate(channels):
                planes = (reader.read(region, fov, channel, z, t) for z in z_levels)
                out[t, c_i, 0] = reduce(planes)  # streamed z; bounded memory
        return out

    # z-selecting: ONE focus solve per (t, fov), shared by every channel.
    ref = channels[0] if reference_channel is None else reference_channel
    if ref not in channels:
        raise ValueError(
            f"reference_channel {ref!r} is not a channel of this acquisition: {channels}"
        )

    out = np.empty((n_t, len(channels), 1, y, x), dtype=meta["dtype"])
    for t in range(n_t):
        planes = (reader.read(region, fov, ref, z, t) for z in z_levels)
        z_star = z_levels[select_index(planes)]   # position -> real z label
        for c_i, channel in enumerate(channels):
            out[t, c_i, 0] = reader.read(region, fov, channel, z_star, t)
            picked_z[(t, channel)] = z_star       # provenance: the z actually consumed
        # Data-checked invariant, not a comment: one z per (t, fov) across ALL channels.
        assert len({picked_z[(t, c)] for c in channels}) == 1, (
            f"channel misregistration at region={region!r} fov={fov} t={t}: "
            f"{ {c: picked_z[(t, c)] for c in channels} }"
        )
    return out


def select_fovs(metadata: dict, n_fovs: Optional[int] = 1) -> dict[str, list[int]]:
    """Pick the FOV(s) to project for each well.

    Selection is positional — the first ``n_fovs`` FOVs of each well, in the reader's sorted
    ``fovs_per_region`` order (so it never depends on a literal, possibly 1-based, filename
    FOV label).

    ``n_fovs=None`` means **all FOVs in each well** (IMA-187's mosaic path). That is a
    distinct request from any specific count, and it changes the ragged-plate behaviour on
    purpose: with an explicit count, a well holding fewer FOVs than asked for is bad input and
    raises; with ``None``, each well simply contributes what it has, so one short well (a
    skipped position, an interrupted acquisition) cannot abort a multi-hour plate run.

    Parameters
    ----------
    metadata:
        ``reader.metadata`` from IMA-189.
    n_fovs:
        FOVs per well. ``None`` = all. Default 1.

    Returns
    -------
    dict[str, list[int]]
        ``{region: [fov, ...]}`` for every region. Every list has length ``n_fovs`` unless
        ``n_fovs`` is ``None``, in which case lengths may differ between wells.

    Raises
    ------
    ValueError
        If ``n_fovs`` is an int < 1, or (explicit counts only) a well has fewer than
        ``n_fovs`` FOVs — named loud, never a silent short slice.
    """
    if n_fovs is not None and n_fovs < 1:
        raise ValueError(f"n_fovs must be >= 1 or None (= all), got {n_fovs}")

    fovs_per_region = metadata["fovs_per_region"]
    selected: dict[str, list[int]] = {}
    for region in metadata["regions"]:
        available = fovs_per_region[region]
        if n_fovs is None:
            selected[region] = list(available)      # every FOV; ragged wells are fine
            continue
        if n_fovs > len(available):
            raise ValueError(
                f"n_fovs={n_fovs} requested but region {region!r} has only "
                f"{len(available)} FOV(s): {available}. Pass n_fovs=None to take whatever "
                "each well has instead of requiring a uniform count."
            )
        selected[region] = list(available[:n_fovs])
    return selected


def resolve_n_fovs(metadata: dict, n_fovs: Optional[int]) -> int:
    """Concrete FOV-per-well count for callers that need an int, resolving ``None`` to the max.

    OME-NGFF's ``field_count`` is a single plate-level scalar, so a plate whose wells hold
    different FOV counts has to report one number; the max is the only value that does not
    under-describe some well. Exists so ``None`` never reaches an ``int()`` call (it would
    raise TypeError deep inside the writer, far from the caller that passed it).
    """
    if n_fovs is not None:
        return int(n_fovs)
    per_region = metadata["fovs_per_region"]
    return max((len(per_region[r]) for r in metadata["regions"]), default=0)
