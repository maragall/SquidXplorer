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

import functools
from typing import TYPE_CHECKING, Callable, Iterable, Optional

import numpy as np

if TYPE_CHECKING:  # avoid import cost / cycle at runtime
    from squidmip.reader import SquidReader


# --- IMA-210: which axis does an operator consume? -------------------------------------------
#
# Every operator here has ONE callable shape — ``Iterable[plane] -> plane`` — and differs only in
# which axis the engine groups over before calling it. That single declaration drives the loop:
#
#   PLANE_OP  = frozenset()        plane -> plane. z is NOT consumed, so z SURVIVES at full depth.
#                                  Deconvolution, background subtraction, flat-field (IMA-223/4/5).
#   Z_REDUCER = frozenset({"z"})   all z of one (t, c) -> one plane. z collapses to size 1.
#                                  MIP (``project``) and reference (``project_reference``).
#
# ``consumes`` is WHICH axis an operator eats. It is orthogonal to ``select_index``, which is HOW a
# z-reducer picks within z (combine every plane vs choose one). ``project_reference`` is BOTH a
# ``{"z"}`` consumer and z-selecting; conflating the two is what once landed the channels of one
# FOV on different z. Do not encode "selects" as a separate consumed axis.
#
# {"fov"} is deliberately NOT a member: stitching is inter-FOV and needs each tile's x/y stage
# geometry, which an ``Iterable[plane] -> plane`` callable never sees. That seam is IMA-222's
# ``_REGION_OPERATORS``/``stitch_plate()``, not this table.
PLANE_OP: frozenset[str] = frozenset()
Z_REDUCER: frozenset[str] = frozenset({"z"})
CONSUMABLE_AXES: frozenset[str] = frozenset({"z"})


def normalise_consumes(consumes) -> frozenset[str]:
    """Coerce a ``consumes`` declaration to a frozenset of axis names, refusing anything unsupported.

    Accepts any iterable of axis names (``{"z"}``, ``()``, ``"z"``) so callers are not forced to
    spell ``frozenset`` at every registration site. Refuses by name rather than silently ignoring:
    a projector that believes it consumes an axis the engine does not group over would run over the
    wrong data.

    Raises
    ------
    ValueError
        If an axis is not consumable here. ``"fov"`` gets its own message pointing at IMA-222's
        region-operator seam, because it is the one people reach for and the one this shape of
        callable structurally cannot serve.
    """
    if isinstance(consumes, str):
        consumes = (consumes,)
    axes = frozenset(consumes)
    if "fov" in axes:
        raise ValueError(
            "consumes={'fov'} is not supported by the projector table: a projector is "
            "Iterable[plane] -> plane and never sees a tile's x/y stage geometry, which any "
            "inter-FOV operation (stitching, illumination-field fitting across a well) requires. "
            "Inter-FOV operators belong to the region-operator seam (IMA-222), not here."
        )
    unknown = axes - CONSUMABLE_AXES
    if unknown:
        raise ValueError(
            f"unsupported axis {sorted(unknown)[0]!r} in consumes={sorted(axes)}; this engine "
            f"groups over {sorted(CONSUMABLE_AXES)} only. A plane-op declares consumes=frozenset(), "
            "a z-reduction declares consumes=frozenset({'z'})."
        )
    return axes


def plane_op(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Lift a natural ``plane -> plane`` function into the engine's ``Iterable[plane] -> plane`` shape.

    The point of ONE callable shape is that the engine has ONE loop: it groups the input planes over
    the consumed axes and calls the operator per group. For a plane-op the group is a single plane,
    so the author should not have to unpack an iterable by hand — this adapter does it, and stamps
    ``consumes = PLANE_OP`` on the result so :func:`squidmip.add_projector` infers the declaration::

        add_projector("bgsub", plane_op(subtract_background))    # consumes inferred = frozenset()

    Handing the adapted callable more than one plane raises instead of quietly using the first: that
    can only happen if it was registered as a z-reducer, i.e. a seam bug that would otherwise show up
    as "my background subtraction silently dropped all but one z".
    """
    @functools.wraps(fn)
    def _apply(planes: Iterable[np.ndarray]) -> np.ndarray:
        it = iter(planes)
        try:
            plane = next(it)
        except StopIteration:
            raise ValueError(f"plane-op {getattr(fn, '__name__', fn)!r} requires one plane; "
                             "got an empty iterable.") from None
        if next(it, None) is not None:
            raise ValueError(
                f"plane-op {getattr(fn, '__name__', fn)!r} was handed more than one plane. A "
                "plane-op maps plane -> plane and must be registered with consumes=frozenset(); "
                "registered as a z-reducer it would silently discard every plane but the first."
            )
        return fn(plane)

    _apply.consumes = PLANE_OP      # the declaration, carried on the callable (cf. select_index)
    return _apply


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

# The consumed-axis declarations (IMA-210). BOTH shipped projectors eat z — combining every plane
# (mip) and choosing one (reference) are two ways to consume the SAME axis, not two axes.
project.consumes = Z_REDUCER
project_reference.consumes = Z_REDUCER


def project_well(
    reader: "SquidReader",
    region: str,
    fov: int,
    reduce: Callable[[Iterable[np.ndarray]], np.ndarray] = project,
    reference_channel: Optional[str] = None,
    picked_z: Optional[dict] = None,
    consumes=None,
    t: Optional[int] = None,
) -> np.ndarray:
    """Apply one operator to a FOV's planes for every channel and timepoint.

    Group-by-then-reduce, with the grouping derived from the operator's ``consumes``
    declaration (IMA-210) — never from a per-operator branch:

    ====================  ==================================  ===================
    ``consumes``          group handed to *reduce*            output shape
    ====================  ==================================  ===================
    ``frozenset({"z"})``  every z of one (t, c) — the stack   ``(T, C, 1, Y, X)``
    ``frozenset()``       one plane                           ``(T, C, Nz, Y, X)``
    ====================  ==================================  ===================

    So a z-reduction (MIP, reference) collapses z to 1, and a plane-op (deconvolution,
    background subtraction, flat-field) leaves z at full depth — it is mapped over the planes,
    never routed through the z-reduction. t and c are preserved either way, output is TCZYX in
    the reader's native dtype, and z stays an axis (never removed) so the OME-zarr writer needs
    no special-casing.

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
        (a MIP consumes every z, so no single index describes it) and by plane-ops (which make
        no geometric choice — every plane is kept, at its own z). Callers/tests use it to
        assert the c-alignment invariant on data rather than trusting a comment.
    consumes:
        The operator's consumed-axis declaration (see the table above). ``None`` (default) reads
        it off the callable's ``consumes`` attribute — :func:`plane_op` stamps one — and falls
        back to ``frozenset({"z"})``, the shipped z-reduction contract, so every existing caller
        is byte-for-byte unchanged. :func:`squidmip.add_projector` passes the registry's
        declaration through :func:`squidmip.project_plate`.

    c-alignment (the invariant)
    ---------------------------
    A reference projector is **z-selecting but c-aligned and t-aligned**: it may choose *which*
    z to sample, but all channels of one (t, fov) must be sampled at the SAME z or they do not
    overlay. Rule of thumb: do not re-solve a geometric selection on an axis you do not consume.
    So the focus is estimated ONCE per ``(t, fov)`` on *reference_channel* and that single z is
    read for every channel; the invariant ``len({picked_z[(t, c)] for c in channels}) == 1`` is
    asserted here, per timepoint, not merely documented.
    t:
        Which timepoint to project. ``None`` (default) projects *all* of them, which is
        what every plate-scale caller wants. An int projects only that one and returns
        ``T=1`` — the single-frame consumers (IMA-228's Minerva export) need one timepoint,
        and reading all ``n_t`` to then discard ``n_t - 1`` of them is an ``n_t``-fold
        wasted read of the whole z-stack.

    Returns
    -------
    np.ndarray
        Shape ``(T, n_channels, 1, Y, X)`` where ``T`` is ``n_t`` when ``t is None`` and
        ``1`` otherwise; dtype ``reader.metadata["dtype"]``. Channels are ordered as
        ``reader.metadata["channels"]`` (kept distinct — no z-as-channel collapse).

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

    # The operator's own declaration wins; absent one, the shipped z-reduction contract.
    if consumes is None:
        consumes = getattr(reduce, "consumes", Z_REDUCER)
    consumes = normalise_consumes(consumes)

    if t is None:
        timepoints = tuple(range(n_t))
    else:
        if not 0 <= t < n_t:
            raise ValueError(f"timepoint {t} out of range for an acquisition with n_t={n_t}")
        timepoints = (t,)

    # A z-SELECTING projector advertises how to pick the index (see project_reference).
    select_index = getattr(reduce, "select_index", None)

    if select_index is not None and "z" not in consumes:
        raise ValueError(
            f"{getattr(reduce, '__name__', reduce)!r} carries select_index (it CHOOSES a z) but "
            f"declares consumes={sorted(consumes)}; a z-selecting operator must consume z."
        )

    if select_index is None:
        # THE group-by-then-reduce loop. The consumed axes decide the grouping and nothing else:
        #   z consumed     -> one group per (t, c): the whole stack  -> one output plane, Z=1
        #   z not consumed -> one group per (t, c, z): a single plane -> Z survives at full depth
        # Both cases call the SAME callable shape, so a new plane-op needs no engine edit.
        z_groups = [tuple(z_levels)] if "z" in consumes else [(z,) for z in z_levels]
        out = np.empty((len(timepoints), len(channels), len(z_groups), y, x), dtype=meta["dtype"])
        for t_i, t_src in enumerate(timepoints):
            for c_i, channel in enumerate(channels):
                for k, group in enumerate(z_groups):
                    planes = (reader.read(region, fov, channel, z, t_src) for z in group)
                    out[t_i, c_i, k] = reduce(planes)  # streamed z; bounded memory
        # Nothing lands in picked_z: a combining reduction consumes every z (no single index
        # describes it) and a plane-op chooses nothing (every plane is kept, at its own z).
        return out

    # z-selecting: ONE focus solve per (t, fov), shared by every channel.
    ref = channels[0] if reference_channel is None else reference_channel
    if ref not in channels:
        raise ValueError(
            f"reference_channel {ref!r} is not a channel of this acquisition: {channels}"
        )

    out = np.empty((len(timepoints), len(channels), 1, y, x), dtype=meta["dtype"])
    for t_i, t_src in enumerate(timepoints):
        planes = (reader.read(region, fov, ref, z, t_src) for z in z_levels)
        z_star = z_levels[select_index(planes)]   # position -> real z label
        for c_i, channel in enumerate(channels):
            out[t_i, c_i, 0] = reader.read(region, fov, channel, z_star, t_src)
            picked_z[(t_src, channel)] = z_star   # provenance: the z actually consumed
        # Data-checked invariant, not a comment: one z per (t, fov) across ALL channels.
        assert len({picked_z[(t_src, c)] for c in channels}) == 1, (
            f"channel misregistration at region={region!r} fov={fov} t={t_src}: "
            f"{ {c: picked_z[(t_src, c)] for c in channels} }"
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
