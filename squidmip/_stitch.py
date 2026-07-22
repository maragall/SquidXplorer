"""Stitch operator (IMA-222): the first REAL stitcher wired into the plate.

This module adds a **region operator** — an operation whose unit of work is a whole well
(all of its FOVs at once) rather than one FOV's z-stack — and ships one: ``stitch``, which
registers a well's FOVs against each other and fuses them into a single seamless mosaic.

Why a parallel registry instead of ``_PROJECTORS``
--------------------------------------------------
``_engine._PROJECTORS`` is a **z-reduction** table: ``Iterable[plane] -> plane``. Every entry
sees one channel's z-planes of ONE FOV and nothing else. A stitcher cannot live there. It is
an *inter-FOV* operation: it needs every FOV of the well simultaneously **plus each FOV's x/y
stage geometry**, and that signature carries neither. Registering a stitcher as a "projector"
would be a type lie that only fails at runtime.

So IMA-222 builds a **parallel** table, :data:`_REGION_OPERATORS`, whose entries have the
shape the work actually has::

    RegionOperator = Callable[[SquidReader, str, list[int]], np.ndarray]   # -> (T, C, 1, Y, X)
                                 reader,  region,  fovs

and a :func:`stitch_plate` generator that **mirrors ``project_plate``'s exact contract** —
same keyword names (``n_fovs``/``workers``/``on_error``/``regions``), same bounded in-flight
window, same ``(region, fov, (T, C, 1, Y, X))`` yield — so the viewer's
``_OperatorWorker._on_well`` consumes it with no change to its body.

The one contract difference, stated loud: ``project_plate``'s task is a **FOV**, so a 27-FOV
well yields 27 arrays. ``stitch_plate``'s task is a **REGION**, so a 27-FOV well yields
exactly ONE array — the fused mosaic — reported against the well's first FOV as the anchor
index. That is the whole point of stitching, and it is why ``workers`` defaults *low* here
(one fused 5x6 mosaic of 2084px tiles is ~0.9 GB at 4 channels, versus ~139 MB for one
projected FOV).

The algorithm is NOT reimplemented
----------------------------------
Every step below is Julio's own ``tilefusion``, called in the same order and with the same
parameters as ``TileFusion.run()`` drives them — but on **in-memory arrays**, because
``TileFusion`` is a file->file pipeline that writes a fused OME-Zarr, which is unusable both
for a streaming viewer operator and on a disk-constrained machine::

    projection.project_well          per-FOV z-reduction (MIP), the IMA-183 primitive
    registration.find_adjacent_pairs        which FOVs actually overlap, from stage geometry
    registration.rotation_aware_max_shift   residual-rejection cap, adaptive to tile spacing
    registration.compute_pair_bounds        the overlap strip of each pair, per tile
    registration.register_pairs_batched     phase correlation (upsample 20) + NCC score
    optimization._edges_from_pairwise_metrics    pairwise metrics -> weighted pose-graph edges
    optimization.two_round_optimization     global least-squares solve + blunder rejection
    utils.make_1d_profile                   the Hann feather ramp
    fusion.fuse_plane                       sub-pixel placement + feathered blend, block-wise

Geometry, and the units trap
----------------------------
Positions come from ``metadata["fov_positions_um"]`` — stage **micrometres**, ``{(region,
fov): (x_um, y_um)}``. ``tilefusion`` works in ``(y, x)``, so the pair is swapped on the way
in; a mm value anywhere in this module is a bug (see ``_placement.py``'s units note). The
step is read from the coordinates, never from a config: on the 10x tissue acquisition the
measured stage step is 1410.45 um against a 1.567 mm tile, i.e. ~208 px (~10%) of real
overlap — while ``acquisition parameters.json`` advertises 0.9 mm, which is simply wrong.
Trusting the config would compute a negative overlap and register nothing.
"""

from __future__ import annotations

import contextlib
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Sequence

import numpy as np

from squidmip._engine import _default_workers, _resolve_projector
from squidmip.projection import project_well, select_fovs

if TYPE_CHECKING:  # avoid import cost / cycle at runtime
    from squidmip.reader import SquidReader

# --------------------------------------------------------------------------------------
# tilefusion defaults, copied from TileFusion.run() so this path is parameter-identical to
# Julio's pipeline. They are module constants (not magic numbers at the call site) so a
# future sweep changes them in exactly one greppable place.
# --------------------------------------------------------------------------------------
_DOWNSAMPLE_FACTORS = (1, 1)   # registration MUST be full-res; any downsample coarsens the
#                                sub-pixel shift (see registration._UPSAMPLE_FACTOR's note)
_SSIM_WINDOW = 15              # kept for API compatibility with register_and_score
_REL_THRESH = 0.5              # TileFusion.run(): optimize_shifts(TWO_ROUND_ITERATIVE, ...)
_ABS_THRESH = 2.0
_MIN_OVERLAP_PX = 15           # find_adjacent_pairs default
_BLEND_PX = 128                # feather ramp width. Sized to the MEASURED ~208 px overlap on
#                                the 10x tissue set: the ramp must fit inside the overlap
#                                (a ramp wider than the overlap never reaches full weight and
#                                dims the seam) with margin for registration residual.
_BLOCK_PX = 2048               # fusion scratchpad edge. Bounds peak fusion memory to
#                                C x 2048^2 x 4 B x 2 buffers (~134 MB at C=4) regardless of
#                                mosaic size. fuse_plane's output is block-size independent.


class _NullTimer:
    """Stand-in for ``profiling.stages.StageTimer`` when the caller passes none.

    The library must not hard-depend on the stitcher repo's profiling package just to run;
    callers that DO want timings pass the real ``StageTimer`` (that is Julio's own profiler
    and the only one this code should ever use) and get spans for free.
    """

    @contextlib.contextmanager
    def stage(self, name: str):
        yield


def _positions_yx_um(
    metadata: dict, region: str, fovs: Sequence[int]
) -> list[tuple[float, float]]:
    """``[(y_um, x_um), ...]`` for *fovs*, in tilefusion's (y, x) order.

    ``metadata["fov_positions_um"]`` stores ``(x_um, y_um)``; tilefusion's
    ``find_adjacent_pairs`` / ``fuse_plane`` are ``(y, x)`` throughout. The swap happens
    HERE, once, rather than at four call sites where three of them would eventually be
    right and one would silently transpose the mosaic.

    Raises
    ------
    KeyError
        If any FOV has no recorded stage position. A missing position cannot be guessed:
        placing it at (0, 0) would stack it on the anchor and corrupt every registration
        pair it touches, so this refuses rather than drawing a wrong mosaic.
    """
    positions = metadata["fov_positions_um"]
    missing = [f for f in fovs if (region, f) not in positions]
    if missing:
        raise KeyError(
            f"region {region!r}: no stage position for FOV(s) {missing[:8]}; cannot stitch "
            "without geometry (coordinates.csv and the image filenames disagree)."
        )
    return [(float(positions[(region, f)][1]), float(positions[(region, f)][0])) for f in fovs]


def _pixel_size(metadata: dict) -> tuple[float, float]:
    """Isotropic object-space pixel size as tilefusion's ``(py, px)`` pair, validated."""
    p = metadata.get("pixel_size_um")
    if p is None:
        raise ValueError(
            "pixel_size_um is required to stitch (stage micrometres must become pixels), but "
            "the acquisition metadata has none. Add objective.pixel_size_um to acquisition.yaml."
        )
    p = float(p)
    if not p > 0:
        raise ValueError(f"pixel_size_um must be > 0, got {metadata.get('pixel_size_um')!r}")
    return (p, p)


def _resolve_registration_channel(metadata: dict, registration_channel) -> int:
    """Index of the channel registration runs on (an operator choice, never automatic).

    Mirrors ``TileFusion``'s ``channel_to_use`` policy — one channel drives the geometry and
    every channel is then fused with that ONE solution, because channels of a FOV share a
    sensor and must not be given independent, disagreeing placements.
    """
    names = [c["name"] for c in metadata["channels"]]
    if registration_channel is None:
        return 0
    if isinstance(registration_channel, str):
        if registration_channel not in names:
            raise ValueError(
                f"registration_channel {registration_channel!r} is not a channel of this "
                f"acquisition: {names}"
            )
        return names.index(registration_channel)
    idx = int(registration_channel)
    if not 0 <= idx < len(names):
        raise ValueError(f"registration_channel index {idx} out of range for {len(names)} channels")
    return idx


def solve_offsets_px(
    tiles: np.ndarray,
    positions_yx_um: Sequence[tuple[float, float]],
    pixel_size: tuple[float, float],
    tile_shape: tuple[int, int],
    *,
    registration_channel: int = 0,
    max_workers: Optional[int] = None,
    rel_thresh: float = _REL_THRESH,
    abs_thresh: float = _ABS_THRESH,
    timer=None,
) -> np.ndarray:
    """Register the tiles against each other and return each tile's residual shift in PIXELS.

    The registration half of :func:`stitch_region`, split out so it can be tested on a
    synthetic mosaic with a KNOWN injected offset — "the solver recovered 7.0 px" is a real
    assertion, whereas "the mosaic rendered" is not.

    This is four ``tilefusion`` calls in ``TileFusion.run()``'s order, nothing else:
    ``find_adjacent_pairs`` -> ``rotation_aware_max_shift`` -> ``compute_pair_bounds`` ->
    ``register_pairs_batched``, then the pose-graph solve
    ``_edges_from_pairwise_metrics`` -> ``two_round_optimization``.

    Parameters
    ----------
    tiles:
        ``(n_tiles, C, Y, X)`` — one z-reduced plane stack per FOV, at ONE timepoint.
        Registration is geometry, solved once; it is not re-solved per timepoint.
    positions_yx_um:
        Stage positions in MICROMETRES, ``(y, x)`` (see :func:`_positions_yx_um`).
    pixel_size:
        ``(py, px)`` object-space micrometres per pixel.
    tile_shape:
        ``(Y, X)`` of one FOV.
    registration_channel:
        Channel index driving the geometry (all channels are then placed with this one
        solution — see :func:`_resolve_registration_channel`).
    rel_thresh, abs_thresh:
        BLUNDER REJECTION, handed straight to ``two_round_optimization``. After the first
        least-squares solve, a link whose residual exceeds BOTH ``rel_thresh`` x the median
        residual AND ``abs_thresh`` pixels is dropped, and the pose graph is re-solved
        without it. Both conditions must hold, which is why two numbers rather than one:
        the relative term adapts to how well this acquisition registers overall, and the
        absolute term stops a very clean plate (tiny median) from rejecting links that were
        only ever off by a fraction of a pixel.

        Defaults are ``TileFusion.run()``'s own 0.5 / 2.0, so an unset call is byte-for-byte
        what this module has always done. They are parameters rather than constants because
        maragall/stitcher exposes exactly these two as operator controls ("Outlier rel: N%"
        and "abs: N px") and the stitcher panel had nothing to bind to.
    timer:
        Optional ``profiling.stages.StageTimer``; spans ``register`` and ``optimize``.

    Returns
    -------
    np.ndarray
        ``(n_tiles, 2)`` float — the per-tile ``(dy, dx)`` correction in pixels to ADD to the
        stage-derived position. All-zero when nothing registered (no overlap, or every pair
        rejected), which degrades cleanly to pure coordinate placement rather than raising.
    """
    from tilefusion.optimization import _edges_from_pairwise_metrics, two_round_optimization
    from tilefusion.registration import (
        compute_pair_bounds,
        find_adjacent_pairs,
        register_pairs_batched,
        rotation_aware_max_shift,
    )

    # Refuse a degenerate threshold BEFORE any correlation runs. rel<=0 or abs<=0 makes the
    # rejection test vacuously true for every link, so the second round solves on an empty
    # edge set and hands back all-zero offsets -- which is indistinguishable from "the stage
    # was already perfect". A silently un-registered mosaic that reports success is the
    # failure mode this project has six confirmed instances of.
    if not (np.isfinite(rel_thresh) and rel_thresh > 0):
        raise ValueError(f"rel_thresh must be a positive finite number, got {rel_thresh!r}")
    if not (np.isfinite(abs_thresh) and abs_thresh > 0):
        raise ValueError(f"abs_thresh must be a positive finite number, got {abs_thresh!r}")

    timer = timer or _NullTimer()
    n_tiles = len(positions_yx_um)
    max_workers = max_workers or (os.cpu_count() or 8)

    with timer.stage("register"):
        adjacent_pairs = find_adjacent_pairs(
            list(positions_yx_um), pixel_size, tile_shape, min_overlap=_MIN_OVERLAP_PX
        )
        if not adjacent_pairs:
            # No pair overlaps enough to correlate. Not an error: a sparse/freeform
            # acquisition legitimately has isolated FOVs. Fall back to stage positions.
            return np.zeros((n_tiles, 2), dtype=np.float64)
        max_shift = rotation_aware_max_shift(adjacent_pairs)
        pair_bounds = compute_pair_bounds(adjacent_pairs, tile_shape)

        def read_region(i: int, y_slice: slice, x_slice: slice) -> np.ndarray:
            # The overlap STRIP only — never a whole tile. This is what keeps registration's
            # resident memory proportional to overlap area, not to the mosaic.
            return tiles[i][registration_channel][y_slice, x_slice]

        metrics = register_pairs_batched(
            pair_bounds,
            read_region,
            _DOWNSAMPLE_FACTORS,
            _SSIM_WINDOW,
            max_shift,
            max_workers,
        )

    with timer.stage("optimize"):
        edges = _edges_from_pairwise_metrics(metrics)
        if not edges:
            return np.zeros((n_tiles, 2), dtype=np.float64)
        # Anchor tile 0 at the origin, exactly as TileFusion.optimize_shifts does; the solve
        # is translation-only and otherwise gauge-free.
        return two_round_optimization(
            edges, n_tiles, [0], rel_thresh, abs_thresh, True
        )


def _mosaic_geometry(
    positions_yx_um: Sequence[tuple[float, float]],
    pixel_size: tuple[float, float],
    tile_shape: tuple[int, int],
) -> tuple[tuple[int, int], list[tuple[float, float]]]:
    """``((H, W), [(oy, ox), ...])`` — mosaic size and each tile's FRACTIONAL pixel origin.

    Ported from ``TileFusion._compute_fused_image_space`` + ``_tile_pixel_origins``, minus the
    chunk padding (that exists to align a Zarr write; there is no store here). Origins stay
    fractional so ``fuse_plane`` can honour the sub-pixel registration instead of truncating
    it — truncation to whole pixels is exactly the misalignment registration just removed.
    """
    pos = np.asarray(positions_yx_um, dtype=np.float64)
    py, px = pixel_size
    Y, X = tile_shape
    min_y, min_x = pos.min(axis=0)
    h = int(np.ceil((pos[:, 0].max() + Y * py - min_y) / py))
    w = int(np.ceil((pos[:, 1].max() + X * px - min_x) / px))
    origins = [((y - min_y) / py, (x - min_x) / px) for y, x in pos]
    return (h, w), origins


def stitch_region(
    reader: "SquidReader",
    region: str,
    fovs: Sequence[int],
    *,
    projector: str = "mip",
    register: bool = True,
    registration_channel=None,
    channels: Optional[Sequence[int]] = None,
    blend_px: int = _BLEND_PX,
    block_px: int = _BLOCK_PX,
    max_workers: Optional[int] = None,
    rel_thresh: float = _REL_THRESH,
    abs_thresh: float = _ABS_THRESH,
    geometry: Optional[dict] = None,
    timer=None,
) -> np.ndarray:
    """Z-reduce every FOV of one well, register them, and fuse one seamless mosaic.

    The region operator. Returns the SAME 5-D shape ``project_well`` returns — ``(T, C, 1, Y,
    X)``, native dtype — but Y/X are the whole well's mosaic rather than one FOV, so every
    downstream consumer (the viewer's ``_on_well``, the writer) needs no new case.

    Stages (all timed through *timer*, which is Julio's ``StageTimer``):

    ``project``
        ``project_well`` per FOV — the IMA-183 z-reduction, unchanged. Done once and held,
        because both registration and fusion consume the same planes and re-reading a 10-deep
        z-stack twice per FOV would double the I/O for nothing.
    ``register`` / ``optimize``
        :func:`solve_offsets_px`. Skipped entirely when ``register=False``.
    ``fuse``
        ``tilefusion.fusion.fuse_plane`` per timepoint, sub-pixel placement + Hann feather,
        block-wise at *block_px* so peak fusion memory is bounded by the block, not the mosaic.

    Parameters
    ----------
    register:
        ``False`` gives pure **coordinate placement** — identical code path, identical
        feather, positions straight from the stage. This is the honest control for judging
        whether registration actually helped; it is registered as the ``"coordinate"``
        region operator.
    channels:
        Channel indices to fuse (``None`` = all). A mosaic costs ``C x H x W x 2`` bytes, so
        a one-channel request is the difference between ~0.2 GB and ~0.9 GB on a 27-FOV 10x
        well. Registration always runs on *registration_channel*, whatever this selects.
    blend_px:
        Hann feather ramp width. Must fit inside the real overlap; see :data:`_BLEND_PX`.
    rel_thresh, abs_thresh:
        Blunder rejection, forwarded to :func:`solve_offsets_px` — see its docstring.
        Ignored when ``register=False`` (there is no pose graph to reject links from).
    geometry:
        Optional out-dict for **provenance** (the ``picked_z`` pattern from
        :func:`project_well`): filled with ``offsets_px`` (the solved per-tile correction,
        zeros when ``register=False``), ``origins_px`` (each FOV's fractional top-left in the
        mosaic) and ``shape``. Without it there is no way to say *where* a given FOV landed,
        which makes an A/B against coordinate placement impossible to crop to a common frame.

    Returns
    -------
    np.ndarray
        ``(T, C, 1, H, W)`` native dtype, where ``C == len(channels)``.

    Raises
    ------
    ValueError
        If *fovs* is empty, ``pixel_size_um`` is missing/invalid, or a channel selection is
        out of range.
    KeyError
        If a FOV has no stage position (see :func:`_positions_yx_um`).
    """
    # Refuse a plane-op BEFORE reading anything. This pipeline fuses with z=1 by construction
    # (`out` is allocated with z extent 1, write_block writes [t, :, 0, ...], fuse_plane gets
    # z_level=0), which is correct for a z-reducer whose project_well output is (T, C, 1, Y, X).
    # A plane-op's output is (T, C, Nz, Y, X), so the old `[:, channels, 0]` silently kept plane 0
    # and discarded the rest of the stack -- on exported science data, for bgsub/decon/flatfield,
    # i.e. three of the six registered projectors. On the 10x tissue set that is 9 of 10 planes
    # gone with nothing said. Per-plane fusion is the real fix and needs a z-outer streaming loop
    # (holding every z for 27 FOVs x 4 ch x 2084^2 uint16 is ~9.4 GB), so refuse until then.
    _guard_op = _resolve_projector(projector)
    if not _guard_op.consumes:
        raise NotImplementedError(
            f"operator {getattr(_guard_op, 'name', _guard_op)!r} is a plane-op (consumes=set()), "
            f"and stitching does not yet fuse per z-plane. Stitching it would keep only z-plane 0 "
            f"and silently discard the rest of the stack. Reduce z first (e.g. mip), or use a "
            f"z-reducing operator such as decon3d."
        )

    from tilefusion.fusion import fuse_plane
    from tilefusion.utils import make_1d_profile

    timer = timer or _NullTimer()
    fovs = list(fovs)
    if not fovs:
        raise ValueError(f"region {region!r}: no FOVs to stitch.")

    meta = reader.metadata
    all_channels = [c["name"] for c in meta["channels"]]
    if channels is None:
        channels = list(range(len(all_channels)))
    channels = [int(c) for c in channels]
    bad = [c for c in channels if not 0 <= c < len(all_channels)]
    if bad:
        raise ValueError(f"channel index/indices {bad} out of range for {len(all_channels)} channels")

    pixel_size = _pixel_size(meta)
    tile_shape = tuple(int(v) for v in meta["frame_shape"])
    positions = _positions_yx_um(meta, region, fovs)
    dtype = np.dtype(meta["dtype"])
    n_t = int(meta["n_t"])
    # IMA-210 turned _PROJECTORS values into Operator(fn, consumes) records, so the
    # registry no longer hands back a bare callable. Unpack it the same way
    # _engine.project_plate does; passing the Operator itself raises
    # 'Operator object is not callable' deep inside project_well.
    _op = _resolve_projector(projector)

    reg_c_global = _resolve_registration_channel(meta, registration_channel)
    # Index of the registration channel WITHIN the selected subset. If the operator selected
    # channels that exclude it, register on the first selected channel instead of indexing
    # out of bounds (a silent wrong-channel registration is worse than an explicit fallback).
    reg_c = channels.index(reg_c_global) if reg_c_global in channels else 0

    # (guard for plane-ops lives just after _resolve_projector, so nothing is read first)
    # This whole pipeline is z=1 BY CONSTRUCTION: `out` below is allocated with a z extent of 1,
    # write_block writes [_t, :, 0, ...], and fuse_plane is called with z_level=0. That is correct
    # for a z-REDUCER, whose project_well output is (T, C, 1, Y, X).
    #
    # It is NOT correct for a PLANE-OP. project_well's contract (see its docstring) is:
    #     consumes=frozenset({"z"})  ->  (T, C, 1,  Y, X)   z collapsed
    #     consumes=frozenset()       ->  (T, C, Nz, Y, X)   FULL DEPTH
    # so `[:, channels, 0]` on a plane-op silently kept z-plane 0 and discarded every other plane
    # — on exported science data. bgsub, decon and flatfield are all plane-ops, i.e. three of the
    # six registered projectors, and on the 10x tissue set that threw away 9 planes out of 10
    # with nothing said. Refuse instead: a loud refusal is recoverable, a silently truncated
    # export is not, and this project has six confirmed silent failures already.
    #
    # The real fix is per-plane fusion (IMA-277). It is not a one-liner because memory is the
    # reason this shortcut existed: holding every z for 27 FOVs x 4 channels x 2084^2 uint16 is
    # ~9.4 GB, so it needs a z-outer streaming loop, not a bigger allocation.

    with timer.stage("project"):
        # (n_tiles, T, C, Y, X) native dtype. Guarded above: only z-reducers reach here, so
        # project_well's output is (T, C, 1, Y, X) and index 0 is the whole reduced plane.
        tiles = np.empty((len(fovs), n_t, len(channels), *tile_shape), dtype=dtype)
        for i, fov in enumerate(fovs):
            tiles[i] = project_well(reader, region, fov, reduce=_op.fn,
                                    consumes=_op.consumes)[:, channels, 0]

    offsets = np.zeros((len(fovs), 2), dtype=np.float64)
    if register:
        offsets = solve_offsets_px(
            tiles[:, 0],  # geometry is solved once, on t=0
            positions,
            pixel_size,
            tile_shape,
            registration_channel=reg_c,
            max_workers=max_workers,
            rel_thresh=rel_thresh,
            abs_thresh=abs_thresh,
            timer=timer,
        )
        # Apply the solved correction in micrometres, exactly as TileFusion.run does:
        # position += offset_px * pixel_size.
        positions = [
            (y + float(o[0]) * pixel_size[0], x + float(o[1]) * pixel_size[1])
            for (y, x), o in zip(positions, offsets)
        ]

    (h, w), origins = _mosaic_geometry(positions, pixel_size, tile_shape)
    if geometry is not None:
        geometry.update(
            fovs=list(fovs), offsets_px=offsets, origins_px=origins, shape=(h, w),
            pixel_size_um=pixel_size[0], tile_shape=tile_shape,
        )

    with timer.stage("fuse"):
        y_profile = make_1d_profile(tile_shape[0], blend_px)
        x_profile = make_1d_profile(tile_shape[1], blend_px)
        out = np.zeros((n_t, len(channels), 1, h, w), dtype=dtype)

        def read_tile(idx: int, z_level: int, time_idx: int) -> np.ndarray:
            # float32 because the numba blend kernels accumulate in float32; converting the
            # ONE tile the block is currently consuming keeps this at ~C x tile bytes.
            return tiles[idx][time_idx].astype(np.float32, copy=False)

        for t in range(n_t):

            def write_block(y0, y1, x0, x1, arr, _t=t):
                out[_t, :, 0, y0:y1, x0:x1] = arr

            fuse_plane(
                read_tile=read_tile,
                write_block=write_block,
                origins=origins,
                padded_shape=(h, w),
                tile_shape=tile_shape,
                channels=len(channels),
                y_profile=y_profile,
                x_profile=x_profile,
                block_size=block_px,
                z_level=0,
                time_idx=t,
            )

    return out


def _coordinate_region(reader, region, fovs, **kwargs):
    """Coordinate placement: :func:`stitch_region` with registration disabled (the control)."""
    kwargs["register"] = False
    return stitch_region(reader, region, fovs, **kwargs)


# name -> region operator. The PARALLEL table to _engine._PROJECTORS: entries here take a
# whole well (reader, region, fovs) and return its fused (T, C, 1, Y, X), because inter-FOV
# work cannot be expressed as a z-reduction. Extended via add_region_operator.
RegionOperator = Callable[..., np.ndarray]
_REGION_OPERATORS: dict[str, RegionOperator] = {
    "stitch": stitch_region,
    "coordinate": _coordinate_region,
}


def add_region_operator(name: str, operator: RegionOperator) -> None:
    """Add a named region operator so it can be selected in :func:`stitch_plate`.

    The IMA-222 seam, mirroring :func:`squidmip.add_projector`: a future inter-FOV operation
    (flat-field-corrected stitch, distortion-corrected stitch, per-well segmentation) plugs
    in by name with **zero engine edits**.

    Parameters
    ----------
    name:
        Table key. Non-empty, and must not already exist — a silent clobber of an existing
        operator would be a quiet correctness bug.
    operator:
        ``operator(reader, region, fovs, **kwargs) -> (T, C, 1, Y, X)``.
    """
    if not name:
        raise ValueError("region operator name must be a non-empty string")
    if not callable(operator):
        raise ValueError(f"region operator for {name!r} is not callable: {operator!r}")
    if name in _REGION_OPERATORS:
        raise ValueError(
            f"region operator {name!r} is already defined; pick a distinct name "
            f"(defined: {available_region_operators()})."
        )
    _REGION_OPERATORS[name] = operator


def available_region_operators() -> list[str]:
    """Return the available region-operator names, sorted (``["coordinate", "stitch"]``)."""
    return sorted(_REGION_OPERATORS)


def _resolve_region_operator(name: str) -> RegionOperator:
    """Look up a region operator by name, failing loud (named) on an unknown key."""
    try:
        return _REGION_OPERATORS[name]
    except KeyError:
        raise KeyError(
            f"unknown region operator {name!r}; available: {available_region_operators()}. "
            "Add new ones with squidmip.add_region_operator(name, fn)."
        ) from None


def stitch_plate(
    reader: "SquidReader",
    *,
    n_fovs: Optional[int] = None,
    workers: int | None = 1,
    operator: str = "stitch",
    on_error=None,
    regions=None,
    **operator_kwargs,
) -> Iterator[tuple[str, int, np.ndarray]]:
    """Stitch every selected well of a plate, streaming one fused mosaic per well.

    The region-operator twin of :func:`squidmip.project_plate`, and deliberately the SAME
    contract: same keyword names, same bounded in-flight window, same
    ``(region, fov, (T, C, 1, Y, X))`` yield in completion order. A consumer written against
    ``project_plate`` — notably the viewer's ``_OperatorWorker._on_well`` — drives this
    unchanged.

    The one difference, and it is intrinsic: the task here is a **region**, not a FOV. A
    27-FOV well yields ONE array (its mosaic), reported against ``fovs[0]`` as the anchor
    index, where ``project_plate`` would have yielded 27. Consumers that composite per-FOV
    sub-boxes (IMA-187's mosaic path) must therefore treat a stitched well as single-tile —
    the fused mosaic IS the well.

    Parameters
    ----------
    n_fovs:
        FOVs per well to include. Defaults to ``None`` = **all** — the opposite of
        ``project_plate``'s default of 1, because stitching one FOV is a no-op. Passed
        straight to :func:`squidmip.select_fovs`.
    workers:
        Regions in flight. Defaults to **1**, not the CPU count: peak memory is
        ``workers x`` one fused mosaic, and a 27-FOV 10x well is ~0.9 GB at 4 channels
        (versus ~139 MB for one projected FOV, which is why ``project_plate`` can afford a
        wide window). Raise it only with the mosaic size in hand. Registration and fusion are
        internally parallel regardless, so ``workers=1`` still saturates the CPU.
    operator:
        A region-operator name (default ``"stitch"``; ``"coordinate"`` is the unregistered
        control). See :func:`add_region_operator`.
    regions:
        Optional subset of wells, in the given order (deduplicated) — the preview path.

        Two shapes, and the second is the reason this parameter is not just a list:

        * a **sequence** of region names — each contributes the FOVs ``n_fovs`` selected,
          i.e. the whole well;
        * a **mapping** ``{region: [fov, ...]}`` — each contributes exactly those FOVs, and
          ``n_fovs`` is ignored for it. This is how a caller expresses a FOV *subset within*
          a region (IMA-228's Minerva export of a marquee'd corner of a well). The result is
          still ONE fused mosaic per region — the crop of that region spanned by the given
          FOVs, because :func:`_mosaic_geometry` derives the canvas from the positions it is
          handed. It is NOT one mosaic per FOV; a region is a mosaic, never a FOV.

        Unknown region names are dropped in both shapes (a stale selection is not fatal); a
        region mapped to an empty FOV list contributes no task, like an empty well.
    on_error:
        ``on_error(region, fov, exc)``: opt-in per-well fault isolation. A well that raises is
        reported and SKIPPED instead of aborting the plate. ``None`` (default) is fail-fast.
    **operator_kwargs:
        Forwarded to the operator (``channels=``, ``blend_px=``, ``projector=``, ``timer=``…).

    Yields
    ------
    tuple[str, int, np.ndarray]
        ``(region, anchor_fov, image)``; ``image`` is ``(T, C, 1, H, W)`` native dtype.

    Raises
    ------
    ValueError
        If *workers* < 1, or ``select_fovs`` rejects *n_fovs*.
    KeyError
        If *operator* is not in the region-operator table.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    n_workers = workers if workers is not None else _default_workers()

    op = _resolve_region_operator(operator)

    # Warm the reader's lazy index/metadata single-threaded BEFORE fan-out, exactly as
    # project_plate does, so concurrent read() only touches immutable state.
    meta = reader.metadata
    wells = select_fovs(meta, n_fovs=n_fovs)
    if isinstance(regions, Mapping):
        # Explicit per-region FOV lists: the caller has already decided which FOVs of each
        # well to fuse, so n_fovs does not apply. Intersect with what the acquisition actually
        # has (order and duplicates as given by the caller, minus the ones that don't exist)
        # rather than trusting the request — a selection can outlive the acquisition it came
        # from. Each surviving region is still exactly one task, hence exactly one mosaic.
        available = meta["fovs_per_region"]
        wells = {}
        for region in dict.fromkeys(regions):
            if region not in available:
                continue
            have = set(available[region])
            wells[region] = [int(f) for f in dict.fromkeys(regions[region]) if int(f) in have]
    elif regions is not None:  # subset preview: keep only the requested wells, in their order
        keep = list(dict.fromkeys(regions))
        wells = {r: wells[r] for r in keep if r in wells}
    tasks: Iterator[tuple[str, list[int]]] = iter(
        [(region, list(fovs)) for region, fovs in wells.items() if fovs]
    )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        in_flight: dict = {}

        def _submit_next() -> bool:
            """Submit the next region, if any; False when the task stream is exhausted."""
            try:
                region, fovs = next(tasks)
            except StopIteration:
                return False
            future = pool.submit(op, reader, region, fovs, **operator_kwargs)
            in_flight[future] = (region, fovs[0])
            return True

        for _ in range(n_workers):  # prime the window
            if not _submit_next():
                break

        while in_flight:
            done, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                region, anchor_fov = in_flight.pop(future)
                _submit_next()  # slide the window first, so a SKIPPED well still refills it
                try:
                    image = future.result()
                except Exception as exc:
                    if on_error is None:
                        raise  # default: fail-fast (project_plate's contract, unchanged)
                    on_error(region, anchor_fov, exc)  # opt-in: record + SKIP, keep going
                    continue
                yield region, anchor_fov, image
