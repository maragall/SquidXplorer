"""Stitch region operator: register a well's FOVs against each other and fuse them into
one seamless mosaic. Wraps Julio's ``tilefusion`` pipeline on in-memory arrays;
registration runs on the raw middle z-plane, never on the z operator's output.
"""

from __future__ import annotations

import contextlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Sequence

import numpy as np

from squidxplorer.projection import cast_like
from squidxplorer._engine import (
    _NOT_A_WELL_FAULT,
    MissingOperatorDependency,
    Param,
    _default_workers,
    _resolve_operator,
    add_region_operator,
    operator_available,
)
from squidxplorer._logpane import get_logger
from squidxplorer._placement import PlacedArray, Placement
from squidxplorer._volume import allocate, release
from squidxplorer.projection import (
    LABELS,
    project_well,
    scope_wells,
)

_log = get_logger("stitch")

if TYPE_CHECKING:
    from squidxplorer.reader import SquidReader

# tilefusion defaults, copied from TileFusion.run().
_DOWNSAMPLE_FACTORS = (1, 1)
_SSIM_WINDOW = 15
_REL_THRESH = 0.5
_ABS_THRESH = 2.0
_MIN_OVERLAP_PX = 15
_BLEND_PX = 128    # feather fallback when nothing overlaps; the default is auto_blend_px
_REG_T = 0
_BLOCK_PX = 2048   # fusion scratchpad edge; bounds peak fusion memory regardless of mosaic size


class _NullTimer:
    """Stand-in for ``profiling.stages.StageTimer`` when the caller passes none."""

    @contextlib.contextmanager
    def stage(self, name: str):
        yield


def _positions_yx_um(
    metadata: dict, region: str, fovs: Sequence[int]
) -> list[tuple[float, float]]:
    """``[(y_um, x_um), ...]`` for *fovs*, in tilefusion's (y, x) order."""
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
    """Index of the channel registration runs on (one channel drives all placements)."""
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


def auto_blend_px(
    positions_yx_um: Sequence[tuple[float, float]],
    pixel_size: tuple[float, float],
    tile_shape: tuple[int, int],
) -> int:
    """Feather ramp width measured from the acquisition's real overlap (the GUI's "Auto")."""
    from tilefusion.registration import find_adjacent_pairs

    pairs = find_adjacent_pairs(list(positions_yx_um), pixel_size, tile_shape,
                                min_overlap=_MIN_OVERLAP_PX)
    # find_adjacent_pairs yields (i, j, dy, dx, overlap_y, overlap_x).
    seams = [min(p[4], p[5]) for p in pairs if min(p[4], p[5]) > 0]
    if not seams:
        return _BLEND_PX
    return max(int(np.median(seams)) * 2, 10)


_FF_MAX_TILES = 50   # maragall/stitcher's _FlatfieldWorker n_samples; parity keeps the shared .npy identical
_FF_SEED = 42        # the GUI's seed, so the same acquisition samples the same tiles twice


def estimate_region_flatfield(
    reader: "SquidReader",
    region: str,
    fovs: Sequence[int],
    *,
    channels: Optional[Sequence[int]] = None,
    z_level: Optional[int] = None,
    time_point: int = 0,
    use_darkfield: bool = False,
    max_tiles: int = _FF_MAX_TILES,
) -> dict:
    """``{channel_name: FlatfieldProfile}`` estimated from raw tiles, per channel."""
    from squidxplorer._flatfield import estimate_profile

    meta = reader.metadata
    all_channels = [c["name"] for c in meta["channels"]]
    if channels is None:
        channels = list(range(len(all_channels)))
    if z_level is None:
        # n//2, NOT _contrast.opening_z: this feeds the SHARED flatfield .npy, and parity with
        # maragall/stitcher (which samples nz//2) must keep the two products byte-identical.
        z_level = int(meta["n_z"]) // 2

    fovs = list(fovs)
    n = min(int(max_tiles), len(fovs))
    rng = np.random.default_rng(_FF_SEED)
    picked = [fovs[i] for i in sorted(rng.choice(len(fovs), size=n, replace=False))]

    _log.info("Flatfield: no profile in hand — estimating %d channel profile(s) from %d raw "
              "tile(s) of region %s at z=%d (tilefusion BaSiC). Stitching starts after this.",
              len(channels), n, region, z_level)
    profiles = {}
    for i, c in enumerate(channels, 1):
        name = all_channels[c]
        _log.info("Flatfield: channel %d of %d (%s) — reading %d raw tile(s)…",
                  i, len(channels), name, n)
        stack = np.stack([reader.read(region, f, name, z_level, time_point) for f in picked])
        t0 = time.perf_counter()
        profiles[name] = estimate_profile(stack, use_darkfield=use_darkfield)
        _log.info("Flatfield: channel %d of %d (%s) estimated in %.1f s.",
                  i, len(channels), name, time.perf_counter() - t0)
        del stack
    _log.info(
        "Flatfield: estimated %d channel profile(s) from %d raw tile(s) of region %s at z=%d.",
        len(profiles), n, region, z_level,
    )
    return profiles


def _flatfield_npy_path(reader):
    """Where maragall/stitcher keeps this acquisition's profile, or ``None`` if unknowable."""
    from pathlib import Path

    root = getattr(reader, "source_id", None) or getattr(reader, "_path", None)
    if root is None:
        return None
    root = Path(root)
    if not root.is_dir():
        return root.parent / f"{root.stem}_flatfield.npy"
    inside = root / f"{root.name}_flatfield.npy"
    beside = root.parent / f"{root.name}_flatfield.npy"
    return inside if inside.exists() or not beside.exists() else beside


def _selected_profiles(names: Sequence[str]) -> Optional[dict]:
    """The GUI-selected profiles as ``{channel_name: profile}``, only when they cover every channel."""
    from squidxplorer._flatfield import active_profiles

    installed = active_profiles()
    if not installed:
        return None
    names = [str(n) for n in names]
    picked = {n: installed[n] for n in names if n in installed}
    if len(picked) < len(names):
        _log.warning(
            "Flatfield: the profile(s) selected in the GUI cover %d of this acquisition's %d "
            "channel(s) (%s); %s have none. Using this acquisition's stored or estimated profile "
            "for EVERY channel instead, so one measurement corrects the whole run. Estimate or "
            "load the missing channel(s) in the flat-field tab to use the selection.",
            len(picked), len(names), ", ".join(sorted(picked)) or "none",
            ", ".join(n for n in names if n not in picked))
        return None
    return picked


def resolve_flatfield(reader, region: str, fovs: Sequence[int], *, channels=None,
                      z_level: Optional[int] = None, time_point: int = 0, use_darkfield: bool = False) -> dict:
    """Resolve one profile per channel: GUI selection > stored ``.npy`` > estimate-and-save."""
    from squidxplorer._flatfield import FlatfieldProfile

    meta = reader.metadata
    names = [c["name"] for c in meta["channels"]]
    path = _flatfield_npy_path(reader)

    selected = _selected_profiles(names)
    if selected is not None:
        _log.info("Flatfield: using the profile(s) selected in the GUI — one %dx%d field per "
                  "channel for all %d channel(s) (%s). Clear them to fall back to this "
                  "acquisition's stored profile.",
                  *next(iter(selected.values())).shape, len(names), ", ".join(names))
        return selected

    if path is not None and path.exists():
        _log.info("Flatfield: loading the stored profile from %s…", path.name)
        try:
            profiles = FlatfieldProfile.per_channel_from_npy(path, names)
            _log.info("Flatfield: loaded %d channel profile(s) from %s.", len(profiles), path.name)
            return profiles
        except Exception as exc:
            _log.warning("Flatfield: could not read %s (%s); estimating from tiles instead.",
                         path, exc)

    # Estimate ALL channels so the saved (C, Y, X) .npy is valid and reusable.
    profiles = estimate_region_flatfield(reader, region, fovs, channels=None, z_level=z_level,
                                         time_point=time_point, use_darkfield=use_darkfield)
    if path is not None and len(profiles) == len(names):
        try:
            from tilefusion.flatfield import save_flatfield

            ff = np.stack([profiles[n].flatfield for n in names])
            dfs = [profiles[n].darkfield for n in names]
            df = np.stack(dfs) if all(d is not None for d in dfs) else None
            save_flatfield(path, ff, df)
            _log.info("Flatfield: saved to %s; later runs and maragall/stitcher reuse it.", path)
        except Exception as exc:
            _log.warning("Flatfield: could not save to %s (%s); it will be re-estimated next run.",
                         path, exc)
    return profiles


class _FlatfieldReader:
    """A ``SquidReader`` whose ``read`` hands back illumination-corrected planes."""

    def __init__(self, inner, profiles: dict):
        self._inner = inner
        self._profiles = profiles

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def read(self, region, fov, channel, z_level, time_point=0):
        from squidxplorer._flatfield import correct_flatfield

        plane = self._inner.read(region, fov, channel, z_level, time_point)
        profile = self._profiles.get(str(channel))
        return plane if profile is None else correct_flatfield(plane, profile)


class _SeamSource:
    """The six members ``tilefusion.distortion`` reads off a ``TileFusion``."""

    def __init__(self, tiles, positions_yx_um, pixel_size, tile_shape, metrics,
                 registration_channel, max_workers):
        self._tiles = tiles
        self._pixel_size = np.asarray(pixel_size, float)
        self._tile_positions = np.asarray(positions_yx_um, float)
        self.Y, self.X = int(tile_shape[0]), int(tile_shape[1])
        self.pairwise_metrics = metrics
        self._c = registration_channel
        self.max_workers = max_workers or (os.cpu_count() or 8)

    def _read_tile_region(self, i: int, y_slice: slice, x_slice: slice) -> np.ndarray:
        return self._tiles[i][self._c][y_slice, x_slice]


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
    metrics_out: Optional[dict] = None,
    timer=None,
) -> np.ndarray:
    """Register the tiles against each other and return per-tile ``(dy, dx)`` shifts in pixels."""
    from tilefusion.optimization import _edges_from_pairwise_metrics, two_round_optimization
    from tilefusion.registration import (
        compute_pair_bounds,
        find_adjacent_pairs,
        register_pairs_batched,
        rotation_aware_max_shift,
    )

    # rel<=0 or abs<=0 makes the rejection test vacuously true for every link, so the second
    # round would return all-zero offsets that read as "the stage was already perfect".
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
            # nothing overlaps: fall back to stage positions
            return np.zeros((n_tiles, 2), dtype=np.float64)
        max_shift = rotation_aware_max_shift(adjacent_pairs)
        pair_bounds = compute_pair_bounds(adjacent_pairs, tile_shape)

        def read_region(i: int, y_slice: slice, x_slice: slice) -> np.ndarray:
            # the overlap strip only, never a whole tile
            return tiles[i][registration_channel][y_slice, x_slice]

        metrics = register_pairs_batched(
            pair_bounds,
            read_region,
            _DOWNSAMPLE_FACTORS,
            _SSIM_WINDOW,
            max_shift,
            max_workers,
        )

    if metrics_out is not None:
        metrics_out.update(metrics)

    with timer.stage("optimize"):
        edges = _edges_from_pairwise_metrics(metrics)
        if not edges:
            return np.zeros((n_tiles, 2), dtype=np.float64)
        # anchor tile 0 at the origin, as TileFusion.optimize_shifts does
        offsets = two_round_optimization(edges, n_tiles, [0], rel_thresh, abs_thresh, True)
        return _place_unconstrained_tiles(offsets, edges, metrics, positions_yx_um, pixel_size)


# Minimum registered pairs for the affine fallback (TileFusion's own _MIN_PAIRS_FOR_AFFINE).
_MIN_PAIRS_FOR_AFFINE = 8


def _place_unconstrained_tiles(
    offsets: np.ndarray,
    edges,
    metrics: dict,
    positions_yx_um: Sequence[tuple[float, float]],
    pixel_size: tuple[float, float],
) -> np.ndarray:
    """Place tiles the pose graph left unconstrained, via the global stage->image affine."""
    from tilefusion.optimization import _check_connectivity, fit_stage_to_image_transform

    n_tiles = len(positions_yx_um)
    components = _check_connectivity(edges, n_tiles)
    anchor_component = next((c for c in components if 0 in c), [])
    unconstrained = [t for t in range(n_tiles) if t not in anchor_component]
    if not unconstrained:
        return offsets  # fully connected: the solve already placed everything

    if len(metrics) < _MIN_PAIRS_FOR_AFFINE:
        _log.warning(
            "%d tile(s) unconstrained but only %d registered pair(s) (< %d); leaving them at "
            "stage positions rather than fitting an unreliable affine.",
            len(unconstrained), len(metrics), _MIN_PAIRS_FOR_AFFINE,
        )
        return offsets

    cal = fit_stage_to_image_transform(
        metrics, [tuple(p) for p in positions_yx_um], pixel_size
    )
    M = cal["M"]
    pos = np.asarray(positions_yx_um, dtype=np.float64)
    ps = np.asarray(pixel_size, dtype=np.float64)
    ref = pos[0]
    offsets = np.array(offsets, dtype=np.float64, copy=True)
    for k in unconstrained:
        d = pos[k] - ref
        # Stored as an offset to the isotropic model, consistent with the solved entries.
        offsets[k] = M @ d - d / ps
    _log.warning(
        "Affine calibration: placed %d unconstrained tile(s) %s (scale %.2f px/unit, "
        "rotation %+.3f deg, fit residual %.1f px over %d pairs).",
        len(unconstrained), unconstrained, cal["scale"], cal["rotation_deg"],
        cal["residual_rms"], cal["n_pairs"],
    )
    return offsets


def _mosaic_geometry(
    positions_yx_um: Sequence[tuple[float, float]],
    pixel_size: tuple[float, float],
    tile_shape: tuple[int, int],
) -> tuple[tuple[int, int], list[tuple[float, float]]]:
    """``((H, W), [(oy, ox), ...])`` — mosaic size and each tile's fractional pixel origin."""
    pos = np.asarray(positions_yx_um, dtype=np.float64)
    py, px = pixel_size
    Y, X = int(tile_shape[0]), int(tile_shape[1])
    min_y, min_x = pos.min(axis=0)
    # The tile size is added in PIXELS, never round-tripped through micrometres: the µm
    # round-trip loses one ULP against stage coordinates of order 1e5 and grows the canvas.
    h = int(np.ceil((pos[:, 0].max() - min_y) / py)) + Y
    w = int(np.ceil((pos[:, 1].max() - min_x) / px)) + X
    origins = [((y - min_y) / py, (x - min_x) / px) for y, x in pos]
    return (h, w), origins


def stitch_region(
    reader: "SquidReader",
    region: str,
    fovs: Sequence[int],
    *,
    z_operator: str = "mip",
    register: bool = True,
    registration_channel=None,
    channels: Optional[Sequence[int]] = None,
    blend_px: Optional[int] = None,
    correct_distortion: Optional[bool] = None,
    correct_illumination: Optional[bool] = None,
    flatfield: Optional[dict] = None,
    use_darkfield: bool = False,
    registration_t: int = _REG_T,
    registration_z: Optional[int] = None,
    block_px: int = _BLOCK_PX,
    max_workers: Optional[int] = None,
    rel_thresh: float = _REL_THRESH,
    abs_thresh: float = _ABS_THRESH,
    geometry: Optional[dict] = None,
    timer=None,
) -> np.ndarray:
    """Apply an operator to every FOV of one well, register them, and fuse a seamless mosaic."""
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

    registration_t = int(registration_t)
    if not 0 <= registration_t < n_t:
        raise ValueError(
            f"registration_t={registration_t} is outside this acquisition's {n_t} timepoint(s)")

    # blend_px=None is maragall/stitcher's "Auto": measure the overlap instead of guessing.
    if blend_px is None:
        blend_px = auto_blend_px(positions, pixel_size, tile_shape)
    _op = _resolve_operator(z_operator)

    reg_c_global = _resolve_registration_channel(meta, registration_channel)

    if registration_z is None:
        # n//2, NOT _contrast.opening_z: registration parity with maragall/stitcher is pinned
        # by tests/test_integration.py's geometry match on real z-stacks.
        registration_z = int(meta["n_z"]) // 2
    registration_z = int(registration_z)
    if not 0 <= registration_z < int(meta["n_z"]):
        raise ValueError(
            f"registration_z={registration_z} is outside this acquisition's "
            f"{meta['n_z']} z-plane(s)")

    # z_sources[k] is the acquisition plane producing output plane k (None = let project_well reduce).
    z_sources: list = [None] if "z" in _op.consumes else list(meta["z_levels"])

    # Fusion blends by weighted average, which is meaningless for integer object ids.
    if _op.produces == LABELS:
        raise ValueError(
            f"operator {z_operator!r} produces label images (integer object ids), and fusion blends "
            "overlapping tiles by a weighted average — the mean of two label ids is a third, "
            "nonexistent object, and per-FOV ids collide across every seam. Stitching labels needs "
            "id reconciliation across seams, which this operator does not do. Segment per FOV "
            f"instead (write_plate(operator={z_operator!r})), or stitch an intensity operator."
        )

    # Exactly one of the read path and the operator may flat-field per pass (not idempotent).
    if correct_illumination is not False and getattr(_op.fn, "corrects_illumination", False):
        raise ValueError(
            f"z_operator {z_operator!r} flat-field corrects its input, and stitching's read path is "
            "ALSO correcting (correct_illumination is on by default). The correction is not "
            "idempotent — applying it twice changes ~89% of pixels by up to 23 counts, silently. "
            "Pick ONE: correct_illumination=False to let the operator do it, or a z operator that "
            "does not correct (e.g. 'mip') and let the read path do it, which is where TileFusion "
            "applies it and the only place registration can benefit from it."
        )

    # Wrap the reader BEFORE anything is read, so registration and fusion both see corrected pixels.
    if correct_illumination is None:
        correct_illumination = True
    if correct_illumination:
        if flatfield is None:
            with timer.stage("flatfield"):
                flatfield = resolve_flatfield(
                    reader, region, fovs, channels=sorted(set(channels) | {reg_c_global}),
                    z_level=registration_z, time_point=registration_t,
                    use_darkfield=use_darkfield,
                )
        reader = _FlatfieldReader(reader, flatfield)

    # The raw planes the geometry is solved on: one z, one channel, straight from the reader.
    reg_planes = None
    if register:
        with timer.stage("read_reg"):
            reg_planes = np.empty((len(fovs), 1, *tile_shape), dtype=dtype)
            for i, fov in enumerate(fovs):
                reg_planes[i, 0] = reader.read(
                    region=region, fov=fov, channel=all_channels[reg_c_global],
                    z_level=registration_z, time_point=registration_t,
                )

    offsets = np.zeros((len(fovs), 2), dtype=np.float64)
    metrics: dict = {}
    if register:
        offsets = solve_offsets_px(
            reg_planes,
            positions,
            pixel_size,
            tile_shape,
            registration_channel=0,
            max_workers=max_workers,
            rel_thresh=rel_thresh,
            abs_thresh=abs_thresh,
            metrics_out=metrics,
            timer=timer,
        )
        if correct_illumination and len(fovs) > 1:
            from tilefusion.registration import find_adjacent_pairs

            expected = len(find_adjacent_pairs(
                [(y, x) for y, x in positions], pixel_size, tile_shape,
                min_overlap=_MIN_OVERLAP_PX))
            if len(metrics) < expected:
                _log.warning(
                    "Flatfield is ON and %d of %d adjacent pair(s) in region %s did not "
                    "register. The correction can push a marginal pair under threshold "
                    "(measured: it costs one pair per region on the 10x tissue set). Re-run "
                    "with correct_illumination=False to see whether it is implicated.",
                    expected - len(metrics), expected, region,
                )

        # Apply the solved correction in micrometres: position += offset_px * pixel_size.
        positions = [
            (y + float(o[0]) * pixel_size[0], x + float(o[1]) * pixel_size[1])
            for (y, x), o in zip(positions, offsets)
        ]

    (h, w), origins = _mosaic_geometry(positions, pixel_size, tile_shape)

    placement = Placement(
        origin_um=(min(y for y, _ in positions), min(x for _, x in positions)),
        pixel_size_um=pixel_size[0],
        z_step_um=meta.get("dz_um"),
        shape=(h, w),
        tile_shape=tile_shape,
        fovs=tuple(fovs),
        offsets_px=tuple((float(o[0]), float(o[1])) for o in offsets),
        origins_px=tuple((float(y), float(x)) for y, x in origins),
        reg_channel=all_channels[reg_c_global] if register else None,
        reg_t=registration_t if register else None,
        reg_z=registration_z if register else None,
        illumination_corrected=bool(correct_illumination),
    )

    # Legacy out-dict, kept as a view of the placement for existing callers.
    if geometry is not None:
        geometry.update(
            fovs=list(fovs), offsets_px=offsets, origins_px=origins, shape=(h, w),
            pixel_size_um=pixel_size[0], tile_shape=tile_shape, placement=placement,
        )

    # None means ON wherever it can run, i.e. wherever a pose graph exists to fit.
    if correct_distortion is None:
        correct_distortion = bool(register)
    get_field = None
    if correct_distortion:
        if not register:
            raise ValueError(
                "correct_distortion needs registration: the per-seam elastic fit corrects the "
                "residual left AFTER the global solve, and with register=False there is no "
                "solve and no seam correspondence to fit. Enable registration, or turn "
                "distortion correction off.")
        from tilefusion.distortion import TileWarper, build_seam_corrections

        with timer.stage("distortion"):
            # Fit on the RAW registration planes: the same pixels the global solve saw.
            source = _SeamSource(reg_planes, positions, pixel_size, tile_shape,
                                 metrics, 0, max_workers)
            corrections = build_seam_corrections(source)
            # An unfittable seam gets None from .field() = identity, degrading to plain fusion.
            get_field = TileWarper(corrections, tile_shape[0], tile_shape[1]).field
            # `source` holds reg_planes; both names must be cleared to free them.
            source = None

    # Drop the registration planes before allocating the projected stack.
    reg_planes = None

    y_profile = make_1d_profile(tile_shape[0], blend_px)
    x_profile = make_1d_profile(tile_shape[1], blend_px)
    # Spilled to a scratch file above 2 GB (squidxplorer._volume); still a plain ndarray downstream.
    out = allocate((n_t, len(channels), len(z_sources), h, w), dtype,
                   what=f"region {region!r}'s fused {len(z_sources)}-plane mosaic")

    # z-outer streaming loop: the geometry was solved once above, so every plane lands on the
    # same grid; one acquisition plane's tiles are resident at a time.
    tiles = None
    for z_i, z_src in enumerate(z_sources):
        with timer.stage("project"):
            tiles = None
            tiles = np.empty((len(fovs), n_t, len(channels), *tile_shape), dtype=dtype)
            for i, fov in enumerate(fovs):
                tiles[i] = project_well(reader, region, fov, reduce=_op.fn,
                                        consumes=_op.consumes, z_level=z_src)[:, channels, 0]

        with timer.stage("fuse"):
            def read_tile(idx: int, z_level: int, time_idx: int, _tiles=tiles) -> np.ndarray:
                # _tiles holds exactly this iteration's plane, so fuse_plane's z_level is ignored.
                return _tiles[idx][time_idx].astype(np.float32, copy=False)

            for t in range(n_t):

                def write_block(y0, y1, x0, x1, arr, _t=t, _z=z_i):
                    # Round back to the acquisition dtype, never truncate.
                    out[_t, :, _z, y0:y1, x0:x1] = cast_like(arr, out.dtype)

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
                    get_field=get_field,
                )

        tiles = None          # this plane's tiles go before the next plane's are read
        release(out)          # ...and its written pages leave the resident set (spilled case)
        if len(z_sources) > 1:
            _log.info("Fusion: region %s plane %d of %d fused (same solved offsets as plane 0).",
                      region, z_i + 1, len(z_sources))

    return PlacedArray(out, placement)


RegionOperator = Callable[..., np.ndarray]

#: The scalar knobs `stitch` DECLARES, so `--param`, recipes and generated panels describe it
#: like every other operator. Defaults restate `stitch_region`'s own, with None spelled
#: concretely where None means a fixed value (registration_channel None = index 0,
#: correct_illumination None = on). Knobs whose None default is measured from the data
#: (blend_px, registration_z, correct_distortion) or that cannot change the pixels (block_px,
#: max_workers) stay keyword arguments. rel_thresh/abs_thresh stay kwargs too, undeclared for
#: a measured reason: tilefusion's two_round_optimization clamps rel_thresh <= 1.0 to its own
#: factor 3.0 and floors the cutoff at _BLUNDER_FLOOR_PX (150 px), so neither knob can change
#: the solve until a link is >150 px wrong — a declaration the build-failing probe test
#: (a declared parameter must be able to change the pixels) could never vouch for.
_STITCH_PARAMS = (
    Param("z_operator", "mip",
          "what each FOV's z-stack becomes before fusion; a z-reducer collapses it to one "
          "plane, a plane-op keeps every plane"),
    Param("register", True,
          "solve per-tile offsets from the overlaps; off = stage-coordinate placement"),
    Param("registration_channel", 0,
          "the channel the pose graph is solved on, by index or name; every channel is then "
          "fused with that one solution"),
    Param("registration_t", _REG_T, "the timepoint the pose graph is solved on"),
    Param("correct_illumination", True,
          "flat-field the tiles on the read path, before registration and fusion"),
)


def _stitch_factory(**params):
    """The registered object: called with the declared parameters, returns the region operator."""
    def stitch(reader, region, fovs, **kwargs):
        return stitch_region(reader, region, fovs, **{**params, **kwargs})
    return stitch


#: The keyword arguments `stitch_region` takes BEYOND the declared params — the record's
#: explicit passthrough, so the loop refuses an unknown key by name instead of splatting it.
#: (The comment above _STITCH_PARAMS records why these are not Params.)
_STITCH_ACCEPTS = ("channels", "blend_px", "correct_distortion", "flatfield", "use_darkfield",
                   "registration_z", "block_px", "max_workers", "rel_thresh", "abs_thresh",
                   "geometry", "timer")


def _coordinate_factory(**params):
    """Coordinate placement: :func:`stitch_region` with registration disabled (the control)."""
    def coordinate(reader, region, fovs, **kwargs):
        return stitch_region(reader, region, fovs, register=False, **{**params, **kwargs})
    return coordinate


#: `coordinate` declares the stitch knobs that survive with registration off; the registration
#: family (register itself, registration_channel/_t/_z) is neither declared nor accepted, so
#: passing one is a NAMED refusal instead of a silently ignored number.
_COORDINATE_PARAMS = tuple(p for p in _STITCH_PARAMS
                           if p.name in ("z_operator", "correct_illumination"))
_COORDINATE_ACCEPTS = tuple(a for a in _STITCH_ACCEPTS if a != "registration_z")

add_region_operator("stitch", _stitch_factory, params=_STITCH_PARAMS, requires=("tilefusion",),
                    extra="stitch", accepts=_STITCH_ACCEPTS, inner_param="z_operator")
add_region_operator("coordinate", _coordinate_factory, params=_COORDINATE_PARAMS,
                    requires=("tilefusion",), extra="stitch",
                    accepts=_COORDINATE_ACCEPTS, inner_param="z_operator")


def _accepts_kwarg(fn, name: str) -> bool:
    """Whether *fn* can be called with keyword *name*, directly or through ``**kwargs``."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):   # builtins / C callables have no introspectable signature
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return name in params


def _resolve_region_operator(name: str, operator_kwargs: Optional[dict] = None) -> RegionOperator:
    """The callable a region-operator name resolves to, refusing anything this loop cannot run."""
    from squidxplorer._engine import _resolve_operator, available_region_operators

    operator = _resolve_operator(name)
    if "fov" not in operator.consumes:
        raise KeyError(
            f"{name!r} is a registered operator but not a REGION operator: it declares "
            f"consumes={sorted(operator.consumes)} and takes planes, while the region loop hands its "
            f"operator (reader, region, fovs). Region operators: "
            f"{available_region_operators()}; run {name!r} with squidxplorer.run_plate."
        )
    return operator.with_params(operator_kwargs)


def _stitch_plate(
    reader: "SquidReader",
    *,
    n_fovs: Optional[int] = None,
    workers: int | None = 1,
    operator: str = "stitch",
    on_error=None,
    regions=None,
    **operator_kwargs,
) -> Iterator[tuple[str, int, np.ndarray]]:
    """Stitch every selected well of a plate, streaming ONE fused mosaic per well."""
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    n_workers = workers if workers is not None else _default_workers()

    # ONE validator for both engine arms: declared parameters bind through the factory, keys
    # the record `accepts` pass through to the callable, anything else is refused BY NAME.
    from squidxplorer._engine import split_operator_kwargs

    bound, operator_kwargs = split_operator_kwargs(operator, operator_kwargs)
    op = _resolve_region_operator(operator, bound)
    ok, why = operator_available(operator)
    if not ok:
        raise MissingOperatorDependency(why)

    # Warm the reader's lazy index/metadata single-threaded BEFORE fan-out.
    meta = reader.metadata
    wells = scope_wells(meta, n_fovs, regions)
    tasks: Iterator[tuple[str, list[int]]] = iter(
        [(region, list(fovs)) for region, fovs in wells.items() if fovs]
    )

    # ONE illumination profile for the whole plate, resolved before any well starts, so
    # every well is corrected by the same gain field.
    # correct_illumination is DECLARED, so it may sit in `bound` rather than the loose kwargs;
    # either spelling of "off" must skip the plate-wide estimate.
    wants_illumination = ({**bound, **operator_kwargs}.get("correct_illumination", True)
                          is not False)
    if _accepts_kwarg(op, "flatfield") and wants_illumination \
            and operator_kwargs.get("flatfield") is None and wells:
        spread = [(r, f) for r, fs in wells.items() for f in fs]
        rng = np.random.default_rng(_FF_SEED)
        picked = [spread[i] for i in
                  sorted(rng.choice(len(spread), size=min(_FF_MAX_TILES, len(spread)),
                                    replace=False))]
        try:
            from squidxplorer._flatfield import FlatfieldProfile, estimate_profile

            z = int(meta["n_z"]) // 2       # parity with maragall/stitcher's shared .npy
            names = [c["name"] for c in meta["channels"]]
            path = _flatfield_npy_path(reader)
            selected = _selected_profiles(names)
            if selected is not None:
                operator_kwargs["flatfield"] = selected
                _log.info("Flatfield: using the profile selected in the GUI for the whole plate — "
                          "%d channel(s), one %dx%d gain field each.",
                          len(selected), *next(iter(selected.values())).shape)
            elif path is not None and path.exists():
                _log.info("Flatfield: loading the stored plate-wide profile from %s…", path.name)
                operator_kwargs["flatfield"] = FlatfieldProfile.per_channel_from_npy(path, names)
                _log.info("Flatfield: loaded %d plate-wide profile(s) from %s.",
                          len(names), path.name)
            else:
                _log.info("Flatfield: no stored profile — estimating one plate-wide profile per "
                          "channel from %d tile(s) across %d well(s) at z=%d (tilefusion BaSiC). "
                          "Stitching starts after this.", len(picked), len(wells), z)
                profiles = {}
                for i, name in enumerate(names, 1):
                    _log.info("Flatfield: channel %d of %d (%s) — reading %d raw tile(s)…",
                              i, len(names), name, len(picked))
                    stack = np.stack([reader.read(r, f, name, z, 0) for r, f in picked])
                    t0 = time.perf_counter()
                    profiles[name] = estimate_profile(stack)
                    _log.info("Flatfield: channel %d of %d (%s) estimated in %.1f s.",
                              i, len(names), name, time.perf_counter() - t0)
                    del stack
                operator_kwargs["flatfield"] = profiles
                _log.info("Flatfield: one plate-wide profile per channel from %d tile(s) across "
                          "%d well(s), at z=%d.", len(picked), len(wells), z)
                if path is not None:
                    try:
                        from tilefusion.flatfield import save_flatfield

                        save_flatfield(path, np.stack([profiles[n].flatfield for n in names]), None)
                        _log.info("Flatfield: saved to %s.", path)
                    except Exception as save_exc:
                        _log.warning("Flatfield: could not save to %s (%s); it will be "
                                     "re-estimated next run.", path, save_exc)
        except Exception as exc:
            _log.warning(
                "Flatfield: plate-wide estimate failed (%s); each well will estimate from its "
                "own tiles instead, so wells are not photometrically comparable.", exc)

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
                except _NOT_A_WELL_FAULT:
                    raise  # a missing package is not a corrupt well
                except Exception as exc:
                    if on_error is None:
                        raise  # default: fail-fast (the per-FOV loop's contract)
                    on_error(region, anchor_fov, exc)
                    continue
                yield region, anchor_fov, image
