"""Stitch operator (IMA-222): the first REAL stitcher wired into the plate.

This module adds a **region operator** — an operation whose unit of work is a whole well
(all of its FOVs at once) rather than one FOV's z-stack — and ships one: ``stitch``, which
registers a well's FOVs against each other and fuses them into a single seamless mosaic.

One table, and a declaration that says which loop runs it
---------------------------------------------------------
A stitcher's unit of work is not a plane. It needs every FOV of the well simultaneously **plus
each FOV's x/y stage geometry**, so its callable has the shape the work actually has::

    RegionOperator = Callable[[SquidReader, str, list[int]], np.ndarray]   # -> (T, C, Nz, Y, X)
                                 reader,  region,  fovs

IMA-222 expressed that as a PARALLEL table (``_REGION_OPERATORS``, plus a ``_REGION_REQUIRES``
sidecar shadowing it). It is one table again as of 2026-08-05: ``squidmip.add_region_operator``
files into ``_engine._OPERATORS`` with ``consumes=REGION_OP`` (``{"fov"}``), and
:func:`stitch_plate` selects on that declaration. Nothing about the callable changed — what
changed is that "which loop runs this" is now a declaration on the one record instead of which of
two dicts the name happens to be in. See ``_engine._OPERATORS`` for the three costs the split was
charging.

and a :func:`stitch_plate` generator that **mirrors ``project_plate``'s exact contract** —
same keyword names (``n_fovs``/``workers``/``on_error``/``regions``), same bounded in-flight
window, same ``(region, fov, (T, C, Nz, Y, X))`` yield — so the viewer's
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

    registration.find_adjacent_pairs        which FOVs actually overlap, from stage geometry
    registration.rotation_aware_max_shift   residual-rejection cap, adaptive to tile spacing
    registration.compute_pair_bounds        the overlap strip of each pair, per tile
    registration.register_pairs_batched     phase correlation (upsample 20) + NCC score
    optimization._edges_from_pairwise_metrics    pairwise metrics -> weighted pose-graph edges
    optimization.two_round_optimization     global least-squares solve + blunder rejection
    optimization.fit_stage_to_image_transform    stage->image affine, for tiles the solve
                                            left unconstrained
    projection.project_well          per-FOV z-reduction (MIP), the IMA-183 primitive
    utils.make_1d_profile                   the Hann feather ramp
    fusion.fuse_plane                       sub-pixel placement + feathered blend, block-wise

REGISTRATION READS THE RAW PLANE, NOT THE PROJECTOR OUTPUT
----------------------------------------------------------
The order above is deliberate and was WRONG until it was measured. ``project_well`` sits below
the registration calls because the geometry is solved on the acquisition's own z-plane -- the
middle one, ``n_z // 2``, which is ``TileFusion._middle_z`` -- and the projector's output is
only ever fused, never registered.

Registering on the MIP is not a harmless substitution. Measured on the 10x tissue set
(``test_10x_laser_af_z_stack``, region ``manual0``, 27 FOVs), against ``TileFusion.run()``'s
own solve as the reference:

    registration input      pairs registered    offsets vs reference
    raw middle z-plane      42 / 43             0.00 px   (identical)
    MIP over 10 planes      40 / 43             6.62 px max, 2.00 px RMS

The MIP flattens ten planes of a thick sample into one, and the out-of-focus content it drags
forward decorrelates the overlap strips: two more pairs fail outright and FOV 4 is left with no
registered edge at all, disconnected from the pose graph. So this is not "a different but equally
valid choice of registration image" -- it registers strictly worse AND it disagrees with the
standalone. Solving on the raw plane reproduces ``TileFusion`` exactly.

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
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Sequence

import numpy as np

from squidmip._background import _cast_like
from squidmip._engine import (
    _NOT_A_WELL_FAULT,
    MissingOperatorDependency,
    _default_workers,
    _resolve_operator,
    add_region_operator,
    operator_available,
)
from squidmip._logpane import get_logger
from squidmip._placement import PlacedArray, Placement
from squidmip._volume import allocate, release
from squidmip.projection import (
    LABELS,
    missing_requirements,
    normalise_requires,
    project_well,
    requirement_refusal,
    select_fovs,
)

_log = get_logger("stitch")

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
_BLEND_PX = 128                # feather ramp FALLBACK, used only when nothing overlaps and so
#                                there is no seam to measure (see auto_blend_px). It is no
#                                longer the default: stitch_region defaults to blend_px=None =
#                                Auto, which is maragall/stitcher's GUI default and the only one
#                                that adapts to the acquisition. As a fixed default this number
#                                was wrong even on the set it was tuned for -- 128 px against a
#                                measured 208 px seam, where Auto gives 416.
_REG_T = 0                     # geometry is solved at ONE timepoint. Named so the solve site
#                                and the Placement that reports it cannot drift apart: a
#                                Placement claiming reg_t=0 while the solve moved to another
#                                timepoint would be provenance that lies.
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


def auto_blend_px(
    positions_yx_um: Sequence[tuple[float, float]],
    pixel_size: tuple[float, float],
    tile_shape: tuple[int, int],
) -> int:
    """Feather ramp width MEASURED from the acquisition's real overlap.

    maragall/stitcher's "Auto" checkbox (app.py:1454), same formula: take each adjacent pair's
    SMALLER overlap (the true seam depth -- a pair that overlaps 200 px vertically and 2000 px
    horizontally has a 200 px seam), the median across pairs, doubled, floored at 10.

    This is the control that removes a guess rather than adding a knob. The fixed default is
    sized to ONE acquisition: 128 px against the 10x tissue set's measured ~208 px overlap. A
    ramp wider than the overlap never reaches full weight and dims the seam, so on a denser
    grid the default is actively wrong -- and the user has no way to know without measuring the
    overlap themselves, which is exactly what this does for them.

    Falls back to the module default when nothing overlaps: a sparse/freeform acquisition
    legitimately has isolated FOVs (the same case :func:`solve_offsets_px` degrades to stage
    placement for), and a median over an empty set is not an answer.
    """
    from tilefusion.registration import find_adjacent_pairs

    pairs = find_adjacent_pairs(list(positions_yx_um), pixel_size, tile_shape,
                                min_overlap=_MIN_OVERLAP_PX)
    # find_adjacent_pairs yields (i, j, dy, dx, overlap_y, overlap_x).
    seams = [min(p[4], p[5]) for p in pairs if min(p[4], p[5]) > 0]
    if not seams:
        return _BLEND_PX
    return max(int(np.median(seams)) * 2, 10)


_FF_MAX_TILES = 50   # maragall/stitcher's `_FlatfieldWorker` n_samples (app.py:1122-1171). PARITY
#                      is the reason for this number, not a belief that 50 is optimal: the two
#                      tools read AND WRITE the same `<root>_flatfield.npy` (see
#                      _flatfield_npy_path), so a different cap means whichever ran last silently
#                      owns a profile the other would not have produced.
#
#                      It was briefly 49, from a note reading "limit to n=49 (Ian-stitcher)". Two
#                      things came out of checking that (Julio ruled 50, 2026-08-04):
#
#                        * Ian's number is neither 49 nor 249. ~/CEPHLA/projects/ian-stitcher,
#                          image_stitcher/flatfield_correction.py:11-12, is MAX_FLATFIELD_IMAGES=48
#                          plus MAX_FLATFIELD_IMAGES_PER_T=32 — a per-TIMEPOINT cap we have no
#                          equivalent of. His sample is `random.shuffle` UNSEEDED, so his estimate
#                          is not reproducible run to run; ours is (_FF_SEED).
#                        * The cap only bites plate-wide. Measured, 10x tissue set, 405 nm: per
#                          REGION (27 FOVs) it never binds and 49 vs 50 are BIT-IDENTICAL;
#                          plate-wide (55 tiles) they differ by max|d| 7.8e-3 / RMS 1.7e-3, which
#                          against a field only ~4% deep is ~0.78% of mean gain -- a fifth of the
#                          correction's own magnitude, so not negligible.
#
#                      Revisit if the sample ever needs STRATIFYING across wells: on a 1536-well
#                      plate, 50 tiles out of tens of thousands may not represent the illumination
#                      across the whole plate. stitch_plate already spreads its sample over wells;
#                      estimate_region_flatfield does not.
_FF_SEED = 42        # the GUI's seed, so the same acquisition samples the same tiles twice.


def estimate_region_flatfield(
    reader: "SquidReader",
    region: str,
    fovs: Sequence[int],
    *,
    channels: Optional[Sequence[int]] = None,
    z: Optional[int] = None,
    t: int = 0,
    use_darkfield: bool = False,
    max_tiles: int = _FF_MAX_TILES,
) -> dict:
    """``{channel_name: FlatfieldProfile}`` estimated from RAW tiles — the standalone's recipe.

    maragall/stitcher's ``_FlatfieldWorker`` (app.py:1122-1171), step for step: sample up to
    ``max_tiles`` FOVs with a fixed seed, read ONE raw plane each at the registration z, and run
    ``estimate_flatfield_channel`` per channel. squidmip already wraps that exact estimator as
    :func:`squidmip.estimate_profile`, so the algorithm is Julio's BaSiC port either way.

    RAW is load-bearing and the standalone says so out loud ("flatfield estimation must be
    performed on raw, uncorrected tiles"): estimating from already-corrected pixels measures the
    residual of a correction rather than the illumination, and converges to a unit field.

    Per channel, not per FOV: illumination is a property of the optical path, so every FOV of a
    channel shares one gain field. Memory is bounded by ONE channel's stack (BaSiC downsamples to
    128x128 internally, but the float32 input stack is real — ~0.5 GB for 27 2084^2 tiles), which
    is why the channels are walked in sequence rather than stacked together.
    """
    from squidmip._flatfield import estimate_profile

    meta = reader.metadata
    all_channels = [c["name"] for c in meta["channels"]]
    if channels is None:
        channels = list(range(len(all_channels)))
    if z is None:
        z = int(meta["n_z"]) // 2

    fovs = list(fovs)
    n = min(int(max_tiles), len(fovs))
    rng = np.random.default_rng(_FF_SEED)
    picked = [fovs[i] for i in sorted(rng.choice(len(fovs), size=n, replace=False))]

    # SAY IT BEFORE DOING IT. Every line this function used to emit was in the past tense, so the
    # stage that runs BEFORE the first well is stitched -- and reads up to _FF_MAX_TILES tiles per
    # channel to do it -- was silent for its whole duration and then announced itself as finished.
    # The plate bar cannot cover it either: its unit is the REGION (squidmip._progress.unit_plan),
    # and this runs before any region does, so the only surface that can show it live is the log
    # panel the bar now sits in. One line per channel, at the start and at the end, is the same
    # shape tilefusion's own stage prints have.
    _log.info("Flatfield: no profile in hand — estimating %d channel profile(s) from %d raw "
              "tile(s) of region %s at z=%d (tilefusion BaSiC). Stitching starts after this.",
              len(channels), n, region, z)
    profiles = {}
    for i, c in enumerate(channels, 1):
        name = all_channels[c]
        _log.info("Flatfield: channel %d of %d (%s) — reading %d raw tile(s)…",
                  i, len(channels), name, n)
        stack = np.stack([reader.read(region, f, name, z, t) for f in picked])
        t0 = time.perf_counter()
        profiles[name] = estimate_profile(stack, use_darkfield=use_darkfield)
        _log.info("Flatfield: channel %d of %d (%s) estimated in %.1f s.",
                  i, len(channels), name, time.perf_counter() - t0)
        del stack
    _log.info(
        "Flatfield: estimated %d channel profile(s) from %d raw tile(s) of region %s at z=%d.",
        len(profiles), n, region, z,
    )
    return profiles


def _flatfield_npy_path(reader):
    """Where maragall/stitcher keeps this acquisition's profile, or ``None`` if unknowable.

    Its convention, verbatim (``app.py:1716-1722``), for a directory acquisition — which is what
    Squid writes: ``<root>/<root.name>_flatfield.npy`` first, then
    ``<root.parent>/<root.name>_flatfield.npy`` beside it. Auto-save uses the FIRST of those
    (``app.py:1911``), so the two tools converge on one file.

    Returns the inside path when neither exists, because that is where a new one is written.
    ``None`` when the reader has no path at all (the test fakes), which disables both the lookup
    and the save rather than guessing at a location.
    """
    from pathlib import Path

    root = getattr(reader, "_path", None)
    if root is None:
        return None
    root = Path(root)
    if not root.is_dir():
        return root.parent / f"{root.stem}_flatfield.npy"
    inside = root / f"{root.name}_flatfield.npy"
    beside = root.parent / f"{root.name}_flatfield.npy"
    return inside if inside.exists() or not beside.exists() else beside


def _selected_profiles(names: Sequence[str]) -> Optional[dict]:
    """The profiles the USER selected in the GUI, as ``{channel_name: profile}``, or ``None``.

    THE ONE OWNER OF "the profile the user chose". ``squidmip._flatfield``'s module global
    (``set_profile``/``active_profile``) was read by exactly one consumer — the registered
    ``flatfield`` plane-op — so a profile loaded or estimated in the GUI's flat-field tab had
    **zero effect on stitching**: :func:`resolve_flatfield` went straight to the ``.npy`` lookup
    and, finding nothing, estimated its own. Two unsynchronised answers to "which gain field is
    this plate corrected by", and nothing able to notice they disagreed.

    Fixed HERE rather than by having the GUI push into ``flatfield=``, because the tab that owns
    the chooser is not the tab that starts a stitch: making the GUI set a per-run argument would
    leave the global still owning the plane-op, i.e. two owners again. Reading the global from the
    one place stitching resolves a profile leaves exactly one, and the precedence is total:

        explicit ``flatfield=`` argument  >  GUI-selected profile  >  stored ``.npy``  >  estimate

    The argument stays on top because a caller who names a profile has said something more specific
    than a standing GUI selection (it is also how :func:`stitch_plate` hands ONE plate-wide profile
    to every region). The GUI selection beats the file on disk because the user chose it after the
    file was already there.

    PER CHANNEL, AND ALL-OR-NOTHING (2026-08-06). The global used to be a SINGLE
    ``FlatfieldProfile`` and this function broadcast it over every name — so loading a plate's own
    ``(4, 2084, 2084)`` ``.npy`` through the GUI made stitching correct 488, 561 and 638 with the
    405 gain field, while loading the SAME FILE through ``resolve_flatfield``'s own lookup gave
    each channel its own (measured: identical for 405, max|d| 0.3335 / 0.0237 / 0.1411 for the
    other three). One file, two mosaics, depending on whether the user had clicked "Load
    illumination profile". The global is now ``{channel: profile}`` and this is a lookup.

    THE PARTIAL-COVERAGE RULE, one rule for every caller: the GUI selection is used only when it
    covers EVERY channel of the run. Partial coverage (the auto-estimate installs the ONE channel
    it estimated) returns ``None`` and is logged by name, so the run falls through to this
    acquisition's stored ``.npy`` or its own estimate — for all channels, from one provenance.
    The alternative, mixing a GUI field for one channel with file fields for the others, makes a
    mosaic whose channels were corrected by two different measurements with nothing in the
    artifact to say which; and it would have to be decided the same way at BOTH callers
    (:func:`resolve_flatfield` per region and :func:`stitch_plate` plate-wide) or the two disagree,
    which is the shape of the defect being removed. Silence is what is forbidden here, not
    falling through: the log names exactly which channels were covered and which were not.
    """
    from squidmip._flatfield import active_profiles

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
                      z: Optional[int] = None, t: int = 0, use_darkfield: bool = False) -> dict:
    """Load this acquisition's profile, or estimate it and save it. The standalone's lifecycle.

    ``app.py:1724-1732`` on drop: look for the ``.npy``; found -> load it and tick the box; not
    found -> tick the box, auto-calculate, and ``app.py:1903-1916`` auto-saves the result next to
    the data. So flat-fielding is on by default there, and it is compute-ONCE: the second run of
    the same acquisition reads the file the first run wrote, and so does the standalone.

    Ahead of all of that sit the profiles the user selected in the GUI, when they cover every
    channel of this acquisition — see :func:`_selected_profiles` for the precedence, for the
    partial-coverage rule, and for why the global is read here.

    Either way the answer is ONE PROFILE PER CHANNEL. It has to be: the stored ``.npy`` is
    ``(C, Y, X)`` and its four fields are genuinely different (0.645–1.102 for 488 against
    0.974–1.020 for 405 on the 10x set), so a map that carried one field for every channel
    disagreed with the very file it was loaded from.

    Saving is best-effort, exactly as the standalone's is (it wraps its own save in try/except and
    logs the failure): an acquisition on a read-only share must still stitch. Failing to persist
    costs a re-estimate next run; failing to stitch costs the run.
    """
    from squidmip._flatfield import FlatfieldProfile

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
            # A profile we cannot read is no reason to skip the correction, and no reason to
            # trust a partial one either. Name the file, then estimate.
            _log.warning("Flatfield: could not read %s (%s); estimating from tiles instead.",
                         path, exc)

    # ALL channels, never the caller's subset. Three reasons, and the first is the one that bit:
    # a subset profile cannot be SAVED (the .npy is (C, Y, X) and a partial one would claim to
    # describe channels it never measured), so narrowing here silently disabled the persistence
    # and every run re-estimated. It is also what the standalone does -- its worker loops
    # `range(n_channels)` -- and it makes the artifact reusable by a later run that asks for
    # different channels.
    profiles = estimate_region_flatfield(reader, region, fovs, channels=None, z=z, t=t,
                                         use_darkfield=use_darkfield)
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
    """A ``SquidReader`` whose ``read`` hands back illumination-corrected planes.

    The correction lives in the READ PATH, which is the whole point and the reason this is a
    wrapper rather than a post-processing step on the projected tiles.

    ``TileFusion`` applies flatfield inside ``_read_tile`` and ``_read_tile_region`` — so it
    reaches registration and fusion alike, and it never meets a z-reduction because the standalone
    fuses every z independently. squidmip DOES z-reduce, so correcting the projector's OUTPUT
    instead would raise a question the standalone never has to answer: whether the correction
    commutes with the reducer. For a monotone per-pixel map it does (``_flatfield.py`` makes that
    argument for the MIP), but ``decon3d`` is not monotone and the answer would quietly become
    "no" for it.

    Correcting at the read means the question never arises: every projector, monotone or not, sees
    exactly the pixels ``TileFusion`` would have registered and fused. The commutation property is
    then a nice fact about the MIP rather than a correctness dependency.

    Unknown channels pass through uncorrected rather than raising: a caller may legitimately read
    a channel no profile was estimated for, and a silent identity is the same thing
    ``TileWarper.field`` does for an unfittable seam.
    """

    def __init__(self, inner, profiles: dict):
        self._inner = inner
        self._profiles = profiles

    def __getattr__(self, name):
        # Only fires for attributes this wrapper does not define, so `metadata` (often a lazy
        # property) is delegated rather than snapshotted at construction.
        return getattr(self._inner, name)

    def read(self, region, fov, channel, z, t=0):
        from squidmip._flatfield import correct_flatfield

        plane = self._inner.read(region, fov, channel, z, t)
        profile = self._profiles.get(str(channel))
        return plane if profile is None else correct_flatfield(plane, profile)


class _SeamSource:
    """The six members ``tilefusion.distortion`` reads off a ``TileFusion``.

    squidmip runs tilefusion's pieces on IN-MEMORY arrays (see the module docstring), so there
    is no ``TileFusion`` instance to hand ``build_seam_corrections``. It does not need one:
    across ``distortion.py`` the entire surface it touches is ``_pixel_size``,
    ``_tile_positions``, ``Y``, ``X``, ``pairwise_metrics`` and ``_read_tile_region`` -- so
    this adapter IS the orchestration, and the elastic fit itself stays Julio's code.

    Positions are the REGISTERED ones. build_seam_corrections' own docstring is explicit that
    it corrects the residual left AFTER the global solve; fitting it on raw stage positions
    would re-measure the error registration had just removed.

    *tiles* is the same ``(n_tiles, 1, Y, X)`` raw registration-plane stack the global solve
    consumed, so ``registration_channel`` is 0 here. Both fits see the same pixels by
    construction.
    """

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
        # The overlap STRIP only, exactly like solve_offsets_px's own reader: the elastic fit
        # block-registers along a seam and never wants a whole tile.
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

    # The pairwise metrics are what tilefusion's distortion fit keys off (it reads
    # tf.pairwise_metrics to know which seams exist). They were computed and dropped; handing
    # them back through an out-dict follows the `geometry` provenance pattern already used
    # here, and avoids re-running phase correlation just to re-derive the pair list.
    if metrics_out is not None:
        metrics_out.update(metrics)

    with timer.stage("optimize"):
        edges = _edges_from_pairwise_metrics(metrics)
        if not edges:
            return np.zeros((n_tiles, 2), dtype=np.float64)
        # Anchor tile 0 at the origin, exactly as TileFusion.optimize_shifts does; the solve
        # is translation-only and otherwise gauge-free.
        offsets = two_round_optimization(edges, n_tiles, [0], rel_thresh, abs_thresh, True)
        return _place_unconstrained_tiles(offsets, edges, metrics, positions_yx_um, pixel_size)


# The affine fallback's minimum evidence: a 2-3 DOF global transform needs a handful of spread
# pairs. TileFusion._place_unconstrained_tiles_with_affine's own _MIN_PAIRS_FOR_AFFINE.
_MIN_PAIRS_FOR_AFFINE = 8


def _place_unconstrained_tiles(
    offsets: np.ndarray,
    edges,
    metrics: dict,
    positions_yx_um: Sequence[tuple[float, float]],
    pixel_size: tuple[float, float],
) -> np.ndarray:
    """Place tiles the pose graph left unconstrained, via the global stage->image affine.

    A port of ``TileFusion._place_unconstrained_tiles_with_affine``, which this module used to
    omit. The omission was silent and load-bearing: ``two_round_optimization`` does NOT place a
    tile that registered against nothing -- its own docstring says such a tile "is left for the
    caller's affine/stage-model fallback" and it merely logs a warning -- so every tile whose
    overlaps were too low-texture to register kept a ZERO offset and was fused at its raw,
    miscalibrated stage position. Zero is also what a perfectly-placed tile gets, so nothing
    downstream could tell the two apart.

    The affine is a property of the instrument (stage axes vs sensor axes: scale, rotation,
    shear), fit from the pairs that DID register, so it predicts where an unregistered tile
    belongs far better than the stage does. No-op when the graph is fully connected to the
    anchor, which is the normal case -- on the 10x tissue set's ``manual0`` all 27 FOVs connect,
    which is exactly why this gap survived review.

    Degrades rather than guesses: with fewer than :data:`_MIN_PAIRS_FOR_AFFINE` registered pairs
    the fit is not trustworthy, so the tiles are left at stage positions (what this function
    replaced) and the caller is told.
    """
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
    """``((H, W), [(oy, ox), ...])`` — mosaic size and each tile's FRACTIONAL pixel origin.

    Ported from ``TileFusion._compute_fused_image_space`` + ``_tile_pixel_origins``, minus the
    chunk padding (that exists to align a Zarr write; there is no store here). Origins stay
    fractional so ``fuse_plane`` can honour the sub-pixel registration instead of truncating
    it — truncation to whole pixels is exactly the misalignment registration just removed.

    THE TILE SIZE IS ADDED IN PIXELS, NEVER ROUND-TRIPPED THROUGH MICROMETRES. The canvas must
    cover ``max(origin) + tile``, and because *tile* is already an integer number of pixels that
    is ``ceil(span_px) + tile`` exactly. Written the other way — ``ceil((max + Y*py - min) / py)``
    — the tile's height goes to µm and back around a stage coordinate of order 1e5 µm, and the
    cancellation costs enough precision to land one ULP above the integer. MEASURED on the real
    10x acquisition (stage x from 96813.688 µm, py 0.752, 2084 px tiles): every one of its 55
    single-FOV stitches came back 2085 px on an axis that is exactly 2084 px of data, and
    ``manual1`` fovs [0, 1] came back (2085, 3960) against a true (2084, 3960). The extra row is
    all zeros — a black seam line along the bottom/right of the stitched mosaic that the raw
    preview of the same region does not have, and a mosaic one pixel taller than the box it is
    placed in (``_op_result`` places it by ``mosaic_bbox_um``, which is derived in whole pixels).
    """
    pos = np.asarray(positions_yx_um, dtype=np.float64)
    py, px = pixel_size
    Y, X = int(tile_shape[0]), int(tile_shape[1])
    min_y, min_x = pos.min(axis=0)
    h = int(np.ceil((pos[:, 0].max() - min_y) / py)) + Y
    w = int(np.ceil((pos[:, 1].max() - min_x) / px)) + X
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
    """Apply an operator to every FOV of one well, register them, and fuse a seamless mosaic.

    The region operator. Returns the SAME 5-D shape ``project_well`` returns — ``(T, C, Nz, Y,
    X)``, native dtype — but Y/X are the whole well's mosaic rather than one FOV, so every
    downstream consumer (the viewer's ``_on_well``, the writer) needs no new case. ``Nz``
    follows the operator's own ``consumes`` declaration, exactly as ``project_well``'s does:

    ====================  ===================  ==============================================
    ``consumes``          output ``Nz``        operators
    ====================  ===================  ==============================================
    ``frozenset({"z"})``  1 (z collapsed)      mip, reference, decon3d
    ``frozenset()``       ``n_z`` (per plane)  bgsub, decon, flatfield, spot, cellpose
    ====================  ===================  ==============================================

    PER-PLANE FUSION (IMA-277)
    --------------------------
    A plane-op used to be REFUSED here, because this pipeline fused with z pinned to 1 and
    ``[:, channels, 0]`` would have kept plane 0 and silently discarded the other nine on a
    10-plane acquisition. It now fuses every plane, and the design is the reason it can:

    **The geometry is solved ONCE.** Registration is geometry — it runs on ONE raw plane
    (``registration_z``) at ONE timepoint (``registration_t``), and the ``offsets`` it produces
    are applied to ``positions`` BEFORE the z loop starts. Every plane is then fused from the
    same ``origins`` and the same ``get_field`` distortion warp. That is not an optimisation,
    it is the correctness requirement: two planes solved independently would disagree by their
    respective residuals and the stack would shear with depth. Pixel-identical placement in all
    planes is what "we fuse all 10 z independently" has to mean.

    **The z loop is OUTER and streaming.** For each output plane the tiles are projected
    (``project_well(..., z=)`` reads exactly that one acquisition plane), fused for every
    timepoint, written into ``out``, and dropped before the next plane is read. So the input
    side stays at ONE plane's worth of tiles — ~0.94 GB for 27 FOVs x 4 channels x 2084^2
    uint16, exactly what it was when only one plane existed — rather than the ~9.4 GB the whole
    stack would cost.

    **The output side is spilled, not swapped.** A 10-plane 4-channel mosaic of that well is
    8.79 GB, on a 16 GB machine that is also holding the tiles. :func:`squidmip._volume.allocate`
    backs it with a scratch file above 2 GB and :func:`squidmip._volume.release` drops the
    resident pages after each plane, so the result is still a plain ndarray for every consumer
    while the resident set stays flat in stack depth. A z-reducer's single plane is under the
    threshold, so mip/reference/decon3d allocate exactly as they always have.

    Stages (all timed through *timer*, which is Julio's ``StageTimer``):

    ``read_reg``
        One RAW plane per FOV — ``registration_z``, one channel — read straight from the
        reader. This is what the geometry is solved on, exactly as ``TileFusion`` does; see the
        module docstring for the measurement that says why it is not the projector's output.
    ``project``
        ``project_well`` per FOV — the IMA-183 z-reduction, unchanged. Its output is FUSED,
        never registered.
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
        well. It cannot affect the geometry: registration reads *registration_channel* out of
        the reader itself, so the solved offsets are identical for every channel selection of
        the same region BY CONSTRUCTION, not by an invariant someone has to maintain.
    registration_z:
        RAW z-plane the geometry is solved on. ``None`` (default) = the middle plane,
        ``n_z // 2``, which is ``TileFusion._middle_z``. This indexes the ACQUISITION's z, not
        the projector's output — the projector is not run for registration at all.
    blend_px:
        Hann feather ramp width. ``None`` (default) = **Auto**: measured from this
        acquisition's own overlap by :func:`auto_blend_px`, which is both maragall/stitcher's
        GUI default and what its "Auto" checkbox computes. An int overrides it. The old fixed
        default of :data:`_BLEND_PX` was sized to ONE acquisition and is 128 px against a
        measured 208 px seam here, where Auto gives 416 — so it under-feathered every seam of
        the very dataset it was tuned on.
    correct_illumination:
        Retrospective flat-field correction, applied **in the read path** so registration and
        fusion both see corrected pixels — which is where ``TileFusion`` applies it
        (``_read_tile_region`` feeds the registration strips).

        **On by default** (``None`` = on), which is maragall/stitcher's behaviour, not an
        extension of it: dropping a dataset in looks for a stored profile and, finding none,
        ticks the box and auto-calculates from the tiles (``app.py:1724-1732``).

        Worth knowing before trusting it to help: on the 10x tissue set the correction COSTS a
        registered pair in both regions (manual0 42->41, manual1 41->40), because the vignetting
        there is only ~4.5% deep and dividing it out amplifies noise in the dim corners where the
        marginal pairs live. Mean NCC rises, but a pair falls under threshold and its tile lands
        in the affine fallback. Any pair lost this way is named in a WARNING.
    flatfield:
        ``{channel_name: FlatfieldProfile}`` to use instead of resolving one. ``None`` (the
        default) runs :func:`resolve_flatfield`, whose precedence is: the profile the user
        selected in the GUI (:func:`_selected_profiles`), else the acquisition's stored
        ``<root>_flatfield.npy`` if present, else estimate from raw tiles and save it there.

        :func:`stitch_plate` resolves ONCE and passes the same profile to every region, which is
        the standalone's semantics — its flat-field worker builds a ``TileFusion`` with no region
        filter and hands one ``(C, Y, X)`` array to every region. That matters beyond tidiness:
        illumination is a property of the optical path, not of a well, so a per-region profile
        would divide each well by a different gain and break photometric comparability across the
        plate.
    use_darkfield:
        Also estimate the additive pedestal. Off by default, matching the standalone's own
        default: for fluorescence the pedestal is ~0 and the estimate is the less stable half.
    correct_distortion:
        Per-seam elastic lens-distortion correction, fitted on the REGISTERED positions and
        applied during fusion. **On by default** (``None`` = on wherever it can run, i.e.
        wherever ``register`` is on); ``False`` turns it off, and an explicit ``True`` with
        ``register=False`` raises rather than quietly doing nothing. See the note at the
        ``if correct_distortion is None`` line for why the default is not a plain ``True``.
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
        ``(T, C, Nz, H, W)`` native dtype, where ``C == len(channels)`` and ``Nz`` is 1 for a
        z-reducer and the acquisition's ``n_z`` for a plane-op.

    Raises
    ------
    ValueError
        If *fovs* is empty, ``pixel_size_um`` is missing/invalid, a channel selection is out of
        range, or *projector* would apply the flat-field correction a SECOND time on top of the
        read path's (see the ``correct_illumination`` guard below).
    KeyError
        If a FOV has no stage position (see :func:`_positions_yx_um`).
    """
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

    # The registration TIMEPOINT is now the caller's (Defect 1). It used to be the module
    # constant _REG_T = 0, so on a multi-timepoint acquisition the geometry silently solved at
    # t=0 and the user had no way to see or set it. That is the same defect CLASS as the
    # registration-channel substitution bug fixed above -- a solve running somewhere the user
    # cannot name -- so it is refused rather than clamped: clamping would put a number in the
    # Placement that did not solve anything.
    registration_t = int(registration_t)
    if not 0 <= registration_t < n_t:
        raise ValueError(
            f"registration_t={registration_t} is outside this acquisition's {n_t} timepoint(s)")

    # blend_px=None is maragall/stitcher's "Auto": measure the overlap instead of guessing.
    if blend_px is None:
        blend_px = auto_blend_px(positions, pixel_size, tile_shape)
    # IMA-210 turned _OPERATORS values into Operator(fn, consumes) records, so the
    # registry no longer hands back a bare callable. Unpack it the same way
    # _engine.project_plate does; passing the Operator itself raises
    # 'Operator object is not callable' deep inside project_well.
    _op = _resolve_operator(projector)

    reg_c_global = _resolve_registration_channel(meta, registration_channel)

    # The registration Z-PLANE, the same way registration_t is the caller's. `None` = the middle
    # plane, which is TileFusion._middle_z -- the standalone's own choice, and the one the
    # measurement in the module docstring reproduces to 0.00 px.
    #
    # This indexes the ACQUISITION's z. It used to index nothing at all: registration consumed
    # `tiles`, i.e. the PROJECTOR's output, so on this 10-plane acquisition the pose graph was
    # solved on a MIP. That is the defect this function was carrying -- not a missing knob but a
    # solve running on an image the standalone never registers, which cost 2 of 43 pairs and up
    # to 6.62 px of placement.
    if registration_z is None:
        registration_z = int(meta["n_z"]) // 2
    registration_z = int(registration_z)
    if not 0 <= registration_z < int(meta["n_z"]):
        raise ValueError(
            f"registration_z={registration_z} is outside this acquisition's "
            f"{meta['n_z']} z-plane(s)")

    # THE OUTPUT'S Z EXTENT, from the operator's own declaration and nothing else — the same
    # `consumes` table project_well dispatches on, so the two cannot drift:
    #     consumes={"z"}  -> the operator reduces z          -> Nz = 1,   z_sources = [None]
    #     consumes=set()  -> the operator maps over planes   -> Nz = n_z, z_sources = z_levels
    # `z_sources[k]` is the ACQUISITION plane that produces output plane k (None = "all of them",
    # i.e. let project_well reduce). This list IS the z-outer loop below.
    z_sources: list = [None] if "z" in _op.consumes else list(meta["z_levels"])

    # LABELS CANNOT BE FEATHERED. `fuse_plane` blends overlapping tiles by a Hann-weighted convex
    # combination, which is the right answer for intensity and meaningless for an operator whose
    # pixels are integer OBJECT IDS (`produces="labels"`: spot, cellpose): the mean of label 12 and
    # label 37 is label 24, an object that does not exist. Per-FOV ids also collide between tiles,
    # so even a nearest-tile rule would merge unrelated objects across every seam.
    #
    # That is a real inter-FOV problem (id reconciliation across seams), not a fusion parameter, so
    # it is refused rather than approximated. These operators ARE end-to-end on the per-FOV path
    # (`write_plate(projector="cellpose")`), which is where a label image is meaningful.
    if _op.produces == LABELS:
        raise ValueError(
            f"operator {projector!r} produces label images (integer object ids), and fusion blends "
            "overlapping tiles by a weighted average — the mean of two label ids is a third, "
            "nonexistent object, and per-FOV ids collide across every seam. Stitching labels needs "
            "id reconciliation across seams, which this operator does not do. Segment per FOV "
            f"instead (write_plate(projector={projector!r})), or stitch an intensity operator."
        )

    # THE FLAT-FIELD DOUBLE-APPLY GUARD.
    #
    # Two places can flat-field, and exactly one of them may run per pass:
    #   * the READ path (`correct_illumination`, ON by default) — `_FlatfieldReader` below, which
    #     is where TileFusion applies it and where it must be applied for REGISTRATION to see
    #     corrected strips;
    #   * the OPERATOR (`projector="flatfield"`, or anything built by `_flatfield.flatfield_op`),
    #     which corrects the pixels project_well emits.
    #
    # They could not meet until now: this function refused every plane-op, and `flatfield` is a
    # plane-op. Per-plane fusion removes that refusal, so the combination became reachable in the
    # same commit — and the correction is NOT idempotent. Measured on the 10x tissue set, region
    # manual0, correcting twice changes 88.6% of pixels by up to 23 counts. Nothing downstream
    # could tell: the mosaic renders, the store validates, the numbers are just wrong.
    #
    # So it is refused by DECLARATION, not by name — `_flatfield.CORRECTS_ILLUMINATION` is an
    # attribute on the operator callable, in the same style as `consumes`, because this package
    # hands out flat-field operators under names it does not choose (`flatfield_op(profile)`) and
    # a `== "flatfield"` test would miss every one of them. Both single-correction spellings stay
    # available and the message names them.
    if correct_illumination is not False and getattr(_op.fn, "corrects_illumination", False):
        raise ValueError(
            f"projector {projector!r} flat-field corrects its input, and stitching's read path is "
            "ALSO correcting (correct_illumination is on by default). The correction is not "
            "idempotent — applying it twice changes ~89% of pixels by up to 23 counts, silently. "
            "Pick ONE: correct_illumination=False to let the operator do it, or a projector that "
            "does not correct (e.g. 'mip') and let the read path do it, which is where TileFusion "
            "applies it and the only place registration can benefit from it."
        )

    # ILLUMINATION CORRECTION, wrapped around the reader BEFORE anything is read, so every read
    # below this line -- the registration planes AND project_well's z-stack -- is corrected. That
    # ordering is the whole feature: TileFusion corrects inside _read_tile_region, so its phase
    # correlation runs on corrected strips, and vignetting is a fixed multiplicative pattern
    # present on both tiles of a seam for the correlator to lock onto.
    #
    # Estimated from RAW tiles (`reader`, not the wrapper), which is what the standalone's worker
    # is emphatic about: estimating from corrected pixels measures a correction's residual and
    # converges to a unit field.
    if correct_illumination is None:
        correct_illumination = True
    if correct_illumination:
        if flatfield is None:
            with timer.stage("flatfield"):
                flatfield = resolve_flatfield(
                    reader, region, fovs, channels=sorted(set(channels) | {reg_c_global}),
                    z=registration_z, t=registration_t, use_darkfield=use_darkfield,
                )
        reader = _FlatfieldReader(reader, flatfield)

    # The RAW planes the geometry is solved on: one z, one channel, straight from the reader.
    # Shape (n_tiles, 1, Y, X) so solve_offsets_px's `tiles[i][c]` indexes it with c=0.
    #
    # This is the read TileFusion._read_tile_region does, and it is why the channel selection can
    # no longer reach the solve: `channels` is a fusion concern now, and there is no longer an
    # appended-registration-channel dance to keep the two apart. It costs one extra plane per FOV
    # (2084^2 uint16 = 8.7 MB here, freed before fusion), which is the price of registering on
    # what the standalone registers on.
    reg_planes = None
    if register:
        with timer.stage("read_reg"):
            reg_planes = np.empty((len(fovs), 1, *tile_shape), dtype=dtype)
            for i, fov in enumerate(fovs):
                reg_planes[i, 0] = reader.read(
                    region=region, fov=fov, channel=all_channels[reg_c_global],
                    z=registration_z, t=registration_t,
                )

    offsets = np.zeros((len(fovs), 2), dtype=np.float64)
    metrics: dict = {}
    if register:
        offsets = solve_offsets_px(
            reg_planes,                 # raw plane, ONE timepoint and ONE z: geometry is geometry
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
        # Say what the correction cost, when it costs anything. Flat-fielding is ON by default
        # and on the 10x tissue set it drops a marginal pair per region -- measured, not feared.
        # This CANNOT name the specific pairs without solving twice, and a second full solve is
        # not worth paying for on every run, so it reports the two numbers it does know and says
        # how to get the comparison. Silence here is what would let the trade go unnoticed.
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

        # Apply the solved correction in micrometres, exactly as TileFusion.run does:
        # position += offset_px * pixel_size.
        positions = [
            (y + float(o[0]) * pixel_size[0], x + float(o[1]) * pixel_size[1])
            for (y, x), o in zip(positions, offsets)
        ]

    (h, w), origins = _mosaic_geometry(positions, pixel_size, tile_shape)

    # The placement is built UNCONDITIONALLY (Defect 3). It used to exist only if the caller
    # remembered to pass a `geometry` out-dict, so by default the solved transform was computed
    # once at t=0 and then thrown away -- while the viewer separately re-derived placement from
    # a stage bounding box. Two answers to "where are these pixels", and nothing able to notice
    # when they disagreed. It now rides back attached to the pixels themselves, and records
    # WHICH channel and timepoint solved it.
    placement = Placement(
        origin_um=(min(y for y, _ in positions), min(x for _, x in positions)),
        pixel_size_um=pixel_size[0],
        z_step_um=meta.get("dz_um"),
        shape=(h, w),
        tile_shape=tile_shape,
        fovs=tuple(fovs),
        offsets_px=tuple((float(o[0]), float(o[1])) for o in offsets),
        origins_px=tuple((float(y), float(x)) for y, x in origins),
        # The NAME of the channel actually solved on, not the index the caller passed -- the
        # two are only the same when the selection happens to contain it, which is precisely
        # the assumption that made the old registration bug invisible.
        reg_channel=all_channels[reg_c_global] if register else None,
        reg_t=registration_t if register else None,
        reg_z=registration_z if register else None,
        illumination_corrected=bool(correct_illumination),
    )

    # `geometry` is the legacy out-dict, kept so existing callers (tools/stitch_demo.py) keep
    # working. It is now a VIEW of the placement rather than a second computation -- one source
    # of truth, two spellings, and the dict spelling can be deleted once its callers move.
    if geometry is not None:
        geometry.update(
            fovs=list(fovs), offsets_px=offsets, origins_px=origins, shape=(h, w),
            pixel_size_um=pixel_size[0], tile_shape=tile_shape, placement=placement,
        )

    # PER-SEAM ELASTIC LENS DISTORTION (Defect 1). Julio asked for this control by name.
    #
    # Worth knowing: in maragall/stitcher the checkbox is DEAD -- app.py:1472 builds
    # `distortion_checkbox` and nothing ever reads it, so FusionWorker's enable_distortion
    # default (True) always wins and unchecking it does nothing. So this is not a port of a
    # working control; it is the first place the control actually decides anything, which is
    # why both directions are pinned by tests.
    #
    # Needs the pose graph, so it is registration-only: with register=False there are no
    # pairwise metrics, and a seam fit without a global solve would be measuring the stage
    # error rather than the lens.
    #
    # ON BY DEFAULT (Julio, 2026-08-03: "Correct lens distort should be defaulted to on"). The
    # port carried the standalone's opt-in spelling across while the standalone itself runs the
    # stage unconditionally, so squidmip was the only one of the two NOT correcting distortion
    # unless asked.
    #
    # The default is `None`, not `True`, and that is not hedging. It means ON WHEREVER IT CAN
    # RUN, which is exactly where a pose graph exists to fit the residual of. A plain `True`
    # would make every existing `register=False` caller -- the `coordinate` control operator
    # (see _coordinate_region), Minerva's fusion, the A/B benchmarks -- raise the guard below on
    # a combination none of them asked for. An EXPLICIT True with register=False still raises:
    # that is a user asking for something impossible, and it stays loud.
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
            # The RAW registration planes, not the projected ones: build_seam_corrections
            # block-registers ALONG each seam, so it is the same measurement as the global solve
            # at finer granularity and has to see the same pixels. Fitting it on the MIP while
            # the solve ran on the raw plane would correct the residual of a solve that never
            # happened.
            source = _SeamSource(reg_planes, positions, pixel_size, tile_shape,
                                 metrics, 0, max_workers)
            corrections = build_seam_corrections(source)
            # A tile with no correction gets None from .field() = identity, so an unfittable
            # seam degrades to plain fusion rather than failing the run.
            get_field = TileWarper(corrections, tile_shape[0], tile_shape[1]).field
            # `source` holds `reg_planes`, and Python scoping keeps a function local alive to the
            # end of the call -- so clearing `reg_planes` alone below would free nothing. Both
            # names have to go.
            source = None

    # Registration is done with; drop its planes before allocating the projected stack, so the
    # two never coexist at peak (~0.24 GB against ~0.94 GB on a 27-FOV 4-channel well).
    reg_planes = None

    y_profile = make_1d_profile(tile_shape[0], blend_px)
    x_profile = make_1d_profile(tile_shape[1], blend_px)
    # Spilled to a scratch file above 2 GB (see squidmip._volume): a 10-plane 4-channel mosaic of
    # a 27-FOV 10x well is 8.79 GB and this machine has 16, while the tiles below want ~0.94 GB of
    # it. Still a plain ndarray, so nothing downstream changes; a z-reducer's single plane is under
    # the threshold and allocates exactly as it always did.
    out = allocate((n_t, len(channels), len(z_sources), h, w), dtype,
                   what=f"region {region!r}'s fused {len(z_sources)}-plane mosaic")

    # THE Z-OUTER STREAMING LOOP. One acquisition plane at a time: project it for every FOV, fuse
    # it for every timepoint, drop the tiles, release the written pages, move on. `origins` and
    # `get_field` were solved ONCE, above, and are read-only from here — that is what makes every
    # plane land on the same grid instead of each solving its own residual.
    tiles = None
    for z_i, z_src in enumerate(z_sources):
        with timer.stage("project"):
            # (n_tiles, T, C, Y, X) native dtype, C == len(channels) EXACTLY -- there is no longer
            # a registration channel riding along to be sliced off later, because registration read
            # its own plane above. `z=z_src` pulls exactly ONE acquisition plane for a plane-op
            # (None for a z-reducer, which consumes the whole stack), so project_well's output is
            # (T, C, 1, Y, X) either way and index 0 is the whole plane. These planes are FUSED,
            # never registered.
            #
            # Rebound (not reused) per z, and dropped at the end of the iteration, so two planes'
            # worth of tiles never coexist.
            tiles = None
            tiles = np.empty((len(fovs), n_t, len(channels), *tile_shape), dtype=dtype)
            for i, fov in enumerate(fovs):
                tiles[i] = project_well(reader, region, fov, reduce=_op.fn,
                                        consumes=_op.consumes, z=z_src)[:, channels, 0]

        with timer.stage("fuse"):
            def read_tile(idx: int, z_level: int, time_idx: int, _tiles=tiles) -> np.ndarray:
                # float32 because the numba blend kernels accumulate in float32; converting the
                # ONE tile the block is currently consuming keeps this at ~C x tile bytes.
                #
                # `z_level` is fuse_plane's index INTO `_tiles`, and `_tiles` holds exactly the
                # plane this iteration is fusing — so it is 0 below and this argument is ignored.
                # The z that varies is the OUTER loop's, which is the whole point: the tiles are
                # streamed, not indexed.
                #
                # No channel slice: `_tiles` holds exactly the requested channels, in the requested
                # order. The old `[:len(channels)]` existed to drop a registration channel that had
                # been appended to the projected stack; registration now reads its own raw plane,
                # so the two concerns no longer share an array to be trimmed.
                return _tiles[idx][time_idx].astype(np.float32, copy=False)

            for t in range(n_t):

                def write_block(y0, y1, x0, x1, arr, _t=t, _z=z_i):
                    # ROUND back to the acquisition dtype, never truncate. `arr` is the numba
                    # kernel's float32 feathered blend and `out` is native (uint16 on this data),
                    # so a plain slice assignment would truncate toward zero and bias every pixel
                    # of the mosaic down by half a count -- the exact defect _cast_like was written
                    # for in _background/_decon/_flatfield. Stitch was the one operator writing
                    # around it. Same helper, not a second copy of it: one answer to "what does
                    # casting back to the acquisition dtype do".
                    #
                    # No clipping is at stake here (the blend is a convex combination of the tiles,
                    # so it cannot exceed their range), but _cast_like clips anyway and that costs
                    # nothing on values already in range.
                    out[_t, :, _z, y0:y1, x0:x1] = _cast_like(arr, out.dtype)

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


def _coordinate_region(reader, region, fovs, **kwargs):
    """Coordinate placement: :func:`stitch_region` with registration disabled (the control)."""
    kwargs["register"] = False
    return stitch_region(reader, region, fovs, **kwargs)


# The two shipped REGION operators. One `add_region_operator` call each, filed into the ONE
# operator table (`_engine._OPERATORS`) with `consumes=REGION_OP`; `stitch_plate` finds them by
# reading that declaration. There is no `_REGION_OPERATORS` dict here any more, and no
# `_REGION_REQUIRES` sidecar shadowing it.
RegionOperator = Callable[..., np.ndarray]

add_region_operator("stitch", stitch_region)
add_region_operator("coordinate", _coordinate_region)


def _accepts_kwarg(fn, name: str) -> bool:
    """Whether *fn* can be called with keyword *name*, directly or through ``**kwargs``.

    :func:`stitch_plate` injects a plate-wide ``flatfield=`` into the operator's kwargs, and the
    operator table is an EXTENSION POINT (:func:`squidmip.add_region_operator`) — a third-party
    operator that never heard of flat-fielding would get a TypeError from an argument it did not
    ask for. Asking first keeps the injection additive.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):   # builtins / C callables have no introspectable signature
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return name in params


def _resolve_region_operator(name: str, operator_kwargs: Optional[dict] = None) -> RegionOperator:
    """The CALLABLE a region-operator name resolves to, refusing anything this loop cannot run.

    Two refusals, both read off the one table's declaration and neither off a name:

    * an unknown name — the engine's own resolver says so, listing every operator;
    * a name that IS registered but is not a region operator (``"fov" not in consumes``), which
      before the collapse showed up as "unknown region operator 'mip'" — a sentence that said the
      operator did not exist when it existed and was simply not this kind.

    Deliberately does NOT check ``requires``: this is also the lookup that answers "what is this
    operator" for callers that are not about to run it. The dependency refusal belongs where the
    run starts, and that is :func:`stitch_plate`.
    """
    from squidmip._engine import _resolve_operator, available_region_operators

    operator = _resolve_operator(name)
    if "fov" not in operator.consumes:
        raise KeyError(
            f"{name!r} is a registered operator but not a REGION operator: it declares "
            f"consumes={sorted(operator.consumes)} and takes planes, while stitch_plate hands its "
            f"operator (reader, region, fovs). Region operators: "
            f"{available_region_operators()}; run {name!r} with squidmip.project_plate."
        )
    return operator.with_params(operator_kwargs)


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
    ``(region, fov, (T, C, Nz, Y, X))`` yield in completion order. A consumer written against
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
        ``(region, anchor_fov, image)``; ``image`` is ``(T, C, Nz, H, W)`` native dtype —
        ``Nz`` is 1 for a z-reducing ``projector=`` and the acquisition's depth for a plane-op
        (IMA-277 fuses those per plane).

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

    # DECLARED parameters are BOUND; everything else is passed through to the call. A region
    # operator could declare none until it became a record in the one table, so every kwarg used to
    # be pass-through — and a declared `Param` that arrived as a pass-through kwarg would be
    # swallowed by the operator's own `**kwargs` and silently run at its default, which is exactly
    # the "the panel value reached the console line and not the pixels" defect `params=` exists to
    # end. Split here, once, by asking the declaration.
    declared = {p.name for p in _resolve_operator(operator).params}
    bound = {k: operator_kwargs.pop(k) for k in list(operator_kwargs) if k in declared}
    op = _resolve_region_operator(operator, bound)
    ok, why = operator_available(operator)
    if not ok:
        # BEFORE the reader is warmed and before a single well is submitted. The plane-operator
        # path's `bind_operator` refuses in the same place for the same reason: an operator that cannot
        # run must say so by name, not run every well into the same ImportError and let `on_error`
        # file it as N skips and a successful run.
        raise MissingOperatorDependency(why)

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

    # ONE illumination profile for the whole plate, estimated before any well starts.
    #
    # This is the standalone's semantics: its flat-field worker builds a `TileFusion` with NO
    # region filter and hands the single (C, Y, X) array to every region it then stitches. It is
    # also the only defensible answer -- illumination is a property of the optical path, so a
    # per-well profile would divide each well by a different gain and quietly destroy
    # photometric comparability ACROSS wells, which is most of what a plate is for.
    #
    # Sampling is spread over the wells rather than taken from the first one, so a plate whose
    # first well is empty or saturated does not set the gain for all the others. Skipped when the
    # caller supplied a profile or turned the correction off, and left to the operator (which
    # estimates per region and says so) if estimation raises -- a plate that cannot estimate a
    # profile should still stitch.
    if _accepts_kwarg(op, "flatfield") \
            and operator_kwargs.get("correct_illumination", True) is not False \
            and operator_kwargs.get("flatfield") is None and wells:
        spread = [(r, f) for r, fs in wells.items() for f in fs]
        rng = np.random.default_rng(_FF_SEED)
        picked = [spread[i] for i in
                  sorted(rng.choice(len(spread), size=min(_FF_MAX_TILES, len(spread)),
                                    replace=False))]
        try:
            from squidmip._flatfield import FlatfieldProfile, estimate_profile

            z = int(meta["n_z"]) // 2
            names = [c["name"] for c in meta["channels"]]
            path = _flatfield_npy_path(reader)
            selected = _selected_profiles(names)
            if selected is not None:
                # The profile the user picked in the GUI's flat-field tab. ONE owner: the same
                # global the registered `flatfield` plane-op reads, so the two operators correct
                # by the same gain field instead of by two independently resolved ones. See
                # _selected_profiles for the full precedence.
                operator_kwargs["flatfield"] = selected
                # ONE PROFILE PER CHANNEL, and the line says so. It used to say "applied to all
                # %d channel(s)", which was an accurate description of a defect: `_selected_profiles`
                # broadcast a single field over every channel. It now returns a per-channel map or
                # nothing at all, so there is no longer a case this sentence could describe.
                _log.info("Flatfield: using the profile selected in the GUI for the whole plate — "
                          "%d channel(s), one %dx%d gain field each.",
                          len(selected), *next(iter(selected.values())).shape)
            elif path is not None and path.exists():
                # The acquisition already carries a profile -- ours from a previous run, or the
                # standalone's. Reuse beats re-deriving: it is what makes this compute-once, and
                # it is the only way the two tools agree on the SAME gain field rather than two
                # independently estimated ones.
                _log.info("Flatfield: loading the stored plate-wide profile from %s…", path.name)
                # `per_channel_from_npy`, not a second `from_npy(channel=i)` loop. Mapping a
                # channel NAME to a plane INDEX of the stored (C, Y, X) file is exactly one rule,
                # and it lived open-coded in two places here and in `resolve_flatfield` while the
                # GUI's own "Load illumination profile" had a third answer (plane 0, for every
                # channel). One reader is what stops the plate-wide stitch, the per-region stitch
                # and the GUI disagreeing about which gain field a channel gets.
                operator_kwargs["flatfield"] = FlatfieldProfile.per_channel_from_npy(path, names)
                _log.info("Flatfield: loaded %d plate-wide profile(s) from %s.",
                          len(names), path.name)
            else:
                # Before, not after (see estimate_region_flatfield): this stage runs BEFORE the
                # first region is submitted to the pool, so nothing else in the GUI moves while it
                # runs -- the plate bar's unit is the region and the region count is still 0.
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
                    raise  # a missing package is not a corrupt well -- see project_plate
                except Exception as exc:
                    if on_error is None:
                        raise  # default: fail-fast (project_plate's contract, unchanged)
                    on_error(region, anchor_fov, exc)  # opt-in: record + SKIP, keep going
                    continue
                yield region, anchor_fov, image
