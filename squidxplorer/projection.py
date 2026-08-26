"""Per-FOV projection primitives and the operator declarations they carry.

Output is 5-D (T, C, Z, Y, X) in native dtype; a z-reducer collapses Z to 1, a plane-op
keeps it at full depth. The z iterator is ``metadata["z_levels"]``, never ``range(n_z)``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # avoid import cost / cycle at runtime
    from squidxplorer.reader import SquidReader


# Which axis an operator consumes decides the engine's grouping:
#   PLANE_OP  — plane -> plane, z survives at full depth.
#   Z_REDUCER — all z of one (t, c) -> one plane.
#   REGION_OP — a whole well's FOVs -> one fused array; not a member of CONSUMABLE_AXES,
#               because project_well cannot group over it.
PLANE_OP: frozenset[str] = frozenset()
Z_REDUCER: frozenset[str] = frozenset({"z"})
REGION_OP: frozenset[str] = frozenset({"fov"})
CONSUMABLE_AXES: frozenset[str] = frozenset({"z"})


# What the output pixels MEAN: INTENSITY is windowed/colormapped; LABELS are integer
# object ids and must never be windowed or interpolated.
INTENSITY: str = "intensity"
LABELS: str = "labels"
RESULT_KINDS: frozenset[str] = frozenset({INTENSITY, LABELS})


class MissingDependency(RuntimeError):
    """A registry entry declared a package that is not importable."""


def missing_requirements(requires: Iterable[str]) -> list[str]:
    """The module names in *requires* that are not importable right now, in declaration order."""
    import importlib.util

    missing = []
    for module in requires:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):     # parent package absent, or a malformed name
            found = False
        if not found:
            missing.append(module)
    return missing


def requirement_refusal(kind: str, name: str, missing: Sequence[str]) -> str:
    """The one refusal sentence every registry uses: what is missing, and how to install it."""
    return (f"{kind} {name!r} needs {', '.join(missing)}, which "
            f"{'are' if len(missing) > 1 else 'is'} not installed "
            f"(pip install {' '.join(missing)})")


def normalise_requires(requires) -> tuple[str, ...]:
    """Coerce a ``requires`` declaration to a tuple of module names, refusing anything else."""
    if isinstance(requires, str):
        requires = (requires,)
    names = tuple(requires)
    for module in names:
        if not isinstance(module, str) or not module:
            raise ValueError(
                f"requires must be importable MODULE names (e.g. ('petakit',)); got {module!r}")
    return names


def normalise_produces(produces) -> str:
    """Coerce a ``produces`` declaration to a known result-kind string, refusing anything else."""
    kind = str(produces)
    if kind not in RESULT_KINDS:
        raise ValueError(
            f"unknown result kind {produces!r}; this engine knows {sorted(RESULT_KINDS)}. "
            f"An operator whose pixels measure light declares produces={INTENSITY!r}; one whose "
            f"pixels are integer object ids declares produces={LABELS!r}."
        )
    return kind


def labels_op(fn: Callable[[Iterable[np.ndarray]], np.ndarray]) -> Callable[..., np.ndarray]:
    """Stamp ``produces = LABELS`` on an already-shaped operator callable."""
    fn.produces = LABELS
    return fn


def normalise_consumes(consumes) -> frozenset[str]:
    """Coerce a ``consumes`` declaration to a frozenset of axis names, refusing anything unsupported."""
    if isinstance(consumes, str):
        consumes = (consumes,)
    axes = frozenset(consumes)
    if "fov" in axes:
        raise ValueError(
            "consumes={'fov'} is not supported by the operator table: an operator is "
            "Iterable[plane] -> plane and never sees a tile's x/y stage geometry, which any "
            "inter-FOV operation (stitching, illumination-field fitting across a well) requires. "
            "Register it with squidxplorer.add_region_operator(name, fn) instead: that stamps "
            "consumes=REGION_OP on the SAME registry record, and the region loop reads it."
        )
    unknown = axes - CONSUMABLE_AXES
    if unknown:
        raise ValueError(
            f"unsupported axis {sorted(unknown)[0]!r} in consumes={sorted(axes)}; this engine "
            f"groups over {sorted(CONSUMABLE_AXES)} only. A plane-op declares consumes=frozenset(), "
            "a z-reduction declares consumes=frozenset({'z'})."
        )
    return axes


def cast_like(values: np.ndarray, dtype, *, copy: bool = True) -> np.ndarray:
    """Cast a float result back to the acquisition dtype, rounding and clipping integers.

    Round (not truncate) and clip (not wrap); ``copy=False`` rounds in place and refuses a
    non-float buffer, where an in-place ``rint`` would silently do nothing.
    """
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if copy:
            values = np.clip(np.rint(values), info.min, info.max)
        elif not np.issubdtype(values.dtype, np.floating):
            raise ValueError(
                f"cast_like(copy=False) needs a floating-point buffer to round in place; got "
                f"{values.dtype}. In-place rounding of an integer array is a silent no-op."
            )
        else:
            np.rint(values, out=values)
            np.clip(values, info.min, info.max, out=values)
    return values.astype(dtype, copy=False)


def plane_op(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Lift a ``plane -> plane`` function into the engine's ``Iterable[plane] -> plane`` shape."""
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


# for_channel(acquisition_path, channel_name) -> operator: an attribute on the callable that
# lets project_well specialise an operator once per channel (e.g. a per-channel PSF).
def acquisition_path(reader) -> Optional[str]:
    """The acquisition folder a reader reads, or ``None`` for a reader that does not say.

    ``source_id`` is the contract's identity member; ``_path`` survives as the fallback for
    doubles and readers written before it was declared.
    """
    path = getattr(reader, "source_id", None) or getattr(reader, "_path", None)
    return None if path is None else str(path)


def bind_channel(reduce, path: Optional[str], channel: str):
    """Specialise *reduce* to *channel* when it declares it can be; otherwise hand it back."""
    declare = getattr(reduce, "for_channel", None)
    if declare is None:
        return reduce
    bound = declare(path, channel)
    if not callable(bound):
        raise ValueError(
            f"{getattr(reduce, '__name__', reduce)!r}.for_channel({channel!r}) returned "
            f"{bound!r}, which is not callable; it must return an operator."
        )
    before = getattr(reduce, "consumes", None)
    after = getattr(bound, "consumes", before)
    if before is not None and frozenset(after) != frozenset(before):
        raise ValueError(
            f"{getattr(reduce, '__name__', reduce)!r}.for_channel({channel!r}) returned an "
            f"operator consuming {sorted(after)}, but the registered one consumes "
            f"{sorted(before)}. Specialising to a channel must not change the output shape."
        )
    return bound


def _refuse_all_copied_through(reduce, per_channel: dict) -> None:
    """A per-channel specialisation may declare ``copies_through`` (its reason, a string)
    when it hands THAT channel back unchanged - decon does for a channel with no emission
    line (Julio, 2026-08-25: "copy BF through unchanged with a named log line"). A run in
    which EVERY channel would be copied has nothing to compute and is refused by name
    rather than written as a copy of its input. Declaration-driven: no operator name here."""
    reasons = {c: getattr(op, "copies_through", None) for c, op in per_channel.items()}
    if reasons and all(reasons.values()):
        detail = "; ".join(f"{c}: {why}" for c, why in reasons.items())
        raise ValueError(
            f"{getattr(reduce, '__name__', reduce)!r} would copy every channel through "
            f"unchanged ({detail}): nothing to compute, so the run is refused rather than "
            "written as a copy of its input."
        )


def operator_halo_px(reduce, nz: int) -> int:
    """The lateral halo (px) *reduce* DECLARES an ROI window needs around it so the window's
    interior equals the whole-field result: ``halo_px`` on the callable, an int or a callable
    of the stack depth. Undeclared means 0 (a MIP's pixel depends on its own column only).
    Negative is refused by name."""
    declared = getattr(reduce, "halo_px", None)
    if declared is None:
        return 0
    halo = int(declared(int(nz)) if callable(declared) else declared)
    if halo < 0:
        raise ValueError(
            f"{getattr(reduce, '__name__', reduce)!r} declares halo_px={halo}, which is not a "
            "width; a halo is 0 or more pixels.")
    return halo


def read_window(reader, region, fov, channel, z_level, time_point, window):
    """One plane's ``(r0, r1, c0, c1)`` window: the reader's own ``read_window`` when it has
    one (the Zarr reader reads only the covering chunks), else the plane read whole and
    sliced. Either way the caller is handed the window and nothing else."""
    r0, r1, c0, c1 = (int(v) for v in window)
    own = getattr(reader, "read_window", None)
    if callable(own):
        plane = np.asarray(own(region, fov, channel, z_level, time_point, (r0, r1, c0, c1)))
    else:
        plane = np.asarray(reader.read(region, fov, channel, z_level, time_point))[r0:r1, c0:c1]
    if plane.shape != (r1 - r0, c1 - c0):
        raise ValueError(
            f"region {region!r} FOV {fov} {channel!r} z={z_level}: the reader handed back "
            f"shape {plane.shape} for window rows {r0}:{r1}, cols {c0}:{c1}")
    return plane


def _clamp_window(window, frame_shape) -> tuple:
    """A ``(r0, r1, c0, c1)`` window inside *frame_shape*, refused by name when it is not."""
    y, x = (int(v) for v in frame_shape)
    try:
        r0, r1, c0, c1 = (int(v) for v in window)
    except (TypeError, ValueError):
        raise ValueError(f"window must be (r0, r1, c0, c1) frame pixels, got {window!r}") from None
    if not (0 <= r0 < r1 <= y and 0 <= c0 < c1 <= x):
        raise ValueError(
            f"window rows {r0}:{r1}, cols {c0}:{c1} does not fit inside the {y}x{x} frame "
            "(a window is clamped to the frame BEFORE it reaches the engine).")
    return r0, r1, c0, c1


def project(planes: Iterable[np.ndarray]) -> np.ndarray:
    """Maximum-intensity project an iterable of planes into one plane, streaming and dtype-preserving."""
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
    """Tenengrad focus measure: sum of squared gradient magnitude, higher = sharper."""
    gy, gx = np.gradient(plane.astype(np.float32, copy=False))
    return float(np.square(gx).sum() + np.square(gy).sum())


# (`select_reference_z` / `project_reference` — the Tenengrad z-SELECTING reducer behind the
# shelved `reference` operator — were deleted 2026-08-24 with project_well's whole select_index
# arm; `_tenengrad` stays because the GUI's z-slider autofocus reads it. Git history reinstates.)

project.consumes = Z_REDUCER


def project_well(
    reader: "SquidReader",
    region: str,
    fov: int,
    reduce: Callable[[Iterable[np.ndarray]], np.ndarray] = project,
    consumes=None,
    time_point: Optional[int] = None,
    z_level: Optional[int] = None,
    window: Optional[tuple] = None,
) -> np.ndarray:
    """Apply one operator to a FOV's planes for every channel and timepoint.

    The grouping comes from the operator's ``consumes`` declaration: a z-reducer collapses z
    to 1, a plane-op keeps z at full depth. ``time_point``/``z_level`` restrict the run to one
    timepoint / one acquisition plane (``z_level=`` is plane-ops only).

    ``window=(r0, r1, c0, c1)`` (frame pixels) is ruling z, sub-FOV decon: the operator runs
    on the window PLUS the halo it declares (:func:`operator_halo_px`, clamped to the frame),
    reads only that (:func:`read_window`), and the halo is trimmed, so the output's xy IS the
    window while z stays whole. Every FOV of an ROI run carries its own window.
    """
    meta = reader.metadata
    channels = [c["name"] for c in meta["channels"]]
    z_levels = meta["z_levels"]
    n_t = meta["n_t"]
    y, x = meta["frame_shape"]

    if consumes is None:
        consumes = getattr(reduce, "consumes", Z_REDUCER)
    consumes = normalise_consumes(consumes)

    if time_point is None:
        timepoints = tuple(range(n_t))
    else:
        if not 0 <= time_point < n_t:
            raise ValueError(f"timepoint {time_point} out of range for an acquisition with n_t={n_t}")
        timepoints = (time_point,)

    # A z-consuming operator that DECLARES ``keeps_depth`` (on the callable) returns the whole
    # PROCESSED stack — decon: true 3-D deconvolution whose every plane the user examines — so
    # the output depth is the input's while the operator still sees all z in one call.
    keeps_depth = bool(getattr(reduce, "keeps_depth", False)) and "z" in consumes

    # One acquisition plane; refused for EVERY z-consumer: "the MIP of one plane" is a
    # different result, and a depth-keeping solve over one plane treats out-of-focus light
    # as in-plane blur (Julio, 2026-08-26: a preview shows plane z of the FULL solve).
    if z_level is not None:
        if "z" in consumes:
            raise ValueError(
                f"z_level={z_level} selects ONE acquisition plane, which is only meaningful for a "
                f"plane-op. {getattr(reduce, '__name__', reduce)!r} declares "
                f"consumes={sorted(consumes)} - it CONSUMES z, so restricting it to one plane "
                "would silently change what it computes. Drop z_level=, or use a plane-op."
            )
        if z_level not in z_levels:
            raise ValueError(
                f"z_level={z_level} is not one of this acquisition's z levels {list(z_levels)}")
        z_levels = [z_level]

    # z consumed -> one group per (t, c); z not consumed -> one group per (t, c, z).
    z_groups = [tuple(z_levels)] if "z" in consumes else [(z_level,) for z_level in z_levels]
    out_depth = len(z_levels) if keeps_depth else len(z_groups)
    if window is not None:
        r0, r1, c0, c1 = _clamp_window(window, (y, x))
        out_h, out_w = r1 - r0, c1 - c0
    else:
        out_h, out_w = y, x
    out = np.empty((len(timepoints), len(channels), out_depth, out_h, out_w), dtype=meta["dtype"])
    # One specialisation per channel for operators declaring `for_channel`.
    path = acquisition_path(reader) if hasattr(reduce, "for_channel") else None
    per_channel = {c: bind_channel(reduce, path, c) for c in channels}
    _refuse_all_copied_through(reduce, per_channel)
    for t_i, t_src in enumerate(timepoints):
        for c_i, channel in enumerate(channels):
            op = per_channel[channel]
            if window is None:
                read_h, read_w, trim = y, x, (slice(None), slice(None))

                def _read(z, t, _ch=channel):
                    return reader.read(region, fov, _ch, z, t)
            else:
                # The window plus THIS channel's halo (its own PSF), clamped to the frame:
                # at an edge the windowed solve sees the same edge the whole-field one does.
                halo = operator_halo_px(op, len(z_levels))
                hr0, hr1 = max(0, r0 - halo), min(y, r1 + halo)
                hc0, hc1 = max(0, c0 - halo), min(x, c1 + halo)
                read_h, read_w = hr1 - hr0, hc1 - hc0
                trim = (slice(r0 - hr0, r0 - hr0 + out_h), slice(c0 - hc0, c0 - hc0 + out_w))

                def _read(z, t, _ch=channel, _w=(hr0, hr1, hc0, hc1)):
                    return read_window(reader, region, fov, _ch, z, t, _w)
            for k, group in enumerate(z_groups):
                planes = (_read(z_level, t_src) for z_level in group)
                if keeps_depth:
                    stack = np.asarray(op(planes))
                    if stack.shape != (out_depth, read_h, read_w):
                        raise ValueError(
                            f"{getattr(reduce, '__name__', reduce)!r} declares keeps_depth "
                            f"and so owes a ({out_depth}, {read_h}, {read_w}) stack; it "
                            f"returned shape {stack.shape}.")
                    out[t_i, c_i, :] = stack[(slice(None),) + trim]
                else:
                    out[t_i, c_i, k] = np.asarray(op(planes))[trim]  # streamed z; bounded memory
    return out


def select_fovs(metadata: dict, n_fovs: Optional[int] = 1) -> dict[str, list[int]]:
    """``{region: [fov, ...]}``: the first *n_fovs* FOVs of each well; ``None`` means all (ragged ok)."""
    if n_fovs is not None and n_fovs < 1:
        raise ValueError(f"n_fovs must be >= 1 or None (= all), got {n_fovs}")

    fovs_per_region = metadata["fovs_per_region"]
    selected: dict[str, list[int]] = {}
    for region in metadata["regions"]:
        available = fovs_per_region[region]
        if n_fovs is None:
            selected[region] = list(available)
            continue
        if n_fovs > len(available):
            raise ValueError(
                f"n_fovs={n_fovs} requested but region {region!r} has only "
                f"{len(available)} FOV(s): {available}. Pass n_fovs=None to take whatever "
                "each well has instead of requiring a uniform count."
            )
        selected[region] = list(available[:n_fovs])
    return selected


def scope_wells(metadata: dict, n_fovs: Optional[int], regions) -> "dict[str, list[int]]":
    """``{region: [fov, ...]}`` for a run: *regions* is None (whole acquisition), a sequence of
    names, or a mapping ``{region: fovs}``. Always intersected with what the acquisition has."""
    from collections.abc import Mapping

    wells = select_fovs(metadata, n_fovs=n_fovs)
    if isinstance(regions, Mapping):
        available = metadata["fovs_per_region"]
        out: "dict[str, list[int]]" = {}
        for region in dict.fromkeys(regions):
            if region not in available:
                continue
            have = set(available[region])
            out[region] = [int(f) for f in dict.fromkeys(regions[region]) if int(f) in have]
        return out
    if regions is None:
        return wells
    keep = list(dict.fromkeys(regions))
    return {r: wells[r] for r in keep if r in wells}


