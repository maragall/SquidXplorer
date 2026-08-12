"""Spot detection — a simple nuclei counter, registered as a plane-op, plus the
segmenter registry that Cellpose and friends plug into."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Iterable, Optional

import numpy as np

from squidxplorer._engine import Param, add_operator
from squidxplorer.projection import (
    MissingDependency,
    labels_op,
    missing_requirements,
    normalise_requires,
    plane_op,
    requirement_refusal,
)

# The layer key the UI files this operator's results under; the registry and the UI share it.
LAYER_KEY: str = "spot"
LAYER_LABEL: str = "spot detection (nuclei)"


def mask_layer_name(channel: str) -> str:
    """napari layer label for the ``Labels`` mask derived from *channel*. ONE spelling."""
    return f"{channel} · nuclei mask"


def centroid_layer_name(channel: str) -> str:
    """napari layer label for the ``Points`` centroids derived from *channel*. ONE spelling."""
    return f"{channel} · nuclei centroids"


@dataclass(frozen=True)
class SpotParams:
    """The four knobs, all in pixels. Defaults are tuned for a 10x nucleus (~15-25 px)."""

    #: Gaussian denoise before thresholding.
    sigma_px: float = 2.0
    #: Connected components with fewer pixels than this are noise, not cells.
    min_area_px: int = 30
    #: ``peak_local_max(min_distance=)`` — how close two nuclei centres may be.
    min_distance_px: int = 7
    #: Split touching nuclei with the distance-transform watershed.
    split_touching: bool = True

    def validate(self) -> "SpotParams":
        """Raise on a parameter that cannot mean anything, rather than clamping it silently."""
        if not self.sigma_px > 0:
            raise ValueError(f"sigma_px must be > 0, got {self.sigma_px!r}")
        if self.min_area_px < 0:
            raise ValueError(f"min_area_px must be >= 0, got {self.min_area_px!r}")
        if self.min_distance_px < 1:
            raise ValueError(f"min_distance_px must be >= 1, got {self.min_distance_px!r}")
        return self


DEFAULT_PARAMS = SpotParams()


@dataclass(frozen=True)
class SpotResult:
    """One plane's answer: labels, centroids and count are one truth in three shapes."""

    #: int32 label image, same shape as the input. 0 = background.
    labels: np.ndarray
    #: ``(count, 2)`` float array of ``(row, col)`` centroids.
    centroids: np.ndarray
    #: How many nuclei. ``== labels.max() == len(centroids)``.
    count: int


class SpotDetectionCancelled(RuntimeError):
    """The caller's ``should_stop`` said stop. Raised, never returned as a partial result."""


#: The stages, in order, with the label the busy indicator shows; also the progress denominator.
STAGES: tuple[str, ...] = (
    "smoothing",
    "thresholding",
    "removing specks",
    "distance transform",
    "finding nuclei centres",
    "splitting touching nuclei",
    "measuring",
)


class MissingSegmenterDependency(MissingDependency):
    """A registered segmenter's optional package is not importable. NAMED, never silent."""


@dataclass(frozen=True)
class Segmenter:
    """One registered segmentation algorithm: ``fn(plane, params, *, on_stage, should_stop) -> SpotResult``."""

    name: str
    fn: Callable[..., "SpotResult"]
    #: Importable module names this segmenter needs, e.g. ``("cellpose",)``.
    requires: tuple[str, ...] = ()
    #: One line for the UI.
    blurb: str = ""
    #: The :class:`SpotParams` field names this algorithm ACTUALLY READS; ``None`` means all.
    honours: Optional[tuple[str, ...]] = None


_SEGMENTERS: dict[str, Segmenter] = {}

#: The segmenter used when the caller does not name one.
DEFAULT_SEGMENTER: str = "otsu-watershed"


def add_segmenter(name: str, fn, *, requires=(), blurb: str = "", honours=None) -> None:
    """Register a segmentation algorithm under *name*."""
    if not name:
        raise ValueError("segmenter name must be a non-empty string")
    if not callable(fn):
        raise ValueError(f"segmenter for {name!r} is not callable: {fn!r}")
    if name in _SEGMENTERS:
        raise ValueError(
            f"segmenter {name!r} is already defined; pick a distinct name "
            f"(defined: {available_segmenters()})."
        )
    if honours is not None:
        honours = tuple(str(n) for n in honours)
        known = {f.name for f in fields(SpotParams)}
        unknown = [n for n in honours if n not in known]
        if unknown:
            raise ValueError(
                f"segmenter {name!r} declares it honours {unknown}, which SpotParams does not "
                f"have (it has {sorted(known)}). A declared parameter that no field backs is a "
                f"control that cannot reach the pixels, which is the failure this declaration "
                f"exists to prevent."
            )
    _SEGMENTERS[name] = Segmenter(name, fn, normalise_requires(requires), blurb, honours)


def segmenter_honours(algorithm: str) -> tuple[str, ...]:
    """The :class:`SpotParams` field names *algorithm* reads. Every field, unless it says less."""
    seg = _SEGMENTERS.get(str(algorithm))
    declared = None if seg is None else seg.honours
    return tuple(f.name for f in fields(SpotParams)) if declared is None else declared


def available_segmenters() -> list[str]:
    """Every registered segmenter, INCLUDING ones whose dependency is not installed."""
    return sorted(_SEGMENTERS)


def segmenter_available(name: str) -> tuple[bool, str]:
    """``(ok, reason_if_not)`` — is this segmenter's dependency importable right now?"""
    seg = _SEGMENTERS.get(name)
    if seg is None:
        return False, f"unknown segmenter {name!r}; available: {available_segmenters()}"
    missing = missing_requirements(seg.requires)
    if missing:
        return False, requirement_refusal("segmenter", name, missing)
    return True, ""


def resolve_segmenter(name: str) -> Segmenter:
    """Look up a segmenter, failing LOUD and by name on an unknown key or a missing package."""
    seg = _SEGMENTERS.get(name)
    if seg is None:
        raise KeyError(
            f"unknown segmenter {name!r}; available: {available_segmenters()}. "
            "Add new ones with squidxplorer._spots.add_segmenter(name, fn)."
        )
    ok, why = segmenter_available(name)
    if not ok:
        raise MissingSegmenterDependency(why)
    return seg


def result_from_labels(labels: np.ndarray) -> SpotResult:
    """Build the :class:`SpotResult` contract from any segmenter's LABEL IMAGE."""
    from skimage import measure, segmentation

    labels = np.ascontiguousarray(labels, dtype=np.int32)
    labels, _fwd, _inv = segmentation.relabel_sequential(labels)
    count = int(labels.max())
    if count == 0:
        return SpotResult(labels, np.zeros((0, 2), dtype=np.float64), 0)
    props = measure.regionprops_table(labels, properties=("label", "centroid"))
    centroids = np.column_stack([props["centroid-0"], props["centroid-1"]]).astype(np.float64)
    return SpotResult(labels, centroids, count)


def detect_spots(plane: np.ndarray, params: Optional[SpotParams] = None, *,
                 algorithm: str = DEFAULT_SEGMENTER,
                 on_stage=None, should_stop=None) -> SpotResult:
    """Count and outline the nuclei in one 2-D plane, with the named *algorithm*."""
    params = (params or DEFAULT_PARAMS).validate()

    plane = np.asarray(plane)
    if plane.ndim != 2:
        raise ValueError(
            f"spot detection needs a 2-D plane (y, x); got shape {plane.shape!r} "
            f"({plane.ndim}-D). A z-stack is segmented plane by plane by the engine, which "
            "is why this operator declares consumes=frozenset()."
        )

    seg = resolve_segmenter(algorithm)
    return seg.fn(plane, params, on_stage=on_stage, should_stop=should_stop)


def skimage_watershed(plane: np.ndarray, params: SpotParams, *,
                      on_stage=None, should_stop=None) -> SpotResult:
    """The default segmenter: scikit-image's published Otsu + distance-watershed nuclei recipe."""
    import scipy.ndimage as ndi
    from skimage import feature, filters, measure, morphology, segmentation

    def _stage(name: str) -> None:
        """Announce a stage and honour a cancel, in that order — one call per step."""
        if should_stop is not None and should_stop():
            raise SpotDetectionCancelled(f"spot detection cancelled before {name!r}")
        if on_stage is not None:
            on_stage(name, STAGES.index(name), len(STAGES))

    empty = SpotResult(np.zeros(plane.shape, dtype=np.int32),
                       np.zeros((0, 2), dtype=np.float64), 0)

    # 1. denoise. float32 suffices; `gaussian` copies, so the caller's buffer is untouched.
    _stage("smoothing")
    smooth = filters.gaussian(plane.astype(np.float32, copy=False), sigma=params.sigma_px,
                              preserve_range=True)

    # 2. threshold. A constant plane has no bimodal histogram: report "nothing here".
    _stage("thresholding")
    lo, hi = float(smooth.min()), float(smooth.max())
    if not hi > lo:
        return empty
    mask = smooth > filters.threshold_otsu(smooth)

    # 3. drop the specks. skimage 0.26 renamed min_size -> max_size and made the comparison
    #    inclusive, so `max_size = min_area_px - 1` reproduces the old min_size semantics exactly.
    _stage("removing specks")
    if params.min_area_px > 1:
        mask = morphology.remove_small_objects(mask, max_size=params.min_area_px - 1)
    if not mask.any():
        return empty

    if not params.split_touching:
        labels = measure.label(mask)
    else:
        # 4. watershed split: distance transform -> local maxima as markers -> watershed the
        #    negated distance under the mask.
        _stage("distance transform")
        distance = ndi.distance_transform_edt(mask)
        _stage("finding nuclei centres")
        peaks = feature.peak_local_max(
            distance, min_distance=params.min_distance_px, labels=mask, exclude_border=False,
        )
        _stage("splitting touching nuclei")
        if len(peaks) == 0:
            # Every component is smaller than the peak footprint: fall back to plain
            # connected components rather than returning zero cells.
            labels = measure.label(mask)
        else:
            marker_mask = np.zeros(distance.shape, dtype=bool)
            marker_mask[tuple(peaks.T)] = True
            markers = measure.label(marker_mask)
            labels = segmentation.watershed(-distance, markers, mask=mask)

    # 5. count and centroids, derived from `labels` by the shared helper every segmenter uses.
    _stage("measuring")
    return result_from_labels(labels)


def spots_op(params: Optional[SpotParams] = None, *,
             algorithm: str = DEFAULT_SEGMENTER) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build the engine-facing plane-op: plane -> label image, in the input's dtype."""
    params = (params or DEFAULT_PARAMS).validate()

    def _spots(p: np.ndarray) -> np.ndarray:
        res = detect_spots(p, params, algorithm=algorithm)
        dtype = np.asarray(p).dtype
        if np.issubdtype(dtype, np.integer):
            ceiling = int(np.iinfo(dtype).max)
            if res.count > ceiling:
                raise ValueError(
                    f"spot detection found {res.count} nuclei, which does not fit the "
                    f"acquisition's {dtype} label container (max {ceiling}). Writing it would "
                    "wrap round and report a WRONG cell count that looks correct. Raise "
                    "min_area_px, or run this operator on a wider dtype."
                )
        return res.labels.astype(dtype, copy=False)

    # The callable's name reaches the layer key, the console line and the recipe, so it names
    # only the parameters this algorithm actually reads.
    shown = ",".join(f"{n}={getattr(params, n)}" for n in segmenter_honours(algorithm))
    _spots.__name__ = f"spot({algorithm}{',' + shown if shown else ''})"
    # plane_op stamps consumes=frozenset() (z survives); labels_op stamps produces="labels".
    return labels_op(plane_op(_spots))


# Registrations. Two tables, two questions: add_segmenter -> which algorithm counts the nuclei;
# add_operator -> which operator the engine and the UI offer.

add_segmenter(
    DEFAULT_SEGMENTER, skimage_watershed,
    blurb="scikit-image Otsu + distance-transform watershed. Fast, no model, no GPU.",
)

# Cellpose is registered unconditionally so it is always listed; it becomes the default only
# when importable (see ``preferred_segmenter``). The adapter lives in _cellpose.py.
from squidxplorer._cellpose import (
    SEGMENTER_NAME as _CELLPOSE,
    register as _register_cellpose,
    register_operator as _register_cellpose_operator,
)

_register_cellpose()


def preferred_segmenter() -> str:
    """The segmenter to use when the caller does not name one: Cellpose when installed."""
    ok, _why = segmenter_available(_CELLPOSE)
    return _CELLPOSE if ok else DEFAULT_SEGMENTER

#: The engine parameters a segmentation operator declares, derived from :class:`SpotParams` so
#: the dataclass stays the one place the knobs and their defaults are written down.
_SPOT_PARAM_BLURBS: dict[str, str] = {
    "sigma_px": "Gaussian denoise radius, in pixels, before thresholding.",
    "min_area_px": "Objects smaller than this many pixels are noise, not cells.",
    "min_distance_px": "Closest two nuclei centres may be, in pixels. Cellpose reads twice this "
                       "as the expected nucleus diameter.",
    "split_touching": "Split touching nuclei with the distance-transform watershed.",
}
SPOT_PARAMS: tuple[Param, ...] = tuple(
    Param(f.name, f.default, _SPOT_PARAM_BLURBS.get(f.name, ""))
    for f in fields(SpotParams)
)


def segmentation_operator(algorithm: str) -> Callable[..., Callable]:
    """The FACTORY an ``add_operator`` entry registers for *algorithm*."""
    def _build(**kwargs) -> Callable:
        return spots_op(SpotParams(**kwargs), algorithm=algorithm)

    _build.__name__ = f"spots_op[{algorithm}]"
    return _build


add_operator(LAYER_KEY, segmentation_operator(DEFAULT_SEGMENTER), params=SPOT_PARAMS)

_register_cellpose_operator()
