"""Spot detection — nuclei counting. One operator per algorithm: the Otsu-watershed default
lives here, and siblings like Cellpose register through :func:`add_segmentation_operator`."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from squidxplorer._engine import Param, add_operator
from squidxplorer.projection import labels_op, plane_op

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


def result_from_labels(labels: np.ndarray) -> SpotResult:
    """Build the :class:`SpotResult` contract from any algorithm's LABEL IMAGE."""
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
                 segment: Optional[Callable[..., SpotResult]] = None,
                 on_stage=None, should_stop=None) -> SpotResult:
    """Count and outline the nuclei in one 2-D plane.

    *segment* is the algorithm — ``fn(plane, params, *, on_stage, should_stop) -> SpotResult``
    — defaulting to :func:`skimage_watershed`.
    """
    params = (params or DEFAULT_PARAMS).validate()

    plane = np.asarray(plane)
    if plane.ndim != 2:
        raise ValueError(
            f"spot detection needs a 2-D plane (y, x); got shape {plane.shape!r} "
            f"({plane.ndim}-D). A z-stack is segmented plane by plane by the engine, which "
            "is why this operator declares consumes=frozenset()."
        )

    return (segment or skimage_watershed)(plane, params, on_stage=on_stage, should_stop=should_stop)


def skimage_watershed(plane: np.ndarray, params: SpotParams, *,
                      on_stage=None, should_stop=None) -> SpotResult:
    """The default algorithm: scikit-image's published Otsu + distance-watershed nuclei recipe."""
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

    # 5. count and centroids, derived from `labels` by the shared helper every algorithm uses.
    _stage("measuring")
    return result_from_labels(labels)


def spots_op(params: Optional[SpotParams] = None, *,
             segment: Optional[Callable[..., SpotResult]] = None,
             label: str = LAYER_KEY,
             reads: Optional[tuple[str, ...]] = None,
             ) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build the engine-facing plane-op: plane -> label image, in the input's dtype.

    *reads* names the :class:`SpotParams` fields *segment* actually uses; every field, unless
    it says less.
    """
    params = (params or DEFAULT_PARAMS).validate()
    if reads is None:
        reads = tuple(f.name for f in fields(SpotParams))

    def _spots(p: np.ndarray) -> np.ndarray:
        res = detect_spots(p, params, segment=segment)
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
    shown = ",".join(f"{n}={getattr(params, n)}" for n in reads)
    _spots.__name__ = f"{label}({shown})" if shown else label
    # plane_op stamps consumes=frozenset() (z survives); labels_op stamps produces="labels".
    return labels_op(plane_op(_spots))


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


def add_segmentation_operator(name: str, segment: Callable[..., SpotResult], *,
                              params: Sequence[Param], requires=(), extra=None) -> None:
    """One algorithm, one operator: register *segment* under *name*, declaring exactly the
    :class:`SpotParams` fields it reads (*params*, filtered from :data:`SPOT_PARAMS`)."""
    declared = tuple(params)
    reads = tuple(p.name for p in declared)
    known = {f.name for f in fields(SpotParams)}
    unknown = [n for n in reads if n not in known]
    if unknown:
        raise ValueError(
            f"segmentation operator {name!r} declares {unknown}, which SpotParams does not "
            f"have (it has {sorted(known)}). A declared parameter that no field backs is a "
            f"control that cannot reach the pixels, which is the failure this declaration "
            f"exists to prevent."
        )

    def _factory(**kwargs) -> Callable:
        return spots_op(SpotParams(**kwargs), segment=segment, label=name, reads=reads)

    _factory.__name__ = f"spots_op[{name}]"
    add_operator(name, _factory, params=declared, requires=requires, extra=extra)


add_segmentation_operator(LAYER_KEY, skimage_watershed, params=SPOT_PARAMS)

# Registered unconditionally so it is always listed; requires=("cellpose",) makes a missing
# package a named refusal at run time. The adapter lives in _cellpose.py.
from squidxplorer._cellpose import register_operator as _register_cellpose_operator

_register_cellpose_operator()
