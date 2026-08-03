"""Spot detection — a simple, fast nuclei counter, registered as a PLANE-OP (IMA-spot-detect).

WHAT THIS IS FOR
----------------
Spencer, verbatim: *"add a simple 'spot detection' for Nuclei just to test the interface"* …
*"this is just to test the interface. I expect we'll be able to adapt additional, more complex,
segmentation later as operators."*

So the value here is the SEAM, not the segmenter. This module deliberately ships the most boring
published recipe that produces the three things an analysis operator has to produce:

    a MASK      -> a napari ``Labels`` layer   (what was found)
    CENTROIDS   -> a napari ``Points`` layer   (where)
    a COUNT     -> a number in the status line (how many)

When Cellpose (or Fractal's ``Threshold Segmentation``, or anything else) arrives, it replaces
:func:`detect_spots` and nothing else moves: the registration, the worker, the layer types and
the readout are all already in place and all already tested.

THE ALGORITHM IS NOT MINE
-------------------------
It is scikit-image's own published nuclei recipe, from
**"Segment human cells (in mitosis)"** —
https://scikit-image.org/docs/stable/auto_examples/applications/plot_human_mitosis.html
which is the example the scikit-image project itself points at for counting nuclei (it reports
317 nuclei / mitotic index 0.091 on its fixture). Its steps, and ours:

    ============================================  =========================================
    the example                                   here
    ============================================  =========================================
    ``filters.rank.mean`` (smooth)                ``filters.gaussian(sigma=sigma_px)``
    ``filters.threshold_multiotsu(classes=3)``    ``filters.threshold_otsu``
    ``morphology.remove_small_objects``           same
    ``ndi.distance_transform_edt``                same
    ``feature.peak_local_max(min_distance=7)``    same, ``min_distance=min_distance_px``
    ``measure.label`` (of the peak mask)          same
    ``segmentation.watershed(-distance, ...)``    same
    ``measure.regionprops`` (count / properties)  ``measure.regionprops_table`` (centroids)
    ============================================  =========================================

Two deliberate deviations, both stated rather than hidden:

* **``threshold_otsu``, not ``threshold_multiotsu(classes=3)``.** The example needs three classes
  because it separates *dividing* nuclei (very bright) from ordinary ones; it then still uses
  ``thresholds[0]`` — the background/foreground split — to build the mask it watersheds. We want
  only that split, and two-class Otsu is the direct, cheaper way to ask for it.
* **``filters.gaussian``, not ``filters.rank.mean``.** ``rank.mean`` is uint-only and needs an
  explicit footprint; ``gaussian`` takes a scalar sigma, which is the one knob a user can
  reason about. Same job (denoise before thresholding).

WHY NOT ``blob_log`` / ``blob_dog``
-----------------------------------
https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html is the
other obvious candidate and is genuinely simpler — ``len(blob_dog(img))`` *is* the count. It was
rejected for one reason: it returns points and radii only, never a mask. A ``Labels`` layer is
the thing every real segmentation operator after this one will produce (Cellpose, Fractal's
``Threshold Segmentation``, ``napari-segment-blobs-and-things-with-membranes`` — all of them
return ``napari.types.LabelsData``), so an interface test that cannot carry a ``Labels`` layer
would not be testing the interface we actually have to build.

PRIOR ART FOR THE SEAM
----------------------
* **napari-segment-blobs-and-things-with-membranes** (haesleinhuepf) — every operator is a plain
  function annotated ``-> "napari.types.LabelsData"``; the return TYPE is the whole declaration
  of what layer the result becomes. TAKEN: the result is a ``Labels`` layer, and the operator is
  a plain array->array function with no viewer in its signature.
* **cellpose-napari** ``_dock_widget.py`` — runs the model on a ``@thread_worker`` and pushes
  ``viewer.add_labels(masks, ...)`` from the ``returned`` callback, so the GUI never blocks.
  TAKEN: the off-thread + push-on-completion shape (as a ``QThread``, which is what this codebase
  already uses). NOT taken: it surfaces no count at all, only "masks updated" in the logger.
  Spencer asked for the number, so the count goes in the layer NAME, the status readout, and
  ``Points.features`` (see below) rather than nowhere.
* **Fractal** (``fractal-tasks-core``) — a task declares itself once, as data
  (``name``/``category: "Segmentation"``/``input_types``/``output_types``), and per-object
  measurements live in a separate *feature table* keyed by label value, not smuggled into the
  image. TAKEN: the declaration is the ``Operator`` registry entry at the bottom of this file
  (one ``add_projector`` call, zero engine edits), and the per-object record rides on the
  ``Points`` layer's ``features`` keyed by ``label`` — which is the same "row per object, indexed
  by label id" contract, expressed in the shape napari already has.

WHY ``consumes = frozenset()`` (a plane-op, NOT a z-reducer)
------------------------------------------------------------
Segmentation is a per-plane MAP: every z-plane is segmented on its own terms and z survives at
full depth. Declaring ``{"z"}`` would mean "this operator eats the z axis", which for a
segmenter could only be implemented by segmenting one plane and calling it the well's answer —
i.e. silently discarding every cell that was in focus on a different plane.

Collapsing z is ``mip``'s job. "MIP, then count" is a two-operator chain, which is exactly how
Fractal and CellProfiler compose this (a projection task followed by a segmentation task), and
it keeps each operator's declaration honest.

THE ONE PLACE THIS CAN LOSE INFORMATION, AND IT SAYS SO
--------------------------------------------------------
``project_well`` allocates its ``(T, C, Z, Y, X)`` output in the acquisition's NATIVE dtype and
writes each operator's plane into it. A label image is not intensity data: on a uint8
acquisition a 300-nucleus field would wrap round and report a wrong count while looking
perfectly fine. :func:`spots_op` therefore checks the label count against the input dtype and
raises, naming the dtype and both numbers. ``project_plate`` surfaces that with the region and
FOV attached, so a region that cannot be segmented SAYS SO BY NAME.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Iterable, Optional

import numpy as np

from squidmip._engine import Param, add_projector
from squidmip.projection import labels_op, plane_op

# The layer key the UI files this operator's results under, so the registry and the UI cannot
# drift apart on the spelling (same discipline as _background.LAYER_KEY).
#
# NOTE the name: "spot", not "otsu-watershed" and not "skimage". The operator's identity is WHAT
# IT PRODUCES (a label image + centroids + a count), never which library produced it. Julio:
# *"we can import some nice cell detection algos. Like good ones"* — Cellpose and StarDist both
# return exactly this contract, so they are siblings in the segmenter table below, not a rewrite.
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
    """The four knobs, all in PIXELS so they mean something without a microscope in the room.

    Defaults are tuned for a 10x nucleus (~15-25 px across) and are deliberately not
    magic-number-free: this is an interface test, and a user who needs different numbers changes
    them here or passes their own ``SpotParams``.
    """

    #: Gaussian denoise before thresholding. The example's ``rank.mean`` equivalent.
    sigma_px: float = 2.0
    #: Connected components with fewer pixels than this are noise, not cells.
    min_area_px: int = 30
    #: ``peak_local_max(min_distance=)`` — how close two nuclei centres may be. The example uses 7.
    min_distance_px: int = 7
    #: Split touching nuclei with the distance-transform watershed. Off = plain connected
    #: components (faster, fuses anything that touches).
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
    """One plane's answer, as ONE truth in three shapes.

    ``count`` is not a separately maintained tally — it is derived from ``labels`` in
    :func:`detect_spots` and the three are asserted consistent by
    ``test_the_count_is_the_number_of_distinct_labels_not_a_second_tally``. Two representations
    of one number that can disagree is this codebase's most-repeated defect shape.
    """

    #: int32 label image, same shape as the input. 0 = background (napari renders it transparent).
    labels: np.ndarray
    #: ``(count, 2)`` float array of ``(row, col)`` centroids — napari's own 2D world axis order.
    centroids: np.ndarray
    #: How many nuclei. ``== labels.max() == len(centroids)``.
    count: int


class SpotDetectionCancelled(RuntimeError):
    """The caller's ``should_stop`` said stop. Raised, never returned as a partial result.

    A half-finished segmentation that looks like a finished one is the silent failure this
    project bans, so cancellation is an exception rather than a short answer.
    """


#: The stages, in order, with the label the busy indicator shows. The list is the progress
#: DENOMINATOR, so a stage added here updates ``progress(done, total)`` automatically — there is
#: no second hardcoded total to keep in sync.
STAGES: tuple[str, ...] = (
    "smoothing",
    "thresholding",
    "removing specks",
    "distance transform",
    "finding nuclei centres",
    "splitting touching nuclei",
    "measuring",
)


class MissingSegmenterDependency(RuntimeError):
    """A registered segmenter's optional package is not importable. NAMED, never silent.

    Cellpose and StarDist are heavyweight optional dependencies. A segmenter that quietly drops
    out of the list when its package is missing is indistinguishable from one that was never
    written — so every registered segmenter is ALWAYS listed by
    :func:`available_segmenters`, and the refusal happens, by name, at
    :func:`resolve_segmenter`.
    """


@dataclass(frozen=True)
class Segmenter:
    """One registered segmentation algorithm.

    ``fn`` has the signature of :func:`skimage_watershed`::

        fn(plane: np.ndarray, params: SpotParams, *, on_stage, should_stop) -> SpotResult

    That is the whole contract, and it is deliberately the shape Cellpose and StarDist already
    have: they take a 2-D plane, they return a LABEL IMAGE, and the count and the centroids are
    derived from it. A Cellpose entry is therefore ~20 lines wrapping ``model.eval(...)``
    and one :func:`add_segmenter` call — no change to the operator, the worker, the layers or
    the readout.
    """

    name: str
    fn: Callable[..., "SpotResult"]
    #: Importable module names this segmenter needs, e.g. ``("cellpose",)``. Checked at resolve
    #: time so a missing one is a named refusal rather than a silent absence.
    requires: tuple[str, ...] = ()
    #: One line for the UI. Says out loud when something is slow or wants a GPU.
    blurb: str = ""


_SEGMENTERS: dict[str, Segmenter] = {}

#: The segmenter used when the caller does not name one.
DEFAULT_SEGMENTER: str = "otsu-watershed"


def add_segmenter(name: str, fn, *, requires=(), blurb: str = "") -> None:
    """Register a segmentation algorithm under *name*.

    This is the seam a real segmenter plugs into::

        def cellpose_nuclei(plane, params, *, on_stage=None, should_stop=None):
            from cellpose import models
            if should_stop is not None and should_stop():
                raise SpotDetectionCancelled("cancelled before cellpose")
            if on_stage is not None:
                on_stage("running cellpose", 0, 1)
            masks, _flows, _styles = models.CellposeModel(gpu=True).eval(
                plane, diameter=params.min_distance_px * 2)
            return result_from_labels(masks)

        add_segmenter("cellpose", cellpose_nuclei, requires=("cellpose",),
                      blurb="Cellpose (slow; wants a GPU)")

    Raises
    ------
    ValueError
        On an empty name, a non-callable, or a name already taken — a silent clobber of a
        registered segmenter would be a quiet correctness bug (same rule as ``add_projector``).
    """
    if not name:
        raise ValueError("segmenter name must be a non-empty string")
    if not callable(fn):
        raise ValueError(f"segmenter for {name!r} is not callable: {fn!r}")
    if name in _SEGMENTERS:
        raise ValueError(
            f"segmenter {name!r} is already defined; pick a distinct name "
            f"(defined: {available_segmenters()})."
        )
    _SEGMENTERS[name] = Segmenter(name, fn, tuple(requires), blurb)


def available_segmenters() -> list[str]:
    """Every registered segmenter, INCLUDING ones whose dependency is not installed.

    Listing only the importable ones would make "cellpose is not installed" and "nobody wrote a
    cellpose operator" look identical in the UI. Pair with :func:`segmenter_available` to grey a
    row out with a reason instead of dropping it.
    """
    return sorted(_SEGMENTERS)


def segmenter_available(name: str) -> tuple[bool, str]:
    """``(ok, reason_if_not)`` — is this segmenter's dependency importable right now?"""
    import importlib.util

    seg = _SEGMENTERS.get(name)
    if seg is None:
        return False, f"unknown segmenter {name!r}; available: {available_segmenters()}"
    missing = [m for m in seg.requires if importlib.util.find_spec(m) is None]
    if missing:
        return False, (f"segmenter {name!r} needs {', '.join(missing)}, which "
                       f"{'are' if len(missing) > 1 else 'is'} not installed "
                       f"(pip install {' '.join(missing)})")
    return True, ""


def resolve_segmenter(name: str) -> Segmenter:
    """Look up a segmenter, failing LOUD and by name on an unknown key or a missing package."""
    seg = _SEGMENTERS.get(name)
    if seg is None:
        raise KeyError(
            f"unknown segmenter {name!r}; available: {available_segmenters()}. "
            "Add new ones with squidmip._spots.add_segmenter(name, fn)."
        )
    ok, why = segmenter_available(name)
    if not ok:
        raise MissingSegmenterDependency(why)
    return seg


def result_from_labels(labels: np.ndarray) -> SpotResult:
    """Build the :class:`SpotResult` contract from any segmenter's LABEL IMAGE.

    This is what makes a real segmenter cheap to add: Cellpose's ``model.eval`` and StarDist's
    ``predict_instances`` both return a label array, and everything else the UI needs — the
    count, the centroids, the sequential relabelling — is derived from it HERE, once, so no
    segmenter has to reimplement (or disagree about) it.
    """
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
    """Count and outline the nuclei in one 2-D plane, with the named *algorithm*.

    THE dispatch point. The UI, the worker and the engine registration all call this and none of
    them names a library, so swapping in Cellpose is a change to *algorithm* and nothing else.

    See the module docstring for the default recipe and its source. Returns a
    :class:`SpotResult`.

    Parameters
    ----------
    on_stage:
        Optional ``on_stage(label: str, done: int, total: int)``, called as each stage of the
        recipe starts. This is the progress seam the GUI worker turns into signals; the pure
        function stays Qt-free.
    should_stop:
        Optional zero-argument predicate polled between stages. Returns True -> raise
        :class:`SpotDetectionCancelled`. Cancellation is BETWEEN stages, not inside one: the
        skimage calls are single C-level calls with no interruption point, so on a 27 Mpx region
        mosaic the worst-case latency is one watershed (~5 s measured). A click still returns
        immediately because none of this is on the GUI thread.

    Raises
    ------
    ValueError
        If *plane* is not 2-D, or *params* is invalid. Never returns a partial or guessed answer.
    SpotDetectionCancelled
        If *should_stop* returned True.
    KeyError, MissingSegmenterDependency
        If *algorithm* is unknown, or its optional package is not installed.
    """
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
    """The default segmenter: scikit-image's published Otsu + distance-watershed nuclei recipe.

    Source and the two deliberate deviations are in the module docstring. Registered below as
    ``"otsu-watershed"``; it is a peer of any future ``"cellpose"`` entry, not a base class.
    """
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

    # 1. denoise. float32, not float64: half the transient on a 2084^2 plane, ample precision
    #    for a threshold. `gaussian` copies, so the caller's buffer (and the raw TIFF behind it)
    #    is untouched -- the read-only-dataset rule, enforced by a test.
    _stage("smoothing")
    smooth = filters.gaussian(plane.astype(np.float32, copy=False), sigma=params.sigma_px,
                              preserve_range=True)

    # 2. threshold. A CONSTANT plane has no bimodal histogram; Otsu on it is meaningless, so say
    #    "nothing here" instead of returning whatever the tie-break produced. (A blank well is a
    #    legitimate result, not an error -- it must not abort a plate run.)
    _stage("thresholding")
    lo, hi = float(smooth.min()), float(smooth.max())
    if not hi > lo:
        return empty
    mask = smooth > filters.threshold_otsu(smooth)

    # 3. drop the specks. skimage 0.26 RENAMED min_size -> max_size and changed the comparison
    #    from "strictly smaller" to "smaller than or equal", so `max_size = min_area_px - 1`
    #    reproduces the old, documented `min_size = min_area_px` semantics exactly. Passing
    #    min_size= here still works but is deprecated; passing max_size=min_area_px would be an
    #    off-by-one that silently deletes cells of exactly the minimum size.
    _stage("removing specks")
    if params.min_area_px > 1:
        mask = morphology.remove_small_objects(mask, max_size=params.min_area_px - 1)
    if not mask.any():
        return empty

    if not params.split_touching:
        labels = measure.label(mask)
    else:
        # 4. the watershed split, verbatim from the example: distance transform -> local maxima
        #    -> label those as markers -> watershed the NEGATED distance under the mask.
        #    This step is not optional on real tissue: measured on region `manual0` of the 10x
        #    slide, the 405 channel thresholds into ONE confluent component, so without the
        #    watershed the whole 5731x4793 mosaic counts as 1 nucleus instead of 400.
        _stage("distance transform")
        distance = ndi.distance_transform_edt(mask)
        _stage("finding nuclei centres")
        peaks = feature.peak_local_max(
            distance, min_distance=params.min_distance_px, labels=mask, exclude_border=False,
        )
        _stage("splitting touching nuclei")
        if len(peaks) == 0:
            # Every component is smaller than the peak footprint. Fall back to plain connected
            # components rather than returning zero cells for a plane that visibly has some.
            labels = measure.label(mask)
        else:
            marker_mask = np.zeros(distance.shape, dtype=bool)
            marker_mask[tuple(peaks.T)] = True
            markers = measure.label(marker_mask)
            labels = segmentation.watershed(-distance, markers, mask=mask)

    # 5. the count and the centroids, both derived from `labels` by the SHARED helper every
    #    segmenter uses -- so a Cellpose entry gets identical counting semantics for free and
    #    cannot disagree with this one about what "how many" means.
    _stage("measuring")
    return result_from_labels(labels)


def spots_op(params: Optional[SpotParams] = None, *,
             algorithm: str = DEFAULT_SEGMENTER) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build the engine-facing **plane-op**: plane -> label image.

    This is the FACTORY behind a registry entry, not the registration itself. It used to be called
    at registration time with a frozen parameter set — ``add_projector("spot_tight",
    spots_op(SpotParams(min_area_px=80)))`` — i.e. one registry entry per parameter set. It is now
    reached through :func:`segmentation_operator`, so ONE entry takes its numbers at run time::

        project_plate(reader, projector="spot", operator_kwargs={"min_area_px": 80})

    The returned callable carries ``consumes = frozenset()`` (z survives at full depth) and
    ``produces = "labels"`` (the pixels are object ids, so the delivery path builds a napari
    Labels layer rather than windowing them as fluorescence).

    *algorithm* is resolved LAZILY, inside the call, on purpose: registering a Cellpose operator
    must not import cellpose (or claim a GPU) at ``import squidmip`` time, and a plate run that
    reaches an uninstalled segmenter must fail with
    :class:`MissingSegmenterDependency` naming the package — with the region and FOV attached by
    ``project_plate`` — rather than at interpreter start with no context.

    The label image comes back in the INPUT's dtype, because ``project_well`` writes it into a
    native-dtype ``(T, C, Z, Y, X)`` buffer. If the nuclei count does not fit that dtype the
    result would wrap and report a wrong number while looking fine, so that case raises.
    """
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

    _spots.__name__ = (f"spot({algorithm},sigma_px={params.sigma_px},"
                       f"min_area_px={params.min_area_px},"
                       f"split_touching={params.split_touching})")
    # TWO declarations on one callable. `plane_op` stamps consumes=frozenset() (z survives);
    # `labels_op` stamps produces="labels" (the pixels are OBJECT IDS, not light). The second one
    # is what stops the delivery path handing a segmentation to `add_image`, where the fluorescence
    # auto-window treats label 37 as 37 photons and the mask renders as a near-black gradient with
    # an opaque background over the mosaic it is supposed to annotate.
    return labels_op(plane_op(_spots))


# ---------------------------------------------------------------------------------------------
# Registrations. Two tables, two different questions:
#   add_segmenter  -> WHICH ALGORITHM counts the nuclei      (skimage now, cellpose/stardist next)
#   add_projector  -> WHICH OPERATOR the engine and the UI offer  (peer of mip / bgsub / decon)
#
# Adding Cellpose is one add_segmenter call plus (optionally) one add_projector call. It touches
# neither the engine, nor _SpotWorker, nor the napari layers, nor the readout.
# ---------------------------------------------------------------------------------------------

add_segmenter(
    DEFAULT_SEGMENTER, skimage_watershed,
    blurb="scikit-image Otsu + distance-transform watershed. Fast, no model, no GPU.",
)

# Cellpose is the PREFERRED nuclei segmenter (Spencer: "I was thinking Cellpose"; Julio: "cellpose
# should trump this"). Registered here unconditionally so it is always listed; it becomes the
# default only when its package is importable (see ``preferred_segmenter``). The registration is
# in _cellpose.py so the heavyweight adapter stays out of this module's import path.
from squidmip._cellpose import (
    SEGMENTER_NAME as _CELLPOSE,
    register as _register_cellpose,
    register_operator as _register_cellpose_operator,
)

_register_cellpose()


def preferred_segmenter() -> str:
    """The segmenter to use when the caller does not name one.

    Cellpose TRUMPS the traditional otsu-watershed when it is installed — it is the "good" model
    Spencer and Julio both named — but the traditional one is the honest fallback so the feature
    never becomes "install a 2 GB dependency or get nothing". The choice is by availability, not a
    hardcoded winner, so a machine without Cellpose still counts nuclei.
    """
    ok, _why = segmenter_available(_CELLPOSE)
    return _CELLPOSE if ok else DEFAULT_SEGMENTER

#: The engine parameters a segmentation operator declares, derived from :class:`SpotParams` so the
#: dataclass stays the ONE place the knobs and their defaults are written down. Adding a field to
#: ``SpotParams`` adds it to every registered segmentation operator with no second edit; spelling
#: them out here again is how the two would drift.
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
    """The FACTORY an ``add_projector`` entry registers for *algorithm*.

    ``add_projector(name, factory, params=SPOT_PARAMS)`` calls this with the declared defaults to
    build the entry's default binding, and calls it again with a run's ``operator_kwargs``. So
    ``spot`` and ``cellpose`` are one entry each that can be RUN at a different ``min_area_px``,
    rather than the ``add_projector("spot_tight", spots_op(SpotParams(min_area_px=80)))`` this
    module's own docstring used to recommend — which is one registry entry per parameter set, and
    is how a table of eight operators becomes a table of forty.

    *algorithm* is closed over rather than declared as a parameter, deliberately: it names WHICH
    OPERATOR this is (``spot`` is the traditional recipe, ``cellpose`` is the model), and an
    operator that can be re-pointed at a different algorithm at run time would make the layer key,
    the console line and the cache recipe all describe something other than what ran.
    """
    def _build(**kwargs) -> Callable:
        return spots_op(SpotParams(**kwargs), algorithm=algorithm)

    _build.__name__ = f"spots_op[{algorithm}]"
    return _build


# The whole engine registration: one call, no engine edit. `spot` is now a peer of mip / bgsub /
# decon / flatfield in the SAME table, and therefore appears in `runnable_operators()` for free.
#
# It now declares TWO more things about itself, and both are read by generic code that has never
# heard of spot detection: `produces="labels"` (inferred off the callable, stamped by `labels_op`
# in `spots_op`) picks the napari layer type, and `params=SPOT_PARAMS` lets one entry be run at a
# different min_area_px instead of needing a second entry.
add_projector(LAYER_KEY, segmentation_operator(DEFAULT_SEGMENTER), params=SPOT_PARAMS)

# ...and Cellpose, the SECOND consumer of the same two declarations. Two consumers is the point:
# one is a hypothetical seam. See `_cellpose.register_operator`.
_register_cellpose_operator()
