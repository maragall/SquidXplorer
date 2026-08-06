"""Cellpose as a SquidHCS segmenter — the second entry in the ``add_segmenter`` table.

Spencer Schwarz (CSO): *"I was thinking Cellpose."* Julio: *"we can import some nice cell
detection algos. Like good ones."* This is that — the generalist Cellpose model run ZERO-SHOT
(pretrained, no transfer learning), producing the SAME contract the traditional otsu-watershed
segmenter produces (a label image, from which the count and centroids are derived once, in
``result_from_labels``). Nothing downstream — the operator, the worker, the layers, the readout —
knows Cellpose from skimage; the only difference the user sees is the algorithm name and a
better mask.

WHY A THIN ADAPTER AND NOT A REWRITE. ``_spots.py`` deliberately made "n algorithms for one job"
a one-call seam: a segmenter takes a 2-D plane and returns a label array. Cellpose's ``model.eval``
already returns a label array, so this file is ~a wrapper plus one ``add_segmenter`` call. See the
seam's own docstring in ``_spots.add_segmenter``.

GPU ON BOTH MAC AND WINDOWS. Cellpose runs on torch. ``gpu=True`` makes Cellpose's own device
picker select CUDA on Windows/Linux and Metal (MPS) on Apple Silicon, falling back to CPU when
neither is present — Cellpose never raises for a missing GPU, it just runs slower. We ask for the
GPU, then LOG which device Cellpose actually chose, so a demo that is silently on CPU is visible in
the log panel rather than a mystery-slow run.

NO SILENT FAILURES. Cellpose is a heavyweight OPTIONAL dependency. The segmenter is registered
unconditionally (``requires=("cellpose",)``), so it is always LISTED; the refusal, if the package
is absent, happens by name at ``resolve_segmenter`` — never a row that quietly disappears.
"""

from __future__ import annotations

import logging

import numpy as np

from squidmip._engine import add_projector
from squidmip._spots import (
    SpotDetectionCancelled,
    SpotParams,
    SpotResult,
    add_segmenter,
    result_from_labels,
)

from squidmip._logpane import get_logger

log = get_logger("cellpose")

#: The registered name. Its identity is the ALGORITHM, not the package — matched to the table's
#: "otsu-watershed" sibling.
SEGMENTER_NAME = "cellpose"

#: Cellpose's own progress is opaque (one ``eval`` call), so we report a single indeterminate
#: stage rather than inventing sub-steps we cannot measure — the same honesty rule the busy
#: indicator uses for an unknown total.
_STAGE = "running cellpose"


def _pick_model(gpu: bool):
    """Build a nuclei Cellpose model, tolerant of the API shift across Cellpose 2/3/4.

    Cellpose 2/3 expose ``models.Cellpose(model_type="nuclei")``; Cellpose 4 (the CPSAM
    generalist) dropped ``model_type`` and exposes ``models.CellposeModel(gpu=...)``. We try the
    nuclei-specialised model first (better on nuclei, which is what "spot detection (nuclei)"
    means) and fall back to the generalist, so the adapter does not pin a Cellpose version.
    """
    from cellpose import models

    if hasattr(models, "Cellpose"):
        try:
            return models.Cellpose(gpu=gpu, model_type="nuclei"), "nuclei"
        except (TypeError, ValueError):
            pass
    return models.CellposeModel(gpu=gpu), "cpsam"


def _log_device() -> None:
    """Say, once per run, which device Cellpose will actually use — so a silently-CPU demo shows."""
    try:
        from cellpose import core

        gpu = bool(core.use_gpu())
        log.info("cellpose device: %s", "GPU (CUDA/MPS)" if gpu else "CPU (no GPU detected)")
    except Exception:                          # noqa: BLE001 - a probe must never break detection
        pass


def cellpose_nuclei(plane: np.ndarray, params: SpotParams, *,
                    on_stage=None, should_stop=None) -> SpotResult:
    """Segment nuclei in one 2-D plane with Cellpose, returning the standard ``SpotResult``.

    ``params.min_distance_px`` is the nearest allowed nucleus spacing; twice that is a reasonable
    expected DIAMETER to hand Cellpose, which is what its ``diameter`` argument wants. Cancellation
    is checked before the (uninterruptible) ``eval`` call and raised, never returned as a partial
    result — a half-done segmentation that looks whole is the silent failure this project bans.
    """
    if should_stop is not None and should_stop():
        raise SpotDetectionCancelled("cancelled before cellpose")
    if on_stage is not None:
        on_stage(_STAGE, 0, 1)

    _log_device()
    model, kind = _pick_model(gpu=True)
    diameter = max(1.0, float(params.min_distance_px) * 2.0)

    # channels=[0, 0] = a single greyscale channel for both "cytoplasm" and "nucleus" slots, which
    # is correct for our single-channel plane. Cellpose 4's eval ignores channels; passing it is
    # harmless there and required on 2/3, so it is always supplied.
    try:
        out = model.eval(np.asarray(plane), diameter=diameter, channels=[0, 0])
    except TypeError:
        out = model.eval(np.asarray(plane), diameter=diameter)   # Cellpose 4: no channels kwarg
    masks = out[0] if isinstance(out, (tuple, list)) else out

    if should_stop is not None and should_stop():
        raise SpotDetectionCancelled("cancelled after cellpose")
    if on_stage is not None:
        on_stage(_STAGE, 1, 1)

    result = result_from_labels(masks)
    log.info("cellpose (%s): %d nuclei on a %s plane", kind, result.count, plane.shape)
    return result


def register() -> None:
    """Add Cellpose to the segmenter table. Idempotent-safe: ``add_segmenter`` refuses a duplicate,
    so this is called exactly once, from ``_spots``' registration block."""
    add_segmenter(
        SEGMENTER_NAME, cellpose_nuclei, requires=("cellpose",),
        blurb="Cellpose — pretrained generalist, zero-shot (slow on CPU; wants a GPU)",
    )


#: The name Cellpose holds in the ENGINE's operator table, i.e. in ``available_projectors()``,
#: ``runnable_operators()``, the CLI's ``--projector`` and the viewer's operator list. Deliberately
#: the same string as :data:`SEGMENTER_NAME`: the two tables answer different questions about the
#: SAME algorithm, and giving it two spellings is how a log line and a layer key start disagreeing.
OPERATOR_NAME = SEGMENTER_NAME


def register_operator() -> None:
    """Add Cellpose to the ENGINE's operator table — the half that was missing.

    Cellpose has been a registered SEGMENTER (``add_segmenter``, above) and nothing else. That
    table answers "which algorithm counts the nuclei" for the GUI's one Detect-nuclei button; it is
    not the operator table, so ``available_projectors()`` had never heard of Cellpose, no card, no
    dropdown, no CLI ``--projector cellpose``, and no plate-scale run. It was the analysis feature
    bolted on BESIDE the operator system rather than into it.

    This is what plugging it in costs, in full: one call, no engine edit, no delivery-path edit,
    no napari edit. Everything that makes it work is a DECLARATION the generic code reads —

        consumes  frozenset()   inferred from `plane_op`   -> segmented per plane, z survives
        produces  "labels"      inferred from `labels_op`  -> delivered as a napari Labels layer
        params    SPOT_PARAMS   declared                   -> runnable at a different diameter
        requires  ("cellpose",) declared                   -> refused BY NAME when absent, listed
                                                              either way

    The `requires` line is the SEGMENTER declaration ten lines above, repeated on the operator
    entry, and it has to be: the two tables are asked by different callers. `resolve_segmenter`
    guards the GUI's Detect-nuclei button; `bind_operator` guards `--projector cellpose` and every
    plate-scale run, and until this it guarded nothing — a plate run of cellpose on a machine
    without cellpose raised ImportError per well, which `on_error` filed as a skip, and the run
    reported success having produced no masks at all.

    NO TORCH AT IMPORT TIME. Registering here builds the default binding, which is a closure over
    ``SpotParams`` and the string ``"cellpose"``; the ``from cellpose import models`` lives inside
    ``cellpose_nuclei``, one call deeper, and runs when a plane is actually segmented. Measured on
    the developer machine: ``import squidmip`` 0.146 s, ``import cellpose`` 0.52 s (torch alone
    0.498 s). Registering eagerly and importing lazily is what keeps the second number off app
    startup, and it is why this operator needs no lazy-registration machinery.
    """
    from squidmip._spots import SPOT_PARAMS, segmentation_operator

    add_projector(OPERATOR_NAME, segmentation_operator(SEGMENTER_NAME), params=SPOT_PARAMS,
                  requires=("cellpose",))
