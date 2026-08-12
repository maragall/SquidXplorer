"""Cellpose as a SquidHCS segmenter: a thin zero-shot adapter over the pretrained model."""

from __future__ import annotations

import logging

import numpy as np

from squidxplorer._engine import add_operator
from squidxplorer._spots import (
    SpotDetectionCancelled,
    SpotParams,
    SpotResult,
    add_segmenter,
    result_from_labels,
)

from squidxplorer._logpane import get_logger

log = get_logger("cellpose")

SEGMENTER_NAME = "cellpose"

#: The ``SpotParams`` fields :func:`cellpose_nuclei` actually reads.
HONOURED_PARAMS: tuple[str, ...] = ("min_distance_px",)

_STAGE = "running cellpose"


def _pick_model(gpu: bool):
    """Build a nuclei Cellpose model, tolerant of the API shift across Cellpose 2/3/4."""
    from cellpose import models

    if hasattr(models, "Cellpose"):
        try:
            return models.Cellpose(gpu=gpu, model_type="nuclei"), "nuclei"
        except (TypeError, ValueError):
            pass
    return models.CellposeModel(gpu=gpu), "cpsam"


def _log_device() -> None:
    """Log which device Cellpose will actually use."""
    try:
        from cellpose import core

        gpu = bool(core.use_gpu())
        log.info("cellpose device: %s", "GPU (CUDA/MPS)" if gpu else "CPU (no GPU detected)")
    except Exception:                          # noqa: BLE001 - a probe must never break detection
        pass


def cellpose_nuclei(plane: np.ndarray, params: SpotParams, *,
                    on_stage=None, should_stop=None) -> SpotResult:
    """Segment nuclei in one 2-D plane with Cellpose, returning the standard ``SpotResult``."""
    if should_stop is not None and should_stop():
        raise SpotDetectionCancelled("cancelled before cellpose")
    if on_stage is not None:
        on_stage(_STAGE, 0, 1)

    _log_device()
    model, kind = _pick_model(gpu=True)
    diameter = max(1.0, float(params.min_distance_px) * 2.0)

    # channels=[0, 0]: single greyscale channel; required on Cellpose 2/3, absent on 4.
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
    """Add Cellpose to the segmenter table."""
    add_segmenter(
        SEGMENTER_NAME, cellpose_nuclei, requires=("cellpose",), honours=HONOURED_PARAMS,
        blurb="Cellpose — pretrained generalist, zero-shot (slow on CPU; wants a GPU)",
    )


#: Same string as :data:`SEGMENTER_NAME`: one algorithm, two tables.
OPERATOR_NAME = SEGMENTER_NAME


def register_operator() -> None:
    """Add Cellpose to the engine's operator table; cellpose itself imports lazily."""
    from squidxplorer._spots import SPOT_PARAMS, segmentation_operator, segmenter_honours

    # Filtered from SPOT_PARAMS so each default and blurb stays defined once, in SpotParams.
    honoured = frozenset(segmenter_honours(SEGMENTER_NAME))
    add_operator(OPERATOR_NAME, segmentation_operator(SEGMENTER_NAME),
                  params=tuple(p for p in SPOT_PARAMS if p.name in honoured),
                  requires=("cellpose",))
