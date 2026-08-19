"""Cellpose nuclei segmentation as its own operator: a thin zero-shot adapter over the
pretrained model."""

from __future__ import annotations


import numpy as np

from squidxplorer._spots import (
    SpotDetectionCancelled,
    SpotParams,
    SpotResult,
    result_from_labels,
)

from squidxplorer._logpane import get_logger

log = get_logger("cellpose")

OPERATOR_NAME = "cellpose"

#: The ``SpotParams`` fields :func:`cellpose_nuclei` actually reads.
READS: tuple[str, ...] = ("min_distance_px",)

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


def register_operator() -> None:
    """Add Cellpose to the engine's operator table; cellpose itself imports lazily."""
    from squidxplorer._spots import SPOT_PARAMS, add_segmentation_operator

    # Filtered from SPOT_PARAMS so each default and blurb stays defined once, in SpotParams.
    add_segmentation_operator(OPERATOR_NAME, cellpose_nuclei,
                              params=tuple(p for p in SPOT_PARAMS if p.name in READS),
                              requires=("cellpose",), extra="segment")
