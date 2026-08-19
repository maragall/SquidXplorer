"""Qt-free plate-cell geometry, out of the ``_plate_overview`` widget module.

The plate NAVIGATOR is a widget; where a plane lands inside a plate cell is arithmetic. The
non-widget consumers (``_workers``'s preview/computed-plate threads, ``_ingest``) import the
geometry from here so reading a plate cell's layout never costs a Qt import.
``_plate_overview`` imports these back for its own painting; ``tests/test_layering.py``
enforces that this module stays Qt-free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from squidxplorer._montage import _area_downsample

#: The plate cell edge in pixels: every well thumbnail is fitted into a ``_CELL x _CELL`` box.
_CELL = 88


def _fit_cell(a: np.ndarray) -> np.ndarray:
    """Resize a 2D plane to exactly (_CELL, _CELL) for the montage tile."""
    if a.shape == (_CELL, _CELL):
        return a
    if a.shape[0] >= _CELL and a.shape[1] >= _CELL:
        return _area_downsample(a, _CELL, _CELL)
    yi = (np.arange(_CELL) * a.shape[0]) // _CELL
    xi = (np.arange(_CELL) * a.shape[1]) // _CELL
    return a[yi][:, xi].astype(np.float32)


def _fit_box(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a 2D plane to exactly (h, w), the arbitrary-target sibling of :func:`_fit_cell`."""
    h, w = max(1, int(h)), max(1, int(w))
    if a.shape == (h, w):
        return a
    if a.shape[0] >= h and a.shape[1] >= w:
        return _area_downsample(a, h, w)
    yi = (np.arange(h) * a.shape[0]) // h
    xi = (np.arange(w) * a.shape[1]) // w
    return a[yi][:, xi].astype(np.float32)


def _box_union(a, b):
    """Union of two ``(top, left, h, w)`` boxes; ``a`` may be None (nothing accumulated yet)."""
    if a is None:
        return tuple(int(v) for v in b)
    top = min(a[0], b[0])
    left = min(a[1], b[1])
    bottom = max(a[0] + a[2], b[0] + b[2])
    right = max(a[1] + a[3], b[1] + b[3])
    return (int(top), int(left), int(bottom - top), int(right - left))


def resolve_plate_root(path) -> tuple[Path, bool]:
    """(path, is_plate): is_plate True when *path* already holds an OME-zarr plate."""
    p = Path(path)
    if (p / "plate.ome.zarr").is_dir() or (p.name.endswith(".zarr") and (p / "zarr.json").exists()):
        return p, True
    return p, False


def _mosaic_boxes(meta: dict) -> dict:
    """``{(region, fov): (top, left, h, w)}`` — every FOV's box inside its _CELL thumbnail.

    Pure geometry, delegated to :mod:`squidxplorer._placement`. Returns ``{}`` when the
    acquisition has no stage positions or no pixel size, the signal to keep the single-tile
    path. Placement failures for one region are contained to that region.
    """
    from squidxplorer._placement import cell_boxes, fov_offsets_px

    positions = meta.get("fov_positions_um") or {}
    if not positions or meta.get("pixel_size_um") in (None, 0):
        return {}
    frame_shape = meta["frame_shape"]
    out: dict = {}
    for region in meta["regions"]:
        fovs = meta["fovs_per_region"][region]
        if len(fovs) < 2:
            continue
        try:
            offsets = fov_offsets_px(positions, region, fovs, meta.get("pixel_size_um"))
            for fov, box in cell_boxes(offsets, frame_shape, _CELL).items():
                out[(region, fov)] = box
        except (KeyError, ValueError):
            continue
    return out


def content_box(shape, h: int = _CELL, w: int = _CELL) -> tuple[int, int, int, int]:
    """``(top, left, height, width)``: where a *shape*-shaped plane lands in an ``h`` x ``w`` box.

    Applies ``_placement.cell_boxes``' rule (``s = min(box/mh, box/mw)``, then centre) to the
    whole mosaic rather than to its individual FOVs, so a fused mosaic and the raw mosaic of the
    same region land in the same place, at the same size, in the same cell.
    """
    h, w = max(1, int(h)), max(1, int(w))
    mh, mw = max(1, int(shape[0])), max(1, int(shape[1]))
    s = min(h / mh, w / mw)
    ih = max(1, min(h, int(round(mh * s))))
    iw = max(1, min(w, int(round(mw * s))))
    return (h - ih) // 2, (w - iw) // 2, ih, iw
