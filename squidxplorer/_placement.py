"""Coordinate placement: stage micrometres -> pixel offsets.

The pure, GUI-free half of the multi-FOV mosaic: arithmetic on ``fov_positions_um`` plus
``pixel_size_um``. The origin is per region, never plate-wide.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable, Mapping, Optional

import numpy as np

from squidxplorer._conventions import acq_um

# Stage +y maps to image +row (downward).
_Y_SIGN = 1

# Placement mode vocabulary: one spelling, read by the plate model, viewer label and exports.

STAGE = "stage"
"""Cells at the positions the stage recorded."""

COMPACT = "compact"
"""Cells packed evenly, closing the space between regions."""

PLACEMENT_MODES = (STAGE, COMPACT)
"""Every mode there is."""

DEFAULT_PLACEMENT_MODE = STAGE
"""The default; the choice is never persisted per acquisition."""


def normalize_placement_mode(mode) -> str:
    """Validate a placement mode, raising on anything else. Never silently defaults."""
    if mode is None:
        return DEFAULT_PLACEMENT_MODE
    m = str(mode).strip().lower()
    if m not in PLACEMENT_MODES:
        raise ValueError(
            f"placement must be one of {PLACEMENT_MODES}, got {mode!r}. Refusing to fall back to "
            f"{DEFAULT_PLACEMENT_MODE!r}: a mode label that disagrees with the geometry it "
            "describes is a mis-measurement waiting to end up in a figure."
        )
    return m


def placement_mode_label(mode) -> str:
    """The persistent on-screen text for *mode*: ``"stage"`` or ``"compact"``."""
    return normalize_placement_mode(mode)


def _require_pixel_size(pixel_size_um: Optional[float]) -> float:
    """Validate the um->px conversion factor, refusing None/non-positive values.

    ``acq_um``: placement is in LEVEL-0 acquisition pixels, so an ``AcqPitchUm`` unwraps here
    and a ``DisplayPitchUm`` (a fused layer's own decimated pitch) is refused by name.
    """
    if pixel_size_um is None:
        raise ValueError(
            "pixel_size_um is required to place FOVs by stage coordinate, but the acquisition "
            "metadata has none. Without it, micrometres cannot be converted to pixels and every "
            "FOV would be drawn at the same spot. Add objective.pixel_size_um to acquisition.yaml."
        )
    p = acq_um(pixel_size_um)
    if not p > 0:
        raise ValueError(f"pixel_size_um must be > 0, got {pixel_size_um!r}.")
    return p


def fov_offsets_px(
    positions_um: Mapping[tuple, tuple],
    region: str,
    fovs: Iterable[int],
    pixel_size_um: Optional[float],
) -> dict[int, tuple[int, int]]:
    """Pixel offset of each FOV's top-left corner, relative to the region's own mosaic origin.

    Returns ``{fov: (row_px, col_px)}``, both >= 0, top-left-most FOV at ``(0, 0)``.
    """
    p = _require_pixel_size(pixel_size_um)
    fovs = list(fovs)
    if not fovs:
        raise ValueError(f"region {region!r}: no FOVs to place.")

    missing = [f for f in fovs if (region, f) not in positions_um]
    if missing:
        raise KeyError(
            f"region {region!r}: no stage position for FOV(s) {missing[:8]} "
            f"(have {sum(1 for k in positions_um if k[0] == region)} of {len(fovs)}). "
            "coordinates.csv and the image filenames disagree; refusing to draw a mosaic with holes."
        )

    xs = {f: float(positions_um[(region, f)][0]) for f in fovs}
    ys = {f: float(positions_um[(region, f)][1]) for f in fovs}
    x0, y0 = min(xs.values()), min(ys.values())

    out: dict[int, tuple[int, int]] = {}
    for f in fovs:
        col = (xs[f] - x0) / p
        row = (ys[f] - y0) / p * _Y_SIGN
        out[f] = (int(round(row)), int(round(col)))
    return out


def mosaic_extent_px(
    offsets: Mapping[int, tuple[int, int]],
    frame_shape: tuple[int, int],
) -> tuple[int, int]:
    """Full-resolution ``(height, width)`` of the mosaic that *offsets* + *frame_shape* describe."""
    if not offsets:
        raise ValueError("no offsets: nothing to size a mosaic from.")
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    h = max(r for r, _ in offsets.values()) + fh
    w = max(c for _, c in offsets.values()) + fw
    return int(h), int(w)


def cell_boxes(
    offsets: Mapping[int, tuple[int, int]],
    frame_shape: tuple[int, int],
    cell_px: int,
) -> dict[int, tuple[int, int, int, int]]:
    """Scale full-res offsets into a ``cell_px`` x ``cell_px`` thumbnail cell.

    Returns ``{fov: (top, left, height, width)}`` in cell pixels, aspect preserved and
    centred; every box is clamped to at least 1x1 px.
    """
    if cell_px < 1:
        raise ValueError(f"cell_px must be >= 1, got {cell_px}")
    mh, mw = mosaic_extent_px(offsets, frame_shape)
    fh, fw = int(frame_shape[0]), int(frame_shape[1])

    s = min(cell_px / mh, cell_px / mw)          # uniform scale; no aspect distortion
    off_y = (cell_px - mh * s) / 2.0             # centre the mosaic in the cell
    off_x = (cell_px - mw * s) / 2.0

    boxes: dict[int, tuple[int, int, int, int]] = {}
    for fov, (row, col) in offsets.items():
        top = int(round(off_y + row * s))
        left = int(round(off_x + col * s))
        h = max(1, int(round(fh * s)))
        w = max(1, int(round(fw * s)))
        top = max(0, min(top, cell_px - 1))       # keep the box inside the cell
        left = max(0, min(left, cell_px - 1))
        h = min(h, cell_px - top)
        w = min(w, cell_px - left)
        boxes[fov] = (top, left, h, w)
    return boxes


# Placement for a FUSED mosaic: the solved transform travels with the pixels.


@dataclass(frozen=True)
class Placement:
    """Where a fused mosaic's pixels are, and what put them there."""

    origin_um: tuple[float, float]
    """``(y_um, x_um)`` stage position of the mosaic's top-left corner."""

    pixel_size_um: float
    """Object-space pixel size used to convert micrometres to pixels. Always > 0."""

    z_step_um: "float | None"
    """Z step, carried so a 3-D consumer does not have to re-ask the reader. None when unknown."""

    shape: tuple[int, int]
    """``(height, width)`` of the fused mosaic in pixels."""

    tile_shape: tuple[int, int]
    """``(height, width)`` of one FOV."""

    fovs: tuple[int, ...]
    """The FOVs composing the mosaic, in the order the offsets/origins are given."""

    offsets_px: tuple[tuple[float, float], ...]
    """Per-FOV ``(dy, dx)`` correction the solve ADDED to the stage position."""

    origins_px: tuple[tuple[float, float], ...]
    """Per-FOV fractional ``(y, x)`` top-left within the mosaic, after correction."""

    reg_channel: "str | None"
    """NAME of the channel registration solved on; ``None`` if nothing was registered."""

    reg_t: "int | None"
    """Timepoint the transform was solved at; ``None`` if nothing was registered."""

    reg_z: "int | None" = None
    """RAW z-plane the transform was solved on; ``None`` if nothing was registered."""

    illumination_corrected: bool = False
    """Whether these pixels were flat-field corrected before registration and fusion."""

    def __post_init__(self):
        if not self.pixel_size_um > 0:
            raise ValueError(f"pixel_size_um must be > 0, got {self.pixel_size_um!r}")
        n = len(self.fovs)
        if len(self.offsets_px) != n or len(self.origins_px) != n:
            raise ValueError(
                f"fovs has {n} entries but offsets_px has {len(self.offsets_px)} and "
                f"origins_px has {len(self.origins_px)}; they describe the same tiles, so a "
                "disagreement means one of them is for a different mosaic."
            )

    @property
    def bbox_um(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` stage micrometres these pixels cover — X FIRST.

        Done once here because ``origin_um`` is ``(y, x)`` and ``bbox_um`` is ``(x, y, x, y)``:
        opposite axis orders, and an unflipped copy transposes a mosaic without raising.
        """
        y0, x0 = self.origin_um
        h, w = self.shape
        p = float(self.pixel_size_um)
        return (float(x0), float(y0), float(x0) + w * p, float(y0) + h * p)

    @property
    def registered(self) -> bool:
        """Whether a solve ran — read off ``reg_channel``, never off the offsets (which can be all-zero)."""
        return self.reg_channel is not None


class PlacedArray(np.ndarray):
    """An ``ndarray`` that carries its :class:`Placement`, so the geometry cannot be dropped."""

    def __new__(cls, array, placement: Placement):
        if not isinstance(placement, Placement):
            raise TypeError(
                f"placement must be a Placement, got {type(placement).__name__}. The whole "
                "point is that the geometry travelling with the pixels is a checked value, "
                "not another loose dict."
            )
        obj = np.asarray(array).view(cls)
        obj.placement = placement
        return obj

    def __array_finalize__(self, obj):
        # Views and slices are still THOSE pixels in that frame, so the placement follows.
        if obj is not None:
            self.placement = getattr(obj, "placement", None)
