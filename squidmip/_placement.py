"""Coordinate placement: stage micrometres -> pixel offsets (IMA-187).

The pure, GUI-free half of the multi-FOV mosaic. Everything here is arithmetic on
``fov_positions_um`` (from the reader) plus one scalar, ``pixel_size_um`` — no Qt, no I/O, and
no inspection of image CONTENT — so placement correctness is asserted numerically in tests
rather than eyeballed on a rendered plate.

(numpy is imported, for :class:`PlacedArray`'s ndarray subclass only. The property this module
actually depends on is that nothing here reads pixels or touches a GUI; an earlier version of
this note said "no numpy dependency" flat, which stopped being true the moment Placement
needed to travel attached to an array.)

Why that matters: placement bugs do not raise. They draw a plausible-but-wrong picture.
The three that actually happen::

    scale error      every offset off by a constant factor -> mosaic uniformly too tight/loose
    Y-axis flip      stage y mapped to decreasing row      -> mosaic mirrored vertically
    wrong origin     origin taken plate-wide, not per-region -> every tile shifted by a constant

A test that counts tiles catches none of them; a test that compares integer pixel offsets
catches all three. Hence this module.

Units: input positions are stage MICROMETRES (``metadata["fov_positions_um"]``; the reader
converts from the mm that coordinates.csv records). This module used to carry the ``* 1000``
mm->µm conversion itself, which meant an unsuffixed millimetre value travelled through the
metadata dict — the exact silent-1000x hazard the ``_um`` naming rule exists to prevent.

Geometry::

    stage (um), origin = per-region min          image (px), origin = mosaic top-left
    ┌─────────────────────────►  +x              ┌─────────────────────────►  +col
    │   (x0,y0)   (x1,y0)                        │   fov0      fov1
    │      ▪         ▪                           │    ▪         ▪
    │   (x0,y1)   (x1,y1)                        │   fov6      fov7
    │      ▪         ▪                           │    ▪         ▪
    ▼  +y                                        ▼  +row

    col_px = (x_um - min_x_um) / pixel_size_um
    row_px = (y_um - min_y_um) / pixel_size_um * _Y_SIGN

``_Y_SIGN`` is +1: Squid rasters +x/+y and image rows increase downward, so increasing stage
y maps to increasing row. It is a named module constant rather than an inline sign so the
convention is greppable and a future stage with the opposite handedness is a one-line change
with an obvious test to flip.

The origin is **per region**, not plate-wide: each well's mosaic is laid out in its own local
frame, and the well's position on the plate comes from the row/column grid the plate view
already draws. Mixing the two coordinate systems (stage-absolute FOVs inside a grid-placed
cell) is exactly the "wrong origin" bug above.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable, Mapping, Optional

import numpy as np

# Stage +y maps to image +row (downward). See the module docstring.
_Y_SIGN = 1


def _require_pixel_size(pixel_size_um: Optional[float]) -> float:
    """Validate the mm->px conversion factor, failing loud on the values that silently ruin a mosaic.

    ``pixel_size_um`` is ``Optional`` throughout the metadata layer (``_acquisition.py`` returns
    ``objective.get("pixel_size_um")``, which is ``None`` on a dataset without it). A ``None`` here
    would make every offset ``None`` or zero and collapse the mosaic into a single stacked pile —
    with no error. Refuse instead.
    """
    if pixel_size_um is None:
        raise ValueError(
            "pixel_size_um is required to place FOVs by stage coordinate, but the acquisition "
            "metadata has none. Without it, micrometres cannot be converted to pixels and every "
            "FOV would be drawn at the same spot. Add objective.pixel_size_um to acquisition.yaml."
        )
    p = float(pixel_size_um)
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

    Parameters
    ----------
    positions_um:
        ``{(region, fov): (x_um, y_um)}`` — ``reader.metadata["fov_positions_um"]``, stage
        MICROMETRES. Passing millimetres here silently shrinks the mosaic 1000x.
    region:
        The well / region being laid out.
    fovs:
        The FOVs of *region* to place (typically ``metadata["fovs_per_region"][region]``).
    pixel_size_um:
        Object-space pixel size. Required; see :func:`_require_pixel_size`.

    Returns
    -------
    dict[int, tuple[int, int]]
        ``{fov: (row_px, col_px)}``, both >= 0, with the top-left-most FOV at ``(0, 0)``.

    Raises
    ------
    KeyError
        If a requested FOV has no recorded position — a silent skip would leave a hole in the
        mosaic that looks like a failed acquisition.
    ValueError
        If *fovs* is empty, or *pixel_size_um* is unusable.
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
    """Full-resolution ``(height, width)`` of the mosaic that *offsets* + *frame_shape* describe.

    The extent is the bounding box of every placed frame, so it accounts for the real overlap
    between neighbours rather than assuming a dense grid.
    """
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

    Returns ``{fov: (top, left, height, width)}`` in cell pixels. This is what the plate view
    consumes: the mosaic is composited at THUMBNAIL scale, never at full resolution, so a
    36-FOV well costs ~``cell_px``^2 rather than the ~1 GB a full-res composite would.

    The mosaic is fitted to the cell preserving aspect ratio and centred, so a non-square
    mosaic (a 6x4 acquisition, a freeform strip) is not stretched. Every box is clamped to at
    least 1x1 px, so a FOV can never silently vanish at small cell sizes.
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


# ══════════════════════════════════════════════════════════════════════════════════════════
# Placement: ONE source of truth for "where are these pixels" (Defect 3).
#
# Everything above answers that question for the PLATE VIEW, from stage coordinates alone.
# What follows answers it for a FUSED MOSAIC, and carries the solved transform that produced
# it.
#
# The problem this replaces. `stitch_region` solved its transform once, at t=0, and then
# discarded it unless the caller happened to pass a `geometry` out-dict — an OPTIONAL side
# channel. So by default the answer to "where did this mosaic land, and what solved it"
# did not exist at all. Meanwhile the viewer re-derived placement independently from a stage
# bounding box. Two mechanisms, two answers, and nothing that could ever notice they had
# diverged.
#
# A value object fixes the "optional" half: placement is now produced unconditionally. The
# `PlacedArray` below fixes the "side channel" half by making it ride WITH the pixels, so a
# consumer cannot receive the array and miss the geometry.
#
# `reg_channel` / `reg_t` are the point, not decoration. They record WHICH channel and
# timepoint solved the transform, so the data carries its own provenance. That is the other
# half of the registration fix in `_stitch`: once registration is guaranteed to run on the
# requested channel, the output should be able to SAY so rather than leaving it to be
# inferred from the arguments the caller believes it passed.
# ══════════════════════════════════════════════════════════════════════════════════════════



@dataclass(frozen=True)
class Placement:
    """Where a fused mosaic's pixels are, and what put them there.

    A frozen dataclass of plain tuples rather than a pydantic model or a numpy-holding record:
    this is a small immutable value that travels attached to an array, it needs to be cheap to
    construct once per region, and tuples make it comparable and safely shareable across the
    worker threads that produce it. (``Acquisition`` is pydantic because it validates messy
    external input; this is built internally from values already checked.)
    """

    origin_um: tuple[float, float]
    """``(y_um, x_um)`` stage position of the mosaic's top-left corner — the frame origin."""

    pixel_size_um: float
    """Object-space pixel size used to convert micrometres to pixels. Always > 0."""

    z_step_um: "float | None"
    """Z step, carried so a consumer rendering this in 3-D does not have to re-ask the
    reader (and thereby become a second source of truth). ``None`` when unknown."""

    shape: tuple[int, int]
    """``(height, width)`` of the fused mosaic in pixels."""

    tile_shape: tuple[int, int]
    """``(height, width)`` of one FOV."""

    fovs: tuple[int, ...]
    """The FOVs composing the mosaic, in the order the offsets/origins are given."""

    offsets_px: tuple[tuple[float, float], ...]
    """Per-FOV ``(dy, dx)`` correction the solve ADDED to the stage position. All-zero for
    pure coordinate placement — and legitimately all-zero for a real solve that found no
    usable overlap, which is why ``registered`` is not inferred from these."""

    origins_px: tuple[tuple[float, float], ...]
    """Per-FOV fractional ``(y, x)`` top-left within the mosaic, after correction."""

    reg_channel: "str | None"
    """NAME of the channel registration solved on; ``None`` if nothing was registered.

    A name, never an index: an index re-breaks the moment the channel selection or the
    reader's axis order changes, which is exactly the bug this data is meant to make
    impossible to hide."""

    reg_t: "int | None"
    """Timepoint the transform was solved at; ``None`` if nothing was registered."""

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
    def registered(self) -> bool:
        """Whether a solve ran — read off ``reg_channel``, NOT off the offsets.

        A genuine solve can return all-zero offsets (no overlap, or every pair rejected) and
        that is still a registered placement. Inferring this from the numbers would silently
        relabel those as coordinate placement.
        """
        return self.reg_channel is not None


class PlacedArray(np.ndarray):
    """An ``ndarray`` that carries its :class:`Placement`.

    A numpy subclass rather than a ``(array, placement)`` tuple, and that is a deliberate
    trade. A tuple return would be more explicit, but it changes the arity of the region
    operator contract, and ``stitch_plate`` yields these straight into the viewer's worker and
    the OME-Zarr writer. This repo has already shipped a test that silently died by
    destructuring a 2-tuple after the function grew a third return value, so widening that
    contract while other work is in flight in the viewer is the expensive way to be right.

    A ``PlacedArray`` IS an ndarray, so every existing consumer is untouched, and the geometry
    can no longer be dropped on the floor. Consumers that want it ask for ``.placement``; a
    consumer that asks a plain ndarray gets an ``AttributeError``, which is loud, rather than
    a ``None`` that would be silent.
    """

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
