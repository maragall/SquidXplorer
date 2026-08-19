"""Bricking: how a volume too big for ONE GL texture is cut into many that each fit.

Pure geometry and policy — no napari, no Qt, no numpy arrays read — so it is testable headless.
One uniform stride across bricks (>= 1 texel per screen pixel), bounded resident bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from squidxplorer._conventions import acq_um

#: Default brick edge in level-0 voxels: a granularity choice (eviction granule, loader unit of
#: work), not a texture assumption — ``plan`` clamps it down to the live GL limit.
DEFAULT_BRICK_EDGE = 1024


@dataclass(frozen=True)
class Brick:
    """One block of a volume, as a half-open LEVEL-0 voxel window. Immutable: a brick is an
    identity (the loader caches on it), not a buffer."""

    iy: int
    ix: int
    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0

    @property
    def width(self) -> int:
        return self.c1 - self.c0

    @property
    def key(self) -> tuple:
        return (self.iy, self.ix, self.r0, self.r1, self.c0, self.c1)

    def sampled_shape(self, nz: int, step: int = 1) -> tuple:
        """The (z, y, x) shape this brick occupies once strided by *step* (``ceil(n / step)``)."""
        s = max(1, int(step))
        return (int(nz), _ceil_div(self.height, s), _ceil_div(self.width, s))

    def nbytes(self, nz: int, itemsize: int, step: int = 1) -> int:
        z, h, w = self.sampled_shape(nz, step)
        return int(z) * int(h) * int(w) * int(itemsize)

    def translate_um(self, origin_um: Sequence[float], py: float, px: float) -> tuple:
        """This brick's (z, y, x) micrometre translate for ``add_image``.

        The brick's origin is independent of the stride; z is not tiled, so it carries the
        volume's z origin.
        """
        oz, oy, ox = (float(v) for v in origin_um)
        return (oz, oy + self.r0 * float(py), ox + self.c0 * float(px))

    def bounds_um(self, origin_yx_um: Sequence[float], py: float, px: float) -> tuple:
        """(y0, x0, y1, x1) world micrometres this brick covers — what the cull tests."""
        oy, ox = (float(v) for v in origin_yx_um)
        return (oy + self.r0 * float(py), ox + self.c0 * float(px),
                oy + self.r1 * float(py), ox + self.c1 * float(px))


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


#: Voxels of OVERLAP between neighbouring bricks: gives each brick its neighbour's first texel so
#: edge interpolation is against real data. Free under the max blend (max(v, v) == v); under
#: additive the same overlap would double-paint the seam.
BRICK_HALO = 1


def plan(height: int, width: int, *, limit: int, edge: int = DEFAULT_BRICK_EDGE,
         halo: int = BRICK_HALO) -> tuple:
    """Tile a ``height x width`` level-0 plane into bricks that each fit ONE GL texture.

    *limit* is the live ``GL_MAX_3D_TEXTURE_SIZE``. Bricks overlap by *halo* on their far side;
    origins stay on multiples of the edge.
    """
    h, w = int(height), int(width)
    if h <= 0 or w <= 0:
        return ()
    hl = max(0, int(halo))
    e = max(1, min(int(edge), int(limit)))
    # A plane that fits ONE brick has no neighbours, so it gets no halo — otherwise the
    # single-texture fast path would be split in two for nothing.
    if h <= e and w <= e:
        return (Brick(iy=0, ix=0, r0=0, r1=h, c0=0, c1=w),)
    # The halo extends the END, never the origin stride: origins must stay on multiples of a
    # power-of-two edge or ``uniform_step`` loses the shared sampling lattice.
    while e + hl > int(limit) and e > 1:
        e //= 2
    out: list = []
    for iy, r0 in enumerate(range(0, h, e)):
        for ix, c0 in enumerate(range(0, w, e)):
            out.append(Brick(iy=iy, ix=ix, r0=r0, r1=min(r0 + e + hl, h),
                             c0=c0, c1=min(c0 + e + hl, w)))
    return tuple(out)


def fits_single_texture(height: int, width: int, nz: int, limit: int) -> bool:
    """Whether this volume still fits the ONE-texture fast path."""
    return max(int(height), int(width), int(nz)) <= int(limit)


def uniform_step(um_per_screen_px: float, px_um: float, *, max_step: int = 64) -> int:
    """The ONE stride every brick uses, from the camera's current scale.

    One texel per screen pixel, floored (never below 1), powers of two only so bricks share one
    global sampling lattice at every join.
    """
    try:
        ratio = float(um_per_screen_px) / float(px_um)
    except (TypeError, ZeroDivisionError):
        return 1
    if not math.isfinite(ratio) or ratio <= 1.0:
        return 1
    step = 1
    while step * 2 <= min(int(max_step), math.floor(ratio)):
        step *= 2
    return int(step)


def intersects(bounds_um: Sequence[float], view_um: Sequence[float]) -> bool:
    """Does a brick's (y0, x0, y1, x1) overlap the camera's (y0, x0, y1, x1)? Touching counts as
    out: a brick sharing only an edge contributes no pixel."""
    by0, bx0, by1, bx1 = (float(v) for v in bounds_um)
    vy0, vx0, vy1, vx1 = (float(v) for v in view_um)
    return (bx0 < vx1) and (bx1 > vx0) and (by0 < vy1) and (by1 > vy0)


def cull(bricks: Iterable[Brick], *, origin_um: Sequence[float], py: float, px: float,
         view_um: Optional[Sequence[float]], margin_um: float = 0.0) -> tuple:
    """The bricks the camera can actually see, nearest the view centre FIRST.

    ``view_um=None`` means "no camera yet" and keeps every brick, in plan order.
    """
    items = list(bricks)
    if view_um is None:
        return tuple(items)
    vy0, vx0, vy1, vx1 = (float(v) for v in view_um)
    m = float(margin_um)
    padded = (vy0 - m, vx0 - m, vy1 + m, vx1 + m)
    cy, cx = (vy0 + vy1) / 2.0, (vx0 + vx1) / 2.0
    keep = [b for b in items if intersects(b.bounds_um(origin_um, py, px), padded)]

    def _d(b: Brick) -> float:
        by0, bx0, by1, bx1 = b.bounds_um(origin_um, py, px)
        return ((by0 + by1) / 2.0 - cy) ** 2 + ((bx0 + bx1) / 2.0 - cx) ** 2

    keep.sort(key=_d)
    return tuple(keep)


def clamp_bbox_um(bbox_um: Sequence[float], px_um: float, limit: int) -> tuple:
    """Shrink an ROI box to what ONE GL texture can render, keeping its top-left anchor.

    The anchor is the corner the drag started from, so a box that hits the ceiling stops growing.
    Returns ``(bbox, clamped)``; *limit* is the live queried texture size. *px_um* is the
    ACQUISITION pitch (:class:`~squidxplorer._conventions.AcqPitchUm` or a bare float; a
    ``DisplayPitchUm`` is refused by name): the clamp counts level-0 pixels because a drawn
    ROI's 3D reads whole FOV planes off the reader, and at the displayed pitch the same sentence
    would promise one texture over a 16-brick, 4x read.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox_um)
    span = float(limit) * acq_um(px_um)
    if span <= 0:
        return ((x0, y0, x1, y1), False)
    nx1 = x0 + min(x1 - x0, span) if x1 >= x0 else x0 - min(x0 - x1, span)
    ny1 = y0 + min(y1 - y0, span) if y1 >= y0 else y0 - min(y0 - y1, span)
    clamped = (abs(nx1 - x1) > 1e-9) or (abs(ny1 - y1) > 1e-9)
    return ((x0, y0, nx1, ny1), clamped)


def ceiling_line(limit: int, px_um: float, *, measured: bool) -> str:
    """What this GPU can trace, in the user's units, said out loud.

    *px_um* is the ACQUISITION pitch (``AcqPitchUm`` or float; ``DisplayPitchUm`` refused by
    name), the same unit :func:`clamp_bbox_um` counts in, or the two sentences disagree.
    """
    where = "measured on this GPU" if measured else "ASSUMED — the canvas could not be asked"
    p = acq_um(px_um)
    one = float(limit) * p
    return (f"3D ceiling: GL_MAX_3D_TEXTURE_SIZE = {int(limit)} px ({where}). One texture holds "
            f"{int(limit)}x{int(limit)} px = {one:.0f}x{one:.0f} um at native {p:g} um/px; "
            f"larger ROIs are BRICKED into as many textures as they need, so the renderable size "
            f"is bounded by memory rather than by this number. A GPU reporting a larger limit "
            f"raises the single-texture size and lowers the brick count.")


@dataclass(frozen=True)
class Budget:
    """What a bricked render is allowed to hold resident, and what it actually costs."""

    step: int
    bricks: tuple
    bytes_resident: int
    bytes_limit: int
    dropped: int = 0

    @property
    def within(self) -> bool:
        return self.bytes_resident <= self.bytes_limit


def plan_budget(bricks: Sequence[Brick], *, nz: int, itemsize: int, step: int,
                bytes_limit: int, n_channels: int = 1) -> Budget:
    """Fit the visible bricks under *bytes_limit*, COARSENING the stride rather than truncating.

    Dropping bricks would leave holes that read as empty tissue; a coarser stride is uniform and
    named. ``dropped`` stays 0 unless the stride ceiling is hit and the set still does not fit.
    """
    s = max(1, int(step))
    limit = max(1, int(bytes_limit))
    nchan = max(1, int(n_channels))
    items = list(bricks)

    def _cost(stride: int) -> int:
        return sum(b.nbytes(nz, itemsize, stride) for b in items) * nchan

    while _cost(s) > limit and s < 64:
        s *= 2
    resident = _cost(s)
    dropped = 0
    if resident > limit:
        # Stride is exhausted. Trim from the BACK, which ``cull`` ordered farthest-from-centre.
        keep: list = []
        running = 0
        per = nchan
        for b in items:
            cost = b.nbytes(nz, itemsize, s) * per
            if running + cost > limit and keep:
                dropped += 1
                continue
            keep.append(b)
            running += cost
        items, resident = keep, running
    return Budget(step=s, bricks=tuple(items), bytes_resident=int(resident),
                  bytes_limit=int(limit), dropped=int(dropped))
