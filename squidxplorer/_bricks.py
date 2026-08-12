"""Bricking: how a volume too big for ONE GL texture is cut into many that each fit.

WHY. napari renders 3D from ONE GL texture and refuses any axis over GL_MAX_3D_TEXTURE_SIZE --
measured 2048 on this Mac (``max_texture_sizes == (16384, 2048)``, read off the live canvas, and
Metal's own limit is 2048 too, so "use the GPU harder" does not lift it). A stitched region on the
10x set is 11538 x 9645, so until now a 3D request for anything bigger than one field was REFUSED
(``_napari3d.open_native_3d_volume`` raised) -- the user could draw an ROI that could not be seen.

THE FIX IS MANY TEXTURES, NOT A NEW RENDERER. napari gives one texture per Image layer, so tiling
the volume into blocks that each fit the limit and adding one ``add_image`` per block -- positioned
with ``translate`` and the same ``(dz, py, px)`` scale -- composes them back into one continuous
volume inside the existing napari canvas. No vispy canvas, no second renderer, nothing to re-sync.

THREE PROPERTIES THIS MODULE OWNS, and they are in tension; resolving them is the whole design.

1. FULL RESOLUTION, never a silent downsample. See ``uniform_step``: the stride is chosen so there
   is at least one texel per SCREEN pixel. The texture limit never forces a stride, because a brick
   is <= the limit at stride 1 by construction. So whenever the display could actually resolve
   native detail, the stride IS 1. Zoomed out, a coarser stride is not a downsample of what you can
   see -- it is declining to upload voxels smaller than a pixel -- and it converges to 1 as you zoom.

2. MEMORY-SAFE. This is the load-bearing consequence of (1), and it is why bricking is bounded
   rather than merely deferred. At screen-adequate stride the resident voxel count is
   ~ (screen pixels) x nz REGARDLESS OF ROI SIZE: zoom out and the stride grows exactly as fast as
   the brick count does. A 11538 x 9645 x 10 region is 2.2 GB/channel materialised whole; the
   visible set at any zoom is tens of MB. ``plan_budget`` states that bound and enforces it.

3. FAST. Uniform stride across bricks (not per-brick LOD) is deliberate: neighbouring bricks at
   different strides crack at the join, and the slab is only nz deep so per-brick LOD buys nothing.
   One stride for the whole volume means no crack can exist, and the cull below keeps the loader's
   queue to what the camera can actually see.

Pure geometry and policy: no napari, no Qt, no numpy arrays read. Everything here is a decision
about WHICH voxels, so it is testable headless -- which matters because the GL limit itself cannot
be probed without a GL context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

#: Default brick edge in level-0 voxels. NOT the texture limit, and NOT this machine's number.
#:
#: Frame time is flat in brick count (measured on this Mac: 4 bricks 4.93 ms vs 16 bricks 5.20 ms
#: median over 20 rotated redraws), so nothing is bought by making bricks as large as the texture
#: allows -- while a smaller brick is a smaller EVICTION GRANULE and a smaller unit of work for the
#: loader, which is what keeps first-pixels early. 1024 x 1024 x nz at uint16 and nz=10 is 21 MB: a
#: granule the budget can actually steer. A 2048 brick is 84 MB and four of them overshoot a laptop
#: budget in one step.
#:
#: PORTABILITY. This is a GRANULARITY choice, not a texture assumption, which is why it is safe on
#: hardware that reports something other than the 2048 measured here. ``plan`` clamps it DOWN to
#: whatever the live ``GL_MAX_3D_TEXTURE_SIZE`` says, so a limit smaller than this (some integrated
#: GPUs) yields more, smaller bricks rather than a brick the driver refuses. A limit far larger
#: (desktop NVIDIA commonly reports 16384) does not make bricks bigger, and should not: the eviction
#: and latency arguments above are about RAM and time, which a big GPU does not change. What a large
#: limit DOES change is how often bricking happens at all -- ``fits_single_texture`` sends far more
#: ROIs down the one-layer fast path there, which is the intended portable behaviour.
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
        """The (z, y, x) shape this brick occupies once strided by *step*, matching ``arr[::step]``
        semantics exactly -- ``ceil(n / step)``, which is what numpy yields."""
        s = max(1, int(step))
        return (int(nz), _ceil_div(self.height, s), _ceil_div(self.width, s))

    def nbytes(self, nz: int, itemsize: int, step: int = 1) -> int:
        z, h, w = self.sampled_shape(nz, step)
        return int(z) * int(h) * int(w) * int(itemsize)

    def translate_um(self, origin_um: Sequence[float], py: float, px: float) -> tuple:
        """This brick's (z, y, x) micrometre translate for ``add_image``.

        napari puts sample (0, 0, 0) at ``translate`` and sample i at ``translate + i * scale``, so
        with ``scale = (dz, py * step, px * step)`` the strided sample k lands on level-0 voxel
        ``r0 + k * step`` -- the brick's own origin is independent of the stride, and that is why
        changing stride never moves a brick. z is not tiled, so it carries the volume's z origin."""
        oz, oy, ox = (float(v) for v in origin_um)
        return (oz, oy + self.r0 * float(py), ox + self.c0 * float(px))

    def bounds_um(self, origin_yx_um: Sequence[float], py: float, px: float) -> tuple:
        """(y0, x0, y1, x1) world micrometres this brick covers -- what the cull tests.

        Takes the (y, x) origin, not the (z, y, x) one ``translate_um`` takes: culling is a purely
        2-D question here because z is never tiled, and asking for a z the caller would have to
        invent is how a coordinate bug gets in."""
        oy, ox = (float(v) for v in origin_yx_um)
        return (oy + self.r0 * float(py), ox + self.c0 * float(px),
                oy + self.r1 * float(py), ox + self.c1 * float(px))


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


#: Voxels of OVERLAP between neighbouring bricks. This is the last of the seam, and it is free.
#:
#: A GL texture CLAMPS at its edge, so with linear interpolation the samples in the outermost half-
#: texel of a brick interpolate against the brick's own edge texel instead of against the neighbour
#: that is really there. Measured on the 10x set, 2048 px ROI as 1 texture vs 16 bricks, identical
#: camera: 397 of 1,064,828 pixels (0.037%) differed by more than 2/255, and amplifying the
#: difference 16x showed them lying exactly on the brick grid. Small, but structured -- and
#: structured error is what the eye finds.
#:
#: One voxel of overlap gives each brick its neighbour's first texel, so the interpolation is
#: against real data and the join disappears. The reason this is FREE rather than a double-count is
#: the max blend equation (``_napari3d.pin_max_compositing``): the shared voxel carries the same
#: value in both bricks, and max(v, v) == v. Under ADDITIVE the same overlap would paint that voxel
#: twice and put a bright line exactly where the seam used to be -- so the halo and the blend mode
#: are one decision, not two.
BRICK_HALO = 1


def plan(height: int, width: int, *, limit: int, edge: int = DEFAULT_BRICK_EDGE,
         halo: int = BRICK_HALO) -> tuple:
    """Tile a ``height x width`` level-0 plane into bricks that each fit ONE GL texture.

    *limit* is the live ``GL_MAX_3D_TEXTURE_SIZE`` (read off the canvas, never assumed -- see
    ``_napari_pane._live_max_3d_texture``). The edge is clamped to it, so a machine reporting a
    smaller limit than our default simply gets more, smaller bricks rather than a brick napari
    would refuse. Bricks are emitted row-major, which is the order the loader prefers when nothing
    else ranks them.

    Bricks OVERLAP by *halo* voxels on their far side (see ``BRICK_HALO``); the stride between brick
    ORIGINS is still the edge, so ``translate`` is unaffected and the tiling still covers the plane
    exactly once at every origin. The halo is included in the edge budget, so a brick never exceeds
    *limit* even with it.
    """
    h, w = int(height), int(width)
    if h <= 0 or w <= 0:
        return ()
    hl = max(0, int(halo))
    e = max(1, min(int(edge), int(limit)))
    # A plane that fits ONE brick has no neighbours, so it gets no halo -- and must not, or the
    # single-texture fast path would be split in two for nothing. Measured: subtracting the halo
    # from the edge unconditionally turned a 2048 px ROI on a 2048 px limit into 4 bricks.
    if h <= e and w <= e:
        return (Brick(iy=0, ix=0, r0=0, r1=h, c0=0, c1=w),)
    # THE HALO EXTENDS THE END, NEVER THE ORIGIN STRIDE. Brick origins must stay on multiples of a
    # power-of-two edge or ``uniform_step`` loses the shared sampling lattice it depends on (see
    # its docstring). Shrinking the edge to make room instead -- e -> e/2 -- keeps it a power of
    # two. Measured, when the halo was taken out of the edge instead: origins fell on 511, 1022,
    # ... and the stride-4 render disagreed with native on 3.8% of pixels, against 0.007% at
    # stride 1 where the misalignment cannot bite.
    while e + hl > int(limit) and e > 1:
        e //= 2
    out: list = []
    for iy, r0 in enumerate(range(0, h, e)):
        for ix, c0 in enumerate(range(0, w, e)):
            out.append(Brick(iy=iy, ix=ix, r0=r0, r1=min(r0 + e + hl, h),
                             c0=c0, c1=min(c0 + e + hl, w)))
    return tuple(out)


def fits_single_texture(height: int, width: int, nz: int, limit: int) -> bool:
    """Whether this volume still fits the ONE-texture fast path.

    Keeping this path is not an optimisation, it is the contract: ``_napari3d``'s single-layer
    recipe is gallery-view's, it works, and a volume that does not need bricking must not be
    bricked -- one layer means one contrast widget and no compositing question at all."""
    return max(int(height), int(width), int(nz)) <= int(limit)


def uniform_step(um_per_screen_px: float, px_um: float, *, max_step: int = 64) -> int:
    """The ONE stride every brick uses, from the camera's current scale.

    ONE TEXEL PER SCREEN PIXEL is the rule. *um_per_screen_px* is how much stage the camera puts
    under one screen pixel (napari's ``camera.zoom`` is screen px per world unit, so this is
    ``1 / zoom`` in micrometre world units); *px_um* is the acquisition's voxel pitch. Their ratio
    is how many voxels are hiding behind one pixel, and uploading more than one of those is paying
    for detail the display cannot show.

    ``floor``, and never below 1: rounding DOWN means we err toward more resolution than the screen
    strictly needs, so the visible image is never softer than the source. Step 1 is native, and step
    1 is what any zoom at or past 1:1 yields -- which is the sense in which this "converges to full
    res" rather than settling at coarse. *max_step* is a floor on how coarse this may ever go, so a
    pathological camera cannot ask for a 1-voxel volume.

    The texture limit deliberately does NOT appear here. A brick is <= the limit at step 1 by
    construction (``plan``), so the GPU never forces a stride; only the screen does. That is the
    difference between this and the downsample napari does on its own.

    POWERS OF TWO ONLY, and this is a correctness constraint rather than a tidiness one. Each brick
    strides from its OWN origin (``arr[::step]`` starting at ``r0``), so bricks share one global
    sampling lattice only when ``step`` divides the brick edge. At step 3 with a 1024 edge the
    brick starting at column 1024 would sample 1024, 1027, ... where the global lattice wants
    1026 -- every join misaligned by up to ``step - 1`` voxels, which is a seam that appears only
    when zoomed out and would have been maddening to trace. The edge is a power of two, so
    restricting the stride to powers of two makes the lattice exact at every join, for free. It
    also stops the stride thrashing between adjacent values on a slow zoom.
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

    A brick that is not visible must not be resident -- that is the memory rule, and it is also the
    latency rule, because a loader that fetches offscreen bricks is spending the queue the visible
    ones are waiting in. *margin_um* pads the view so a small pan does not immediately stall on a
    brick that was one pixel outside.

    Ordering is centre-out because the user is looking at the middle of their own ROI: with a
    bounded resident set the LAST brick to arrive is the one most likely to be evicted, and it
    should be a corner rather than the thing under the cursor. ``view_um=None`` means "no camera
    yet" and keeps every brick, in plan order -- the honest answer before the first draw.
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

    THIS IS THE SANCTIONED FALLBACK, and it is a constraint on DRAWING rather than a rendering
    subsystem. Julio: "If (a) doesn't work well and with responsiveness then (b) will be very easy
    to implement for the sake of practicality." The guarantee it buys is the one his complaint asks
    for -- **anything you can draw, you can render, at full native resolution, from one texture** --
    and it buys it by construction rather than by machinery: no level-of-detail, no culling, no
    eviction, nothing to be janky.

    The anchor is the corner the drag STARTED from, so a box that hits the ceiling stops growing
    instead of jumping; that is what makes the limit felt while drawing rather than reported after.

    Returns ``(bbox, clamped)``. *limit* is the LIVE queried texture size, so this ceiling rises on
    better hardware -- 2048 px = 1540 um here, 16384 px = 12321 um on a desktop NVIDIA, 512x the
    volume. Nothing here is a constant.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox_um)
    span = float(limit) * float(px_um)
    if span <= 0:
        return ((x0, y0, x1, y1), False)
    nx1 = x0 + min(x1 - x0, span) if x1 >= x0 else x0 - min(x0 - x1, span)
    ny1 = y0 + min(y1 - y0, span) if y1 >= y0 else y0 - min(y0 - y1, span)
    clamped = (abs(nx1 - x1) > 1e-9) or (abs(ny1 - y1) > 1e-9)
    return ((x0, y0, nx1, ny1), clamped)


def ceiling_line(limit: int, px_um: float, *, measured: bool) -> str:
    """What this GPU can trace, in the user's units, said out loud.

    Julio: "the focus will be in understanding how to leverage more powerful GPUs to be able to
    trace larger ROIs". That only becomes actionable if the ceiling is VISIBLE, so this turns the
    live ``GL_MAX_3D_TEXTURE_SIZE`` into the two numbers a user actually reasons about: how big an
    ROI still renders from a single texture, and how big one renders once bricked.

    2048 is the APPLE figure and nothing here assumes it. Desktop NVIDIA commonly reports 16384 --
    8x per axis, 512x the volume -- and on such a machine this line says so, which is the whole
    point: the same build gets a bigger ceiling on better hardware and tells you it did. *measured*
    distinguishes a value read off the live canvas from the fallback, because "your GPU says 16384"
    and "we could not ask and are assuming 2048" are different facts.
    """
    where = "measured on this GPU" if measured else "ASSUMED — the canvas could not be asked"
    one = float(limit) * float(px_um)
    return (f"3D ceiling: GL_MAX_3D_TEXTURE_SIZE = {int(limit)} px ({where}). One texture holds "
            f"{int(limit)}x{int(limit)} px = {one:.0f}x{one:.0f} um at native {px_um:g} um/px; "
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

    Two ways to spend less: show fewer bricks, or show every brick at a coarser stride. Only the
    second one is honest. Dropping bricks leaves HOLES in the volume -- the user sees black where
    their data is and has no way to know it is a budget artefact rather than empty tissue -- whereas
    a coarser stride is uniform, visible as softness, and named in the log. So this doubles the
    stride until the visible set fits, and reports the stride it landed on.

    Each doubling divides resident bytes by 4 (y and x both), so this terminates fast: three
    doublings is a 64x reduction. ``dropped`` stays 0 unless the stride hits its ceiling and the set
    STILL does not fit, which is the only case where a brick is withheld -- and the caller is
    expected to say so out loud rather than draw a quiet hole.
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
