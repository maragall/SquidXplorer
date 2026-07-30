"""The world-space enumerator: which tile goes where on the plate montage, in stage micrometres.

Task 7 of the GUI backlog plan (2026-07-29), the DISPLAY half of deep zoom. The cost half landed
in ``68d769b`` (``_platecache`` + ``_tilesource.CompositePlateSource``): a fit-to-plate tile went
from 8.448 s to 1.44 s first and 0.14 ms steady, and a 1536-well reopen from 15.2 s to 0.075 s.
Nothing here is about cost. What was left undone, and why, in the words of the agent that built
the cache:

    The world-space enumerator that would DISPLAY coarse rungs on the montage. ``_visible_fov_tiles``
    is keyed ``(region, fov)`` and a plate rung is keyed by a world grid cell; the montage's uniform
    cell grid and the ladder's stage micrometres agree only inside a cell.

THE MISMATCH, EXACTLY
---------------------
``PlateOverview`` draws a UNIFORM grid: every well gets a ``cd`` x ``cd`` square, whatever its real
stage position, and its mosaic is letterboxed into that square (``_placement.cell_boxes``). The
tile ladder is in STAGE MICROMETRES: wells sit where the stage put them, millimetres of empty
carrier between them, and a plate-rung tile is a fixed square of world that routinely spans several
wells and the gaps between them.

So there is no single affine from world µm to widget px. There is one PER REGION, and that is not a
limitation to work around, it is the geometry: inside one cell the montage is a uniform scaling of
that region's own bounding box, so the two agree there exactly. This module therefore places a
plate-rung tile once per region it overlaps, CLIPPED to that region, and draws the matching
sub-rectangle of the tile. A tile spanning four wells is four draws with four source rectangles,
and the empty carrier between them is never drawn at all -- which is correct, because in the
montage that space does not exist.

WHY ``Extent`` IS THE CURRENCY, AND WHERE IT IS NOT
---------------------------------------------------
:class:`squidmip._address.Extent` is "WHICH SLAB a result covers, in the acquisition's own
coordinates", and ``bbox_um`` is a rectangle in stage micrometres scoped to one ``region_id``.
That is precisely what a placed tile is here: not a ``(region, fov)`` pair (a plate tile is not one
field and may be a hundred), not a bare world box (the same box means two different widget
rectangles in two different cells), but A REGION PLUS A RECTANGLE. Two placements are the same
placement when they cover the same slab of the same region, and ``Extent`` already decides that
without a tolerance.

Where it is NOT the currency: fetching. A tile's fetch identity must name the RUNG it came from,
because the same slab exists at every rung at a different resolution, and that is
``_tiling.TileDescriptor``. So :class:`PlacedTile` carries both -- ``extent`` is what it covers,
``level``/``key`` are where the pixels come from -- and neither is derivable from the other.

THE LETTERBOX IS RECOVERABLE FROM THE ASPECT RATIO ALONE
--------------------------------------------------------
``cell_boxes`` fits a region's mosaic into its cell with ``s = min(cell/mh, cell/mw)`` and centres
it. The mosaic's full-res extent ``(mh, mw)`` is the region's world bounding box divided by the
pixel size, so the same letterbox falls out of the world box's ASPECT RATIO with no call into
``_placement`` and no second table to keep in step. ``_viewer._cell_source`` already relies on this
identity for the montage blit ("Since the cell rect and the letterbox come from the SAME aspect
ratio, the inner box is recoverable from the rect alone"); :func:`fit_preserving_aspect` is that
same rule written once, in world units, so the overlay lands ON the thumbnail rather than near it.

THE RUNG CLAMP, WHICH IS THE WHOLE "NO VISIBLE JUMP" MECHANISM
--------------------------------------------------------------
``Geometry.pick_level`` picks the coarsest rung still finer than the screen. On a RAW acquisition
that is not sufficient, because a plate rung's NOMINAL scale and the resolution its pixels actually
carry are different numbers. ``CompositePlateSource`` serves plate rungs from the persisted preview
cells, which are ``_CELL`` px per well; a rung whose nominal scale is finer than that holds cell
pixels grown by nearest neighbour (``_tilesource._resample`` grows by nearest ON PURPOSE, because
interpolating would be a lie about resolution). Drawing that over the montage would replace a
smoothly scaled thumbnail with a blockier version of the same pixels: a visible regression, at the
exact moment the feature is supposed to be invisible.

:func:`clamp_to_content` therefore refuses to pick a rung finer than the content behind it. The
consequence is the honest one: the coarse layer is the montage's information, area-averaged into
world tiles, and it can never be sharper than the cells it is made of. When a finer producer feeds
those cells the clamp lifts by itself, because it is derived from ``cell_px`` rather than assumed.

WHAT THE REFERENCE DOES, AND WHY ITS CONTINUITY IS NOT A RENDERING TRICK
------------------------------------------------------------------------
``maragall/ndviewer/ndviewer_hcs`` is the tool in the "MIP Navigator" video. Read rather than
watched, it has NO tile ladder and NO rungs: ``gui_config.py`` computes ONE downsample factor for
the session (``TARGET_PIXEL_SIZE_UM = 10.0`` µm/px, ``downsample_factor =
sample_pixel_size_um / TARGET_PIXEL_SIZE_UM``), ``PlateStackBuilder`` assembles the WHOLE plate at
that one resolution into a multi-page TIFF, and the viewer memory-maps it and hands it to napari.
``get_page(t, z)`` is a memmap index.

So its zoom feels continuous because there is exactly one layer and nothing to cross. That is worth
knowing precisely: 10 µm/px is roughly our own montage resolution on a single-FOV well (a 666 µm
field over 88 px is 7.6 µm/px) and about 2-3x finer on a multi-FOV well. The reference is not
winning on resolution. It is winning by never switching, and everything in this module that looks
like caution -- the content clamp, drawing the coarse layer as a FLOOR rather than as a replacement,
keeping already-decoded fine tiles on screen when the zoom pulls back past them -- exists to make a
ladder behave like the one thing it is imitating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Optional, Sequence

import numpy as np

from squidmip._address import Extent

#: A placement thinner than this many widget pixels is dropped. Below one pixel there is nothing to
#: draw, and a sub-pixel source rectangle rounds to an empty QRect, which paints nothing but still
#: costs a fetch.
MIN_DRAW_PX = 1.0


# --- geometry: one region's world slab, and the widget rectangle it is drawn in -----------------

def region_extent(ladder, region: str) -> Optional[Extent]:
    """The region's slab: its ``region_id`` plus the union of its FOV frames in stage µm.

    The same rectangle ``_tilesource.region_bbox_um`` returns, carried as an :class:`Extent` so it
    can be compared, hashed and logged as one value instead of as a bare 4-tuple that says nothing
    about which region it belongs to.
    """
    boxes = [b for k, b in ladder.fov_bboxes.items() if str(k[0]) == str(region)]
    if not boxes:
        return None
    return Extent(region_id=str(region),
                  bbox_um=(min(b[0] for b in boxes), min(b[1] for b in boxes),
                           max(b[2] for b in boxes), max(b[3] for b in boxes)))


def fit_preserving_aspect(rect: Sequence[float], world_w: float, world_h: float) -> tuple:
    """``(x, y, w, h)``: *world* fitted into *rect*, aspect preserved, centred.

    ``_placement.cell_boxes``' rule (``s = min(cell/mh, cell/mw)``, then centre) in world units.
    A freeform holder whose cell rectangle ALREADY has the mosaic's aspect ratio gets its whole
    rectangle back, which is what ``_cell_source`` draws into, so one formula covers both holders.
    """
    x, y, w, h = (float(v) for v in rect)
    if not (world_w > 0 and world_h > 0 and w > 0 and h > 0):
        raise ValueError(f"cannot fit a {world_w}x{world_h} µm slab into a {w}x{h} px rect")
    s = min(w / world_w, h / world_h)
    iw, ih = world_w * s, world_h * s
    return (x + (w - iw) / 2.0, y + (h - ih) / 2.0, iw, ih)


@dataclass(frozen=True)
class CellPlacement:
    """One region's :class:`Extent` and the widget rectangle the montage draws it in.

    ``inner`` is the letterboxed mosaic inside the cell -- the rectangle that actually holds
    pixels. Everything about world-to-widget goes through it, so a tile can never be placed into
    the letterbox bars, which hold no acquired pixels and in the montage are background.
    """

    extent: Extent
    cell: tuple                     # (x, y, w, h) widget px of the whole cell
    inner: tuple                    # (x, y, w, h) widget px of the letterboxed mosaic

    @classmethod
    def of(cls, extent: Extent, cell_rect: Sequence[float]) -> "CellPlacement":
        if extent.bbox_um is None:
            raise ValueError(f"{extent.region_id}: a placement needs a bbox_um to place")
        x0, y0, x1, y1 = extent.bbox_um
        cell = tuple(float(v) for v in cell_rect)
        return cls(extent, cell, fit_preserving_aspect(cell, x1 - x0, y1 - y0))

    @property
    def um_per_px(self) -> float:
        """World micrometres per WIDGET pixel, at the current zoom. The number every rung
        decision is made from -- deliberately not the plate's ``cd``, which says nothing about how
        much world a region's cell covers when regions differ in size."""
        x0, _y0, x1, _y1 = self.extent.bbox_um
        return (x1 - x0) / max(self.inner[2], 1e-9)

    def to_widget(self, bbox_um: Sequence[float]) -> tuple:
        """A world rectangle as ``(x, y, w, h)`` widget px inside this cell."""
        wx0, wy0, wx1, wy1 = self.extent.bbox_um
        ix, iy, iw, ih = self.inner
        sx, sy = iw / (wx1 - wx0), ih / (wy1 - wy0)
        x0, y0, x1, y1 = (float(v) for v in bbox_um)
        return (ix + (x0 - wx0) * sx, iy + (y0 - wy0) * sy, (x1 - x0) * sx, (y1 - y0) * sy)

    def to_world(self, rect: Sequence[float]) -> Optional[tuple]:
        """A widget rectangle as a world bbox, clipped to this region. ``None`` if they miss.

        This is the viewport transform the enumerator needs and the montage never had: it turns
        "what is on screen" into "which micrometres of this region", which is the only form the
        ladder can be culled against.
        """
        wx0, wy0, wx1, wy1 = self.extent.bbox_um
        ix, iy, iw, ih = self.inner
        rx, ry, rw, rh = (float(v) for v in rect)
        sx, sy = (wx1 - wx0) / iw, (wy1 - wy0) / ih
        a = (wx0 + (rx - ix) * sx, wy0 + (ry - iy) * sy,
             wx0 + (rx + rw - ix) * sx, wy0 + (ry + rh - iy) * sy)
        return intersect(a, self.extent.bbox_um)


def intersect(a: Sequence[float], b: Sequence[float]) -> Optional[tuple]:
    """The overlap of two ``(x0, y0, x1, y1)`` boxes, or ``None``. Strict: touching is not
    overlapping, matching ``_tilesource.fovs_overlapping`` and ``_tiling.select_tiles``' cull."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None


# --- the rung: what the screen asks for, clamped to what the pixels actually carry --------------

def content_um_per_px(extent: Extent, cell_px: int) -> float:
    """The resolution the persisted preview CELLS carry for this region, in µm per pixel.

    ``cell_boxes`` maps the mosaic's longer side onto the full ``cell_px``, so the cell's
    resolution is the longer world side over ``cell_px`` -- and that, not the rung's nominal
    scale, is the finest real detail a plate rung can hold on a raw acquisition.
    """
    x0, y0, x1, y1 = extent.bbox_um
    return max(x1 - x0, y1 - y0) / max(int(cell_px), 1)


def clamp_to_content(ladder, level: int, content_scale: float) -> Optional[int]:
    """Move *level* coarser until its rung is no finer than *content_scale*. ``None`` if it cannot.

    Refusing to draw is a real outcome and must not be faked: a ladder whose coarsest rung is
    still finer than the cells (a tiny sample, one FOV) has no plate rung worth drawing, and the
    montage is then already the best picture available.
    """
    n = len(ladder.geometry.levels)
    lvl = int(level)
    while lvl < n and ladder.geometry.levels[lvl].scale_um_per_px < content_scale:
        lvl += 1
    if lvl >= n or ladder.is_fov_level(lvl):
        return None
    return lvl


def plate_level_for(ladder, placement: CellPlacement, cell_px: int, *,
                    current_level: Optional[int] = None) -> Optional[int]:
    """Which PLATE rung to draw in this cell, or ``None`` to leave the montage alone.

    Two rules, in order: the screen's own request (``pick_level``, with its hysteresis deadband so
    a zoom parked on a boundary does not thrash the fetch queue), then :func:`clamp_to_content`.
    """
    lvl = ladder.geometry.pick_level(placement.um_per_px, current_level)
    return clamp_to_content(ladder, lvl, content_um_per_px(placement.extent, cell_px))


# --- the enumerator ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PlacedTile:
    """One tile, and where it goes: what it COVERS (``extent``) and where it CAME FROM.

    ``target`` is widget pixels. ``source`` is ``(fx, fy, fw, fh)`` as FRACTIONS of the tile
    image, not pixels, so the drawing code multiplies by the array it actually got and a tile
    served at an unexpected size cannot silently shift the picture.
    """

    extent: Extent                  # WHICH SLAB of WHICH REGION this rectangle shows
    level: int
    key: Hashable
    bbox_um: tuple                  # the whole tile's world box: the fetch descriptor's box
    target: tuple                   # (x, y, w, h) widget px
    source: tuple                   # (fx, fy, fw, fh) fractions of the tile image

    @property
    def whole_tile(self) -> bool:
        fx, fy, fw, fh = self.source
        return fx <= 0.0 and fy <= 0.0 and fw >= 1.0 and fh >= 1.0


def plate_tiles(ladder, placements: Iterable[CellPlacement], *, cell_px: int,
                viewport: Optional[Sequence[float]] = None,
                current_level: Optional[int] = None,
                min_draw_px: float = MIN_DRAW_PX) -> list:
    """Every plate-rung tile visible in *placements*, placed and clipped. Pure; no I/O, no Qt.

    *viewport* is a widget-pixel ``(x, y, x1, y1)`` box used only to CULL. Clipping is to the
    REGION and never to the viewport, so panning does not change a tile's placement (and therefore
    does not churn the descriptors the fetcher is working on) -- the painter clips to the window.
    """
    out: list = []
    for pl in placements:
        if viewport is not None:
            ix, iy, iw, ih = pl.inner
            if intersect((ix, iy, ix + iw, iy + ih), viewport) is None:
                continue
        lvl = plate_level_for(ladder, pl, cell_px, current_level=current_level)
        if lvl is None:
            continue
        level = ladder.geometry.levels[lvl]
        world = pl.extent.bbox_um
        b = level.bboxes
        hit = ((b[:, 0] < world[2]) & (b[:, 2] > world[0])
               & (b[:, 1] < world[3]) & (b[:, 3] > world[1]))
        for i in np.flatnonzero(hit):
            tile = tuple(float(v) for v in b[int(i)])
            part = intersect(tile, world)
            if part is None:
                continue
            target = pl.to_widget(part)
            if target[2] < min_draw_px or target[3] < min_draw_px:
                continue
            tw, th = tile[2] - tile[0], tile[3] - tile[1]
            source = ((part[0] - tile[0]) / tw, (part[1] - tile[1]) / th,
                      (part[2] - part[0]) / tw, (part[3] - part[1]) / th)
            out.append(PlacedTile(
                extent=Extent(region_id=pl.extent.region_id, bbox_um=part),
                level=lvl, key=level.keys[int(i)], bbox_um=tile,
                target=target, source=source))
    return out


def source_pixels(source: Sequence[float], shape: Sequence[int]) -> tuple:
    """A fractional source rect as integer pixels ``(x, y, w, h)`` of an array of *shape*.

    Clamped to at least 1x1 and to the array: a fraction that rounds to zero width would paint
    nothing, and a fraction that rounds one pixel past the end would raise inside the painter.
    """
    h, w = int(shape[0]), int(shape[1])
    fx, fy, fw, fh = (float(v) for v in source)
    x = int(np.clip(round(fx * w), 0, max(w - 1, 0)))
    y = int(np.clip(round(fy * h), 0, max(h - 1, 0)))
    return (x, y, int(np.clip(round(fw * w), 1, w - x)), int(np.clip(round(fh * h), 1, h - y)))
