"""ONE address for "give me pixels": a world-micrometre request, composed over the tile seam.

Seven surfaces spell the question "pixels for this part of the acquisition" seven ways (the
reader's ``(region, fov, channel, z_level, time_point)``, tile descriptors, loupe window
pixels, per-region fusers, gallery cells, 3D brick windows, layer-data closures). This module
is the CONVERGENCE point the 2026-08-15 deepening plan committed to: a request in world
micrometres plus ``(channel, time_point)``, with the LADDER PICK INSIDE — the caller says what
it wants to see and how many pixels it can spend, never which resolution level that is.

Deliberately built ON the existing deep seam rather than beside it: level choice is
:meth:`~squidxplorer._tiling.Geometry.pick_level` via :func:`~squidxplorer._tiling.select_tiles`,
pixels come from any :class:`~squidxplorer._tiling.TileSource` adapter (the reader-backed and
the written-store-backed one both), and placement is :func:`~squidxplorer._tilesource._paste_field`
— THE compositor. No new placement rule: the plate contract names "a third placement rule" as
the defect shape this repo has the most of, because the error renders as a plausible image.

What deliberately stays OUT (and where it lives): z selection rides the tile SOURCE today
(``TileDescriptor`` carries no z; moving it onto the request is the tile path's own change,
with the same reconciliation ``set_time_point`` needed for ``t``), and blur-fallback/caching
stay in :class:`~squidxplorer._tiling.TileCache` — this entry reads strictly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squidxplorer._tiling import TileSource, select_tiles
from squidxplorer._tilesource import PlateLadder, _paste_field


@dataclass(frozen=True)
class PixelRequest:
    """What a consumer wants to SEE, in the world's own units.

    ``bbox_um`` is ``(x0, y0, x1, y1)`` in the tile ladder's CORNER-convention world frame
    (:attr:`PlateLadder.world_bbox_um`'s frame — FOV frame corners, NOT the fov-centre family
    ``mosaic_bbox_um`` speaks; the plate contract documents the two and this type names which
    one it takes). ``out_px`` is a CEILING ``(h, w)``: the raster comes back square-pixeled to
    the bbox's aspect, at most this size, and says its own pitch.
    """

    bbox_um: tuple
    out_px: tuple
    channel: str
    time_point: int = 0


@dataclass(frozen=True)
class PixelResult:
    """The pixels AND what they are: raster, its true pitch, and the window it shows."""

    pixels: np.ndarray          # (h, w), the tiles' own dtype (zeros where nothing covers)
    um_per_px: float            # the raster's true pitch — never assume the request's ceiling
    bbox_um: tuple              # the window shown, echoed from the request


def read_pixels(source: TileSource, ladder: PlateLadder, request: PixelRequest) -> PixelResult:
    """Compose one world window from *source* at the ladder level its target resolution earns."""
    x0, y0, x1, y1 = (float(v) for v in request.bbox_um)
    h_cap, w_cap = (int(v) for v in request.out_px)
    if h_cap < 1 or w_cap < 1:
        raise ValueError(f"out_px must be a positive (h, w) ceiling, got {request.out_px!r}")
    if not (np.isfinite([x0, y0, x1, y1]).all() and x1 > x0 and y1 > y0):
        raise ValueError(
            f"bbox_um must be a finite (x0, y0, x1, y1) window with x1 > x0 and y1 > y0, "
            f"got {request.bbox_um!r}")

    # Square pixels, sized to the bbox aspect under the caller's ceiling.
    um_per_px = max((x1 - x0) / w_cap, (y1 - y0) / h_cap)
    out_w = max(1, int(round((x1 - x0) / um_per_px)))
    out_h = max(1, int(round((y1 - y0) / um_per_px)))

    descs = select_tiles(request.bbox_um, um_per_px, ladder.geometry,
                         channels=(str(request.channel),),
                         time_point=int(request.time_point))
    dst = None
    for desc in descs:
        plane = np.asarray(source.read_tile(desc))
        if dst is None:
            dst = np.zeros((out_h, out_w), dtype=plane.dtype)
        _paste_field(dst, request.bbox_um, um_per_px, plane, desc.bbox_um)
    if dst is None:
        # Nothing covers the window: an EMPTY answer, in a stated dtype, not a crash — the
        # caller asked about a part of the world this acquisition never imaged.
        dst = np.zeros((out_h, out_w), dtype=np.float32)
    return PixelResult(pixels=dst, um_per_px=um_per_px, bbox_um=(x0, y0, x1, y1))
