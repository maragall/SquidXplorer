"""Tissue acquisitions drawn as GLASS SLIDES, side by side, from stage micrometres (IMA-265).

Why this module exists
----------------------
A hand-drawn tissue run is not a plate of wells, and until now it was drawn as one: the freeform
path (IMA-253) placed each region's *mosaic bounding box* as a cell inside a generic rounded
"carrier body". That is honest about where the tissue is and says nothing at all about what the
sample IS. Julio, on the result: "the tissues should emulate slides and should be side by side,
not an abstract representation of a carrier".

So the picture drawn here is: **one slide per physical slide, at true size, carrying its tissues
at their true positions**. The slide is the object the operator handled; the tissue is the ink on
it. Two regions on one piece of glass share one slide, and two regions on different glass are two
slides side by side -- and which of those you get is *measured*, never assumed.

Where the numbers come from (nothing here is invented)
------------------------------------------------------
* **The slide footprint, 25 x 75 mm portrait.** Squid ships the artwork for its own 4-up holder,
  ``images/4 slide carrier_1509x1010.png`` at 0.084665 mm/px. Its four slots measure ~23.5 mm
  across by ~73.7 mm along and stand UPRIGHT, side by side across x -- so a slide's short axis is
  x and its long axis is y, which is also what ``_plate._VENDORED_MM["4 slide carrier"]``
  independently says (``rows=1, cols=4``, ``cell_size 25 mm``, ``spacing 27 mm``). 25 x 75 mm is
  the ISO 8037-1 / universal microscope slide, so the art and the standard agree.
* **The tissue rectangles.** ``_plate.region_stage_boxes_um`` -- the union of every FOV footprint,
  in stage micrometres. Same measurement the mosaic itself is composited from.
* **Which regions share a slide.** Purely their RELATIVE separation: a set of regions is one slide
  when the union of their boxes fits inside one slide footprint. Deliberately NOT the carrier's
  ``a1_x_um``/pitch, because ``_plate`` documents the 4-up carrier's origin as "a LAYOUT
  approximation, not a measured calibration" -- keying the picture off it would let a 1 mm
  calibration error split one slide into two. Translation-invariance is a test.

What is drawn (the vocabulary is the WELL PLATE's, deliberately)
----------------------------------------------------------------
The well plate's rendering is the house style and it is not up for renegotiation here: flat fills,
2 px strokes, one accent, a chamfered orientation corner, dashed = empty. A slide reuses all of
it, so the two holders read as one product:

    slide body      flat dark glass, thin cool-grey stroke  (the well plate's holder body)
    label end       a frosted band at the slide's TOP, solid, the way every slide scanner draws
                    it (QuPath / Aperio ImageScope / OMERO thumbnails all mark the label end so
                    the slide has an orientation) -- and it is where the region name goes
    chamfer         same corner cue the plate body already uses

The tissues themselves are NOT drawn here. They are cells, and the overview paints them through
the exact path it always did (status dots, red ROI frame, hover, marquee, control frame, loupe,
double-click), so this module can only ever add a backdrop -- it cannot take a feature away.

Everything is returned in the overview's GRID UNITS, through ONE similarity transform shared by
slides and tissues, so there is a single coordinate system and the slide cannot drift out of
register with the cells drawn on it.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

# ISO 8037-1 / the universal microscope slide, and what Squid's 4-up carrier art measures.
# ACROSS is the short axis (x, the direction slides are racked in), ALONG is the long axis (y).
SLIDE_ACROSS_UM = 25000.0
SLIDE_ALONG_UM = 75000.0

# Fraction of the slide's long axis taken by the frosted label end. Real slides are ~20 mm of a
# 75 mm slide; this is the drawn band, and it is where the region name is written.
LABEL_FRAC = 20.0 / 75.0

Rect = tuple[float, float, float, float]


# --------------------------------------------------------------------------- footprint

def slide_footprint_um(geometry) -> tuple[float, float]:
    """``(across_um, along_um)`` of one slide, from the holder's geometry where it declares one.

    ``PlateGeometry.cell_size_um`` is "well diameter / SLIDE WIDTH", so a carrier that records a
    real slot width is drawn at ITS width rather than at the standard. ``glass slide``'s vendored
    row carries 0.0 (it has no slot), and a 0 um slide is not a drawing -- that falls back.
    """
    across = getattr(geometry, "cell_size_um", 0.0) or 0.0
    try:
        across = float(across)
    except (TypeError, ValueError):
        across = 0.0
    return (across if across > 0 else SLIDE_ACROSS_UM), SLIDE_ALONG_UM


# --------------------------------------------------------------------------- grouping

def group_onto_slides(boxes_um: Mapping[str, Rect], across_um: float, along_um: float
                      ) -> list[list[str]]:
    """Partition regions into the physical slides they sit on. ``[[region, ...], ...]``.

    Greedy in stage order (x then y): walk the regions left-to-right and keep adding to the
    current slide while the group's UNION still fits one slide footprint. Testing the union and
    not the pairwise gap is what stops three regions 12 mm apart from chaining into a 24 mm-wide
    "slide" -- single linkage would happily fuse a row that no piece of glass could hold.

    Slides come back in stage order, so the drawing reads left-to-right exactly as the carrier
    does. Depends only on differences between boxes, never on their absolute stage position.
    """
    order = sorted(boxes_um, key=lambda r: (boxes_um[r][0], boxes_um[r][1], r))
    groups: list[list[str]] = []
    for region in order:
        if groups and _fits(boxes_um, groups[-1] + [region], across_um, along_um):
            groups[-1].append(region)
        else:
            groups.append([region])
    return groups


def _fits(boxes_um: Mapping[str, Rect], regions: Iterable[str],
          across_um: float, along_um: float) -> bool:
    x0, y0, x1, y1 = _union(boxes_um, regions)
    return (x1 - x0) <= across_um and (y1 - y0) <= along_um


def _union(boxes_um: Mapping[str, Rect], regions: Iterable[str]
           ) -> tuple[float, float, float, float]:
    rs = [boxes_um[r] for r in regions]
    return (min(b[0] for b in rs), min(b[1] for b in rs),
            max(b[0] + b[2] for b in rs), max(b[1] + b[3] for b in rs))


# --------------------------------------------------------------------------- slide rectangles

def slide_rects_um(boxes_um: Mapping[str, Rect], across_um: float, along_um: float
                   ) -> list[Rect]:
    """One ``(x, y, w, h)`` slide rectangle per physical slide, in stage micrometres.

    The slide is the FULL footprint whatever the tissue is -- that is the point of drawing a
    slide instead of a bounding box, and it is what makes a 7 mm tissue and a 15 mm tissue look
    genuinely different on identical glass. It is centred on the tissue it carries, since the
    absolute slot position is not trustworthy enough to place it by (see the module docstring).

    A tissue LARGER than a standard slide grows its slide to contain it rather than being clipped:
    a picture that hides pixels the operator acquired is worse than an off-spec rectangle.
    """
    out: list[Rect] = []
    for group in group_onto_slides(boxes_um, across_um, along_um):
        x0, y0, x1, y1 = _union(boxes_um, group)
        w, h = max(across_um, x1 - x0), max(along_um, y1 - y0)
        out.append(((x0 + x1) / 2.0 - w / 2.0, (y0 + y1) / 2.0 - h / 2.0, w, h))
    return out


# --------------------------------------------------------------------------- the fitted layout

def slide_layout(boxes_um: Mapping[str, Rect], rows: int, cols: int, geometry=None
                 ) -> tuple[dict[str, Rect], list[Rect]]:
    """``({region: rect}, [slide rect, ...])`` in GRID UNITS -- the overview's coordinate system.

    ONE similarity transform is applied to slides and tissues alike, so relative size and relative
    offset survive and the tissue cannot drift off its slide. What is fitted into the declared
    ``cols x rows`` box is the union of the SLIDES, not of the tissues: fitting the tissues (what
    ``_plate.freeform_layout`` does, correctly, for a holder with no slide art) would push the
    slide bodies straight off the widget, because a slide is an order of magnitude bigger than the
    tissue on it.

    Returns ``({}, [])`` for degenerate input -- no boxes, or a union with no area on either axis.
    The caller keeps its nominal grid rather than dividing by ~zero and drawing a wrong picture.
    """
    if not boxes_um:
        return {}, []
    across_um, along_um = slide_footprint_um(geometry)
    slides = slide_rects_um(boxes_um, across_um, along_um)
    if not slides:
        return {}, []

    x0 = min(s[0] for s in slides)
    y0 = min(s[1] for s in slides)
    uw = max(s[0] + s[2] for s in slides) - x0
    uh = max(s[1] + s[3] for s in slides) - y0
    if not (uw > 0 and uh > 0):
        return {}, []
    s = min(cols / uw, rows / uh)
    ox, oy = (cols - uw * s) / 2.0, (rows - uh * s) / 2.0

    def _to_grid(r: Rect) -> Rect:
        return (ox + (r[0] - x0) * s, oy + (r[1] - y0) * s, r[2] * s, r[3] * s)

    return {r: _to_grid(b) for r, b in boxes_um.items()}, [_to_grid(r) for r in slides]


# --------------------------------------------------------------------------- painting

def paint_slides(p, rects: Iterable[Rect], label_frac: float = LABEL_FRAC,
                 labels: Optional[Iterable[str]] = None) -> None:
    """Draw the slide bodies (Qt imported lazily so the geometry above stays Qt-free).

    *rects* are widget pixels. Nothing here reads or writes overview state, so the slides are
    strictly a backdrop: every gesture the plate carries is painted after this, by the code that
    always painted it.
    """
    from qtpy.QtCore import QRectF, Qt
    from qtpy.QtGui import QColor, QFont, QPen

    rects = list(rects)
    if not rects:
        return
    p.save()
    for i, (x, y, w, h) in enumerate(rects):
        if not (w > 0 and h > 0):
            continue
        body = QRectF(x, y, w, h)
        # The glass: flat, a shade above the plate background, with the same cool-grey 2 px stroke
        # the well plate's holder body uses. No gradient, no bevel -- brutalist, like the plate.
        p.setBrush(QColor(24, 28, 36))
        p.setPen(QPen(QColor(90, 100, 116), 2))
        p.drawRect(body)
        # The FROSTED LABEL END at the top: solid, lighter, the orientation cue every slide
        # viewer draws (QuPath / ImageScope / OMERO all mark it). Without it a slide is just a
        # tall rectangle and has no up.
        lh = max(2.0, min(h * float(label_frac), h))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(58, 66, 80))
        p.drawRect(QRectF(x, y, w, lh))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(90, 100, 116), 2))
        p.drawLine(int(x), int(y + lh), int(x + w), int(y + lh))
        # Same chamfered corner the plate body carries, so both holders orient the same way.
        c = min(12.0, w * 0.25, h * 0.06)
        if c > 1.0:
            p.setPen(QPen(QColor(120, 132, 150), 2))
            p.drawLine(int(x), int(y + c), int(x + c), int(y))
        text = _label_at(labels, i)
        if text and lh > 10:
            p.setPen(QColor(198, 208, 222))
            # PIXELS, not points: a point size resolves against the paint device's per-screen
            # logicalDpiY, so this label changed apparent size between a laptop panel and an
            # external monitor while every other size in the app (all logical px) held still.
            # 10 px is what 10 pt already rendered as at macOS's 72 dpi, so nothing moves here.
            label_font = QFont("Helvetica Neue")
            label_font.setPixelSize(10)
            label_font.setWeight(QFont.DemiBold)
            p.setFont(label_font)
            p.drawText(QRectF(x, y, w, lh), int(Qt.AlignCenter), text)
    p.restore()


def overview_slide_layout(plate) -> tuple[Optional[dict], Optional[list]]:
    """``({(row, col): tissue_rect}, [slide_rect, ...])`` in GRID UNITS, or ``(None, None)``.

    The one call the overview makes. Returns non-None only for a :class:`~squidxplorer._plate.
    SlideCarrier` that carried its stage boxes -- a real tissue acquisition with coordinates. A
    well plate, or a slide carrier with no measured geometry (regions in report order, no stage
    coordinates), returns ``(None, None)`` and the overview keeps the grid it already had.

    The tissue rectangles come back keyed by ``(row, col)`` -- the overview's cell key -- so the
    result drops straight into ``PlateOverview._layout`` and the slide art registers with the
    cells in one coordinate system.
    """
    try:
        from squidxplorer._plate import SlideCarrier
    except Exception:                       # pragma: no cover - import guard
        return None, None
    if not isinstance(plate, SlideCarrier):
        return None, None
    boxes = getattr(plate, "stage_boxes_um", None) or {}
    if not boxes:
        return None, None
    tissues, slides = slide_layout(boxes, plate.rows, plate.cols,
                                   getattr(plate, "geometry", None))
    if not tissues:
        return None, None
    by_rc: dict[tuple[int, int], Rect] = {}
    for cid, rect in tissues.items():
        try:
            by_rc[plate.cell_index(cid)] = rect
        except KeyError:                    # a region with no cell -> nothing to place it in
            return None, None
    return by_rc, slides


def _label_at(labels, i: int) -> str:
    if labels is None:
        return ""
    seq = list(labels)
    return str(seq[i]) if i < len(seq) else ""
