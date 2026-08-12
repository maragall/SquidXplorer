"""Tissue acquisitions drawn as glass slides, side by side, from stage micrometres."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

# ISO 8037-1 slide: ACROSS is the short axis (x), ALONG is the long axis (y).
SLIDE_ACROSS_UM = 25000.0
SLIDE_ALONG_UM = 75000.0

# Fraction of the slide's long axis taken by the frosted label end.
LABEL_FRAC = 20.0 / 75.0

Rect = tuple[float, float, float, float]


def slide_footprint_um(geometry) -> tuple[float, float]:
    """(across_um, along_um) of one slide: the holder's own slide width, else the ISO standard."""
    across = getattr(geometry, "cell_size_um", 0.0) or 0.0
    try:
        across = float(across)
    except (TypeError, ValueError):
        across = 0.0
    return (across if across > 0 else SLIDE_ACROSS_UM), SLIDE_ALONG_UM


def group_onto_slides(boxes_um: Mapping[str, Rect], across_um: float, along_um: float
                      ) -> list[list[str]]:
    """Partition regions into the physical slides they sit on, in stage order (x then y).

    Tests the running GROUP's union against the footprint, not pairwise gaps, so three regions
    spaced just under the limit can't chain into a union no single slide would hold.
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


def slide_rects_um(boxes_um: Mapping[str, Rect], across_um: float, along_um: float
                   ) -> list[Rect]:
    """One (x, y, w, h) slide rect per physical slide, centred on the tissue it carries.

    Grows past the standard footprint rather than clipping a tissue larger than a slide.
    """
    out: list[Rect] = []
    for group in group_onto_slides(boxes_um, across_um, along_um):
        x0, y0, x1, y1 = _union(boxes_um, group)
        w, h = max(across_um, x1 - x0), max(along_um, y1 - y0)
        out.append(((x0 + x1) / 2.0 - w / 2.0, (y0 + y1) / 2.0 - h / 2.0, w, h))
    return out


def slide_layout(boxes_um: Mapping[str, Rect], rows: int, cols: int, geometry=None
                 ) -> tuple[dict[str, Rect], list[Rect]]:
    """({region: rect}, [slide rect, ...]) in GRID UNITS, one similarity transform for both.

    Fits the SLIDE union to the cols x rows box, not the tissue union, so the slide bodies stay
    on screen. Returns ({}, []) on degenerate input (no boxes, or zero-area union).
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


def paint_slides(p, rects: Iterable[Rect], label_frac: float = LABEL_FRAC,
                 labels: Optional[Iterable[str]] = None) -> None:
    """Draw the slide bodies (Qt imported lazily so the geometry above stays Qt-free)."""
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
        p.setBrush(QColor(24, 28, 36))
        p.setPen(QPen(QColor(90, 100, 116), 2))
        p.drawRect(body)
        # frosted label end: orientation cue, matching other slide viewers
        lh = max(2.0, min(h * float(label_frac), h))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(58, 66, 80))
        p.drawRect(QRectF(x, y, w, lh))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(90, 100, 116), 2))
        p.drawLine(int(x), int(y + lh), int(x + w), int(y + lh))
        # same chamfered corner as the plate body
        c = min(12.0, w * 0.25, h * 0.06)
        if c > 1.0:
            p.setPen(QPen(QColor(120, 132, 150), 2))
            p.drawLine(int(x), int(y + c), int(x + c), int(y))
        text = _label_at(labels, i)
        if text and lh > 10:
            p.setPen(QColor(198, 208, 222))
            # pixel size, not point size: point size resolves per-screen DPI and drifted
            # between displays; 10px matches what 10pt already rendered on macOS.
            label_font = QFont("Helvetica Neue")
            label_font.setPixelSize(10)
            label_font.setWeight(QFont.DemiBold)
            p.setFont(label_font)
            p.drawText(QRectF(x, y, w, lh), int(Qt.AlignCenter), text)
    p.restore()


def overview_slide_layout(plate) -> tuple[Optional[dict], Optional[list]]:
    """({(row, col): tissue_rect}, [slide_rect, ...]), or (None, None) for a well plate or a
    slide carrier with no measured stage coordinates."""
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
