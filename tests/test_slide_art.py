"""The SLIDE layout: tissues drawn on real glass slides, side by side (IMA-265).

The rule under test is that everything visible is derived from stage micrometres and the
vendored slide footprint -- never from region NAMES, never from a slot count, and never from
the carrier art's absolute origin (which `_plate` itself documents as an approximation).
"""

import pytest
from qtpy.QtWidgets import QApplication

from squidxplorer import _slide_art as SA


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------- the slide footprint

def test_slide_footprint_is_the_iso_glass_slide():
    """25 x 75 mm, portrait: short axis across, long axis along. Measured off Squid's own
    `4 slide carrier_1509x1010.png` (four upright slots, ~23.5 x 73.7 mm at 0.084665 mm/px)
    and matching ISO 8037-1.

    MUTATION: swap the two constants and the slide renders landscape -> red.
    """
    assert SA.SLIDE_ACROSS_UM == 25000.0
    assert SA.SLIDE_ALONG_UM == 75000.0
    assert SA.SLIDE_ACROSS_UM < SA.SLIDE_ALONG_UM


def test_footprint_prefers_the_carrier_geometry_over_the_constant():
    """A holder that declares its slide width wins -- the constant is the FALLBACK, so a
    non-standard carrier is drawn at its own size instead of at 25 mm.

    MUTATION: return the constant unconditionally -> red.
    """
    from squidxplorer._plate import PlateGeometry

    g = PlateGeometry(name="x", rows=1, cols=4, a1_x_um=0, a1_y_um=0,
                      pitch_x_um=27000.0, pitch_y_um=27000.0, cell_size_um=18000.0)
    assert SA.slide_footprint_um(g) == (18000.0, 75000.0)
    assert SA.slide_footprint_um(None) == (25000.0, 75000.0)


def test_a_zero_cell_size_falls_back_rather_than_drawing_a_zero_width_slide():
    """`glass slide`'s vendored row has cell_size 0. A 0 um slide is not a drawing.

    MUTATION: drop the `> 0` guard -> a zero-width slide -> red.
    """
    from squidxplorer._plate import PlateGeometry

    g = PlateGeometry(name="glass slide", rows=1, cols=1, a1_x_um=0, a1_y_um=0,
                      pitch_x_um=0.0, pitch_y_um=0.0, cell_size_um=0.0)
    assert SA.slide_footprint_um(g) == (25000.0, 75000.0)


# --------------------------------------------------------------------- grouping onto slides

def test_regions_that_fit_one_slide_share_it():
    """The real acquisition: manual0/manual1 are 7.2 x 8.6 mm, 1.1 mm apart in x and 10.9 mm
    apart in y. Their union is 8.3 x 19.5 mm, which fits inside 25 x 75 mm -- they are two
    tissues on ONE physical slide, and drawing two slides would be a fiction.

    MUTATION: group every region separately -> two slides -> red.
    """
    boxes = {"manual0": (96814.0, 10186.0, 7209.0, 8619.0),
             "manual1": (97937.0, 21113.0, 7209.0, 8619.0)}
    assert SA.group_onto_slides(boxes, 25000.0, 75000.0) == [["manual0", "manual1"]]


def test_regions_too_far_apart_get_their_own_slides_ordered_by_stage_x():
    """30 mm apart across the slide's SHORT axis cannot be one 25 mm slide. Two slides, and
    they come back in stage order so the drawing reads left-to-right like the carrier does.

    MUTATION: sort by name -> "a" before "b" regardless of position -> red (b is at lower x).
    """
    boxes = {"b": (0.0, 0.0, 2000.0, 2000.0), "a": (30000.0, 0.0, 2000.0, 2000.0)}
    assert SA.group_onto_slides(boxes, 25000.0, 75000.0) == [["b"], ["a"]]


def test_grouping_uses_only_relative_separation_not_the_carrier_origin():
    """Translating every region by a metre changes nothing: the grouping cannot depend on
    `a1_x_um`, which `_plate` documents as an unmeasured approximation.

    MUTATION: assign slots by absolute position (x // pitch) -> translation changes the answer.
    """
    boxes = {"p": (0.0, 0.0, 5000.0, 5000.0), "q": (9000.0, 0.0, 5000.0, 5000.0)}
    far = {k: (x + 1_000_000.0, y, w, h) for k, (x, y, w, h) in boxes.items()}
    assert SA.group_onto_slides(boxes, 25000.0, 75000.0) == \
           SA.group_onto_slides(far, 25000.0, 75000.0) == [["p", "q"]]


def test_a_long_chain_does_not_fuse_into_one_oversized_slide():
    """Single-linkage on NEIGHBOURS would chain three 9 mm-spaced tissues into a 27 mm group
    that no 25 mm slide holds. The union must fit the slide, not merely each hop.

    MUTATION: test pairwise gaps instead of the group union -> one group -> red.
    """
    boxes = {f"r{i}": (i * 13000.0, 0.0, 1000.0, 1000.0) for i in range(3)}
    groups = SA.group_onto_slides(boxes, 25000.0, 75000.0)
    assert groups == [["r0", "r1"], ["r2"]]
    for g in groups:                       # every group really does fit a slide
        xs = [boxes[r][0] for r in g] + [boxes[r][0] + boxes[r][2] for r in g]
        assert max(xs) - min(xs) <= 25000.0


def test_no_boxes_is_no_slides():
    assert SA.group_onto_slides({}, 25000.0, 75000.0) == []


# --------------------------------------------------------------------- slide rectangles (um)

def test_a_slide_is_the_full_footprint_centred_on_the_tissue_it_carries():
    """The slide is 25 x 75 mm whatever the tissue is -- that is the whole point of drawing a
    slide rather than a bounding box, and it is why two regions of different size look different
    ON a slide of the same size.

    MUTATION: size the slide to the tissue -> the assert on w/h -> red.
    """
    boxes = {"t": (100000.0, 40000.0, 6000.0, 8000.0)}
    (x, y, w, h), = SA.slide_rects_um(boxes, 25000.0, 75000.0)
    assert (w, h) == (25000.0, 75000.0)
    assert x + w / 2 == pytest.approx(103000.0)      # centred on the tissue's centre
    assert y + h / 2 == pytest.approx(44000.0)


def test_a_tissue_bigger_than_a_slide_grows_the_slide_instead_of_being_clipped():
    """A drawing that hides pixels the operator acquired is worse than an off-spec slide.

    MUTATION: return the fixed footprint unconditionally -> the tissue overflows -> red.
    """
    boxes = {"t": (0.0, 0.0, 40000.0, 10000.0)}
    (x, y, w, h), = SA.slide_rects_um(boxes, 25000.0, 75000.0)
    assert w >= 40000.0 and h == 75000.0
    assert x <= 0.0 and x + w >= 40000.0


# --------------------------------------------------------------------- the fitted layout

def test_layout_places_tissues_inside_their_slide_and_fits_the_SLIDES_to_the_grid():
    """ONE similarity transform for slides and tissues alike, and it is the SLIDES that are
    fitted to the rows x cols box -- fitting the tissues (what the old freeform layout did)
    would push the slide art off the widget.

    MUTATION: fit the tissue union instead -> the slide exceeds the grid box -> red.
    """
    boxes = {"manual0": (96814.0, 10186.0, 7209.0, 8619.0),
             "manual1": (97937.0, 21113.0, 7209.0, 8619.0)}
    tissues, slides = SA.slide_layout(boxes, rows=2, cols=1, geometry=None)

    # Two regions on one slide each: `slide_layout` returning nothing would make both loops
    # below iterate zero times and the whole containment claim vacuous.
    assert len(slides) >= 1, slides
    assert set(tissues) == {"manual0", "manual1"}, sorted(tissues)
    for rect in slides:                             # every slide inside the declared grid box
        assert rect[0] >= -1e-9 and rect[1] >= -1e-9
        assert rect[0] + rect[2] <= 1 + 1e-9
        assert rect[1] + rect[3] <= 2 + 1e-9
    for rect in tissues.values():                   # every tissue inside SOME slide
        assert any(s[0] - 1e-9 <= rect[0] and s[1] - 1e-9 <= rect[1]
                   and rect[0] + rect[2] <= s[0] + s[2] + 1e-9
                   and rect[1] + rect[3] <= s[1] + s[3] + 1e-9 for s in slides)


def test_the_transform_preserves_relative_size_and_relative_offset():
    """Two tissues of different physical size must get different-sized rectangles, in the same
    ratio; and the gap between them must stay in proportion. This is the property that makes the
    picture a MEASUREMENT rather than an illustration.

    MUTATION: scale x and y independently -> the aspect assert -> red.
    """
    boxes = {"small": (0.0, 0.0, 2000.0, 2000.0),
             "big": (0.0, 10000.0, 4000.0, 4000.0)}
    tissues, _ = SA.slide_layout(boxes, rows=2, cols=1, geometry=None)
    s, b = tissues["small"], tissues["big"]
    assert b[2] / s[2] == pytest.approx(2.0)
    assert b[3] / s[3] == pytest.approx(2.0)
    assert s[2] / s[3] == pytest.approx(1.0)                     # aspect preserved
    assert (b[1] - s[1]) / s[3] == pytest.approx(10000.0 / 2000.0)


def test_degenerate_geometry_yields_no_layout_so_the_caller_keeps_the_nominal_grid():
    """A single point (no extent) has no scale to preserve; dividing by ~zero would be a
    silently wrong picture.

    MUTATION: drop the guard -> ZeroDivisionError or an infinite scale -> red.
    """
    assert SA.slide_layout({}, rows=1, cols=1, geometry=None) == ({}, [])


def test_layout_survives_a_zero_area_region():
    """A region with one FOV and no recorded extent must not blow the transform up."""
    boxes = {"a": (0.0, 0.0, 0.0, 0.0), "b": (5000.0, 0.0, 1000.0, 1000.0)}
    tissues, slides = SA.slide_layout(boxes, rows=1, cols=1, geometry=None)
    assert set(tissues) == {"a", "b"} and slides


# --------------------------------------------------------------------- the overview bridge

def test_overview_layout_from_a_real_slide_carrier():
    """The end-to-end path: a SlideCarrier built from the real acquisition's metadata yields a
    tissue layout keyed by (row, col) plus one slide (both tissues share one piece of glass).

    MUTATION: stop passing stage_boxes_um into SlideCarrier -> (None, None) -> red.
    """
    from squidxplorer._plate import build_plate

    meta = {
        "regions": ["manual0", "manual1"],
        "fovs_per_region": {"manual0": [0], "manual1": [0]},
        "fov_positions_um": {("manual0", 0): (96814.0, 10186.0),
                             ("manual1", 0): (97937.0, 21113.0)},
        "frame_shape": (2084, 2084), "pixel_size_um": 0.752,
        "wellplate_format": "glass slide",
    }
    plate = build_plate(meta)
    by_rc, slides = SA.overview_slide_layout(plate)
    assert by_rc is not None and len(by_rc) == 2
    assert len(slides) == 1                        # both tissues on ONE slide
    assert set(by_rc) == {plate.cell_index("manual0"), plate.cell_index("manual1")}


def test_overview_layout_is_none_for_a_well_plate():
    """A well plate must fall straight through -- the slide path cannot touch the design Julio
    drew for the plate.

    MUTATION: drop the SlideCarrier isinstance guard -> a well plate gets slide art -> red.
    """
    from squidxplorer._plate import build_plate

    wells = ["A1", "A2", "B1"]
    plate = build_plate({"regions": wells, "fovs_per_region": {w: [0] for w in wells},
                         "wellplate_format": "96 well plate"})
    assert SA.overview_slide_layout(plate) == (None, None)


def test_overview_layout_is_none_without_stage_coordinates():
    """A carrier placed by report order (no coordinates) has nothing to draw a true-size slide
    from, so it keeps its nominal grid rather than inventing a scale.
    """
    from squidxplorer._plate import SlideCarrier

    c = SlideCarrier.from_format("4 slide carrier", cell_ids=["manual0", "manual1"])
    assert SA.overview_slide_layout(c) == (None, None)


# --------------------------------------------------------------------- painting (offscreen)

def test_paint_slides_draws_and_does_not_raise(qapp):
    """The art actually renders. Qt swallows paint exceptions, so this asserts on PIXELS: the
    slide body must put non-background ink on the canvas.

    MUTATION: make paint_slides a no-op -> the canvas stays uniform -> red.
    """
    from qtpy.QtGui import QColor, QImage, QPainter

    img = QImage(300, 300, QImage.Format_RGB888)
    img.fill(QColor("#0d1117"))
    p = QPainter(img)
    SA.paint_slides(p, [(20.0, 20.0, 80.0, 240.0), (120.0, 20.0, 80.0, 240.0)])
    p.end()

    colors = {img.pixel(x, y) for x in range(0, 300, 3) for y in range(0, 300, 3)}
    assert len(colors) > 1, "paint_slides put no ink on the canvas"


def test_paint_slides_with_nothing_to_draw_is_a_no_op(qapp):
    from qtpy.QtGui import QColor, QImage, QPainter

    img = QImage(60, 60, QImage.Format_RGB888)
    img.fill(QColor("#0d1117"))
    p = QPainter(img)
    SA.paint_slides(p, [])
    p.end()
    assert {img.pixel(x, y) for x in range(60) for y in range(60)} == {img.pixel(0, 0)}


# ------------------------------------------------ the overview adopts slides via set_carrier

def _slide_overview(qapp):
    """A PlateOverview wired exactly as PlateWindow wires it, for the real tissue metadata."""
    from squidxplorer import _viewer as V
    from squidxplorer._plate import build_plate

    meta = {
        "regions": ["manual0", "manual1"],
        "fovs_per_region": {"manual0": [0], "manual1": [0]},
        "fov_positions_um": {("manual0", 0): (96814.0, 10186.0),
                             ("manual1", 0): (97937.0, 21113.0)},
        "frame_shape": (2084, 2084), "pixel_size_um": 0.752,
        "wellplate_format": "glass slide",
    }
    plate = build_plate(meta)
    rows, cols, wells, order = plate.viewer_grid()
    cl = plate.cell_layout()
    layout = {plate.cell_index(cid): rect for cid, rect in cl.items()}
    ov = V.PlateOverview(rows, cols, wells, layout=layout)
    ov.set_carrier(plate)            # <- the branch under test
    ov.resize(600, 480)
    return ov, plate


def test_set_carrier_draws_NO_true_scale_slide_art_and_keeps_the_even_layout(qapp):
    """The inverse of the test that lived here (`test_set_carrier_adopts_the_slide_layout`).

    That test asserted `set_carrier` populated `_slides` from `overview_slide_layout`, drawing the
    glass slides at true micron scale and placing the mosaics at their real stage positions. Commit
    2b8fbc5 (Julio, 2026-07-23) dropped that: a 25 mm slide dwarfed an 8 mm tissue and the pair
    stacked into a tall, tiny column. `_viewer.set_carrier` now sets `self._slides = None`
    unconditionally and leaves the plate's EVEN carrier layout (`even_carrier_layout`) alone.

    So this pins the decision instead of the retired feature: no slide bodies, and the layout the
    carrier was constructed with survives `set_carrier` untouched. `_slide_art`'s own functions are
    unchanged and still covered by every test above; only the viewer wiring went.

    MUTATION: re-adopt the slide layout in set_carrier -> `_slides` is populated -> red.
    """
    ov, plate = _slide_overview(qapp)
    assert ov._slides is None, "true-scale slide art came back; 2b8fbc5 removed it deliberately"
    assert ov._carrier_slide is True, "the holder is still known to be a slide carrier"
    # The even layout is intact: equal cells, side by side, none on top of another.
    cells = [ov._layout[plate.cell_index(r)] for r in ("manual0", "manual1")]
    assert cells[0][2:] == cells[1][2:], "the two tissue cells are not equal any more"
    ax, ay, aw, ah = cells[0]
    bx, by, bw, bh = cells[1]
    assert (min(ax + aw, bx + bw) - max(ax, bx)) <= 1e-9 or (
        min(ay + ah, by + bh) - max(ay, by)) <= 1e-9, "the two tissue cells overlap"


def test_every_gesture_still_resolves_a_cell_on_the_slide_layout(qapp):
    """The features must survive: hit-test (hover/click/loupe/double-click all start here),
    marquee selection, and the cell rect the red ROI / control frame draw from. Each is exercised
    through the SAME public path the widget uses, on the slide-transformed layout.

    MUTATION: make _cell ignore _layout -> the point no longer lands on manual0 -> red.
    """
    ov, plate = _slide_overview(qapp)
    for region in ("manual0", "manual1"):
        rc = plate.cell_index(region)
        rx, ry, rw, rh = ov._cell_rect(*rc)
        cx, cy = rx + rw / 2, ry + rh / 2
        # 1) point hit-test (hover, click, loupe, double-click)
        cell = ov._cell(cx, cy)
        assert cell is not None and cell["well_id"] == region
        # 2) marquee rubber-band selection
        assert rc in ov._cells_in(rx + 1, ry + 1, rx + rw - 1, ry + rh - 1)
    # 3) the frames the ROI / control / selection draw from land on real, non-empty rects
    for region in ("manual0", "manual1"):
        r = ov._cell_rect(*plate.cell_index(region))
        assert r[2] > 0 and r[3] > 0


def test_status_dots_and_frames_paint_on_a_slide_without_raising(qapp):
    """The whole overlay stack (status dots, selection wash, control frame, red ROI box, hover
    dot) paints over the slide art. Qt swallows paint exceptions, so render into a pixmap and
    assert it did not blow up AND put ink down.
    """
    from qtpy.QtGui import QColor, QPixmap

    ov, plate = _slide_overview(qapp)
    ov._control = plate.cell_index("manual0")
    ov._sel = plate.cell_index("manual1")
    ov._hover = plate.cell_index("manual0")
    ov._selection = {plate.cell_index("manual1")}
    pm = QPixmap(600, 480)
    pm.fill(QColor("#0d1117"))
    ov.render(pm)
    img = pm.toImage()
    colors = {img.pixel(x, y) for x in range(0, 600, 5) for y in range(0, 480, 5)}
    assert len(colors) > 3            # slide + tissue tint + frames all left distinct ink
