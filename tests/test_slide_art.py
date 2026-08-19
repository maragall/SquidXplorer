"""The slide-carrier layout on the plate overview.

True-scale slide ART was retired in 2b8fbc5 (a real-size slide dwarfed a small tissue) and the
`_slide_art` module was deleted with it on 2026-08-19; the carrier keeps the even cell layout.
These tests pin that retirement and the gestures on a slide-carrier dataset.
"""

import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


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
    ov.set_carrier(plate)
    ov.resize(600, 480)
    return ov, plate


def test_set_carrier_draws_NO_true_scale_slide_art_and_keeps_the_even_layout(qapp):
    """Pins the deliberate retirement (2b8fbc5, module deleted 2026-08-19): a true-size slide
    dwarfed a small tissue, so the carrier keeps the plate's even layout."""
    ov, plate = _slide_overview(qapp)
    assert not hasattr(ov, "_slides"), "true-scale slide art came back; it was retired deliberately"
    assert ov._carrier_slide is True, "the holder is still known to be a slide carrier"
    cells = [ov._layout[plate.cell_index(r)] for r in ("manual0", "manual1")]
    assert cells[0][2:] == cells[1][2:], "the two tissue cells are not equal any more"
    ax, ay, aw, ah = cells[0]
    bx, by, bw, bh = cells[1]
    assert (min(ax + aw, bx + bw) - max(ax, bx)) <= 1e-9 or (
        min(ay + ah, by + bh) - max(ay, by)) <= 1e-9, "the two tissue cells overlap"


def test_every_gesture_still_resolves_a_cell_on_the_slide_layout(qapp):
    """Hit-test, marquee selection, and the ROI/control frame rects, all on the slide layout."""
    ov, plate = _slide_overview(qapp)
    for region in ("manual0", "manual1"):
        rc = plate.cell_index(region)
        rx, ry, rw, rh = ov._cell_rect(*rc)
        cx, cy = rx + rw / 2, ry + rh / 2
        cell = ov._cell(cx, cy)
        assert cell is not None and cell["well_id"] == region
        assert rc in ov._cells_in(rx + 1, ry + 1, rx + rw - 1, ry + rh - 1)
    for region in ("manual0", "manual1"):
        r = ov._cell_rect(*plate.cell_index(region))
        assert r[2] > 0 and r[3] > 0


def test_status_dots_and_frames_paint_on_a_slide_without_raising(qapp):
    """Qt swallows paint exceptions, so render into a pixmap and assert ink was put down."""
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
    assert len(colors) > 3
