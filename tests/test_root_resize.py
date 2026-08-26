"""The root window resizes, and the type comes with it."""

from __future__ import annotations

import re

import pytest

from squidxplorer._viewer import _scale_qss_fonts

QSS = "QPushButton{background:#161b22;color:#c9d1d9;font-size:12px;padding:3px 10px;}"


def _font_px(qss: str) -> int:
    m = re.search(r"font-size:(\d+)px", qss)
    assert m, f"no font-size found in {qss!r}"
    return int(m.group(1))


# --- the stylesheet rewriter ------------------------------------------------------------------

def test_font_size_scales_from_the_authored_value_and_nothing_else_moves():
    out = _scale_qss_fonts(QSS, 2.0)
    assert _font_px(out) == 24
    assert "padding:3px 10px" in out, "padding must not scale — see the hairline note"
    assert "background:#161b22" in out
    assert _font_px(_scale_qss_fonts(QSS, 1.0)) == 12, "x1 must return the authored size"


def test_point_sizes_are_left_alone():
    """A pt size is already device-independent; scaling it would double-apply the DPI."""
    assert _scale_qss_fonts("QLabel{font-size:11pt;}", 3.0) == "QLabel{font-size:11pt;}"


def test_type_never_shrinks_below_legibility():
    assert _font_px(_scale_qss_fonts(QSS, 0.1)) == 8


def test_a_sheet_with_no_font_size_is_returned_unchanged():
    plain = "QFrame{border:1px solid #232b3a;}"
    assert _scale_qss_fonts(plain, 2.0) == plain


# --- the window itself ------------------------------------------------------------------------

@pytest.fixture
def root(qapp_or_skip):
    from squidxplorer._viewer import PlateWindow

    win = PlateWindow(None)
    yield win
    win.close()


@pytest.fixture
def qapp_or_skip():
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


def test_the_root_window_is_resizable_above_a_floor(root):
    """The regression: setFixedSize pinned max == min, so the frame had no grab handles."""
    assert root.maximumWidth() > root.minimumWidth()
    assert root.maximumHeight() > root.minimumHeight()
    assert root.minimumWidth() >= 420
    assert root.minimumHeight() >= 520


def test_the_default_size_never_exceeds_the_screen(root):
    """850 logical is 1700 physical at 200%. A fixed size that cannot fit is not a shape."""
    from qtpy.QtWidgets import QApplication

    avail = QApplication.primaryScreen().availableGeometry()
    w, h = root._default_root_size()
    assert w <= max(root.minimumWidth(), avail.width())
    assert h <= max(root.minimumHeight(), avail.height())


def test_widening_enlarges_the_type_and_returning_to_the_design_width_restores_it(root):
    """Caching the original sheet is what stops repeated resizes compounding the size."""
    root.resize(root._DESIGN_W, 700)
    root._rescale_fonts()
    small = _font_px(root._open_sel_btn.styleSheet())

    root.resize(root._DESIGN_W * 2, 700)
    root._rescale_fonts()
    assert _font_px(root._open_sel_btn.styleSheet()) > small

    root.resize(root._DESIGN_W, 700)
    root._rescale_fonts()
    assert _font_px(root._open_sel_btn.styleSheet()) == 12


def test_the_scale_is_clamped_at_both_ends(root):
    root.resize(200, 200)
    assert root._ui_scale() >= 0.85
    root.resize(6000, 700)
    assert root._ui_scale() <= 2.0
