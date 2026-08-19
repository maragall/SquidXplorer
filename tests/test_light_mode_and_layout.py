"""Three UI defects, pinned by the mechanism behind each rather than by a screenshot:

1. every colour rule must state ink and ground together (a bare `background:` with no `color:`
   lets the OS palette supply black-on-dark text)
2. all type is sized in pixels, never points, so it does not change size across screens
3. operator card descriptions are elided to the card's width, with the full text on the tooltip
"""

from __future__ import annotations

import inspect
import re
import sys

import pytest

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)

from qtpy.QtGui import QColor, QFont, QGuiApplication, QPalette  # noqa: E402
from qtpy.QtWidgets import QApplication, QComboBox  # noqa: E402

from squidxplorer import _plate_overview as PO  # noqa: E402
from squidxplorer import _qtstyle, _slide_art  # noqa: E402
from squidxplorer._fontscale import window_screen  # noqa: E402
from squidxplorer._region_viewer import RegionViewer  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


def _rules(qss: str) -> "dict[str, str]":
    """``{selector: declarations}`` for a flat (non-nested) Qt style sheet."""
    return {m.group(1).strip(): m.group(2) for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", qss)}


@pytest.mark.parametrize("qss, who", [
    (RegionViewer._COMBO_CHIP_QSS, "the view window's operator dropdown"),
    (_qtstyle.COMBO_QSS, "the shared combo chrome"),
])
def test_a_combo_states_its_foreground_wherever_it_states_a_background(qss, who):
    checked = 0
    for selector, decls in _rules(qss).items():
        if "background" not in decls:
            continue
        checked += 1
        assert "color:" in decls, f"{who}: {selector!r} sets a background but no colour"
    assert checked, f"{who}: no rule in this style sheet sets a background at all"


@pytest.mark.parametrize("qss", [RegionViewer._COMBO_CHIP_QSS, _qtstyle.COMBO_QSS])
def test_the_popup_list_is_styled_and_not_just_the_closed_combo(qss):
    """The popup is a separate top-level view; the ``QComboBox`` selector does not reach it."""
    view = _rules(qss).get("QComboBox QAbstractItemView")
    assert view is not None, "the drop-down list is unstyled — it will use the OS palette"
    assert "color:" in view and "background:" in view


def test_the_operator_dropdown_ink_does_not_come_from_a_light_palette(qapp):
    """Resolve the combo's foreground under a black-on-light palette: the sheet's own ink must
    win, not the palette's."""
    light = QPalette()
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        light.setColor(role, QColor("#000000"))

    combo = QComboBox()
    combo.setPalette(light)
    combo.setStyleSheet(RegionViewer._COMBO_CHIP_QSS)
    combo.addItem("Maximum Intensity Projection")
    combo.resize(200, 28)
    combo.show()
    qapp.processEvents()
    resolved = combo.palette().color(QPalette.ButtonText)
    combo.hide()

    assert resolved.name() != "#000000", "the OS light palette is still supplying the ink"
    assert resolved.lightness() > 128, "the ink must be light against the combo's dark ground"


@pytest.mark.parametrize("module", [PO, _slide_art])
def test_no_painted_label_is_measured_in_points(module):
    """A QFont point size resolves against the paint device's per-screen DPI, so it is the one
    size that changes apparent size between a laptop panel and an external monitor."""
    code = "\n".join(ln for ln in inspect.getsource(module).splitlines()
                     if not ln.lstrip().startswith(("#", "#:")))
    assert not re.search(r"QFont\(\s*[\"'][^\"']+[\"']\s*,\s*\d", code), \
        f"{module.__name__} still sizes a font in points; use setPixelSize"


def test_the_plate_labels_carry_a_pixel_size(qapp):
    font = PO._plate_font(PO._LABEL_PX, QFont.DemiBold)
    assert font.pixelSize() == PO._LABEL_PX
    assert font.pointSize() == -1, "a pixel-sized font must not also carry a point size"


def test_window_screen_falls_back_to_the_primary_rather_than_none(qapp):
    assert window_screen(None) is QGuiApplication.primaryScreen()


def test_window_screen_prefers_the_screen_the_widget_is_actually_on(qapp):
    class _OnItsOwnScreen:
        def screen(self):
            return "the-external-monitor"

    assert window_screen(_OnItsOwnScreen()) == "the-external-monitor"


def test_a_view_window_is_placed_relative_to_its_own_screen_not_the_desktop_origin():
    """``move(120, 90)`` is a global coordinate that pins every view to whichever display owns
    (0, 0); the offsets are unchanged, only what they are measured from."""
    src = inspect.getsource(RegionViewer.__init__)
    assert "availableGeometry().topLeft()" in src
    assert "self.move(120 + off, 90 + off)" not in src


_BLURB = ("Register every FOV of a well against its neighbours and fuse one seamless mosaic "
          "per well, instead of trusting the stage coordinates alone.")
_LABEL = "Stitch (register + fuse)"


@pytest.fixture
def card(qapp):
    c = _qtstyle.operator_card(_LABEL, _BLURB)
    c.setStyleSheet(_qtstyle.CARD_QSS)
    c.resize(300, 54)                      # about the Process pane's real width
    c.show()
    qapp.processEvents()
    yield c
    c.close()


def test_the_description_is_elided_to_the_card_and_fits_it(card):
    blurb_line = card.text().split("\n")[1]
    assert blurb_line != _BLURB, "the full blurb is still being handed to the button"
    assert blurb_line.endswith("…"), "an elision must SAY that there is more"
    assert card.fontMetrics().horizontalAdvance(blurb_line) <= 300


def test_eliding_never_loses_the_text_it_hides(card):
    """Elide with a tooltip rather than shorten at the registry: nothing is deleted."""
    assert card.toolTip() == f"{_LABEL}\n{_BLURB}"


def test_a_wider_card_shows_more_of_the_description(card, qapp):
    narrow = card.text().split("\n")[1]
    card.resize(900, 54)
    qapp.processEvents()
    assert len(card.text().split("\n")[1]) > len(narrow)


def test_the_card_never_demands_its_unelided_width_from_the_layout(card):
    """A card that asks for 900 px inside a 300 px pane trades clipping for a scrollbar."""
    assert card.minimumWidth() == 0
    assert card.minimumSizeHint().width() <= 300


def test_the_cards_in_the_operator_dock_all_elide(qapp, monkeypatch):
    """The cards live in the views window's dock since 2026-08-19; still the plate's launcher."""
    from squidxplorer._operations import _OPERATIONS
    from squidxplorer._viewer import PlateWindow

    win = PlateWindow(None)
    try:
        pane = win._build_operator_cards()      # what _install_operator_dock hands each dock
        cards = win._op_cards
        # "galleryview" is not an operator key; it's a View-menu action now.
        assert set(cards) == {op.key for op in _OPERATIONS}
        assert pane._op_cards is cards
        for key, c in cards.items():
            assert hasattr(c, "_retext"), f"the {key!r} card is a plain QPushButton again"
            assert c.toolTip(), f"the {key!r} card lost the full text its elision hides"
    finally:
        win.close()


# vispy installs a process-wide Qt message handler, so a stylesheet Qt's CSS parser rejects on
# ANY widget surfaces as a vispy log warning. Checked over the whole window because the failure
# is a class of bug (Python brace-escaping vs. Qt's grammar), not one string.

def _qt_parse_failures(qapp, build, resizes=((1400, 900), (900, 700), (1600, 1000))):
    """Build a window, resize it, and collect Qt's "could not parse" complaints."""
    from qtpy.QtCore import qInstallMessageHandler

    seen: "list[str]" = []
    previous = qInstallMessageHandler(lambda *a: seen.append(str(a[-1])))
    try:
        win = build()
        try:
            win.show()
            qapp.processEvents()
            for w, h in resizes:
                win.resize(w, h)
                qapp.processEvents()
            win.grab()                       # force a real polish + paint of every child
            qapp.processEvents()
        finally:
            win.close()
    finally:
        qInstallMessageHandler(previous)
    return [m for m in seen if "parse stylesheet" in m]


def test_no_widget_in_the_plate_window_has_an_unparseable_stylesheet(qapp):
    from squidxplorer._viewer import PlateWindow

    failures = _qt_parse_failures(qapp, lambda: PlateWindow(None))
    assert failures == [], (
        "Qt refused to parse a stylesheet, so that widget is rendering unstyled and the app "
        f"spams the log on every repolish: {failures[:3]}")


def test_the_log_panel_toggle_closes_its_qss_block_exactly_once(qapp):
    """The specific brace bug: ``}}`` in the non-f-string half of a split literal."""
    from squidxplorer._logpanel import LogPanel

    panel = LogPanel()
    try:
        qss = panel._toggle.styleSheet()
        assert qss.count("{") == qss.count("}"), f"unbalanced braces in {qss!r}"
        assert not qss.endswith("}}"), "the sheet still closes one brace too many"
    finally:
        panel.deleteLater()
