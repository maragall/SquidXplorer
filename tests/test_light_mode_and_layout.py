"""Three reported UI defects, and the assertions that keep them fixed.

Julio, on macOS in LIGHT mode:

1. "When my computer is in light mode, in-window operator dropdown text goes black, which clobber
   the contrast with background."
2. "cross-monitor UI fixes: keep font/window sizing more consistent across laptop + external
   monitor setups"
3. "Remove or shorten operator descriptions in the main window because the labels are getting
   clipped"

WHAT IS ACTUALLY ASSERTABLE OFFSCREEN
-------------------------------------
Whether it LOOKS right is a screenshot, not a unit test -- the same line `test_root_resize` draws.
What IS assertable is the mechanism behind each report:

1. Every colour rule states INK AND GROUND TOGETHER. The bug was never a wrong colour: the view
   window's operator combo declared no ``color`` at all, so Qt took the foreground from the OS
   palette (black in light mode) while the background came from the row's selector-less
   ``background:#0b0e14``, which Qt applies to every descendant. One half from us, one half from
   the platform. The regression test is that neither half is left unstated -- including the popup
   ``QAbstractItemView``, which the ``QComboBox`` selector never reaches.
2. Type is measured in PIXELS, never points. A point resolves against the paint device's
   per-screen ``logicalDpiY``, so a pt-sized label is the one thing in this app that changes
   apparent size when the window moves to another monitor.
3. The card's description is ELIDED to the card's width and the full text is on the tooltip. Not
   shortened at the registry: the blurbs are where the app says what an operator does.
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

from squidmip import _plate_overview as PO  # noqa: E402
from squidmip import _qtstyle, _slide_art  # noqa: E402
from squidmip._fontscale import window_screen  # noqa: E402
from squidmip._region_viewer import RegionViewer  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)
    return app


# --- 1. light mode: no colour rule may state only half of a pair -------------------------------

def _rules(qss: str) -> "dict[str, str]":
    """``{selector: declarations}`` for a flat (non-nested) Qt style sheet."""
    return {m.group(1).strip(): m.group(2) for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", qss)}


@pytest.mark.parametrize("qss, who", [
    (RegionViewer._COMBO_CHIP_QSS, "the view window's operator dropdown"),
    (_qtstyle.COMBO_QSS, "the shared combo chrome"),
])
def test_a_combo_states_its_foreground_wherever_it_states_a_background(qss, who):
    """The defect exactly: a background with no foreground, so the OS palette supplies the ink."""
    for selector, decls in _rules(qss).items():
        if "background" in decls and "background-color" not in decls:
            assert "color:" in decls, f"{who}: {selector!r} sets a background but no colour"


@pytest.mark.parametrize("qss", [RegionViewer._COMBO_CHIP_QSS, _qtstyle.COMBO_QSS])
def test_the_popup_list_is_styled_and_not_just_the_closed_combo(qss):
    """The popup is a separate top-level view; the ``QComboBox`` selector does not reach it."""
    view = _rules(qss).get("QComboBox QAbstractItemView")
    assert view is not None, "the drop-down list is unstyled — it will use the OS palette"
    assert "color:" in view and "background:" in view


def test_the_operator_dropdown_ink_does_not_come_from_a_light_palette(qapp):
    """Measured, not argued: resolve the combo's foreground under a BLACK-on-light palette.

    Before the fix the sheet named no colour, so this returned the palette's #000000 over the
    near-black background inherited from the row — Julio's report. After it, the sheet's own ink.
    """
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


# --- 2. cross-monitor: pixels, not points; this window's screen, not the primary one ------------

@pytest.mark.parametrize("module", [PO, _slide_art])
def test_no_painted_label_is_measured_in_points(module):
    """``QFont(family, N)`` is a POINT size, resolved against the paint device's per-screen DPI.

    Every other size in this GUI is a logical pixel, which ``enable_hidpi`` makes device
    independent. A pt size is the one thing that changes apparent size between a laptop panel and
    an external monitor. Comment lines are skipped: the note explaining this quotes the old call.
    """
    code = "\n".join(ln for ln in inspect.getsource(module).splitlines()
                     if not ln.lstrip().startswith(("#", "#:")))
    assert not re.search(r"QFont\(\s*[\"'][^\"']+[\"']\s*,\s*\d", code), \
        f"{module.__name__} still sizes a font in points; use setPixelSize"


def test_the_plate_labels_carry_a_pixel_size(qapp):
    font = PO._plate_font(PO._LABEL_PX, QFont.DemiBold)
    assert font.pixelSize() == PO._LABEL_PX
    assert font.pointSize() == -1, "a pixel-sized font must not also carry a point size"


def test_window_screen_falls_back_to_the_primary_rather_than_none(qapp):
    """The fallback is the OLD behaviour, so a widget with no window handle still gets an answer."""
    assert window_screen(None) is QGuiApplication.primaryScreen()


def test_window_screen_prefers_the_screen_the_widget_is_actually_on(qapp):
    class _OnItsOwnScreen:
        def screen(self):
            return "the-external-monitor"

    assert window_screen(_OnItsOwnScreen()) == "the-external-monitor"


def test_a_view_window_is_placed_relative_to_its_own_screen_not_the_desktop_origin():
    """``move(120, 90)`` is a GLOBAL coordinate: it pins every view to whichever display owns
    (0, 0). The offsets are unchanged; what changed is what they are measured from."""
    src = inspect.getsource(RegionViewer.__init__)
    assert "availableGeometry().topLeft()" in src
    assert "self.move(120 + off, 90 + off)" not in src


# --- 3. the operator cards: elided, with the full text on the tooltip ---------------------------

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
    """Elide-with-a-tooltip rather than shorten-at-the-registry: nothing is deleted."""
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


def test_the_cards_in_the_process_pane_all_elide(qapp, monkeypatch):
    """The wiring, not just the widget: every operator card in the main window is an eliding one."""
    from squidmip._operations import _OPERATIONS
    from squidmip._viewer import PlateWindow

    win = PlateWindow(None)
    try:
        cards = win._op_cards
        # The operator stack is the operator REGISTRY and nothing else. "galleryview" used to be
        # here as a bare extra key; it arranges windows rather than pixels and is a View-menu
        # action now, so a stray key in this dict is once again a real defect.
        assert set(cards) == {op.key for op in _OPERATIONS}
        for key, c in cards.items():
            assert hasattr(c, "_retext"), f"the {key!r} card is a plain QPushButton again"
            assert c.toolTip(), f"the {key!r} card lost the full text its elision hides"
    finally:
        win.close()
