"""The views window's OPERATOR DOCK: a collapsible right-edge dock, napari-style (2026-08-19).

Julio's mock moved the plate's Operators card column INTO the views window: a vertical dock on
the right edge, collapsed by default to a thin titled grip. It holds two things, top to bottom:

* the CURRENT VIEW's operator surface (`RegionViewer.operator_panel()` — the old "Operators for
  this window" toolbar), swapped as tabs change through :meth:`OperatorDock.show_window_panel`;
* the BULK card launcher the plate builds (`PlateWindow._build_operator_cards`), whose cards
  open their panels in the plate window exactly as before — only the launcher moved.

History that binds the shape: Julio previously REJECTED a collapsible "operators" chip in the
window's centre-top toolbar (it was reverted). This right-edge dock is the explicitly requested
different thing; do not fold it back into a toolbar.

Wired ONCE at deck/window construction by `ViewerManager` through `operator_dock_installer`,
never per region. Qt-light: everything here renders offscreen, so the layout tests drive it.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import (
    QDockWidget, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

#: The collapsed grip's width — thin enough to be a margin, wide enough to hit with a mouse.
GRIP_PX = 22

#: The expanded dock's width bounds. The cards' blurbs elide below ~220 px (_qtstyle records it).
OPEN_MIN_PX, OPEN_MAX_PX = 230, 360

_BG = "#0b0e14"
_HEADER_QSS = "color:#8b949e;font-size:10px;font-weight:700;border:none;"
_TOGGLE_QSS = (
    "QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
    "border-radius:4px;padding:1px 6px;font-size:11px;}"
    "QPushButton:hover{background:#21262d;}"
)


class _VerticalGrip(QPushButton):
    """The collapsed dock: a thin vertical button reading "◂ Operators" bottom-up."""

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label = text
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(GRIP_PX)
        self.setToolTip("Open the Operators panel (bulk cards + this view's operator controls).")
        self.setStyleSheet(
            "QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
            "border-radius:0px;}"
            "QPushButton:hover{background:#21262d;}")

    def paintEvent(self, e):                             # noqa: N802 - Qt naming
        super().paintEvent(e)
        p = QPainter(self)
        p.rotate(-90)
        # After rotate(-90) the x axis runs UP the widget: draw in (-height, width) space.
        p.drawText(-self.height() + 6, 0, self.height() - 12, self.width(),
                   int(Qt.AlignCenter), self._label)
        p.end()


class OperatorDock(QDockWidget):
    """A right-edge dock with two states: a thin grip (default) and the operator surface."""

    def __init__(self, host, *, cards: Optional[QWidget] = None) -> None:
        super().__init__("Operators", host)
        self.setObjectName("squidxplorer_operator_dock")
        self.setAllowedAreas(Qt.RightDockWidgetArea)
        # Never closable and never floatable: a launcher you can lose is the navigator's defect
        # in a new place. Collapse is the one gesture, and it is ours, not QDockWidget's.
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self._grip = _VerticalGrip("◂ Operators", self)
        self._grip.clicked.connect(lambda *_: self.set_collapsed(False))

        header = QWidget(self)
        header.setStyleSheet(f"background:{_BG};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 4, 6, 4)
        hl.setSpacing(6)
        cap = QLabel("OPERATORS")
        cap.setStyleSheet(_HEADER_QSS)
        hl.addWidget(cap, 1)
        self._collapse_btn = QPushButton("▸")
        self._collapse_btn.setToolTip("Collapse the Operators panel to its edge grip.")
        self._collapse_btn.setCursor(Qt.PointingHandCursor)
        self._collapse_btn.setStyleSheet(_TOGGLE_QSS)
        self._collapse_btn.clicked.connect(lambda *_: self.set_collapsed(True))
        hl.addWidget(self._collapse_btn, 0)
        self._header = header

        body = QWidget(self)
        body.setStyleSheet(f"background:{_BG};")
        bv = QVBoxLayout(body)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(4)
        #: The CURRENT view's operator panel, one page per live view. Pages are owned by their
        #: views (`RegionViewer.dispose` deletes its panel); Qt drops a destroyed page from the
        #: stack on its own.
        self._panels = QStackedWidget(body)
        self._panels.setVisible(False)                   # costs no pixels until a view hands one in
        bv.addWidget(self._panels, 0)
        self._body = body
        self._body_l = bv
        self._cards: Optional[QWidget] = None
        if cards is not None:
            self.set_cards(cards)
        bv.addStretch(0)
        self.setWidget(self._body)

        self._collapsed = True
        self._apply_state()
        try:
            host.addDockWidget(Qt.RightDockWidgetArea, self)
        except Exception:                                # noqa: BLE001 - a host without dock areas
            pass

    # -- the two halves ---------------------------------------------------------------------
    def set_cards(self, cards: QWidget) -> None:
        """Adopt the plate-built card launcher as the dock's lower half."""
        if self._cards is not None:
            self._body_l.removeWidget(self._cards)
            self._cards.deleteLater()
        self._cards = cards
        self._body_l.insertWidget(1, cards, 1)

    def set_cards_enabled(self, flag: bool) -> None:
        for card in getattr(self._cards, "_op_cards", {}).values():
            try:
                card.setEnabled(bool(flag))
            except Exception:                            # noqa: BLE001 - a dead card is not news
                pass

    def show_window_panel(self, panel: Optional[QWidget]) -> None:
        """Show *panel* (the current view's operator surface) above the cards."""
        if panel is None:
            self._panels.setVisible(False)
            return
        if self._panels.indexOf(panel) < 0:
            self._panels.addWidget(panel)
        self._panels.setCurrentWidget(panel)
        self._panels.setVisible(True)

    # -- collapse ---------------------------------------------------------------------------
    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        self._apply_state()

    def _apply_state(self) -> None:
        if self._collapsed:
            self.setTitleBarWidget(self._grip)
            self._grip.setVisible(True)
            self._header.setVisible(False)
            self._body.setVisible(False)
            self.setFixedWidth(GRIP_PX)
        else:
            self.setTitleBarWidget(self._header)
            self._header.setVisible(True)
            self._grip.setVisible(False)
            self._body.setVisible(True)
            self.setMinimumWidth(OPEN_MIN_PX)            # undo the grip's fixed width
            self.setMaximumWidth(OPEN_MAX_PX)
