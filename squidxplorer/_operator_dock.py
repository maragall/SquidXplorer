"""The views window's OPERATOR DOCK: a collapsible right-edge dock, napari-style (2026-08-19).

Julio's mock moved the plate's Operators card column INTO the views window: a vertical dock on
the right edge, collapsed by default to a thin grip. It holds ONE thing: the BULK card launcher
the plate builds (`PlateWindow._build_operator_cards`), whose cards open their panels in the
plate window exactly as before — only the launcher moved.

The per-window operator surface (`RegionViewer.operator_panel()`) does NOT live here any more
(Julio, 2026-08-19: "The operators for this window row should also be on the left vertical dock.
The bulk processing is what is solutioned on the right vertical column.") — each view docks its
own panel into napari's LEFT column beside the 2D/3D·ROI chips, so this dock no longer swaps
panels on tab changes.

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
    QDockWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout,
    QWidget,
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
    """The collapsed dock: a thin vertical button reading "◂ Operators" bottom-up.

    It is the dock's WHOLE content while collapsed — a full-height dark tab hugging the right
    edge — never a title-bar button over an empty column (Julio, 2026-08-19: the title-bar
    version left "a whole dock white column only for that button")."""

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label = text
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(GRIP_PX)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip("Open the bulk-processing Operators panel (the plate's operator cards; "
                        "this view's own operator controls live in its left column).")
        # Theme-matched, edge-shaped: the window's own background, one hairline on the left.
        # A boxed light-gray button here reads as a stray widget, not a window edge.
        self.setStyleSheet(
            f"QPushButton{{background:{_BG};color:#8b949e;border:none;"
            "border-left:1px solid #30363d;border-radius:0px;}"
            "QPushButton:hover{background:#161b22;color:#c9d1d9;}")

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

        # A dark ground on the dock itself: whatever sliver Qt paints around the content must be
        # the theme's, never the platform's white window color.
        self.setStyleSheet(f"QDockWidget{{background:{_BG};border:none;}}")

        self._grip = _VerticalGrip("◂ Operators", self)
        self._grip.clicked.connect(lambda *_: self.set_collapsed(False))
        #: A zero-height title bar for the collapsed state, so the grip is the WHOLE column —
        #: the old arrangement (grip AS title bar, body hidden) left the dock's empty content
        #: area painted platform-white for the full window height.
        self._no_title = QWidget(self)
        self._no_title.setFixedHeight(0)

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
        # NO per-view panel stack here: the current view's operator surface lives in that view's
        # LEFT column (see the module docstring); this dock is the bulk-processing cards only.
        self._body = body
        self._body_l = bv
        self._cards: Optional[QWidget] = None
        if cards is not None:
            self.set_cards(cards)
        bv.addStretch(0)
        #: ONE content widget with two pages, so collapse never leaves an empty dock frame for
        #: the platform to paint white: page 0 is the full-height grip, page 1 the card body.
        self._stack = QStackedWidget(self)
        self._stack.setStyleSheet(f"background:{_BG};")
        self._stack.addWidget(self._grip)
        self._stack.addWidget(self._body)
        self.setWidget(self._stack)

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
        self._body_l.insertWidget(0, cards, 1)

    def set_cards_enabled(self, flag: bool) -> None:
        for card in getattr(self._cards, "_op_cards", {}).values():
            try:
                card.setEnabled(bool(flag))
            except Exception:                            # noqa: BLE001 - a dead card is not news
                pass

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
            # The grip is the dock's WHOLE content, under a zero-height title bar — never a
            # title-bar button over a hidden body, which left the dock's full-height content
            # area painted in the platform's white (Julio, 2026-08-19: "a whole dock white
            # column only for that button").
            self.setTitleBarWidget(self._no_title)
            self._header.setVisible(False)
            self._stack.setCurrentWidget(self._grip)
            self.setFixedWidth(GRIP_PX)
        else:
            self.setTitleBarWidget(self._header)
            self._header.setVisible(True)
            self._stack.setCurrentWidget(self._body)
            self.setMinimumWidth(OPEN_MIN_PX)            # undo the grip's fixed width
            self.setMaximumWidth(OPEN_MAX_PX)
