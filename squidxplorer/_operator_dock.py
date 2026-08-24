"""The views window's OPERATOR DOCK: a collapsible right-edge dock, napari-style (2026-08-19).

Julio's mock moved the plate's Operators card column INTO the views window: a vertical dock on
the right edge, collapsed by default to a thin grip. It holds the BULK card launcher the plate
builds (`PlateWindow._build_operator_cards`) AND, since 2026-08-24, the panel a clicked card
opens: Julio — "When I click on an operator on the operator collapsible dock, the UI appears in
the plate window, rather than as a tab in that same collapsible dock." A card click now shows
that operator's panel on this dock's own panel page (one at a time, "◂ operators" goes back);
the plate's `_left_tabs` route survives only for non-card users (menus, published QC results).

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
    """A right-edge dock: a thin grip (default), the card launcher, or one opened panel."""

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

        # -- the PANEL page (Julio, 2026-08-24: "the UI appears in the plate window, rather
        # -- than as a tab in that same collapsible dock"): a clicked card's operator panel
        # -- opens HERE, one at a time, with a back-to-cards affordance on top. The widget is
        # -- the plate's own (it stays in the plate's _op_tabs registry); this dock only hosts
        # -- and reparents it, exactly the discipline the ViewDeck uses for whole views. ------
        panel_page = QWidget(self)
        panel_page.setStyleSheet(f"background:{_BG};")
        pv = QVBoxLayout(panel_page)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)
        back_row = QWidget(panel_page)
        back_row.setStyleSheet(f"background:{_BG};")
        brl = QHBoxLayout(back_row)
        brl.setContentsMargins(8, 4, 6, 4)
        brl.setSpacing(6)
        self.back_btn = QPushButton("◂ operators")
        self.back_btn.setToolTip("Back to the operator cards.")
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.setStyleSheet(_TOGGLE_QSS)
        self.back_btn.clicked.connect(lambda *_: self.show_cards())
        brl.addWidget(self.back_btn, 0)
        self._panel_title = QLabel("")
        self._panel_title.setStyleSheet(_HEADER_QSS)
        brl.addWidget(self._panel_title, 1)
        pv.addWidget(back_row, 0)
        self._panel_l = QVBoxLayout()
        self._panel_l.setContentsMargins(0, 0, 0, 0)
        pv.addLayout(self._panel_l, 1)
        self._panel_page = panel_page
        self._panel: Optional[QWidget] = None
        #: Which expanded page the user was on, so a collapse/expand round trip lands back
        #: where it left (the panel, not the cards, when a panel was open).
        self._on_panel_page = False

        #: ONE content widget with three pages, so collapse never leaves an empty dock frame
        #: for the platform to paint white: page 0 is the full-height grip, page 1 the card
        #: body, page 2 an opened operator's panel.
        self._stack = QStackedWidget(self)
        self._stack.setStyleSheet(f"background:{_BG};")
        self._stack.addWidget(self._grip)
        self._stack.addWidget(self._body)
        self._stack.addWidget(self._panel_page)
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

    # -- the panel page -----------------------------------------------------------------------
    def panel(self) -> Optional[QWidget]:
        """The operator panel this dock is currently hosting, or ``None``."""
        return self._panel

    def show_panel(self, panel: QWidget, title: str) -> None:
        """Host *panel* on the dock's panel page and put that page on screen, expanded.

        One panel at a time: a previously hosted one is released (it stays alive in the
        plate's ``_op_tabs`` registry, which owns panel lifetime; this dock never disposes)."""
        if self._panel is not None and self._panel is not panel:
            self.release_panel()
        if self._panel is None:
            self._panel = panel
            self._panel_l.addWidget(panel)
            panel.setVisible(True)
        self._panel_title.setText(str(title).upper())
        self._on_panel_page = True
        self.set_collapsed(False)
        self._stack.setCurrentWidget(self._panel_page)

    def show_cards(self) -> None:
        """Back to the card launcher. The hosted panel stays alive on its hidden page — its
        state (a decon sweep, half-typed parameters) survives the round trip."""
        self._on_panel_page = False
        if not self._collapsed:
            self._stack.setCurrentWidget(self._body)

    def release_panel(self) -> Optional[QWidget]:
        """Detach and return the hosted panel without disposing it (the plate owns disposal)."""
        panel, self._panel = self._panel, None
        if panel is not None:
            self._panel_l.removeWidget(panel)
            panel.setParent(None)
            panel.setVisible(False)
        self._on_panel_page = False
        if self._stack.currentWidget() is self._panel_page:
            self._stack.setCurrentWidget(self._body if not self._collapsed else self._grip)
        self._panel_title.setText("")
        return panel

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
            self._stack.setCurrentWidget(
                self._panel_page if (self._panel is not None and self._on_panel_page)
                else self._body)
            self.setMinimumWidth(OPEN_MIN_PX)            # undo the grip's fixed width
            self.setMaximumWidth(OPEN_MAX_PX)
