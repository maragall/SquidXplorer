"""Detachable tabs and the float-out window (IMA-209/IMA-237) — the pane-shell machinery.

Three widgets, no product knowledge between them:

    _DetachTabBar   the GESTURE: a drag that leaves the bar means "float this tab out"
    _DetachTabs     a QTabWidget wearing that bar, telling the handler WHICH widget fired
    _FloatWindow    the free-floating top-level window a detached tab lives in

All the detach POLICY (what may float, how it is disposed, how it re-docks) stays in
``PlateWindow._detach_tab`` — the seam the offscreen tests drive directly. These classes only
notice the gesture and carry the widget; they were never viewer logic and are lifted out of
``_viewer.py`` unchanged.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QPushButton, QTabBar, QTabWidget, QVBoxLayout, QWidget,
)

from squidmip._qtstyle import BG, BTN_QSS, dark_palette


class _DetachTabBar(QTabBar):
    """Tab bar that detaches a tab when it's dragged OUT of the bar (ImageJ-style float-out,
    IMA-209). Gesture only — all detach logic lives in PlateWindow._detach_tab (the seam the
    offscreen tests drive directly). A drag that stays inside the bar keeps Qt's normal tab
    behavior.

    ``first_detachable`` is where the detachable range starts: 1 in the process console, whose
    index 0 is the permanent 'Process wells' home tab, but 0 in IMA-237's exploration pane, where
    every tab is a user-opened subset and the first one is no more special than the fifth."""

    def __init__(self, on_detach, parent=None, first_detachable: int = 1):
        super().__init__(parent)
        self._on_detach = on_detach
        self._first_detachable = first_detachable
        self._press_pos = None
        self._press_index = -1

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press_pos = e.pos()
            self._press_index = self.tabAt(e.pos())
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if (self._press_pos is not None and self._press_index >= self._first_detachable
                and (e.pos() - self._press_pos).manhattanLength() >= QApplication.startDragDistance()
                and not self.rect().contains(e.pos())):
            idx = self._press_index
            self._press_pos, self._press_index = None, -1      # fire once per press
            # Defer: _detach_tab calls removeTab, and mutating the bar from inside its own
            # mouseMoveEvent (mid-drag, pressed-index state live) is re-entrant — crash bait.
            QTimer.singleShot(0, lambda: self._on_detach(idx))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._press_pos, self._press_index = None, -1
        super().mouseReleaseEvent(e)


class _DetachTabs(QTabWidget):
    """QTabWidget with a detachable tab bar. Qt requires setTabBar BEFORE any tab is added,
    so the custom bar is installed here in __init__ rather than at the call site.

    IMA-237 put a SECOND bar in the window (the exploration pane). Rather than fork a copy of
    IMA-209's detach machinery — this codebase already paid for that once, with three parallel
    _plate.py files — the bar tells the handler WHICH tab widget fired, so one _detach_tab serves
    both. ``on_detach(index, tabs)``."""

    def __init__(self, on_detach, first_detachable: int = 1):
        super().__init__()
        self.setTabBar(_DetachTabBar(lambda i: on_detach(i, self), self,
                                     first_detachable=first_detachable))


class _FloatWindow(QWidget):
    """A detached operator tab as a free-floating top-level window (IMA-209).

    Owns no logic: PlateWindow hands it the live tab widget and two callbacks. 'Re-dock'
    returns the widget to the tab bar (the SAME object — a live CLI keeps its shell and
    history); closing the window disposes the widget through the same cleanup path as
    closing its tab (PlateWindow._dispose_tab_widget)."""

    def __init__(self, title, content, on_close, on_redock):
        super().__init__()
        self._tab_title = title            # verbatim, for re-dock (never parsed back out)
        self.setWindowTitle(f"{title} — SquidMIP")
        self._content = content
        self._on_close = on_close
        # Scoped dark chrome — palette + stylesheet only, never app-wide (see dark_palette).
        # NO per-widget Fusion style here: _left_tabs needs it for its TAB STRIP rendering, but a
        # float has no strip, and a Python-owned QStyle on a deleteLater'd widget can be GC'd
        # first — ~QWidget then unpolishes a dangling style (segfault, found by the test suite).
        self.setPalette(dark_palette())
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background:{BG};color:#e6edf3;")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 5, 8, 5)
        h.addStretch(1)
        dock = QPushButton("Re-dock")
        dock.setStyleSheet(BTN_QSS)
        dock.setToolTip("Return this view to the main window's tab bar")
        dock.clicked.connect(on_redock)
        h.addWidget(dock)
        v.addWidget(bar)
        v.addWidget(content, 1)
        self.resize(560, 480)

    def content(self):
        """The widget this window is holding (None once taken) — lets the window ask WHAT floated
        without reaching into a private attribute."""
        return self._content

    def take_content(self):
        """Detach and return the live widget (re-dock / app-exit); the window becomes an empty
        shell whose close is then a plain close (see closeEvent's guard)."""
        w, self._content = self._content, None
        if w is not None:
            w.setParent(None)
        return w

    def closeEvent(self, e):
        if self._content is not None:      # re-dock/app-exit already emptied us otherwise
            self._on_close()
        super().closeEvent(e)
