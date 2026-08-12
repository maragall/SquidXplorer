"""Detachable tabs and the float-out window — the pane-shell machinery.

Three widgets, no product knowledge between them:

    _DetachTabBar   the gesture: a drag that leaves the bar means "float this tab out"
    _DetachTabs     a QTabWidget wearing that bar, telling the handler which widget fired
    _FloatWindow    the free-floating top-level window a detached tab lives in

All detach POLICY (what may float, how it is disposed, how it re-docks) stays in
PlateWindow._detach_tab. These classes only notice the gesture and carry the widget.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication, QHBoxLayout, QPushButton, QTabBar, QTabWidget, QVBoxLayout, QWidget,
)

from squidxplorer._qtstyle import BG, BTN_QSS, dark_palette


class _DetachTabBar(QTabBar):
    """Detaches a tab when dragged out of the bar (ImageJ-style float-out).

    ``first_detachable`` is where the detachable range starts: 1 when index 0 is a permanent
    home tab, 0 when every tab is equally a user-opened subset.
    """

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
            # deferred: mutating the bar inside its own mouseMoveEvent is re-entrant
            QTimer.singleShot(0, lambda: self._on_detach(idx))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._press_pos, self._press_index = None, -1
        super().mouseReleaseEvent(e)


class _DetachTabs(QTabWidget):
    """QTabWidget with a detachable tab bar.

    A window can hold more than one of these; the bar tells the handler which tab widget
    fired so one on_detach(index, tabs) serves all of them.
    """

    def __init__(self, on_detach, first_detachable: int = 1):
        super().__init__()
        self.setTabBar(_DetachTabBar(lambda i: on_detach(i, self), self,
                                     first_detachable=first_detachable))


class _FloatWindow(QWidget):
    """A detached operator tab as a free-floating top-level window.

    Owns no logic: PlateWindow hands it the live tab widget and two callbacks. Re-dock returns
    the same widget object to the tab bar; closing disposes it through the same cleanup path as
    closing its tab.
    """

    def __init__(self, title, content, on_close, on_redock):
        super().__init__()
        self._tab_title = title            # verbatim, for re-dock (never parsed back out)
        self.setWindowTitle(f"{title} — SquidXplorer")
        self._content = content
        self._on_close = on_close
        # scoped dark chrome only, never app-wide: no per-widget Fusion style here, since a
        # Python-owned QStyle on a deleteLater'd widget can be GC'd first (segfault)
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
        """The widget this window is holding (None once taken)."""
        return self._content

    def take_content(self):
        """Detach and return the live widget; the window becomes an empty shell."""
        w, self._content = self._content, None
        if w is not None:
            w.setParent(None)
        return w

    def closeEvent(self, e):
        if self._content is not None:      # re-dock/app-exit already emptied us otherwise
            self._on_close()
        super().closeEvent(e)
