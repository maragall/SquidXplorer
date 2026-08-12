"""The one place the window's dark chrome is defined — colours, stylesheets, palette.

Presentation only; imports nothing from ``_viewer``, so it is importable from anywhere.
"""

from __future__ import annotations

import re

from qtpy.QtGui import QColor, QPalette

BG = "#070a0f"
#: Grid ink, the current-FOV box, muted copy, and the accent (selection, focus, links).
GRID, RED, MUTED, ACCENT = QColor(0, 0, 0), QColor("#ff2d2d"), QColor("#8b98ad"), QColor("#58a6ff")
#: Translucent accent wash over a selected well.
SEL_FILL = QColor(88, 166, 255, 90)
#: The control well's persistent frame.
CONTROL_BLUE = QColor("#7fd4ff")

#: Processing-status hue coding; colorblind-safe (blue/amber) with a shape cue for failure.
STATUS = {
    "empty":      QColor("#b7bcc4"),   # not yet processed
    "processing": QColor("#f59e0b"),   # amber — running now
    "done":       QColor("#3b82f6"),   # blue — MIP computed
    "failed":     QColor("#ef4444"),   # red outline + x cross
}

#: ndviewer defaults to light; theme its Qt chrome dark to match.
NDV_DARK = (
    "QWidget{background:#0b0e14;color:#e6edf3;}"
    "QLabel{color:#e6edf3;background:transparent;}"
    "QSlider::groove:horizontal{background:#232b3a;height:4px;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#58a6ff;width:12px;margin:-5px 0;border-radius:6px;}"
    "QPushButton{background:#131824;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;padding:3px 8px;}"
)

#: Tab bar for a pane's own strip — never a global strip across the window.
TABS_DARK = (
    "QTabWidget{background:#070a0f;}"
    "QTabWidget::pane{border:1px solid #c9d1d9;background:#070a0f;top:-1px;}"
    "QTabBar{background:#070a0f;}"
    "QTabBar::tab{background:#0b0e14;color:#8b98ad;padding:6px 13px;border:1px solid #232b3a;"
    "border-bottom:none;margin-right:2px;font-weight:700;font-size:12px;}"
    "QTabBar::tab:selected{background:#131b2b;color:#e6edf3;}"
)

#: An operator "card" in the Process pane (Cellpose-style pick-an-operation).
CARD_QSS = (
    "QPushButton{background:#0d1420;color:#e6edf3;border:1px solid #232b3a;border-radius:10px;"
    "text-align:left;padding:9px 13px;font-size:13px;}"
    "QPushButton:hover{border-color:#58a6ff;background:#111a2b;}"
    "QPushButton:disabled{color:#57606a;border-color:#1a2130;}"
)
BTN_QSS = (
    "QPushButton{background:#131824;color:#e6edf3;border:1px solid #232b3a;border-radius:8px;"
    "padding:7px 12px;font-weight:700;} QPushButton:hover{border-color:#58a6ff;}"
    "QPushButton:disabled{color:#57606a;}"
)
#: A combo AND its popup: the popup is a separate top-level view, so every rule states ink and
#: ground together — no half of a pair may come from the OS theme.
COMBO_QSS = ("QComboBox{background:#0d1420;color:#e6edf3;border:1px solid #232b3a;"
             "border-radius:6px;padding:5px 8px;}"
             "QComboBox:disabled{background:#0d1420;color:#57606a;}"
             "QComboBox QAbstractItemView{background:#0d1420;color:#e6edf3;"
             "border:1px solid #232b3a;selection-background-color:#1c2b44;"
             "selection-color:#e6edf3;outline:none;}")
#: Checkbox with a visible white outline on the box.
CHECK_QSS = (
    "QCheckBox{color:#e6edf3;spacing:7px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #c9d1d9;border-radius:3px;background:#0d1420;}"
    "QCheckBox::indicator:checked{background:#58a6ff;border:1px solid #c9d1d9;}"
)
TERM_QSS = ("QPlainTextEdit{background:#05070b;color:#8bffd0;border:none;"
            "font-family:'SF Mono','Menlo',monospace;font-size:12px;padding:10px;}")
#: The command line inside an embedded terminal.
TERM_INPUT_QSS = (
    "QLineEdit{background:#05070b;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;"
    "padding:6px 8px;font-family:'SF Mono','Menlo',monospace;font-size:12px;}")
#: The plate's right-click dropdown.
MENU_QSS = ("QMenu{background:#0d1420;color:#e6edf3;border:1px solid #232b3a;font-size:16px;}"
            "QMenu::item{padding:7px 18px;}"
            "QMenu::item:selected{background:#1c2b44;}"
            "QMenu::item:disabled{color:#57606a;}")

# Legibility floor for empty-state copy: 16 arcmin at desk distance.
EMPTY_BODY_PX = 15
EMPTY_HEAD_PX = 19   # heading, one step up from body

#: Strip ANSI CSI/OSC escapes + stray control bytes so shell output renders clean.
ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x00-\x08\x0e-\x1f]")


def dark_palette() -> QPalette:
    """A dark palette for ONE widget subtree — never applied app-wide (it bleeds into ndviewer)."""
    dark, base, text, mut = (QColor(7, 10, 20), QColor(11, 14, 20),
                             QColor(230, 237, 243), QColor(87, 96, 109))
    pal = QPalette()
    pal.setColor(QPalette.Window, dark)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, base)
    pal.setColor(QPalette.AlternateBase, dark)
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, QColor(19, 24, 36))
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.ToolTipBase, base)
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.Highlight, QColor(88, 166, 255))
    pal.setColor(QPalette.HighlightedText, dark)
    for grp in (QPalette.Disabled,):
        pal.setColor(grp, QPalette.Text, mut)
        pal.setColor(grp, QPalette.ButtonText, mut)
        pal.setColor(grp, QPalette.WindowText, mut)
    return pal


_OPERATOR_CARD_CLS = None


def operator_card(label: str, blurb: str):
    """An operator card whose description elides to the card's width, full text on hover."""
    return _operator_card_cls()(label, blurb)


def _operator_card_cls():
    """Build (once) the eliding card class; lazy so this module never imports QtWidgets at load."""
    global _OPERATOR_CARD_CLS
    if _OPERATOR_CARD_CLS is not None:
        return _OPERATOR_CARD_CLS

    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtWidgets import QPushButton, QSizePolicy

    #: ``QEvent.Type.FontChange`` under Qt6/qtpy, ``QEvent.FontChange`` under PyQt5.
    _FONT_CHANGE = getattr(getattr(QEvent, "Type", QEvent), "FontChange")

    class _OperatorCard(QPushButton):
        #: The card's horizontal chrome: CARD_QSS padding + border, with slack.
        _CHROME_PX = 30

        def __init__(self, label: str, blurb: str) -> None:
            super().__init__()
            self._label, self._blurb = str(label), str(blurb)
            self.setToolTip(f"{self._label}\n{self._blurb}")
            # Never demand the unelided width from the layout.
            self.setMinimumWidth(0)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            self._retext()

        def _retext(self) -> None:
            avail = self.width() - self._CHROME_PX
            if avail <= 0:                       # not laid out yet; resizeEvent will do it
                self.setText(f"{self._label}\n{self._blurb}")
                return
            fm = self.fontMetrics()
            self.setText(f"{fm.elidedText(self._label, Qt.ElideRight, avail)}\n"
                         f"{fm.elidedText(self._blurb, Qt.ElideRight, avail)}")

        def resizeEvent(self, e):                # noqa: N802 - Qt's spelling
            super().resizeEvent(e)
            self._retext()

        def changeEvent(self, e):                # noqa: N802 - Qt's spelling
            super().changeEvent(e)
            if e is not None and e.type() == _FONT_CHANGE:
                self._retext()

    _OPERATOR_CARD_CLS = _OperatorCard
    return _OPERATOR_CARD_CLS


def hline():
    """A thin horizontal divider (a 1px framed line) for separating sections in a pane."""
    from qtpy.QtWidgets import QFrame

    ln = QFrame()
    ln.setFrameShape(QFrame.HLine)
    ln.setStyleSheet("color:#232b3a;background:#232b3a;max-height:1px;")
    return ln
