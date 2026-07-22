"""The ONE place the window's dark chrome is defined — colours, stylesheets, palette.

WHY THIS MODULE EXISTS
----------------------
These constants used to live at the top of ``_viewer.py``, and ``_op_panels.py`` reached back
into ``squidmip._viewer`` at call time to borrow three of them::

    from squidmip._viewer import _BTN_QSS, _CHECK_QSS, _COMBO_QSS   # inside a function body

That import had to be deferred into a function precisely because it was circular: ``_viewer``
imports ``_op_panels``. A style constant is a leaf fact; a leaf fact that can only be reached
through the 7,000-line god object is the god object recruiting new dependents.

Every value here is presentation only. Nothing in this module imports ``_viewer``, so it can be
imported at module scope from anywhere in the package, and there is exactly one definition of
each colour rather than one per widget that wanted it.

The palette is deliberately SCOPED, never applied app-wide — see :func:`dark_palette`.
"""

from __future__ import annotations

import re

from PyQt5.QtGui import QColor, QPalette

# --------------------------------------------------------------------------------------
# Palette constants
# --------------------------------------------------------------------------------------
BG = "#070a0f"
#: Grid ink, the current-FOV box, muted copy, and the accent (selection, focus, links).
GRID, RED, MUTED, ACCENT = QColor(0, 0, 0), QColor("#ff2d2d"), QColor("#8b98ad"), QColor("#58a6ff")
#: Translucent accent wash over a SELECTED well (IMA-221).
SEL_FILL = QColor(88, 166, 255, 90)
#: The CONTROL WELL's persistent frame (IMA-248/IMA-260). Light blue, and deliberately NOT RED:
#: the red box is the transient current-FOV, the blue frame is a pinned reference.
CONTROL_BLUE = QColor("#7fd4ff")

#: Processing-status hue coding, adopted from Hongquan Li's record-zstack-viewer plate navigator.
#: Deliberately colorblind-safe (blue/amber, never red/green) with a shape cue for failure (the x).
STATUS = {
    "empty":      QColor("#b7bcc4"),   # not yet processed
    "processing": QColor("#f59e0b"),   # amber — running now
    "done":       QColor("#3b82f6"),   # blue — MIP computed
    "failed":     QColor("#ef4444"),   # red outline + x cross
}

# --------------------------------------------------------------------------------------
# Stylesheets
# --------------------------------------------------------------------------------------
#: ndviewer defaults to light; theme its Qt chrome dark (bg AND text) to match.
NDV_DARK = (
    "QWidget{background:#0b0e14;color:#e6edf3;}"
    "QLabel{color:#e6edf3;background:transparent;}"
    "QSlider::groove:horizontal{background:#232b3a;height:4px;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#58a6ff;width:12px;margin:-5px 0;border-radius:6px;}"
    "QPushButton{background:#131824;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;padding:3px 8px;}"
)

#: Tab bar for a pane's OWN strip — never a global strip across the window.
TABS_DARK = (
    "QTabWidget{background:#070a0f;}"
    "QTabWidget::pane{border:1px solid #c9d1d9;background:#070a0f;top:-1px;}"  # thin white outline
    "QTabBar{background:#070a0f;}"                                            # black strip, not white
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
COMBO_QSS = ("QComboBox{background:#0d1420;color:#e6edf3;border:1px solid #232b3a;"
             "border-radius:6px;padding:5px 8px;}")
#: Checkbox with a visible white outline on the box.
CHECK_QSS = (
    "QCheckBox{color:#e6edf3;spacing:7px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #c9d1d9;border-radius:3px;background:#0d1420;}"
    "QCheckBox::indicator:checked{background:#58a6ff;border:1px solid #c9d1d9;}"
)
TERM_QSS = ("QPlainTextEdit{background:#05070b;color:#8bffd0;border:none;"
            "font-family:'SF Mono','Menlo',monospace;font-size:12px;padding:10px;}")
#: The command line inside an embedded terminal. One definition, two terminal implementations
#: (_Terminal and _ProcTerminal) — they used to carry a verbatim copy each.
TERM_INPUT_QSS = (
    "QLineEdit{background:#05070b;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;"
    "padding:6px 8px;font-family:'SF Mono','Menlo',monospace;font-size:12px;}")
#: The plate's right-click dropdown (IMA-260). Sized at 16 px — a menu is read at a glance with the
#: cursor already on it, so it does not carry the empty-state copy's read-from-your-chair floor.
MENU_QSS = ("QMenu{background:#0d1420;color:#e6edf3;border:1px solid #232b3a;font-size:16px;}"
            "QMenu::item{padding:7px 18px;}"
            "QMenu::item:selected{background:#1c2b44;}"
            "QMenu::item:disabled{color:#57606a;}")

# --------------------------------------------------------------------------------------
# Legibility floor for read-at-a-distance copy (project-wide constraint)
# --------------------------------------------------------------------------------------
# The spec is angular, not typographic: 16 arcmin MINIMUM, 20 arcmin optimal, which this project
# has already converted to 17.3 px at 60 cm and 28.8 px at 1 m. Scaling the floor to 1 m gives
# 28.8 * 16/20 = 23.0 px, so 24 px clears the 16-arcmin minimum at BOTH seating distances, and
# 30 px clears the 20-arcmin optimum at 1 m. Empty-state copy is exactly the text a user reads
# while leaning back from the big monitor, so it is sized for the far case, not the near one.
EMPTY_BODY_PX = 15   # 16 arcmin at ~40 cm. The old 24 px assumed a 1 m viewing distance on a
#                      large monitor; Julio is on a SMALL one and called it "huge text". A pane
#                      read at desk distance does not need the across-the-room size.
EMPTY_HEAD_PX = 19   # heading, one step up from body

#: Strip ANSI CSI/OSC escapes + stray control bytes so shell output renders clean in a
#: QPlainTextEdit (we run the shell with TERM=dumb to minimise these, but zsh still emits a few).
ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x00-\x08\x0e-\x1f]")


def dark_palette() -> QPalette:
    """A dark palette for ONE widget subtree (a tab widget, a float window) — never the app.

    The tab strip's empty area (behind/beside the tabs) is painted by the STYLE from the palette,
    not by our stylesheets, so in macOS LIGHT mode it rendered white. We fix it by giving the TAB
    WIDGET a Fusion style + this dark palette — scoped to that widget subtree, NOT the whole app.
    Applying it app-wide bled into the embedded ndviewer and hid its per-channel colour swatches
    (the cmap combo indicators), which is why this is deliberately not global.
    """
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


def hline():
    """A thin horizontal divider (a 1px framed line) for separating sections in a pane."""
    from PyQt5.QtWidgets import QFrame

    ln = QFrame()
    ln.setFrameShape(QFrame.HLine)
    ln.setStyleSheet("color:#232b3a;background:#232b3a;max-height:1px;")
    return ln
