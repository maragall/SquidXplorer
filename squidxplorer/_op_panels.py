"""The shared panel toolkit for the ONE parameter surface: the inline slot.

The bespoke operator PAGES are shelved whole (Julio, 2026-08-25: "You have like pages per
operators when you open the controls. I don't want that. You should shelf those operator
pages.") - StitcherPanel, DeconQCPanel, the QC sweep (DeconQCResultView, its worker) and
their kwargs converters are gone; reinstating starts from git history. Every operator's
controls are built from its ``params`` declaration by :mod:`squidxplorer._param_panel`
(headline knobs plus the "advanced parameters" disclosure), inserted inline under the
view's operators row. What survives here is the toolkit those panels share and the one
vocabulary item the z-handling combo needs.

This module never touches a tab bar or a dock; the host places the panel.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from squidxplorer import _qtstyle

#: The Z-handling combo's spelling of ``z_operator=None``: fuse every acquired plane unchanged,
#: no inner operator. NOT a registry name (the registered `keepz` identity was shelved
#: 2026-08-24); the panel maps this label to None before anything asks the registry.
KEEP_EVERY_PLANE = "keep every z plane"


def z_operator_choice(text: str):
    """The z-handling combo's text as ``stitch_region``'s ``z_operator`` value: the
    keep-every-plane label spells ``None`` (every acquired plane, fused unchanged)."""
    return None if str(text) == KEEP_EVERY_PLANE else str(text)


_BG = "#0d1117"
_SUB = "color:#8b98ad;font-size:11px;"


def _apply_qss(root: QWidget) -> None:
    """Style every control in *root* with the app-wide dark theme (squidxplorer._qtstyle)."""
    for w in root.findChildren(QPushButton):
        w.setStyleSheet(_qtstyle.BTN_QSS)
        w.setCursor(Qt.PointingHandCursor)
    for w in root.findChildren(QComboBox):
        w.setStyleSheet(_qtstyle.COMBO_QSS)
    for w in root.findChildren(QSpinBox):
        w.setStyleSheet(_qtstyle.COMBO_QSS)
    for w in root.findChildren(QCheckBox):
        w.setStyleSheet(_qtstyle.CHECK_QSS)


def _wrapped(text: str, style: str) -> QLabel:
    """Word-wrapped QLabel that reserves the height its wrapping needs (a plain one doesn't)."""
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(style)
    lab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    lab.setMinimumHeight(lab.fontMetrics().height() * 2)
    return lab


def _row(*widgets) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setSpacing(6)
    for w in widgets:
        lay.addWidget(w) if isinstance(w, QWidget) else lay.addLayout(w)
    lay.addStretch(1)
    return lay


class _Panel(QWidget):
    """Common shell: a scrollable body and a status line. Quiet by default (verbosity
    strip, Julio 2026-08-25): no title heading, no description paragraph - the operators
    row already names the operator, and prose belongs in tooltips or the log."""

    def __init__(self, host, title: str, blurb: str) -> None:
        super().__init__()
        self.host = host
        self.title = str(title)                # kept as DATA (tab labels, tests); not drawn
        self.setStyleSheet(f"background:{_BG};color:#e6edf3;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QWidget()
        self.v = QVBoxLayout(body)
        self.v.setContentsMargins(10, 6, 10, 6)
        self.v.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(body)
        outer.addWidget(scroll)

        self.status = _wrapped("", "color:#d29922;font-size:11px;")

    def say(self, text: str) -> None:
        """Put a sentence in front of the user; also routed to the host's own readout."""
        self.status.setText(text)
        say = getattr(self.host, "say", None)
        if callable(say):
            say(text)
