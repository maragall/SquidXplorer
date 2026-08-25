"""The log panel widget shown at the bottom of the window: a header (RAM, activity, level tally)
that is always visible, and a collapsible body holding the log lines. The rules of what a log line
is, its colour, and that it is bounded live in :mod:`squidxplorer._logpane` (no Qt, unit-tested);
this module is only the Qt surface.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)


def _shrinkable(label: QLabel) -> QLabel:
    """Let a header label shrink below its text width instead of widening the pane it lives
    under. Ignored horizontal policy makes it report a minimum width of 0."""
    sp = label.sizePolicy()
    sp.setHorizontalPolicy(QSizePolicy.Ignored)
    label.setSizePolicy(sp)
    label.setMinimumWidth(0)
    return label

from squidxplorer._activity import ActivityLog
from squidxplorer._logpane import MAX_LINES, DEFAULT_LEVEL, LogBus, color_for
from squidxplorer._measure import human_bytes

#: 1 s: fast enough that "busy" looks live, slow enough the poll itself is invisible.
MEMORY_POLL_MS = 1000

_MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
_BG = "#0d1117"
_HEADER_BG = "#161b22"
_MUTED = "#8b949e"


def memory_line() -> str:
    """This process's RSS against the machine's free/total memory."""
    try:
        import psutil

        rss = psutil.Process().memory_info().rss
        vm = psutil.virtual_memory()
        return (f"mem {human_bytes(rss)}  ·  {human_bytes(vm.available)} free "
                f"of {human_bytes(vm.total)}")
    except Exception:                       # noqa: BLE001 - psutil missing/refused: say so, don't crash
        return "mem -"


class _LogBridge(QObject):
    """Thread hop: a worker logs, the bus calls us on the worker thread, we emit a signal Qt
    delivers queued onto the GUI thread. Nothing else touches the widget off-thread."""

    line = Signal(str, str)             # (level_name, formatted_line)


class LogPanel(QWidget):
    """The bottom-right log panel. Safe to construct and render headless. Owns nothing global; pass
    it a :class:`~squidxplorer._logpane.LogBus` and :class:`~squidxplorer._activity.ActivityLog` and
    it becomes a sink of both, or stays an inert valid widget with neither."""

    #: The panel only notices the "open in new window" gesture; PlateWindow owns the window.
    float_requested = Signal()

    def __init__(self, bus: Optional[LogBus] = None, activity: Optional[ActivityLog] = None,
                 *, level: int = DEFAULT_LEVEL, max_lines: int = MAX_LINES,
                 start_collapsed: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._activity = activity
        self._bridge = _LogBridge()
        self._counts = {"WARNING": 0, "ERROR": 0, "CRITICAL": 0}

        self.setStyleSheet(f"background:{_BG};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        header.setStyleSheet(f"background:{_HEADER_BG};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 3, 8, 3)
        hl.setSpacing(12)

        self._toggle = QPushButton()
        self._toggle.setFlat(True)
        self._toggle.setCursor(Qt.PointingHandCursor)
        # ONE closing brace: the f-string's `{{` collapses to `{`, but the second line is a plain
        # literal so its `}}` stays two braces and the sheet ends `font-size:11px;}}` — Qt then
        # fails to parse the whole sheet and warns "Could not parse stylesheet" on every repolish
        # (labelled `WARN vispy:` only because vispy installs the process-wide Qt message handler).
        self._toggle.setStyleSheet(
            f"QPushButton{{color:#c3ccd9;border:none;background:transparent;font-family:{_MONO};"
            "font-size:11px;}")
        self._toggle.clicked.connect(self.toggle)
        hl.addWidget(self._toggle)

        self._activity_lbl = QLabel("idle")
        self._activity_lbl.setStyleSheet(
            f"color:#c3ccd9;font-family:{_MONO};font-size:11px;background:transparent;")
        hl.addWidget(_shrinkable(self._activity_lbl), 1)

        self._tally_lbl = QLabel("")
        self._tally_lbl.setStyleSheet(
            f"font-family:{_MONO};font-size:11px;background:transparent;")
        hl.addWidget(_shrinkable(self._tally_lbl))

        self._mem_lbl = QLabel(memory_line())
        self._mem_lbl.setStyleSheet(
            f"color:{_MUTED};font-family:{_MONO};font-size:11px;background:transparent;")
        hl.addWidget(_shrinkable(self._mem_lbl))

        self._float_btn = QPushButton("⧉")
        self._float_btn.setFlat(True)
        self._float_btn.setCursor(Qt.PointingHandCursor)
        self._float_btn.setToolTip("Open the log in a new window")
        self._float_btn.setStyleSheet(
            f"QPushButton{{color:#c3ccd9;border:none;background:transparent;"
            f"font-family:{_MONO};font-size:11px;}}")
        self._float_btn.clicked.connect(lambda *_: self.float_requested.emit())
        hl.addWidget(self._float_btn)

        self.setMinimumWidth(0)
        header.setMinimumWidth(0)
        outer.addWidget(header)

        # Memory + run-progress bars adopted from OpenViewList (not rebuilt): they keep their
        # existing signal wiring, so reparenting them here moves pixels, not plumbing.
        self._status = QWidget()
        self._status.setStyleSheet(f"background:{_HEADER_BG};")
        self._status_l = QVBoxLayout(self._status)
        self._status_l.setContentsMargins(8, 2, 8, 3)
        self._status_l.setSpacing(2)
        self._status.setMinimumWidth(0)
        self._status.setVisible(False)      # nothing adopted yet: costs no pixels
        outer.addWidget(self._status)

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(int(max_lines))   # Qt drops the oldest block: bounded, free
        self._view.setFont(QFont("Menlo", 10))
        self._view.setStyleSheet(
            f"QPlainTextEdit{{background:{_BG};color:#c3ccd9;border:none;"
            f"font-family:{_MONO};font-size:11px;}}")
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._view.setMinimumHeight(0)
        self._view.setMinimumWidth(0)
        outer.addWidget(self._view, 1)

        self._bridge.line.connect(self._append)           # queued: worker thread -> GUI thread

        self._mem_timer = QTimer(self)
        self._mem_timer.setInterval(MEMORY_POLL_MS)
        self._mem_timer.timeout.connect(self._refresh_memory)

        self._collapsed = False
        if bus is not None:
            self.attach_bus(bus, level=level)
        if activity is not None:
            self.attach_activity(activity)
        if start_collapsed:
            self.set_collapsed(True)
        else:
            self._sync_toggle_text()

    def adopt_status_row(self, memory_caption: QWidget, memory_bar: QWidget,
                         work_caption: QWidget, work_bar: QWidget) -> None:
        """Re-home the window's memory/run-progress widgets into this panel. Idempotent: adding a
        widget to a layout it's already in is a no-op move, not a duplicate."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(_shrinkable(memory_caption))
        row.addWidget(memory_bar, 1)
        self._status_l.addLayout(row)
        self._status_l.addWidget(work_caption)
        self._status_l.addWidget(work_bar)
        self._status.setVisible(True)

    def attach_bus(self, bus: LogBus, *, level: int = DEFAULT_LEVEL) -> None:
        bus.subscribe(self._on_record)      # called on the LOGGING thread — hop via the bridge

    def attach_activity(self, activity: ActivityLog) -> None:
        self._activity = activity
        activity.subscribe(self._on_activity)   # fires immediately with current state

    def start(self) -> None:
        """Begin the memory poll. Separate from construction so a headless test can build the
        widget without a running timer to chase down."""
        self._refresh_memory()
        self._mem_timer.start()

    def stop(self) -> None:
        self._mem_timer.stop()

    def _on_record(self, level_name: str, line: str) -> None:
        # Runs on whatever thread logged; do nothing but emit — the append happens on the GUI
        # thread via the queued signal.
        self._bridge.line.emit(level_name, line)

    def _append(self, level_name: str, line: str) -> None:
        colour = color_for(level_name)
        safe = (line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        self._view.appendHtml(f'<span style="color:{colour};white-space:pre;">{safe}</span>')
        up = str(level_name).upper()
        if up in self._counts:
            self._counts[up] += 1
            self._refresh_tally()

    def _refresh_tally(self) -> None:
        warn = self._counts["WARNING"]
        err = self._counts["ERROR"] + self._counts["CRITICAL"]
        if not warn and not err:
            self._tally_lbl.setText("")
            return
        parts = []
        if err:
            parts.append(f'<span style="color:{color_for("ERROR")};">{err} error'
                         f'{"s" if err != 1 else ""}</span>')
        if warn:
            parts.append(f'<span style="color:{color_for("WARNING")};">{warn} warning'
                         f'{"s" if warn != 1 else ""}</span>')
        self._tally_lbl.setText("  ·  ".join(parts))

    def _on_activity(self, log: ActivityLog) -> None:
        sentence = log.sentence()
        self._activity_lbl.setText(sentence or "idle")

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        """Collapsing drops the vertical size hint to the header's height so the splitter hands the
        space back to the panes, instead of leaving a grey gap."""
        self._collapsed = bool(collapsed)
        self._view.setVisible(not self._collapsed)
        if self._collapsed:
            self.setMaximumHeight(self.sizeHint().height())
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            self.setMaximumHeight(16777215)     # Qt's QWIDGETSIZE_MAX — no cap
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._sync_toggle_text()

    def _sync_toggle_text(self) -> None:
        self._toggle.setText("▸ Log" if self._collapsed else "▾ Log")

    def _refresh_memory(self) -> None:
        self._mem_lbl.setText(memory_line())

    def text(self) -> str:
        return self._view.toPlainText()

    def line_count(self) -> int:
        return self._view.blockCount() if self._view.toPlainText() else 0
