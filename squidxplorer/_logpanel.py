"""The log panel widget shown at the bottom of the window: a header (RAM, activity, level tally)
that is always visible, and a collapsible body holding the log lines. The rules of what a log line
is, its colour, and that it is bounded live in :mod:`squidxplorer._logpane` (no Qt, unit-tested);
this module is only the Qt surface.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import QEvent, QObject, Qt, QTimer, Signal
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
#: The panel's one font, in PIXELS and set as a QFont (never a stylesheet font): a stylesheet
#: font resolves at polish, after the slot height was fixed from the wrong metrics (measured:
#: 66 px at construction, 94 px once polished, two lines shown of the three promised).
_FONT_PX = 11
#: The header ("Log" + the status word) one step below the body (Julio, 2026-08-25, live on
#: 862 px: "the 'log idle' text size could be smaller, to give more height to the actual log").
_HEADER_FONT_PX = 10


def _mono_font(px: int = _FONT_PX):
    f = QFont()
    f.setFamilies([n.strip().strip("'") for n in _MONO.split(",")])
    f.setPixelSize(px)
    return f
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

    #: A FIXED slot: the header plus exactly LINES lines of the log's own font, scrollable;
    #: never collapsed, never floated (Julio, 2026-08-25: "Re-docking the logger doesn't
    #: work... the logger fullscreen idea will cause complications downstream"). Three lines
    #: at first (132 px of log left the layer list one channel tall on 862 px); five once the
    #: header and the chip grid shrank to pay for them (Julio: "My log Height is a bit too
    #: small").
    LINES = 5

    def __init__(self, bus: Optional[LogBus] = None, activity: Optional[ActivityLog] = None,
                 *, level: int = DEFAULT_LEVEL, max_lines: int = MAX_LINES,
                 parent=None) -> None:
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
        hl.setContentsMargins(8, 2, 8, 2)
        hl.setSpacing(12)

        self._title = QLabel("Log")
        self._title.setFont(_mono_font(_HEADER_FONT_PX))
        self._title.setStyleSheet(
            f"color:#c3ccd9;background:transparent;")
        hl.addWidget(self._title)
        self._activity_lbl = QLabel("idle")
        self._activity_lbl.setFont(_mono_font(_HEADER_FONT_PX))
        self._activity_lbl.setStyleSheet(
            f"color:#c3ccd9;background:transparent;")
        hl.addWidget(_shrinkable(self._activity_lbl), 1)

        self._tally_lbl = QLabel("")
        self._tally_lbl.setFont(_mono_font(_HEADER_FONT_PX))
        self._tally_lbl.setStyleSheet(
            f"background:transparent;")
        hl.addWidget(_shrinkable(self._tally_lbl))

        self._mem_lbl = QLabel(memory_line())
        self._mem_lbl.setFont(_mono_font(_HEADER_FONT_PX))
        self._mem_lbl.setStyleSheet(
            f"color:{_MUTED};background:transparent;")
        hl.addWidget(_shrinkable(self._mem_lbl))

        self.setMinimumWidth(0)
        header.setMinimumWidth(0)
        self._header = header
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
        self._view.setFont(_mono_font())
        self._view.setStyleSheet(
            f"QPlainTextEdit{{background:{_BG};color:#c3ccd9;border:none;"
            "}")
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._view.setMinimumHeight(0)
        self._view.setMinimumWidth(0)
        outer.addWidget(self._view, 1)

        self._bridge.line.connect(self._append)           # queued: worker thread -> GUI thread

        self._mem_timer = QTimer(self)
        self._mem_timer.setInterval(MEMORY_POLL_MS)
        self._mem_timer.timeout.connect(self._refresh_memory)

        self.setFixedHeight(self.slot_px())
        if bus is not None:
            self.attach_bus(bus, level=level)
        if activity is not None:
            self.attach_activity(activity)

    def adopt_status_row(self, work_caption: QWidget, work_bar: QWidget) -> None:
        """Re-home the run-progress caption and bar into this slot. Idempotent: adding a widget
        already in this layout is a no-op for Qt."""
        self._status_l.addWidget(work_caption)
        self._status_l.addWidget(work_bar)
        self._status.setVisible(True)
        # The panel just grew; a collapsed cap frozen at the pre-adoption size CLIPS the
        # band (Julio, live 2026-08-25: the "2%" run bar cut mid-label, no reachable
        # summon toggle). Re-derive the cap from what the band now holds.

    def slot_px(self) -> int:
        """The slot's height: the header plus LINES lines of the view's polished font."""
        fm = self._view.fontMetrics()
        return int(self._header.sizeHint().height() + self.LINES * fm.lineSpacing()
                   + 2 * int(self._view.document().documentMargin())
                   + 2 * self._view.frameWidth())

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
        # The timer may be C++-dead: hosted in a view (one window, 2026-08-25) this panel can
        # be destroyed with that view, and an exception out of a closeEvent aborts the process.
        try:
            self._mem_timer.stop()
        except RuntimeError:
            pass

    def _on_record(self, level_name: str, line: str) -> None:
        # Runs on whatever thread logged; do nothing but emit — the append happens on the GUI
        # thread via the queued signal.
        self._bridge.line.emit(level_name, line)

    def _append(self, level_name: str, line: str) -> None:
        colour = color_for(level_name)
        safe = (line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        self._view.appendHtml(f'<span style="color:{colour};white-space:pre;">{safe}</span>')
        # The banner strips are retired (2026-08-25): while COLLAPSED, this band's one line
        # is the latest entry, so a refusal is still noticed without expanding.
        self._latest_line = str(line)
        self._refresh_header_line()
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

    #: The last log line shown (console-formatted), for the collapsed band's one line.
    _latest_line = ""

    def _on_activity(self, log: ActivityLog) -> None:
        self._activity_sentence = log.sentence() or ""
        self._refresh_header_line()

    _activity_sentence = ""

    def _refresh_header_line(self) -> None:
        """The header's middle label: the activity sentence (the body scrolls the lines)."""
        self._activity_lbl.setText(self._activity_sentence or "idle")

    def _refresh_memory(self) -> None:
        self._mem_lbl.setText(memory_line())

    def text(self) -> str:
        return self._view.toPlainText()

    def line_count(self) -> int:
        return self._view.blockCount() if self._view.toPlainText() else 0
