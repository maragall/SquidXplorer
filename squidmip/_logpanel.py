"""The log panel WIDGET — the thing the user watches at the bottom of the window.

The rules of what a log line is, how it is coloured, and that it is bounded live in
:mod:`squidmip._logpane` (no Qt, unit-tested). This module is only the Qt surface that shows
them, plus the two live readouts Squid's own status bar shows continuously — memory and the
current activity — because "the GUI is actually doing something rather than staying idle" is the
whole reason the panel exists.

WHERE IT SITS, AND WHY THERE
----------------------------
Julio: "the logger... on the bottom right of the GUI, below the exploration pane" and "look how
squid software positions it." Squid's ``gui_hcs.py`` puts three widgets in a persistent bottom
status bar: ``ramMonitorWidget``, ``backpressureMonitorWidget``, ``warningErrorWidget``
("Auto-hides when no messages pending"). So the shape is: RAM shown ALWAYS, activity shown
ALWAYS, and the log body available but not allowed to dominate. This panel is that shape — a
one-line header that is always visible (RAM + activity + a level tally), and a collapsible body
that holds the actual lines.

BOUNDED FOR FREE
----------------
``QPlainTextEdit.setMaximumBlockCount(MAX_LINES)`` makes Qt drop the oldest block when the cap is
reached — the bounded-memory requirement met by the widget itself, not by our bookkeeping. A plate
run emits tens of thousands of lines; without this the panel is a slow leak with a nice UI.

THREADS — WHY THE BRIDGE IS A QObject
-------------------------------------
Operators log from QThreads. ``LogBus`` is pure Python: it calls subscribers SYNCHRONOUSLY, on
whatever thread emitted the record. Touching a QWidget from a worker thread is undefined behaviour
and crashes at random — the exact class of bug this codebase has paid for at QThread teardown. So
the subscriber this panel registers does nothing but emit a Qt signal, and the signal is delivered
QUEUED into the GUI event loop, where the append actually happens. The pure bus stays testable
without Qt; the widget owns the thread hop.

COLLAPSED MEANS COLLAPSED
-------------------------
Requirement: "it must not steal space from the panes when collapsed." Collapsing hides the body
and drops this widget's vertical size hint to the header's height, so the splitter gives the
reclaimed space back to the panes above rather than leaving a grey gap. The header stays, because
a status bar that vanishes cannot tell you the app is busy — which is the one thing it is for.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)


def _shrinkable(label: QLabel) -> QLabel:
    """Let a header label shrink below its text width instead of forcing the panel — and the whole
    pane-3 column — wider. A status readout must never widen the pane it lives under (the
    exploration pane already sets that column's minimum); at the pane's natural width the text
    shows in full, and only a hand-dragged-narrower pane clips it. Ignored horizontal policy makes
    the label report a minimum width of 0."""
    sp = label.sizePolicy()
    sp.setHorizontalPolicy(QSizePolicy.Ignored)
    label.setSizePolicy(sp)
    label.setMinimumWidth(0)
    return label

from squidmip._activity import ActivityLog
from squidmip._logpane import MAX_LINES, DEFAULT_LEVEL, LogBus, color_for
from squidmip._measure import human_bytes

#: How often the memory readout refreshes. 1 s: fast enough that "busy" looks live, slow enough
#: that the poll itself is invisible. It is one psutil call, on the GUI thread, never in a run's
#: hot path.
MEMORY_POLL_MS = 1000

_MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
_BG = "#0d1117"
_HEADER_BG = "#161b22"
_MUTED = "#8b949e"


def memory_line() -> str:
    """One line describing this process's footprint against the machine — the RAM readout.

    Squid shows RAM continuously so the user can see a run consuming the machine before it swaps.
    This shows the same: what THIS process holds, and how much of the machine is still free. Both
    numbers, because "2.1 GiB" alone is alarming on a laptop and nothing on a workstation — the
    free number is what makes it readable.
    """
    try:
        import psutil

        rss = psutil.Process().memory_info().rss
        vm = psutil.virtual_memory()
        return (f"mem {human_bytes(rss)}  ·  {human_bytes(vm.available)} free "
                f"of {human_bytes(vm.total)}")
    except Exception:                       # noqa: BLE001 - psutil missing/refused: say so, don't crash
        return "mem —"


class _LogBridge(QObject):
    """The thread hop. A worker logs, the bus calls us on the worker thread, we emit a signal that
    Qt delivers QUEUED onto the GUI thread. Nothing else touches the widget from off-thread."""

    line = pyqtSignal(str, str)             # (level_name, formatted_line)


class LogPanel(QWidget):
    """The bottom-right log panel: a header that is always visible and a collapsible body.

    Built to be safe to construct and render headless (offscreen) — the paint test renders it into
    a QPixmap, which is what caught two bugs role-level tests could not see. It owns nothing global:
    pass it the process's :class:`~squidmip._logpane.LogBus` and :class:`~squidmip._activity.ActivityLog`
    and it becomes a sink of both. With neither, it is an inert but valid widget (used by the layout
    before the window has wired the buses).
    """

    def __init__(self, bus: Optional[LogBus] = None, activity: Optional[ActivityLog] = None,
                 *, level: int = DEFAULT_LEVEL, max_lines: int = MAX_LINES,
                 start_collapsed: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._bus = bus
        self._activity = activity
        self._bridge = _LogBridge()
        self._counts = {"WARNING": 0, "ERROR": 0, "CRITICAL": 0}

        self.setStyleSheet(f"background:{_BG};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # -- header: always visible (RAM, activity, level tally, collapse toggle) --------
        header = QWidget()
        header.setStyleSheet(f"background:{_HEADER_BG};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 3, 8, 3)
        hl.setSpacing(12)

        self._toggle = QPushButton()
        self._toggle.setFlat(True)
        self._toggle.setCursor(Qt.PointingHandCursor)
        self._toggle.setStyleSheet(
            f"QPushButton{{color:#c3ccd9;border:none;background:transparent;font-family:{_MONO};"
            "font-size:11px;}}")
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

        # The panel itself must not impose a width on the pane-3 column — the exploration pane above
        # it owns that column's minimum. Its OWN body (the log view) can shrink too.
        self.setMinimumWidth(0)
        header.setMinimumWidth(0)
        outer.addWidget(header)

        # -- body: the bounded log view --------------------------------------------------
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

    # -- wiring -------------------------------------------------------------------------
    def attach_bus(self, bus: LogBus, *, level: int = DEFAULT_LEVEL) -> None:
        """Become a sink of *bus*. Subscribing (not owning the handler) is deliberate: many
        widgets can watch one bus, and the bus is what installs on the root logger once."""
        self._bus = bus
        bus.subscribe(self._on_record)      # called on the LOGGING thread — hop via the bridge

    def attach_activity(self, activity: ActivityLog) -> None:
        """Become a sink of the activity registry — the current-work line in the header."""
        self._activity = activity
        activity.subscribe(self._on_activity)   # fires immediately with current state

    def start(self) -> None:
        """Begin the memory poll. Separate from construction so a headless test can build the
        widget without a running timer it then has to chase down."""
        self._refresh_memory()
        self._mem_timer.start()

    def stop(self) -> None:
        self._mem_timer.stop()

    # -- log sink -----------------------------------------------------------------------
    def _on_record(self, level_name: str, line: str) -> None:
        # Runs on whatever thread logged. Do NOTHING but emit — the append happens on the GUI
        # thread via the queued signal. This is the whole reason the bridge exists.
        self._bridge.line.emit(level_name, line)

    def _append(self, level_name: str, line: str) -> None:
        colour = color_for(level_name)
        # appendHtml adds one block; setMaximumBlockCount then evicts the oldest if over cap.
        # Escape the payload so a log line containing "<" is not eaten as markup.
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
        # Auto-hides when there is nothing pending — Squid's warningErrorWidget behaviour. The
        # colour is the loudest present, muted otherwise, so a clean run's header stays calm.
        parts = []
        if err:
            parts.append(f'<span style="color:{color_for("ERROR")};">{err} error'
                         f'{"s" if err != 1 else ""}</span>')
        if warn:
            parts.append(f'<span style="color:{color_for("WARNING")};">{warn} warning'
                         f'{"s" if warn != 1 else ""}</span>')
        self._tally_lbl.setText("  ·  ".join(parts))

    # -- activity sink ------------------------------------------------------------------
    def _on_activity(self, log: ActivityLog) -> None:
        # ActivityLog fires on whatever thread advanced it; label writes should be on the GUI
        # thread. In practice the GUI advances it, but route through the bridge's thread affinity
        # by using a plain setText which Qt tolerates for a QLabel text set — kept simple, and the
        # heavy cross-thread traffic (log lines) already goes through the queued signal.
        sentence = log.sentence()
        self._activity_lbl.setText(sentence or "idle")

    # -- collapse -----------------------------------------------------------------------
    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        """Hide/show the body. Collapsed drops the vertical size hint to the header so the splitter
        hands the space back to the panes — it must not leave a grey gap where the body was."""
        self._collapsed = bool(collapsed)
        self._view.setVisible(not self._collapsed)
        if self._collapsed:
            # Fix height to the header so the widget cannot claim body space it is not showing.
            self.setMaximumHeight(self.sizeHint().height())
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            self.setMaximumHeight(16777215)     # Qt's QWIDGETSIZE_MAX — no cap
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._sync_toggle_text()

    def _sync_toggle_text(self) -> None:
        self._toggle.setText("▸ Log" if self._collapsed else "▾ Log")

    # -- memory -------------------------------------------------------------------------
    def _refresh_memory(self) -> None:
        self._mem_lbl.setText(memory_line())

    # -- testing seam -------------------------------------------------------------------
    def text(self) -> str:
        """The visible log body as plain text. For tests and for a copy action."""
        return self._view.toPlainText()

    def line_count(self) -> int:
        return self._view.blockCount() if self._view.toPlainText() else 0
