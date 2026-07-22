"""HCS viewer — a post-acquisition, well-plate viewer for Squid acquisitions (IMA-185).

A single professional Qt window, isolated from the Squid acquisition software. This tool runs on
acquisitions that are ALREADY on disk (post-acquisition), so there is no live-follow machinery — it
opens a completed scan and lets you navigate it and apply post-processing operators to it.

    drop a Squid acquisition folder
      -> TOP-LEFT: the PROCESS console — a Cellpose-style stack of operators to run on the plate
               (MIP, Record z-stack; more land here as cards) plus a "to be added" roadmap and an
               "Open CLI" button (a standalone stub window; the visible seam to IMA-186's headless
               CLI). Operators gather any parameters through dialogs — MIP prompts for a destination,
               Record prompts for scope + folder — so the pane is self-contained, no tabs.
      -> BOTTOM-LEFT (<= half the display): a low-resolution PLATE OVERVIEW — one cell per well, laid
               out in true plate row-major (A,B,...,Z,AA,...). Each well is HUE-CODED by its PROCESSING
               status (Hongquan Li's record-zstack-viewer palette): grey = not processed, amber =
               processing, blue = done, red-x = failed. The CURRENT well in view is a red box; the
               cursor's well (as you move around) is a red dot. Wheel-zoom + drag-pan; double-click
               opens a well; PRESS-AND-HOLD raises a loupe (IMA-208) that overlays the well's real
               pixels — read from the acquisition's TIFFs, or from the written pyramid once an
               operator has persisted one — magnified relative to the current plate zoom and capped
               at native resolution, with a µm scale bar when the pixel size is known.
      -> RIGHT (>= half): ndviewer_light EMBEDDED (dark-themed) — the per-FOV 4D detail, full height.
               DOUBLE-CLICK a well and its RAW z-stack (all z, all channels) opens here by pointing
               ndviewer at the acquisition's existing TIFFs (register_image with the raw paths) — zero
               bytes copied, nothing written to disk. The z / t sliders are the real acquisition axes.

The plate is the spatial navigator; ndviewer handles the per-FOV z-stack. "Processing" here means
post-processing: MIP is operator #1, and more operators stack behind the same menu (the moment a
second operator lands this is a general HCS viewer, not just a MIP tool).

Design notes:
- ndviewer_light is the embedded detail viewer (its LightweightViewer QWidget + push API); PyQt5 to
  match its stack. PyQt5 is imported here, never in squidmip/__init__, so the pipeline stays Qt-free.
- Nothing is written to the user's disk: the detail view reads the acquisition's own read-only
  TIFFs. Memory is NOT one-well-at-a-time on the plate side: PlateOverview retains the whole plate
  with its CHANNEL AXIS intact — one (C, nr*88, nc*88) NATIVE-DTYPE store per displayed layer — so
  a channel toggle or a contrast drag recomposites from pixels already in RAM instead of re-reading
  or re-projecting anything. That is ~95 MB for a 1536wp at C=4 uint16 (native dtype, so half what
  float32 would cost), and it MULTIPLIES per layer: raw + one operator layer is ~190 MB. Allocated
  lazily, only for a layer that actually receives tiles. On top sits a grid-sized RGB canvas (~36 MB)
  per layer and one transient float32 buffer during a full-resolution recomposite. Bounded by the
  plate format (<=1536 wells), not by z/frame size. What IS one-well-at-a-time is project_plate's
  producer (workers x one ~139 MB well) and the detail viewer's LRU-bounded decoded planes.
- Hit-testing / cell fitting are pure functions (unit-testable); widgets run headless under
  QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import (
    Qt, QProcess, QProcessEnvironment, QRectF, QSocketNotifier, QThread, QTimer, pyqtSignal,
)
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen, QPixmap, QRegion
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenu, QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox,
    QSplitter, QStackedWidget, QStyleFactory, QTabBar, QTabWidget, QVBoxLayout, QWidget,
)

from squidmip._engine import _default_workers, available_projectors
from squidmip._layers import OperationStack
from squidmip._minerva import MINERVA_HOME_ENV as _MINERVA_HOME_ENV
from squidmip._montage import _area_downsample, _hex_to_rgb01, _window, composite
from squidmip._output import parse_well_id
from squidmip._plate import PlateBuildError, build_plate
from squidmip._plate_shape import PlateShapeError

# (`_SUPPORTED_PLATES` and `resolve_plate_format` used to live here. `build_plate` (IMA-214) is now
#  the single format-resolution path — override > measured > declared > inferred — so both were dead
#  leftovers that could only ever disagree with it. Deleted rather than left as a second opinion.)
_CELL = 88                # per-well px in the low-res overview (1536wp -> ~4224x2816)
_PUSH_PX = 512             # per-well px pushed to the ndviewer scan-slider (downsampled -> bounded RAM)
_HDR, _COLH = 46, 30       # left / top label margins (px)
_PAD = 16                  # breathing room around the plate
_VIEWER_WORKERS = min(6, _default_workers())   # adapt to the machine, but CAP at 6: the producer's
                           # peak RAM is ~workers x one-well (~139 MB each on a 1536wp), and projection
                           # throughput scales only sublinearly past ~6 threads — so more workers buys
                           # little speed for linearly more memory. 6 balances both, leaves GUI cores.
_BG = "#070a0f"
_GRID, _RED, _MUTED, _ACCENT = QColor(0, 0, 0), QColor("#ff2d2d"), QColor("#8b98ad"), QColor("#58a6ff")
_SEL_FILL = QColor(88, 166, 255, 90)   # translucent accent wash over a SELECTED well (IMA-221)
_MIN_PREVIEW_BOX_PX = 4    # smallest FOV box (of _CELL) the RAW preview will bother mosaicking
#                            (IMA-253): below this a field is a speck, and reading one plane per
#                            field to draw specks is pure cost. The operator path is unaffected.
_CLICK_SLOP = 3                       # px of travel below which a Shift-drag counts as a click
#                                        (matches the pan threshold, so the two gestures agree)
_CONTROL_BLUE = QColor("#7fd4ff")     # the CONTROL WELL's persistent frame (IMA-248/IMA-260).
#                                       Light blue, and deliberately NOT _RED: the red box is the
#                                       transient current-FOV, the blue frame is a pinned reference.

# --- Legibility floor for read-at-a-distance copy (project-wide constraint) --------------------
# The spec is angular, not typographic: 16 arcmin MINIMUM, 20 arcmin optimal, which this project
# has already converted to 17.3 px at 60 cm and 28.8 px at 1 m. Scaling the floor to 1 m gives
# 28.8 * 16/20 = 23.0 px, so 24 px clears the 16-arcmin minimum at BOTH seating distances, and
# 30 px clears the 20-arcmin optimum at 1 m. Empty-state copy is exactly the text a user reads
# while leaning back from the big monitor, so it is sized for the far case, not the near one.
_EMPTY_BODY_PX = 15                   # 16 arcmin at ~40 cm. The old 24 px assumed a 1 m viewing
#                                       distance on a large monitor; Julio is on a SMALL one and
#                                       called it "huge text". A pane read at desk distance does
#                                       not need the across-the-room size.
_EMPTY_HEAD_PX = 19                   # heading, one step up from body

# The empty exploration pane's copy (IMA-260). Framed as an EXAMPLE of what you might do, never as
# an instruction: Julio asked for "example usage", so the pane shows one concrete path and then
# says out loud that it is only an example. Primary line first (right-click -> Control Well),
# secondary second (Shift-drag). Plain sentences, no UI jargon.
_EMPTY_EXPLORE_HEAD = "Exploration pane"
_EMPTY_EXPLORE_LEDE = "Nothing here yet. Here is an example of how you might use this pane."
_EMPTY_EXPLORE_PRIMARY = (
    "For example, you can right-click a well on the plate and choose Control Well from the menu. "
    "That well opens here and stays, so you can compare the other wells against it.")
_EMPTY_EXPLORE_SECONDARY = (
    "You can also hold Shift and drag across the plate to pick a few wells. They open here in "
    "their own tab.")
_EMPTY_EXPLORE_CODA = "These are only examples. Whatever you open lands in this pane."
_EXPLORE_W = 380                      # pane 3's width on open, in px (see PlateWindow.__init__)

# Processing-status hue coding, adopted from Hongquan Li's record-zstack-viewer plate navigator.
# Deliberately colorblind-safe (blue/amber, never red/green) with a shape cue for failure (the x).
_STATUS = {
    "empty":      QColor("#b7bcc4"),   # not yet processed
    "processing": QColor("#f59e0b"),   # amber — running now
    "done":       QColor("#3b82f6"),   # blue — MIP computed
    "failed":     QColor("#ef4444"),   # red outline + x cross
}
_NDV_DARK = (  # ndviewer defaults to light; theme its Qt chrome dark (bg AND text) to match
    "QWidget{background:#0b0e14;color:#e6edf3;}"
    "QLabel{color:#e6edf3;background:transparent;}"
    "QSlider::groove:horizontal{background:#232b3a;height:4px;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#58a6ff;width:12px;margin:-5px 0;border-radius:6px;}"
    "QPushButton{background:#131824;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;padding:3px 8px;}"
)

# Tab bar for the top-left pane ONLY (its bar sits at the top of that pane, like the plate pane's
# title bar — NOT a global strip across the window). Home tab = the operator list; operators open
# their own UI tab beside it.
_TABS_DARK = (
    "QTabWidget{background:#070a0f;}"
    "QTabWidget::pane{border:1px solid #c9d1d9;background:#070a0f;top:-1px;}"  # thin white outline
    "QTabBar{background:#070a0f;}"                                            # black strip, not white
    "QTabBar::tab{background:#0b0e14;color:#8b98ad;padding:6px 13px;border:1px solid #232b3a;"
    "border-bottom:none;margin-right:2px;font-weight:700;font-size:12px;}"
    "QTabBar::tab:selected{background:#131b2b;color:#e6edf3;}"
)
_CARD_QSS = (   # an operator "card" in the Process pane (Cellpose-style pick-an-operation)
    "QPushButton{background:#0d1420;color:#e6edf3;border:1px solid #232b3a;border-radius:10px;"
    "text-align:left;padding:9px 13px;font-size:13px;}"
    "QPushButton:hover{border-color:#58a6ff;background:#111a2b;}"
    "QPushButton:disabled{color:#57606a;border-color:#1a2130;}"
)
_BTN_QSS = (
    "QPushButton{background:#131824;color:#e6edf3;border:1px solid #232b3a;border-radius:8px;"
    "padding:7px 12px;font-weight:700;} QPushButton:hover{border-color:#58a6ff;}"
    "QPushButton:disabled{color:#57606a;}"
)
_COMBO_QSS = "QComboBox{background:#0d1420;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;padding:5px 8px;}"
_CHECK_QSS = (   # checkbox with a visible white outline on the box
    "QCheckBox{color:#e6edf3;spacing:7px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #c9d1d9;border-radius:3px;background:#0d1420;}"
    "QCheckBox::indicator:checked{background:#58a6ff;border:1px solid #c9d1d9;}"
)
_TERM_QSS = ("QPlainTextEdit{background:#05070b;color:#8bffd0;border:none;"
             "font-family:'SF Mono','Menlo',monospace;font-size:12px;padding:10px;}")
# The plate's right-click dropdown (IMA-260). Sized at 16 px — a menu is read at a glance with the
# cursor already on it, so it does not carry the empty-state copy's read-from-your-chair floor.
_MENU_QSS = ("QMenu{background:#0d1420;color:#e6edf3;border:1px solid #232b3a;font-size:16px;}"
             "QMenu::item{padding:7px 18px;}"
             "QMenu::item:selected{background:#1c2b44;}"
             "QMenu::item:disabled{color:#57606a;}")


def _signal_names(cls) -> tuple:
    """Every pyqtSignal declared on *cls* or its bases, by attribute name.

    ``pyqtSignal`` is a class attribute until Qt binds it per-instance, so the class object is
    where the declarations are discoverable. Excludes ``finished``/``started`` — QThread's own,
    which the retire path connects deliberately and must not tear down.
    """
    from PyQt5.QtCore import pyqtSignal as _sig
    seen, out = set(), []
    for klass in cls.__mro__:
        for name, value in vars(klass).items():
            if name in seen or name in ("finished", "started"):
                continue
            if isinstance(value, _sig) or type(value).__name__ in ("pyqtSignal", "unbound_signal"):
                seen.add(name)
                out.append(name)
    return tuple(out)


def _hline():
    """A thin horizontal divider (a 1px framed line) for separating sections in a pane."""
    from PyQt5.QtWidgets import QFrame as _QFrame
    ln = _QFrame()
    ln.setFrameShape(_QFrame.HLine)
    ln.setStyleSheet("color:#232b3a;background:#232b3a;max-height:1px;")
    return ln
# Strip ANSI CSI/OSC escapes + stray control bytes so shell output renders clean in the QPlainTextEdit
# (we run the shell with TERM=dumb to minimise these, but zsh still emits a few).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x00-\x08\x0e-\x1f]")


class _CmdEdit(QLineEdit):
    """A command input with up/down history recall (so re-running a `squidmip …` line is one key)."""

    def __init__(self, terminal):
        super().__init__()
        self._term = terminal

    def keyPressEvent(self, e):
        h = self._term._history
        if e.key() == Qt.Key_Up and h:
            self._term._hpos = max(0, self._term._hpos - 1)
            self.setText(h[self._term._hpos])
        elif e.key() == Qt.Key_Down and h:
            self._term._hpos = min(len(h), self._term._hpos + 1)
            self.setText(h[self._term._hpos] if self._term._hpos < len(h) else "")
        else:
            super().keyPressEvent(e)


class _Terminal(QWidget):
    """A real, interactive shell embedded in the Process-wells pane — IMA-186's `squidmip` CLI, live.

    A login shell on a pseudo-terminal (so it echoes input and behaves like a real terminal): type a
    command, press Enter, see its output. `squidmip` is aliased to THIS app's interpreter, so the batch
    MIP command runs here even though the console script isn't pip-installed. Pre-seeded with a how-to
    banner (MIP every well; `--tiff` writes FIJI-openable TIFFs). Scrollback is capped (bounded RAM),
    and the shell is killed when the tab or the window closes (no orphan process).

    PTY-backed, so it needs a Unix-y OS; ``build`` falls back to a static command preview elsewhere.
    """

    def __init__(self, cwd: Optional[str], banner: list, setup_cmds: Optional[list] = None, parent=None):
        super().__init__(parent)
        self._pid = None
        self._fd = None
        self._notifier = None
        self._history: list[str] = []
        self._hpos = 0
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._out = QPlainTextEdit()
        self._out.setReadOnly(True)
        self._out.setMaximumBlockCount(4000)   # capped scrollback — output can never grow unbounded
        self._out.setStyleSheet(_TERM_QSS)
        v.addWidget(self._out, 1)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(8, 6, 8, 8)
        rl.setSpacing(6)
        tag = QLabel("$")
        tag.setStyleSheet("color:#58a6ff;font-weight:800;font-family:'SF Mono','Menlo',monospace;")
        self._in = _CmdEdit(self)
        self._in.setStyleSheet(
            "QLineEdit{background:#05070b;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;"
            "padding:6px 8px;font-family:'SF Mono','Menlo',monospace;font-size:12px;}")
        self._in.setPlaceholderText("type a command and press Enter  (e.g. squidmip … --tiff)")
        self._in.returnPressed.connect(self._send)
        rl.addWidget(tag)
        rl.addWidget(self._in, 1)
        v.addWidget(row)
        self._start(cwd, banner, setup_cmds or [])

    def _start(self, cwd, banner, setup_cmds):
        import pty
        shell = os.environ.get("SHELL", "/bin/zsh")
        env = dict(os.environ)
        env["TERM"] = "dumb"        # minimise escape sequences; still a real interactive shell
        env["PS1"] = "$ "
        # put the venv's Scripts/bin on PATH so the `squidmip` console script resolves directly.
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
        try:
            self._pid, self._fd = pty.fork()
        except Exception as e:      # no PTY (e.g. Windows) — degrade to a disabled, informative pane
            self._out.setPlainText(f"(embedded terminal unavailable on this platform: {e})")
            self._in.setEnabled(False)
            return
        if self._pid == 0:          # CHILD → becomes the shell (only chdir/exec between fork and exec)
            try:
                if cwd and os.path.isdir(cwd):
                    os.chdir(cwd)
                os.execvpe(shell, [shell, "-i"], env)
            except Exception:
                os._exit(127)
        import fcntl                # PARENT: read the master fd non-blocking, driven by Qt's event loop
        import struct
        import termios
        try:                        # a wide PTY so long commands don't wrap into garbled cursor escapes
            fcntl.ioctl(self._fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 400, 0, 0))
        except Exception:
            pass
        fl = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        self._notifier = QSocketNotifier(self._fd, QSocketNotifier.Read, self)
        self._notifier.activated.connect(self._read)
        # Banner is DISPLAY text — print it straight into the pane (NOT echo'd through the shell, which
        # duplicates + line-wraps it). setup_cmds (e.g. the squidmip alias) run silently in the shell.
        self._append("\n".join(banner) + "\n")
        for cmd in setup_cmds:
            self._write(cmd + "\n")

    def _read(self):
        try:
            data = os.read(self._fd, 8192)
        except BlockingIOError:
            return                       # notifier fired but no data ready yet — keep listening
        except (OSError, TypeError):
            data = b""                   # EIO / fd closed -> the child shell is gone
        if not data:
            if self._notifier is not None:
                self._notifier.setEnabled(False)
            return
        self._append(data.decode(errors="replace"))

    def _append(self, text: str):
        """Append text to the output pane (ANSI escapes + carriage returns stripped), scrolled to end."""
        text = _ANSI_RE.sub("", text).replace("\r", "")
        cur = self._out.textCursor()
        cur.movePosition(cur.End)
        cur.insertText(text)
        self._out.setTextCursor(cur)
        self._out.ensureCursorVisible()

    def _write(self, s: str):
        if self._fd is not None:
            try:
                os.write(self._fd, s.encode())
            except OSError:
                pass

    def _send(self):
        cmd = self._in.text()
        self._in.clear()
        if cmd.strip():
            self._history.append(cmd)
        self._hpos = len(self._history)
        self._write(cmd + "\n")     # the PTY echoes it back, so it appears after the shell prompt

    def shutdown(self):
        """Kill the shell (and its group) and release the fd. Idempotent; safe to call on tab/window close."""
        if self._notifier is not None:
            self._notifier.setEnabled(False)
            self._notifier = None
        if self._pid:
            import signal
            for killer in (lambda: os.killpg(os.getpgid(self._pid), signal.SIGTERM),
                           lambda: os.kill(self._pid, signal.SIGTERM)):
                try:
                    killer()
                    break
                except OSError:
                    continue
            try:
                os.waitpid(self._pid, os.WNOHANG)
            except OSError:
                pass
            self._pid = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def closeEvent(self, e):
        self.shutdown()
        super().closeEvent(e)


class _ProcTerminal(QWidget):
    """An interactive shell in the pane via QProcess — works on Windows (cmd.exe) AND Unix ($SHELL),
    no PTY needed. Type a command, it runs, output streams back. Not a full VT100 (pipes don't echo,
    so we echo the typed line ourselves), but `squidmip …` and any command work. `squidmip` is aliased
    to this app's interpreter. Used where a PTY is unavailable (i.e. on Windows)."""

    def __init__(self, cwd, banner: list, setup_cmds: list, parent=None):
        super().__init__(parent)
        self._nl = "\r\n" if sys.platform == "win32" else "\n"
        self._history: list[str] = []
        self._hpos = 0
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._out = QPlainTextEdit()
        self._out.setReadOnly(True)
        self._out.setMaximumBlockCount(4000)
        self._out.setStyleSheet(_TERM_QSS)
        v.addWidget(self._out, 1)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(8, 6, 8, 8)
        rl.setSpacing(6)
        tag = QLabel("$")
        tag.setStyleSheet("color:#58a6ff;font-weight:800;font-family:'SF Mono','Menlo',monospace;")
        self._in = _CmdEdit(self)
        self._in.setStyleSheet(
            "QLineEdit{background:#05070b;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;"
            "padding:6px 8px;font-family:'SF Mono','Menlo',monospace;font-size:12px;}")
        self._in.setPlaceholderText("type a command and press Enter  (e.g. squidmip … --tiff)")
        self._in.returnPressed.connect(self._send)
        rl.addWidget(tag)
        rl.addWidget(self._in, 1)
        v.addWidget(row)

        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyRead.connect(self._read)
        self._proc.finished.connect(lambda *a: self._append("\n[shell exited]\n"))
        # put the venv's Scripts/bin on PATH so the `squidmip` console script resolves directly.
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PATH", os.path.dirname(sys.executable) + os.pathsep + env.value("PATH"))
        self._proc.setProcessEnvironment(env)
        if cwd and os.path.isdir(cwd):
            self._proc.setWorkingDirectory(cwd)
        shell = "cmd.exe" if sys.platform == "win32" else os.environ.get("SHELL", "/bin/sh")
        self._proc.start(shell, [])
        self._proc.waitForStarted(3000)
        self._append("\n".join(banner) + "\n")
        for c in setup_cmds:            # e.g. the squidmip alias/doskey — run silently
            self._write(c)

    def running(self) -> bool:
        return self._proc.state() != QProcess.NotRunning

    def _read(self):
        data = bytes(self._proc.readAll())
        self._append(_ANSI_RE.sub("", data.decode(errors="replace")).replace("\r", ""))

    def _append(self, text: str):
        cur = self._out.textCursor()
        cur.movePosition(cur.End)
        cur.insertText(text)
        self._out.setTextCursor(cur)
        self._out.ensureCursorVisible()

    def _write(self, s: str):
        if self.running():
            self._proc.write((s + self._nl).encode())

    def _send(self):
        cmd = self._in.text()
        self._in.clear()
        if cmd.strip():
            self._history.append(cmd)
        self._hpos = len(self._history)
        self._append("> " + cmd + "\n")   # pipes don't echo input, so show it ourselves
        self._write(cmd)

    def shutdown(self):
        if self.running():
            self._proc.kill()
            self._proc.waitForFinished(1500)

    def closeEvent(self, e):
        self.shutdown()
        super().closeEvent(e)


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
        # Scoped dark chrome — palette + stylesheet only, never app-wide (see _dark_palette).
        # NO per-widget Fusion style here: _left_tabs needs it for its TAB STRIP rendering, but a
        # float has no strip, and a Python-owned QStyle on a deleteLater'd widget can be GC'd
        # first — ~QWidget then unpolishes a dangling style (segfault, found by the test suite).
        self.setPalette(_dark_palette())
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 5, 8, 5)
        h.addStretch(1)
        dock = QPushButton("Re-dock")
        dock.setStyleSheet(_BTN_QSS)
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


@dataclass(frozen=True)
class Operation:
    """One post-processing operator declared in ONE place — the 'operation template'. Adding a feature
    is a single entry here plus one ``_build_<x>_tab`` method; the console builds a card + menu item
    from ``label``/``blurb``, clicking it opens the tab that ``build_tab`` (a PlateWindow method name)
    returns, and every status text (progress, done) derives from ``label``. A flat record, not a
    subclass tree — no new operator ever edits scattered texts or the dispatch."""
    key: str
    label: str
    blurb: str
    build_tab: str        # name of the PlateWindow method that builds this operator's UI tab

# The operator registry. MIP is operator #1; append an Operation + write its `_build_*_tab` and both
# the console cards and the Process-well-plates menu grow automatically.
_OPERATIONS = (
    Operation("mip", "Maximum Intensity Projection",
              "Collapse each well's z-stack to one max-intensity image; save a navigable OME-Zarr plate.",
              "_build_mip_tab"),
    Operation("stitch", "Stitch (register + fuse)",
              "Register every FOV of a well against its neighbours and fuse one seamless mosaic "
              "per well, instead of trusting the stage coordinates alone.",
              "_build_stitch_tab"),
    Operation("minerva", "Open in Minerva Author",
              "Export the selected FOVs to Minerva-ingestable OME-TIFFs and open Minerva Author on them.",
              "_build_minerva_tab"),
    # IMA-223/224/225 -- the PLANE-OPS. Unlike mip/stitch these keep z at full depth, so they get
    # _build_plane_op_tab (preview only) rather than _build_run_tab: write_plate's _validate_image
    # accepts Z == 1 only and would fail LOUD on save. Loud is correct; offering the button is not.
    # The blurb said "the microscope's Gaussian PSF ... no explicit kernel". Both halves were
    # false and had been since IMA-247 deleted the reimplementation: the kernel is a VECTORIAL
    # PSF computed from the acquisition's own optics (NA 0.3 on this scope), and it is very much
    # explicit. A card that describes the wrong algorithm is how a user picks the wrong operator.
    Operation("decon", "Deconvolution (Richardson-Lucy)",
              "Sharpen against a vectorial PSF computed from this acquisition's own optics (NA, "
              "emission wavelength, pixel size, z-step) -- not an assumed Gaussian. Richardson-Lucy "
              "is semi-convergent, so the iteration count is chosen by eye against a turbo x-z / "
              "y-z view rather than defaulted.",
              "_build_decon_tab"),
    Operation("bgsub", "Background subtraction",
              "Remove the smooth out-of-focus haze from every plane with a rolling ball (ImageJ's "
              "algorithm). A LAYER: the raw is untouched on disk and one toggle away.",
              "_build_bgsub_tab"),
    Operation("flatfield", "Flat-field correction",
              "Divide out the objective's illumination profile so the corners match the centre. "
              "Needs an illumination profile (.npy) from the stitcher or estimated from the plate.",
              "_build_flatfield_tab"),
)
_OPERATIONS_BY_KEY = {op.key: op for op in _OPERATIONS}

# Roadmap cards shown under "TO BE ADDED", as (label, blurb). Empty: everything currently on the
# roadmap that we're willing to advertise has shipped as a real Operation above. Add an entry when
# a next operator (e.g. the Nautilus agent) is close enough to promise.
_TO_BE_ADDED: list = []


# --- pure geometry (unit-testable, no Qt display) -------------------------------------------

def well_at(rows, cols, by_rc, px: float, py: float, cell_disp: float) -> Optional[dict]:
    """Map a plate pixel (px, py) at *cell_disp* px/well to a cell, or None if out of bounds.

    ``by_rc`` maps (row_index, col_index) -> well_id for acquired wells (else the cell is 'empty').
    Pixels are relative to the plate's top-left (label margins already removed by the caller).
    """
    if px < 0 or py < 0:
        return None
    ci, ri = int(px // cell_disp), int(py // cell_disp)
    if ci >= len(cols) or ri >= len(rows):
        return None
    return {"row_index": ri, "col_index": ci, "row": rows[ri], "col": cols[ci],
            "well_id": by_rc.get((ri, ci))}


def cells_in_rect(rows, cols, by_rc, x0: float, y0: float, x1: float, y1: float,
                  cell_disp: float) -> list:
    """Every ACQUIRED cell whose square meets the drag rect (x0,y0)-(x1,y1), row-major sorted.

    Same plate-pixel space as ``well_at`` (label margins already removed by the caller). The rect
    is NORMALIZED first, so an up-left drag selects exactly what the equivalent down-right drag
    does. Out-of-grid edges clamp instead of inventing cells, and a cell is returned only when
    ``by_rc`` holds a well there — a marquee over a sparse plate never selects the un-acquired
    positions the grey dots mark.

        by_rc = {(0,0):A1, (1,1):B2}          drag (0,0)->(39,39) at 20px/cell
        +-------+-------+
        |  A1   |  A2   |   -> [(0,0), (1,1)]   A2/B1 are plate positions, not acquisitions
        |  (B1) |  B2   |
        +-------+-------+
    """
    if cell_disp <= 0:
        return []
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)      # normalize: any drag direction is equal
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    if hi_x < 0 or hi_y < 0:                             # entirely above/left of the plate
        return []
    c0, c1 = int(max(0.0, lo_x) // cell_disp), int(max(0.0, hi_x) // cell_disp)
    r0, r1 = int(max(0.0, lo_y) // cell_disp), int(max(0.0, hi_y) // cell_disp)
    c1, r1 = min(c1, len(cols) - 1), min(r1, len(rows) - 1)   # clamp at the far edge
    return [(ri, ci) for ri in range(r0, r1 + 1) for ci in range(c0, c1 + 1) if (ri, ci) in by_rc]


def _fit_cell(a: np.ndarray) -> np.ndarray:
    """Resize a 2D plane to EXACTLY (_CELL, _CELL) for the montage tile.

    Area-downsample when larger (the common case: a ~768px tile -> 88); nearest-upscale a tiny
    frame so the tile shape is always (_CELL, _CELL) (guards the <88px-frame crash the review found).
    """
    if a.shape == (_CELL, _CELL):
        return a
    if a.shape[0] >= _CELL and a.shape[1] >= _CELL:
        return _area_downsample(a, _CELL, _CELL)
    yi = (np.arange(_CELL) * a.shape[0]) // _CELL
    xi = (np.arange(_CELL) * a.shape[1]) // _CELL
    return a[yi][:, xi].astype(np.float32)


def _fit_box(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a 2D plane to EXACTLY (h, w) — the arbitrary-target sibling of :func:`_fit_cell`.

    Used to place one FOV into its box inside a multi-FOV mosaic cell (IMA-187), where each box
    is a fraction of _CELL and generally not square. Same policy as _fit_cell: area-downsample
    when shrinking (the normal case — a 2084px frame into a ~15px box), nearest-sample when
    upscaling, so a tiny synthetic frame in a test can never crash the render path.
    """
    h, w = max(1, int(h)), max(1, int(w))
    if a.shape == (h, w):
        return a
    if a.shape[0] >= h and a.shape[1] >= w:
        return _area_downsample(a, h, w)
    yi = (np.arange(h) * a.shape[0]) // h
    xi = (np.arange(w) * a.shape[1]) // w
    return a[yi][:, xi].astype(np.float32)


# The Squid well-plate formats we fit a plate to (well count -> (rows, cols)). An acquisition whose
# format isn't one of these falls back to a present-only grid (see _plate_grid).
_PLATE_DIMS = {4: (2, 2), 6: (2, 3), 12: (3, 4), 24: (4, 6), 96: (8, 12),
               384: (16, 24), 1536: (32, 48)}


def _row_letter(i: int) -> str:
    """0->A, 25->Z, 26->AA, ... (plate row labels)."""
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _plate_grid(wellplate_format) -> Optional[tuple[list, list]]:
    """Full (rows, cols) label grid for a Squid wellplate format, so the plate view shows every
    position evenly spaced (present wells fill; absent stay blank) rather than collapsing gaps.
    Returns None for an unknown/absent format (caller falls back to present-only)."""
    import re
    m = re.search(r"(\d+)", str(wellplate_format or ""))
    dims = _PLATE_DIMS.get(int(m.group(1))) if m else None
    if not dims:
        return None
    nr, nc = dims
    return [_row_letter(i) for i in range(nr)], [str(c) for c in range(1, nc + 1)]


def resolve_plate_root(path) -> tuple[Path, bool]:
    """(path, is_plate): is_plate True when *path* already holds an OME-zarr plate (not a raw
    acquisition); False for a raw acquisition (the case this viewer opens)."""
    p = Path(path)
    if (p / "plate.ome.zarr").is_dir() or (p.name.endswith(".zarr") and (p / "zarr.json").exists()):
        return p, True
    return p, False


def exploration_tab_key(acq_id: str, regions) -> str:
    """Stable, CONTENT-ADDRESSED id for an exploration tab (IMA-205).

    Same acquisition + same SET of regions -> the same key, whichever order the user selected
    them in. That gives dedupe for free: re-selecting the same wells focuses the tab that is
    already open instead of piling up duplicates on every stray drag.

    ``acq_id`` is hashed in deliberately. Well ids repeat across plates ("B2" exists on every
    one), so a key built from regions alone would let a selection on a NEW acquisition dedupe
    onto a stale tab left over from the old one. ``ingest`` also closes exploration tabs, but
    the key has to be safe on its own rather than relying on that.
    """
    uniq = sorted(set(regions))
    if not uniq:
        raise ValueError("an exploration tab needs at least one region")
    digest = hashlib.sha1("\x1f".join([acq_id, *uniq]).encode("utf-8")).hexdigest()[:10]
    return f"exp:{digest}"


def exploration_tab_label(regions) -> str:
    """Human-readable tab title — 'B2–B5 (4)'. The hash is the internal key, never the label."""
    uniq = sorted(set(regions))
    if not uniq:
        return "exploration"
    if len(uniq) == 1:
        return uniq[0]
    return f"{uniq[0]}–{uniq[-1]} ({len(uniq)})"


def operator_layer_key(op_key: str, tab_key: Optional[str]) -> str:
    """Layer id an operator's results are filed under.

    Plate-wide runs keep the bare operator key ("mip") so existing behaviour is byte-identical.
    A run scoped to an exploration tab gets "<op>@<tab_key>" — without this, two tabs running
    the same operator both write into PlateOverview._op_canvas["mip"] and silently overwrite
    each other's tiles."""
    return f"{op_key}@{tab_key}" if tab_key else op_key


def runnable_operators() -> list[str]:
    """Every operator ``run_operator`` can stream live (IMA-226) — read off the ENGINE registry,
    never off ``_OPERATIONS``.

    The two lists are not the same set and never were:

    * ``reference`` is a registered projector with no card, so ``_OPERATIONS_BY_KEY[key].label``
      raised a bare ``KeyError`` out of the event loop the moment anything asked to run it.
    * ``minerva`` is a card that is NOT an operator — it is an export hand-off. Handing its key to
      the engine dies with a raw ``KeyError: unknown projector 'minerva'`` in the status line.

    Both are cured by asking the engine what it can run. A z-reducer and a region operator stream
    through the SAME ``_OperatorWorker`` (it already branches ``project_plate``/``stitch_plate`` on
    ``available_region_operators``), so both belong in one list.
    """
    from squidmip import available_projectors, available_region_operators

    return sorted(set(available_projectors()) | set(available_region_operators()))


def operator_label(key: str) -> str:
    """Human label for an operator: its card's if it has one, else the registry name itself.

    A card is presentation, not capability (IMA-226) — an operator with no card must still be
    runnable and must still name itself in the status line and the layer stack."""
    op = _OPERATIONS_BY_KEY.get(key)
    return op.label if op is not None else key


class _RunningContrast:
    """Per-channel global contrast that updates as wells stream in (histogram over tiles so far).

    Each channel also carries an auto/manual LATCH (IMA-206). The histogram keeps growing while a
    run streams, so an untouched channel keeps auto-scaling; the first time the user drags that
    channel's contrast it latches MANUAL and the next well to land can no longer stomp the window
    they just set. ``set_auto`` unlatches it back onto the running histogram.
    """

    def __init__(self, n_ch: int, dmax: float, pct=(1.0, 99.8), bins=512):
        self._bins, self._dmax, self._pct = bins, max(1.0, float(dmax)), pct
        self._hist = [np.zeros(bins, dtype=np.int64) for _ in range(n_ch)]
        self._manual: dict[int, tuple[float, float]] = {}   # ch -> the window the USER latched
        # ch -> the window the OWNING VIEWER (ndviewer_light) resolved and is rendering with.
        # Deliberately NOT the same dict as _manual: see set_followed.
        self._followed: dict[int, tuple[float, float]] = {}

    @property
    def dmax(self) -> float:
        return self._dmax

    def add(self, ch: int, tile: np.ndarray):
        idx = np.clip((tile.ravel() / self._dmax * self._bins).astype(int), 0, self._bins - 1)
        self._hist[ch] += np.bincount(idx, minlength=self._bins)

    def set_manual(self, ch: int, lo: float, hi: float):
        """Latch *ch* to a user-set window (hi is kept above lo so _window never divides by zero)."""
        self._manual[ch] = (float(lo), float(max(hi, lo + 1)))

    def set_auto(self, ch: int):
        """Unlatch *ch* — it goes back to following the running histogram."""
        self._manual.pop(ch, None)

    def is_manual(self, ch: int) -> bool:
        """Did the USER latch this channel? Never true merely because the viewer autoscaled."""
        return ch in self._manual

    def set_followed(self, ch: int, lo: float, hi: float):
        """Record the window the OWNING VIEWER resolved for *ch* (IMA-261).

        THIS IS NOT A LATCH, AND THE DISTINCTION IS THE WHOLE POINT
        ------------------------------------------------------------
        The first version of this recorded ndv's window by calling ``set_manual``, which read as
        "the user has taken manual control of this channel". It was wrong twice over, and both
        showed on screen:

          * ndv autoscales on its own, at open, before the user has touched anything — so every
            channel came up latched MANUAL and the plate's running histogram was permanently
            overridden. Auto-contrast was dead from the first frame.
          * ``resolve`` puts a manual latch above everything, so under SCOPE_PER_REGION every cell
            resolved to ndv's one global window. All 1536 wells were painted identically while the
            plate still drew the amber "wells NOT comparable" badge over the top. The control did
            nothing and the caveat was a lie.

        A followed window is a SINK recording what the owner is showing. A manual latch is a
        POLICY decision, and only the user makes it — the sink never writes policy back into the
        model. Same numbers, different authority, and the authority is what ``resolve`` reads.
        """
        self._followed[ch] = (float(lo), float(max(hi, lo + 1)))

    def clear_followed(self, ch: int):
        self._followed.pop(ch, None)

    def is_followed(self, ch: int) -> bool:
        return ch in self._followed

    def resolve(self, ch: int, auto: tuple[float, float],
                follow: bool = True) -> tuple[float, float]:
        """THE precedence rule, in one place (IMA-242, extended by IMA-261).

            user latch  >  the owning viewer's window  >  whatever the caller computed

        Every renderer derives its own *auto* window legitimately — the plate from the running
        histogram, a per-region cell from exact percentiles over that cell, the loupe from a
        well's coarse plane. What they must NOT each decide for themselves is whether the user's
        gesture outranks that, because a renderer that forgets to ask is a renderer where the
        control silently does nothing. One rule, one place, three callers.

        *follow* is how a caller says "I am not rendering the viewer's global view". Only
        SCOPE_PER_REGION passes False, and it means exactly what the user asked for by choosing
        per-region: derive this cell's window from this cell's pixels. A user's explicit latch
        still wins even then — that is a decision about the channel, not about the scope.
        """
        if ch in self._manual:
            return self._manual[ch]
        if follow and ch in self._followed:
            return self._followed[ch]
        return auto

    def window(self, ch: int) -> tuple[float, float]:
        """(lo, hi) for this channel: user latch, else the viewer's, else the running histogram."""
        return self.resolve(ch, self._auto_window(ch))

    def _auto_window(self, ch: int) -> tuple[float, float]:
        """The running histogram's window for *ch*, ignoring any latch.

        A DEGENERATE window (hi <= lo) is returned DELIBERATELY when both percentiles land in the
        same bin — ``_window``'s ``span <= 0`` guard renders that black, which is the honest
        answer when there is no contrast.
        """
        h = self._hist[ch]
        tot = h.sum()
        if tot == 0:
            return 0.0, self._dmax
        cdf = np.cumsum(h) / tot
        lo = np.searchsorted(cdf, self._pct[0] / 100.0) / self._bins * self._dmax
        hi = np.searchsorted(cdf, self._pct[1] / 100.0) / self._bins * self._dmax
        # Do NOT widen a collapsed window to lo+1. That +1 is one DATA unit, while a histogram bin
        # is dmax/bins wide (~128 on uint16) — so for a blank, dead or saturated channel both
        # percentiles land in one bin and (v - lo)/1 clips to 1.0: the well rendered FULL WHITE and
        # read as signal. Blank wells are normal on a partially acquired plate. Handing the
        # degenerate window through instead makes _window's guard reachable, and they render black.
        return float(lo), float(hi)

    def windows(self) -> list[tuple[float, float]]:
        return [self.window(ch) for ch in range(len(self._hist))]


# --- contrast scope (IMA-207): how wide a net each contrast window is computed over ------------
#
#   GLOBAL      one window per channel across the whole plate. Wells stay COMPARABLE — a dim well
#               looks dim — but a dim region can be crushed to black beside a bright one.
#   PER_REGION  one window per channel PER CELL. Every region fills its own range, so a dim and a
#               bright region are both readable at once — at the cost of comparability: two wells
#               that look identical may differ by orders of magnitude. That is why the active
#               scope is drawn INTO the plate (see paintEvent) rather than living only in a
#               dropdown that a screenshot would crop out.
#
# Scope is a DISPLAY control, NOT a run parameter. Flipping it re-composites from the native-dtype
# tiles PlateOverview already retains (IMA-206's per-layer store); it never re-runs the plate,
# because a 1536wp run is minutes and that would make the control unusable.
#
# PER_FOV is deliberately absent. It slots into `_scoped_windows`' bucketing when someone wants
# per-field windows inside a mosaic cell — no other change.
SCOPE_GLOBAL = "global"
SCOPE_PER_REGION = "per-region"
SCOPES = (SCOPE_GLOBAL, SCOPE_PER_REGION)

_PCT = (1.0, 99.8)   # clip the darkest 1% / brightest 0.2% so hot pixels don't crush the window


def _pct_window(a: np.ndarray, pct=_PCT) -> tuple[float, float]:
    """EXACT percentile window over *a*.

    Exactness is the point. ``_RunningContrast`` quantizes to a bin ~dmax/bins wide (~128 counts on
    uint16), so a dim region spanning a few hundred counts collapses into two or three bins and its
    window comes out garbage — precisely the region PER_REGION exists to rescue. The histogram
    stays the live during-run approximation; this is what the final render uses.

    A degenerate result (hi <= lo) is returned as-is on purpose: ``_window`` renders it black.
    """
    if a.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(a, pct)
    return float(lo), float(hi)


# --- loupe (IMA-208): press-and-hold magnifier over the plate ------------------------------
#
# The plate montage CANNOT be the loupe's source: a tile is _CELL (88) px per well, a ~47x
# downsample of a ~4168px field (see _fit_cell), so magnifying it yields interpolation, not
# pixels. The loupe therefore reads the real data behind whatever layer is on screen — the
# acquisition's TIFFs in raw mode, a windowed read of the written pyramid otherwise.
#
# Magnification is derived from the CURRENT plate zoom (so it is dynamic, per the spec) and
# capped at native resolution (so it never invents detail):
#
#     s_plate = cd / well_px            screen px per image px, at the current plate zoom
#     s_loupe = min(1.0, MAG * s_plate) screen px per image px inside the inset (cap = native)
#     M       = s_loupe / s_plate       actual magnification, in (1, MAG]
#     L       = coarsest pyramid level whose own pixels are still >= s_loupe
#
# Reading level L instead of level 0 is what keeps this cheap: L is chosen so the level's
# pixels land ~1:1 on the inset's screen pixels, so the crop is a few hundred px per side no
# matter how far out the plate is zoomed.

_LOUPE_PX = 240            # inset size on screen (px)
_LOUPE_MAG = 8.0           # target magnification over the plate's current scale
_LOUPE_HOLD_MS = 350       # press-and-hold dwell before the loupe arms
_LOUPE_SLOP = 3            # cursor may drift this many px while arming (matches the pan threshold)
_LOUPE_CACHE = 8           # decoded crops kept (small: a crop is a few MB, not a whole well)
_LOUPE_MAX_CROP = 2 * _LOUPE_PX   # ceiling on the RETURNED array's side, in px
# Why a ceiling at all, when level selection is supposed to bound the read: a source can run OUT
# of levels. Raw TIFFs have no pyramid on disk (n_levels == 1), and a written field below
# _PYRAMID_MIN_YX collapses to level 0 alone, so loupe_level clamps to 0 and the crop becomes
# inset/s_loupe — the WHOLE field. Measured on the 2084 px synthetic plate at fit: a 1826 px
# crop, 4 channels, 26.7 MB, composited ON THE GUI THREAD at ~118 ms per cursor move, and the
# worker's LRU — keyed on (well, level, y0, x0, h, w), i.e. a new key for every pixel of motion
# — held eight of them (213 MB). A 4168 px field is 4x worse on both counts. The fix is decimation, not truncation: the requested RECTANGLE
# still defines the region the inset covers (truncating it would silently change the
# magnification), but a source returns it at no more than this many samples per side. The inset
# is 240 px on screen; beyond 2x that, nobody can see the difference.


def _fov_of_well(well_id, fovs_per_region=None) -> int:
    """The FOV index the plate addresses for ``well_id`` — the single seam for multi-FOV.

    The plate hit-test resolves a WELL, never a FOV, so today this is always 0: the viewer is
    one-FOV-per-well (the library folded IMA-187, the viewer has not). Everything that needs a
    FOV goes through here rather than writing a bare ``0``, so when the plate grows FOV
    sub-cells there is one place to change and ``test_fov_seam_is_single_fov`` fails loudly
    instead of the loupe silently magnifying FOV 0 of every position."""
    if fovs_per_region:
        fovs = fovs_per_region.get(well_id)
        if fovs:
            return int(fovs[0])
    return 0


def loupe_scale(cd: float, well_px: int, mag: float = _LOUPE_MAG,
                inset_px: int = _LOUPE_PX) -> tuple[float, float]:
    """(s_loupe, M) for a plate showing ``cd`` screen px per well of ``well_px`` image px.

    ``s_loupe`` is clamped to 1.0 — one screen pixel per level-0 image pixel is as far as
    honest magnification goes; past that we would be upsampling, which is the very thing
    the montage already does badly. ``M`` is what the user actually gains, in [1, mag].

    Two lower clamps, both learned the hard way:

    * Once the user has wheel-zoomed the plate PAST native (a well drawn bigger than its own
      pixel count), the 1.0 cap alone would put the inset BELOW the plate's own scale — a
      loupe that shrinks what it points at. Floor at the plate's scale; the caller labels
      that case "native", since there is no detail left to reveal.
    * A fixed target magnification does not survive a 1536-well plate. At fit, a well is ~10
      screen px, so 8x fills only ~85 px of a 240 px inset and the rest would have to come
      from neighbouring wells. Floor at ``inset_px / well_px`` so the inset shows AT MOST one
      whole well — which is also what the gesture means: look closely at *this* well. On a
      1536wp that yields ~22x rather than 8x, still derived entirely from the plate's zoom."""
    well_px = max(1, int(well_px))
    s_plate = max(1e-9, float(cd) / well_px)
    fill_well = float(inset_px) / well_px             # scale at which one well fills the inset
    # Order matters: cap at native FIRST, then floor at the plate's own scale. Capping last
    # would drag a plate that is already past native back down to 1.0 and demagnify.
    s_loupe = max(s_plate, min(1.0, max(mag * s_plate, fill_well)))
    return s_loupe, s_loupe / s_plate


def loupe_level(s_loupe: float, n_levels: int) -> int:
    """Coarsest pyramid level whose native resolution still satisfies ``s_loupe``.

    Level L is downsampled by 2**L, so its pixels carry scale 1/2**L relative to level 0. We
    want the largest L with 2**-L >= s_loupe, i.e. L <= log2(1/s_loupe). Clamped into the
    levels that actually exist (a small field writes a single level — see _PYRAMID_MIN_YX)."""
    s = min(1.0, max(1e-9, float(s_loupe)))
    return int(max(0, min(int(np.floor(np.log2(1.0 / s))), max(0, int(n_levels) - 1))))


def loupe_crop_px(s_loupe: float, level: int, inset_px: int = _LOUPE_PX) -> int:
    """Image pixels to read AT ``level`` to fill an ``inset_px`` square inset."""
    eff = max(1e-9, float(s_loupe) * (2 ** int(level)))   # screen px per level-``level`` px
    return int(max(1, np.ceil(inset_px / eff)))


def loupe_decimation(crop_px: int, max_px: int = _LOUPE_MAX_CROP) -> int:
    """Power-of-two stride that brings a ``crop_px``-wide read down to <= ``max_px`` samples.

    Applied by the SOURCE, after the rectangle is fixed: the region the inset covers is set by
    the crop rect and must not change, only the sample count within it."""
    step = 1
    while crop_px // step > max(1, int(max_px)):
        step *= 2
    return step


def loupe_clamp_crop(y0: int, x0: int, h: int, w: int, ny: int, nx: int):
    """Fit a crop rect inside a ``ny`` x ``nx`` field: shift the ORIGIN in, keep the extent.

    Every source must do this, and the reason it is a free function rather than four lines
    repeated per source is IMA-208's primary bug: ``_ZarrLoupeSource`` clamped and
    ``_RawLoupeSource`` did not, so raw mode — the DEFAULT on every folder open — passed a
    negative origin straight into a numpy slice. ``a[-427:1399]`` is not an error, it is an
    EMPTY array, so the inset said "no pixels here" over the ~75% of every well whose crop
    starts left of or above the field. Clamping the origin (rather than truncating the extent,
    which would return a 1 px sliver at an edge) keeps the inset full near the field border."""
    ny, nx = max(1, int(ny)), max(1, int(nx))
    h, w = max(1, min(int(h), ny)), max(1, min(int(w), nx))
    return max(0, min(int(y0), ny - h)), max(0, min(int(x0), nx - w)), h, w


def loupe_um_per_screen_px(pixel_size_um, s_loupe: float):
    """µm per SCREEN pixel inside the inset, or None when the pixel size isn't trustworthy.

    Returns None rather than a guess. ``_output.py`` writes 1.0 into the multiscales scale for
    BOTH "unknown" and a genuine 1.0 µm/px, so a computed plate cannot distinguish them (see
    TODOS.md) — callers pass None for that case. A microscopy tool that displays a confidently
    wrong micron figure is worse than one that admits it doesn't know."""
    if pixel_size_um is None:
        return None
    p = float(pixel_size_um)
    if not np.isfinite(p) or p <= 0:
        return None
    return p / max(1e-9, float(s_loupe))


def _nice_scale_um(rough: float) -> float:
    """Round a scale-bar length to a 1/2/5 x 10^n figure, the way a microscope overlay would."""
    rough = max(1e-6, float(rough))
    decade = 10.0 ** np.floor(np.log10(rough))
    for step in (1.0, 2.0, 5.0, 10.0):
        if rough <= step * decade:
            return step * decade
    return 10.0 * decade


def _fmt_um(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:g} mm"
    return f"{v:g} µm" if v >= 1 else f"{v * 1000:g} nm"


# IMA-242: `_composite_rgb` and `_percentile_window` used to live here — a second compositor and a
# second percentile rule, each a hand-synced twin of `composite` and `_pct_window`. They had drifted
# apart in exactly the way that shape always drifts:
#
#   * `_percentile_window` widened a degenerate window to (lo, lo + 1); `_pct_window` deliberately
#     does NOT, because +1 is one DATA unit and (v - lo)/1 clips to 1.0 — a blank or saturated
#     channel rendered FULL WHITE and read as signal. The loupe had the bug the plate had fixed.
#   * `_composite_rgb` took no channel mask, so unticking a channel removed it from the plate and
#     left it in the loupe.
#   * Neither consulted the manual latch, so dragging a contrast slider moved the plate and left
#     the loupe showing the old window forever.
#
# Both are now gone. `composite` is the one compositor, `_pct_window` the one percentile rule, and
# `_RunningContrast.resolve` the one place the manual-outranks-auto precedence is decided.


_LOUPE_WIN_LOCK = threading.Lock()   # guards the per-source window memo (worker thread writes)


class _LoupeSource:
    """Where the loupe's real pixels come from for the layer currently on the plate.

    Availability is per (source, WELL) — never per layer key. A layer key cannot express what
    is actually on disk: ``OperationStack.add`` dedupes by key, so a saved run and a later
    unsaved preview collapse into one "mip" layer while ``_processed_plate`` still points at
    the older save. Ask the source about the specific well instead."""

    n_levels = 1
    well_px = 1
    pixel_size_um = None

    def available(self, well_id) -> tuple[bool, str]:
        """(ok, reason-if-not). ``reason`` is shown to the user verbatim."""
        return False, "no pixel source"

    def read_crop(self, well_id, level, y0, x0, h, w):
        """(C, y, x) crop at ``level``, CLAMPED into the field (see loupe_clamp_crop) and
        decimated to at most _LOUPE_MAX_CROP samples per side. Runs on the worker thread."""
        raise NotImplementedError

    def coarse(self, well_id):
        """A small whole-field (C, y, x) plane used ONLY to derive the contrast window."""
        raise NotImplementedError

    def window(self, well_id):
        """Per-channel contrast window for a well, mirroring the tile's rule.

        Computed HERE, on the loupe worker thread, and memoised per well — never on the GUI
        thread. It used to be derived in ``_on_loupe_crop`` by calling ``coarse()``, which for
        raw meant decoding a whole TIFF plane inside a paint-driven slot AND touching the same
        plane cache the worker was writing (two threads, no lock, one well's pixels labelled as
        another's). One owner, one thread."""
        with _LOUPE_WIN_LOCK:
            cache = self.__dict__.setdefault("_win_cache", {})
            hit = cache.get(well_id)
        if hit is not None:
            return hit
        coarse = self.coarse(well_id)
        win = [_pct_window(coarse[c]) for c in range(coarse.shape[0])]
        with _LOUPE_WIN_LOCK:
            cache[well_id] = win
        return win


class _RawLoupeSource(_LoupeSource):
    """Raw-acquisition source: the loupe works the moment a folder is open, before any operator.

    Reads the same representative plane per channel that _PreviewWorker already reads, so the
    inset shows exactly the data the raw plate tile was built from. Individual TIFFs hold one
    plane per file and aren't tiled, so a crop means decoding that plane — hence the one-well
    plane cache. Bounded to a single well's channels (~C x frame bytes)."""

    def __init__(self, reader, meta, fov_of):
        self._reader, self._meta, self._fov_of = reader, meta, fov_of
        ny, nx = meta["frame_shape"]
        self.well_px = int(min(ny, nx))
        self.n_levels = 1                      # raw TIFFs have no pyramid ON DISK
        self.pixel_size_um = meta.get("pixel_size_um")
        self._channels = [c["name"] for c in meta["channels"]]
        zs = meta["z_levels"]
        self._z = zs[len(zs) // 2]             # mid plane, as the preview does
        self._lock = threading.RLock()         # _planes is touched by the worker AND the GUI thread
        self._cache_key = None
        self._cache = None
        self._coarse: dict[str, np.ndarray] = {}

    def available(self, well_id) -> tuple[bool, str]:
        if well_id in self._meta["regions"]:
            return True, ""
        return False, "no image for this well"

    def _planes(self, well_id):
        """The well's (C, y, x) planes, decoded once and cached.

        Held under a lock for the whole check-decode-publish sequence. Unsynchronised, the two
        callers (worker thread reading a crop, GUI thread deriving a window) could interleave
        between the key test and the store and hand back ANOTHER well's pixels labelled as the
        well under the cursor — a wrong-image bug in a microscopy tool, not a glitch. The GUI
        thread no longer calls in at all (see _LoupeSource.window), but the lock stays: the class
        must be correct for its callers, not for today's call sites."""
        with self._lock:
            if self._cache_key != well_id:
                fov = self._fov_of(well_id)
                planes = np.stack([
                    np.asarray(self._reader.read(well_id, fov, ch, self._z))
                    for ch in self._channels])
                self._cache, self._cache_key = planes, well_id
            return self._cache

    def read_crop(self, well_id, level, y0, x0, h, w):
        """Level is always 0 here — raw has no pyramid — so the whole burden of bounding the
        work falls on decimation. At plate fit the rect IS most of the field (2084 px on the
        synthetic plate); area-averaging it down to <= _LOUPE_MAX_CROP happens HERE, on the
        worker thread, so what crosses to the GUI thread to be composited is a 456 px square
        (3.3 MB, 11 ms) instead of a 1826 px one (26.7 MB, 118 ms) — which is also what the
        worker's LRU then caches."""
        p = self._planes(well_id)
        ny, nx = p.shape[-2], p.shape[-1]
        y0, x0, h, w = loupe_clamp_crop(y0, x0, h, w, ny, nx)   # NEGATIVE origin -> empty slice
        crop = p[:, y0:y0 + h, x0:x0 + w]
        step = loupe_decimation(max(h, w))
        if step == 1:
            return crop
        oh, ow = max(1, h // step), max(1, w // step)
        # float32, not _area_downsample's float64 default: the compositor casts to float32 anyway,
        # and this array crosses a thread boundary and sits in the worker's LRU.
        return np.stack([_area_downsample(crop[c], oh, ow).astype(np.float32, copy=False)
                         for c in range(crop.shape[0])])

    def coarse(self, well_id):
        if well_id not in self._coarse:
            p = self._planes(well_id)
            self._coarse[well_id] = np.stack(
                [_area_downsample(p[c], _CELL, _CELL) for c in range(p.shape[0])])
        return self._coarse[well_id]


class _ZarrLoupeSource(_LoupeSource):
    """Written-plate source: a WINDOWED tensorstore read of one pyramid level.

    Deliberately NOT _ComputedPlateWorker._read, which pulls a whole plane (~139 MB per well at
    level 0 on a 1536wp) — right for its one-pass streaming job, ruinous for a gesture that
    re-reads as the cursor moves. Arrays are chunked (1, 1, 1, <=1024, <=1024) (_zarr_store)
    precisely so a viewer can read a region, so a loupe crop touches a handful of chunks.

    ``written`` is the set of wells this run has actually persisted. It grows as wells land, so
    the loupe works on completed wells DURING a long run, and a subset save / failed well is
    reported as "not written yet" instead of magnifying some other well's pixels."""

    def __init__(self, base, path_of, fov_of, levels, well_px, pixel_size_um, written=None):
        self._base = str(base)
        self._path_of, self._fov_of = path_of, fov_of
        self._levels = list(levels) if levels is not None else None   # None -> discover on first use
        self.n_levels = max(1, len(self._levels)) if self._levels else 1
        self.well_px = int(well_px)
        self.pixel_size_um = pixel_size_um
        self._written = written                # None = every well (a plate opened from disk)
        self._handles: dict[tuple, object] = {}
        self._coarse: dict[str, np.ndarray] = {}

    def mark_written(self, well_id):
        """A well just landed on disk. Availability grows DURING a run — which is exactly when
        someone is watching the plate fill and wants to glance at what already finished."""
        if self._written is not None:
            self._written.add(well_id)

    def available(self, well_id) -> tuple[bool, str]:
        if self._written is not None and well_id not in self._written:
            return False, "not written yet"
        if self._path_of(well_id) is None:
            return False, "no image for this well"
        return True, ""

    def _resolve_levels(self, well_id):
        """Read the field's multiscales once, to learn how many pyramid levels exist.

        Deferred because a run that is still writing hasn't declared its levels yet — and how
        many there are depends on the field size (_PYRAMID_MIN_YX collapses small fields to a
        single level, which is exactly what the test fixtures hit)."""
        if self._levels is not None:
            return self._levels
        field = f"{self._base}/{self._path_of(well_id)}/{self._fov_of(well_id)}"
        try:
            ome = json.loads((Path(field) / "zarr.json").read_text())["attributes"]["ome"]
            self._levels = [ds["path"] for ds in ome["multiscales"][0]["datasets"]]
        except Exception:
            self._levels = ["0"]               # a field always has a full-res array 0
        self.n_levels = max(1, len(self._levels))
        return self._levels

    def _open(self, well_id, level):
        levels = self._resolve_levels(well_id)
        level = max(0, min(int(level), len(levels) - 1))
        key = (well_id, level)
        if key not in self._handles:
            import tensorstore as ts
            path = f"{self._base}/{self._path_of(well_id)}/{self._fov_of(well_id)}/{levels[level]}"
            self._handles[key] = ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}}).result()
        return self._handles[key]

    def read_crop(self, well_id, level, y0, x0, h, w):
        arr = self._open(well_id, level)
        ny, nx = arr.shape[-2], arr.shape[-1]
        # Clamp the ORIGIN so the window stays whole near an edge (shift it in), rather than
        # truncating the extent — clamping y0 to ny-1 first would return a 1px sliver.
        y0, x0, h, w = loupe_clamp_crop(y0, x0, h, w, ny, nx)
        # A field below _PYRAMID_MIN_YX writes level 0 alone, so level selection cannot bound
        # this read; stride it in tensorstore so the I/O itself shrinks, not just the result.
        step = loupe_decimation(max(h, w))
        return np.asarray(
            arr[0, :, 0, y0:y0 + h:step, x0:x0 + w:step].read().result())

    def coarse(self, well_id):
        if well_id not in self._coarse:
            arr = self._open(well_id, self.n_levels - 1)          # coarsest level = cheapest
            self._coarse[well_id] = np.asarray(arr[0, :, 0].read().result())
        return self._coarse[well_id]


class _LoupeWorker(QThread):
    """Serves loupe crops off the GUI thread, coalescing to the NEWEST request.

    Only the latest cursor position matters: if the user sweeps across three wells while a read
    is in flight, the two intermediate reads are worthless. One pending slot (overwritten by
    each new request) IS the coalescing. Results carry the generation they were asked for, so a
    late arrival for a stale position is dropped by the widget rather than flashing."""

    ready = pyqtSignal(int, str, object, object, object)  # (gen, well, crop|None, window|None, err)

    def __init__(self, source: _LoupeSource):
        super().__init__()
        self._source = source
        self._cv = threading.Condition()
        self._pending = None
        self._stop = False
        self._cache: dict[tuple, np.ndarray] = {}
        self._order: list[tuple] = []

    def request(self, gen, well_id, level, y0, x0, h, w):
        with self._cv:
            self._pending = (gen, well_id, level, y0, x0, h, w)
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _cached(self, key):
        hit = self._cache.get(key)
        if hit is not None:
            self._order.remove(key)
            self._order.append(key)
        return hit

    def _store(self, key, val):
        self._cache[key] = val
        self._order.append(key)
        while len(self._order) > _LOUPE_CACHE:
            self._cache.pop(self._order.pop(0), None)

    def run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                gen, well_id, level, y0, x0, h, w = self._pending
                self._pending = None
            key = (well_id, level, y0, x0, h, w)
            try:
                crop = self._cached(key)
                if crop is None:
                    crop = self._source.read_crop(well_id, level, y0, x0, h, w)
                    self._store(key, crop)
                # The contrast window belongs on this side too: deriving it on the GUI thread
                # meant a paint-driven slot could decode a whole TIFF plane (IMA-208).
                try:
                    win = self._source.window(well_id)
                except Exception:
                    win = None                        # the widget falls back to a flat window
                self.ready.emit(gen, well_id, crop, win, None)
            except Exception as e:                    # a racing writer / deleted plate / bad path
                self.ready.emit(gen, well_id, None, None, f"{type(e).__name__}: {e}")


# --- plate overview widget (one cell per well; hue-coded status; fit-to-view) ---------------

class PlateOverview(QWidget):
    """The low-res plate: an RGB canvas of MIP tiles, a per-well status hue, a red box, and a
    press-and-hold LOUPE that overlays real acquisition pixels for the well under the cursor
    (IMA-208 — the montage itself is far too coarse to magnify; see the loupe block above).

    The RGB canvas is only what is CURRENTLY shown. What the widget actually owns (IMA-206) is a
    per-layer ``(C, nr*_CELL, nc*_CELL)`` native-dtype store — the plate with its channel axis
    still intact — plus a channel mask and a per-channel contrast window. Producers hand over
    per-channel tiles and this widget does all compositing, so toggling a channel or dragging its
    contrast re-composites from the retained pixels: no reader I/O, no re-projection, ever.
    """

    hovered = pyqtSignal(str)              # region id (or "" off-plate), for the window's readout
    wellActivated = pyqtSignal(str, int)   # (well_id, fov_index) double-clicked -> load in ndviewer
    selectionChanged = pyqtSignal(list)    # acquired well ids the operator picked (row-major)
    controlRequested = pyqtSignal(str)     # right-click menu: set this region as the CONTROL WELL
                                           # ("" = clear it). The plate ASKS; the window owns the
                                           # answer and hands it back via set_control — the widget
                                           # never sets its own frame (IMA-248's one-owner rule).
    marqueeSelected = pyqtSignal(list)     # ...and specifically by a Shift-DRAG: opens an exploration
                                           # tab (IMA-205). Shift+CLICK refines the selection one well
                                           # at a time and deliberately does NOT fire it — otherwise
                                           # every corrective click spawns another tab.

    def __init__(self, rows, cols, wells: dict, layout: Optional[dict] = None):
        """``wells``: (row_index, col_index) -> well_id for every acquired well (drawn grey until
        processed). Tiles/status arrive as an operator runs.

        ``layout`` (IMA-253) is ``{(row_index, col_index): (x, y, w, h)}`` in GRID UNITS for a
        holder whose cells are placed by real geometry rather than by a uniform pitch -- a freeform
        tissue slide, where each region's cell is its own mosaic's bounding box. ``None`` is the
        uniform grid every well plate is, and keeps the single-blit fast path a 1536-well plate
        needs. Cells absent from the map fall back to their nominal ``(c, r, 1, 1)`` square, which
        is what an EMPTY slot (no stage coordinates to place it by) can honestly be drawn as.
        """
        super().__init__()
        self._rows, self._cols = list(rows), list(cols)
        self._layout: Optional[dict] = ({tuple(k): tuple(float(v) for v in val)
                                         for k, val in layout.items()} if layout else None)
        self._nr, self._nc = len(self._rows), len(self._cols)
        self._by_rc: dict[tuple, str] = dict(wells)            # every acquired well (for status + hit-test)
        self._status: dict[tuple, str] = {rc: "empty" for rc in wells}
        self._tiles: set[tuple] = set()                        # cells that have a tile painted (any layer)
        self._tiles_by_layer: dict[str, set] = {}              # layer -> cells with an image there
        self._canvas = QImage(self._nc * _CELL, self._nr * _CELL, QImage.Format_RGB888)
        self._canvas.fill(QColor(_BG))
        self._final = None            # global-contrast recomposite of the ACTIVE layer (or None)
        # Layer stack render: the base ("raw") is self._canvas; each operator draws into its own
        # per-layer canvas/final. self._active is the layer the plate currently shows (LayersTab picks
        # it via set_active_layer). Keeps memory to one small montage-canvas per layer used.
        self._op_canvas: dict[str, QImage] = {}
        self._op_final: dict[str, QImage] = {}
        self._final_arr: dict[str, np.ndarray] = {}   # keeps each recomposited RGB alive: QImage
        #                                               WRAPS the numpy buffer, it does not copy it
        self._active = "raw"
        # --- the channel axis (IMA-206) — set_channels declares it; empty until then -----------
        self._labels: list[str] = []      # channel display names, for the channel bar
        self._colors = None               # (C, 3) float RGB, the RESOLVED display_color per channel
        self._dtype = np.uint16           # store dtype: the acquisition's native dtype (half the RAM)
        self._store: dict[str, np.ndarray] = {}   # layer -> (C, nr*_CELL, nc*_CELL), allocated lazily
        # --- what a contrast change is allowed to touch (IMA-261) ------------------------------
        # A contrast window is a POINT transform, so it commutes with subsampling: windowing the
        # display-sized thumbnail is bit-identical to windowing the whole plate and subsampling
        # afterwards (see squidmip._montage._window_lut). The only thing that must be re-derived
        # per tick is therefore the composite of the DISPLAY-SIZED buffer. These two caches hold
        # everything upstream of that, so a drag re-reads no store and re-percentiles nothing:
        self._disp: dict[str, tuple] = {}      # layer -> (step, contiguous (C, h, w) thumbnail)
        self._cell_auto: dict[str, dict] = {}  # layer -> {(ri, ci): [per-channel AUTO window]}
        #                                        SCOPE_PER_REGION's exact percentiles, which depend
        #                                        on the PIXELS only — never on the contrast.
        self._mask = None                 # (C,) bool: which channels composite into the plate
        self._contrast = None             # _RunningContrast: global per-channel window + auto/manual
        self._scope = SCOPE_GLOBAL        # contrast scope (IMA-207): a DISPLAY control. A flip
        #                                   re-composites from the store above — it never re-runs.
        self._full = QTimer(self)         # coalesces the full-res recomposite behind a gesture
        self._full.setSingleShot(True)    # (a drag repaints at DISPLAY res; full-res lands once)
        self._full.timeout.connect(self._on_full_timeout)
        self._scaled = None           # cached pixmap of (final|canvas) scaled to the current zoom;
        self._scaled_cd = None        # rebuilt only when zoom (cd) or the source image changes — so
        #                               a hover/pan repaint blits 1:1 instead of re-resampling 12 MP.
        self._cd = float(_CELL)       # displayed px/well (fit baseline, then wheel-zoomed)
        self._ox = self._oy = _PAD    # top-left of the plate within the widget (pan-able)
        self._hover = None
        self._sel = None              # well selected from the ndviewer FOV slider
        # SELECTION (IMA-221) is a DIFFERENT concept from _sel above: _sel is "the one well the
        # detail viewer is showing" (red box, driven by the FOV slider); _selection is "the set the
        # operator picked" (tint, driven by Shift-gestures). Never merge them — the red box must
        # survive selecting, and selecting must survive scrubbing.
        self._selection: set = set()  # acquired (row_index, col_index) the user picked. A SET:
        #                               paintEvent membership-tests it once per cell, 1536x on a 1536wp.
        self._marquee = None          # (x0, y0, x1, y1) widget px while a Shift-drag is in flight
        self._marquee_add = False     # this drag unions (Shift+Alt) rather than replaces
        self._control = None          # (row_index, col_index) of the CONTROL WELL, or None. A
        #                               MIRROR of PlateWindow._control_well, written only by
        #                               set_control — the window owns the identity (IMA-248).
        self._press = None            # (x, y, ox, oy) at left-press, for drag-to-pan
        self._panning = False
        self._user_view = False       # True once the user wheel-zooms/pans (stop auto-fitting)
        self._boxes: dict = {}        # (region, fov) -> (top, left, h, w) in cell px; {} = single-FOV
        self._boxed_regions: set = set()   # regions whose cell holds a LETTERBOXED mosaic, not one tile
        # -- carrier geometry (IMA-220, redrawn for IMA-253: geometry, not a photograph) --
        self._carrier = None          # the _plate.PlateGeometry to draw the holder outline from
        self._carrier_slide = False   # slot-shaped cells (a slide carrier) vs round wells
        self._tile_rgn = None         # cached QRegion of cells that HAVE an image, at pan origin
        self._tile_rgn_key = None     # (cd, active layer, n tiled cells) the cached region was built for
        # -- loupe (IMA-208) --
        self._loupe_src = None        # _LoupeSource for the ACTIVE layer, or None (loupe disabled)
        self._loupe_worker = None
        self._loupe = None            # armed/live state dict, or None when idle
        self._loupe_gen = 0           # bumped per request; late results for older gens are dropped
        self._loupe_img = None        # QImage currently shown in the inset
        self._loupe_note = ""         # user-visible reason when the loupe can't show pixels
        self._loupe_win = {}          # well_id -> per-channel window, mirroring the tile's rule
        self._loupe_colors = None     # (C, 3) float RGB, set with the source
        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.setInterval(_LOUPE_HOLD_MS)
        self._hold.timeout.connect(self._arm_loupe)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)   # so focusOutEvent can actually fire (see below)
        self.setMinimumSize(240, 200)

    # -- loupe wiring --
    def set_loupe_source(self, source, colors=None):
        """Point the loupe at the data behind the ACTIVE layer. ``None`` disables the gesture.

        Called whenever what the plate is showing changes identity — a new acquisition, an
        operator run persisting, a layer switch, a preview superseding a saved run. Re-pointing
        is what stops a stale run's pixels appearing under a newer run's tiles."""
        self._dismiss_loupe()
        if self._loupe_worker is not None:
            self._loupe_worker.stop()
            self._loupe_worker.wait(2000)
            self._loupe_worker = None
        self._loupe_src = source
        self._loupe_colors = colors
        self._loupe_win.clear()
        if source is not None:
            self._loupe_worker = _LoupeWorker(source)
            self._loupe_worker.ready.connect(self._on_loupe_crop)
            self._loupe_worker.start()

    def _arm_loupe(self):
        """Hold timer fired: the press became a loupe. Only reachable while still ARMED."""
        self._hold.stop()             # a pending fire must never re-arm and blank a LIVE loupe:
                                      # _arm_loupe clears _loupe_img, so a second arm 350 ms after
                                      # the first shows an empty inset until the re-read lands.
        if self._press is None or self._panning:
            return
        x, y = self._press[0], self._press[1]
        c = self._cell(x, y)
        if not c or not c["well_id"]:
            return
        self._loupe = {"well": c["well_id"], "x": x, "y": y}
        self._loupe_img, self._loupe_note = None, ""
        self._request_loupe(x, y)
        self.update()

    def _dismiss_loupe(self):
        if self._loupe is not None or self._loupe_img is not None or self._loupe_note:
            self._loupe = self._loupe_img = None
            self._loupe_note = ""
            self.update()

    def _loupe_geometry(self, x, y):
        """Map a widget point to (well_id, level, crop rect, s_loupe, M) — or None if off-plate."""
        src = self._loupe_src
        c = self._cell(x, y)
        if src is None or not c or not c["well_id"]:
            return None
        s_loupe, mag = loupe_scale(self._cd, src.well_px)
        level = loupe_level(s_loupe, src.n_levels)
        crop = loupe_crop_px(s_loupe, level)
        # cursor -> position within the cell -> image px at level 0 -> image px at ``level``
        ax, ay = self._ox + _HDR, self._oy + _COLH
        fy = (y - (ay + c["row_index"] * self._cd)) / max(1e-9, self._cd)
        fx = (x - (ax + c["col_index"] * self._cd)) / max(1e-9, self._cd)
        span = max(1, src.well_px >> level)
        cy, cx = int(fy * span), int(fx * span)
        # Clamp HERE as well as in the source. Two reasons beyond belt-and-braces: the request
        # that reaches the worker is then always a rectangle that exists, and a hold near a field
        # edge produces the SAME key as the cursor drifts, so the LRU hits instead of decoding a
        # fresh full-field crop per pixel of motion.
        y0, x0, h, w = loupe_clamp_crop(cy - crop // 2, cx - crop // 2, crop, crop, span, span)
        return c["well_id"], level, (y0, x0, h, w), s_loupe, mag

    def _request_loupe(self, x, y):
        geo = self._loupe_geometry(x, y)
        if geo is None:                    # dragged onto the margin / an un-acquired cell
            if self._loupe_img is not None or not self._loupe_note:
                self._loupe_img, self._loupe_note = None, "no well here"
                self.update()
            return
        if self._loupe_worker is None:
            return
        well, level, (y0, x0, h, w), _s, _m = geo
        ok, why = self._loupe_src.available(well)
        if not ok:
            self._loupe_img, self._loupe_note = None, why
            self.update()
            return
        self._loupe_gen += 1
        self._loupe_worker.request(self._loupe_gen, well, level, y0, x0, h, w)

    def _on_loupe_crop(self, gen, well_id, crop, window, error):
        """A crop landed. Drop it unless it is the newest request and the loupe is still up.

        Everything expensive already happened on the worker thread: this slot only windows and
        colours a <= _LOUPE_MAX_CROP square. It must stay that way — it runs inside the paint
        loop of a widget the user is dragging across."""
        if gen != self._loupe_gen or self._loupe is None:
            return
        if error is not None or crop is None or crop.size == 0:
            self._loupe_img, self._loupe_note = None, error or "no pixels here"
            self.update()
            return
        # Mirror the TILE's contrast rule on the WELL's pixels (computed by the source, per well)
        # — never percentiles of the crop under the cursor, which would make brightness lurch as
        # the cursor moves and make the inset look like different data.
        #
        # That AUTO window is then resolved through the plate's one contrast model (IMA-242), so a
        # channel the user latched with the slider shows the user's window here too. Before, the
        # loupe kept its own memo and the inset went on displaying the pre-drag contrast — two
        # representations of one truth, never synced.
        auto = window if window is not None else self._loupe_win.get(well_id)
        if auto is None:
            auto = [(0.0, 1.0)] * crop.shape[0]
        self._loupe_win[well_id] = auto              # memo the AUTO window, never the resolved one
        win = ([self._contrast.resolve(c, auto[c]) for c in range(len(auto))]
               if self._contrast is not None else list(auto))
        colors = self._loupe_colors
        if colors is None:
            colors = np.ones((crop.shape[0], 3), np.float32)
        # The same compositor the plate uses, with the same channel mask: unticking a channel must
        # remove it from the inset as well, or the loupe contradicts the plate it sits on top of.
        planes = np.stack([crop[c].astype(np.float32) for c in range(crop.shape[0])])
        mask = self._mask if (self._mask is not None
                              and len(self._mask) == crop.shape[0]) else None
        rgb = composite(planes, colors, win, mask)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()   # copy: rgb is transient
        self._loupe_img, self._loupe_note = img, ""
        self.update()

    def _fit(self):
        """Reset the view: the whole plate fits the widget, centered (zoom = 1)."""
        if self._nr == 0 or self._nc == 0:
            return
        w, h = self.width(), self.height()
        self._cd = max(2.0, min((w - _HDR - 2 * _PAD) / self._nc, (h - _COLH - 2 * _PAD) / self._nr))
        self._ox = max(_PAD, (w - _HDR - self._nc * self._cd) / 2)
        self._oy = max(_PAD, (h - _COLH - self._nr * self._cd) / 2)

    def _fit_cd(self) -> float:
        w, h = self.width(), self.height()
        return max(2.0, min((w - _HDR - 2 * _PAD) / self._nc, (h - _COLH - 2 * _PAD) / self._nr))

    def _canvas_for(self, layer: str) -> QImage:
        if layer == "raw":
            return self._canvas
        cv = self._op_canvas.get(layer)
        if cv is None:
            cv = QImage(self._nc * _CELL, self._nr * _CELL, QImage.Format_RGB888)
            cv.fill(QColor(_BG))
            self._op_canvas[layer] = cv
        return cv

    def _active_source(self) -> QImage:
        return self._final or self._canvas_for(self._active)

    # -- the channel axis: store, mask, per-channel contrast (IMA-206) --
    def set_channels(self, labels, colors: np.ndarray, dtype=np.uint16, pct=(1.0, 99.8)):
        """Declare the acquisition's channels — the per-channel store/mask/contrast start here.

        *colors* is the (C, 3) float RGB of the RESOLVED ``display_color`` (the acquisition's YAML
        first, the wavelength fallback map second — resolve_channels already settled that), so the
        plate is tinted exactly the way every other compositing site tints it.
        """
        self._labels = [str(x) for x in labels]
        self._colors = np.asarray(colors, dtype=np.float32)
        self._dtype = np.dtype(dtype)
        self._mask = np.ones(len(self._labels), dtype=bool)     # every channel on by default (OV8)
        dmax = float(np.iinfo(self._dtype).max) if self._dtype.kind in "ui" else 1.0
        self._contrast = _RunningContrast(len(self._labels), dmax, pct=pct)
        self._store.clear()

    def _store_for(self, layer: str) -> Optional[np.ndarray]:
        """The layer's (C, H, W) plate store, allocated on first tile (one per layer, lazily —
        each layer that supports toggling costs its own ~95 MB at 1536wp x 4ch uint16)."""
        if self._colors is None:
            return None
        st = self._store.get(layer)
        if st is None:
            st = np.zeros((len(self._colors), self._nr * _CELL, self._nc * _CELL), self._dtype)
            self._store[layer] = st
        return st

    def channel_windows(self) -> list:
        """The effective (lo, hi) per channel — the latched manual window, else the running one."""
        return self._contrast.windows() if self._contrast is not None else []

    # -- contrast scope (IMA-207) --
    def contrast_scope(self) -> str:
        return self._scope

    def set_contrast_scope(self, scope: str) -> bool:
        """Switch how wide a net each contrast window is computed over, and REPAINT — never re-run.

        Returns False when there is nothing retained to re-composite yet; the scope is still
        recorded, so the next tile to land renders under it.
        """
        if scope not in SCOPES:
            raise ValueError(f"unknown contrast scope {scope!r}; expected one of {SCOPES}")
        if scope == self._scope:
            return False
        self._scope = scope
        self.update()                 # the badge changes even with nothing to re-composite
        if self._store.get(self._active) is None:
            return False
        self.recomposite(self._active)
        return True

    def _invalidate_pixels(self, layer: str, rc=None):
        """The layer's STORE changed. Drop everything derived from its pixels.

        Called from the one place that writes pixels (``add_tile``) and from ``reset_layer``.
        These caches are keyed on PIXELS alone, so a contrast change must never come through
        here — keeping the two invalidations apart is what lets a drag re-read nothing.

        *rc* narrows the per-cell percentile drop to the one cell that was written; the display
        thumbnail is a whole-plate copy, so it always goes.
        """
        self._disp.pop(layer, None)
        if rc is None:
            self._cell_auto.pop(layer, None)
        else:
            self._cell_auto.get(layer, {}).pop(rc, None)

    def _disp_store(self, layer: str, store: np.ndarray, step: int) -> np.ndarray:
        """A C-CONTIGUOUS (C, h, w) thumbnail of *store* at subsampling *step*, cached.

        This is the "precompute once at ingest, remap cheaply per tick" half of IMA-261. Building
        it costs one strided copy of the whole store (~2.6 ms on a 1536-well plate); it is
        rebuilt only when the pixels change or the zoom changes, so a continuous contrast drag
        pays for it ZERO times and every tick composites straight out of it.

        Contiguity is not a detail: ``store[:, ::4, ::4]`` is a strided VIEW, and both the table
        lookup and the BLAS reduce in ``composite`` would silently materialise their own copy of
        it on every single tick.
        """
        if step <= 1:
            return store          # already the thing itself; caching it would just double the RAM
        hit = self._disp.get(layer)
        if hit is not None and hit[0] == step:
            return hit[1]
        thumb = np.ascontiguousarray(store[:, ::step, ::step])
        self._disp[layer] = (step, thumb)
        return thumb

    def _cell_auto_windows(self, layer: str, store: np.ndarray, ri: int, ci: int) -> list:
        """The AUTO (exact-percentile) window per channel for ONE cell, cached on the pixels.

        ``np.percentile`` over a cell is the expensive half of SCOPE_PER_REGION — on a 1536-well
        plate it is 1536 cells x C channels of sorting per repaint, which measured ~268 ms and is
        why per-region contrast was unusable under a drag. It depends on the cell's PIXELS only,
        so it survives every contrast change and is recomputed only when a tile lands there.
        """
        cache = self._cell_auto.setdefault(layer, {})
        hit = cache.get((ri, ci))
        if hit is not None:
            return hit
        y0, x0 = ri * _CELL, ci * _CELL
        cell = store[:, y0:y0 + _CELL, x0:x0 + _CELL]
        wins = [_pct_window(cell[ch]) for ch in range(cell.shape[0])]
        cache[(ri, ci)] = wins
        return wins

    def _cell_windows(self, store: np.ndarray, ri: int, ci: int, layer: Optional[str] = None) -> list:
        """Per-channel windows for ONE cell, computed exactly over that cell's own pixels.

        A latched channel (D4) keeps the user's manual window even under per-region: the user
        setting a window explicitly outranks any automatic scoping.

        THE VIEWER'S GLOBAL WINDOW DOES NOT OUTRANK PER-REGION (``follow=False``). Choosing
        per-region IS the instruction "give every well its own range"; letting the array viewer's
        one global window win here made all 1536 cells identical while the plate went on drawing
        the amber "wells NOT comparable" badge — the control did nothing and the caveat lied.
        """
        n_ch = store.shape[0]
        c = self._contrast
        if c is not None and all(c.is_manual(ch) for ch in range(n_ch)):
            # Every channel carries a USER latch, so `resolve` will discard every auto window it
            # is handed. Don't compute 1536 x C percentiles to throw all of them away. Note this
            # tests is_manual, not is_followed: a followed window is exactly what per-region must
            # ignore, so it can never license skipping the percentiles.
            return [c.resolve(ch, (0.0, 0.0)) for ch in range(n_ch)]
        auto = self._cell_auto_windows(layer or self._active, store, ri, ci)
        if c is None:
            return auto
        return [c.resolve(ch, auto[ch], follow=False) for ch in range(len(auto))]

    def _composite_per_region(self, store: np.ndarray, step: int) -> np.ndarray:
        """Composite the plate one CELL at a time, each under its own window (SCOPE_PER_REGION).

        Only cells that actually HAVE a tile are composited; an untiled cell's zero padding would
        percentile to a degenerate window and, worse, is not data. Everything else stays black,
        exactly as the global path leaves it.

        *step* is the same sub-sampling stride the global path uses for a quick repaint. Cell
        bounds are converted into the strided frame by ceil-division rather than by dividing the
        cell size, so a stride that does not divide _CELL cannot drift cells apart by a pixel.
        Cells are cut out of the SAME cached thumbnail the global path composites, and
        ``thumb[:, ys:ye]`` is by construction ``store[:, ys*step:ye*step:step]`` — the same
        pixels the old per-cell re-stride produced, without re-striding 1536 times.
        """
        thumb = self._disp_store(self._active, store, step)
        h, w = thumb.shape[1], thumb.shape[2]
        cells = sorted(self._tiles_by_layer.get(self._active, set()))
        per_cell = [(rc, self._cell_windows(store, rc[0], rc[1])) for rc in cells]
        # EVERY CHANNEL LATCHED == EVERY CELL THE SAME WINDOW == ONE COMPOSITE (IMA-261).
        # `_RunningContrast.resolve` returns the manual window whatever the caller computed, so
        # once the array viewer owns all C channels the per-cell percentiles are all overridden
        # and every cell resolves to the identical window. Compositing 1536 cells separately then
        # costs 1536 numpy dispatches to paint pixels that one call would paint identically --
        # measured 46 ms vs 4 ms. This is not a special case in the CONTRAST rule (there is still
        # exactly one, `resolve`); it is only a refusal to do the same arithmetic 1536 times.
        if per_cell and all(w2 == per_cell[0][1] for _, w2 in per_cell):
            full = composite(thumb, self._colors, per_cell[0][1], self._mask)
            rgb = np.zeros_like(full)                     # untiled cells are NOT data: they stay
            for (ri, ci), _ in per_cell:                  # black, exactly as the loop below does
                ys, ye = -(-ri * _CELL // step), -(-(ri + 1) * _CELL // step)
                xs, xe = -(-ci * _CELL // step), -(-(ci + 1) * _CELL // step)
                ye, xe = min(ye, h), min(xe, w)
                rgb[ys:ye, xs:xe] = full[ys:ye, xs:xe]    # rect copies; a boolean mask over the
            return rgb                                    # whole canvas measured 10x slower
        rgb = np.zeros((h, w, 3), np.uint8)
        for (ri, ci), wins in per_cell:
            ys, ye = -(-ri * _CELL // step), -(-(ri + 1) * _CELL // step)     # ceil-div bounds
            xs, xe = -(-ci * _CELL // step), -(-(ci + 1) * _CELL // step)
            ye, xe = min(ye, h), min(xe, w)
            if ys >= ye or xs >= xe:
                continue
            rgb[ys:ye, xs:xe] = composite(thumb[:, ys:ye, xs:xe], self._colors, wins, self._mask)
        return rgb

    def set_channel_visible(self, ch: int, on: bool):
        """Toggle a channel in/out of the plate composite. Recomposites from the RETAINED store —
        no reader I/O, no re-projection (that is the whole point of keeping the channel axis)."""
        if self._mask is None or not (0 <= ch < len(self._mask)):
            return
        self._mask[ch] = bool(on)
        self._refresh()

    def set_channel_window(self, ch: int, lo: float, hi: float):
        """Re-window one channel and repaint. LATCHES that channel manual (D4) so the wells still
        streaming in can't stomp the window the user just set."""
        if self._contrast is None or not (0 <= ch < len(self._mask)):
            return
        self._contrast.set_manual(ch, lo, hi)
        self._refresh()

    def follow_channel_window(self, ch: int, lo: float, hi: float):
        """Render *ch* with the window the OWNING ARRAY VIEWER resolved, and repaint (IMA-261).

        The sink half of the one-owner contract. It does NOT latch the channel manual: ndv
        autoscales on its own, at open and on every data change, so recording that as a user
        gesture would kill the plate's own auto-contrast before the user had touched anything,
        and would outrank the per-region scope the user explicitly selected. See
        ``_RunningContrast.set_followed``.
        """
        if self._contrast is None or not (0 <= ch < len(self._mask)):
            return
        self._contrast.set_followed(ch, lo, hi)
        self._refresh()

    def set_channel_auto(self, ch: int):
        """Unlatch a channel: it goes back to auto-scaling off the running histogram."""
        if self._contrast is None or not (0 <= ch < len(self._mask)):
            return
        self._contrast.set_auto(ch)
        self._refresh()

    def _refresh(self):
        """A user gesture: repaint NOW at display resolution, then land full-res once it settles.

        The invariant is that no gesture ever touches the full (C, 2816, 4224) canvas — a slider
        drag composites the sub-sampled view the screen can actually show (a few thousand px),
        and the single full-res pass is coalesced behind the last event.
        """
        self.recomposite(quick=True)
        self._full.start(150)
        self._refresh_loupe()

    def _refresh_loupe(self):
        """Re-render the loupe inset under the contrast that just changed (IMA-242).

        The inset holds a rendered QImage, so repainting the plate alone would leave it showing
        the PRE-drag contrast until the cursor happened to move — the plate and the magnifier of
        the plate disagreeing about the same pixels. Re-issuing the request is cheap: the worker
        memoises crops, so this re-colours the bytes it already has and re-reads nothing.
        """
        if self._loupe is None or self._loupe_worker is None:
            return
        try:
            self._request_loupe(self._loupe["x"], self._loupe["y"])
        except Exception:
            pass          # a contrast drag must never fail because the loupe could not re-render

    def _on_full_timeout(self):
        """The coalescing timer fired. Guarded: a pending recomposite must not outlive the widget
        (the plate is torn down and rebuilt on every open, and the timer is queued, not immediate)."""
        try:
            self.recomposite(quick=False)
        except RuntimeError:
            pass   # the C++ widget went away while the full-res pass was still pending

    def recomposite(self, layer: Optional[str] = None, *, quick: bool = False):
        """Rebuild a layer's plate image from its store, at the current mask + windows.

        ``quick=True`` composites a strided view at roughly the on-screen resolution (cheap enough
        to run on every slider tick); the default walks the whole store — the end-of-stream pass.
        """
        layer = layer or self._active
        store = self._store.get(layer)
        if store is None or self._colors is None:
            return
        step = max(1, int(round(_CELL / max(1.0, self._cd)))) if quick else 1
        if self._scope == SCOPE_PER_REGION:
            rgb = self._composite_per_region(store, step)
        else:
            rgb = composite(self._disp_store(layer, store, step), self._colors,
                            self.channel_windows(), self._mask)
        self._final_arr[layer] = rgb          # hold the buffer: the QImage below only wraps it
        h, w, _ = rgb.shape
        self.set_final(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888), layer)

    def hideEvent(self, e):
        self._full.stop()   # a plate on its way out must not repaint from a queued timer
        super().hideEvent(e)

    def reset_layer(self, layer: str):
        """Forget a layer's retained pixels — its store, its recomposite and its painted canvas.

        Called before a run streams into a layer that already has one. Without it a mosaic re-run
        that lands FEWER fields (a smaller selection, a failed well) would leave the previous run's
        FOVs standing inside the same cell, blended into the new ones with no way to tell which is
        which — and the ~95 MB store would never be freed either.
        """
        self._store.pop(layer, None)
        self._invalidate_pixels(layer)
        self._final_arr.pop(layer, None)
        self._op_final.pop(layer, None)
        self._tiles_by_layer.pop(layer, None)
        if layer == "raw":
            self._canvas.fill(QColor(_BG))
        else:
            self._op_canvas.pop(layer, None)
        if layer == self._active:
            self._final = None
            self._scaled = None
            self.update()

    # -- data in --
    def add_tile(self, ri: int, ci: int, well_id: str, tile: np.ndarray, layer: str = "raw",
                 box=None):
        """Take one PER-CHANNEL tile ``(C, h, w)`` (native dtype), retain it in the layer's store,
        feed the running contrast, and re-composite that whole cell.

        ``box`` is ``(top, left, h, w)`` in cell px for a multi-FOV mosaic (IMA-187): the tile is
        one FIELD inside the cell, so it is written at that offset and the cell is re-composited
        around it, which is what makes the seams between neighbouring FOVs update as they land.
        ``box=None`` is the historical single-tile path — one field fills the cell at (0, 0).

        CONTRAST IS FED THE TILE, NEVER THE STORE SLICE. A mosaic cell is zero-padded wherever no
        FOV lands (margins, gaps between fields); feeding those zeros to the running histogram
        would pin the 1st percentile at 0 for the whole plate and silently wash every well out.
        Only real acquired pixels get a vote.
        """
        if (ri, ci) not in self._by_rc:    # ignore a stale tile from a retired run / foreign cell
            return
        store = self._store_for(layer)
        if store is None:                  # no channel axis declared yet -> nothing to composite
            return
        tile = np.asarray(tile)
        y0, x0 = ri * _CELL, ci * _CELL
        top, left = (int(box[0]), int(box[1])) if box is not None else (0, 0)
        th, tw = tile.shape[1], tile.shape[2]      # place by ACTUAL shape: a field smaller than
        store[:, y0 + top:y0 + top + th,           # the cell must not broadcast-crash
              x0 + left:x0 + left + tw] = tile
        self._invalidate_pixels(layer, (ri, ci))   # these pixels are new: nothing derived survives
        for c_i in range(tile.shape[0]):
            self._contrast.add(c_i, tile[c_i])     # real FOV pixels only — see the docstring
        wins = (self._cell_windows(store, ri, ci, layer) if self._scope == SCOPE_PER_REGION
                else self.channel_windows())      # per-region windows THIS cell, not the plate
        cell = composite(store[:, y0:y0 + _CELL, x0:x0 + _CELL], self._colors, wins, self._mask)
        img = QImage(cell.data, _CELL, _CELL, 3 * _CELL, QImage.Format_RGB888)
        p = QPainter(self._canvas_for(layer))
        p.drawImage(x0, y0, img)           # drawImage COPIES, so `cell` may die after p.end()
        p.end()
        if self._op_final.pop(layer, None) is not None:   # a new tile supersedes the old recomposite
            self._final_arr.pop(layer, None)              # -> back to the streamed canvas
            if layer == self._active:
                self._final = None
        self._tiles.add((ri, ci))
        self._tiles_by_layer.setdefault(layer, set()).add((ri, ci))   # per-layer: drives the grey dots
        if layer == self._active:         # only the shown layer needs a repaint / cache rebuild
            self._scaled = None
            self.update()

    def set_status(self, ri: int, ci: int, state: str):
        if (ri, ci) not in self._status:   # never let a foreign/stale key leak into the status map
            return
        self._status[(ri, ci)] = state
        self.update()

    def set_all_status(self, state: str):
        for rc in self._status:
            self._status[rc] = state
        self.update()

    def set_final(self, img: QImage, layer: str = "raw"):
        self._op_final[layer] = img
        if layer == self._active:
            self._final = img
            self._scaled = None       # source changed -> the scaled cache is stale
            self.update()

    def set_active_layer(self, layer: str):
        """Show a layer (LayersTab toggle/reorder). Swaps in its montage + streamed canvas."""
        self._active = layer
        self._final = self._op_final.get(layer)   # None for "raw" -> falls back to the base canvas
        self._scaled = None
        self.update()
        if layer in self._store:     # bring it up to the CURRENT mask/windows: its canvas was blitted
            self.recomposite(layer)  # cell-by-cell, with whatever mask was set when each tile landed

    def drop_layer(self, layer: str):
        """Forget a layer entirely and FREE its canvas (IMA-205: an exploration tab's layers die
        with the tab). ``_canvas_for`` lazily allocates a full plate-sized RGB888 image per layer
        (nc*_CELL x nr*_CELL — tens of MB on a 1536wp), so without this a closed tab's montage
        stays resident forever. Falls back to the base layer if the dropped one was showing.

        The per-channel STORE (IMA-206) is the bigger half — ~95 MB of retained (C, H, W) pixels
        per layer — so it goes too. Dropping the canvas while the store survived would look like a
        fix and leak the majority of the memory."""
        if layer == "raw":
            return
        self._op_canvas.pop(layer, None)
        self._op_final.pop(layer, None)
        self._store.pop(layer, None)
        self._invalidate_pixels(layer)
        self._final_arr.pop(layer, None)
        self._tiles_by_layer.pop(layer, None)
        self._tiles = set().union(*self._tiles_by_layer.values()) if self._tiles_by_layer else set()
        if self._active == layer:
            self.set_active_layer("raw")
        else:
            self.update()

    def status_snapshot(self) -> dict:
        """Copy of the per-well status map — the window saves one per exploration tab so a tab's
        amber/failed dots follow its own run, not whatever ran last (IMA-205)."""
        return dict(self._status)

    def set_status_map(self, status: dict):
        """Restore a snapshot. Foreign keys are ignored, same as ``set_status``."""
        for rc, state in status.items():
            if rc in self._status:
                self._status[rc] = state
        self.update()

    def select(self, ri: int, ci: int):
        """Move the red box to a well (driven by the ndviewer FOV slider)."""
        self._sel = (ri, ci)
        self.update()

    def resizeEvent(self, e):
        self._user_view = False       # a resize re-fits (drop any zoom/pan)
        self._fit()
        self.update()

    # -- mouse: wheel-zoom anchored at cursor, left-drag pan (Hongquan's navigator gestures),
    #    and press-and-hold loupe (IMA-208). The left button now means three different things
    #    depending on TIMING, so the rules live here as a diagram rather than as scattered flags:
    #
    #                        ┌───────────────────────────────────────────┐
    #                        │                  IDLE                     │
    #                        └──────────────────┬────────────────────────┘
    #          left-press on an acquired cell   │   (off-plate / empty: never arms)
    #                                           ▼
    #                        ┌───────────────────────────────────────────┐
    #                        │  ARMED   _hold running (_LOUPE_HOLD_MS)   │
    #                        │  cursor must stay within _LOUPE_SLOP px   │
    #                        └───┬───────────────────────┬───────────────┘
    #          move > slop       │                       │  timer fires
    #          (kill the timer)  │                       │
    #                            ▼                       ▼
    #                  ┌──────────────────┐   ┌──────────────────────────┐
    #                  │       PAN        │   │          LOUPE           │
    #                  │  drag the plate  │   │  inset follows cursor;   │
    #                  │  (unchanged)     │   │  pan is DEAD while up;   │
    #                  └────────┬─────────┘   │  hover + wheel suppressed│
    #                           │             └────────────┬─────────────┘
    #                           │ release                  │ release / dragged off the widget
    #                           │                          │ / leave / focus-out
    #                           ▼                          ▼
    #                        ┌───────────────────────────────────────────┐
    #                        │                  IDLE                     │
    #                        └───────────────────────────────────────────┘
    #
    #    Two edges worth stating because they are easy to regress:
    #      * SLOW PAN stays a pan. Press, dwell past the timer, then drag — the timer only runs
    #        while the cursor is still, and any move past the slop kills it. A press that has
    #        already become a loupe is dismissed on release, so the next drag pans normally.
    #      * DOUBLE-CLICK must cancel the timer. Qt delivers press/release/dblclick/release, and
    #        the second press re-arms; without the cancel you would open the detail viewer AND
    #        raise a loupe from one gesture.
    def _cell(self, x, y):
        if self._layout is not None:
            return self._freeform_cell(x, y)
        px, py = x - (self._ox + _HDR), y - (self._oy + _COLH)
        return well_at(self._rows, self._cols, self._by_rc, px, py, self._cd)

    def _freeform_cell(self, x, y):
        """Hit-test a geometrically placed holder: the first cell whose own rect contains (x, y).

        Placed cells are tested FIRST and in reverse paint order, so a click in the small area
        where two regions' bounding boxes overlap resolves to the one drawn on top — the same
        last-one-wins rule ``_fov_at`` uses inside a mosaic. Nominal (empty-slot) rects are only
        consulted when no real region claims the point, so an empty slot can never shadow a region
        that overlaps it. Freeform holders have a handful of cells, so a linear scan is free.
        """
        placed = [rc for rc in self._by_rc if rc in self._layout]
        for rc in list(reversed(placed)) + [rc for rc in self._by_rc if rc not in self._layout]:
            rx, ry, rw, rh = self._cell_rect(*rc)
            if rx <= x < rx + rw and ry <= y < ry + rh:
                ri, ci = rc
                return {"row_index": ri, "col_index": ci, "row": self._rows[ri],
                        "col": self._cols[ci], "well_id": self._by_rc.get(rc)}
        return None

    def _cells_in(self, x0, y0, x1, y1) -> list:
        """Widget px -> acquired cells, via the pure helper (same margin removal as _cell)."""
        if self._layout is not None:
            lo_x, hi_x = min(x0, x1), max(x0, x1)
            lo_y, hi_y = min(y0, y1), max(y0, y1)
            hits = []
            for rc in self._by_rc:
                rx, ry, rw, rh = self._cell_rect(*rc)
                if rx < hi_x and rx + rw > lo_x and ry < hi_y and ry + rh > lo_y:
                    hits.append(rc)
            return sorted(hits)
        ox, oy = self._ox + _HDR, self._oy + _COLH
        return cells_in_rect(self._rows, self._cols, self._by_rc,
                             x0 - ox, y0 - oy, x1 - ox, y1 - oy, self._cd)

    # -- selection API (IMA-221) --
    def selected_wells(self) -> list:
        """The selection as acquired well ids, in plate row-major order."""
        return [self._by_rc[rc] for rc in sorted(self._selection)]

    def clear_selection(self):
        """Drop the whole selection and tell listeners (used on re-ingest)."""
        if self._selection:
            self._selection = set()
            self.selectionChanged.emit([])
            self.update()

    def wheelEvent(self, e):
        if self._marquee is not None:
            return          # a marquee owns the drag; zooming would slide the plate under the rect
        if self._loupe is not None:      # zooming the plate under a live loupe would fight it
            return
        mx, my = e.x() - (self._ox + _HDR), e.y() - (self._oy + _COLH)    # cursor in plate px
        new_cd = self._cd * (1.0015 ** e.angleDelta().y())
        new_cd = max(self._fit_cd(), min(self._fit_cd() * 40, new_cd))    # never zoom out past fit
        scale = new_cd / self._cd
        self._ox = e.x() - _HDR - mx * scale         # keep the point under the cursor fixed
        self._oy = e.y() - _COLH - my * scale
        self._cd = new_cd
        self._user_view = True
        self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        # SHIFT owns every selection gesture (IMA-221). Keeping selection off the plain click is
        # what makes double-click safe: Qt delivers press+release BEFORE mouseDoubleClickEvent, so
        # a plain-click toggle would silently flip a well every time you opened one. (Ctrl is out:
        # on macOS Ctrl+click is right-click and Qt maps Cmd -> ControlModifier.)
        if e.modifiers() & Qt.ShiftModifier:
            self._marquee = (e.x(), e.y(), e.x(), e.y())
            self._marquee_add = bool(e.modifiers() & Qt.AltModifier)   # Shift+Alt = union
            self._press = None                                          # ...so this drag never pans
            self._panning = False
            self.update()
            return          # a Shift-drag is a selection, never a pan and never a loupe
        self._press = (e.x(), e.y(), self._ox, self._oy)
        self._panning = False
        c = self._cell(e.x(), e.y())
        if self._loupe_src is not None and c and c["well_id"]:   # ARM (never off-plate/empty)
            self._hold.start()

    def mouseMoveEvent(self, e):
        if self._loupe is not None:                  # LOUPE: the inset tracks; panning is dead
            # Drag off the widget and the loupe must go. leaveEvent CANNOT do this: Qt grabs the
            # mouse for the duration of a press, so no leave is delivered until the button comes
            # up — the inset used to stay pinned over the neighbouring pane showing stale pixels.
            # The grab is also why this works: move events keep arriving, with coordinates
            # outside rect(), which is the signal.
            if not self.rect().contains(e.x(), e.y()):
                self._hold.stop()
                self._dismiss_loupe()
                return
            self._loupe["x"], self._loupe["y"] = e.x(), e.y()
            self._request_loupe(e.x(), e.y())
            self.update()
            return
        if self._marquee is not None and (e.buttons() & Qt.LeftButton):
            x0, y0, _, _ = self._marquee          # grow the rubber band; emit NOTHING until release
            self._marquee = (x0, y0, e.x(), e.y())
            self.update()
            return
        if self._press is not None and (e.buttons() & Qt.LeftButton):
            dx, dy = e.x() - self._press[0], e.y() - self._press[1]
            if abs(dx) + abs(dy) > 3:
                self._panning = True
                self._hold.stop()                    # moved -> this press is a pan, not a hold
            if self._panning:
                self._ox, self._oy = self._press[2] + dx, self._press[3] + dy
                self._user_view = True
                self.update()
                return
        c = self._cell(e.x(), e.y())                 # hover (only when not dragging)
        new_hover = (c["row_index"], c["col_index"]) if c else None
        if new_hover == self._hover:                 # still the same cell -> no repaint (kills the
            return                                   # per-pixel repaint storm; only cross-cell moves repaint)
        self._hover = new_hover
        self.hovered.emit((c["well_id"] or (c["row"] + c["col"] + "  ·  empty")) if c else "")
        self.update()

    def mouseReleaseEvent(self, e):
        self._hold.stop()
        # Only the LEFT release commits a selection. The gesture is opened by a left press, but Qt
        # delivers a release for whichever button went up — so a right-click while a Shift-drag is
        # in flight would otherwise silently toggle/replace the selection.
        if self._marquee is not None and e.button() == Qt.LeftButton:
            x0, y0, x1, y1 = self._marquee
            add, self._marquee, self._marquee_add = self._marquee_add, None, False
            dragged = abs(x1 - x0) + abs(y1 - y0) > _CLICK_SLOP
            if not dragged:                                     # Shift+CLICK -> toggle ONE well
                hit = self._cell(x1, y1)
                if hit and hit["well_id"]:
                    self._selection ^= {(hit["row_index"], hit["col_index"])}
            elif add:
                self._selection |= set(self._cells_in(x0, y0, x1, y1))
            else:
                self._selection = set(self._cells_in(x0, y0, x1, y1))
            # ONE emission per gesture. A live emit would rebuild a 1536-item list per mouse-move
            # on a 1536wp; the rubber band already gave the user live feedback.
            wells = self.selected_wells()
            self.selectionChanged.emit(wells)
            if dragged:                    # "hold shift [and drag] to open an exploration tab"
                self.marqueeSelected.emit(wells)
            self.update()
        self._press = None
        self._panning = False
        self._dismiss_loupe()                        # release always dismisses

    def leaveEvent(self, e):
        self._hold.stop()                            # cursor left mid-hold: release may never come
        self._dismiss_loupe()
        self._hover = None
        # Drop any in-flight marquee too. If the grab is lost mid-drag (modal dialog, alt-tab) no
        # release ever arrives, and a stranded _marquee both paints a dashed rect forever and makes
        # wheelEvent's mid-marquee guard disable zoom permanently.
        self._marquee = None
        self._marquee_add = False
        self.hovered.emit("")
        self.update()

    def set_mosaic_boxes(self, boxes: dict):
        """Adopt the per-FOV cell boxes so a double-click can resolve WHICH FOV was hit.

        Also tells the paint path WHICH cells hold a letterboxed mosaic rather than a
        cell-filling single tile (see :meth:`_cell_source`), so the two can never disagree about
        the same cell: they read one dict.
        """
        self._boxes = dict(boxes or {})
        self._boxed_regions = {r for r, _f in self._boxes}
        self.update()

    # -- carrier geometry (IMA-220 -> IMA-253: DRAWN, not blitted) --
    def set_carrier(self, plate, images_dir=None):
        """Adopt *plate*'s geometry so the holder can be DRAWN behind the cells.

        This used to blit Squid's carrier PHOTOGRAPH. It no longer does, and the reason is
        registration, not taste. A PNG lives in its own pixel space and has to be brought into the
        cell grid's space through three calibration constants (``a1_x_pixel``, ``a1_x_mm``,
        ``mm_per_pixel``) that must agree with the geometry the cells are laid out from. When they
        disagree, nothing raises — you get a plausible picture with the wells in the wrong places,
        which is exactly what shipped, and it is unfixable in general for a FREEFORM holder because
        there is no photograph of "two tissues wherever the operator happened to put them".

        Drawing the outline, the slot/well boundaries and the empty-vs-occupied state from
        :class:`~squidmip._plate.PlateGeometry` puts the holder in the SAME coordinate system as
        the cells, so it cannot misalign, it cannot vanish on pan or zoom (there is no separately
        positioned blit to drift), and an acquisition with no artwork on disk renders identically
        to one with artwork. ``plate.art()`` and the whole PNG registry stay in ``_plate.py`` for
        an optional skin; they are simply not on this path.

        *images_dir* is accepted and ignored, so callers that passed one still work.
        """
        self._carrier = getattr(plate, "geometry", None) if plate is not None else None
        try:
            from squidmip._plate import SlideCarrier
            self._carrier_slide = isinstance(plate, SlideCarrier)
        except Exception:
            self._carrier_slide = False
        self.update()

    # -- cell rectangles: the ONE place a (row, col) becomes widget pixels (IMA-253) --
    def _cell_rect(self, ri: int, ci: int) -> tuple:
        """Widget-pixel ``(x, y, w, h)`` of cell (ri, ci) at the current zoom/pan.

        Uniform plates return the historical ``(ax + ci*cd, ay + ri*cd, cd, cd)`` exactly. A
        freeform holder returns the region's own rectangle: its mosaic's bounding box, scaled by
        the same single transform for every region, so relative offset and relative scale are
        preserved and two regions of different size get different-sized cells.
        """
        cd = self._cd
        ax, ay = self._ox + _HDR, self._oy + _COLH
        if self._layout is not None:
            r = self._layout.get((ri, ci))
            if r is not None:
                return (ax + r[0] * cd, ay + r[1] * cd, r[2] * cd, r[3] * cd)
        return (ax + ci * cd, ay + ri * cd, cd, cd)

    def _cell_source(self, ri: int, ci: int) -> tuple:
        """The sub-rectangle of the montage canvas that ``_cell_rect(ri, ci)`` shows.

        The store keeps every cell as one ``_CELL`` x ``_CELL`` square. A MOSAIC is LETTERBOXED into
        it (``_placement.cell_boxes`` centres it, preserving aspect), so the bars must be excluded
        or the mosaic would be stretched back into them. A single tile — one FOV, or a region
        operator's already-fused result — FILLS the block, so the whole block is the source. Which
        of the two a cell holds is read from ``self._boxes``, the same dict the tiles were placed
        by, so the blit and the pixels cannot disagree.

        Since the cell rect and the letterbox come from the SAME aspect ratio, the inner box is
        recoverable from the rect alone: no second bookkeeping table to fall out of sync.
        """
        full = (ci * _CELL, ri * _CELL, _CELL, _CELL)
        if self._layout is None or self._by_rc.get((ri, ci)) not in self._boxed_regions:
            return full
        r = self._layout.get((ri, ci))
        if r is None or not (r[2] > 0 and r[3] > 0):
            return full
        a = r[2] / r[3]                                    # target aspect == mosaic aspect
        iw = _CELL * min(1.0, a)
        ih = _CELL * min(1.0, 1.0 / a)
        return (ci * _CELL + (_CELL - iw) / 2.0, ri * _CELL + (_CELL - ih) / 2.0, iw, ih)

    def _tiled_region(self) -> "QRegion":
        """The cells that HAVE an image on the active layer, as a QRegion at pan origin (0, 0).

        The montage canvas is opaque _BG wherever no tile landed, so blitting it whole would paint
        the carrier art out. Clipping the blit to this region is what lets the background show
        through the empty wells. Cached and only translated on pan: a full 1536wp is 1536 rects,
        and rebuilding that union on every hover repaint would be the one thing that makes the
        plate feel slow.
        """
        cells = self._tiles_by_layer.get(self._active, set())
        key = (self._cd, self._active, len(cells))
        if self._tile_rgn is None or self._tile_rgn_key != key:
            cd = self._cd
            rgn = QRegion()
            for ri, ci in cells:
                rgn = rgn.united(QRegion(int(ci * cd), int(ri * cd),
                                         int(cd) + 1, int(cd) + 1))   # +1: no hairline seams
            self._tile_rgn, self._tile_rgn_key = rgn, key
        return self._tile_rgn

    def _fov_at(self, c: dict, e) -> int:
        """FOV index under the cursor within cell *c*, or 0 when there is no mosaic to resolve.

        Inverts the placement transform: find where the click landed inside the cell (in _CELL
        units), then pick the FOV whose box contains it. Boxes overlap by ~9% at the seams, so
        the LAST match wins — matching the draw order in ``_OperatorWorker._on_well``, where
        later FOVs paint over earlier ones. Without that agreement a click in a seam would open
        a different FOV than the one visibly on top.
        """
        region = c["well_id"]
        if not region or not self._boxes:
            return 0
        ri, ci = c["row_index"], c["col_index"]
        rx, ry, rw, rh = self._cell_rect(ri, ci)
        sx, sy, sw, sh = self._cell_source(ri, ci)
        if not (rw > 0 and rh > 0):
            return 0
        # position within the cell, normalised to the _CELL-px space the boxes live in. Going via
        # the cell's SOURCE rect is what keeps the hit-test agreeing with the blit on a freeform
        # holder, where the drawn rect is the mosaic's box and not the whole square block.
        fx = (e.x() - rx) / rw * sw + (sx - ci * _CELL)
        fy = (e.y() - ry) / rh * sh + (sy - ri * _CELL)
        hit = 0
        for (r, fov), (top, left, h, w) in self._boxes.items():
            if r == region and top <= fy < top + h and left <= fx < left + w:
                hit = fov
        return hit

    def focusOutEvent(self, e):
        # Only reachable because __init__ sets ClickFocus: with the default NoFocus this widget
        # never held focus, so this handler was dead code pretending to cover "window
        # deactivated mid-hold". A press now focuses the plate, so losing focus is a real signal.
        self._hold.stop()
        self._dismiss_loupe()
        super().focusOutEvent(e)

    # -- CONTROL WELL (IMA-248, made reachable by IMA-260's empty-state example) -------------------
    def set_control(self, well_id: Optional[str]):
        """Mirror the window's control-well identity onto the plate. ``None``/unknown clears it.

        Deliberately a setter and not a toggle: there is ONE control at a time and ONE thing that
        decides which, so the plate cannot drift out of agreement with the exploration pane by
        answering a click locally."""
        rc = next((k for k, v in self._by_rc.items() if v == well_id), None) if well_id else None
        if rc == self._control:
            return
        self._control = rc
        self.update()

    def control_well(self) -> Optional[str]:
        """The control's region id as the PLATE currently draws it — for asserting that all three
        views agree by reading them, rather than by trusting that they were all told."""
        return self._by_rc.get(self._control) if self._control is not None else None

    def contextMenuEvent(self, e):
        """Right-click a region -> the dropdown the empty pane's example points at.

        This menu IS the example: 'right-click a well and choose Control Well'. It exists on the
        plate rather than in a menu bar because that is the sentence the user said, and an example
        that names a gesture the app does not have is worse than no example at all."""
        c = self._cell(e.x(), e.y())
        well = c["well_id"] if c else None
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_QSS)
        if well:
            if self._control is not None and self._by_rc.get(self._control) == well:
                act = menu.addAction(f"Clear Control Well ({well})")
                act.triggered.connect(lambda *_: self.controlRequested.emit(""))
            else:
                act = menu.addAction(f"Control Well  ·  set {well} as the reference")
                act.triggered.connect(lambda *_, w=well: self.controlRequested.emit(w))
        else:
            act = menu.addAction("Control Well")     # off-plate: say why it is unavailable
            act.setEnabled(False)
        if self._control is not None and self._by_rc.get(self._control) != well:
            clear = menu.addAction(f"Clear Control Well ({self._by_rc.get(self._control)})")
            clear.triggered.connect(lambda *_: self.controlRequested.emit(""))
        self._context_menu = menu           # keep a ref so offscreen tests can drive the actions
        menu.popup(e.globalPos())
        e.accept()

    def mouseDoubleClickEvent(self, e):
        # Qt sends press/release/dblclick — the second press already re-armed the hold timer, so
        # kill it here or one double-click both opens the well AND raises a loupe.
        self._hold.stop()
        self._dismiss_loupe()
        c = self._cell(e.x(), e.y())
        if c and c["well_id"]:
            self.wellActivated.emit(c["well_id"], self._fov_at(c, e))

    # -- paint --
    def paintEvent(self, _):
        if not self._user_view:          # auto-fit until the user first zooms/pans
            self._fit()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(_BG))
        cd, nr, nc = self._cd, self._nr, self._nc
        ax, ay = self._ox + _HDR, self._oy + _COLH   # plate top-left (after label margins)
        W, H = nc * cd, nr * cd
        tiled = self._tiles_by_layer.get(self._active, set())
        # THE HOLDER (IMA-253), behind everything: drawn from the plate's own geometry, in the
        # cells' own coordinate system. No photograph, so nothing to calibrate and nothing to
        # drift on pan/zoom; see set_carrier.
        self._paint_carrier(p, tiled)
        if self._layout is not None:
            # FREEFORM: each region's cell is its own rectangle, so the montage is blitted per
            # cell rather than as one grid-aligned image. A handful of regions, one drawImage each.
            src = self._active_source()
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            for rc in sorted(tiled):
                if rc not in self._by_rc:
                    continue
                p.drawImage(QRectF(*self._cell_rect(*rc)), src, QRectF(*self._cell_source(*rc)))
        else:
            # Blit the montage from a cached pixmap scaled to the current zoom. The expensive
            # smooth resample runs ONCE per zoom/source-change (not every repaint) — pan/hover
            # just re-blit.
            w, h = max(1, int(W)), max(1, int(H))
            if (self._scaled is None or self._scaled_cd != cd
                    or self._scaled.width() != w or self._scaled.height() != h):
                self._scaled = QPixmap.fromImage(self._active_source()).scaled(
                    w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self._scaled_cd = cd
            if len(tiled) < nr * nc:
                # The montage canvas is opaque _BG wherever no tile landed, so let it cover only
                # the cells that actually have pixels — otherwise it paints the holder out.
                p.save()
                p.setClipRegion(self._tiled_region().translated(int(ax), int(ay)))
                p.drawPixmap(int(ax), int(ay), self._scaled)
                p.restore()
            else:
                p.drawPixmap(int(ax), int(ay), self._scaled)

        # per-cell DOT over the WHOLE plate grid (so a sparse acquisition still shows the full plate
        # shape — e.g. 32x48 for 1536, 16x24 for 384 — with grey dots on the un-acquired wells):
        # amber = processing, red x = failed, GREY = no image on the active layer, no dot once a cell
        # HAS an image (the image speaks for itself). Dot size is a capped absolute size.
        d = min(max(3.0, cd * 0.36), 15.0)
        active_tiles = self._tiles_by_layer.get(self._active, set())
        for ri in range(nr):
            for ci in range(nc):
                state = self._status.get((ri, ci), "empty")
                has_img = (ri, ci) in active_tiles
                x0, y0, cw, ch = self._cell_rect(ri, ci)
                ex, ey = int(x0 + (cw - d) / 2), int(y0 + (ch - d) / 2)
                if state == "processing":                   # amber dot
                    p.setPen(Qt.NoPen)
                    p.setBrush(_STATUS["processing"])
                    p.drawEllipse(ex, ey, int(d), int(d))
                elif state == "failed":                     # red x within the dot box
                    p.setPen(QPen(_STATUS["failed"], max(1.5, min(cd * 0.09, 3.0))))
                    p.drawLine(ex, ey, ex + int(d), ey + int(d))
                    p.drawLine(ex + int(d), ey, ex, ey + int(d))
                elif not has_img:                           # grey dot: an empty plate position
                    p.setPen(Qt.NoPen)
                    p.setBrush(_STATUS["empty"])
                    p.drawEllipse(ex, ey, int(d), int(d))
                # else: has an image on the active layer -> no dot
        p.setBrush(Qt.NoBrush)

        if self._selection:            # SELECTED wells = translucent accent wash (IMA-221). Drawn
            p.setPen(Qt.NoPen)         # under the grid/labels so it reads as a highlight, and kept
            p.setBrush(_SEL_FILL)      # visually distinct from _sel's red BOX and _hover's red DOT.
            for ri, ci in self._selection:
                rx, ry, rw, rh = self._cell_rect(ri, ci)
                p.drawRect(int(rx), int(ry), int(rw), int(rh))
            p.setBrush(Qt.NoBrush)

        if self._layout is None:
            p.setPen(QPen(_GRID, 3))   # black grid lines between wells (multi-FOV mosaics sit INSIDE a cell)
            for c in range(nc + 1):
                p.drawLine(int(ax + c * cd), int(ay), int(ax + c * cd), int(ay + H))
            for r in range(nr + 1):
                p.drawLine(int(ax), int(ay + r * cd), int(ax + W), int(ay + r * cd))
        # (a freeform holder has no shared grid lines to draw — its cells are individually placed
        #  rectangles, and _paint_carrier already outlined each one.)
        p.setFont(QFont("Helvetica Neue", 11, QFont.DemiBold))
        if self._layout is not None:
            # A freeform region is named, not numbered, and the gutter is sized for "A".."AF" —
            # "manual0" gets sliced in half there. Its own cell is the only place wide enough and
            # the only place that is unambiguous when cells are individually positioned.
            for rc, region in self._by_rc.items():
                rx, ry, rw, _rh = self._cell_rect(*rc)
                p.setPen(_ACCENT if self._hover == rc else _MUTED)
                p.drawText(QRectF(rx, ry - _COLH, max(rw, 60.0), _COLH),
                           int(Qt.AlignCenter), str(region))
        # Column/row labels THIN OUT as cells shrink so they never overlap (a 48-col 1536wp would
        # otherwise cram "1..48" into a few px). Always draw the hovered row/col so hover still
        # reads. Skipped entirely for a freeform holder: its rows and columns are an internal
        # bookkeeping key, not something on the glass, and the names are already on the cells.
        cstep = max(1, int(np.ceil(22.0 / cd)))
        rstep = max(1, int(np.ceil(18.0 / cd)))
        for c in range(nc if self._layout is None else 0):
            hov = bool(self._hover and self._hover[1] == c)
            if c % cstep and not hov:
                continue
            p.setPen(_ACCENT if hov else _MUTED)
            p.drawText(int(ax + c * cd), int(self._oy), int(cd), _COLH, Qt.AlignCenter, str(self._cols[c]))
        for r in range(nr if self._layout is None else 0):
            hov = bool(self._hover and self._hover[0] == r)
            if r % rstep and not hov:
                continue
            p.setPen(_ACCENT if hov else _MUTED)
            p.drawText(int(self._ox), int(ay + r * cd), _HDR, int(cd), Qt.AlignCenter, str(self._rows[r]))
        if self._control is not None and self._control in self._by_rc:
            # THE CONTROL WELL: a PERSISTENT light-blue frame, labelled, drawn UNDER the red box so
            # the transient current-FOV marker still reads when the two land on the same cell. The
            # label is what stops "a blue box" from being a mystery a week later.
            cx, cy, cw, ch = self._cell_rect(*self._control)
            p.setPen(QPen(_CONTROL_BLUE, 3))
            p.setBrush(Qt.NoBrush)
            p.drawRect(int(cx) + 1, int(cy) + 1, int(cw) - 2, int(ch) - 2)
            p.setFont(QFont("Helvetica Neue", 10, QFont.Bold))
            p.drawText(QRectF(cx, cy + 2, max(cw, 50.0), 16.0), int(Qt.AlignCenter), "Control")
            p.setFont(QFont("Helvetica Neue", 11, QFont.DemiBold))
        if self._sel is not None:          # the CURRENT well in the detail viewer = a red BOX
            p.setPen(QPen(_RED, 2))
            p.setBrush(Qt.NoBrush)
            sx, sy, sw, sh = self._cell_rect(*self._sel)
            p.drawRect(int(sx), int(sy), int(sw), int(sh))
        if self._hover is not None:        # where the cursor is, moving around the plate = a red DOT
            ri, ci = self._hover           # SAME geometry as the status dots -> overlays them exactly
            x0, y0, hw, hh = self._cell_rect(ri, ci)
            ex, ey = int(x0 + (hw - d) / 2), int(y0 + (hh - d) / 2)
            p.setPen(Qt.NoPen)
            p.setBrush(_RED)
            p.drawEllipse(ex, ey, int(d), int(d))
        if self._marquee is not None:      # live drag rectangle while Shift-dragging
            mx0, my0, mx1, my1 = self._marquee
            p.setPen(QPen(_ACCENT, 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(int(min(mx0, mx1)), int(min(my0, my1)),
                       int(abs(mx1 - mx0)), int(abs(my1 - my0)))
        if self._loupe is not None:        # press-and-hold magnifier, over everything else
            self._paint_loupe(p)
        # CONTRAST SCOPE BADGE (IMA-207), drawn INTO the plate. Per-region contrast stretches every
        # well to its own range, so a dim and a bright well can look identical — a screenshot of
        # this plate is scientifically wrong if read as relative signal. The badge travels with the
        # pixels; a dropdown would be cropped out of that screenshot. Nothing is drawn for global:
        # there is no caveat to give when the wells are comparable.
        if self._scope != SCOPE_GLOBAL:
            label = f"contrast: {self._scope}  ·  wells NOT comparable"
            p.setFont(QFont("Helvetica Neue", 10, QFont.Bold))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label) if hasattr(fm, "horizontalAdvance") else fm.width(label)
            bw, bh = tw + 18, 24
            bx, by = self.width() - bw - 10, self.height() - bh - 10
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 176, 0, 235))            # amber: a caveat, not an error
            p.drawRoundedRect(bx, by, bw, bh, 6, 6)
            p.setPen(QPen(QColor("#1b1300")))
            p.drawText(bx + 9, by + bh - 7, label)
        # a fine outer white frame around the whole plate view
        p.setPen(QPen(QColor("#c9d1d9"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.end()

    def _paint_carrier(self, p: QPainter, tiled: set):
        """Draw the sample holder: body outline, per-cell boundary, empty vs occupied (IMA-253).

        Everything here comes out of the geometry the cells themselves are placed from, so there is
        exactly one coordinate system and the holder cannot drift out of register with the wells —
        which is the failure a separately-positioned photograph kept producing, and could not
        avoid. It also means an acquisition with no artwork on disk renders IDENTICALLY to one with
        artwork, because artwork is no longer consulted.

        Three states, deliberately distinct, because "which slots are empty" was the exact question
        the photograph answered badly:

            occupied, imaged   the pixels themselves (drawn over this)
            occupied, waiting  a solid accent-tinted boundary + fill
            empty slot         a DASHED, dim boundary and no fill

        Skipped entirely below a few px per cell: at 1536-well zoom the boundaries are smaller than
        the status dots, so drawing 1536 of them would cost a repaint and show nothing.
        """
        if self._carrier is None:
            return
        cd = self._cd
        ax, ay = self._ox + _HDR, self._oy + _COLH
        # The holder BODY: the union of every cell rectangle, padded by the margin the geometry
        # implies (half a pitch beyond the outer cell centres on a well plate).
        rects = [self._cell_rect(r, c) for r in range(self._nr) for c in range(self._nc)]
        if not rects:
            return
        bx0 = min(r[0] for r in rects)
        by0 = min(r[1] for r in rects)
        bx1 = max(r[0] + r[2] for r in rects)
        by1 = max(r[1] + r[3] for r in rects)
        pad = max(4.0, cd * 0.18)
        body = QRectF(bx0 - pad, by0 - pad, (bx1 - bx0) + 2 * pad, (by1 - by0) + 2 * pad)
        p.setBrush(QColor(28, 32, 40))
        p.setPen(QPen(QColor(90, 100, 116), 2))
        p.drawRoundedRect(body, min(10.0, pad), min(10.0, pad))
        # An orientation cue instead of a picture of one: the A1 / first-slot corner is chamfered,
        # the way a real plate's notched corner reads.
        p.setPen(QPen(QColor(120, 132, 150), 2))
        ch = min(14.0, pad * 2.0)
        p.drawLine(int(body.left()), int(body.top() + ch), int(body.left() + ch), int(body.top()))
        if cd < 6.0:                     # boundaries smaller than the status dots: not worth it
            return
        for ri in range(self._nr):
            for ci in range(self._nc):
                rx, ry, rw, rh = self._cell_rect(ri, ci)
                occupied = (ri, ci) in self._by_rc
                if occupied and (ri, ci) in tiled:
                    continue             # the acquired pixels will cover it; do not tint them
                if occupied:
                    p.setPen(QPen(_ACCENT, max(1.0, min(cd * 0.03, 2.0))))
                    p.setBrush(QColor(56, 139, 253, 40))
                else:
                    p.setPen(QPen(QColor(74, 84, 100), 1, Qt.DashLine))
                    p.setBrush(Qt.NoBrush)
                if self._carrier_slide:  # a slide slot is a rectangle; a well is round
                    p.drawRect(QRectF(rx, ry, rw, rh))
                else:
                    # Well diameter relative to pitch, straight from sample_formats.csv, so a 96wp
                    # reads as fat wells and a 1536wp as pinpricks — the real difference between them.
                    g = self._carrier
                    f = (g.cell_size_um / g.pitch_x_um) if g.pitch_x_um else 0.8
                    f = float(min(max(f, 0.15), 1.0))
                    p.drawEllipse(QRectF(rx + rw * (1 - f) / 2, ry + rh * (1 - f) / 2, rw * f, rh * f))
        p.setBrush(Qt.NoBrush)

    def _paint_loupe(self, p: QPainter):
        """The inset: real pixels, a µm scale bar when the pixel size is known, or the reason
        there are no pixels. Offset from the cursor so the hand never covers what it points at,
        and clamped inside the widget so it stays whole at the plate's edges."""
        x, y = self._loupe["x"], self._loupe["y"]
        s = _LOUPE_PX
        bx = x + 18 if x + 18 + s < self.width() else x - 18 - s
        by = y + 18 if y + 18 + s < self.height() else y - 18 - s
        bx = int(max(2, min(bx, self.width() - s - 2)))
        by = int(max(2, min(by, self.height() - s - 2)))
        p.fillRect(bx, by, s, s, QColor("#05070b"))
        if self._loupe_img is not None:
            p.save()
            p.setClipRect(bx, by, s, s)
            p.drawPixmap(bx, by, QPixmap.fromImage(self._loupe_img).scaled(
                s, s, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation))   # 1:1-ish: no smoothing
            p.restore()
        else:
            p.setPen(_MUTED)
            p.setFont(QFont("Helvetica Neue", 11))
            p.drawText(bx, by, s, s, Qt.AlignCenter | Qt.TextWordWrap,
                       self._loupe_note or "reading …")
        geo = self._loupe_geometry(x, y)
        if geo is not None and self._loupe_img is not None:
            _w, _l, _r, s_loupe, mag = geo
            um_px = loupe_um_per_screen_px(getattr(self._loupe_src, "pixel_size_um", None), s_loupe)
            p.setFont(QFont("Helvetica Neue", 10, QFont.DemiBold))
            if um_px is None:
                # No trustworthy pixel size: say so rather than draw a bar that would be fiction.
                p.setPen(_MUTED)
                p.drawText(bx + 8, by + s - 10, "scale unknown")
            else:
                target = _nice_scale_um(um_px * (s * 0.4))     # ~40% of the inset, rounded to 1/2/5
                bar = int(round(target / um_px))
                p.setPen(QPen(QColor("#e6edf3"), 2))
                p.drawLine(bx + 10, by + s - 14, bx + 10 + bar, by + s - 14)
                p.setPen(QColor("#e6edf3"))
                p.drawText(bx + 10, by + s - 18, f"{_fmt_um(target)}")
            p.setPen(_ACCENT)
            label = f"{self._loupe['well']}  ·  {mag:.1f}×" if mag >= 1.05 else \
                    f"{self._loupe['well']}  ·  native"
            p.drawText(bx + 8, by + 16, label)
        p.setPen(QPen(QColor("#c9d1d9"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(bx, by, s, s)


def _mosaic_boxes(meta: dict) -> dict:
    """``{(region, fov): (top, left, h, w)}`` — every FOV's box inside its _CELL thumbnail.

    Pure geometry, delegated to :mod:`squidmip._placement` (which is Qt-free and unit-tested
    against exact pixel offsets). Returns ``{}`` when the acquisition has no stage positions or
    no pixel size, which is the signal for the caller to keep the historical single-tile path —
    a mosaic is simply not derivable without both, and guessing would draw a wrong picture.

    Placement failures for ONE region are contained: that region falls back to single-tile
    rendering rather than aborting a whole-plate run. The reader has already fail-loud checked
    the CSV/filename agreement, so anything reaching here is a genuine per-region oddity
    (e.g. a region with images but no coordinate rows).
    """
    from squidmip._placement import cell_boxes, fov_offsets_px

    positions = meta.get("fov_positions_um") or {}
    if not positions or meta.get("pixel_size_um") in (None, 0):
        return {}
    frame_shape = meta["frame_shape"]
    out: dict = {}
    for region in meta["regions"]:
        fovs = meta["fovs_per_region"][region]
        if len(fovs) < 2:
            continue                     # a single-FOV well fills its cell; no mosaic needed
        try:
            offsets = fov_offsets_px(positions, region, fovs, meta.get("pixel_size_um"))
            for fov, box in cell_boxes(offsets, frame_shape, _CELL).items():
                out[(region, fov)] = box
        except (KeyError, ValueError):
            continue                     # this region renders single-tile; the rest still mosaic
    return out


def region_mosaic_extent_px(meta: dict, regions: Optional[list] = None) -> Optional[tuple]:
    """Full-resolution ``(height, width)`` bounding box of the mosaics a REGION operator will fuse.

    IMA-245. A region operator (``available_region_operators()``) yields ONE fused mosaic per
    region, whose extent is the bounding box of the region's coordinate-placed frames — NOT the
    frame shape. Anything that has to size a surface for that result (the array viewer's canvas,
    the push planes fed into it) must ask this, or it sizes a mosaic as a frame.

    Returns the max extent over *regions* (``None`` = every region in the acquisition), because
    one array viewer serves the whole run and its canvas is declared once. Returns ``None`` when
    the acquisition carries no stage positions / no pixel size — the same "not derivable, do not
    guess" signal :func:`_mosaic_boxes` returns ``{}`` for.
    """
    from squidmip._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    if not positions or meta.get("pixel_size_um") in (None, 0):
        return None
    frame_shape = meta["frame_shape"]
    scoped = list(meta["regions"]) if regions is None else list(regions)
    best = None
    for region in scoped:
        fovs = meta["fovs_per_region"].get(region) or []
        if not fovs:
            continue
        try:
            offsets = fov_offsets_px(positions, region, fovs, meta.get("pixel_size_um"))
            h, w = mosaic_extent_px(offsets, frame_shape)
        except (KeyError, ValueError):
            continue                     # this region contributes nothing; the rest still size it
        best = (h, w) if best is None else (max(best[0], h), max(best[1], w))
    return best


def push_shape_for(meta: dict, region_op: bool, regions: Optional[list] = None) -> tuple:
    """The ``(height, width)`` of every plane pushed into the array viewer for this run.

    A per-FOV operator pushes a FRAME, so the surface is the frame shape. A REGION operator pushes
    a whole-region MOSAIC, so the surface is the mosaic EXTENT. Either is scaled into a ``_PUSH_PX``
    box PRESERVING ASPECT — a freeform 27-FOV strip is not a square, and squashing it into one
    (which is what a fixed ``(_PUSH_PX, _PUSH_PX)`` surface did) is how it arrives unrecognisable.

    Aspect is preserved the same way :func:`squidmip._placement.cell_boxes` preserves it for the
    plate thumbnail, so the plate and the array viewer describe one geometry rather than two.
    Falls back to the square when the extent is not derivable (no stage positions / pixel size);
    the caller reports that fallback rather than showing a silently squashed mosaic.
    """
    extent = region_mosaic_extent_px(meta, regions) if region_op else None
    if extent is None:                              # per-FOV op, or a region op with no geometry
        extent = tuple(meta["frame_shape"])
    mh, mw = int(extent[0]), int(extent[1])
    s = min(_PUSH_PX / mh, _PUSH_PX / mw, 1.0)     # never UPSCALE: a push is a bounded thumbnail
    return (max(1, int(round(mh * s))), max(1, int(round(mw * s))))


def _fit_letterboxed(a: np.ndarray, h: int, w: int, dtype) -> np.ndarray:
    """Scale a 2D plane into an exactly ``(h, w)`` canvas, aspect preserved and centred.

    The array viewer's canvas is declared ONCE per run (``start_acquisition``), so every push has
    to be that exact shape — while two regions of one acquisition can have differently shaped
    mosaics. Letterboxing is the only way to satisfy both without stretching one of them, and it
    is the policy the plate cell already uses (``cell_boxes`` centres a mosaic in its cell).
    """
    h, w = max(1, int(h)), max(1, int(w))
    s = min(h / a.shape[0], w / a.shape[1])
    ih = max(1, min(h, int(round(a.shape[0] * s))))
    iw = max(1, min(w, int(round(a.shape[1] * s))))
    out = np.zeros((h, w), dtype)
    out[(h - ih) // 2:(h - ih) // 2 + ih,
        (w - iw) // 2:(w - iw) // 2 + iw] = _fit_box(a, ih, iw)
    return out


# --- channel bar: one row per channel, under the plate overview -----------------------------

class _ChannelBar(QWidget):
    """Per-channel VISIBILITY for the plate, one compact row per channel, plus a contrast READOUT.

    A row is  <color dot> [x] <name>  …  <lo – hi>.  The checkbox masks that channel out of the
    plate composite; PlateOverview recomposites from its retained per-channel store, so nothing
    is re-read and nothing is re-projected.

    THERE ARE NO CONTRAST SLIDERS HERE, AND THERE MUST NOT BE (IMA-261)
    -------------------------------------------------------------------
    This strip used to carry a low/high QSlider pair and an "auto" button per channel —
    duplicating the contrast control the embedded ndviewer_light array viewer already has, two
    hand-widths apart on the same screen. Two controls over one quantity is the shape this
    project has now shipped a defect in four times, and it had already gone wrong here: the plate
    followed these sliders, the array viewer followed its own, and the SAME channel was displayed
    at two different windows side by side.

    Contrast therefore has exactly ONE owner — the central array viewer — and this strip only
    REPORTS the window that owner resolved (``set_window``, driven by
    ``LightweightViewer.contrastChanged`` → ``PlateWindow._on_detail_contrast``). A readout is not
    a second control surface: it cannot be dragged, it cannot disagree, and it is what makes the
    sync visible on screen instead of merely asserted in a commit message.
    """

    def __init__(self, labels, colors: np.ndarray, overview: PlateOverview):
        super().__init__()
        self._overview = overview
        self._rows = []           # per channel: (checkbox, contrast readout label)
        self.setStyleSheet(f"background:{_BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 5, 12, 6)
        lay.setSpacing(3)
        for c_i, label in enumerate(labels):
            row = QHBoxLayout()
            row.setSpacing(8)
            dot = QLabel("\u25cf")   # the channel's own LUT color, so the strip reads as a legend too
            dot.setStyleSheet("color:rgb({});".format(",".join(str(int(v * 255)) for v in colors[c_i])))
            box = QCheckBox(str(label))
            box.setChecked(True)
            box.setStyleSheet(_CHECK_QSS)
            box.toggled.connect(lambda on, i=c_i: self._overview.set_channel_visible(i, on))
            win = QLabel("\u2014")
            win.setStyleSheet("color:#8b949e;")   # dimmer than a control: this REPORTS, never sets
            win.setToolTip("contrast window, owned by the array viewer on the right \u2014 set it there")
            row.addWidget(dot)
            row.addWidget(box)
            row.addStretch(1)
            row.addWidget(win)
            lay.addLayout(row)
            self._rows.append((box, win))

    def set_window(self, ch: int, lo: float, hi: float):
        """Show the window the CENTRAL VIEWER resolved for *ch*. Display only — it sets nothing."""
        if 0 <= ch < len(self._rows):
            self._rows[ch][1].setText(f"{lo:g} \u2013 {hi:g}")


# --- operator worker: stream a projection over the plate, fill row-major -------------------

class _OperatorWorker(QThread):
    """Runs an operator (MIP) over the plate AND persists it as a navigable multiscale OME-Zarr plate
    (``write_plate``), filling one thumbnail per well as each is written. Projection + pyramid write
    run in write_plate's bounded producer/writer pools; our ``_on_well`` renders the plate tile and
    is called FROM THOSE WRITER THREADS, several at once — so only the done-counter needs ``_lock``
    (the expensive per-channel downsample happens OUTSIDE it, so downsampling still parallelises).
    The worker is deliberately THIN: it emits one native-dtype tile per FIELD — the whole 88x88 cell
    for a single-FOV well, or just that FOV's sub-cell box for a mosaic (IMA-187) — and keeps no
    pixels of its own. PlateOverview owns the per-channel store, the contrast and the compositing,
    so the channel toggle works on every entry path, not only after a run. Memory stays O(engine +
    write workers) wells in flight. The written ``plate.ome.zarr`` is the durable, re-openable artifact.
    """

    tileReady = pyqtSignal(int, int, str, object, object)   # (ri, ci, well_id, (C,h,w) native tile,
    #                                                          box=(top,left,h,w) in cell px | None)
    progress = pyqtSignal(int, int)                 # (done, total)
    streamEnded = pyqtSignal()                      # every well landed -> recomposite the whole plate
    writtenReady = pyqtSignal(str)                  # path of the written plate.ome.zarr
    wellFailed = pyqtSignal(int, int)               # (ri, ci) of a well SKIPPED on a read error
    pushReady = pyqtSignal(int, object)             # (fov_idx, [per-channel ~512px plane]) for the slider
    failed = pyqtSignal(str)                        # whole-run failure (not a per-well skip)
    finished_ok = pyqtSignal()

    def __init__(self, operator: str, reader, meta, fov_index: dict, out_dir: str,
                 regions=None, save: bool = True, n_fovs=1, operator_kwargs=None):
        # No nr/nc: the worker no longer builds a plate-sized montage of its own (IMA-206 moved the
        # canvas into PlateOverview), so it has no use for the plate's shape.
        super().__init__()
        self._operator = operator
        self._reader, self._meta = reader, meta
        self._fov_index = fov_index
        self._out_dir = out_dir
        self._regions = regions          # None = whole plate; a list = subset preview (those wells only)
        self._save = save                # False = PREVIEW: compute + push to the viewer, write NOTHING
        self._n_fovs = n_fovs            # None = every FOV per well -> coordinate-placed mosaic tiles
        # Per-run parameters for a REGION operator (registration on/off, registration channel,
        # feather width, blunder thresholds, channel subset) -- the stitcher panel in pane 1
        # sets these. Carried on BOTH branches of run(): a setting tuned on a preview and then
        # dropped on the save would be thrown away at exactly the moment it is written to disk.
        # A projector has no equivalent seam (its parameters are baked in at registration), and
        # write_plate refuses these for one by name rather than accepting and dropping them.
        self._operator_kwargs = dict(operator_kwargs or {})
        # Per-region FOV boxes inside the _CELL thumbnail (IMA-187). Computed ONCE up front from
        # the reader's stage positions, because every arriving FOV needs its box and the geometry
        # never changes mid-run. Empty dict => no positions => the historical single-tile path.
        # IMA-222: a REGION operator (stitch) returns ONE fused mosaic per well, not one array
        # per FOV, so there are no per-FOV sub-boxes to composite into -- the mosaic IS the cell.
        # A non-empty _boxes here would slot a whole-well mosaic into a single FOV's sub-rectangle.
        from squidmip import available_region_operators

        self._region_op = self._operator in available_region_operators()
        self._boxes = {} if (self._region_op or n_fovs == 1) else _mosaic_boxes(meta)
        # IMA-245: the shape of what this run PUSHES to the array viewer. A region operator pushes
        # a whole-region mosaic, so the surface is the mosaic extent (aspect preserved), not the
        # frame. Computed here, once, and read back by the window through `push_shape` — the window
        # declares the viewer's canvas from the SAME number the worker fills, so the producer and
        # the consumer cannot describe two different rectangles.
        self._push_shape = push_shape_for(meta, self._region_op, regions)
        # True when a region run wanted the mosaic extent and could not derive it (no stage
        # positions / no pixel size), so the push falls back to the square frame surface. The
        # window turns this into a readout line: a squashed mosaic must not look like a correct one.
        self._push_shape_estimated = bool(self._region_op
                                          and region_mosaic_extent_px(meta, regions) is None)
        self._total = len(regions) if regions is not None else len(meta["regions"])
        self._channels = [c["name"] for c in meta["channels"]]
        self._dtype = np.dtype(meta["dtype"])
        self._lock = threading.Lock()             # guards _done (on_well runs on writer threads)
        self._done = 0
        self._seen_fovs: dict[tuple, set] = {}    # (ri,ci) -> FOVs composited so far, for progress
        self._failed_regions: set = set()         # regions whose fields raised (IMA-226: report it)
        self._stop = threading.Event()            # set by the window to end the run cleanly

    @property
    def mosaic_boxes(self) -> dict:
        """``{(region, fov): (top, left, h, w)}`` this run composites into ({} = single-tile path).

        The plate view must hit-test against the SAME boxes the worker paints into, so it reads
        them from here rather than recomputing — a second `_mosaic_boxes(meta)` call would be a
        second chance to disagree, and a disagreement opens a different FOV than the one clicked.
        """
        return self._boxes

    @property
    def push_shape(self) -> tuple:
        """``(h, w)`` of every plane this run pushes to the array viewer (IMA-245).

        Read by the window to size ``start_acquisition``. Same reasoning as ``mosaic_boxes``:
        recomputing it there would be a second chance to disagree, and a disagreement here is a
        black viewer — the push is rejected for the wrong shape and the rejection is invisible.
        """
        return self._push_shape

    @property
    def push_shape_estimated(self) -> bool:
        """True when a REGION run could not derive its mosaic extent and fell back to the square
        frame surface. The window reports it; a squashed mosaic must not pass for a correct one."""
        return self._push_shape_estimated

    @property
    def landed(self) -> int:
        """Wells that actually produced pixels. IMA-226: a live run whose every well raised used to
        finish "✓ · 1 well" with an empty plate behind it — flat-field with no illumination profile
        raises per field, `_on_error` painted the dots red, and the success message printed anyway.
        The status line must not claim a result the plate does not have."""
        with self._lock:
            return self._done

    @property
    def skipped(self) -> int:
        """Regions where at least one field raised and was skipped."""
        with self._lock:
            return len(self._failed_regions)

    def stop(self):
        """Ask the run to stop; write_plate polls this and abandons after in-flight wells drain."""
        self._stop.set()

    def _on_well(self, region, fov, image):
        """Called per written FIELD (on a write_plate WRITER THREAD): composite the plate thumbnail.

        Single-FOV (the historical path) fills the whole _CELL tile. Multi-FOV composites this
        FOV into its coordinate-derived box inside the SAME cell, accumulating across the calls
        for that region — so a 36-FOV well is built from 36 arrivals rather than 36 overwrites.
        Compositing is at THUMBNAIL scale throughout: the cell is _CELL x _CELL no matter how
        many FOVs land in it, so a mosaic well costs the same memory as a single-FOV well.
        """
        info = self._fov_index[region]
        ri, ci, well_id = *info["rc"], info["well_id"]
        well = image[0, :, 0]  # (C, Y, X)
        box = self._boxes.get((region, fov))
        n_c = len(self._channels)

        # Downsample OUTSIDE the lock (the expensive part stays parallel). Single-FOV fills the
        # whole cell; a mosaic FOV is fitted to its own sub-cell box and carries that box along,
        # so the widget can slot it in at the right offset without knowing the geometry.
        if box is None:
            tiles = [_fit_cell(well[c_i]) for c_i in range(n_c)]
            bh = bw = _CELL
        else:
            top, left, bh, bw = box
            tiles = [_fit_box(well[c_i], bh, bw) for c_i in range(n_c)]
        raw = np.empty((n_c, bh, bw), self._dtype)   # native dtype (half the RAM)
        for c_i, ds in enumerate(tiles):
            raw[c_i] = ds
        with self._lock:                          # shared counter -> serialize (the cheap part)
            seen = self._seen_fovs.setdefault((ri, ci), set())
            was_empty = not seen
            seen.add(fov)
            if was_empty:                          # count WELLS, not fields, so the bar still
                self._done += 1                    # reads "n of n wells" on a 36-FOV plate
            done = self._done
        # per-channel + its box; the widget windows, places and composites (IMA-206 + IMA-187)
        self.tileReady.emit(ri, ci, well_id, raw, box)
        self.progress.emit(done, self._total)
        # feed the ndviewer growing slider: one ~512px plane per channel, in memory (register_array),
        # so scrubbing the processed wells is instant and z-collapsed (nz=1). Downsampled -> bounded.
        # ...at `push_shape`: the frame square for a per-FOV operator, the aspect-preserved mosaic
        # extent for a REGION operator (IMA-245). Squashing a region mosaic into the frame square
        # is what put a whole-well stitch into the array viewer as an unreadable rectangle.
        ph, pw = self._push_shape
        push = [_fit_letterboxed(well[c_i], ph, pw, self._dtype)
                for c_i in range(len(self._channels))]
        self.pushReady.emit(info["idx"], push)

    def _on_error(self, region, fov, exc):
        """A well's projection failed (corrupt/missing plane): SKIP it, mark its dot failed, keep the
        run alive. One bad file must not abort a whole-plate run."""
        with self._lock:
            self._failed_regions.add(region)
        info = self._fov_index.get(region)
        if info is not None:
            self.wellFailed.emit(*info["rc"])

    def run(self):
        try:
            projector = self._operator
            if self._save:
                # write_plate picks its own stream from the operator, so a region operator (stitch)
                # persists through the same call: both twins yield (region, fov, (T,C,1,Y,X)), and
                # the disk guard sizes a region write from real mosaic extents rather than frames.
                from squidmip import write_plate  # persist + project in one bounded, streaming pass

                write_plate(self._reader, self._out_dir, n_fovs=self._n_fovs, workers=_VIEWER_WORKERS,
                            projector=projector, tiff=False, on_well=self._on_well,
                            stop=self._stop.is_set, on_error=self._on_error, regions=self._regions,
                            operator_kwargs=self._operator_kwargs or None)
                if self._stop.is_set():
                    return  # window closing / re-opening; drop out cleanly (no final/written emit)
                self.streamEnded.emit()
                self.writtenReady.emit(str(Path(self._out_dir) / "plate.ome.zarr"))
            else:
                # PREVIEW: run the engine over the subset and push each result to the plate + slider,
                # writing NOTHING to disk (so testing an operator on a few wells costs no disk + only
                # the subset's compute). Same math as the saved run — a faithful preview.
                if self._region_op:
                    # IMA-222: a region operator's unit of work is the WELL, so stitch_plate yields
                    # one fused mosaic per region. It mirrors project_plate's contract exactly
                    # (bounded in-flight window, regions=, on_error=, and the same
                    # (region, fov, (T, C, 1, Y, X)) yield), so the loop below is UNCHANGED.
                    # workers=1: peak memory is workers x one fused mosaic (~0.9 GB on a 27-FOV 10x
                    # well), not the ~139 MB of one projected FOV. Saving takes the write_plate
                    # branch above, which dispatches to stitch_plate itself.
                    from squidmip import stitch_plate

                    stream = stitch_plate(self._reader, workers=1, operator=projector,
                                          n_fovs=None, on_error=self._on_error,
                                          regions=self._regions, **self._operator_kwargs)
                else:
                    from squidmip import project_plate

                    stream = project_plate(self._reader, workers=_VIEWER_WORKERS, projector=projector,
                                           n_fovs=self._n_fovs, on_error=self._on_error,
                                           regions=self._regions)
                try:
                    for region, fov, image in stream:
                        if self._stop.is_set():
                            return
                        self._on_well(region, fov, image)
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                if self._stop.is_set():
                    return
                self.streamEnded.emit()
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class _MinervaWorker(QThread):
    """Export the selection to Minerva-ingestable files, then start Minerva Author (IMA-228).

    Two stages with deliberately different failure semantics::

        export  ──ok──▶  launch  ──ok──▶  exported(paths) + launched(True)
           │                │
           │                └──fail──▶  exported(paths) + launched(False)   ← files still good
           └──fail──▶  failed(msg)                                          ← nothing written

    A launch failure must NEVER invalidate a successful export: Minerva Author lives in a
    separate checkout that may not be installed, and the OME-TIFF on disk is the deliverable.
    The user always gets the story path either way, because Minerva has no deep link — the
    file is picked by hand in its "Select File" dialog.
    """
    progress = pyqtSignal(int, int)          # (done, total) FOVs exported
    exported = pyqtSignal(object)            # [(ome_path, story_path), ...]
    launched = pyqtSignal(bool)              # did a Minerva server end up answering?
    failed = pyqtSignal(str)
    finished_ok = pyqtSignal()

    def __init__(self, reader, selection, out_dir, projector: str, t: int = 0, launch: bool = True):
        super().__init__()
        self._reader = reader
        self._selection = list(selection)
        self._out_dir = out_dir
        self._projector = projector
        self._t = t
        self._launch = launch
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip import _minerva
        try:
            pairs = []

            def on_progress(done, total):
                self.progress.emit(done, total)

            # Export REGION by REGION — the export unit is a fused mosaic per region, so this
            # is also the finest granularity a stop can act on. A stop between regions takes
            # effect promptly; every file already written stays on disk and is reported.
            grouped = _minerva.group_selection(self._selection)
            for i, (region, fovs) in enumerate(grouped.items()):
                if self._stop.is_set():
                    break
                pairs.extend(
                    _minerva.export_selection(
                        self._reader, [(region, f) for f in fovs], self._out_dir,
                        t=self._t, projector=self._projector,
                    )
                )
                on_progress(i + 1, len(grouped))
            self.exported.emit(pairs)
            if pairs and self._launch and not self._stop.is_set():
                # should_stop: the liveness wait is up to 90 s and closeEvent joins this thread.
                # Without it, closing mid-poll froze the GUI for the rest of the wait (84 s).
                self.launched.emit(
                    _minerva.launch_minerva(pairs[0][1], should_stop=self._stop.is_set))
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class _MosaicWorker(QThread):
    """Fuse one region's FOVs into a mosaic per channel, OFF the GUI thread.

    A 28-FOV tissue region is ~940 MB of TIFF to read. Doing that in ``ingest`` would freeze the
    window for seconds on open, which is precisely the "opens instantly" property IMA-260 bought.
    Results arrive per channel so the first channel paints while the rest are still being read.
    """

    ready = pyqtSignal(str, str, object, object)   # region, channel, plane, bbox_um|None
    problem = pyqtSignal(str)
    finished_count = pyqtSignal(int)

    def __init__(self, reader, meta, region, channels, parent=None):
        super().__init__(parent)
        self._reader, self._meta = reader, meta
        self._region = region
        self._channels = list(channels)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip._mosaic_source import fuse_region_pyramid, mosaic_bbox_um

        try:
            bbox = mosaic_bbox_um(self._meta, self._region)
        except Exception as exc:                    # noqa: BLE001
            self.problem.emit(f"mosaic placement failed: {exc}")
            bbox = None

        n = 0
        for ch in self._channels:
            if self._stop.is_set():
                break
            try:
                # A LAZY MULTISCALE PYRAMID of (z, y, x) levels — the same shape of data the
                # written-OME-Zarr path has always handed napari via open_pyramid. napari fetches
                # only the clipped visible region of the level matching the current zoom, so a
                # fit-to-window view costs a coarse level (~0.9 MB) instead of a full-resolution
                # fused plane (54.9 MB on the real 10x region), per channel, per z step.
                #
                # napari's own dimension slider is still the z control: every level keeps the z
                # axis at full length and only y/x are coarsened. Only the visible (level, z) is
                # ever materialised, and _mosaic_source's bounded cache keeps a revisited one.
                res = fuse_region_pyramid(self._reader, self._meta, self._region, ch)
            except Exception as exc:                # noqa: BLE001 - reported, never swallowed
                self.problem.emit(f"{self._region}/{ch}: {type(exc).__name__}: {exc}")
                continue
            if res is None:
                self.problem.emit(
                    f"{self._region}: no stage positions / pixel size — mosaic not derivable."
                )
                continue
            levels, _step, _nz = res
            self.ready.emit(self._region, ch, levels, bbox)
            n += 1
        self.finished_count.emit(n)


class _PreviewWorker(QThread):
    """Fast RAW preview so the plate shows imagery the moment it opens — before any operator runs
    (the "downsample the plate before opening" step). Reads ONE representative z-plane per channel
    per FOV (not the whole stack), area-downsamples, and hands the per-channel tile to the plate.
    Cheap relative to a full projection; parallel reads. Status stays 'empty' (grey frame) — this is
    a preview, not a processed result. A later operator overwrites each tile. Like the other
    producers it keeps the CHANNEL AXIS intact all the way to the widget, so the channel toggle
    works on a freshly-opened acquisition, before any operator has run.

    A REGION IS A MOSAIC, NOT A FOV (IMA-253/IMA-249). This used to read exactly one representative
    FOV per region and stretch it over the region's whole cell, so the real 10x tissue acquisition
    showed two lone frames pretending to be two 27- and 28-FOV mosaics, and the mosaic only ever
    appeared *after* an operator run. It now composites every FOV of a region into that region's
    cell at its coordinate-derived box (``_placement.cell_boxes`` — the same geometry the operator
    path uses, so preview and result describe one layout).

    The cost is driven by the REAL FOV COUNT PER REGION, which is the only way both datasets stay
    fast: the 1536-well fixture is 1536 regions x 1 FOV, so it reads 1536 planes per channel exactly
    as before, takes the identical single-tile code path (``box=None``), and cannot get slower. The
    tissue slide is 2 regions x ~27 FOVs, so it reads 55. Work is emitted per FOV as it lands, so
    cells fill progressively and the UI never blocks on a whole mosaic.
    """

    tileReady = pyqtSignal(int, int, str, object, object)   # (ri, ci, well_id, tile, box|None)
    streamEnded = pyqtSignal()                      # preview complete -> recomposite the whole plate

    def __init__(self, reader, meta, fov_index: dict, order: list, mosaic: bool = True):
        super().__init__()
        self._reader, self._meta = reader, meta
        self._fov_index, self._order = fov_index, order
        self._channels = [c["name"] for c in meta["channels"]]
        self._dtype = np.dtype(meta["dtype"])
        self._mosaic = bool(mosaic)
        self._stop = threading.Event()

    def _plan(self) -> list:
        """``[(region, fov, box|None), ...]`` — the read list, in plate order, FOVs in stage order.

        ``box=None`` means "this FOV fills its cell": a single-FOV region, or an acquisition with no
        stage coordinates / no pixel size, where a mosaic is not derivable and guessing one would
        draw a wrong picture. That is the historical path, and it stays byte-identical for it.

        A region whose FOVs are so widely spread that each one lands in fewer than
        ``_MIN_PREVIEW_BOX_PX`` cell pixels also previews single-tile. Reading N planes to paint N
        specks is all cost and no picture — and at that scale the "mosaic" and the single tile are
        visually the same thing anyway. The operator path still mosaics it; this is a PREVIEW
        budget, not a change to the geometry.
        """
        from squidmip._placement import cell_boxes, fov_offsets_px

        positions = self._meta.get("fov_positions_um") or {}
        px = self._meta.get("pixel_size_um")
        plan: list = []
        for region in self._order:
            fovs = list(self._meta["fovs_per_region"][region])
            boxes: dict = {}
            if self._mosaic and len(fovs) > 1 and positions and px not in (None, 0):
                try:
                    boxes = cell_boxes(fov_offsets_px(positions, region, fovs, px),
                                       self._meta["frame_shape"], _CELL)
                except (KeyError, ValueError):
                    boxes = {}       # this region previews single-tile; the rest still mosaic
                if any(min(b[2], b[3]) < _MIN_PREVIEW_BOX_PX for b in boxes.values()):
                    boxes = {}
            if boxes:
                plan.extend((region, f, boxes[f]) for f in fovs if f in boxes)
            else:
                plan.append((region, fovs[0], None))
        return plan

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            from concurrent.futures import ThreadPoolExecutor
            zs = self._meta["z_levels"]
            z_mid = zs[len(zs) // 2]      # a mid-stack plane is a fair single-plane preview
            plan = self._plan()

            def load(item):
                region, fov, box = item
                h, w = (_CELL, _CELL) if box is None else (box[2], box[3])
                fit = _fit_cell if box is None else (lambda a: _fit_box(a, h, w))
                return region, box, [fit(self._reader.read(region, fov, ch, z_mid)
                                         .astype(np.float32)) for ch in self._channels]

            with ThreadPoolExecutor(max_workers=_VIEWER_WORKERS) as ex:
                for region, box, tiles in ex.map(load, plan):   # plate order preserved
                    if self._stop.is_set():
                        return
                    ri, ci = self._fov_index[region]["rc"]
                    self.tileReady.emit(ri, ci, region,
                                        np.stack(tiles).astype(self._dtype), box)
            if not self._stop.is_set():
                self.streamEnded.emit()   # the running window is mature now -> one clean recomposite
        except Exception:
            pass   # preview is best-effort; the operator run is the authoritative result


class _ComputedPlateWorker(QThread):
    """Read a previously-written OME-Zarr plate back into the viewer (no recompute).

    Streams each well from disk: a coarse pyramid level -> the plate thumbnail, and a ~512px level ->
    the ndviewer slider (register_array). Bounded (one well in flight); reads via tensorstore off the
    GUI thread so opening a big computed plate never freezes the window. Emits per-channel tiles, so
    a reopened plate is windowed GLOBALLY by the widget's running contrast exactly like a live run —
    it used to take percentiles per well, which made a dim well and a bright well look identical and
    silently broke the one thing a plate overview is for (comparing wells at a glance)."""

    tileReady = pyqtSignal(int, int, str, object)   # (ri, ci, well_id, (C, cell, cell) native tile)
    pushReady = pyqtSignal(int, object)             # (fov_idx, [per-channel ~512px plane])
    progress = pyqtSignal(int, int)
    streamEnded = pyqtSignal()                      # plate fully loaded -> recomposite globally
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, base, wells, coarse_lvl, push_lvl, dtype):
        super().__init__()
        self._base = base                 # plate.ome.zarr path
        self._wells = wells               # [(well_id, wellpath, fov, ri, ci, flat_idx)]
        self._coarse, self._push = coarse_lvl, push_lvl
        self._dtype = dtype
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _read(self, wellpath, fov, level):
        import tensorstore as ts
        path = f"{self._base}/{wellpath}/{fov}/{level}"
        arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": path}}).result()
        return np.asarray(arr[0, :, 0].read().result())   # (C, y, x) at t=0, z=0

    def run(self):
        try:
            n = len(self._wells)
            for i, (wid, wpath, fov, ri, ci, idx) in enumerate(self._wells, 1):
                if self._stop.is_set():
                    return
                coarse = self._read(wpath, fov, self._coarse)             # thumbnail source (C,y,x)
                tile = np.stack([_fit_cell(plane.astype(np.float32)) for plane in coarse])
                self.tileReady.emit(ri, ci, wid, tile.astype(self._dtype))
                push_src = self._read(wpath, fov, self._push)             # detail-slider source (C,Y,X)
                # ...at the declared push canvas exactly (IMA-245): a pyramid level smaller than
                # _PUSH_PX used to be pushed at its own size, which the viewer silently refused.
                push = [_fit_letterboxed(push_src[c], _PUSH_PX, _PUSH_PX, self._dtype)
                        for c in range(push_src.shape[0])]
                self.pushReady.emit(idx, push)
                self.progress.emit(i, n)
            if not self._stop.is_set():
                self.streamEnded.emit()   # every well in the store -> one global-window recomposite
                self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# --- main window: plate overview | embedded ndviewer ----------------------------------------

class _ExplorationTab(QWidget):
    """One 'exploration' tab: a saved FOV/region subset plus the operator UI scoped to it (IMA-205).

    Multi-instance by design — one per selection — which is why it does NOT reuse an operator
    tab's fixed key. Identity is content-addressed (``exploration_tab_key``), so re-selecting the
    same wells focuses this tab instead of opening a second copy of it.

        selection {B2,B3,B4}
              |
              v  exploration_tab_key(acq, regions) -> "exp:1a2b3c"
        _ExplorationTab(regions)  --run--> run_operator(op, regions=..., tab_key="exp:1a2b3c")
              |                                              |
              |                                    layer "<op>@exp:1a2b3c"
              +--close--> stop run, drop layers, free canvases
    """

    def __init__(self, regions: list, tab_key: str, parent=None):
        super().__init__(parent)
        self.regions = list(regions)
        self.tab_key = tab_key
        self.status: dict = {}      # this tab's plate dots, restored when it becomes active
        self.sync_note = None       # set by _build_exploration_tab; the "not synced yet" banner
        self.sync_pending = False   # True while this tab is in front but the view still shows a run

    def set_sync_pending(self, pending: bool):
        """Say out loud that this tab is in FRONT but the plate/detail beside it still belong to a
        run that is finishing. A tab which silently shows someone else's wells is the whole bug —
        the banner is the honest state until _on_run_drained catches the view up."""
        self.sync_pending = bool(pending)
        if self.sync_note is not None:
            self.sync_note.setVisible(self.sync_pending)

    def shutdown(self):
        """Called by _close_op_tab (duck-typed, like the CLI terminal's). The window does the real
        teardown in _discard_exploration — this exists so the hasattr(w, 'shutdown') path is safe."""
        return


def _make_mosaic_pane():
    """Build pane 2's napari mosaic viewer, or report why it could not be built.

    Returns ``(pane_or_None, mode, message)``. Import failures are caught here rather than at
    module import so that a machine without napari still opens the window with ndviewer_light —
    and with a VISIBLE sentence saying so, never a silent downgrade.
    """
    try:
        from squidmip._napari_pane import make_pane

        return make_pane()
    except Exception as exc:                     # noqa: BLE001 - surfaced, not swallowed
        return None, "ndv", f"napari viewer unavailable ({type(exc).__name__}: {exc}) — using ndviewer_light."


class PlateWindow(QMainWindow):
    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("MIP tool")
        self.resize(1600, 950)
        self._worker = None           # the operator (MIP) run
        self._preview = None          # the raw preview fill on open
        self._minerva = None          # the Minerva export + Author launch (IMA-228)
        self._retired = []            # workers asked to stop; kept alive until they actually finish
        self._overview = None
        self._reader = None
        self._meta = None
        self._mosaic_worker = None    # fuses a region's FOVs for pane 2, off the GUI thread
        self._mosaic_region = None    # region currently shown in the napari mosaic pane
        self._fov_index = {}
        self._selected_regions = []   # wells picked on the plate (IMA-221); scopes an operator run
        self._pushed = set()          # wells whose raw z-stack is already registered in the detail viewer
        self._channel_bar = None      # per-channel toggle + contrast strip under the plate (IMA-206)

        # File menu: a reliable "Open acquisition folder" (drag-drop can be blocked on Windows by the
        # GL child pane or an elevation mismatch, so this is the always-works path).
        file_menu = self.menuBar().addMenu("&File")
        open_act = QAction("&Open acquisition folder…", self)
        open_act.triggered.connect(self._open_acquisition_dialog)
        file_menu.addAction(open_act)
        open_hcs = QAction("Open a &computed MIP (.hcs)…", self)
        open_hcs.triggered.connect(self._open_computed)
        file_menu.addAction(open_hcs)

        # Process-well-plates menu (operators). MIP is #1; disabled until an acquisition is open.
        self._op_actions = {}
        proc_menu = self.menuBar().addMenu("&Process well-plates")
        for op in _OPERATIONS:
            act = QAction("&" + op.label, self)
            act.setEnabled(False)
            act.triggered.connect(lambda _=False, k=op.key: self._activate_operator(k))
            proc_menu.addAction(act)
            self._op_actions[op.key] = act

        self._acq_name = ""           # acquisition folder name, shown as the Process-pane title
        self._current_well = None     # the well currently shown in the detail viewer (for Record)
        self._current_fov = 0         # the FOV of that region on screen (IMA-250: autofocus ranks IT)
        self._acq_path = None         # the opened acquisition dir (persist writes next to it)
        self._processed_plate = None  # path of the written plate.ome.zarr once an operator persists it
        self._plate_mode = "raw"      # what the plate view is showing — shown in the plate-pane title
        self._plate_format = None     # the format the plate is laid out with (declared or inferred)
        self._plate_format_override = None   # manual override; also read from SQUIDMIP_WELLPLATE_FORMAT
        self._op_stack = OperationStack()   # the toggleable layer stack (base + applied operators)
        self._active_op_key = None    # operator whose tiles are streaming into its layer right now
        self._layers_tab = None       # the Layers tab widget, once opened
        self._order = []              # well order = the detail's FOV-slider order
        self._op_tabs = {}            # key -> operator-UI widget currently open as a tab in _left_tabs
        self._floating = {}           # key -> _FloatWindow holding that operator's UI detached
                                      # (a key lives in exactly ONE of the two dicts, never both)
        self._push_index = None       # global plate idx -> current run's slider position (None = identity)
        # IMA-245: the (h, w) canvas the array viewer was last declared with, and the sticky reason
        # a push could not be shown. None = no array canvas declared (the raw path registers file
        # paths, not arrays), which is the signal for _on_push to skip the shape check.
        self._push_shape = None
        self._push_problem = None
        self._readout_base = ""
        self._dropped_pushes = 0
        self._active_exploration = None   # the exploration tab currently in front, if any
        self._control_well = None     # THE control well (IMA-248/IMA-260): one region id, owned
        #                               here, mirrored onto the plate frame and pane 3's pinned tab
        self._tabs_muted = False      # suppress _on_tab_changed during bulk teardown (ingest)
        self._run_out_dir = None      # output dir of the in-flight SAVE run (for partial cleanup)
        self._run_tab_key = None      # exploration tab that owns the in-flight run, if any
        self._pending_resync = False  # a tab switch was deferred because a run was live (IMA-205 bugs)
        self._loupe_sources = {}      # layer key -> _LoupeSource backing that layer's pixels (IMA-208)

        # THREE HORIZONTAL PANES on one monitor (IMA-237). Tabs live inside a pane (their bar sits at
        # the pane's top, like the plate pane's title bar) — never a global strip across the window.
        # Any detachable tab can be DRAGGED OUT of its bar into a free-floating window (ImageJ-style;
        # see _detach_tab, which serves BOTH bars):
        #   PANE 1 = plate view + the controls with the tabs. A vertical split: on top the PROCESS
        #            console, a QTabWidget with a "Process wells" home tab (operator list) and one tab
        #            per operator you open (MIP -> where-to-save UI; Record -> recorder UI); below it
        #            the HCS PLATE view, whose title bar names the plate.
        #   PANE 2 = the initial viewer: the ndviewer_light array viewer, full height. ONE widget
        #            instance (never tabbed), but NOT plate-fixed: its FOV slider FOLLOWS the active
        #            exploration tab, which re-points it at that tab's subset; no exploration tab
        #            restores the whole plate (_on_tab_changed). ndviewer's only retarget seam is
        #            start_acquisition, which resets the viewer, so computed frames do not survive a
        #            switch; raw plane paths are re-registered so it isn't black.
        #   PANE 3 = the EXPLORATION pane: one tab per Shift-dragged FOV subset (IMA-205/221).
        #
        # Pane 3 is VISIBLE FROM OPEN (IMA-260), reversing IMA-237's reveal-on-first-drag. The
        # saving of a fifth of the monitor bought undiscoverability: you cannot find a pane that is
        # not there, so nobody found the Shift-drag that was the only way to make it appear. It now
        # opens showing EXAMPLE USAGE (_build_explore_empty) and swaps to the tab bar the moment it
        # holds real content — a pane that teaches costs its width back immediately.
        # Exploration tabs moved OUT of the process console to get here: the console is pane 1, and
        # pane 1 is not where the user asked exploration to live.

        # top-left: the process console (build the home tab first — it owns self._readout, which
        # _make_detail_viewer writes to if ndviewer is unavailable).
        self._left_tabs = _DetachTabs(self._detach_tab)
        # Dark the tab widget's own canvas (the strip behind/beside the tabs rendered white in macOS
        # light mode). Scope a Fusion style + dark palette to THIS widget subtree only — NOT the app,
        # which would bleed into the embedded ndviewer and hide its per-channel colour swatches.
        self._fusion_style = QStyleFactory.create("Fusion")   # keep a ref: setStyle doesn't own it
        if self._fusion_style is not None:
            self._left_tabs.setStyle(self._fusion_style)
        self._left_tabs.setPalette(_dark_palette())
        self._left_tabs.setAutoFillBackground(True)
        self._left_tabs.setStyleSheet(_TABS_DARK)
        self._left_tabs.setTabsClosable(True)
        self._left_tabs.tabCloseRequested.connect(self._close_op_tab)
        self._left_tabs.currentChanged.connect(self._on_tab_changed)
        self._left_tabs.addTab(self._build_process_pane(), "Process wells")
        self._left_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)  # home tab isn't closable

        # PANE 3: the exploration pane. Same _DetachTabs class as the console (one detach seam, not
        # two), but every tab is detachable — it has no permanent home tab to protect.
        self._explore_tabs = _DetachTabs(self._detach_tab, first_detachable=0)
        if self._fusion_style is not None:
            self._explore_tabs.setStyle(self._fusion_style)
        self._explore_tabs.setPalette(_dark_palette())
        self._explore_tabs.setAutoFillBackground(True)
        self._explore_tabs.setStyleSheet(_TABS_DARK)
        self._explore_tabs.setTabsClosable(True)
        self._explore_tabs.tabCloseRequested.connect(
            lambda i: self._close_op_tab(i, self._explore_tabs))
        self._explore_tabs.currentChanged.connect(self._on_tab_changed)
        # Pane 3 is a two-page STACK, not a bare tab bar: page 0 is the example-usage empty state,
        # page 1 is the tab bar. One widget owns "what pane 3 shows", so the empty copy and the
        # tabs can never be on screen together and can never both be off it. _explore_tabs.isHidden()
        # stays truthful for free — QStackedWidget hides the page that is not current — which is
        # exactly what "there are no exploration tabs" means, and what callers already read.
        self._explore_empty = self._build_explore_empty()
        self._explore_pane = QStackedWidget()
        self._explore_pane.setStyleSheet(f"background:{_BG};")
        self._explore_pane.addWidget(self._explore_empty)   # page 0: example usage
        self._explore_pane.addWidget(self._explore_tabs)    # page 1: real content
        # Wide enough to set 24 px copy without one word per line — the legibility floor is a floor
        # on the TEXT, and text you have to read a syllable at a time is not legible either.
        self._explore_pane.setMinimumWidth(360)

        # right (pane 2): the central viewer. napari renders the region MOSAIC as a multiscale
        # pyramid — the reason for the move, and the unit the user actually looks at (IMA-265).
        # ndviewer_light is the fallback viewer and, in fallback mode, the per-FOV z-stack detail
        # viewer; `self._detail` refers to it and every push path below is unchanged.
        #
        # It is NOT constructed when napari is the viewer. Building it anyway "just for the push
        # paths" leaves its LUT sliders in the widget tree, where they are a SECOND control
        # surface over contrast — invisible to the user, but two widgets that can move one value
        # is the defect either way, and the IMA-268 gate walks the tree and actuates everything it
        # finds. Measured with that gate: origin/main FAILS with 8 sliders + 4 auto buttons in the
        # plate view; with this, "contrast: 0 sliders, 0 auto buttons". napari's mosaic pane
        # supersedes the per-FOV detail viewer — it shows the region mosaic WITH a z slider — and
        # every `self._detail` path already guards on None.
        self._mosaic_pane, viewer_mode, viewer_msg = _make_mosaic_pane()
        self._detail = None if viewer_mode == "napari" else self._make_detail_viewer()
        if self._detail is not None:   # connect the FOV slider -> red box ONCE (not per ingest)
            slider = getattr(self._detail, "_fov_slider", None)
            if slider is not None:
                slider.valueChanged.connect(self._on_fov_slider)
            # CONTRAST HAS ONE OWNER, AND IT IS THE ARRAY VIEWER (IMA-261). Connected ONCE here,
            # for the same reason the FOV slider is: the detail viewer is a singleton that
            # outlives every ingest, and a per-ingest connect would stack duplicate slots.
            sig = getattr(self._detail, "contrastChanged", None)
            if sig is not None:
                sig.connect(self._on_detail_contrast)
            elif viewer_mode != "napari":
                # Fail LOUD, in the readout: a silent no-sync is the bug. Only when ndviewer_light
                # is ACTUALLY the viewer, though — under napari the contrast owner is
                # layer.contrast_limits and this signal is irrelevant, so warning about it there
                # would be a false alarm, and a warning that cries wolf gets ignored.
                self._readout.setText(
                    "this ndviewer_light has no contrastChanged signal — the plate cannot follow "
                    "the array viewer's contrast (needs the IMA-261 build of ndviewer_light)")

        if viewer_mode == "napari" and self._mosaic_pane is not None:
            self._right_widget = self._mosaic_pane
        elif self._detail is not None:
            self._right_widget = self._detail
            # A fallback the user cannot see is a silent failure. Say it in the status line.
            if viewer_msg:
                self._readout.setText(viewer_msg)
        else:
            ph = QLabel(
                (viewer_msg + "\n" if viewer_msg else "")
                + "ndviewer_light unavailable — pip install ndviewer_light"
            )
            ph.setAlignment(Qt.AlignCenter)
            ph.setWordWrap(True)
            ph.setStyleSheet("color:#8b98ad;")
            self._right_widget = ph

        # bottom-left: plate view (drop target until an acquisition opens). Its FIXED title bar names
        # the wellplate we're on (the acquisition) — the plate's identity lives with the plate.
        self._plate_title = QLabel("well plate")   # plate name; shows the hovered well (large) on hover
        self._plate_title.setStyleSheet(           # the BAR below now carries background + border
            "color:#e6edf3;font-size:17px;font-weight:800;padding:9px 14px;border:none;")
        # CONTRAST SCOPE selector (IMA-207), in the plate's OWN title bar because it governs the
        # plate, not the run — flipping it re-composites the retained tiles and never re-runs.
        self._scope_combo = QComboBox()
        self._scope_combo.setStyleSheet(_COMBO_QSS)
        self._scope_combo.addItems(list(SCOPES))
        self._scope_combo.setToolTip(
            "How wide a net each contrast window is computed over.\n"
            "global — one window across the plate; wells stay comparable.\n"
            "per-region — every well fills its own range, so a dim and a bright well are both\n"
            "readable, but they are NO LONGER comparable.")
        self._scope_combo.currentTextChanged.connect(self._on_scope_changed)
        _scope_lbl = QLabel("contrast")
        _scope_lbl.setStyleSheet("color:#8b98ad;font-size:12px;border:none;")
        plate_title_bar = QWidget()
        plate_title_bar.setStyleSheet("background:#0b0e14;border-bottom:1px solid #232b3a;")
        _tb = QHBoxLayout(plate_title_bar)
        _tb.setContentsMargins(0, 0, 12, 0)
        _tb.setSpacing(8)
        _tb.addWidget(self._plate_title, 1)
        _tb.addWidget(_scope_lbl)
        _tb.addWidget(self._scope_combo)
        self._drop = QLabel("Drop a Squid acquisition folder here\n\n"
                            "then pick an operator in  Process wells")
        self._drop.setAlignment(Qt.AlignCenter)
        self._drop.setStyleSheet("color:#8b98ad;font-size:16px;border:2px dashed #232b3a;border-radius:12px;margin:24px;")
        plate_host = QWidget()
        plate_host.setStyleSheet(f"background:{_BG};")
        self._left_l = QVBoxLayout(plate_host)
        self._left_l.setContentsMargins(0, 0, 0, 0)
        self._left_l.setSpacing(0)
        self._left_l.addWidget(plate_title_bar)
        self._left_l.addWidget(self._drop, 1)    # the plate overview replaces this on ingest

        left_col = QSplitter(Qt.Vertical)
        left_col.setStyleSheet("QSplitter::handle{background:#232b3a;height:1px;}")
        left_col.setChildrenCollapsible(False)
        left_col.addWidget(self._left_tabs)
        left_col.addWidget(plate_host)
        left_col.setStretchFactor(0, 0)
        left_col.setStretchFactor(1, 1)
        left_col.setSizes([340, 610])
        left_col.setChildrenCollapsible(True)
        left_col.setHandleWidth(6)

        # the right (array viewer) pane gets a thin white outline via a 1px-margin frame
        right_frame = QFrame()
        right_frame.setStyleSheet("QFrame{background:#6e7681;}")   # shows through the 1px margin = outline
        rfl = QVBoxLayout(right_frame)
        rfl.setContentsMargins(1, 1, 1, 1)
        rfl.setSpacing(0)
        rfl.addWidget(self._right_widget, 1)
        # A small control bar UNDER the detail viewer (below its FOV slider): "focus reference plane"
        # jumps the z-slider to the current FOV's sharpest plane (Tenengrad) — a per-FOV autofocus, not
        # a plate-wide save.
        detail_bar = QWidget()
        detail_bar.setStyleSheet(f"background:{_BG};")
        dbl = QHBoxLayout(detail_bar)
        dbl.setContentsMargins(8, 5, 8, 5)
        self._focus_btn = QPushButton("Focus reference plane")
        self._focus_btn.setStyleSheet(_BTN_QSS)
        self._focus_btn.setToolTip("Jump the z-slider to the sharpest plane of the FOV in view")
        self._focus_btn.clicked.connect(self._focus_reference_plane)
        dbl.addWidget(self._focus_btn)
        dbl.addStretch(1)
        rfl.addWidget(detail_bar)

        outer = QSplitter(Qt.Horizontal)
        outer.setStyleSheet("QSplitter::handle{background:#232b3a;width:1px;}")
        outer.setChildrenCollapsible(False)
        outer.addWidget(left_col)                  # pane 1: plate + controls with the tabs
        outer.addWidget(right_frame)               # pane 2: the initial viewer
        outer.addWidget(self._explore_pane)        # pane 3: exploration — VISIBLE from open
        outer.setSizes([600, 620, _EXPLORE_W])     # three real panes on a 1600 px window
        # Draggable and COLLAPSIBLE. Julio: "Since the GUI is so large, I need to be able to
        # collapse windows and be able to drag the splitters." A fixed pane on a large monitor
        # is dead space he cannot reclaim; a collapsible one lets him give the canvas the whole
        # window when he is inspecting a mosaic.
        outer.setChildrenCollapsible(True)
        outer.setHandleWidth(6)
        for _i in range(outer.count()):
            outer.setCollapsible(_i, True)
            outer.setStretchFactor(_i, 1 if _i == 1 else 0)
        # Stretch: panes 1 and 2 share the window; pane 3 gets 0 so a window RESIZE grows the plate
        # and the viewer, never the exploration strip. Requirement 5 — pane 3 must not squash the
        # plate view — is why its width is a CONSTANT taken once at construction rather than a
        # share: the pane never widens behind the user's back, and never has to be carved out of a
        # neighbour later, because it was there from the first frame.
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 1)
        outer.setStretchFactor(2, 0)
        self._split = outer
        self.setCentralWidget(outer)
        self._sync_explore_pane()                  # page 0 (the example copy) is what an empty pane shows

        self.setAcceptDrops(True)
        if initial_path:
            self.ingest(initial_path)

    # -- top-left process console: pick an operator (params via dialogs), open the CLI stub ---------
    def _build_process_pane(self) -> QWidget:
        """The 'Process wells' console: a compact one-line status, a scrollable stack of operator cards
        (MIP, Record z-stack) plus a 'to be added' roadmap, and an 'Open CLI' button. Operators gather
        any parameters through dialogs, so this pane is self-contained — no tabs."""
        pane = QWidget()
        pane.setStyleSheet(f"background:{_BG};")
        v = QVBoxLayout(pane)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(7)

        # a single small subtitle (also the live status line — tests read this)
        self._readout = QLabel("Drop a Squid acquisition, then pick an operator to run on the plate.")
        self._readout.setStyleSheet("color:#8b98ad;font-size:12px;")
        self._readout.setWordWrap(True)
        v.addWidget(self._readout)

        def _section(text):
            lab = QLabel(text)
            lab.setStyleSheet("color:#57606a;font-size:10px;font-weight:800;letter-spacing:1.5px;padding-top:8px;")
            return lab

        stack = QWidget()
        sv = QVBoxLayout(stack)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(7)
        sv.addWidget(_section("OPERATORS"))
        self._op_cards = {}
        for op in _OPERATIONS:
            card = QPushButton(f"{op.label}\n{op.blurb}")
            card.setEnabled(False)                         # enabled once an acquisition loads
            card.setCursor(Qt.PointingHandCursor)
            card.setStyleSheet(_CARD_QSS)
            card.setMinimumHeight(54)
            card.clicked.connect(lambda _=False, k=op.key: self._activate_operator(k))
            sv.addWidget(card)
            self._op_cards[op.key] = card
        if _TO_BE_ADDED:                                    # only show the section when it has cards
            sv.addWidget(_section("TO BE ADDED"))
            for label, blurb in _TO_BE_ADDED:
                soon = QPushButton(f"{label}\n{blurb}")
                soon.setEnabled(False)
                soon.setStyleSheet(_CARD_QSS + "QPushButton:disabled{color:#57606a;border-style:dashed;}")
                soon.setMinimumHeight(46)
                sv.addWidget(soon)
        sv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(stack)
        v.addWidget(scroll, 1)

        open_btn = QPushButton("Open a computed MIP…")     # load a previously written .hcs plate
        open_btn.setStyleSheet(_BTN_QSS)
        open_btn.clicked.connect(self._open_computed)
        v.addWidget(open_btn)
        self._raw_btn = QPushButton("Return to raw view")  # only shown once an operator has run
        self._raw_btn.setStyleSheet(_BTN_QSS)
        self._raw_btn.clicked.connect(self._return_to_raw)
        self._raw_btn.hide()
        v.addWidget(self._raw_btn)
        cli_btn = QPushButton("Open CLI")                  # opens a CLI tab within this pane (ABOVE Layers)
        cli_btn.setStyleSheet(_BTN_QSS)
        cli_btn.clicked.connect(lambda: self._open_op_tab("cli", "CLI", self._build_cli_tab))
        v.addWidget(cli_btn)
        layers_btn = QPushButton("Layers")                 # toggle/reorder applied operation layers
        layers_btn.setStyleSheet(_BTN_QSS)
        layers_btn.clicked.connect(lambda: self._open_op_tab("layers", "Layers", self._build_layers_tab))
        v.addWidget(layers_btn)
        return pane

    # -- operator UIs live as tabs INSIDE pane 1 (home tab + one per opened operator); exploration
    # -- tabs live in pane 3. Both bars share every path below — *tabs* says which one. -----------
    def _build_explore_empty(self) -> QWidget:
        """Pane 3 with nothing in it: EXAMPLE USAGE, not a blank strip (IMA-260).

        The pane is visible from open, so 'empty' is a state a user will actually look at, and a
        blank column teaches nothing — the Shift-drag and the right-click that fill this pane are
        both invisible gestures with no button anywhere. So the empty state names one concrete
        path (right-click -> Control Well, the primary), then a second (Shift-drag), then says
        plainly that these are only examples. It is illustration, not instruction: the user asked
        to be shown a way in, not told what to do.

        Every string here is sized at or above the project's legibility floor — see
        _EMPTY_BODY_PX. Copy nobody can read from their chair is a blank pane with extra steps.
        """
        w = QWidget()
        w.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)

        head = QLabel(_EMPTY_EXPLORE_HEAD)
        head.setWordWrap(True)
        head.setStyleSheet(f"color:#e6edf3;font-size:{_EMPTY_HEAD_PX}px;font-weight:800;")
        v.addWidget(head)

        for text, color in ((_EMPTY_EXPLORE_LEDE, "#c3ccd9"),
                            (_EMPTY_EXPLORE_PRIMARY, "#e6edf3"),      # PRIMARY: Control Well
                            (_EMPTY_EXPLORE_SECONDARY, "#c3ccd9"),    # secondary: Shift-drag
                            (_EMPTY_EXPLORE_CODA, "#8b98ad")):
            lab = QLabel(text)
            lab.setWordWrap(True)
            lab.setStyleSheet(f"color:{color};font-size:{_EMPTY_BODY_PX}px;line-height:150%;")
            v.addWidget(lab)
        v.addStretch(1)
        return w

    def explore_empty_text(self) -> str:
        """Everything pane 3 is currently SAYING while empty — '' once it holds content.

        One reader for the whole empty state, so a check cannot pass by finding a label that is on
        the widget but not on the screen: it returns text only while the empty page is the page
        pane 3 is showing."""
        if self._explore_pane.currentWidget() is not self._explore_empty:
            return ""
        head = self._explore_empty.findChildren(QLabel)
        return "\n".join(lab.text() for lab in head)

    def _sync_explore_pane(self):
        """Show the tab bar once pane 3 holds a tab, and the EXAMPLE COPY whenever it does not.

        Pane 3 keeps its width either way (IMA-260) — it is a permanent third column, so this is a
        page swap inside it, never a collapse. Both directions matter: the copy has to come back
        when the last tab closes, or a user who explores once and tidies up is left with the blank
        strip the empty state exists to prevent."""
        page = self._explore_tabs if self._explore_tabs.count() > 0 else self._explore_empty
        if self._explore_pane.currentWidget() is not page:
            self._explore_pane.setCurrentWidget(page)

    def _open_op_tab(self, key: str, title: str, builder, tabs=None):
        """Open (or focus) a UI as a tab. Built lazily, once. *tabs* is the bar it belongs in —
        the process console by default, pane 3 for exploration tabs.
        If the UI is currently detached (see _detach_tab), focus its floating window instead —
        never rebuild: for the CLI that would mean a second live shell."""
        tabs = self._left_tabs if tabs is None else tabs
        win = self._floating.get(key)
        if win is not None:
            win.raise_()
            win.activateWindow()
            return
        w = self._op_tabs.get(key)
        if w is None:
            w = builder()
            self._op_tabs[key] = w
            tabs.addTab(w, title)
            self._sync_explore_pane()
        tabs.setCurrentWidget(w)

    def _close_op_tab(self, index: int, tabs=None):
        tabs = self._left_tabs if tabs is None else tabs
        if index == 0 and tabs is self._left_tabs:         # 'Process wells' home tab — never closable
            return
        w = tabs.widget(index)
        tabs.removeTab(index)
        self._dispose_tab_widget(w)
        self._sync_explore_pane()

    def _dispose_tab_widget(self, w):
        """The ONE teardown path for an operator UI — tab close, float close, and app exit all
        route here so they can't drift: registry pop, stale-ref clear, shell kill, delete.

        An exploration tab owns MORE than a widget (a possibly-live run and a set of plate layers),
        so its extra teardown hangs off this same path rather than off the tab-close caller — a
        float-close or an app exit must free it exactly as a tab close does (IMA-209 + IMA-205)."""
        if isinstance(w, _ExplorationTab):                 # stop its run + free its layers FIRST
            self._discard_exploration(w)
        for k, v in list(self._op_tabs.items()):
            if v is w:
                del self._op_tabs[k]
        if w is self._layers_tab:                          # drop the stale ref so refresh no-ops
            self._layers_tab = None
            self._layers_box = None
        if hasattr(w, "shutdown"):                         # a live terminal — kill its shell first
            w.shutdown()
        w.deleteLater()

    # -- drag a tab out -> free-floating window (IMA-209); Re-dock returns it ---------------------
    def _detach_tab(self, index: int, tabs=None):
        """Detach the tab at `index` of `tabs` into a _FloatWindow. ALL detach logic lives here (the
        drag in _DetachTabBar is a thin, deferred caller) so the offscreen tests drive it directly.
        Returns the new window, or None when the tab can't detach (home tab / unregistered).

        ONE implementation serves both bars (IMA-237): pane 3's tabs float out through this exact
        path, and re-dock to the bar they came from. *tabs* defaults to the process console, so
        IMA-209's callers and tests are unchanged."""
        tabs = self._left_tabs if tabs is None else tabs
        if index <= 0 and tabs is self._left_tabs:   # the process console's home tab never detaches
            return None
        if index < 0:
            return None
        w = tabs.widget(index)
        if w is not None and w is self._op_tabs.get(self.CONTROL_KEY):
            return None      # the CONTROL tab is pinned: floating it would leave pane 3 claiming
            #                  no control while the plate still wears the blue frame (IMA-248)
        key = next((k for k, v in self._op_tabs.items() if v is w), None)
        if key is None:
            return None
        title = tabs.tabText(index)
        tabs.removeTab(index)
        del self._op_tabs[key]
        # _layers_tab is deliberately NOT cleared: the widget lives on in the float and
        # _refresh_layers_tab writes into it directly, so a floating Layers keeps updating.
        # `*_` is load-bearing: on_redock is connected to QPushButton.clicked, which passes
        # `checked=False` and would land on a bare `lambda k=key:` AS k — so the Re-dock button
        # called _redock(False), found no such key in _floating, and returned silently. The button
        # had been dead since IMA-209 because every test called _redock(key) directly instead of
        # clicking it. Swallow the signal's argument and keep the key bound.
        win = _FloatWindow(title, w,
                           on_close=lambda *_, k=key: self._on_float_closed(k),
                           on_redock=lambda *_, k=key: self._redock(k))
        win._home_tabs = tabs        # re-dock returns it to the bar it was dragged out of
        self._floating[key] = win
        win.show()
        self._sync_explore_pane()    # pane 3 collapses if that was its last tab
        return win

    def _on_float_closed(self, key: str):
        """User closed the floating window: same fate as closing the tab."""
        win = self._floating.pop(key, None)
        if win is None:
            return
        w = win.take_content()
        if w is not None:
            self._dispose_tab_widget(w)
        self._sync_explore_pane()

    def _redock(self, key: str):
        """Re-dock button: return the floated widget to the tab bar — the SAME object, so a live
        CLI keeps its shell and history (close-and-reopen would kill both)."""
        win = self._floating.pop(key, None)
        if win is None:
            return
        title = win._tab_title
        # `is None`, never `or`: an EMPTY QTabWidget is falsy in PyQt, so `_home_tabs or _left_tabs`
        # sent every re-dock from a just-emptied pane 3 into the process console instead.
        tabs = getattr(win, "_home_tabs", None)
        if tabs is None:
            tabs = self._left_tabs                           # back to the bar it came from
        w = win.take_content()                             # empties the window: its close is plain
        win.close()
        win.deleteLater()
        if w is None:
            return
        self._op_tabs[key] = w
        tabs.addTab(w, title)
        self._sync_explore_pane()
        tabs.setCurrentWidget(w)

    def _discard_exploration(self, tab: "_ExplorationTab"):
        """Tear down one exploration tab's work: stop its run if it owns the live one, then drop
        every layer it produced and FREE the plate canvases behind them.

        Without this the worker keeps computing into a layer nobody can reach, and each abandoned
        layer keeps a full plate-sized RGB canvas resident (tens of MB on a 1536wp) — silent
        growth on the app's headline gesture, with no error anywhere."""
        stopped = False
        if self._run_tab_key == tab.tab_key and self._busy():
            self._stop_worker()          # _retire: disconnects signals, then lets the thread drain
            self._note_partial_output()  # a stopped SAVE run leaves a half-written .hcs on disk
            self._run_tab_key = None
            stopped = True
        if self._active_exploration is tab:
            # BUG 1: the tab in front is being deleted. Leaving _active_exploration pointing at it
            # strands the whole view — _on_tab_changed would later park status onto a dead widget,
            # and _push_index / the FOV slider stay scoped to a subset nobody can see. Drop the ref
            # NOW and ask for a re-sync; if a run is still draining, _on_run_drained does it once
            # the thread is actually gone (a stopped run keeps _busy() True for a while, which is
            # exactly why the deferred path exists).
            self._active_exploration = None
            self._request_resync()
        gone = self._op_stack.remove_suffix(f"@{tab.tab_key}")
        if self._overview is not None:
            for layer in gone:
                self._overview.drop_layer(layer)
        if self._active_op_key in gone:
            self._active_op_key = None
            self._plate_mode = "raw"
            if self._acq_name:
                self._plate_title.setText(f"{self._acq_name}   ·   raw")
        self._refresh_layers_tab()
        if stopped:
            self._readout.setText(f"stopped {exploration_tab_label(tab.regions)} — tab closed mid-run")

    def _note_partial_output(self):
        """A save run stopped mid-write leaves a partial `.hcs`. Drop an INCOMPLETE marker in it so
        a later 'Open a computed MIP…' can refuse it instead of presenting a truncated plate as a
        finished one (resolve_plate_root only looks for plate.ome.zarr, which a partial still has)."""
        out = self._run_out_dir
        self._run_out_dir = None
        if not out:
            return
        try:
            p = Path(out)
            if p.exists():
                (p / "INCOMPLETE").write_text(
                    "This plate was stopped mid-write and is NOT complete.\n"
                    "Re-run the operator to produce a full plate.\n")
        except OSError:
            pass       # best-effort: never let cleanup bookkeeping break teardown

    def _close_exploration_tabs(self):
        """Close every exploration tab. Called on ingest: a tab's regions belong to the acquisition
        it was opened from, and _fov_index is about to be rebuilt for a different plate.

        Muted: each removeTab emits currentChanged, and letting _on_tab_changed re-point the detail
        at the OUTGOING acquisition mid-teardown is pure waste (ingest rebuilds it all anyway)."""
        self._tabs_muted = True
        try:
            for i in range(self._explore_tabs.count() - 1, -1, -1):
                if isinstance(self._explore_tabs.widget(i), _ExplorationTab):
                    self._close_op_tab(i, self._explore_tabs)
            # ...and the ones dragged out into floating windows (IMA-209). A float is off the tab
            # bar but NOT off the plate: it still owns layers and can still own the live run.
            for key, win in list(self._floating.items()):
                if isinstance(win.content(), _ExplorationTab):
                    self._floating.pop(key, None)
                    w = win.take_content()
                    win.close()
                    win.deleteLater()
                    if w is not None:
                        self._dispose_tab_widget(w)
        finally:
            self._tabs_muted = False

    def open_exploration_tab(self, regions) -> Optional[str]:
        """Open (or focus) the exploration tab for ``regions``. Returns its key, or None.

        The UI entry point is IMA-221's Shift-drag marquee, via ``_on_marquee_selected``; it is also
        callable programmatically (and by tests). Identity is content-addressed, so dragging the
        same wells twice focuses the existing tab rather than opening a duplicate."""
        if self._reader is None or self._overview is None:
            self._readout.setText("open an acquisition first")
            return None
        regions = list(dict.fromkeys(regions))            # de-dupe, keep first-seen order
        if not regions:
            self._readout.setText("empty selection — nothing to explore")
            return None
        unknown = [r for r in regions if r not in self._fov_index]
        if unknown:
            self._readout.setText(f"{len(unknown)} region(s) are not in this acquisition: {unknown[:3]}")
            return None
        key = exploration_tab_key(self._acq_name, regions)
        # PANE 3 (IMA-237), not the process console: the Shift-drag that opens this tab is also what
        # REVEALS the exploration pane, which is why it is the gesture and not a menu item.
        self._open_op_tab(key, exploration_tab_label(regions),
                          lambda: self._build_exploration_tab(regions, key),
                          tabs=self._explore_tabs)
        return key

    # -- CONTROL WELL (IMA-248's unit, corrected from FOV to WELL; reachable because IMA-260's
    # -- empty-state example points at it) ---------------------------------------------------------
    CONTROL_KEY = "control"           # the pinned tab's registry key: there is only ever one

    def set_control_well(self, well_id: Optional[str]):
        """Pin *well_id* as THE control: the reference region every other well is compared against.

        A control well is standard HCS practice — you read a treated well against an untreated one
        — and under IMA-253's model a well IS a region (a mosaic of FOVs), which is why the unit
        here is the well and not a single field.

        ONE piece of state, ``self._control_well``, owned here. The plate and the exploration pane
        are told from it; neither keeps its own answer and neither is asked. That is the whole
        design constraint of IMA-248: this project has already shipped four bugs whose shape was
        two places holding the same fact and disagreeing.

        ``None`` or ``""`` clears it. Setting a different well releases the previous one — the
        release is implicit in there being one variable, not a step that can be forgotten."""
        well_id = well_id or None
        if well_id is not None and well_id not in self._fov_index:
            self._readout.setText(f"{well_id} is not a region of this acquisition")
            return
        self._control_well = well_id
        if self._overview is not None:
            self._overview.set_control(well_id)          # the frame is DERIVED, never independent
        self._sync_control_tab()
        self._readout.setText(f"control well: {well_id}" if well_id else "control well cleared")

    def control_well(self) -> Optional[str]:
        """The one control-well identity. Every view answers this question by asking here."""
        return self._control_well

    def _sync_control_tab(self):
        """Make pane 3's pinned first tab agree with ``self._control_well``.

        The control is a PINNED tab (IMA-248): always index 0, and with no close button — you clear
        a control from the plate, where you set it, not by tidying a tab away and leaving a blue
        frame behind on a well that is no longer anything."""
        old = self._op_tabs.get(self.CONTROL_KEY)
        if old is not None and getattr(old, "regions", None) != ([self._control_well]
                                                                 if self._control_well else None):
            idx = self._explore_tabs.indexOf(old)        # a DIFFERENT (or no) control: retire it
            if idx >= 0:
                self._explore_tabs.removeTab(idx)
            self._dispose_tab_widget(old)
        if not self._control_well:
            self._sync_explore_pane()
            return
        if self._op_tabs.get(self.CONTROL_KEY) is None:
            w = self._build_exploration_tab([self._control_well], self.CONTROL_KEY)
            self._op_tabs[self.CONTROL_KEY] = w
            self._explore_tabs.addTab(w, f"Control · {self._control_well}")
        idx = self._explore_tabs.indexOf(self._op_tabs[self.CONTROL_KEY])
        if idx > 0:
            self._explore_tabs.tabBar().moveTab(idx, 0)  # pinned FIRST, ahead of every subset tab
        self._explore_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)   # ...and not closable
        self._sync_explore_pane()
        self._explore_tabs.setCurrentIndex(0)

    def _current_exploration(self) -> Optional["_ExplorationTab"]:
        """The exploration tab the plate and viewer follow: pane 3's FRONT tab, or None when pane 3
        is empty (IMA-237).

        Before pane 3 existed, "which tab is in front" was a single question with a single answer,
        because exploration tabs shared the process console's bar. Now the console and pane 3 are
        side by side and both are visible at once, so scope is owned by pane 3 alone — opening the
        Layers tab in pane 1 must not silently un-scope the viewer beside it."""
        if self._explore_tabs.count() == 0:
            return None
        w = self._explore_tabs.currentWidget()
        return w if isinstance(w, _ExplorationTab) else None

    def _on_tab_changed(self, index: int = -1, force: bool = False):
        """The plate + detail follow the ACTIVE tab (IMA-205).

        An exploration tab claims to be scoped to its subset, so the plate's status dots and the
        detail's FOV slider have to agree with it — otherwise the tab says '4 wells' while the
        viewer beside it lists all 1536, and scrubbing lands on wells the tab never selected.

        A LIVE run is the one thing we won't retarget under: the worker is pushing into the slider
        this call would rebuild. So the switch is DEFERRED, not dropped (``_request_resync``) —
        dropping it is what left the front tab lying about what the viewer shows (BUG 2), because
        nothing re-emits ``currentChanged`` when the run later drains.

        ``force=True`` re-runs the sync from ``_on_run_drained`` even when there is no outgoing
        exploration tab to park — after a mid-run tab close there ISN'T one, and that is precisely
        the case that has to fall back to the whole plate (BUG 1).

        Honest limitation: ndviewer's only retarget seam is ``start_acquisition``, which RESETS the
        viewer. Computed frames pushed via register_array are in-memory and do not survive the
        switch; we re-register the subset's RAW plane paths (cheap — paths only) so the pane shows
        real imagery rather than black. Re-run the operator in the tab to recompute its frames."""
        if self._reader is None or self._overview is None or self._tabs_muted:
            return
        if self._busy():
            self._request_resync()   # never retarget the slider a live run is pushing into — LATER
            return
        w = self._current_exploration()      # pane 3 owns scope now — not the index we were handed
        prev = self._active_exploration
        if prev is not None and self._overview is not None:
            prev.status = self._overview.status_snapshot()      # park the outgoing tab's dots
        if w is not None:
            self._active_exploration = w
            self._setup_raw_detail(order=w.regions)
            self._overview.set_all_status("empty")
            self._overview.set_status_map(w.status)
            top = next((ly.key for ly in reversed(self._op_stack.layers())
                        if ly.key.endswith(f"@{w.tab_key}")), None)
            self._overview.set_active_layer(top or "raw")
            w.set_sync_pending(False)
            # NB: do NOT reset _push_index here — _setup_raw_detail just built the subset map for
            # this tab's slider, and clearing it would send register_image straight back to global
            # plate indices (the exact off-by-a-lot this whole path exists to prevent).
        else:
            if prev is None and not force:
                return                   # home -> operator tab: the plate is already plate-wide
            self._active_exploration = None
            self._setup_raw_detail(order=None)
            self._overview.set_all_status("empty")
            self._overview.set_active_layer(self._active_op_key or "raw")

    def _request_resync(self):
        """Remember that the plate/detail need to catch up with the front tab once the run drains.

        Both IMA-205 bugs are the same missing edge: a tab switch that arrives while a run is live
        is silently discarded, and no later event re-delivers it. The pending flag IS that later
        event; ``_on_run_drained`` fires it as soon as the last worker thread actually exits."""
        self._pending_resync = True
        w = self._current_exploration()
        if w is not None:
            # say so IN THE TAB rather than in _readout: the run's progress writes _readout on every
            # well, so a note there would be gone before the user could read it.
            w.set_sync_pending(True)

    def _on_run_drained(self):
        """A worker thread has exited. Deliver any tab switch that was deferred while it ran.

        Fires on QThread.finished, so it also covers a run that was STOPPED (closing a tab mid-run)
        — ``_stop_worker`` returns immediately but the thread keeps going until its current well is
        done, and ``_busy()`` stays True for all of that window."""
        if self._busy():
            return                       # another (retired) worker is still draining — wait for it
        self._run_tab_key = None
        if not self._pending_resync:
            return
        self._pending_resync = False
        self._on_tab_changed(force=True)

    def _activate_operator(self, key: str):
        """Operator card / menu clicked: open the operator's UI tab. Fully generic — driven by the
        Operation template, so a new operator needs no edit here (just a registry entry + build_tab)."""
        if self._reader is None or self._overview is None:
            self._readout.setText("open an acquisition first")
            return
        op = _OPERATIONS_BY_KEY.get(key)
        if op is not None:
            self._open_op_tab(op.key, op.label, getattr(self, op.build_tab))

    def _op_tab_shell(self, title: str, blurb: str) -> tuple:
        """A standard operator-UI tab body: title + blurb, returns (widget, vbox) to fill."""
        w = QWidget()
        w.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        v = QVBoxLayout(w)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)
        t = QLabel(title); t.setStyleSheet("font-size:16px;font-weight:800;")
        v.addWidget(t)
        b = QLabel(blurb); b.setWordWrap(True); b.setStyleSheet("color:#8b98ad;font-size:12px;")
        v.addWidget(b)
        return w, v

    def _build_mip_tab(self) -> QWidget:
        return self._build_run_tab(_OPERATIONS_BY_KEY["mip"])

    def _build_stitch_tab(self) -> QWidget:
        """maragall/stitcher's control surface, in pane 1 (IMA-decon-stitch-ui).

        This used to be `_build_run_tab` -- a destination picker and a "first N wells"
        spinner, with NO stitcher controls at all. Julio: "Right now I'm blocked in testing
        the post-processing because Stitcher doesn't have that maragall/Stitcher interface
        embedded in our top-left subpane." What a user tunes on a registration/fusion run
        (registration on/off, registration channel, feather width, blunder thresholds,
        which channels to fuse) now lives in `_op_panels.StitcherPanel` and travels to both
        the preview and the saved run through `operator_kwargs`.
        """
        from squidmip._op_panels import StitcherPanel

        return StitcherPanel(self)

    def _build_decon_tab(self) -> QWidget:
        """The RL semi-convergence loop's controls (IMA-252 + IMA-decon-stitch-ui).

        The controls are here in pane 1; the picture they produce -- the deconvolved 2-D
        image in turbo with the x-z and y-z strips concatenated -- opens as a tab in PANE 3
        via :meth:`publish_qc_result`. It was `_build_plane_op_tab` (a preview button and
        nothing else), which gave no way to choose an iteration count at all.
        """
        from squidmip._op_panels import DeconQCPanel

        return DeconQCPanel(self)

    # -- the host surface the pane-1 operator panels use -----------------------------------
    #
    # Deliberately three small methods rather than handing a panel the whole window: if a
    # panel starts needing more than this, that is a coupling worth seeing in a diff.

    def say(self, text: str) -> None:
        """Put an operator panel's sentence in the window's status line."""
        if text:
            self._run_readout(text)

    def explore_scopes(self) -> list:
        """``[(label, regions), ...]`` for every subset currently parked in pane 3.

        These become SCOPE VALUES on the pane-1 panels, not buttons over in pane 3. A UI
        audit found two operator registries launching the same operators from panes 1 and 3
        with different labels and different `save` defaults, and they had already diverged
        in production; a third caller would have made that worse rather than better.
        """
        scopes = []
        for i in range(self._explore_tabs.count()):
            w = self._explore_tabs.widget(i)
            if isinstance(w, _ExplorationTab):
                scopes.append((exploration_tab_label(w.regions), list(w.regions)))
        for win in self._floating.values():                # detached tabs count too
            w = win.content()
            if isinstance(w, _ExplorationTab):
                scopes.append((exploration_tab_label(w.regions), list(w.regions)))
        return scopes

    def publish_qc_result(self, widget: QWidget, title: str) -> None:
        """Show *widget* as a result tab in PANE 3.

        THE seam between the pane-1 controls and pane 3. It is deliberately one method wide
        and it introduces no new tab machinery: `_open_op_tab` with `tabs=self._explore_tabs`
        is exactly how exploration tabs already get there, so the pane-3 owner has nothing
        to merge. Keyed by title so re-running the same subject reuses its tab instead of
        stacking a new one per iteration.
        """
        self._open_op_tab(f"qc:{title}", title, lambda w=widget: w, tabs=self._explore_tabs)

    def _build_bgsub_tab(self) -> QWidget:
        return self._build_plane_op_tab(_OPERATIONS_BY_KEY["bgsub"])

    def _build_flatfield_tab(self) -> QWidget:
        # The one plane-op that cannot run without an argument: a flat-field with no illumination
        # profile has no sane default (an identity field would silently do nothing while the UI
        # said "corrected"), so the operator raises until one is loaded. The chooser is that load.
        return self._build_plane_op_tab(_OPERATIONS_BY_KEY["flatfield"], profile_chooser=True)

    def _build_plane_op_tab(self, op, profile_chooser: bool = False) -> QWidget:
        """Generic PLANE-OP tab (IMA-223/224/225): preview on a subset, never save.

        A plane-op maps plane -> plane and does NOT consume z (IMA-210), so its output keeps the
        z-stack at full depth -- and write_plate's _validate_image accepts Z == 1 only. So this
        builder deliberately omits the "Run on the whole plate" / destination half of
        _build_run_tab: there is nothing to write yet. The moment the OME-Zarr writer learns
        Z > 1, this method can simply forward to _build_run_tab and disappear.

        The preview path itself is unchanged and needs no worker edit: _OperatorWorker's save=False
        branch streams project_plate, and _on_well already indexes image[0, :, 0] -- for a plane-op
        that is the FIRST z-plane, corrected, which is exactly what a preview should show.
        """
        w, v = self._op_tab_shell(op.label, op.blurb)
        v.addWidget(_hline())

        state = {"profile": None}
        if profile_chooser:
            prof_lbl = QLabel("(no illumination profile loaded)")
            prof_lbl.setWordWrap(True)
            prof_lbl.setStyleSheet("color:#8b98ad;font-size:12px;")

            def load_profile():
                path, _ = QFileDialog.getOpenFileName(
                    self, "Load illumination profile", "", "Illumination profile (*.npy)")
                if not path:
                    return
                from squidmip import FlatfieldProfile
                from squidmip._flatfield import set_profile
                try:
                    profile = FlatfieldProfile.from_npy(path)
                except Exception as exc:                     # bad file -> say so, keep the tab alive
                    prof_lbl.setText(f"could not load {Path(path).name}: {exc}")
                    return
                frame = tuple(self._reader.metadata["frame_shape"]) if self._reader else None
                if frame is not None and profile.shape != frame:
                    prof_lbl.setText(f"profile is {profile.shape}, this acquisition's frames are "
                                     f"{frame} -- wrong profile for this plate")
                    return
                set_profile(profile)
                state["profile"] = path
                prof_lbl.setText(f"{Path(path).name}  {profile.shape}")
                prev.setEnabled(True)

            pick_prof = QPushButton("Load illumination profile (.npy)…")
            pick_prof.setStyleSheet(_BTN_QSS)
            pick_prof.clicked.connect(load_profile)
            v.addWidget(pick_prof)
            v.addWidget(prof_lbl)
            v.addWidget(_hline())

        prev_lbl = QLabel("Preview (subset)")
        prev_lbl.setStyleSheet("color:#57606a;font-size:10px;font-weight:800;letter-spacing:1.5px;padding-top:6px;")
        v.addWidget(prev_lbl)
        n_wells = max(1, len(self._order))
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("First"))
        spin = QSpinBox(); spin.setRange(1, n_wells); spin.setValue(min(4, n_wells))
        spin.setStyleSheet(_COMBO_QSS)
        row.addWidget(spin); row.addWidget(QLabel("wells")); row.addStretch(1)
        v.addLayout(row)

        prev = QPushButton("Preview"); prev.setStyleSheet(_BTN_QSS)
        prev.setEnabled(not profile_chooser)          # flat-field waits for its profile
        prev.clicked.connect(
            lambda: self.run_operator(op.key, out_parent=None,
                                      regions=self._order[:spin.value()], save=False))
        v.addWidget(prev)

        note = QLabel("Preview only: this operator keeps the z-stack at full depth, and the "
                      "OME-Zarr writer accepts one z per field today, so there is nothing to "
                      "save yet. The raw acquisition is never modified.")
        note.setWordWrap(True); note.setStyleSheet("color:#8b98ad;font-size:11px;")
        v.addWidget(note)
        v.addStretch(1)
        return w

    def _build_run_tab(self, op) -> QWidget:
        """Generic projector-operator tab (MIP, …): pick a destination, run over the whole plate → a
        navigable OME-Zarr plate. ONE builder for every z-reduction operator — a new one needs no new
        tab code. Per-tab state lives in a closure (no per-operator instance attrs)."""
        w, v = self._op_tab_shell(op.label, op.blurb + " Pick a destination with room — output can be large.")
        state = {"dir": None}
        dir_lbl = QLabel("(no folder chosen)"); dir_lbl.setWordWrap(True)
        dir_lbl.setStyleSheet("color:#8b98ad;font-size:12px;")
        run = QPushButton("Run on the whole plate"); run.setStyleSheet(_BTN_QSS); run.setEnabled(False)

        def pick():
            d = QFileDialog.getExistingDirectory(self, f"Save {op.label} plate to folder")
            if not d:
                return
            state["dir"] = d
            ok, est_gb, _ = self._check_disk(Path(d) / f"{self._acq_name}.hcs")
            dir_lbl.setText(f"{d}\n~{est_gb:.0f} GB needed" + ("" if ok else "  (not enough free space)"))
            run.setEnabled(True)

        pick_btn = QPushButton("Choose output folder…"); pick_btn.setStyleSheet(_BTN_QSS)
        pick_btn.clicked.connect(pick)

        v.addWidget(_hline())
        run.clicked.connect(lambda: self.run_operator(op.key, out_parent=state["dir"]))
        v.addWidget(pick_btn); v.addWidget(dir_lbl); v.addWidget(run)

        # PREVIEW on a subset — test the operator on the first N wells without committing the whole
        # plate's compute + disk. Default: don't save (compute + push to the viewer only).
        v.addWidget(_hline())
        prev_lbl = QLabel("Preview (subset)")
        prev_lbl.setStyleSheet("color:#57606a;font-size:10px;font-weight:800;letter-spacing:1.5px;padding-top:6px;")
        v.addWidget(prev_lbl)
        n_wells = max(1, len(self._order))
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("First"))
        spin = QSpinBox(); spin.setRange(1, n_wells); spin.setValue(min(4, n_wells))
        spin.setStyleSheet(_COMBO_QSS)
        row.addWidget(spin); row.addWidget(QLabel("wells")); row.addStretch(1)
        v.addLayout(row)
        save_cb = QCheckBox("Save previews to disk"); save_cb.setStyleSheet(_CHECK_QSS)
        v.addWidget(save_cb)
        prev = QPushButton("Preview"); prev.setStyleSheet(_BTN_QSS); prev.setEnabled(False)

        def do_preview():
            save = save_cb.isChecked()
            dest = None
            if save:
                dest = state["dir"] or QFileDialog.getExistingDirectory(self, f"Save {op.label} preview to folder")
                if not dest:
                    return
            # "first N wells" is just one way to build a region list, so the prefix policy lives
            # here (in the UI that owns the spinner) rather than as a second subset parameter.
            self.run_operator(op.key, out_parent=dest, regions=self._order[:spin.value()], save=save)

        prev.clicked.connect(do_preview)
        v.addWidget(prev)
        v.addStretch(1)
        # both run buttons enable once an acquisition is open (the tab is only reachable then, but be safe)
        for b in (run, prev):
            b.setEnabled(self._reader is not None)
        return w

    def _build_exploration_tab(self, regions: list, tab_key: str) -> QWidget:
        """The exploration tab body: the selection, an operator menu scoped to it, and the Minerva
        hook. Operators run in PREVIEW (save=False) by default — exploring a subset should cost
        compute, not disk. 'Save this subset' persists just these wells."""
        w = _ExplorationTab(regions, tab_key)
        w.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        v = QVBoxLayout(w)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(9)
        t = QLabel(f"Exploration · {exploration_tab_label(regions)}")
        t.setStyleSheet("font-size:16px;font-weight:800;")
        v.addWidget(t)
        b = QLabel(f"{len(regions)} region(s) selected. Operators here run on this subset only.")
        b.setWordWrap(True)
        b.setStyleSheet("color:#8b98ad;font-size:12px;")
        v.addWidget(b)

        listing = QLabel(", ".join(regions))       # the tab must LIST exactly what it is scoped to
        listing.setWordWrap(True)
        listing.setStyleSheet("color:#57606a;font-size:11px;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(listing)
        scroll.setMaximumHeight(90)
        v.addWidget(scroll)
        w.listing = listing                        # tests assert the tab lists exactly its regions

        note = QLabel("A run is still finishing — the plate and viewer beside this tab still show "
                      "it. They will switch to this subset when it is done.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#d29922;font-size:11px;")
        note.setVisible(False)
        v.addWidget(note)
        w.sync_note = note
        w.set_sync_pending(w.sync_pending)

        v.addWidget(_hline())
        lab = QLabel("RUN ON THIS SUBSET")
        lab.setStyleSheet("color:#57606a;font-size:10px;font-weight:800;letter-spacing:1.5px;padding-top:4px;")
        v.addWidget(lab)
        # IMA-226: one live-preview button per RUNNABLE operator, off the engine registry rather
        # than off _OPERATIONS. That is the same edit in both directions: `reference` gains a
        # button it never had (it has no card, so the card loop skipped it), and `minerva` loses
        # one it should never have had (it is an export hand-off, not an operator — clicking it
        # handed "minerva" to the engine and the run died with a raw KeyError in the status line).
        for k in runnable_operators():
            btn = QPushButton(f"{operator_label(k)} (preview)")
            btn.setStyleSheet(_BTN_QSS)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(
                lambda _=False, k=k: self.run_operator(k, regions=regions, save=False,
                                                       tab_key=tab_key))
            v.addWidget(btn)

        save_btn = QPushButton("Save this subset to disk…")
        save_btn.setStyleSheet(_BTN_QSS)
        save_btn.clicked.connect(
            lambda: self.run_operator(_OPERATIONS[0].key, regions=regions, save=True, tab_key=tab_key))
        v.addWidget(save_btn)

        v.addWidget(_hline())
        minerva = QPushButton("Open in Minerva")
        minerva.setStyleSheet(_BTN_QSS + "QPushButton:disabled{color:#57606a;border-style:dashed;}")
        minerva.setEnabled(False)     # IMA-228 owns the squid2minerva bridge; this is its mount point
        minerva.setToolTip("Coming with IMA-228 — exports the selection as OME-TIFF and launches "
                           "minerva-author")
        v.addWidget(minerva)
        w.minerva_btn = minerva
        v.addStretch(1)
        return w

    def _build_minerva_tab(self) -> QWidget:
        """Minerva Author hand-off (IMA-228): export the SELECTION, then open Author on it.

        Scope comes from :meth:`minerva_selection` — the plate's selected FOVs/wells, else the
        well open in the detail viewer, which means every FOV of it. One file pair per FOV
        (Minerva opens one 2D image at a time and SquidMIP has no stitcher).
        """
        op = _OPERATIONS_BY_KEY["minerva"]
        w, v = self._op_tab_shell(
            op.label,
            "Writes an OME-TIFF plus a Minerva story for every selected FOV, then starts Minerva "
            "Author. Minerva has no deep link, so pick the .story.json below in its “Select File” "
            "dialog — the colours and contrast are already applied.",
        )
        state = {"dir": None, "pairs": []}

        dir_lbl = QLabel("(defaults to a minerva_export folder in your home directory)")
        dir_lbl.setWordWrap(True)
        dir_lbl.setStyleSheet("color:#8b98ad;font-size:12px;")

        # Projection mode — the salesperson tool (squid2minerva convert.py) offers --mip/--z, so
        # hardcoding one here would be a capability regression. Driven by the projector registry.
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("Projection"))
        proj = QComboBox(); proj.setStyleSheet(_COMBO_QSS)
        proj.addItems(available_projectors())
        proj.setCurrentText("mip")
        row.addWidget(proj); row.addStretch(1)

        launch_cb = QCheckBox("Open Minerva Author after exporting")
        launch_cb.setStyleSheet(_CHECK_QSS)
        launch_cb.setChecked(True)

        path_lbl = QLabel("")
        path_lbl.setWordWrap(True)
        path_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_lbl.setStyleSheet("color:#8b98ad;font-size:11px;")
        copy_btn = QPushButton("Copy story path"); copy_btn.setStyleSheet(_BTN_QSS); copy_btn.hide()
        reveal_btn = QPushButton("Show in folder"); reveal_btn.setStyleSheet(_BTN_QSS); reveal_btn.hide()

        def pick():
            d = QFileDialog.getExistingDirectory(self, "Save the Minerva export to folder")
            if not d:
                return
            state["dir"] = d
            dir_lbl.setText(d)

        def on_exported(pairs):
            state["pairs"] = pairs
            if not pairs:
                return
            path_lbl.setText("\n".join(str(story) for _, story in pairs))
            copy_btn.show(); reveal_btn.show()

        def do_copy():
            if state["pairs"]:
                QApplication.clipboard().setText("\n".join(str(s) for _, s in state["pairs"]))
                self._readout.setText("story path copied")

        def do_reveal():
            if state["pairs"]:
                from squidmip._minerva import reveal
                reveal(state["pairs"][0][1])

        pick_btn = QPushButton("Choose output folder…"); pick_btn.setStyleSheet(_BTN_QSS)
        pick_btn.clicked.connect(pick)
        run = QPushButton("Export the selected FOVs"); run.setStyleSheet(_BTN_QSS)
        run.clicked.connect(lambda: self.run_minerva_export(
            out_dir=state["dir"], projector=proj.currentText(),
            launch=launch_cb.isChecked(), on_exported=on_exported,
        ))
        copy_btn.clicked.connect(do_copy)
        reveal_btn.clicked.connect(do_reveal)

        v.addWidget(pick_btn); v.addWidget(dir_lbl)
        v.addLayout(row); v.addWidget(launch_cb); v.addWidget(run)
        v.addWidget(_hline()); v.addWidget(path_lbl); v.addWidget(copy_btn); v.addWidget(reveal_btn)
        v.addStretch(1)
        run.setEnabled(self._reader is not None)
        return w

    def minerva_selection(self) -> list:
        """The ``[(region, fov), ...]`` the user actually selected — never a silent stand-in.

        The requirement is "open minerva-author with the selected region(s)", so this reads the
        selection instead of inventing one. Exactly two sources, in order:

        1. :meth:`selected_region_fovs` — **this window's** selection. ``PlateOverview`` is
           display-only: it maps grid cells to well ids and emits them, and ``PlateWindow`` is
           where they land (``_on_selection_changed`` -> ``_selected_regions``) because
           expanding a well to its FOVs needs ``fovs_per_region``, which only this side has.
           So we call our own method directly. The previous version probed the overview too and
           fell back to ``PlateOverview.selected_wells()``; the overview never had a
           ``selected_region_fovs`` and the fallback was what made the export appear to work at
           all — a duck-typed chain standing in for reading the selection from its owner.
        2. The region open in the detail viewer (``_current_well``): every FOV of it.

        Note the unit. The pairs are ``(region, fov)`` but the export groups them BY REGION and
        fuses each into one mosaic — a region is a mosaic containing an array of FOVs, never a
        FOV. Selecting a whole region yields all its FOVs here and one fused mosaic downstream.

        Nothing selected returns ``[]`` — the caller says so rather than exporting fov 0 of 36
        and calling it "the selected well".
        """
        fovs_per_region = (self._meta or {}).get("fovs_per_region", {}) or {}

        def expand(regions) -> list:
            out = []
            for region in regions:
                out.extend((str(region), int(f)) for f in fovs_per_region.get(str(region), []))
            return out

        sel = [(str(r), int(f)) for r, f in self.selected_region_fovs()
               if int(f) in fovs_per_region.get(str(r), [])]
        if sel:
            return sel
        if self._current_well:
            return expand([self._current_well])
        return []

    def run_minerva_export(self, out_dir=None, projector: str = "mip", launch: bool = True,
                           on_exported=None, t: int = 0, selection=None):
        """Export the user's selection for Minerva Author and (optionally) open it.

        Runs off the GUI thread: projecting a well is real I/O plus compute, and starting
        Minerva Author polls a port for up to 90 s. Tests call this directly with launch=False.
        *selection* overrides :meth:`minerva_selection` (tests and future callers).
        """
        if self._reader is None or self._meta is None:
            self._readout.setText("open an acquisition first")
            return
        if self._minerva is not None and self._minerva.isRunning():
            self._readout.setText("already exporting — let the current export finish first")
            return

        sel = list(selection) if selection is not None else self.minerva_selection()
        if not sel:
            self._readout.setText(
                "nothing selected — pick the well or FOVs to export "
                "(double-click a well on the plate), then export again")
            return

        # The export unit is a REGION (one fused mosaic each), so count regions, not FOVs.
        regions = list(dict.fromkeys(r for r, _ in sel))
        what = (f"{len(regions)} mosaic{'s' if len(regions) != 1 else ''} "
                f"({', '.join(regions)}, {len(sel)} FOVs)")
        n_t = self._meta.get("n_t", 1) or 1
        t_note = f" (t={t} of {n_t})" if n_t > 1 else ""
        self._minerva = w = _MinervaWorker(
            self._reader, sel, out_dir, projector, t=t, launch=launch)

        def on_launched(ok):
            if ok:
                self._readout.setText(
                    f"✓ Minerva Author open — pick a .story.json ({what}{t_note} exported)")
            else:
                self._readout.setText(
                    f"✓ exported {what}{t_note} — Minerva Author not found "
                    f"(set ${_MINERVA_HOME_ENV} to an explorer checkout)")

        def on_exported_readout(pairs):
            # Report what LANDED, not what was asked for: a stop mid-export writes fewer.
            if not pairs:
                self._readout.setText("nothing exported")
                return
            done = regions[: len(pairs)]
            note = "" if len(pairs) == len(regions) else f" of {len(regions)} (stopped)"
            self._readout.setText(
                f"✓ exported {len(pairs)} mosaic{'s' if len(pairs) != 1 else ''}{note} from "
                f"{', '.join(done)}{t_note} → {Path(pairs[0][0]).parent}")

        w.progress.connect(
            lambda d, n: self._readout.setText(f"● Minerva export · {d}/{n} mosaics"))
        if on_exported is not None:
            w.exported.connect(on_exported)
        w.exported.connect(on_exported_readout)
        w.launched.connect(on_launched)
        w.failed.connect(lambda m: self._readout.setText(f"Minerva export failed: {m}"))
        self._readout.setText(f"● Minerva export · {what}{t_note} …")
        w.start()

    def _build_layers_tab(self) -> QWidget:
        """The Layers tab: the OperationStack as a list of toggleable, reorderable layers. The topmost
        enabled layer is what the plate shows. Base 'raw' plus each operator you have run."""
        w = QWidget(); w.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        self._layers_box = QVBoxLayout(w)
        self._layers_box.setContentsMargins(14, 12, 14, 12); self._layers_box.setSpacing(6)
        self._layers_tab = w
        self._refresh_layers_tab()
        return w

    def _refresh_layers_tab(self):
        box = getattr(self, "_layers_box", None)
        if self._layers_tab is None or box is None:
            return
        while box.count():                       # rebuild from the current stack
            item = box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        title = QLabel("Layers (top shows on the plate)")
        title.setStyleSheet("font-size:14px;font-weight:800;")
        box.addWidget(title)
        for ly in reversed(self._op_stack.layers()):   # topmost first
            base = ly.key == "raw"
            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox(ly.label + ("  (base)" if base else ""))
            cb.setChecked(ly.enabled); cb.setStyleSheet("color:#e6edf3;")
            cb.toggled.connect(lambda on, k=ly.key: self._on_layer_toggle(k, on))
            up = QPushButton("↑"); up.setStyleSheet(_BTN_QSS); up.setFixedWidth(34)
            up.clicked.connect(lambda _=False, k=ly.key: self._on_layer_move(k, +1))
            dn = QPushButton("↓"); dn.setStyleSheet(_BTN_QSS); dn.setFixedWidth(34)
            dn.clicked.connect(lambda _=False, k=ly.key: self._on_layer_move(k, -1))
            if base:
                # IMA-227: raw is the layer every transform is recoverable TO — "each transform is
                # a LAYER, the raw is never destroyed". OperationStack refuses to disable or
                # reorder it; the controls must SAY so rather than accepting a click the model
                # then ignores, which reads as a broken checkbox.
                cb.setEnabled(False)
                cb.setToolTip("The base layer. Untick the transforms above to see it.")
                up.setEnabled(False); dn.setEnabled(False)
            h.addWidget(cb, 1); h.addWidget(up); h.addWidget(dn)
            box.addWidget(row)
        box.addStretch(1)

    def _on_layer_toggle(self, key, enabled):
        self._op_stack.toggle(key, enabled)
        self._apply_layers()

    def _on_layer_move(self, key, delta):
        self._op_stack.move(key, delta)
        self._apply_layers()
        self._refresh_layers_tab()

    def _apply_layers(self):
        """Show the topmost enabled layer on the plate; keep the title in sync.

        ``top_enabled()`` cannot be None now that raw is undisableable, but this used to no-op on
        None and leave the plate showing a layer the tab said was OFF. Fall back to raw explicitly
        instead of silently doing nothing: the plate must never render something no enabled layer
        accounts for."""
        top = self._op_stack.top_enabled()
        if top is None:
            top = next((ly for ly in self._op_stack.layers() if ly.key == "raw"), None)
        if top is not None and self._overview is not None:
            self._overview.set_active_layer(top.key)
            self._plate_mode = "raw" if top.key == "raw" else top.label
            self._plate_title.setText(f"{self._acq_name}   ·   {self._plate_mode}")
            self._update_loupe_source()

    # -- loupe sources (IMA-208) --------------------------------------------------------------
    # One source per LAYER, registered when that layer's pixels get a real home, and dropped the
    # moment they don't. This is the "run identity" the review insisted on: the layer KEY alone
    # can't be trusted, because OperationStack.add dedupes by key — save a MIP, return to raw,
    # then run an unsaved preview and the same "mip" key now shows preview tiles while
    # _processed_plate still names the older save. Re-registering on every transition is what
    # keeps the inset showing the same run the tiles came from.

    def _release_loupe_sources(self):
        """Drop every source AND join the read thread that serves them.

        The one call every "the plate is being replaced" path must make. Assigning
        ``self._loupe_sources = {}`` (which _open_computed did) only forgets the sources: the
        _LoupeWorker QThread lives on the OVERVIEW, so the old overview walked off with a running
        thread and its ~35 MB plane cache on every plate open — confirmed still isRunning() after
        the overview was replaced. Only PlateOverview.set_loupe_source(None) stops and joins it."""
        if self._overview is not None:
            self._overview.set_loupe_source(None)
        self._loupe_sources = {}

    def _set_loupe_source(self, layer_key, source):
        self._loupe_sources[layer_key] = source
        self._update_loupe_source()

    def _drop_loupe_source(self, layer_key):
        self._loupe_sources.pop(layer_key, None)
        self._update_loupe_source()

    def _update_loupe_source(self):
        """Point the plate at the source for whatever layer is on screen right now."""
        if self._overview is None:
            return
        active = getattr(self._overview, "_active", "raw")
        source = self._loupe_sources.get(active)
        if source is self._overview._loupe_src:
            return                                   # unchanged: don't churn the worker thread
        colors = None
        if self._meta and self._meta.get("channels"):
            colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in self._meta["channels"]])
        self._overview.set_loupe_source(source, colors)

    def _build_cli_tab(self) -> QWidget:
        """A LIVE, interactive shell in the pane: run the `squidmip` batch CLI (IMA-186) right here.
        Pre-seeded with the how-to (MIP every well; `--tiff` -> FIJI-openable TIFFs). `squidmip` is
        aliased to this app's interpreter so it runs regardless of PATH/conda. Falls back to a static
        command preview where a PTY isn't available (e.g. Windows)."""
        # Input must be a RAW acquisition folder; if the current path is a computed .hcs plate (or
        # none), show a placeholder rather than a wrong path.
        p = str(self._acq_path) if self._acq_path else ""
        acq = p if (p and ".hcs" not in p and not p.endswith(".ome.zarr")) else "<your acquisition folder>"
        py = sys.executable
        win = sys.platform == "win32"
        banner = [
            "==========================================================",
            "  Process a whole plate from the command line",
            "==========================================================",
            "",
            "  Same MIP as the buttons, on every well. Copy a line and press Enter.",
            "",
            "  - Flatten every well + save FIJI-openable TIFFs:",
            f'      python -m squidmip "{acq}" --tiff',
            "",
            "  - Try just the first 8 wells first (quick, little disk):",
            f'      python -m squidmip "{acq}" --limit 8 --tiff',
            "",
            "  - Choose where to save:",
            f'      python -m squidmip "{acq}" --limit 8 --tiff --output-folder ~/Downloads',
            "",
            "  - All options:   python -m squidmip --help",
            "",
        ]
        # The terminals put the venv's Scripts/bin on PATH, so the `squidmip` console script resolves
        # directly — no alias needed (doskey is unreliable in a piped cmd.exe anyway).
        setup: list = []
        cwd = str(self._acq_path.parent) if self._acq_path else str(Path.home())
        if not win:                              # Unix: a real PTY terminal
            try:
                t = _Terminal(cwd, banner, setup_cmds=setup)
                if t._fd is not None:
                    return t
            except Exception:
                pass
        try:                                     # Windows (+ Unix fallback): a QProcess shell
            t = _ProcTerminal(cwd, banner, setup)
            if t.running():
                return t
        except Exception:
            pass
        term = QPlainTextEdit(); term.setReadOnly(True)   # last resort: static, copy-paste preview
        term.setStyleSheet(_TERM_QSS)
        term.setPlainText(
            "Process a whole plate from the command line\n"
            "──────────────────────────────\n"
            "Open a terminal, then paste (no conda needed — this is the app's own Python):\n\n"
            f'    "{py}" -m squidmip "{acq}" --limit 8 --tiff --output-folder ~/Downloads\n\n'
            "This flattens the first 8 wells (MIP) and saves TIFFs you can open in FIJI.\n"
            "Drop --limit 8 to do the whole plate. Add --help to see all options.\n")
        return term

    def _enable_operators(self, flag: bool):
        for a in self._op_actions.values():
            a.setEnabled(flag)
        for c in getattr(self, "_op_cards", {}).values():
            c.setEnabled(flag)

    def _make_detail_viewer(self):
        try:
            from ndviewer_light.core import LightweightViewer
            v = LightweightViewer(None)   # empty -> push mode (we register raw z-planes on demand)
            v.setStyleSheet(_NDV_DARK)    # ndviewer defaults to light; match the plate view
            # Use ndviewer's OWN FOV slider as the scan navigator (upstreamed control — no external
            # slider). Its valueChanged drives the red box (_on_fov_slider), and the plate's double-click
            # drives it back (go_to_well_fov); both stay in sync. Hide only the "n per well" subset
            # control (an IMA-191 extra that would just clutter the z-stack detail here).
            sub = getattr(v, "_subset_container", None)
            if sub is not None:
                sub.hide()
            return v
        except Exception as e:
            self._readout.setText(f"ndviewer_light unavailable: {e}")
            return None

    # -- drag & drop --
    def _open_acquisition_dialog(self):
        """File > Open: pick a Squid acquisition folder (the reliable alternative to drag-drop)."""
        d = QFileDialog.getExistingDirectory(self, "Open a Squid acquisition folder")
        if d:
            self.ingest(d)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):   # some Windows setups require dragMove to also accept, or drop is refused
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            e.acceptProposedAction()
            self.ingest(urls[0].toLocalFile())

    # -- open an acquisition (no processing yet — that's the Process menu) --
    def ingest(self, path: str):
        from squidmip import open_reader
        from squidmip._output import plate_metadata

        p, is_plate = resolve_plate_root(path)
        if is_plate:
            self._readout.setText("this is already a written plate — drop a raw Squid acquisition")
            return
        # stop any in-flight run/preview/export and clear prior state before opening a new
        # acquisition. _stop_minerva matters as much as the other two: a Minerva worker left
        # running holds the OLD reader and would keep exporting (and launching) against an
        # acquisition the window no longer shows.
        self._stop_worker()
        self._stop_preview()
        self._stop_minerva()
        # Exploration tabs belong to the acquisition they were opened from: their region sets and
        # layer keys point at a _fov_index that is about to be rebuilt for a different plate.
        self._close_exploration_tabs()
        self._active_exploration = None
        self._control_well = None     # a control is a region OF THIS acquisition; the next plate
        #                               has no reference until the user picks one on it
        self._push_index = None
        self._run_tab_key = None
        self._reader = self._meta = None
        self._fov_index = {}
        self._selected_regions = []   # wells picked on the plate (IMA-221); scopes an operator run
        self._pushed = set()
        self._current_well = None
        self._current_fov = 0
        self._enable_operators(False)
        if self._overview is not None:
            self._release_loupe_sources()   # join the read thread BEFORE dropping its owner
            self._overview.setParent(None)
            self._overview.deleteLater()
            self._overview = None
        if self._channel_bar is not None:     # its channels belong to the plate we just dropped
            self._channel_bar.setParent(None)
            self._channel_bar.deleteLater()
            self._channel_bar = None
        self._readout.setText("scanning acquisition …")
        QApplication.processEvents()
        try:
            reader = open_reader(str(p))
            meta = reader.metadata
        except Exception as e:   # not a Squid acquisition / unreadable -> report, don't crash the app
            self._readout.setText(f"not a readable Squid acquisition: {e}")
            self._drop.show()
            return
        # Resolve the layout format ONCE: an explicit override wins, then the declared field, then
        # inference from the well ids (IMA-219 — two real acquisitions carry no format at all).
        # Never fatal: an un-inferable plate keeps the declared value and falls through the guard.
        # Resolve the sample holder ONCE (IMA-214). build_plate handles wells AND slides: a slide
        # carrier is a Plate whose cells are the freeform region ids, so a glass-slide/tissue
        # acquisition reaches this widget by the same path a 384wp does. It also reconciles a
        # declared format against the MEASURED stage pitch, so a mis-declared plate cannot lay out
        # at the wrong scale.
        try:
            plate = build_plate(meta, override=self._plate_format_override)
        except (PlateShapeError, PlateBuildError) as e:
            self._readout.setText(f"cannot lay out this acquisition: {e}")
            self._drop.show()
            return
        self._plate = plate
        self._plate_format = fmt = plate.format_name
        self._reader, self._meta = reader, meta
        self._acq_name = Path(p).name
        self._acq_path = Path(p)
        self._processed_plate = None
        rows, cols, wells, order = plate.viewer_grid()
        for idx, region in enumerate(order):
            self._fov_index[region] = {"idx": idx, "well_id": region, "rc": plate.cell_index(region)}

        self._order = order                          # well order = the detail's FOV-slider order
        # A freeform holder places its cells by GEOMETRY (IMA-253): the plate hands over one
        # rectangle per region, in grid units, and the overview draws exactly those. A well plate
        # returns None here and keeps the uniform grid it has always had.
        cl = plate.cell_layout() if hasattr(plate, "cell_layout") else None
        layout = ({plate.cell_index(cid): rect for cid, rect in cl.items()} if cl else None)
        self._overview = PlateOverview(rows, cols, wells, layout=layout)
        # Carrier art behind the cells (IMA-220). Hand over the PLATE, not its name: `plate` is what
        # build_plate RESOLVED (measured pitch beat the 2x2's mis-declared "384 well plate"), so the
        # background can only ever be drawn at the same scale the grid is laid out at.
        self._overview.set_carrier(plate)
        self._overview.set_contrast_scope(self._scope_combo.currentText())   # IMA-207
        self._selected_regions = []                  # a new acquisition starts with nothing picked
        self._overview.hovered.connect(self._on_hover)
        self._overview.wellActivated.connect(self.activate_well)
        self._overview.selectionChanged.connect(self._on_selection_changed)
        self._overview.marqueeSelected.connect(self._on_marquee_selected)
        self._overview.controlRequested.connect(self.set_control_well)
        self._plate_mode = "raw"                     # a freshly-opened plate shows raw previews
        self._plate_title.setText(f"{self._acq_name}   ·   raw")   # bottom-left plate-pane title
        self._op_stack.reset()                       # fresh layer stack (base only)
        self._active_op_key = None
        if getattr(self, "_raw_btn", None):
            self._raw_btn.hide()                     # raw view on open -> nothing to return from
        self._refresh_layers_tab()
        self._drop.hide()
        self._left_l.addWidget(self._overview, 1)   # fills the pane and self-fits — no scrollbars
        self._install_channel_bar(meta["channels"], meta["dtype"])

        self._setup_raw_detail()
        self._load_mosaic()          # pane 2 shows the region MOSAIC from the moment it opens

        self._enable_operators(True)

        # The loupe works from the moment the folder opens — the raw layer's real pixels are the
        # acquisition's own TIFFs, the same planes the preview below is about to downsample. No
        # operator run is required to look closely at a well.
        self._loupe_sources = {"raw": _RawLoupeSource(
            reader, meta, lambda w: _fov_of_well(w, meta.get("fovs_per_region")))}
        self._update_loupe_source()

        # The mosaic geometry is known the moment the acquisition opens — it is pure arithmetic on
        # coordinates.csv — so hand it to the plate NOW rather than waiting for an operator run
        # (IMA-249: it was only ever set from run_operator, which is why the plate looked like a
        # grid of lone frames until something was run). The preview below composites into exactly
        # these boxes.
        self._overview.set_mosaic_boxes(_mosaic_boxes(meta))

        # fast RAW preview: fill the plate with downsampled thumbnails immediately (grey dots),
        # in the SAME row-major order the operator will later process them in.
        self._preview = _PreviewWorker(reader, meta, self._fov_index, order)
        self._preview_order = list(order)
        self._preview.tileReady.connect(self._on_preview_tile)
        self._preview.streamEnded.connect(lambda: self._recomposite("raw"))
        self._preview.start()   # (the detail already landed on order[0] via _setup_raw_detail)
        # top-left = LIVE STATUS (what's happening / what's shown); the plate name is the pane title
        # Multi-FOV policy (IMA-187): an operator run processes EVERY FOV and composites them into
        # the well's cell by stage coordinate. The raw preview above is still one FOV per well (it
        # reads a single plane per well precisely to stay fast), so say which one you're looking at.
        multi = sum(1 for r in order if len(meta["fovs_per_region"][r]) > 1)
        note = (f" · {multi} multi-FOV region(s), previewing as mosaics" if multi else "")
        # "live" retired: this is a POST-ACQUISITION viewer. Nothing here is streaming off a
        # scope -- the acquisition is finished and on disk, and calling it live invited exactly
        # the wrong mental model of what the operators below are doing.
        self._readout.setText(f"loaded · {len(self._fov_index)} wells · double-click to open{note}")

    def _load_mosaic(self, region: Optional[str] = None, op: str = "raw"):
        """Show one region's fused MOSAIC in pane 2, one napari layer per channel.

        The unit displayed is a mosaic, never a single FOV (IMA-265). This runs on OPEN, before
        any operator: a raw acquisition has no pyramid on disk, so the region's FOVs are fused
        by stage position. Once an operator has written an OME-Zarr, ``_load_mosaic_zarr`` shows
        that pyramid lazily instead, as a SECOND processing layer, so the before/after toggle is
        just a visibility flip.
        """
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False):
            return
        if self._reader is None or self._meta is None:
            return
        region = region or (self._order[0] if getattr(self, "_order", None) else None)
        if region is None:
            return

        prior = getattr(self, "_mosaic_worker", None)
        if prior is not None and prior.isRunning():
            prior.stop()
            prior.wait(2000)

        self._mosaic_region = region
        pane.mosaic.remove_op(op)
        channels = [c["name"] for c in self._meta["channels"]]
        w = _MosaicWorker(self._reader, self._meta, region, channels, parent=self)
        w.ready.connect(lambda r, ch, plane, bbox: self._on_mosaic_plane(op, r, ch, plane, bbox))
        w.problem.connect(lambda msg: pane.say(msg))
        w.finished_count.connect(lambda n: self._on_mosaic_done(op, region, n))
        self._mosaic_worker = w
        w.start()

    def _on_mosaic_plane(self, op: str, region: str, channel: str, levels, bbox_um):
        """One channel of the mosaic arrived, as a LAZY PYRAMID. Add it as a napari layer.

        ``levels`` is always the list napari's ``multiscale=True`` contract wants — highest
        resolution first — even when the mosaic is too small to have a second rung, so there is
        one code path here and no sniffing of what arrived.
        """
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False):
            return
        if getattr(self, "_mosaic_region", None) != region:
            return                                  # a later region won the race; drop this one
        from squidmip._napari_pane import _colormap_for

        # NO contrast_limits: napari autoscales, and napari OWNS contrast.
        #
        # We used to pass _pct_window's percentile window. Two things were wrong with that.
        # First it duplicated napari's job — napari computes its own percentile autoscale and
        # exposes it on the layer, so passing ours meant two percentile rules over one quantity,
        # which is this project's most-repeated defect shape. Second it made the composite
        # unreadable: a window like 561 -> 576..4032 sends every mid-tone tissue pixel to full
        # intensity in THAT channel, and with additive blending four saturated channels sum to
        # white. Julio, repeatedly: "Channel blending still sucks."
        #
        # Julio: "Napari has so many pre-built features that you're not leveraging." This is one.
        # napari autoscales on add and the user retunes with the layer's own contrast slider,
        # which is also the single owner the plate now follows. A LAZY z-stack makes this doubly
        # right: computing our own window would have to reduce over z, materialising the stack on
        # the GUI thread; napari autoscales from the visible plane instead.
        # multiscale=True is what makes the pyramid a pyramid. Without it napari treats the list
        # as one array to stack, or takes level 0 and renders exactly as slowly as before — the
        # levels would exist and buy nothing.
        pane.mosaic.add_mosaic(
            op, channel, levels,
            colormap=_colormap_for(channel),
            multiscale=True,
            bbox_um=bbox_um,
            z_scale_um=(self._meta or {}).get("dz_um"),
        )

    def _on_mosaic_done(self, op: str, region: str, n: int):
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False):
            return
        if n == 0:
            pane.say(f"{region}: no mosaic could be built (see the message above).")
            return
        pane.say("")
        try:
            pane.mosaic.show_op(op)
            pane.mosaic.model.reset_view()
        except Exception:                            # noqa: BLE001 - view framing is cosmetic
            pass
        self._bind_napari_contrast()

    def _bind_napari_contrast(self):
        """Point the EXISTING contrast sink (IMA-261) at napari instead of ndviewer_light.

        No new contrast model, and no second sink: this reuses ``_on_detail_contrast``, which
        already lands in the plate's FOLLOW path via ``follow_channel_window`` rather than its
        manual latch. That distinction is the one that matters — treating an owner's autoscale as
        a user gesture is what latched every channel MANUAL on open, killed the plate's running
        auto-contrast from the first frame, and left SCOPE_PER_REGION painting every well under
        one global window while the amber "wells NOT comparable" badge lied over the top.

        ``MosaicLayers.on_user_contrast`` additionally filters out OUR OWN writes (the percentile
        window set at add time, and link propagation), so only a real change of the owner's
        resolved window arrives here.
        """
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False) or self._meta is None:
            return
        if getattr(self, "_napari_contrast_bound", False):
            return          # connect ONCE: the pane outlives every ingest, like the detail viewer
        index = {c["name"]: i for i, c in enumerate(self._meta["channels"])}

        def _sink(channel: str, lo: float, hi: float):
            ch = index.get(channel)
            if ch is not None:
                self._on_detail_contrast(ch, lo, hi)

        pane.mosaic.on_user_contrast(_sink)
        self._napari_contrast_bound = True

    def _setup_raw_detail(self, order: Optional[list] = None):
        """Point the detail viewer at the RAW acquisition: full z-stack, full frame, FOV slider.

        ``order=None`` is the whole plate (open / 'Return to raw view'). An exploration tab passes
        its own region subset so the slider lists exactly the wells that tab is scoped to.
        Registers each well's raw plane PATHS up front (cheap — paths only, no image I/O) so
        scrubbing shows a real (lazily read + cached) image per well instead of black."""
        if self._detail is None or self._reader is None:
            return
        meta, reader = self._meta, self._reader
        order = self._order if order is None else list(order)
        h, w = meta["frame_shape"]
        channels = [c["name"] for c in meta["channels"]]
        # pixel_size_um/dz_um are what make ndv's 3D button render this z-stack with the
        # right geometry (IMA-255). This is the ONLY call site that declares a real n_z —
        # the processed/mosaic ones below declare n_z=1, where a volume is meaningless — so
        # it is the only one that needs them. Omitting them renders the stack isotropic:
        # on the tissue set that is dz 1.5um against pixel 0.752um, i.e. 2x squashed in z.
        # Passed positionally-by-keyword and NOT guarded: a stale ndviewer_light without
        # these parameters must fail loudly here rather than silently drop back to
        # isotropic. See tests/test_viewer_3d.py.
        self._detail.start_acquisition(channels, meta["n_z"], h, w, [f"{r}:0" for r in order],
                                       pixel_size_um=meta.get("pixel_size_um"),
                                       dz_um=meta.get("dz_um"))
        self._push_shape = None       # raw mode registers PATHS, not arrays — no array canvas here
        self._push_problem = None
        # Re-scope the RAW preview to the same wells the slider now lists. Without this the
        # producer (a full-plate _PreviewWorker) and the consumer (_push_index, built from
        # `order`) describe different well lists, and every push outside the subset is discarded.
        # That is the bug that made the FOV slider stop advancing after an exploration tab was
        # opened: the slider showed only the well that had already loaded.
        if getattr(self, "_preview", None) is not None and order != getattr(self, "_preview_order", None):
            self._stop_preview()
            self._preview = _PreviewWorker(reader, meta, self._fov_index, order)
            self._preview_order = list(order)
            self._preview.tileReady.connect(self._on_preview_tile)
            self._preview.streamEnded.connect(lambda: self._recomposite("raw"))
            self._preview.start()
        self._pushed = set()
        # The raw slider is 1:1 with `order`, so pushes must map into THAT, not the plate index.
        # Identity when the slider IS the whole plate; a subset map otherwise.
        #
        # REGRESSION GUARD (found on the live GUI): this map is consumed by _on_push, which DROPS
        # any push it cannot translate. That is correct for a stale run, but it silently de-scoped
        # the main view: opening an exploration tab on one well left _push_index == {0: 0} while
        # the full-plate preview kept emitting global indices 1..N-1, so every other well was
        # discarded and the FOV slider stopped advancing past the well that was clicked. The map
        # and the producer must describe the SAME well list, so record it and re-scope the raw
        # preview to match rather than leaving the two to disagree.
        self._push_order = list(order)
        self._push_index = (None if order == self._order
                            else {self._fov_index[r]["idx"]: pos for pos, r in enumerate(order)})
        if hasattr(self._detail, "register_images_bulk"):
            entries = []
            for pos, well in enumerate(order):
                w_idx = pos            # position IN THIS SLIDER (== plate idx only for a full plate)
                fov = meta["fovs_per_region"][well][0]
                for z_i, z in enumerate(meta["z_levels"]):
                    for ch in channels:
                        try:
                            path, page = reader.plane_ref(well, fov, ch, z)   # (file, page) — OME-safe
                            entries.append((0, w_idx, z_i, ch, path, page))
                        except (KeyError, IndexError, OSError):
                            continue
            self._detail.register_images_bulk(entries)
            self._pushed.update(order)   # every well is registered; double-click just navigates
        if order:                        # land on the first well so the viewer isn't blank
            self.activate_well(order[0], 0)

    def _return_to_raw(self):
        """Stop previewing/processing and restore the raw downsampled view across the whole plate."""
        if self._reader is None or self._overview is None:
            return
        self._stop_worker()
        self._active_op_key = None
        if getattr(self, "_raw_btn", None):
            self._raw_btn.hide()                             # nothing to return from now
        self._plate_mode = "raw"
        self._plate_title.setText(f"{self._acq_name}   ·   raw")
        self._overview.set_active_layer("raw")
        # The raw preview is itself a MOSAIC now (IMA-253), so returning to it restores the
        # acquisition's own boxes rather than clearing them — clearing them broke both the paint
        # (a mosaic redrawn as if it filled its cell) and the double-click FOV hit-test.
        self._overview.set_mosaic_boxes(_mosaic_boxes(self._meta))
        self._update_loupe_source()                          # back to the acquisition's own pixels
        for rc in list(self._overview._status):
            self._overview.set_status(*rc, "empty")
        self._refresh_layers_tab()
        self._setup_raw_detail()
        # resume the raw thumbnail fill — the operator run stopped the preview partway, so re-run it to
        # finish downsampling every well's raw tile (idempotent: it just re-renders the raw layer).
        self._stop_preview()
        self._preview = _PreviewWorker(self._reader, self._meta, self._fov_index, self._order)
        self._preview_order = list(self._order)
        self._preview.tileReady.connect(self._on_preview_tile)
        self._preview.streamEnded.connect(lambda: self._recomposite("raw"))
        self._preview.start()
        self._readout.setText("raw view")

    def _open_computed(self):
        """Open a previously-written .hcs plate (OME-Zarr) and VISUALISE it — no recompute.

        Reads the plate/well/image OME metadata, lays out the plate, and streams each well from disk
        (a coarse pyramid level -> plate thumbnail, a ~512px level -> the ndviewer slider). Read-only."""
        import json
        d = QFileDialog.getExistingDirectory(self, "Open a computed .hcs plate")
        if not d:
            return
        base = Path(d)
        zroot = base / "plate.ome.zarr"
        if not (zroot / "zarr.json").exists():
            zroot = base if (base / "zarr.json").exists() and base.name.endswith(".zarr") else zroot
        if not (zroot / "zarr.json").exists():
            self._readout.setText("not an .hcs plate — pick a folder containing plate.ome.zarr")
            return
        # A run stopped mid-write leaves a real-looking plate.ome.zarr with only some wells in it.
        # Refuse it by name rather than silently presenting a truncated plate as a finished one.
        if (base / "INCOMPLETE").exists():
            self._readout.setText(
                f"{base.name} was stopped mid-write and is incomplete — re-run the operator "
                f"(delete the INCOMPLETE marker to open it anyway)")
            return
        try:
            plate = json.loads((zroot / "zarr.json").read_text())["attributes"]["ome"]["plate"]
            rows = [r["name"] for r in plate["rows"]]
            cols = [c["name"] for c in plate["columns"]]
            wells_meta = sorted(plate["wells"], key=lambda w: (w["rowIndex"], w["columnIndex"]))
            w0 = wells_meta[0]["path"]

            def _fov_path(well_path, default=None):
                """Each well declares its OWN first image; do not assume well 0's id fits all.

                Reusing well 0's fov path for every well silently renders the wrong image on a
                plate whose wells carry differing image ids. No dataset produces that today, so
                it stayed latent — but the loupe reads through this same mapping, and a loupe
                that magnifies a different well than the one under the cursor is precisely the
                failure the FOV seam exists to prevent."""
                try:
                    meta_w = json.loads((zroot / well_path / "zarr.json").read_text())
                    return meta_w["attributes"]["ome"]["well"]["images"][0]["path"]
                except Exception:
                    return default

            fov0 = _fov_path(w0)
            ome0 = json.loads((zroot / w0 / fov0 / "zarr.json").read_text())["attributes"]["ome"]
            levels = [ds["path"] for ds in ome0["multiscales"][0]["datasets"]]
            ms0 = ome0["multiscales"][0]
            # Pixel size, recovered from the level-0 coordinate transform. The writer collapses an
            # unknown pixel size to 1.0 (_output.py), so a plate reporting exactly 1.0 is
            # AMBIGUOUS — treat it as unknown and let the loupe say so rather than draw a scale
            # bar that might be fiction. See TODOS.md for the writer-side fix.
            px_um = None
            try:
                sc = ms0["datasets"][0]["coordinateTransformations"][0]["scale"]
                cand = float(sc[-1])
                px_um = cand if cand > 0 and abs(cand - 1.0) > 1e-9 else None
            except Exception:
                px_um = None
            chans = ome0.get("omero", {}).get("channels", [])
            channels = [{"name": c.get("label", f"ch{i}"), "display_color": "#" + c["color"].lstrip("#")}
                        for i, c in enumerate(chans)]
        except Exception as e:
            self._readout.setText(f"could not read plate metadata: {e}")
            return
        if not channels:
            self._readout.setText("plate has no channel metadata (omero) — cannot open")
            return

        self._stop_worker()
        self._stop_preview()
        self._release_loupe_sources()             # a new plate: no source (and no thread) survives
        self._acq_name, self._acq_path = base.name, base
        self._processed_plate = str(zroot)
        self._reader = None                       # a computed plate has no raw reader
        self._meta = {"channels": channels, "z_levels": [0], "n_z": 1, "n_t": 1,
                      "pixel_size_um": px_um,
                      "regions": [f"{rows[w['rowIndex']]}{cols[w['columnIndex']]}" for w in wells_meta]}
        # a computed plate replaces the whole session: drop exploration tabs (their regions belong
        # to the raw acquisition) and go back to identity push indexing over the full plate.
        self._close_exploration_tabs()
        self._active_exploration = None
        self._control_well = None                 # ...including the control (a raw-plate region)
        self._push_index = None
        self._run_tab_key = None
        wells_rc, self._fov_index, self._order, worker_wells = {}, {}, [], []
        well_paths, well_fovs = {}, {}
        for idx, w in enumerate(wells_meta):
            ri, ci = w["rowIndex"], w["columnIndex"]
            wid = f"{rows[ri]}{cols[ci]}"
            fov = _fov_path(w["path"], fov0)          # per-well, not well 0's for everyone
            wells_rc[(ri, ci)] = wid
            self._fov_index[wid] = {"rc": (ri, ci), "idx": idx, "well_id": wid}
            self._order.append(wid)
            well_paths[wid], well_fovs[wid] = w["path"], fov
            worker_wells.append((wid, w["path"], fov, ri, ci, idx))

        if self._overview is not None:
            self._overview.setParent(None); self._overview.deleteLater()
        self._overview = PlateOverview(rows, cols, wells_rc)
        # A written plate carries no stage coordinates and no declared format, so build_plate falls
        # through to inferring the format from the well ids — which is the right and only evidence
        # here. It can fail (a plate whose wells fit no standard format); carrier art is decoration,
        # so a failure means "no background", never "cannot open the plate".
        try:
            self._plate = build_plate(self._meta, override=self._plate_format_override)
        except (PlateShapeError, PlateBuildError):
            self._plate = None
        self._overview.set_carrier(self._plate)
        self._overview.set_contrast_scope(self._scope_combo.currentText())   # IMA-207
        self._overview.hovered.connect(self._on_hover)
        self._overview.wellActivated.connect(self.activate_well)
        self._overview.controlRequested.connect(self.set_control_well)
        self._active_op_key = "computed"
        if getattr(self, "_raw_btn", None):
            self._raw_btn.hide()                      # a computed plate has no raw to return to
        self._plate_mode = "computed MIP"
        self._plate_title.setText(f"{self._acq_name}   ·   computed MIP")
        self._op_stack.reset(); self._op_stack.add("computed", "computed MIP")
        self._overview.set_active_layer("computed")
        self._refresh_layers_tab()
        self._drop.hide()
        self._left_l.addWidget(self._overview, 1)
        self._install_channel_bar(channels, np.uint16)
        self._enable_operators(False)             # no raw data -> operators stay disabled

        if self._detail is not None:
            self._detail.start_acquisition([c["name"] for c in channels], 1, _PUSH_PX, _PUSH_PX,
                                           [f"{w}:0" for w in self._order])
        # A written plate is read back per FOV, so its pushes are frames at the push square.
        self._push_shape = (_PUSH_PX, _PUSH_PX)
        self._push_problem = None
        self._dropped_pushes = 0
        # Every well came from disk, so the loupe is available across the whole plate here.
        try:
            import tensorstore as _ts
            _a = _ts.open({"driver": "zarr3", "kvstore": {"driver": "file",
                          "path": f"{zroot}/{w0}/{fov0}/{levels[0]}"}}).result()
            _well_px = int(min(_a.shape[-2], _a.shape[-1]))
        except Exception:
            _well_px = _PUSH_PX
        self._set_loupe_source("computed", _ZarrLoupeSource(
            str(zroot), path_of=well_paths.get, fov_of=well_fovs.get,
            levels=levels, well_px=_well_px, pixel_size_um=px_um, written=None))
        coarse_lvl = levels[-1]                                   # coarsest -> tiny thumbnail
        push_lvl = levels[min(3, len(levels) - 1)]                # ~512px level for the detail slider
        self._worker = _ComputedPlateWorker(str(zroot), worker_wells, coarse_lvl, push_lvl, np.uint16)
        self._worker.tileReady.connect(self._on_tile)
        self._worker.pushReady.connect(self._on_push)
        self._worker.streamEnded.connect(lambda: self._recomposite("computed"))
        self._worker.progress.connect(
            lambda i, n: self._readout.setText(f"loading computed plate — {i}/{n} wells"))
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(
            lambda: self._readout.setText(f"✓ computed MIP · {len(self._order)} wells (read-only)"))
        self._readout.setText(f"loading computed plate · {len(self._order)} wells …")
        self._worker.start()

    # -- run a post-processing operator over the whole plate (persists a navigable OME-Zarr plate) --
    def _busy(self) -> bool:
        """True while ANY operator run is still alive — including one that was STOPPED but is
        still draining.

        ``_stop_worker`` clears ``self._worker`` immediately, while ``_retire`` keeps the thread
        running (and referenced in ``self._retired``) until it finishes its in-flight well —
        destroying a running QThread aborts the app, so it cannot do otherwise. Checking
        ``self._worker`` alone therefore lets a second run start against the same reader the
        moment a tab is closed, which is exactly the routine path IMA-205 introduces."""
        if self._worker is not None and self._worker.isRunning():
            return True
        return any(w.isRunning() for w in self._retired)

    def run_operator(self, key: str, out_parent: Optional[str] = None,
                     regions: Optional[list] = None, save: bool = True,
                     tab_key: Optional[str] = None, operator_kwargs: Optional[dict] = None):
        """Run a projector operator (MIP / reference) over the plate, or over a subset of it.

        ``regions=None`` runs the whole plate. A list runs exactly those regions, in that order —
        this is the ONE way to express a subset (the old ``preview_limit=N`` was a prefix-only
        special case of it; the preview spinner now passes ``self._order[:n]`` itself).

        save=False is PREVIEW: compute + stream results into the plate + ndviewer slider, writing
        NOTHING to disk (no folder, no disk-space cost). save=True persists a navigable OME-Zarr;
        combined with a subset it saves just those regions. Tests pass out_parent to skip the dialog.

        ``tab_key`` scopes the run to an exploration tab: results are filed under the layer
        ``<op>@<tab_key>`` so two tabs running the same operator do not overwrite each other.
        """
        if self._reader is None or self._overview is None:
            return
        if self._busy():
            self._readout.setText("already processing — let the current run finish first")
            return
        # IMA-226: gate on the ENGINE registry, not on the card table. `_OPERATIONS_BY_KEY[key]`
        # raised a bare KeyError for `reference` (a registered projector with no card) and let
        # `minerva` (a card that is not an operator) through to die inside the engine instead.
        # Refuse BY NAME here, in the readout, the same way an unknown region is refused below.
        if key not in runnable_operators():
            self._readout.setText(
                f"'{key}' is not a runnable operator — this viewer can run: "
                f"{', '.join(runnable_operators())}")
            return
        label = operator_label(key)
        # Scope the run: an explicit `regions` list wins (an exploration tab and the preview spinner
        # both build one), else a plate SELECTION scopes it (IMA-221), else the whole plate.
        # regions=None keeps the existing whole-plate path byte-for-byte.
        from_selection = regions is None and bool(self._selected_regions)
        if from_selection:
            regions = list(self._selected_regions)
        if regions is not None:
            regions = list(regions)
            if not regions:
                self._readout.setText("empty selection — nothing to run")
                return
            unknown = [r for r in regions if r not in self._fov_index]
            if unknown:      # fail NAMED, not with a bare KeyError out of the status loop below
                self._readout.setText(
                    f"{len(unknown)} region(s) are not in this acquisition: {unknown[:3]}")
                return
        if regions is None:
            scope = "the whole plate"
        elif from_selection:
            scope = f"{len(regions)} selected well(s)"
        else:
            scope = f"{len(regions)} well" + ("s" if len(regions) != 1 else "")
        out_dir = est_gb = None
        if save:
            # Ask WHERE to persist: output can be hundreds of GB, so let the user aim it at a roomy
            # disk rather than silently filling the acquisition's. Tests pass out_parent.
            if out_parent is None:
                out_parent = QFileDialog.getExistingDirectory(self, f"Save {label} plate to folder")
                if not out_parent:
                    return
            out_dir = Path(out_parent) / f"{self._acq_name}.hcs"
            # Estimate the bytes THIS RUN writes: a subset writes len(regions)/n_wells of a plate.
            # Previously the guard was computed plate-wide and then skipped entirely for subsets
            # (`if not ok and regions is None`), so a 500-well subset save got no check at all.
            ok, est_gb, msg = self._check_disk(out_dir, regions=regions)
            if not ok:
                self._readout.setText(msg)
                return
        self._stop_preview()                                 # the operator supersedes the raw preview
        if regions is not None:                              # amber only the wells we'll actually run
            for r in regions:
                self._overview.set_status(*self._fov_index[r]["rc"], "processing")
        else:
            self._overview.set_all_status("processing")      # amber across the plate
        self._plate_mode = label                             # plate now shows this operator's result
        self._plate_title.setText(f"{self._acq_name}   ·   {label}"
                                 + (f"   ·   {exploration_tab_label(regions)}" if tab_key else ""))
        layer_key = operator_layer_key(key, tab_key)
        self._active_op_key = layer_key                      # tiles stream into this layer
        if getattr(self, "_raw_btn", None):
            self._raw_btn.show()                             # now there's a processed view to return from
        stack_label = label if not tab_key else f"{label} · {exploration_tab_label(regions)}"
        self._op_stack.add(layer_key, stack_label)           # push the operator layer onto the stack
        self._overview.set_active_layer(layer_key)           # show it
        # Loupe source for this run. A SAVED run gets a zarr source whose written-well set grows
        # as wells land (so the loupe works mid-run on what's finished); a PREVIEW writes nothing,
        # so the layer gets no source and the gesture reports that rather than magnifying the
        # previous run's pixels through the same reused layer key.
        if save and out_dir is not None:
            ny, nx = self._meta["frame_shape"]
            fovs = self._meta.get("fovs_per_region")
            self._set_loupe_source(layer_key, _ZarrLoupeSource(
                str(Path(out_dir) / "plate.ome.zarr"),
                path_of=lambda w: "/".join(str(x) for x in parse_well_id(w)),
                fov_of=lambda w: _fov_of_well(w, fovs),
                levels=None,                                 # discovered from the first written field
                well_px=min(ny, nx), pixel_size_um=self._meta.get("pixel_size_um"),
                written=set()))
        else:
            self._drop_loupe_source(layer_key)
        self._refresh_layers_tab()
        # switch the detail to processed mode: z collapsed (nz=1 -> ndv drops the z-slider), frames at
        # the push size. The slider lists THIS RUN's regions — for a subset that is the subset, not the
        # whole plate (it used to always be self._order, so a subset preview built a 1536-entry slider
        # of which 4 were ever filled).
        run_order = self._order if regions is None else regions
        # _OperatorWorker emits the GLOBAL plate index (fov_index[region]["idx"]) with every push.
        # The slider we just built is indexed 0..len(run_order)-1, so translate on the way in —
        # without this every push on a subset run lands at the wrong slot or out of range.
        self._push_index = (None if regions is None
                            else {self._fov_index[r]["idx"]: pos for pos, r in enumerate(run_order)})
        # n_fovs=None = EVERY FOV in each well (IMA-187). Anything else (the historical 1) makes
        # `_boxes` empty in the worker and the plate falls back to one thumbnail per well, which is
        # the whole feature not rendering. The overview then adopts the worker's boxes so a
        # double-click on a mosaic cell resolves the FOV under the cursor instead of always 0.
        # Built BEFORE start_acquisition (IMA-245): the worker owns this run's push geometry and the
        # viewer's canvas is declared from it, so there is one rectangle, not two that agree by luck.
        self._worker = _OperatorWorker(key, self._reader, self._meta, self._fov_index,
                                       str(out_dir) if out_dir else "", regions=regions, save=save,
                                       n_fovs=None, operator_kwargs=operator_kwargs)
        self._overview.set_mosaic_boxes(self._worker.mosaic_boxes)
        # IMA-245: size the array viewer to what this run actually pushes. A REGION operator
        # (stitch, coordinate) pushes one FUSED MOSAIC per region, so the canvas is the mosaic
        # extent — declaring the frame square here handed the viewer a rectangle the mosaic does
        # not have, and the reported symptom was a black central viewer with no error anywhere.
        self._push_shape = self._worker.push_shape
        self._push_problem = None                            # sticky readout warning (see _on_push)
        self._dropped_pushes = 0                             # per RUN: this run's unrouted pushes
        if self._worker.push_shape_estimated:
            self._note_push_problem(
                "no stage positions / pixel size — the array viewer is sized as a frame, so the "
                "fused mosaic is shown squashed to that shape")
        if self._detail is not None:
            ph, pw = self._push_shape
            self._detail.start_acquisition([c["name"] for c in self._meta["channels"]], 1,
                                           ph, pw, [f"{r}:0" for r in run_order])
        self._run_out_dir = str(out_dir) if (save and out_dir) else None   # for partial-output cleanup
        self._run_tab_key = tab_key
        # A re-run must not composite on top of the LAST run's pixels: with a mosaic, a run that
        # lands fewer FOVs would otherwise leave the previous run's fields standing in the same
        # cell, blended into the new ones. Drop this layer's store before the first tile arrives.
        # ...keyed by the LAYER, not the bare operator key: an exploration tab files its results
        # under "<op>@<tab_key>" (IMA-205), and resetting "mip" from a tab run would wipe the
        # plate-wide layer instead of the tab's own.
        self._overview.reset_layer(layer_key)
        dest = f" → {out_dir.name}" if save else " (preview — not saved)"
        self._worker.tileReady.connect(self._on_tile)
        self._worker.pushReady.connect(self._on_push)
        self._worker.progress.connect(
            lambda d, t: self._run_readout(f"● {label} · {d}/{t} wells{dest}"))
        self._worker.streamEnded.connect(lambda k=layer_key: self._recomposite(k))
        self._worker.writtenReady.connect(self._on_written)
        self._worker.wellFailed.connect(                     # a skipped well -> red x, run continues
            lambda ri, ci: self._overview.set_status(ri, ci, "failed") if self._overview else None)
        self._worker.failed.connect(self._on_failed)
        # IMA-226: report what the plate ACTUALLY got. A run where every well raised (flat-field
        # with no profile is the routine case) still reaches finished_ok — the per-well on_error
        # path is what keeps one bad file from aborting a plate — and used to print "✓" over an
        # empty plate. Landed==0 is a failure however politely the engine returned.
        def _done_msg(w=self._worker):
            if w.landed == 0:
                self._run_readout(
                    f"⚠ {label} · {scope} produced nothing — all {w.skipped or self._worker._total} "
                    f"well(s) were skipped (see the red markers)")
            elif w.skipped:
                self._run_readout(
                    f"✓ {label} · {scope}{dest} — {w.skipped} well(s) skipped"
                    + ("  (re-openable OME-Zarr)" if save else ""))
            else:
                self._run_readout(
                    f"✓ {label} · {scope}{dest}" + ("  (re-openable OME-Zarr)" if save else ""))

        self._worker.finished_ok.connect(_done_msg)
        # a run that FINISHED wrote a complete plate — forget the path so a later stop can never
        # retroactively flag it incomplete
        self._worker.finished_ok.connect(lambda: setattr(self, "_run_out_dir", None))
        # QThread.finished (not finished_ok): it fires for a FAILED or STOPPED run too, and a tab
        # switch deferred during any of those still has to be delivered. _retire only disconnects
        # the worker's own signals, so this survives a stop.
        self._worker.finished.connect(self._on_run_drained)
        self._run_readout(f"● {label} · {scope}{dest} …")
        self._worker.start()

    def _check_disk(self, out_dir, regions: Optional[list] = None) -> tuple[bool, float, str]:
        """Estimate the persisted plate size and refuse if it won't fit (with headroom). Returns
        (ok, estimate_GB, message). Estimate = per-well projection (T·C·Y·X·itemsize) × 1.34 (the exact
        4/3 geometric sum of the 2× pyramid tail), UNCOMPRESSED. The projection collapses Z only, so
        every timepoint is preserved — a time-lapse plate writes n_t as many bytes, so n_t MUST be in
        the estimate (omitting it under-counts n_t× and lets a multi-hour time-lapse run fill the disk
        mid-write — the exact failure this guards). We do NOT discount for zstd: real fluorescence
        compresses unpredictably (often <1.2×), so assuming compression would under-estimate. An
        over-estimate only ever asks for a roomier disk, which is the safe way to be wrong."""
        import shutil
        m = self._meta
        ny, nx = m["frame_shape"]
        # Count FIELDS, not wells: the run projects every FOV (n_fovs=None), so a 36-FOV plate
        # writes 36x what a per-well count predicts. Under-counting here is how a multi-hour run
        # fills the disk mid-write with the guard reporting "plenty of room".
        # regions=None means the whole plate; a subset writes only ITS wells' fields, so count over
        # exactly those — the guard must still RUN for subsets (500 selected wells is not a rounding
        # error, and it used to be skipped entirely).
        scoped = list(self._fov_index) if regions is None else [r for r in regions if r in self._fov_index]
        n_fields = sum(len(m["fovs_per_region"][r]) for r in scoped)
        est = int(n_fields * m.get("n_t", 1) * len(m["channels"]) * ny * nx
                  * np.dtype(m["dtype"]).itemsize * 1.34)
        gb = 1024 ** 3
        try:
            free = shutil.disk_usage(Path(out_dir).parent).free
        except OSError:
            return True, est / gb, ""      # can't stat the disk — don't block
        if est > free * 0.9:
            what = "MIP" if regions is None else f"this {len(scoped)}-well run"
            return False, est / gb, (f"{what} would persist ~{est/gb:.0f} GB to {Path(out_dir).parent} "
                                     f"but only {free/gb:.0f} GB free — free space or pick another disk.")
        return True, est / gb, ""

    def _on_written(self, plate_path: str):
        """The operator finished persisting: remember the written plate (re-openable artifact)."""
        self._processed_plate = plate_path

    def _on_preview_tile(self, ri, ci, well_id, tile, box=None):
        """One preview FIELD landed. ``box`` slots it into the region's mosaic (IMA-253); ``None``
        is the single-tile path, where the field fills the cell."""
        if self._overview is not None:                       # raw preview fills the base ("raw") layer
            self._overview.add_tile(ri, ci, well_id, tile, layer="raw", box=box)

    def _on_tile(self, ri, ci, well_id, tile, box=None):
        """A field landed. ``box`` is None for the single-tile producers (_ComputedPlateWorker emits
        a 4-arg signal, which PyQt matches against this default) and a sub-cell box for a mosaic."""
        if self._overview is None:
            return
        layer = self._active_op_key or "raw"
        self._overview.add_tile(ri, ci, well_id, tile, layer=layer, box=box)
        self._overview.set_status(ri, ci, "done")           # blue
        src = self._loupe_sources.get(layer)                 # this well is now on disk -> loupe-able
        if isinstance(src, _ZarrLoupeSource):
            src.mark_written(well_id)

    def _on_push(self, fov_idx, planes):
        """A computed result's bounded planes -> the array viewer (in-memory register_array, LRU
        bounded). z collapsed (nz=1). One push per FOV for a per-FOV operator, one per REGION —
        the fused mosaic — for a region operator (IMA-245).

        ``fov_idx`` is the GLOBAL plate index. The slider is built from the CURRENT RUN's regions,
        so for a subset run it is only len(regions) long and the global index has to be translated
        (``_push_index``). Dropping an untranslatable push is deliberate: a push whose position we
        cannot resolve belongs to a run whose slider is gone, and guessing would paint one well's
        image onto another well's slot.

        NOTHING here is dropped silently (IMA-245). Every way a push can fail to land — no viewer,
        a viewer with no ``register_array``, an index this run's slider has no slot for, a plane
        whose shape is not the canvas we declared, or a rejection from the viewer itself — counts
        into ``_dropped_pushes`` AND says so in the readout. A black viewer with no error is what
        made the reported defect take a human to find; the swallowed ``except Exception: pass``
        below it was the last place that could have spoken and did not."""
        if self._detail is None:
            self._drop_push("there is no array viewer in this window to show the result in")
            return
        if not hasattr(self._detail, "register_array"):
            # The routine cause: an ndviewer_light build without the register_array push API. Every
            # computed result is then unshowable, which looks exactly like a viewer that is black.
            self._drop_push("this ndviewer_light build has no register_array — computed results "
                            "cannot reach the array viewer (upgrade ndviewer_light)")
            return
        pos = fov_idx if self._push_index is None else self._push_index.get(fov_idx)
        if pos is None:
            self._drop_push(f"a result for plate index {fov_idx} has no slot in this run's "
                            f"viewer — it belongs to a run whose slider is gone")
            return
        want = getattr(self, "_push_shape", None)
        channels = [c["name"] for c in self._meta["channels"]]
        for c_i, plane in enumerate(planes):
            got = tuple(np.asarray(plane).shape)
            if want is not None and got != tuple(want):
                # The producer and the declared canvas disagree — the defect class this whole file
                # keeps meeting. Say which two numbers disagree; do not push a plane the viewer
                # will reject without telling anyone.
                self._drop_push(f"the result is {got[0]}x{got[1]} but the array viewer was "
                                f"declared {want[0]}x{want[1]}")
                return
            try:
                self._detail.register_array(0, pos, 0, channels[c_i], plane)
            except Exception as e:      # one bad push must not break the run — but it must be said
                self._drop_push(f"the array viewer rejected the result: {type(e).__name__}: {e}")
                return

    def _drop_push(self, why: str):
        """Count an unrouted push and put the reason in the readout (IMA-245)."""
        self._dropped_pushes = getattr(self, "_dropped_pushes", 0) + 1
        self._note_push_problem(why)

    def _note_push_problem(self, why: str):
        """Make ``why`` a STICKY suffix on this run's readout, so a later progress/success line
        cannot overwrite it. A run that finished computing but could not display its result is not
        a success, and the '✓' must not be the last word on it."""
        if getattr(self, "_push_problem", None) == why:
            return
        self._push_problem = why
        self._run_readout(getattr(self, "_readout_base", self._readout.text()))

    def _run_readout(self, text: str):
        """Set the run's status line, re-appending any push problem this run has hit."""
        self._readout_base = text
        why = getattr(self, "_push_problem", None)
        self._readout.setText(text + (f"   ·   ⚠ {why}" if why else ""))

    def _on_failed(self, msg):
        if self._overview is not None:
            for rc, state in list(self._overview._status.items()):
                if state == "processing":
                    self._overview.set_status(*rc, "failed")  # red x on wells that didn't finish
        self._readout.setText(f"failed: {msg}")

    def _recomposite(self, layer: str):
        """End of a producer's stream: rebuild that layer once at full resolution, now that the
        running global window has seen every well (early wells were windowed by a young histogram)."""
        if self._overview is not None:
            self._overview.recomposite(layer)

    def _install_channel_bar(self, channels, dtype):
        """Declare the plate's channel axis and (re)build the per-channel toggle/contrast strip.

        Colors are the RESOLVED ``display_color`` — resolve_channels already applied the precedence
        (the acquisition's YAML first, the wavelength fallback map second), so the plate is tinted
        exactly like every other compositing site.
        """
        if self._overview is None:
            return
        colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in channels])
        self._overview.set_channels([c.get("display_name") or c["name"] for c in channels],
                                    colors, dtype)
        if self._channel_bar is not None:
            self._channel_bar.setParent(None)
            self._channel_bar.deleteLater()
        self._channel_bar = _ChannelBar(self._overview._labels, colors, self._overview)
        self._left_l.addWidget(self._channel_bar)   # sits UNDER the plate, in the same pane
        # A fresh plate must ALREADY agree with the array viewer, not merely agree from the next
        # gesture on: the viewer keeps whatever window it had, and a plate that waited for the
        # user to touch the slider would open showing a different window from the one on screen.
        self._adopt_detail_contrast()

    def _adopt_detail_contrast(self):
        """Pull the array viewer's CURRENT per-channel windows onto the plate (IMA-261)."""
        get = getattr(self._detail, "channel_windows", None) if self._detail is not None else None
        if get is None:
            return
        for ch, (lo, hi) in get().items():
            self._on_detail_contrast(ch, lo, hi)

    # -- navigation links --
    # -- selection (IMA-221): the widget picks wells, THIS window knows what a well contains ----
    def _on_selection_changed(self, wells: list):
        """PlateOverview is display-only — it maps grid cells to well ids and nothing more. The
        metadata lives here, so the expansion to (region, fov) happens here too.

            PlateOverview            PlateWindow
            [cells] --wells--> [_order sort] --fovs_per_region--> [(region, fov), ...]

        Today every well yields one FOV (the viewer is 1-FOV: write_plate(n_fovs=1)), so the pairs
        read [(B3, 0)]. The PAIR shape is the point: when per-FOV selection becomes possible (it
        needs FOV geometry that metadata doesn't carry yet) consumers don't change.
        """
        picked = set(wells)
        self._selected_regions = [w for w in self._order if w in picked]   # plate row-major

    def _on_marquee_selected(self, wells: list):
        """Shift-DRAG released on the plate -> open the exploration tab for that subset (IMA-205).

        This is the user's sentence end to end: "hold shift to open an 'exploration' tab with the
        selected FOV subset". The marquee (IMA-221) is the only gesture wired here — Shift+CLICK
        toggles single wells while you refine a selection, and opening a tab per corrective click
        would bury the one you actually wanted (each distinct set is a distinct content-addressed
        tab, so they would NOT dedupe).

        An empty drag (over blank plate) is a miss, not a request: return quietly rather than
        writing 'empty selection' over whatever the readout is saying."""
        if not wells:
            return
        self.open_exploration_tab(wells)

    def selected_region_fovs(self) -> list:
        """The current selection as (region, fov) pairs — the payload IMA-205 will consume."""
        per = (self._meta or {}).get("fovs_per_region", {})
        return [(r, f) for r in self._selected_regions for f in (per.get(r) or [0])]

    def _on_scope_changed(self, scope: str):
        """Contrast scope picked (IMA-207). A DISPLAY control: re-composite, never re-run.

        Deliberately does not touch the operator, the reader or any worker. The plate re-windows
        from the native-dtype tiles PlateOverview already retains, so the flip is instant even on a
        1536wp — where re-running would be minutes and would make the control unusable.
        """
        if self._overview is None:
            return
        self._overview.set_contrast_scope(scope)
        if scope == SCOPE_GLOBAL:
            self._readout.setText("contrast: global — wells are comparable")
        else:
            self._readout.setText(f"contrast: {scope} — each well fills its own range, "
                                  "wells are NOT comparable")

    def _on_hover(self, text: str):
        # BOTTOM-LEFT plate title bar: "<acq>  ·  <mode>" (mode = raw / the operator that processed it),
        # plus the hovered well when the cursor is over the plate.
        base = f"{self._acq_name or 'well plate'}   ·   {self._plate_mode}"
        self._plate_title.setText(f"{base}   ·   {text}" if text else base)

    def _slider_pos(self, well_id: str) -> Optional[int]:
        """Where ``well_id`` sits in the detail's CURRENT FOV slider, or None if it isn't in it.

        The slider is whole-plate by default (position == plate index) but an exploration tab
        scopes it to a subset, where the two diverge. Everything that hands ndviewer an index —
        register_image, register_array, go-to — has to translate through here."""
        info = self._fov_index.get(well_id)
        if info is None:
            return None
        if self._push_index is None:
            return info["idx"]
        return self._push_index.get(info["idx"])

    def activate_well(self, well_id: str, fov_index: int):
        """Double-click -> show the well in the ndviewer. In RAW mode (no operator run yet) push the
        well's raw z-stack lazily (the true z-stack, zero bytes copied). In PROCESSED mode (an operator
        has run, the slider already holds the results) just navigate the slider to that well."""
        if well_id not in self._fov_index:
            return
        # Re-point pane 2's mosaic FIRST, and independently of the detail viewer: with napari as
        # the viewer ndviewer_light may not be installed at all, and an early return on
        # `_detail is None` would silently make double-click do nothing.
        if well_id != getattr(self, "_mosaic_region", None):
            self._load_mosaic(region=well_id)
        if self._detail is None:
            return
        # Resolve the slider position BEFORE moving the red box. The box says "this is the well you
        # are looking at"; if the detail's slider does not contain the well (an exploration tab
        # scopes it to a subset) we cannot show it, and moving the box anyway is how you get a red
        # box on one well and another well's pixels beside it — silently.
        idx = self._slider_pos(well_id)
        if idx is None:
            self._readout.setText(
                f"{well_id} is not in this tab's subset — switch to 'Process wells' to open it")
            return
        self._current_well = well_id
        self._current_fov = fov_index                  # the FOV ON SCREEN (IMA-250 (b))
        if self._overview is not None:                 # current well at view = the red BOX
            self._overview.select(*self._fov_index[well_id]["rc"])
        if self._active_op_key is not None or self._reader is None:   # processed/computed: already pushed
            self._detail.go_to_well_fov(well_id, fov_index)
            return
        if well_id not in self._pushed:
            fov = self._meta["fovs_per_region"][well_id][0]
            for z_i, z in enumerate(self._meta["z_levels"]):
                for ch in (c["name"] for c in self._meta["channels"]):
                    try:
                        path, page = self._reader.plane_ref(well_id, fov, ch, z)
                        self._detail.register_image(0, idx, z_i, ch, path, page)
                    except (KeyError, IndexError, OSError, RuntimeError):
                        continue   # a genuinely-missing plane / closed viewer shouldn't block the rest
            self._pushed.add(well_id)
        # Region ids are not necessarily well ids: a slide carrier's are freeform ("R2C3",
        # "region_A", "tissue-1"). This used to rebuild the id as f"{row}{col}" from
        # parse_well_id, which RAISES on all of those, inside a bare except that swallowed it -
        # so a double-click moved the red box and never navigated, silently. The id is only a
        # label for the detail viewer's well/FOV combo, so it is passed through untouched.
        self._detail.go_to_well_fov(well_id, fov_index)

    def _on_fov_slider(self, flat_idx: int):
        """ndviewer FOV slider moved -> move the red box on the plate to that well."""
        if self._detail is None or self._overview is None:
            return
        labels = getattr(self._detail, "_fov_labels", None)
        if not labels or not (0 <= flat_idx < len(labels)):
            return
        info = self._fov_index.get(labels[flat_idx].split(":")[0])
        if info:
            self._overview.select(*info["rc"])

    def _on_detail_contrast(self, ch: int, lo: float, hi: float):
        """The CENTRAL ARRAY VIEWER re-windowed channel *ch*. Make the plate show that window.

        This is the whole of the cross-repo sync (IMA-261), and it is deliberately one-way:
        ndviewer_light owns contrast, the plate follows. `(lo, hi)` are the numbers ndv handed its
        own canvas — not a re-derivation from a slider position, not a percentile recomputed here
        — so "the plate and the viewer show the same window" is true by construction rather than
        by two rules being kept in step.

        It lands in the plate's FOLLOW path, NOT in its manual latch. ndv autoscales by itself —
        at open, and again whenever the displayed data changes — so treating each broadcast as a
        user gesture latched every channel MANUAL before anyone had touched anything: the plate's
        running auto-contrast was dead from the first frame, and SCOPE_PER_REGION painted every
        well under ndv's one global window while the plate still drew the amber "wells NOT
        comparable" badge over the top. A sink records what the owner resolved; only the user sets
        policy. `_RunningContrast.resolve` is still the single precedence rule.
        """
        if self._overview is None:
            return
        n_ch = len(self._overview._labels)
        if not (0 <= ch < n_ch):
            return          # ndv drew a channel the plate does not have (RGB mode, or a re-ingest)
        self._overview.follow_channel_window(ch, float(lo), float(hi))
        if self._channel_bar is not None:
            self._channel_bar.set_window(ch, float(lo), float(hi))

    def _focus_reference_plane(self):
        """Jump the detail viewer's z-slider to the CURRENT FOV's sharpest plane (Tenengrad autofocus).

        Per-FOV, on demand — nothing is saved. Ranks focus on downsampled planes so it stays snappy.
        This is the reference-plane feature in the viewer (not a plate-wide save operator)."""
        if self._reader is None or self._current_well is None or self._detail is None:
            self._readout.setText("double-click a well first, then focus its reference plane")
            return
        if not hasattr(self._detail, "set_current_index"):
            return
        from squidmip.projection import _tenengrad
        well = self._current_well
        # The FOV IN VIEW, not the region's first one (IMA-250). This is a per-FOV autofocus, so
        # ranking field 0 while the viewer shows field 12 reports the sharpest plane of pixels the
        # user is not looking at. Falls back to the region's first FOV when the one on screen is
        # not one of its own (a freshly-scoped detail, or a region with a single field).
        fovs = self._meta["fovs_per_region"][well]
        fov = self._current_fov if self._current_fov in fovs else fovs[0]
        chan = self._meta["channels"][0]["name"]        # rank on one representative channel
        best_z_i, best_f = 0, -1.0
        for z_i, z in enumerate(self._meta["z_levels"]):
            try:
                plane = self._reader.read(well, fov, chan, z)
            except Exception:
                continue
            f = _tenengrad(_area_downsample(plane, 512, 512).astype(np.float32))  # downsample = fast
            if f > best_f:
                best_f, best_z_i = f, z_i
        try:
            self._detail.set_current_index("z_level", best_z_i)
        except Exception:
            pass
        self._readout.setText(f"{well}:{fov} focused on reference plane z={best_z_i} (sharpest)")

    def _retire(self, w):
        """Retire a worker thread WITHOUT ever destroying a running QThread (that aborts the app).
        Disconnect its signals first so a tile already queued before the stop can't paint onto a
        freshly-opened plate (the cross-plate corruption the review found); then keep a reference
        alive until it actually finishes (stop() returns after the current item, which is bounded).

        The signal list is DISCOVERED from the worker class, not hardcoded. It used to be a literal
        tuple of names, which silently failed open: a worker declaring a signal absent from that
        tuple kept it connected through teardown and could paint onto the next plate — the very bug
        this method exists to prevent, re-armed by every new worker. Introspection makes a new
        worker correct by construction."""
        if w is None:
            return
        for name in _signal_names(type(w)):
            sig = getattr(w, name, None)
            if sig is not None:
                try:
                    sig.disconnect()
                except TypeError:
                    pass             # nothing connected — fine
        if w.isRunning():
            w.stop()
            self._retired.append(w)
            w.finished.connect(lambda: self._retired.remove(w) if w in self._retired else None)
            # _busy() counts EVERY retired thread (the raw preview included), so a deferred tab
            # switch can only be delivered once the last one exits — hook them all, not just the
            # operator run, or the resync waits for an event that never comes.
            w.finished.connect(self._on_run_drained)

    def _stop_worker(self):
        self._retire(self._worker)
        self._worker = None

    def _stop_preview(self):
        self._retire(self._preview)
        self._preview = None

    def _stop_minerva(self):
        self._retire(self._minerva)
        self._minerva = None

    def showEvent(self, e):
        """Take a GUI slot the moment this window becomes VISIBLE.

        The cap cannot live in ``main()`` alone: every proof script and debug launcher builds a
        ``PlateWindow`` directly and never goes through it, which is exactly how Julio's screen
        filled up. This is the one call every visible window makes, whoever constructed it.

        A refusal closes the window rather than raising: an exception out of showEvent leaves a
        half-built top-level on screen, which is the state we are trying to prevent.
        """
        if _gui_cap_applies() and getattr(self, "_gui_slot", None) is None:
            try:
                self._gui_slot = acquire_gui_slot()
            except GuiAlreadyOpen as exc:
                print(f"squidmip-view: {exc}", file=sys.stderr)
                self._gui_slot = None
                QTimer.singleShot(0, self.close)   # unwind out of showEvent first, then close
                return
        super().showEvent(e)

    def closeEvent(self, e):
        release_gui_slot(getattr(self, "_gui_slot", None))   # let the next window open
        self._gui_slot = None
        self._stop_worker()          # stop the run cleanly; nothing on disk to clean up (no cache)
        self._stop_preview()
        self._stop_minerva()         # files already written stay; only the launch poll is abandoned
        for key in list(self._floating):   # floated tabs are top-levels of their own — Qt won't
            win = self._floating.pop(key)  # close them for us, and each may hold a live shell
            w = win.take_content()
            if w is not None:
                self._dispose_tab_widget(w)
            win.close()
        self._release_loupe_sources()   # joins the loupe read thread
        for w in list(self._op_tabs.values()):
            if hasattr(w, "shutdown"):
                w.shutdown()         # kill any live embedded terminal's shell
        for w in list(self._retired):
            w.wait()                 # join before exit — never leave a QThread running at teardown
        super().closeEvent(e)


def _dark_palette() -> QPalette:
    """A dark palette for the Process pane's tab widget ONLY (see PlateWindow.__init__).

    The tab strip's empty area (behind/beside the tabs) is painted by the STYLE from the palette, not
    by our stylesheets, so in macOS LIGHT mode it rendered white. We fix it by giving the TAB WIDGET a
    Fusion style + this dark palette — scoped to that widget subtree, NOT the whole app. Applying it
    app-wide bled into the embedded ndviewer and hid its per-channel colour swatches (the cmap combo
    indicators), which is why this is deliberately not global."""
    dark, base, text, mut = QColor(7, 10, 20), QColor(11, 14, 20), QColor(230, 237, 243), QColor(87, 96, 109)
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


def _rss_mb() -> tuple:
    """(peak_MB, current_MB_or_None). Peak = the OS high-water mark (ru_maxrss), so it is exact even
    without sampling. Current RSS needs psutil (optional). Returns (0, None) where resource is absent."""
    peak = 0.0
    try:
        import resource
        m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = m / (1024 * 1024) if sys.platform == "darwin" else m / 1024   # darwin: bytes, linux: KB
    except Exception:
        pass
    cur = None
    try:
        import os as _os

        import psutil
        cur = psutil.Process(_os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    return peak, cur


def _install_footprint_monitor(app, win):
    """Track the process memory footprint and PRINT THE PEAK when the GUI closes or crashes.

    A light QTimer prints a live line every few seconds so you can watch the footprint as you drive
    the GUI (open a plate, run MIP, scrub FOVs); the peak is the OS high-water mark, so the final
    number is exact regardless of sampling. Wired to app-quit (normal close), atexit, and the
    excepthook (crash) — so a peak is always reported. Unix only (no-ops where `resource` is absent)."""
    import atexit

    state = {"peak": 0.0, "done": False}

    def _live():
        peak, cur = _rss_mb()
        state["peak"] = max(state["peak"], peak)
        cur_s = f", current {cur:.0f} MB" if cur is not None else ""
        print(f"[footprint] peak {state['peak']:.0f} MB{cur_s}", flush=True)

    def _final(reason: str):
        if state["done"]:
            return
        state["done"] = True
        peak, _ = _rss_mb()
        state["peak"] = max(state["peak"], peak)
        print(f"\n[footprint] FINAL peak RSS: {state['peak']:.0f} MB  ({reason})", flush=True)

    timer = QTimer()
    timer.timeout.connect(_live)
    timer.start(5000)
    win._footprint_timer = timer            # keep a reference alive
    app.aboutToQuit.connect(lambda: _final("window closed"))
    atexit.register(lambda: _final("process exit"))
    _orig_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        _final(f"CRASH: {exc_type.__name__}: {exc}")
        _orig_hook(exc_type, exc, tb)

    sys.excepthook = _hook


# --- how many GUI windows may be open AT ONCE, across processes (Julio, IMA-window-cap) -----
#
# Every agent proof run opened another PlateWindow and left it there, until the screen was full
# and swap was at 5.8 GB of 7. Nothing in the app said no, so the cap has to live HERE, at the
# one place a real instance starts -- not in whatever script happened to launch it.
#
# flock on a slot file, deliberately NOT a pidfile. A pidfile must be cleaned up, and a GUI that
# is killed or crashes never cleans up; that is precisely how these runs ended, so a pidfile
# would have wedged the app permanently shut. The kernel drops an flock when the holder dies
# however it dies, so a crashed window frees its slot with no recovery path to get wrong.

DEFAULT_MAX_GUI = 1


class GuiAlreadyOpen(RuntimeError):
    """Refusing to open another GUI window: the cap is already used up."""


class _GuiSlot:
    """A held slot. Keep the reference alive: closing ``fd`` releases the lock."""

    __slots__ = ("fd", "path")

    def __init__(self, fd: int, path: Path) -> None:
        self.fd = fd
        self.path = path


def gui_slot_limit() -> int:
    """How many GUI windows may be open at once. ``SQUIDMIP_MAX_GUI`` overrides."""
    raw = os.environ.get("SQUIDMIP_MAX_GUI")
    if not raw:
        return DEFAULT_MAX_GUI
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_GUI


def _gui_lock_dir() -> Path:
    d = Path(os.environ.get("SQUIDMIP_GUI_LOCK_DIR")
             or (Path.home() / ".cache" / "squidmip"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def acquire_gui_slot() -> _GuiSlot:
    """Take one of the ``gui_slot_limit()`` slots, or raise :class:`GuiAlreadyOpen`.

    Returns a handle whose lifetime IS the reservation -- hold it for as long as the window
    lives. Never blocks: a GUI that hangs waiting for another GUI to exit is a worse bug than
    the one this prevents.
    """
    import fcntl

    limit = gui_slot_limit()
    lock_dir = _gui_lock_dir()
    for slot in range(limit):
        path = lock_dir / f"gui-{slot}.lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)          # somebody else holds this slot; try the next one
            continue
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        return _GuiSlot(fd, path)

    raise GuiAlreadyOpen(
        f"a SquidMIP window is already open ({limit} allowed at once). Close it first, or "
        f"raise the cap with SQUIDMIP_MAX_GUI=<n>. Lock dir: {lock_dir}"
    )


def release_gui_slot(handle: Optional[_GuiSlot]) -> None:
    """Give the slot back. Idempotent, and safe on an already-closed handle."""
    if handle is None:
        return
    try:
        os.close(handle.fd)       # closing the fd releases the flock
    except OSError:
        pass                      # already closed (or the holder died) -- the lock is gone either way


def _gui_cap_applies() -> bool:
    """The cap guards REAL windows only.

    Offscreen runs (the test suite, and every automated proof) never put anything on Julio's
    screen, and capping them would serialise the suite for no benefit.
    """
    return os.environ.get("QT_QPA_PLATFORM") != "offscreen"


def main(dataset_path: str = None):
    path = dataset_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    slot = None
    if _gui_cap_applies():
        try:
            slot = acquire_gui_slot()
        except GuiAlreadyOpen as e:
            print(f"squidmip-view: {e}", file=sys.stderr)
            sys.exit(1)
    app = QApplication.instance() or QApplication(sys.argv)
    win = PlateWindow(path)
    _install_footprint_monitor(app, win)
    win._gui_slot = slot                  # the reservation lives as long as the window
    win.show()
    if not app.property("_squidmip_test"):
        try:
            sys.exit(app.exec_())
        finally:
            release_gui_slot(slot)
    return win


if __name__ == "__main__":
    main()
