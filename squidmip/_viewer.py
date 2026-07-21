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
  TIFFs. Memory is NOT one-well-at-a-time on the plate side: the MIP run retains one downsampled
  88x88xC float32 tile PER WELL (_OperatorWorker._raw) for the final global-contrast montage, so
  the plate-side footprint is O(n_wells x C) (~190 MB for a 1536wp, C=4), plus a grid-sized RGB
  canvas (~36 MB) and a transient float32 montage buffer at run end. Bounded by the plate format
  (<=1536 wells), not by z/frame size. What IS one-well-at-a-time is project_plate's producer
  (workers x one ~139 MB well) and the detail viewer's LRU-bounded decoded planes.
- Hit-testing / cell fitting are pure functions (unit-testable); widgets run headless under
  QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QSocketNotifier, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QStyleFactory, QTabBar, QTabWidget, QVBoxLayout, QWidget,
)

from squidmip._engine import _default_workers
from squidmip._layers import OperationStack
from squidmip._montage import _area_downsample, _hex_to_rgb01, _window
from squidmip._output import parse_well_id

_SUPPORTED_PLATES = ("384", "1536")   # well-plate formats the tool currently accepts (no stitching yet)
_CELL = 88                 # per-well px in the low-res overview (1536wp -> ~4224x2816)
_PUSH_PX = 512             # per-well px pushed to the ndviewer scan-slider (downsampled -> bounded RAM)
_HDR, _COLH = 46, 30       # left / top label margins (px)
_PAD = 16                  # breathing room around the plate
_VIEWER_WORKERS = min(6, _default_workers())   # adapt to the machine, but CAP at 6: the producer's
                           # peak RAM is ~workers x one-well (~139 MB each on a 1536wp), and projection
                           # throughput scales only sublinearly past ~6 threads — so more workers buys
                           # little speed for linearly more memory. 6 balances both, leaves GUI cores.
_BG = "#070a0f"
_GRID, _RED, _MUTED, _ACCENT = QColor(0, 0, 0), QColor("#ff2d2d"), QColor("#8b98ad"), QColor("#58a6ff")

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
)
_OPERATIONS_BY_KEY = {op.key: op for op in _OPERATIONS}

# Roadmap slots shown (disabled) in the Process console so the direction is visible: hand-off to
# Minerva Author, and a locally-run agent (Nautilus) that builds the operator you ask it for.
# Roadmap cards shown under "TO BE ADDED". Empty for now — we do NOT advertise Minerva Author or
# Nautilus in the UI. Add a card as (label, blurb) when a real next operator is ready to surface.
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


class _RunningContrast:
    """Per-channel global contrast that updates as wells stream in (histogram over tiles so far)."""

    def __init__(self, n_ch: int, dmax: float, pct=(1.0, 99.8), bins=512):
        self._bins, self._dmax, self._pct = bins, max(1.0, float(dmax)), pct
        self._hist = [np.zeros(bins, dtype=np.int64) for _ in range(n_ch)]

    def add(self, ch: int, tile: np.ndarray):
        idx = np.clip((tile.ravel() / self._dmax * self._bins).astype(int), 0, self._bins - 1)
        self._hist[ch] += np.bincount(idx, minlength=self._bins)

    def window(self, ch: int) -> tuple[float, float]:
        h = self._hist[ch]
        tot = h.sum()
        if tot == 0:
            return 0.0, self._dmax
        cdf = np.cumsum(h) / tot
        lo = np.searchsorted(cdf, self._pct[0] / 100.0) / self._bins * self._dmax
        hi = np.searchsorted(cdf, self._pct[1] / 100.0) / self._bins * self._dmax
        return float(lo), float(max(hi, lo + 1))


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


def _composite_rgb(planes, colors, windows) -> np.ndarray:
    """(C, y, x) planes -> (y, x, 3) float RGB in [0, 1], one window per channel.

    The single definition of "channels to colour" for the viewer. Windowing stays with the
    caller: the plate's three renderers legitimately source their window differently (streaming
    running-contrast, per-well percentiles, the loupe mirroring a tile's window), and folding
    that in here would force a policy where there isn't one."""
    out = None
    for c_i, plane in enumerate(planes):
        lo, hi = windows[c_i]
        contrib = _window(plane, lo, hi if hi > lo else lo + 1)[:, :, None] * colors[c_i][None, None, :]
        out = contrib if out is None else out + contrib
    if out is None:
        return np.zeros((1, 1, 3), np.float32)
    return np.clip(out, 0, 1)


def _percentile_window(plane, pct=(1.0, 99.8)) -> tuple[float, float]:
    """The plate's per-well contrast rule, in one place (mirrors _ComputedPlateWorker)."""
    lo, hi = float(np.percentile(plane, pct[0])), float(np.percentile(plane, pct[1]))
    return lo, hi if hi > lo else lo + 1


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
        """(C, y, x) crop at ``level``, clipped to the field. Runs on the loupe worker thread."""
        raise NotImplementedError

    def coarse(self, well_id):
        """A small whole-field (C, y, x) plane used ONLY to derive the contrast window."""
        raise NotImplementedError


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
        self.n_levels = 1                      # raw TIFFs have no pyramid
        self.pixel_size_um = meta.get("pixel_size_um")
        self._channels = [c["name"] for c in meta["channels"]]
        zs = meta["z_levels"]
        self._z = zs[len(zs) // 2]             # mid plane, as the preview does
        self._cache_key = None
        self._cache = None

    def available(self, well_id) -> tuple[bool, str]:
        if well_id in self._meta["regions"]:
            return True, ""
        return False, "no image for this well"

    def _planes(self, well_id):
        if self._cache_key != well_id:
            fov = self._fov_of(well_id)
            self._cache = np.stack([
                np.asarray(self._reader.read(well_id, fov, ch, self._z)) for ch in self._channels])
            self._cache_key = well_id
        return self._cache

    def read_crop(self, well_id, level, y0, x0, h, w):
        p = self._planes(well_id)              # level is always 0 here (n_levels == 1)
        return p[:, y0:y0 + h, x0:x0 + w]

    def coarse(self, well_id):
        p = self._planes(well_id)
        return np.stack([_area_downsample(p[c], _CELL, _CELL) for c in range(p.shape[0])])


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
        h, w = max(1, min(int(h), ny)), max(1, min(int(w), nx))
        y0 = max(0, min(int(y0), ny - h))
        x0 = max(0, min(int(x0), nx - w))
        return np.asarray(arr[0, :, 0, y0:y0 + h, x0:x0 + w].read().result())

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

    ready = pyqtSignal(int, str, object, object)    # (gen, well_id, crop|None, error|None)

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
                self.ready.emit(gen, well_id, crop, None)
            except Exception as e:                    # a racing writer / deleted plate / bad path
                self.ready.emit(gen, well_id, None, f"{type(e).__name__}: {e}")


# --- plate overview widget (one cell per well; hue-coded status; fit-to-view) ---------------

class PlateOverview(QWidget):
    """The low-res plate: an RGB canvas of MIP tiles, a per-well status hue, a red box, and a
    press-and-hold LOUPE that overlays real acquisition pixels for the well under the cursor
    (IMA-208 — the montage itself is far too coarse to magnify; see the loupe block above)."""

    hovered = pyqtSignal(str)              # region id (or "" off-plate), for the window's readout
    wellActivated = pyqtSignal(str, int)   # (well_id, fov_index) double-clicked -> load in ndviewer

    def __init__(self, rows, cols, wells: dict):
        """``wells``: (row_index, col_index) -> well_id for every acquired well (drawn grey until
        processed). Tiles/status arrive as an operator runs."""
        super().__init__()
        self._rows, self._cols = list(rows), list(cols)
        self._nr, self._nc = len(self._rows), len(self._cols)
        self._by_rc: dict[tuple, str] = dict(wells)            # every acquired well (for status + hit-test)
        self._status: dict[tuple, str] = {rc: "empty" for rc in wells}
        self._tiles: set[tuple] = set()                        # cells that have a tile painted (any layer)
        self._tiles_by_layer: dict[str, set] = {}              # layer -> cells with an image there
        self._canvas = QImage(self._nc * _CELL, self._nr * _CELL, QImage.Format_RGB888)
        self._canvas.fill(QColor(_BG))
        self._final = None            # crisp global-contrast montage of the ACTIVE layer (or None)
        # Layer stack render: the base ("raw") is self._canvas; each operator draws into its own
        # per-layer canvas/final. self._active is the layer the plate currently shows (LayersTab picks
        # it via set_active_layer). Keeps memory to one small montage-canvas per layer used.
        self._op_canvas: dict[str, QImage] = {}
        self._op_final: dict[str, QImage] = {}
        self._active = "raw"
        self._scaled = None           # cached pixmap of (final|canvas) scaled to the current zoom;
        self._scaled_cd = None        # rebuilt only when zoom (cd) or the source image changes — so
        #                               a hover/pan repaint blits 1:1 instead of re-resampling 12 MP.
        self._cd = float(_CELL)       # displayed px/well (fit baseline, then wheel-zoomed)
        self._ox = self._oy = _PAD    # top-left of the plate within the widget (pan-able)
        self._hover = None
        self._sel = None              # well selected from the ndviewer FOV slider
        self._press = None            # (x, y, ox, oy) at left-press, for drag-to-pan
        self._panning = False
        self._user_view = False       # True once the user wheel-zooms/pans (stop auto-fitting)
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
        return c["well_id"], level, (cy - crop // 2, cx - crop // 2, crop, crop), s_loupe, mag

    def _request_loupe(self, x, y):
        geo = self._loupe_geometry(x, y)
        if geo is None or self._loupe_worker is None:
            return
        well, level, (y0, x0, h, w), _s, _m = geo
        ok, why = self._loupe_src.available(well)
        if not ok:
            self._loupe_img, self._loupe_note = None, why
            self.update()
            return
        self._loupe_gen += 1
        self._loupe_worker.request(self._loupe_gen, well, level, y0, x0, h, w)

    def _on_loupe_crop(self, gen, well_id, crop, error):
        """A crop landed. Drop it unless it is the newest request and the loupe is still up."""
        if gen != self._loupe_gen or self._loupe is None:
            return
        if error is not None or crop is None or crop.size == 0:
            self._loupe_img, self._loupe_note = None, error or "no pixels here"
            self.update()
            return
        win = self._loupe_win.get(well_id)
        if win is None:
            # Mirror the TILE's contrast rule on the WELL's pixels — never percentiles of the crop
            # under the cursor, which would make brightness lurch as the cursor moves and make the
            # inset look like different data. There is no shared window object to borrow: the
            # streaming _RunningContrast dies with its worker and computed plates stretch per well.
            try:
                coarse = self._loupe_src.coarse(well_id)
                win = [_percentile_window(coarse[c]) for c in range(coarse.shape[0])]
            except Exception:
                win = [(0.0, 1.0)] * crop.shape[0]
            self._loupe_win[well_id] = win
        colors = self._loupe_colors
        if colors is None:
            colors = np.ones((crop.shape[0], 3), np.float32)
        rgb = (_composite_rgb([crop[c].astype(np.float32) for c in range(crop.shape[0])],
                              colors, win) * 255).astype(np.uint8)
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

    # -- data in --
    def add_tile(self, ri: int, ci: int, well_id: str, rgb: np.ndarray, layer: str = "raw"):
        if (ri, ci) not in self._by_rc:    # ignore a stale tile from a retired run / foreign cell
            return
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        img = QImage(rgb.data, _CELL, _CELL, 3 * _CELL, QImage.Format_RGB888)
        p = QPainter(self._canvas_for(layer))
        p.drawImage(ci * _CELL, ri * _CELL, img)
        p.end()
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
    #                           │ release                  │ release / leave / focus-out
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
        px, py = x - (self._ox + _HDR), y - (self._oy + _COLH)
        return well_at(self._rows, self._cols, self._by_rc, px, py, self._cd)

    def wheelEvent(self, e):
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
        if e.button() == Qt.LeftButton:
            self._press = (e.x(), e.y(), self._ox, self._oy)
            self._panning = False
            c = self._cell(e.x(), e.y())
            if self._loupe_src is not None and c and c["well_id"]:   # ARM (never off-plate/empty)
                self._hold.start()

    def mouseMoveEvent(self, e):
        if self._loupe is not None:                  # LOUPE: the inset tracks; panning is dead
            self._loupe["x"], self._loupe["y"] = e.x(), e.y()
            self._request_loupe(e.x(), e.y())
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
        self._press = None
        self._panning = False
        self._dismiss_loupe()                        # release always dismisses

    def leaveEvent(self, e):
        self._hold.stop()                            # cursor left mid-hold: release may never come
        self._dismiss_loupe()
        self._hover = None
        self.hovered.emit("")
        self.update()

    def focusOutEvent(self, e):
        self._hold.stop()                            # window deactivated mid-hold: same reasoning
        self._dismiss_loupe()
        super().focusOutEvent(e)

    def mouseDoubleClickEvent(self, e):
        # Qt sends press/release/dblclick — the second press already re-armed the hold timer, so
        # kill it here or one double-click both opens the well AND raises a loupe.
        self._hold.stop()
        self._dismiss_loupe()
        c = self._cell(e.x(), e.y())
        if c and c["well_id"]:
            self.wellActivated.emit(c["well_id"], _fov_of_well(c["well_id"]))

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
        # Blit the montage from a cached pixmap scaled to the current zoom. The expensive smooth
        # resample runs ONCE per zoom/source-change (not every repaint) — pan/hover just re-blit.
        w, h = max(1, int(W)), max(1, int(H))
        if self._scaled is None or self._scaled_cd != cd or self._scaled.width() != w or self._scaled.height() != h:
            self._scaled = QPixmap.fromImage(self._active_source()).scaled(
                w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self._scaled_cd = cd
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
                x0, y0 = ax + ci * cd, ay + ri * cd
                ex, ey = int(x0 + (cd - d) / 2), int(y0 + (cd - d) / 2)
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

        p.setPen(QPen(_GRID, 3))       # black grid lines between wells (room for multi-FOV, IMA-187)
        for c in range(nc + 1):
            p.drawLine(int(ax + c * cd), int(ay), int(ax + c * cd), int(ay + H))
        for r in range(nr + 1):
            p.drawLine(int(ax), int(ay + r * cd), int(ax + W), int(ay + r * cd))
        # Column/row labels THIN OUT as cells shrink so they never overlap (a 48-col 1536wp would
        # otherwise cram "1..48" into a few px). Always draw the hovered row/col so hover still reads.
        p.setFont(QFont("Helvetica Neue", 11, QFont.DemiBold))
        cstep = max(1, int(np.ceil(22.0 / cd)))
        rstep = max(1, int(np.ceil(18.0 / cd)))
        for c in range(nc):
            hov = bool(self._hover and self._hover[1] == c)
            if c % cstep and not hov:
                continue
            p.setPen(_ACCENT if hov else _MUTED)
            p.drawText(int(ax + c * cd), int(self._oy), int(cd), _COLH, Qt.AlignCenter, str(self._cols[c]))
        for r in range(nr):
            hov = bool(self._hover and self._hover[0] == r)
            if r % rstep and not hov:
                continue
            p.setPen(_ACCENT if hov else _MUTED)
            p.drawText(int(self._ox), int(ay + r * cd), _HDR, int(cd), Qt.AlignCenter, str(self._rows[r]))
        if self._sel is not None:          # the CURRENT well in the detail viewer = a red BOX
            ri, ci = self._sel
            p.setPen(QPen(_RED, 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(int(ax + ci * cd), int(ay + ri * cd), int(cd), int(cd))
        if self._hover is not None:        # where the cursor is, moving around the plate = a red DOT
            ri, ci = self._hover           # SAME geometry as the status dots -> overlays them exactly
            x0, y0 = ax + ci * cd, ay + ri * cd
            ex, ey = int(x0 + (cd - d) / 2), int(y0 + (cd - d) / 2)
            p.setPen(Qt.NoPen)
            p.setBrush(_RED)
            p.drawEllipse(ex, ey, int(d), int(d))
        if self._loupe is not None:        # press-and-hold magnifier, over everything else
            self._paint_loupe(p)
        # a fine outer white frame around the whole plate view
        p.setPen(QPen(QColor("#c9d1d9"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.end()

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


# --- operator worker: stream a projection over the plate, fill row-major -------------------

class _OperatorWorker(QThread):
    """Runs an operator (MIP) over the plate AND persists it as a navigable multiscale OME-Zarr plate
    (``write_plate``), filling one thumbnail per well as each is written. Projection + pyramid write
    run in write_plate's bounded producer/writer pools; our ``_on_well`` renders the plate tile and
    is called FROM THOSE WRITER THREADS, several at once — so the shared running-contrast, the ``_raw``
    tile store, and the done-counter are guarded by ``_lock`` (the expensive per-channel downsample
    happens OUTSIDE the lock, so downsampling still parallelises). Memory stays O(engine + write
    workers) wells in flight plus one 88x88xC native-dtype tile per well retained in ``_raw`` for the
    final montage. The written ``plate.ome.zarr`` is the durable, re-openable artifact.
    """

    tileReady = pyqtSignal(int, int, str, object)   # (row_index, col_index, well_id, rgb tile)
    progress = pyqtSignal(int, int)                 # (done, total)
    finalReady = pyqtSignal(object)                 # final global-contrast montage (H, W, 3) uint8
    writtenReady = pyqtSignal(str)                  # path of the written plate.ome.zarr
    wellFailed = pyqtSignal(int, int)               # (ri, ci) of a well SKIPPED on a read error
    pushReady = pyqtSignal(int, object)             # (fov_idx, [per-channel ~512px plane]) for the slider
    failed = pyqtSignal(str)                        # whole-run failure (not a per-well skip)
    finished_ok = pyqtSignal()

    def __init__(self, operator: str, reader, meta, fov_index: dict, nr: int, nc: int, out_dir: str,
                 regions=None, save: bool = True):
        super().__init__()
        self._operator = operator
        self._reader, self._meta = reader, meta
        self._fov_index, self._nr, self._nc = fov_index, nr, nc
        self._out_dir = out_dir
        self._regions = regions          # None = whole plate; a list = subset preview (those wells only)
        self._save = save                # False = PREVIEW: compute + push to the viewer, write NOTHING
        self._total = len(regions) if regions is not None else len(meta["regions"])
        self._channels = [c["name"] for c in meta["channels"]]
        self._colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in meta["channels"]])
        self._dtype = np.dtype(meta["dtype"])
        self._contrast = _RunningContrast(len(self._channels), float(np.iinfo(self._dtype).max))
        self._raw: dict[tuple, np.ndarray] = {}   # (ri,ci) -> (C, _CELL, _CELL) tiles, for the final montage
        self._lock = threading.Lock()             # guards _contrast/_raw/_done (on_well runs on writer threads)
        self._done = 0
        self._stop = threading.Event()            # set by the window to end the run cleanly

    def stop(self):
        """Ask the run to stop; write_plate polls this and abandons after in-flight wells drain."""
        self._stop.set()

    def _on_well(self, region, fov, image):
        """Called per written well (on a write_plate WRITER THREAD): render the plate thumbnail."""
        info = self._fov_index[region]
        ri, ci, well_id = *info["rc"], info["well_id"]
        well = image[0, :, 0]  # (C, Y, X)
        tiles = [_fit_cell(well[c_i]) for c_i in range(len(self._channels))]  # downsample OUTSIDE lock
        raw = np.empty((len(tiles), _CELL, _CELL), self._dtype)              # native dtype (half the RAM)
        with self._lock:                          # shared contrast/raw/counter -> serialize (cheap part)
            for c_i, ds in enumerate(tiles):
                raw[c_i] = ds
                self._contrast.add(c_i, ds)
            wins = [self._contrast.window(c_i) for c_i in range(len(tiles))]  # streaming global window
            self._raw[(ri, ci)] = raw
            self._done += 1
            done = self._done
        rgb = _composite_rgb(tiles, self._colors, wins)      # colour OUTSIDE the lock
        self.tileReady.emit(ri, ci, well_id, (rgb * 255).astype(np.uint8))
        self.progress.emit(done, self._total)
        # feed the ndviewer growing slider: one ~512px plane per channel, in memory (register_array),
        # so scrubbing the processed wells is instant and z-collapsed (nz=1). Downsampled -> bounded.
        push = [_area_downsample(well[c_i], _PUSH_PX, _PUSH_PX).astype(self._dtype)
                for c_i in range(len(self._channels))]
        self.pushReady.emit(info["idx"], push)

    def _on_error(self, region, fov, exc):
        """A well's projection failed (corrupt/missing plane): SKIP it, mark its dot failed, keep the
        run alive. One bad file must not abort a whole-plate run."""
        info = self._fov_index.get(region)
        if info is not None:
            self.wellFailed.emit(*info["rc"])

    def run(self):
        try:
            if self._save:
                from squidmip import write_plate  # persist + project in one bounded, streaming pass

                write_plate(self._reader, self._out_dir, n_fovs=1, workers=_VIEWER_WORKERS,
                            projector=self._operator, tiff=False, on_well=self._on_well,
                            stop=self._stop.is_set, on_error=self._on_error, regions=self._regions)
                if self._stop.is_set():
                    return  # window closing / re-opening; drop out cleanly (no final/written emit)
                self.finalReady.emit(self._final_montage())
                self.writtenReady.emit(str(Path(self._out_dir) / "plate.ome.zarr"))
            else:
                # PREVIEW: run the engine over the subset and push each result to the plate + slider,
                # writing NOTHING to disk (so testing an operator on a few wells costs no disk + only
                # the subset's compute). Same math as the saved run — a faithful preview.
                from squidmip import project_plate

                stream = project_plate(self._reader, workers=_VIEWER_WORKERS, projector=self._operator,
                                       on_error=self._on_error, regions=self._regions)
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
                self.finalReady.emit(self._final_montage())
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _final_montage(self) -> np.ndarray:
        wins = [self._contrast.window(ch) for ch in range(len(self._channels))]
        canvas = np.zeros((self._nr * _CELL, self._nc * _CELL, 3), np.float32)
        for (ri, ci), raw in self._raw.items():
            y0, x0 = ri * _CELL, ci * _CELL
            for ch in range(raw.shape[0]):
                lo, hi = wins[ch]
                canvas[y0:y0 + _CELL, x0:x0 + _CELL] += _window(raw[ch], lo, hi)[:, :, None] * self._colors[ch][None, None, :]
        # clip/scale IN PLACE — avoids 3 grid-sized float32 copies (a ~430 MB transient on a 1536wp)
        np.clip(canvas, 0, 1, out=canvas)
        canvas *= 255
        return canvas.astype(np.uint8)


class _PreviewWorker(QThread):
    """Fast RAW preview so the plate shows imagery the moment it opens — before any operator runs
    (the "downsample the plate before opening" step). Reads ONE representative z-plane per channel
    per well (not the whole stack), area-downsamples, and composites by display colour. Cheap
    relative to a full projection; parallel reads; one well's planes at a time. Status stays 'empty'
    (grey frame) — this is a preview, not a processed result. A later operator overwrites each tile.
    """

    tileReady = pyqtSignal(int, int, str, object)   # (row_index, col_index, well_id, rgb tile)

    def __init__(self, reader, meta, fov_index: dict, order: list):
        super().__init__()
        self._reader, self._meta = reader, meta
        self._fov_index, self._order = fov_index, order
        self._channels = [c["name"] for c in meta["channels"]]
        self._colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in meta["channels"]])
        self._contrast = _RunningContrast(len(self._channels), float(np.iinfo(np.dtype(meta["dtype"])).max))
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            from concurrent.futures import ThreadPoolExecutor
            zs = self._meta["z_levels"]
            z_mid = zs[len(zs) // 2]      # a mid-stack plane is a fair single-plane preview

            def load(region):
                fov = self._meta["fovs_per_region"][region][0]
                return region, [_fit_cell(self._reader.read(region, fov, ch, z_mid).astype(np.float32))
                                for ch in self._channels]

            with ThreadPoolExecutor(max_workers=_VIEWER_WORKERS) as ex:
                for region, tiles in ex.map(load, self._order):   # row-major order preserved
                    if self._stop.is_set():
                        return
                    rgb = np.zeros((_CELL, _CELL, 3), np.float32)
                    for c_i, ds in enumerate(tiles):
                        self._contrast.add(c_i, ds)
                        lo, hi = self._contrast.window(c_i)
                        rgb += _window(ds, lo, hi)[:, :, None] * self._colors[c_i][None, None, :]
                    ri, ci = self._fov_index[region]["rc"]
                    self.tileReady.emit(ri, ci, region, (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        except Exception:
            pass   # preview is best-effort; the operator run is the authoritative result


class _ComputedPlateWorker(QThread):
    """Read a previously-written OME-Zarr plate back into the viewer (no recompute).

    Streams each well from disk: a coarse pyramid level -> the plate thumbnail, and a ~512px level ->
    the ndviewer slider (register_array). Bounded (one well in flight); reads via tensorstore off the
    GUI thread so opening a big computed plate never freezes the window."""

    tileReady = pyqtSignal(int, int, str, object)   # (ri, ci, well_id, rgb tile)
    pushReady = pyqtSignal(int, object)             # (fov_idx, [per-channel ~512px plane])
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, base, wells, colors, coarse_lvl, push_lvl, dtype):
        super().__init__()
        self._base = base                 # plate.ome.zarr path
        self._wells = wells               # [(well_id, wellpath, fov, ri, ci, flat_idx)]
        self._colors = colors             # (C, 3) float RGB
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
                tiles = [_fit_cell(plane.astype(np.float32)) for plane in coarse]
                wins = [_percentile_window(t) for t in tiles]   # per-well stretch (no global pass)
                rgb = _composite_rgb(tiles, self._colors, wins)
                self.tileReady.emit(ri, ci, wid, (rgb * 255).astype(np.uint8))
                push_src = self._read(wpath, fov, self._push)             # detail-slider source (C,Y,X)
                push = [_area_downsample(push_src[c], _PUSH_PX, _PUSH_PX).astype(self._dtype)
                        for c in range(push_src.shape[0])]
                self.pushReady.emit(idx, push)
                self.progress.emit(i, n)
            if not self._stop.is_set():
                self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# --- main window: plate overview | embedded ndviewer ----------------------------------------

class PlateWindow(QMainWindow):
    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("MIP tool")
        self.resize(1600, 950)
        self._worker = None           # the operator (MIP) run
        self._preview = None          # the raw preview fill on open
        self._retired = []            # workers asked to stop; kept alive until they actually finish
        self._overview = None
        self._reader = None
        self._meta = None
        self._fov_index = {}
        self._pushed = set()          # wells whose raw z-stack is already registered in the detail viewer
        self._final_arr = None        # keep the final montage array alive for its QImage

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
        self._acq_path = None         # the opened acquisition dir (persist writes next to it)
        self._processed_plate = None  # path of the written plate.ome.zarr once an operator persists it
        self._plate_mode = "raw"      # what the plate view is showing — shown in the plate-pane title
        self._op_stack = OperationStack()   # the toggleable layer stack (base + applied operators)
        self._active_op_key = None    # operator whose tiles are streaming into its layer right now
        self._layers_tab = None       # the Layers tab widget, once opened
        self._order = []              # well order = the detail's FOV-slider order
        self._op_tabs = {}            # key -> operator-UI widget currently open as a tab in _left_tabs
        self._loupe_sources = {}      # layer key -> _LoupeSource backing that layer's pixels (IMA-208)

        # THREE-PANE layout. Tabs live ONLY inside the top-left pane (their bar sits at the pane's top,
        # like the plate pane's title bar) — never a global strip across the window:
        #   top-left  = the PROCESS console, a QTabWidget: a "Process wells" home tab (operator list),
        #               and one tab per operator you open (MIP -> where-to-save UI; Record -> recorder
        #               UI). The right pane stays a plain singleton, so operator UIs live here.
        #   bottom-left = the HCS PLATE view (<= half the display wide); its title bar names the plate.
        #   right     = the ndviewer_light array viewer, full height (a singleton — no tabs).

        # top-left: the process console (build the home tab first — it owns self._readout, which
        # _make_detail_viewer writes to if ndviewer is unavailable).
        self._left_tabs = QTabWidget()
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
        self._left_tabs.addTab(self._build_process_pane(), "Process wells")
        self._left_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)  # home tab isn't closable

        # right: the ndviewer array viewer directly (a singleton — no tab)
        self._detail = self._make_detail_viewer()
        if self._detail is not None:   # connect the FOV slider -> red box ONCE (not per ingest)
            slider = getattr(self._detail, "_fov_slider", None)
            if slider is not None:
                slider.valueChanged.connect(self._on_fov_slider)
            self._right_widget = self._detail
        else:
            ph = QLabel("ndviewer_light unavailable — pip install ndviewer_light")
            ph.setAlignment(Qt.AlignCenter)
            ph.setStyleSheet("color:#8b98ad;")
            self._right_widget = ph

        # bottom-left: plate view (drop target until an acquisition opens). Its FIXED title bar names
        # the wellplate we're on (the acquisition) — the plate's identity lives with the plate.
        self._plate_title = QLabel("well plate")   # plate name; shows the hovered well (large) on hover
        self._plate_title.setStyleSheet(
            "color:#e6edf3;font-size:17px;font-weight:800;padding:9px 14px;"
            "background:#0b0e14;border-bottom:1px solid #232b3a;")
        self._drop = QLabel("Drop a Squid acquisition folder here\n\n"
                            "then pick an operator in  Process wells")
        self._drop.setAlignment(Qt.AlignCenter)
        self._drop.setStyleSheet("color:#8b98ad;font-size:16px;border:2px dashed #232b3a;border-radius:12px;margin:24px;")
        plate_host = QWidget()
        plate_host.setStyleSheet(f"background:{_BG};")
        self._left_l = QVBoxLayout(plate_host)
        self._left_l.setContentsMargins(0, 0, 0, 0)
        self._left_l.setSpacing(0)
        self._left_l.addWidget(self._plate_title)
        self._left_l.addWidget(self._drop, 1)    # the plate overview replaces this on ingest

        left_col = QSplitter(Qt.Vertical)
        left_col.setStyleSheet("QSplitter::handle{background:#232b3a;height:1px;}")
        left_col.setChildrenCollapsible(False)
        left_col.addWidget(self._left_tabs)
        left_col.addWidget(plate_host)
        left_col.setStretchFactor(0, 0)
        left_col.setStretchFactor(1, 1)
        left_col.setSizes([340, 610])

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
        outer.addWidget(left_col)
        outer.addWidget(right_frame)
        outer.setSizes([760, 760])                 # divider fixed at the middle of the window
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 1)
        self._split = outer
        self.setCentralWidget(outer)

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

    # -- operator UIs live as tabs INSIDE the top-left pane (home tab + one per opened operator) ----
    def _open_op_tab(self, key: str, title: str, builder):
        """Open (or focus) an operator's UI as a tab beside 'Process wells'. Built lazily, once."""
        w = self._op_tabs.get(key)
        if w is None:
            w = builder()
            self._op_tabs[key] = w
            self._left_tabs.addTab(w, title)
        self._left_tabs.setCurrentWidget(w)

    def _close_op_tab(self, index: int):
        if index == 0:                                     # 'Process wells' home tab — never closable
            return
        w = self._left_tabs.widget(index)
        self._left_tabs.removeTab(index)
        for k, v in list(self._op_tabs.items()):
            if v is w:
                del self._op_tabs[k]
        if w is self._layers_tab:                          # drop the stale ref so refresh no-ops
            self._layers_tab = None
            self._layers_box = None
        if hasattr(w, "shutdown"):                         # a live terminal — kill its shell first
            w.shutdown()
        w.deleteLater()

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
            self.run_operator(op.key, out_parent=dest, preview_limit=spin.value(), save=save)

        prev.clicked.connect(do_preview)
        v.addWidget(prev)
        v.addStretch(1)
        # both run buttons enable once an acquisition is open (the tab is only reachable then, but be safe)
        for b in (run, prev):
            b.setEnabled(self._reader is not None)
        return w

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
            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox(ly.label); cb.setChecked(ly.enabled); cb.setStyleSheet("color:#e6edf3;")
            cb.toggled.connect(lambda on, k=ly.key: self._on_layer_toggle(k, on))
            up = QPushButton("↑"); up.setStyleSheet(_BTN_QSS); up.setFixedWidth(34)
            up.clicked.connect(lambda _=False, k=ly.key: self._on_layer_move(k, +1))
            dn = QPushButton("↓"); dn.setStyleSheet(_BTN_QSS); dn.setFixedWidth(34)
            dn.clicked.connect(lambda _=False, k=ly.key: self._on_layer_move(k, -1))
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
        """Show the topmost enabled layer on the plate; keep the title in sync."""
        top = self._op_stack.top_enabled()
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
        # stop any in-flight run/preview and clear prior state before opening a new acquisition
        self._stop_worker()
        self._stop_preview()
        self._reader = self._meta = None
        self._fov_index = {}
        self._pushed = set()
        self._current_well = None
        self._final_arr = None
        self._enable_operators(False)
        if self._overview is not None:
            self._overview.setParent(None)
            self._overview.deleteLater()
            self._overview = None
        self._readout.setText("scanning acquisition …")
        QApplication.processEvents()
        try:
            reader = open_reader(str(p))
            meta = reader.metadata
        except Exception as e:   # not a Squid acquisition / unreadable -> report, don't crash the app
            self._readout.setText(f"not a readable Squid acquisition: {e}")
            self._drop.show()
            return
        fmt = str(meta.get("wellplate_format", ""))
        if not any(s in fmt for s in _SUPPORTED_PLATES):   # scope guard: supported formats only
            self._readout.setText(f"only 384- and 1536-well plates are supported right now — this is a {fmt or 'other'}")
            self._drop.show()
            return
        self._reader, self._meta = reader, meta
        self._acq_name = Path(p).name
        self._acq_path = Path(p)
        self._processed_plate = None

        # Order wells in TRUE plate row-major (A,B,...,Z,AA,...). NOT lexicographic ("AA" < "B").
        # This parses region ids as well ids — guard it: a readable acquisition whose regions are
        # NOT well-plate ids (glass slide, manual/coordinate names, "R2C3", "0") must report, not
        # crash. parse_well_id / plate_metadata raise ValueError on those. (contract: never crash.)
        try:
            plate = plate_metadata(meta["regions"], field_count=1)["plate"]
            present_rows = [r["name"] for r in plate["rows"]]
            present_cols = [c["name"] for c in plate["columns"]]
        except (ValueError, KeyError) as e:
            self._reader = self._meta = None
            self._readout.setText(
                f"not a well-plate acquisition — regions like {list(meta['regions'])[:3]} aren't "
                f"well ids (e.g. B2); the MIP tool needs a well plate. ({type(e).__name__})")
            self._drop.show()
            return
        # Prefer the FULL plate-format grid (every position, evenly spaced — no collapsed gaps).
        # Fall back to present-only when the format is unknown or the wells don't fit it.
        grid = _plate_grid(meta.get("wellplate_format"))
        if grid and set(present_rows) <= set(grid[0]) and set(present_cols) <= set(grid[1]):
            rows, cols = grid
        else:
            rows, cols = present_rows, present_cols
        row_of = {r: i for i, r in enumerate(rows)}   # plate order: A=0,B=1,...,Z=25,AA=26,...
        col_of = {c: i for i, c in enumerate(cols)}

        def _rc(region):
            rr, cc = parse_well_id(region)
            return (row_of[rr], col_of[cc])
        order = sorted(meta["regions"], key=_rc)
        wells = {}
        for idx, region in enumerate(order):
            rc = _rc(region)
            self._fov_index[region] = {"idx": idx, "well_id": region, "rc": rc}
            wells[rc] = region

        self._order = order                          # well order = the detail's FOV-slider order
        self._overview = PlateOverview(rows, cols, wells)
        self._overview.hovered.connect(self._on_hover)
        self._overview.wellActivated.connect(self.activate_well)
        self._plate_mode = "raw"                     # a freshly-opened plate shows raw previews
        self._plate_title.setText(f"{self._acq_name}   ·   raw")   # bottom-left plate-pane title
        self._op_stack.reset()                       # fresh layer stack (base only)
        self._active_op_key = None
        if getattr(self, "_raw_btn", None):
            self._raw_btn.hide()                     # raw view on open -> nothing to return from
        self._refresh_layers_tab()
        self._drop.hide()
        self._left_l.addWidget(self._overview, 1)   # fills the pane and self-fits — no scrollbars

        self._setup_raw_detail()

        self._enable_operators(True)

        # The loupe works from the moment the folder opens — the raw layer's real pixels are the
        # acquisition's own TIFFs, the same planes the preview below is about to downsample. No
        # operator run is required to look closely at a well.
        self._loupe_sources = {"raw": _RawLoupeSource(
            reader, meta, lambda w: _fov_of_well(w, meta.get("fovs_per_region")))}
        self._update_loupe_source()

        # fast RAW preview: fill the plate with downsampled thumbnails immediately (grey dots),
        # in the SAME row-major order the operator will later process them in.
        self._preview = _PreviewWorker(reader, meta, self._fov_index, order)
        self._preview.tileReady.connect(self._on_preview_tile)
        self._preview.start()   # (the detail already landed on order[0] via _setup_raw_detail)
        # top-left = LIVE STATUS (what's happening / what's shown); the plate name is the pane title
        # Multi-FOV policy (IMA-191): current scope is one FOV per well (a well = a condition, exactly
        # Nick's "n=1 per condition"). When wells hold >1 FOV we SAMPLE the first and say so — honest
        # interim until high-throughput stitching (stitch -> MIP -> one composite/well) lands.
        multi = sum(1 for r in order if len(meta["fovs_per_region"][r]) > 1)
        note = (f" · {multi} multi-FOV well(s): sampling 1 FOV/well (stitching TBD)" if multi else "")
        self._readout.setText(f"live · {len(self._fov_index)} wells · double-click to open{note}")

    def _setup_raw_detail(self):
        """Point the detail viewer at the RAW acquisition: full z-stack, full frame, whole-plate FOV
        slider. Registers every well's raw plane PATHS up front (cheap — paths only, no image I/O) so
        scrubbing shows a real (lazily read + cached) image per well instead of black. Shared by open
        (ingest) and 'Return to raw view'."""
        if self._detail is None or self._reader is None:
            return
        meta, reader, order = self._meta, self._reader, self._order
        h, w = meta["frame_shape"]
        channels = [c["name"] for c in meta["channels"]]
        self._detail.start_acquisition(channels, meta["n_z"], h, w, [f"{r}:0" for r in order])
        self._pushed = set()
        if hasattr(self._detail, "register_images_bulk"):
            entries = []
            for well in order:
                w_idx = self._fov_index[well]["idx"]
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
        self._update_loupe_source()                          # back to the acquisition's own pixels
        for rc in list(self._overview._status):
            self._overview.set_status(*rc, "empty")
        self._refresh_layers_tab()
        self._setup_raw_detail()
        # resume the raw thumbnail fill — the operator run stopped the preview partway, so re-run it to
        # finish downsampling every well's raw tile (idempotent: it just re-renders the raw layer).
        self._stop_preview()
        self._preview = _PreviewWorker(self._reader, self._meta, self._fov_index, self._order)
        self._preview.tileReady.connect(self._on_preview_tile)
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
        self._loupe_sources = {}                  # a new plate: no source survives from the old one
        self._acq_name, self._acq_path = base.name, base
        self._processed_plate = str(zroot)
        self._reader = None                       # a computed plate has no raw reader
        self._meta = {"channels": channels, "z_levels": [0], "n_z": 1, "n_t": 1,
                      "pixel_size_um": px_um,
                      "regions": [f"{rows[w['rowIndex']]}{cols[w['columnIndex']]}" for w in wells_meta]}
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
        self._overview.hovered.connect(self._on_hover)
        self._overview.wellActivated.connect(self.activate_well)
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
        self._enable_operators(False)             # no raw data -> operators stay disabled

        if self._detail is not None:
            self._detail.start_acquisition([c["name"] for c in channels], 1, _PUSH_PX, _PUSH_PX,
                                           [f"{w}:0" for w in self._order])
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

        colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in channels])
        coarse_lvl = levels[-1]                                   # coarsest -> tiny thumbnail
        push_lvl = levels[min(3, len(levels) - 1)]                # ~512px level for the detail slider
        self._worker = _ComputedPlateWorker(str(zroot), worker_wells, colors, coarse_lvl, push_lvl,
                                            np.uint16)
        self._worker.tileReady.connect(self._on_tile)
        self._worker.pushReady.connect(self._on_push)
        self._worker.progress.connect(
            lambda i, n: self._readout.setText(f"loading computed plate — {i}/{n} wells"))
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(
            lambda: self._readout.setText(f"✓ computed MIP · {len(self._order)} wells (read-only)"))
        self._readout.setText(f"loading computed plate · {len(self._order)} wells …")
        self._worker.start()

    # -- run a post-processing operator over the whole plate (persists a navigable OME-Zarr plate) --
    def run_operator(self, key: str, out_parent: Optional[str] = None,
                     preview_limit: Optional[int] = None, save: bool = True):
        """Run a projector operator (MIP / reference) over the plate.

        preview_limit=N runs on only the first N wells (a subset) — a cheap way to test an operator.
        save=False is PREVIEW: compute + stream results into the plate + ndviewer slider, writing
        NOTHING to disk (no folder, no disk-space cost). save=True persists a navigable OME-Zarr;
        combined with preview_limit it saves just that subset. Tests pass out_parent to skip the dialog.
        """
        if self._reader is None or self._overview is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._readout.setText("already processing — let the current run finish first")
            return
        label = _OPERATIONS_BY_KEY[key].label
        regions = self._order[:preview_limit] if preview_limit is not None else None
        scope = f"first {len(regions)} wells" if regions is not None else "the whole plate"
        out_dir = est_gb = None
        if save:
            # Ask WHERE to persist: output can be hundreds of GB, so let the user aim it at a roomy
            # disk rather than silently filling the acquisition's. Tests pass out_parent.
            if out_parent is None:
                out_parent = QFileDialog.getExistingDirectory(self, f"Save {label} plate to folder")
                if not out_parent:
                    return
            out_dir = Path(out_parent) / f"{self._acq_name}.hcs"
            ok, est_gb, msg = self._check_disk(out_dir)   # whole-plate estimate; a subset only uses less
            if not ok and regions is None:
                self._readout.setText(msg)
                return
        self._stop_preview()                                 # the operator supersedes the raw preview
        if regions is not None:                              # amber only the wells we'll actually run
            for r in regions:
                self._overview.set_status(*self._fov_index[r]["rc"], "processing")
        else:
            self._overview.set_all_status("processing")      # amber across the plate
        self._plate_mode = label                             # plate now shows this operator's result
        self._plate_title.setText(f"{self._acq_name}   ·   {label}")
        self._active_op_key = key                            # tiles stream into this layer
        if getattr(self, "_raw_btn", None):
            self._raw_btn.show()                             # now there's a processed view to return from
        self._op_stack.add(key, label)                       # push the operator layer onto the stack
        self._overview.set_active_layer(key)                 # show it
        # Loupe source for this run. A SAVED run gets a zarr source whose written-well set grows
        # as wells land (so the loupe works mid-run on what's finished); a PREVIEW writes nothing,
        # so the layer gets no source and the gesture reports that rather than magnifying the
        # previous run's pixels through the same reused layer key.
        if save and out_dir is not None:
            ny, nx = self._meta["frame_shape"]
            fovs = self._meta.get("fovs_per_region")
            self._set_loupe_source(key, _ZarrLoupeSource(
                str(Path(out_dir) / "plate.ome.zarr"),
                path_of=lambda w: "/".join(str(x) for x in parse_well_id(w)),
                fov_of=lambda w: _fov_of_well(w, fovs),
                levels=None,                                 # discovered from the first written field
                well_px=min(ny, nx), pixel_size_um=self._meta.get("pixel_size_um"),
                written=set()))
        else:
            self._drop_loupe_source(key)
        self._refresh_layers_tab()
        # switch the detail to processed mode: z collapsed (nz=1 -> ndv drops the z-slider), frames at
        # the push size, same well order. Each computed well is pushed into the growing slider below.
        if self._detail is not None:
            self._detail.start_acquisition([c["name"] for c in self._meta["channels"]], 1,
                                           _PUSH_PX, _PUSH_PX, [f"{r}:0" for r in self._order])
        self._worker = _OperatorWorker(key, self._reader, self._meta, self._fov_index,
                                       self._overview._nr, self._overview._nc,
                                       str(out_dir) if out_dir else "", regions=regions, save=save)
        dest = f" → {out_dir.name}" if save else " (preview — not saved)"
        self._worker.tileReady.connect(self._on_tile)
        self._worker.pushReady.connect(self._on_push)
        self._worker.progress.connect(
            lambda d, t: self._readout.setText(f"● {label} · {d}/{t} wells{dest}"))
        self._worker.finalReady.connect(self._set_final)
        self._worker.writtenReady.connect(self._on_written)
        self._worker.wellFailed.connect(                     # a skipped well -> red x, run continues
            lambda ri, ci: self._overview.set_status(ri, ci, "failed") if self._overview else None)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(lambda: self._readout.setText(
            f"✓ {label} · {scope}{dest}" + ("  (re-openable OME-Zarr)" if save else "")))
        self._readout.setText(f"● {label} · {scope}{dest} …")
        self._worker.start()

    def _check_disk(self, out_dir) -> tuple[bool, float, str]:
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
        est = int(len(self._fov_index) * m.get("n_t", 1) * len(m["channels"]) * ny * nx
                  * np.dtype(m["dtype"]).itemsize * 1.34)
        gb = 1024 ** 3
        try:
            free = shutil.disk_usage(Path(out_dir).parent).free
        except OSError:
            return True, est / gb, ""      # can't stat the disk — don't block
        if est > free * 0.9:
            return False, est / gb, (f"MIP would persist ~{est/gb:.0f} GB to {Path(out_dir).parent} "
                                     f"but only {free/gb:.0f} GB free — free space or pick another disk.")
        return True, est / gb, ""

    def _on_written(self, plate_path: str):
        """The operator finished persisting: remember the written plate (re-openable artifact)."""
        self._processed_plate = plate_path

    def _on_preview_tile(self, ri, ci, well_id, rgb):
        if self._overview is not None:                       # raw preview fills the base ("raw") layer
            self._overview.add_tile(ri, ci, well_id, rgb, layer="raw")

    def _on_tile(self, ri, ci, well_id, rgb):
        if self._overview is None:
            return
        layer = self._active_op_key or "raw"
        self._overview.add_tile(ri, ci, well_id, rgb, layer=layer)
        self._overview.set_status(ri, ci, "done")           # blue
        src = self._loupe_sources.get(layer)                 # this well is now on disk -> loupe-able
        if isinstance(src, _ZarrLoupeSource):
            src.mark_written(well_id)

    def _on_push(self, fov_idx, planes):
        """A computed well's ~512px channels -> the ndviewer growing slider (in-memory register_array,
        LRU bounded). z collapsed (nz=1). No-op if the detail has no register_array (older ndv / stub)."""
        if self._detail is None or not hasattr(self._detail, "register_array"):
            return
        channels = [c["name"] for c in self._meta["channels"]]
        for c_i, plane in enumerate(planes):
            try:
                self._detail.register_array(0, fov_idx, 0, channels[c_i], plane)
            except Exception:
                pass   # one bad push must not break the run

    def _on_failed(self, msg):
        if self._overview is not None:
            for rc, state in list(self._overview._status.items()):
                if state == "processing":
                    self._overview.set_status(*rc, "failed")  # red x on wells that didn't finish
        self._readout.setText(f"failed: {msg}")

    def _set_final(self, rgb):
        if self._overview is None:
            return
        self._final_arr = np.ascontiguousarray(rgb)
        h, w, _ = self._final_arr.shape
        self._overview.set_final(QImage(self._final_arr.data, w, h, 3 * w, QImage.Format_RGB888),
                                 layer=self._active_op_key or "raw")

    # -- navigation links --
    def _on_hover(self, text: str):
        # BOTTOM-LEFT plate title bar: "<acq>  ·  <mode>" (mode = raw / the operator that processed it),
        # plus the hovered well when the cursor is over the plate.
        base = f"{self._acq_name or 'well plate'}   ·   {self._plate_mode}"
        self._plate_title.setText(f"{base}   ·   {text}" if text else base)

    def activate_well(self, well_id: str, fov_index: int):
        """Double-click -> show the well in the ndviewer. In RAW mode (no operator run yet) push the
        well's raw z-stack lazily (the true z-stack, zero bytes copied). In PROCESSED mode (an operator
        has run, the slider already holds the results) just navigate the slider to that well."""
        if self._detail is None or well_id not in self._fov_index:
            return
        self._current_well = well_id
        if self._overview is not None:                 # current well at view = the red BOX
            self._overview.select(*self._fov_index[well_id]["rc"])
        idx = self._fov_index[well_id]["idx"]
        if self._active_op_key is not None or self._reader is None:   # processed/computed: already pushed
            try:
                row, col = parse_well_id(well_id)
                self._detail.go_to_well_fov(f"{row}{col}", fov_index)
            except Exception:
                pass
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
        row, col = parse_well_id(well_id)
        try:
            self._detail.go_to_well_fov(f"{row}{col}", fov_index)
        except Exception:
            pass

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
        fov = self._meta["fovs_per_region"][well][0]
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
        self._readout.setText(f"{well}: focused on reference plane z={best_z_i} (sharpest)")

    def _retire(self, w):
        """Retire a worker thread WITHOUT ever destroying a running QThread (that aborts the app).
        Disconnect its signals first so a tile already queued before the stop can't paint onto a
        freshly-opened plate (the cross-plate corruption the review found); then keep a reference
        alive until it actually finishes (stop() returns after the current item, which is bounded)."""
        if w is None:
            return
        for name in ("tileReady", "progress", "finalReady", "writtenReady", "wellFailed", "pushReady", "failed", "finished_ok"):
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

    def _stop_worker(self):
        self._retire(self._worker)
        self._worker = None

    def _stop_preview(self):
        self._retire(self._preview)
        self._preview = None

    def closeEvent(self, e):
        self._stop_worker()          # stop the run cleanly; nothing on disk to clean up (no cache)
        self._stop_preview()
        if self._overview is not None:
            self._overview.set_loupe_source(None)   # joins the loupe read thread
        self._loupe_sources = {}
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


def main(dataset_path: str = None):
    path = dataset_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    app = QApplication.instance() or QApplication(sys.argv)
    win = PlateWindow(path)
    _install_footprint_monitor(app, win)
    win.show()
    if not app.property("_squidmip_test"):
        sys.exit(app.exec_())
    return win


if __name__ == "__main__":
    main()
