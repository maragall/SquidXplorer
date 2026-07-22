"""The embedded shell — IMA-186's ``squidmip`` CLI, live inside the Process pane.

Two implementations of one widget contract, because the platform decides which is possible:

    _Terminal        pty.fork() + QSocketNotifier   a REAL interactive terminal (Unix)
    _ProcTerminal    QProcess                        works everywhere, including Windows

They share a base (:class:`_ShellPane`) that owns everything neither of them should have an
opinion about: the read-only scrollback with its capped block count, the ``$`` prompt row, the
command line, and the history recall. Before the split existed, the two classes carried a
verbatim copy each of the layout, the stylesheet string, the ``_append`` cursor dance and the
``_send`` history bookkeeping — four facts with two owners apiece, which is the defect shape this
project is named after. The subclasses now hold only what genuinely differs: how bytes get out
(``_write``), how they come back, and how the child dies (``shutdown``).

Lifted out of ``_viewer.py`` unchanged in behaviour. Nothing here knows about the plate, the
reader, or napari — it is a shell in a box, and it was never part of the viewer's job.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from PyQt5.QtCore import QProcess, QProcessEnvironment, QSocketNotifier, Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout, QWidget,
)

from squidmip._qtstyle import ANSI_RE, TERM_INPUT_QSS, TERM_QSS

#: Scrollback cap, in blocks. Output can never grow unbounded.
_MAX_BLOCKS = 4000


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


class _ShellPane(QWidget):
    """The parts of an embedded shell that do not depend on HOW the shell is spawned.

    Owns: the scrollback view, the input row, the command history, and the append/send rules.
    Subclasses implement :meth:`_write` (send a line to the child) and :meth:`shutdown`.
    """

    #: Subclasses set this to False when the transport echoes typed input back on its own (a PTY
    #: does; a pipe does not). It decides whether ``_send`` prints the command itself.
    _transport_echoes = True

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[str] = []
        self._hpos = 0
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._out = QPlainTextEdit()
        self._out.setReadOnly(True)
        self._out.setMaximumBlockCount(_MAX_BLOCKS)
        self._out.setStyleSheet(TERM_QSS)
        v.addWidget(self._out, 1)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(8, 6, 8, 8)
        rl.setSpacing(6)
        tag = QLabel("$")
        tag.setStyleSheet(
            "color:#58a6ff;font-weight:800;font-family:'SF Mono','Menlo',monospace;")
        self._in = _CmdEdit(self)
        self._in.setStyleSheet(TERM_INPUT_QSS)
        self._in.setPlaceholderText("type a command and press Enter  (e.g. squidmip … --tiff)")
        self._in.returnPressed.connect(self._send)
        rl.addWidget(tag)
        rl.addWidget(self._in, 1)
        v.addWidget(row)

    # -- output ------------------------------------------------------------------------
    def _append(self, text: str):
        """Append to the output pane (ANSI escapes + carriage returns stripped), scrolled to end."""
        text = ANSI_RE.sub("", text).replace("\r", "")
        cur = self._out.textCursor()
        cur.movePosition(cur.End)
        cur.insertText(text)
        self._out.setTextCursor(cur)
        self._out.ensureCursorVisible()

    # -- input -------------------------------------------------------------------------
    def _write(self, s: str):   # pragma: no cover - abstract
        raise NotImplementedError

    def _send(self):
        cmd = self._in.text()
        self._in.clear()
        if cmd.strip():
            self._history.append(cmd)
        self._hpos = len(self._history)
        if not self._transport_echoes:
            self._append("> " + cmd + "\n")   # pipes don't echo input, so show it ourselves
        self._write((cmd + "\n") if self._transport_echoes else cmd)

    # -- teardown ----------------------------------------------------------------------
    def shutdown(self):   # pragma: no cover - abstract
        raise NotImplementedError

    def closeEvent(self, e):
        self.shutdown()
        super().closeEvent(e)


class _Terminal(_ShellPane):
    """A real, interactive shell embedded in the Process-wells pane — IMA-186's `squidmip` CLI, live.

    A login shell on a pseudo-terminal (so it echoes input and behaves like a real terminal): type a
    command, press Enter, see its output. `squidmip` is aliased to THIS app's interpreter, so the batch
    MIP command runs here even though the console script isn't pip-installed. Pre-seeded with a how-to
    banner (MIP every well; `--tiff` writes FIJI-openable TIFFs). Scrollback is capped (bounded RAM),
    and the shell is killed when the tab or the window closes (no orphan process).

    PTY-backed, so it needs a Unix-y OS; ``build`` falls back to a static command preview elsewhere.
    """

    _transport_echoes = True

    def __init__(self, cwd: Optional[str], banner: list, setup_cmds: Optional[list] = None,
                 parent=None):
        super().__init__(parent)
        self._pid = None
        self._fd = None
        self._notifier = None
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
        except OSError:
            # A PTY that will not take a window size still works; it just wraps at 80 columns.
            # Narrow to OSError so a genuine bug (bad struct, wrong fd type) is not swallowed.
            self._append("(could not widen the terminal — long lines will wrap)\n")
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

    def _write(self, s: str):
        if self._fd is not None:
            try:
                os.write(self._fd, s.encode())
            except OSError as exc:
                # The child is gone. Say so once, in the pane the user is looking at, instead of
                # letting Enter do nothing forever.
                self._append(f"\n[the shell is not accepting input: {exc}]\n")

    def shutdown(self):
        """Kill the shell (and its group) and release the fd. Idempotent; safe on tab/window close."""
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
                pass                     # already reaped, or never ours — nothing left to do
            self._pid = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass                     # already closed — the point was to release it
            self._fd = None


class _ProcTerminal(_ShellPane):
    """An interactive shell in the pane via QProcess — works on Windows (cmd.exe) AND Unix ($SHELL),
    no PTY needed. Type a command, it runs, output streams back. Not a full VT100 (pipes don't echo,
    so we echo the typed line ourselves), but `squidmip …` and any command work. `squidmip` is aliased
    to this app's interpreter. Used where a PTY is unavailable (i.e. on Windows)."""

    _transport_echoes = False

    def __init__(self, cwd, banner: list, setup_cmds: list, parent=None):
        super().__init__(parent)
        self._nl = "\r\n" if sys.platform == "win32" else "\n"

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
        self._append(data.decode(errors="replace"))

    def _write(self, s: str):
        if self.running():
            self._proc.write((s + self._nl).encode())

    def shutdown(self):
        if self.running():
            self._proc.kill()
            self._proc.waitForFinished(1500)
