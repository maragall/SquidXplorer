"""The embedded shell inside the Process pane.

_Terminal (pty.fork + QSocketNotifier, Unix) and _ProcTerminal (QProcess, everywhere)
share the _ShellPane base: scrollback, prompt row, command line, history.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from qtpy.QtCore import QProcess, QProcessEnvironment, QSocketNotifier, Qt
from qtpy.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout, QWidget,
)

from squidxplorer._qtstyle import ANSI_RE, TERM_INPUT_QSS, TERM_QSS

#: Scrollback cap, in blocks.
_MAX_BLOCKS = 4000


class _CmdEdit(QLineEdit):
    """A command input with up/down history recall."""

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
    """The parts of an embedded shell that do not depend on how the shell is spawned."""

    #: False when the transport does not echo typed input (a PTY does; a pipe does not).
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
        self._in.setPlaceholderText("type a command and press Enter  (e.g. squidxplorer … --tiff)")
        self._in.returnPressed.connect(self._send)
        rl.addWidget(tag)
        rl.addWidget(self._in, 1)
        v.addWidget(row)

    def _append(self, text: str):
        """Append to the output pane (ANSI escapes + carriage returns stripped), scrolled to end."""
        text = ANSI_RE.sub("", text).replace("\r", "")
        cur = self._out.textCursor()
        cur.movePosition(cur.End)
        cur.insertText(text)
        self._out.setTextCursor(cur)
        self._out.ensureCursorVisible()

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

    def shutdown(self):   # pragma: no cover - abstract
        raise NotImplementedError

    def closeEvent(self, e):
        self.shutdown()
        super().closeEvent(e)


class _Terminal(_ShellPane):
    """A real interactive shell on a pseudo-terminal (Unix only)."""

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
        env["TERM"] = "dumb"        # minimise escape sequences
        env["PS1"] = "$ "
        # venv's Scripts/bin on PATH so the `squidxplorer` console script resolves directly
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
        try:
            self._pid, self._fd = pty.fork()
        except Exception as e:      # no PTY (e.g. Windows) — degrade to a disabled pane
            self._out.setPlainText(f"(embedded terminal unavailable on this platform: {e})")
            self._in.setEnabled(False)
            return
        if self._pid == 0:          # child becomes the shell
            try:
                if cwd and os.path.isdir(cwd):
                    os.chdir(cwd)
                os.execvpe(shell, [shell, "-i"], env)
            except Exception:
                os._exit(127)
        import fcntl                # parent: read the master fd non-blocking via Qt's event loop
        import struct
        import termios
        try:                        # a wide PTY so long commands don't wrap
            fcntl.ioctl(self._fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 400, 0, 0))
        except OSError:
            self._append("(could not widen the terminal — long lines will wrap)\n")
        fl = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        self._notifier = QSocketNotifier(self._fd, QSocketNotifier.Read, self)
        self._notifier.activated.connect(self._read)
        # banner is display text, printed straight into the pane; setup_cmds run silently
        self._append("\n".join(banner) + "\n")
        for cmd in setup_cmds:
            self._write(cmd + "\n")

    def _read(self):
        try:
            data = os.read(self._fd, 8192)
        except BlockingIOError:
            return                       # no data ready yet — keep listening
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
                self._append(f"\n[the shell is not accepting input: {exc}]\n")

    def shutdown(self):
        """Kill the shell (and its group) and release the fd. Idempotent."""
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
                pass                     # already reaped
            self._pid = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass                     # already closed
            self._fd = None


class _ProcTerminal(_ShellPane):
    """An interactive shell via QProcess — works on Windows (cmd.exe) and Unix ($SHELL), no PTY."""

    _transport_echoes = False

    def __init__(self, cwd, banner: list, setup_cmds: list, parent=None):
        super().__init__(parent)
        self._nl = "\r\n" if sys.platform == "win32" else "\n"

        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyRead.connect(self._read)
        self._proc.finished.connect(lambda *a: self._append("\n[shell exited]\n"))
        # venv's Scripts/bin on PATH so the `squidxplorer` console script resolves directly
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PATH", os.path.dirname(sys.executable) + os.pathsep + env.value("PATH"))
        self._proc.setProcessEnvironment(env)
        if cwd and os.path.isdir(cwd):
            self._proc.setWorkingDirectory(cwd)
        shell = "cmd.exe" if sys.platform == "win32" else os.environ.get("SHELL", "/bin/sh")
        self._proc.start(shell, [])
        self._proc.waitForStarted(3000)
        self._append("\n".join(banner) + "\n")
        for c in setup_cmds:            # run silently
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
