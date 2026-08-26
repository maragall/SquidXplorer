"""The GUI's log console: a bridge from stdlib ``logging`` (and captured stdout) to the window.

The stdlib root logger is the source so third-party libraries appear with no wiring;
``capture_stdout_to_log`` covers libraries that print instead. Names and line format
match Squid's ``squid/logging.py`` so the two streams interleave in one console.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from typing import Any, Iterator, Optional

#: Lines retained in the view.
MAX_LINES = 2000

#: The stdlib root logger: what makes a third-party library appear without being wired.
ROOT = ""

#: Our logger namespace, a child of Squid's.
XPLORER_ROOT = "squid.xplorer"

DEFAULT_LEVEL = logging.INFO

# Squid's line layout, copied verbatim from squid/logging.py (not imported: Squid is not a
# dependency). test_logpane.py pins this copy against logging.Formatter, so drift fails loudly.
LOG_FORMAT = ("%(asctime)s.%(msecs)03d - %(thread_id)d - %(name)s - %(levelname)s - %(message)s "
              "(%(filename)s:%(lineno)d)")
LOG_DATEFORMAT = "%Y-%m-%d %H:%M:%S"

#: Record attributes carrying "what happened to what" as objects, not prose.
VIEW_FIELD = "view_id"
ADDRESS_FIELD = "address"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Our logger for *name*, inside Squid's hierarchy: ``squid.xplorer.<name>``."""
    root = logging.getLogger(XPLORER_ROOT)
    return root if name is None else root.getChild(str(name))


def thread_id_filter(record: logging.LogRecord) -> bool:
    """Inject ``thread_id``, Squid's custom record field, on the emitting thread."""
    record.thread_id = threading.get_native_id()
    return True


def _address_prefix(view_id: Any = None, address: Any = None) -> str:
    """``"[3] A1 fov 2  "``, or ``""``. The bracket is the VIEW, the rest is the ADDRESS."""
    parts = []
    if view_id is not None:
        parts.append(f"[{view_id}]")
    if address is not None:
        label = address.label() if hasattr(address, "label") else str(address)
        if label:
            parts.append(label)
    return (" ".join(parts) + "  ") if parts else ""


def format_console(record: logging.LogRecord) -> str:
    """One short line for one record, for the on-screen console (Squid's full line does not fit)."""
    when = time.strftime("%H:%M:%S", time.localtime(record.created))
    # Strip only OUR prefix; any other logger name is news and stays whole.
    name = record.name or "?"
    leaf = name[len(XPLORER_ROOT) + 1:] if name.startswith(XPLORER_ROOT + ".") else name
    try:
        message = record.getMessage()
    except Exception as exc:                # noqa: BLE001 - a bad format string must not kill the log
        message = f"<unformattable log message: {exc!r}>"
    return f"{when} {record.levelname[:4]:<4} {leaf}: {message}"


def format_record(record: logging.LogRecord) -> str:
    """One line for one record, in Squid's layout, byte for byte.

    Built by hand rather than with ``logging.Formatter`` so a bad message format
    string is guarded here instead of raising inside the formatter.
    """
    when = time.strftime(LOG_DATEFORMAT, time.localtime(record.created))
    # Records built by hand lack thread_id; fall back to the record's own thread ident.
    tid = getattr(record, "thread_id", None)
    if tid is None:
        tid = record.thread or 0
    try:
        message = record.getMessage()
    except Exception as exc:                # noqa: BLE001 - a bad format string must not kill the log
        message = f"<unformattable log record from {record.name}: {exc}>"
    return (f"{when}.{int(record.msecs):03d} - {int(tid)} - {record.name} - {record.levelname} - "
            f"{message} ({record.filename}:{record.lineno})")


#: A message carrying one of these reads as a refusal or a failure.
STATUS_WARN_MARKS = (
    "fail", "could not", "cannot", "error", "not an .hcs", "no plate",
    "already processing", "empty selection", "open an acquisition", "open a view",
    "pick wells first", "not in the current region order", "nowhere to put",
    "no region is open", "no acquisition open", "stranded",
)


def status_level(text: str) -> int:
    """WARNING for refusal-shaped status text, INFO otherwise - the one classification."""
    low = str(text).lower()
    return logging.WARNING if any(m in low for m in STATUS_WARN_MARKS) else logging.INFO


class StatusReadout:
    """A status surface that IS a log line (2026-08-19, plate; 2026-08-25, every banner).

    Julio: "log messages that show around the GUI and not in the log" was the plate
    complaint; "I don't like the red strip that appears above the window" retired the view
    banner the same way. Every ``setText`` lands in the console (refusal-shaped sentences
    at WARNING, status at INFO; a repeat of the current text is dropped so idempotent
    writers do not spam), and ``text()`` stays the seam tools/gates and tests assert on.
    """

    def __init__(self, logger) -> None:
        self._log = logger
        self._text = ""

    def setText(self, text) -> None:                 # noqa: N802 - keeps the QLabel spelling
        text = str(text or "")
        if text == self._text:
            return
        self._text = text
        if not text:
            return
        self._log.log(status_level(text), "%s", text)

    def text(self) -> str:
        return self._text


class ViewLog(logging.LoggerAdapter):
    """A window's logger: every line it emits carries the view id and the address.

    The prefix goes inside the message so the line survives Squid's formatter; the
    record also keeps ``view_id`` and ``address`` as attributes.
    """

    def __init__(self, logger: logging.Logger, view_id: Any, address: Any = None) -> None:
        super().__init__(logger, {})
        self.view_id = view_id
        self.address = address

    def at(self, address: Any) -> "ViewLog":
        """The same view, addressed more precisely."""
        return ViewLog(self.logger, self.view_id, address)

    def process(self, msg, kwargs):
        extra = dict(kwargs.get("extra") or {})
        # setdefault, so a per-call address beats the adapter's standing one.
        if self.view_id is not None:
            extra.setdefault(VIEW_FIELD, self.view_id)
        if self.address is not None:
            extra.setdefault(ADDRESS_FIELD, self.address)
        kwargs["extra"] = extra
        prefix = _address_prefix(extra.get(VIEW_FIELD), extra.get(ADDRESS_FIELD))
        return f"{prefix}{msg}", kwargs

    # -- the two lines an action produces ----------------------------------------------
    def started(self, action: str, *, address: Any = None) -> None:
        """``[3] A1 fov 2  decon(sigma=2.0)  started``"""
        self.info("%s  started", action, extra=_extra(address))

    def done(self, action: str, seconds: float, *, address: Any = None) -> None:
        """``[3] A1 fov 2  decon(sigma=2.0)  done in 1.4 s``"""
        self.info("%s  done in %.1f s", action, float(seconds), extra=_extra(address))

    def failed(self, action: str, reason: str, *, address: Any = None) -> None:
        """``[3] A1 fov 2  decon(sigma=2.0)  failed: ...``"""
        self.warning("%s  failed: %s", action, reason, extra=_extra(address))


def _extra(address: Any) -> dict:
    return {} if address is None else {ADDRESS_FIELD: address}


# print() -> logging.

#: Logger name carrying captured ``print`` output; not ours, so not under XPLORER_ROOT.
STDOUT_LOGGER = "stdout"

#: Line prefixes that carry no information (tilefusion's "=" / "-" section rules).
_DECORATION_PREFIXES = ("=", "-")

#: Per-thread partial line; the buffer is thread-local so concurrent regions cannot splice half-lines.
_capture_state = threading.local()

#: Guards the ``sys.stdout`` swap, its refcount, and the active logger across threads.
_install_lock = threading.Lock()
_install_count = 0
_capture_logger: Optional[logging.Logger] = None


class _StdoutToLog:
    """The ``sys.stdout`` stand-in: a logger while a run is in flight, a passthrough otherwise.

    ``print`` calls ``write`` once per fragment and again for the newline, so writes are buffered
    per thread and emitted only on a completed line — otherwise one print becomes two log records,
    the second of which is empty.
    """

    def __init__(self, stream) -> None:
        self._stream = stream

    #: The real stream this wraps. Named so :func:`capture_stdout_to_log` can tell "already
    #: installed" from "someone else replaced sys.stdout while we were running".
    @property
    def wrapped(self):
        return self._stream

    def write(self, s) -> int:
        logger = _capture_logger
        if logger is None:
            return self._stream.write(s)
        text = getattr(_capture_state, "buf", "") + str(s)
        lines = text.split("\n")
        _capture_state.buf = lines.pop()            # the tail has no newline yet
        for line in lines:
            line = line.strip()
            if not line or line.startswith(_DECORATION_PREFIXES):
                continue
            # "%s" with an argument: a captured line with a stray "%d" must not act as a format spec.
            logger.info("%s", line)
        return len(str(s))

    def flush(self) -> None:
        # Never flush a partial captured line into the log; it joins the next write.
        if _capture_logger is None:
            self._stream.flush()

    def __getattr__(self, name):
        # isatty/encoding/fileno/… belong to the real stream.
        return getattr(self._stream, name)


@contextlib.contextmanager
def capture_stdout_to_log(logger_name: str = STDOUT_LOGGER) -> Iterator[None]:
    """While this is open, ``print`` becomes an INFO record on *logger_name*.

    Process-scoped, because pooled work leaves the calling thread; nesting is counted, not stacked.
    """
    global _install_count, _capture_logger

    with _install_lock:
        if not isinstance(sys.stdout, _StdoutToLog):
            sys.stdout = _StdoutToLog(sys.stdout)
        if _install_count == 0:
            _capture_logger = logging.getLogger(logger_name)
        _install_count += 1
    try:
        yield
    finally:
        with _install_lock:
            _install_count -= 1
            # Only the last capture restores, and only if sys.stdout is still ours (capsys/napari swap it too).
            if _install_count <= 0:
                _install_count = 0
                _capture_logger = None
                if isinstance(sys.stdout, _StdoutToLog):
                    sys.stdout = sys.stdout.wrapped


class _QtBridgeHandler(logging.Handler):
    """One handler per logger, shared by every bus, so a line is delivered once to N sinks."""

    def __init__(self) -> None:
        super().__init__()
        self.buses: list = []

    def emit(self, record: logging.LogRecord) -> None:
        # Never raise out of a log handler.
        for bus in list(self.buses):
            try:
                bus.emit_record(record)
            except Exception:               # noqa: BLE001 - logging must not break the caller
                pass


class LogBus:
    """The seam between ``logging`` and the widget; holds no history of its own. Qt-free on purpose."""

    def __init__(self, level: int = DEFAULT_LEVEL) -> None:
        self.level = int(level)
        self._subscribers: list = []
        self._handler: Optional[logging.Handler] = None

    def subscribe(self, callback) -> None:
        """``callback(level_name, formatted_line)``, called for every record that passes."""
        self._subscribers.append(callback)

    def emit_record(self, record: logging.LogRecord) -> None:
        if record.levelno < self.level:
            return
        # Subscribers receive both the compact console line and the full Squid line.
        line = format_console(record)
        full = format_record(record)
        for cb in list(self._subscribers):
            # One broken sink must not silence the panel for the others.
            try:
                try:
                    cb(record.levelname, line, full)
                except TypeError:
                    cb(record.levelname, line)      # older two-argument sinks still work
            except Exception:               # noqa: BLE001 - a sink's bug is not the log's problem
                pass

    # -- installation ------------------------------------------------------------------
    def install(self, logger_name: str = ROOT) -> logging.Handler:
        """Attach to the stdlib logger. Idempotent: installing twice does not double every line."""
        if self._handler is not None:
            return self._handler

        logger = logging.getLogger(logger_name)
        handler = next((h for h in logger.handlers if isinstance(h, _QtBridgeHandler)), None)
        if handler is None:
            handler = _QtBridgeHandler()
            handler.addFilter(thread_id_filter)
            handler.setLevel(self.level)
            logger.addHandler(handler)
        elif self.level < handler.level:
            # A second bus wanting MORE than the first must not be starved by the shared handler.
            handler.setLevel(self.level)
        if self not in handler.buses:
            handler.buses.append(self)
        if logger.level == logging.NOTSET or logger.level > self.level:
            logger.setLevel(self.level)
        self._handler = handler
        return handler

    def uninstall(self, logger_name: str = ROOT) -> None:
        """Detach; the shared handler is removed only when the last bus leaves."""
        handler = self._handler
        if handler is None:
            return
        self._handler = None
        buses = getattr(handler, "buses", None)
        if buses is not None and self in buses:
            buses.remove(self)
        if not buses:
            logging.getLogger(logger_name).removeHandler(handler)


#: How each level is coloured in the view; muted on purpose.
LEVEL_COLORS = {
    "DEBUG": "#6e7681",
    "INFO": "#c3ccd9",
    "WARNING": "#e3b341",
    "ERROR": "#f85149",
    "CRITICAL": "#f85149",
}


def color_for(level_name: str) -> str:
    return LEVEL_COLORS.get(str(level_name).upper(), LEVEL_COLORS["INFO"])
