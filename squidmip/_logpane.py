"""What the application is DOING, in words, at the bottom of the window.

Julio: "what happened to the logger that we were going to add on the bottom right of the GUI,
below the exploration pane... this is great for the customers, because it shows them that the GUI
is actually doing something rather than staying idle." And: "The logger is super important."

WHY PYTHON'S ``logging`` IS THE SOURCE, AND NOT A SIGNAL OF OUR OWN
------------------------------------------------------------------
The obvious design is a ``pyqtSignal(str)`` that our code emits into a text box. maragall/stitcher
does exactly that and it works — for the code that remembers to emit.

But this application's whole value is ORCHESTRATING OTHER PEOPLE'S LIBRARIES: tilefusion, petakit,
bgsub, and next Cellpose or StarDist. None of them will ever emit our signal. They all already use
``logging``, because every serious Python library does. So the source of truth is the stdlib root
logger, and this module attaches a handler to it. The consequence is the point:

    a library we have never heard of, dropped in as a new operator, appears in the log with no
    work at all.

That is the same property Julio asked for on the data model — "plug and play different algos" —
applied to the thing the user watches.

BOUNDED, BECAUSE EVERYTHING HERE IS BOUNDED
-------------------------------------------
A log that grows without limit is a memory leak with a nice UI. A plate run emits one line per
well; at 1536 wells x several operators that is tens of thousands of lines, each held as a Qt text
block. The view keeps a fixed number of blocks (``MAX_LINES``) and drops the oldest, and the
handler never accumulates anything of its own. This project's first principle is
data-intensiveness with bounded memory; a debug panel does not get an exemption.

THREADS
-------
Operators run on QThreads and log from them. A Qt signal emitted from a non-GUI thread to a
receiver living in the GUI thread is delivered QUEUED by Qt — it lands in the GUI event loop
rather than touching a widget from the worker. That is why the bus is a QObject with a signal
instead of the handler writing to the widget directly: writing a QWidget from a worker thread is
undefined behaviour and crashes at random, which is precisely the class of bug this codebase has
already paid for twice with QThread teardown.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

#: Lines retained in the view. ~200 KB of text at typical line lengths — enough to scroll back
#: through a long run, small enough that it can never be the reason we run out of memory.
MAX_LINES = 2000

#: The logger namespaces we attach to. The ROOT logger is deliberate: it is what makes a
#: third-party library appear without being told about us.
ROOT = ""

#: Levels below this are dropped before they reach the view. DEBUG is for a terminal, not for a
#: scientist watching a demo.
DEFAULT_LEVEL = logging.INFO


def format_record(record: logging.LogRecord) -> str:
    """One line for one record: ``HH:MM:SS  LEVEL  logger: message``.

    The logger NAME is kept because it is what tells the user (and us, in a bug report) that a
    line came from tilefusion rather than from us. An unattributed log line is a rumour.
    """
    # time.strftime over record.created, NOT logging.Formatter(...).format(record): Formatter
    # calls record.getMessage() internally, so a bad format string ("%s and %s" with one
    # argument - the classic accidental logging crash) raises from the TIMESTAMP line, before any
    # guard around the message. Caught by a test.
    when = time.strftime("%H:%M:%S", time.localtime(record.created))
    name = record.name.split(".")[0] if record.name else "squidmip"
    try:
        message = record.getMessage()
    except Exception as exc:                # noqa: BLE001 - a bad format string must not kill the log
        message = f"<unformattable log record from {record.name}: {exc}>"
    return f"{when}  {record.levelname:<7} {name}: {message}"


class LogBus:
    """THE seam between ``logging`` and the widget. Holds no history of its own.

    Not a QObject at import time on purpose — this module must stay importable without Qt so the
    formatting and filtering rules can be tested headless. ``attach`` is what binds it to Qt.
    """

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
        line = format_record(record)
        for cb in list(self._subscribers):
            # One broken sink must not silence the panel for the others. Caught by a test: a
            # single raising subscriber swallowed every subsequent line.
            try:
                cb(record.levelname, line)
            except Exception:               # noqa: BLE001 - a sink's bug is not the log's problem
                pass

    # -- installation ------------------------------------------------------------------
    def install(self, logger_name: str = ROOT) -> logging.Handler:
        """Attach to the stdlib logger. Idempotent: installing twice does not double every line.

        The root logger's own level is RAISED to ours only if it is currently higher, never
        lowered below WARNING silently — turning on DEBUG globally for someone else's library is
        not ours to decide, and it would flood the panel with noise the user cannot act on.
        """
        if self._handler is not None:
            return self._handler

        bus = self

        class _QtBridgeHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                # NEVER raise out of a log handler: an exception here surfaces as a mangled
                # traceback from whatever unrelated code happened to be logging.
                try:
                    bus.emit_record(record)
                except Exception:           # noqa: BLE001 - logging must not break the caller
                    pass

        handler = _QtBridgeHandler()
        handler.setLevel(self.level)
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        if logger.level == logging.NOTSET or logger.level > self.level:
            logger.setLevel(self.level)
        self._handler = handler
        return handler

    def uninstall(self, logger_name: str = ROOT) -> None:
        """Detach. Called on window close so a closed window's panel stops receiving records."""
        if self._handler is None:
            return
        logging.getLogger(logger_name).removeHandler(self._handler)
        self._handler = None


#: How each level is coloured in the view. Muted on purpose: a log that shouts at INFO teaches the
#: user to ignore it, and then WARNING and ERROR have nowhere left to go.
LEVEL_COLORS = {
    "DEBUG": "#6e7681",
    "INFO": "#c3ccd9",
    "WARNING": "#e3b341",
    "ERROR": "#f85149",
    "CRITICAL": "#f85149",
}


def color_for(level_name: str) -> str:
    return LEVEL_COLORS.get(str(level_name).upper(), LEVEL_COLORS["INFO"])
