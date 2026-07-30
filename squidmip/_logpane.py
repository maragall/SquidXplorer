"""What the application is DOING, in words, at the bottom of the window.

Julio: "what happened to the logger that we were going to add on the bottom right of the GUI,
below the exploration pane... this is great for the customers, because it shows them that the GUI
is actually doing something rather than staying idle." And: "The logger is super important."

WHY PYTHON'S ``logging`` IS THE SOURCE, AND NOT A SIGNAL OF OUR OWN
------------------------------------------------------------------
The obvious design is a ``Signal(str)`` that our code emits into a text box. maragall/stitcher
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

SPEAKING SQUID'S LANGUAGE, NOT ONLY SHARING ITS DATA MODEL (Julio, 2026-07-29)
------------------------------------------------------------------------------
:mod:`squidmip._address` is written under the law that Squid models the physical world and we
model the processing of what it recorded, so the words must agree. That law governs the logger
too, in two places.

**The NAME.** Squid's logging root is ``"squid"`` and every one of its modules is a child of it
(``squid/logging.py:_squid_root_logger_name``). We are a Squid tool, so ``squid.xplorer.<module>``
is the truthful name and it merges for free the day SquidXplorer runs inside Squid. The obvious
objection is that emitting under a ``squid`` root is confusing when Squid is not installed, and v1
is standalone. Weighed, and adopted anyway, for a reason that is not sentiment: Squid documents
``set_stdout_log_level`` as the way "all squid code" controls squid-only logging, and it works by
walking the handlers of the ``squid`` root. A tool that names itself outside that hierarchy cannot
be turned down by it. When Squid is absent, ``logging.getLogger("squid")`` is a bare, handler-less
logger that forwards to the stdlib root exactly as before, so the entire cost of the choice is
thirteen characters in a name field. Use :func:`get_logger`, never ``logging.getLogger`` with a
literal.

**The FORMAT.** :data:`LOG_FORMAT` and :data:`LOG_DATEFORMAT` are copied VERBATIM from
``squid/logging.py``, where they are exported as public aliases with the comment "for use by other
modules". They are copied and not imported: Squid is not a dependency of v1 and must not become
one. **``squid/logging.py`` is the source of truth and this copy must be kept in step with it** —
``test_logpane.py`` pins our rendering against ``logging.Formatter(LOG_FORMAT, LOG_DATEFORMAT)``,
so a drift in the layout is a red test rather than a discovery at the merge. The point of matching
is one console: when v1's viewer replaces Squid's mosaic and multi-channel views, both streams
print in one format and interleave correctly instead of one of them looking like debris.

``thread_id`` is a CUSTOM record field of theirs, injected by a filter on the handler, so a record
that lacks it blows up their formatter. We inject it the same way and from the same call
(``threading.get_native_id()``), so a record of ours can never break a formatter of theirs.

**The address goes inside the message.** Squid's format has no address field, so ``[3] A1 fov 2``
is prepended to ``%(message)s`` by :class:`ViewLog` at emission. That keeps our lines legible
under THEIR formatter rather than requiring ours, and the record still carries ``view_id`` and
``address`` as attributes, so a consumer that wants the objects reads the objects.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

#: Lines retained in the view. ~200 KB of text at typical line lengths — enough to scroll back
#: through a long run, small enough that it can never be the reason we run out of memory.
MAX_LINES = 2000

#: The logger namespaces we attach to. The ROOT logger is deliberate: it is what makes a
#: third-party library appear without being told about us. Note this is the STDLIB root, which is
#: a strict superset of Squid's ``squid`` root, so attaching here shows both streams; attaching to
#: ``squid`` instead would silently drop tilefusion, petakit and everything else we orchestrate.
ROOT = ""

#: Our logger namespace, a child of Squid's. See the docstring for why this and not "squidmip".
XPLORER_ROOT = "squid.xplorer"

#: Levels below this are dropped before they reach the view. DEBUG is for a terminal, not for a
#: scientist watching a demo.
DEFAULT_LEVEL = logging.INFO

# --- Squid's line layout, copied verbatim from squid/logging.py -------------------------------
#
# SOURCE OF TRUTH: Cephla-Lab/Squid, software/squid/logging.py (_baseline_log_format /
# _baseline_log_dateformat, exported there as LOG_FORMAT / LOG_DATEFORMAT "for use by other
# modules"). Copied rather than imported because Squid is not a dependency of v1. KEEP IN STEP:
# test_logpane.py renders a record through logging.Formatter(LOG_FORMAT, LOG_DATEFORMAT) and
# asserts format_record() agrees character for character, so drift fails loudly.
LOG_FORMAT = ("%(asctime)s.%(msecs)03d - %(thread_id)d - %(name)s - %(levelname)s - %(message)s "
              "(%(filename)s:%(lineno)d)")
LOG_DATEFORMAT = "%Y-%m-%d %H:%M:%S"

#: Record attributes carrying the two halves of "what happened to what". The view is a WINDOW (an
#: ordinal, ours); the address is an :class:`squidmip._address.Address` or ``Extent`` (acquisition
#: coordinates, Squid's words). Kept as attributes as well as in the message so a consumer that
#: wants the objects gets the objects rather than parsing prose back out of a string.
VIEW_FIELD = "view_id"
ADDRESS_FIELD = "address"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Our logger for *name*, inside Squid's hierarchy: ``squid.xplorer.<name>``.

    Mirrors ``squid.logging.get_logger`` deliberately, including the ``None`` case returning the
    namespace root. Every module in this package should call this instead of
    ``logging.getLogger("squidmip.something")``; see the naming section of the module docstring.
    """
    root = logging.getLogger(XPLORER_ROOT)
    return root if name is None else root.getChild(str(name))


def thread_id_filter(record: logging.LogRecord) -> bool:
    """Inject ``thread_id``, Squid's custom record field, exactly as ``squid/logging.py`` does.

    It runs as a handler filter, so it executes on the thread that emitted the record and reports
    that thread rather than whichever thread formats it later. Without this a record of ours
    reaching a formatter of theirs raises on ``%(thread_id)d``.
    """
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


#: Squid's layout is about 110 characters wide once the timestamp, thread id, dotted logger name,
#: level, message and `(file:line)` are all present. The root window is 596 logical pixels and its
#: top row is capped at 240, so a full Squid line does not come close to fitting: it wraps two or
#: three times and the console becomes unreadable exactly when you need it.
#:
#: So the CONSOLE gets a compact layout and `format_record` above keeps Squid's byte for byte. That
#: is not a compromise, it is the normal split: the console is for watching, and the full line is
#: what goes to a file and into a bug report, which is where Squid-format compatibility actually
#: matters (one file, both streams, correctly interleaved, when SquidXplorer runs inside Squid).
#:
#: What is dropped and why it is safe to drop ON SCREEN: the date (you are watching now), the
#: thread id (an implementation detail while watching), the `squid.xplorer.` prefix (every line in
#: this console has it), and `(file:line)` (a code pointer, not a user fact). What is KEPT: the
#: time, the level, the view id and address prefix, and the full message. The logger's leaf name is
#: kept because it is what tells you a line came from tilefusion rather than from us, and an
#: unattributed log line is a rumour.
def format_console(record: logging.LogRecord) -> str:
    """One SHORT line for one record, for the on-screen console. See the note above."""
    when = time.strftime("%H:%M:%S", time.localtime(record.created))
    # Strip OUR prefix, never take the leaf. A first attempt took `name.rsplit(".")[-1]`, which
    # turned `squid.xplorer.viewer` into `viewer` (right) and `tilefusion.optimization` into
    # `optimization` (wrong: it dropped the library name, which is the only reason attribution is
    # here at all). test_logpanel.py caught it. The rule is that `squid.xplorer.` is redundant
    # because every line in this console carries it; any OTHER name is news and stays whole.
    name = record.name or "?"
    # XPLORER_ROOT, not ROOT: ROOT is the stdlib root "" on purpose, which is what makes a
    # third-party library appear without being wired, so stripping ROOT would strip nothing.
    leaf = name[len(XPLORER_ROOT) + 1:] if name.startswith(XPLORER_ROOT + ".") else name
    try:
        message = record.getMessage()
    except Exception as exc:                # noqa: BLE001 - a bad format string must not kill the log
        message = f"<unformattable log message: {exc!r}>"
    return f"{when} {record.levelname[:4]:<4} {leaf}: {message}"


def format_record(record: logging.LogRecord) -> str:
    """One line for one record, in Squid's layout, byte for byte.

    The logger NAME is kept in full because it is what tells the user (and us, in a bug report)
    that a line came from tilefusion rather than from us. An unattributed log line is a rumour.

    Built by hand rather than with ``logging.Formatter(LOG_FORMAT).format(record)``, for the
    reason a test already pins: Formatter calls ``record.getMessage()`` internally, so a bad
    format string ("%s and %s" with one argument, the classic accidental logging crash) raises
    from inside the FORMATTER, before any guard we could put around the message. The layout is
    identical; only the failure mode differs.
    """
    when = time.strftime(LOG_DATEFORMAT, time.localtime(record.created))
    # thread_id is Squid's field, injected by our handler's filter. A record built by hand (a test,
    # another framework's bridge) will not have it, so fall back to the stdlib's own thread ident
    # rather than reporting the FORMATTING thread, which would be a lie.
    tid = getattr(record, "thread_id", None)
    if tid is None:
        tid = record.thread or 0
    try:
        message = record.getMessage()
    except Exception as exc:                # noqa: BLE001 - a bad format string must not kill the log
        message = f"<unformattable log record from {record.name}: {exc}>"
    return (f"{when}.{int(record.msecs):03d} - {int(tid)} - {record.name} - {record.levelname} - "
            f"{message} ({record.filename}:{record.lineno})")


class ViewLog(logging.LoggerAdapter):
    """A window's logger: every line it emits carries the view id and the address.

    This is the consumer that PROVES the address model. A per-window logger can lean on implicit
    context ("the user knows which window they are looking at"). One global console, printing
    lines from every window with ROI children and possibly two acquisitions open, cannot: it has
    to say WHAT HAPPENED TO WHAT, and "what" needs a stable address. That is why Task 1 built the
    address and the console together rather than one after the other.

    The prefix is prepended to the MESSAGE, not added as a format field, so the line survives
    Squid's formatter unchanged (their layout has no address field). The record also keeps
    ``view_id`` and ``address`` as attributes, which is what tests assert against: a test that
    matches a formatted string is testing the formatter.
    """

    def __init__(self, logger: logging.Logger, view_id: Any, address: Any = None) -> None:
        super().__init__(logger, {})
        self.view_id = view_id
        self.address = address

    def at(self, address: Any) -> "ViewLog":
        """The same view, addressed more precisely. Cheap: windows re-address per action."""
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
        """The third outcome, and it must exist: an action that starts and then says nothing is
        indistinguishable from one still running."""
        self.warning("%s  failed: %s", action, reason, extra=_extra(address))


def _extra(address: Any) -> dict:
    return {} if address is None else {ADDRESS_FIELD: address}


class _QtBridgeHandler(logging.Handler):
    """ONE handler per logger, however many buses subscribe to it.

    THE DOUBLE-HANDLER HAZARD, and why this is a list rather than a single bus. Two handlers of
    ours on one logger means every line arrives twice, and a console that duplicates every event
    is worse than no console: the user cannot tell one event from two. That is not hypothetical
    here — one :class:`LogBus` is built per root window, and the day a second window exists two
    buses install on the same stdlib root.

    The obvious guard, "refuse the second install", trades a duplicated line for a silent one: the
    second bus's panel would show nothing at all. So the handler is shared instead. It is found by
    ``isinstance`` on the logger's existing handlers, and a bus attaches ITSELF to the one that is
    already there. One handler, one delivery, N sinks.

    Squid's own handler is a separate concern and needs no guard: it sits on the ``squid`` logger
    and writes to stdout, while ours sits on the stdlib root and writes to a widget. A line
    reaching both is one line in each of two different places, which is what "one console shows
    both streams" is supposed to mean.
    """

    def __init__(self) -> None:
        super().__init__()
        self.buses: list = []

    def emit(self, record: logging.LogRecord) -> None:
        # NEVER raise out of a log handler: an exception here surfaces as a mangled traceback
        # from whatever unrelated code happened to be logging.
        for bus in list(self.buses):
            try:
                bus.emit_record(record)
            except Exception:               # noqa: BLE001 - logging must not break the caller
                pass


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
        # The console gets the compact layout; format_record's full Squid line is what a file
        # handler and a bug report want. Subscribers receive BOTH, so a sink that wants the long
        # form is not forced to re-derive it.
        line = format_console(record)
        full = format_record(record)
        for cb in list(self._subscribers):
            # One broken sink must not silence the panel for the others. Caught by a test: a
            # single raising subscriber swallowed every subsequent line.
            try:
                try:
                    cb(record.levelname, line, full)
                except TypeError:
                    cb(record.levelname, line)      # older two-argument sinks still work
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

        logger = logging.getLogger(logger_name)
        # ONE bridge handler per logger, shared by every bus. See _QtBridgeHandler for why this is
        # a share and not a refusal.
        handler = next((h for h in logger.handlers if isinstance(h, _QtBridgeHandler)), None)
        if handler is None:
            handler = _QtBridgeHandler()
            # Squid's custom record field, injected here on the EMITTING thread, exactly as
            # squid/logging.py does it, so a record of ours never breaks a formatter of theirs.
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
        """Detach. Called on window close so a closed window's panel stops receiving records.

        The shared handler outlives us if another bus is still on it; it is removed only when the
        last bus leaves, or a second window closing would silence the first."""
        handler = self._handler
        if handler is None:
            return
        self._handler = None
        buses = getattr(handler, "buses", None)
        if buses is not None and self in buses:
            buses.remove(self)
        if not buses:
            logging.getLogger(logger_name).removeHandler(handler)


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
