"""The log panel's rules, tested without a window.

Julio: "The logger is super important... this is great for the customers, because it shows them
that the GUI is actually doing something rather than staying idle."

The property that matters is not "text appears". It is that the log CANNOT LIE and CANNOT GROW:
a third-party library we orchestrate shows up without being told about us, a log handler never
breaks the code that called it, and the history is bounded.
"""

from __future__ import annotations

import logging

import pytest

from squidmip._logpane import (
    MAX_LINES,
    LogBus,
    color_for,
    format_record,
)


def _record(msg="hello", level=logging.INFO, name="squidmip", args=()):
    return logging.LogRecord(name=name, level=level, pathname=__file__, lineno=1,
                             msg=msg, args=args, exc_info=None)


@pytest.fixture
def bus():
    b = LogBus()
    yield b
    b.uninstall()


def test_a_line_says_when_what_and_WHO(bus):
    """The logger NAME is kept deliberately: it is what tells the user a line came from
    tilefusion rather than from us. An unattributed log line is a rumour."""
    line = format_record(_record("fusing region manual0", name="tilefusion.optimization"))
    assert "tilefusion" in line
    assert "INFO" in line
    assert "fusing region manual0" in line
    assert line[:2].isdigit() and line[2] == ":", f"no timestamp: {line!r}"


def test_a_THIRD_PARTY_library_appears_without_being_told_about_us(bus):
    """THE design property, and the reason this attaches to the stdlib root logger instead of
    using a signal of our own.

    This application orchestrates other people's libraries - tilefusion, petakit, bgsub, and next
    Cellpose or StarDist. None of them will ever emit OUR signal, and all of them already use
    `logging`. So a library nobody has wired up must still show up.

    MUTATION: install on "squidmip" instead of the root logger and this goes red.
    """
    seen = []
    bus.subscribe(lambda level, line: seen.append(line))
    bus.install()

    logging.getLogger("some_library_we_have_never_heard_of").info("doing something")

    assert any("some_library_we_have_never_heard_of" in ln for ln in seen), (
        f"a third-party library's log never reached the panel: {seen}"
    )


def test_installing_twice_does_not_double_every_line(bus):
    """`_bind_napari_contrast` had exactly this bug shape once. A log panel that duplicates every
    line is worse than none: the user cannot tell one event from two."""
    seen = []
    bus.subscribe(lambda level, line: seen.append(line))
    bus.install()
    bus.install()

    logging.getLogger("squidmip.test").info("once")
    assert len([ln for ln in seen if "once" in ln]) == 1, f"line was duplicated: {seen}"


def test_debug_is_dropped_but_warnings_and_errors_are_not(bus):
    seen = []
    bus.subscribe(lambda level, line: seen.append((level, line)))
    bus.install()

    log = logging.getLogger("squidmip.test")
    log.debug("noise for a terminal")
    log.info("something happened")
    log.warning("something is off")
    log.error("something failed")

    levels = [lvl for lvl, _ in seen]
    assert "DEBUG" not in levels, "DEBUG reached the panel; it is for a terminal, not a demo"
    assert levels.count("INFO") == 1
    assert "WARNING" in levels and "ERROR" in levels


def test_a_broken_log_call_does_NOT_break_OUR_seam(bus):
    """A handler that raises surfaces as a mangled traceback from whatever unrelated code happened
    to be logging. Ours must never be able to break the thing being logged about.

    Driven through `emit_record` rather than through `logging.info(...)` ON PURPOSE. A bad format
    string ("%s and %s" with one argument - the classic accidental logging crash) makes EVERY
    handler attached to the root logger raise, including pytest's own capture handler, so going
    through the global call would test pytest rather than us. What we own is this seam.

    MUTATION: remove the try/except around getMessage in format_record and this goes red.
    """
    seen = []
    bus.subscribe(lambda level, line: seen.append(line))

    bad = _record("value is %s and %s", args=("only-one",))
    bus.emit_record(bad)                      # must not raise
    assert seen and "unformattable" in seen[-1], (
        f"a bad format string was not reported as such: {seen}"
    )

    bus.emit_record(_record("still alive"))
    assert any("still alive" in ln for ln in seen)


def test_a_subscriber_that_raises_does_not_stop_the_others(bus):
    """One broken sink must not silence the panel."""
    good = []
    bus.subscribe(lambda level, line: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.subscribe(lambda level, line: good.append(line))
    bus.install()

    logging.getLogger("squidmip.test").info("reaches the second sink")
    assert any("reaches the second sink" in ln for ln in good), (
        "one raising subscriber silenced the whole panel"
    )


def test_uninstall_stops_delivery(bus):
    """A closed window's panel must stop receiving records, or the handler outlives the widget."""
    seen = []
    bus.subscribe(lambda level, line: seen.append(line))
    bus.install()
    logging.getLogger("squidmip.test").info("before")
    bus.uninstall()
    logging.getLogger("squidmip.test").info("after")

    assert any("before" in ln for ln in seen)
    assert not any("after" in ln for ln in seen), "records kept arriving after uninstall"


def test_the_history_is_BOUNDED(bus):
    """A log that grows without limit is a memory leak with a nice UI.

    A plate run emits a line per well; 1536 wells x several operators is tens of thousands of
    lines. This project's first principle is bounded memory, and a debug panel gets no exemption.
    """
    assert 100 <= MAX_LINES <= 20000, (
        f"MAX_LINES is {MAX_LINES}: either too small to scroll back through a run, or large "
        "enough to be a memory problem of its own"
    )


def test_levels_are_visually_distinct_but_INFO_does_not_shout():
    """A log that shouts at INFO teaches the user to ignore it, and then WARNING and ERROR have
    nowhere left to go."""
    assert color_for("ERROR") != color_for("INFO")
    assert color_for("WARNING") != color_for("INFO")
    assert color_for("CRITICAL") == color_for("ERROR")
    assert color_for("something-unknown") == color_for("INFO")
