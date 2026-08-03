"""The one global console has to be readable in a 596 px portrait root.

2026-07-29. Task 1 made the log a fixed tab in the operators strip, which was right, and it
exposed two things that make a console unreadable rather than merely ugly.

**The line was too long.** Squid's layout, which we copy verbatim so one file can hold both
streams when SquidXplorer runs inside Squid, is about 110 characters once the date, thread id,
dotted logger name, level, message and ``(file:line)`` are all present. The root window is 596
logical pixels wide. A full Squid line wraps two or three times, so ten records fill the strip and
none of them can be read at a glance.

**The strip was too short.** It is capped at 240 px because the plate is the star and an
uncapped strip balloons on the operator cards' size hint. 240 px is roughly ten lines, which is a
status light, not a log.

The fix is split, and the split is the normal one rather than a compromise: the CONSOLE gets a
compact layout and the FULL Squid line is what a file handler and a bug report get, which is where
format compatibility actually matters. Then the strip grows while the Log tab is in front, because
selecting that tab is the user saying they are reading it.

What the compact form drops, and why each is safe to drop ON SCREEN: the date (you are watching
now), the thread id (an implementation detail while watching), the ``squid.xplorer.`` prefix (every
line in this console has it), and ``(file:line)`` (a code pointer, not a user fact). What it keeps:
the time, the level, the logger's LEAF name, and the whole message including the view id and
address prefix. The leaf name stays because it is what tells you a line came from tilefusion rather
than from us, and an unattributed log line is a rumour.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from squidmip._logpane import LOG_FORMAT, format_console, format_record

#: The root window's default width. A console line has to be readable at this width.
_ROOT_WIDTH_PX = 596

#: Roughly how many characters fit across that width in the console's monospace face. Deliberately
#: generous: the point is to catch a line that cannot possibly fit, not to pin a font metric.
_CHARS_THAT_FIT = 78


def _record(msg="[3] A1 fov 2  decon(sigma=2.0)  done in 1.4 s", name="squid.xplorer.viewer"):
    r = logging.LogRecord(name, logging.INFO, "/x/squidmip/_viewer.py", 1234, msg, None, None)
    r.funcName = "run_operator"
    return r


def test_the_full_squid_line_really_is_too_long_for_the_root():
    """The premise. If this ever fails, the compact form may no longer be needed."""
    full = format_record(_record())
    assert len(full) > _CHARS_THAT_FIT, (
        "Squid's line now fits the root window; re-examine whether format_console is still earning "
        "its keep"
    )


def test_the_console_line_fits():
    assert len(format_console(_record())) <= _CHARS_THAT_FIT


def test_the_console_keeps_what_a_reader_needs():
    line = format_console(_record())
    assert "16:" in line or ":" in line, "no time"
    assert "INFO" in line, "no level"
    assert "viewer" in line, "no attribution: an unattributed log line is a rumour"
    assert "[3] A1 fov 2" in line, "the view id and address prefix were dropped"
    assert "decon(sigma=2.0)" in line, "the message was truncated"


def test_the_console_drops_only_what_is_safe_to_drop_on_screen():
    line = format_console(_record())
    assert "squid.xplorer." not in line, "every line here has that prefix; it is noise"
    assert "_viewer.py:1234" not in line, "a code pointer is not a user fact"
    assert "2026-" not in line, "the date is not news while you are watching"


def test_a_bad_format_string_still_does_not_kill_the_console():
    """The same guarantee format_record makes. A logging bug must not silence the log."""
    r = logging.LogRecord("squid.xplorer.x", logging.ERROR, "f.py", 1, "%s and %s", ("one",), None)
    line = format_console(r)
    assert "unformattable" in line


def test_the_full_line_is_still_byte_identical_to_squids_layout():
    """The console being compact must not have weakened the thing Squid compatibility rests on."""
    r = _record()
    r.thread_id = 4242
    expected = logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S").format(r)
    assert format_record(r) == expected


def test_the_reading_height_is_now_the_only_height():
    """REPLACES test_the_strip_grows_while_the_console_is_in_front, 2026-08-03.

    That test pinned ``_sync_top_row_height``: the strip's cap swapped to ``_TOP_ROW_READING_PX``
    (520) while the Log TAB was in front and back to ``_TOP_ROW_COMPACT_PX`` (240) otherwise,
    because 240 px is about ten lines, which is a status light rather than a log.

    The log is no longer a tab (Julio's 2026-08-03 restack: Operator above, Log below, both
    visible), so there is no tab selection left to infer intent from. 520 is now the ONE cap and
    the Operator/Log boundary is a QSplitter handle the user drags. The mechanism is deleted, not
    adapted: its own docstring said it was "deliberately not a remembered setting and not a drag
    handle", and a drag handle is precisely what replaced it.

    The argument this test still has to carry is the SIZING one, which did not go away: a strip
    holding two stacked panels cannot be sized like a strip holding one tab. Splitting 240 px two
    ways is what leaves the log about five lines. Measured offscreen at 596x850 with the cap forced
    back to 240: Operator 122 px, Log 112 px. With 520: Operator 315 px, Log 199 px.
    """
    pytest.importorskip("qtpy")
    import squidmip._viewer as V

    assert not hasattr(V, "_TOP_ROW_READING_PX"), "the second height is back"
    assert not hasattr(V.PlateWindow, "_sync_top_row_height"), "the tab-driven height swap is back"
    assert V._TOP_ROW_COMPACT_PX == 520, (
        "the strip holds Operator AND Log now; 240 px split two ways is a five-line log")
