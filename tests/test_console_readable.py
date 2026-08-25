"""The console's compact line must fit the 596 px root; the full Squid line stays byte-identical."""
from __future__ import annotations

import logging
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from squidxplorer._logpane import LOG_FORMAT, format_console, format_record

_ROOT_WIDTH_PX = 596

#: Roughly how many monospace characters fit across the root's width, generously.
_CHARS_THAT_FIT = 78


def _record(msg="[3] A1 fov 2  decon(sigma=2.0)  done in 1.4 s", name="squid.xplorer.viewer"):
    r = logging.LogRecord(name, logging.INFO, "/x/squidxplorer/_viewer.py", 1234, msg, None, None)
    r.funcName = "run_operator"
    return r


def test_the_console_line_fits_where_squids_full_line_does_not():
    assert len(format_record(_record())) > _CHARS_THAT_FIT, "re-examine whether format_console earns its keep"
    assert len(format_console(_record())) <= _CHARS_THAT_FIT


def test_the_console_keeps_what_a_reader_needs():
    line = format_console(_record())
    assert "16:" in line or ":" in line, "no time"
    assert "INFO" in line, "no level"
    assert "viewer" in line, "no attribution: an unattributed log line is a rumour"
    assert "[3] A1 fov 2" in line, "the view id and address prefix were dropped"
    assert "decon(sigma=2.0)" in line, "the message was truncated"
    assert "squid.xplorer." not in line, "every line here has that prefix; it is noise"
    assert "_viewer.py:1234" not in line, "a code pointer is not a user fact"
    assert "2026-" not in line, "the date is not news while you are watching"


def test_a_bad_format_string_still_does_not_kill_the_console():
    """A logging bug must not silence the log."""
    r = logging.LogRecord("squid.xplorer.x", logging.ERROR, "f.py", 1, "%s and %s", ("one",), None)
    line = format_console(r)
    assert "unformattable" in line


def test_the_full_line_is_still_byte_identical_to_squids_layout():
    """The compact console must not weaken what Squid compatibility rests on."""
    r = _record()
    r.thread_id = 4242
    expected = logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S").format(r)
    assert format_record(r) == expected


def test_the_band_opens_below_its_ceiling():
    pytest.importorskip("qtpy")
    import squidxplorer._viewer as V

    assert V._BAND_DEFAULT_PX < V._BAND_MAX_PX, (
        "the band opens at its ceiling, so the cap is sizing it again rather than bounding it")
    assert V._BAND_DEFAULT_PX >= 300, (
        "the band holds Operator AND Log; split two ways, less than this is a five-line log")


def test_no_gui_string_carries_an_em_or_en_dash():
    """Julio (2026-08-24): "there should be no em dashes in GUI"."""
    import ast
    from pathlib import Path

    import squidxplorer

    offenders: list[str] = []
    for path in sorted(Path(squidxplorer.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) \
                        and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docstrings \
                    and ("—" in node.value or "–" in node.value):
                offenders.append(f"{path.name}:{node.lineno} {node.value!r:.80}")
    assert not offenders, (
        "em/en dashes in string literals reach the GUI; use commas, colons, periods or "
        "hyphens instead:\n" + "\n".join(offenders))
