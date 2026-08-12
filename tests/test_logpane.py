"""The log panel's rules, tested without a window.

Julio: "The logger is super important... this is great for the customers, because it shows them
that the GUI is actually doing something rather than staying idle."

The property that matters is not "text appears". It is that the log CANNOT LIE and CANNOT GROW:
a third-party library we orchestrate shows up without being told about us, a log handler never
breaks the code that called it, and the history is bounded.
"""

from __future__ import annotations

import logging
import sys

import pytest

from squidxplorer._logpane import (
    MAX_LINES,
    STDOUT_LOGGER,
    LogBus,
    capture_stdout_to_log,
    color_for,
    format_record,
)


def _record(msg="hello", level=logging.INFO, name="squidxplorer", args=()):
    return logging.LogRecord(name=name, level=level, pathname=__file__, lineno=1,
                             msg=msg, args=args, exc_info=None)


@pytest.fixture
def bus():
    b = LogBus()
    yield b
    b.uninstall()


def test_a_line_says_when_what_and_WHO(bus):
    """The logger NAME is kept deliberately: it is what tells the user a line came from
    tilefusion rather than from us. An unattributed log line is a rumour.

    Updated 2026-07-29: the timestamp is now Squid's ``YYYY-MM-DD HH:MM:SS.mmm`` rather than a
    bare ``HH:MM:SS``, because the console renders in Squid's layout. The exact layout is pinned
    against ``logging.Formatter(LOG_FORMAT)`` in ``tests/test_address.py``; this test only asks
    that the four facts a line owes the reader are still in it."""
    line = format_record(_record("fusing region manual0", name="tilefusion.optimization"))
    assert "tilefusion" in line
    assert "INFO" in line
    assert "fusing region manual0" in line
    assert line[:4].isdigit() and line[4] == "-", f"no timestamp: {line!r}"


def test_a_THIRD_PARTY_library_appears_without_being_told_about_us(bus):
    """THE design property, and the reason this attaches to the stdlib root logger instead of
    using a signal of our own.

    This application orchestrates other people's libraries - tilefusion, petakit, bgsub, and next
    Cellpose or StarDist. None of them will ever emit OUR signal, and all of them already use
    `logging`. So a library nobody has wired up must still show up.

    MUTATION: install on "squidxplorer" instead of the root logger and this goes red.
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

    logging.getLogger("squidxplorer.test").info("once")
    assert len([ln for ln in seen if "once" in ln]) == 1, f"line was duplicated: {seen}"


def test_debug_is_dropped_but_warnings_and_errors_are_not(bus):
    seen = []
    bus.subscribe(lambda level, line: seen.append((level, line)))
    bus.install()

    log = logging.getLogger("squidxplorer.test")
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

    logging.getLogger("squidxplorer.test").info("reaches the second sink")
    assert any("reaches the second sink" in ln for ln in good), (
        "one raising subscriber silenced the whole panel"
    )


def test_uninstall_stops_delivery(bus):
    """A closed window's panel must stop receiving records, or the handler outlives the widget."""
    seen = []
    bus.subscribe(lambda level, line: seen.append(line))
    bus.install()
    logging.getLogger("squidxplorer.test").info("before")
    bus.uninstall()
    logging.getLogger("squidxplorer.test").info("after")

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


# ---------------------------------------------------------------------------------------
# print() -> logging: the stitcher's own progress lines
# ---------------------------------------------------------------------------------------


def test_a_LIBRARYS_PRINT_reaches_the_panel(bus):
    """The reported defect: "stitcher logger not showing the same logs as maragall/stitcher".

    tilefusion keeps its module loggers for warnings and says what it is DOING with bare print
    (registration.py:274, optimization.py:254, distortion.py:245). The bus is on the root logger,
    so those lines could never reach it. maragall/stitcher's GUI sees them only because it swaps
    sys.stdout (gui/app.py:580); this is our half of the same trade.
    """
    seen = []
    bus.subscribe(lambda level, line, full=None: seen.append(line))
    bus.install()

    with capture_stdout_to_log():
        print("Parallel registration: 12 pairs in 3 batches")

    assert any("Parallel registration: 12 pairs" in ln for ln in seen), seen
    # ATTRIBUTED, and attributed HONESTLY: the line is not ours, so it is not named as ours.
    assert any(f"{STDOUT_LOGGER}:" in ln for ln in seen), seen


def test_capture_survives_the_THREAD_POOL_the_work_actually_runs_on(bus):
    """The reason this is run-scoped and not thread-scoped.

    stitch_plate submits each region to a ThreadPoolExecutor (_stitch.py:863), so every tilefusion
    print happens on a POOL thread rather than on the worker that opened the capture. A
    thread-local switch would have captured exactly nothing, which is the bug this pins.
    """
    from concurrent.futures import ThreadPoolExecutor

    seen = []
    bus.subscribe(lambda level, line, full=None: seen.append(line))
    bus.install()

    with capture_stdout_to_log():
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda i: print(f"region {i} fused"), range(2)))

    assert sum("fused" in ln for ln in seen) == 2, seen


def test_print_goes_back_to_the_TERMINAL_when_the_run_ends(bus, capsys):
    """A GUI capture that outlives its run would silence the CLI for the life of the process."""
    seen = []
    bus.subscribe(lambda level, line, full=None: seen.append(line))
    bus.install()

    before = sys.stdout
    with capture_stdout_to_log():
        print("during the run")
    assert sys.stdout is before, "sys.stdout was not handed back"

    print("after the run")
    assert any("during the run" in ln for ln in seen), seen
    assert not any("after the run" in ln for ln in seen), "capture outlived its run"
    assert "after the run" in capsys.readouterr().out


def test_decoration_and_blank_lines_are_DROPPED(bus):
    """tilefusion frames its sections with rules of '=' (core.py:1162). A bounded 2000-line panel
    spends its budget on facts, and maragall/stitcher's own shim drops exactly these
    (gui/app.py:589-591), so both logs agree on what a line IS."""
    seen = []
    bus.subscribe(lambda level, line, full=None: seen.append(line))
    bus.install()

    with capture_stdout_to_log():
        print("=" * 60)
        print("")
        print("   ")
        print("Fusing tiles...")

    assert sum("Fusing tiles" in ln for ln in seen) == 1
    assert not any("======" in ln for ln in seen), seen


def test_nesting_restores_only_ONCE(bus):
    """_coordinate_region calls stitch_region, and a saved run wraps a preview's machinery, so the
    capture nests. An inner exit that restored the stream would silence the rest of the run."""
    before = sys.stdout
    with capture_stdout_to_log():
        with capture_stdout_to_log():
            pass
        assert sys.stdout is not before, "the inner exit tore down the outer capture"
    assert sys.stdout is before


# --------------------------------------------------------------------------------------
# WHERE THE CAPTURE IS ACTUALLY WIRED, which is the half that was missing.
#
# Julio, 2026-08-03: "I don't believe you're using the same exact algorithm as maragall/stitcher.
# ... The reason that I don't believe you is that when I run a preview it doesn't show may
# standalone stitchers log messages on the master log. This tell me it was a partial integration."
#
# The algorithm was never the problem — `tilefusion` is imported from his own checkout, so it is
# byte-identical code. The WIRING was partial: `capture_stdout_to_log()` was opened inside
# `_OperatorWorker.run` alone, so every path that is not an operator run still printed into a
# terminal nobody watches. These two tests are the guard on that, one per worker, because "wired on
# one of N producers" is exactly the defect being fixed and it cannot be caught by testing the
# context manager in isolation (which the tests above already do thoroughly).
#
# The workers are imported INSIDE each test so this module stays importable without Qt, which is
# the property its docstring claims.
# --------------------------------------------------------------------------------------

def _run_worker_capturing(worker) -> list:
    """Run *worker* in this thread with a bus installed; return the console lines it produced."""
    seen: list = []
    b = LogBus()
    b.subscribe(lambda level, line, full=None: seen.append(line))
    b.install()
    try:
        worker.run()
    finally:
        b.uninstall()
    return seen


class _PrintingReader:
    """A reader that prints while it reads. tilefusion says what it is DOING with bare ``print``
    (registration.py:274, optimization.py:254, distortion.py:245), so this is the honest shape."""

    def __init__(self, path):
        self._path = str(path)

    def read(self, region, fov, channel, z, t=0):
        import numpy as np

        print(f"Parallel registration: {region} fov {fov}")
        return np.zeros((8, 8), dtype=np.uint16)


def test_the_RAW_PREVIEW_captures_print_into_the_log(tmp_path):
    """The reported gap, at the worker that had it.

    A raw preview is not an operator run, so it never entered the capture and everything the reader
    (or anything it imports) printed went to a terminal instead of the panel.
    """
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    import squidxplorer._viewer as V

    QApplication.instance() or QApplication([])
    (tmp_path / "acq").mkdir()
    meta = {"channels": [{"name": "c0"}], "dtype": "uint16", "z_levels": [0, 1, 2],
            "regions": ["A1"], "fovs_per_region": {"A1": [0]}, "frame_shape": (8, 8),
            "pixel_size_um": 1.0, "fov_positions_um": {}}
    worker = V._PreviewWorker(_PrintingReader(tmp_path / "acq"), meta,
                              {"A1": {"rc": (0, 0)}}, ["A1"], cache=None)

    seen = _run_worker_capturing(worker)

    assert any("Parallel registration: A1 fov 0" in ln for ln in seen), seen
    # ATTRIBUTED HONESTLY, the same as the operator run's capture: a line a library printed is not
    # a line we logged, so it is named `stdout` and not `squid.xplorer.*`.
    assert any(f"{STDOUT_LOGGER}:" in ln for ln in seen), seen


def test_the_capture_is_handed_BACK_after_a_preview_too(tmp_path):
    """A capture that outlived the pass would silence the terminal for the life of the process —
    the property `test_print_goes_back_to_the_TERMINAL_when_the_run_ends` pins for the operator
    worker, asserted again here because it is a different `run` method holding the context."""
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    import squidxplorer._viewer as V

    QApplication.instance() or QApplication([])
    (tmp_path / "acq").mkdir()
    meta = {"channels": [{"name": "c0"}], "dtype": "uint16", "z_levels": [0],
            "regions": ["A1"], "fovs_per_region": {"A1": [0]}, "frame_shape": (8, 8),
            "pixel_size_um": 1.0, "fov_positions_um": {}}
    worker = V._PreviewWorker(_PrintingReader(tmp_path / "acq"), meta,
                              {"A1": {"rc": (0, 0)}}, ["A1"], cache=None)
    before = sys.stdout
    worker.run()
    assert sys.stdout is before, "the preview did not hand sys.stdout back"
