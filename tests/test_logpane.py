"""The log panel's rules, tested without a window."""

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
    line = format_record(_record("fusing region manual0", name="tilefusion.optimization"))
    assert "tilefusion" in line
    assert "INFO" in line
    assert "fusing region manual0" in line
    assert line[:4].isdigit() and line[4] == "-", f"no timestamp: {line!r}"


def test_a_THIRD_PARTY_library_appears_without_being_told_about_us(bus):
    seen = []
    bus.subscribe(lambda level, line: seen.append(line))
    bus.install()

    logging.getLogger("some_library_we_have_never_heard_of").info("doing something")

    assert any("some_library_we_have_never_heard_of" in ln for ln in seen), (
        f"a third-party library's log never reached the panel: {seen}"
    )


def test_installing_twice_does_not_double_every_line(bus):
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
    """Driven through emit_record directly: a raising handler via logging.info() would also
    break pytest's own capture handler, testing pytest rather than us."""
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
    good = []
    bus.subscribe(lambda level, line: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.subscribe(lambda level, line: good.append(line))
    bus.install()

    logging.getLogger("squidxplorer.test").info("reaches the second sink")
    assert any("reaches the second sink" in ln for ln in good), (
        "one raising subscriber silenced the whole panel"
    )


def test_uninstall_stops_delivery(bus):
    seen = []
    bus.subscribe(lambda level, line: seen.append(line))
    bus.install()
    logging.getLogger("squidxplorer.test").info("before")
    bus.uninstall()
    logging.getLogger("squidxplorer.test").info("after")

    assert any("before" in ln for ln in seen)
    assert not any("after" in ln for ln in seen), "records kept arriving after uninstall"


def test_the_history_is_BOUNDED(bus):
    assert 100 <= MAX_LINES <= 20000, (
        f"MAX_LINES is {MAX_LINES}: either too small to scroll back through a run, or large "
        "enough to be a memory problem of its own"
    )


def test_levels_are_visually_distinct_but_INFO_does_not_shout():
    assert color_for("ERROR") != color_for("INFO")
    assert color_for("WARNING") != color_for("INFO")
    assert color_for("CRITICAL") == color_for("ERROR")
    assert color_for("something-unknown") == color_for("INFO")


def test_a_LIBRARYS_PRINT_reaches_the_panel(bus):
    """tilefusion reports progress via bare print(), not logging, so the root-logger hook
    alone never sees it; stdout capture is the other half."""
    seen = []
    bus.subscribe(lambda level, line, full=None: seen.append(line))
    bus.install()

    with capture_stdout_to_log():
        print("Parallel registration: 12 pairs in 3 batches")

    assert any("Parallel registration: 12 pairs" in ln for ln in seen), seen
    assert any(f"{STDOUT_LOGGER}:" in ln for ln in seen), seen


def test_capture_survives_the_THREAD_POOL_the_work_actually_runs_on(bus):
    """Run-scoped, not thread-local: the region loop's prints happen on pool threads, not the
    thread that opened the capture."""
    from concurrent.futures import ThreadPoolExecutor

    seen = []
    bus.subscribe(lambda level, line, full=None: seen.append(line))
    bus.install()

    with capture_stdout_to_log():
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda i: print(f"region {i} fused"), range(2)))

    assert sum("fused" in ln for ln in seen) == 2, seen


def test_print_goes_back_to_the_TERMINAL_when_the_run_ends(bus, capsys):
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
    """Capture nests because a saved run wraps a preview's machinery; an inner exit must not
    tear down the outer capture."""
    before = sys.stdout
    with capture_stdout_to_log():
        with capture_stdout_to_log():
            pass
        assert sys.stdout is not before, "the inner exit tore down the outer capture"
    assert sys.stdout is before


def _run_worker_capturing(worker) -> list:
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
    def __init__(self, path):
        self._path = str(path)

    def read(self, region, fov, channel, z_level, time_point=0):
        import numpy as np

        print(f"Parallel registration: {region} fov {fov}")
        return np.zeros((8, 8), dtype=np.uint16)


def test_the_RAW_PREVIEW_captures_print_into_the_log(tmp_path):
    """A raw preview is not an operator run, so it needs its own capture hookup."""
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    import squidxplorer._workers as W

    QApplication.instance() or QApplication([])
    (tmp_path / "acq").mkdir()
    meta = {"channels": [{"name": "c0"}], "dtype": "uint16", "z_levels": [0, 1, 2],
            "regions": ["A1"], "fovs_per_region": {"A1": [0]}, "frame_shape": (8, 8),
            "pixel_size_um": 1.0, "fov_positions_um": {}}
    worker = W._PreviewWorker(_PrintingReader(tmp_path / "acq"), meta,
                              {"A1": {"rc": (0, 0)}}, ["A1"], cache=None)

    seen = _run_worker_capturing(worker)

    assert any("Parallel registration: A1 fov 0" in ln for ln in seen), seen
    assert any(f"{STDOUT_LOGGER}:" in ln for ln in seen), seen


def test_the_capture_is_handed_BACK_after_a_preview_too(tmp_path):
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    import squidxplorer._workers as W

    QApplication.instance() or QApplication([])
    (tmp_path / "acq").mkdir()
    meta = {"channels": [{"name": "c0"}], "dtype": "uint16", "z_levels": [0],
            "regions": ["A1"], "fovs_per_region": {"A1": [0]}, "frame_shape": (8, 8),
            "pixel_size_um": 1.0, "fov_positions_um": {}}
    worker = W._PreviewWorker(_PrintingReader(tmp_path / "acq"), meta,
                              {"A1": {"rc": (0, 0)}}, ["A1"], cache=None)
    before = sys.stdout
    worker.run()
    assert sys.stdout is before, "the preview did not hand sys.stdout back"
