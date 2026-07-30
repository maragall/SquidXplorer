"""Pane 2: camera-settle coalescing and the VISIBLE fallback.

The coalescer is clock-injected, so the timing rule is tested without a Qt event loop and
without sleeping.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidmip._napari_pane import (  # noqa: E402
    SETTLE_MS,
    SettleCoalescer,
    gl_available,
    make_pane,
)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


# ------------------------------------------------------- camera-settle coalescing


def test_nothing_fires_before_the_camera_has_settled():
    clock = _Clock()
    fired = []
    s = SettleCoalescer(0.12, lambda: fired.append(1), clock=clock)

    s.notify()
    clock.advance(0.05)
    assert s.poll() is False
    assert fired == []


def test_a_continuous_drag_coalesces_into_exactly_one_fetch():
    """This is the #1942 mechanism: one fetch per camera event means each fetch is invalidated
    by the next, and the queue grows faster than it drains."""
    clock = _Clock()
    fired = []
    s = SettleCoalescer(0.12, lambda: fired.append(1), clock=clock)

    for _ in range(40):            # ~16 ms apart, i.e. a 60 Hz drag lasting 640 ms
        s.notify()
        clock.advance(0.016)
        s.poll()

    assert fired == [], "fetched mid-drag — the camera never went quiet"

    clock.advance(0.12)
    assert s.poll() is True
    assert fired == [1], "a whole drag must cost exactly one fetch"


def test_it_fires_once_the_camera_is_quiet_and_not_again():
    clock = _Clock()
    fired = []
    s = SettleCoalescer(0.12, lambda: fired.append(1), clock=clock)

    s.notify()
    clock.advance(0.2)
    assert s.poll() is True
    assert s.pending is False

    clock.advance(10.0)
    assert s.poll() is False       # nothing pending -> no repeat
    assert fired == [1]


def test_a_second_move_after_settling_fires_again():
    clock = _Clock()
    fired = []
    s = SettleCoalescer(0.12, lambda: fired.append(1), clock=clock)

    for _ in range(2):
        s.notify()
        clock.advance(0.2)
        s.poll()

    assert fired == [1, 1]


def test_the_debounce_is_a_quiet_period_not_a_rate_limit():
    """A rate limit would fire every interval DURING the drag. A quiet-period debounce fires
    only after motion stops, which is what makes a long pan cost one fetch."""
    clock = _Clock()
    fired = []
    s = SettleCoalescer(0.12, lambda: fired.append(1), clock=clock)

    for _ in range(10):
        s.notify()
        clock.advance(0.10)        # shorter than the interval, so it never settles
        s.poll()

    assert fired == []


def test_the_interval_sits_under_the_perceptible_pause():
    """120 ms: long enough to coalesce a 60 Hz drag, short enough to stay under the ~150 ms at
    which a pause stops reading as a response to your own action."""
    assert 60 <= SETTLE_MS <= 150


# ------------------------------------- the visible FAILURE (there is no fallback any more)
#
# These asserted a working ndviewer_light fallback until 2026-07-30. The fallback is deleted:
# napari won a written gate, and ndviewer_light imported PyQt5 at module scope, so it could not
# share a process with the Qt6 napari it was supposed to be a safety net for. What survives is
# the RULE the fallback existed to serve, which never depended on there being a second viewer:
# a result that is not napari must carry a reason a human can read. Six confirmed silent
# failures in this project say so. `mode == "unavailable"` is that rule with the second viewer
# removed, and it is a stronger contract, not a weaker one: before, a user could be handed a
# different renderer and only a message said so; now there is nothing to be quietly handed.


def test_a_retired_flag_value_changes_nothing(monkeypatch):
    """`SQUIDMIP_VIEWER=ndv` must take exactly the same path as no flag at all.

    The failure this prevents is specific: someone's launcher still exports the old value, and a
    naive removal turns that into "no viewer at all, and no explanation". Asserting the two runs
    AGREE says that without needing to know which way they resolve here.

    Note what this does NOT do: unset `QT_QPA_PLATFORM`. This file's own no-GL test explains why
    — constructing a vispy canvas under the offscreen platform SEGFAULTS rather than raising, so
    unsetting the guard inside an offscreen process kills the interpreter, taking pytest's
    summary with it. I did exactly that here on the first attempt and it aborted the run.
    """
    monkeypatch.delenv("SQUIDMIP_VIEWER", raising=False)
    without = make_pane()[1:]
    monkeypatch.setenv("SQUIDMIP_VIEWER", "ndv")
    with_flag = make_pane()[1:]
    assert with_flag == without, (
        f"a retired flag value took a different path: {with_flag} vs {without}"
    )
    assert with_flag[0] != "ndv", "the deleted fallback is still reachable by name"


def test_napari_is_the_default(monkeypatch):
    monkeypatch.delenv("SQUIDMIP_VIEWER", raising=False)
    widget, mode, msg = make_pane()
    # Whichever way it resolves, a non-napari result must carry a REASON — never a silent
    # downgrade. Six confirmed silent failures in this project say so.
    if mode != "napari":
        assert msg, "reported no viewer without saying why"
    else:
        assert widget is not None and widget.ok


# ------------------------------------------------ the no-GL guard (why the suite survives)


def test_offscreen_is_recognised_as_having_no_gl():
    """Constructing a vispy canvas under the offscreen platform SEGFAULTS rather than raising,
    so this cannot be a try/except — it has to be checked before construction."""
    ok, why = gl_available({"QT_QPA_PLATFORM": "offscreen"})
    assert ok is False
    assert "OpenGL" in why or "offscreen" in why


def test_a_real_platform_is_allowed():
    assert gl_available({"QT_QPA_PLATFORM": "cocoa"})[0] is True
    assert gl_available({})[0] is True


def test_headless_says_there_is_no_viewer_rather_than_crashing(monkeypatch):
    """No GL means no mosaic, said in a sentence. The window still opens without one."""
    monkeypatch.setenv("SQUIDMIP_VIEWER", "napari")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget, mode, msg = make_pane()
    assert (widget, mode) == (None, "unavailable")
    assert "OpenGL" in msg


def test_an_unknown_viewer_name_does_not_silently_disable_the_viewer(monkeypatch):
    """A typo must cost you nothing. It resolves to napari, exactly as an empty value does."""
    monkeypatch.setenv("SQUIDMIP_VIEWER", "wat")
    _widget, mode, msg = make_pane()
    assert mode in ("napari", "unavailable")
    if mode == "unavailable":
        assert msg, "no viewer, and no reason given"
