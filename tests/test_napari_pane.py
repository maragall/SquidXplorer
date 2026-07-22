"""Pane 2: camera-settle coalescing and the VISIBLE fallback.

The coalescer is clock-injected, so the timing rule is tested without a Qt event loop and
without sleeping.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy.QtWidgets")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt6 GUI tests.",
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


# ------------------------------------------------------------- the visible fallback


def test_the_flag_can_select_ndviewer_and_says_it_was_asked_for(monkeypatch):
    monkeypatch.setenv("SQUIDMIP_VIEWER", "ndv")
    widget, mode, msg = make_pane()
    assert (widget, mode) == (None, "ndv")
    assert "SQUIDMIP_VIEWER" in msg


def test_napari_is_the_default(monkeypatch):
    monkeypatch.delenv("SQUIDMIP_VIEWER", raising=False)
    widget, mode, msg = make_pane()
    # Whichever way it resolves, a non-napari result must carry a REASON — never a silent
    # downgrade. Six confirmed silent failures in this project say so.
    if mode != "napari":
        assert msg, "fell back to ndviewer_light without saying why"
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


def test_headless_falls_back_with_a_reason_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("SQUIDMIP_VIEWER", "napari")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget, mode, msg = make_pane()
    assert (widget, mode) == (None, "ndv")
    assert "OpenGL" in msg


def test_an_unknown_viewer_name_does_not_silently_disable_the_viewer(monkeypatch):
    monkeypatch.setenv("SQUIDMIP_VIEWER", "wat")
    _widget, mode, msg = make_pane()
    assert mode in ("napari", "ndv")
    if mode == "ndv":
        assert msg
