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

from squidmip import _napari_pane  # noqa: E402
from squidmip._napari_pane import (  # noqa: E402
    SETTLE_MS,
    MosaicPane,
    SettleCoalescer,
    gl_available,
    make_pane,
    max_3d_texture_line,
)
from squidmip._napari_view import _DEFAULT_MAX_3D_TEXTURE  # noqa: E402


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


# ------------------------------------- the GPU 3D texture limit, made knowable


class _Canvas:
    """The vispy canvas as the pane reads it: (2d, 3d) maximum texture sizes."""

    def __init__(self, sizes):
        self.max_texture_sizes = sizes


class _Pane:
    """A stand-in for the parts of MosaicPane ``_live_max_3d_texture`` touches. The method is
    called unbound against it, so this pins the reading rule without a GL context -- which is the
    whole difficulty: offscreen there is no GL context to have."""

    def __init__(self, canvas=None):
        self.canvas = canvas
        self._viewer = None


@pytest.fixture(autouse=True)
def _forget_what_was_already_said(monkeypatch):
    """The announcement is per process, so a previous test's value must not silence this one's."""
    monkeypatch.setattr(_napari_pane, "_MAX_3D_TEXTURE_SAID", None)


def test_the_gpu_reported_limit_is_used_rather_than_the_fallback():
    """Julio asked whether the limit is hardcoded. It is not: a GPU that reports 4096 gets 4096,
    which is FOUR TIMES the native 3D area of the 2048 every design document has assumed."""
    assert MosaicPane._live_max_3d_texture(_Pane(_Canvas((16384, 4096)))) == 4096


def test_with_no_canvas_the_fallback_is_the_one_the_renderer_uses():
    """One number, one owner. A second literal here would drift from the one _napari_view applies
    when it actually picks a level, and the two would disagree silently."""
    assert MosaicPane._live_max_3d_texture(_Pane()) == _DEFAULT_MAX_3D_TEXTURE


def test_the_limit_is_announced_and_says_whether_it_was_measured(caplog):
    """Neither the owner nor an agent can see this number: it needs a GL context, so it cannot be
    probed offscreen, and every figure in the ROI design uses the 2048 fallback on faith. An
    unmeasured number that the interface silently depends on is the thing to fix here."""
    with caplog.at_level("INFO"):
        MosaicPane._live_max_3d_texture(_Pane(_Canvas((16384, 4096))))
    assert "4096" in caplog.text and "read from the GPU" in caplog.text

    caplog.clear()
    _napari_pane._MAX_3D_TEXTURE_SAID = None
    with caplog.at_level("INFO"):
        MosaicPane._live_max_3d_texture(_Pane())
    assert "assuming" in caplog.text, (
        "an assumed limit reads exactly like a measured one, so nobody can tell whether the "
        "figure the design is built on is this machine's or a guess")


def test_the_limit_is_not_announced_again_on_every_toggle(caplog):
    """This is read on every 2D/3D change. Repeating it would bury the log the user is meant to
    read the run's own lines in."""
    pane = _Pane(_Canvas((16384, 2048)))
    with caplog.at_level("INFO"):
        for _ in range(5):
            MosaicPane._live_max_3d_texture(pane)
    assert caplog.text.count("GL_MAX_3D_TEXTURE_SIZE") == 1


def test_a_limit_that_changes_is_announced_again(caplog):
    """The first read can fall back before the canvas has drawn and succeed afterwards. That
    correction is the single line anybody reading this log most needs to see."""
    with caplog.at_level("INFO"):
        MosaicPane._live_max_3d_texture(_Pane())                        # no canvas yet
        MosaicPane._live_max_3d_texture(_Pane(_Canvas((16384, 4096))))  # canvas has drawn
    assert caplog.text.count("GL_MAX_3D_TEXTURE_SIZE") == 2
    assert "4096" in caplog.text


def test_the_sentence_names_what_the_number_costs_the_user():
    """A bare integer in a log is not knowledge. The line has to say what it decides, because the
    person reading it is deciding how big an ROI to draw."""
    line = max_3d_texture_line(2048, measured=True)
    assert "2048" in line and "ROI" in line
