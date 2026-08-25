"""Pane 2: camera-settle coalescing and the VISIBLE fallback."""

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

from squidxplorer import _napari_pane  # noqa: E402
from squidxplorer._napari_pane import (  # noqa: E402
    SETTLE_MS,
    MosaicPane,
    SettleCoalescer,
    gl_available,
    make_pane,
    max_3d_texture_line,
)
from squidxplorer._napari_view import _DEFAULT_MAX_3D_TEXTURE  # noqa: E402


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def test_a_continuous_drag_coalesces_into_exactly_one_fetch():
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
    assert s.poll() is False
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


def test_the_interval_sits_under_the_perceptible_pause():
    assert 60 <= SETTLE_MS <= 150


# The ndviewer_light fallback was deleted (it imported PyQt5 at module scope, incompatible with
# Qt6 napari). What survives: a non-napari result must always carry a human-readable reason.


def test_a_retired_flag_value_changes_nothing(monkeypatch):
    """`SQUIDXPLORER_VIEWER=ndv` must take exactly the same path as no flag at all."""
    monkeypatch.delenv("SQUIDXPLORER_VIEWER", raising=False)
    without = make_pane()[1:]
    monkeypatch.setenv("SQUIDXPLORER_VIEWER", "ndv")
    with_flag = make_pane()[1:]
    assert with_flag == without, (
        f"a retired flag value took a different path: {with_flag} vs {without}"
    )
    assert with_flag[0] != "ndv", "the deleted fallback is still reachable by name"


def test_napari_is_the_default(monkeypatch):
    monkeypatch.delenv("SQUIDXPLORER_VIEWER", raising=False)
    widget, mode, msg = make_pane()
    if mode != "napari":
        assert msg, "reported no viewer without saying why"
    else:
        assert widget is not None and widget.ok


def test_offscreen_is_recognised_as_having_no_gl():
    """Constructing a vispy canvas under the offscreen platform SEGFAULTS rather than raising, so this cannot be a try/except — it has to be checked before"""
    ok, why = gl_available({"QT_QPA_PLATFORM": "offscreen"})
    assert ok is False
    assert "OpenGL" in why or "offscreen" in why


def test_a_real_platform_is_allowed():
    assert gl_available({"QT_QPA_PLATFORM": "cocoa"})[0] is True
    assert gl_available({})[0] is True


def test_headless_says_there_is_no_viewer_rather_than_crashing(monkeypatch):
    """No GL means no mosaic, said in a sentence. The window still opens without one."""
    monkeypatch.setenv("SQUIDXPLORER_VIEWER", "napari")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget, mode, msg = make_pane()
    assert (widget, mode) == (None, "unavailable")
    assert "OpenGL" in msg


class _Canvas:
    """The vispy canvas as the pane reads it: (2d, 3d) maximum texture sizes."""

    def __init__(self, sizes):
        self.max_texture_sizes = sizes


class _Pane:
    """Stand-in for the parts of MosaicPane `_live_max_3d_texture` touches; called unbound against it, since offscreen there is no GL context to construct one."""

    def __init__(self, canvas=None):
        self.canvas = canvas
        self._viewer = None


@pytest.fixture(autouse=True)
def _forget_what_was_already_said(monkeypatch):
    """The announcement is per process, so a previous test's value must not silence this one's."""
    monkeypatch.setattr(_napari_pane, "_MAX_3D_TEXTURE_SAID", None)


def test_the_gpu_reported_limit_is_used_rather_than_the_fallback():
    """Not hardcoded: a GPU that reports 4096 gets 4096, not the 2048 fallback."""
    assert MosaicPane._live_max_3d_texture(_Pane(_Canvas((16384, 4096)))) == 4096


def test_with_no_canvas_the_fallback_is_the_one_the_renderer_uses():
    """One number, one owner — must match the fallback _napari_view actually applies."""
    assert MosaicPane._live_max_3d_texture(_Pane()) == _DEFAULT_MAX_3D_TEXTURE


def test_the_limit_is_announced_and_says_whether_it_was_measured(caplog):
    """The GPU limit needs a GL context, so it cannot be probed offscreen; an unmeasured number must not read like a measured one."""
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


def test_the_limit_is_announced_once_per_change_not_per_toggle(caplog):
    """The first read can fall back before the canvas has drawn, then succeed afterwards."""
    pane = _Pane(_Canvas((16384, 4096)))
    with caplog.at_level("INFO"):
        MosaicPane._live_max_3d_texture(_Pane())                        # no canvas yet
        for _ in range(5):
            MosaicPane._live_max_3d_texture(pane)                       # canvas has drawn
    assert caplog.text.count("GL_MAX_3D_TEXTURE_SIZE") == 2
    assert "4096" in caplog.text


def test_the_sentence_names_what_the_number_costs_the_user():
    """The line must say what the number decides, not just state it."""
    line = max_3d_texture_line(2048, measured=True)
    assert "2048" in line and "ROI" in line
