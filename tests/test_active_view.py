"""Which view is ACTIVE — and the registry learning it from the user, not only from itself."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # headless Qt; must precede any Qt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt GUI tests.",
        allow_module_level=True,
    )

from qtpy.QtCore import QEvent  # noqa: E402

from squidxplorer import _viewer as V  # noqa: E402
from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)


def _activate(monkeypatch, win, other=None):
    """Say this window is the active one and hand Qt's activation event to the real handler."""
    monkeypatch.setattr(win, "isActiveWindow", lambda: True, raising=False)
    if other is not None:
        monkeypatch.setattr(other, "isActiveWindow", lambda: False, raising=False)
    win.changeEvent(QEvent(QEvent.ActivationChange))


def _plate_with_two_views(qapp, root):
    win = V.PlateWindow(None)
    win.ingest(str(root))
    order = list(win._order)
    a = win._viewer_manager.open(order[:1])
    b = win._viewer_manager.open(order[1:2] or order[:1])
    assert a is not None and b is not None
    _drain_until(qapp, lambda: a._pane is not None and b._pane is not None, timeout=10)
    return win, a, b


def test_activating_a_window_makes_it_the_active_view_and_moves_the_plate_wash(
        qapp, napari_pane_stub, squid_dataset, monkeypatch):
    """THE DEFECT, stated directly; ``viewFocused`` is what repaints the plate's per-view hue."""
    root, _ = squid_dataset
    win, a, b = _plate_with_two_views(qapp, root)
    mgr = win._viewer_manager
    seen = []
    mgr.viewFocused.connect(lambda regions: seen.append(list(regions)))
    try:
        assert mgr.focused_id == b.window_id, "the newest window should start focused"
        _activate(monkeypatch, a, other=b)
        assert mgr.focused_id == a.window_id, "activating a window did not move the focus"
        assert mgr.active_view() is a
        assert seen, "activating a window announced nothing"
        assert seen[-1] == list(a._regions), "the wash moved to the wrong window's regions"
    finally:
        shutdown_plate_window(qapp, win)


def test_reactivating_the_already_focused_window_announces_nothing(qapp, napari_pane_stub,
                                                                   squid_dataset, monkeypatch):
    """THE PING-PONG GUARD, and the reason ``focus`` sets ``_focused_id`` before it activates."""
    root, _ = squid_dataset
    win, a, b = _plate_with_two_views(qapp, root)
    mgr = win._viewer_manager
    try:
        _activate(monkeypatch, a, other=b)
        seen = []
        mgr.viewFocused.connect(lambda regions: seen.append(list(regions)))
        _activate(monkeypatch, a, other=b)
        _activate(monkeypatch, a, other=b)
        assert seen == [], f"a redundant activation re-announced {len(seen)} time(s)"
    finally:
        shutdown_plate_window(qapp, win)


def test_deactivating_does_not_clear_the_active_view(qapp, napari_pane_stub, squid_dataset,
                                                     monkeypatch):
    """CLICKING THE PLATE MUST NOT COST THE PLATE ITS TARGET."""
    root, _ = squid_dataset
    win, a, b = _plate_with_two_views(qapp, root)
    mgr = win._viewer_manager
    try:
        _activate(monkeypatch, a, other=b)
        monkeypatch.setattr(a, "isActiveWindow", lambda: False, raising=False)
        a.changeEvent(QEvent(QEvent.ActivationChange))
        assert mgr.focused_id == a.window_id, "losing activation cleared the active view"
        assert mgr.active_view() is a
    finally:
        shutdown_plate_window(qapp, win)


def test_deactivating_still_halts_playback(qapp, napari_pane_stub, squid_dataset, monkeypatch):
    """The job ``changeEvent`` already had."""
    root, _ = squid_dataset
    win, a, b = _plate_with_two_views(qapp, root)
    stopped = []

    class _Playing:
        """A control that is playing."""

        is_playing = True

        def stop(self):
            stopped.append(self)

    real_slider = a._slider
    try:
        a._slider = _Playing()
        monkeypatch.setattr(a, "isActiveWindow", lambda: False, raising=False)
        a.changeEvent(QEvent(QEvent.ActivationChange))
        assert stopped, "deactivating a window no longer halts its playback"
    finally:
        a._slider = real_slider
        shutdown_plate_window(qapp, win)


def test_there_is_no_active_view_with_nothing_open(qapp, napari_pane_stub, squid_dataset):
    """The answer the plate needs before it decides a click means anything."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    try:
        assert win._viewer_manager.active_view() is None
    finally:
        shutdown_plate_window(qapp, win)


def test_closing_the_active_view_leaves_no_active_view(qapp, napari_pane_stub, squid_dataset):
    """A stale id here would hand the plate a window that no longer exists."""
    root, _ = squid_dataset
    win, a, b = _plate_with_two_views(qapp, root)
    mgr = win._viewer_manager
    try:
        mgr.focus(b.window_id)
        b.close()
        qapp.processEvents()
        assert mgr.active_view() is not b, "the registry still hands out a closed window"
    finally:
        shutdown_plate_window(qapp, win)
