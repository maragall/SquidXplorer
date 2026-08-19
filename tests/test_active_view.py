"""Which view is ACTIVE — and the registry learning it from the user, not only from itself.

THE DEFECT THIS CLOSES. ``ViewerManager._focused_id`` was written by five places, and every one of
them was the app moving focus: ``_spawn``, ``focus``, ``set_selected``, ``clear_focus``,
``collapse_all``. None of them was the USER moving it. ``RegionViewer.changeEvent`` already saw
every activation — it used the signal to halt playback in a window nobody was watching — and told
the registry nothing.

So clicking a view's title bar changed nothing the app believed. The plate kept washing whichever
window was focused last time the app decided, and anything reading ``focused_id`` as "the window
the user is looking at" read the contrast of a window the user had left. It was a lie about the
user, told by a registry that only watched itself.

That matters now because the plate is becoming a navigator: "click a well to move the active view"
is only meaningful if "the active view" is the window the user actually last worked in.

WHY OFFSCREEN TESTS DRIVE ``changeEvent`` DIRECTLY. Real activation needs a window manager, and
there is none under ``QT_QPA_PLATFORM=offscreen`` — ``activateWindow()`` and ``isActiveWindow()``
do not describe anything. So these tests state the window's answer (monkeypatching
``isActiveWindow``) and hand Qt's own event in, which exercises the real handler rather than a
paraphrase of it.
"""

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


def test_activating_a_window_makes_it_the_active_view(qapp, napari_pane_stub, squid_dataset,
                                                      monkeypatch):
    """THE DEFECT, stated directly. Spawning b left b focused; the user going back to a must move
    the registry with them."""
    root, _ = squid_dataset
    win, a, b = _plate_with_two_views(qapp, root)
    mgr = win._viewer_manager
    try:
        assert mgr.focused_id == b.window_id, "the newest window should start focused"
        _activate(monkeypatch, a, other=b)
        assert mgr.focused_id == a.window_id, "activating a window did not move the focus"
        assert mgr.active_view() is a
    finally:
        shutdown_plate_window(qapp, win)


def test_activating_a_window_moves_the_plate_wash(qapp, napari_pane_stub, squid_dataset,
                                                  monkeypatch):
    """``viewFocused`` is what repaints the plate's per-view hue. Recording the focus without
    announcing it would leave the plate showing the previous window's wells."""
    root, _ = squid_dataset
    win, a, b = _plate_with_two_views(qapp, root)
    mgr = win._viewer_manager
    seen = []
    mgr.viewFocused.connect(lambda regions: seen.append(list(regions)))
    try:
        _activate(monkeypatch, a, other=b)
        assert seen, "activating a window announced nothing"
        assert seen[-1] == list(a._regions), "the wash moved to the wrong window's regions"
    finally:
        shutdown_plate_window(qapp, win)


def test_reactivating_the_already_focused_window_announces_nothing(qapp, napari_pane_stub,
                                                                   squid_dataset, monkeypatch):
    """THE PING-PONG GUARD, and the reason ``focus`` sets ``_focused_id`` before it activates.

    ``focus()`` calls ``activateWindow()``, which fires ``changeEvent``, which lands in
    ``note_focus``. Without the unchanged-id early return that is a loop with the window manager —
    and a plate that repaints its hue on every frame of it."""
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


def test_focus_does_not_ping_pong_through_the_window_manager(qapp, napari_pane_stub, squid_dataset):
    """The same guard, exercised through the real ``focus()`` path rather than a synthetic event."""
    root, _ = squid_dataset
    win, a, b = _plate_with_two_views(qapp, root)
    mgr = win._viewer_manager
    mgr.focus(a.window_id)
    seen = []
    mgr.viewFocused.connect(lambda regions: seen.append(list(regions)))
    try:
        mgr.focus(a.window_id)
        qapp.processEvents()
        assert len(seen) <= 1, f"focusing the focused window announced {len(seen)} times"
    finally:
        shutdown_plate_window(qapp, win)


def test_deactivating_does_not_clear_the_active_view(qapp, napari_pane_stub, squid_dataset,
                                                     monkeypatch):
    """CLICKING THE PLATE MUST NOT COST THE PLATE ITS TARGET.

    Deactivation is not "no view is focused" — it is usually the user reaching for the plate, and
    the plate is how the focused view gets driven. Clearing on the way out would mean the target
    disappeared at the exact moment it was needed. ``_focused_id`` therefore means LAST-focused."""
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
    """The job ``changeEvent`` already had. Adding the registry report must not displace it — a
    window nobody is watching still has to stop drawing."""
    root, _ = squid_dataset
    win, a, b = _plate_with_two_views(qapp, root)
    stopped = []

    class _Playing:
        """A control that is playing. Stands IN FOR the slider rather than mutating it: the real
        ``is_playing`` is a read-only property over the animation thread, and a test that could
        set it would be describing a slider this app does not have."""

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
        # PUT THE REAL SLIDER BACK BEFORE TEARDOWN. `dispose` joins the slider's animation thread
        # through this attribute, so a window torn down while the stub is installed destroys a live
        # QThread and aborts the interpreter (0xC0000409, measured). Not monkeypatch: its undo runs
        # after this block, which is already too late.
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
