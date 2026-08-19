"""The plate follows the per-region windows' napari; it owns no central pane of its own."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)

from qtpy.QtWidgets import QApplication

from squidxplorer import _viewer as V


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


class _FakeMosaic:
    """The SUBSCRIPTION surface of ``MosaicLayers``, and its gesture calls, and nothing else."""

    def __init__(self, resolved=None) -> None:
        self.contrast_cbs: list = []
        self.visibility_cbs: list = []
        self.colormap_cbs: list = []
        self.op_cbs: list = []
        # napari's own autoscale at open: readable, never fired as a gesture.
        self.resolved: dict = dict(resolved or {})

    def contrast(self, channel: str):
        return self.resolved.get(channel)

    def on_user_contrast(self, cb) -> None:
        self.contrast_cbs.append(cb)

    def on_user_visibility(self, cb) -> None:
        self.visibility_cbs.append(cb)

    def on_user_colormap(self, cb) -> None:
        self.colormap_cbs.append(cb)

    def on_user_op(self, cb) -> None:
        self.op_cbs.append(cb)

    def user_drags_contrast(self, channel: str, lo: float, hi: float) -> None:
        for cb in list(self.contrast_cbs):
            cb(channel, lo, hi)

    def user_clicks_eye(self, channel: str, on: bool) -> None:
        for cb in list(self.visibility_cbs):
            cb(channel, on)

    def user_picks_colormap(self, channel: str, rgb) -> None:
        for cb in list(self.colormap_cbs):
            cb(channel, rgb)

    def user_shows_layer(self, op: str, on: bool = True) -> None:
        """The user ticked (or unticked) a PROCESSING LAYER in this window's layer tree."""
        for cb in list(self.op_cbs):
            cb(op, on)


class _FakeWindow:
    """A ``RegionViewer`` as the PLATE reads one: a window id and a napari pane."""

    def __init__(self, window_id: int = 1, resolved=None) -> None:
        self.window_id = int(window_id)
        self.mosaic = _FakeMosaic(resolved)
        self._pane = type("_Pane", (), {"ok": True, "mosaic": self.mosaic})()


def _open_plate(squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    return win


def _channels(win) -> list:
    return [c["name"] for c in win._meta["channels"]]


def _spawn(win, window_id: int = 1, resolved=None) -> _FakeWindow:
    """A window opens: the manager announces it exactly as ``_spawn`` does."""
    child = _FakeWindow(window_id, resolved)
    win._viewer_manager.windowOpened.emit(child)
    return child


def test_the_root_really_has_no_central_napari_pane_to_bind(qapp, squid_dataset):
    """Pinned as "the attribute does not exist", not "is None" — None hid twenty dead methods."""
    win = _open_plate(squid_dataset)
    try:
        assert not hasattr(win, "_mosaic_pane"), (
            "a central mosaic pane exists again; the window->plate follow path now has two "
            "candidate sources and the precedence has to be decided rather than inherited"
        )
    finally:
        win.close()


def test_a_gesture_in_a_window_leaves_the_plate_alone(qapp, squid_dataset):
    """Live look-following is shelved: gestures in a window change nothing on the plate."""
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        ch_name = _channels(win)[0]
        before_window = win._overview._contrast.window(0)
        before_mask = list(win._overview._mask)

        child.mosaic.user_drags_contrast(ch_name, 11.0, 222.0)
        child.mosaic.user_clicks_eye(ch_name, False)

        assert win._overview._contrast.window(0) == before_window, (
            "a contrast drag in a window still reaches the plate")
        assert list(win._overview._mask) == before_mask, (
            "an eye icon in a window still reaches the plate")
        assert not win._overview._contrast.is_followed(0), (
            "the plate is still following a window's resolved window")
    finally:
        win.close()


def test_the_lut_copy_paste_pair_is_shelved_and_nothing_else_moves_the_plates_look(
        qapp, squid_dataset):
    """2026-08-19, Julio: "Shelf the LUT logic completely." The plate's copy/paste pair, the
    window pair and the shared clipboard are DELETED — absence pinned with `not hasattr`, the
    repo's convention — and a window gesture still leaves the plate alone (the test above), so
    the plate's look now changes only with the acquisition's own declarations and the layer
    picks."""
    import squidxplorer._lut_clipboard as LC
    import squidxplorer._region_viewer as RV

    win = _open_plate(squid_dataset)
    try:
        assert not hasattr(win, "_plate_copy_luts"), "the plate copy handler is back"
        assert not hasattr(win, "_plate_paste_luts"), "the plate paste handler is back"
        assert not hasattr(win, "_plate_copy_lut_btn") and not hasattr(win, "_plate_paste_lut_btn")
        assert not hasattr(LC, "CLIPBOARD"), "the shared LUT clipboard is back"
        assert not hasattr(RV, "_LUT_CLIPBOARD"), "the clipboard alias is back"
    finally:
        win.close()


def test_a_channel_the_plate_does_not_have_is_ignored_rather_than_raising(qapp, squid_dataset):
    """An unknown channel must be dropped quietly — an exception here dies inside a Qt slot."""
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        child.mosaic.user_drags_contrast("a channel that is not in this acquisition", 1.0, 2.0)
        child.mosaic.user_clicks_eye("a channel that is not in this acquisition", False)
        assert not win._overview._contrast.is_followed(0)
    finally:
        win.close()


def test_picking_an_operator_layer_in_a_window_moves_the_plate_onto_it(qapp, squid_dataset):
    win = _open_plate(squid_dataset)
    try:
        win._op_stack.add("mip", "Maximum Intensity Projection")   # as a run would
        win._apply_layers()
        assert win._overview._active == "mip"

        child = _spawn(win)
        child.mosaic.user_shows_layer("mip", False)
        assert win._overview._active == "raw", (
            "hiding the operator layer in the window left the plate on it")

        child.mosaic.user_shows_layer("mip", True)
        assert win._overview._active == "mip", (
            "showing the operator layer in the window did not bring the plate back to it")
    finally:
        win.close()


def test_the_plate_layers_tab_agrees_with_what_the_window_asked_for(qapp, squid_dataset):
    win = _open_plate(squid_dataset)
    try:
        win._op_stack.add("mip", "Maximum Intensity Projection")
        win._apply_layers()
        child = _spawn(win)
        child.mosaic.user_shows_layer("mip", False)
        assert [ly.enabled for ly in win._op_stack.layers() if ly.key == "mip"] == [False]
    finally:
        win.close()


def test_a_window_layer_the_plate_never_ran_is_ignored_rather_than_raising(qapp, squid_dataset):
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        before = win._overview._active
        child.mosaic.user_shows_layer("nothing-the-plate-ran", True)
        assert win._overview._active == before
    finally:
        win.close()


def test_every_open_window_is_bound_not_only_the_first(qapp, squid_dataset):
    win = _open_plate(squid_dataset)
    try:
        first, second = _spawn(win, 1), _spawn(win, 2)
        assert 1 in win._followed_windows and 2 in win._followed_windows, (
            "a second window was not bound")
        assert len(first.mosaic.op_cbs) == 1 and len(second.mosaic.op_cbs) == 1, (
            "the plate did not subscribe to both windows")
    finally:
        win.close()


def test_the_last_gesture_wins_when_two_windows_touch_one_channel(qapp, squid_dataset):
    """Deliberate: the plate paints with whatever window the user last resolved."""
    win = _open_plate(squid_dataset)
    try:
        win._op_stack.add("mip", "Maximum Intensity Projection")
        win._apply_layers()
        first, second = _spawn(win, 1), _spawn(win, 2)

        first.mosaic.user_shows_layer("mip", False)
        assert win._overview._active == "raw", "the FIRST window's layer choice was not followed"
        second.mosaic.user_shows_layer("mip", True)
        assert win._overview._active == "mip", "the SECOND window was never bound"
    finally:
        win.close()


def test_binding_the_same_window_twice_does_not_stack_a_second_subscription(qapp, squid_dataset):
    """``MosaicLayers`` cannot unsubscribe, so a window offered twice must not gain a second copy."""
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        n = len(child.mosaic.op_cbs)
        assert n == 1

        win._bind_window_contrast(child)     # the same window offered again
        win._bind_window_contrast(child)     # ...and again
        assert len(child.mosaic.op_cbs) == n, (
            f"the plate subscribed {len(child.mosaic.op_cbs)} times to one window"
        )
    finally:
        win.close()


def test_a_window_that_came_up_without_napari_is_skipped_quietly(qapp, squid_dataset):
    """A window with no napari pane has nothing to follow and must not break the rest."""
    win = _open_plate(squid_dataset)
    try:
        blind = _FakeWindow(3)
        blind._pane = None
        win._viewer_manager.windowOpened.emit(blind)      # must not raise
        assert 3 not in win._followed_windows
    finally:
        win.close()
