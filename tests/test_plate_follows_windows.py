"""The plate has to follow the WINDOWS' napari, because there is no central napari left.

Task 8.1, 2026-07-29. The requirement is Julio's, and it was quoted verbatim inside the method
that could not run:

    "there shouldn't be any controls for the plate view. It just reacts to toggles and contrast
    adjustments in napari."

WHAT WAS WRONG. ``PlateWindow._bind_napari_contrast`` bound the plate to ``self._mosaic_pane``,
the ONE central napari pane. The decentralization deleted that pane and left
``self._mosaic_pane = None`` unconditionally in ``__init__``, so the method's first guard was
permanently true: it returned before subscribing to anything. Contrast drags, eye-icon toggles and
colormap changes inside a window changed that window and NOTHING else, and no test named it,
because every test that touched the method assigned a stub pane onto a bare ``__new__`` shell and
so never met the guard the real window hits.

The sentinel and every method that guarded on it were deleted on 2026-08-06, including
``_bind_napari_contrast`` itself: once ``_on_mosaic_done`` went, its sweep over every open window
had no caller left. The binding that has always done the work is
``ViewerManager.windowOpened -> _bind_window_contrast``, connected in ``__init__``, and that is
what this file drives.

WHAT IS PINNED. The sources are now the per-region ``RegionViewer`` windows registered in
``ViewerManager``, so:

* a window is bound the moment the manager spawns it (``windowOpened``), with the real root
  window and its real manager;
* a contrast gesture in a window lands in the plate's FOLLOW path, and NOT in its manual latch.
  That distinction is load-bearing: napari autoscales on its own at open, so recording an owner's
  autoscale as a user gesture latched every channel MANUAL before anyone had touched anything and
  killed the plate's running auto-contrast from the first frame;
* channel VISIBILITY follows too (Julio asked for the toggles as well as the contrast), and so
  does the colormap;
* a second window is bound as well as the first, and a window bound twice does not stack a second
  subscription.

WHAT IS DELIBERATELY NOT HERE. Whether a napari event counts as a user gesture at all is
``MosaicLayers``' decision (it filters our own writes, including the percentile window set at add
time and link propagation) and is pinned in ``tests/test_napari_view.py``. This file starts one
step later: given that the owner reported a gesture, where do the numbers land.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)

from qtpy.QtWidgets import QApplication

from squidmip import _viewer as V


@pytest.fixture(scope="module")
def qapp():
    # Held for the module, and the process's app is pinned by squidmip._viewer.qt_app() — see
    # tests/test_window_lifetime.py for why a fixture must not be the only owner.
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)
    return app


class _FakeMosaic:
    """The SUBSCRIPTION surface of ``MosaicLayers``, and its gesture calls, and nothing else.

    It records what the plate subscribed and can fire it, which is exactly what the real
    ``MosaicLayers`` does once it has decided a napari event was a user gesture.
    """

    def __init__(self, resolved=None) -> None:
        self.contrast_cbs: list = []
        self.visibility_cbs: list = []
        self.colormap_cbs: list = []
        self.op_cbs: list = []
        # What napari has ALREADY resolved for each channel in this window -- its own autoscale at
        # open. Deliberately NOT reachable through the gesture helpers below: an autoscale is not a
        # gesture, the sinks filter it out, and it is therefore the one state that can only be
        # READ. `_adopt_window_view` is the reader.
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

    # -- what napari reports when the user actually gestures in this window ------------------
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
    """A ``RegionViewer`` as the PLATE reads one: a window id and a napari pane.

    A real ``RegionViewer`` builds a real napari canvas, and building one in a process that
    already holds PyQt5 widgets loads a second Qt binding and aborts the interpreter rather than
    failing a test (see the note at the top of ``tests/test_channel_bar.py``). The plate touches
    exactly two attributes of a window, so those two are what this carries.
    """

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


def _copy(luts: dict) -> None:
    """Put a window's LUT snapshot on the shared clipboard, as `Copy LUTs` does.

    The dict shape is `RegionViewer._per_channel_luts`': `clim`, `cmap`, `rgb`, and -- since
    2026-08-06 -- `on`, the channel's visibility, which travels WITH the look rather than as a
    separate gesture.
    """
    from squidmip._region_viewer import _LUT_CLIPBOARD
    _LUT_CLIPBOARD.clear()
    _LUT_CLIPBOARD.update(luts)


def _spawn(win, window_id: int = 1, resolved=None) -> _FakeWindow:
    """A window opens: the manager announces it exactly as ``_spawn`` does."""
    child = _FakeWindow(window_id, resolved)
    win._viewer_manager.windowOpened.emit(child)
    return child


# --------------------------------------------------------------- the premise of the whole bug


def test_the_root_really_has_no_central_napari_pane_to_bind(qapp, squid_dataset):
    """The plate must own no napari surface at all. If this fails, the pane is back and the
    precedence between it and the windows has to be DECIDED rather than inherited.

    Pinned as "the attribute does not exist" rather than "the attribute is None", because None was
    the state that let twenty dead methods sit here for two weeks looking like a feature.
    """
    win = _open_plate(squid_dataset)
    try:
        assert not hasattr(win, "_mosaic_pane"), (
            "a central mosaic pane exists again; the window->plate follow path now has two "
            "candidate sources and the precedence has to be decided rather than inherited"
        )
    finally:
        win.close()


# ------------------------------------------- the LOOK: copy/paste, never a live subscription
#
# Julio, 2026-08-06: *"we're shelving the interactive contrast synch. What we do is that whichever
# lookup table we have for the window, we copy it and it reflects on the plate, with whichever
# channels were turned on on the window. And the plate image shouldn't change unless we paste a
# LUT."*
#
# This file used to assert the opposite -- eight tests that a drag, an eye icon or a colormap
# change in ANY window landed on the plate immediately -- and each of them was an honest
# description of what the code did. What none of them could express is the property that made it
# unusable: the plate followed *whichever window the user last gestured in*, so with several
# windows open its look was decided by a history with no surface anywhere. The tests passed
# because each one had exactly one window.
#
# The tests below pin the replacement, and the first one is the whole of it: gestures in a window
# now change NOTHING on the plate.


def test_a_gesture_in_a_window_leaves_the_plate_alone(qapp, squid_dataset):
    """The shelving, stated as a property. Contrast, eye icon and colormap all together, because
    all three used to land and the requirement is about the plate's look as a whole.

    MUTATION: restore any of the three subscriptions in ``_bind_window_contrast`` and this goes
    red on that quantity.
    """
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


def test_pasting_a_windows_luts_is_what_moves_the_plate(qapp, squid_dataset):
    """...and the explicit gesture DOES land. Shelving the subscription without this would be
    removing the feature rather than replacing it."""
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        ch_name = _channels(win)[0]
        child.mosaic.user_drags_contrast(ch_name, 11.0, 222.0)

        _copy({ch_name: {"clim": (11.0, 222.0), "cmap": None, "rgb": None, "on": True}})
        win._plate_paste_luts()

        assert win._overview._contrast.window(0) == (11.0, 222.0), (
            "pasting the window's LUTs did not put its contrast on the plate")
    finally:
        win.close()


def test_a_paste_carries_the_channels_the_window_had_lit(qapp, squid_dataset):
    """*"with whichever channels were turned on on the window."* Visibility travels WITH the LUT:
    a copied look with every channel's window and none of its on/off state is not the look that
    was on screen."""
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        names = _channels(win)
        _copy({names[0]: {"clim": None, "cmap": None, "rgb": None, "on": False},
               names[1]: {"clim": None, "cmap": None, "rgb": None, "on": True}})
        win._plate_paste_luts()

        assert not win._overview._mask[0], (
            "the paste did not carry the window's channel on/off state to the plate")
        assert win._overview._mask[1], "the paste turned off a channel the window had lit"
    finally:
        win.close()


def test_a_paste_can_never_black_the_plate_out(qapp, squid_dataset):
    """The never-go-black floor still wins over a paste. A window with everything switched off is
    a legitimate thing to have; emptying the NAVIGATOR from it is not, because the plate has no
    controls of its own to fill it back in from."""
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        _copy({n: {"clim": None, "cmap": None, "rgb": None, "on": False}
               for n in _channels(win)})
        win._plate_paste_luts()

        assert any(bool(v) for v in win._overview._mask), (
            "pasting an all-off window emptied the plate")
    finally:
        win.close()


def test_a_channel_the_plate_does_not_have_is_ignored_rather_than_raising(qapp, squid_dataset):
    """A window can show a layer whose channel this plate has no column for (a re-ingest, an
    operator's own output). It must be dropped quietly: an exception here is raised inside a Qt
    slot, where it kills the gesture rather than reporting anything."""
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        child.mosaic.user_drags_contrast("a channel that is not in this acquisition", 1.0, 2.0)
        child.mosaic.user_clicks_eye("a channel that is not in this acquisition", False)
        assert not win._overview._contrast.is_followed(0)
    finally:
        win.close()

# ------------------------------------------------------- the processing layer, Julio 2026-08-03


def test_picking_an_operator_layer_in_a_window_moves_the_plate_onto_it(qapp, squid_dataset):
    """Julio: "after I click an operator layer in our window, the thumbnails don't update."

    The three sinks above are all per CHANNEL, and a window's layer tree picks a PROCESSING
    LAYER. There was no sink for that at all, so the plate went on showing whatever layer the
    last run left active however the user drove the window's tree.

    MUTATION: drop the ``on_user_op`` binding and this goes red.
    """
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
    """The tab's checkboxes are a second surface over one stack. A window toggling a layer that
    left the tab reading the old state would be the "view disagrees with its own controls" defect
    ``OperationStack.toggle`` was written to end."""
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
    """A window can carry operator groups this plate has no layer for. Ignoring it must not raise
    out of a Qt slot (an unhandled exception in one aborts the process)."""
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        before = win._overview._active
        child.mosaic.user_shows_layer("nothing-the-plate-ran", True)
        assert win._overview._active == before
    finally:
        win.close()


# ------------------------------------------------------------------- many windows, one plate


def test_every_open_window_is_bound_not_only_the_first(qapp, squid_dataset):
    """The old binding was a once-per-process latch over a single pane. Windows are plural, and a
    second window whose gestures went nowhere would be the same defect one window later.

    Asserted on the LAYER sink, which is the subscription that survived the 2026-08-06 shelving of
    the live look-following. The property under test is about BINDING, not about contrast, so it
    is unaffected by which quantities the plate still follows.
    """
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
    """Deliberate, and stated so it is a decision rather than an accident: a window IS a view of a
    subset of this plate, so the plate paints with whatever window the user last resolved. The
    alternative — one privileged window — is the central pane the decentralization removed."""
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
    """``MosaicLayers`` keeps a LIST of callbacks and cannot unsubscribe, so a re-bind is
    permanent: a window offered twice must not gain a second copy of the plate on its sink.

    The second offer used to come from ``_bind_napari_contrast``, a sweep over every open window
    called from the mosaic-done path; both went on 2026-08-06 with the dead pane. The requirement
    did not go with them -- ``windowOpened`` is emitted per window and nothing stops a caller
    offering the same window again -- so it is pinned directly on the method that carries the
    ``_followed_windows`` guard.
    """
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
    """``RegionViewer`` shows a red "napari viewer unavailable" panel instead of a pane when napari
    is missing, leaving ``_pane`` None. That window has nothing to follow, and the plate must not
    fail to open the rest of them over it."""
    win = _open_plate(squid_dataset)
    try:
        blind = _FakeWindow(3)
        blind._pane = None
        win._viewer_manager.windowOpened.emit(blind)      # must not raise
        assert 3 not in win._followed_windows
    finally:
        win.close()
