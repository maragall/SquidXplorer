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

WHAT IS PINNED. The sources are now the per-region ``RegionViewer`` windows registered in
``ViewerManager``, so:

* a window is bound the moment the manager spawns it (``windowOpened``), with the real root
  window and its real manager, and with ``_mosaic_pane`` still None — the premise of the bug;
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


def _spawn(win, window_id: int = 1, resolved=None) -> _FakeWindow:
    """A window opens: the manager announces it exactly as ``_spawn`` does."""
    child = _FakeWindow(window_id, resolved)
    win._viewer_manager.windowOpened.emit(child)
    return child


# --------------------------------------------------------------- the premise of the whole bug


def test_the_root_really_has_no_central_napari_pane_to_bind(qapp, squid_dataset):
    """If this ever fails, the central pane is back and 8.1's rewiring should be re-examined."""
    win = _open_plate(squid_dataset)
    try:
        assert win._mosaic_pane is None, (
            "a central mosaic pane exists again; _bind_napari_contrast now has two candidate "
            "sources and the precedence has to be decided rather than inherited"
        )
    finally:
        win.close()


# ------------------------------------------------------------------------- contrast, the ask


def test_a_contrast_drag_in_a_window_repaints_the_plate(qapp, squid_dataset):
    """The user-visible behaviour: drag a channel's contrast in a window, the plate follows.

    MUTATION: put back the ``_mosaic_pane is None`` guard (or drop the ``windowOpened``
    connection) and this goes red, which is the state the code shipped in.
    """
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        ch_name = _channels(win)[0]

        child.mosaic.user_drags_contrast(ch_name, 11.0, 222.0)

        contrast = win._overview._contrast
        assert contrast.window(0) == (11.0, 222.0), (
            f"the plate is not rendering channel 0 with the window the user set in the window: "
            f"{contrast.window(0)}"
        )
    finally:
        win.close()


def test_the_plate_adopts_the_window_it_opens_with_before_any_gesture(qapp, squid_dataset):
    """Julio, with a screenshot: "loupe not contrast synched with window ... the yellow vs green."

    The sink above only reports a CHANGE, and deliberately so. But napari's autoscale at open is
    not a change, it is the initial state, so the moment that matters most -- the window a region
    comes up with -- is the one moment no sink can ever report. ``_adopt_centre_view`` was written
    to pull it and is gated on ``self._mosaic_pane``, which the test above pins as permanently
    None: the pull was orphaned when the central pane was removed, and until the user happened to
    drag a slider the plate painted from its running histogram while the window painted from
    napari's autoscale. The loupe magnifies the plate, so it inherited the disagreement.

    MUTATION: drop the ``_adopt_window_view`` call from ``_bind_window_contrast`` (or point it back
    at ``self._mosaic_pane``) and this goes red while every gesture test above stays green -- which
    is exactly the state the code shipped in.
    """
    win = _open_plate(squid_dataset)
    try:
        ch_name = _channels(win)[0]
        _spawn(win, resolved={ch_name: (321.0, 4321.0)})     # no gesture: it just opened

        contrast = win._overview._contrast
        assert contrast.window(0) == (321.0, 4321.0), (
            "the window came up showing (321, 4321) and the plate is painting "
            f"{contrast.window(0)}; nobody has touched a slider, so no sink will ever say so")
        assert contrast.is_followed(0) and not contrast.is_manual(0), (
            "the pull latched the channel MANUAL: an owner's autoscale is not a user gesture")
    finally:
        win.close()


def test_the_windows_contrast_is_followed_and_never_latched_manual(qapp, squid_dataset):
    """A window's resolved window is a SINK reading, not a policy decision by the user.

    napari autoscales by itself, at open and whenever the displayed data changes. Recording that
    as a user gesture is what latched every channel MANUAL before anyone had touched anything: the
    plate's running auto-contrast was dead from the first frame, and per-region scope resolved
    every cell to one global window while the plate drew a "wells NOT comparable" badge over the
    top. Same numbers, different authority, and ``_RunningContrast.resolve`` reads the authority.

    MUTATION: make the sink call ``set_channel_window`` (the manual latch) instead of
    ``follow_channel_window`` and this goes red while the test above stays green.
    """
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        child.mosaic.user_drags_contrast(_channels(win)[0], 5.0, 500.0)

        contrast = win._overview._contrast
        assert contrast.is_followed(0), "the window's contrast never reached the plate's sink"
        assert not contrast.is_manual(0), (
            "the plate latched channel 0 MANUAL from an owner's window; auto-contrast is now dead "
            "for every well that streams in afterwards"
        )
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


# ------------------------------------------------------------- visibility and colour, also asked


def test_an_eye_icon_toggle_in_a_window_hides_that_channel_on_the_plate(qapp, squid_dataset):
    """Julio asked for the TOGGLES as well as the contrast. napari's eye icon is the only control
    over channel visibility now — the plate's own checkboxes are gone — so the plate has to be a
    sink of it.

    MUTATION: drop the ``on_user_visibility`` binding and this goes red.
    """
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        ch_name = _channels(win)[0]

        child.mosaic.user_clicks_eye(ch_name, False)
        assert bool(win._overview._mask[0]) is False, "the channel is still in the plate composite"

        child.mosaic.user_clicks_eye(ch_name, True)
        assert bool(win._overview._mask[0]) is True, "the channel never came back"
    finally:
        win.close()


def test_a_colormap_change_in_a_window_re_tints_the_plate(qapp, squid_dataset):
    """Julio: "I change channel colormap in napari and plate view doesn't react." Same sink
    shape: napari owns the colour, the plate follows it."""
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        child.mosaic.user_picks_colormap(_channels(win)[0], (0.0, 1.0, 0.0))
        assert tuple(float(v) for v in win._overview._colors[0]) == (0.0, 1.0, 0.0)
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
    second window whose gestures went nowhere would be the same defect one window later."""
    win = _open_plate(squid_dataset)
    try:
        first, second = _spawn(win, 1), _spawn(win, 2)
        names = _channels(win)

        first.mosaic.user_drags_contrast(names[0], 1.0, 100.0)
        second.mosaic.user_drags_contrast(names[1], 7.0, 700.0)

        contrast = win._overview._contrast
        assert contrast.window(0) == (1.0, 100.0)
        assert contrast.window(1) == (7.0, 700.0), "the second window's gesture went nowhere"
    finally:
        win.close()


def test_the_last_gesture_wins_when_two_windows_touch_one_channel(qapp, squid_dataset):
    """Deliberate, and stated so it is a decision rather than an accident: a window IS a view of a
    subset of this plate, so the plate paints with whatever window the user last resolved. The
    alternative — one privileged window — is the central pane the decentralization removed."""
    win = _open_plate(squid_dataset)
    try:
        first, second = _spawn(win, 1), _spawn(win, 2)
        ch_name = _channels(win)[0]

        first.mosaic.user_drags_contrast(ch_name, 1.0, 100.0)
        second.mosaic.user_drags_contrast(ch_name, 2.0, 200.0)

        assert win._overview._contrast.window(0) == (2.0, 200.0)
    finally:
        win.close()


def test_binding_the_same_window_twice_does_not_stack_a_second_subscription(qapp, squid_dataset):
    """``MosaicLayers`` keeps a LIST of callbacks and cannot unsubscribe, so a re-bind is
    permanent. ``_bind_napari_contrast`` is also called from the mosaic-done path, so it has to be
    idempotent or every region load adds another copy of the plate to the window's sink."""
    win = _open_plate(squid_dataset)
    try:
        child = _spawn(win)
        n = len(child.mosaic.contrast_cbs)
        assert n == 1

        win._bind_window_contrast(child)     # the same window offered again
        win._bind_napari_contrast()          # ...and the sweep over every open window
        assert len(child.mosaic.contrast_cbs) == n, (
            f"the plate subscribed {len(child.mosaic.contrast_cbs)} times to one window"
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
