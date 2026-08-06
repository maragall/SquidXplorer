"""Playback on the TIME axis: the picture must actually move, and never run ahead of the read.

`squidmip/_time_point.py` said, from the day the timepoint bar landed, that "there is no
time-reduction operator … the real operation on that axis is playback and export". This file is
the playback half, and it is written against the two ways it can be wrong RATHER THAN against the
happy path, because both have live precedent in this repo:

1. **It animates a still image.** Until 2026-08-04 `_MosaicWorker` never received a `t`, so a
   window whose bar said 3 rendered timepoint 0 — and a play button on top of that would have
   walked the slider over byte-identical pixels while looking perfectly convincing. So the
   acceptance test here reads PIXELS out of the pane and fails if the frames repeat. Every
   assertion about "it played" that does not look at pixels is worthless on this axis.

2. **It queues instead of dropping.** A frame costs a mosaic load. A free-running 10 fps timer
   starts ten loads for every one that finishes, and the viewer falls further behind the slider
   the longer you watch. napari's own playback is debounced on the render for exactly this
   reason, and the gate is REUSED — `frame_done()` is called when the mosaic is on screen, so
   playback self-limits to the rate the data can be read at. Dropping a frame is correct;
   queueing it is not, and that is pinned below by counting the loads that start while none has
   finished.

The fixture is `multi_time_point_dataset` (Nt=3, pixel value = t*100 + z*10 + channel), the only
multi-timepoint acquisition in this suite. A consumer stuck at t=0 differs from the truth by a
visible multiple of 100, which is why the fixture encodes it that way.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("napari", reason="playback is napari's own dims playback")

from squidmip import open_reader                                        # noqa: E402
from squidmip._time_point import NoPlaybackError, TimePointBar          # noqa: E402
from tests.conftest import (                                            # noqa: E402
    N_TIME_POINTS,
    TIME_SERIES_CHANNELS,
    TIME_SERIES_REGION,
    time_series_pixel_value,
)


@pytest.fixture(scope="module")
def qapp():
    """This module's QApplication.

    Declared here rather than taken from pytest-qt: the suite runs under
    ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``, where pytest-qt's ``qapp`` does not exist and a test
    asking for it ERRORS instead of running.
    """
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def make_bar(qapp):
    """Build TimePointBars and JOIN napari's animation thread afterwards.

    Not hygiene: Qt aborts the process with SIGABRT on "QThread: Destroyed while thread is still
    running", which kills pytest before it can print why a test failed.
    """
    made = []

    def _make(**kw):
        bar = TimePointBar(**kw)
        made.append(bar)
        return bar

    yield _make
    for bar in made:
        bar.shutdown()


def _pump(qapp, predicate, seconds=5.0):
    deadline = time.time() + seconds
    while time.time() < deadline and not predicate():
        qapp.processEvents()
        time.sleep(0.005)
    return predicate()


# --- the control ---------------------------------------------------------------------------

def test_playback_is_napari_s_own_and_not_a_qtimer(make_bar):
    """We are not reinventing playback, and we are not putting a timer on the GUI loop.

    A hand-rolled ``QTimer`` gets two things wrong that napari's ``AnimationThread`` does not: it
    ticks on the event loop the canvas draws on, and it free-runs. Asserting the widget IS
    napari's is how "somebody replaced it with a timer" becomes a failing test rather than a
    stutter nobody can reproduce.
    """
    from napari._qt.widgets.qt_dims_slider import AnimationThread, QtDimSliderWidget

    bar = make_bar(playback=True)
    bar.set_count(3)
    assert bar.playback is not None
    assert isinstance(bar.playback.dim_slider, QtDimSliderWidget)
    assert bar.playback.dim_slider.play_button is not None, "no play button = we built our own"
    assert bar.fps > 0
    bar.play(fps=20)
    assert isinstance(getattr(bar.playback.qt_dims, "_animation_thread"), AnimationThread), (
        "playback is not running on napari's off-thread animation")
    bar.stop()


def test_the_axis_napari_is_walking_is_the_TIME_axis(make_bar):
    """The dims model is 3-D with two dummy displayed axes; axis 0 must be the timepoint.

    Named rather than assumed: `_region_nav`'s slider is the same shape over "region", and two
    controls whose axis label is a copy-paste of the other's is precisely how a timepoint bar
    ends up driving regions.
    """
    bar = make_bar(playback=True)
    assert tuple(bar.playback._dims.axis_labels)[0] == "time_point"


def test_a_one_timepoint_acquisition_refuses_to_play_out_loud(make_bar):
    """Silently ignoring the click is the dead-button failure this project keeps re-finding."""
    bar = make_bar(playback=True)
    bar.set_count(1)
    said = []
    bar.on_problem(said.append)
    bar.play(fps=10)
    assert not bar.is_playing
    assert said and "timepoint" in said[0], said


def test_the_plate_s_bar_has_no_playback_and_says_why(make_bar):
    """The plate does NOT animate time, and the refusal names the reason.

    `_PreviewWorker`'s persistent cell cache is keyed (token, region) with no timepoint, so a
    plate that played would show timepoint 0's pixels under a moving label — worse than the bug
    it would look like it was fixing. There is therefore no play button, and calling `play()`
    anyway is a PROGRAMMING error that raises rather than a user gesture that no-ops.
    """
    bar = make_bar(playback=False)
    bar.set_count(3)
    assert bar.playback is None
    assert not bar.is_playing
    with pytest.raises(NoPlaybackError) as excinfo:
        bar.play(fps=10)
    assert "timepoint" in str(excinfo.value)
    bar.frame_done()            # unconditional at the call site: must be a no-op, not a crash


@pytest.mark.parametrize("playback", [False, True])
def test_both_skins_are_the_same_control(make_bar, playback):
    """One class, two skins, ONE set of semantics.

    The playback skin replaces our QSlider with napari's dims widget, and a skin that answered
    differently about where you are, what fires a callback, or where the end is would be the
    second timepoint implementation the module docstring forbids. So the gesture rules are
    asserted against both.
    """
    seen = []
    bar = make_bar(on_change=seen.append, playback=playback)

    bar.set_count(3)
    assert bar.count == 3 and not bar.isHidden()
    assert bar.slider.maximum() == 2, "the slider is not the length of what it navigates"

    bar.set_time_point(2)                       # following somebody else
    assert bar.time_point == 2 and seen == [], "a programmatic move was reported as a gesture"
    assert "3" in bar.label.text()

    bar.set_time_point_from_user(1)             # a drag
    assert bar.time_point == 1 and seen == [1]

    bar.set_time_point_from_user(99)            # past the end: clamped, not refused
    assert bar.time_point == 2 and seen == [1, 2]

    seen.clear()
    bar.set_count(1)                            # a re-ingest is not a gesture
    assert seen == [] and bar.isHidden()


def test_a_playback_step_is_a_gesture_because_it_must_reload(make_bar, qapp):
    """Playback and a drag take the SAME path into the loader.

    A separate "animation" path is how the two end up disagreeing about which timepoint is on
    screen. So a frame advance fires ``on_change`` exactly as a drag does — and that is also
    what makes the pixel test below possible at all.
    """
    seen = []
    bar = make_bar(on_change=seen.append, playback=True)
    bar.set_count(N_TIME_POINTS)
    bar.on_problem(lambda text: seen.append(f"PROBLEM: {text}"))
    bar.play(fps=30)
    assert _pump(qapp, lambda: len(seen) >= 1), "playback never reached the loader"
    bar.stop()
    assert all(isinstance(v, int) for v in seen), seen


def test_playback_loops_over_the_series(make_bar, qapp):
    """It must keep going and WRAP, whatever napari's user-wide loop setting says.

    ``QtDimSliderWidget`` reads ``application.playback_mode`` at construction, so on a machine
    where the user last watched a movie in "once" mode the time axis would advance one frame and
    stop. The hostile value is forced here because behaviour alone cannot tell the two apart on
    a machine already set to "loop".
    """
    from napari.settings import get_settings

    settings = get_settings().application
    was = settings.playback_mode
    settings.playback_mode = "once"
    try:
        visited = []
        bar = make_bar(playback=True)
        bar.set_count(3)
        bar._on_change = lambda t: (visited.append(t), bar.frame_done())
        assert bar.playback.dim_slider.loop_mode.value == "once", "the hostile setting did not take"
        bar.play(fps=30)
        assert _pump(qapp, lambda: len(visited) >= 7), (
            f"playback stopped early under loop_mode 'once': {visited}")
        assert 0 in visited[1:], f"playback never wrapped round the series: {visited}"
        bar.stop()
    finally:
        settings.playback_mode = was


def test_playback_never_runs_ahead_of_the_loading(make_bar, qapp):
    """THE BACKLOG TEST. Nothing finishes, so exactly one frame may ever be requested.

    This is the property that separates "playback" from "a timer": at 60 fps with the gate held
    shut, a free-running implementation asks for ~60 mosaic loads per second and every one of
    them is work the user will never see.
    """
    asked = []
    bar = make_bar(on_change=asked.append, playback=True)
    bar.set_count(N_TIME_POINTS)
    bar.play(fps=60)                                # NOBODY calls frame_done
    _pump(qapp, lambda: False, seconds=1.0)         # let a free-running timer do its worst
    bar.stop()
    qapp.processEvents()
    assert len(asked) == 1, (
        f"playback requested {len(asked)} timepoints while none had finished loading; "
        "the render gate is not holding")


def test_a_stalled_playback_says_so_instead_of_looking_pressed(make_bar, qapp):
    bar = make_bar(playback=True)
    bar.set_count(3)
    bar.playback.STALL_GRACE_S = 0.2
    said = []
    bar.on_problem(said.append)
    bar.play(fps=30)
    assert _pump(qapp, lambda: bool(said), seconds=6.0), "a stall was never reported"
    assert "not finished loading" in said[0]
    assert not bar.is_playing


# --- through the real window, on real pixels -------------------------------------------------

def _added_values(pane, channel):
    """Every distinct pixel value this pane has been handed for *channel*, in arrival order.

    Reads the LEVELS the window actually added, materialised at z=0 — the same expression
    `tests/test_time_point.py` uses for the worker. A test that watched the slider instead would
    pass against a viewer that animates a still image, which is the exact defect this file
    exists for.
    """
    out = []
    for op, ch, levels, _kw in pane.mosaic.added:
        if ch != channel:
            continue
        try:
            value = int(np.asarray(levels[0][0]).max())
        except Exception:                            # noqa: BLE001 - not a raw mosaic level
            continue
        if not out or out[-1] != value:
            out.append(value)
    return out


def test_playing_a_window_renders_a_DIFFERENT_frame_per_timepoint(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """THE ACCEPTANCE TEST. Press play; the pixels on screen must change with the timepoint.

    Written on pixels rather than on the slider on purpose: this feature's predecessor bug was a
    window that navigated timepoints perfectly and rendered timepoint 0 every time. Playback that
    paints the same frame three times fails here and passes every test that only watches state.
    """
    from squidmip._region_viewer import ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    channel = TIME_SERIES_CHANNELS[0]
    try:
        win = mgr.open([TIME_SERIES_REGION])
        assert win is not None
        pane = napari_pane_stub[0]
        bar = win._time_point_bar
        assert bar.count == N_TIME_POINTS
        assert bar.playback is not None, "a window's timepoint bar must be able to play"
        assert _pump(qapp, lambda: bool(_added_values(pane, channel))), "the first frame never landed"

        bar.play(fps=20)
        # Every timepoint of the series, so a loop that only ever shows t=0 and t=1 fails too.
        want = {time_series_pixel_value(t, 0, 0) for t in range(N_TIME_POINTS)}
        assert _pump(qapp, lambda: want.issubset(set(_added_values(pane, channel))), seconds=20.0), (
            f"playback painted {sorted(set(_added_values(pane, channel)))}, wanted every one of "
            f"{sorted(want)} — the window is animating a still image")
        bar.stop()
        qapp.processEvents()

        # And it walked them IN ORDER rather than jumping about: consecutive frames differ, which
        # is what "the blob moves" means to the person watching.
        values = _added_values(pane, channel)
        assert all(a != b for a, b in zip(values, values[1:])), values
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_a_superseded_load_cannot_repaint_the_window_with_its_own_timepoint(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """Cancellation is by GENERATION, and the region check cannot do it.

    A timepoint change keeps the region, so a stale worker and the current one agree about
    `region` and differ only in `t`. Without the generation the retired read lands last and the
    window shows the older frame under the newer label — the same class of silent divergence as
    the bug this axis already had once.
    """
    from squidmip._region_viewer import ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    channel = TIME_SERIES_CHANNELS[0]
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        assert _pump(qapp, lambda: bool(_added_values(pane, channel)))

        stale_gen = win._load_gen
        win._time_point_bar.set_time_point(2)
        win._load_mosaic(TIME_SERIES_REGION)            # the CURRENT load, gen bumped
        assert _pump(qapp, lambda: _added_values(pane, channel)[-1]
                     == time_series_pixel_value(2, 0, 0), seconds=10.0)

        before = list(_added_values(pane, channel))
        # The retired worker finishing LATE, exactly as it would from its own thread.
        win._on_plane(TIME_SERIES_REGION, channel, [np.zeros((1, 4, 4), dtype=np.uint16)],
                      None, None, gen=stale_gen)
        win._on_done(TIME_SERIES_REGION, 1, gen=stale_gen)
        assert _added_values(pane, channel) == before, "a superseded load repainted the window"
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_a_superseded_load_does_not_open_the_playback_gate(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """The other half of the same bug: a stale completion must not let the next frame start.

    If it did, the gate would open once per RETIRED read as well as per real one, and playback
    would go back to queueing — which is the failure the gate exists to prevent, arriving through
    the back door.
    """
    from squidmip._region_viewer import ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    try:
        win = mgr.open([TIME_SERIES_REGION])
        opened = []
        win._time_point_bar.frame_done = lambda: opened.append(True)
        win._on_done(TIME_SERIES_REGION, 1, gen=win._load_gen - 1)
        assert opened == [], "a superseded load opened the frame gate"
        win._on_done(TIME_SERIES_REGION, 1, gen=win._load_gen)
        assert opened == [True], "the current load did NOT open the frame gate"
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_a_reload_reuses_the_layers_instead_of_destroying_them(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """A timepoint step must NOT tear the raw layers down and build them again.

    `add_mosaic` reuses a layer it can find (6465069, "I can't cycle rapidly through these
    mosaics"), and `_load_mosaic` used to remove them first, which guaranteed the slow path.
    Measured with a real napari canvas on sim_5d_2x2_t3: 165-265 ms of GUI thread per channel to
    rebuild, against a 10-13 ms read. That is the difference between ~0.75 fps and ~4.5 fps, and
    between an 800 ms freeze per frame and a 400 ms one.
    """
    from squidmip._region_viewer import _RAW_OP, ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    channel = TIME_SERIES_CHANNELS[0]
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        assert _pump(qapp, lambda: bool(_added_values(pane, channel)))
        pane.mosaic.removed.clear()

        win._time_point_bar.set_time_point_from_user(1)
        assert _pump(qapp, lambda: _added_values(pane, channel)[-1]
                     == time_series_pixel_value(1, 0, 0), seconds=10.0)
        assert _RAW_OP not in pane.mosaic.removed, (
            "the reload destroyed the raw layers; every frame now pays a full rebuild")
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


# --- the reuse rule, against a REAL napari ViewerModel ---------------------------------------
#
# EVERY TEST ABOVE THIS LINE RUNS AGAINST `napari_pane_stub`, whose `add_mosaic` RECORDS THE CALL
# AND RETURNS. `tests/conftest.py` says so where the stub is defined: "napari's own rendering is
# not exercised here, and never was under offscreen." So nothing downstream of `add_mosaic` runs
# in those tests — including the layer-REUSE path, which is where the hazard below lives. A green
# suite proves nothing about it, which is exactly how the bug these two tests exist for got in.
#
# THE HAZARD. `add_mosaic` reuses a layer it can find, and `_reuse_layer` does not refuse a shape
# it cannot survive: it assigns `layer.data` and napari raises somewhere downstream. Driven
# against a real `ViewerModel`, three ordinary region transitions crash:
#
#     deeper -> shallower pyramid   IndexError: index 1 is out of bounds for axis 0 with size 1
#     2D -> 3D                      IndexError: index 2 is out of bounds for axis 1 with size 2
#     3D -> 2D                      ValueError: operands could not be broadcast together
#
# and "a big region then a small one" is ordinary plate navigation: on the 10x tissue set manual0
# is 27 FOVs and manual1 is 28, so their fused mosaics differ in extent.
#
# THE RULE, therefore, is not "reuse" or "remove" but the DISTINCTION between them:
#
#     a REGION change     -> a different mosaic, different shape -> REMOVE, then add fresh
#     a TIMEPOINT change  -> the same region at the same extent  -> REUSE, which is the point
#
# Both halves are asserted, because a fix in either direction alone is wrong: removing always
# costs 165-265 ms of GUI thread per channel per frame and undoes 6465069, and reusing always
# crashes on the first region change.

def _real_mosaic():
    from napari.components import ViewerModel

    from squidmip._napari_view import MosaicLayers

    return MosaicLayers(ViewerModel())


#: One region per SHAPE, in an order that walks every transition the perf agent found crashing:
#: two levels -> one level (deeper -> shallower), 3D -> 2D, then 2D -> 3D.
_SHAPE_WALK = {
    "two_levels": [np.zeros((2, 64, 64), np.uint16), np.zeros((2, 32, 32), np.uint16)],
    "one_level": [np.zeros((2, 32, 32), np.uint16)],
    "flat_2d": [np.zeros((16, 16), np.uint16)],
    "deep_3d": [np.zeros((3, 16, 16), np.uint16)],
}


def _catch_layer_failures(win):
    """Record what `_on_plane` raises instead of letting it reach Qt. Returns the list.

    NOT decoration, and not a swallow. PyQt turns an exception raised inside a slot invoked from
    C++ into `qFatal`, so a napari failure here ABORTS the interpreter — and an abort takes
    pytest's summary line with it, which is the exact failure mode that hid 51 real failures in
    this suite for weeks. Catching at the boundary turns "the process died, no idea why" into a
    named assertion carrying napari's own message.
    """
    failures = []
    real_on_plane = win._on_plane

    def guarded(*args, **kwargs):
        try:
            real_on_plane(*args, **kwargs)
        except BaseException as exc:                 # noqa: BLE001 - reported, never swallowed
            failures.append(f"{type(exc).__name__}: {exc}")

    win._on_plane = guarded
    return failures


def _shape_worker_class(shapes):
    """A `_MosaicWorker` stand-in that emits a CHOSEN pyramid, synchronously.

    Synchronous on purpose: a real worker's result arrives through a QUEUED connection, so a test
    driving it through the event loop would have to pump and could not tell "not yet" from "never".
    """
    from qtpy.QtCore import QObject, Signal

    class _ShapeWorker(QObject):
        ready = Signal(str, str, object, object, object)
        problem = Signal(str)
        finished_count = Signal(int)
        finished = Signal()

        def __init__(self, reader, meta, region, channels, z_index=0, parent=None, t=0):
            super().__init__(parent)
            self._region = str(region)
            self._channels = list(channels)

        def isRunning(self):
            return False

        def stop(self):
            pass

        def start(self):
            levels = shapes[self._region]
            for channel in self._channels:
                self.ready.emit(self._region, channel, levels, None, None)
            self.finished_count.emit(len(self._channels))
            self.finished.emit()

    return _ShapeWorker


def test_a_region_change_never_hands_napari_a_layer_of_another_shape(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """Walk four region shapes through the REAL `MosaicLayers`. None of them may raise.

    MUTATION: drop the region test from `_load_mosaic`'s `remove_op` (reuse across every reload)
    and this fails on the first transition with napari's own IndexError. That is the whole reason
    the removal is conditional rather than gone.
    """
    from squidmip import _viewer as V
    from squidmip._region_viewer import ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    real_worker = V._MosaicWorker
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        assert _pump(qapp, lambda: bool(pane.mosaic.added)), "the window never loaded"
        pane.mosaic = _real_mosaic()                 # REAL napari from here on

        V._MosaicWorker = _shape_worker_class(_SHAPE_WALK)
        failures = _catch_layer_failures(win)
        regions = list(_SHAPE_WALK)
        win._cursor.set_order(regions)
        previous = None
        for index, region in enumerate(regions):
            win._cursor.set_index(index)
            if win._load_timer is not None:
                win._load_timer.stop()               # we drive the loads, not the debounce
            win._load_mosaic(region)
            # STOP AT THE FIRST BAD TRANSITION. A half-assigned napari layer takes the process
            # down on the NEXT touch, so walking on would replace a readable failure with an abort.
            assert not failures, (
                f"{previous} -> {region} handed napari a layer of another shape: {failures[0]}")
            layer = pane.mosaic.find("raw", TIME_SERIES_CHANNELS[0])
            assert layer is not None, f"{region}: nothing was added"
            assert np.asarray(layer.data[0]).shape == _SHAPE_WALK[region][0].shape, (
                f"{region}: the layer kept the previous region's shape")
            previous = region
    finally:
        V._MosaicWorker = real_worker
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_a_timepoint_change_keeps_the_very_same_layer_object(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """The other half: the same region at another timepoint must REUSE, not rebuild.

    Asserted on OBJECT IDENTITY against a real ViewerModel, because that is what makes it cheap
    (165-265 ms of GUI thread per channel is the rebuild) and what keeps every subscription bound
    to the layer alive — contrast, visibility and colormap all subscribe to layer objects, and
    each of those has already broken once when a reload destroyed the object underneath them.
    """
    from squidmip import _viewer as V
    from squidmip._region_viewer import ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    real_worker = V._MosaicWorker
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        assert _pump(qapp, lambda: bool(pane.mosaic.added))
        pane.mosaic = _real_mosaic()

        same_shape = {TIME_SERIES_REGION: [np.zeros((2, 32, 32), np.uint16)]}
        V._MosaicWorker = _shape_worker_class(same_shape)
        win._load_mosaic(TIME_SERIES_REGION)
        first = pane.mosaic.find("raw", TIME_SERIES_CHANNELS[0])
        assert first is not None

        for time_point in (1, 2, 0):
            win._time_point_bar.set_time_point(time_point)
            win._load_mosaic(TIME_SERIES_REGION)
            assert pane.mosaic.find("raw", TIME_SERIES_CHANNELS[0]) is first, (
                f"timepoint {time_point} destroyed and rebuilt the layer instead of reusing it")
    finally:
        V._MosaicWorker = real_worker
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_a_load_that_produces_nothing_DOES_drop_the_stale_layers(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """The other half of not removing up front: a failed load must not leave the old frame up.

    Reusing layers is only safe while a reload actually produces pixels. When it produces none,
    what is on screen belongs to another timepoint and is now sitting under this one's label,
    which is the exact class of silent lie this axis already had once.
    """
    from squidmip._region_viewer import _RAW_OP, ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        assert _pump(qapp, lambda: bool(pane.mosaic.added))
        pane.mosaic.removed.clear()
        win._on_done(TIME_SERIES_REGION, 0, gen=win._load_gen)
        assert _RAW_OP in pane.mosaic.removed, (
            "a load that built nothing left the previous timepoint's pixels on screen")
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_the_camera_is_not_re_framed_on_every_frame(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """Framing follows the REGION, not the timepoint.

    `reset_view` cost 85-130 ms of GUI thread per frame (measured), but the reason it is
    conditional is not the milliseconds: a timepoint step reloads the same region at the same
    stage coordinates, so re-framing drags the user's zoom back to fit on every frame. You
    cannot watch a blob move at 1:1 if the camera keeps pulling out.
    """
    from squidmip._region_viewer import ViewerManager

    class _Camera:
        def __init__(self):
            self.frames = 0

        def reset_view(self):
            self.frames += 1

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    channel = TIME_SERIES_CHANNELS[0]
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        camera = pane.mosaic.model = _Camera()
        assert _pump(qapp, lambda: bool(_added_values(pane, channel)))

        win._on_done(TIME_SERIES_REGION, 2, gen=win._load_gen)      # first framing of this region
        assert camera.frames >= 1, "the region was never framed at all"
        was = camera.frames
        for time_point in (1, 2):
            win._time_point_bar.set_time_point_from_user(time_point)
            assert _pump(qapp, lambda: _added_values(pane, channel)[-1]
                         == time_series_pixel_value(time_point, 0, 0), seconds=10.0)
        assert camera.frames == was, (
            f"the camera was re-framed {camera.frames - was} times while only the timepoint moved")
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_a_finished_mosaic_worker_is_released_rather_than_piling_up(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """One QThread per frame, kept alive by its Qt parent, is a pile that grows with playback.

    Measured before this changed: 78 live `_MosaicWorker` objects after 78 playback frames, none
    of them reclaimable by `gc.collect()`. It is the same accumulation `tools/run_suite_chunked.py`
    diagnosed as the reason this suite cannot run in one process — and playback turns "one per
    navigation" into "one per frame".
    """
    from squidmip._region_viewer import ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    channel = TIME_SERIES_CHANNELS[0]
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        assert _pump(qapp, lambda: bool(_added_values(pane, channel)))
        for time_point in (1, 2, 0):
            win._time_point_bar.set_time_point_from_user(time_point)
            assert _pump(qapp, lambda: _added_values(pane, channel)[-1]
                         == time_series_pixel_value(time_point, 0, 0), seconds=10.0)
        assert _pump(qapp, lambda: win._worker is None, seconds=5.0), (
            "the window is still holding its finished worker; nothing will ever free it")
        assert win._retired_workers == [], "superseded workers were never reaped"
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_the_window_does_not_block_the_ui_thread_to_supersede_a_load(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """`_load_mosaic` must not `wait()` on the worker it is replacing.

    Measured before this changed: `stop()` only sets an Event that `_MosaicWorker` polls BETWEEN
    channels, so the wait ran for as long as the current channel's fuse plus contrast seed took,
    on the GUI thread, up to a 2 s cap — per scrub step, per playback frame. The source is
    asserted because the cost is invisible on a 4x4 fixture and brutal on a 27-FOV region.
    """
    import inspect

    from squidmip._region_viewer import RegionViewer

    src = inspect.getsource(RegionViewer._load_mosaic)
    assert ".wait(" not in src, (
        "_load_mosaic blocks the GUI thread waiting for the load it is superseding")
    assert "_retire_worker" in src, "the superseded worker is not held anywhere; Qt will abort"
