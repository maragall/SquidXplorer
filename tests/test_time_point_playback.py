"""Playback on the TIME axis: pixels must actually change per frame, and never queue ahead of the read."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("napari", reason="playback is napari's own dims playback")


@pytest.fixture(autouse=True)
def _sync_slicing_for_determinism(monkeypatch):
    """This file's pump-conditions were written for SYNCHRONOUS slicing and flake under the async default (three different tests across runs)."""
    from napari.components import ViewerModel

    orig = ViewerModel.__init__

    def patched(self, *a, **k):
        orig(self, *a, **k)
        self._layer_slicer._force_sync = True

    monkeypatch.setattr(ViewerModel, "__init__", patched)

from squidxplorer import open_reader                                        # noqa: E402
from squidxplorer._time_point import NoPlaybackError, TimePointBar          # noqa: E402
from tests.conftest import (                                            # noqa: E402
    N_TIME_POINTS,
    TIME_SERIES_CHANNELS,
    TIME_SERIES_REGION,
    time_series_pixel_value,
)


@pytest.fixture(scope="module")
def qapp():
    """This module's QApplication."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def make_bar(qapp):
    """Build TimePointBars and JOIN napari's animation thread afterwards."""
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


def test_playback_is_napari_s_own_and_not_a_qtimer(make_bar):
    """Asserting the widget IS napari's animation thread, not a hand-rolled QTimer that free-runs on the GUI loop."""
    from napari._qt.widgets.qt_dims_slider import AnimationThread, QtDimSliderWidget

    bar = make_bar(playback=True)
    bar.set_count(3)
    assert bar.playback is not None
    assert isinstance(bar.playback.dim_slider, QtDimSliderWidget)
    assert bar.playback.dim_slider.play_button is not None, "no play button = we built our own"
    assert tuple(bar.playback._dims.axis_labels)[0] == "time_point", "axis 0 must be the TIME axis"
    assert bar.fps > 0
    bar.play(fps=20)
    assert isinstance(getattr(bar.playback.qt_dims, "_animation_thread"), AnimationThread), (
        "playback is not running on napari's off-thread animation")
    bar.stop()


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
    """The plate does NOT animate time; calling `play()` anyway is a programming error that raises, not a no-op."""
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
    """One class, two skins (plain QSlider vs napari's dims widget), one set of gesture semantics."""
    seen = []
    bar = make_bar(on_change=seen.append, playback=playback)

    bar.set_count(3)
    assert bar.count == 3 and not bar.isHidden()
    assert bar.slider.maximum() == 2, "the slider is not the length of what it navigates"

    bar.set_time_point(2)                       # following somebody else
    assert bar.time_point == 2 and seen == [], "a programmatic move was reported as a gesture"
    assert "3" in bar.label.text() and "time_point" in bar.label.text(), bar.label.text()

    bar.set_time_point_from_user(1)             # a drag
    assert bar.time_point == 1 and seen == [1]

    bar.set_time_point_from_user(99)            # past the end: clamped, not refused
    assert bar.time_point == 2 and seen == [1, 2]

    seen.clear()
    bar.set_count(5)
    bar.set_time_point(4)
    bar.set_count(2)                            # resizing down clamps the position
    assert bar.time_point <= 1 and bar.slider.maximum() == 1 and seen == []
    bar.set_count(1)                            # a re-ingest is not a gesture
    assert seen == [] and bar.isHidden()


def test_a_playback_step_is_a_gesture_because_it_must_reload(make_bar, qapp):
    """A frame advance fires `on_change` exactly as a drag does, through the same loader path."""
    seen = []
    bar = make_bar(on_change=seen.append, playback=True)
    bar.set_count(N_TIME_POINTS)
    bar.on_problem(lambda text: seen.append(f"PROBLEM: {text}"))
    bar.play(fps=30)
    assert _pump(qapp, lambda: len(seen) >= 1), "playback never reached the loader"
    bar.stop()
    assert all(isinstance(v, int) for v in seen), seen


def test_playback_loops_over_the_series(make_bar, qapp):
    """Must wrap even under a hostile `playback_mode="once"`, which `QtDimSliderWidget` reads at construction."""
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
    """Nothing finishes, so exactly one frame may ever be requested: this is what separates playback from a timer."""
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


def _added_values(pane, channel):
    """Every distinct pixel value painted for *channel*, in arrival order."""
    from squidxplorer._napari_view import key_of, pyramid_levels

    histories = getattr(pane, "_test_value_history", None)
    if histories is None:
        histories = pane._test_value_history = {}
    if channel not in histories:
        rec = histories[channel] = []

        def _grab(layer):
            k = key_of(layer)
            if k is None or k.channel != channel:
                return
            try:
                levels = pyramid_levels(layer.data)
                level0 = levels[0] if levels else layer.data
                value = int(np.asarray(level0[0]).max())
            except Exception:                        # noqa: BLE001 - not a raw mosaic level
                return
            if not rec or rec[-1] != value:
                rec.append(value)

        def _hook(layer):
            layer.events.data.connect(lambda e, ly=layer: _grab(ly))
            _grab(layer)

        viewer = pane._viewer
        for ly in list(viewer.layers):
            _hook(ly)
        viewer.layers.events.inserted.connect(lambda e: _hook(e.value))
    return list(histories[channel])


def test_playing_a_window_renders_a_DIFFERENT_frame_per_timepoint(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """Press play; the pixels on screen must change with the timepoint, not just the slider state."""
    from squidxplorer._region_viewer import ViewerManager

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
        want = {time_series_pixel_value(t, 0, 0) for t in range(N_TIME_POINTS)}
        assert _pump(qapp, lambda: want.issubset(set(_added_values(pane, channel))), seconds=20.0), (
            f"playback painted {sorted(set(_added_values(pane, channel)))}, wanted every one of "
            f"{sorted(want)} — the window is animating a still image")
        bar.stop()
        qapp.processEvents()

        values = _added_values(pane, channel)
        assert all(a != b for a, b in zip(values, values[1:])), values
    finally:
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_a_superseded_load_cannot_repaint_the_window_with_its_own_timepoint(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """Cancellation is by GENERATION: a stale and a current worker agree on `region` and differ only in `t`."""
    from squidxplorer._region_viewer import ViewerManager

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
        opened = []
        win._time_point_bar.frame_done = lambda: opened.append(True)
        win._on_plane(TIME_SERIES_REGION, channel, [np.zeros((1, 4, 4), dtype=np.uint16)],
                      None, None, gen=stale_gen)
        win._on_done(TIME_SERIES_REGION, 1, gen=stale_gen)
        assert _added_values(pane, channel) == before, "a superseded load repainted the window"
        assert opened == [], "a superseded load opened the frame gate"
        win._on_done(TIME_SERIES_REGION, 1, gen=win._load_gen)
        assert opened == [True], "the current load did NOT open the frame gate"
    finally:
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_a_reload_reuses_the_layers_instead_of_destroying_them(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """A timepoint step must NOT tear the raw layers down and build them again (165-265 ms/channel to rebuild)."""
    from squidxplorer._region_viewer import _RAW_OP, ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    channel = TIME_SERIES_CHANNELS[0]
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        assert _pump(qapp, lambda: bool(_added_values(pane, channel)))
        before_layer = pane.mosaic.find(_RAW_OP, channel)
        assert before_layer is not None

        win._time_point_bar.set_time_point_from_user(1)
        assert _pump(qapp, lambda: _added_values(pane, channel)[-1]
                     == time_series_pixel_value(1, 0, 0), seconds=10.0)
        assert pane.mosaic.find(_RAW_OP, channel) is before_layer, (
            "the reload destroyed the raw layers; every frame now pays a full rebuild")
    finally:
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


# The tests below run against a REAL napari ViewerModel, not `napari_pane_stub`, because the
# layer-REUSE hazard lives downstream of `add_mosaic` where the stub never goes: `_reuse_layer`
# assigns `layer.data` without checking the shape can survive, and napari raises on region
# transitions of different extent (deeper->shallower pyramid, 2D->3D, 3D->2D). The rule is the
# DISTINCTION: a REGION change (different shape) must REMOVE and add fresh; a TIMEPOINT change
# (same shape) must REUSE. Both halves are asserted below since a fix in only one direction is wrong.

def _real_mosaic():
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    return MosaicLayers(ViewerModel())


#: One region per SHAPE, ordered to walk every crashing transition: deeper->shallower, 3D->2D, 2D->3D.
_SHAPE_WALK = {
    "two_levels": [np.zeros((2, 64, 64), np.uint16), np.zeros((2, 32, 32), np.uint16)],
    "one_level": [np.zeros((2, 32, 32), np.uint16)],
    "flat_2d": [np.zeros((16, 16), np.uint16)],
    "deep_3d": [np.zeros((3, 16, 16), np.uint16)],
}


def _catch_layer_failures(win):
    """Record what `_on_plane` raises instead of letting it reach Qt."""
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
    """A `_MosaicWorker` stand-in that emits a CHOSEN pyramid, synchronously (a real worker's result arrives via a queued connection, so a pumping test"""
    from qtpy.QtCore import QObject, Signal

    class _ShapeWorker(QObject):
        ready = Signal(str, str, object, object, object)
        problem = Signal(str)
        finished_count = Signal(int)
        finished = Signal()

        def __init__(self, reader, meta, region, channels, parent=None, time_point=0):
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
    """Walk four region shapes through the REAL `MosaicLayers`. None of them may raise."""
    from squidxplorer import _workers as W
    from squidxplorer._region_viewer import ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    real_worker = W._MosaicWorker
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        assert _pump(qapp, lambda: bool(len(pane._viewer.layers))), "the window never loaded"
        pane.mosaic = _real_mosaic()                 # REAL napari from here on

        W._MosaicWorker = _shape_worker_class(_SHAPE_WALK)
        failures = _catch_layer_failures(win)
        regions = list(_SHAPE_WALK)
        win._cursor.set_order(regions)
        previous = None
        for index, region in enumerate(regions):
            win._cursor.set_index(index)
            if win._load_timer is not None:
                win._load_timer.stop()               # we drive the loads, not the debounce
            win._load_mosaic(region)
            assert not failures, (
                f"{previous} -> {region} handed napari a layer of another shape: {failures[0]}")
            layer = pane.mosaic.find("raw", TIME_SERIES_CHANNELS[0])
            assert layer is not None, f"{region}: nothing was added"
            assert np.asarray(layer.data[0]).shape == _SHAPE_WALK[region][0].shape, (
                f"{region}: the layer kept the previous region's shape")
            previous = region
    finally:
        W._MosaicWorker = real_worker
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_a_load_that_produces_nothing_DOES_drop_the_stale_layers(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """A failed load must not leave the previous timepoint's pixels on screen under the new label."""
    from squidxplorer._region_viewer import _RAW_OP, ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        assert _pump(qapp, lambda: bool(len(pane._viewer.layers)))
        assert pane.mosaic.find(_RAW_OP, TIME_SERIES_CHANNELS[0]) is not None
        win._on_done(TIME_SERIES_REGION, 0, gen=win._load_gen)
        assert pane.mosaic.find(_RAW_OP, TIME_SERIES_CHANNELS[0]) is None, (
            "a load that built nothing left the previous timepoint's pixels on screen")
    finally:
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_the_camera_is_not_re_framed_on_every_frame(
    multi_time_point_dataset, napari_pane_stub, qapp, monkeypatch
):
    """Framing follows the REGION, not the timepoint: re-framing on every frame drags the user's zoom back to fit."""
    from squidxplorer._region_viewer import ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    channel = TIME_SERIES_CHANNELS[0]
    try:
        win = mgr.open([TIME_SERIES_REGION])
        pane = napari_pane_stub[0]
        frames = []
        real_reset = type(pane._viewer).reset_view
        monkeypatch.setattr(type(pane._viewer), "reset_view",
                            lambda self, *a, **k: (frames.append(1),
                                                   real_reset(self, *a, **k))[1])
        assert _pump(qapp, lambda: bool(_added_values(pane, channel)))

        win._on_done(TIME_SERIES_REGION, 2, gen=win._load_gen)      # first framing of this region
        assert len(frames) >= 1, "the region was never framed at all"
        was = len(frames)
        for time_point in (1, 2):
            win._time_point_bar.set_time_point_from_user(time_point)
            assert _pump(qapp, lambda: _added_values(pane, channel)[-1]
                         == time_series_pixel_value(time_point, 0, 0), seconds=10.0)
        assert len(frames) == was, (
            f"the camera was re-framed {len(frames) - was} times while only the timepoint moved")
    finally:
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_a_finished_mosaic_worker_is_released_rather_than_piling_up(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    """One QThread per frame, kept alive by its Qt parent, is a pile that grows with playback."""
    from squidxplorer._region_viewer import ViewerManager

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
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


