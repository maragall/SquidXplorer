"""The region is the navigation unit, and exactly one object owns which region is current."""

from __future__ import annotations

import pytest

from squidxplorer._region_nav import RegionCursor


# --------------------------------------------------------------------------------------
# The cursor: the single owner
# --------------------------------------------------------------------------------------

def test_empty_cursor_has_no_region():
    c = RegionCursor()
    assert c.index is None
    assert c.region is None
    assert c.count == 0
    assert not c.activated


def test_set_order_selects_the_first_region_and_announces_it():
    c = RegionCursor()
    seen = []
    c.subscribe(lambda i, r: seen.append((i, r)))
    c.set_order(["A1", "A2", "B1"])
    assert c.index == 0 and c.region == "A1"
    assert seen == [(0, "A1")], "loading a plate must announce the region it landed on"


def test_set_index_moves_and_announces_once():
    c = RegionCursor()
    c.set_order(["A1", "A2", "B1"])
    seen = []
    c.subscribe(lambda i, r: seen.append((i, r)))
    c.set_index(2)
    assert (c.index, c.region) == (2, "B1")
    assert seen == [(2, "B1")]


def test_setting_the_same_index_does_not_re_announce():
    """A slider that echoes its own value back would ping-pong forever with the widget that set it."""
    c = RegionCursor()
    c.set_order(["A1", "A2"])
    seen = []
    c.subscribe(lambda i, r: seen.append(i))
    c.set_index(1)
    c.set_index(1)
    c.set_index(1)
    assert seen == [1]


def test_set_region_and_set_index_are_the_same_move():
    c = RegionCursor()
    c.set_order(["A1", "A2", "B1"])
    c.set_region("B1")
    assert c.index == 2
    a = []
    c.subscribe(lambda i, r: a.append((i, r)))
    c.set_index(2)          # already there by id — must NOT fire again
    assert a == []


def test_set_region_rejects_an_unknown_region():
    c = RegionCursor()
    c.set_order(["A1", "A2"])
    with pytest.raises(KeyError):
        c.set_region("Z9")
    assert c.region == "A1", "a failed move must not leave the cursor half-moved"


def test_set_index_out_of_range_is_refused_not_clamped():
    """Clamping is a silent failure: the caller believes it moved to 99 and the cursor is at 2."""
    c = RegionCursor()
    c.set_order(["A1", "A2", "B1"])
    with pytest.raises(IndexError):
        c.set_index(99)
    with pytest.raises(IndexError):
        c.set_index(-1)
    assert c.index == 0


def test_step_wraps_so_playback_can_loop():
    c = RegionCursor()
    c.set_order(["A1", "A2", "B1"])
    c.set_index(2)
    c.step(1)
    assert c.index == 0
    c.step(-1)
    assert c.index == 2


def test_reordering_keeps_you_on_the_same_region_when_it_survives():
    c = RegionCursor()
    c.set_order(["A1", "A2", "B1", "B2"])
    c.set_region("B1")
    moved = []
    c.subscribe(lambda i, r: moved.append(r))
    # The surviving region is deliberately NOT first in the new order, so an index-0 snap fails.
    c.set_order(["A2", "B1", "B2"])
    assert c.region == "B1" and c.index == 1
    assert moved == [], "staying on the same region is not a navigation; nothing may reload"


def test_reordering_that_drops_the_current_region_lands_on_the_first():
    c = RegionCursor()
    c.set_order(["A1", "A2"])
    c.set_region("A2")
    seen = []
    c.subscribe(lambda i, r: seen.append(r))
    c.set_order(["B1", "B2"])
    assert c.region == "B1"
    assert seen == ["B1"], "the move to a different region must be announced"


def test_clearing_the_order_clears_the_cursor():
    c = RegionCursor()
    c.set_order(["A1"])
    c.set_order([])
    assert c.index is None and c.region is None and not c.activated


# --------------------------------------------------------------------------------------
# `activated` — "the user explicitly opened a region", which is NOT "a region is displayed"
# --------------------------------------------------------------------------------------

def test_a_plate_load_displays_a_region_without_activating_it():
    """If merely opening a plate counted as activation, every run would narrow to region 0."""
    c = RegionCursor()
    c.set_order(["A1", "A2"])
    assert c.region == "A1"
    assert not c.activated


def test_activate_marks_it_and_moves_there():
    c = RegionCursor()
    c.set_order(["A1", "A2"])
    c.activate("A2")
    assert c.activated and c.region == "A2"
    c.deactivate()
    assert not c.activated
    assert c.region == "A2", "deactivating is not a navigation; the frame must not move"


def test_activating_the_region_already_shown_still_marks_it_activated():
    """Double-clicking the region already on screen must count."""
    c = RegionCursor()
    c.set_order(["A1", "A2"])
    assert c.region == "A1" and not c.activated
    c.activate("A1")
    assert c.activated


# --------------------------------------------------------------------------------------
# Subscribers
# --------------------------------------------------------------------------------------

def test_every_subscriber_is_told(monkeypatch):
    c = RegionCursor()
    c.set_order(["A1", "A2"])
    a, b = [], []
    c.subscribe(lambda i, r: a.append(r))
    c.subscribe(lambda i, r: b.append(r))
    c.set_index(1)
    assert a == ["A2"] and b == ["A2"]


def test_a_subscriber_that_raises_does_not_silence_the_others():
    c = RegionCursor()
    c.set_order(["A1", "A2"])
    good = []

    def boom(i, r):
        raise RuntimeError("subscriber exploded")

    c.subscribe(boom)
    c.subscribe(lambda i, r: good.append(r))
    problems = []
    c.on_problem(problems.append)
    c.set_index(1)
    assert good == ["A2"], "the surviving subscriber must still have been told"
    assert len(problems) == 1 and "subscriber exploded" in problems[0]


def test_a_subscriber_failure_with_no_problem_sink_is_raised_not_swallowed():
    c = RegionCursor()
    c.set_order(["A1", "A2"])
    c.subscribe(lambda i, r: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        c.set_index(1)


# --------------------------------------------------------------------------------------
# The napari-backed slider widget
# --------------------------------------------------------------------------------------

napari = pytest.importorskip("napari", reason="the region slider is napari's own dims slider")
pytest.importorskip("qtpy")


@pytest.fixture
def qapp():
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def make_slider():
    """Build RegionSliders and JOIN napari's animation thread afterwards, or Qt SIGABRTs pytest."""
    from squidxplorer._region_nav import RegionSlider
    made = []

    def _make():
        s = RegionSlider()
        made.append(s)
        return s

    yield _make
    for s in made:
        s.shutdown()


def test_slider_bindings_are_verified_not_trusted():
    """napari moved its Qt access path twice between 0.5 and 0.8."""
    from squidxplorer._region_nav import REQUIRED_PLAYBACK_BINDINGS, NapariPlaybackError, verify_playback_bindings

    verify_playback_bindings()                     # the real ones must be there
    assert REQUIRED_PLAYBACK_BINDINGS, "the binding list must not be empty"

    class _Empty:
        pass

    dotted = REQUIRED_PLAYBACK_BINDINGS[0][0]
    with pytest.raises(NapariPlaybackError):
        verify_playback_bindings(modules={dotted: _Empty()})


def test_slider_is_napari_s_own_dims_slider_with_a_play_button(qapp, make_slider):
    """The widget IS napari's QtDims over a napari Dims model."""
    from squidxplorer._region_nav import RegionSlider
    from napari._qt.widgets.qt_dims_slider import QtDimSliderWidget

    s = make_slider()
    s.set_count(5)
    assert isinstance(s.dim_slider, QtDimSliderWidget)
    assert s.dim_slider.play_button is not None, "no play button = we would have to build one"
    assert s.fps > 0


def test_slider_moves_the_cursor_and_the_cursor_moves_the_slider(qapp, make_slider):
    c = RegionCursor()
    c.set_order(["A1", "A2", "B1", "B2"])
    s = make_slider()
    s.bind(c)

    s.set_index_from_user(3)                  # a user drag
    assert c.index == 3, "the slider did not move the cursor"

    c.set_index(1)                            # e.g. a double-click on the plate
    assert s.index == 1, "the cursor did not move the slider"


def test_binding_sizes_the_slider_to_the_plate(qapp, make_slider):
    c = RegionCursor()
    s = make_slider()
    s.bind(c)
    c.set_order(["A1", "A2", "B1"])
    assert s.count == 3
    c.set_order(["A1"])
    assert s.count == 1


def _pump(qapp, predicate, seconds=5.0):
    import time
    deadline = time.time() + seconds
    while time.time() < deadline and not predicate():
        qapp.processEvents()
        time.sleep(0.005)
    return predicate()


def test_playback_is_napari_s_and_it_walks_regions(qapp, make_slider):
    """Pressing play must walk regions, and keep going; frame_done stands in for the mosaic finishing."""
    c = RegionCursor()
    c.set_order([f"R{i}" for i in range(8)])
    s = make_slider()
    s.bind(c)
    visited = []
    c.subscribe(lambda i, r: (visited.append(i), s.frame_done()))

    assert not s.is_playing
    s.play(fps=30)
    # `is_playing` right after play() is a race against napari's animation thread; assert on moves.
    assert _pump(qapp, lambda: len(visited) >= 4), f"playback stalled at {visited}"
    assert s.is_playing
    assert _pump(qapp, lambda: len(visited) >= 9), f"playback stopped early at {visited}"
    assert 0 in visited[1:], f"playback never wrapped round the plate: {visited}"
    s.shutdown()
    qapp.processEvents()
    assert not s.is_playing


def test_playback_loops_even_when_napari_is_configured_to_play_once(qapp, make_slider):
    """The loop mode must not be inherited from the user's global napari setting."""
    from napari.settings import get_settings

    settings = get_settings().application
    was = settings.playback_mode
    settings.playback_mode = "once"
    try:
        c = RegionCursor()
        c.set_order([f"R{i}" for i in range(4)])
        s = make_slider()                       # constructed UNDER the hostile setting
        s.bind(c)
        assert s.dim_slider.loop_mode.value == "once", "the hostile setting did not take"
        visited = []
        c.subscribe(lambda i, r: (visited.append(i), s.frame_done()))
        s.play(fps=30)
        assert _pump(qapp, lambda: len(visited) >= 6), (
            f"playback stopped early under loop_mode 'once': {visited}")
        s.shutdown()
    finally:
        settings.playback_mode = was


def test_playback_never_runs_ahead_of_the_loading(qapp, make_slider):
    """napari drops frames while the render gate is closed; a fast timer must not queue loads."""
    c = RegionCursor()
    c.set_order([f"R{i}" for i in range(8)])
    s = make_slider()
    s.bind(c)
    moves = []
    c.subscribe(lambda i, r: moves.append(i))      # NOBODY calls frame_done: nothing finishes

    s.play(fps=60)
    _pump(qapp, lambda: False, seconds=1.0)        # let a free-running timer do its worst
    s.stop()
    qapp.processEvents()
    assert len(moves) == 1, (
        f"playback requested {len(moves)} regions while none had finished loading; "
        "the render gate is not holding"
    )


def test_a_stalled_playback_says_so_instead_of_looking_pressed(qapp, make_slider):
    c = RegionCursor()
    c.set_order([f"R{i}" for i in range(4)])
    s = make_slider()
    s.bind(c)
    s.STALL_GRACE_S = 0.2
    said = []
    s.on_problem(said.append)
    s.play(fps=30)
    assert _pump(qapp, lambda: bool(said), seconds=5.0), "a stall was never reported"
    assert "not finished loading" in said[0]
    assert not s.is_playing
    s.shutdown()


def test_playing_an_empty_plate_says_so_rather_than_doing_nothing(qapp, make_slider):
    s = make_slider()
    said = []
    s.on_problem(said.append)
    s.play(fps=10)
    assert not s.is_playing
    assert said and "no regions" in said[0].lower()


def test_playing_a_single_region_plate_says_so(qapp, make_slider):
    """Silently ignoring the click is the dead-button failure mode."""
    c = RegionCursor()
    c.set_order(["only"])
    s = make_slider()
    s.bind(c)
    said = []
    s.on_problem(said.append)
    s.play(fps=10)
    assert not s.is_playing
    assert said and "one region" in said[0].lower()
