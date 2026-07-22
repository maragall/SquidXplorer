"""The REGION is the navigation unit, and exactly ONE object owns which region is current.

These tests exist because "a second copy of the current selection, hand-synced" is this
project's dominant defect shape (4+ confirmed instances: the FOV slider vs the red box, the
plate's contrast vs ndv's, ``_push_index`` vs the plate index, the loupe's compositor vs the
plate's). The cursor is the fix, so the tests are written against the property that makes it a
fix — that there is no way to move one of the two and not the other — rather than against the
happy path.
"""

from __future__ import annotations

import pytest

from squidmip._region_nav import RegionCursor


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
    """A re-announce is not cosmetic: every subscriber reloads a mosaic, and a slider that
    echoes its own value back would ping-pong forever with the widget that set it."""
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
    """Clamping is a silent failure: the caller believes it moved to 99 and the cursor is at 2.
    Six confirmed silent failures in this project say this must be loud."""
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
    """An exploration tab re-scopes the plate order. Landing back on index 0 would silently
    move the red frame off the region the user was looking at."""
    c = RegionCursor()
    c.set_order(["A1", "A2", "B1", "B2"])
    c.set_region("B1")
    moved = []
    c.subscribe(lambda i, r: moved.append(r))
    # The surviving region is deliberately NOT first in the new order. With it first, a cursor
    # that snapped to index 0 would land on it by coincidence and this test would pass while
    # asserting nothing — which is how it was first written, and the mutation run caught it.
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
    """`_selection_regions` scopes an operator run to the activated region. If merely OPENING a
    plate counted as activation, every run would silently narrow to region 0."""
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
    """Double-clicking the region that happens to be on screen must count. Otherwise the first
    double-click after open does nothing at all, which is how a dead button gets shipped."""
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
    """One broken subscriber must not take the red frame down with it, and the failure must be
    reported rather than swallowed — this project has six confirmed log-and-continue bugs."""
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
pytest.importorskip("PyQt5")


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def make_slider():
    """Build RegionSliders and JOIN napari's animation thread afterwards.

    Not hygiene: Qt aborts the process with SIGABRT on "QThread: Destroyed while thread is
    still running", which kills pytest before it can print why a test failed. The first run of
    this file exited 134 with no summary at all for exactly that reason.
    """
    from squidmip._region_nav import RegionSlider
    made = []

    def _make():
        s = RegionSlider()
        made.append(s)
        return s

    yield _make
    for s in made:
        s.shutdown()


def test_slider_bindings_are_verified_not_trusted():
    """napari moved its Qt access path twice between 0.5 and 0.8. A bound-but-absent symbol is
    how `_voxel_scale` ran every time and did nothing for its whole life."""
    from squidmip._region_nav import REQUIRED_PLAYBACK_BINDINGS, NapariPlaybackError, verify_playback_bindings

    verify_playback_bindings()                     # the real ones must be there
    assert REQUIRED_PLAYBACK_BINDINGS, "the binding list must not be empty"

    class _Empty:
        pass

    dotted = REQUIRED_PLAYBACK_BINDINGS[0][0]
    with pytest.raises(NapariPlaybackError):
        verify_playback_bindings(modules={dotted: _Empty()})


def test_slider_is_napari_s_own_dims_slider_with_a_play_button(qapp, make_slider):
    """We are not reinventing playback. The widget IS napari's QtDims over a napari Dims model,
    so the play button, the fps popup and the loop modes are napari's, not ours."""
    from squidmip._region_nav import RegionSlider
    from napari._qt.widgets.qt_dims_slider import QtDimSliderWidget

    s = make_slider()
    s.set_count(5)
    assert isinstance(s.dim_slider, QtDimSliderWidget)
    assert s.dim_slider.play_button is not None, "no play button = we would have to build one"
    assert s.fps > 0


def test_slider_moves_the_cursor_and_the_cursor_moves_the_slider(qapp, make_slider):
    """The two must not be able to disagree — that is the whole point of the cursor."""
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
    """The acceptance test for requirement 3: pressing play must walk REGIONS, and keep going.

    ``frame_done`` stands in for the mosaic finishing — in the app that is what closes the
    loop, and playing without it would prove nothing about the real path.
    """
    c = RegionCursor()
    c.set_order([f"R{i}" for i in range(8)])
    s = make_slider()
    s.bind(c)
    visited = []
    c.subscribe(lambda i, r: (visited.append(i), s.frame_done()))

    assert not s.is_playing
    s.play(fps=30)
    # NOT asserted here: `is_playing` reads napari's animation-thread event, which the thread
    # only clears once it is scheduled. Checking it on the line after play() is a race, and a
    # racy assertion in a suite this size is a flake nobody will ever diagnose. What matters is
    # that regions actually move, which is what the pumps below prove.
    assert _pump(qapp, lambda: len(visited) >= 4), f"playback stalled at {visited}"
    assert s.is_playing
    # It must KEEP going and WRAP. The loop mode is asserted directly as well as behaviourally:
    # it comes from napari's USER-WIDE `playback_mode` setting, so a machine configured to
    # "once" would step one region and stop. Behaviour alone cannot tell those apart on a
    # machine that happens to be set to "loop", which the mutation run proved.
    assert _pump(qapp, lambda: len(visited) >= 9), f"playback stopped early at {visited}"
    assert 0 in visited[1:], f"playback never wrapped round the plate: {visited}"
    s.shutdown()
    qapp.processEvents()
    assert not s.is_playing


def test_playback_loops_even_when_napari_is_configured_to_play_once(qapp, make_slider):
    """The loop mode must not be inherited from the user's global napari setting.

    ``QtDimSliderWidget`` reads ``application.playback_mode`` at construction, so on a
    machine where the user last watched a movie in "once" mode the region axis would advance a
    single region and stop. Behaviour alone cannot catch that on a machine already set to
    "loop" — the mutation run proved it — so the setting is forced to the hostile value here.
    """
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
    """napari drops frames while the render gate is closed. That is the property that stops a
    10 fps timer queueing ten region loads for every one that finishes, so it is pinned."""
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
    """A one-region plate has nothing to play through. Silently ignoring the click is the
    dead-button failure mode this brief exists to remove."""
    c = RegionCursor()
    c.set_order(["only"])
    s = make_slider()
    s.bind(c)
    said = []
    s.on_problem(said.append)
    s.play(fps=10)
    assert not s.is_playing
    assert said and "one region" in said[0].lower()
