"""The working layout: a narrow plate, and a view window over every well filling the rest.

THE REQUEST (Spencer, 2026-08-10): "Plate view should always load at about one-fifth the width of
the current screen, and the full height. When a plate is loaded by any manner, a default child
should populate with all wells selected... The idea is to provide a view that users are used to
navigating with other software suites."

WHAT THIS COSTS, because it was measured before it was taken. Plate cells are square and at full
height the plate is WIDTH-limited, so narrowing the root shrinks every well: 96-well 43.2 -> 28.5
px, 1536-well 10.8 -> 7.1. Accepted on the reading that the plate is now a NAVIGATOR — the pixels
live in the view beside it — and the wheel still zooms with fit as the floor.

THE SHAPE OF THIS FILE. The offscreen platform reports ONE 800x600 screen, so a literal like
`assert child.width() == 1500` could not be written here and a literal that did pass would pin
nothing. So the arithmetic is pure and tested across six real screen shapes without building a
window at all, and the window tests assert only RELATIONSHIPS — against `availableGeometry()` and
against the root's own frame — which hold on any screen.

Geometry assertions are on a really-shown, really-sized window, the discipline
`test_main_window_layout` states: widgets can all be `isVisible()` and one of them can be 0 px.
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

from qtpy.QtCore import QRect  # noqa: E402

from squidmip import _viewer as V  # noqa: E402
from squidmip._fontscale import beside_rect, default_root_width  # noqa: E402
from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)


MIN_W, DESIGN_W = 420, 596

#: (label, work-area width, work-area height). Real shapes, including both bounds binding.
SCREENS = [
    ("workstation 4K @200%", 1920, 1032),
    ("laptop", 1440, 852),
    ("small laptop", 1280, 800),
    ("2560 wide", 2560, 1400),
    ("4K @100%", 3840, 2120),
    ("offscreen", 800, 600),
]


# --- pure arithmetic ----------------------------------------------------------------------------

@pytest.mark.parametrize("label,w,h", SCREENS)
def test_the_root_width_is_bounded_at_both_ends(label, w, h):
    """Never below the minimum (its own controls do not fit), never above the design width (past
    it the extra goes to gutters, not to plate)."""
    got = default_root_width(w, MIN_W, DESIGN_W)
    assert MIN_W <= got <= DESIGN_W, f"{label}: {got} outside [{MIN_W}, {DESIGN_W}]"


@pytest.mark.parametrize("label,w,h", SCREENS)
def test_the_root_and_the_view_tile_the_screen_exactly(label, w, h):
    """THE ASSERTION THAT ENCODES "the other four fifths": no gap and no overlap, on every screen.
    A share that is merely "about right" either leaves a strip of desktop or hides part of one
    window under the other, and both look like a bug rather than a layout."""
    avail = QRect(0, 0, w, h)
    root_w = default_root_width(w, MIN_W, DESIGN_W)
    root = QRect(0, 0, root_w, h)
    child = beside_rect(avail, root)
    assert child.left() == root.right() + 1, f"{label}: gap or overlap at the seam"
    assert root.width() + child.width() == w, (
        f"{label}: {root.width()} + {child.width()} != {w}")


def test_a_wide_screen_gets_about_one_fifth():
    """Where neither bound binds, the request is honoured literally."""
    avail_w = 2560
    got = default_root_width(avail_w, MIN_W, DESIGN_W)
    assert abs(got / avail_w - 0.2) < 0.02, f"{got}/{avail_w} is not about a fifth"


def test_a_laptop_degrades_upward_rather_than_squeezing():
    """The floor binds here, and the DOCUMENTED consequence is a ~30/70 split rather than 20/80.
    Pinned as a number so a future change to the floor shows up here instead of silently reshaping
    every laptop's layout."""
    got = default_root_width(1440, MIN_W, DESIGN_W)
    assert got == MIN_W
    assert 0.27 < got / 1440 < 0.31, f"{got}/1440 = {got / 1440:.2f}, expected ~0.29"


def test_the_view_never_leaves_the_work_area():
    """Including from a root the user has dragged nearly full width — the floor takes over and the
    rect is pulled back inside rather than hanging off the right edge."""
    avail = QRect(0, 0, 1920, 1032)
    child = beside_rect(avail, QRect(0, 0, 1900, 1032))
    assert child.width() >= avail.width() // 3, "the view was squeezed to a sliver"
    assert child.right() <= avail.right(), "the view ran off the screen"
    assert child.left() >= avail.left()


def test_the_view_matches_the_roots_top_and_height():
    """They read as one layout rather than two adjacent windows, and stay in step if the root's
    height changes."""
    avail = QRect(0, 0, 1920, 1032)
    root = QRect(0, 0, 420, 900)
    child = beside_rect(avail, root)
    assert child.top() == root.top() and child.height() == root.height()


# --- real windows -------------------------------------------------------------------------------

def test_the_default_layout_opens_one_view_over_every_well(qapp, napari_pane_stub, squid_dataset):
    """THE REQUEST. Every well, so the plate can navigate anywhere without opening anything."""
    root, _ = squid_dataset
    win = V.PlateWindow(None, default_layout=True)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    qapp.processEvents()
    try:
        windows = win._viewer_manager.windows
        assert len(windows) == 1, f"expected one default view, got {len(windows)}"
        assert windows[0]._regions == list(win._order), "the default view is not over every well"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_default_view_does_not_select_anything(qapp, napari_pane_stub, squid_dataset):
    """`select_all()` was the obvious way to say "every well" and would have been wrong: it writes
    the operator's selection, so every run would silently have been scoped to the whole plate. The
    same invariant `test_nav_wiring` pins for merely opening a plate."""
    root, _ = squid_dataset
    win = V.PlateWindow(None, default_layout=True)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    qapp.processEvents()
    try:
        assert win._overview.selected_wells() == [], "the default view selected wells"
        assert win._selected_regions == [], "the default view scoped the operator run"
        assert win._current_well is None, "the default view counted as the user opening a region"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_default_view_costs_one_region_load_however_many_it_holds(
        qapp, napari_pane_stub, squid_dataset):
    """THE ASSERTION THAT MAKES "startup is independent of plate size" A FACT. A window renders one
    region at a time, so holding 2 or 1536 costs the same to open. Counted rather than reasoned:
    the claim is about work done, and only a count can say."""
    root, _ = squid_dataset
    win = V.PlateWindow(None, default_layout=True)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    qapp.processEvents()
    view = win._viewer_manager.windows[0]
    loads = []
    real = view._load_mosaic
    view._load_mosaic = lambda r, *a, **k: (loads.append(str(r)), real(r, *a, **k))[1]
    try:
        _drain_until(qapp, lambda: view._shown_region is not None, timeout=10)
        assert len(view._regions) >= 2, "fixture needs several regions for this to mean anything"
        assert len(loads) <= 1, f"opening loaded {len(loads)} regions: {loads}"
    finally:
        view._load_mosaic = real
        shutdown_plate_window(qapp, win)


def test_the_root_takes_the_layout_geometry(qapp, napari_pane_stub, squid_dataset):
    """THE OTHER HALF OF THE REQUEST: "about one-fifth the width of the current screen, and the
    full height". Asserted against the work area, so it holds on any screen.

    The MOVE matters as much as the size and is easy to forget: Windows cascades the first
    top-level, so the root does not start at the left edge, and "the view fills the rest" would
    then push the view off the right of the screen. Added after a mutation that skipped the whole
    geometry step left every other test in this file green."""
    from squidmip._fontscale import window_screen

    root, _ = squid_dataset
    win = V.PlateWindow(None, default_layout=True)
    # DISPLACED FIRST, on purpose. The constructor already applies `_default_root_size`, and
    # offscreen a window starts at (0,0), so a window left where it was built satisfies both
    # assertions below without the layout step running at all — measured: a mutation that removed
    # the whole geometry call left this file green until this displacement was added.
    win.move(180, 140)
    win.resize(700, 420)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    qapp.processEvents()
    try:
        screen = window_screen(win)
        if screen is None:
            pytest.skip("no screen to size against")
        avail = screen.availableGeometry()
        assert win.width() == default_root_width(avail.width(), win.minimumWidth(), win._DESIGN_W)
        assert win.height() == max(win.minimumHeight(), avail.height() - 80), "not full height"
        assert win.frameGeometry().topLeft() == avail.topLeft(), (
            "the root was not moved to the corner of the work area")
    finally:
        shutdown_plate_window(qapp, win)


def test_the_view_is_placed_beside_the_plate(qapp, napari_pane_stub, squid_dataset):
    """Relative to the work area and the root's own frame — never a literal, because the offscreen
    screen is not the user's and a literal would pin nothing anywhere else.

    WIDTH IS ASSERTED THROUGH THE WINDOW'S OWN MINIMUM, and that is not a hedge. Offscreen there is
    no font rendering, so text widgets report inflated size hints and a view's `minimumWidth` comes
    out around 3000 px — against a work area a quarter of that. Measured on the real workstation
    the same window reports 1224 against a 1920 work area and lands on exactly the requested rect
    (1498x982 at 422,0). So the rule the app actually follows is "the requested rect, unless the
    window cannot be that small", and asserting the requested rect verbatim here would be pinning
    an offscreen artefact rather than the layout.
    """
    from squidmip._fontscale import window_screen

    root, _ = squid_dataset
    win = V.PlateWindow(None, default_layout=True)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    qapp.processEvents()
    try:
        view = win._viewer_manager.windows[0]
        screen = window_screen(win)
        if screen is None:
            pytest.skip("no screen to place against")
        want = beside_rect(screen.availableGeometry(), win.frameGeometry())
        got = view.geometry()
        assert got.left() == want.left(), "the view does not start where the plate ends"
        assert got.left() >= win.frameGeometry().right(), "the view overlaps the plate"
        assert got.height() == want.height(), "the view is not the plate's height"
        assert got.width() == max(want.width(), view.minimumWidth()), (
            "the view is neither the width asked for nor its own minimum")
    finally:
        shutdown_plate_window(qapp, win)


def test_a_second_acquisition_replaces_the_default_view(qapp, napari_pane_stub, squid_dataset,
                                                        multipage_dataset):
    """Every open view holds the READER it was built with, so keeping the old default view across
    a re-ingest would show the previous acquisition's pixels under the new plate — a wrong-data
    bug, not a stale-layout one."""
    root, _ = squid_dataset
    other, _ = multipage_dataset            # a (root, arrays) pair, like squid_dataset
    win = V.PlateWindow(None, default_layout=True)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    qapp.processEvents()
    first_id = win._default_view_id
    assert first_id is not None
    try:
        win.ingest(str(other))
        qapp.processEvents()
        assert win._acq_name != "acq", "the second acquisition was refused; test proves nothing"
        assert win._default_view_id != first_id, "the default view survived a new acquisition"
        assert all(w.window_id != first_id for w in win._viewer_manager.windows), (
            "the previous acquisition's view is still open")
    finally:
        shutdown_plate_window(qapp, win)


def test_the_root_geometry_is_not_reapplied_on_a_second_acquisition(
        qapp, napari_pane_stub, squid_dataset, multipage_dataset):
    """The layout is an opening position, not a policy. A root the user has dragged and resized
    must not snap back because they dropped another folder on it."""
    root, _ = squid_dataset
    win = V.PlateWindow(None, default_layout=True)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    qapp.processEvents()
    try:
        other, _ = multipage_dataset
        win.resize(700, 500)
        win.move(37, 41)
        qapp.processEvents()
        moved = win.geometry()
        win.ingest(str(other))
        qapp.processEvents()
        assert win.geometry() == moved, "a second acquisition moved the window the user had placed"
    finally:
        shutdown_plate_window(qapp, win)


def test_a_refused_acquisition_opens_no_view(qapp, napari_pane_stub, squid_dataset, tmp_path):
    """A LOAD THAT FAILED IS NOT A LOAD. `ingest` refuses in three places and every one returns
    before the hook, so a folder that is not an acquisition must leave the layout exactly as it
    was — no new view, and the previous one still standing.

    Written because the first draft of the test above passed a `(root, arrays)` tuple to `ingest`
    by mistake, which the app duly refused; the refusal path was being exercised by accident and
    nothing was asserting on it."""
    root, _ = squid_dataset
    win = V.PlateWindow(None, default_layout=True)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    qapp.processEvents()
    first_id = win._default_view_id
    try:
        win.ingest(str(tmp_path / "not-an-acquisition"))
        qapp.processEvents()
        assert win._default_view_id == first_id, "a refused load replaced the default view"
        assert [w.window_id for w in win._viewer_manager.windows] == [first_id], (
            "a refused load changed which windows are open")
    finally:
        shutdown_plate_window(qapp, win)


# --- the opt-out, which is what protects every other test in the suite --------------------------

def test_without_the_flag_nothing_opens_and_nothing_moves(qapp, napari_pane_stub, squid_dataset):
    """THE DEFAULT, and the reason it is a constructor keyword rather than a test-mode branch.

    ~120 existing tests call `ingest()`; almost none use `napari_pane_stub`. An unconditional
    auto-open would make every one of them build a real napari QtViewer. It also keeps
    `test_no_orphan_windows`'s premise true — "a headless test never calls show(), so anything
    visible here showed ITSELF" — for the construction it actually describes."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    before = win.geometry()
    win.ingest(str(root))
    qapp.processEvents()
    try:
        assert win._viewer_manager.windows == [], "a bare PlateWindow opened a view"
        assert win.geometry() == before, "a bare PlateWindow moved itself"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_default_view_is_not_counted_in_the_close_all_warning(qapp, napari_pane_stub,
                                                                  squid_dataset):
    """Warning about the app's own furniture on every quit is how people learn to click a dialog
    through — taking the real warning with it."""
    root, _ = squid_dataset
    win = V.PlateWindow(None, default_layout=True)
    win.show()
    qapp.processEvents()
    win.ingest(str(root))
    qapp.processEvents()
    try:
        assert len(win._viewer_manager.windows) == 1
        assert win._open_view_count() == 0, "the auto-opened view would trigger a quit dialog"
        win._viewer_manager.open(list(win._order)[:1])       # now the USER opens one
        qapp.processEvents()
        assert win._open_view_count() == 1, "a user-opened view is not being counted"
    finally:
        shutdown_plate_window(qapp, win)
