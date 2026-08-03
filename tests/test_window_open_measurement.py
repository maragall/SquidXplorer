"""Opening a window is MEASURED: the clock is wired to the real open, not only to a class.

Julio, round two of the operator GUI feedback: "If we can speed up window loading time, that would
be good. But I understand that this laptop is resource Limited."

Nothing in the app could say whether it had got faster. The one measured interval was an operator
run, whose clock starts on a Run button that opening a window never presses, so a complaint about
window loading time could not be turned into a number and a change to it could not be shown to have
helped. ``to-do/2026-08-03-roi-viewport-rendering-design.md`` section 3.6 asks for the second clock
and increment 0 puts it first, ahead of every rendering change, on the grounds that the rest of that
design should not be built on a guess about where the time goes.

WHAT IS PINNED HERE, AND WHY IT IS SEPARATE FROM tests/test_run_measurement.py
------------------------------------------------------------------------------
That file pins the clock itself, against a hand-moved clock, with no Qt. This one pins the WIRING:
that a real ``ViewerManager.open`` produces a record, that an ROI child is distinguishable from a
whole-region open, and that the things a window does AFTER it has opened -- loading another region,
being closed -- do not each record another open. A clock that is right and hooked to the wrong
signal passes every test in that file and reports nonsense.

Per docs/adr/0001-ci-gates-work-not-time.md nothing here asserts a duration. These tests assert
that a number was recorded and that it was recorded once.

NOT TESTED, and stated rather than left to be found: that first paint is taken where the layer is
ADDED rather than where the worker emits it. Both placements produce a number, and only a stalled
interface tells them apart, which needs a real interface under load.
"""

from __future__ import annotations

import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import pytest  # noqa: E402

# The ONE guard in this file, and it is an ENVIRONMENT gate rather than a skipped assertion, exactly
# as every other GUI test module here does it: PyQt is an optional extra and
# ``squidmip._region_viewer`` imports it at module scope, so without it there is nothing to test.
pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidmip._measure import METRICS, OK, WINDOW_OPEN  # noqa: E402
from squidmip._region_viewer import ViewerManager  # noqa: E402

from .conftest import REGIONS  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixture)


@pytest.fixture(autouse=True)
def _own_the_metrics_log():
    """These tests read THE process-wide log, because that is the log the app writes and the file
    ``persist_runs`` drains. Cleared around each test so one test's opens cannot be counted by the
    next; the log is a diagnostic, so no other test depends on what is in it."""
    METRICS.clear()
    yield
    METRICS.clear()


@pytest.fixture
def manager(qapp, napari_pane_stub, squid_dataset):
    """A real ViewerManager over the real reader, with no PlateWindow in the way.

    Same shape as the fixture in tests/test_view_settings.py, and for the same reason: the windows
    it hands out set WA_DeleteOnClose, so the drain and the collect belong here, with the app alive.
    """
    from squidmip import open_reader

    root, _arrays = squid_dataset
    reader = open_reader(str(root))
    mgr = ViewerManager(reader, reader.metadata)
    try:
        yield mgr
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()
        gc.collect()
        qapp.processEvents()


def _opens():
    """Every window-open record the app has written, newest last."""
    return [m for m in METRICS if m.operator == WINDOW_OPEN]


def _loaded(qapp, win):
    """Wait until this window's mosaic is on screen and its settings have been applied."""
    assert _drain_until(qapp, lambda: win._settings_applied, timeout=30), (
        f"view {win.window_id} never finished loading")
    return win


def test_opening_a_region_window_is_measured(qapp, manager):
    """THE point of the increment: a window open leaves a record with a number on it."""
    win = manager.open([REGIONS[0]])
    _loaded(qapp, win)

    assert len(_opens()) == 1, "opening a window recorded nothing, so loading time is still unmeasured"
    m = _opens()[0]
    assert m.outcome == OK
    assert REGIONS[0] in m.target, "the record does not say which window it describes"
    assert m.first_paint_seconds is not None, (
        "no first paint was recorded, so the record cannot distinguish 'slow to show anything' "
        "from 'quick to show, slow to finish' — which is the distinction being investigated")
    assert m.seconds >= m.first_paint_seconds, (
        "the whole open finished before its first layer appeared; the two ends are crossed")


def test_the_clock_covers_building_the_window_not_only_loading_it(qapp, manager, napari_pane_stub,
                                                                  monkeypatch):
    """Constructing the napari pane is time the user waits, and on the complaint being investigated
    it may be most of it (the backlog plan records 91 MB to 419 MB just opening a 9-well plate). A
    clock started after the window exists would report every part of the wait except that one, and
    would keep reporting a healthy number while the app got slower to open.

    Asserted as ORDER, not as duration: at the moment the clock is created, no pane has been built
    yet. ADR-0001 forbids the direct form of this assertion, and the order is what makes the number
    honest anyway.
    """
    from squidmip import _measure

    panes_at_start = []
    real = _measure.WindowOpen
    monkeypatch.setattr(_measure, "WindowOpen",
                        lambda *a, **kw: (panes_at_start.append(len(napari_pane_stub)),
                                          real(*a, **kw))[1])

    win = manager.open([REGIONS[0]])
    _loaded(qapp, win)

    assert len(napari_pane_stub) == 1, "no pane was built, so this test proved nothing"
    assert panes_at_start == [0], (
        "the clock started after the window's pane was built, so it cannot see the cost of "
        "building it")


def test_an_roi_child_open_is_measured_and_names_itself_as_one(qapp, manager):
    """An ROI child reads a corner, a region window reads the region, and the whole ROI-viewport
    design turns on that difference in cost. Two records that read alike cannot show it."""
    parent = manager.open([REGIONS[0]])
    _loaded(qapp, parent)
    child = manager.open_child([REGIONS[0]], roi_bbox=(0.0, 0.0, 5.0, 5.0),
                               parent_id=parent.window_id)
    _loaded(qapp, child)

    targets = [m.target for m in _opens()]
    assert len(targets) == 2
    assert targets[1].startswith("ROI in "), (
        "an ROI child's record is indistinguishable from a whole-region window's")
    assert not targets[0].startswith("ROI in ")


def test_loading_another_mosaic_in_an_open_window_is_not_another_open(qapp, manager):
    """A window that navigates regions, or moves timepoint, runs the same loader again. Recording
    each one as an open would make the log say the user opened six windows when they opened one and
    moved the slider, and every subsequent load is warm — so the average would drift downward on its
    own and report an improvement nobody made."""
    win = manager.open([REGIONS[0], REGIONS[1]])
    _loaded(qapp, win)
    before = len(win._pane.mosaic.added)

    win._load_mosaic(win.current_region())
    assert _drain_until(qapp, lambda: len(win._pane.mosaic.added) > before, timeout=30), (
        "the second load never landed, so this test proved nothing")

    assert len(_opens()) == 1


def test_closing_a_window_that_already_loaded_does_not_record_a_second_open(qapp, manager):
    """Every window is closed eventually, and the close path records the give-up case. Recording
    both ends would double every successful open in the log."""
    win = manager.open([REGIONS[0]])
    _loaded(qapp, win)
    win.close()
    for _ in range(20):
        qapp.processEvents()

    assert len(_opens()) == 1
    assert _opens()[0].outcome == OK, "a completed open was overwritten by its own close"
