"""Opening a window is MEASURED: the clock is wired to the real open, not only to a class.

Separate from tests/test_run_measurement.py, which pins the clock itself against a hand-moved
clock with no Qt. This file pins the WIRING: that a real ViewerManager.open produces a record,
that an ROI child is distinguishable from a whole-region open, and that things a window does
after opening (loading another region, closing) do not each record another open.

Per docs/adr/0001-ci-gates-work-not-time.md nothing here asserts a duration — only that a number
was recorded, and recorded once.

Not tested, stated rather than left to be found: whether first paint is taken where the layer is
added or where the worker emits it. Both placements produce a number; telling them apart needs a
real interface under load.
"""

from __future__ import annotations

import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidxplorer._measure import METRICS, OK, WINDOW_OPEN  # noqa: E402
from squidxplorer._region_viewer import ViewerManager  # noqa: E402

from .conftest import REGIONS  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixture)


@pytest.fixture(autouse=True)
def _own_the_metrics_log():
    """These tests read the process-wide log the app writes; cleared per test so opens don't
    leak across tests."""
    METRICS.clear()
    yield
    METRICS.clear()


@pytest.fixture
def manager(qapp, napari_pane_stub, squid_dataset):
    """A real ViewerManager over the real reader, with no PlateWindow in the way."""
    from squidxplorer import open_reader

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
    win = manager.open([REGIONS[0]])
    _loaded(qapp, win)

    assert len(_opens()) == 1, "opening a window recorded nothing, so loading time is still unmeasured"
    m = _opens()[0]
    assert m.outcome == OK
    assert REGIONS[0] in m.target, "the record does not say which window it describes"
    assert m.first_paint_seconds is not None, (
        "no first paint was recorded, so the record cannot distinguish 'slow to show anything' "
        "from 'quick to show, slow to finish'")
    assert m.seconds >= m.first_paint_seconds, (
        "the whole open finished before its first layer appeared; the two ends are crossed")


def test_the_clock_covers_building_the_window_not_only_loading_it(qapp, manager, napari_pane_stub,
                                                                  monkeypatch):
    """Constructing the napari pane is time the user waits (91 MB to 419 MB just opening a
    9-well plate); a clock started after the window exists would miss it entirely.

    Asserted as order, not duration, per ADR-0001: at the moment the clock is created, no pane
    has been built yet.
    """
    from squidxplorer import _measure

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
    """An ROI child reads a corner, a region window reads the region; two records that read
    alike cannot show that difference in cost."""
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
    """A window that navigates regions, or moves timepoint, runs the same loader again;
    recording each as an open would make the log claim windows the user never opened."""
    win = manager.open([REGIONS[0], REGIONS[1]])
    _loaded(qapp, win)
    # The reload lands as in-place data replacement (the reuse path) or an insertion; both fire
    # events on the REAL model, which is the observable now that the stub's add-log is gone.
    landed = []
    for ly in list(win._pane._viewer.layers):
        ly.events.data.connect(lambda e: landed.append(1))
    win._pane._viewer.layers.events.inserted.connect(lambda e: landed.append(1))

    win._load_mosaic(win.current_region())
    assert _drain_until(qapp, lambda: bool(landed), timeout=30), (
        "the second load never landed, so this test proved nothing")

    assert len(_opens()) == 1


def test_closing_a_window_that_already_loaded_does_not_record_a_second_open(qapp, manager):
    """Every window is closed eventually; recording both ends would double every successful
    open in the log."""
    win = manager.open([REGIONS[0]])
    _loaded(qapp, win)
    win.close()
    for _ in range(20):
        qapp.processEvents()

    assert len(_opens()) == 1
    assert _opens()[0].outcome == OK, "a completed open was overwritten by its own close"
