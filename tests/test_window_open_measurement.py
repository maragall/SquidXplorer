"""Opening a window is MEASURED: the clock is wired to the real open, not only to a class."""

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
    """These tests read the process-wide log the app writes; cleared per test so opens don't leak across tests."""
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
    win.close()
    for _ in range(20):
        qapp.processEvents()
    assert len(_opens()) == 1 and _opens()[0].outcome == OK, "the close overwrote or doubled the open"


def test_the_clock_covers_building_the_window_not_only_loading_it(qapp, manager, napari_pane_stub,
                                                                  monkeypatch):
    """Constructing the napari pane is time the user waits (91 MB to 419 MB just opening a 9-well plate); a clock started after the window exists would miss"""
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
    """An ROI child reads a corner, a region window reads the region; two records that read alike cannot show that difference in cost."""
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
    """A window that navigates regions, or moves timepoint, runs the same loader again; recording each as an open would make the log claim windows the user"""
    win = manager.open([REGIONS[0], REGIONS[1]])
    _loaded(qapp, win)
    landed = []
    for ly in list(win._pane._viewer.layers):
        ly.events.data.connect(lambda e: landed.append(1))
    win._pane._viewer.layers.events.inserted.connect(lambda e: landed.append(1))

    win._load_mosaic(win.current_region())
    assert _drain_until(qapp, lambda: bool(landed), timeout=30), (
        "the second load never landed, so this test proved nothing")

    assert len(_opens()) == 1


