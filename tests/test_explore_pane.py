"""Pane 3 — the EXPLORATION pane, driven headless through the real widgets.

What is gated here, in the owner's words:

* "The right pane is essentially a COPY of the central pane, but it occurs on a SUBSET." — so
  the tab hosts a viewer built by pane 2's OWN constructor, EMBEDDED in the tab. Never a second
  viewer implementation, and never a separate top-level napari window ("the control well is
  opening a separate napari window where it should have actually popped up in the exploration
  pane").
* "Preview runs can open a TAB on the exploration pane so that they look at how it is behaving."
  — and several coexist, for comparison.
* "There should be a slider under, and there should be controls that we can run on the subsets of
  FOVs that we're passing into the exploration pane."
* Minerva Author runs from here, on THIS TAB's subset.
* "What we set in our exploration pane should always be visualizing."

The napari canvas needs OpenGL, which the offscreen Qt platform does not provide, so the tab's
viewer is a recording stub — exactly the technique ``tests/test_viewer.py`` already uses for
ndviewer_light. What is under test is the SEAM: that the pane asks for a viewer, embeds it,
aims it at its own subset, and puts real layers on it as results arrive.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede the PyQt import

import numpy as np
import pytest

pytest.importorskip("PyQt5")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

import time  # noqa: E402

from PyQt5.QtCore import Qt, pyqtSignal  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel, QSlider, QWidget  # noqa: E402

from squidmip import _explore as E  # noqa: E402
from squidmip import _viewer as V  # noqa: E402


# --- stubs -------------------------------------------------------------------------------------

class _StubMosaic:
    """Stands in for ``MosaicLayers``. Records what was put on the canvas."""

    def __init__(self):
        self.added = []        # (op, channel, shape, bbox_um)
        self.removed = []
        self.shown = []

    def add_mosaic(self, op, channel, data, **kw):
        self.added.append((op, channel, np.asarray(data).shape, kw.get("bbox_um")))
        return object()

    def remove_op(self, op):
        self.removed.append(op)
        return []

    def show_op(self, op):
        self.shown.append(op)
        return []

    def ops(self):
        return list(dict.fromkeys(op for op, _c, _s, _b in self.added))


class _StubPane(QWidget):
    """Stands in for ``MosaicPane`` — a QWidget with the same surface pane 3 drives."""

    def __init__(self):
        super().__init__()
        self.mosaic = _StubMosaic()
        self.ok = True
        self.failure = None
        self.said = []

    def say(self, text):
        self.said.append(text)


class _StubDetail(QWidget):
    """Minimal ndviewer_light stand-in (the real one segfaults offscreen)."""

    contrastChanged = pyqtSignal(int, float, float)

    def __init__(self):
        super().__init__()
        self._fov_slider = QSlider(Qt.Horizontal, self)
        self.acquisitions = []

    def start_acquisition(self, channels, nz, h, w, labels, pixel_size_um=None, dz_um=None):
        self.acquisitions.append(list(labels))
        self._fov_slider.setMaximum(max(0, len(labels) - 1))

    def register_images_bulk(self, entries):
        pass

    def register_image(self, *a, **k):
        pass

    def go_to_well_fov(self, *a, **k):
        pass


def _drain(app, win, timeout=30):
    """Let a run finish and its thread actually exit — `_busy()` stays True while it drains."""
    t0 = time.time()
    while win._busy() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    return not win._busy()


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)
    return app


@pytest.fixture
def panes(monkeypatch):
    """Every exploration tab gets a recording stub viewer instead of a real napari window."""
    made = []

    def _make(self):
        p = _StubPane()
        made.append(p)
        return p, "napari", ""

    monkeypatch.setattr(V.PlateWindow, "_make_explore_viewer", _make)
    monkeypatch.setattr(V.PlateWindow, "_make_detail_viewer", lambda self: _StubDetail())
    return made


@pytest.fixture
def win(qapp, panes, squid_dataset):
    root, _ = squid_dataset
    w = V.PlateWindow(None)
    w.ingest(str(root))
    yield w
    w.close()


# --- the tab IS a viewer, embedded ---------------------------------------------------------------

def test_exploration_tab_hosts_a_viewer_on_its_own_subset(win, panes):
    key = win.open_exploration_tab(["B3"])
    tab = win._op_tabs[key]
    assert tab.viewer is not None, "an exploration tab with no viewer is a control strip, not a pane"
    assert tab.viewer in panes


def test_the_tabs_viewer_is_embedded_never_a_separate_window(win):
    """Julio: 'the control well is opening a SEPARATE NAPARI WINDOW where it should have actually
    popped up in the exploration pane.'

    MUTATION: drop the ``addWidget`` that puts the viewer in the tab's layout and this goes red —
    an unparented QWidget is exactly what shows up as its own floating window.
    """
    key = win.open_exploration_tab(["B3"])
    tab = win._op_tabs[key]
    assert tab.viewer.parent() is not None
    assert tab.viewer.isWindow() is False
    # ...and it is inside THIS tab, not merely inside something.
    assert tab.viewer in tab.findChildren(QWidget)


def test_the_control_well_lands_in_pane_3_not_in_its_own_window(win):
    win.set_control_well("B2")
    tab = win._op_tabs[win.CONTROL_KEY]
    assert win._explore_tabs.indexOf(tab) == 0        # pinned first, in pane 3
    assert tab.viewer is not None
    assert tab.viewer.isWindow() is False
    assert win.control_well() == "B2"


def test_a_viewer_that_cannot_be_built_is_stated_in_the_tab(win, monkeypatch):
    """NO SILENT FAILURE: a tab with no viewer must SAY why, in a sentence the user can read."""
    monkeypatch.setattr(V.PlateWindow, "_make_explore_viewer",
                        lambda self: (None, "ndv", "napari needs OpenGL — no viewer here."))
    key = win.open_exploration_tab(["B3"])
    tab = win._op_tabs[key]
    assert tab.viewer is None
    said = "\n".join(l.text() for l in tab.findChildren(QLabel))
    assert "OpenGL" in said


def test_the_default_viewer_constructor_is_pane_2s(qapp, monkeypatch):
    """One viewer implementation, not two. ``_make_explore_viewer`` must DELEGATE.

    MUTATION: give it its own ``MosaicPane()`` construction and this goes red — which is the
    duplication the brief calls out ('two divergent viewer implementations').
    """
    calls = []
    monkeypatch.setattr(V, "_make_mosaic_pane", lambda: (calls.append(1), (None, "ndv", "x"))[1])
    w = V.PlateWindow.__new__(V.PlateWindow)          # no Qt window needed to test the delegation
    assert V.PlateWindow._make_explore_viewer(w) == (None, "ndv", "x")
    assert calls == [1]


# --- the slider under it ------------------------------------------------------------------------

def test_the_tab_has_a_slider_under_the_viewer_covering_its_regions(win):
    key = win.open_exploration_tab(["B2", "B3"])
    tab = win._op_tabs[key]
    assert isinstance(tab.slider, QSlider)
    assert tab.slider.minimum() == 0
    assert tab.slider.maximum() == 1                  # one position per region of the subset
    assert tab.cursor.region == "B2"


def test_moving_the_slider_aims_the_tabs_own_viewer_at_that_region(win):
    key = win.open_exploration_tab(["B2", "B3"])
    tab = win._op_tabs[key]
    tab.slider.setValue(1)
    assert tab.cursor.region == "B3"
    assert "B3" in tab.region_label.text()


def test_the_slider_never_touches_the_centre_pane(win):
    """Pane 3 is a copy on a SUBSET; scrubbing it must not re-aim pane 2 underneath the user."""
    key = win.open_exploration_tab(["B2", "B3"])
    tab = win._op_tabs[key]
    before = win._mosaic_region
    tab.slider.setValue(1)
    assert win._mosaic_region == before


# --- preview runs open tabs ------------------------------------------------------------------------

def test_a_preview_run_opens_a_tab_in_the_exploration_pane(win, tmp_path):
    n0 = win._explore_tabs.count()
    win.run_operator("mip", regions=["B2"], save=False)
    assert win._explore_tabs.count() == n0 + 1
    tab = win._explore_tabs.currentWidget()
    assert isinstance(tab, V._ExplorationTab)
    assert tab.tab_key.startswith(E.PREVIEW_PREFIX)
    assert tab.regions == ["B2"]
    win._stop_worker()


def test_two_preview_runs_coexist_as_two_tabs(qapp, win):
    """'Multiple preview runs coexist as multiple tabs for comparison.'"""
    n0 = win._explore_tabs.count()
    win.run_operator("mip", regions=["B2"], save=False)
    assert _drain(qapp, win)
    win.run_operator("reference", regions=["B2"], save=False)
    assert _drain(qapp, win)
    assert win._explore_tabs.count() == n0 + 2
    keys = [win._explore_tabs.widget(i).tab_key for i in range(win._explore_tabs.count())]
    assert len(set(keys)) == 2


def test_re_running_the_same_preview_reuses_its_tab(qapp, win):
    n0 = win._explore_tabs.count()
    win.run_operator("mip", regions=["B2"], save=False)
    assert _drain(qapp, win)
    win.run_operator("mip", regions=["B2"], save=False)
    assert _drain(qapp, win)
    assert win._explore_tabs.count() == n0 + 1


def test_a_saved_run_does_not_open_a_preview_tab(win, tmp_path):
    """A run that writes to disk is not a preview; pane 3 must not fill up with them."""
    n0 = win._explore_tabs.count()
    win.run_operator("mip", out_parent=str(tmp_path), regions=["B2"], save=True)
    assert win._explore_tabs.count() == n0
    win._stop_worker()


def test_a_whole_plate_preview_does_not_open_a_preview_tab(win):
    """Pane 3 shows a SUBSET. A plate-wide run belongs to the centre pane."""
    n0 = win._explore_tabs.count()
    win.run_operator("mip", regions=None, save=False)
    assert win._explore_tabs.count() == n0
    win._stop_worker()


# --- results appear AS THEY COMPUTE ----------------------------------------------------------------

def _field(win, region, channels=2, cell=None):
    """A field arriving from the operator worker, exactly as ``tileReady`` emits it."""
    cell = cell or V._CELL
    info = win._fov_index[region]
    raw = np.full((channels, cell, cell), 7, np.uint16)
    return (*info["rc"], info["well_id"], raw, None)


def test_each_computed_region_becomes_its_own_layer_as_it_lands(win):
    """Julio: 'layers don't update in the napari mosaic... you INSTANTIATE AN ACTUAL LAYER.'

    So a run over two regions puts two layer groups on the tab's canvas, the first appearing
    before the second is computed — not one layer whose pixels are replaced at the end.

    MUTATION: file both regions under one op name and this goes red.
    """
    win.run_operator("mip", regions=["B2", "B3"], save=False)
    tab = win._explore_tabs.currentWidget()
    added0 = len(tab.viewer.mosaic.added)
    win._on_tile(*_field(win, "B2"))
    ops_after_first = set(op for op, _c, _s, _b in tab.viewer.mosaic.added[added0:])
    assert ops_after_first == {E.subset_layer_op(V.operator_label("mip"), "B2")}
    win._on_tile(*_field(win, "B3"))
    ops_after_second = set(op for op, _c, _s, _b in tab.viewer.mosaic.added[added0:])
    assert ops_after_second == {E.subset_layer_op(V.operator_label("mip"), "B2"), E.subset_layer_op(V.operator_label("mip"), "B3")}
    win._stop_worker()


def test_a_computed_region_lands_as_one_layer_per_channel(win):
    win.run_operator("mip", regions=["B2"], save=False)
    tab = win._explore_tabs.currentWidget()
    added0 = len(tab.viewer.mosaic.added)
    win._on_tile(*_field(win, "B2", channels=2))
    chans = [c for _op, c, _s, _b in tab.viewer.mosaic.added[added0:]]
    assert len(chans) == 2 and len(set(chans)) == 2
    win._stop_worker()


def test_the_tab_says_how_far_the_run_has_got(win):
    win.run_operator("mip", regions=["B2", "B3"], save=False)
    tab = win._explore_tabs.currentWidget()
    win._on_progress(1, 2)
    assert tab.progress.text() == E.progress_sentence(V.operator_label("mip"), 1, 2)
    win._on_progress(2, 2)
    assert "complete" in tab.progress.text()
    win._stop_worker()


def test_progress_never_says_live(win):
    win.run_operator("mip", regions=["B2"], save=False)
    tab = win._explore_tabs.currentWidget()
    win._on_progress(1, 1)
    assert "live" not in tab.progress.text().lower()
    win._stop_worker()


def test_a_field_for_a_region_outside_the_tab_is_not_drawn_on_it(win):
    """The tab claims a subset. Painting a foreign region on it would make that claim false."""
    win.run_operator("mip", regions=["B2"], save=False)
    tab = win._explore_tabs.currentWidget()
    added0 = len(tab.viewer.mosaic.added)
    win._on_tile(*_field(win, "B3"))
    assert tab.viewer.mosaic.added[added0:] == []
    win._stop_worker()


# --- Minerva, on THIS tab's subset -------------------------------------------------------------

def test_the_minerva_button_is_live_and_scoped_to_the_tabs_subset(win, monkeypatch):
    """The button used to be permanently disabled behind a 'coming with IMA-228' tooltip while
    the export it names had already shipped — a dead control is a silent failure with a label."""
    seen = {}
    monkeypatch.setattr(V.PlateWindow, "run_minerva_export",
                        lambda self, **kw: seen.update(kw))
    key = win.open_exploration_tab(["B3"])
    tab = win._op_tabs[key]
    assert tab.minerva_btn.isEnabled()
    tab.minerva_btn.click()
    assert seen["selection"] == E.subset_selection(["B3"], win._meta["fovs_per_region"])


def test_minerva_from_the_tab_ignores_the_plate_selection(win, monkeypatch):
    """'Controls that we can run on the subsets of FOVs that we're PASSING INTO the exploration
    pane' — the tab's own subset, not whatever is highlighted on the plate."""
    seen = {}
    monkeypatch.setattr(V.PlateWindow, "run_minerva_export",
                        lambda self, **kw: seen.update(kw))
    win._selected_regions = ["B2"]
    key = win.open_exploration_tab(["B3"])
    win._op_tabs[key].minerva_btn.click()
    assert set(r for r, _f in seen["selection"]) == {"B3"}


def test_minerva_refuses_a_region_it_cannot_expand_and_says_so(win, monkeypatch):
    key = win.open_exploration_tab(["B3"])
    tab = win._op_tabs[key]
    monkeypatch.setitem(win._meta, "fovs_per_region", {"B3": []})
    tab.minerva_btn.click()
    assert "B3" in win._readout.text()
    assert win._minerva is None            # nothing was started


# --- pane 3 is a RESULT surface, not a second control surface ------------------------------------

def test_the_side_pane_has_no_operator_launcher_of_its_own(win):
    """Julio: 'we have the controls for the whole dataset on the left, but those controls are
    repeated for the subset on the right pane. Maybe it's not a good idea for there to be
    repetition of knowledge in our user interface.'

    It had already cost this codebase: pane 1 launched operators off the ``_OPERATIONS`` card
    table and pane 3 off ``runnable_operators()``, with different labels and different ``save``
    defaults, and the two drifted in production. There is now ONE catalogue, in pane 1.

    MUTATION: put any operator button back on the tab and this goes red.
    """
    from PyQt5.QtWidgets import QPushButton

    key = win.open_exploration_tab(["B2", "B3"])
    tab = win._op_tabs[key]
    labels = {b.text() for b in tab.findChildren(QPushButton)}
    for op in V.runnable_operators():
        assert not any(V.operator_label(op).lower() in t.lower() for t in labels), (
            f"pane 3 still launches {op!r} — that is the duplicated control surface")


def test_the_left_panel_owns_the_scope_and_lists_every_scope(win):
    assert [win._scope_run.itemText(i) for i in range(win._scope_run.count())] == list(E.RUN_SCOPES)
    assert win._scope_run.currentText() == E.SCOPE_SELECTION


def test_running_at_subset_scope_reads_the_subset_parked_in_pane_3(win, monkeypatch):
    """One owner (pane 3), one reader (pane 1) — not two writers."""
    win.open_exploration_tab(["B2", "B3"])
    win._scope_run.setCurrentText(E.SCOPE_SUBSET)
    assert win.parked_subset() == ["B2", "B3"]
    win.run_operator("mip", save=False)
    assert win._worker is not None
    assert win._worker._regions == ["B2", "B3"]
    win._stop_worker()


def test_running_at_subset_scope_with_an_empty_side_pane_refuses_out_loud(win):
    win._scope_run.setCurrentText(E.SCOPE_SUBSET)
    win.run_operator("mip", save=False)
    assert win._worker is None
    assert "side pane" in win._readout.text()


def test_the_default_scope_keeps_the_historical_selection_behaviour(win, tmp_path):
    win._selected_regions = ["B3"]
    win.run_operator("mip", out_parent=str(tmp_path))
    assert win._worker._regions == ["B3"]
    win._stop_worker()


# --- teardown -------------------------------------------------------------------------------------

def test_closing_a_tab_frees_its_viewer(win):
    """A napari viewer per tab is real memory and a real GL context. Leaking one per Shift-drag
    is how a session dies after twenty selections."""
    key = win.open_exploration_tab(["B3"])
    tab = win._op_tabs[key]
    pane = tab.viewer
    idx = win._explore_tabs.indexOf(tab)
    win._close_op_tab(idx, win._explore_tabs)
    assert tab.viewer is None
    assert pane.parent() is None
