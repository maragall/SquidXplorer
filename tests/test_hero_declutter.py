"""Hero declutter (team feedback, 2026-08-25): the image and the plate view are the hero
features. Operator controls and the non-essential view chips start COLLAPSED behind slim
summon affordances (the app's grip pattern), the log slot starts collapsed, and an ordinary
open-and-preview session produces a SHORT log. State is per view and session-scoped: no
prefs file.
"""

from __future__ import annotations

import logging

import pytest
from qtpy.QtWidgets import QApplication

import squidxplorer._viewer as V
from tests.conftest import shutdown_plate_window

pytestmark = pytest.mark.usefixtures("qapp")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _drain_until(app, pred, timeout=60):
    import time

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        if pred():
            return True
    return False


def _open_view(qapp, root, n_views=1):
    win = V.PlateWindow(None)
    win.ingest(str(root))
    mgr = win._viewer_manager
    order = list(win._order)
    views = [mgr.open([order[i % len(order)]]) for i in range(n_views)]
    _drain_until(qapp, lambda: all(v._pane is not None for v in views), timeout=10)
    return win, views


# --- the operator surface starts collapsed --------------------------------------------------


def test_the_operator_surface_starts_collapsed_behind_a_summon_bar(qapp, napari_pane_stub,
                                                                   squid_dataset):
    root, _ = squid_dataset
    win, views = _open_view(qapp, root)
    try:
        v = views[0]
        fold = v._operators_fold
        assert fold.collapsed, "the operator surface must start collapsed"
        assert fold.grip.isVisibleTo(v), "the summon affordance is not on screen"
        panel = v.operator_panel()
        assert not panel.isVisibleTo(v), "the operator panel is showing while collapsed"
        fold.grip.click()
        assert not fold.collapsed
        assert panel.isVisibleTo(v), "summon did not reveal the operator panel"
        assert v._btn_preview.isVisibleTo(v)
        assert v._btn_run_plate.isVisibleTo(v)
        fold.grip.click()
        assert fold.collapsed and not panel.isVisibleTo(v), (
            "collapse did not fold the operator surface back behind the affordance")
    finally:
        shutdown_plate_window(qapp, win)


def test_the_fold_state_is_per_view(qapp, napari_pane_stub, squid_dataset):
    root, _ = squid_dataset
    win, views = _open_view(qapp, root, n_views=2)
    try:
        a, b = views
        a._operators_fold.grip.click()
        assert not a._operators_fold.collapsed
        assert b._operators_fold.collapsed, "expanding one view expanded another"
    finally:
        shutdown_plate_window(qapp, win)


def test_inserting_a_param_panel_summons_the_operator_surface(qapp, napari_pane_stub,
                                                              squid_dataset):
    """A parameter panel inserted into a collapsed fold would be invisible: the insert summons."""
    root, _ = squid_dataset
    win, views = _open_view(qapp, root)
    try:
        v = views[0]
        assert v._operators_fold.collapsed
        combo = v._op_combo
        combo.setCurrentIndex(next(k for k in range(combo.count())
                                   if combo.itemData(k) == "stitch"))
        v._show_operator_controls()
        qapp.processEvents()
        assert v._inserted_panel is not None
        assert not v._operators_fold.collapsed, (
            "inserting the operator's controls left the fold collapsed over them")
    finally:
        shutdown_plate_window(qapp, win)


# --- the chips fold to the essentials -------------------------------------------------------


def test_only_the_2d_3d_essentials_show_until_the_controls_are_summoned(qapp, napari_pane_stub,
                                                                        squid_dataset):
    root, _ = squid_dataset
    win, views = _open_view(qapp, root)
    try:
        v = views[0]
        fold = v._controls_fold
        assert fold.collapsed, "the view controls must start collapsed"
        assert v._btn_2d.isVisibleTo(v), "2D is an essential and must stay visible"
        assert v._btn_3d.isVisibleTo(v), "3D is an essential and must stay visible"
        for name in ("_btn_focus", "_btn_record", "_btn_png", "_btn_fovs",
                     "_btn_copy_luts", "_btn_paste_luts"):
            assert not getattr(v, name).isVisibleTo(v), f"{name} is showing while collapsed"
        fold.grip.click()
        for name in ("_btn_focus", "_btn_record", "_btn_png", "_btn_fovs",
                     "_btn_copy_luts", "_btn_paste_luts"):
            chip = getattr(v, name)
            assert chip.isVisibleTo(v), f"summon did not reveal {name}"
    finally:
        shutdown_plate_window(qapp, win)


def test_summon_controls_expands_both_folds(qapp, napari_pane_stub, squid_dataset):
    """The one entry GATE 3 (and any headless driver) uses to reach every control."""
    root, _ = squid_dataset
    win, views = _open_view(qapp, root)
    try:
        v = views[0]
        v.summon_controls()
        assert not v._controls_fold.collapsed
        assert not v._operators_fold.collapsed
    finally:
        shutdown_plate_window(qapp, win)


# --- the log slot starts collapsed ----------------------------------------------------------


def test_the_log_starts_collapsed_and_view_menu_summons_it(qapp, napari_pane_stub,
                                                           squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    try:
        assert win._log_panel.collapsed, "the log must start collapsed (quiet by default)"
        win.show_log()
        assert not win._log_panel.collapsed, "View > Log did not summon the log"
    finally:
        shutdown_plate_window(qapp, win)


def test_a_hosted_log_keeps_its_height_cap_across_a_collapse_cycle(qapp, napari_pane_stub,
                                                                   squid_dataset):
    """Expanding a hosted log must respect the 3/4-of-plate-slot cap, not grow unbounded."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    mgr = win._viewer_manager
    mgr.tabbed_views = True
    views = [mgr.open([list(win._order)[0]])]
    _drain_until(qapp, lambda: views[0]._pane is not None, timeout=10)
    try:
        box = win._plate_slot_box
        cap = int(box.PLATE_SLOT_PX * 3 / 4)
        panel = win._log_panel
        panel.set_collapsed(False)
        assert panel.maximumHeight() == cap, (
            "an expanded hosted log lost the 3/4-of-plate-slot cap")
        panel.set_collapsed(True)
        panel.set_collapsed(False)
        assert panel.maximumHeight() == cap, (
            "a collapse cycle lost the hosted height cap")
    finally:
        shutdown_plate_window(qapp, win)


# --- the log diet ---------------------------------------------------------------------------

#: An ordinary open-and-preview session's ceiling of INFO lines. The point is a SHORT log:
#: facts a user acts on, not narration (team feedback 2026-08-25). Measured 3 on the fixture
#: after the diet (scan status, wells-loaded status, window-open measure line); the headroom
#: covers once-per-process lines a real pane adds (GPU ceiling, color provenance). Raise this
#: only with a reason written next to the new number.
ORDINARY_SESSION_INFO_CAP = 8


def test_an_ordinary_open_and_preview_session_produces_a_short_log(qapp, napari_pane_stub,
                                                                   squid_dataset):
    root, _ = squid_dataset
    records: list[logging.LogRecord] = []

    class _Spy(logging.Handler):
        def emit(self, record):
            records.append(record)

    spy = _Spy(level=logging.DEBUG)
    logger = logging.getLogger("squid.xplorer")
    logger.addHandler(spy)
    try:
        win = V.PlateWindow(None)
        win.ingest(str(root))
        _drain_until(qapp, lambda: win._overview is not None
                     and len(win._overview._tiles) >= 1, timeout=60)
        mgr = win._viewer_manager
        views = [mgr.open([list(win._order)[0]])]
        _drain_until(qapp, lambda: views[0]._pane is not None, timeout=10)
        for _ in range(20):
            qapp.processEvents()
        shutdown_plate_window(qapp, win)
    finally:
        logger.removeHandler(spy)
    info = [r for r in records if r.levelno == logging.INFO]
    lines = "\n".join(f"  {r.name}: {r.getMessage()}" for r in info)
    assert len(info) <= ORDINARY_SESSION_INFO_CAP, (
        f"an ordinary open-and-preview session logged {len(info)} INFO line(s), over the "
        f"cap of {ORDINARY_SESSION_INFO_CAP}:\n{lines}")
