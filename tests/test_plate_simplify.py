"""The 2026-08-19 plate/views simplification (Julio's annotated mocks), pinned.

Plate window: the readout strip is a LOG LINE, the selection caption is the status bar, Select
all is a menu action beside the plate's own Cmd/Ctrl-A, and the top bar keeps only the
acquisition name, the layer combo and Open view. Views window: 3D opens in a NEW tab, a spawn
un-minimises the deck, a reveal keeps a maximised deck maximised, and the operator panel is the
home of the Detect row.
"""

from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede the Qt import

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QWidget  # noqa: E402

import squidxplorer._viewer as V  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_view_deck import _tabbed_plate  # noqa: E402
from .test_viewer import qapp  # noqa: E402,F401  (fixture)


# --- the readout is a log line -------------------------------------------------------------------

def test_a_readout_message_lands_in_the_log_records(qapp, caplog):
    """Julio: "log messages that show around the GUI and not in the log". Every setText that
    used to paint the strip is a log record now — INFO for status, WARNING for refusals."""
    win = V.PlateWindow(None)
    try:
        with caplog.at_level(logging.INFO):
            win._readout.setText("scanning acquisition …")
        assert any(r.getMessage() == "scanning acquisition …" and r.levelno == logging.INFO
                   for r in caplog.records), "a status sentence never reached the log"

        caplog.clear()
        with caplog.at_level(logging.INFO):
            win._readout.setText("open an acquisition first")
        assert any(r.levelno == logging.WARNING and "open an acquisition" in r.getMessage()
                   for r in caplog.records), "a refusal logged below WARNING (or not at all)"

        # `.text()` still answers — tools/gates.py and _gui_commands read the last sentence.
        assert win._readout.text() == "open an acquisition first"

        caplog.clear()
        with caplog.at_level(logging.INFO):
            win._readout.setText("open an acquisition first")   # the same sentence again
        assert not caplog.records, "an idempotent setText spammed a duplicate log line"
    finally:
        win.close()


def test_the_readout_is_not_a_widget_any_more(qapp):
    """Pinned as class identity: a QLabel readout is the strip coming back."""
    win = V.PlateWindow(None)
    try:
        assert not isinstance(win._readout, QWidget), "the readout strip is back as a widget"
        assert not hasattr(win._readout, "setStyleSheet"), (
            "something can style the readout; a shim that quacks like a widget will be re-added "
            "to a layout sooner or later")
    finally:
        win.close()


# --- the top row keeps three things ---------------------------------------------------------------

def test_the_top_bar_keeps_name_layer_and_open_view_only(qapp):
    win = V.PlateWindow(None)
    try:
        for gone in ("_select_all_btn", "_plate_copy_lut_btn", "_plate_paste_lut_btn"):
            assert not hasattr(win, gone), f"{gone} is back on the top bar"
        bar = win._plate_title.parentWidget()
        assert win._open_sel_btn.parentWidget() is bar, "Open view left the plate title bar"
        assert win._view_combo.parentWidget() is bar
        # The selection caption lives in the STATUS BAR now, not in a row of its own.
        assert win._selection_label.parentWidget() is win.statusBar()
    finally:
        win.close()


def test_select_all_is_a_menu_action_and_the_shortcut_survives(qapp, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    try:
        win.ingest(str(root))
        win._select_all_act.trigger()
        assert win._selected_regions == list(win._order), (
            "View > Select All Wells did not select every occupied well")
        # The existing Cmd/Ctrl-A lives on the plate widget itself; verified at the source so a
        # refactor cannot silently drop the second way in.
        import inspect

        from squidxplorer._plate_overview import PlateOverview

        src = inspect.getsource(PlateOverview.keyPressEvent)
        assert "select_all" in src and "Key_A" in src, "the plate lost its Cmd/Ctrl-A select-all"
    finally:
        win.close()


# --- views window: 3D is a new tab, the deck respects its window states ---------------------------

def test_3d_opens_in_a_new_tab_and_the_2d_view_stays(qapp, napari_pane_stub, squid_dataset,
                                                     monkeypatch):
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    v = views[0]
    try:
        # The no-ROI branch bricks the WHOLE region in the child's own canvas (2026-08-19);
        # the single-FOV popout is only the fallback for a window without one.
        opened = []
        import squidxplorer._napari3d as N3D
        monkeypatch.setattr(N3D, "open_native_3d",
                            lambda *a, **k: opened.append((a, k)) or None)
        before = deck.count()
        v._open_3d()
        qapp.processEvents()
        assert deck.count() == before + 1, "3D replaced the view instead of opening a new tab"
        child = deck.current_page()
        assert child is not v, "the 3D tab is not the current page"
        assert child.parent_id == v.window_id
        assert child.display_name.startswith("3D ·")
        from squidxplorer._brick_view import BrickedVolume

        assert isinstance(child._native3d, BrickedVolume), \
            "the new tab never opened its in-window volume"
        assert not opened, "3D fell back to the single-FOV popout despite an in-window canvas"
        assert v._render_mode == "2d", "the 2D view was flipped to 3D under the user"
        assert child._render_mode == "3d"
    finally:
        shutdown_plate_window(qapp, win)


def test_a_view_opened_while_the_deck_is_minimised_restores_it(qapp, napari_pane_stub,
                                                               squid_dataset):
    """One measured shape of "tabs do not come back": `show()` does not un-minimise, so a view
    opened into a minimised deck landed in a window the user could not see."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        views[0].collapse()
        qapp.processEvents()
        assert deck.isMinimized(), "collapse() did not minimise the deck holding the tab"
        second = mgr.open([list(win._order)[0]])
        qapp.processEvents()
        assert second is not None
        assert not deck.isMinimized(), "a new view was docked into a deck that stayed minimised"
    finally:
        shutdown_plate_window(qapp, win)


def test_reveal_keeps_a_maximised_deck_maximised(qapp, napari_pane_stub, squid_dataset):
    """`showNormal()` unconditionally also DE-maximises: every programmatic reveal yanked a
    full-screen views window back to its normal size."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        deck.setWindowState(Qt.WindowMaximized)
        qapp.processEvents()
        views[0].reveal()
        qapp.processEvents()
        assert deck.windowState() & Qt.WindowMaximized, (
            "reveal() de-maximised the deck; showNormal must be reserved for a minimised one")
    finally:
        shutdown_plate_window(qapp, win)


# --- the operator panel is the home of the per-window controls -----------------------------------

def test_the_operator_panel_adopts_the_panes_detect_row(qapp, napari_pane_stub, squid_dataset):
    """The Detect row moves out of the window body into the operator surface (the dock)."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    v = views[0]
    try:
        row = QWidget()
        v._pane.detect_row = row
        v._op_panel = None                       # rebuild: the stub pane had no row at spawn
        panel = v.operator_panel()
        p = row.parentWidget()
        while p is not None and p is not panel:
            p = p.parentWidget()
        assert p is panel, "the pane's Detect row was not adopted into the operator panel"
        # The panel carries the whole per-window operator surface.
        assert v._op_combo is not None and v._save_chk is not None
        assert v._btn_controls is not None
    finally:
        shutdown_plate_window(qapp, win)
