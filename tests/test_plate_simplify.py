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

def test_the_operator_panel_carries_no_detect_row(qapp, napari_pane_stub, squid_dataset):
    """Absence pin (2026-08-24): the Detect surface was shelved with spot/cellpose. A stray
    ``detect_row`` on a pane is NOT adopted, and the per-window operator surface survives it."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    v = views[0]
    try:
        row = QWidget()
        v._pane.detect_row = row                 # a foreign attribute nothing reads any more
        v._op_panel = None                       # rebuild
        panel = v.operator_panel()
        p = row.parentWidget()
        while p is not None and p is not panel:
            p = p.parentWidget()
        assert p is None, "the shelved Detect row was adopted into the operator panel"
        # The panel still carries the whole per-window operator surface.
        assert v._op_combo is not None and v._save_chk is not None
        assert v._btn_controls is not None
    finally:
        shutdown_plate_window(qapp, win)


def test_the_operator_panel_docstring_names_its_home(qapp, napari_pane_stub, squid_dataset):
    """The Detect-row adoption above holds wherever the panel lives; its home is the view's own
    LEFT column now (Julio: "The operators for this window row should also be on the left
    vertical dock")."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    v = views[0]
    try:
        panel = v.operator_panel()
        # The panel is a descendant of the VIEW (its left column container), never of the deck's
        # right-edge bulk dock.
        p = panel.parentWidget()
        while p is not None and p is not v:
            p = p.parentWidget()
        assert p is v, "the operator panel is not inside its own view"
    finally:
        shutdown_plate_window(qapp, win)


# --- viewer space: the top toolbar is gone, the controls live in the left column ------------------

def test_the_full_width_top_toolbar_is_gone_and_the_chips_live_in_the_left_column(
        qapp, napari_pane_stub, squid_dataset):
    """Julio (2026-08-19): the 2D/3D·ROI cluster "should be on the left column, where the
    controls are, so that we can free up the viewer space to the top. No need to have a full
    horizontal dock." The builder is pinned ABSENT the repo's way; the chips keep their names
    (tests and GATE 3 find them by attribute) and stay actuatable."""
    import squidxplorer._region_viewer as RV

    assert not hasattr(RV.RegionViewer, "_build_top_row"), "the full-width top row is back"

    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    v = views[0]
    try:
        box = v._view_controls
        assert box is not None
        # Every chip is inside the control block, and the block is inside the VIEW (with the
        # headless pane the left column falls back into the window body — same ancestry).
        for name in ("_btn_2d", "_btn_3d", "_btn_focus", "_btn_record", "_btn_fovs",
                     "_btn_copy_luts", "_btn_paste_luts"):
            chip = getattr(v, name)
            p = chip.parentWidget()
            while p is not None and p is not box:
                p = p.parentWidget()
            assert p is box, f"{name} left the 2D/3D·ROI block"
        p = box.parentWidget()
        while p is not None and p is not v:
            p = p.parentWidget()
        assert p is v, "the control block is not inside its own view"
        # The progress bar stays in the window BODY: a run must stay visible regardless of the
        # left column's state.
        assert v._op_progress is not None and v._op_progress.isHidden()
    finally:
        shutdown_plate_window(qapp, win)


# --- the two-button LUT clipboard and the paste-parity rule ---------------------------------------

def test_a_lut_paste_reaches_the_plate_and_the_two_agree(qapp, napari_pane_stub, squid_dataset):
    """Julio (2026-08-19): "Make sure that we don't have the issue where we copy luts and plate
    contrast is different from the window contrast." The PASTE is the one event the plate
    follows (a drag still leaves it alone — pinned in test_plate_follows_windows): after a
    paste, each channel's plate window equals the view's own contrast_limits, through the
    FOLLOW path (never the manual latch), and the plate's channel COLOURS are untouched so a
    stain-LUT channel keeps its LUT rendering."""
    import numpy as np

    import squidxplorer._lut_clipboard as LC

    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    v = views[0]
    try:
        names = [c["name"] for c in win._meta["channels"]]
        mosaic = v._pane.mosaic
        for ch in names:
            if mosaic.find("raw", ch) is None:
                mosaic.add_mosaic("raw", ch, np.full((8, 8), 500, dtype=np.uint16))
        colors_before = [win._overview.channel_rgb(i) for i in range(len(names))]
        LC.CLIPBOARD.clear()
        LC.CLIPBOARD.update({ch: {"clim": (120.0, 900.0), "cmap": None, "rgb": None, "on": None}
                             for ch in names})
        # NO processEvents between paste and assert: the paste and the plate's follow are
        # SYNCHRONOUS (a direct signal on one thread), and pumping events here lets the view's
        # deferred region load clear and rebuild the very layers under test.
        v._paste_luts()
        for i, ch in enumerate(names):
            assert mosaic.contrast(ch) == (120.0, 900.0), f"the paste never landed on {ch}"
            assert win._overview._contrast.window(i) == (120.0, 900.0), (
                f"the plate's {ch} window differs from the pasted view's")
            assert not win._overview._contrast.is_manual(i), (
                "a paste latched the plate manual; it must ride the FOLLOW path")
        assert [win._overview.channel_rgb(i) for i in range(len(names))] == colors_before, (
            "a paste moved the plate's channel colours; a stain-LUT channel would lose its "
            "LUT rendering")
    finally:
        shutdown_plate_window(qapp, win)


def test_an_empty_clipboard_paste_is_a_refusal_not_a_noop(qapp, napari_pane_stub, squid_dataset):
    import squidxplorer._lut_clipboard as LC

    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    v = views[0]
    try:
        LC.CLIPBOARD.clear()
        said = []
        v._say = said.append
        v._paste_luts()
        assert said and "empty" in said[-1], "an empty paste said nothing"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_plate_paste_button_applies_the_clipboard(qapp, squid_dataset):
    """The one plate-side LUT control: paste latches the clipboard's clim per channel name."""
    import numpy as np

    import squidxplorer._viewer as V
    from squidxplorer import _lut_clipboard

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    try:
        win.ingest(str(root))
        ch0 = win._meta["channels"][0]["name"]
        _lut_clipboard.CLIPBOARD.clear()
        win._paste_luts_onto_plate()
        assert "empty" in win._readout.text()
        _lut_clipboard.CLIPBOARD[ch0] = {"clim": (12.0, 345.0), "on": True}
        win._paste_luts_onto_plate()
        assert win._overview.channel_windows()[0] == (12.0, 345.0)
        # a deliberate paste latches manual: streamed tiles cannot stomp it
        win._overview._contrast.add(0, np.full((8, 8), 999, dtype=np.uint16))
        assert win._overview.channel_windows()[0] == (12.0, 345.0)
    finally:
        _lut_clipboard.CLIPBOARD.clear()
        win._stop_worker()
        win.close()


def test_copy_then_paste_round_trips_between_two_views(qapp, napari_pane_stub, squid_dataset):
    """Two buttons, one dict: copy in one view, paste in another, same contrast."""
    import numpy as np

    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    a, b = views
    try:
        names = [c["name"] for c in win._meta["channels"]]
        for v in (a, b):
            for ch in names:
                if v._pane.mosaic.find("raw", ch) is None:
                    v._pane.mosaic.add_mosaic("raw", ch,
                                              np.full((8, 8), 500, dtype=np.uint16))
        a._pane.mosaic.set_contrast(names[0], 33.0, 333.0)
        a._copy_luts()
        b._paste_luts()
        assert b._pane.mosaic.contrast(names[0]) == (33.0, 333.0), (
            "the copied window never reached the second view")
    finally:
        shutdown_plate_window(qapp, win)


# --- reconstructed color is a View-menu switch ----------------------------------------------------

def _color_recorded_gray_acq(tmp_path):
    """A gray-recorded color acquisition: 2-D BMP planes plus the mosaic sidecar calling it RGB."""
    import numpy as np
    from PIL import Image

    root = tmp_path / "acq_gray_rgb"
    mv = root / "0" / "mosaic_view"
    mv.mkdir(parents=True)
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 255, (16, 16), dtype=np.uint8)).save(
        root / "0" / "manual_0_0_BF_LED_matrix_full.bmp")
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.418\nz_stack:\n  nz: 1\ntime_series:\n  nt: 1\n")
    (root / "acquisition_channels.yaml").write_text(
        "channels:\n- name: BF LED matrix full\n  display_color: '#FFFFFF'\n")
    t = np.tile(np.linspace(0.2, 1.0, 200), (120, 1))
    png = np.stack([255 * t ** 0.3, 255 * t, 255 * t ** 0.6], axis=-1).astype(np.uint8)
    Image.fromarray(png).save(mv / "mosaic_2um_x.png")
    (mv / "mosaic_2um.yaml").write_text(
        "rgb_channel_names:\n- 20x BF LED matrix full\nrgb_view_files:\n- mosaic_2um_x.png\n")
    return root


def test_reconstructed_color_menu_action_round_trips(qapp, tmp_path):
    """View > Reconstructed Color re-ingests under the flipped flag: off is honest gray (no
    stain LUT, yaml color), on brings the estimated LUT and its label back."""
    from squidxplorer import _stain

    root = _color_recorded_gray_acq(tmp_path)
    win = V.PlateWindow(None)
    try:
        win.ingest(str(root))
        assert win._recon_color_act.isChecked(), "reconstruction must default ON"
        assert win._meta["channels"][0].get("display_lut") is not None
        assert "color: estimated" in win._readout.text(), (
            "the ingest status line never named the derived color")

        win._recon_color_act.trigger()           # off: falls all the way to gray
        chs = win._meta["channels"]
        assert len(chs) == 1 and chs[0].get("display_lut") is None
        assert chs[0].get("color_source") is None
        assert chs[0]["display_color"] == "#FFFFFF"
        assert "color:" not in win._readout.text()
        assert not _stain.reconstruction_enabled()

        win._recon_color_act.trigger()           # back on: the LUT and its label return
        assert win._meta["channels"][0].get("display_lut") is not None
        assert "color: estimated" in win._readout.text()
    finally:
        _stain.set_reconstruction(None)
        shutdown_plate_window(qapp, win)
