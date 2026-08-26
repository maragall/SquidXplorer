"""Hero declutter (team feedback, 2026-08-25): the image and the plate view are the hero features."""

from __future__ import annotations

import logging

import numpy as np
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


# --- the chips fold to the essentials -------------------------------------------------------


def test_the_2d_button_is_gone_whole_and_a_3d_tab_disables_its_own_3d(qapp, napari_pane_stub,
                                                                      squid_dataset):
    """Julio, 2026-08-25: "There should not be 2D button since we make separate tabs for the 3d view." A 2D tab IS 2D and a 3D tab IS 3D; nothing switches"""
    import squidxplorer._region_viewer as RV

    assert not hasattr(RV.RegionViewer, "_view_roi_2d"), "the 2D chip's handler is back"
    assert not hasattr(RV.RegionViewer, "_set_ndisplay"), (
        "the in-place mode switch is back")
    root, _ = squid_dataset
    win, views = _open_view(qapp, root)
    try:
        v = views[0]
        assert not hasattr(v, "_btn_2d"), "the 2D button is back"
        v.note_volume_tab()
        assert not v._btn_3d.isEnabled(), "a 3D tab still offers to open 3D from itself"
        assert v._btn_3d.toolTip(), "a disabled chip must say why"
    finally:
        shutdown_plate_window(qapp, win)


# --- the log slot starts collapsed ----------------------------------------------------------


# --- the collapsed log is a REAL band, never a clipped sliver -------------------------------
# Julio, live GUI 2026-08-25: "Can't see log. Blank frame." The collapsed cap was frozen at
# construction-time sizeHint, BEFORE adopt_status_row grew the panel (header + memory/run
# bars), so the band rendered as a clipped sliver ("2%" cut mid-label) with no reachable
# summon toggle.


# --- napari-native chrome is minimized ------------------------------------------------------


def test_native_chrome_is_minimized(qapp):
    """The pure half of the pane's chrome diet, proven on a plain Qt window shaped like the embedded napari one: the status bar hides, dock title bars slim"""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QComboBox, QDockWidget, QFormLayout, QLabel, QMainWindow, QSlider, QStatusBar, QWidget,
    )

    from squidxplorer._napari_pane import NATIVE_HIDDEN_ROWS, minimize_native_chrome

    win = QMainWindow()
    win.setStatusBar(QStatusBar())
    dock = QDockWidget("layer controls", win)
    dock.setWidget(QWidget())
    win.addDockWidget(Qt.LeftDockWidgetArea, dock)
    controls = QWidget()
    form = QFormLayout(controls)
    keep_label, keep_field = QLabel("contrast limits:"), QSlider()
    form.addRow(keep_label, keep_field)
    folded = {}
    for text in NATIVE_HIDDEN_ROWS:
        lab, fld = QLabel(text), QComboBox()
        form.addRow(lab, fld)
        folded[text] = (lab, fld)
    dock.widget().setLayout(QFormLayout())  # not the controls form; just a body
    win.show()
    qapp.processEvents()

    hidden = minimize_native_chrome(win, [controls])
    qapp.processEvents()

    assert not win.statusBar().isVisibleTo(win), "napari's status bar is still showing"
    tb = dock.titleBarWidget()
    assert tb is not None and tb.maximumHeight() == 0, (
        "the dock still spends a title bar")
    for text, (lab, fld) in folded.items():
        assert not lab.isVisibleTo(controls) and not fld.isVisibleTo(controls), (
            f"the {text!r} row is still showing")
    assert keep_label.isVisibleTo(controls) and keep_field.isVisibleTo(controls), (
        "a kept layer-controls row was hidden too"
    )
    assert hidden, "nothing was reported hidden; the inventory must be named"
    assert form.verticalSpacing() <= 2, "the form still spends vertical spacing between rows"


def test_the_layer_controls_diet_keeps_at_most_the_three_touched_rows(qapp):
    """Julio, live 2026-08-25: "Layer controls, too much height." The resting blade shows ONLY what a life-science user touches: contrast limits,"""
    from qtpy.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget

    from squidxplorer._napari_pane import hide_native_rows

    napari_image_rows = (            # the probe's inventory of QtImageControls, 2026-08-25
        "opacity:", "blending:", "contrast limits:", "auto-contrast:", "gamma:",
        "colormap:", "projection mode:", "interpolation:", "depiction:",
    )
    controls = QWidget()
    form = QFormLayout(controls)
    rows = {}
    for text in napari_image_rows:
        lab, fld = QLabel(text), QComboBox()
        form.addRow(lab, fld)
        rows[text] = lab
    hide_native_rows(controls)
    controls.show()
    qapp.processEvents()
    visible = sorted(t for t, lab in rows.items() if lab.isVisibleTo(controls))
    assert visible == ["auto-contrast:", "colormap:", "contrast limits:"], (
        f"the resting blade must keep ONLY contrast limits, auto-contrast and colormap; "
        f"it shows {visible}")


# --- decon is just decon: the preview scope follows the tab ---------------------------------
# Julio (2026-08-25): "2d decon vs 3d decon. That's confusing. It should just say decon. if
# it's on 3D mode it runs it on all panes, if it's on 2d mode it runs it only on that one."


def _depthkeep_operator():
    """A core depth-keeping z-consumer (decon's declaration shape, no petakit needed)."""
    import numpy as np

    from squidxplorer import add_operator

    def _stack(planes):
        return np.asarray([np.asarray(p) for p in planes])

    _stack.keeps_depth = True
    add_operator("depthkeep_probe", _stack, consumes={"z"})
    return "depthkeep_probe"


def _z_reader(n_z=3):
    import numpy as np

    from tests.conftest import FakeReader

    meta = {
        "regions": ["A1"], "channels": [{"name": "488", "display_color": "#00FF00"}],
        "z_levels": list(range(n_z)), "n_z": n_z, "n_t": 1, "dtype": "uint16",
        "frame_shape": (4, 4), "pixel_size_um": 1.0, "dz_um": 1.0,
        "fovs_per_region": {"A1": [0]},
        "fov_positions_um": {("A1", 0): (0.0, 0.0)},
    }
    planes = lambda region, fov, ch, z, t: np.full((4, 4), 10 * z + 1, np.uint16)
    return FakeReader(meta, planes)


def test_a_preview_and_a_save_both_read_the_full_stack(tmp_path):
    """Every preview is the full solve (Julio, 2026-08-26): a depth-keeping preview, a
    reducer's preview and a save read every plane; the dispatch carries no plane knob."""
    import inspect

    from squidxplorer._dispatch import run_operator_once

    assert "preview_z_level" not in inspect.signature(run_operator_once).parameters
    op = _depthkeep_operator()
    reader = _z_reader()
    got = []
    result = run_operator_once(reader, operator=op, save=False, owed=1, regions=["A1"],
                               n_fovs=None, on_well=lambda r, f, img: got.append(img))
    assert result.landed == 1
    assert sorted({k[3] for k in reader.reads}) == [0, 1, 2]
    assert got[0].shape[2] == 3, "a preview keeps every plane"

    reducer_reader = _z_reader()
    run_operator_once(reducer_reader, operator="mip", save=False, owed=1, regions=["A1"],
                      n_fovs=None, on_well=lambda r, f, img: None)
    assert sorted({k[3] for k in reducer_reader.reads}) == [0, 1, 2]

    save_reader = _z_reader()
    run_operator_once(save_reader, operator=op, save=True, owed=1, regions=["A1"],
                      n_fovs=None, out_dir=str(tmp_path), on_well=lambda r, f, img: None)
    assert sorted({k[3] for k in save_reader.reads}) == [0, 1, 2]


def test_no_gui_string_says_2d_or_3d_decon():
    """The operator is just "decon" everywhere a user reads (Julio, 2026-08-25); the mode is how we VIEW it."""
    import ast
    from pathlib import Path

    import squidxplorer

    banned = ("2d decon", "3d decon", "volume solve")
    offenders: list[str] = []
    for path in sorted(Path(squidxplorer.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) \
                        and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docstrings \
                    and any(b in node.value.lower() for b in banned):
                offenders.append(f"{path.name}:{node.lineno} {node.value!r:.80}")
    assert not offenders, (
        "user-facing strings must say just 'decon'; the 2D/3D words are how we view it:\n"
        + "\n".join(offenders))


# --- the banner strip is retired: every say IS a log line -----------------------------------
# Julio (2026-08-25): "I don't like the red strip that appears above the window when I run
# an operator. That should appear in the logger."


def test_the_banner_strip_is_gone_and_say_is_a_log_line(qapp, caplog):
    import inspect

    import squidxplorer._napari_pane as NP

    assert "_banner" not in inspect.getsource(NP), "the banner strip is back"
    pane = NP.model_pane_class()()
    with caplog.at_level(logging.INFO, logger="squid.xplorer"):
        pane.say("could not dock the view controls")
    assert pane.readout.text() == "could not dock the view controls"
    assert pane.said[-1] == "could not dock the view controls"
    rec = caplog.records[-1]
    assert rec.levelno == logging.WARNING, "a refusal-shaped say must land at WARNING"
    assert "could not dock" in rec.getMessage()


def test_a_view_say_logs_classified_with_its_address(qapp, napari_pane_stub, squid_dataset,
                                                     caplog):
    root, _ = squid_dataset
    win, views = _open_view(qapp, root)
    try:
        v = views[0]
        with caplog.at_level(logging.INFO, logger="squid.xplorer"):
            v._say("could not start decon: no reader.")
        rec = caplog.records[-1]
        assert rec.levelno == logging.WARNING
        assert "could not start decon: no reader." in rec.getMessage()
        assert v._pane.said[-1] == "could not start decon: no reader.", (
            "the recording seam went hungry")
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


# --- the empty launch: a drop target, one line, quiet ------------------------------------------

#: An empty launch may say this many INFO lines at most (measured 0 on 2026-08-25).
EMPTY_LAUNCH_INFO_CAP = 2

EMPTY_HERO_LINE = "drop an acquisition folder here, or File > Open"


def _empty_launch(qapp):
    """What `main()` builds with no dataset: the working layout over nothing."""
    win = V.PlateWindow(None, default_layout=True, tabbed_views=True)
    win.show()
    for _ in range(10):
        qapp.processEvents()
    return win


def test_an_empty_launch_shows_the_hero_as_a_drop_target_with_one_line(qapp):
    win = _empty_launch(qapp)
    try:
        assert win.acceptDrops(), "the plate window does not accept drops"
        assert win._drop.isVisibleTo(win), "the drop target is not on screen"
        assert win._drop.text() == EMPTY_HERO_LINE
        assert "\n" not in win._drop.text(), "the empty state must be ONE centred line"
        for name in ("_view_caption", "_view_combo", "_open_sel_btn"):
            assert not getattr(win, name).isVisibleTo(win), f"{name} is shown with no data"
        assert not win._left_tabs.isVisibleTo(win), "the operator band is open with no data"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_data_bound_title_controls_appear_with_the_acquisition(qapp, napari_pane_stub,
                                                                  squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    try:
        assert not win._open_sel_btn.isVisibleTo(win)
        win.ingest(str(root))
        qapp.processEvents()
        assert not win._drop.isVisibleTo(win), "the drop target survived the ingest"
        for name in ("_view_caption", "_view_combo", "_open_sel_btn"):
            assert getattr(win, name).isVisibleTo(win), f"{name} is hidden after the ingest"
    finally:
        shutdown_plate_window(qapp, win)


def test_a_drop_on_the_empty_hero_reaches_ingest(qapp, tmp_path, monkeypatch):
    from qtpy.QtCore import QMimeData, QPointF, QUrl, Qt
    from qtpy.QtGui import QDragEnterEvent, QDropEvent

    win = V.PlateWindow(None)
    got: list[str] = []
    monkeypatch.setattr(win, "ingest", lambda p: got.append(str(p)))
    try:
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(tmp_path))])
        assert not win._drop.acceptDrops()
        pos = QPointF(win._drop.geometry().center())
        enter = QDragEnterEvent(pos.toPoint(), Qt.CopyAction, mime, Qt.LeftButton,
                                Qt.NoModifier)
        QApplication.sendEvent(win, enter)
        assert enter.isAccepted(), "the drag was refused at the hero"
        drop = QDropEvent(pos, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(win, drop)
        assert got == [str(tmp_path)], f"the drop did not reach ingest: {got}"
    finally:
        shutdown_plate_window(qapp, win)


def test_an_empty_launch_logs_at_most_a_couple_of_info_lines(qapp):
    records: list[logging.LogRecord] = []

    class _Spy(logging.Handler):
        def emit(self, record):
            records.append(record)

    spy = _Spy(level=logging.DEBUG)
    logger = logging.getLogger("squid.xplorer")
    logger.addHandler(spy)
    try:
        win = _empty_launch(qapp)
        shutdown_plate_window(qapp, win)
    finally:
        logger.removeHandler(spy)
    info = [r for r in records if r.levelno == logging.INFO]
    lines = "\n".join(f"  {r.name}: {r.getMessage()}" for r in info)
    assert len(info) <= EMPTY_LAUNCH_INFO_CAP, (
        f"an empty launch logged {len(info)} INFO line(s), over the cap of "
        f"{EMPTY_LAUNCH_INFO_CAP}:\n{lines}")


# --- ONE window: the deck takes the work area, the plate window hides -------------------------
# Julio, live on bf982a2 (2026-08-25): "I still see the blank screen where the old plate window
# used to go. The new one window should take up the whole screen. Not in fullscreen, but we add
# the space where the old plate window was." Measured: the plate window still on screen at
# x=0 width 420 and the deck beside it at x=420 width 1050 on a 1470-wide work area.
#
# THE CAUSE, reproduced offscreen: with the path given to the constructor the default view
# opened from the plate's own showEvent, so the hosting's hide() ran INSIDE the show. Qt maps
# the platform window after the show event returns, so the widget read hidden (isVisible()
# False) while its QWindow stayed on screen (windowHandle().isVisible() True). The tests
# below therefore assert the WINDOW HANDLE, which is what the user sees.


def _plate_is_off_screen(win) -> bool:
    handle = win.windowHandle()
    return not win.isVisible() and (handle is None or not handle.isVisible())


def _assert_frame_is_the_work_area(window, avail):
    """The frame (what the user sees) equals the available geometry, through the window's own minimum where offscreen size hints inflate it (see"""
    frame, client = window.frameGeometry(), window.geometry()
    assert frame.topLeft() == avail.topLeft(), (
        f"the deck's frame starts at {frame.topLeft()}, not the work area's {avail.topLeft()}")
    margin_w = frame.width() - client.width()
    margin_h = frame.height() - client.height()
    want_w = max(avail.width() - margin_w, window.minimumWidth())
    want_h = max(avail.height() - margin_h, window.minimumHeight())
    assert client.width() == want_w, f"deck width {client.width()} != work area {want_w}"
    assert client.height() == want_h, f"deck height {client.height()} != work area {want_h}"


def test_the_working_layout_is_one_window_and_the_deck_takes_the_work_area(
        qapp, napari_pane_stub, squid_dataset):
    from squidxplorer._fontscale import window_screen

    root, _ = squid_dataset
    win = V.PlateWindow(str(root), default_layout=True, tabbed_views=True)
    win.show()
    _drain_until(qapp, lambda: win._viewer_manager.deck(create=False) is not None, timeout=10)
    for _ in range(20):
        qapp.processEvents()
    try:
        deck = win._viewer_manager.deck(create=False)
        assert deck is not None, "the working layout opened no deck"
        assert deck.isVisible()
        assert _plate_is_off_screen(win), (
            "the plate window is still on screen beside the deck (widget visible "
            f"{win.isVisible()}, window handle visible "
            f"{win.windowHandle() is not None and win.windowHandle().isVisible()})")
        screen = window_screen(deck)
        if screen is None:
            pytest.skip("no screen to size against")
        _assert_frame_is_the_work_area(deck, screen.availableGeometry())
    finally:
        shutdown_plate_window(qapp, win)


def test_the_plate_window_shows_again_once_the_last_view_closes(qapp, napari_pane_stub,
                                                                squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(str(root), default_layout=True, tabbed_views=True)
    win.show()
    _drain_until(qapp, lambda: win._viewer_manager.deck(create=False) is not None, timeout=10)
    for _ in range(20):
        qapp.processEvents()
    try:
        assert _plate_is_off_screen(win)
        for view in list(win._viewer_manager.windows):
            view.request_close()
        _drain_until(qapp, lambda: not win._viewer_manager.windows, timeout=10)
        for _ in range(10):
            qapp.processEvents()
        assert win.isVisible(), "with no view left the plate window must be the surface"
    finally:
        shutdown_plate_window(qapp, win)


# --- ruling l: the ROI chip becomes "go to ROI" once a box is drawn ---------------------------
# Julio, 2026-08-25: "When I click ROI and draw window, then the ROI button temporarily changes
# to the go to roi arrow so that I don't have to open the controls to go to the ROI."


def _rect_inside(meta, region):
    from squidxplorer._mosaic_source import mosaic_bbox_um

    x0, y0, x1, y1 = mosaic_bbox_um(meta, region)
    ya, yb = y0 + (y1 - y0) * 0.3, y0 + (y1 - y0) * 0.6
    xa, xb = x0 + (x1 - x0) * 0.3, x0 + (x1 - x0) * 0.6
    return np.array([[ya, xa], [ya, xb], [yb, xb], [yb, xa]])


def test_the_roi_chip_turns_into_go_to_roi_once_a_box_is_drawn(qapp, napari_pane_stub,
                                                              squid_dataset):
    root, _ = squid_dataset
    win, (v,) = _open_view(qapp, root)
    mgr = win._viewer_manager
    try:
        assert v._btn_roi.text() == "▭ ROI"
        v._btn_roi.click()                               # start drawing
        layer = v._roi_layer
        assert layer is not None, "the ROI chip did not start an ROI layer"
        rect = _rect_inside(win._meta, v.current_region())
        layer.data = [rect]                              # the user finished the rectangle
        qapp.processEvents()
        assert v._btn_roi.text() == "→ ROI", "the chip did not turn into the go-to-ROI arrow"
        assert len(v._btn_roi.toolTip().split(". ")) == 1, "the arrow's tooltip is one sentence"
        n_before = len(mgr.windows)
        v._btn_roi.click()                               # the arrow opens the ROI child
        _drain_until(qapp, lambda: len(mgr.windows) == n_before + 1, timeout=10)
        assert len(mgr.windows) == n_before + 1, "clicking the arrow did not open the ROI child"
        assert v._btn_roi.text() == "▭ ROI", "a used ROI must hand the chip back to drawing"
        layer.data = [rect, rect + 2.0]                  # a SECOND box is drawn
        qapp.processEvents()
        assert v._btn_roi.text() == "→ ROI", "a new box must offer the arrow again"
        v._clear_rois()
        qapp.processEvents()
        assert v._btn_roi.text() == "▭ ROI", "clearing the ROIs must hand the chip back"
    finally:
        shutdown_plate_window(qapp, win)


# --- ruling n: a scoped preview lands a layer for its own fields ------------------------------


def test_a_scoped_preview_lands_a_layer_and_an_unscoped_one_still_refuses_holes(
        qapp, napari_pane_stub, squid_dataset, monkeypatch):
    from squidxplorer._mosaic_source import mosaic_fov_bboxes_um
    from squidxplorer._run import OperatorRun

    root, _ = squid_dataset
    win, (v,) = _open_view(qapp, root)
    try:
        region = v.current_region()
        fovs = list(win._meta["fovs_per_region"][region])
        assert len(fovs) >= 2, "the fixture must have a multi-FOV region"
        delivered = []
        monkeypatch.setattr(win, "_deliver_operator_result",
                            lambda op, res: delivered.append((op, res)))
        win._active_op_key = "mip"
        planes = np.ones((len(win._meta["channels"]),) + tuple(win._meta["frame_shape"]),
                         np.uint16)

        def _run(scope):
            return OperatorRun(key="mip", layer_key="mip", label="mip", action="mip", dest="",
                               address=None, requester=None, is_partial=scope is not None,
                               t0=0.0, scope=scope)

        win._run = _run({region: [fovs[0]]})     # the ROI preview: one field of N
        win._on_result(region, fovs[0], planes)
        assert len(delivered) == 1, "a scoped preview must land its layer at 1 of N"
        assert delivered[0][1].extent.bbox_um == mosaic_fov_bboxes_um(
            win._meta, region)[fovs[0]].bbox(), "the layer must sit on the scoped field"

        win._run = _run(None)                    # the unscoped run: still whole or nothing
        win._on_result(region, fovs[0], planes)
        assert len(delivered) == 1, "an unscoped run at 1 of N must still refuse holes"
    finally:
        shutdown_plate_window(qapp, win)


# --- ruling m: the plate view and the log go UNDER the layer controls and the layer list ------
# Julio, 2026-08-25: "The plate view and the logger should be under the contrast adjustment
# stuff and the layer toggle, not above."


# --- ruling o: the left column's real estate ------------------------------------------------
# Julio, 2026-08-25, screenshot: "Top left corner, realstate not being allocated efficiently."


def test_a_progress_report_after_the_run_ended_does_not_resurrect_the_bar(qapp, napari_pane_stub,
                                                                          squid_dataset):
    """The bar sat at '4%' under the log after a failed ROI decon run."""
    from squidxplorer._progress import ProgressReport

    root, _ = squid_dataset
    win, (v,) = _open_view(qapp, root)
    try:
        v.operator_started("decon")
        v.operator_progress(ProgressReport(label="decon", done=1, total=27, unit="field"))
        assert not v._op_progress.isHidden(), "a running run shows its bar"
        v.operator_failed("decon", "1 region(s) landed no layer")
        assert v._op_progress.isHidden(), "a failed run must take its bar down"
        v.operator_progress(ProgressReport(label="decon", done=2, total=27, unit="field"))
        assert v._op_progress.isHidden(), "a late report after the run ended put the bar back"
        v.operator_started("decon")
        v.operator_progress(ProgressReport(label="decon", done=3, total=27, unit="field"))
        v.operator_done("decon", 1.0)
        assert v._op_progress.isHidden(), "a finished run must take its bar down"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_shapes_controls_rows_and_tool_grid_are_chrome(qapp):
    """The ROI rectangle's look is the app's, so napari's Shapes styling rows and its shape-tool button grid are chrome; an Image form's tool grid stays."""
    from qtpy.QtWidgets import QComboBox, QFormLayout, QGridLayout, QLabel, QPushButton, QWidget

    from squidxplorer._napari_pane import NATIVE_HIDDEN_ROWS, hide_native_rows

    shapes_rows = ("edge width:", "edge color:", "face color:", "display text:")
    for text in shapes_rows:
        assert text in NATIVE_HIDDEN_ROWS, f"{text!r} is not in the chrome inventory"

    def _form(rows):
        controls = QWidget()
        form = QFormLayout(controls)
        grid = QGridLayout()
        buttons = [QPushButton(f"tool {i}") for i in range(3)]
        for i, b in enumerate(buttons):
            grid.addWidget(b, 0, i)
        form.addRow(grid)
        controls.button_grid = grid
        kept = {}
        for text in rows:
            lab, fld = QLabel(text), QComboBox()
            form.addRow(lab, fld)
            kept[text] = (lab, fld)
        controls.show()
        qapp.processEvents()
        return controls, buttons, kept

    shapes, tools, rows = _form(("opacity:",) + shapes_rows)
    hidden = hide_native_rows(shapes)
    qapp.processEvents()
    for text in shapes_rows:
        lab, fld = rows[text]
        assert not lab.isVisibleTo(shapes) and not fld.isVisibleTo(shapes), f"{text!r} shows"
    assert all(not b.isVisibleTo(shapes) for b in tools), "the shape-tool grid is still showing"
    assert "shape tools" in hidden, "the hidden grid must be named in the inventory"

    image, tools, rows = _form(("opacity:", "contrast limits:", "colormap:"))
    hide_native_rows(image)
    qapp.processEvents()
    assert all(b.isVisibleTo(image) for b in tools), "an Image form's tool grid was hidden"
    for text in ("contrast limits:", "colormap:"):
        lab, _ = rows[text]
        assert lab.isVisibleTo(image)


def test_the_layer_controls_container_takes_only_what_its_page_needs(qapp):
    """napari's controls container is a QStackedWidget whose hint is the TALLEST page (an Image form, 289 px measured), so a Shapes page sat over a blank area."""
    from qtpy.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

    from squidxplorer._napari_pane import fit_controls_container

    stack = QStackedWidget()
    tall = QWidget()
    tv = QVBoxLayout(tall)
    for i in range(12):
        tv.addWidget(QLabel(f"row {i}"))
    short = QWidget()
    sv = QVBoxLayout(short)
    sv.addWidget(QLabel("edge width"))
    stack.addWidget(tall)
    stack.addWidget(short)
    stack.setCurrentWidget(short)
    stack.show()
    qapp.processEvents()
    assert stack.sizeHint().height() >= tall.sizeHint().height()
    fit_controls_container(stack)
    qapp.processEvents()
    assert stack.maximumHeight() == short.sizeHint().height(), (
        f"the container keeps {stack.maximumHeight()} px for a page needing "
        f"{short.sizeHint().height()}")
    stack.setCurrentWidget(tall)
    fit_controls_container(stack)
    assert stack.maximumHeight() == tall.sizeHint().height()


# --- ruling p: the left column's height goes to the layer list, our docks are content-sized --
# Live on 2888349 (coordinator's screenshot): ~130 px blank under the operators band, ~80 px
# blank under the layer controls, the layer list squeezed to two rows, and the log band gone
# (the plate/log dock got only the plate's fixed 240 px). The dock area was handing spare
# height to OUR docks, which top-align their content and paint the rest blank.


def _left_docks(win):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDockWidget

    return [d for d in win.findChildren(QDockWidget)
            if win.dockWidgetArea(d) == Qt.LeftDockWidgetArea]


def _settle(qapp, n=30):
    for _ in range(n):
        qapp.processEvents()


# --- ruling q: fstack is an operator with a card ---------------------------------------------
# Julio: "Why isn't the fstack stuff integrated as an operator?" Measured: registered and CLI-
# runnable, no card, so the view's dropdown (built from the cards) could not select it.


def test_every_runnable_operator_has_a_card():
    from squidxplorer import runnable_operators
    from squidxplorer._operations import CLI_ONLY_OPERATORS, _OPERATIONS_BY_KEY

    assert CLI_ONLY_OPERATORS == frozenset(), "every survivor has a card; the CLI-only set stays empty"
    missing = [n for n in runnable_operators()
               if n not in _OPERATIONS_BY_KEY and n not in CLI_ONLY_OPERATORS]
    assert not missing, f"runnable operator(s) without a card, invisible to the GUI: {missing}"


# --- ruling r: a preview's layer has the asking view's raw extent and lands only there --------
# Julio, live on 2888349: "when I run decon on an ROI, it adds the full FOV to the layer. When
# I go to a tab that's the mosaic, it still enables me to click decon and look at a single ROI
# in comparison to the whole mosaic. In other words, decon layer is != raw view".


def _scoped_result(win, region, fov, op="mip"):
    from squidxplorer._op_result import RegionResultAccumulator

    meta = win._meta
    acc = RegionResultAccumulator(op, region, meta, [c["name"] for c in meta["channels"]],
                                  fovs=[fov])
    acc.add(fov, np.full((len(meta["channels"]),) + tuple(meta["frame_shape"]), 7, np.uint16))
    return acc.result()


def _layer_bbox(layer):
    ty, tx = float(layer.translate[-2]), float(layer.translate[-1])
    sy, sx = float(layer.scale[-2]), float(layer.scale[-1])
    h, w = layer.data.shape[-2], layer.data.shape[-1]
    return (tx, ty, tx + w * sx, ty + h * sy), (h, w)


def test_an_roi_child_shows_a_preview_cropped_to_its_own_box(qapp, napari_pane_stub,
                                                            squid_dataset):
    from squidxplorer._mosaic_source import mosaic_fov_bboxes_um

    root, _ = squid_dataset
    win, (parent,) = _open_view(qapp, root)
    mgr = win._viewer_manager
    try:
        region = parent.current_region()
        fov = list(win._meta["fovs_per_region"][region])[0]
        px = float(win._meta["pixel_size_um"])
        x0, y0, x1, y1 = mosaic_fov_bboxes_um(win._meta, region)[fov].bbox()
        roi = (x0 + 1 * px, y0 + 1 * px, x0 + 3 * px, y0 + 3 * px)   # a 2 x 2 px window in a 4 px field
        child = mgr.open_child([region], roi_bbox=roi, parent_id=parent.window_id)
        _drain_until(qapp, lambda: child._pane is not None, timeout=10)
        _settle(qapp)
        added = child.deliver_result("mip", _scoped_result(win, region, fov), visible=True)
        assert added >= 1
        channel = win._meta["channels"][0]["name"]
        layer = child._pane.mosaic.find("mip", channel)
        assert layer is not None
        bbox, shape = _layer_bbox(layer)
        for got, want in zip(bbox, roi):
            assert abs(got - want) <= px, f"layer bbox {bbox} is not the ROI {roi}"
        assert shape == (2, 2), f"the layer is {shape}, not the ROI's 2 x 2 px window"
    finally:
        shutdown_plate_window(qapp, win)


def test_a_preview_lands_only_in_the_view_that_asked(qapp, napari_pane_stub, squid_dataset,
                                                    monkeypatch):
    from squidxplorer._run import OperatorRun

    root, _ = squid_dataset
    win, (asker, other) = _open_view(qapp, root, n_views=2)
    try:
        region = asker.current_region()
        fov = list(win._meta["fovs_per_region"][region])[0]
        got = {}
        for v in (asker, other):
            monkeypatch.setattr(v, "deliver_result",
                                lambda op, res, visible, _v=v: got.setdefault(_v.window_id, []).append(visible) or 1)
        win._active_op_key = "mip"
        win._run = OperatorRun(key="mip", layer_key="mip", label="mip", action="mip", dest="",
                               address=None, requester=asker, is_partial=True, t0=0.0,
                               scope={region: [fov]})
        planes = np.ones((len(win._meta["channels"]),) + tuple(win._meta["frame_shape"]),
                         np.uint16)
        win._on_result(region, fov, planes)
        assert got.get(asker.window_id) == [True], "the asking view did not get its layer"
        assert other.window_id not in got, "a view that did not ask received the layer"
        from squidxplorer._recipe import acquisition_version, cached_operator_results

        assert not list(cached_operator_results(region, acquisition_version(win._reader))), (
            "a scoped (ROI) result was cached under the whole region")
    finally:
        shutdown_plate_window(qapp, win)


def test_a_mosaic_tab_preview_covers_the_whole_mosaic(qapp, napari_pane_stub, squid_dataset):
    from squidxplorer._mosaic_source import mosaic_bbox_um
    from squidxplorer._op_result import RegionResultAccumulator

    root, _ = squid_dataset
    win, (v,) = _open_view(qapp, root)
    try:
        region = v.current_region()
        meta = win._meta
        acc = RegionResultAccumulator("mip", region, meta, [c["name"] for c in meta["channels"]])
        for fov in meta["fovs_per_region"][region]:
            acc.add(fov, np.full((len(meta["channels"]),) + tuple(meta["frame_shape"]), 3,
                                 np.uint16))
        assert v.deliver_result("mip", acc.result(), visible=True) >= 1
        layer = v._pane.mosaic.find("mip", meta["channels"][0]["name"])
        bbox, _ = _layer_bbox(layer)
        want = mosaic_bbox_um(meta, region)
        px = float(meta["pixel_size_um"])
        for got, w in zip(bbox, want):
            assert abs(got - w) <= px, f"mosaic-tab layer bbox {bbox} != mosaic {want}"
    finally:
        shutdown_plate_window(qapp, win)


# --- ruling s: "auto-contrast" is OUR window rule on the pixels on screen --------------------
# Julio: "the napari autocontrast SUCKS for the G7 dataset". Measured on G7 488 / FOV 1 / z 7:
# napari min/max (6416, 65520), clipped 0; auto_contrast on the coarsest rung (18451, 63849),
# clipped 0.14%; auto_contrast full-res (18296, 65520), clipped 0; _pct_window full-res
# (9120, 58384), clipped 0.2%; mode 15888, p99 30000, p99.9 = 65520 (the plane saturates).


def _sparse_plane(seed=0):
    rng = np.random.default_rng(seed)
    plane = rng.normal(100.0, 5.0, (1024, 1024)).clip(0, 65535).astype(np.uint16)
    spots = []
    for _ in range(20):
        y, x = rng.integers(20, 1000, 2)
        plane[y:y + 5, x:x + 5] = rng.integers(2800, 3200)
        spots.append(plane[y:y + 5, x:x + 5].copy())
    plane[512, 512] = 65535                      # one hot pixel
    return plane, np.concatenate([s.ravel() for s in spots])


def test_our_auto_contrast_windows_a_sparse_field_on_its_objects_where_min_max_cannot():
    from squidxplorer._contrast import auto_contrast

    plane, spot_px = _sparse_plane()
    p99 = float(np.percentile(spot_px, 99))
    lo, hi = auto_contrast(plane)
    clipped = float((plane > hi).mean())
    assert hi >= 0.98 * p99, f"the ceiling {hi} sits below the objects' p99 {p99} (2% subsample tolerance)"
    assert hi <= 1.2 * p99, f"the ceiling {hi} follows the lone hot pixel, not the objects"
    assert clipped < 0.005, f"{clipped:.2%} of pixels clipped"
    assert lo > 100.0, "the floor must clear the background"
    napari_hi = float(plane.max())                # napari's once/continuous: the slice's max
    assert napari_hi > 1.2 * p99, "the pin would not distinguish our rule from min/max"


def test_the_seed_reads_the_finest_rung_its_budget_allows():
    from squidxplorer._contrast import SEED_MAX_PX, sample_plane

    fine = np.zeros((4000, 4000), np.uint16)     # 16 Mpx: over budget
    mid = np.zeros((2000, 2000), np.uint16)      # 4 Mpx: fits
    coarse = np.zeros((500, 500), np.uint16)
    assert sample_plane([fine, mid, coarse]).shape == coarse.shape, "no budget: the coarsest"
    assert sample_plane([fine, mid, coarse], max_px=SEED_MAX_PX).shape == mid.shape
    assert sample_plane([fine, mid, coarse], max_px=10).shape == coarse.shape


def test_the_continuous_autoscale_button_is_hidden_and_once_stays(qapp):
    from qtpy.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QPushButton, QWidget

    from squidxplorer._napari_pane import hide_native_rows

    controls = QWidget()
    form = QFormLayout(controls)
    holder = QWidget()
    row = QHBoxLayout(holder)
    once, cont = QPushButton("once"), QPushButton("continuous")
    row.addWidget(once)
    row.addWidget(cont)
    form.addRow(QLabel("auto-contrast:"), holder)
    controls.show()
    qapp.processEvents()
    hidden = hide_native_rows(controls)
    assert "continuous autoscale" in hidden
    assert once.isVisibleTo(controls) and not cont.isVisibleTo(controls)


def test_the_floor_clears_a_poisson_background():
    """G7 561 / FOV 1 / z 7: mode 896, sigma 18.7, max 3840; mode + 2 sigma rendered 16.7% of the background as speckle."""
    from squidxplorer._contrast import auto_contrast

    rng = np.random.default_rng(3)
    plane = rng.poisson(900.0, (1024, 1024)).astype(np.uint16)
    background = plane.copy()
    cells = []
    for _ in range(30):
        y, x = rng.integers(10, 1010, 2)
        plane[y:y + 6, x:x + 6] = rng.integers(2500, 3800)
        cells.append(plane[y:y + 6, x:x + 6].copy())
    cell_px = np.concatenate([c.ravel() for c in cells])
    lo, hi = auto_contrast(plane)
    above = float((background > lo).mean())
    assert above < 0.01, f"{above:.2%} of the background renders above black (floor {lo})"
    assert hi >= float(np.percentile(cell_px, 99)) * 0.98, f"the ceiling {hi} cuts the cells"


# --- ruling t: the param slot holds CONTROLS only, never a sentence -------------------------
# Julio: "The operators still have like a 'controls' page that it just has like BS AI text".


def _is_sentence(text: str) -> bool:
    t = text.strip()
    return bool(t) and (t.endswith(".") or len(t.split()) > 6)


# --- ruling u: one channel's checkbox is one channel's checkbox -------------------------------
# Julio, live on G7 after an ROI decon preview: "When I turn off layer 561 for decon, the whole
# layer turns off." Compute side settled headless (all three channels solved and delivered).


def test_hiding_one_result_channel_leaves_the_op_s_other_channels_and_raw_alone(
        qapp, napari_pane_stub, squid_dataset):
    from qtpy.QtCore import Qt

    from squidxplorer._layer_tree import MosaicTree

    root, _ = squid_dataset
    win, (v,) = _open_view(qapp, root)
    try:
        region = v.current_region()
        fov = list(win._meta["fovs_per_region"][region])[0]
        mosaic = v._pane.mosaic
        channels = [c["name"] for c in win._meta["channels"]]
        assert len(channels) >= 2
        _drain_until(qapp, lambda: len(mosaic.channels("raw")) == len(channels), timeout=10)
        assert v.deliver_result("mip", _scoped_result(win, region, fov), visible=True) >= 2
        _settle(qapp)
        for ch in channels:
            assert mosaic.find("mip", ch) is not None, f"no (mip, {ch}) layer"
            assert len(mosaic.layers_for("mip", ch)) == 1, "one layer per (op, channel) in 2D"
        raw_before = {ch: bool(mosaic.find("raw", ch).visible) for ch in channels}
        tree = MosaicTree(mosaic)
        m = tree.model()
        op_row = next(i for i, (op, _chs) in enumerate(m._rows) if op == "mip")
        ch_row = m._rows[op_row][1].index(channels[0])
        idx = m.index(ch_row, 0, m.index(op_row, 0))
        assert m._key_at(idx) == ("mip", channels[0]), "the channel row is not keyed as one"
        assert m.setData(idx, Qt.Unchecked, Qt.CheckStateRole)
        _settle(qapp)
        assert not mosaic.find("mip", channels[0]).visible
        for ch in channels[1:]:
            assert mosaic.find("mip", ch).visible, (
                f"hiding (mip, {channels[0]}) also hid (mip, {ch}): the whole op went dark")
        assert {ch: bool(mosaic.find("raw", ch).visible) for ch in channels} == raw_before, (
            "raw was touched by a result channel's checkbox")
        mosaic.find("mip", channels[1]).visible = False
        _settle(qapp)
        mosaic.find("mip", channels[1]).visible = True
        _settle(qapp)
        assert mosaic.find("mip", channels[1]).visible
        assert not mosaic.find("mip", channels[0]).visible, "re-lighting one channel lit another"
    finally:
        shutdown_plate_window(qapp, win)


# =============================================================================================
# Ruling v (Julio, 2026-08-25): the REVERSAL of "hidden by default and summoned". Nothing
# collapses; everything is visible, well-sized, one spacing. v2: the log is a fixed slot, never
# floated. v4: the LUT clipboard is shelved whole. w: ONE parameter surface. x: napari's menu
# bar is chrome. y: the one bar is the run bar. v1: a slider spans its OWN channel.
# =============================================================================================


def test_nothing_collapses_any_more_absence_pins():
    import squidxplorer._region_viewer as RV
    import squidxplorer._lut_clipboard as LC
    import squidxplorer._napari_pane as NP
    from squidxplorer._logpanel import LogPanel

    assert not hasattr(RV, "_FoldSection"), "_FoldSection is back"
    for name in ("set_operators_collapsed", "set_controls_collapsed", "summon_controls",
                 "_show_operator_controls", "_copy_luts", "_paste_luts", "lutsPasted",
                 "_refresh_controls_note", "_params_summary", "_refresh_quick_iterations"):
        assert not hasattr(RV.RegionViewer, name), f"RegionViewer.{name} is back"
    for name in ("grip", "toggle", "GRIP_PX"):
        assert not hasattr(V._PlateSlotBox, name), f"the plate slot is collapsible again ({name})"
    for name in ("set_collapsed", "toggle", "collapsed", "set_expanded_cap", "float_requested",
                 "collapsedChanged"):
        assert not hasattr(LogPanel, name), f"LogPanel.{name} is back"
    for name in ("_float_log", "_redock_log", "_on_log_collapsed", "_rehand_band", "show_log",
                 "_paste_luts_onto_plate", "_follow_window_luts", "_LOG_FLOAT_KEY"):
        assert not hasattr(V.PlateWindow, name), f"PlateWindow.{name} is back"
    for name in ("CLIPBOARD", "copy_luts", "paste_luts"):
        assert not hasattr(LC, name), f"the LUT clipboard is back ({name})"
    for name in ("_DockFitter", "watch_dock_fit", "stretch_dock", "hoist_left_dock",
                 "append_left_dock", "fit_dock_to_content"):
        assert not hasattr(NP, name), f"the dock-fit mechanism is back ({name})"
    assert isinstance(RV.COLUMN_PX, int), "one spacing constant"


def test_every_chip_and_the_operator_row_are_visible_at_rest(qapp, napari_pane_stub,
                                                            squid_dataset):
    from qtpy.QtWidgets import QAbstractButton, QLabel

    import squidxplorer._region_viewer as RV

    root, _ = squid_dataset
    win, (v,) = _open_view(qapp, root)
    try:
        v.show()
        _settle(qapp)
        for name in ("_btn_3d", "_btn_roi", "_btn_fovs", "_btn_focus", "_btn_record", "_btn_png",
                     "_op_combo", "_btn_preview", "_btn_run_plate"):
            assert getattr(v, name).isVisibleTo(v), f"{name} is not visible at rest"
        for name in ("_btn_controls", "_btn_copy_luts", "_btn_paste_luts", "_btn_auto",
                     "_iter_spin", "_controls_note"):
            assert not hasattr(v, name), f"{name} is back"
        # Julio, live on 862 px (2026-08-25): "The SquidXplorer buttons at the top of the left
        # dock should be smaller as well". Measured headless before: 26 px chips at a 15 px
        # QFont (the 11 px lived only in the stylesheet), grid 105 px; after: 22 px chips at
        # an 11 px QFont, grid 92 px, still three per row.
        for name in ("_btn_3d", "_btn_roi", "_btn_fovs", "_btn_focus", "_btn_record", "_btn_png"):
            chip = getattr(v, name)
            assert chip.height() == RV.CHIP_PX == 22, (name, chip.height())
            assert chip.font().pixelSize() == RV.CHIP_FONT_PX == 11, (name, chip.font().pixelSize())
        assert v._view_controls.sizeHint().height() <= 92, v._view_controls.sizeHint().height()
        combo = v._op_combo

        def pick(key):
            combo.setCurrentIndex(next(i for i in range(combo.count()) if combo.itemData(i) == key))
            _settle(qapp)

        pick("decon")
        panel = v._inserted_panel
        assert panel is not None and panel.isVisibleTo(v), "decon's controls did not follow"
        assert sorted(panel.widgets) == ["iterations"], sorted(panel.widgets)
        assert panel.ni_combo.currentText() == "1.000 (air)"
        area = [w for w in v.operator_panel().findChildren((QLabel, QAbstractButton))
                if not w.isHidden()]
        blob = " ".join(w.text().lower() for w in area) + " " + \
            panel.widgets["iterations"].prefix().lower()
        assert blob.count("iterations") == 1, f"'iterations' appears {blob.count('iterations')}x"
        panel.widgets["iterations"].setValue(9)
        assert win.operator_kwargs_for("decon")["iterations"] == 9, "the panel is not the store"
        pick("mip")
        assert v._inserted_panel is None, "a parameter-less operator shows nothing"
        pick("fstack")
        assert v._inserted_panel is not None and v._inserted_panel.adv_btn is not None
        assert v._inserted_panel.adv_btn.text() == "advanced parameters"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_log_is_a_fixed_slot_with_no_bar_at_rest_and_one_bar_during_a_run(
        qapp, napari_pane_stub, squid_dataset):
    from qtpy.QtWidgets import QProgressBar

    from squidxplorer._logpanel import LogPanel
    from squidxplorer._progress import ProgressReport
    from tests.test_view_deck import _tabbed_plate

    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    try:
        log = win._log_panel
        assert log.height() == log.slot_px()
        assert not hasattr(log, "_float_btn") and not hasattr(log, "_toggle")
        v = deck.current_page()
        assert v._hosts_plate_slots and log.isVisibleTo(v), "the log slot is not in the view"

        def bars():
            return [b for b in log.findChildren(QProgressBar) if not b.isHidden()]

        assert not bars(), "a bar shows with no run live (y)"
        mgr.set_run_progress(ProgressReport(label="decon", done=2, total=9, unit="field"))
        _settle(qapp)
        assert len(bars()) == 1, f"{len(bars())} bars during a run"
        mgr.set_run_progress(None)
        _settle(qapp)
        assert not bars(), "the bar survived the run's end"
        # The addendum: switching tabs re-homes the plate slot at its full height, visible
        # (on a deck tall enough to hold the column; a short one shrinks the plate slot).
        deck.resize(900, 1100)
        for page in (views[1], views[0]):
            deck._tabs.setCurrentWidget(page)
            _settle(qapp)
        cur = deck.current_page()
        box = win._plate_slot_box
        assert box.height() == V._PlateSlotBox.PLATE_SLOT_PX
        assert cur._plate_log_host.isAncestorOf(box), "the plate slot is not in the current view"
        assert win._overview.isVisibleTo(cur), "the overview is not showing after a tab switch"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_log_is_five_lines_and_the_layer_list_shows_every_channel_of_raw(
        qapp, napari_pane_stub, squid_dataset):
    """Measured on Julio's 862 px screen (2026-08-25): the layer list, the stretch consumer,
    ended with ~80 px and ONE of three raw channels behind a scrollbar while the log slot held
    135 px of a three-line message. The log slot is the header plus LINES lines of its own
    font (three at first; FIVE once the header dropped to a 10 px font and the chip grid
    shrank: "My log Height is a bit too small", measured 66 -> 89 px with the body 39 -> 65);
    the list's minimum is a group header plus the largest group's channel rows; and a screen
    too short for both shrinks the PLATE slot (a navigator; the list is a control), never
    below its floor."""
    from squidxplorer._logpanel import _FONT_PX, _HEADER_FONT_PX, LogPanel
    from tests.test_view_deck import _tabbed_plate

    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        v = deck.current_page()
        log = win._log_panel
        view = log._view
        fm = view.fontMetrics()
        want = (log._header.sizeHint().height() + LogPanel.LINES * fm.lineSpacing()
                + 2 * int(view.document().documentMargin()) + 2 * view.frameWidth())
        assert LogPanel.LINES == 5
        assert log.height() == log.slot_px() == want, (log.height(), log.slot_px(), want)
        # The header is one step below the body, as explicit QFonts (a stylesheet font
        # resolves at polish, after the slot height was fixed from the wrong metrics).
        assert _HEADER_FONT_PX < _FONT_PX
        for lbl in (log._title, log._activity_lbl):
            assert lbl.font().pixelSize() == _HEADER_FONT_PX, lbl.font().pixelSize()
        assert view.font().pixelSize() == _FONT_PX
        assert log._header.sizeHint().height() <= 16, log._header.sizeHint().height()

        tree = v._pane.layer_tree
        assert tree is not None and v._left_col.isAncestorOf(tree), "no layer list in the column"
        # The view's own raw preview lands on its worker's schedule; let it land first, then
        # add a third channel so the raw group is three channels (the fixture has two).
        assert _drain_until(qapp, lambda: len(v._pane.mosaic.channels("raw")) >= 2, timeout=20)
        v._pane.mosaic.add_mosaic("raw", "third", np.full((16, 16), 300, np.uint16),
                                  bbox_um=(0.0, 0.0, 16.0, 16.0))
        _settle(qapp)
        assert len(v._pane.mosaic.channels("raw")) == 3
        row = tree.sizeHintForRow(0)
        assert row > 0
        assert tree.minimumHeight() >= 4 * row, (tree.minimumHeight(), row)
        deck.resize(900, 1100)
        deck.show()
        _settle(qapp)
        assert tree.height() >= 4 * row, f"list {tree.height()} px shows less than 4 rows of {row}"
        box = win._plate_slot_box
        assert box.height() == V._PlateSlotBox.PLATE_SLOT_PX, "a tall screen keeps the full plate"

        deck.resize(900, 300)                    # far too short: something has to give
        _settle(qapp)
        assert tree.height() >= tree.minimumHeight(), "the list gave up rows"
        assert log.height() == want, "the log slot moved"
        assert V._PlateSlotBox.PLATE_SLOT_MIN_PX <= box.height() < V._PlateSlotBox.PLATE_SLOT_PX, (
            f"the plate slot is {box.height()} px; it is what shrinks, down to "
            f"{V._PlateSlotBox.PLATE_SLOT_MIN_PX}")
    finally:
        shutdown_plate_window(qapp, win)


def test_naparis_menu_bar_is_chrome_and_the_decks_menu_is_file_and_view(qapp, napari_pane_stub,
                                                                       squid_dataset):
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QAction, QKeySequence
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QMainWindow

    from squidxplorer._napari_pane import minimize_native_chrome
    from tests.test_view_deck import _tabbed_plate

    win = QMainWindow()
    fired = []
    act = QAction("Open File(s)...", win)
    act.setShortcut(QKeySequence.StandardKey.Open)
    act.triggered.connect(lambda *_: fired.append("napari open"))
    win.menuBar().addMenu("&File").addAction(act)
    win.show()
    qapp.processEvents()
    hidden = minimize_native_chrome(win)
    assert "menu bar" in hidden
    bar = win.menuBar()
    assert bar.isHidden() and not bar.isNativeMenuBar() and not bar.actions()
    QTest.keyClick(win, Qt.Key_O, Qt.ControlModifier)
    qapp.processEvents()
    assert not fired, "a hidden napari menu's shortcut still fires"

    root, _ = squid_dataset
    plate, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    try:
        menus = [a.menu().title() for a in deck.menuBar().actions() if a.menu() is not None]
        assert menus == ["&File", "&View"], menus
        file_actions = [a.text() for a in deck.menuBar().actions()[0].menu().actions()]
        assert file_actions == ["&Open Acquisition…", "&Quit"], file_actions
    finally:
        shutdown_plate_window(qapp, plate)


def test_a_dim_channel_s_slider_spans_its_own_maximum_not_the_saturated_channel_s():
    from squidxplorer import _bitdepth

    depth = _bitdepth.new_dataset(np.uint16)
    heard = []
    depth.on_change(lambda ch, lo, hi: heard.append((ch, lo, hi)))
    sat = np.full((64, 64), 65520, np.uint16)
    dim = np.full((64, 64), 900, np.uint16)
    dim[3, 3] = 3840
    depth.observe_array(sat, "Fluorescence_488_nm_Ex")
    depth.observe_array(dim, "Fluorescence_561_nm_Ex")
    assert _bitdepth.range_for(np.uint16, "Fluorescence_488_nm_Ex") == (0.0, 65535.0)
    lo, hi = _bitdepth.range_for(np.uint16, "Fluorescence_561_nm_Ex")
    assert 3840 <= hi <= 3840 * 1.10, f"the dim channel's range top {hi} is not within 10% of its max"
    assert [h[0] for h in heard] == ["Fluorescence_488_nm_Ex", "Fluorescence_561_nm_Ex"]
    depth.observe_array(np.full((4, 4), 100, np.uint16), "Fluorescence_561_nm_Ex")
    assert _bitdepth.range_for(np.uint16, "Fluorescence_561_nm_Ex") == (lo, hi), "a range narrowed"
    _bitdepth.new_dataset(None)


def test_naparis_once_button_runs_our_rule_on_an_app_layer(qapp, napari_pane_stub, squid_dataset):
    from squidxplorer._contrast import auto_contrast

    root, _ = squid_dataset
    win, (v,) = _open_view(qapp, root)
    try:
        assert not hasattr(v, "_btn_auto")
        mosaic = v._pane.mosaic
        plane, spot_px = _sparse_plane()
        layer = mosaic.add_result("intensity", "demo", "Fluorescence_561_nm_Ex", plane,
                                  bbox_um=(0.0, 0.0, 1024.0, 1024.0), visible=True)
        want = auto_contrast(np.asarray(mosaic.displayed_sample(layer)))
        layer.reset_contrast_limits()            # what napari's once button calls
        _drain_until(qapp, lambda: tuple(layer.contrast_limits) == pytest.approx(tuple(want)),
                     timeout=20)
        lo, hi = layer.contrast_limits
        assert (lo, hi) == pytest.approx(want), f"once landed {(lo, hi)}, not our window {want}"
        p99 = float(np.percentile(spot_px, 99))
        assert 0.98 * p99 <= hi <= 1.2 * p99 and hi < float(plane.max())
    finally:
        shutdown_plate_window(qapp, win)
