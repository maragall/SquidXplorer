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


def test_only_the_3d_and_roi_essentials_show_until_the_controls_are_summoned(qapp,
                                                                             napari_pane_stub,
                                                                             squid_dataset):
    root, _ = squid_dataset
    win, views = _open_view(qapp, root)
    try:
        v = views[0]
        fold = v._controls_fold
        assert fold.collapsed, "the view controls must start collapsed"
        assert v._btn_3d.isVisibleTo(v), "3D is an essential and must stay visible"
        assert v._btn_roi.isVisibleTo(v), (
            "ROI is top-level (Julio, 2026-08-25: 'The ROI button shouldn't be hidden "
            "behind controls.')")
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


def test_the_2d_button_is_gone_whole_and_a_3d_tab_disables_its_own_3d(qapp, napari_pane_stub,
                                                                      squid_dataset):
    """Julio, 2026-08-25: "There should not be 2D button since we make separate tabs for
    the 3d view." A 2D tab IS 2D and a 3D tab IS 3D; nothing switches modes in place."""
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


# --- the collapsed log is a REAL band, never a clipped sliver -------------------------------
# Julio, live GUI 2026-08-25: "Can't see log. Blank frame." The collapsed cap was frozen at
# construction-time sizeHint, BEFORE adopt_status_row grew the panel (header + memory/run
# bars), so the band rendered as a clipped sliver ("2%" cut mid-label) with no reachable
# summon toggle.


def test_a_collapsed_log_shows_its_whole_header_and_status_rows(qapp, napari_pane_stub,
                                                                squid_dataset):
    win = V.PlateWindow(None)
    try:
        panel = win._log_panel
        assert panel.collapsed
        panel.layout().activate()
        need = panel.layout().sizeHint().height()
        assert panel.maximumHeight() >= need, (
            f"the collapsed log is clipped: cap {panel.maximumHeight()} px against "
            f"{need} px of header + status rows - the summon toggle and the progress "
            f"bars are cut mid-pixel")
        assert panel._toggle.isVisibleTo(panel), "the summon toggle is not in the band"
        assert panel._status.isVisibleTo(panel), (
            "the adopted memory/run bars are not in the collapsed band")
    finally:
        shutdown_plate_window(qapp, win)


def test_a_hosted_collapsed_log_is_a_reachable_band(qapp, napari_pane_stub, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    mgr = win._viewer_manager
    mgr.tabbed_views = True
    views = [mgr.open([list(win._order)[0]])]
    _drain_until(qapp, lambda: views[0]._pane is not None, timeout=10)
    for _ in range(10):
        qapp.processEvents()
    try:
        panel = win._log_panel
        v = views[0]
        assert panel.collapsed
        assert panel._toggle.isVisibleTo(v), (
            "the hosted collapsed log has no summon affordance on screen")
        panel.layout().activate()
        assert panel.maximumHeight() >= panel.layout().sizeHint().height(), (
            "the hosted collapsed log is clipped")
    finally:
        shutdown_plate_window(qapp, win)


def test_the_view_column_cannot_claim_more_height_than_its_content(qapp, napari_pane_stub,
                                                                   squid_dataset):
    """The blank frame: the docked left column kept the height its collapsed content no
    longer needed, a dead band between the plate slot and the layer controls. A Maximum
    vertical policy makes the dock hand freed space to the other slots (qSmartMaxSize
    caps a no-grow policy at the size hint)."""
    from qtpy.QtWidgets import QSizePolicy

    root, _ = squid_dataset
    win, views = _open_view(qapp, root)
    try:
        col = views[0]._left_col
        assert col.sizePolicy().verticalPolicy() == QSizePolicy.Maximum, (
            "the left column can grow past its content and paints the slack as a blank band")
    finally:
        shutdown_plate_window(qapp, win)


# --- napari-native chrome is minimized ------------------------------------------------------


def test_native_chrome_is_minimized(qapp):
    """The pure half of the pane's chrome diet, proven on a plain Qt window shaped like the
    embedded napari one: the status bar hides, dock title bars slim to nothing, and the
    layer-controls rows the app manages elsewhere fold away while the kept rows stay."""
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
    """Julio, live 2026-08-25: "Layer controls, too much height." The resting blade shows
    ONLY what a life-science user touches: contrast limits, auto-contrast, colormap. The
    pin is a row-count budget over napari 0.6.6's real image-controls labels, never a
    pixel number."""
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


def test_a_2d_preview_solves_exactly_the_plane_in_view():
    from squidxplorer._dispatch import run_operator_once

    op = _depthkeep_operator()
    reader = _z_reader()
    got = []
    result = run_operator_once(reader, operator=op, save=False, owed=1,
                               regions=["A1"], n_fovs=None,
                               on_well=lambda r, f, img: got.append(img),
                               preview_z_level=1)
    assert result.landed == 1
    assert sorted({z for (_r, _f, _c, z, _t) in reader.reads}) == [1], (
        f"a 2D preview must read ONLY the in-view plane; read z {sorted({k[3] for k in reader.reads})}")
    assert got[0].shape[2] == 1, "the restricted preview must yield a 1-plane result"


def test_a_3d_preview_and_a_reducer_preview_keep_the_full_stack():
    from squidxplorer._dispatch import run_operator_once

    op = _depthkeep_operator()
    reader = _z_reader()
    got = []
    run_operator_once(reader, operator=op, save=False, owed=1, regions=["A1"], n_fovs=None,
                      on_well=lambda r, f, img: got.append(img), preview_z_level=None)
    assert sorted({k[3] for k in reader.reads}) == [0, 1, 2]
    assert got[0].shape[2] == 3, "an unrestricted preview keeps every plane"

    # A z-REDUCER ignores the restriction by declaration: the MIP of one plane is a
    # different result, so the guard must never let preview_z_level reach it.
    reducer_reader = _z_reader()
    run_operator_once(reducer_reader, operator="mip", save=False, owed=1, regions=["A1"],
                      n_fovs=None, on_well=lambda r, f, img: None, preview_z_level=1)
    assert sorted({k[3] for k in reducer_reader.reads}) == [0, 1, 2], (
        "a reducer's preview must still consume every plane")


def test_a_save_always_runs_the_full_stack(tmp_path):
    from squidxplorer._dispatch import run_operator_once

    op = _depthkeep_operator()
    reader = _z_reader()
    run_operator_once(reader, operator=op, save=True, owed=1, regions=["A1"], n_fovs=None,
                      out_dir=str(tmp_path), on_well=lambda r, f, img: None,
                      preview_z_level=1)
    assert sorted({k[3] for k in reader.reads}) == [0, 1, 2], (
        "Run on plate must deconvolve the full stack whatever tab asked")


def test_the_region_arm_refuses_a_z_restriction_by_name():
    import pytest as _pytest

    import squidxplorer

    reader = _z_reader()
    with _pytest.raises(ValueError, match="z_operator"):
        list(squidxplorer.run_plate(reader, operator="stitch", z_level=1))


def test_no_gui_string_says_2d_or_3d_decon():
    """The operator is just "decon" everywhere a user reads (Julio, 2026-08-25); the mode is
    how we VIEW it. Same sweep shape as the em-dash guard: non-docstring literals only."""
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


def test_the_collapsed_log_band_shows_the_latest_entry(qapp):
    from squidxplorer._logpane import LogBus
    from squidxplorer._logpanel import LogPanel

    bus = LogBus()
    panel = LogPanel(bus, None, start_collapsed=True)
    logger = logging.getLogger("squid.xplorer.test_band")
    rec = logger.makeRecord("squid.xplorer.test_band", logging.WARNING, __file__, 1,
                            "stitch refused: no positions", (), None)
    bus.emit_record(rec)
    for _ in range(5):
        QApplication.processEvents()
    assert "stitch refused: no positions" in panel._activity_lbl.text(), (
        "a collapsed log band must show the latest entry so a refusal is noticed "
        "without expanding")
    panel.set_collapsed(False)
    assert panel._activity_lbl.text() in ("idle", ""), (
        "expanded, the header goes back to the activity sentence; the body has the lines")


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
