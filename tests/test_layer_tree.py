"""The grouped layer tree, and the 2D/3D button.

Qt/GL parts run in a clean subprocess (offscreen ships no GL and would segfault the session);
the pure-logic parts run in-process.
"""

from __future__ import annotations

import json
import os

import qtpy
import pathlib
import subprocess
import sys

import pytest

napari = pytest.importorskip("napari")

REPO = pathlib.Path(__file__).resolve().parent.parent


def _run_qt(script_body: str, tmp_path, marker: str):
    """Run *script_body* in a clean Qt process and return the dict it printed after *marker*.

    An exception in our code FAILS the test; only a GL-less box (no marker line) skips.
    """
    script = tmp_path / f"{marker.lower()}_check.py"
    script.write_text(_PREAMBLE.replace("__MARKER__", marker) + script_body + _POSTAMBLE.replace("__MARKER__", marker))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # The gate exports offscreen (no GL) for the whole suite; let Qt pick the real platform.
    env.pop("QT_QPA_PLATFORM", None)
    # Pin the child to the binding the parent is using; unpinned, qtpy defaults to PySide6 and
    # loading two bindings aborts the interpreter.
    env["QT_API"] = os.environ.get("QT_API") or qtpy.API_NAME.lower()

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO), env=env,
    )
    failed = [ln for ln in proc.stdout.splitlines() if ln.startswith(marker + "FAIL ")]
    if failed:
        pytest.fail("Qt check raised:\n" + json.loads(failed[0][len(marker) + 5:]))
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith(marker + " ")]
    if not line:
        pytest.skip(
            f"napari's Qt canvas could not be constructed here (rc={proc.returncode}); "
            f"stderr tail: {proc.stderr[-400:]}"
        )
    return json.loads(line[0][len(marker) + 1:])


_PREAMBLE = r"""
import json, os, sys, traceback
os.environ.setdefault("QT_API", "pyqt5")
import numpy as np
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget
app = QApplication.instance() or QApplication([])
out = {}
try:
    from squidxplorer._napari_pane import MosaicPane
"""

_POSTAMBLE = r"""
    print("__MARKER__ " + json.dumps(out))
except BaseException:
    print("__MARKER__FAIL " + json.dumps(traceback.format_exc()))
sys.stdout.flush()
os._exit(0)
"""


_NDISPLAY_SCRIPT = r"""
    host = QWidget()
    host.resize(1440, 900)          # Julio's monitor is small; check at the width he uses.
    lay = QVBoxLayout(host)
    pane = MosaicPane()
    lay.addWidget(pane)
    host.show()
    app.processEvents()

    from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton

    btn = pane.ndisplay_button
    top_of_pane = btn.mapTo(pane, btn.rect().topLeft()).y() if btn is not None else -1

    out["is_napari_widget_class"] = isinstance(btn, QtViewerPushButton)
    out["visible"] = bool(btn.isVisible())
    # "Visible" is not enough: the napari one is visible too, 752 px down. It has to be near the
    # top, where a short pane still shows it.
    out["y_within_pane"] = top_of_pane
    out["pane_height"] = pane.height()

    before = int(pane.mosaic.model.dims.ndisplay)
    btn.click(); app.processEvents()
    after = int(pane.mosaic.model.dims.ndisplay)
    checked_in_3d = bool(btn.isChecked())
    btn.click(); app.processEvents()
    back = int(pane.mosaic.model.dims.ndisplay)

    out["toggle"] = [before, after, back]
    out["checked_follows_dims"] = checked_in_3d
    # napari's dims is the ONE owner of 2D/3D. Move it from the model and our button must follow
    # without anybody hand-syncing it.
    pane.mosaic.model.dims.ndisplay = 3
    app.processEvents()
    out["follows_model_write"] = bool(btn.isChecked())
    pane.mosaic.model.dims.ndisplay = 2
    app.processEvents()
    out["unchecks_on_model_write"] = bool(btn.isChecked())
    out["tooltip"] = btn.toolTip()
"""


def test_the_3d_button_is_naparis_own_and_is_kept_alive_but_hidden(tmp_path):
    """napari's own 2D/3D button: hidden (3D is the ROI popout) but still tracking dims."""
    got = _run_qt(_NDISPLAY_SCRIPT, tmp_path, "NDISPLAY")

    assert got["is_napari_widget_class"] is True, "we rebuilt a button instead of reusing napari's"
    assert got["visible"] is False, (
        "the embedded 3D toggle is back on screen. 3D is the ROI native popout; an embedded "
        "toggle is the control that was explicitly deleted"
    )
    before, after, back = got["toggle"]
    assert [before, after, back] == [2, 3, 2], "clicking it does not actually change ndisplay"
    assert got["checked_follows_dims"] is True
    # One owner: dims. The button READS it, it does not keep a second copy.
    assert got["follows_model_write"] is True
    assert got["unchecks_on_model_write"] is False
    from squidxplorer._napari_pane import NDISPLAY_TOOLTIP

    assert got["tooltip"] == NDISPLAY_TOOLTIP
    assert "ROI" in got["tooltip"], "the tooltip must name a real way to reach full resolution"
    assert "AGAVE" not in got["tooltip"], "AGAVE is cancelled; do not advertise it"


from qtpy.QtCore import Qt                                          # noqa: E402
from qtpy.QtWidgets import QApplication                             # noqa: E402

import numpy as np                                                   # noqa: E402

from squidxplorer._napari_view import MosaicLayers                       # noqa: E402
from squidxplorer._layer_tree import MosaicTree                          # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _img(seed=0, shape=(8, 8)):
    return np.random.default_rng(seed).integers(0, 4000, shape, dtype="uint16")


@pytest.fixture
def mosaic():
    """raw and stitched, four channels each; stitched (added second) is the lit group."""
    from napari.components import ViewerModel

    m = MosaicLayers(ViewerModel())
    for i, op in enumerate(("raw", "stitched")):
        for j, ch in enumerate(("405", "488", "561", "638")):
            m.add_mosaic(op, ch, _img(i * 10 + j))
    return m


@pytest.fixture
def tree(qapp, mosaic):
    return MosaicTree(mosaic)


def _op_index(tree, row):
    return tree.model().index(row, 0)


def _ch_index(tree, op_row, ch_row):
    return tree.model().index(ch_row, 0, _op_index(tree, op_row))


def _op_index_of(tree, op):
    """The group row for *op*, found by name (display order is napari's to choose)."""
    m = tree.model()
    for r in range(m.rowCount()):
        if m.data(m.index(r, 0), Qt.DisplayRole) == op:
            return m.index(r, 0)
    raise AssertionError(f"no group row for {op!r}")


def _ch_index_of(tree, op, channel):
    """The channel row for (*op*, *channel*), found by name."""
    m = tree.model()
    parent = _op_index_of(tree, op)
    for r in range(m.rowCount(parent)):
        if m.data(m.index(r, 0, parent), Qt.DisplayRole) == channel:
            return m.index(r, 0, parent)
    raise AssertionError(f"no channel row for {op!r}/{channel!r}")


def test_the_tree_is_two_levels_processing_layer_then_channels(tree, mosaic):
    m = tree.model()
    assert m.rowCount() == 2, "processing layers are the top level"
    # Topmost first — napari's own layer-list convention.
    assert [m.data(_op_index(tree, r), Qt.DisplayRole) for r in range(2)] == ["stitched", "raw"]
    for r, op in enumerate(["stitched", "raw"]):
        assert m.rowCount(_op_index(tree, r)) == 4
        assert [
            m.data(_ch_index(tree, r, c), Qt.DisplayRole) for c in range(4)
        ] == list(reversed(mosaic.channels(op)))


def test_the_tree_reads_visibility_off_the_layer_and_keeps_no_copy(tree, mosaic):
    """The tree is a VIEW: napari's Image layer owns ``visible``."""
    m = tree.model()
    assert m.data(_ch_index_of(tree, "stitched", "405"), Qt.CheckStateRole) == Qt.Checked

    # Change it BEHIND the tree's back, the way napari's own layer list does.
    mosaic.find("stitched", "405").visible = False
    assert m.data(_ch_index_of(tree, "stitched", "405"), Qt.CheckStateRole) == Qt.Unchecked, (
        "the tree is holding its own copy of visibility instead of reading the layer"
    )


def test_an_external_visibility_change_repaints_the_row(tree, mosaic, qapp):
    m = tree.model()
    seen = []
    m.dataChanged.connect(lambda tl, br, roles=None: seen.append(tl))
    mosaic.find("raw", "488").visible = False
    qapp.processEvents()
    assert seen, "changing layer.visible elsewhere left the tree's checkbox stale"


def test_toggling_a_processing_layer_toggles_its_four_channels(tree, mosaic):
    m = tree.model()
    assert m.setData(_op_index_of(tree, "stitched"), Qt.Unchecked, Qt.CheckStateRole) is True
    assert [ly.visible for ly in mosaic.group("stitched")] == [False] * 4
    assert [ly.visible for ly in mosaic.group("raw")] == [False] * 4, (
        "hiding a group LIT another one -- a checkbox going off must never turn anything on")

    m.setData(_op_index_of(tree, "stitched"), Qt.Checked, Qt.CheckStateRole)
    assert [ly.visible for ly in mosaic.group("stitched")] == [True] * 4


def test_checking_a_processing_layer_darkens_the_one_it_replaces(tree, mosaic):
    """The group checkbox is a switch, not an accumulator: two operators of one channel sum."""
    m = tree.model()
    m.setData(_op_index_of(tree, "raw"), Qt.Checked, Qt.CheckStateRole)

    assert [ly.visible for ly in mosaic.group("raw")] == [True] * 4
    assert [ly.visible for ly in mosaic.group("stitched")] == [False] * 4, (
        "both operators of every channel are lit at once, so each channel is summed twice")
    assert mosaic.visible_op() == "raw"


def test_a_group_check_state_is_derived_from_its_channels_not_stored(tree, mosaic):
    m = tree.model()
    assert m.data(_op_index_of(tree, "stitched"), Qt.CheckStateRole) == Qt.Checked

    mosaic.find("stitched", "561").visible = False
    assert m.data(_op_index_of(tree, "stitched"), Qt.CheckStateRole) == Qt.PartiallyChecked, (
        "one hidden channel out of four is neither on nor off"
    )
    for ly in mosaic.group("stitched"):
        ly.visible = False
    assert m.data(_op_index_of(tree, "stitched"), Qt.CheckStateRole) == Qt.Unchecked


def test_toggling_one_channel_writes_that_layer_only(tree, mosaic):
    m = tree.model()
    m.setData(_ch_index_of(tree, "stitched", "561"), Qt.Unchecked, Qt.CheckStateRole)
    assert [ly.visible for ly in mosaic.group("stitched")] == [True, True, False, True]


def test_a_checkbox_write_logs_the_identity_and_every_layer_it_changed(tree, mosaic, caplog):
    """Ruling u diagnostic (Julio: "when I turn off layer 561 for decon, the whole decon layer
    turns off"; headless pins say each identity toggles alone). ONE DEBUG line per checkbox
    write: the identity written and every layer whose `visible` changed as a consequence."""
    import logging

    m = tree.model()
    with caplog.at_level(logging.DEBUG, logger="squid.xplorer"):
        m.setData(_ch_index_of(tree, "stitched", "561"), Qt.Unchecked, Qt.CheckStateRole)
    lines = [r.getMessage() for r in caplog.records
             if r.levelno == logging.DEBUG and r.getMessage().startswith("layer checkbox")]
    assert len(lines) == 1, caplog.text
    assert "stitched/561 -> off" in lines[0] and "changed 1 layer" in lines[0], lines[0]
    assert "stitched/561" in lines[0].split("changed", 1)[1]
    assert "stitched/488" not in lines[0].split("changed", 1)[1]
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="squid.xplorer"):
        m.setData(_op_index_of(tree, "stitched"), Qt.Unchecked, Qt.CheckStateRole)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("layer checkbox")]
    assert len(lines) == 1 and "stitched (group) -> off" in lines[0] and "changed 3 layer" in lines[0], lines


# The same tree over a flat mosaic and over a bricked volume; scene builders are shared with
# test_viewer_3d.py so there is one definition of what a 3D scene is.

from .conftest import build_flat_scene, build_volume_scene  # noqa: E402

SCENES = [
    pytest.param(build_flat_scene, id="2D-flat-mosaic"),
    pytest.param(build_volume_scene, id="3D-bricked-volume"),
]
SCENE_OP, SCENE_CHANNELS = "raw", ("488", "561")


@pytest.fixture
def scene_tree(qapp, request):
    """``(tree, mosaic)`` for whichever scene the test was parametrized with."""
    from napari.components import ViewerModel

    m = MosaicLayers(ViewerModel())
    request.param(m, SCENE_OP, SCENE_CHANNELS)
    return MosaicTree(m), m


@pytest.mark.parametrize("scene_tree", SCENES, indirect=True)
def test_the_tree_shows_one_group_with_one_row_per_channel_in_both_modes(scene_tree):
    tree, _mosaic = scene_tree
    m = tree.model()
    assert m.rowCount() == 1, "one processing layer is on screen, so one group row"
    group = _op_index_of(tree, SCENE_OP)
    assert m.rowCount(group) == len(SCENE_CHANNELS)
    assert {m.data(m.index(r, 0, group), Qt.DisplayRole)
            for r in range(m.rowCount(group))} == set(SCENE_CHANNELS)


@pytest.mark.parametrize("scene_tree", SCENES, indirect=True)
def test_the_group_checkbox_reaches_every_layer_in_both_modes(scene_tree):
    tree, mosaic = scene_tree
    m = tree.model()
    group = _op_index_of(tree, SCENE_OP)

    m.setData(group, Qt.Unchecked, Qt.CheckStateRole)
    assert not any(ly.visible for ly in mosaic.ours()), (
        "part of the scene is still lit with the group switched off")
    assert m.data(group, Qt.CheckStateRole) == Qt.Unchecked

    m.setData(group, Qt.Checked, Qt.CheckStateRole)
    assert all(ly.visible for ly in mosaic.ours())
    assert m.data(group, Qt.CheckStateRole) == Qt.Checked


@pytest.mark.parametrize("scene_tree", SCENES, indirect=True)
def test_a_channel_checkbox_reaches_every_layer_of_that_channel_in_both_modes(scene_tree):
    tree, mosaic = scene_tree
    m = tree.model()
    off, on = SCENE_CHANNELS[0], SCENE_CHANNELS[1]

    m.setData(_ch_index_of(tree, SCENE_OP, off), Qt.Unchecked, Qt.CheckStateRole)

    assert not any(ly.visible for ly in mosaic.layers_for(SCENE_OP, off)), (
        f"{off} is still partly on screen after its checkbox was cleared")
    assert all(ly.visible for ly in mosaic.layers_for(SCENE_OP, on)), (
        "clearing one channel darkened another")
    assert m.data(_ch_index_of(tree, SCENE_OP, off), Qt.CheckStateRole) == Qt.Unchecked
    assert m.data(_op_index_of(tree, SCENE_OP), Qt.CheckStateRole) == Qt.PartiallyChecked


def test_the_tree_survives_layers_being_destroyed_and_recreated(tree, mosaic, qapp):
    """Identity is (op, channel) out of layer.metadata, so a rebuilt layer is the same row."""
    m = tree.model()
    for op in list(mosaic.ops()):
        mosaic.remove_op(op)
    qapp.processEvents()
    assert m.rowCount() == 0, "the tree kept rows for layers that no longer exist"

    for i, op in enumerate(("raw", "stitched")):
        for j, ch in enumerate(("405", "488", "561", "638")):
            mosaic.add_mosaic(op, ch, _img(100 + i * 10 + j))
    qapp.processEvents()

    assert m.rowCount() == 2
    assert m.rowCount(_op_index_of(tree, "raw")) == 4
    m.setData(_ch_index_of(tree, "raw", "405"), Qt.Unchecked, Qt.CheckStateRole)
    assert mosaic.find("raw", "405").visible is False, (
        "the tree is still driving the DESTROYED layer object -- the contrast-sync bug again"
    )

    # The subscription has to be rebuilt too, not just the rows.
    seen = []
    m.dataChanged.connect(lambda tl, br, roles=None: seen.append(tl))
    mosaic.find("stitched", "638").visible = False
    qapp.processEvents()
    assert seen, (
        "after layers were recreated the tree stopped hearing visibility changes -- it is "
        "still subscribed to the destroyed objects"
    )


def test_foreign_layers_never_appear_in_the_tree(tree, mosaic, qapp):
    m = tree.model()
    mosaic.model.add_image(_img(7), name="somebody else's layer")
    qapp.processEvents()
    assert m.rowCount() == 2
    assert [m.data(_op_index(tree, r), Qt.DisplayRole) for r in range(2)] == ["stitched", "raw"]


def _brick(mosaic, op, channel, iy, ix):
    """One brick layer, stamped exactly as ``BrickedVolume._add_layer`` stamps it."""
    from squidxplorer._napari_view import MosaicKey

    return mosaic.model.add_image(
        _img(iy * 3 + ix, (4, 8, 8)), name=f"{channel} B{iy},{ix}",
        metadata=MosaicKey(op, channel).as_metadata())


def test_a_channel_row_switches_off_EVERY_layer_of_that_pair(tree, mosaic, qapp):
    """The row stands for the (op, channel) pair, not for one layer object."""
    bricks = [_brick(mosaic, "stitched", "561", iy, ix)
              for iy, ix in ((0, 0), (0, 1), (1, 0))]
    qapp.processEvents()
    m = tree.model()
    idx = _ch_index_of(tree, "stitched", "561")
    assert m.data(idx, Qt.CheckStateRole) == Qt.Checked

    m.setData(idx, Qt.Unchecked, Qt.CheckStateRole)

    lit = [ly for ly in mosaic.layers_for("stitched", "561") if ly.visible]
    assert lit == [], (
        f"{len(lit)} of {len(bricks) + 1} layer(s) answering to stitched/561 are still on screen "
        "after the channel was switched off")
    assert m.data(idx, Qt.CheckStateRole) == Qt.Unchecked


def test_a_channel_row_reports_PARTIAL_when_its_layers_disagree(tree, mosaic, qapp):
    """The row state is derived from ALL its layers, never from the first."""
    _brick(mosaic, "stitched", "488", 0, 0)
    second = _brick(mosaic, "stitched", "488", 0, 1)
    qapp.processEvents()
    m = tree.model()
    second.visible = False
    assert m.data(_ch_index_of(tree, "stitched", "488"), Qt.CheckStateRole) == Qt.PartiallyChecked


def test_selecting_a_channel_row_selects_every_brick_of_it(qapp, mosaic):
    _brick(mosaic, "stitched", "638", 0, 0)
    _brick(mosaic, "stitched", "638", 0, 1)
    tree = MosaicTree(mosaic)
    tree.setCurrentIndex(_ch_index_of(tree, "stitched", "638"))
    qapp.processEvents()
    assert len(mosaic.model.layers.selection) == 3, (
        f"the row selected {len(mosaic.model.layers.selection)} layer(s); the pair has 3")


def test_checkboxes_are_actually_offered_to_the_user(tree):
    """Answering CheckStateRole without ItemIsUserCheckable renders unclickable checkboxes."""
    m = tree.model()
    for idx in (_op_index(tree, 0), _ch_index(tree, 0, 0)):
        assert m.flags(idx) & Qt.ItemIsUserCheckable
        assert m.flags(idx) & Qt.ItemIsEnabled


_MOUNT_SCRIPT = r"""
    host = QWidget()
    host.resize(1440, 900)
    lay = QVBoxLayout(host)
    pane = MosaicPane()
    lay.addWidget(pane)
    host.show()
    app.processEvents()

    for op in ("raw", "stitched"):
        for ch in ("405", "488", "561", "638"):
            pane.mosaic.add_mosaic(op, ch, np.zeros((16, 16), dtype="uint16"))
    app.processEvents()

    from qtpy.QtCore import Qt as _Qt
    tree = pane.layer_tree

    def descends_from(child, ancestor):
        node = child
        while node is not None:
            if node is ancestor:
                return True
            node = node.parent()
        return False

    out["tree_exists"] = tree is not None
    out["tree_visible"] = bool(tree.isVisible())
    out["tree_is_in_our_pane"] = descends_from(tree, pane)
    out["rows"] = tree.model().rowCount()
    out["children_of_first"] = tree.model().rowCount(tree.model().index(0, 0))

    # napari's FLAT LAYER LIST must be HIDDEN (the tree replaces it), while its LAYER CONTROLS
    # -- the contrast/gamma/colormap panel, a different surface -- must survive. And the canvas
    # must still be its window's central widget: adding a dock must not repeat 506c813.
    win = pane._native_window
    _qtv = getattr(pane._viewer.window, "_qt_viewer", None)
    _dock = getattr(_qtv, "dockLayerList", None) if _qtv is not None else None
    out["flat_layer_list_visible"] = bool(_dock.isVisible()) if _dock is not None else None
    out["layer_controls_still_there"] = len([
        w for w in win.findChildren(QWidget) if "QtLayerControls" in type(w).__name__
    ]) if win is not None else 0
    out["canvas_still_inside_napari_window"] = descends_from(pane.canvas, win)

    # The gesture: hide a whole processing layer from the tree. Found BY NAME: the tree lists
    # topmost-first (napari's own convention), so a row index is not a stable way to say
    # "the stitched group".
    _m = tree.model()
    idx = next(_m.index(r, 0) for r in range(_m.rowCount())
               if _m.data(_m.index(r, 0), _Qt.DisplayRole) == "stitched")
    # ...but FIRST the switch: checking the raw group must darken stitched, because both lit is
    # one channel summed twice (see test_checking_a_processing_layer_darkens_the_one_it_replaces).
    _rawidx = next(_m.index(r, 0) for r in range(_m.rowCount())
                   if _m.data(_m.index(r, 0), _Qt.DisplayRole) == "raw")
    tree.model().setData(_rawidx, _Qt.Checked, _Qt.CheckStateRole)
    app.processEvents()
    out["switched_to"] = [bool(l.visible) for l in pane.mosaic.group("raw")]
    out["switched_away_from"] = [bool(l.visible) for l in pane.mosaic.group("stitched")]

    tree.model().setData(idx, _Qt.Unchecked, _Qt.CheckStateRole)
    app.processEvents()
    out["group_hidden"] = [bool(l.visible) for l in pane.mosaic.group("stitched")]
    out["other_group_untouched"] = [bool(l.visible) for l in pane.mosaic.group("raw")]

    # ... and the reverse direction: napari's own list is still an owner, and the tree follows.
    pane.mosaic.find("raw", "405").visible = False
    app.processEvents()
    _raw = next(_m.index(r, 0) for r in range(_m.rowCount())
                if _m.data(_m.index(r, 0), _Qt.DisplayRole) == "raw")
    _405 = next(_m.index(r, 0, _raw) for r in range(_m.rowCount(_raw))
                if _m.data(_m.index(r, 0, _raw), _Qt.DisplayRole).endswith("405"))
    # Qt5 hands back a plain int here; Qt6 hands back a Qt.CheckState enum, and int() refuses it.
    # .value where there is one, the value itself otherwise, so this reads under either binding.
    def _check_state(index):
        v = _m.data(index, _Qt.CheckStateRole)
        return int(getattr(v, "value", v))

    out["leaf_state_after_external_change"] = _check_state(_405)
    out["group_state_after_external_change"] = _check_state(_raw)
"""


def test_the_tree_is_mounted_beside_naparis_own_controls(tmp_path):
    """The tree ships inside the real pane, and napari's own list survives next to it."""
    got = _run_qt(_MOUNT_SCRIPT, tmp_path, "MOUNT")

    assert got["tree_exists"] is True
    # Since ruling v3 (2026-08-25) the pane BUILDS the tree and the view's one plain column
    # hosts it (`native_column_widgets`); a bare pane holds it undocked, so visibility is the
    # view's to assert, not this probe's.
    assert got["rows"] == 2
    assert got["children_of_first"] == 4
    # The grouped tree REPLACES the flat list; napari's LAYER CONTROLS stay (they own contrast).
    assert got["flat_layer_list_visible"] is False, (
        "napari's flat layer list is still showing, so the layer explosion is still on screen"
    )
    assert got["layer_controls_still_there"] >= 1, (
        "napari's layer controls disappeared -- contrast has no owner on screen"
    )
    # Adding a dock must never move the canvas out of napari's window.
    assert got["canvas_still_inside_napari_window"] is True
    assert got["switched_to"] == [True] * 4
    assert got["switched_away_from"] == [False] * 4, (
        "checking raw left stitched lit, so every channel is being drawn twice, additively")
    assert got["group_hidden"] == [False] * 4
    assert got["other_group_untouched"] == [True] * 4
    assert got["leaf_state_after_external_change"] == 0        # Qt.Unchecked
    assert got["group_state_after_external_change"] == 1       # Qt.PartiallyChecked


def test_the_rows_are_painted_by_naparis_own_delegate(qapp, mosaic):
    """napari's LayerDelegate by class — a silent fallback to Qt's default must go red."""
    from squidxplorer import _layer_tree as LT

    tree = LT.MosaicTree(mosaic)
    delegate = tree.itemDelegate()
    assert type(delegate).__module__.startswith("napari."), (
        f"rows are painted by {type(delegate).__module__}.{type(delegate).__name__}, "
        "not by napari -- the tree fell back to Qt's default look"
    )
    assert type(delegate).__name__ == "LayerDelegate"


def test_the_eye_rules_reach_the_tree_and_not_only_naparis_own_list():
    """The ::indicator (eye) and ::item (gutter) rules must be rewritten onto QTreeView."""
    from squidxplorer import _layer_tree as LT

    sheet = (
        'QListView { background: #000000; }\n'
        'QtLayerList::item { margin: 2px 2px 2px 28px; }\n'
        'QtLayerList::indicator { image: url("theme_dark:/visibility_off.svg"); }\n'
        'QtLayerList::indicator:checked { image: url("theme_dark:/visibility.svg"); }\n'
    )
    out = LT._napari_stylesheet(sheet)

    assert "QTreeView::indicator:checked" in out, "the eye never reaches the tree"
    checked = out.split("QTreeView::indicator:checked")[-1]
    assert 'url("theme_dark:/visibility.svg")' in checked, (
        "the tree got an indicator rule with no eye image in it"
    )
    # The gutter is load-bearing: without it the thumbnail covers the indicator.
    assert "QTreeView::item" in out and "28px" in out.split("QTreeView::item")[-1], (
        "no left margin on the tree's items -- the thumbnail will paint over the eye"
    )
    assert "QListView {" in out and "QtLayerList::indicator {" in out


def test_the_model_serves_every_role_that_delegate_paints_from(qapp, mosaic):
    """A missing role degrades the row silently: no thumbnail, or a folder icon on a channel."""
    from squidxplorer import _layer_tree as LT

    assert LT._NAPARI_ROLES, "napari's delegate roles did not resolve; rows cannot be painted"
    view = LT.MosaicTree(mosaic)        # held: the model is parented to it and dies with it
    model = view.model()
    group = _op_index_of(view, "raw")
    channel = _ch_index_of(view, "raw", "405")

    # a PROCESSING LAYER is a group -> napari paints a folder, open when expanded
    item = model.data(group, LT._NAPARI_ROLES["item"])
    assert item is not None and item.is_group() is True

    # a CHANNEL is the real napari layer -> it gets the image icon and its own thumbnail
    assert model.data(channel, LT._NAPARI_ROLES["item"]) is mosaic.find("raw", "405")
    thumb = model.data(channel, LT._NAPARI_ROLES["thumbnail"])
    assert thumb is not None and thumb.width() > 0, "no thumbnail: the row will paint empty"

    # loaded, or napari starts a loading GIF that never stops
    assert model.data(channel, LT._NAPARI_ROLES["loaded"]) is True
    # and the row must be tall enough for the thumbnail napari draws into it
    assert model.data(channel, Qt.SizeHintRole).height() >= 30


def test_the_rows_actually_PAINT_without_raising(qapp, mosaic):
    """Render the view for real; Qt swallows paint() exceptions, so an excepthook catches them."""
    import sys

    from qtpy.QtGui import QPixmap

    from squidxplorer import _layer_tree as LT

    view = LT.MosaicTree(mosaic)
    view.resize(300, 400)
    view.expandAll()

    caught = []
    original = sys.excepthook
    sys.excepthook = lambda *a: caught.append(a)
    try:
        pixmap = QPixmap(view.size())
        view.render(pixmap)          # drives delegate.paint() over every visible row
    finally:
        sys.excepthook = original

    assert not caught, f"painting a row raised: {caught[0][1] if caught else ''}"


def test_selecting_a_channel_row_selects_that_layer_in_napari(qapp, mosaic):
    from squidxplorer import _layer_tree as LT

    tree = LT.MosaicTree(mosaic)
    tree.setCurrentIndex(_ch_index_of(tree, "raw", "488"))

    selection = mosaic.model.layers.selection
    assert [ly.name for ly in selection] == ["raw · 488"]
    assert selection.active is mosaic.find("raw", "488"), (
        "no ACTIVE layer, so napari's contrast panel has nothing to show"
    )


def test_selecting_a_processing_layer_selects_all_of_its_channels(qapp, mosaic):
    """Setting selection.active would call select_only() and collapse the set to one layer."""
    from squidxplorer import _layer_tree as LT

    tree = LT.MosaicTree(mosaic)
    tree.setCurrentIndex(_op_index_of(tree, "raw"))

    assert sorted(ly.name for ly in mosaic.model.layers.selection) == [
        "raw · 405", "raw · 488", "raw · 561", "raw · 638",
    ]


def test_the_tree_lists_topmost_first_like_naparis_own_layer_list(qapp, mosaic):
    """napari owns the display order (last added on top); the tree mirrors it."""
    from squidxplorer import _layer_tree as LT

    tree = LT.MosaicTree(mosaic)
    model = tree.model()
    ours = [model.data(model.index(r, 0), Qt.DisplayRole) for r in range(model.rowCount())]

    # napari's own order, as its list widget shows it: last added first.
    napari_order = []
    for layer in reversed(list(mosaic.model.layers)):
        op = layer.metadata["squidxplorer"]["op"]
        if op not in napari_order:
            napari_order.append(op)
    assert ours == napari_order, f"tree shows {ours}, napari shows {napari_order}"

    raw = _op_index_of(tree, "raw")
    channels = [model.data(model.index(r, 0, raw), Qt.DisplayRole)
                for r in range(model.rowCount(raw))]
    assert channels == list(reversed(mosaic.channels("raw"))), "channels are not topmost-first"
