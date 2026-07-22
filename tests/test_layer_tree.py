"""The grouped layer tree, and the 2D/3D button that has to be reachable on a small screen.

Why a separate file. ``tests/test_napari_view.py`` is being edited by several agents at once;
everything this branch adds lives here so the two never collide.

Why subprocesses. napari's canvas is vispy/GL and the gate runs the suite under
``QT_QPA_PLATFORM=offscreen``, which ships no GL — constructing a canvas under it does not
raise, it SEGFAULTS the session. ``test_napari_view.py`` already solved this: run the Qt part
in a clean subprocess with the platform plugin left alone, so a crash is a test failure rather
than a dead run, and a genuinely GL-less box skips with the reason attached. The pure-logic
parts below need neither Qt nor napari and run in-process.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

napari = pytest.importorskip("napari")

REPO = pathlib.Path(__file__).resolve().parent.parent


def _run_qt(script_body: str, tmp_path, marker: str):
    """Run *script_body* in a clean Qt process and return the dict it printed after *marker*.

    An exception inside OUR code prints ``<marker>FAIL`` and FAILS the test. Only a box with no
    GL at all produces no marker line and skips. A skip and a bug must never look the same —
    that is how the embedding check read green for its whole life while asserting nothing.
    """
    script = tmp_path / f"{marker.lower()}_check.py"
    script.write_text(_PREAMBLE.replace("__MARKER__", marker) + script_body + _POSTAMBLE.replace("__MARKER__", marker))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # The gate exports offscreen for the whole suite and offscreen has no GL, so inheriting it
    # guarantees a segfault and a permanent skip. Let Qt pick the real platform.
    env.pop("QT_QPA_PLATFORM", None)
    # squidmip imports PyQt5; qtpy defaults to PySide6 here and loading both aborts the process
    # long before any assertion runs.
    env["QT_API"] = "pyqt5"

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
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
app = QApplication.instance() or QApplication([])
out = {}
try:
    from squidmip._napari_pane import MosaicPane
"""

_POSTAMBLE = r"""
    print("__MARKER__ " + json.dumps(out))
except BaseException:
    print("__MARKER__FAIL " + json.dumps(traceback.format_exc()))
sys.stdout.flush()
os._exit(0)
"""


# ---------------------------------------------------------------- the 2D/3D button
#
# The button is not missing from napari — it is napari's own ``QtViewerButtons.ndisplayButton``,
# and a probe of the embedded window found it present and visible at y=752 inside a 900 px host:
# the LAST row of the left dock column, under a layer list that grows with every layer. Julio is
# on a small monitor and has asked for a visible 3D toggle twice. So the fix is not to build a
# button, it is to put NAPARI'S button somewhere that does not scroll off: a fixed row at the top
# of pane 2.

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
"""


def test_the_3d_button_is_naparis_own_and_sits_where_a_short_pane_shows_it(tmp_path):
    """A 2D/3D toggle Julio can actually see, built out of napari's own button.

    Asked for twice. napari HAS the button — bottom of the left dock column, below a layer list
    that grows with every layer added, so on a small screen it is simply not on screen. Lifting
    napari's own widget into a fixed row at the top of the pane fixes reachability without
    inventing a second control: the button we show and the one napari docks drive the same
    ``viewer.dims.ndisplay``, so they cannot disagree.
    """
    got = _run_qt(_NDISPLAY_SCRIPT, tmp_path, "NDISPLAY")

    assert got["is_napari_widget_class"] is True, "we rebuilt a button instead of reusing napari's"
    assert got["visible"] is True
    assert 0 <= got["y_within_pane"] <= 80, (
        f"the 3D button is {got['y_within_pane']} px down a {got['pane_height']} px pane — "
        "that is the same 'present but off the bottom' failure it was meant to fix"
    )
    before, after, back = got["toggle"]
    assert [before, after, back] == [2, 3, 2], "clicking it does not actually change ndisplay"
    assert got["checked_follows_dims"] is True
    # One owner: dims. The button READS it, it does not keep a second copy.
    assert got["follows_model_write"] is True
    assert got["unchecks_on_model_write"] is False


# ---------------------------------------------------------------- the grouped tree
#
# 5 processing layers x 4 channels + 4 raw = 24 rows in a FLAT LayerList. napari 0.6.6 has no
# layer groups (zero LayerGroup symbols; upstream #2229 open since Feb 2021), and
# `channel_axis=` provably splits into one layer per channel, so 4-layers-per-operator is
# idiomatic napari rather than our mistake. Both shipped precedents -- brainglobe's
# napari-experimental and PartSeg -- answer this by REPLACING THE LAYER-LIST UI, not by capping
# the layer count. This is that: a two-level view over the same layers.

from PyQt5.QtCore import Qt                                          # noqa: E402
from PyQt5.QtWidgets import QApplication                             # noqa: E402

import numpy as np                                                   # noqa: E402

from squidmip._napari_view import MosaicLayers                       # noqa: E402
from squidmip._layer_tree import MosaicTree                          # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _img(seed=0, shape=(8, 8)):
    return np.random.default_rng(seed).integers(0, 4000, shape, dtype="uint16")


@pytest.fixture
def mosaic():
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
    """The group row for *op*, found BY NAME.

    Positional lookup couples every test to the display order, and the display order is napari's
    to choose (topmost layer first), not ours. Asking by name is what these tests actually mean.
    """
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
    """24 flat rows become 5 collapsible ones. That is the whole point."""
    m = tree.model()
    assert m.rowCount() == 2, "processing layers are the top level"
    # TOPMOST FIRST -- napari's own layer-list convention, which this tree now mirrors rather
    # than inventing a second order. The fixture adds raw then stitched, so stitched is on top.
    assert [m.data(_op_index(tree, r), Qt.DisplayRole) for r in range(2)] == ["stitched", "raw"]
    for r, op in enumerate(["stitched", "raw"]):
        assert m.rowCount(_op_index(tree, r)) == 4
        assert [
            m.data(_ch_index(tree, r, c), Qt.DisplayRole) for c in range(4)
        ] == list(reversed(mosaic.channels(op)))


def test_the_tree_reads_visibility_off_the_layer_and_keeps_no_copy(tree, mosaic):
    """THE constraint. Two representations of one truth, hand-synced, is this project's
    dominant defect shape (4+ confirmed, most recently the contrast sync silently killed by
    layer recreation). The tree is a VIEW: napari's Image layer owns ``visible``."""
    m = tree.model()
    assert m.data(_ch_index_of(tree, "raw", "405"), Qt.CheckStateRole) == Qt.Checked

    # Change it BEHIND the tree's back, the way napari's own layer list does.
    mosaic.find("raw", "405").visible = False
    assert m.data(_ch_index_of(tree, "raw", "405"), Qt.CheckStateRole) == Qt.Unchecked, (
        "the tree is holding its own copy of visibility instead of reading the layer"
    )


def test_an_external_visibility_change_repaints_the_row(tree, mosaic, qapp):
    """Reading the truth is not enough if nothing tells Qt to re-read it. Without this the
    checkbox is correct only until someone touches napari's own list."""
    m = tree.model()
    seen = []
    m.dataChanged.connect(lambda tl, br, roles=None: seen.append(tl))
    mosaic.find("raw", "488").visible = False
    qapp.processEvents()
    assert seen, "changing layer.visible elsewhere left the tree's checkbox stale"


def test_toggling_a_processing_layer_toggles_its_four_channels(tree, mosaic):
    """The before/after-stitching gesture, at group level."""
    m = tree.model()
    assert m.setData(_op_index_of(tree, "stitched"), Qt.Unchecked, Qt.CheckStateRole) is True
    assert [ly.visible for ly in mosaic.group("stitched")] == [False] * 4
    assert [ly.visible for ly in mosaic.group("raw")] == [True] * 4, "it toggled the wrong group"

    m.setData(_op_index_of(tree, "stitched"), Qt.Checked, Qt.CheckStateRole)
    assert [ly.visible for ly in mosaic.group("stitched")] == [True] * 4


def test_a_group_check_state_is_derived_from_its_channels_not_stored(tree, mosaic):
    """napari-experimental keeps ``GroupLayer._visible`` and documents the consequence: nothing
    syncs it upward, so a group checkbox drifts out of step with its own contents. We derive it
    instead -- there is no group state to drift."""
    m = tree.model()
    assert m.data(_op_index_of(tree, "raw"), Qt.CheckStateRole) == Qt.Checked

    mosaic.find("raw", "561").visible = False
    assert m.data(_op_index_of(tree, "raw"), Qt.CheckStateRole) == Qt.PartiallyChecked, (
        "one hidden channel out of four is neither on nor off"
    )
    for ly in mosaic.group("raw"):
        ly.visible = False
    assert m.data(_op_index_of(tree, "raw"), Qt.CheckStateRole) == Qt.Unchecked


def test_toggling_one_channel_writes_that_layer_only(tree, mosaic):
    m = tree.model()
    m.setData(_ch_index_of(tree, "raw", "561"), Qt.Unchecked, Qt.CheckStateRole)
    assert [ly.visible for ly in mosaic.group("raw")] == [True, True, False, True]


def test_the_tree_survives_layers_being_destroyed_and_recreated(tree, mosaic, qapp):
    """_load_mosaic (_viewer.py:5092) destroys and recreates every layer on each region change.
    That already killed the contrast sync silently, because the subscription was bound to layer
    OBJECTS that no longer existed. Identity here is (op, channel) out of layer.metadata, so a
    rebuilt layer is the same row -- and the checkbox drives the NEW object."""
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

    # And the SUBSCRIPTION has to be rebuilt too, not just the rows. Subscribing once at
    # construction is exactly what killed on_user_contrast: it kept listening to layers that
    # no longer existed and reported nothing, forever, without an error.
    seen = []
    m.dataChanged.connect(lambda tl, br, roles=None: seen.append(tl))
    mosaic.find("stitched", "638").visible = False
    qapp.processEvents()
    assert seen, (
        "after layers were recreated the tree stopped hearing visibility changes -- it is "
        "still subscribed to the destroyed objects"
    )


def test_foreign_layers_never_appear_in_the_tree(tree, mosaic, qapp):
    """A points layer a plugin added is not one of our mosaics. Tolerated, not shown, and above
    all not crashed on -- key_of returns None for anything without our metadata."""
    m = tree.model()
    mosaic.model.add_image(_img(7), name="somebody else's layer")
    qapp.processEvents()
    assert m.rowCount() == 2
    assert [m.data(_op_index(tree, r), Qt.DisplayRole) for r in range(2)] == ["stitched", "raw"]


def test_checkboxes_are_actually_offered_to_the_user(tree):
    """A model that answers CheckStateRole but does not set ItemIsUserCheckable renders a tree
    with no checkboxes at all -- readable, unclickable, and green under every test above."""
    m = tree.model()
    for idx in (_op_index(tree, 0), _ch_index(tree, 0, 0)):
        assert m.flags(idx) & Qt.ItemIsUserCheckable
        assert m.flags(idx) & Qt.ItemIsEnabled


# ------------------------------------------------- the tree, mounted in the real pane
#
# ALONGSIDE napari's own controls, not instead of them. napari-experimental's ethos is that the
# main layer list should only add/remove layers; PartSeg goes further and deletes napari's docks
# outright (dockLayerList.deleteLater()). We do neither: dc0f288 embeds the REAL napari window
# precisely because hand-rebuilt controls were rejected as "not napari", and the two surfaces
# cannot conflict here because both write the same layer.visible. Mounting through napari's own
# public Window.add_dock_widget puts the tree where napari puts its own panels.

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

    from PyQt5.QtCore import Qt as _Qt
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
    out["leaf_state_after_external_change"] = int(_m.data(_405, _Qt.CheckStateRole))
    out["group_state_after_external_change"] = int(
        _m.data(_raw, _Qt.CheckStateRole)
    )
"""


def test_the_tree_is_mounted_beside_naparis_own_controls(tmp_path):
    """The tree ships inside the real pane, and napari's own list survives next to it."""
    got = _run_qt(_MOUNT_SCRIPT, tmp_path, "MOUNT")

    assert got["tree_exists"] is True
    assert got["tree_visible"] is True
    assert got["tree_is_in_our_pane"] is True, "the tree was mounted somewhere the user cannot see"
    assert got["rows"] == 2
    assert got["children_of_first"] == 4
    # The grouped tree REPLACES the flat list. Julio: "Why do we have the layer list tab in our
    # napari variant if we don't want the number of layers to explode precisely?" Keeping a tab
    # that shows all 24 rows defeats the reason the tree exists.
    assert got["flat_layer_list_visible"] is False, (
        "napari's flat layer list is still showing, so the layer explosion is still on screen"
    )
    # ...but napari's LAYER CONTROLS are a different surface and must stay: they own contrast.
    assert got["layer_controls_still_there"] >= 1, (
        "napari's layer controls disappeared -- contrast has no owner on screen"
    )
    # 506c813, again: adding a dock must never move the canvas out of napari's window.
    assert got["canvas_still_inside_napari_window"] is True
    assert got["group_hidden"] == [False] * 4
    assert got["other_group_untouched"] == [True] * 4
    assert got["leaf_state_after_external_change"] == 0        # Qt.Unchecked
    assert got["group_state_after_external_change"] == 1       # Qt.PartiallyChecked


# ------------------------------------- napari PAINTS these rows; we do not imitate its look
#
# Julio: "I still don't like these napari layer UX. The original napari layer widgets were way
# more beautiful." The gap was never styling. Qt's default item view draws a native checkbox and
# a string; napari's LayerDelegate draws an eye, a per-type icon and the layer's live thumbnail.
# Imitating that would be a second renderer to keep in step with napari's own.

def test_the_rows_are_painted_by_naparis_own_delegate(qapp, mosaic):
    """Not "a delegate is set" -- napari's, by class. A fallback that silently kept Qt's default
    would leave the ugly rows on screen with every test still green."""
    from squidmip import _layer_tree as LT

    tree = LT.MosaicTree(mosaic)
    delegate = tree.itemDelegate()
    assert type(delegate).__module__.startswith("napari."), (
        f"rows are painted by {type(delegate).__module__}.{type(delegate).__name__}, "
        "not by napari -- the tree fell back to Qt's default look"
    )
    assert type(delegate).__name__ == "LayerDelegate"


def test_the_model_serves_every_role_that_delegate_paints_from(qapp, mosaic):
    """The delegate is only as good as the roles behind it. Miss one and the row degrades
    silently: no thumbnail, or a folder icon on a channel, with nothing raised."""
    from squidmip import _layer_tree as LT

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
    """Render the view for real, into a pixmap.

    THE test that was missing. Serving the delegate's roles is not the same as surviving its
    paint: `LayerDelegate._paint_thumbnail` calls `index.model().sourceModel().all_loaded()`,
    because in napari the view always sits behind a QSortFilterProxyModel. Ours does not, so
    every repaint raised AttributeError -- 54 tracebacks in a single launch -- while the
    role-level tests stayed green, because reading a role never paints a row.

    Qt swallows exceptions raised inside paint(), so this installs an excepthook to catch them.

    MUTATION: remove `sourceModel` or `all_loaded` from the model and this goes red.
    """
    import sys

    from PyQt5.QtGui import QPixmap

    from squidmip import _layer_tree as LT

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


# ------------------------------- selecting a row here selects the LAYER, so contrast follows
#
# Julio: "when I click on the mosaic layers I cannot adjust contrast, I would have to go from
# the processing layers." napari's contrast/gamma/colormap panel renders whatever is in
# viewer.layers.selection. Our tree had its own Qt selection and never touched napari's, so
# clicking a channel highlighted a row and changed nothing the controls could see.

def test_selecting_a_channel_row_selects_that_layer_in_napari(qapp, mosaic):
    """MUTATION: drop the currentChanged connection and this goes red."""
    from squidmip import _layer_tree as LT

    tree = LT.MosaicTree(mosaic)
    tree.setCurrentIndex(_ch_index_of(tree, "raw", "488"))

    selection = mosaic.model.layers.selection
    assert [ly.name for ly in selection] == ["raw · 488"]
    assert selection.active is mosaic.find("raw", "488"), (
        "no ACTIVE layer, so napari's contrast panel has nothing to show"
    )


def test_selecting_a_processing_layer_selects_all_of_its_channels(qapp, mosaic):
    """A group means its channels, so selecting it selects them -- and napari then shows the
    shared controls for the set.

    MUTATION: set `selection.active` for the group case and this goes red. napari's `active`
    setter calls select_only(), which silently collapses a four-channel selection to one; that
    is what it did before this test existed.
    """
    from squidmip import _layer_tree as LT

    tree = LT.MosaicTree(mosaic)
    tree.setCurrentIndex(_op_index_of(tree, "raw"))

    assert sorted(ly.name for ly in mosaic.model.layers.selection) == [
        "raw · 405", "raw · 488", "raw · 561", "raw · 638",
    ]


def test_the_tree_lists_topmost_first_like_naparis_own_layer_list(qapp, mosaic):
    """Julio: "stack order is inverted -> that points to a bad data model on channels."

    One list, two display rules: napari renders its LayerList reversed (last added on top), and
    our tree rendered it in insertion order. Whichever is "right", having two is the defect.
    napari owns the order; we mirror it.

    MUTATION: drop either `reversed(...)` in `refresh` and this goes red.
    """
    from squidmip import _layer_tree as LT

    tree = LT.MosaicTree(mosaic)
    model = tree.model()
    ours = [model.data(model.index(r, 0), Qt.DisplayRole) for r in range(model.rowCount())]

    # napari's own order, as its list widget shows it: last added first.
    napari_order = []
    for layer in reversed(list(mosaic.model.layers)):
        op = layer.metadata["squidmip"]["op"]
        if op not in napari_order:
            napari_order.append(op)
    assert ours == napari_order, f"tree shows {ours}, napari shows {napari_order}"

    raw = _op_index_of(tree, "raw")
    channels = [model.data(model.index(r, 0, raw), Qt.DisplayRole)
                for r in range(model.rowCount(raw))]
    assert channels == list(reversed(mosaic.channels("raw"))), "channels are not topmost-first"
