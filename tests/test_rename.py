"""Renaming a window moves the LABEL and moves nothing else.

A window's identity is RegionViewer.window_id, a per-process monotonic int. The title is a
rendering of that identity plus a mutable label, and nothing anywhere parses it. Rename safety
is asserted at the three places a rename could plausibly break something: logging, the data
model (ViewerManager._windows keyed by the int), and the navigator (rows carry the int under
Qt.UserRole).

Not pinned, and a real limitation: a rename does not survive a restart.
"""

from __future__ import annotations

import gc
import logging
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede the Qt import

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from squidxplorer._logpane import VIEW_FIELD  # noqa: E402
from squidxplorer._region_viewer import OpenViewList, ViewerManager  # noqa: E402

from .conftest import REGIONS  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


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


def _rows(nav) -> "dict[int, object]":
    """Every navigator row, nested ones included, keyed by the window id under Qt.UserRole."""
    out: dict = {}

    def walk(item):
        wid = item.data(0, Qt.UserRole)
        if wid is not None:
            out[int(wid)] = item
        for i in range(item.childCount()):
            walk(item.child(i))

    for i in range(nav._tree.topLevelItemCount()):
        walk(nav._tree.topLevelItem(i))
    return out


def test_rename_does_not_move_the_log_id(qapp, manager, caplog):
    """The bracket in a log line is built from window_id at construction, so a rename cannot
    reach it."""
    win = manager.open([REGIONS[0]])
    assert win is not None
    wid = win.window_id

    assert manager.rename(wid, "Deconvolution trial") is True

    with caplog.at_level(logging.INFO):
        win.log.info("mosaic loaded")
    rec = [r for r in caplog.records if hasattr(r, VIEW_FIELD)][-1]

    assert getattr(rec, VIEW_FIELD) == wid, "the rename moved the structured view id"
    assert rec.getMessage().startswith(f"[{wid}]"), (
        "the rename moved the log prefix, so a log line no longer points at its window")
    assert "Deconvolution trial" not in rec.getMessage(), (
        "the LABEL leaked into the log prefix; the prefix is the id and only the id")


def test_rename_does_not_move_the_target(qapp, manager):
    """Same window, same regions, same id after a rename."""
    win = manager.open([REGIONS[0], REGIONS[1]])
    wid = win.window_id
    before = manager.view_for(wid)

    assert manager.rename(wid, "the batch I care about") is True

    after = manager.view_for(wid)
    assert after is not None, "the window became unreachable by its id after a rename"
    assert after.window_id == wid
    assert after.regions == before.regions, "the rename moved the regions an operator would run on"
    assert manager._windows[wid] is win, "the registry key moved"
    assert manager.make_default(wid) is True, "an id-keyed manager call stopped resolving"
    assert manager.windows == [win]


def test_rename_reaches_the_navigator_and_keeps_the_bracket(qapp, manager):
    win = manager.open([REGIONS[0]])
    wid = win.window_id
    nav = OpenViewList(manager)
    assert _rows(nav)[wid].text(0) == f"[{wid}] {REGIONS[0]}"

    manager.rename(wid, "Deconvolution trial")

    row = _rows(nav)[wid]
    assert row.text(0) == f"[{wid}] Deconvolution trial", (
        "the navigator did not repaint, or it dropped the bracket")
    assert int(row.data(0, Qt.UserRole)) == wid, (
        "the row's identity moved with its text; selecting it would raise the wrong window")
    assert win.windowTitle() == f"[{wid}] Deconvolution trial"


def test_the_bracket_is_not_editable(qapp, manager):
    win = manager.open([REGIONS[0]])
    wid = win.window_id

    manager.rename(wid, "[99] pretend")

    assert win.windowTitle() == f"[{wid}] [99] pretend"
    assert win.window_id == wid
    assert manager.view_for(wid).window_id == wid


def test_a_blank_name_is_a_refusal_not_a_reset(qapp, manager):
    """An empty box means "I changed my mind", not "wipe the region-derived name"."""
    win = manager.open([REGIONS[0]])
    wid = win.window_id
    before = win.display_name

    assert manager.rename(wid, "   ") is False
    assert manager.rename(wid, "") is False
    assert win.display_name == before
    assert win.windowTitle() == f"[{wid}] {before}"

    assert manager.rename(wid, "renamed") is True
    assert manager.rename(wid, None) is True, "None is the explicit undo"
    assert win.display_name == before, "restoring the derived name did not work"


def test_renaming_an_unknown_id_refuses_rather_than_appearing_to_work(qapp, manager):
    win = manager.open([REGIONS[0]])
    assert manager.rename(win.window_id + 999, "ghost") is False


def test_two_windows_may_share_a_name_because_the_bracket_disambiguates(qapp, manager):
    a, b = manager.open([REGIONS[0]]), manager.open([REGIONS[1]])
    manager.rename(a.window_id, "same")
    manager.rename(b.window_id, "same")

    assert a.display_name == b.display_name == "same"
    assert a.windowTitle() != b.windowTitle()
    assert {v.window_id for v in manager.views()} == {a.window_id, b.window_id}


def test_an_roi_childs_parent_reference_keeps_pointing_at_the_id(qapp, manager):
    """The "◂ view N" suffix references the stable id, not the mutable label."""
    parent = manager.open([REGIONS[0]])
    child = manager.open_child([REGIONS[0]], roi_bbox=(0.0, 0.0, 10.0, 10.0),
                               parent_id=parent.window_id)

    manager.rename(parent.window_id, "renamed parent")

    assert f"◂ view {parent.window_id}" in child.display_name
    assert child.parent_id == parent.window_id


def test_the_view_name_is_the_label_without_the_bracket(qapp, manager):
    """View.name is the label alone; _run_scope.describe_view_target composes id + name."""
    win = manager.open([REGIONS[0]])
    wid = win.window_id
    manager.rename(wid, "Deconvolution trial")

    view = manager.view_for(wid)
    assert view.name == "Deconvolution trial"
    assert not view.name.startswith("["), "the id is back inside the name field"
    assert f"[{view.window_id}] {view.name}" == win.windowTitle(), (
        "the printer's composition no longer reproduces the title bar")
