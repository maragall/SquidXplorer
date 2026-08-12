""""Close selected views" acts on the SELECTION, and on all of it."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication, QPushButton, QTreeWidgetItem  # noqa: E402

from squidxplorer._region_viewer import OpenViewList, ViewerManager  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


class _FakeWindow:
    """A ``RegionViewer`` as the registry and the tree read one."""

    def __init__(self, window_id: int, title: str, parent_id=None) -> None:
        self.window_id = int(window_id)
        self.parent_id = parent_id
        self._title = title
        self.closes = 0
        self.manager = None

    def windowTitle(self) -> str:           # noqa: N802 - Qt naming
        return self._title

    def close(self) -> None:
        """DEREGISTERS, which is the whole hazard `_close_selected` is written against."""
        self.closes += 1
        if self.manager is not None:
            self.manager._on_window_closed(self)

    def showNormal(self) -> None:           # noqa: N802 - Qt naming
        pass

    def raise_(self) -> None:
        pass

    def activateWindow(self) -> None:       # noqa: N802 - Qt naming
        pass


def _nav(*wins):
    """A real manager holding *wins*, and a navigator over it."""
    manager = ViewerManager()
    for w in wins:
        manager._windows[w.window_id] = w
        w.manager = manager
    return manager, OpenViewList(manager)


def _close_button(nav) -> QPushButton:
    """The SELECTION close button as the USER finds it: by its face, not by a private attribute."""
    found = [b for b in nav.findChildren(QPushButton) if b.text() == "Close selected views"]
    assert len(found) == 1, (
        f"expected one 'Close selected views' button, found "
        f"{[b.text() for b in nav.findChildren(QPushButton)]}")
    return found[0]


def _items(nav) -> "dict[int, QTreeWidgetItem]":
    """Every row in the tree, nested ones included, keyed by window id."""
    out: "dict[int, QTreeWidgetItem]" = {}

    def walk(item):
        wid = item.data(0, Qt.UserRole)
        if wid is not None:
            out[int(wid)] = item
        for i in range(item.childCount()):
            walk(item.child(i))

    tree = nav._tree
    for i in range(tree.topLevelItemCount()):
        walk(tree.topLevelItem(i))
    return out


def _select_only(nav, *ids) -> None:
    nav._tree.clearSelection()
    rows = _items(nav)
    for wid in ids:
        rows[wid].setSelected(True)


def test_the_button_says_views_plural(qapp):
    _, nav = _nav(_FakeWindow(1, "[1] A1"))
    assert _close_button(nav).text() == "Close selected views"


def test_clicking_it_closes_every_selected_view(qapp):
    a, b, c = _FakeWindow(1, "[1] A1"), _FakeWindow(2, "[2] A6"), _FakeWindow(3, "[3] D1")
    _, nav = _nav(a, b, c)

    _select_only(nav, 1, 3)
    _close_button(nav).click()

    assert (a.closes, c.closes) == (1, 1), "a selected view was left open"
    assert b.closes == 0, "an UNSELECTED view was closed"


def test_it_closes_nested_roi_children_when_they_are_selected(qapp):
    parent = _FakeWindow(1, "[1] A1")
    child = _FakeWindow(2, "[2] A1 ROI", parent_id=1)
    _, nav = _nav(parent, child)
    assert 2 in _items(nav), "the nested child never appeared as a row"

    _select_only(nav, 2)
    _close_button(nav).click()

    assert child.closes == 1
    assert parent.closes == 0


def test_it_ignores_the_focus_rectangle_and_obeys_the_selection(qapp):
    a, b = _FakeWindow(1, "[1] A1"), _FakeWindow(2, "[2] A6")
    _, nav = _nav(a, b)
    rows = _items(nav)

    nav._tree.setCurrentItem(rows[1])
    _select_only(nav, 2)
    assert nav._tree.currentItem() is rows[1], "the test did not manage to split the two"

    _close_button(nav).click()

    assert b.closes == 1, "the selected view did not close"
    assert a.closes == 0, "the button closed the CURRENT row instead of the selected one"


def _collapse_button(nav) -> QPushButton:
    found = [b for b in nav.findChildren(QPushButton) if b.text().startswith("Collapse")]
    assert len(found) == 1
    return found[0]


def test_close_is_greyed_out_until_something_is_selected_and_says_why(qapp):
    a = _FakeWindow(1, "[1] A1")
    _, nav = _nav(a)
    btn = _close_button(nav)

    nav._tree.clearSelection()
    assert not btn.isEnabled(), "a button with nothing to close looked clickable"
    assert "nothing to close" in btn.toolTip(), btn.toolTip()

    _select_only(nav, 1)
    assert btn.isEnabled(), "the button stayed dead after a row was selected"
    assert "shift/ctrl-click" in btn.toolTip()


def test_collapse_all_is_greyed_out_when_no_view_is_open_and_says_why(qapp):
    _, empty = _nav()
    btn = _collapse_button(empty)
    assert not btn.isEnabled()
    assert "nothing to minimise" in btn.toolTip(), btn.toolTip()

    _, one = _nav(_FakeWindow(1, "[1] A1"))
    assert _collapse_button(one).isEnabled()


def test_it_closes_nothing_when_nothing_is_selected(qapp):
    a = _FakeWindow(1, "[1] A1")
    _, nav = _nav(a)
    nav._tree.setCurrentItem(_items(nav)[1])
    nav._tree.clearSelection()

    _close_button(nav).click()

    assert a.closes == 0
