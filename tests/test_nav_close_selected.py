""""Close selected views" acts on the SELECTION, and on all of it.

The navigator tree has been ``ExtendedSelection`` since 1073999, and the plate wash already
followed the whole selection (``_on_selection_changed`` reads ``selectedItems()``). The close
button did not: it read ``currentItem()``, so selecting four views and pressing it closed one and
left three open.

Two distinct defects, and the rename only names the first:

* **Plural.** All selected rows close, not just one.
* **Selection, not focus.** ``currentItem()`` is the focus rectangle, which is a different
  question from "what is selected" -- it survives ctrl-clicking a row back OFF, so the old button
  could close a view the user had just deselected.

The button is CLICKED here rather than the handler being called. That is the lesson recorded in
``tools/walkthrough.py``: "a Re-dock button was broken from the day it shipped and no test
noticed, because every test called the handler directly instead of clicking." A test that calls
``nav._close_selected()`` stays green if the button is wired to nothing at all, and this button
was wired but wrong, which is the same class of miss.

The MANAGER is real. Only the windows are fakes, because a real ``RegionViewer`` builds a napari
GL canvas that cannot exist under ``QT_QPA_PLATFORM=offscreen`` (the whole suite's platform). So
the path under test is navigator -> real ``ViewerManager.close`` -> the window, and the only
pretend part is the thing at the end that gets closed.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede the Qt import

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication, QPushButton, QTreeWidgetItem  # noqa: E402

from squidmip._region_viewer import OpenViewList, ViewerManager  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)
    return app


class _FakeWindow:
    """A ``RegionViewer`` as the registry and the tree read one.

    ``close`` COUNTS rather than just recording a bool: closing the same window twice would be a
    real defect (the second call means the loop is reading a stale row), and a bool cannot see it.
    """

    def __init__(self, window_id: int, title: str, parent_id=None) -> None:
        self.window_id = int(window_id)
        self.parent_id = parent_id
        self._title = title
        self.closes = 0

    def windowTitle(self) -> str:           # noqa: N802 - Qt naming
        return self._title

    def close(self) -> None:
        self.closes += 1

    # raise_views() touches these the moment a row is selected
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
        manager._windows[w.window_id] = w   # what _spawn does, minus the GL canvas
    return manager, OpenViewList(manager)


def _close_button(nav) -> QPushButton:
    """The button as the USER finds it: by its face, not by a private attribute."""
    found = [b for b in nav.findChildren(QPushButton) if b.text().startswith("Close")]
    assert len(found) == 1, f"expected one Close button, found {[b.text() for b in found]}"
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


# ------------------------------------------------------------------ the name on the button


def test_the_button_says_views_plural(qapp):
    """The label is half the fix. "Close view" on a multi-select list told the user the button was
    singular, and it was -- naming it honestly is what makes the behaviour below discoverable."""
    _, nav = _nav(_FakeWindow(1, "[1] A1"))
    assert _close_button(nav).text() == "Close selected views"


# ------------------------------------------------------------------ it closes ALL of them


def test_clicking_it_closes_every_selected_view(qapp):
    """The reported defect: select several, press it, and only one closes."""
    a, b, c = _FakeWindow(1, "[1] A1"), _FakeWindow(2, "[2] A6"), _FakeWindow(3, "[3] D1")
    _, nav = _nav(a, b, c)

    _select_only(nav, 1, 3)
    _close_button(nav).click()

    assert (a.closes, c.closes) == (1, 1), "a selected view was left open"
    assert b.closes == 0, "an UNSELECTED view was closed"


def test_it_closes_nested_roi_children_when_they_are_selected(qapp):
    """ROI windows nest under their parent, so a selected child is a row like any other and must
    close on its own without dragging its parent with it."""
    parent = _FakeWindow(1, "[1] A1")
    child = _FakeWindow(2, "[2] A1 ROI", parent_id=1)
    _, nav = _nav(parent, child)
    assert 2 in _items(nav), "the nested child never appeared as a row"

    _select_only(nav, 2)
    _close_button(nav).click()

    assert child.closes == 1
    assert parent.closes == 0


# --------------------------------------------------- selection, not the focus rectangle


def test_it_ignores_the_focus_rectangle_and_obeys_the_selection(qapp):
    """``currentItem()`` is not ``selectedItems()``.

    Current follows the keyboard/focus and stays put when a row is ctrl-clicked back off, so the
    old handler would close a view the user had explicitly deselected. Pinned by making the two
    disagree: current is A, the selection is B.
    """
    a, b = _FakeWindow(1, "[1] A1"), _FakeWindow(2, "[2] A6")
    _, nav = _nav(a, b)
    rows = _items(nav)

    nav._tree.setCurrentItem(rows[1])       # focus rectangle on A ...
    _select_only(nav, 2)                    # ... selection on B alone
    assert nav._tree.currentItem() is rows[1], "the test did not manage to split the two"

    _close_button(nav).click()

    assert b.closes == 1, "the selected view did not close"
    assert a.closes == 0, "the button closed the CURRENT row instead of the selected one"


def test_it_closes_nothing_when_nothing_is_selected(qapp):
    """An empty selection is not "fall back to whatever is current". With no rows selected the
    button has nothing to act on, and doing something anyway is how the deselected-row case above
    used to happen."""
    a = _FakeWindow(1, "[1] A1")
    _, nav = _nav(a)
    nav._tree.setCurrentItem(_items(nav)[1])
    nav._tree.clearSelection()

    _close_button(nav).click()

    assert a.closes == 0
