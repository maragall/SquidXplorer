"""The way BACK to the plate, from a window the plate opened.

Spencer, 2026-07-30: the plate "can get lost easily". It is the smallest window on the desktop
and every view spawned from it is larger, so after two or three wells the thing you navigate FROM
is underneath everything you navigated TO. ``Collapse all`` is not the answer -- it minimises the
VIEWS and leaves the plate wherever it already was, which may be behind them.

Two seams, pinned separately because they fail separately:

* ``ViewerManager.raise_plate`` -- does the registry find the plate at all. It reaches it through
  its own Qt parent (``PlateWindow`` builds it as ``ViewerManager(parent=self)``), so this pins
  the assumption that the parent is the plate, which is the part a refactor would quietly break.
* the BUTTON -- is it on the window, and is it wired. Clicked, not called: the lesson in
  ``tools/walkthrough.py`` is that a handler-calling test stays green against a button connected
  to nothing.

The button test builds a REAL ``RegionViewer`` through the real registry, with ``napari_pane_stub``
replacing the one seam that needs a GL context. Same arrangement as ``tests/test_view_settings.py``.
"""

from __future__ import annotations

import gc
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede the Qt import

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)

from qtpy.QtWidgets import QApplication, QPushButton, QWidget  # noqa: E402

from squidmip._region_viewer import ViewerManager  # noqa: E402

from .conftest import REGIONS  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)
    return app


class _FakePlate(QWidget):
    """A ``PlateWindow`` as the registry raises one: counts what was asked of it.

    Counting rather than flagging, so "raised twice" is visible -- and so ``restored`` can be
    asserted to stay at ZERO for a plate that was never minimised, which is the un-maximise
    regression this guards.
    """

    def __init__(self) -> None:
        super().__init__()
        self.raised = 0
        self.activated = 0
        self.restored = 0
        self._pretend_minimised = False

    def isMinimized(self) -> bool:           # noqa: N802 - Qt naming
        return self._pretend_minimised

    def raise_(self) -> None:
        self.raised += 1

    def activateWindow(self) -> None:        # noqa: N802 - Qt naming
        self.activated += 1

    def showNormal(self) -> None:            # noqa: N802 - Qt naming
        self.restored += 1
        self._pretend_minimised = False


# --------------------------------------------------- the registry finds the plate through Qt


def test_it_raises_the_window_that_owns_the_registry(qapp):
    """PlateWindow builds the manager as ViewerManager(parent=self), so the Qt parent IS the
    plate. If a refactor reparents the manager, this is the test that says so."""
    plate = _FakePlate()
    mgr = ViewerManager(parent=plate)
    try:
        assert mgr.raise_plate() is True
        assert (plate.raised, plate.activated) == (1, 1)
    finally:
        mgr._mem_timer.stop()


def test_it_does_not_un_maximise_a_plate_that_was_not_minimised(qapp):
    """Raising a window and RESIZING it are different requests; this button makes only the first.

    ``focus()`` calls showNormal unconditionally and is right to -- it restores collapsed VIEWS.
    Copying that here would drop a maximised plate back to its restored size on every click.
    """
    plate = _FakePlate()
    mgr = ViewerManager(parent=plate)
    try:
        assert mgr.raise_plate() is True
        assert plate.restored == 0, "the plate was un-maximised just to bring it forward"
    finally:
        mgr._mem_timer.stop()


def test_it_restores_a_plate_that_really_was_minimised(qapp):
    """The other half: a minimised plate cannot be raised into view without showNormal."""
    plate = _FakePlate()
    plate._pretend_minimised = True
    mgr = ViewerManager(parent=plate)
    try:
        assert mgr.raise_plate() is True
        assert plate.restored == 1
        assert plate.raised == 1
    finally:
        mgr._mem_timer.stop()


def test_it_reports_failure_when_there_is_no_plate(qapp):
    """A standalone manager has no parent. Returning False lets the window SAY so; returning True
    would give the user a button that appears to work and does nothing."""
    mgr = ViewerManager()
    try:
        assert mgr.raise_plate() is False
    finally:
        mgr._mem_timer.stop()


# ------------------------------------------------------- the button, on a real child window


@pytest.fixture
def manager(qapp, napari_pane_stub, squid_dataset):
    """A real ViewerManager over the real reader, parented to a fake plate.

    Same shape as tests/test_view_settings.py's fixture: real registry, real RegionViewer, with
    only the GL pane stubbed. The parent is what ``raise_plate`` will go looking for.
    """
    from squidmip import open_reader

    root, _arrays = squid_dataset
    reader = open_reader(str(root))
    plate = _FakePlate()
    mgr = ViewerManager(reader, reader.metadata, parent=plate)
    mgr._test_plate = plate
    try:
        yield mgr
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()
        gc.collect()
        qapp.processEvents()


def _plate_button(win) -> QPushButton:
    """The button as the user finds it: by its face."""
    found = [b for b in win.findChildren(QPushButton) if "plate" in b.text()]
    assert len(found) == 1, f"expected one plate button, found {[b.text() for b in found]}"
    return found[0]


def test_every_child_window_carries_the_button(qapp, manager):
    win = manager.open([REGIONS[0]])
    assert win is not None
    assert _plate_button(win).text() == "▣ plate"


def test_clicking_it_raises_the_plate(qapp, manager):
    """The whole point, through the real wiring: button -> handler -> registry -> plate."""
    win = manager.open([REGIONS[0]])
    assert win is not None
    plate = manager._test_plate
    before = plate.raised

    _plate_button(win).click()

    assert plate.raised == before + 1, "the button did not reach the plate"
    assert plate.activated >= 1
