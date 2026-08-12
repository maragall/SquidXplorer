"""The way back to the plate, from a window the plate opened.

Two seams pinned separately: ``ViewerManager.raise_plate`` (the registry finds the plate
through its Qt parent) and the button itself, clicked rather than called.
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

from squidxplorer._region_viewer import ViewerManager  # noqa: E402

from .conftest import REGIONS  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


class _FakePlate(QWidget):
    """A ``PlateWindow`` as the registry raises one: counts what was asked of it."""

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


def test_it_raises_the_window_that_owns_the_registry(qapp):
    """The Qt parent IS the plate; a refactor that reparents the manager fails here."""
    plate = _FakePlate()
    mgr = ViewerManager(parent=plate)
    try:
        assert mgr.raise_plate() is True
        assert (plate.raised, plate.activated) == (1, 1)
    finally:
        mgr._mem_timer.stop()


def test_it_does_not_un_maximise_a_plate_that_was_not_minimised(qapp):
    """Raising a window and resizing it are different requests; this button makes only the first."""
    plate = _FakePlate()
    mgr = ViewerManager(parent=plate)
    try:
        assert mgr.raise_plate() is True
        assert plate.restored == 0, "the plate was un-maximised just to bring it forward"
    finally:
        mgr._mem_timer.stop()


def test_it_restores_a_plate_that_really_was_minimised(qapp):
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
    """A standalone manager has no parent; False lets the window say so."""
    mgr = ViewerManager()
    try:
        assert mgr.raise_plate() is False
    finally:
        mgr._mem_timer.stop()


@pytest.fixture
def manager(qapp, napari_pane_stub, squid_dataset):
    """A real ViewerManager over the real reader, parented to a fake plate."""
    from squidxplorer import open_reader

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
    """Through the real wiring: button -> handler -> registry -> plate."""
    win = manager.open([REGIONS[0]])
    assert win is not None
    plate = manager._test_plate
    before = plate.raised

    _plate_button(win).click()

    assert plate.raised == before + 1, "the button did not reach the plate"
    assert plate.activated >= 1


# The controls chip: the plate comes forward always; the tab opens for the operator
# in the window's dropdown; a window with no results still raises and says what is missing.


def _controls_button(win) -> QPushButton:
    found = [b for b in win.findChildren(QPushButton) if "controls" in b.text()]
    assert len(found) == 1, f"expected one controls button, found {[b.text() for b in found]}"
    return found[0]


def _selects(win, key: str, label: str = "") -> None:
    """Put *key* in this window's "Operators for this window" dropdown and select it."""
    combo = win._op_combo
    combo.clear()
    combo.addItem(label or key, key)
    combo.setCurrentIndex(0)
    combo.setEnabled(True)


def _holds(win, ops):
    """Make the window report *ops* as the processing layers it holds, the way the pane would."""
    class _Mosaic:
        def ops(self):
            return ["raw", *ops]          # raw is always there
        def visible_op(self):
            return ops[0] if ops else "raw"
    win._pane.mosaic = _Mosaic()


def _wired(manager):
    plate = manager._test_plate
    plate.activated_ops = []
    plate._activate_operator = plate.activated_ops.append
    return plate


def test_every_child_window_carries_the_controls_button(qapp, manager):
    win = manager.open([REGIONS[0]])
    assert win is not None
    assert _controls_button(win).text() == "⚙ controls"


def test_it_opens_the_tab_for_THE_OPERATOR_IN_THE_DROPDOWN(qapp, manager):
    """One operator: the one the Run button beside it would use."""
    win = manager.open([REGIONS[0]])
    plate = _wired(manager)
    _holds(win, ["decon", "bgsub"])          # a history of what has run, not the chip's question
    _selects(win, "stitch", "Stitch (register + fuse)")
    before = plate.raised

    _controls_button(win).click()

    assert plate.raised == before + 1, "the controls button did not raise the plate"
    assert plate.activated_ops == ["stitch"], (
        f"the chip opened {plate.activated_ops}, not the operator in the dropdown")


def test_it_says_what_the_operator_will_run_with(qapp, manager):
    """The chip's own line names the mode and the parameters."""
    win = manager.open([REGIONS[0]])
    _wired(manager)
    _selects(win, "stitch", "Stitch (register + fuse)")
    said_before = len(win._pane.said)

    _controls_button(win).click()

    said = " ".join(win._pane.said[said_before:])
    assert "Stitch" in said, f"the chip did not name the operator: {said!r}"
    assert "2D" in said or "3D" in said, f"the chip did not say which mode it would run in: {said!r}"


def test_the_chip_refuses_out_loud_when_no_operator_is_selected(qapp, manager):
    """An empty dropdown still gets the trip back to the plate, and is told why."""
    win = manager.open([REGIONS[0]])
    plate = _wired(manager)
    win._op_combo.clear()
    before = plate.raised

    _controls_button(win).click()

    assert plate.raised == before + 1, "a window with no operator selected lost its trip back"
    assert plate.activated_ops == [], f"it opened a tab anyway: {plate.activated_ops}"


def test_the_plate_COMES_FORWARD_even_when_the_window_holds_no_operator(qapp, manager):
    """The raise must never be gated on finding an operator."""
    win = manager.open([REGIONS[0]])
    plate = _wired(manager)
    _holds(win, [])                                   # nothing run on this window yet
    before, said_before = plate.raised, len(win._pane.said)

    _controls_button(win).click()

    assert plate.raised == before + 1, "the plate did not come forward on a window with no results"
    assert plate.activated_ops == [], "there was nothing to open"
    said = " ".join(win._pane.said[said_before:]).lower()
    assert "run" in said or "nothing" in said, f"it did not say what was missing: {said!r}"


def test_it_reads_the_operators_off_the_LAYERS_the_window_really_holds(qapp, manager):
    """No ``_holds``: the chip must find the operators through the production path."""
    import numpy as np

    from squidxplorer._result import Extent, Result

    win = manager.open([REGIONS[0]])
    plate = _wired(manager)
    plane = np.zeros((4, 4), "uint16")
    for op in ("decon", "bgsub"):
        win.deliver_result(op, Result.of(Extent(region_id=win.current_region()), [plane],
                                         channels=("405",), z_depth=1, pixel_size_um=1.0,
                                         dtype=plane.dtype, kind="intensity"), visible=True)
    assert win._window_operators() == ["decon", "bgsub"]

    _selects(win, "decon")
    _controls_button(win).click()

    assert plate.activated_ops == ["decon"], (
        f"the chip opened {plate.activated_ops}, not the operator the dropdown names")


def test_a_panel_that_raises_is_NAMED_not_swallowed(qapp, manager):
    """If the panel cannot open, the window says which operator and what went wrong."""
    win = manager.open([REGIONS[0]])
    plate = manager._test_plate
    plate.activated_ops = []

    def _activate(op):
        raise RuntimeError("boom")

    plate._activate_operator = _activate
    _selects(win, "decon")
    said_before = len(win._pane.said)

    _controls_button(win).click()

    said = " ".join(win._pane.said[said_before:])
    assert "decon" in said and "boom" in said, f"the failure was not named: {said!r}"
