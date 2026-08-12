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

from squidxplorer._region_viewer import ViewerManager  # noqa: E402

from .conftest import REGIONS  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
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
    """The whole point, through the real wiring: button -> handler -> registry -> plate."""
    win = manager.open([REGIONS[0]])
    assert win is not None
    plate = manager._test_plate
    before = plate.raised

    _plate_button(win).click()

    assert plate.raised == before + 1, "the button did not reach the plate"
    assert plate.activated >= 1


# --------------------------------------------------- ⚙ controls: the plate AND the operator tabs
#
# Julio, 2026-08-05, correcting the first cut: "the controls is actually for the 'operators for
# this window'. And it is not bringing up the plateview window."
#
# Both corrections were one mistake. v1 asked `visible_op()` -- ONE operator, the lit one -- and
# made RAISING THE PLATE CONDITIONAL on finding it, so a window showing raw got nothing at all: no
# tab, and not even the trip back that `▣ plate` gives unconditionally.
#
# The rule these pin, in order of importance:
#   1. the plate comes forward ALWAYS, whatever is loaded;
#   2. every operator the window HOLDS gets a tab, plural, off the layers' own declared identity;
#   3. a window with no results still raises, and says what is missing.


def _controls_button(win) -> QPushButton:
    found = [b for b in win.findChildren(QPushButton) if "controls" in b.text()]
    assert len(found) == 1, f"expected one controls button, found {[b.text() for b in found]}"
    return found[0]


def _selects(win, key: str, label: str = "") -> None:
    """Put *key* in this window's "Operators for this window" dropdown and select it.

    THE chip's target since 2026-08-06. Julio: *"Controls should be in the 'operators for this
    window' to show the UI controls for the operator in the dropdown and apply the newly set
    parameters."* The chip used to open a tab per operator whose LAYER was in the pane, which
    answers a different question -- that is a history of what has been RUN, while the dropdown is
    what is about to be, and the Run button beside the chip uses the dropdown.
    """
    combo = win._op_combo
    combo.clear()
    combo.addItem(label or key, key)
    combo.setCurrentIndex(0)
    combo.setEnabled(True)


def _holds(win, ops):
    """Make the window report *ops* as the processing layers it holds, the way the pane would."""
    class _Mosaic:
        def ops(self):
            return ["raw", *ops]          # raw is always there and must never be offered as one
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
    """ONE operator: the one the Run button beside it would use.

    MUTATION: point `_show_operator_controls` back at `_window_operators()` and this goes red --
    it would open decon and bgsub, the two whose layers happen to be in the pane, and NOT stitch,
    which is what the window is about to run.
    """
    win = manager.open([REGIONS[0]])
    plate = _wired(manager)
    _holds(win, ["decon", "bgsub"])          # a history of what has run: NOT the chip's question
    _selects(win, "stitch", "Stitch (register + fuse)")
    before = plate.raised

    _controls_button(win).click()

    assert plate.raised == before + 1, "the controls button did not raise the plate"
    assert plate.activated_ops == ["stitch"], (
        f"the chip opened {plate.activated_ops}, not the operator in the dropdown")


def test_it_says_what_the_operator_will_run_with(qapp, manager):
    """*"The control button should print a small text to it's side saying what the UI parameters
    are set to."* The chip's own line names the mode and the parameters, so pressing Run is not a
    guess about which values are in force."""
    win = manager.open([REGIONS[0]])
    _wired(manager)
    _selects(win, "stitch", "Stitch (register + fuse)")
    said_before = len(win._pane.said)

    _controls_button(win).click()

    said = " ".join(win._pane.said[said_before:])
    assert "Stitch" in said, f"the chip did not name the operator: {said!r}"
    assert "2D" in said or "3D" in said, f"the chip did not say which mode it would run in: {said!r}"


def test_the_chip_refuses_out_loud_when_no_operator_is_selected(qapp, manager):
    """A window whose dropdown is empty still gets the trip back to the plate, and is told why
    there is nothing to tune -- the refusal `_show_operator_controls` has always made, moved onto
    the dropdown along with the question."""
    win = manager.open([REGIONS[0]])
    plate = _wired(manager)
    win._op_combo.clear()
    before = plate.raised

    _controls_button(win).click()

    assert plate.raised == before + 1, "a window with no operator selected lost its trip back"
    assert plate.activated_ops == [], f"it opened a tab anyway: {plate.activated_ops}"


def test_the_plate_COMES_FORWARD_even_when_the_window_holds_no_operator(qapp, manager):
    """THE REGRESSION, in the words it was reported in: "it is not bringing up the plateview
    window". The raise is the half that always works, so it must never be gated on the half that
    does not. v1 returned before raising and the chip looked dead."""
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
    """No ``_holds``: the layers arrive the way a finished run delivers them, and the chip has to
    find them through the production ``_window_operators`` -> ``mosaic.ops()`` path.

    Every other test in this section replaces ``win._pane.mosaic`` with a hand-written object that
    has an ``ops()``, because the shared ``StubMosaic`` had none — so ``_window_operators``'s
    ``except Exception: return []`` swallowed an AttributeError and every one of them was really
    testing the replacement. The stub answers ``ops()`` now; this is the test that needed it.
    """
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
    """A dead click is the defect this chip was written twice to avoid. If the panel cannot open,
    the window says which operator and what went wrong -- it does not fall silent."""
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
