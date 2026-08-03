"""Building the root window must not put a window on the user's desktop that nobody asked for.

2026-07-29, Task 9. Julio: a stray untitled window, about 129x59, floating with no home, on both
launches of a session. This is it, and the mechanism is a one-liner with a big consequence.

    self._focus_btn = QPushButton("Focus reference plane")     # no parent
    self._focus_btn.hide()

In Qt, a widget constructed with NO PARENT and never added to a layout is a TOP-LEVEL WINDOW. The
``hide()`` on the next line is why this was thought to be harmless. It is not, because
``_sync_focus_button`` then ran on every ingest and did::

    btn.setVisible(len((self._meta or {}).get("z_levels", [])) > 1)

So the button un-hid itself, as a bare 178x30 frameless window with no title, for any acquisition
with more than one z level -- which is most of them. Measured here before the fix, on the standard
z-stack fixture: ``QPushButton "Focus reference plane", top level, visible True, 178x30``. Nothing
else in the process was visible, because a headless test never calls ``show()``.

The backlog entry guessed at this and the plan recorded ``_update_focus_button`` as having "zero
call sites". Both were half right. The button really was an orphan and the reference-plane control
really did move onto each window's own z-slider in ``d07db43``; but the sync method was NOT
uncalled, and it is precisely what made the orphan visible. A hidden orphan is untidy. An orphan
that un-hides itself is a bug report.

WHAT IS PINNED HERE

1. Constructing and ingesting into a root window makes NOTHING visible. This is the general rule
   rather than the specific button, and it is exactly as strong as it should be: the caller owns
   ``show()``. Any future widget built without a parent and then made visible fails this, which is
   the whole class of defect rather than the one instance of it.
2. The dead reference-plane chain on ``PlateWindow`` is gone, and stays gone. The button, the sync
   method, the handler and the z-slider helper it fed all had exactly one entry point between them
   and it was the orphan's ``clicked``.
3. A widget handed to ``publish_qc_result`` ends up INSIDE the shown window and VISIBLE, not
   merely inside a tab bar. The existing test for that seam
   (``test_a_decon_qc_result_opens_as_a_tab_in_pane_3``) asserted ``_explore_tabs.indexOf(view)
   >= 0`` and stayed green for six weeks while the pane holding those tabs was in no layout at
   all. Membership of a container is not reachability; ancestry up to the window, plus
   ``isVisible()`` on a really-shown window, is.

NOT pinned, deliberately: the several other parentless widgets ``PlateWindow`` keeps as hidden
orphans so that old call sites still resolve (a ``QComboBox``, the "3D native" and "Return to raw
view" buttons). They are documented as such in the source, they are never made visible, and rule 1
above is what holds them to that.

THE OTHER HALF OF THE SAME COIN (2026-08-03)

An orphan that never shows is not automatically harmless, and ``_explore_pane`` — the
``QStackedWidget`` this docstring used to list above as a benign hidden orphan — is the proof.
``publish_qc_result`` posts the deconvolution QC result into its tab bar, so for six weeks that
result was built, tabbed and shown to nobody: invisible instead of floating, which is why it never
tripped rule 1 and why nobody filed it as a stray window. Rule 3 below is the mirror of rule 1: a
widget the code hands to the USER must be REACHABLE, not merely constructed. Both failures are the
same missing question — is this thing parented into the window? — asked from opposite ends.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from qtpy.QtWidgets import QApplication  # noqa: E402

from squidmip import _viewer as V  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import qapp, stub_detail  # noqa: E402,F401  (fixtures)


def _visible_top_levels():
    """Every top-level widget Qt currently considers visible, as readable descriptions."""
    out = []
    for w in QApplication.topLevelWidgets():
        try:
            if not w.isVisible():
                continue
        except RuntimeError:                      # a wrapper whose C++ half is already gone
            continue
        label = getattr(w, "text", None)
        out.append(f"{type(w).__name__}(title={w.windowTitle()!r}, "
                   f"text={label() if callable(label) else ''!r}, "
                   f"{w.width()}x{w.height()})")
    return out


def test_building_and_ingesting_shows_no_window_the_caller_did_not_open(
        qapp, stub_detail, squid_dataset):
    """A headless test never calls show(), so anything visible here showed ITSELF.

    MUTATION: put back ``self._focus_btn = QPushButton(...)`` with no parent plus a
    ``setVisible(True)`` on a z-stack -> the button appears in this list -> red.
    """
    root, _ = squid_dataset
    before = set(_visible_top_levels())

    win = V.PlateWindow(None)
    win.ingest(str(root))

    assert len(win._meta["z_levels"]) > 1, (
        "fixture invalid: the orphan only un-hid itself on a MULTI-z acquisition, so a "
        "single-plane fixture would pass this test without testing anything")
    strays = sorted(set(_visible_top_levels()) - before)
    assert strays == [], f"the root put {len(strays)} window(s) on screen by itself: {strays}"
    shutdown_plate_window(qapp, win)


def _ancestry(widget):
    """The chain of parents from *widget* upward, as type names — for a readable failure."""
    chain, w = [], widget
    while w is not None:
        chain.append(type(w).__name__)
        w = w.parentWidget()
    return " -> ".join(chain)


def test_a_published_qc_result_is_really_inside_the_window_and_really_visible(
        qapp, stub_detail, squid_dataset):
    """Rule 3. The decon QC view is the picture the whole iterate-and-look loop exists for.

    Between 2b8fbc5 (which took pane 3 out of the layout) and this commit, `publish_qc_result`
    put it in `_explore_tabs`, `_explore_tabs` sat in `_explore_pane`, and `_explore_pane` had
    no parent and was never shown — so the tab existed and the picture did not reach a screen.
    Julio asked for the feature he had already paid for: "we should be able to toggle the turbo
    colormap mini-gui where we click on there image".

    MUTATION: drop `body.addWidget(self._explore_pane)` from `PlateWindow.__init__` -> the
    ancestry assertion fails with the view's chain ending at a top-level QStackedWidget -> red.
    """
    from squidmip._op_panels import DeconQCResultView

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.resize(900, 900)
    win.show()                       # the caller opens the window; nothing else may open itself
    win.ingest(str(root))
    qapp.processEvents()

    view = DeconQCResultView("B2/0/c0")
    win.publish_qc_result(view, "Decon QC · B2/0/c0")
    qapp.processEvents()

    # 1. ANCESTRY: the view is a descendant of the window, not of a stray top level.
    ancestors = []
    w = view
    while w is not None:
        ancestors.append(w)
        w = w.parentWidget()
    assert win in ancestors, (
        "the QC result is not inside the plate window at all — it is in a tab bar in a pane "
        f"nothing parented. Its chain is: {_ancestry(view)}")
    assert win.centralWidget() in ancestors, (
        f"the QC result hangs off the window but outside its central widget: {_ancestry(view)}")

    # 2. VISIBILITY: on a shown window, every link in that chain is shown too. `isVisible()` is
    #    False for a widget inside a hidden pane, which is exactly the state this bug was in.
    assert view.isVisible(), (
        "the QC result is parented but not on screen — pane 3 is hidden with a tab in it")
    assert view.width() > 0 and view.height() > 0, "the QC result has no geometry"

    # 3. ...and pane 3 STANDS DOWN again when its last tab goes, rather than leaving a strip of
    #    example copy where the plate used to be.
    win._close_op_tab(win._explore_tabs.indexOf(view), win._explore_tabs)
    qapp.processEvents()
    assert not win._explore_pane.isVisible(), (
        "pane 3 kept the plate's room after its last tab closed")

    shutdown_plate_window(qapp, win)


def test_a_preview_run_s_exploration_tab_does_not_drag_napari_back_under_the_plate(
        qapp, stub_detail, squid_dataset):
    """The reveal is scoped ON PURPOSE, and the scope is stated here so it cannot drift.

    `_open_preview_tab` still opens an `_ExplorationTab` in the same bar, and that tab embeds a
    second napari mosaic. Taking the embedded viewer out of the root window is exactly what the
    decentralization did, so showing the QC result must not smuggle it back in. Those tabs stay
    where 2b8fbc5 left them; making them visible again is a separate decision for someone who
    can watch it happen on a screen.
    """
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.resize(900, 900)
    win.show()
    win.ingest(str(root))

    tab = V._ExplorationTab(["B2"], "exp:test")
    win._open_op_tab("exp:test", "B2", lambda: tab, tabs=win._explore_tabs)
    qapp.processEvents()

    assert win._explore_tabs.indexOf(tab) >= 0, "the fixture did not open the tab it meant to"
    assert not win._explore_pane.isVisible(), (
        "an exploration/preview tab pulled pane 3 open — the deck shows QC results, not a "
        "second embedded viewer")
    shutdown_plate_window(qapp, win)


def test_the_dead_reference_plane_chain_is_not_on_the_plate_window(qapp, stub_detail,
                                                                  squid_dataset):
    """Focus is per-window now (each window's own z-slider, ``d07db43``). The plate's copy was
    reachable only through the orphan button, so every link in it was dead code with a live
    ``setEnabled`` habit. Named one by one so a re-introduction is a failing test, not a review
    comment."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    for gone in ("_focus_btn", "_sync_focus_button", "_focus_reference_plane",
                 "_on_focus_problem", "_on_reference_plane", "_set_z_index"):
        assert not hasattr(win, gone), (
            f"PlateWindow.{gone} is back. The reference plane lives on each window's z-slider; "
            "a second copy on the plate is what produced the orphan window.")
    shutdown_plate_window(qapp, win)


def test_the_focus_worker_itself_survives_because_the_windows_use_it(qapp, stub_detail,
                                                                    squid_dataset):
    """Deleting the chain must not take the Tenengrad worker with it: ``RegionViewer`` imports
    ``_viewer._FocusWorker`` by name, so removing it would break the control that REPLACED the
    orphan."""
    assert hasattr(V, "_FocusWorker")
    from squidmip._region_viewer import RegionViewer

    assert hasattr(RegionViewer, "_focus_reference_plane")
    assert hasattr(RegionViewer, "_on_reference_plane")
