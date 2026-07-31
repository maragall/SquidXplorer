"""The four Spencer NEXT_STEPS items landed on 2026-07-30, each pinned by what it fixes.

* "Close view" closed ONE view while the tree was multi-select.
* Arrow keys did nothing in the Window navigator, because the tree never had keyboard focus.
* Child windows did not rescale their type on resize; only the root did.
* Startup was silent, so a slow launch and a crash looked identical.

**Scope, deliberately narrow.** `tests/test_root_resize.py` already owns the stylesheet arithmetic
and the ROOT window's behaviour (clamping, non-compounding, font-size-only rewriting). This file
covers only what moving that code into `_fontscale` made newly possible, plus the three items that
had no coverage at all. Re-asserting the arithmetic here would give two files an opinion about one
rule, which is how they drift.

No napari, no dataset, no `PlateWindow`: the heavier GUI files abort the interpreter on import
through their fixture chain, and an abort takes pytest's summary with it, which is exactly how
failures hid in this repo for weeks.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

from squidmip._fontscale import rescale_fonts  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ------------------------------------------------- font scaling reaches a NON-root window


def test_any_window_can_scale_its_own_type(qapp):
    """The point of moving this out of `PlateWindow`: a window that is not the root can call it.

    A `RegionViewer` is a separate TOP-LEVEL window, so it never appeared in
    `PlateWindow.findChildren(QWidget)` and its type stayed put while the root's grew. This is
    that fix at its smallest: a plain QWidget, no relation to the root, scaling correctly.
    """
    win = QWidget()                                  # NOT a child of anything
    win.resize(2200, 900)                            # 2x the design width
    label = QLabel("hello", win)
    label.setStyleSheet("QLabel{font-size:10px;}")

    assert rescale_fonts(win) is True
    assert "font-size:20px" in label.styleSheet(), label.styleSheet()


def test_two_windows_scale_independently(qapp):
    """Decentralised windows must not be forced to agree about type size.

    Dragging one view wide must not enlarge the type in another. A module-level scale, or state
    hung off the root, would couple them; the state lives on each window for this reason.
    """
    small, large = QWidget(), QWidget()
    small.resize(1100, 800)
    large.resize(2200, 800)
    a, b = QLabel("a", small), QLabel("b", large)
    a.setStyleSheet("QLabel{font-size:10px;}")
    b.setStyleSheet("QLabel{font-size:10px;}")

    rescale_fonts(small)
    rescale_fonts(large)
    assert "font-size:10px" in a.styleSheet(), "resizing one window changed another's type"
    assert "font-size:20px" in b.styleSheet()


def test_the_child_window_class_actually_rescales_on_resize():
    """The wiring, not just the helper: RegionViewer must call it from resizeEvent.

    MUTATION: delete `RegionViewer.resizeEvent` -> red. Without this the helper exists and is
    correct and nothing ever calls it, which is the state the whole item was reported in.
    """
    import inspect

    from squidmip._region_viewer import RegionViewer

    assert hasattr(RegionViewer, "resizeEvent"), "the child window never rescales"
    assert "rescale_fonts" in inspect.getsource(RegionViewer.resizeEvent)


# ------------------------------------------------- the startup splash


def test_the_splash_is_skipped_when_nobody_can_see_it(qapp):
    """Headless and under tests it returns None, rather than a dummy that must be humoured."""
    from squidmip._viewer import _startup_splash

    qapp.setProperty("_squidmip_test", True)
    try:
        assert _startup_splash(qapp) is None
    finally:
        qapp.setProperty("_squidmip_test", False)


def test_the_splash_never_stops_the_app_opening():
    """A splash is cosmetic. If it raises, the app must still start.

    MUTATION: drop the try/except in `_startup_splash` -> red. A decoration that can prevent
    launch is worse than no decoration.
    """
    from squidmip._viewer import _startup_splash

    class Hostile:
        def property(self, _name):
            raise RuntimeError("boom")

    assert _startup_splash(Hostile()) is None


def test_the_splash_is_shown_before_the_slow_constructor():
    """Ordering IS the feature. Built after `PlateWindow(...)`, it appears once the wait is over.

    napari's import happens inside the PlateWindow constructor, which is the several-second
    silence Spencer reported, so anything hung off the window itself is too late by definition.
    """
    import inspect

    from squidmip import _viewer

    # CODE lines only: the comment above the call names PlateWindow, so a raw string index
    # finds the prose before the statement and the assertion passes or fails for the wrong reason.
    code = [ln.split("#")[0] for ln in inspect.getsource(_viewer.main).splitlines()]
    splash_at = next(i for i, ln in enumerate(code) if "_startup_splash(" in ln)
    build_at = next(i for i, ln in enumerate(code) if "PlateWindow(" in ln)
    assert splash_at < build_at, (
        "the splash is created AFTER the slow constructor, so it can never be seen during it"
    )


# ------------------------------------------------- shared GL contexts (Qt6 step 3)


def test_windows_share_gl_contexts():
    """N windows means N napari canvases; without sharing, the second can render black.

    Asserted on the source rather than by opening two canvases, because the failure is a
    rendering artefact that offscreen Qt cannot reproduce. Stated plainly so nobody reads this as
    proof that multi-canvas rendering works: it proves only that the attribute is requested.
    """
    import inspect

    from squidmip import _viewer

    src = inspect.getsource(_viewer.enable_hidpi)
    assert "AA_ShareOpenGLContexts" in src, (
        "enable_hidpi does not share GL contexts, so a second napari canvas can render black"
    )


# ------------------------------------------------- the Window navigator


# "Close selected views" is covered by tests/test_nav_close_selected.py (Spencer, PR #10), which
# supersedes the two source-inspection tests that were here. His CLICKS the button through a real
# ViewerManager; mine only read the source with inspect.getsource, and a source check stays green
# even if the button is wired to nothing at all. That is the exact failure the walkthrough records
# ("a Re-dock button was broken from the day it shipped and no test noticed, because every test
# called the handler directly instead of clicking"), so the weaker pair is deleted rather than kept
# alongside: two files with an opinion about one rule is how they drift.


def test_the_navigator_tree_can_take_keyboard_focus():
    """Arrow keys were never the missing piece: FOCUS was.

    A QTreeWidget moves its current row on up/down for free, and selection already raises the
    window. With no focus policy the arrows went wherever focus happened to be, so a complete
    feature looked absent.
    """
    import inspect

    from squidmip._region_viewer import OpenViewList

    src = inspect.getsource(OpenViewList.__init__)
    assert "setFocusPolicy" in src, "the navigator tree cannot take keyboard focus"
    assert hasattr(OpenViewList, "showEvent"), "nothing ever hands the tree focus"


def test_opening_the_navigator_does_not_reorder_the_users_windows():
    """`setCurrentItem` selects, and selecting raises. Guard it, or merely showing the panel
    reshuffles every open window in front of the user."""
    import inspect

    from squidmip._region_viewer import OpenViewList

    src = inspect.getsource(OpenViewList.showEvent)
    assert "_syncing" in src, "showEvent selects without the guard, so it raises windows"
