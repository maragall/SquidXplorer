"""Four fixes: "close view" ignored multi-select, arrow keys did nothing in the navigator
(the tree never had keyboard focus), child windows did not rescale their font type on resize,
and startup was silent so a slow launch and a crash looked identical.

Scope deliberately narrow: tests/test_root_resize.py already owns the stylesheet arithmetic and
the root window's own behaviour. No napari, no dataset, no PlateWindow: the heavier GUI files
abort the interpreter on import through their fixture chain, taking pytest's summary with it.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

from squidxplorer._fontscale import rescale_fonts  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_any_window_can_scale_its_own_type(qapp):
    """A RegionViewer is a separate top-level window, so it never appeared in
    PlateWindow.findChildren(QWidget) and its type stayed put while the root's grew."""
    win = QWidget()                                  # not a child of anything
    win.resize(2200, 900)                            # 2x the design width
    label = QLabel("hello", win)
    label.setStyleSheet("QLabel{font-size:10px;}")

    assert rescale_fonts(win) is True
    assert "font-size:20px" in label.styleSheet(), label.styleSheet()


def test_two_windows_scale_independently(qapp):
    """Decentralised windows must not be forced to agree about type size."""
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
    """MUTATION: delete RegionViewer.resizeEvent -> red. Without this the helper is correct and
    nothing ever calls it."""
    import inspect

    from squidxplorer._region_viewer import RegionViewer

    assert hasattr(RegionViewer, "resizeEvent"), "the child window never rescales"
    assert "rescale_fonts" in inspect.getsource(RegionViewer.resizeEvent)


def test_the_splash_is_skipped_when_nobody_can_see_it(qapp):
    """Headless and under tests it returns None, rather than a dummy that must be humoured."""
    from squidxplorer._viewer import _startup_splash

    qapp.setProperty("_squidxplorer_test", True)
    try:
        assert _startup_splash(qapp) is None
    finally:
        qapp.setProperty("_squidxplorer_test", False)


def test_the_splash_never_stops_the_app_opening():
    """MUTATION: drop the try/except in _startup_splash -> red."""
    from squidxplorer._viewer import _startup_splash

    class Hostile:
        def property(self, _name):
            raise RuntimeError("boom")

    assert _startup_splash(Hostile()) is None


def test_the_splash_is_shown_before_the_slow_constructor():
    """napari's import happens inside the PlateWindow constructor, which is the several-second
    silence reported; anything hung off the window itself is too late by definition."""
    import inspect

    from squidxplorer import _viewer

    # code lines only, so a comment mentioning PlateWindow doesn't fool the search
    code = [ln.split("#")[0] for ln in inspect.getsource(_viewer.main).splitlines()]
    splash_at = next(i for i, ln in enumerate(code) if "_startup_splash(" in ln)
    build_at = next(i for i, ln in enumerate(code) if "PlateWindow(" in ln)
    assert splash_at < build_at, (
        "the splash is created AFTER the slow constructor, so it can never be seen during it"
    )


def test_windows_share_gl_contexts():
    """N windows means N napari canvases; without sharing, the second can render black.

    Asserted on the source, since offscreen Qt cannot reproduce the rendering artefact — this
    proves only that the attribute is requested, not that multi-canvas rendering works.
    """
    import inspect

    from squidxplorer import _viewer

    src = inspect.getsource(_viewer.enable_hidpi)
    assert "AA_ShareOpenGLContexts" in src, (
        "enable_hidpi does not share GL contexts, so a second napari canvas can render black"
    )


# The Window navigator (OpenViewList) was DELETED on 2026-08-19: the ViewDeck's tabs superseded
# its list, View > Close All Views carries close-all, and StatusRow carries the memory/run bars.
# tests/test_view_deck.py pins the absence and the surviving jobs; the keyboard-focus and
# raise-on-show tests that lived here died with the widget they described.


def test_the_window_navigator_stays_deleted():
    import squidxplorer._region_viewer as RV

    assert not hasattr(RV, "OpenViewList"), (
        "the Window navigator widget is back; the deck's tabs are its replacement (2026-08-19)")
