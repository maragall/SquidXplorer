"""Per-window font scaling, the startup splash, and shared GL contexts."""
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


def test_any_top_level_window_scales_its_own_type_independently(qapp):
    """A RegionViewer is a separate top-level, so it never appeared in PlateWindow.findChildren(QWidget) and its type stayed put while the root's grew."""
    small, large = QWidget(), QWidget()
    small.resize(1100, 800)
    large.resize(2200, 800)
    a, b = QLabel("a", small), QLabel("b", large)
    a.setStyleSheet("QLabel{font-size:10px;}")
    b.setStyleSheet("QLabel{font-size:10px;}")

    rescale_fonts(small)
    assert rescale_fonts(large) is True
    assert "font-size:10px" in a.styleSheet(), "resizing one window changed another's type"
    assert "font-size:20px" in b.styleSheet(), b.styleSheet()


def test_the_child_window_class_actually_rescales_on_resize():
    """MUTATION: delete RegionViewer.resizeEvent -> red."""
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
    """napari's import happens inside the PlateWindow constructor, which is the several-second silence reported; anything hung off the window itself is too"""
    import inspect

    from squidxplorer import _viewer

    code = [ln.split("#")[0] for ln in inspect.getsource(_viewer.main).splitlines()]
    splash_at = next(i for i, ln in enumerate(code) if "_startup_splash(" in ln)
    build_at = next(i for i, ln in enumerate(code) if "PlateWindow(" in ln)
    assert splash_at < build_at, (
        "the splash is created AFTER the slow constructor, so it can never be seen during it"
    )


def test_windows_share_gl_contexts():
    """N windows means N napari canvases; without sharing, the second can render black."""
    import inspect

    from squidxplorer import _viewer

    src = inspect.getsource(_viewer.enable_hidpi)
    assert "AA_ShareOpenGLContexts" in src, (
        "enable_hidpi does not share GL contexts, so a second napari canvas can render black"
    )
