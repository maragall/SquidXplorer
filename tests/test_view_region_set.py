"""``RegionViewer._regions`` is the CURSOR's order, not a field beside it."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # headless Qt; must precede any Qt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt GUI tests.",
        allow_module_level=True,
    )

from squidxplorer import _viewer as V  # noqa: E402
from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)


def _plate_with_view(qapp, root, regions=None):
    win = V.PlateWindow(None)
    win.ingest(str(root))
    want = list(regions) if regions is not None else list(win._order)[:1]
    view = win._viewer_manager.open(want)
    assert view is not None
    _drain_until(qapp, lambda: view._pane is not None, timeout=10)
    return win, view


def test_the_region_set_follows_a_rescope_and_the_seed_the_registry_and_the_title_do_not_drift(
        qapp, napari_pane_stub, squid_dataset):
    """THE POINT: `_regions` reads through the cursor; the seed, and the title derived from it, stay put."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, regions=["B2"])
    mgr = win._viewer_manager
    try:
        assert view._regions == ["B2"] and view._seed_regions == ["B2"]
        assert mgr.views()[0].regions == ("B2",)
        title = view.windowTitle()
        view._cursor.set_order(["B2", "B3"])
        assert view._regions == ["B2", "B3"], "the region set did not follow the cursor"
        assert view._seed_regions == ["B2"], "the seed is not immutable"
        assert mgr.views()[0].regions == ("B2", "B3"), "views() served a stale region set"
        assert view.windowTitle() == title, "a re-scope renamed the window"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_region_set_is_read_only(qapp, napari_pane_stub, squid_dataset):
    """Assigning must RAISE, not silently create the second copy this replaces."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, regions=["B2"])
    try:
        with pytest.raises(AttributeError):
            view._regions = ["B3"]
    finally:
        shutdown_plate_window(qapp, win)
