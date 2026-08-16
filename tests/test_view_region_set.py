"""``RegionViewer._regions`` is the CURSOR's order, not a field beside it.

WHY THIS EXISTS. The set of regions a window can reach used to be a list captured in ``__init__``
and never touched again, which was true for exactly as long as a window's scope was fixed at the
moment it opened. The plate is becoming a navigator: it can point an already-open window at a
region the window was not opened over, and that re-scopes the cursor.

A field kept in step with the cursor by hand is the "second copy" that ``_region_nav``'s design
forbids, and the places that read this set are precisely the ones that would show it wrong —
``ViewerManager.views()`` (what an operator run is scoped to) and the ``viewFocused`` payload (what
the plate paints). Reading THROUGH the cursor means there is nothing to keep in step, and that is
the property these tests pin.

The split is deliberate and both halves are asserted here:

    _seed_regions   what this window was OPENED over — historical, immutable, names the window
    _regions        where it can go NOW — the cursor's order, live

This change is behaviour-neutral on its own: nothing re-scopes a window yet, so the two agree for
every window the app currently builds. The tests drive ``_cursor.set_order`` directly, which is how
the rest of the suite already moves a cursor, so they pin the wiring before anything depends on it.
"""

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


def test_a_fresh_window_reports_what_it_was_opened_over(qapp, napari_pane_stub, squid_dataset):
    """The behaviour-neutral half. Nothing re-scopes a window yet, so the seed and the live set
    must agree for every window the app builds today — if they ever disagree at open time, the
    cursor was seeded from something other than the request."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, regions=["B2"])
    try:
        assert view._regions == ["B2"]
        assert view._seed_regions == ["B2"]
    finally:
        shutdown_plate_window(qapp, win)


def test_the_region_set_follows_a_rescope(qapp, napari_pane_stub, squid_dataset):
    """THE POINT. Growing the cursor's order grows what the window reports it can reach, with
    nothing to update by hand."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, regions=["B2"])
    try:
        view._cursor.set_order(["B2", "B3"])
        assert view._regions == ["B2", "B3"], "the region set did not follow the cursor"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_seed_does_not_follow_a_rescope(qapp, napari_pane_stub, squid_dataset):
    """The other half of the split. "What it was opened over" is history; a window that later
    wandered somewhere else was still opened over B2."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, regions=["B2"])
    try:
        view._cursor.set_order(["B2", "B3"])
        assert view._seed_regions == ["B2"], "the seed is not immutable"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_region_set_is_read_only(qapp, napari_pane_stub, squid_dataset):
    """Assigning must RAISE, not silently create the second copy this replaces.

    Same shape as ``test_nav_wiring``'s rule for ``_mosaic_region``: a writable attribute over a
    cursor is an invitation to set it and an invitation to drift."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, regions=["B2"])
    try:
        with pytest.raises(AttributeError):
            view._regions = ["B3"]
    finally:
        shutdown_plate_window(qapp, win)


def test_the_registry_view_follows_a_rescope(qapp, napari_pane_stub, squid_dataset):
    """THE PAYOFF, and the reason this is a property rather than a field plus a refresh call.

    ``ViewerManager.views()`` is what an operator run reads to decide its scope. It builds from
    ``win._regions`` and nothing tells it a window was re-scoped — so with a field it would have
    served a stale answer until someone remembered to add a refresh, and the failure would have
    been a run scoped to the wrong wells rather than an exception."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, regions=["B2"])
    mgr = win._viewer_manager
    try:
        assert mgr.views()[0].regions == ("B2",)
        view._cursor.set_order(["B2", "B3"])
        assert mgr.views()[0].regions == ("B2", "B3"), "views() served a stale region set"
    finally:
        shutdown_plate_window(qapp, win)


def test_a_rescope_does_not_rename_the_window(qapp, napari_pane_stub, squid_dataset):
    """The window's name is derived from the SEED. The ``[wid]`` title is the only visible join
    between a log line and a window on the desktop, so it must not change under the user because
    the plate moved the view somewhere."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root, regions=["B2"])
    try:
        before = view.windowTitle()
        view._cursor.set_order(["B2", "B3"])
        assert view.windowTitle() == before, "a re-scope renamed the window"
    finally:
        shutdown_plate_window(qapp, win)
