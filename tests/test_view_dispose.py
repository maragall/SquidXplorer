"""``RegionViewer.dispose`` — the teardown, separated from the close EVENT."""

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


def _plate_with_view(qapp, root):
    """A plate with one open view over its first region, drained onto the screen."""
    win = V.PlateWindow(None)
    win.ingest(str(root))
    view = win._viewer_manager.open(list(win._order)[:1])
    assert view is not None, "the fixture acquisition did not open a view"
    _drain_until(qapp, lambda: view._pane is not None, timeout=10)
    return win, view


def test_disposing_a_view_shuts_its_napari_pane_down(qapp, napari_pane_stub, squid_dataset):
    """THE LEAK."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root)
    pane = view._pane
    try:
        assert pane.shutdowns == 0, "the pane was shut down before anything disposed it"
        view.dispose()
        assert pane.shutdowns == 1, "dispose did not shut the napari pane down — the leak is back"
    finally:
        shutdown_plate_window(qapp, win)


def test_dispose_is_idempotent(qapp, napari_pane_stub, squid_dataset):
    """A view can be disposed by its owner and THEN still receive a closeEvent."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root)
    pane = view._pane
    try:
        view.dispose()
        view.dispose()
        view.dispose()
        assert pane.shutdowns == 1, f"pane shut down {pane.shutdowns} times; dispose is not idempotent"
    finally:
        shutdown_plate_window(qapp, win)


def test_closing_a_view_still_disposes_it(qapp, napari_pane_stub, squid_dataset):
    """The extraction must not change what CLOSING does."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root)
    pane = view._pane
    try:
        view.close()
        qapp.processEvents()
        assert view._disposed, "closing a view no longer disposes it"
        assert pane.shutdowns == 1, "closing a view no longer shuts its napari pane down"
    finally:
        shutdown_plate_window(qapp, win)


def test_closing_a_view_still_deregisters_it(qapp, napari_pane_stub, squid_dataset):
    """The OTHER job closeEvent was doing."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root)
    mgr = win._viewer_manager
    wid = view.window_id
    try:
        assert wid in [w.window_id for w in mgr.windows]
        view.close()
        qapp.processEvents()
        assert wid not in [w.window_id for w in mgr.windows], "the registry still holds a closed view"
    finally:
        shutdown_plate_window(qapp, win)


def test_a_pane_that_fails_to_shut_down_does_not_strand_the_registry(
        qapp, napari_pane_stub, squid_dataset):
    """Every teardown step is wrapped separately so one failure cannot strand the rest."""
    root, _ = squid_dataset
    win, view = _plate_with_view(qapp, root)
    mgr = win._viewer_manager
    wid = view.window_id

    def _boom():
        raise RuntimeError("napari refused to close")

    view._pane.shutdown = _boom
    try:
        view.dispose()
        assert view._disposed
        assert wid not in [w.window_id for w in mgr.windows], (
            "a pane that failed to shut down took the deregistration down with it")
    finally:
        shutdown_plate_window(qapp, win)
