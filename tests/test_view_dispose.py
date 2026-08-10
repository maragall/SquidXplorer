"""``RegionViewer.dispose`` — the teardown, separated from the close EVENT.

WHAT THIS IS FOR. ``closeEvent`` was doing two different jobs: joining this window's threads, and
telling the registry the window is gone. Only a top-level window ever receives a close event, so
both jobs were reachable by exactly one route, and the napari half of the first job was not being
done at all: ``MosaicPane.shutdown`` had ZERO callers anywhere in ``squidmip/``, ``tests/`` or
``tools/``, while its own docstring said "every owner of a pane calls this before deleteLater()".

The consequence is in that docstring too: ``deleteLater()`` on the Qt wrapper does not close the
napari Viewer, because napari keeps every Viewer in its own instance registry — so a GL context and
tens of MB leaked per window CLOSED, which "killed a session after twenty of them".

These tests are therefore mostly about a CALL HAPPENING, not about a return value. That is a weak
shape for a test and it is deliberate here: the defect was silence. Note that ``dispose`` wraps
every teardown step in ``except Exception`` — on purpose, because a step that throws must not
strand the joins after it — which means a missing method would be swallowed and read as success.
That is exactly why ``StubPane.shutdown`` COUNTS rather than no-ops (``conftest``).
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

from squidmip import _viewer as V  # noqa: E402
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
    """THE LEAK. The pane's shutdown is the only thing that closes the napari Viewer and drops it
    from napari's instance registry; nothing called it."""
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
    """A view can be disposed by its owner and THEN still receive a closeEvent. Joining a QThread
    twice or closing a napari Viewer twice is the shape that aborts the interpreter rather than
    raising, so the second call must do nothing at all."""
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
    """The extraction must not change what CLOSING does. This is the regression guard for the
    refactor itself: closeEvent delegates, so every existing caller keeps its behaviour."""
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
    """The OTHER job closeEvent was doing. ``closed`` is what ``ViewerManager`` listens on to drop
    the window from its registry, and it now fires from ``dispose`` — so it must still reach the
    manager by the ordinary close route."""
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
    """Every teardown step is wrapped separately so one failure cannot strand the rest. The napari
    shutdown runs LAST and immediately before ``closed`` is emitted, which makes it the step most
    able to lose the deregistration if it were allowed to propagate."""
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
