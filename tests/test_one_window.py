"""ONE WINDOW: the plate view and the log render as slots in the view window's left column."""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication

import squidxplorer._viewer as V
from tests.conftest import shutdown_plate_window

pytestmark = pytest.mark.usefixtures("qapp")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _tabbed(qapp, root, n_views=1):
    win = V.PlateWindow(None)
    win.ingest(str(root))
    mgr = win._viewer_manager
    mgr.tabbed_views = True
    order = list(win._order)
    views = [mgr.open([order[i % len(order)]]) for i in range(n_views)]
    for _ in range(10):
        qapp.processEvents()
    return win, mgr, mgr.deck(create=False), views


def _is_inside(widget, ancestor) -> bool:
    p = widget
    while p is not None:
        if p is ancestor:
            return True
        p = p.parentWidget()
    return False


def test_the_plate_view_and_log_are_hosted_in_the_view_window(qapp, napari_pane_stub,
                                                              squid_dataset):
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed(qapp, root)
    try:
        v = views[0]
        assert getattr(v, "_hosts_plate_slots", False), (
            "opening a tabbed view did not host the plate slots")
        assert _is_inside(win._overview, v), "the plate view is not inside the view window"
        assert _is_inside(win._log_panel, v), "the log is not inside the view window"
    finally:
        shutdown_plate_window(qapp, win)


def test_closing_every_view_returns_the_plate_and_log_home(qapp, napari_pane_stub,
                                                           squid_dataset):
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed(qapp, root)
    try:
        mgr.close_all()
        for _ in range(10):
            qapp.processEvents()
        assert not getattr(win, "_plate_hosted", True), "the plate still thinks it is hosted"
        assert _is_inside(win._overview, win), "the plate view did not come home"
        assert _is_inside(win._log_panel, win), "the log did not come home"
    finally:
        shutdown_plate_window(qapp, win)


def test_a_reingest_lands_the_new_overview_in_the_live_slot(qapp, napari_pane_stub,
                                                            squid_dataset):
    """Ingest REBUILDS the overview; while hosted, the new one must land in the slot, not in the hidden plate window."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed(qapp, root)
    try:
        v = views[0]
        win.ingest(str(root))
        for _ in range(5):
            qapp.processEvents()
        assert _is_inside(win._overview, win._plate_slot_box), (
            "the re-ingested overview did not land in the plate slot")
        assert _is_inside(win._overview, v)
    finally:
        shutdown_plate_window(qapp, win)


def test_the_default_layout_hides_the_plate_window_once_hosted(qapp, napari_pane_stub,
                                                               squid_dataset):
    """The plate window DIES as a window in the working layout: hosted, it hides; with the views gone it shows again (the app must keep a surface)."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.default_layout = True
    win._layout_applied = True                    # geometry policy is not under test
    win.show()
    win.ingest(str(root))
    mgr = win._viewer_manager
    mgr.tabbed_views = True
    views = [mgr.open([list(win._order)[0]])]
    for _ in range(10):
        qapp.processEvents()
    try:
        assert win.isHidden(), "the working layout still shows the plate window"
        mgr.close_all()
        for _ in range(10):
            qapp.processEvents()
        assert not win.isHidden(), "with no views left the plate window must come back"
    finally:
        shutdown_plate_window(qapp, win)


def test_the_deck_accepts_a_dropped_acquisition_for_the_plate(qapp, napari_pane_stub,
                                                              squid_dataset, tmp_path):
    """One window means the DECK is the drop target too: a dropped folder reaches the plate's own ingest."""
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed(qapp, root)
    try:
        got = []
        win.ingest = lambda p: got.append(str(p))
        from qtpy.QtCore import QMimeData, QUrl
        from qtpy.QtGui import QDropEvent
        from qtpy.QtCore import QPointF, Qt

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(root))])
        event = QDropEvent(QPointF(10, 10), Qt.CopyAction, mime, Qt.LeftButton,
                           Qt.NoModifier)
        deck.dropEvent(event)
        assert got == [str(root)], "the drop never reached the plate's ingest"
    finally:
        shutdown_plate_window(qapp, win)
