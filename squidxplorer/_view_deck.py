"""ViewDeck — many open views as TABS in one window, one of them current.

Spencer, 2026-08-10: "I'd like the ability to 'tab' Napari child windows. So that I can have many
selections open (as tabs) but only work with one tab at a time."

WHY A NEW TOP-LEVEL AND NOT PART OF THE ROOT. The root's band is already three panels sharing about
300 px of width, and `tests/test_main_window_layout` pins that shape in real pixels. The working
layout also wants the big window to be the VIEWS and the small one to be the plate, so putting the
deck inside the root would have it the wrong way round.

WHAT MAKES THIS POSSIBLE AT ALL. A `RegionViewer` already contains a napari `QMainWindow` that was
reparented into it and had its flags flipped to `Qt.Widget` (`_napari_pane._embed_native_window`).
So the trick is proven in this codebase; the deck applies the SAME three calls one level further
out. Crucially the napari canvas is never touched — `_napari_pane` states outright that
`setParent()` on it guts the window — because the whole `RegionViewer` assembly moves as one
object.

VERIFIED BEFORE THIS WAS WRITTEN, on the real machine, because the suite runs with no OpenGL and
therefore cannot check any of it: four napari canvases render inside one QTabWidget, exactly one
page is visible at a time, this container's minimumSizeHint is 925x725 against a 1920x1032 work
area, and a page survives 20 dock/undock cycles with the same pane and the same napari Viewer.

THE COST, measured at the same time: about 88 MB per live view. Tabs make it cheap to open twelve
because everything looks like one window, and twelve is about a gigabyte. That is what
`_SOFT_TAB_CAP` and the status line below are for — the expense is invisible in a tab strip in a
way it is not on a desktop full of windows, so the deck says it out loud.
"""

from __future__ import annotations

from typing import List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QMainWindow, QStatusBar

from squidxplorer._qt_tabs import _DetachTabs

#: Views open in one deck before it starts naming the cost. Not a limit — nothing is refused. At
#: ~88 MB a view (measured) this is where a deck is holding about half a gigabyte, which is the
#: point at which a user who cannot see the cost should be told it.
_SOFT_TAB_CAP = 6

#: Measured on the workstation, 2026-08-10: RSS with four live views minus the plate alone,
#: divided by four. Used only to phrase the status line, never to decide anything.
_MB_PER_VIEW = 88


class ViewDeck(QMainWindow):
    """A window holding many :class:`~squidxplorer._region_viewer.RegionViewer` pages as tabs.

    Owns no view logic. It knows how to hold pages, which one is current, and how to give one back
    as a top-level window; everything about what a view IS stays in ``RegionViewer``.
    """

    #: A page became current (int window_id). The manager listens and records the focus; the deck
    #: deliberately does not reach into the registry itself.
    pageActivated = Signal(int)

    def __init__(self, parent=None, index: int = 1):
        super().__init__(parent)
        #: 1-based, and part of how a view says WHERE it lives. One deck is just "views"; a second
        #: one has to be tellable from the first, which is what the navigator's location column
        #: reads. Phase 2 (dragging tabs between decks) is the case this exists for.
        self.deck_index = int(index)
        self.setWindowTitle("SquidXplorer views" if self.deck_index == 1
                            else f"SquidXplorer views {self.deck_index}")
        #: True from the first moment of teardown. `_DetachTabBar` fires its detach through
        #: ``QTimer.singleShot(0, lambda: ...)`` — a self-capturing lambda on a process-global
        #: timer — so a press that lands just before this window closes would otherwise call into
        #: a deleted deck. That is the exact shape of `test_window_lifetime`'s BUG 3.
        self._closing = False
        #: Programmatic tab changes must not be mistaken for the user picking a tab. Same guard,
        #: same reason, as ``PlateWindow._tabs_muted`` one module away.
        self._muted = False
        self._tabs = _DetachTabs(self._detach_page, first_detachable=0)
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(True)
        # A CLOSE BUTTON ON EVERY TAB. Spencer, on first use: "we could use a more obvious close
        # tab button." Closing a view was reachable only from the navigator's "Close selected
        # views" or the deck's own title bar — one of which is in the other window and the other of
        # which closes everything. The affordance belongs where the thing being closed is.
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self._tabs.currentChanged.connect(self._on_current_changed)
        self.setCentralWidget(self._tabs)
        self.setStatusBar(QStatusBar())
        #: The plate window this deck fronts for while it is the ONE window, or None.
        self._plate = None
        self._refresh_status()

    # -- the app surface while the plate window hides (one window, 2026-08-25) -------------
    def bind_plate(self, plate) -> None:
        """Make this deck an app surface for *plate*: drop-to-open and the essential menu
        actions forward to the plate's own methods (the books stay there)."""
        if plate is None or self._plate is plate:
            return
        self._plate = plate
        self.setAcceptDrops(True)
        bar = self.menuBar()
        # The ONE window's menu (Julio, 2026-08-25, ruling x): File (Open, Quit) and View,
        # nothing else; napari's own menu bar is hidden chrome.
        file_menu = bar.addMenu("&File")
        act_open = file_menu.addAction("&Open Acquisition…")
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(lambda *_: self._plate_call("_open_acquisition_dialog"))
        act_quit = file_menu.addAction("&Quit")
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(lambda *_: self._plate_call("close"))
        view_menu = bar.addMenu("&View")
        act_all = view_menu.addAction("Select &All Wells")
        act_all.triggered.connect(lambda *_: self._plate_call("_select_all_wells"))
        act_close = view_menu.addAction("Close All Vie&ws")
        act_close.triggered.connect(lambda *_: self._plate_call("_close_all_views"))
        act_plate = view_menu.addAction("&Plate Window")
        act_plate.setToolTip("Bring the (hidden) plate window back on screen.")
        act_plate.triggered.connect(lambda *_: self._show_plate())

    def _plate_call(self, name: str) -> None:
        plate = self._plate
        fn = getattr(plate, name, None) if plate is not None else None
        if callable(fn):
            try:
                fn()
            except RuntimeError:
                self._plate = None               # the plate's C++ half is gone

    def _show_plate(self) -> None:
        plate = self._plate
        if plate is None:
            return
        try:
            plate._hidden_for_one_window = False
            plate.show()
            plate.raise_()
        except RuntimeError:
            self._plate = None

    def dragEnterEvent(self, event):             # noqa: N802 - Qt naming
        if self._plate is not None and event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event):                  # noqa: N802 - Qt naming
        plate = self._plate
        urls = event.mimeData().urls() if event.mimeData() is not None else []
        if plate is not None and urls:
            path = urls[0].toLocalFile()
            if path:
                event.acceptProposedAction()
                try:
                    plate.ingest(path)
                except RuntimeError:
                    self._plate = None
                return
        super().dropEvent(event)

    # -- what it is holding ---------------------------------------------------------------
    def count(self) -> int:
        return self._tabs.count()

    def pages(self) -> "List":
        return [self._tabs.widget(i) for i in range(self._tabs.count())]

    def current_page(self):
        return self._tabs.currentWidget()

    def index_of(self, page) -> int:
        return self._tabs.indexOf(page)

    # -- holding and letting go -----------------------------------------------------------
    def dock_page(self, page, index: Optional[int] = None) -> int:
        """Take *page* as a tab. The SAME object — it keeps its napari viewer, its layers, its
        in-flight worker and its window id.

        The three calls are `_embed_native_window`'s, in its order: reparent, flip the flags to
        ``Qt.Widget`` so Qt stops treating it as a top-level, then hand it to the tab widget.
        """
        page.setParent(self._tabs)
        page.setWindowFlags(Qt.Widget)
        page._host = self
        self._muted = True
        try:
            at = (self._tabs.addTab(page, page.windowTitle()) if index is None
                  else self._tabs.insertTab(int(index), page, page.windowTitle()))
        finally:
            self._muted = False
        self._tabs.setCurrentIndex(at)
        self._refresh_status()
        return at

    def undock_page(self, page) -> None:
        """Give *page* back as a free-standing window, still alive and still registered.

        ``removeTab`` does NOT reparent — the page stays a child of the tab widget's internal
        ``QStackedWidget`` — so the explicit ``setParent(None)`` is required, not defensive.
        ``_FloatWindow`` already knows this; the same discipline applies here.
        """
        i = self._tabs.indexOf(page)
        if i >= 0:
            self._muted = True
            try:
                self._tabs.removeTab(i)
            finally:
                self._muted = False
        page._host = None
        page.setParent(None)
        page.setWindowFlags(Qt.Window)
        page.resize(*page._default_view_size())
        page.show()
        page.raise_()
        page.activateWindow()
        self._refresh_status()

    def close_page(self, page) -> None:
        """Close one view: untab it, then dispose it. `dispose` is what joins its threads, shuts
        its napari pane down and tells the registry it is gone — and a tab removal fires no close
        event, so nothing else would."""
        i = self._tabs.indexOf(page)
        if i >= 0:
            self._muted = True
            try:
                self._tabs.removeTab(i)
            finally:
                self._muted = False
        page._host = None
        page.setParent(None)        # removeTab does NOT reparent; see undock_page
        page.dispose()
        page.deleteLater()
        self._refresh_status()
        self._close_if_empty()

    def set_current(self, page) -> bool:
        i = self._tabs.indexOf(page)
        if i < 0:
            return False
        self._tabs.setCurrentIndex(i)
        return True

    def refresh_titles(self) -> None:
        """Refresh every tab's text from its page's current title (a title change emits
        ``windowsChanged``; without this the tab keeps the old text and the ``[id] name`` join
        between a log line and a view goes stale on this surface only."""
        for i in range(self._tabs.count()):
            page = self._tabs.widget(i)
            if page is not None:
                self._tabs.setTabText(i, page.windowTitle())

    # -- signals in ------------------------------------------------------------------------
    def _on_current_changed(self, index: int) -> None:
        """A different page is on screen: exactly one view draws, and the app learns which.

        ONLY the current page of an ACTIVE deck is the active view, so this both stops the outgoing
        page and starts the incoming one. That is the "windows not being manipulated halt their
        draw" rule landing for free: `QStackedWidget` hides the others, so their canvases stop
        painting as well, which a desktop full of top-level views never did.
        """
        if self._muted or self._closing:
            return
        current = self._tabs.widget(index) if index >= 0 else None
        for page in self.pages():
            try:
                page.set_active(page is current and self.isActiveWindow())
            except Exception:                    # noqa: BLE001 - a page mid-teardown is not news
                pass
        if current is not None:
            self.pageActivated.emit(int(current.window_id))

    def changeEvent(self, event):                # noqa: N802 - Qt naming
        from qtpy.QtCore import QEvent

        if event.type() == QEvent.ActivationChange and not self._closing:
            active = self.isActiveWindow()
            page = self.current_page()
            if page is not None:
                try:
                    page.set_active(active)
                except Exception:                # noqa: BLE001
                    pass
                if active:
                    self.pageActivated.emit(int(page.window_id))
        super().changeEvent(event)

    def short_name(self) -> str:
        """How a view names this deck when asked where it lives. Short on purpose: it sits in a
        column beside a window title that is already long."""
        return "views" if self.deck_index == 1 else f"views {self.deck_index}"

    def _on_tab_close_requested(self, index: int) -> None:
        """The tab's own close button. Goes through `close_page`, which is the same path the
        navigator and the plate use — so a view closed here is disposed and deregistered exactly
        as one closed anywhere else, rather than merely removed from a tab bar."""
        if self._closing or not (0 <= index < self._tabs.count()):
            return
        page = self._tabs.widget(index)
        if page is not None:
            self.close_page(page)

    def _detach_page(self, index: int, tabs) -> None:
        """The drag-out gesture. Guarded first, because the bar defers this through a
        self-capturing lambda on a global timer and the deck may be gone by the time it fires."""
        if self._closing or not (0 <= index < self._tabs.count()):
            return
        page = self._tabs.widget(index)
        if page is not None:
            self.undock_page(page)

    # -- the cost, said out loud ------------------------------------------------------------
    def _refresh_status(self) -> None:
        n = self._tabs.count()
        bar = self.statusBar()
        if bar is None:
            return
        if n == 0:
            bar.clearMessage()
            return
        text = f"{n} view{'s' if n != 1 else ''} open"
        if n > _SOFT_TAB_CAP:
            # NAMED, NOT REFUSED. Nothing here stops the user; a cap that blocked work would be
            # worse than the memory. What the tab strip hides is that each of these holds its own
            # napari viewer and GL context, so the number is what the strip cannot show.
            text += (f" · about {n * _MB_PER_VIEW} MB - each view holds its own napari viewer; "
                     f"close the ones you are done with")
        bar.showMessage(text)

    # -- teardown ----------------------------------------------------------------------------
    def _close_if_empty(self) -> None:
        """A deck with no views is furniture. It is also a TOP-LEVEL, and a top-level that outlives
        the plate is how the process survives with nothing on screen still holding the
        single-instance lock — the bug `PlateWindow.closeEvent` documents.

        HIDDEN rather than destroyed, and reused if another view opens: Qt's quit-on-last-window
        rule counts VISIBLE windows, so hiding settles the lifetime question, and tearing down a
        container to rebuild it is the churn this codebase's segfault history is made of. Destroying
        it here was tried against the tabbed-teardown abort and changed nothing, so the cheaper
        behaviour stands.
        """
        if not self._closing and self._tabs.count() == 0:
            self.close()

    def closeEvent(self, event):                 # noqa: N802 - Qt naming
        """Closing the deck closes every view in it, through the same dispose a top-level uses."""
        self._closing = True
        for page in self.pages():
            i = self._tabs.indexOf(page)
            if i >= 0:
                self._tabs.removeTab(i)
            page._host = None
            page.setParent(None)
            try:
                page.dispose()
            except Exception:                    # noqa: BLE001 - one bad page must not strand the rest
                pass
            page.deleteLater()
        super().closeEvent(event)
