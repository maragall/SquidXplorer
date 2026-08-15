"""The tab / dock / float MECHANISM, out of ``PlateWindow`` (2026-08-14).

``_qt_tabs`` stays the widget layer (the drag gesture, the bar, the float window) and keeps no
policy, exactly as its docstring promises. This module is the layer above: the two registries
(``op_tabs``: key -> docked widget, ``floating``: key -> ``_FloatWindow``) and the one float
mechanism every floatable widget uses.

The generalisation that earns the module: ``_float_log`` / ``_redock_log`` used to be a
near-duplicate specialisation of ``_detach_tab`` / ``_redock`` for one widget, because the log's
HOME is not a tab bar (it re-docks into the right column's splitter) and its CLOSE policy is not
disposal (a console is a live sink on the process-wide root logger; deleting it loses it for
good). Both differences are now parameters of :meth:`float_out` — ``restore`` says where the
widget goes home, ``dispose_on_close`` says what closing the window means — so the log case is
one call through the general mechanism and the specialised copy is deleted.

``restore`` is a CLOSURE over the exact home object, which also retires the old
``_home_tabs or _left_tabs`` hazard by construction: an empty ``QTabWidget`` is falsy in PyQt,
so any boolean test on the home was a re-dock sent to the wrong bar. There is no boolean test
left to get wrong.

``PlateWindow`` keeps thin delegates (tests actuate ``_detach_tab`` / ``_redock`` /
``_float_log`` / ``_redock_log`` by name) and aliases ``_op_tabs`` / ``_floating`` to the SAME
dict objects owned here, because tests and ``closeEvent`` index them directly.
"""

from __future__ import annotations

from typing import Callable, Optional

from squidxplorer._qt_tabs import _FloatWindow


class TabManager:
    """The registries and the float mechanism. POLICY (what may float, what disposal means for
    a given widget, where the log lives) stays with the owner; this class carries the moves."""

    def __init__(self, *, default_tabs: Callable, fixed_tabs: int,
                 dispose: Callable) -> None:
        #: key -> operator-UI widget currently open as a tab.
        self.op_tabs: dict = {}
        #: key -> _FloatWindow holding that widget detached.
        self.floating: dict = {}
        self._default_tabs = default_tabs   # zero-arg callable -> the band's tab bar
        self._fixed_tabs = fixed_tabs
        #: The owner's ONE teardown path for a widget (registry pop included via `forget`).
        self._dispose = dispose

    # -- docked tabs -----------------------------------------------------------------------------
    def open_tab(self, key: str, title: str, builder: Callable, tabs=None) -> None:
        """Open (or focus) a UI as a tab. Built lazily, once. If the UI is currently detached,
        focus its floating window instead — never rebuild, so a widget's live state survives."""
        tabs = self._default_tabs() if tabs is None else tabs
        win = self.floating.get(key)
        if win is not None:
            win.raise_()
            win.activateWindow()
            return
        w = self.op_tabs.get(key)
        if w is None:
            w = builder()
            self.op_tabs[key] = w
            tabs.addTab(w, title)
        tabs.setCurrentWidget(w)

    def close_tab(self, index: int, tabs=None) -> None:
        default = self._default_tabs()
        tabs = default if tabs is None else tabs
        if index < self._fixed_tabs and tabs is default:   # the Operators home tab
            return
        w = tabs.widget(index)
        tabs.removeTab(index)
        self._dispose(w)

    def forget(self, widget) -> None:
        """Drop every registry entry for *widget* — the owner's dispose path calls this so a
        torn-down widget cannot be found again."""
        for k, v in list(self.op_tabs.items()):
            if v is widget:
                del self.op_tabs[k]

    # -- the ONE float mechanism -----------------------------------------------------------------
    def float_out(self, key: str, title: str, widget, *, restore: Callable,
                  dispose_on_close: bool, home_tabs=None) -> Optional[_FloatWindow]:
        """Float *widget* under *key*; raise the existing float instead of building a second.

        ``restore(widget)`` puts the widget back in its home on re-dock. ``dispose_on_close``
        says what the window's close button means: True routes through the owner's dispose path
        (an operator tab's fate), False RE-DOCKS (the log's fate — a live sink must survive its
        window).
        """
        win = self.floating.get(key)
        if win is not None:                 # already out: raise it, never build a second
            win.raise_()
            win.activateWindow()
            return win
        # `*_` is load-bearing: on_redock is connected to QPushButton.clicked, which passes
        # `checked=False` and would land on a bare `lambda k=key:` AS k — so the Re-dock button
        # called redock(False), found no such key in `floating`, and returned silently. The
        # button had been dead since IMA-209 because every test called redock(key) directly
        # instead of clicking it. Swallow the signal's argument and keep the key bound.
        on_close = ((lambda *_, k=key: self._float_closed(k)) if dispose_on_close
                    else (lambda *_, k=key: self.redock(k)))
        win = _FloatWindow(title, widget, on_close=on_close,
                           on_redock=lambda *_, k=key: self.redock(k))
        win._home_tabs = home_tabs          # informational; `restore` is what actually goes home
        win._restore = restore
        self.floating[key] = win
        win.show()
        return win

    def detach(self, index: int, tabs=None) -> Optional[_FloatWindow]:
        """Detach the tab at *index* of *tabs* into a float. Returns the new window, or None
        when the tab can't detach (home tab / unregistered). *tabs* defaults to the one bar."""
        default = self._default_tabs()
        tabs = default if tabs is None else tabs
        if index < self._fixed_tabs and tabs is default:
            return None                  # the Operators home tab is fixed: it never detaches
        if index < 0:
            return None
        w = tabs.widget(index)
        key = next((k for k, v in self.op_tabs.items() if v is w), None)
        if key is None:
            return None
        title = tabs.tabText(index)
        tabs.removeTab(index)
        del self.op_tabs[key]

        def restore(widget, k=key, t=title, bar=tabs):
            # Back to the bar it was dragged out of — the SAME object, so live state survives.
            self.op_tabs[k] = widget
            bar.addTab(widget, t)
            bar.setCurrentWidget(widget)

        return self.float_out(key, title, w, restore=restore, dispose_on_close=True,
                              home_tabs=tabs)

    def redock(self, key: str) -> None:
        """Return the floated widget to its home — the SAME object, so its live state survives
        (close-and-reopen would kill it). Idempotent: an absent key is a no-op."""
        win = self.floating.pop(key, None)
        if win is None:
            return
        restore = getattr(win, "_restore", None)
        w = win.take_content()              # empties the window: its close is plain
        win.close()
        win.deleteLater()
        if w is None or restore is None:
            return
        restore(w)

    def _float_closed(self, key: str) -> None:
        """User closed a disposing float: same fate as closing the tab."""
        win = self.floating.pop(key, None)
        if win is None:
            return
        w = win.take_content()
        if w is not None:
            self._dispose(w)
