"""What the application is DOING right now, in one place.

Spencer Schwarz (CSO): "responsiveness is important. And an indicator when its working."

The complaint this answers is not that work is slow -- fusing a 5731x4793 mosaic takes as long
as it takes. It is that a busy application and a wedged application LOOK IDENTICAL. Julio, on
the mosaic load: "When I click on a different mosaic, say for my image, it takes some time to
load", and separately, on a GUI that had silently died: feedback was given on a window that was
no longer running. A user cannot tell "working" from "broken" unless the app says so.

WHY A REGISTRY AND NOT A FLAG ON THE WIDGET
-------------------------------------------
This codebase's dominant defect shape is "two representations of one truth, hand-synced" -- 4+
confirmed instances, including the contrast that stopped following after one region change and
the two operator registries that drifted in production. A ``self._busy = True`` set at five call
sites and cleared at four is that defect with a new name, and the missed path leaves a spinner
running forever over an idle application, which is worse than no indicator at all: it teaches
the user that the indicator lies.

So there is ONE owner. Work announces itself by starting an activity and ending it; every widget
that shows activity SUBSCRIBES. Nothing else stores whether the app is busy.

Pure Python -- no Qt -- so the rules are testable without a window, and so a failure here cannot
be swallowed by Qt's habit of eating exceptions raised inside a slot.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional


class Activity:
    """One unit of work the user should be told about.

    ``total is None`` means the size is UNKNOWN, which is the honest state for a mosaic fuse
    (the work is one lazy graph, not N countable steps). An unknown total drives an
    INDETERMINATE indicator rather than a fake percentage -- a progress bar that invents a
    denominator is a lie that gets believed.
    """

    __slots__ = ("key", "label", "done", "total")

    def __init__(self, key: str, label: str, total: Optional[int] = None) -> None:
        self.key = key
        self.label = label
        self.done = 0
        self.total = total

    @property
    def determinate(self) -> bool:
        return self.total is not None and self.total > 0

    def sentence(self) -> str:
        """What to show a human. Never a bare number: the label says what is working."""
        if self.determinate:
            return f"{self.label} · {self.done}/{self.total}"
        return f"{self.label} …"

    def __repr__(self) -> str:                                   # pragma: no cover - debugging
        return f"<Activity {self.key} {self.sentence()}>"


class ActivityLog:
    """THE registry of in-flight work. One instance per window.

    Re-entrant by KEY: starting a key that is already running REPLACES it rather than stacking,
    because the callers are event handlers that can legitimately fire twice (a region change
    while the previous fuse is still draining). Ending a key that is not running is a no-op, not
    an error -- teardown paths call ``end`` defensively and must not raise on the way out.
    """

    def __init__(self) -> None:
        self._items: dict[str, Activity] = {}
        self._subs: list[Callable[["ActivityLog"], None]] = []

    # -- reading ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Activity]:
        return iter(list(self._items.values()))

    @property
    def busy(self) -> bool:
        return bool(self._items)

    def get(self, key: str) -> Optional[Activity]:
        return self._items.get(key)

    def current(self) -> Optional[Activity]:
        """The activity an indicator should show when it can only show one.

        A DETERMINATE one wins: it can show real progress, and "3/28 wells" tells the user more
        than a spinner. Otherwise the first one started.
        """
        items = list(self._items.values())
        if not items:
            return None
        for a in items:
            if a.determinate:
                return a
        return items[0]

    def sentence(self) -> str:
        """One line for the whole log, including how much else is going on."""
        a = self.current()
        if a is None:
            return ""
        rest = len(self._items) - 1
        return a.sentence() + (f"  (+{rest} more)" if rest else "")

    # -- writing ----------------------------------------------------------------------
    def start(self, key: str, label: str, total: Optional[int] = None) -> Activity:
        a = Activity(key, label, total)
        self._items[key] = a
        self._fire()
        return a

    def advance(self, key: str, done: int, total: Optional[int] = None) -> None:
        """Report progress. Silently ignored for a key that is not running.

        Ignored rather than raising because progress signals arrive asynchronously and can
        outlive their activity by one delivery -- a worker's last ``progress`` can be queued
        behind its own ``finished``. Raising there would turn a benign race into a crash inside
        a Qt slot.
        """
        a = self._items.get(key)
        if a is None:
            return
        a.done = int(done)
        if total is not None:
            a.total = int(total)
        self._fire()

    def end(self, key: str) -> None:
        if self._items.pop(key, None) is not None:
            self._fire()

    def clear(self) -> None:
        """Everything stopped -- used on teardown, and as the one recovery from a leaked key."""
        if self._items:
            self._items.clear()
            self._fire()

    # -- subscribing -------------------------------------------------------------------
    def subscribe(self, callback: Callable[["ActivityLog"], None]) -> None:
        """Be told whenever the picture changes. Called immediately with the current state, so a
        widget built while work is already running does not start out blank and wrong."""
        self._subs.append(callback)
        callback(self)

    def _fire(self) -> None:
        for cb in list(self._subs):
            cb(self)
