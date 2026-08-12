"""One registry of the application's in-flight work; widgets subscribe.

Pure Python, no Qt, so the rules are testable without a window.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional


class Activity:
    """One unit of work the user should be told about; ``total is None`` means indeterminate."""

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
        """What to show a human."""
        if self.determinate:
            return f"{self.label} · {self.done}/{self.total}"
        return f"{self.label} …"

    def __repr__(self) -> str:                                   # pragma: no cover - debugging
        return f"<Activity {self.key} {self.sentence()}>"


class ActivityLog:
    """The registry of in-flight work, one per window; re-entrant by key, ``end`` is a no-op."""

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
        """The activity an indicator should show when it can only show one; determinate wins."""
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
        """Report progress; silently ignored for a key that is not running."""
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
        """Everything stopped; used on teardown."""
        if self._items:
            self._items.clear()
            self._fire()

    # -- subscribing -------------------------------------------------------------------
    def subscribe(self, callback: Callable[["ActivityLog"], None]) -> None:
        """Be told whenever the picture changes; called immediately with the current state."""
        self._subs.append(callback)
        callback(self)

    def _fire(self) -> None:
        for cb in list(self._subs):
            cb(self)
