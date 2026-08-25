"""ONE launch seam beside the one retire seam.

Teardown was generalised first: ``PlateWindow._retire`` discovers a worker's signals from its
class (via :func:`signal_names`, which lives here now) so a NEW worker is disconnected correctly
by construction. Startup never got the same treatment — ten call sites hand-wrote their own
``.connect()`` blocks and bookkeeping slots, and the two flat-field slots fell off the pattern
entirely for a while (``PlateWindow._stop_flatfield``'s docstring records the SIGABRT that cost).

:func:`launch` is the startup half: it wires the standard callback set onto whichever signal
names the worker declares, registers the worker on the owner's slot attribute — the handle the
existing teardown (``_retire(getattr(self, slot, None))``) already understands — and starts the
thread. Worker CONSTRUCTION stays at the call site on purpose: the class names resolved there
(``_viewer``'s module globals, ``_region_viewer``'s lazy imports from ``_viewer``) are the seams
tests monkeypatch with spies, and moving construction here would silently stop those spies from
intercepting.

Two tolerances, both measured against what exists rather than invented:

* **Name synonyms.** Worker classes name the same two events differently — success is ``done``
  on some and ``finished_ok`` on others, failure is ``problem`` or ``failed``. ``launch``
  resolves each against the worker it was handed, like ``_retire`` introspects rather than
  hardcodes.
* **Stubs.** Test stand-ins range from real ``QThread`` subclasses to plain objects whose
  "signals" are shared no-op recorders, so resolution probes the INSTANCE for a connectable
  attribute instead of trusting class-level ``Signal`` declarations.

Anything else is exact: a signal name the worker does not declare is refused BY NAME, never
guessed — a guessed connection is a callback that never fires, which is precisely the silent
teardown failure ``_retire``'s introspection was built to end.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Union

#: The two spellings each standard event has across the worker classes. Tolerated here so this
#: card does not rename signals on every worker; a worker declaring BOTH would be ambiguous and
#: none does.
DONE_SYNONYMS = ("done", "finished_ok")
PROBLEM_SYNONYMS = ("problem", "failed")

_Callback = Union[Callable, Iterable[Callable]]


def signal_names(cls) -> tuple:
    """Every Signal declared on *cls* or its bases, excluding QThread's own finished/started."""
    from qtpy.QtCore import Signal as _sig
    seen, out = set(), []
    for klass in cls.__mro__:
        for name, value in vars(klass).items():
            if name in seen or name in ("finished", "started"):
                continue
            if isinstance(value, _sig) or type(value).__name__ in ("Signal", "unbound_signal"):
                seen.add(name)
                out.append(name)
    return tuple(out)


def _connectable(worker, name: str):
    """The signal called *name* on *worker*, or None when nothing connectable answers to it."""
    sig = getattr(worker, name, None)
    return sig if sig is not None and callable(getattr(sig, "connect", None)) else None


def _connect(sig, callback: _Callback) -> None:
    """Connect one callable, or each of an iterable of them, in the order given."""
    callbacks = [callback] if callable(callback) else list(callback)
    for cb in callbacks:
        sig.connect(cb)


def _connect_synonym(worker, synonyms: tuple, callback: _Callback, what: str) -> None:
    for name in synonyms:
        sig = _connectable(worker, name)
        if sig is not None:
            _connect(sig, callback)
            return
    raise AttributeError(
        f"{type(worker).__name__} declares no {what} signal - expected one of "
        f"{'/'.join(synonyms)}. Wire it by its exact name via signals={{...}} if it uses "
        f"another word, or declare the standard one on the worker.")


def launch(owner, worker, *, slot: str,
           on_done: Optional[_Callback] = None,
           on_problem: Optional[_Callback] = None,
           on_progress: Optional[_Callback] = None,
           signals: Optional[Mapping[str, _Callback]] = None,
           on_finished: Optional[_Callback] = None) -> Any:
    """Wire, register and start *worker* — THE way an owner's QThread worker begins.

    * ``on_done`` / ``on_problem`` connect to whichever of the standard synonyms
      (:data:`DONE_SYNONYMS` / :data:`PROBLEM_SYNONYMS`) the worker declares; a worker declaring
      neither is refused by name.
    * ``on_progress`` connects to ``progress`` (the one name every worker that reports uses).
    * ``signals`` wires worker-specific signals by their EXACT names (``tileReady``,
      ``resultReady``, ``ready`` …). Each value is a callable or an ordered iterable of them.
    * ``on_finished`` connects to the thread's own ``finished`` — the one signal that fires for
      an ok, failed and stopped run alike, which is why teardown bookkeeping belongs on it.
    * The worker lands on ``owner.<slot>`` BEFORE it starts, so the busy checks and the retire
      seam can see it from its first instant.

    Returns the worker, started.
    """
    if signals:
        for name, callback in signals.items():
            sig = _connectable(worker, name)
            if sig is None:
                raise AttributeError(
                    f"{type(worker).__name__} declares no signal named '{name}' - refusing to "
                    f"guess. Declared: {', '.join(signal_names(type(worker))) or '(none found)'}.")
            _connect(sig, callback)
    if on_done is not None:
        _connect_synonym(worker, DONE_SYNONYMS, on_done, "completion")
    if on_problem is not None:
        _connect_synonym(worker, PROBLEM_SYNONYMS, on_problem, "failure")
    if on_progress is not None:
        _connect_synonym(worker, ("progress",), on_progress, "progress")
    if on_finished is not None:
        finished = _connectable(worker, "finished")
        if finished is None:
            raise AttributeError(
                f"{type(worker).__name__} has no 'finished' signal to carry teardown "
                f"bookkeeping - is it a QThread?")
        _connect(finished, on_finished)
    setattr(owner, slot, worker)
    worker.start()
    return worker


def stop_slot(owner, slot: str) -> None:
    """Retire whatever worker sits on ``owner.<slot>`` (through the owner's own retire seam)
    and clear the slot. The one spelling of "this named worker stops now", so no slot is
    special at teardown."""
    owner._retire(getattr(owner, slot, None))
    setattr(owner, slot, None)
