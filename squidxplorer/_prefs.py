"""USER PREFERENCES THAT OUTLIVE THE PROCESS -- one tiny JSON file, one reader, one writer.

This is deliberately NOT :class:`squidxplorer._region_viewer.ViewSettings`, and the distinction is the
reason this module exists rather than a field being added there. ``ViewSettings`` holds one
WINDOW's baseline and the overrides made in it, in memory, for the life of that window; its whole
design is that a change in one window never reaches another and never reaches the next session
("propagating one window's change to the others fights the independence the decentralization
bought"). A preference is the opposite thing: it is a decision the user makes ONCE, about the
application, and expects to still be true tomorrow.

The first of them is ``warn_close_all``. A "don't show me this again" checkbox that does not
survive a restart is not a preference, it is a lie with a checkbox on it -- so the storage had to
exist before the dialog could honestly offer one.

DELIBERATELY SMALL. No schema, no migration, no defaults table: a caller passes the default it
wants at the point of asking, so a preference that has never been set and one this build does not
know are the same answer, and deleting the file is always a complete reset. Anything that needs
more structure than "a name and a JSON value" is not a preference and belongs where its own rules
live.

EVERY FAILURE IS SURVIVABLE AND SAID. A read that cannot parse returns the caller's default and
logs; a write that cannot land logs and returns False. Neither raises: a preferences file on a
read-only home directory, or one a previous version left half-written, must not stop the
application from starting -- the cost of losing a checkbox state is a dialog the user has to
dismiss again, and the cost of raising is an application that will not open.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("squid.xplorer.prefs")

#: The one file. ``SQUIDXPLORER_PREFS`` overrides it, which is what makes this testable without
#: writing to the developer's real home directory -- and what lets a kiosk deployment point every
#: instance at a read-only shipped file.
_ENV = "SQUIDXPLORER_PREFS"


def prefs_path() -> Path:
    """Where preferences live. ``~/.squidxplorer/prefs.json`` unless ``SQUIDXPLORER_PREFS`` says otherwise.

    ``~/.squidxplorer`` is the directory ``_plugins.py`` already names for user-level squidxplorer state,
    so this adds a file to an existing home rather than a second dot-directory.
    """
    override = os.environ.get(_ENV)
    if override:
        return Path(override)
    return Path.home() / ".squidxplorer" / "prefs.json"


def _load() -> dict:
    path = prefs_path()
    try:
        if not path.exists():
            return {}
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                     # noqa: BLE001 - unreadable prefs are not fatal
        log.warning("preferences at %s could not be read (%s: %s); using defaults. "
                    "Delete the file to reset it.", path, type(exc).__name__, exc)
        return {}
    if not isinstance(blob, dict):
        log.warning("preferences at %s are %s, not an object; using defaults.",
                    path, type(blob).__name__)
        return {}
    return blob


def get(name: str, default: Any = None) -> Any:
    """The stored value for *name*, or *default* when it has never been set.

    The default is the CALLER'S, passed at the point of asking, so this module holds no table of
    them to fall out of step with the code that reads them.
    """
    return _load().get(str(name), default)


def set(name: str, value: Any) -> bool:            # noqa: A001 - `prefs.set(...)` is the point
    """Store *value* under *name*. Returns whether it actually landed on disk.

    Written whole, every time: the file holds a handful of scalars, so read-modify-write costs
    nothing and there is no partial-update path to get wrong. The return value exists because a
    caller offering "don't show me this again" needs to know whether the promise was kept -- a
    checkbox that silently fails to persist is the defect this module's docstring names.
    """
    path = prefs_path()
    blob = _load()
    blob[str(name)] = value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written to a sibling and RENAMED, so an interrupted write cannot leave the file
        # truncated -- which is the one way this module could turn a lost checkbox into a lost
        # file. os.replace is atomic within a filesystem.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(blob, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception as exc:                     # noqa: BLE001 - an unwritable home is not fatal
        log.warning("preference %r could not be saved to %s (%s: %s); it will apply to this "
                    "session only.", name, path, type(exc).__name__, exc)
        return False
