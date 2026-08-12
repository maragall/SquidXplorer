"""User preferences that outlive the process — one tiny JSON file, one reader, one writer.

Distinct from ViewSettings, which holds one window's baseline in memory for that window's life.
A preference is a decision the user makes once, about the application, expected to still hold
tomorrow (e.g. warn_close_all's "don't show me this again").

Deliberately small: no schema, no migration, no defaults table — a caller passes its own
default at the point of asking. Every failure is survivable and logged, never raised: a
preferences file on a read-only home, or one half-written by a previous version, must not stop
the application from starting.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("squid.xplorer.prefs")

#: SQUIDXPLORER_PREFS overrides the path, for tests and kiosk deployments.
_ENV = "SQUIDXPLORER_PREFS"


def prefs_path() -> Path:
    """~/.squidxplorer/prefs.json unless SQUIDXPLORER_PREFS says otherwise."""
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
    """The stored value for *name*, or *default* when it has never been set."""
    return _load().get(str(name), default)


def set(name: str, value: Any) -> bool:            # noqa: A001 - `prefs.set(...)` is the point
    """Store *value* under *name*. Returns whether it actually landed on disk."""
    path = prefs_path()
    blob = _load()
    blob[str(name)] = value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # written to a sibling and renamed so an interrupted write cannot truncate the file
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(blob, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception as exc:                     # noqa: BLE001 - an unwritable home is not fatal
        log.warning("preference %r could not be saved to %s (%s: %s); it will apply to this "
                    "session only.", name, path, type(exc).__name__, exc)
        return False
