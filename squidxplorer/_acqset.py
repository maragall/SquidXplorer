"""Folder-of-acquisitions support: discovery, the window's set bookkeeping, and the
one-operator-over-every-acquisition SAVE loop.

Qt-free. An "acquisition" is exactly what ``open_reader`` accepts (the reader's own detection,
never re-derived here); a SET is a folder that is not itself an acquisition but holds >= 2
immediate child acquisitions, name-sorted. One operator per bulk run, run sequentially: the
engine already refuses chained operators, and N parallel acquisitions is N times the RAM.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Optional


class AcqSetError(ValueError):
    """A folder that is neither one acquisition nor a set of them, refused by name."""


def is_acquisition(path) -> bool:
    """True when ``open_reader`` would accept *path* as a raw acquisition."""
    from squidxplorer import open_reader
    # Lazy: resolve_plate_root is pure pathlib but lives in a Qt-importing module.
    from squidxplorer._plate_overview import resolve_plate_root

    p = Path(path)
    if not p.is_dir():
        return False
    if resolve_plate_root(p)[1]:
        return False    # a written plate is an OUTPUT; ingest refuses it with this same test
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    # mixed-layout warning is detection detail here
            open_reader(str(p))               # readers are lazy: acceptance costs one listing
        return True
    except Exception:   # noqa: BLE001, open_reader refuses by raising; any raise means "not one"
        return False


def discover_acquisitions(folder) -> "list[Path]":
    """``[folder]`` for one acquisition, the name-sorted child acquisitions for a set.

    One level only: a grandchild acquisition never makes its parent a set member. A folder
    that is neither raises :class:`AcqSetError` naming what was found.
    """
    p = Path(folder)
    if not p.is_dir():
        raise AcqSetError(f"{p} is not a directory")
    if is_acquisition(p):
        return [p]
    children = sorted((c for c in p.iterdir() if c.is_dir()), key=lambda c: c.name)
    members = [c for c in children if is_acquisition(c)]
    if len(members) >= 2:
        return members
    raise AcqSetError(
        f"{p} is neither a Squid acquisition nor a folder of them: "
        f"{len(members)} of {len(children)} immediate subfolder(s) are acquisitions, "
        "and a multi-acquisition folder needs at least 2.")


def note_set(win, path) -> str:
    """Record the set (if any) on *win* and return the member path ingest should open.

    Cycling passes a member of the loaded set: the index moves, the set stays. A fresh drop
    re-discovers; a plain acquisition or an unreadable folder clears the set and passes
    through so ingest refuses it with its own sentence.
    """
    p = Path(path)
    current = getattr(win, "_acq_set", None)
    if current and p in current:
        win._acq_set_index = current.index(p)
        return str(p)
    win._acq_set, win._acq_set_index = None, 0
    if not p.is_dir() or is_acquisition(p):
        return str(p)
    try:
        members = discover_acquisitions(p)
    except AcqSetError:
        return str(p)
    win._acq_set, win._acq_set_index = members, 0
    return str(members[0])


def run_over_set(paths, *, operator: str, out_parent,
                 parameters: Optional[dict] = None,
                 log: Optional[Callable[[str], None]] = None,
                 stop: Optional[Callable[[], bool]] = None) -> dict:
    """SAVE *operator* once per acquisition, sequentially, with the SAME parameters.

    Per-acquisition fault isolation, same philosophy as per-well: one failing acquisition is
    logged by name and the loop continues. A stop request ends the whole set run. Readers
    need no explicit close: TIFF handles ride the process-wide LRU. Returns
    ``{"ok", "partial", "failed", "total", "stopped"}``.
    """
    from squidxplorer import _acq_output, _fused_output, _measure, open_reader
    from squidxplorer._dispatch import run_operator_once
    from squidxplorer._engine import operator_saves_copy

    say = log or (lambda _line: None)
    paths = [Path(p) for p in paths]
    n = len(paths)
    counts = {"ok": 0, "partial": 0, "failed": 0, "total": n, "stopped": False}
    for i, p in enumerate(paths, start=1):
        if stop is not None and stop():
            counts["stopped"] = True
            say(f"set run stopped before acquisition {i} of {n} ({p.name})")
            break
        head = f"acquisition {i} of {n}: {operator} on {p.name}"
        try:
            reader = open_reader(str(p))   # un-padded: a save keeps skipping missing wells
            owed = len(reader.metadata.get("fovs_per_region") or {})
            out_dir = None
            if not operator_saves_copy(operator):
                # run_operator's own destination naming, per acquisition under one parent.
                acq_format = (
                    _acq_output.acquisition_format_dst(reader, operator) is not None
                    or _fused_output.fused_format_dst(reader, operator) is not None)
                out_dir = Path(out_parent) / (f"{operator}_{p.name}" if acq_format
                                              else f"{p.name}.hcs")
            result = run_operator_once(
                reader, operator=operator, save=True, owed=owed, out_dir=out_dir,
                regions=None, n_fovs=None, parameters=parameters, stop=stop)
        except Exception as exc:   # noqa: BLE001, isolation: named, counted, loop continues
            counts["failed"] += 1
            say(f"{head} … failed: {type(exc).__name__}: {exc}")
            continue
        detail = f" ({result.detail})" if result.detail else ""
        say(f"{head} … {result.outcome}{detail}")
        if result.stopped:
            counts["stopped"] = True
            break
        counts["ok" if result.outcome == _measure.OK else "partial"] += 1
    say(f"set run finished: {counts['ok']} ok, {counts['partial']} partial, "
        f"{counts['failed']} failed of {n} acquisition(s)"
        + (" (stopped early)" if counts["stopped"] else ""))
    return counts
