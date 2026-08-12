"""The ONE place that SPELLS a plate path from its parts.

The other legitimate route to a field is to descend the plate metadata (``wells[].path`` ->
``well.images[].path`` -> ``datasets[].path``), which reads every name rather than assuming it;
``_montage`` and ``_tilesource`` do that. What is banned is inventing the path a third way.


Gap 5 of the three-viewers review. Before this module the plate layout was reconstructed by string
formatting at four sites in ``_viewer.py`` (``:1030``, ``:1045``, ``:3726``, ``:6835``), three of
which handed the string straight to a store open and never went near ``reader.py``. Four copies of
one rule is four chances for the rule to change in three places, and it is what made the version
gate in :mod:`squidxplorer.contract.version` worth building: a declared version is decorative unless
there is a single seam where the declared layout is actually used.

The layout itself is the STABLE half of ``docs/plate-contract.md``::

    <plate.ome.zarr>/{row}/{col}/{fov}/{level}

``wellpath`` is the NGFF ``plate.wells[].path``, i.e. ``"{row}/{col}"`` already joined, because
that is the form the plate metadata records and the form every caller already has. It is used
verbatim, never re-derived: ``_output.parse_well_id`` writes ``B2 -> B/2`` and never ``B/02``, and
the spec permits any alphanumeric row and column NAME, so recomposing it from a well id would be a
second, subtly different implementation of the same rule.

Separator is a forward slash even on Windows, deliberately. These strings are consumed by
TensorStore's ``file`` kvstore, which takes POSIX-style paths on every platform, and by ``Path``,
which accepts them everywhere. ``os.path.join`` here would produce backslashes that the kvstore
does not want.
"""

from __future__ import annotations

from typing import Optional


def field_path(base, wellpath, fov, level: Optional[object] = None) -> str:
    """Path of one field's image group, or of one pyramid level inside it.

    ``level=None`` gives the FIELD GROUP (the directory holding ``zarr.json`` with ``multiscales``
    and ``omero``); a level gives the ARRAY inside it. Both callers exist: the loupe reads the
    group to discover how many levels a field has, then opens one of them.

    The level is stringified rather than indexed, because ``datasets[].path`` is a NAME the store
    chooses. SquidXplorer writes ``"0"``, ``"1"``, ..., but a conforming store may write anything, so a
    caller that has read the multiscales hands the recorded path through unchanged.
    """
    parts = [str(base).rstrip("/"), str(wellpath).strip("/"), str(fov)]
    if level is not None:
        parts.append(str(level))
    return "/".join(p for p in parts if p)


def field_levels(field_dir) -> list:
    """The pyramid level NAMES a field declares, coarsest last. Falls back to ``["0"]``.

    ``multiscales[0].datasets[*].path``, read through ``reader._group_attrs`` so the v0.4 versus
    v0.5 attribute difference lives in exactly one place. ``_viewer._ZarrLoupeSource`` used to
    hand-parse ``zarr.json -> attributes -> ome -> multiscales[0] -> datasets[*].path`` itself
    behind a bare ``except Exception``, which is a fifth copy of the layout AND an unnamed
    fallback. The fallback is real and is now written down (``docs/plate-contract.md``, optional
    section): a multi-level pyramid is OPTIONAL, small fields are written single-level on purpose,
    and a field always has a full-resolution array ``"0"``.

    The reader import is deferred: ``reader`` imports this package to compare the contract stamp.
    """
    from pathlib import Path

    from squidxplorer.reader import _group_attrs

    try:
        datasets = (_group_attrs(Path(field_dir)).get("multiscales") or [{}])[0].get("datasets")
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        datasets = None
    levels = [str(d["path"]) for d in (datasets or []) if isinstance(d, dict) and "path" in d]
    return levels or ["0"]
