"""The one place that spells a plate path (``<plate.ome.zarr>/{row}/{col}/{fov}/{level}``).

Separator is a forward slash even on Windows: TensorStore's ``file`` kvstore wants POSIX paths.
"""

from __future__ import annotations

from typing import Optional


def field_path(base, wellpath, fov, level: Optional[object] = None) -> str:
    """Path of one field's image group (``level=None``) or of one pyramid level inside it."""
    parts = [str(base).rstrip("/"), str(wellpath).strip("/"), str(fov)]
    if level is not None:
        parts.append(str(level))
    return "/".join(p for p in parts if p)


def field_levels(field_dir) -> list:
    """The pyramid level names a field declares, coarsest last. Falls back to ``["0"]``."""
    # Deferred import: reader imports this package to compare the contract stamp.
    from pathlib import Path

    from squidxplorer.reader import _group_attrs

    try:
        datasets = (_group_attrs(Path(field_dir)).get("multiscales") or [{}])[0].get("datasets")
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        datasets = None
    levels = [str(d["path"]) for d in (datasets or []) if isinstance(d, dict) and "path" in d]
    return levels or ["0"]
