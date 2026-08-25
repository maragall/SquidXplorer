"""Operator discovery: how an operator that lives in another package gets into the registries.

Python packaging entry points, group "squidxplorer.operators". A package declares:

    [project.entry-points."squidxplorer.operators"]
    my-operators = "my_package:register"

and on `import squidxplorer` we import my_package, call its register(), and its operators land
in the same tables as the built-ins. The built-ins keep their hardcoded imports; discovery runs
after them, so a name collision with a built-in is refused rather than overwriting it.

A plugin that fails to import or raises from register() aborts `import squidxplorer` with an
OperatorPluginError naming the plugin and the underlying error, rather than silently missing.
SQUIDXPLORER_NO_PLUGINS=1 skips discovery entirely.
"""

from __future__ import annotations

import os
from typing import Optional

GROUP: str = "squidxplorer.operators"
DISABLE_ENV: str = "SQUIDXPLORER_NO_PLUGINS"


class OperatorPluginError(RuntimeError):
    """A declared operator plugin could not be loaded."""


def _entry_points(group: str) -> list:
    """Declared entry points in *group*, sorted by name so load order is deterministic."""
    from importlib.metadata import entry_points

    return sorted(entry_points(group=group), key=lambda ep: (ep.name, ep.value))


def _describe(ep) -> str:
    """"my-operators" (my_package:register, from my-package 0.2.0)."""
    dist = getattr(ep, "dist", None)
    origin = ""
    if dist is not None:
        origin = f", from {dist.metadata['Name']} {dist.version}"
    return f"{ep.name!r} ({ep.value}{origin})"


def load_operator_plugins(group: str = GROUP) -> list[str]:
    """Import and run every declared operator plugin. Returns the entry-point names loaded.

    Each entry point is loaded and, if the result is callable, called with no arguments:
    my_package:register is a function that makes the add_operator calls; a bare module name
    registers as an import side effect instead, like this package's own built-ins.
    """
    if os.environ.get(DISABLE_ENV):
        return []

    loaded: list[str] = []
    for ep in _entry_points(group):
        try:
            target = ep.load()
        except Exception as exc:                     # noqa: BLE001 - re-raised, named, chained
            raise OperatorPluginError(
                f"operator plugin {_describe(ep)} could not be imported: "
                f"{type(exc).__name__}: {exc}. "
                f"Fix or uninstall the package that declares it; "
                f"{DISABLE_ENV}=1 starts squidxplorer without any plugins."
            ) from exc
        if callable(target):
            try:
                target()
            except Exception as exc:                 # noqa: BLE001 - re-raised, named, chained
                raise OperatorPluginError(
                    f"operator plugin {_describe(ep)} raised while registering its operators: "
                    f"{type(exc).__name__}: {exc}. "
                    f"A name that is already taken is the common cause - every registrar refuses a "
                    f"duplicate rather than clobbering it. "
                    f"{DISABLE_ENV}=1 starts squidxplorer without any plugins."
                ) from exc
        loaded.append(ep.name)
    return loaded


def declared_operator_plugins(group: str = GROUP) -> list[tuple[str, str, Optional[str]]]:
    """[(entry_point_name, target, distribution), ...] for every declared plugin. Imports nothing."""
    out = []
    for ep in _entry_points(group):
        dist = getattr(ep, "dist", None)
        out.append((ep.name, ep.value, dist.metadata["Name"] if dist is not None else None))
    return out
