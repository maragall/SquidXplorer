"""Operator discovery: how an operator that lives in ANOTHER package gets into the registries.

THE PROBLEM THIS CLOSES
-----------------------
Adding an operator was already cheap *inside* this repo — one ``add_projector`` call, no engine
edit. It was impossible *outside* it. ``squidmip/__init__.py`` ends in a hardcoded side-effect
import list (``from squidmip import _background, _decon, _flatfield, _spots``), and that list is
the ONLY thing that makes a registration run. An operator in a package we do not ship appears in
``available_projectors()``, the CLI's ``--projector``, the viewer's operator list and ``ops list``
— never. There was no mechanism, not a limited one.

So a community contributor could not adapt their own repo to work with SquidXplorer without
editing SquidXplorer. That is the whole point of the operator template.

THE SEAM
--------
Python packaging entry points, group ``squidmip.operators``. A package declares::

    [project.entry-points."squidmip.operators"]
    my-operators = "my_package:register"

and on ``import squidmip`` we import ``my_package``, call its ``register()``, and its operators are
in the same tables as ``mip``. Nothing else changes: the operator is declared with the same
``consumes`` / ``produces`` / ``params`` / ``requires`` record, dispatched by the same loop, drawn
by the same delivery path, and validated by the same tests (``test_operator_declaration`` is
parametrised over ``available_projectors()``, so a plugin registered at import time is checked by
tests written before it existed).

Chosen over the alternatives for the reason each was rejected:

* **A scan of a plugin directory** (``~/.squidmip/operators/*.py``) — needs its own installer, its
  own version story, and executes whatever is in a folder. Entry points get all of that from pip.
* **A config file listing modules to import** — a second source of truth beside the installed
  environment, and it goes stale exactly when someone uninstalls a package.
* **napari's own plugin manager (npe2)** — the right answer if these were napari WIDGETS. They are
  not: an operator is a headless ``Iterable[plane] -> plane`` that the CLI runs with no Qt in the
  process, and npe2 would make the headless pipeline depend on the GUI stack.

Prior art: this is what scikit-image's ``skimage.io`` plugins, pytest's ``pytest11``, napari's
``napari.manifest`` and Fractal's task manifests all do. The group name is spelled the way pytest's
is — one group, one namespace, package-declared.

ADDITIVE, NEVER A REPLACEMENT
-----------------------------
The built-ins keep their hardcoded imports. They ship in this package; routing them through the
entry-point machinery would make ``mip`` depend on installed metadata being intact, which is a
worse failure mode for zero benefit. Discovery runs AFTER them, so a plugin sees a fully populated
registry and a name collision with a built-in is refused by ``add_projector`` rather than
overwriting ``mip``.

FAIL LOUD AND NAMED
-------------------
A plugin that raises on import, or whose ``register()`` raises, or whose entry point does not
resolve, aborts ``import squidmip`` with an :class:`OperatorPluginError` naming the plugin, its
entry-point target, and the underlying error.

This is a deliberate, uncomfortable choice, and it is the same one the rest of this codebase makes:
the alternative is a ``try/except: pass`` that produces an application which silently does not have
the operator you installed, which is indistinguishable from an operator that was never written —
the exact defect ``requires=`` exists to end, reintroduced one layer up. A broken plugin is a
broken installation and it says so, with the uninstall command in the message.

The escape hatch is an environment variable, not a code path: ``SQUIDMIP_NO_PLUGINS=1`` skips
discovery entirely, so a user whose app will not start can still start it and uninstall the
offender. It is named in the error message.
"""

from __future__ import annotations

import os
from typing import Optional

#: The entry-point group a package declares its operators under. One group for every registry:
#: a plugin's ``register()`` may call ``add_projector``, ``add_region_operator`` and
#: ``add_segmenter`` as it likes. Splitting the group per table would force a contributor to know
#: which of our five tables their idea belongs in before they can declare anything.
GROUP: str = "squidmip.operators"

#: Set to a truthy value to skip discovery. The stated escape hatch for a broken plugin.
DISABLE_ENV: str = "SQUIDMIP_NO_PLUGINS"


class OperatorPluginError(RuntimeError):
    """A declared operator plugin could not be loaded. Names the plugin and what went wrong.

    Raised out of ``import squidmip``. See this module's docstring for why that is loud rather
    than skipped, and for the ``SQUIDMIP_NO_PLUGINS=1`` escape hatch.
    """


def _entry_points(group: str) -> list:
    """The declared entry points in *group*, sorted by name so load order is deterministic.

    Deterministic because two plugins can collide on an operator name, and a collision that
    depends on filesystem iteration order would be reproducible on one machine and not another.
    """
    from importlib.metadata import entry_points

    return sorted(entry_points(group=group), key=lambda ep: (ep.name, ep.value))


def _describe(ep) -> str:
    """``"my-operators" (my_package:register, from my-package 0.2.0)`` — everything needed to act.

    The distribution is what a user actually uninstalls, and it is NOT derivable from the module
    name, so the message carries it. ``ep.dist`` is absent on some older metadata backends; the
    name and target are the parts that matter and they are always there.
    """
    dist = getattr(ep, "dist", None)
    origin = ""
    if dist is not None:
        origin = f", from {dist.metadata['Name']} {dist.version}"
    return f"{ep.name!r} ({ep.value}{origin})"


def load_operator_plugins(group: str = GROUP) -> list[str]:
    """Import and run every declared operator plugin. Returns the entry-point names loaded.

    Called once, from the bottom of ``squidmip/__init__.py``, after the built-in registrations.

    Each entry point is loaded (``ep.load()``, which imports the module and resolves the attribute)
    and, if the result is callable, CALLED with no arguments. Both spellings are supported and both
    are honest:

    * ``my_package:register`` — a function that makes the ``add_projector`` calls. Preferred, and
      what the template ships: the registration is an explicit, testable, importable function.
    * ``my_package`` (a module object) — registration happens as an import side effect, which is
      exactly how this package's own ``_decon`` / ``_background`` / ``_flatfield`` do it. Not
      callable, so nothing is called.

    Raises
    ------
    OperatorPluginError
        If any plugin fails to import, fails to resolve, or raises from ``register()``. Named,
        never skipped. The message carries the plugin, its target, its distribution and the
        original error, and the original exception is chained as ``__cause__``.
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
                f"{DISABLE_ENV}=1 starts squidmip without any plugins."
            ) from exc
        if callable(target):
            try:
                target()
            except Exception as exc:                 # noqa: BLE001 - re-raised, named, chained
                raise OperatorPluginError(
                    f"operator plugin {_describe(ep)} raised while registering its operators: "
                    f"{type(exc).__name__}: {exc}. "
                    f"A name that is already taken is the common cause — every registrar refuses a "
                    f"duplicate rather than clobbering it. "
                    f"{DISABLE_ENV}=1 starts squidmip without any plugins."
                ) from exc
        loaded.append(ep.name)
    return loaded


def declared_operator_plugins(group: str = GROUP) -> list[tuple[str, str, Optional[str]]]:
    """``[(entry_point_name, target, distribution), ...]`` for every declared plugin.

    A read-only inventory for ``ops list`` and for a bug report: "which operators are not ours"
    is the first question when a run behaves unlike the shipped app. Imports nothing.
    """
    out = []
    for ep in _entry_points(group):
        dist = getattr(ep, "dist", None)
        out.append((ep.name, ep.value, dist.metadata["Name"] if dist is not None else None))
    return out
