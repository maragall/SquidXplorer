"""The SquidXplorer operator template — a complete, installable example plugin.

Copy this directory, rename the package, replace ``_stdev.py`` with your algorithm, keep the
shape. See ``../README.md`` for the contract this file implements: one ``register()`` function,
named by the ``squidxplorer.operators`` entry point in ``pyproject.toml``, called once during
``import squidxplorer``.
"""

from __future__ import annotations

from squidxplorer_operator_template._stdev import (
    DEFAULT_DDOF,
    DEFAULT_SMOOTH_SIGMA,
    stdev_op,
)

# `from squidxplorer import ...` at MODULE scope deadlocks: SquidXplorer loads this plugin from
# inside `import squidxplorer`, so a module-scope import of squidxplorer is re-entrant and sees a
# half-initialised module. Import squidxplorer INSIDE register() instead. Your own algorithm
# module should not import squidxplorer at all, so it stays testable without the app.

__all__ = ["OPERATOR_NAME", "register", "stdev_op"]

#: The name this operator holds in every registry-driven surface. Pick it for the ALGORITHM, not
#: the package, and keep it unique — `add_operator` refuses a name that is already taken.
OPERATOR_NAME: str = "stdev"


def register() -> None:
    """Register this package's operators. Called by SquidXplorer during ``import squidxplorer``.

    The four declarations: ``consumes`` (which axis you eat, and therefore the engine's loop and
    your output's shape), ``produces`` (what your pixels MEAN, and therefore the napari layer
    type), ``params`` (what one entry can be RUN with — non-empty makes the registered object a
    factory), and ``requires`` (the importable module names you need, so a missing dependency
    refuses by name instead of failing per-well deep inside the run).
    """
    from squidxplorer import Param, add_operator       # INSIDE the function — see the note above

    add_operator(
        OPERATOR_NAME,
        stdev_op,                                  # a FACTORY, because params= is non-empty
        consumes=frozenset({"z"}),                 # z-reducer: (T, C, Nz, Y, X) -> (T, C, 1, Y, X)
        produces="intensity",                      # a measurement of light -> napari Image layer
        params=(
            Param("smooth_sigma", DEFAULT_SMOOTH_SIGMA,
                  "Gaussian low-pass applied to each plane first, in pixels. 0 disables it; "
                  "without it the result is dominated by per-plane shot noise."),
            Param("ddof", DEFAULT_DDOF,
                  "Delta degrees of freedom. 0 = population standard deviation (divide by N), "
                  "which is right when the z-stack IS the population."),
        ),
        requires=("scipy",),                       # refused BY NAME when absent; listed either way
    )
