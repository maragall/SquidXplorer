"""The SquidXplorer operator template — a complete, installable example plugin.

Copy this directory, rename the package, replace ``_stdev.py`` with your algorithm, keep the
shape. See ``../README.md`` for the contract this file implements.

WHAT THIS FILE IS
-----------------
One function, ``register()``, named by the ``squidxplorer.operators`` entry point in
``pyproject.toml``. SquidXplorer calls it once, during ``import squidxplorer``, after its own
operators are registered. Everything your package contributes happens inside it.

That is the entire integration surface. There is no base class to subclass, no interface to
implement, no manifest to keep in sync, and nothing to edit in SquidXplorer.
"""

from __future__ import annotations

from squidxplorer_operator_template._stdev import (
    DEFAULT_DDOF,
    DEFAULT_SMOOTH_SIGMA,
    stdev_op,
)

# NOTE THE IMPORT THAT IS NOT HERE. `from squidxplorer import Param, add_projector` at MODULE scope
# looks natural and deadlocks: SquidXplorer loads this plugin from inside `import squidxplorer`, so a
# module-scope import of squidxplorer is re-entrant. Whichever import happens first, the other sees a
# half-initialised module, and it surfaces as
#
#   OperatorPluginError: operator plugin 'my-operators' (my_package:register) could not be
#   imported: AttributeError: partially initialized module 'my_package' has no attribute
#   'register' (most likely due to a circular import)
#
# which is loud and named, and still a wasted afternoon. Import squidxplorer INSIDE `register()`. By
# the time it is called, everything a plugin needs is bound. Your own algorithm module
# (`_stdev.py`) should not import squidxplorer at all — it is plain numpy, which is also what makes it
# testable without the app.

__all__ = ["OPERATOR_NAME", "register", "stdev_op"]

#: The name this operator holds in every registry-driven surface: ``available_projectors()``, the
#: CLI's ``--projector``, the viewer's operator list, ``list_operators``, the layer it writes into.
#:
#: Pick it for the ALGORITHM, not for your package. Two spellings of one operator (one in a log
#: line, one on a layer) is how they start disagreeing. And it must be unique across the whole
#: installed environment: ``add_projector`` REFUSES a name that is already taken rather than
#: clobbering it, so a collision is a loud failure at import, which is the correct outcome — but
#: prefixing an unusually generic name ("threshold", "filter") is polite to the next plugin.
OPERATOR_NAME: str = "stdev"


def register() -> None:
    """Register this package's operators. Called by SquidXplorer during ``import squidxplorer``.

    THE FOUR DECLARATIONS, and what each one decides. Nothing in SquidXplorer branches on your
    operator's NAME — a test in that repo (``test_no_module_branches_on_an_operator_name``) fails
    the build if anything does. Everything generic code needs to know about your operator is one
    of these:

    ``consumes``
        WHICH AXIS you eat, and therefore the engine's loop and your output's shape.
        ``frozenset({"z"})`` (the default, and what this operator declares) means you are handed a
        whole z-stack per (t, c) and return one plane, so z collapses to 1. ``frozenset()`` means
        you are handed one plane at a time and z survives at full depth. Inferred from the callable
        when ``squidxplorer.plane_op`` stamped it; declared explicitly here because this is a z-reducer
        built by hand, and the explicit spelling is the readable one.

    ``produces``
        What your output PIXELS MEAN, and therefore which napari layer type they become.
        ``"intensity"`` (the default) = a measurement of light: windowed, colormapped, blended.
        ``"labels"`` = integer OBJECT IDS: 0 renders transparent, never windowed, never
        interpolated. Declaring this wrong is not cosmetic — a segmentation delivered as intensity
        arrives auto-windowed as if label 37 were 37 photons, which is a defect this project has
        actually shipped.

    ``params``
        What your ONE entry can be RUN with. Because ``params`` is non-empty, the object passed to
        ``add_projector`` is read as a FACTORY: it is called with these defaults to build the
        default binding, and called again per run with a caller's ``operator_kwargs``. A ``Param``
        is a name, a default and one line of prose — deliberately no type, no range, no widget
        hint. The moment it carries a widget hint it has become the GUI's schema and two places own
        the same fact.

    ``requires``
        The importable MODULE names you need. Your operator is always LISTED whether or not they
        are installed, because "scipy is missing" and "nobody wrote this operator" must not look
        identical. What changes is that a run REFUSES BY NAME, before it touches a single well,
        with a sentence naming the package and the pip command. Declare a module here if your code
        imports it, INCLUDING a lazy import several calls deep — an undeclared lazy import is
        exactly the failure this argument exists to convert into a refusal, because per-well fault
        isolation would otherwise record it as one skip per well and finish the run reporting
        success with nothing written.

        This operator declares ``scipy`` because ``_stdev.py`` imports ``scipy.ndimage`` inside the
        loop. Note that it declares it even though ``smooth_sigma=0`` would never reach that
        import: ``requires`` describes what the entry needs AT ITS DECLARED DEFAULTS, and the
        default is 1.0. Under-declaring is the dangerous direction.
    """
    from squidxplorer import Param, add_projector      # INSIDE the function — see the note above

    add_projector(
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
