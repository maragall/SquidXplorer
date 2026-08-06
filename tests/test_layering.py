"""The layering rules, enforced by reading the imports rather than by remembering them.

Gap 6 of the GUI backlog plan (2026-07-29), step 4 of the split of ``squidmip/_viewer.py``.

WHY THIS FILE EXISTS
--------------------
The discipline these tests pin already holds. That is the problem. It holds INFORMALLY: it is
written down in three module docstrings and in ``_recipe.py``, it is true today, and nothing
anywhere would notice the day it stopped being true. ``_viewer.py`` is the proof of what that
costs. It went from 2,081 lines to 8,388 in 199 commits, one reasonable-looking addition at a
time, and the bill arrived as a whole abandoned Qt6 migration whose payload could not be rebased
past it.

An import is the cheapest thing in the world to add and the most expensive thing to remove, because
by the time you want it gone there are call sites depending on what it dragged in. So these tests
read the AST of every module in the package and check four properties of the import graph. None of
them is a style preference; each one is load-bearing:

1. **napari is never imported at module scope.** This is why ``pip install squidmip`` without the
   ``[gui]`` extra gives a working headless CLI. napari costs seconds to import and pulls in a Qt
   binding; if any module reached for it on import, the pipeline would depend on the GUI stack.
2. **napari appears only in the modules whose job is napari.** The rule the plan asks for. It has
   two named exceptions, below, and naming them is the point: a debt you can see is a debt.
3. **Qt is imported only by the declared GUI modules.** The domain layer (reader, placement,
   engine, operators, fusion, output, cache) is genuinely Qt-free today. A test is what keeps it
   that way when someone needs "just a QTimer" in the engine.
4. **The modules cut out of ``_viewer.py`` do not import it back.** A re-export makes a back-edge
   invisible: ``from squidmip._viewer import X`` inside a function body would compile, pass every
   test, and quietly restore the cycle the split existed to remove.

These are checked with ``ast``, not by importing the modules, so the test costs milliseconds and
works on a machine with no Qt installed at all.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import squidmip

PKG = pathlib.Path(squidmip.__file__).parent

#: The Qt bindings, by top-level module name. Membership is what "imports Qt" means below.
QT_BINDINGS = frozenset({"PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy"})

#: Every module allowed to import a Qt binding: the GUI layer, and nothing else. This list IS the
#: boundary. Adding a module here is a deliberate statement that it is part of the desktop app and
#: can never be imported by the headless pipeline; that is a decision, so it should cost an edit
#: here and a line in a commit message rather than happening as a side effect of one import.
GUI_MODULES = frozenset({
    "_fontscale",        # type that grows with its window. Qt-facing by nature (it rewrites
    #                      widget stylesheets), and shared so CHILD windows get it too: it lives
    #                      outside `_viewer` only to avoid a circular import, not to escape this
    #                      boundary.
    "_brick_view",       # the IN-WINDOW 3D renderer. Qt-facing on purpose: it owns a QThread that
    #                      reads bricks off the UI thread, and it drives the pane's live napari
    #                      canvas. The Qt-free half of bricking is `_bricks` (geometry, stride and
    #                      budget policy) and it stays below this boundary so the decisions that
    #                      cannot be probed without a GL context are still testable headless.
    "_gallery_window",   # Gallery View's WINDOW: the grid of QLabels, its worker, its controls.
    #                      Its producer is `_gallery`, which is Qt-free and stays on the other side
    #                      of this boundary — that split is the point, not an accident: the scope,
    #                      the crop, the fuse and the contrast are all testable without a window,
    #                      and the window holds nothing but layout.
    "_layer_tree",       # the grouped layer tree (a QTreeView over napari's item model)
    "_logpanel",         # the log PANEL (the Qt widget). `_logpane`, no 'l', is the Qt-free bus.
    "_napari_pane",
    "_op_panels",
    "_param_panel",      # the panel BUILT FROM an operator's `params` declaration. Qt-facing by
    #                      nature: it is one widget per declared Param. The declaration itself
    #                      (`_engine.Param`) stays Qt-free, which is what lets the CLI, the recipe
    #                      writer and the composer read the same record headless.
    "_plate_overview",   # gap 6 step 1: the plate navigator and its geometry
    "_qt_tabs",
    "_qtstyle",
    "_region_nav",
    "_region_viewer",
    "_slide_art",
    "_terminal",
    "_time_point",       # Task 4b: the timepoint bar. A QWidget shared by the plate and every
    #                      window, so one definition rather than two that can drift about which
    #                      timepoint you are looking at.
    "_viewer",
    "_workers",          # gap 6 step 2: the plate window's eight background threads
})

#: Modules whose job is napari. The rule is that napari is theirs alone.
def _is_napari_module(stem: str) -> bool:
    return stem.startswith("_napari")


#: The two modules that import napari and are not named for it. Both are PORTS of napari's own
#: widget classes rather than users of the napari public API, which is why they cannot be expressed
#: through a ``_napari_*`` facade without the facade becoming a copy of napari's private layout:
#:
#:   _layer_tree   napari._qt.containers item roles, LayerDelegate, and napari.qt stylesheets.
#:                 It rebuilds napari's own layer list as a tree; the whole point is that the
#:                 rendering is napari's, not ours.
#:   _region_nav   napari.components.Dims and napari._qt.widgets.qt_dims.QtDims, i.e. the real
#:                 napari dims slider, reused rather than reimplemented.
#:
#: Every one of those imports is already inside a function body, so neither module makes napari a
#: load-time dependency. They are listed here so the exception is VISIBLE and dated rather than
#: discovered later as "the rule was never true". If either is refactored behind a `_napari_*`
#: module, delete its entry; do not add a third without saying why here.
NAPARI_EXCEPTIONS = frozenset({"_layer_tree", "_region_nav"})

#: The modules lifted out of ``_viewer.py``, and what each may lift out of the others. `_viewer` is
#: absent from every value on purpose: that absence is the property under test.
#: (`_qtstyle`, `_terminal` and `_qt_tabs` came out at e1d6018; `_plate_overview`, `_workers` and
#: `_operations` at gap 6. The rule is the same for all six.)
CUT_OUT_OF_VIEWER = frozenset({
    "_operations", "_plate_overview", "_qt_tabs", "_qtstyle", "_terminal", "_workers",
})


def _modules() -> list[pathlib.Path]:
    """Every source file in the package, including subpackages."""
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path: pathlib.Path) -> list[tuple[str, bool]]:
    """Every module name *path* imports, as (dotted name, at_module_scope).

    ``at_module_scope`` is False for an import inside a function, method or class body -- a
    deferred import, which costs nothing until the code runs. The distinction is the whole reason
    napari can be a hard dependency of the app and no dependency at all of the pipeline.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    deferred: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for inner in ast.walk(node):
                deferred.add(id(inner))
    out: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        at_module_scope = id(node) not in deferred
        if isinstance(node, ast.Import):
            out.extend((alias.name, at_module_scope) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.module, at_module_scope))
    return out


def _top(dotted: str) -> str:
    return dotted.split(".")[0]


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_napari_is_never_imported_at_module_scope(path):
    """A module-scope ``import napari`` anywhere makes the headless CLI depend on the GUI stack.

    This holds for the ``_napari_*`` modules too, which is not an accident: ``_napari3d`` says so
    in a comment at its own import ("lazy: heavy import, and a machine without napari still runs
    the 2D app"). The rule is about WHEN, and it has no exceptions.
    """
    offenders = [name for name, at_module_scope in _imports(path)
                 if at_module_scope and _top(name) == "napari"]
    assert not offenders, (
        f"{path.name} imports napari at module scope ({offenders}). Move it inside the function "
        f"that needs it: napari is a [gui] extra, and the headless pipeline must import without it."
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_napari_belongs_to_the_napari_modules(path):
    """napari is imported by ``_napari_*.py`` and by the two named ports, and by nothing else."""
    if _is_napari_module(path.stem) or path.stem in NAPARI_EXCEPTIONS:
        return
    offenders = sorted({name for name, _ in _imports(path) if _top(name) == "napari"})
    assert not offenders, (
        f"{path.name} imports napari ({offenders}) and is not a `_napari_*` module. Put the napari "
        f"call behind one of those, or add {path.stem!r} to NAPARI_EXCEPTIONS in this file WITH the "
        f"reason it cannot be. The list is the visible cost of the exception."
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_only_the_gui_layer_imports_qt(path):
    """Everything outside GUI_MODULES is Qt-free, deferred imports included.

    ``squidmip/_operations.py`` (gap 6 step 3) is the newest member of the Qt-free side and the
    reason this test was written when it was: it is the operator registry, one dataclass and four
    pure functions, and it spent its life inside a QMainWindow module where nothing could tell.
    """
    if path.stem in GUI_MODULES:
        return
    offenders = sorted({name for name, _ in _imports(path) if _top(name) in QT_BINDINGS})
    assert not offenders, (
        f"{path.name} imports Qt ({offenders}) and is not in GUI_MODULES. The pipeline installs "
        f"without PyQt5; a Qt import here breaks that. If it really is a GUI module, add "
        f"{path.stem!r} to GUI_MODULES in this file and say so in the commit."
    )


@pytest.mark.parametrize("stem", sorted(CUT_OUT_OF_VIEWER))
def test_what_was_cut_out_of_the_viewer_does_not_import_it_back(stem):
    """No back-edge to ``_viewer``, at module scope or deferred.

    ``_viewer`` re-exports every one of these names, so a back-edge is invisible at the call site:
    ``from squidmip._viewer import _fit_cell`` inside a method body would work today and would put
    the cycle straight back. This is the one test that actually keeps the split split.
    """
    path = PKG / f"{stem}.py"
    assert path.exists(), f"{stem} is gone; update CUT_OUT_OF_VIEWER"
    offenders = sorted({name for name, _ in _imports(path) if "_viewer" in name})
    assert not offenders, (
        f"{stem} imports {offenders}. It was cut OUT of _viewer; importing it back restores the "
        f"cycle. Move the shared thing down into a module both can depend on."
    )


def test_the_gui_module_list_has_no_stale_entries():
    """GUI_MODULES, NAPARI_EXCEPTIONS and CUT_OUT_OF_VIEWER name modules that exist.

    A list of names is only a boundary while the names are real. Left unchecked, a rename turns an
    entry into a permanent silent hole in whichever rule it belonged to.
    """
    live = {p.stem for p in _modules()}
    for name, declared in (("GUI_MODULES", GUI_MODULES),
                           ("NAPARI_EXCEPTIONS", NAPARI_EXCEPTIONS),
                           ("CUT_OUT_OF_VIEWER", CUT_OUT_OF_VIEWER)):
        missing = sorted(declared - live)
        assert not missing, f"{name} names modules that no longer exist: {missing}"


def test_every_gui_module_actually_imports_qt():
    """GUI_MODULES is an ALLOWLIST, so an entry that needs no Qt is an allowance nobody is using.

    Kept honest in both directions on purpose: the list has to shrink when a widget module becomes
    Qt-free, or it slowly becomes "the modules we gave up on".
    """
    unused = []
    for stem in sorted(GUI_MODULES):
        path = PKG / f"{stem}.py"
        if not any(_top(name) in QT_BINDINGS for name, _ in _imports(path)):
            unused.append(stem)
    assert not unused, (
        f"these are listed in GUI_MODULES but import no Qt: {unused}. They are now part of the "
        f"Qt-free layer; take them off the list so the boundary keeps meaning something."
    )
