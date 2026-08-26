"""The layering rules, enforced by reading the imports rather than by remembering them: napari is never imported at module scope, napari belongs only to"""

from __future__ import annotations

import ast
import pathlib

import pytest

import squidxplorer

PKG = pathlib.Path(squidxplorer.__file__).parent

#: The Qt bindings, by top-level module name. Membership is what "imports Qt" means below.
QT_BINDINGS = frozenset({"PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy"})

#: Every module allowed to import a Qt binding: the GUI layer, and nothing else.
GUI_MODULES = frozenset({
    "_fontscale",        # lives outside `_viewer` only to avoid a circular import
    "_brick_view",       # the in-window 3D renderer; owns a QThread reading bricks off the UI
                          # thread. The Qt-free half (geometry, stride, budget policy) is `_bricks`.
    "_gallery_window",   # Gallery View's window; its producer `_gallery` is Qt-free on purpose
    "_ingest",           # the acquisition-open pipeline, cut out of `_viewer` (2026-08-13)
    "_acqset_gui",       # the QThread over `_acqset.run_over_set`; `_acqset` itself is Qt-free
    "_worker_lifecycle", # the worker launch/stop seam: wires Signals, starts QThreads (2026-08-14)
    "_layer_tree",       # the grouped layer tree (a QTreeView over napari's item model)
    "_loupe",            # the LOUPE ENGINE: magnification arithmetic, sources, the coalescing
                          # QThread worker and the shared inset painter. Extracted from
                          # `_plate_overview` so the canvas loupe could not become a SECOND loupe.
    "_napari_loupe",     # the canvas loupe's GESTURE and floating inset (an overlay on the vispy
                          # canvas); everything about pixels is `_loupe`'s.
    "_fov_nav",          # the FOV axis: napari's dims slider walking a region's fields with the
                          # CAMERA. Where a FOV is lives in `_mosaic_source.mosaic_fov_bboxes_um`,
                          # what framing one means in `_napari_view.camera_for_bbox_um` — Qt-free.
    "_logpanel",         # the log PANEL (the Qt widget). `_logpane`, no 'l', is the Qt-free bus.
    "_napari_pane",
    "_op_panels",
    "_param_panel",      # built from an operator's `params` declaration; the declaration itself
                          # (`_engine.Param`) stays Qt-free so the CLI can read it headless.
    "_plate_overview",   # the plate navigator and its geometry
    "_qt_tabs",
    "_qtstyle",
    "_region_nav",
    "_region_viewer",
    "_time_point",       # the timepoint bar, shared by the plate and every window
    "_view_deck",        # the TAB DECK that holds view windows as pages. Qt-facing with nothing
    #                      underneath it to split off: it is a QMainWindow wearing a QTabWidget,
    #                      and every line is reparenting, tab indices and window activation. What
    #                      would have been its Qt-free half — which view is current, and what that
    #                      means for who draws — already lives in `RegionViewer` and `ViewerManager`
    #                      rather than being copied here.
    "_viewer",
    "_workers",          # the plate window's background threads
})


def _is_napari_module(stem: str) -> bool:
    return stem.startswith("_napari")


#: The two modules that import napari without being named for it — both PORT napari's own widget
#: classes rather than use its public API, so they cannot be expressed through a `_napari_*` facade
#: without the facade becoming a copy of napari's private layout. Both imports are inside function
#: bodies, so neither makes napari a load-time dependency; delete an entry if it stops applying,
#: and do not add a third without saying why here.
NAPARI_EXCEPTIONS = frozenset({"_layer_tree", "_region_nav"})

#: The modules lifted out of ``_viewer.py``, and what each may lift out of the others. `_viewer` is
#: absent from every value on purpose: that absence is the property under test.
CUT_OUT_OF_VIEWER = frozenset({
    "_ingest", "_operations", "_plate_overview", "_qt_tabs", "_qtstyle", "_run", "_workers",
})


def _modules() -> list[pathlib.Path]:
    """Every source file in the package, including subpackages."""
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path: pathlib.Path) -> list[tuple[str, bool]]:
    """Every module name *path* imports, as (dotted name, at_module_scope)."""
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
    """A module-scope ``import napari`` anywhere makes the headless CLI depend on the GUI stack."""
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
    """Everything outside GUI_MODULES is Qt-free, deferred imports included."""
    if path.stem in GUI_MODULES:
        return
    offenders = sorted({name for name, _ in _imports(path) if _top(name) in QT_BINDINGS})
    assert not offenders, (
        f"{path.name} imports Qt ({offenders}) and is not in GUI_MODULES. The pipeline installs "
        f"without PyQt5; a Qt import here breaks that. If it really is a GUI module, add "
        f"{path.stem!r} to GUI_MODULES in this file and say so in the commit."
    )


# `test_only_the_gui_layer_imports_qt` and `test_napari_belongs_to_the_napari_modules` RETURN for
# every name in these allowlists, so nothing checks the allowlists themselves; the tests below do.

@pytest.mark.parametrize("stem", sorted(GUI_MODULES))
def test_every_gui_exemption_is_still_earned(stem):
    """A name in GUI_MODULES must exist, and must still import Qt. Otherwise delete the entry."""
    path = PKG / f"{stem}.py"
    assert path.is_file(), (
        f"GUI_MODULES names {stem!r} and squidxplorer/{stem}.py does not exist. The exemption outlived "
        f"the module: delete the entry.")
    imports = sorted({name for name, _ in _imports(path) if _top(name) in QT_BINDINGS})
    assert imports, (
        f"squidxplorer/{stem}.py is exempted from the Qt boundary and imports no Qt at all any more. "
        f"It is Qt-free now: remove {stem!r} from GUI_MODULES so the boundary starts covering it.")


@pytest.mark.parametrize("stem", sorted(NAPARI_EXCEPTIONS))
def test_every_napari_exemption_is_still_earned(stem):
    """The same rule for the napari list, which is two names and so the easier one to forget."""
    path = PKG / f"{stem}.py"
    assert path.is_file(), (
        f"NAPARI_EXCEPTIONS names {stem!r} and squidxplorer/{stem}.py does not exist.")
    imports = sorted({name for name, _ in _imports(path) if _top(name) == "napari"})
    assert imports, (
        f"squidxplorer/{stem}.py is on NAPARI_EXCEPTIONS and imports no napari any more. Remove "
        f"{stem!r} from the list so the `_napari_*` rule starts covering it.")


@pytest.mark.parametrize("stem", sorted(CUT_OUT_OF_VIEWER))
def test_what_was_cut_out_of_the_viewer_does_not_import_it_back(stem):
    """No back-edge to ``_viewer``, at module scope or deferred — `_viewer` re-exports every one of these names, so a back-edge would be invisible at the"""
    path = PKG / f"{stem}.py"
    assert path.exists(), f"{stem} is gone; update CUT_OUT_OF_VIEWER"
    offenders = sorted({name for name, _ in _imports(path) if "_viewer" in name})
    assert not offenders, (
        f"{stem} imports {offenders}. It was cut OUT of _viewer; importing it back restores the "
        f"cycle. Move the shared thing down into a module both can depend on."
    )


