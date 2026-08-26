"""The functions-over-the-window interface is DECLARED, and the declaration is enforced."""

from __future__ import annotations

import ast
import pathlib

import pytest

import squidxplorer
from squidxplorer._window_contract import WINDOW_CONTRACTS

PKG = pathlib.Path(squidxplorer.__file__).parent


def _win_accesses(path: pathlib.Path) -> tuple[set, set]:
    """(called, uncalled): every attribute the module reaches on a name ``win``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called: set = set()
    uncalled: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            v = node.func.value
            if isinstance(v, ast.Name) and v.id == "win":
                called.add(node.func.attr)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "win" and node.attr not in called:
            uncalled.add(node.attr)
    return called, uncalled


def _declared(proto: type) -> tuple[set, set]:
    """(methods, attributes) a Protocol declares, dunders excluded."""
    methods = {n for n, v in vars(proto).items()
               if callable(v) and not n.startswith("__")}
    attributes = set(getattr(proto, "__annotations__", {}))
    return methods, attributes


@pytest.mark.parametrize("stem", sorted(WINDOW_CONTRACTS))
def test_every_window_access_is_declared(stem):
    """An attribute a helper reaches on the window must be in its Protocol, by name."""
    called, uncalled = _win_accesses(PKG / f"{stem}.py")
    methods, attributes = _declared(WINDOW_CONTRACTS[stem])
    missing = sorted((called | uncalled) - (methods | attributes))
    assert not missing, (
        f"{stem}.py reaches win.{missing} and {WINDOW_CONTRACTS[stem].__name__} does not "
        f"declare them. Add each to the Protocol in _window_contract.py: the point of the "
        f"contract is that this interface never grows silently."
    )


@pytest.mark.parametrize("stem", sorted(WINDOW_CONTRACTS))
def test_no_declared_name_is_stale(stem):
    """A Protocol entry nobody accesses is a contract for nothing: delete it with the access."""
    called, uncalled = _win_accesses(PKG / f"{stem}.py")
    methods, attributes = _declared(WINDOW_CONTRACTS[stem])
    stale = sorted((methods | attributes) - (called | uncalled))
    assert not stale, (
        f"{WINDOW_CONTRACTS[stem].__name__} declares {stale} and {stem}.py no longer reaches "
        f"them on the window. Remove the entries so the Protocol stays the real interface."
    )


@pytest.mark.parametrize("stem", sorted(WINDOW_CONTRACTS))
def test_a_called_name_is_declared_as_a_method(stem):
    """``win.attr(...)`` is a method of the window; declaring it as bare data hides that."""
    called, _ = _win_accesses(PKG / f"{stem}.py")
    methods, _attributes = _declared(WINDOW_CONTRACTS[stem])
    demoted = sorted(called - methods)
    assert not demoted, (
        f"{stem}.py calls win.{demoted} but {WINDOW_CONTRACTS[stem].__name__} declares them "
        f"as attributes. Declare each as a method so the contract says what it is."
    )


def test_the_contract_module_is_typing_only():
    """``_window_contract`` imports typing and nothing heavier: it must never cost a Qt import."""
    tree = ast.parse((PKG / "_window_contract.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    allowed = {"__future__", "typing"}
    assert imported <= allowed, (
        f"_window_contract.py imports {sorted(imported - allowed)}; it is a declaration, "
        f"not a runtime dependency."
    )
