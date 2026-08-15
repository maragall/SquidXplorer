"""Headless tests for scripts/installer, loaded by path because scripts/ is not a package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_INSTALLER = Path(__file__).resolve().parents[1] / "scripts" / "installer"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"installer_{name}", _INSTALLER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod    # @dataclass resolves types via sys.modules
    spec.loader.exec_module(mod)
    return mod


menu = _load("menu")
bootstrap = _load("bootstrap")


def _probe_ok():
    return True, ""


def _row(rows, extra):
    return next(r for r in rows if r.extra == extra)


# the menu derives from the registry

def test_the_menu_groups_operators_by_their_declared_extra():
    rows = menu.build_menu(probe=_probe_ok)
    assert rows[0].extra == "core"
    assert "mip" in _row(rows, "core").operators
    stitch = _row(rows, "stitch")
    assert {"stitch", "coordinate"} <= set(stitch.operators)
    assert "tilefusion" in stitch.requires
    assert "cellpose" in _row(rows, "segment").operators


def test_an_operator_registered_with_an_extra_appears_in_its_own_row():
    from squidxplorer._engine import _OPERATORS, add_operator

    add_operator("_menu_test_op", lambda planes: next(iter(planes)),
                 requires=("nosuchpkg",), extra="video")
    try:
        row = _row(menu.build_menu(probe=_probe_ok), "video")
        assert row.operators == ("_menu_test_op",)
        assert row.requires == ("nosuchpkg",)
    finally:
        del _OPERATORS["_menu_test_op"]


def test_defaults_match_the_plan_decon_checked_segment_unchecked():
    rows = menu.build_menu(probe=_probe_ok)
    for extra in ("core", "stitch", "decon"):
        assert _row(rows, extra).checked, f"{extra} should start checked"
    assert not _row(rows, "segment").checked


# the CUDA probe

def test_a_failed_cuda_probe_shades_decon_with_the_probe_reason():
    rows = menu.build_menu(probe=lambda: (False, "no CUDA 12 driver on this Mac"))
    decon = _row(rows, "decon")
    assert not decon.enabled
    assert not decon.checked
    assert decon.reason == "no CUDA 12 driver on this Mac"
    assert "no CUDA 12 driver on this Mac" in menu.render(rows)


def test_the_probe_names_the_missing_driver(monkeypatch):
    monkeypatch.setattr(menu.shutil, "which", lambda name: None)
    ok, why = menu.cuda12_available()
    assert not ok
    assert "nvidia-smi" in why


# the bootstrapper

def test_dry_run_emits_the_uv_commands_for_the_chosen_extras(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/opt/uv/uv")
    rc = bootstrap.main(["--extras", "stitch,decon", "--source", "dist/squidxplorer.whl",
                         "--env", str(tmp_path / "env"), "--dry-run"])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"/opt/uv/uv venv {tmp_path / 'env'}"
    assert lines[1].startswith("/opt/uv/uv pip install --python ")
    assert "squidxplorer[decon,stitch] @ dist/squidxplorer.whl" in lines[1]


def test_a_rerun_reuses_the_existing_env(tmp_path):
    env = tmp_path / "env"
    py = bootstrap.env_python(env)
    py.parent.mkdir(parents=True)
    py.touch()
    cmds = bootstrap.commands("/opt/uv/uv", ["segment"], "x.whl", env)
    assert [c[1] for c in cmds] == ["pip"], "an existing env must not be recreated"


def test_core_only_installs_the_bare_package(tmp_path):
    cmds = bootstrap.commands("/opt/uv/uv", ["core"], "x.whl", tmp_path / "env")
    assert cmds[-1][-1] == "squidxplorer @ x.whl"


def test_without_uv_the_refusal_carries_the_install_hint(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: None)
    rc = bootstrap.main(["--source", "x.whl", "--env", str(tmp_path / "env"), "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "uv not found" in err
    assert "astral.sh" in err
