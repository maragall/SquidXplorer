"""Headless tests for scripts/installer, loaded by path because scripts/ is not a package."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_INSTALLER = Path(__file__).resolve().parents[1] / "scripts" / "installer"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"installer_{name}", _INSTALLER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod    # @dataclass resolves types via sys.modules
    spec.loader.exec_module(mod)
    return mod


bootstrap = _load("bootstrap")
sys.modules["bootstrap"] = bootstrap    # menu.py does `from bootstrap import ...`
menu = _load("menu")


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
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: None)
    ok, why = menu.cuda12_available()
    assert not ok
    assert "nvidia-smi" in why


# click-path defaults: double-clicked with no flags, every argument resolves itself

def test_the_default_source_is_the_wheel_beside_the_program(tmp_path):
    assert bootstrap.default_source(tmp_path) is None
    (tmp_path / "squidxplorer-0.1.0-py3-none-any.whl").touch()
    assert bootstrap.default_source(tmp_path).name == "squidxplorer-0.1.0-py3-none-any.whl"


def test_the_default_env_lives_under_the_users_app_data():
    env = bootstrap.default_env()
    assert "squidxplorer" in str(env).lower()
    assert str(env).startswith(str(Path.home()))


def test_default_extras_gate_decon_on_the_probe():
    assert bootstrap.default_extras(probe=lambda: (True, ""))[0] == ["gui", "stitch", "decon"]
    extras, note = bootstrap.default_extras(probe=lambda: (False, "no CUDA 12 here"))
    assert extras == ["gui", "stitch"], "a desktop install must always carry the gui extra"
    assert "no CUDA 12 here" in note


def test_without_a_wheel_or_source_the_refusal_says_so(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bootstrap, "default_source", lambda beside=None: None)
    rc = bootstrap.main(["--env", str(tmp_path / "env"), "--dry-run"])
    assert rc == 2
    assert "--source" in capsys.readouterr().err


def test_defaulted_flags_still_dry_run_the_uv_commands(tmp_path, capsys, monkeypatch):
    wheel = tmp_path / "squidxplorer-0.1.0-py3-none-any.whl"
    wheel.touch()
    monkeypatch.setattr(bootstrap, "program_dir", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "default_extras", lambda: (["stitch"], ""))
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/opt/uv/uv")
    rc = bootstrap.main(["--env", str(tmp_path / "env"), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "squidxplorer[stitch] @ " in out and wheel.name not in out.splitlines()[0]


# the bootstrapper

def test_dry_run_emits_the_uv_commands_for_the_chosen_extras(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/opt/uv/uv")
    rc = bootstrap.main(["--extras", "stitch,decon", "--source", "dist/squidxplorer.whl",
                         "--env", str(tmp_path / "env"), "--dry-run"])
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("/opt/uv/uv")]
    assert lines[0] == f"/opt/uv/uv venv --python {bootstrap.ENV_PYTHON} {tmp_path / 'env'}"
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


def test_the_env_is_built_from_a_PINNED_managed_python_never_the_machines(tmp_path):
    """The first customer install (Katana rig, 2026-08-16) died because `uv venv` built the env
    from Ubuntu 22.04's system Python 3.10 and the wheel needs >=3.11. The pin makes uv download
    a managed CPython, so the installer works on a machine with no (or the wrong) Python."""
    cmds = bootstrap.commands("/opt/uv/uv", ["core"], "x.whl", tmp_path / "env")
    venv = cmds[0]
    assert venv[1] == "venv"
    assert venv[2:4] == ["--python", bootstrap.ENV_PYTHON], \
        "the venv step no longer pins its interpreter; the system python decides again"


def _fake_env(tmp_path, version_line: str):
    """An env whose `bin/python` answers *version_line* to the stale_env probe (posix shim)."""
    env = tmp_path / "env"
    py = bootstrap.env_python(env)
    py.parent.mkdir(parents=True)
    py.write_text(f"#!/bin/sh\necho {version_line}\n")
    py.chmod(0o755)
    return env


@pytest.mark.skipif(os.name != "posix", reason="the shim interpreter is a shell script")
def test_a_stale_old_python_env_is_named_not_reused(tmp_path):
    """The venv step is skipped whenever the env exists, so an env built wrong once would fail
    every future install identically — the guard names it so _run can recreate it."""
    reason = bootstrap.stale_env(_fake_env(tmp_path, "3.10"))
    assert reason is not None and "3.10" in reason and bootstrap.ENV_PYTHON in reason


@pytest.mark.skipif(os.name != "posix", reason="the shim interpreter is a shell script")
def test_a_matching_env_is_reused(tmp_path):
    assert bootstrap.stale_env(_fake_env(tmp_path, bootstrap.ENV_PYTHON)) is None


@pytest.mark.skipif(os.name != "posix", reason="the shim interpreter is a shell script")
def test_a_NEWER_python_env_is_also_recreated(tmp_path):
    """Exact-minor match, not a floor: a 3.12 env satisfies requires-python, but psfmodels has
    no cp312 wheel, so adding the decon pack to it later would source-build on a machine with no
    compiler — the interpreter decides which wheels exist, so one installer means one
    interpreter."""
    reason = bootstrap.stale_env(_fake_env(tmp_path, "3.12"))
    assert reason is not None and "3.12" in reason and bootstrap.ENV_PYTHON in reason


def test_an_absent_env_is_not_stale(tmp_path):
    assert bootstrap.stale_env(tmp_path / "no-such-env") is None


def test_no_dependency_needs_a_git_binary_on_the_customer_machine():
    """`git+https` references make uv shell out to a `git` binary — present on every CI runner,
    absent on a bare lab machine. The SHA pins live in archive-tarball URLs instead, which
    GitHub serves over plain HTTPS."""
    import pathlib

    pyproject = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    offenders = [ln for ln in pyproject.splitlines()
                 if "@ git+" in ln and not ln.lstrip().startswith("#")]
    assert not offenders, \
        f"git+ dependencies reappeared ({offenders}); the frozen installer will fail on " \
        f"machines without git"


def test_the_linux_gui_libs_note_is_linux_only():
    assert bootstrap._linux_gui_libs_note("darwin") is None
    assert bootstrap._linux_gui_libs_note("win32") is None


def test_without_uv_the_refusal_carries_the_install_hint(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: None)
    rc = bootstrap.main(["--source", "x.whl", "--env", str(tmp_path / "env"), "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "uv not found" in err
    assert "astral.sh" in err


# the one-file payload: the frozen binary carries its wheel and uv inside


def test_the_bundled_payload_wheel_beats_the_one_beside_the_program(tmp_path, monkeypatch):
    payload = tmp_path / "unpacked" / "payload"
    payload.mkdir(parents=True)
    (payload / "squidxplorer-0.2.0-py3-none-any.whl").touch()
    beside = tmp_path / "beside"
    beside.mkdir()
    (beside / "squidxplorer-0.1.0-py3-none-any.whl").touch()
    monkeypatch.setattr(bootstrap.sys, "_MEIPASS", str(tmp_path / "unpacked"), raising=False)
    monkeypatch.setattr(bootstrap, "program_dir", lambda: beside)
    assert bootstrap.default_source().name == "squidxplorer-0.2.0-py3-none-any.whl"


def test_without_a_payload_the_wheel_beside_the_program_still_serves(tmp_path, monkeypatch):
    (tmp_path / "squidxplorer-0.1.0-py3-none-any.whl").touch()
    monkeypatch.setattr(bootstrap, "program_dir", lambda: tmp_path)
    assert bootstrap.default_source().name == "squidxplorer-0.1.0-py3-none-any.whl"


def test_find_uv_prefers_the_bundled_uv_over_path(tmp_path, monkeypatch):
    payload = tmp_path / "unpacked" / "payload"
    payload.mkdir(parents=True)
    name = "uv.exe" if sys.platform == "win32" else "uv"
    (payload / name).touch()
    monkeypatch.setattr(bootstrap.sys, "_MEIPASS", str(tmp_path / "unpacked"), raising=False)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda n: "/opt/uv/uv")
    found = bootstrap.find_uv()
    assert found == str(payload / name)
    if os.name == "posix":
        assert os.access(found, os.X_OK), "onefile extraction may drop the mode; find_uv restores it"


def test_find_uv_falls_back_to_path(monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda n: "/opt/uv/uv")
    assert bootstrap.find_uv() == "/opt/uv/uv"


def test_double_click_is_interactive_a_piped_run_is_not(monkeypatch):
    class _Tty:
        def isatty(self):
            return True

    class _Pipe:
        def isatty(self):
            return False

    monkeypatch.setattr(bootstrap.sys, "argv", ["SquidXplorer-Setup"])
    monkeypatch.setattr(bootstrap.sys, "stdin", _Tty())
    assert bootstrap._interactive(None)
    monkeypatch.setattr(bootstrap.sys, "stdin", _Pipe())
    assert not bootstrap._interactive(None), "a CI or piped run must never block on Enter"
    monkeypatch.setattr(bootstrap.sys, "stdin", _Tty())
    assert not bootstrap._interactive(["--dry-run"]), "explicit argv is a scripted call"


# launchers: what a finished install leaves behind, per platform


def test_the_macos_app_wraps_the_envs_viewer(tmp_path):
    env = tmp_path / "env"
    made = bootstrap._macos_app(env, apps_dir=tmp_path / "Applications")
    app = tmp_path / "Applications" / "SquidXplorer.app"
    assert made == f"app created: {app}"
    runner = app / "Contents" / "MacOS" / "SquidXplorer"
    assert str(env / "bin" / "squidxplorer-view") in runner.read_text()
    if os.name == "posix":
        assert runner.stat().st_mode & 0o111, "the bundle's executable must be executable"
    plist = (app / "Contents" / "Info.plist").read_text()
    assert "SquidXplorer" in plist and "com.cephla.squidxplorer" in plist


def test_a_macos_app_rerun_repoints_at_the_env(tmp_path):
    env = tmp_path / "env"
    assert bootstrap._macos_app(env, apps_dir=tmp_path / "Applications")
    assert bootstrap._macos_app(env, apps_dir=tmp_path / "Applications"), \
        "a rerun over the existing bundle must not fail"


def test_the_linux_menu_entry_points_at_the_envs_viewer(tmp_path):
    env = tmp_path / "env"
    made = bootstrap._linux_desktop_entry(env, apps_dir=tmp_path / "applications")
    entry = tmp_path / "applications" / "squidxplorer.desktop"
    assert made == f"menu entry created: {entry}"
    text = entry.read_text()
    assert f"Exec={env / 'bin' / 'squidxplorer-view'} %f" in text, \
        "the %f is the drag-and-open half: a dropped folder must arrive as sys.argv[1]"
    assert "Terminal=false" in text
    assert "[Desktop Entry]" in text
    if os.name == "posix":
        assert entry.stat().st_mode & 0o111


def test_the_linux_menu_entry_carries_the_installed_icon(tmp_path, monkeypatch):
    """The icon is COPIED beside the env: a one-file build's payload dir vanishes on exit, so an
    Icon= pointing there would lose its art on the next login."""
    env = tmp_path / "env"
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "squidxplorer.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(bootstrap, "payload_dirs", lambda: [payload])
    bootstrap._linux_desktop_entry(env, apps_dir=tmp_path / "applications")
    text = (tmp_path / "applications" / "squidxplorer.desktop").read_text()
    assert f"Icon={env.parent / 'squidxplorer.png'}" in text
    assert (env.parent / "squidxplorer.png").exists(), "the icon must outlive the payload dir"


def test_a_missing_icon_still_makes_a_launcher(tmp_path, monkeypatch):
    """The icon is cosmetic; an empty payload must not cost the user the launcher itself."""
    monkeypatch.setattr(bootstrap, "payload_dirs", lambda: [tmp_path / "nowhere"])
    made = bootstrap._linux_desktop_entry(tmp_path / "env", apps_dir=tmp_path / "applications")
    assert made is not None
    assert "Icon=" not in (tmp_path / "applications" / "squidxplorer.desktop").read_text()


def test_a_failed_launcher_is_said_and_never_fatal(tmp_path, capsys):
    blocker = tmp_path / "not-a-dir"
    blocker.touch()
    made = bootstrap._linux_desktop_entry(tmp_path / "env", apps_dir=blocker / "applications")
    assert made is None
    assert "squidxplorer-view" in capsys.readouterr().err, \
        "the failure must name the viewer to launch directly"


def test_create_launcher_dispatches_on_the_platform(monkeypatch):
    monkeypatch.setattr(bootstrap, "_macos_app", lambda env: f"mac:{env}")
    monkeypatch.setattr(bootstrap, "_linux_desktop_entry", lambda env: f"linux:{env}")
    monkeypatch.setattr(bootstrap, "_windows_shortcut", lambda env: f"win:{env}")
    assert bootstrap.create_launcher(Path("e"), platform="darwin") == "mac:e"
    assert bootstrap.create_launcher(Path("e"), platform="linux") == "linux:e"
    assert bootstrap.create_launcher(Path("e"), platform="win32") == "win:e"
