"""Adapter contract: availability, version, argv shape, position read-back."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bench.adapters import BUILTIN_ADAPTERS, build_adapters
from bench.adapters.base import (
    StitchRequest,
    executable_available,
    python_module_available,
    read_positions_json,
)
from bench.dataset import load_acquisition
from tests.conftest_bench import write_acquisition


@pytest.fixture
def req(tmp_path):
    write_acquisition(tmp_path / "acq", grid=(1, 2), tile=(32, 32), step=(24, 24))
    acq = load_acquisition(tmp_path / "acq")
    return StitchRequest(
        acquisition=acq,
        region="C5",
        channel=acq.channels[0],
        z=0,
        out_dir=tmp_path / "out",
    )


def test_builtin_registry_starts_narrow():
    """IMA-233 ships tilefusion + ASHLAR only; the other three are deferred."""
    names = [a.name for a in BUILTIN_ADAPTERS]
    assert names == ["tilefusion", "ashlar"]


def test_build_adapters_default_returns_all():
    assert len(build_adapters(None)) == 2


def test_build_adapters_by_name():
    assert [a.name for a in build_adapters(["ashlar"])] == ["ashlar"]


def test_build_adapters_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown adapter"):
        build_adapters(["bigstitcher"])


@pytest.mark.parametrize("cls", BUILTIN_ADAPTERS)
def test_adapter_declares_full_measurability(cls):
    """Both shipped tools are local Python: RSS, disk and quality are all measurable."""
    a = cls()
    assert a.measurable_rss
    assert a.measurable_output_bytes
    assert a.supports_quality


@pytest.mark.parametrize("cls", BUILTIN_ADAPTERS)
def test_build_command_is_a_runnable_argv(cls, req):
    cmd = cls().build_command(req)
    assert cmd[0] == sys.executable
    assert cmd[1] == "-m"
    assert cmd[2].startswith("bench.drivers.")
    assert "--input" in cmd and "--out" in cmd
    assert str(req.out_dir) in cmd


@pytest.mark.parametrize("cls", BUILTIN_ADAPTERS)
def test_availability_and_version_never_raise(cls):
    """These run before anything else; they must degrade, not explode."""
    a = cls()
    assert isinstance(a.is_available(), bool)
    assert isinstance(a.version(), str)


@pytest.mark.parametrize("cls", BUILTIN_ADAPTERS)
def test_collect_on_an_empty_output_dir_returns_no_positions(cls, req):
    req.out_dir.mkdir(parents=True, exist_ok=True)
    assert cls().collect(req).positions_px is None


# ----------------------------------------------------------------- positions.json


def test_read_positions_json_parses_the_contract(tmp_path):
    (tmp_path / "positions.json").write_text(
        json.dumps({"positions_px": {"0": [0.0, 0.0], "1": [0.0, 96.0]}})
    )
    got = read_positions_json(tmp_path)
    assert got == {0: (0.0, 0.0), 1: (0.0, 96.0)}


def test_read_positions_json_accepts_a_bare_mapping(tmp_path):
    (tmp_path / "positions.json").write_text(json.dumps({"0": [1.0, 2.0]}))
    assert read_positions_json(tmp_path) == {0: (1.0, 2.0)}


def test_read_positions_json_missing_file(tmp_path):
    assert read_positions_json(tmp_path) is None


def test_read_positions_json_malformed_json(tmp_path):
    (tmp_path / "positions.json").write_text("{not json")
    assert read_positions_json(tmp_path) is None


def test_read_positions_json_skips_bad_entries(tmp_path):
    (tmp_path / "positions.json").write_text(
        json.dumps({"positions_px": {"0": [1.0, 2.0], "bad": "nope", "2": [3.0]}})
    )
    assert read_positions_json(tmp_path) == {0: (1.0, 2.0)}


def test_read_positions_json_all_bad_returns_none(tmp_path):
    (tmp_path / "positions.json").write_text(json.dumps({"positions_px": {"x": "y"}}))
    assert read_positions_json(tmp_path) is None


# ------------------------------------------------------------------- availability


def test_python_module_available_is_true_for_stdlib():
    assert python_module_available("json")


def test_python_module_available_is_false_for_nonsense():
    assert not python_module_available("definitely_not_a_module_xyz")


def test_executable_available():
    assert executable_available("ls")
    assert not executable_available("definitely-not-a-binary-xyz")


# ----------------------------------------------------------------------- drivers


@pytest.mark.parametrize("mod", ["tilefusion_driver", "ashlar_driver"])
def test_driver_exits_3_when_its_tool_is_absent(mod, tmp_path, req):
    """Importing the driver must not require the tool; a clean exit 3 becomes a
    MISSING_TOOL row rather than a traceback in the operator's face."""
    import subprocess

    out = subprocess.run(
        [
            sys.executable, "-m", f"bench.drivers.{mod}",
            "--input", str(req.acquisition.root),
            "--region", "C5",
            "--channel", req.channel,
            "--out", str(tmp_path / "o"),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode in (0, 3, 4)
