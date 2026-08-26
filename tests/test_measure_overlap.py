"""Unit tests for scripts/measure_overlap.py, loaded by path because scripts/ is not a package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "measure_overlap", Path(__file__).resolve().parents[1] / "scripts" / "measure_overlap.py"
)
mo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mo)


# _modal_step
def test_modal_step_finds_the_pitch_through_gaps_and_duplicates():
    assert mo._modal_step([0.0, 0.7056, 1.4112, 2.1168]) == pytest.approx(0.7056, abs=1e-4)
    assert mo._modal_step([0.0, 0.7056, 1.4112, 2.8224]) == pytest.approx(0.7056, abs=1e-4)
    assert mo._modal_step([0.0, 0.0, 0.7056, 0.7056, 1.4112]) == pytest.approx(0.7056, abs=1e-4)
    assert mo._modal_step([1.23]) == 0.0 and mo._modal_step([]) == 0.0


# _pixel_size_um
def test_json_sensor_pitch_is_divided_by_magnification(tmp_path):
    (tmp_path / "acquisition parameters.json").write_text(
        json.dumps({"sensor_pixel_size_um": 7.4571427, "objective": {"magnification": 20.0}})
    )
    px, src = mo._pixel_size_um(tmp_path)
    assert px == pytest.approx(0.37286, abs=1e-4), "must be object-space, not sensor pitch"
    assert "20x" in src, "the conversion must be visible in the reported source"


def test_yaml_pixel_size_is_taken_as_is(tmp_path):
    """acquisition.yaml already stores object-space; dividing again would be a second bug."""
    (tmp_path / "acquisition.yaml").write_text("objective:\n  pixel_size_um: 0.325\n")
    px, src = mo._pixel_size_um(tmp_path)
    assert px == pytest.approx(0.325)
    assert "object-space" in src
    (tmp_path / "acquisition parameters.json").write_text(
        json.dumps({"sensor_pixel_size_um": 7.45, "objective": {"magnification": 20.0}}))
    assert mo._pixel_size_um(tmp_path)[0] == pytest.approx(0.325), "yaml wins over legacy json"


def test_sensor_pitch_without_magnification_refuses_to_guess(tmp_path):
    """Returning the raw pitch here would silently be a 10-20x error. Report None instead."""
    (tmp_path / "acquisition parameters.json").write_text(
        json.dumps({"sensor_pixel_size_um": 7.45})
    )
    px, src = mo._pixel_size_um(tmp_path)
    assert px is None
    assert "cannot convert" in src
    assert mo._pixel_size_um(tmp_path / "empty") == (None, "not found")


# survey — end to end on a synthetic acquisition
def _write_acq(root: Path, *, with_original: bool, n=3, step=0.7056, region="C5", z_levels=1):
    root.mkdir(parents=True, exist_ok=True)
    (root / "acquisition.yaml").write_text("objective:\n  pixel_size_um: 0.3728571\n")
    rows = [(r, c) for r in range(n) for c in range(n)]
    with (root / "coordinates.csv").open("w", newline="") as fh:
        fh.write("region,x (mm),y (mm),z (mm)\n")
        for r, c in rows:
            fh.write(f"{region},{c * step},{r * step},\n")
    if with_original:
        d = root / "original_coordinates"
        d.mkdir(exist_ok=True)
        with (d / "original_coordinates_0.csv").open("w", newline="") as fh:
            fh.write("region,fov,z_level,x (mm),y (mm),z (um),time\n")
            for z in range(z_levels):
                for i, (r, c) in enumerate(rows):
                    fh.write(f"{region},{i},{z},{c * step},{r * step},0,t\n")


def test_survey_reports_grid_and_overlap(tmp_path, monkeypatch):
    root = tmp_path / "acq"
    _write_acq(root, with_original=True, n=3)
    monkeypatch.setattr(mo, "_frame_shape", lambda _r: (2084, 2084))

    out = mo.survey(root)
    assert out["has_original_coordinates"] is True
    assert out["has_explicit_fov_column"] is True
    e = out["regions"]["C5"]
    assert e["n_fov"] == 9 and e["grid"] == [3, 3] and e["rectangular"]
    assert e["overlap_frac"][0] == pytest.approx(0.092, abs=0.005)
    assert e["overlap_frac"][1] == pytest.approx(0.092, abs=0.005)


def test_survey_filters_multi_z_instead_of_miscounting(tmp_path, monkeypatch):
    root = tmp_path / "acq"
    _write_acq(root, with_original=True, n=3, z_levels=10)
    monkeypatch.setattr(mo, "_frame_shape", lambda _r: (2084, 2084))

    out = mo.survey(root)
    assert out["regions"]["C5"]["n_fov"] == 9, "must be 9 FOVs, not 90 rows"
    assert any("multi-z" in n for n in out["notes"])
    bare = tmp_path / "bare"
    _write_acq(bare, with_original=False)
    out = mo.survey(bare)
    assert out["has_original_coordinates"] is False
    assert any("NO original_coordinates" in n for n in out["notes"])


def test_survey_flags_declared_grid_disagreement(tmp_path, monkeypatch):
    root = tmp_path / "acq"
    _write_acq(root, with_original=True, n=3)
    (root / "acquisition parameters.json").write_text(json.dumps({"Nx": 1, "Ny": 1}))
    monkeypatch.setattr(mo, "_frame_shape", lambda _r: (2084, 2084))

    out = mo.survey(root)
    assert any("DO NOT derive grid geometry" in n for n in out["notes"])


def test_survey_tolerates_no_frame_and_still_reports_grid(tmp_path, monkeypatch):
    root = tmp_path / "acq"
    _write_acq(root, with_original=True, n=3)
    monkeypatch.setattr(mo, "_frame_shape", lambda _r: None)

    out = mo.survey(root)
    assert out["regions"]["C5"]["grid"] == [3, 3]
    assert out["regions"]["C5"]["overlap_frac"] is None
