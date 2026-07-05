"""Tests for scalar params + position parsing across coordinates.csv schema versions."""

import json

from squidmip._acquisition import load_acquisition_params, load_positions


def test_load_params(tmp_path):
    (tmp_path / "acquisition parameters.json").write_text(
        json.dumps(
            {"Nz": 3, "Nt": 1, "dz(um)": 1.031, "objective": {"magnification": 20.0}, "sensor_pixel_size_um": 3.76}
        )
    )
    p = load_acquisition_params(tmp_path)
    assert p["n_z_declared"] == 3
    assert p["dz_um"] == 1.031
    assert p["pixel_size_um"] == 3.76 / 20.0  # 0.188


def test_load_params_absent_is_all_none(tmp_path):
    p = load_acquisition_params(tmp_path)
    assert p["n_z_declared"] is None
    assert p["pixel_size_um"] is None


def test_positions_legacy_schema_enumerates_fov(tmp_path):
    # legacy: no fov column -> fov is the per-region row index
    (tmp_path / "coordinates.csv").write_text(
        "region,x (mm),y (mm),z (mm)\nB2,1.0,2.0,\nB2,3.0,2.0,\nB3,1.0,5.0,\n"
    )
    pos = load_positions(tmp_path / "coordinates.csv")
    assert pos[("B2", 0)] == (1.0, 2.0)
    assert pos[("B2", 1)] == (3.0, 2.0)
    assert pos[("B3", 0)] == (1.0, 5.0)


def test_positions_current_schema_uses_fov_column(tmp_path):
    # current: explicit fov + z_level; first row per (region,fov) wins
    (tmp_path / "coordinates.csv").write_text(
        "region,fov,z_level,x (mm),y (mm),z (um),time\n"
        "B2,0,0,1.0,2.0,0.0,0\n"
        "B2,0,1,1.0,2.0,1.5,0\n"
        "B2,1,0,3.0,2.0,0.0,0\n"
    )
    pos = load_positions(tmp_path / "coordinates.csv")
    assert pos[("B2", 0)] == (1.0, 2.0)
    assert pos[("B2", 1)] == (3.0, 2.0)
    assert len(pos) == 2  # z rows collapsed


def test_positions_absent_file_is_empty(tmp_path):
    assert load_positions(tmp_path / "nope.csv") == {}
