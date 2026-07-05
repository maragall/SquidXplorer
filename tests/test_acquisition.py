"""Tests for scalar acquisition metadata sourcing (acquisition.yaml primary, JSON fallback)."""

import json

from squidmip._acquisition import load_acquisition_metadata

_ACQ_YAML = """\
objective:
  pixel_size_um: 0.325
sample:
  wellplate_format: 24 well plate
z_stack:
  nz: 3
  delta_z_mm: 0.001031
time_series:
  nt: 2
"""

_JSON = {
    "Nz": 3,
    "Nt": 1,
    "dz(um)": 1.031,
    "objective": {"magnification": 20.0},
    "sensor_pixel_size_um": 3.76,
}


def test_prefers_acquisition_yaml(tmp_path):
    (tmp_path / "acquisition.yaml").write_text(_ACQ_YAML)
    (tmp_path / "acquisition parameters.json").write_text(json.dumps(_JSON))
    m = load_acquisition_metadata(tmp_path)
    assert m["source"] == "acquisition.yaml"
    assert m["pixel_size_um"] == 0.325           # stored value, NOT recomputed 3.76/20=0.188
    assert m["dz_um"] == 0.001031 * 1000          # mm -> um
    assert m["n_z_declared"] == 3
    assert m["n_t_declared"] == 2
    assert m["wellplate_format"] == "24 well plate"


def test_json_fallback_recomputes_pixel_size(tmp_path):
    # no acquisition.yaml -> legacy flat JSON; pixel size recomputed sensor/magnification
    (tmp_path / "acquisition parameters.json").write_text(json.dumps(_JSON))
    m = load_acquisition_metadata(tmp_path)
    assert m["source"] == "acquisition parameters.json"
    assert m["pixel_size_um"] == 3.76 / 20.0      # 0.188
    assert m["dz_um"] == 1.031
    assert m["n_z_declared"] == 3
    assert m["n_t_declared"] == 1
    assert m["wellplate_format"] is None          # legacy sidecar has no plate format


def test_json_fallback_applies_tube_lens_correction(tmp_path):
    # a non-design tube lens must scale the effective magnification (sensor px != image px)
    (tmp_path / "acquisition parameters.json").write_text(
        json.dumps(
            {
                "objective": {"magnification": 20.0, "tube_lens_f_mm": 180.0},
                "tube_lens_mm": 200.0,
                "sensor_pixel_size_um": 3.76,
                "Nz": 1,
                "Nt": 1,
            }
        )
    )
    m = load_acquisition_metadata(tmp_path)
    # effective mag = 20 * (200/180); FOV pixel = sensor / effective_mag
    assert m["pixel_size_um"] == 3.76 / (20.0 * 200.0 / 180.0)


def test_no_sidecar_all_none(tmp_path):
    m = load_acquisition_metadata(tmp_path)
    assert m["source"] is None
    assert m["pixel_size_um"] is None
    assert m["n_z_declared"] is None
    assert m["wellplate_format"] is None
