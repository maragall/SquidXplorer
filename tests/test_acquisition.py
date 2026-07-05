"""Tests for scalar acquisition metadata — acquisition.yaml is the single required format."""

import pytest

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


def test_reads_acquisition_yaml(tmp_path):
    (tmp_path / "acquisition.yaml").write_text(_ACQ_YAML)
    m = load_acquisition_metadata(tmp_path)
    assert m["pixel_size_um"] == 0.325            # stored, binning-aware (not recomputed)
    assert m["dz_um"] == 0.001031 * 1000          # mm -> um
    assert m["n_z_declared"] == 3
    assert m["n_t_declared"] == 2
    assert m["wellplate_format"] == "24 well plate"


def test_missing_yaml_raises(tmp_path):
    # single format, loud on absence: no silent JSON recompute, no None-degradation.
    with pytest.raises(FileNotFoundError, match="acquisition.yaml"):
        load_acquisition_metadata(tmp_path)


def test_legacy_json_is_ignored(tmp_path):
    # a lone legacy 'acquisition parameters.json' must NOT be silently accepted.
    (tmp_path / "acquisition parameters.json").write_text('{"Nz": 3}')
    with pytest.raises(FileNotFoundError, match="acquisition.yaml"):
        load_acquisition_metadata(tmp_path)


def test_metadata_keys_exact_no_dead_attributes(tmp_path):
    # Guard against dead / leftover keys after edits (e.g. the removed 'source' key): the dict
    # must be EXACTLY the fields the reader consumes — surfaced in reader.metadata
    # (pixel_size_um, dz_um, wellplate_format) or used for its Nz/Nt cross-check
    # (n_z_declared, n_t_declared). Nothing else, so no attribute goes stale/unused.
    (tmp_path / "acquisition.yaml").write_text(_ACQ_YAML)
    m = load_acquisition_metadata(tmp_path)
    assert set(m) == {
        "pixel_size_um",
        "n_z_declared",
        "dz_um",
        "n_t_declared",
        "wellplate_format",
    }
