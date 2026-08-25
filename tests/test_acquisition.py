"""Tests for scalar acquisition metadata — acquisition.yaml first, the legacy JSON as fallback."""

import json

import pytest

from squidxplorer._acquisition import load_acquisition_metadata

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
    assert set(m) == {"pixel_size_um", "n_z_declared", "dz_um", "n_t_declared",
                      "wellplate_format"}


#: Real legacy shape, from Squid's pre-yaml writer (mirrors conftest._PARAMS).
_LEGACY = {
    "Nz": 3,
    "Nt": 2,
    "dz(um)": 1.5,
    "objective": {"magnification": 20.0, "NA": 0.8},
    "sensor_pixel_size_um": 3.76,
}


def test_missing_both_files_raises_naming_both(tmp_path):
    with pytest.raises(FileNotFoundError, match="acquisition.yaml.*parameters.json"):
        load_acquisition_metadata(tmp_path)


def test_a_legacy_acquisition_loads_with_a_warning(tmp_path):
    """The old format is a supported SOURCE now, never a silent one: the pixel size is derived (sensor / magnification) and the warning says so, because the"""
    (tmp_path / "acquisition parameters.json").write_text(json.dumps(_LEGACY))
    with pytest.warns(UserWarning, match="legacy.*binning"):
        m = load_acquisition_metadata(tmp_path)
    assert m["pixel_size_um"] == pytest.approx(3.76 / 20.0)
    assert m["dz_um"] == 1.5                       # already micrometres in the legacy format
    assert m["n_z_declared"] == 3
    assert m["n_t_declared"] == 2
    assert m["wellplate_format"] is None
    assert set(m) == {"pixel_size_um", "n_z_declared", "dz_um", "n_t_declared",
                      "wellplate_format"}, "both readings must produce the same keys"


def test_the_yaml_outranks_a_legacy_file_beside_it(tmp_path):
    """A post-yaml acquisition ships BOTH files; the yaml's stored, binning-aware pixel size must win over the legacy derivation."""
    (tmp_path / "acquisition.yaml").write_text(_ACQ_YAML)
    (tmp_path / "acquisition parameters.json").write_text(json.dumps(_LEGACY))
    m = load_acquisition_metadata(tmp_path)
    assert m["pixel_size_um"] == 0.325, "the legacy derivation (0.188) outranked the yaml"


def test_a_legacy_file_missing_the_optics_still_loads_but_knows_no_pixel_size(tmp_path):
    (tmp_path / "acquisition parameters.json").write_text('{"Nz": 3}')
    with pytest.warns(UserWarning, match="pixel size is unknown"):
        m = load_acquisition_metadata(tmp_path)
    assert m["pixel_size_um"] is None              # unknown, never guessed
    assert m["n_z_declared"] == 3


def test_a_corrupt_legacy_file_is_refused_by_name(tmp_path):
    (tmp_path / "acquisition parameters.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_acquisition_metadata(tmp_path)


def test_an_old_acquisition_opens_end_to_end_through_the_reader(squid_dataset):
    """THE CUSTOMER CASE: a real (synthetic) acquisition with the yaml DELETED — exactly what an old dataset looks like on disk — must open through the"""
    from squidxplorer import open_reader

    root, _ = squid_dataset
    (root / "acquisition.yaml").unlink()
    with pytest.warns(UserWarning, match="legacy"):
        meta = open_reader(str(root)).metadata
    assert meta["pixel_size_um"] == pytest.approx(3.76 / 20.0)   # conftest._PARAMS optics
    assert meta["n_z"] >= 1 and meta["regions"], "the reader did not assemble the acquisition"
