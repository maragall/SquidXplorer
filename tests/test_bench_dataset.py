"""Acquisition parsing: identity from filenames, geometry from coordinates.csv.

The failure this file cares most about is the quiet one: a fixture directory that still
exists but whose contents are gone. `load_acquisition` must say so in words, because
the alternative -- letting it through to die inside tifffile -- is how the SquidMIP
integration suite ended up red and misdiagnosed.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.dataset import AcquisitionError, load_acquisition
from tests.conftest_bench import write_acquisition, write_broken_symlink_farm


def test_parses_a_synthetic_acquisition(tmp_path):
    gt = write_acquisition(tmp_path, grid=(2, 3), tile=(64, 64), step=(48, 48))
    acq = load_acquisition(tmp_path)
    assert acq.regions == ["C5"]
    assert acq.frame_shape == (64, 64)
    assert acq.dtype == "uint16"
    assert acq.fovs("C5") == list(range(6))
    assert acq.n_tiles == 6
    assert acq.pixel_size_um == pytest.approx(gt["pixel_um"])


def test_positions_px_round_trips_the_ground_truth(tmp_path):
    """mm in the CSV must convert back to the pixel offsets the tiles were cut at."""
    gt = write_acquisition(tmp_path, grid=(2, 3), tile=(64, 64), step=(48, 48))
    acq = load_acquisition(tmp_path)
    got = acq.positions_px("C5")
    for fov, (y, x) in gt["positions_px"].items():
        assert got[fov][0] == pytest.approx(y, abs=0.01)
        assert got[fov][1] == pytest.approx(x, abs=0.01)


def test_handles_the_monkey_layout_with_an_explicit_fov_column(tmp_path):
    gt = write_acquisition(tmp_path, grid=(1, 3), tile=(64, 64), step=(48, 48), with_fov_column=True)
    acq = load_acquisition(tmp_path)
    got = acq.positions_px("C5")
    assert sorted(got) == [0, 1, 2]
    assert got[2][1] == pytest.approx(gt["positions_px"][2][1], abs=0.01)


def test_channel_names_containing_underscores_survive_parsing(tmp_path):
    write_acquisition(tmp_path, grid=(1, 2), tile=(32, 32), step=(24, 24))
    acq = load_acquisition(tmp_path)
    assert acq.channels == ["Fluorescence_405_nm_Ex"]


def test_multiple_channels_and_z_levels(tmp_path):
    write_acquisition(
        tmp_path,
        grid=(1, 2),
        tile=(32, 32),
        step=(24, 24),
        channels=["Fluorescence_405_nm_Ex", "Fluorescence_488_nm_Ex"],
        z_levels=2,
    )
    acq = load_acquisition(tmp_path)
    assert len(acq.channels) == 2
    assert acq.z_levels == [0, 1]
    assert acq.n_tiles == 2 * 2 * 2


def test_read_returns_the_pixels(tmp_path):
    write_acquisition(tmp_path, grid=(1, 2), tile=(32, 32), step=(24, 24))
    acq = load_acquisition(tmp_path)
    plane = acq.read("C5", 0, 0, acq.channels[0])
    assert plane.shape == (32, 32)
    assert plane.dtype == np.uint16


# ------------------------------------------------------------------ failure modes


def test_broken_symlink_farm_is_named_precisely(tmp_path):
    """The sim_1536wp failure: directory exists, every link dangles."""
    write_broken_symlink_farm(tmp_path)
    with pytest.raises(AcquisitionError, match="dangling"):
        load_acquisition(tmp_path)


def test_missing_coordinates_csv_explains_why_it_matters(tmp_path):
    write_acquisition(tmp_path, grid=(1, 2), tile=(32, 32), step=(24, 24))
    (tmp_path / "coordinates.csv").unlink()
    with pytest.raises(AcquisitionError, match="no coordinates.csv"):
        load_acquisition(tmp_path)


def test_missing_acquisition_parameters_is_an_error(tmp_path):
    write_acquisition(tmp_path, grid=(1, 2), tile=(32, 32), step=(24, 24))
    (tmp_path / "acquisition parameters.json").unlink()
    with pytest.raises(AcquisitionError, match="acquisition parameters"):
        load_acquisition(tmp_path)


def test_fov_count_mismatch_is_refused_rather_than_guessed(tmp_path):
    """Filenames name 3 FOVs but the CSV places 2. Trusting row order here would
    silently misplace a tile and corrupt every residual downstream."""
    write_acquisition(tmp_path, grid=(1, 3), tile=(32, 32), step=(24, 24))
    lines = (tmp_path / "coordinates.csv").read_text().splitlines()
    (tmp_path / "coordinates.csv").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(AcquisitionError, match="disagree"):
        load_acquisition(tmp_path)


def test_directory_that_does_not_exist(tmp_path):
    with pytest.raises(AcquisitionError, match="not a directory"):
        load_acquisition(tmp_path / "nope")


def test_directory_with_no_tiffs(tmp_path):
    (tmp_path / "0").mkdir()
    with pytest.raises(AcquisitionError, match="no Squid-named TIFFs"):
        load_acquisition(tmp_path)


def test_header_only_coordinates_csv(tmp_path):
    write_acquisition(tmp_path, grid=(1, 2), tile=(32, 32), step=(24, 24))
    (tmp_path / "coordinates.csv").write_text("region,x (mm),y (mm),z (mm)\n")
    with pytest.raises(AcquisitionError, match="is empty"):
        load_acquisition(tmp_path)


def test_coordinates_csv_with_blank_region_rows(tmp_path):
    """Rows present but no usable region -- distinct from a header-only file."""
    write_acquisition(tmp_path, grid=(1, 2), tile=(32, 32), step=(24, 24))
    (tmp_path / "coordinates.csv").write_text("region,x (mm),y (mm),z (mm)\n,0,0,\n")
    with pytest.raises(AcquisitionError, match="no positions"):
        load_acquisition(tmp_path)


def test_coordinates_csv_missing_required_columns(tmp_path):
    write_acquisition(tmp_path, grid=(1, 2), tile=(32, 32), step=(24, 24))
    (tmp_path / "coordinates.csv").write_text("region,foo,bar\nC5,1,2\n")
    with pytest.raises(AcquisitionError, match="lacks region/x/y"):
        load_acquisition(tmp_path)
