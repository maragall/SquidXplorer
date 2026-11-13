"""The movie's measurement pair: the numbers describe exactly the pixels each frame shows."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from squidxplorer import open_reader
from squidxplorer._measurements import (
    STATS_COLUMNS,
    frame_profile_rows,
    frame_stats_rows,
    measurement_paths,
)
from squidxplorer._mosaic_source import fuse_region_mosaic, mosaic_bbox_um
from squidxplorer._video import encoder_problem, record_region, region_movie_frames

_REPO = Path(__file__).resolve().parent.parent


def _make_5d():
    spec = importlib.util.spec_from_file_location(
        "_make_5d_fixture", _REPO / "tools" / "make_5d_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def five_d(tmp_path_factory):
    """1 region x 4 FOV x 3 z x 2 channels x 3 t, 64 px frames."""
    root = tmp_path_factory.mktemp("measure") / "acq5d"
    _make_5d().build(root, ["A1"], n_fovs=4, nz=3, nt=3, size=64)
    # the fixture's yaml declares nt and no interval; a real time series declares both
    yaml_path = root / "acquisition.yaml"
    yaml_path.write_text(yaml_path.read_text() + "  delta_t_s: 1800.0\n", encoding="utf-8")
    reader = open_reader(root)
    return reader, reader.metadata, "A1"


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_stats_ignore_the_uncovered_zeros_of_a_mosaic():
    # Arrange: half the plane is uncovered mosaic area (exact zeros), the rest reads 100 and 300.
    plane = np.zeros((4, 4), np.uint16)
    plane[:, 2] = 100
    plane[:, 3] = 300
    store = plane[None]

    # Act
    (row,) = frame_stats_rows(store, ["ch"], frame=0, time_point=0, z_level=0,
                              time_s=None, pixel_um=2.0)

    # Assert: the mean is over the 8 lit pixels, never dragged down by the 8 uncovered ones.
    assert row["n_pixels"] == 8
    assert row["mean"] == pytest.approx(200.0)
    assert row["min"] == 100 and row["max"] == 300
    assert row["time_s_declared"] == ""


def test_the_profile_runs_in_micrometres_along_each_axis():
    # Arrange: intensity rises along x and is flat along y.
    store = np.tile(np.array([10, 20, 30], np.uint16), (2, 1))[None]

    # Act
    rows = frame_profile_rows(store, ["ch"], frame=0, time_point=0, z_level=0,
                              time_s=0.0, pixel_um=1.5)

    # Assert
    along_x = [(r["distance_um"], r["intensity"]) for r in rows if r["axis"] == "x"]
    along_y = [r["intensity"] for r in rows if r["axis"] == "y"]
    assert along_x == [(0.0, 10.0), (1.5, 20.0), (3.0, 30.0)]
    assert along_y == [20.0, 20.0]


@pytest.mark.skipif(encoder_problem() is not None, reason="no mp4 encoder on this machine")
def test_a_time_movie_lands_with_the_numbers_of_its_own_frames(five_d, tmp_path):
    # Arrange
    reader, meta, region = five_d
    out = tmp_path / "A1_t.mp4"
    names = [c["name"] for c in meta["channels"]]

    # Act
    _path, n = record_region(reader, meta, region, out, axis="t", z_level=1,
                             measurements=True)

    # Assert: one stats row per frame per channel, each equal to the fused plane it describes.
    stats_path, profile_path = measurement_paths(out)
    rows = _read(stats_path)
    assert list(rows[0]) == list(STATS_COLUMNS)
    assert n == 3 and len(rows) == 3 * len(names)
    z_use = list(meta["z_levels"])[1]
    for row in rows:
        fused, step = fuse_region_mosaic(reader, meta, region, row["channel"], z_level=z_use,
                                         time_point=int(row["time_point"]), max_px=1080)
        lit = fused[fused > 0]
        assert int(row["frame"]) == int(row["time_point"])
        # the fixture declares a 1800 s interval; the clock column is that times the timepoint
        assert float(row["time_s_declared"]) == pytest.approx(1800.0 * int(row["time_point"]))
        assert float(row["mean"]) == pytest.approx(float(lit.mean()))
        assert float(row["pixel_um"]) == pytest.approx(float(meta["pixel_size_um"]) * step)
    profile = _read(profile_path)
    assert {r["axis"] for r in profile} == {"x", "y"}
    assert {r["frame"] for r in profile} == {"0", "1", "2"}


def test_an_roi_movie_measures_only_the_box(five_d):
    # Arrange: the left half of the region, in world micrometres.
    reader, meta, region = five_d
    x0, y0, x1, y1 = mosaic_bbox_um(meta, region)
    roi = (x0, y0, (x0 + x1) / 2.0, y1)
    seen = []

    # Act
    whole = next(region_movie_frames(reader, meta, region, axis="t"))
    boxed = next(region_movie_frames(
        reader, meta, region, axis="t", bbox_um=roi,
        on_planes=lambda frame, zi, ti, names, store, step: seen.append(store.shape)))

    # Assert: the frame and the measured planes are the SAME crop, half the region's width.
    assert boxed.shape[0] == whole.shape[0]
    assert abs(boxed.shape[1] - whole.shape[1] / 2.0) <= 1
    assert seen[0][1:] == boxed.shape[:2]
