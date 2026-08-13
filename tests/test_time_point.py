"""The time axis: the fixture that finally exercises it, and the bug it makes visible."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import tensorstore as ts

from squidxplorer import open_reader
from squidxplorer._engine import run_plate
from squidxplorer._output import parse_well_id, write_plate
from squidxplorer.contract import field_path
from tests.conftest import (
    N_TIME_POINTS,
    TIME_SERIES_CHANNELS,
    TIME_SERIES_FOV,
    TIME_SERIES_NZ,
    TIME_SERIES_REGION,
    time_series_pixel_value,
)

_PREVIEW_TIME_POINT = 0


@pytest.fixture(scope="module")
def qapp():
    """This module's QApplication."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


def test_the_fixture_is_squids_own_on_disk_layout(multi_time_point_dataset):
    root, _ = multi_time_point_dataset

    folders = sorted(p for p in root.iterdir() if p.is_dir())
    assert [p.name for p in folders] == ["0", "1", "2"], "unpadded, one folder per timepoint"

    for folder in folders:
        assert (folder / ".done").exists(), f"{folder.name} has no .done marker"
        planes = sorted(p.name for p in folder.glob("*.tiff"))
        assert len(planes) == TIME_SERIES_NZ * len(TIME_SERIES_CHANNELS)
        for name in planes:
            assert name.startswith(f"{TIME_SERIES_REGION}_{TIME_SERIES_FOV}_")


def test_the_two_coordinates_csv_files_are_not_the_same_file(multi_time_point_dataset):
    root, _ = multi_time_point_dataset

    planned = (root / "coordinates.csv").read_text().splitlines()
    executed = (root / "0" / "coordinates.csv").read_text().splitlines()

    assert planned[0] == "region,x (mm),y (mm),z (mm)"
    assert executed[0] == "region,fov,z_level,x (mm),y (mm),z (um),time"
    assert planned[0] != executed[0]

    assert len(planned) == 1 + 1
    assert len(executed) == 1 + TIME_SERIES_NZ
    assert "time" not in planned[0] and "z (um)" not in planned[0]

    for time_point in range(N_TIME_POINTS):
        assert (root / str(time_point) / "coordinates.csv").exists()


def test_the_fixture_declares_nt_in_both_sidecars(multi_time_point_dataset):
    root, _ = multi_time_point_dataset

    params = json.loads((root / "acquisition parameters.json").read_text())
    assert params["Nt"] == N_TIME_POINTS

    text = (root / "acquisition.yaml").read_text()
    assert "time_series:" in text and f"nt: {N_TIME_POINTS}" in text


def test_the_reader_reports_three_timepoints(multi_time_point_dataset):
    root, _ = multi_time_point_dataset
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        meta = open_reader(root).metadata
    assert not [str(w.message) for w in caught], "the fixture must not make the reader warn"
    assert meta["n_t"] == N_TIME_POINTS
    assert meta["n_z"] == TIME_SERIES_NZ
    assert meta["regions"] == [TIME_SERIES_REGION]
    assert meta["fovs_per_region"] == {TIME_SERIES_REGION: [TIME_SERIES_FOV]}
    assert len(meta["channels"]) == len(TIME_SERIES_CHANNELS)


def test_every_timepoint_holds_its_own_pixels(multi_time_point_dataset):
    root, planes = multi_time_point_dataset
    reader = open_reader(root)
    channel_names = [c["name"] for c in reader.metadata["channels"]]

    seen = set()
    for time_point in range(N_TIME_POINTS):
        for z_level in range(TIME_SERIES_NZ):
            for channel_index, channel in enumerate(channel_names):
                got = reader.read(TIME_SERIES_REGION, TIME_SERIES_FOV, channel, z_level,
                                  time_point=time_point)
                want = time_series_pixel_value(time_point, z_level, channel_index)
                assert got.min() == got.max() == want
                np.testing.assert_array_equal(got, planes[(time_point, z_level, channel)])
                seen.add(want)
    assert len(seen) == N_TIME_POINTS * TIME_SERIES_NZ * len(TIME_SERIES_CHANNELS)


def test_the_engine_keeps_all_three_timepoints(multi_time_point_dataset):
    root, _ = multi_time_point_dataset
    reader = open_reader(root)
    results = list(run_plate(reader, n_fovs=1, workers=1))

    assert len(results) == 1
    region, fov, image = results[0]
    assert (region, fov) == (TIME_SERIES_REGION, TIME_SERIES_FOV)
    assert image.shape == (N_TIME_POINTS, len(TIME_SERIES_CHANNELS), 1, 4, 4)
    for time_point in range(N_TIME_POINTS):
        for channel_index in range(len(TIME_SERIES_CHANNELS)):
            want = max(time_series_pixel_value(time_point, z, channel_index)
                       for z in range(TIME_SERIES_NZ))
            assert image[time_point, channel_index, 0].min() == want
    assert len({int(image[k, 0, 0, 0, 0]) for k in range(N_TIME_POINTS)}) == N_TIME_POINTS


def test_the_region_mosaic_fuses_the_timepoint_it_is_asked_for(multi_time_point_dataset, qapp):
    from squidxplorer._workers import _MosaicWorker

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    meta = reader.metadata
    channels = [c["name"] for c in meta["channels"]]

    def fused(time_point):
        got = {}
        w = _MosaicWorker(reader, meta, TIME_SERIES_REGION, channels, time_point=time_point)
        w.ready.connect(lambda r, ch, levels, bbox, win: got.__setitem__(ch, levels))
        problems = []
        w.problem.connect(problems.append)
        w.run()
        assert problems == [], f"t={time_point}: the worker reported {problems}"
        assert set(got) == set(channels)
        return {ch: np.asarray(levels[0][z]) for ch, levels in got.items()
                for z in [0]}

    at0, at1 = fused(0), fused(1)
    for channel_index, channel in enumerate(channels):
        want0 = time_series_pixel_value(0, 0, channel_index)
        want1 = time_series_pixel_value(1, 0, channel_index)
        assert at0[channel].min() == at0[channel].max() == want0, channel
        assert at1[channel].min() == at1[channel].max() == want1, channel
        assert not np.array_equal(at0[channel], at1[channel]), (
            f"{channel}: t=1 fused to the same pixels as t=0")


def test_a_region_window_fuses_the_timepoint_its_own_bar_shows(
    multi_time_point_dataset, napari_pane_stub, qapp
):
    from squidxplorer import _viewer as V
    from squidxplorer._region_viewer import ViewerManager

    root, _planes = multi_time_point_dataset
    reader = open_reader(root)
    mgr = ViewerManager(reader, reader.metadata)
    try:
        win = mgr.open([TIME_SERIES_REGION])
        assert win is not None
        assert win._time_point_bar.count == N_TIME_POINTS, (
            "the window must offer every timepoint or there is nothing to select")

        seen = []
        real_worker = V._MosaicWorker

        class _Recording(real_worker):
            def __init__(self, *a, **kw):
                seen.append(int(kw.get("time_point", 0)))
                super().__init__(*a, **kw)

            def start(self):
                pass

        V._MosaicWorker = _Recording
        try:
            win._time_point_bar.set_time_point(1)
            win._load_mosaic(TIME_SERIES_REGION)
        finally:
            V._MosaicWorker = real_worker
        assert seen and seen[-1] == 1, (
            f"the window's mosaic worker was built with t={seen}, not the bar's timepoint 1")
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def _preview_read(plate_dir, region, fov):
    """The exact expression the three plate-preview sites use, in one place."""
    row, col = parse_well_id(region)
    arr = ts.open({"driver": "zarr3",
                   "kvstore": {"driver": "file",
                               "path": field_path(plate_dir, f"{row}/{col}", fov, "0")}}).result()
    return np.asarray(arr[0, :, 0].read().result()), arr.shape


def test_the_written_plate_serves_only_the_first_timepoint(multi_time_point_dataset, tmp_path):
    root, _ = multi_time_point_dataset
    reader = open_reader(root)
    write_plate(reader, tmp_path, n_fovs=1, workers=1, tiff=False)
    plate_dir = tmp_path / "plate.ome.zarr"

    preview, shape = _preview_read(plate_dir, TIME_SERIES_REGION, TIME_SERIES_FOV)

    assert shape[0] == N_TIME_POINTS, "the writer must keep every timepoint"
    assert preview.shape == (len(TIME_SERIES_CHANNELS), 4, 4)

    for channel_index in range(len(TIME_SERIES_CHANNELS)):
        want_first = max(time_series_pixel_value(_PREVIEW_TIME_POINT, z, channel_index)
                         for z in range(TIME_SERIES_NZ))
        assert preview[channel_index].min() == preview[channel_index].max() == want_first

    for time_point in range(1, N_TIME_POINTS):
        unreachable = max(time_series_pixel_value(time_point, z, 0)
                          for z in range(TIME_SERIES_NZ))
        assert preview[0].max() != unreachable


def test_the_coarse_preview_read_cannot_tell_three_timepoints_from_one(
    multi_time_point_dataset, squid_dataset, tmp_path
):
    root, _ = multi_time_point_dataset

    truncated = tmp_path / "truncated"
    truncated.mkdir()
    for item in root.iterdir():
        if item.is_dir() and item.name != "0":
            continue
        target = truncated / item.name
        if item.is_dir():
            target.mkdir()
            for f in item.iterdir():
                target.joinpath(f.name).write_bytes(f.read_bytes())
        else:
            target.write_bytes(item.read_bytes())
    params = json.loads((truncated / "acquisition parameters.json").read_text())
    params["Nt"] = 1
    (truncated / "acquisition parameters.json").write_text(json.dumps(params))
    (truncated / "acquisition.yaml").write_text(
        (truncated / "acquisition.yaml").read_text().replace(f"nt: {N_TIME_POINTS}", "nt: 1")
    )

    assert open_reader(root).metadata["n_t"] == N_TIME_POINTS
    assert open_reader(truncated).metadata["n_t"] == 1

    full_dir = tmp_path / "full_out"
    one_dir = tmp_path / "one_out"
    write_plate(open_reader(root), full_dir, n_fovs=1, workers=1, tiff=False)
    write_plate(open_reader(truncated), one_dir, n_fovs=1, workers=1, tiff=False)

    full_preview, full_shape = _preview_read(
        full_dir / "plate.ome.zarr", TIME_SERIES_REGION, TIME_SERIES_FOV)
    one_preview, one_shape = _preview_read(
        one_dir / "plate.ome.zarr", TIME_SERIES_REGION, TIME_SERIES_FOV)

    assert full_shape[0] == N_TIME_POINTS and one_shape[0] == 1
    np.testing.assert_array_equal(full_preview, one_preview)
