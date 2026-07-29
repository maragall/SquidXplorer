"""The time axis: the fixture that finally exercises it, and the bug it makes visible.

READ THIS BEFORE CHANGING ANY ASSERTION IN THIS FILE.

`test_the_written_plate_serves_only_the_first_timepoint` and
`test_the_coarse_preview_read_cannot_tell_three_timepoints_from_one` assert the CURRENT,
WRONG behaviour on purpose. They are green, and they are green while a real bug is live. That
is deliberate and it is stated here so nobody mistakes green for correct: an xfail or a skip is
how 51 failures hid in this suite for weeks, so the bug is documented by a passing test whose
docstring says exactly what is broken and what the assertion must become.

THE BUG. The acquisition, the reader and the projection engine all carry the time axis
correctly: `reader.read(region, fov, channel, z, t)` takes `t`, `reader.metadata["n_t"]` is real,
`project_well(..., t=None)` returns every timepoint and `project_plate` yields the full
`(T, C, 1, Y, X)` array. The plate we write to disk is TCZYX and holds every frame. Then every
consumer throws all but the first frame away by indexing `[0, :, 0]`:

    squidmip/_viewer.py:1108   `_ZarrLoupeSource.coarse`
    squidmip/_viewer.py:3087   `_ComputedPlateWorker._on_well`  (`well = image[0, :, 0]`)
    squidmip/_viewer.py:3866   `_ComputedPlateWorker._read`

and `project_plate(...) -> (region, fov, array)` never mentions `t` in its signature, so nothing
between the engine and the screen has a timepoint to pass. A 40-timepoint plate therefore looks
exactly like a 1-timepoint plate, with no error, no warning and no visible difference.

WHY IT SURVIVED. Nothing in the corpus had more than one timepoint. Every fixture in
`conftest.py` was `Nt=1`, and so is every real acquisition on this machine: the 10x laser-AF
tissue set, the 20x scan and sim_1536wp all record `Nt: 1` in both `acquisition parameters.json`
and `acquisition.yaml`. With `n_t == 1`, `[0, :, 0]` and "the whole dataset" are the same pixels,
so the bug was invisible by construction. `multi_time_point_dataset` is the fix for that.

WHAT CHANGES WHEN THE `t` SLIDER LANDS. The two bug-documenting tests invert:

  * `test_the_written_plate_serves_only_the_first_timepoint` becomes a test that the preview
    read at timepoint `k` returns timepoint `k`'s pixels for every k in range(n_t), and its
    `_PREVIEW_TIME_POINT` constant becomes the slider's position.
  * `test_the_coarse_preview_read_cannot_tell_three_timepoints_from_one` becomes its own
    negation: a 3-timepoint plate and a 1-timepoint plate must NOT produce identical previews.

The tests above them (layout, `n_t`, per-timepoint pixels, the engine's own output) are correct
today and must stay green through that change unaltered.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import tensorstore as ts

from squidmip import open_reader
from squidmip._engine import project_plate
from squidmip._output import parse_well_id, write_plate
from squidmip.contract import field_path
from tests.conftest import (
    N_TIME_POINTS,
    TIME_SERIES_CHANNELS,
    TIME_SERIES_FOV,
    TIME_SERIES_NZ,
    TIME_SERIES_REGION,
    time_series_pixel_value,
)

#: The timepoint the plate preview path is hardcoded to. Not a parameter anywhere yet: that is
#: the bug. When the slider lands this becomes the slider's value.
_PREVIEW_TIME_POINT = 0


# --- the fixture is Squid's layout, not an invention -------------------------------------------

def test_the_fixture_is_squids_own_on_disk_layout(multi_time_point_dataset):
    """One folder per timepoint, unpadded, each with its own executed coordinates.csv and .done.

    Verified against `Squid/software/control/core/multi_point_worker.py`: :744 names the folder
    `f"{time_point:0{FILE_ID_PADDING}}"` with `FILE_ID_PADDING = 0` (`control/_def.py:720`), so
    the names carry no padding; :757 writes the timepoint's coordinates.csv from the frame built
    at :802-805; :785 drops the `.done` marker via `control/utils.py:193`.
    """
    root, _ = multi_time_point_dataset

    folders = sorted(p for p in root.iterdir() if p.is_dir())
    assert [p.name for p in folders] == ["0", "1", "2"], "unpadded, one folder per timepoint"

    for folder in folders:
        assert (folder / ".done").exists(), f"{folder.name} has no .done marker"
        planes = sorted(p.name for p in folder.glob("*.tiff"))
        assert len(planes) == TIME_SERIES_NZ * len(TIME_SERIES_CHANNELS)
        for name in planes:
            # {region}_{fov}_{z_level}_{channel}.tiff, multi_point_worker.py:1108
            assert name.startswith(f"{TIME_SERIES_REGION}_{TIME_SERIES_FOV}_")


def test_the_two_coordinates_csv_files_are_not_the_same_file(multi_time_point_dataset):
    """Root coordinates.csv is the PLAN, each timepoint's is the RECORD. Different columns.

    Two files, one name, two meanings, and conflating them is how a planned position ends up
    presented as a measured one. The root file comes from
    `multi_point_controller.py:735-744` (written before the run, from the scan's intended FOV
    coordinates); the per-timepoint file comes from `multi_point_worker.py:802-805` and :757
    (written after the timepoint finishes, from `stage.get_pos()`, with a wall-clock stamp per
    plane). `reader.py` documents them as "schema (a)" and "schema (b)", which reads as two
    dialects of one file rather than two different files.
    """
    root, _ = multi_time_point_dataset

    planned = (root / "coordinates.csv").read_text().splitlines()
    executed = (root / "0" / "coordinates.csv").read_text().splitlines()

    assert planned[0] == "region,x (mm),y (mm),z (mm)"
    assert executed[0] == "region,fov,z_level,x (mm),y (mm),z (um),time"
    assert planned[0] != executed[0]

    # The plan has one row per FOV; the record has one row per (fov, z_level) and carries time.
    assert len(planned) == 1 + 1
    assert len(executed) == 1 + TIME_SERIES_NZ
    assert "time" not in planned[0] and "z (um)" not in planned[0]

    # And every timepoint records its own, which is the whole point of the per-timepoint file.
    for time_point in range(N_TIME_POINTS):
        assert (root / str(time_point) / "coordinates.csv").exists()


def test_the_fixture_declares_nt_in_both_sidecars(multi_time_point_dataset):
    """`Nt` in acquisition parameters.json and `time_series.nt` in acquisition.yaml agree.

    The reader cross-checks the declared count against the folders found and warns on a
    mismatch, so a fixture that declared 1 while writing 3 would emit a UserWarning and
    misrepresent what a real Squid run looks like.
    """
    root, _ = multi_time_point_dataset

    params = json.loads((root / "acquisition parameters.json").read_text())
    assert params["Nt"] == N_TIME_POINTS

    text = (root / "acquisition.yaml").read_text()
    assert "time_series:" in text and f"nt: {N_TIME_POINTS}" in text


# --- the reader carries the time axis correctly. Everything below leans on this ----------------

def test_the_reader_reports_three_timepoints(multi_time_point_dataset):
    root, _ = multi_time_point_dataset
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        meta = open_reader(root).metadata
    # A fixture whose declared Nt disagreed with its folder count would warn here, and a warning
    # is how a fixture quietly stops describing a real Squid run.
    assert not [str(w.message) for w in caught], "the fixture must not make the reader warn"
    assert meta["n_t"] == N_TIME_POINTS
    assert meta["n_z"] == TIME_SERIES_NZ
    assert meta["regions"] == [TIME_SERIES_REGION]
    assert meta["fovs_per_region"] == {TIME_SERIES_REGION: [TIME_SERIES_FOV]}
    assert len(meta["channels"]) == len(TIME_SERIES_CHANNELS)


def test_every_timepoint_holds_its_own_pixels(multi_time_point_dataset):
    """The fixture is only worth having if t=2 is distinguishable from t=0. Prove it on data."""
    root, planes = multi_time_point_dataset
    reader = open_reader(root)
    channel_names = [c["name"] for c in reader.metadata["channels"]]

    seen = set()
    for time_point in range(N_TIME_POINTS):
        for z_level in range(TIME_SERIES_NZ):
            for channel_index, channel in enumerate(channel_names):
                got = reader.read(TIME_SERIES_REGION, TIME_SERIES_FOV, channel, z_level,
                                  t=time_point)
                want = time_series_pixel_value(time_point, z_level, channel_index)
                assert got.min() == got.max() == want
                np.testing.assert_array_equal(got, planes[(time_point, z_level, channel)])
                seen.add(want)
    # Every plane is a distinct value: no two (t, z, channel) triples can be confused.
    assert len(seen) == N_TIME_POINTS * TIME_SERIES_NZ * len(TIME_SERIES_CHANNELS)


def test_the_engine_keeps_all_three_timepoints(multi_time_point_dataset):
    """`project_plate` yields `(T, C, 1, Y, X)` with T == n_t. The loss is downstream of here.

    Worth pinning separately from the consumer tests below: it is what makes those a display
    bug rather than a data-loss bug, and it is what makes the fix cheap when 4b lands.
    """
    root, _ = multi_time_point_dataset
    reader = open_reader(root)
    results = list(project_plate(reader, n_fovs=1, workers=1))

    assert len(results) == 1
    region, fov, image = results[0]
    assert (region, fov) == (TIME_SERIES_REGION, TIME_SERIES_FOV)
    assert image.shape == (N_TIME_POINTS, len(TIME_SERIES_CHANNELS), 1, 4, 4)
    # MIP over z, so each frame is the max over z_level of that (t, channel).
    for time_point in range(N_TIME_POINTS):
        for channel_index in range(len(TIME_SERIES_CHANNELS)):
            want = max(time_series_pixel_value(time_point, z, channel_index)
                       for z in range(TIME_SERIES_NZ))
            assert image[time_point, channel_index, 0].min() == want
    # And the frames differ across t, which is the property every assertion below depends on.
    assert len({int(image[k, 0, 0, 0, 0]) for k in range(N_TIME_POINTS)}) == N_TIME_POINTS


# --- THE BUG. Both tests below assert the WRONG behaviour on purpose. See the module docstring --

def _preview_read(plate_dir, region, fov):
    """The exact expression the three plate-preview sites use, in one place.

    `_ZarrLoupeSource.coarse` (`_viewer.py:1108`), `_ComputedPlateWorker._read`
    (`_viewer.py:3866`) and `_ComputedPlateWorker._on_well` (`_viewer.py:3087`) all reduce a
    TCZYX field to `(C, Y, X)` with a literal `[0, :, 0]`. That literal is reproduced here
    verbatim rather than parameterised, because the point is that the sites have no parameter:
    when 4b threads a timepoint through them, this helper grows the argument they grow.
    """
    row, col = parse_well_id(region)
    arr = ts.open({"driver": "zarr3",
                   "kvstore": {"driver": "file",
                               "path": field_path(plate_dir, f"{row}/{col}", fov, "0")}}).result()
    return np.asarray(arr[0, :, 0].read().result()), arr.shape


def test_the_written_plate_serves_only_the_first_timepoint(multi_time_point_dataset, tmp_path):
    """BUG, ASSERTED AS-IS: the store holds 3 timepoints and the preview read returns 1.

    The write is correct. `plate.ome.zarr` is TCZYX with T == 3 and every frame in it, which is
    asserted below so a future regression in the WRITER cannot hide behind this test. What is
    wrong is the read: `[0, :, 0]` is a literal at all three preview sites
    (`_viewer.py:1108`, `:3087`, `:3866`), so timepoints 1 and 2 are unreachable from the
    viewer no matter what the user does.

    WHEN 4b LANDS this must become: for every `time_point` in `range(n_t)`, the preview read at
    that timepoint equals that timepoint's pixels. The `_PREVIEW_TIME_POINT` constant becomes
    the slider position, and the final assertion (that timepoints 1 and 2 are unreachable) is
    deleted rather than adjusted.
    """
    root, _ = multi_time_point_dataset
    reader = open_reader(root)
    write_plate(reader, tmp_path, n_fovs=1, workers=1, tiff=False)
    plate_dir = tmp_path / "plate.ome.zarr"

    preview, shape = _preview_read(plate_dir, TIME_SERIES_REGION, TIME_SERIES_FOV)

    # The DATA is all there: nothing is lost on the way to disk.
    assert shape[0] == N_TIME_POINTS, "the writer must keep every timepoint"
    assert preview.shape == (len(TIME_SERIES_CHANNELS), 4, 4)

    # The READ is stuck at t=0. This is the bug.
    for channel_index in range(len(TIME_SERIES_CHANNELS)):
        want_first = max(time_series_pixel_value(_PREVIEW_TIME_POINT, z, channel_index)
                         for z in range(TIME_SERIES_NZ))
        assert preview[channel_index].min() == preview[channel_index].max() == want_first

    # And the later timepoints are unreachable through this path: no argument selects them.
    for time_point in range(1, N_TIME_POINTS):
        unreachable = max(time_series_pixel_value(time_point, z, 0)
                          for z in range(TIME_SERIES_NZ))
        assert preview[0].max() != unreachable


def test_the_coarse_preview_read_cannot_tell_three_timepoints_from_one(
    multi_time_point_dataset, squid_dataset, tmp_path
):
    """BUG, ASSERTED AS-IS: a 3-timepoint plate previews identically to its own first frame.

    Stated the way a user meets it: the plate preview of an acquisition with `Nt=3` is
    byte-identical to the preview of the same acquisition truncated to its first timepoint. The
    two acquisitions differ on disk by two thirds of their frames and the viewer shows the same
    picture for both, with nothing anywhere saying a time axis exists.

    `squid_dataset` is taken as a fixture and deliberately unused, so that the contrast being
    drawn (every other fixture in this suite is Nt=1) is visible at the call site rather than
    only in prose.

    WHEN 4b LANDS this test inverts: the two previews must DIFFER for at least one timepoint,
    because a preview that is the same for every t is exactly what the slider exists to fix.
    """
    root, _ = multi_time_point_dataset

    # Truncate a copy to one timepoint: same acquisition, two thirds of the frames removed.
    truncated = tmp_path / "truncated"
    truncated.mkdir()
    for item in root.iterdir():
        if item.is_dir() and item.name != "0":
            continue                                   # drop timepoints 1 and 2
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

    assert full_shape[0] == N_TIME_POINTS and one_shape[0] == 1   # the stores are NOT the same
    np.testing.assert_array_equal(full_preview, one_preview)      # the previews ARE. The bug.
