"""Squid's downsampled well mosaics (mosaic_view/wells): discovery, pyramid top, plate seed, backfill."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from squidxplorer import _mosaic_source, _wellimage
from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

#: Filename-style names; the widget's layer names use spaces (see the normalisation test).
CHANNELS = ["Fluorescence_405_nm_Ex", "Fluorescence_638_nm_Ex"]
FRAME = (64, 64)
PX = 1.0            # µm/px, so Squid's 2 µm target gives factor 2
GRID = 4            # 4x4 FOVs at a 64 px step: 256 px extent, enough for a coarse rung >= factor
STEP_UM = 64.0


@pytest.fixture(autouse=True)
def _feature_on(monkeypatch):
    """The suite disables well images globally (conftest); these tests are their coverage."""
    monkeypatch.setenv(_wellimage.ENV_ENABLED, "1")
    _wellimage.clear_cache()
    yield
    _wellimage.clear_cache()


def _positions(regions) -> dict:
    out = {}
    for ri, region in enumerate(regions):
        ox = 10_000.0 * ri
        for fov in range(GRID * GRID):
            out[(region, fov)] = (ox + (fov % GRID) * STEP_UM, (fov // GRID) * STEP_UM)
    return out


def _meta(regions=("A1", "B2"), n_z: int = 1) -> dict:
    regions = list(regions)
    return {
        "regions": regions,
        "fovs_per_region": {r: list(range(GRID * GRID)) for r in regions},
        "fov_positions_um": _positions(regions),
        "channels": [{"name": c} for c in CHANNELS],
        "n_z": n_z, "z_levels": list(range(n_z)), "n_t": 1,
        "pixel_size_um": PX, "frame_shape": FRAME, "dtype": "uint16",
    }


class _SmoothReader:
    """Planes are a smooth gradient keyed to ABSOLUTE stage position: a misplaced paste shows
    as a large numeric deviation, while stride-vs-mean sampling stays within a few counts."""

    def __init__(self, path, meta):
        self.source_id = str(path)
        self._meta = meta
        self.reads = 0

    def read(self, region, fov, channel, z_level, time_point=0):
        self.reads += 1
        row, col = fov_offsets_px(self._meta["fov_positions_um"], region,
                                  self._meta["fovs_per_region"][region],
                                  self._meta["pixel_size_um"])[int(fov)]
        yy, xx = np.mgrid[0:FRAME[0], 0:FRAME[1]].astype(np.float32)
        c = CHANNELS.index(str(channel))
        val = 500.0 + 3.0 * (yy + row) + 5.0 * (xx + col) + 400.0 * c + 40.0 * int(z_level)
        return val.astype(np.uint16)


def _acq(tmp_path: Path, name="acq") -> Path:
    root = tmp_path / name
    (root / "0").mkdir(parents=True)
    return root


def _write_squid_well(root: Path, meta: dict, region: str, *, factor: int = 2,
                      names=None, shape=None, value: int = 7) -> Path:
    """Hand-write one well file the way Squid's widget does, independent of our backfill."""
    wells = root / "0" / "mosaic_view" / "wells"
    wells.mkdir(parents=True, exist_ok=True)
    offsets = fov_offsets_px(meta["fov_positions_um"], region,
                             meta["fovs_per_region"][region], meta["pixel_size_um"])
    full_h, full_w = mosaic_extent_px(offsets, FRAME)
    if shape is None:
        shape = (len(CHANNELS), int(round(full_h / factor)), int(round(full_w / factor)))
    stack = np.full(shape, value, dtype=np.uint16)
    res = int(round(meta["pixel_size_um"] * factor))
    path = wells / f"{region}_{res}um.tiff"
    tifffile.imwrite(path, stack, photometric="minisblack",
                     metadata={"axes": "CYX",
                               "Channel": {"Name": names or CHANNELS},
                               "well_id": region})
    return path


# --- discovery -------------------------------------------------------------------------------


def test_discovery_finds_the_file_and_derives_the_factor_from_its_own_size(tmp_path):
    root = _acq(tmp_path)
    meta = _meta()
    _write_squid_well(root, meta, "A1", factor=2, value=321)
    reader = _SmoothReader(root, meta)

    got = _wellimage.downsampled_well(reader, meta, "A1", CHANNELS[0], 0)
    assert got is not None, "a saved well image must be discovered"
    plane, factor = got
    assert factor == 2, "the factor comes from the FILE'S size vs the fused extent"
    assert np.all(plane == 321)
    assert _wellimage.downsampled_well(reader, meta, "B2", CHANNELS[0], 0) is None, \
        "a well with no file reads as absent"


def test_channel_names_match_across_squids_spaces_and_our_underscores(tmp_path):
    root = _acq(tmp_path)
    meta = _meta()
    # Squid's widget saves its LAYER names, which use spaces.
    _write_squid_well(root, meta, "A1", names=["Fluorescence 405 nm Ex",
                                               "Fluorescence 638 nm Ex"])
    got = _wellimage.downsampled_well(_SmoothReader(root, meta), meta, "A1",
                                      "Fluorescence_638_nm_Ex", 0)
    assert got is not None, "spaces vs underscores must not hide a channel"


def test_a_file_whose_size_fits_no_integer_factor_reads_as_absent(tmp_path, caplog):
    root = _acq(tmp_path)
    meta = _meta()
    _write_squid_well(root, meta, "A1", shape=(len(CHANNELS), 41, 97))
    with caplog.at_level("WARNING"):
        got = _wellimage.downsampled_well(_SmoothReader(root, meta), meta, "A1", CHANNELS[0], 0)
    assert got is None
    assert any("fits no integer downsample" in r.message for r in caplog.records), \
        "the refusal must be named, not silent"


def test_a_multi_z_acquisition_never_serves_a_well_image(tmp_path):
    root = _acq(tmp_path)
    meta = _meta(n_z=3)
    _write_squid_well(root, meta, "A1")
    assert _wellimage.load_well_stack(_SmoothReader(root, meta), meta, "A1", 0) is None, \
        "the saved image is ONE z; it may not stand in for a stack"


def test_the_env_kill_switch_turns_the_feature_off(tmp_path, monkeypatch):
    root = _acq(tmp_path)
    meta = _meta()
    _write_squid_well(root, meta, "A1")
    monkeypatch.setenv(_wellimage.ENV_ENABLED, "0")
    assert _wellimage.downsampled_well(_SmoothReader(root, meta), meta, "A1",
                                       CHANNELS[0], 0) is None


# --- the pyramid top -------------------------------------------------------------------------


class _CountingReader(_SmoothReader):
    pass


def test_the_coarse_rung_comes_from_the_well_image_and_never_touches_the_fovs(tmp_path):
    root = _acq(tmp_path)
    meta = _meta()
    reader = _SmoothReader(root, meta)
    assert _wellimage.write_well_images(reader, meta, time_point=0) == 2
    reader.reads = 0

    levels, _step0, _nz = _mosaic_source.fuse_region_pyramid(
        reader, meta, "A1", CHANNELS[0], cache_bytes=64 * 2 ** 20)
    coarse = np.asarray(levels[-1])
    assert coarse.size, "the coarse rung must materialise"
    assert reader.reads == 0, \
        f"first paint must cost ONE small file read, not {reader.reads} FOV decode(s)"

    np.asarray(levels[0])
    assert reader.reads > 0, "the fine rungs still fuse from the FOVs"


def test_the_well_rung_lands_on_the_fused_rungs_geometry(tmp_path):
    """Same origin, same scale: on a smooth set the two rungs agree within sampling noise."""
    root = _acq(tmp_path)
    meta = _meta()
    reader = _SmoothReader(root, meta)
    assert _wellimage.write_well_images(reader, meta, time_point=0) == 2

    plan = _mosaic_source._planned_plane(meta, "A1", 128)
    assert plan is not None
    h, w, step, dt = plan
    assert step >= 2, "this fixture must produce a rung at least as coarse as the factor"
    fused = _mosaic_source._fuse_levels(reader, meta, "A1", CHANNELS[0], 0, 0,
                                        [(128, h, w, step, dt)])[128]

    plane, factor = _wellimage.downsampled_well(reader, meta, "A1", CHANNELS[0], 0)
    derived = _wellimage.resample_plane(plane, factor, int(step), h, w)

    assert derived.shape == fused.shape
    dev = float(np.max(np.abs(fused.astype(np.float32) - derived.astype(np.float32))))
    # The gradient is 3-5 counts/px: a one-frame (or half-frame) misplacement would deviate by
    # hundreds. Area-mean vs stride sampling plus the ±1 px paste rounding stays under this.
    assert dev <= 16.0, f"the well rung is {dev} counts off the fused rung's geometry"


def test_a_corrupt_well_image_falls_back_to_fusing_with_a_named_line(tmp_path, caplog):
    root = _acq(tmp_path)
    meta = _meta()
    reader = _SmoothReader(root, meta)
    wells = root / "0" / "mosaic_view" / "wells"
    wells.mkdir(parents=True)
    (wells / "A1_2um.tiff").write_bytes(b"this is not a TIFF")

    with caplog.at_level("WARNING"):
        levels, _s, _n = _mosaic_source.fuse_region_pyramid(
            reader, meta, "A1", CHANNELS[0], cache_bytes=64 * 2 ** 20)
        coarse = np.asarray(levels[-1])
    assert reader.reads > 0, "the fallback must fuse from the FOVs"
    honest = _mosaic_source._fuse_levels(
        reader, meta, "A1", CHANNELS[0], 0, 0,
        [(128,) + tuple(v for v in _mosaic_source._planned_plane(meta, "A1", 128))])[128]
    assert np.array_equal(coarse, honest), "the fallback must be the ordinary fusion"
    assert any("unreadable" in r.message and "A1_2um.tiff" in r.message
               for r in caplog.records), "the corrupt file must be named in the log"


# --- the backfill ----------------------------------------------------------------------------


def test_the_backfill_writes_what_squid_would_write(tmp_path):
    root = _acq(tmp_path)
    meta = _meta(regions=("A1", "B2", "manual0"))
    meta["fov_positions_um"].update(
        {("manual0", f): (50_000.0 + (f % GRID) * STEP_UM, (f // GRID) * STEP_UM)
         for f in range(GRID * GRID)})
    reader = _SmoothReader(root, meta)

    n = _wellimage.write_well_images(reader, meta, time_point=0)
    assert n == 2, "well ids are written; a non-well region is skipped, as Squid would"
    wells = root / "0" / "mosaic_view" / "wells"
    assert sorted(p.name for p in wells.iterdir()) == ["A1_2um.tiff", "B2_2um.tiff"]

    with tifffile.TiffFile(wells / "A1_2um.tiff") as tif:
        arr = tif.series[0].asarray()
        md = tif.shaped_metadata[0]
    assert arr.ndim == 3 and arr.shape[0] == len(CHANNELS) and arr.dtype == np.uint16
    assert md["axes"] == "CYX"
    assert md["Channel"]["Name"] == CHANNELS
    assert md["well_id"] == "A1"
    assert md["PhysicalSizeX"] == pytest.approx(PX * 2)

    assert _wellimage.write_well_images(reader, meta, time_point=0) == 0, \
        "a second backfill must not rewrite existing files"


def test_the_backfill_downsample_is_the_vendored_area_mean(tmp_path):
    plane = np.arange(36, dtype=np.uint16).reshape(6, 6)
    out = _wellimage.downsample_plane(plane, 3)
    assert out.shape == (2, 2)
    assert out[0, 0] == int(np.mean(plane[:3, :3]))
    assert np.array_equal(_wellimage.downsample_plane(plane, 1), plane)


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0,
                    reason="a chmod-based read-only dir needs a non-root POSIX user")
def test_a_read_only_acquisition_is_left_untouched_without_a_crash(tmp_path):
    root = _acq(tmp_path)
    meta = _meta()
    (root / "0").chmod(0o555)
    try:
        assert _wellimage.write_well_images(_SmoothReader(root, meta), meta,
                                            time_point=0) == 0
        assert not (root / "0" / "mosaic_view").exists()
    finally:
        (root / "0").chmod(0o755)


# --- the plate preview -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _preview(reader, meta, regions):
    from squidxplorer import _workers

    idx = {r: {"rc": (0, i)} for i, r in enumerate(regions)}
    worker = _workers._PreviewWorker(reader, meta, idx, list(regions), cache=None)
    got: list = []
    worker.tileReady.connect(lambda *a: got.append(a))
    worker.run()
    return worker, got


def test_the_plate_preview_seeds_cells_from_well_images_and_skips_the_fov_walk(qapp, tmp_path):
    root = _acq(tmp_path)
    meta = _meta()
    seed = _SmoothReader(root, meta)
    assert _wellimage.write_well_images(seed, meta, time_point=0) == 2

    reader = _SmoothReader(root, meta)
    worker, got = _preview(reader, meta, ["A1", "B2"])
    assert reader.reads == 0, \
        f"the plate view must not walk the FOVs when well images exist ({reader.reads} reads)"
    assert worker.well_image_hits == 2
    assert {a[2] for a in got} == {"A1", "B2"}
    for _ri, _ci, _region, tile, box in got:
        assert box is not None and np.asarray(tile).shape == (len(CHANNELS), box[2], box[3])


def test_the_plate_preview_backfills_an_absent_mosaic_view(qapp, tmp_path):
    root = _acq(tmp_path)
    meta = _meta()
    reader = _SmoothReader(root, meta)
    assert not _wellimage.has_well_images(root, 0)

    _preview(reader, meta, ["A1", "B2"])
    assert _wellimage.has_well_images(root, 0), \
        "a finished preview must leave the acquisition mosaic_view-complete"
    got = _wellimage.downsampled_well(_SmoothReader(root, meta), meta, "A1", CHANNELS[0], 0)
    assert got is not None and got[1] == 2


def test_a_missing_channel_sends_the_well_back_to_the_fov_walk(qapp, tmp_path):
    root = _acq(tmp_path)
    meta = _meta(regions=("A1",))
    _write_squid_well(root, meta, "A1", names=["Fluorescence 405 nm Ex", "Something Else"])

    reader = _SmoothReader(root, meta)
    worker, got = _preview(reader, meta, ["A1"])
    assert worker.well_image_hits == 0
    assert reader.reads > 0, "a file lacking a channel cannot serve the cell"
    assert {a[2] for a in got} == {"A1"}
