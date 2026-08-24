"""The acquisition-format save: one projected plane per FOV, bit-exact, beside the source.

A z-reducing per-FOV operator's SAVE writes the same format as its input acquisition into
``<operator>_<folder>``, so the output is native resolution (never a fused, decimated paste)
and findable without knowing NGFF. Everything else keeps the OME-Zarr writer.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from squidxplorer import open_reader
from squidxplorer._acq_output import acquisition_format_dst, write_acquisition_planes
from squidxplorer._dispatch import run_operator_once

from .conftest import CHANNELS, FOVS, NZ, REGIONS


def _mip(arrays, region, fov, channel):
    return np.max(np.stack([arrays[(region, fov, z, channel)] for z in range(NZ)]), axis=0)


# ------------------------------------------------------------------ the writer itself

def test_output_layout_sidecars_and_one_file_per_fov_channel(squid_dataset, tmp_path):
    root, _ = squid_dataset
    dst = tmp_path / "mip_out"
    summary = write_acquisition_planes(open_reader(root), "mip", dst)
    assert summary["complete"] and not summary["stopped"]
    assert summary["n_fields_written"] == len(REGIONS) * len(FOVS)
    # root sidecars copied, image files never copied
    for name in ("acquisition.yaml", "acquisition_channels.yaml",
                 "acquisition parameters.json", "coordinates.csv"):
        assert (dst / name).is_file(), name
    written = sorted(p.name for p in (dst / "0").iterdir())
    expected = sorted(f"{r}_{f}_0_{c}.tiff" for r in REGIONS for f in FOVS for c in CHANNELS)
    assert written == expected  # z index 0 in every name, extension matches the .tiff input


def test_pixels_are_the_exact_mip_at_native_shape(squid_dataset, tmp_path):
    root, arrays = squid_dataset
    dst = tmp_path / "mip_out"
    write_acquisition_planes(open_reader(root), "mip", dst)
    for region in REGIONS:
        for fov in FOVS:
            for channel in CHANNELS:
                out = tifffile.imread(dst / "0" / f"{region}_{fov}_0_{channel}.tiff")
                want = _mip(arrays, region, fov, channel)
                assert out.dtype == want.dtype and out.shape == want.shape
                np.testing.assert_array_equal(out, want)


def test_the_output_round_trips_through_open_reader(squid_dataset, tmp_path):
    root, arrays = squid_dataset
    dst = tmp_path / "mip_out"
    write_acquisition_planes(open_reader(root), "mip", dst)
    src_meta = open_reader(root).metadata
    out = open_reader(dst)
    meta = out.metadata
    assert meta["n_z"] == 1 and list(meta["z_levels"]) == [0]
    assert meta["regions"] == src_meta["regions"]
    assert meta["fov_positions_um"] == src_meta["fov_positions_um"]
    plane = out.read(REGIONS[0], 0, CHANNELS[0], 0)
    np.testing.assert_array_equal(plane, _mip(arrays, REGIONS[0], 0, CHANNELS[0]))


def test_every_timepoint_is_written_with_its_own_sidecars(multi_time_point_dataset, tmp_path):
    from .conftest import (N_TIME_POINTS, TIME_SERIES_CHANNELS, TIME_SERIES_FOV,
                           TIME_SERIES_NZ, TIME_SERIES_REGION, time_series_pixel_value)

    root, _ = multi_time_point_dataset
    dst = tmp_path / "mip_out"
    write_acquisition_planes(open_reader(root), "mip", dst)
    for t in range(N_TIME_POINTS):
        assert (dst / str(t) / "coordinates.csv").is_file()
        for c_i, channel in enumerate(TIME_SERIES_CHANNELS):
            name = f"{TIME_SERIES_REGION}_{TIME_SERIES_FOV}_0_{channel}.tiff"
            out = tifffile.imread(dst / str(t) / name)
            want = max(time_series_pixel_value(t, z, c_i) for z in range(TIME_SERIES_NZ))
            np.testing.assert_array_equal(out, np.full((4, 4), want, dtype=np.uint16))
    reopened = open_reader(dst).metadata
    assert reopened["n_t"] == N_TIME_POINTS and reopened["n_z"] == 1


def test_a_uint8_bmp_acquisition_keeps_its_own_extension(tmp_path):
    from PIL import Image

    root = tmp_path / "acq_bmp"
    (root / "0").mkdir(parents=True)
    rng = np.random.default_rng(7)
    stack = rng.integers(0, 255, (2, 8, 8), dtype=np.uint8)
    for z in range(2):
        Image.fromarray(stack[z]).save(root / "0" / f"A1_0_{z}_BF_LED_matrix_full.bmp")
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.418\nz_stack:\n  nz: 2\ntime_series:\n  nt: 1\n")
    (root / "acquisition_channels.yaml").write_text(
        "channels:\n- name: BF LED matrix full\n  display_color: '#FFFFFF'\n")
    dst = tmp_path / "mip_bmp"
    write_acquisition_planes(open_reader(root), "mip", dst)
    out_path = dst / "0" / "A1_0_0_BF_LED_matrix_full.bmp"
    assert out_path.is_file(), sorted(p.name for p in (dst / "0").iterdir())
    with Image.open(out_path) as img:
        np.testing.assert_array_equal(np.asarray(img), stack.max(axis=0))
    assert open_reader(dst).metadata["n_z"] == 1  # the copied yaml's nz was rewritten to 1


def test_the_copied_yaml_rewrites_only_the_z_count(squid_dataset, tmp_path):
    root, _ = squid_dataset
    dst = tmp_path / "mip_out"
    write_acquisition_planes(open_reader(root), "mip", dst)
    src_lines = (root / "acquisition.yaml").read_text().splitlines()
    dst_lines = (dst / "acquisition.yaml").read_text().splitlines()
    changed = [(a, b) for a, b in zip(src_lines, dst_lines) if a != b]
    assert changed == [("  nz: 2", "  nz: 1")]


def test_a_stopped_run_stays_under_the_partial_name(squid_dataset, tmp_path):
    root, _ = squid_dataset
    dst = tmp_path / "mip_out"
    summary = write_acquisition_planes(open_reader(root), "mip", dst, stop=lambda: True)
    assert summary["stopped"] and not summary["complete"]
    assert not dst.exists()
    assert summary["path"].endswith(".partial")


# (The reference/hardlink block — the focus_dataset fixture and the four
# _link_selected_planes tests — was deleted 2026-08-24 with the shelved `reference`
# operator and the writer's select_index arm. Git history reinstates.)


# ------------------------------------------------------------------ keep-z (decon-shaped)

def test_a_keep_z_operator_writes_every_plane_under_its_own_index(squid_dataset, tmp_path,
                                                                 identity_operator):
    root, arrays = squid_dataset
    dst = tmp_path / "planes_out"
    summary = write_acquisition_planes(open_reader(root), identity_operator, dst)
    assert summary["complete"] and summary["n_fields_written"] == len(REGIONS) * len(FOVS)
    written = sorted(p.name for p in (dst / "0").iterdir())
    expected = sorted(f"{r}_{f}_{z}_{c}.tiff"
                      for r in REGIONS for f in FOVS for z in range(NZ) for c in CHANNELS)
    assert written == expected                        # z index PRESERVED in every name
    for z in range(NZ):
        out = tifffile.imread(dst / "0" / f"{REGIONS[0]}_0_{z}_{CHANNELS[0]}.tiff")
        np.testing.assert_array_equal(out, arrays[(REGIONS[0], 0, z, CHANNELS[0])])


def test_a_keep_z_save_keeps_the_declared_z_count_and_round_trips(squid_dataset, tmp_path,
                                                                 identity_operator):
    root, arrays = squid_dataset
    dst = tmp_path / "planes_out"
    write_acquisition_planes(open_reader(root), identity_operator, dst)
    # nz NOT rewritten: the output really has every acquired plane.
    assert (dst / "acquisition.yaml").read_text() == (root / "acquisition.yaml").read_text()
    out = open_reader(dst)
    assert out.metadata["n_z"] == NZ and list(out.metadata["z_levels"]) == list(range(NZ))
    plane = out.read(REGIONS[0], 0, CHANNELS[0], NZ - 1)
    np.testing.assert_array_equal(plane, arrays[(REGIONS[0], 0, NZ - 1, CHANNELS[0])])


def test_a_keeps_depth_z_consumer_saves_every_plane_same_size_as_input(squid_dataset, tmp_path):
    """decon3d's save shape (Julio 2026-08-21): the output acquisition is the SAME SIZE as
    the input — every processed z under its own index, nz untouched."""
    from squidxplorer import add_operator

    def flipz(planes):
        return np.asarray(list(planes))[::-1]

    flipz.keeps_depth = True
    add_operator("acqtest_zdepth", flipz, consumes=frozenset({"z"}))
    root, arrays = squid_dataset
    dst = tmp_path / "out"
    manifest = write_acquisition_planes(open_reader(root), "acqtest_zdepth", dst)
    assert manifest["complete"]

    out = open_reader(dst)
    meta = out.metadata
    assert meta["n_z"] == NZ, "the declared z count must survive a depth-keeping save"
    for k, z in enumerate(reversed(list(range(NZ)))):
        np.testing.assert_array_equal(out.read(REGIONS[0], 0, CHANNELS[0], k),
                                      arrays[(REGIONS[0], 0, z, CHANNELS[0])])


def test_a_labels_producer_is_refused_by_name(squid_dataset, tmp_path):
    from squidxplorer import add_operator

    add_operator("acqtest_zlabels", lambda planes: next(iter(planes)),
                 consumes=frozenset({"z"}), produces="labels")
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="acqtest_zlabels.*labels"):
        write_acquisition_planes(open_reader(root), "acqtest_zlabels", tmp_path / "out")


def test_an_unknown_parameter_is_refused_before_any_directory(squid_dataset, tmp_path):
    root, _ = squid_dataset
    dst = tmp_path / "mip_out"
    with pytest.raises((TypeError, ValueError)):
        write_acquisition_planes(open_reader(root), "mip", dst,
                                 operator_kwargs={"no_such_knob": 1})
    assert not dst.exists() and not dst.with_name(dst.name + ".partial").exists()


# ------------------------------------------------------------------ the dispatch routing

def test_saving_mip_honours_the_chosen_folder(squid_dataset, tmp_path):
    # An explicit out_dir (the GUI's chosen folder, the CLI's --out) IS the destination.
    root, arrays = squid_dataset
    out_dir = tmp_path / f"mip_{root.name}"
    result = run_operator_once(open_reader(root), operator="mip", save=True,
                               owed=len(REGIONS), out_dir=out_dir, n_fovs=None)
    assert out_dir.is_dir(), "the save must land where the caller aimed it"
    assert result.out_path == str(out_dir)
    assert result.outcome == "ok" and result.landed == len(REGIONS) * len(FOVS)
    assert not (out_dir / "plate.ome.zarr").exists()
    out = tifffile.imread(out_dir / "0" / f"{REGIONS[0]}_0_0_{CHANNELS[0]}.tiff")
    np.testing.assert_array_equal(out, _mip(arrays, REGIONS[0], 0, CHANNELS[0]))


def test_saving_mip_without_a_destination_defaults_beside_the_source(squid_dataset):
    root, _ = squid_dataset
    result = run_operator_once(open_reader(root), operator="mip", save=True,
                               owed=len(REGIONS), out_dir=None, n_fovs=None)
    dst = root.parent / f"mip_{root.name}"
    assert dst.is_dir() and result.out_path == str(dst)


def test_saving_an_operator_that_keeps_z_takes_the_acquisition_format(squid_dataset, tmp_path,
                                                                     identity_operator):
    root, arrays = squid_dataset
    out_dir = tmp_path / f"planes_{root.name}"
    result = run_operator_once(open_reader(root), operator=identity_operator, save=True,
                               owed=len(REGIONS), out_dir=out_dir, n_fovs=None)
    assert out_dir.is_dir() and result.out_path == str(out_dir)
    assert not (out_dir / "plate.ome.zarr").exists()
    out = tifffile.imread(out_dir / "0" / f"{REGIONS[0]}_0_{NZ - 1}_{CHANNELS[0]}.tiff")
    np.testing.assert_array_equal(out, arrays[(REGIONS[0], 0, NZ - 1, CHANNELS[0])])


def test_the_window_save_toggle_lands_both_formats(qapp, squid_dataset, tmp_path):
    """The in-window save: mip lands the acquisition format at the CHOSEN folder, stitch the
    stitcher's fused OME-TIFF — both through PlateWindow.run_operator, the toggle's real path."""
    pytest.importorskip("tilefusion")
    import squidxplorer._viewer as V

    from .test_viewer import _drain_until

    root, arrays = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    try:
        mip_dst = tmp_path / f"mip_{root.name}"
        win.run_operator("mip", out_parent=str(tmp_path))
        n_files = len(REGIONS) * len(FOVS) * len(CHANNELS)
        assert _drain_until(
            qapp, lambda: mip_dst.is_dir()
            and len(list((mip_dst / "0").glob("*.tiff"))) == n_files
            and (win._worker is None or not win._worker.isRunning()))
        out = tifffile.imread(mip_dst / "0" / f"{REGIONS[0]}_0_0_{CHANNELS[0]}.tiff")
        np.testing.assert_array_equal(out, _mip(arrays, REGIONS[0], 0, CHANNELS[0]))
        assert open_reader(mip_dst).metadata["n_z"] == 1

        # The GUI's acq_format gate knows the fused writer: a stitch save names
        # <operator>_<acq> and writes the stitcher's OME-TIFF acquisition, re-openable.
        fused_dst = tmp_path / f"stitch_{root.name}"
        win.run_operator("stitch", out_parent=str(tmp_path),
                         operator_kwargs={"register": False, "correct_illumination": False})
        assert _drain_until(
            qapp, lambda: (fused_dst / "ome_tiff" / f"{REGIONS[0]}_0.ome.tiff").is_file()
            and (win._worker is None or not win._worker.isRunning()), timeout=120)
        assert not (fused_dst / "plate.ome.zarr").exists()
        assert open_reader(fused_dst).metadata["regions"] == REGIONS
    finally:
        win._stop_worker()
        win.close()


def test_the_gate_is_declaration_driven_not_name_driven(squid_dataset, identity_operator,
                                                        blob_operator):
    root, _ = squid_dataset
    reader = open_reader(root)
    assert acquisition_format_dst(reader, "mip") == root.parent / f"mip_{root.name}"
    assert (acquisition_format_dst(reader, identity_operator)
            == root.parent / f"{identity_operator}_{root.name}")
    assert acquisition_format_dst(reader, "stitch") is None     # region operator
    assert acquisition_format_dst(reader, blob_operator) is None   # labels
