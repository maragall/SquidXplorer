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


# ------------------------------------------------------------------ reference: hardlinks

#: The z whose plane is SHARP (random texture) per FOV; every other plane is flat (Tenengrad 0).
_SHARPEST = {0: 2, 1: 1}
_FOCUS_NZ = 3
_FOCUS_REGION = "B2"

_FOCUS_ACQ_YAML = """\
objective:
  pixel_size_um: 0.325
z_stack:
  nz: 3
time_series:
  nt: 1
"""


@pytest.fixture
def focus_dataset(tmp_path):
    """An acquisition whose sharpest z plane per FOV is KNOWN, so the pick is assertable."""
    from .conftest import _YAML

    root = tmp_path / "acq_focus"
    (root / "0").mkdir(parents=True)
    rng = np.random.default_rng(3)
    arrays: dict = {}
    for fov in FOVS:
        for z in range(_FOCUS_NZ):
            for ch in CHANNELS:
                if z == _SHARPEST[fov]:
                    arr = rng.integers(0, 4000, (8, 8), dtype=np.uint16)   # texture: high gradient
                else:
                    arr = np.full((8, 8), 100 + z, dtype=np.uint16)        # flat: zero gradient
                tifffile.imwrite(root / "0" / f"{_FOCUS_REGION}_{fov}_{z}_{ch}.tiff", arr)
                arrays[(fov, z, ch)] = arr
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_FOCUS_ACQ_YAML)
    lines = ["region,x (mm),y (mm),z (mm)"]
    for _z in range(_FOCUS_NZ):
        for fov, x in ((0, 10.0), (1, 10.5)):
            lines.append(f"{_FOCUS_REGION},{x},20.0,")
    (root / "coordinates.csv").write_text("\n".join(lines) + "\n")
    return root, arrays


def test_reference_save_hardlinks_the_chosen_z(focus_dataset, tmp_path):
    root, arrays = focus_dataset
    dst = tmp_path / "reference_out"
    summary = write_acquisition_planes(open_reader(root), "reference", dst)
    assert summary["complete"] and summary["n_fields_written"] == len(FOVS)
    for fov in FOVS:
        for ch in CHANNELS:
            out_f = dst / "0" / f"{_FOCUS_REGION}_{fov}_0_{ch}.tiff"
            src_f = root / "0" / f"{_FOCUS_REGION}_{fov}_{_SHARPEST[fov]}_{ch}.tiff"
            # A second directory entry for the SAME bytes, and of the RIGHT source plane:
            # every channel of one FOV links the one z the focus rule chose.
            assert out_f.stat().st_ino == src_f.stat().st_ino, (fov, ch)
    out = open_reader(dst)
    assert out.metadata["n_z"] == 1
    np.testing.assert_array_equal(out.read(_FOCUS_REGION, 0, CHANNELS[0], 0),
                                  arrays[(0, _SHARPEST[0], CHANNELS[0])])


def test_reference_falls_back_to_a_real_copy_where_links_are_refused(
        focus_dataset, tmp_path, monkeypatch):
    def _refuse(src, dst, **kw):
        raise OSError("this filesystem refuses hardlinks")

    monkeypatch.setattr("os.link", _refuse)
    root, _ = focus_dataset
    dst = tmp_path / "reference_out"
    summary = write_acquisition_planes(open_reader(root), "reference", dst)
    assert summary["complete"]
    out_f = dst / "0" / f"{_FOCUS_REGION}_0_0_{CHANNELS[0]}.tiff"
    src_f = root / "0" / f"{_FOCUS_REGION}_0_{_SHARPEST[0]}_{CHANNELS[0]}.tiff"
    assert out_f.stat().st_ino != src_f.stat().st_ino     # a real copy, not a link
    assert out_f.read_bytes() == src_f.read_bytes()       # ...of the chosen file, byte-identical


def test_reference_over_shared_plane_files_writes_the_chosen_pixels(focus_dataset, tmp_path):
    # A source whose planes share one file (a multi-page stack, an RGB base) cannot be linked
    # under a single-plane name; the save falls back to the engine and still lands the chosen
    # plane's pixels.
    root, arrays = focus_dataset
    reader = open_reader(root)
    shared = root / "0" / f"{_FOCUS_REGION}_0_0_{CHANNELS[0]}.tiff"
    reader.plane_path = lambda region, fov, channel, z_level, time_point=0: shared
    dst = tmp_path / "reference_out"
    summary = write_acquisition_planes(reader, "reference", dst)
    assert summary["complete"]
    out_f = dst / "0" / f"{_FOCUS_REGION}_0_0_{CHANNELS[0]}.tiff"
    assert out_f.stat().st_ino != shared.stat().st_ino
    np.testing.assert_array_equal(tifffile.imread(out_f), arrays[(0, _SHARPEST[0], CHANNELS[0])])


def test_reference_links_every_timepoint_from_its_own_folder(multi_time_point_dataset, tmp_path):
    from .conftest import N_TIME_POINTS, TIME_SERIES_CHANNELS, TIME_SERIES_REGION

    root, _ = multi_time_point_dataset
    dst = tmp_path / "reference_out"
    summary = write_acquisition_planes(open_reader(root), "reference", dst)
    assert summary["complete"]
    for t in range(N_TIME_POINTS):
        for ch in TIME_SERIES_CHANNELS:
            # Constant planes tie at Tenengrad 0 and the rule keeps the FIRST, z 0.
            out_f = dst / str(t) / f"{TIME_SERIES_REGION}_0_0_{ch}.tiff"
            src_f = root / str(t) / f"{TIME_SERIES_REGION}_0_0_{ch}.tiff"
            assert out_f.stat().st_ino == src_f.stat().st_ino, (t, ch)


# ------------------------------------------------------------------ declaration refusals

def test_a_region_operator_is_refused_by_name(squid_dataset, tmp_path):
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="stitch.*whole well"):
        write_acquisition_planes(open_reader(root), "stitch", tmp_path / "out")


# ------------------------------------------------------------------ keep-z (decon-shaped)

def test_a_keep_z_operator_writes_every_plane_under_its_own_index(squid_dataset, tmp_path):
    root, arrays = squid_dataset
    dst = tmp_path / "keepz_out"
    summary = write_acquisition_planes(open_reader(root), "keepz", dst)
    assert summary["complete"] and summary["n_fields_written"] == len(REGIONS) * len(FOVS)
    written = sorted(p.name for p in (dst / "0").iterdir())
    expected = sorted(f"{r}_{f}_{z}_{c}.tiff"
                      for r in REGIONS for f in FOVS for z in range(NZ) for c in CHANNELS)
    assert written == expected                        # z index PRESERVED in every name
    for z in range(NZ):
        out = tifffile.imread(dst / "0" / f"{REGIONS[0]}_0_{z}_{CHANNELS[0]}.tiff")
        np.testing.assert_array_equal(out, arrays[(REGIONS[0], 0, z, CHANNELS[0])])


def test_a_keep_z_save_keeps_the_declared_z_count_and_round_trips(squid_dataset, tmp_path):
    root, arrays = squid_dataset
    dst = tmp_path / "keepz_out"
    write_acquisition_planes(open_reader(root), "keepz", dst)
    # nz NOT rewritten: the output really has every acquired plane.
    assert (dst / "acquisition.yaml").read_text() == (root / "acquisition.yaml").read_text()
    out = open_reader(dst)
    assert out.metadata["n_z"] == NZ and list(out.metadata["z_levels"]) == list(range(NZ))
    plane = out.read(REGIONS[0], 0, CHANNELS[0], NZ - 1)
    np.testing.assert_array_equal(plane, arrays[(REGIONS[0], 0, NZ - 1, CHANNELS[0])])


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


def test_saving_an_operator_that_keeps_z_takes_the_acquisition_format(squid_dataset, tmp_path):
    root, arrays = squid_dataset
    out_dir = tmp_path / f"keepz_{root.name}"
    result = run_operator_once(open_reader(root), operator="keepz", save=True,
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

        # The GUI still NAMES the folder ".hcs" (its acq_format gate does not know the fused
        # writer yet); an explicit out_dir IS the destination, and the write inside it is the
        # stitcher's OME-TIFF acquisition, re-openable.
        fused_dst = tmp_path / f"{root.name}.hcs"
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


def test_the_gate_is_declaration_driven_not_name_driven(squid_dataset):
    root, _ = squid_dataset
    reader = open_reader(root)
    assert acquisition_format_dst(reader, "mip") == root.parent / f"mip_{root.name}"
    assert acquisition_format_dst(reader, "keepz") == root.parent / f"keepz_{root.name}"
    assert acquisition_format_dst(reader, "stitch") is None     # region operator
    assert acquisition_format_dst(reader, "spot") is None       # labels
