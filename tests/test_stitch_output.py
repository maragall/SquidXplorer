"""The fused save: a region operator's mosaics in the stitcher's own OME-TIFF format.

``write_fused_acquisition`` writes one Squid-style tiled OME-TIFF per region (tilefusion's own
exporter — maragall/stitcher parity by construction) laid out as an acquisition
(``ome_tiff/{region}_0.ome.tiff`` + sidecars) so ``open_reader`` re-opens it: a saved stitch
drops straight back into the GUI. Routing is declaration-driven, never name-driven.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from squidxplorer import add_region_operator, open_reader
from squidxplorer._dispatch import run_operator_once
from squidxplorer._fused_output import fused_format_dst, write_fused_acquisition
from squidxplorer._placement import PlacedArray, Placement

from .conftest import CHANNELS, FOVS, NZ, REGIONS

pytest.importorskip("tilefusion")   # the writer IS tilefusion's exporter

_H, _W = 6, 8


def _probe_value(region_index: int, c: int, z: int, t: int = 0) -> int:
    return t * 5000 + region_index * 1000 + c * 100 + z * 10 + 7


def _probe_mosaic(reader, region, fovs):
    """A deterministic full-depth fused mosaic at NATIVE pitch, with a real Placement."""
    meta = reader.metadata
    idx = list(meta["regions"]).index(region)
    n_t, n_z, n_c = int(meta["n_t"]), int(meta["n_z"]), len(meta["channels"])
    out = np.empty((n_t, n_c, n_z, _H, _W), np.uint16)
    for t in range(n_t):
        for c in range(n_c):
            for z in range(n_z):
                out[t, c, z] = _probe_value(idx, c, z, t)
    fovs = tuple(fovs)
    placement = Placement(
        origin_um=(2000.0 + 10.0 * idx, 1000.0), pixel_size_um=float(meta["pixel_size_um"]),
        z_step_um=meta.get("dz_um"), shape=(_H, _W), tile_shape=(4, 4), fovs=fovs,
        offsets_px=((0.0, 0.0),) * len(fovs), origins_px=((0.0, 0.0),) * len(fovs),
        reg_channel=None, reg_t=None)
    return PlacedArray(out, placement)


@pytest.fixture
def fused_probe():
    add_region_operator("fusedprobe", _probe_mosaic)   # conftest restores the registry
    return "fusedprobe"


# ------------------------------------------------------------------ the writer itself

def test_layout_sidecars_and_one_ome_tiff_per_region(squid_dataset, tmp_path, fused_probe):
    root, _ = squid_dataset
    dst = tmp_path / "fused_out"
    summary = write_fused_acquisition(open_reader(root), fused_probe, dst)
    assert summary["complete"] and not summary["stopped"]
    assert summary["n_fields_written"] == len(REGIONS)     # ONE anchor field per region
    for name in ("acquisition.yaml", "acquisition_channels.yaml", "coordinates.csv"):
        assert (dst / name).is_file(), name
    written = sorted(p.name for p in (dst / "ome_tiff").iterdir())
    assert written == sorted(f"{r}_0.ome.tiff" for r in REGIONS)


def test_the_output_reopens_and_the_fused_pixels_round_trip(squid_dataset, tmp_path, fused_probe):
    root, _ = squid_dataset
    dst = tmp_path / "fused_out"
    write_fused_acquisition(open_reader(root), fused_probe, dst)
    out = open_reader(dst)                                  # "drop it back in the GUI"
    meta = out.metadata
    assert meta["regions"] == REGIONS
    assert meta["fovs_per_region"] == {r: [0] for r in REGIONS}
    assert meta["n_z"] == NZ                                # the probe keeps the whole depth
    assert meta["pixel_size_um"] == pytest.approx(0.325)    # native, from the copied sidecar
    channels = [c["name"] for c in meta["channels"]]
    for r_i, region in enumerate(REGIONS):
        for c_i, channel in enumerate(channels):
            for z in range(NZ):
                plane = np.asarray(out.read(region, 0, channel, z))
                assert plane.shape == (_H, _W)
                np.testing.assert_array_equal(
                    plane, np.full((_H, _W), _probe_value(r_i, c_i, z), np.uint16))


def test_coordinates_carry_each_mosaic_origin(squid_dataset, tmp_path, fused_probe):
    root, _ = squid_dataset
    dst = tmp_path / "fused_out"
    write_fused_acquisition(open_reader(root), fused_probe, dst)
    positions = open_reader(dst).metadata["fov_positions_um"]
    for r_i, region in enumerate(REGIONS):
        x_um, y_um = positions[(region, 0)]
        assert x_um == pytest.approx(1000.0)                # placement.origin_um is (y, x)
        assert y_um == pytest.approx(2000.0 + 10.0 * r_i)


def test_a_stopped_run_stays_under_the_partial_name(squid_dataset, tmp_path, fused_probe):
    root, _ = squid_dataset
    dst = tmp_path / "fused_out"
    summary = write_fused_acquisition(open_reader(root), fused_probe, dst, stop=lambda: True)
    assert summary["stopped"] and not summary["complete"]
    assert not dst.exists() and summary["path"].endswith(".partial")


# ------------------------------------------------------------------ stitch itself

def test_stitch_saves_the_mosaic_of_record_and_reopens(squid_dataset, tmp_path):
    from squidxplorer._stitch import stitch_region

    root, _ = squid_dataset
    reader = open_reader(root)
    kwargs = {"register": False, "correct_illumination": False}
    expected = stitch_region(reader, REGIONS[0], FOVS, **kwargs)   # (1, C, 1, H, W), native
    dst = tmp_path / "stitch_out"
    summary = write_fused_acquisition(reader, "stitch", dst, operator_kwargs=kwargs)
    assert summary["complete"] and summary["n_fields_written"] == len(REGIONS)
    out = open_reader(dst)
    meta = out.metadata
    assert meta["n_z"] == 1                                  # the default z_operator collapses z
    # ...and the copied yaml SAYS so (the declared z count is rewritten like the mip save's).
    declared = yaml.safe_load((dst / "acquisition.yaml").read_text())
    assert declared["z_stack"]["nz"] == 1
    channels = [c["name"] for c in meta["channels"]]
    for c_i, channel in enumerate(channels):
        plane = np.asarray(out.read(REGIONS[0], 0, channel, 0))
        np.testing.assert_array_equal(plane, np.asarray(expected)[0, c_i, 0])


# ------------------------------------------------------------------ declaration refusals

def test_a_per_fov_operator_is_refused_by_name(squid_dataset, tmp_path):
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="mip.*per FOV"):
        write_fused_acquisition(open_reader(root), "mip", tmp_path / "out")


def test_a_copy_saving_operator_is_refused_by_name(squid_dataset, tmp_path):
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="register.*COPY"):
        write_fused_acquisition(open_reader(root), "register", tmp_path / "out")


def test_an_unknown_parameter_is_refused_before_any_directory(
        squid_dataset, tmp_path, fused_probe):
    root, _ = squid_dataset
    dst = tmp_path / "fused_out"
    with pytest.raises((TypeError, ValueError)):
        write_fused_acquisition(open_reader(root), fused_probe, dst,
                                operator_kwargs={"no_such_knob": 1})
    assert not dst.exists() and not dst.with_name(dst.name + ".partial").exists()


def test_a_decimated_look_is_refused_by_name(squid_dataset, tmp_path):
    def _look(reader, region, fovs):
        mosaic = _probe_mosaic(reader, region, fovs)
        placement = mosaic.placement
        from dataclasses import replace
        return PlacedArray(np.asarray(mosaic),
                           replace(placement, pixel_size_um=placement.pixel_size_um * 2))

    add_region_operator("fusedlook", _look)
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="fusedlook.*native"):
        write_fused_acquisition(open_reader(root), "fusedlook", tmp_path / "out")


def test_a_bare_array_stream_is_refused_by_name(squid_dataset, tmp_path):
    add_region_operator("fusedbare",
                        lambda reader, region, fovs: np.zeros((1, 2, 2, _H, _W), np.uint16))
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="fusedbare.*Placement"):
        write_fused_acquisition(open_reader(root), "fusedbare", tmp_path / "out")


# ------------------------------------------------------------------ the dispatch routing

@pytest.mark.parametrize("operator, routes_fused", [
    ("mip", False),          # per-FOV: the acquisition-format writer's
    ("planes", False),       # per-FOV, keeps z (identity fixture): still acquisition-format
    ("blob", False),         # labels (fixture)
    ("register", False),     # saves a registered COPY; its routing is operator_saves_copy's
    ("stitch", True),
])
def test_the_gate_is_declaration_driven_not_name_driven(squid_dataset, operator, routes_fused,
                                                        identity_operator, blob_operator):
    root, _ = squid_dataset
    reader = open_reader(root)
    expected = root.parent / f"{operator}_{root.name}" if routes_fused else None
    assert fused_format_dst(reader, operator) == expected


def test_dispatch_routes_a_region_save_to_the_fused_writer(squid_dataset, fused_probe):
    root, _ = squid_dataset
    result = run_operator_once(open_reader(root), operator=fused_probe, save=True,
                               owed=len(REGIONS), out_dir=None)   # n_fovs: the loop's default
    dst = root.parent / f"{fused_probe}_{root.name}"
    assert dst.is_dir() and result.out_path == str(dst)
    assert result.outcome == "ok" and result.landed == len(REGIONS)
    assert not (dst / "plate.ome.zarr").exists()
    assert (dst / "ome_tiff" / f"{REGIONS[0]}_0.ome.tiff").is_file()


def test_dispatch_honours_the_chosen_folder(squid_dataset, tmp_path, fused_probe):
    root, _ = squid_dataset
    out_dir = tmp_path / f"{fused_probe}_{root.name}"
    result = run_operator_once(open_reader(root), operator=fused_probe, save=True,
                               owed=len(REGIONS), out_dir=out_dir, n_fovs=None)
    assert out_dir.is_dir() and result.out_path == str(out_dir)
    assert open_reader(out_dir).metadata["regions"] == REGIONS
