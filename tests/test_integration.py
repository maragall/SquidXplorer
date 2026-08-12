"""Cross-slot integration tests — the "cross commit" surface for SquidXplorer."""

from __future__ import annotations

import time
import tracemalloc
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

import tensorstore as ts
import tifffile

from squidxplorer import build_montage, open_reader, project_plate, project_well, select_fovs, write_plate
from squidxplorer._output import plate_metadata, split_well, write_from_stream
from tests.test_performance import benchmark_single_well


def _read_zarr_array(path) -> np.ndarray:
    """Read a zarr v3 array back the way ndviewer_light does — via tensorstore."""
    store = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}}, open=True
    ).result()
    return np.asarray(store[...].read().result())

from tests.conftest import SIM_1536WP  # noqa: E402,F401  (path constant; fixture is auto-discovered)


def _assert_well_matches_np_max(reader, region, fov):
    """project_well(region, fov) == np.max over z_levels of the reader's own exact reads."""
    meta = reader.metadata
    out = project_well(reader, region, fov)
    assert out.shape == (meta["n_t"], len(meta["channels"]), 1, *meta["frame_shape"])
    assert out.dtype == meta["dtype"]
    for t in range(meta["n_t"]):
        for c_i, ch in enumerate(c["name"] for c in meta["channels"]):
            ref = np.max(
                np.stack(
                    [reader.read(region, fov, ch, z, t) for z in meta["z_levels"]]
                ),
                axis=0,
            )
            np.testing.assert_array_equal(out[t, c_i, 0], ref)


@pytest.mark.integration
def test_sim1536_metadata_sanity(sim_1536wp):
    meta = open_reader(sim_1536wp).metadata
    assert len(meta["regions"]) == 1536
    assert meta["regions"][0] == "A1" and meta["regions"][-1] == "AF48"
    assert all(fovs == [0] for fovs in meta["fovs_per_region"].values())
    assert len(meta["channels"]) == 4
    assert meta["dtype"] == np.uint16
    assert meta["frame_shape"] == (2084, 2084)
    xs = sorted({round(v[0], 1) for v in meta["fov_positions_um"].values()})
    ys = sorted({round(v[1], 1) for v in meta["fov_positions_um"].values()})
    assert (len(xs), len(ys)) == (48, 32)
    assert round(xs[1] - xs[0]) == 2250 and round(ys[1] - ys[0]) == 2250


@pytest.mark.filterwarnings("ignore:Recorded Nz")
@pytest.mark.integration
def test_sim1536_select_one_fov_per_well(sim_1536wp):
    meta = open_reader(sim_1536wp).metadata
    wells = select_fovs(meta, n_fovs=1)
    assert len(wells) == 1536
    assert all(fovs == [0] for fovs in wells.values())


@pytest.mark.filterwarnings("ignore:Recorded Nz")
@pytest.mark.integration
def test_sim1536_project_sampled_wells_pixel_exact(sim_1536wp):
    reader = open_reader(sim_1536wp)
    regions = reader.metadata["regions"]
    for region in (
        regions[0],
        regions[len(regions) // 2],
        regions[-1],
    ):
        _assert_well_matches_np_max(reader, region, 0)


@pytest.mark.integration
def test_real_acquisition_pipeline_end_to_end(real_dataset):
    reader = open_reader(real_dataset)
    meta = reader.metadata
    assert meta["regions"]
    assert meta["z_levels"]
    assert meta["channels"]
    wells = select_fovs(meta, n_fovs=1)
    assert set(wells) == set(meta["regions"])
    region = meta["regions"][0]
    _assert_well_matches_np_max(reader, region, wells[region][0])


@pytest.mark.integration
def test_real_acquisition_mip_actually_combines_z(real_dataset):
    reader = open_reader(real_dataset)
    meta = reader.metadata
    region = meta["regions"][0]
    fov = meta["fovs_per_region"][region][0]
    z_levels = meta["z_levels"]
    out = project_well(reader, region, fov)
    for t in range(meta["n_t"]):
        for c_i, ch in enumerate(c["name"] for c in meta["channels"]):
            mip = out[t, c_i, 0]
            slices = [reader.read(region, fov, ch, z, t) for z in z_levels]
            for s in slices:
                assert (mip >= s).all()
            assert np.array_equal(mip, np.max(np.stack(slices), axis=0))
            if len(z_levels) > 1:
                assert all(not np.array_equal(mip, s) for s in slices)
                assert (mip > np.stack(slices).min(axis=0)).any()


_SUBSET = 24


def _first_n_projected(reader, n, **kw):
    """Drain the first *n* wells from project_plate into {(region, fov): image}."""
    return {(r, f): img for r, f, img in islice(project_plate(reader, **kw), n)}


@pytest.mark.filterwarnings("ignore:Recorded Nz")
@pytest.mark.integration
def test_ima188_sim1536_parallel_pixel_identical(sim_1536wp):
    reader = open_reader(sim_1536wp)
    projected = _first_n_projected(reader, 6, workers=8)
    assert projected, "engine yielded no wells"
    for (region, fov), img in projected.items():
        np.testing.assert_array_equal(img, project_well(reader, region, fov))


@pytest.mark.filterwarnings("ignore:Recorded Nz")
@pytest.mark.integration
def test_ima188_sim1536_scaling_measured_no_regression(sim_1536wp, capsys):
    import threading
    from unittest import mock

    from squidxplorer import _engine

    reader = open_reader(sim_1536wp)
    regions = reader.metadata["regions"]
    project_well(reader, regions[50], 0)

    def _peak_concurrency(workers):
        real = _engine.project_well
        lock = threading.Lock()
        state = {"cur": 0, "peak": 0}

        def counting(*args, **kwargs):
            with lock:
                state["cur"] += 1
                state["peak"] = max(state["peak"], state["cur"])
            try:
                time.sleep(0.005)
                return real(*args, **kwargs)
            finally:
                with lock:
                    state["cur"] -= 1

        with mock.patch.object(_engine, "project_well", counting):
            produced = _first_n_projected(reader, _SUBSET, workers=workers)
        return state["peak"], produced

    peak_1, got1 = _peak_concurrency(1)
    peak_8, got8 = _peak_concurrency(8)

    with capsys.disabled():
        print(
            f"\n[IMA-188] {_SUBSET} wells: peak concurrent project_well workers=1 -> {peak_1}, "
            f"workers=8 -> {peak_8}. Gate is on concurrency, not wall clock (warm cache is "
            f"bandwidth-bound; the real speedup needs cold/real storage, Decision C)."
        )
    plate = {(r, f) for r in regions for f in reader.metadata["fovs_per_region"][r]}
    assert len(got8) == len(got1) == _SUBSET, "the engine yielded fewer wells than asked"
    assert set(got8) <= plate and set(got1) <= plate, "the engine yielded a well not on the plate"
    assert peak_1 == 1, f"single-thread engine ran {peak_1} wells at once — expected 1"
    assert peak_8 >= 2, f"8-worker engine peaked at {peak_8} concurrent wells — pool serialized"


@pytest.mark.filterwarnings("ignore:Recorded Nz")
@pytest.mark.integration
def test_ima188_sim1536_memory_bounded_by_workers_not_plate(sim_1536wp):
    reader = open_reader(sim_1536wp)
    base = benchmark_single_well(reader, reader.metadata["regions"][0], 0)
    workers = 4

    tracemalloc.start()
    for _ in islice(project_plate(reader, workers=workers), 12):
        pass
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert peak < (workers + 2) * (base["result_bytes"] + 6 * base["plane_bytes"])


@pytest.mark.integration
def test_ima188_real_parallel_pixel_identical(real_dataset):
    reader = open_reader(real_dataset)
    projected = _first_n_projected(reader, 4, workers=4)
    assert projected
    for (region, fov), img in projected.items():
        np.testing.assert_array_equal(img, project_well(reader, region, fov))


@pytest.mark.integration
def test_ima188_real_projector_registry_swap_end_to_end(real_dataset):
    reader = open_reader(real_dataset)
    for region, fov, img in islice(project_plate(reader, workers=4, projector="mip"), 3):
        np.testing.assert_array_equal(img, project_well(reader, region, fov))


def _ref_projected(reader, regions):
    """{region: project_well(...)} computed independently (MIP) for pixel-exact comparison."""
    meta = reader.metadata
    return {r: project_well(reader, r, meta["fovs_per_region"][r][0]) for r in regions}


@pytest.mark.integration
def test_ima184_real_plate_roundtrip(real_dataset, tmp_path):
    core = pytest.importorskip("ndviewer_light.core")
    reader = open_reader(real_dataset)
    meta = reader.metadata

    manifest = write_plate(reader, tmp_path, n_fovs=1, workers=4, tiff=True)
    assert manifest["n_fields_written"] == len(meta["regions"])

    fovs, structure = core.discover_zarr_v3_fovs(tmp_path)
    assert structure == "hcs_plate"
    assert {f["region"] for f in fovs} == set(meta["regions"])

    from tests.ngff_check import assert_valid_ngff_plate

    assert_valid_ngff_plate(tmp_path / "plate.ome.zarr")

    refs = _ref_projected(reader, meta["regions"])
    ch_names = [c["name"] for c in meta["channels"]]
    for region, ref in refs.items():
        row, col = split_well(region)
        fov = meta["fovs_per_region"][region][0]
        np.testing.assert_array_equal(_read_zarr_array(tmp_path / "plate.ome.zarr" / row / col / "0" / "0"), ref)
        for c_i, ch in enumerate(ch_names):
            plane = tifffile.imread(tmp_path / "tiff" / "0" / f"{region}_{fov}_0_{ch}.tiff")
            assert plane.dtype == meta["dtype"]
            np.testing.assert_array_equal(plane, ref[0, c_i, 0])


@pytest.mark.integration
def test_ima184_real_colors_match_reader(real_dataset, tmp_path):
    import json

    reader = open_reader(real_dataset)
    meta = reader.metadata
    write_plate(reader, tmp_path, n_fovs=1, workers=4, tiff=False)

    region = meta["regions"][0]
    row, col = split_well(region)
    field = tmp_path / "plate.ome.zarr" / row / col / "0"
    omero = json.loads((field / "zarr.json").read_text())["attributes"]["ome"]["omero"]
    got = [(c["label"], c["color"]) for c in omero["channels"]]
    want = [(c["display_name"], c["display_color"].lstrip("#")) for c in meta["channels"]]
    assert got == want


@pytest.mark.integration
def test_ima184_real_opens_in_zarr_python(real_dataset, tmp_path):
    zarr = pytest.importorskip("zarr")
    reader = open_reader(real_dataset)
    meta = reader.metadata
    write_plate(reader, tmp_path, n_fovs=1, workers=4, tiff=False)

    grp = zarr.open_group(str(tmp_path / "plate.ome.zarr"), mode="r")
    plate = grp.attrs["ome"]["plate"]
    assert len(plate["wells"]) == len(meta["regions"])
    region = meta["regions"][0]
    row, col = split_well(region)
    arr = grp[f"{row}/{col}/0/0"]
    assert tuple(arr.shape) == (meta["n_t"], len(meta["channels"]), 1, *meta["frame_shape"])


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:Recorded Nz")
def test_ima184_sim1536_plate_metadata_scales(sim_1536wp):
    meta = open_reader(sim_1536wp).metadata
    plate = plate_metadata(meta["regions"], field_count=1)["plate"]
    assert len(plate["wells"]) == 1536
    for well, region in zip(plate["wells"], meta["regions"]):
        assert well["path"].replace("/", "") == region


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:Recorded Nz")
def test_ima184_sim1536_streamed_subset(sim_1536wp, tmp_path):
    core = pytest.importorskip("ndviewer_light.core")
    reader = open_reader(sim_1536wp)

    picked = list(islice(project_plate(reader, n_fovs=1, workers=4), 4))
    assert len(picked) == 4
    submeta = {
        **reader.metadata,
        "regions": [r for r, _, _ in picked],
        "fovs_per_region": {r: [f] for r, f, _ in picked},
    }
    write_from_stream(submeta, iter(picked), tmp_path, n_fovs=1, tiff=False)

    fovs, structure = core.discover_zarr_v3_fovs(tmp_path)
    assert structure == "hcs_plate"
    assert {f["region"] for f in fovs} == {r for r, _, _ in picked}
    for region, fov, img in picked:
        row, col = split_well(region)
        np.testing.assert_array_equal(_read_zarr_array(tmp_path / "plate.ome.zarr" / row / col / "0" / "0"), img)


import json as _json  # noqa: E402  (kept local to this section's helpers)


def _montage_wells(sidecar_path):
    return {w["well_id"]: w for w in _json.loads(Path(sidecar_path).read_text())["wells"]}


@pytest.mark.integration
def test_ima185_real_montage_enumerates_and_renders_all_wells(real_dataset, tmp_path):
    from PIL import Image

    reader = open_reader(real_dataset)
    meta = reader.metadata
    write_plate(reader, tmp_path, n_fovs=1, workers=4, tiff=False)

    manifest = build_montage(tmp_path, cell_px=64)

    assert manifest["n_wells"] == len(meta["regions"])
    wells = _montage_wells(manifest["sidecar"])
    assert set(wells) == set(meta["regions"])

    n_rows, n_cols = manifest["grid"]
    rgb = np.asarray(Image.open(manifest["montage"]))
    assert rgb.shape == (n_rows * 64, n_cols * 64, 3)

    assert rgb.max() > 0
    for w in wells.values():
        cell = rgb[w["y0"] : w["y1"], w["x0"] : w["x1"]]
        assert cell.sum() > 0, f"well {w['well_id']} rendered fully black"

    side = _json.loads(Path(manifest["sidecar"]).read_text())
    assert [c["color"] for c in side["channels"]] == [
        c["display_color"].lstrip("#") for c in meta["channels"]
    ]


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:Recorded Nz")
def test_ima185_sim1536_montage_real_seam_subset(sim_1536wp, tmp_path):
    from PIL import Image

    reader = open_reader(sim_1536wp)
    picked = list(islice(project_plate(reader, n_fovs=1, workers=4), 6))
    submeta = {
        **reader.metadata,
        "regions": [r for r, _, _ in picked],
        "fovs_per_region": {r: [f] for r, f, _ in picked},
    }
    write_from_stream(submeta, iter(picked), tmp_path, n_fovs=1, tiff=False)

    manifest = build_montage(tmp_path, cell_px=48)

    assert manifest["n_wells"] == len(picked)
    assert set(_montage_wells(manifest["sidecar"])) == {r for r, _, _ in picked}
    n_rows, n_cols = manifest["grid"]
    rgb = np.asarray(Image.open(manifest["montage"]))
    assert rgb.shape == (n_rows * 48, n_cols * 48, 3)
    assert rgb.max() > 0


_STITCHER_PARITY_SET = Path(
    "/Users/julioamaragall/Downloads/"
    "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy"
)


@pytest.mark.integration
@pytest.mark.parametrize("region", ["manual0", "manual1"])
def test_stitch_geometry_matches_maragall_stitcher_on_a_real_zstack(region):
    if not _STITCHER_PARITY_SET.is_dir():
        pytest.skip(f"parity acquisition not on this machine: {_STITCHER_PARITY_SET}")
    pytest.importorskip("tilefusion")
    from tilefusion.core import TileFusion

    from squidxplorer import stitch_region

    tf = TileFusion(_STITCHER_PARITY_SET, output_path="/tmp/parity_never_written.ome.zarr",
                    region=region, channel_to_use=0, blend_pixels=(0, 0))
    try:
        tf.refine_tile_positions_with_cross_correlation(
            downsample_factors=tf.downsample_factors, ch_idx=tf.channel_to_use)
        tf.optimize_shifts(method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=2.0,
                           iterative=True)
        reference = np.asarray(tf.global_offsets, dtype=np.float64)
        ref_fovs = [t[1] for t in tf._tile_identifiers]
    finally:
        tf.close()

    reader = open_reader(_STITCHER_PARITY_SET)
    fovs = list(reader.metadata["fovs_per_region"][region])
    assert fovs == ref_fovs, "FOV ordering differs; the offsets are not comparable row-wise"

    geometry: dict = {}
    stitch_region(reader, region, fovs, channels=[0], correct_illumination=False,
                  geometry=geometry)
    ours = np.asarray(geometry["offsets_px"], dtype=np.float64)

    np.testing.assert_allclose(ours, reference, rtol=0, atol=1e-9)

    placement = geometry["placement"]
    assert placement.reg_z == reader.metadata["n_z"] // 2
    assert placement.registered and placement.illumination_corrected is False
