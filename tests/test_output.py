"""IMA-184 output writer — clean-room unit tests (no reader, no data on disk).

Drives ``write_from_stream`` with a fabricated metadata dict + a hand-built
``(region, fov, image)`` stream, then reads the written store back with tensorstore + json
(the same v3 store ndviewer_light reads). The real-seam cross commit (``project_plate`` on
``sim_1536wp`` + hongquan) lives in tests/test_integration.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorstore as ts

from squidmip._output import (
    parse_well_id,
    plate_metadata,
    split_well,
    write_from_stream,
)

CH = [
    {"name": "Fluorescence_638_nm_-_Penta", "display_name": "Fluorescence 638 nm - Penta", "display_color": "#FF0000"},
    {"name": "Fluorescence_405_nm_-_Penta", "display_name": "Fluorescence 405 nm - Penta", "display_color": "#20ADF8"},
]
# B10 present so column sort is natural (2,3,10 not 10,2,3) and no zero-padding is exercised.
REGIONS = ["B2", "B3", "B10"]


def _meta():
    return {
        "regions": REGIONS,
        "fovs_per_region": {r: [0] for r in REGIONS},
        "channels": CH,
        "pixel_size_um": 0.325,
        "frame_shape": (8, 8),
        "dtype": "uint16",
    }


def _image(seed: int, t=1, c=2, y=8, x=8, dtype=np.uint16):
    # deterministic, unique-ish per (seed, t, c) plane, kept small so it also fits uint8
    base = np.arange(y * x).reshape(y, x)
    out = np.empty((t, c, 1, y, x), dtype=dtype)
    for ti in range(t):
        for ci in range(c):
            out[ti, ci, 0] = ((base + seed * 20 + ti * 7 + ci * 3) % 200).astype(dtype)
    return out


def _stream(images: dict):
    # completion order is arbitrary in reality; yield out of plate order on purpose
    for region in ("B3", "B10", "B2"):
        yield region, 0, images[region]


def _read_array(path: Path) -> np.ndarray:
    store = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}}, open=True
    ).result()
    return np.asarray(store[...].read().result())


# --- pure helpers ---------------------------------------------------------------------------

def test_parse_well_id_uppercases_no_padding_roundtrips():
    # vendored Squid semantics: uppercase, multi-letter rows, no zero-padding
    assert parse_well_id("B2") == ("B", "2")
    assert parse_well_id("aa3") == ("AA", "3")  # lowercase -> upper (Squid parse_well_id)
    assert split_well is parse_well_id  # back-compat alias
    for region in ("B2", "H12", "AA1", "AF48"):
        row, col = parse_well_id(region)
        assert row + col == region  # ndviewer reconstructs well_id = row + col


def test_parse_well_id_fails_loud_on_non_plate_region():
    import pytest

    for bad in ("region_1", "1A", "B2C"):  # not <letters><digits> -> refuse, don't mislabel
        with pytest.raises(ValueError):
            parse_well_id(bad)


def test_plate_metadata_natural_column_sort_and_well_paths():
    ome = plate_metadata(REGIONS, field_count=1)["plate"]
    assert [c["name"] for c in ome["columns"]] == ["2", "3", "10"]  # int sort, not lexicographic
    assert [c["name"] for c in ome["rows"]] == ["B"]
    paths = {w["path"] for w in ome["wells"]}
    assert paths == {"B/2", "B/3", "B/10"}  # no zero-padding


# --- full write via the stream --------------------------------------------------------------

def test_write_from_stream_layout_and_pixels(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=True)

    plate = Path(manifest["plate"])
    assert plate.name == "plate.ome.zarr"
    assert manifest["n_wells"] == 3 and manifest["n_fields_written"] == 3

    # plate group metadata
    plate_doc = json.loads((plate / "zarr.json").read_text())
    assert plate_doc["node_type"] == "group"
    assert plate_doc["attributes"]["ome"]["plate"]["field_count"] == 1

    # each well: group metadata + single-level field + omero + pixel-exact full-res array
    for region in REGIONS:
        row, col = parse_well_id(region)
        well_doc = json.loads((plate / row / col / "zarr.json").read_text())
        assert well_doc["attributes"]["ome"]["well"]["images"] == [{"path": "0"}]  # raw fov id 0

        field = plate / row / col / "0"
        field_doc = json.loads((field / "zarr.json").read_text())["attributes"]["ome"]
        ds_paths = [d["path"] for d in field_doc["multiscales"][0]["datasets"]]
        assert ds_paths == ["0"]  # single level, Squid canonical (no pyramid)
        colors = [c["color"] for c in field_doc["omero"]["channels"]]
        assert colors == ["FF0000", "20ADF8"]  # hex without '#', in channel order

        assert np.array_equal(_read_array(field / "0"), images[region])  # full-res pixel-exact
        assert not (field / "1").exists()  # no pyramid level written

    # every group validates against the official OME-NGFF v0.5 pydantic models
    from tests.ngff_check import assert_valid_ngff_plate

    assert_valid_ngff_plate(plate)


def test_large_field_writes_pyramid(tmp_path):
    # A field larger than the pyramid floor (256 px) gets downsample LEVELS: 600 -> 300 -> 150.
    # Level 0 stays full-res pixel-exact; coarser levels are half-size area-averages. Small fields
    # (the other tests, 8x8) collapse to level 0 alone — canonical single-level output unchanged.
    big = {r: _image(i, y=600, x=600) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(big), tmp_path, n_fovs=1, tiff=False)
    assert manifest["levels"] == 3

    field = Path(manifest["plate"]) / "B" / "2" / "0"
    ds_paths = [d["path"] for d in
                json.loads((field / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]["datasets"]]
    assert ds_paths == ["0", "1", "2"]
    assert np.array_equal(_read_array(field / "0"), big["B2"])          # level 0 pixel-exact
    assert _read_array(field / "1").shape == (1, 2, 1, 300, 300)        # half-size
    assert _read_array(field / "2").shape == (1, 2, 1, 150, 150)
    # coarse-level scale reflects the real downsample factor (2x, 4x) in Y,X
    scales = [d["coordinateTransformations"][0]["scale"] for d in
              json.loads((field / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]["datasets"]]
    assert scales[1][-2:] == [0.325 * 2, 0.325 * 2]
    assert scales[2][-2:] == [0.325 * 4, 0.325 * 4]


def test_write_from_stream_individual_tiffs(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=True)

    import tifffile

    tiff_dir = tmp_path / "tiff" / "0"
    for region in REGIONS:
        for c_i, ch in enumerate(CH):
            f = tiff_dir / f"{region}_0_0_{ch['name']}.tiff"
            assert f.exists(), f
            plane = tifffile.imread(f)
            assert plane.dtype == np.uint16  # native dtype preserved
            assert np.array_equal(plane, images[region][0, c_i, 0])  # pixel-exact, z collapsed


def test_tiff_disabled(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=False)
    assert manifest["tiff"] is None
    assert not (tmp_path / "tiff").exists()


def test_uint8_dtype_preserved(tmp_path):
    images = {r: _image(i, dtype=np.uint8) for i, r in enumerate(REGIONS)}
    write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=False)
    arr = _read_array(tmp_path / "plate.ome.zarr" / "B" / "2" / "0" / "0")
    assert arr.dtype == np.uint8
    assert np.array_equal(arr, images["B2"])


def test_fails_loud_on_wrong_shape(tmp_path):
    import pytest

    # z not collapsed (Z=3) -> a seam bug; refuse rather than write a mislabelled field
    bad = np.zeros((1, 2, 3, 8, 8), np.uint16)
    with pytest.raises(ValueError, match="T, C, 1, Y, X"):
        write_from_stream(_meta(), iter([("B2", 0, bad)]), tmp_path, n_fovs=1, tiff=False)


def test_fails_loud_on_channel_count_mismatch(tmp_path):
    import pytest

    # image says 3 channels, metadata lists 2 -> refuse (would mislabel omero)
    bad = np.zeros((1, 3, 1, 8, 8), np.uint16)
    with pytest.raises(ValueError, match="channels"):
        write_from_stream(_meta(), iter([("B2", 0, bad)]), tmp_path, n_fovs=1, tiff=False)


def test_writer_memory_is_bounded_in_well_count(tmp_path):
    """The writer streams: peak RSS is flat in the number of wells (it never holds the plate).

    Feeds a lazy generator of N wells and checks 4x the wells does NOT ~4x the peak — each
    (region, fov, image) is written and released before the next is pulled.
    """
    import tracemalloc

    def stream(n):
        for i in range(n):
            yield f"B{i + 2}", 0, _image(i, y=256, x=256)  # ~256 KB/well, built lazily

    def peak_for(n, dest):
        meta = {
            **_meta(),
            "regions": [f"B{i + 2}" for i in range(n)],
            "fovs_per_region": {f"B{i + 2}": [0] for i in range(n)},
        }
        tracemalloc.start()
        write_from_stream(meta, stream(n), dest, n_fovs=1, tiff=True)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    p4 = peak_for(4, tmp_path / "a")
    p16 = peak_for(16, tmp_path / "b")
    assert p16 < p4 * 2  # 4x wells, <2x peak -> bounded/streaming, not proportional to plate size


# --- IMA-230: disk pre-flight guard + graceful failure ----------------------------------------

import pytest

from squidmip._output import (
    InsufficientDiskSpaceError,
    estimate_write_bytes,
    free_bytes,
    is_incomplete,
    plate_pyramid_factor,
)


def _big_meta(n_regions=4, n_fovs=9, y=2048, x=2048, n_t=1):
    regions = [f"B{i + 2}" for i in range(n_regions)]
    return {
        "regions": regions,
        "fovs_per_region": {r: list(range(n_fovs)) for r in regions},
        "channels": CH,
        "pixel_size_um": 0.325,
        "frame_shape": (y, x),
        "dtype": "uint16",
        "n_t": n_t,
    }


def test_pyramid_factor_is_the_exact_level_sum_not_a_guess():
    # 2048 -> 1024 -> 512 -> 256 : the ladder pyramid_shapes() would write, exactly.
    from squidmip._output import pyramid_shapes

    shapes = pyramid_shapes((2048, 2048))
    y0, x0 = shapes[0]
    assert plate_pyramid_factor((2048, 2048)) == pytest.approx(
        sum(y * x for y, x in shapes) / (y0 * x0))
    assert plate_pyramid_factor((8, 8)) == 1.0            # no pyramid for a tiny field


def test_estimate_scales_with_fields_channels_and_timepoints():
    m = _big_meta()
    base = estimate_write_bytes(m, n_fovs=None)
    # hand-computed: fields x t x c x z x frame_bytes x pyramid x safety (+ a small fixed allowance)
    hand = (4 * 9) * 1 * 2 * 1 * 2048 * 2048 * 2 * plate_pyramid_factor((2048, 2048))
    assert base == pytest.approx(hand * 1.03, rel=1e-3, abs=200 * 1024)
    # n_t multiplies: a 5-timepoint plate writes 5x (the exact under-count the viewer guard fixed)
    assert estimate_write_bytes(_big_meta(n_t=5), n_fovs=None) == pytest.approx(base * 5, rel=1e-3)
    # a subset of regions is scoped, and one FOV per well is 1/9th of nine
    assert estimate_write_bytes(m, n_fovs=None, regions=["B2"]) == pytest.approx(base / 4, rel=1e-2)
    assert estimate_write_bytes(m, n_fovs=1) == pytest.approx(base / 9, rel=1e-2)
    # tiff doubles-ish: a second, UNCOMPRESSED, pyramid-free copy
    assert estimate_write_bytes(m, n_fovs=None, tiff=True) > base


def test_free_bytes_uses_nearest_existing_ancestor(tmp_path):
    # the destination does not exist yet — the guard must still be able to stat the filesystem
    assert free_bytes(tmp_path / "does" / "not" / "exist") == free_bytes(tmp_path)
    assert free_bytes(tmp_path) > 0


def test_refuses_up_front_and_writes_nothing(tmp_path, monkeypatch):
    """The whole point of IMA-230: die BEFORE the first byte, not 94% through."""
    monkeypatch.setattr("squidmip._output.free_bytes", lambda p: 10 * 1024 ** 2)  # 10 MB free
    meta = _big_meta()
    out = tmp_path / "run"
    with pytest.raises(InsufficientDiskSpaceError) as ei:
        write_from_stream(meta, iter([]), out, n_fovs=None)
    msg = str(ei.value)
    assert "GB" in msg and "free" in msg.lower()
    assert not (out / "plate.ome.zarr").exists()   # not one directory created
    assert not out.exists()


def test_headroom_is_configurable_and_reserves_the_last_of_the_disk(tmp_path, monkeypatch):
    meta = _meta()
    est = estimate_write_bytes(meta, n_fovs=1)
    # exactly the estimate free: fits arithmetically, but leaves the disk at zero -> refuse
    monkeypatch.setattr("squidmip._output.free_bytes", lambda p: int(est))
    with pytest.raises(InsufficientDiskSpaceError):
        write_from_stream(meta, iter([]), tmp_path / "a", n_fovs=1)
    # a caller who insists can dial the headroom down
    write_from_stream(meta, iter([]), tmp_path / "b", n_fovs=1, disk_headroom=0.0,
                      min_free_bytes=0)
    assert (tmp_path / "b" / "plate.ome.zarr").exists()


def test_guard_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("squidmip._output.free_bytes", lambda p: 1)
    write_from_stream(_meta(), iter([]), tmp_path / "c", n_fovs=1, check_disk=False)
    assert (tmp_path / "c" / "plate.ome.zarr").exists()


def test_a_failed_field_publishes_nothing_and_marks_the_plate_incomplete(tmp_path, monkeypatch):
    """A killed/failed write must not leave a field that later reads as a valid one."""
    import squidmip._output as O

    real = O._write_field
    calls = {"n": 0}

    def boom(field_dir, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real(field_dir, *a, **k)

    monkeypatch.setattr(O, "_write_field", boom)
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    out = tmp_path / "run"
    with pytest.raises(OSError):
        write_from_stream(_meta(), _stream(images), out, n_fovs=1, write_workers=1)

    plate = out / "plate.ome.zarr"
    assert is_incomplete(plate)                       # the store announces it is not finished
    # every field directory that DOES exist is a complete one (no half-written stragglers)
    for row in plate.iterdir():
        if not row.is_dir():
            continue
        for col in (c for c in row.iterdir() if c.is_dir()):
            for fov in col.iterdir():
                if not fov.is_dir():
                    continue
                assert not fov.name.startswith("."), f"partial left behind: {fov}"
                assert (fov / "0" / "zarr.json").exists() and (fov / "zarr.json").exists()


def test_successful_write_clears_the_incomplete_marker(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1)
    plate = Path(manifest["plate"])
    assert not is_incomplete(plate)
    assert manifest["complete"] is True
    assert not any(p.name.startswith(".") and p.is_dir() for p in plate.rglob("*"))


def test_partial_tiffs_are_never_published(tmp_path, monkeypatch):
    """The 8-byte .ome.tiff that read as a real file: TIFFs land by atomic rename or not at all."""
    import squidmip._output as O

    real_imwrite = O.tifffile.imwrite
    seen = {"n": 0}

    def half_written(path, data, *a, **k):
        seen["n"] += 1
        if seen["n"] == 2:
            Path(path).write_bytes(b"II*\x00\x08\x00\x00\x00")   # 8 bytes, looks like a TIFF
            raise OSError(28, "No space left on device")
        return real_imwrite(path, data, *a, **k)

    monkeypatch.setattr(O.tifffile, "imwrite", half_written)
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    out = tmp_path / "run"
    with pytest.raises(OSError):
        write_from_stream(_meta(), _stream(images), out, n_fovs=1, tiff=True, write_workers=1)
    for tif in (out / "tiff").rglob("*"):
        if tif.is_file():
            assert not tif.name.endswith(".tiff") or tif.stat().st_size > 100


# --- IMA-231: FOV_ROI_table (Fractal / ngio convention) ---------------------------------------

from squidmip._output import (
    _NGIO_COLUMN,
    _ROI_INDEX_KEY,
    field_origin_um,
    fov_roi_records_um,
    write_fov_roi_table,
)

# One region, two FOVs, side by side: centres 500 µm apart on a 100x100 px frame at 1 µm/px.
ROI_POS = {0: (1000.0, 2000.0), 1: (1500.0, 2000.0)}
ROI_FRAME = (100, 100)
ROI_PX = 1.0


def _roi_meta(n_fovs=2):
    return {
        "regions": ["B2"],
        "fovs_per_region": {"B2": list(range(n_fovs))},
        "channels": CH,
        "pixel_size_um": ROI_PX,
        "frame_shape": ROI_FRAME,
        "dtype": "uint16",
        "dz_um": 2.0,
        "n_z": 5,
        "fov_positions_um": {("B2", f): ROI_POS[f] for f in range(n_fovs)},
    }


def _read_table(table_dir: Path):
    """Read the AnnData-encoded table back with zarr-python (an independent reader)."""
    import zarr

    g = zarr.open_group(str(table_dir), mode="r")
    columns = [str(v) for v in g["var"]["_index"][:]]
    index = [str(v) for v in g["obs"][_ROI_INDEX_KEY][:]]
    x = np.asarray(g["X"][:])
    rows = {name: dict(zip(columns, x[i])) for i, name in enumerate(index)}
    for i, name in enumerate(index):
        rows[name]["path_in_well"] = str(g["obs"]["path_in_well"][:][i])
    return dict(g.attrs), index, columns, rows


def test_um_to_ngio_map_is_a_rename_not_a_conversion():
    """Every SquidMIP key ends _um, every ngio column ends _micrometer, same unit both sides."""
    for ours, theirs in _NGIO_COLUMN.items():
        assert ours.endswith("_um"), ours
        # ngio spells the unit "micrometer" (and puts _original AFTER it) — same unit, our _um
        assert theirs.endswith("_micrometer") or theirs.endswith("_micrometer_original"), theirs
        # x_um -> x_micrometer, len_x_um -> len_x_micrometer, x_original_um -> x_micrometer_original
        assert ours.replace("_original_um", "").replace("_um", "") == \
            theirs.replace("_micrometer_original", "").replace("_micrometer", "")


def test_records_use_the_field_origin_corner_not_the_centre():
    recs = fov_roi_records_um([0, 1], ROI_POS, ROI_FRAME, ROI_PX, dz_um=2.0, n_z=5)
    by_fov = {r["path_in_well"]: r for r in recs}
    for fov in (0, 1):
        corner = field_origin_um(ROI_POS[fov], ROI_FRAME, ROI_PX)          # what _multiscales stamps
        assert by_fov[str(fov)]["x_original_um"] == pytest.approx(corner[0])
        assert by_fov[str(fov)]["y_original_um"] == pytest.approx(corner[1])
        # centre 1000 -> corner 950 for a 100 µm frame: half a frame, never the centre itself
        assert by_fov[str(fov)]["x_original_um"] != pytest.approx(ROI_POS[fov][0])
    # extents are the frame's physical size; z is the projected STACK depth (dz x n_z)
    assert by_fov["0"]["len_x_um"] == pytest.approx(100.0)
    assert by_fov["0"]["len_y_um"] == pytest.approx(100.0)
    assert by_fov["0"]["len_z_um"] == pytest.approx(10.0)
    # region-relative origin (fractal reset_origin): the top-left FOV sits at 0,0
    assert (by_fov["0"]["x_um"], by_fov["0"]["y_um"]) == pytest.approx((0.0, 0.0))
    assert by_fov["1"]["x_um"] == pytest.approx(500.0)   # the recorded 500 µm centre pitch


def test_roi_boxes_agree_with_tilesource_field_origin_exactly():
    """IMA-217's tile source and IMA-231's ROI table must place a field at the SAME corner."""
    from squidmip._tilesource import fov_bboxes_um

    boxes = fov_bboxes_um({("B2", f): p for f, p in ROI_POS.items()}, ROI_FRAME, ROI_PX)
    recs = {r["path_in_well"]: r for r in
            fov_roi_records_um([0, 1], ROI_POS, ROI_FRAME, ROI_PX)}
    for fov in (0, 1):
        x0, y0, x1, y1 = boxes[("B2", fov)]
        r = recs[str(fov)]
        assert r["x_original_um"] == pytest.approx(x0)     # same lower corner, to the float
        assert r["y_original_um"] == pytest.approx(y0)
        assert r["len_x_um"] == pytest.approx(x1 - x0)     # same extent
        assert r["len_y_um"] == pytest.approx(y1 - y0)


def test_roi_table_written_on_persist_with_ngio_column_names(tmp_path):
    meta = _roi_meta()
    images = {0: _image(0, y=100, x=100), 1: _image(1, y=100, x=100)}
    stream = iter([("B2", 0, images[0]), ("B2", 1, images[1])])
    manifest = write_from_stream(meta, stream, tmp_path, n_fovs=None, write_workers=1)

    well = Path(manifest["plate"]) / "B" / "2"
    tables = well / "tables"
    # the tables group indexes its members, which is how ngio discovers them
    assert json.loads((tables / "zarr.json").read_text())["attributes"]["tables"] == ["FOV_ROI_table"]

    attrs, index, columns, rows = _read_table(tables / "FOV_ROI_table")
    assert attrs["encoding-type"] == "anndata"
    assert attrs["type"] == "roi_table"
    assert attrs["fractal_table_version"] == "1" and attrs["table_version"] == "1"
    assert attrs["index_key"] == "FieldIndex" and attrs["index_type"] == "str"
    assert index == ["FOV_0", "FOV_1"]                     # ngio FieldIndex convention
    # the six REQUIRED ngio columns, spelled exactly, all present
    assert set(columns) >= {"x_micrometer", "y_micrometer", "z_micrometer",
                            "len_x_micrometer", "len_y_micrometer", "len_z_micrometer"}
    assert {"x_micrometer_original", "y_micrometer_original"} <= set(columns)
    assert rows["FOV_1"]["x_micrometer"] == pytest.approx(500.0)
    assert rows["FOV_1"]["len_x_micrometer"] == pytest.approx(100.0)
    assert rows["FOV_0"]["path_in_well"] == "0"            # points at the field dir on disk

    # a `tables` sibling must not break the plate as an OME-NGFF v0.5 document
    from tests.ngff_check import assert_valid_ngff_plate

    assert_valid_ngff_plate(Path(manifest["plate"]))


def test_written_roi_corners_equal_the_written_ngff_translations(tmp_path):
    """The two artifacts of one write must not disagree about where a field is."""
    meta = _roi_meta()
    stream = iter([("B2", f, _image(f, y=100, x=100)) for f in (0, 1)])
    manifest = write_from_stream(meta, stream, tmp_path, n_fovs=None, write_workers=1)
    well = Path(manifest["plate"]) / "B" / "2"
    _, _, _, rows = _read_table(well / "tables" / "FOV_ROI_table")
    for fov in (0, 1):
        ome = json.loads((well / str(fov) / "zarr.json").read_text())["attributes"]["ome"]
        xf = ome["multiscales"][0]["datasets"][0]["coordinateTransformations"]
        tx, ty = xf[1]["translation"][4], xf[1]["translation"][3]   # (t,c,z,y,x)
        assert rows[f"FOV_{fov}"]["x_micrometer_original"] == pytest.approx(tx)
        assert rows[f"FOV_{fov}"]["y_micrometer_original"] == pytest.approx(ty)


def test_no_roi_table_without_stage_positions(tmp_path):
    meta = {**_roi_meta()}
    meta.pop("fov_positions_um")
    stream = iter([("B2", f, _image(f, y=100, x=100)) for f in (0, 1)])
    manifest = write_from_stream(meta, stream, tmp_path, n_fovs=None, write_workers=1)
    assert not (Path(manifest["plate"]) / "B" / "2" / "tables").exists()


def test_millimetres_in_a_um_key_are_refused(tmp_path):
    """The 1000x defect fixed on main today must not be reintroducible through this door."""
    mm_positions = {0: (1.0, 2.0), 1: (1.5, 2.0)}           # mm wearing a _um key
    recs = fov_roi_records_um([0, 1], mm_positions, ROI_FRAME, ROI_PX)
    with pytest.raises(ValueError, match="millimetres"):
        write_fov_roi_table(tmp_path / "img", recs)


def test_roi_table_reads_back_under_anndata_if_available(tmp_path):
    """Independent reader check: anndata itself, when the environment has it (not a squidmip dep)."""
    anndata = pytest.importorskip("anndata")
    recs = fov_roi_records_um([0, 1], ROI_POS, ROI_FRAME, ROI_PX, dz_um=2.0, n_z=5)
    table = write_fov_roi_table(tmp_path / "img", recs)
    adata = anndata.read_zarr(str(table))
    assert list(adata.obs_names) == ["FOV_0", "FOV_1"]
    assert "x_micrometer" in list(adata.var_names)
    assert float(adata[:, "len_x_micrometer"].X[0, 0]) == pytest.approx(100.0)
