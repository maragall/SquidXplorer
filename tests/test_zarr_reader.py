"""IMA-229: Zarr input — OME-NGFF HCS plate + non-HCS image groups through the SAME reader seam.

``reader.py`` used to raise ``NotImplementedError`` for Zarr. Squid writes ``plate.ome.zarr``, and
SquidMIP's own writer (``_output.py``) emits exactly that, so a MIP output was not re-openable by
the tool that produced it.

The contract under test is *parity*, not a new API: ``metadata`` carries the same twelve keys with
the same meanings, and ``read(region, fov, channel, z, t)`` has the same signature and returns the
same kind of 2-D native-dtype plane, whether the bytes came from TIFFs or from a Zarr store. Every
consumer (engine, CLI, viewer) therefore works unchanged. A forked parallel API would have been the
easy thing and the wrong thing.

Layout adopted (see the reader docstring for the spec citations):

    HCS      plate.ome.zarr/{row}/{col}/{fov}/{level}      rows/columns/wells + well.images
    non-HCS  zarr/{region_id}/{level}                      a bare multiscales image group

Fixtures here are TINY (4x4 to 8x8 px, a few KB) and live in ``tmp_path``. The v0.5 fixture is
produced by SquidMIP's own writer, so this doubles as a round-trip test; the v0.4 fixture is
hand-written zarr v2 (``.zgroup``/``.zattrs``), because the two NGFF versions put the metadata in
different places and a reader that only handles one of them cannot open half the real stores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts

from squidmip import open_reader
from squidmip.reader import SquidZarrReader

_CH = [
    {"name": "Fluorescence_488_nm_-_Penta", "display_name": "Fluorescence 488 nm - Penta",
     "display_color": "#00FF00", "exposure_time_ms": None, "excitation_nm": 488.0},
    {"name": "Fluorescence_638_nm_-_Penta", "display_name": "Fluorescence 638 nm - Penta",
     "display_color": "#FF0000", "exposure_time_ms": None, "excitation_nm": 638.0},
]


# --- fixtures ---------------------------------------------------------------------------------

def _write_v05_plate(out_dir, regions=("B2", "B3"), fovs=(0, 1), shape=(1, 2, 1, 8, 8)):
    """A real v0.5 plate written by SquidMIP's own writer. Returns {(region, fov): array}."""
    from squidmip._output import write_from_stream

    n_t, n_c, n_z, y, x = shape
    arrays = {}
    for i, region in enumerate(regions):
        for j, fov in enumerate(fovs):
            arrays[(region, fov)] = (
                np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + 100 * i + 10 * j
            )
    meta = {
        "regions": list(regions),
        "fovs_per_region": {r: list(fovs) for r in regions},
        "channels": _CH,
        "pixel_size_um": 0.3728571,
        "dz_um": 1.5,
        "n_t": n_t,
        "n_z": n_z,
    }
    stream = ((r, f, arrays[(r, f)]) for r in regions for f in fovs)
    write_from_stream(meta, stream, out_dir, n_fovs=len(fovs))
    return arrays


def _zarr_v2_array(path, data):
    """Write one zarr **v2** array (``.zarray`` + chunks) — the on-disk shape NGFF v0.4 uses."""
    store = ts.open(
        {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)},
         "metadata": {"shape": list(data.shape), "chunks": list(data.shape),
                      "dtype": data.dtype.str}},
        create=True, delete_existing=True,
    ).result()
    store[...].write(np.ascontiguousarray(data)).result()


def _v2_group(path: Path, attrs: dict | None = None):
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (path / ".zattrs").write_text(json.dumps(attrs or {}))


def _v04_multiscales(pixel_size_um=0.5, dz_um=2.0, translation=None, unit="micrometer"):
    ds = {"path": "0", "coordinateTransformations": [
        {"type": "scale", "scale": [1.0, 1.0, dz_um, pixel_size_um, pixel_size_um]}]}
    if translation is not None:
        ds["coordinateTransformations"].append({"type": "translation", "translation": translation})
    return [{
        "version": "0.4", "name": "0",
        "axes": [{"name": "t", "type": "time", "unit": "second"},
                 {"name": "c", "type": "channel"},
                 {"name": "z", "type": "space", "unit": unit},
                 {"name": "y", "type": "space", "unit": unit},
                 {"name": "x", "type": "space", "unit": unit}],
        "datasets": [ds],
    }]


def _omero():
    return {"channels": [{"label": c["display_name"], "color": c["display_color"].lstrip("#"),
                          "active": True} for c in _CH]}


def _write_v04_plate(root: Path, wells=(("B", "2"), ("B", "3")), fovs=("0", "1"),
                     shape=(1, 2, 3, 4, 4), translations=None, unit="micrometer"):
    """A hand-written NGFF **v0.4** (zarr v2) HCS plate: .zattrs['plate'], .zattrs['well']."""
    rows = sorted({r for r, _ in wells})
    cols = sorted({c for _, c in wells}, key=int)
    _v2_group(root, {"plate": {
        "version": "0.4", "name": "p",
        "rows": [{"name": r} for r in rows], "columns": [{"name": c} for c in cols],
        "wells": [{"path": f"{r}/{c}", "rowIndex": rows.index(r), "columnIndex": cols.index(c)}
                  for r, c in wells],
        "field_count": len(fovs)}})
    arrays = {}
    for row, col in wells:
        _v2_group(root / row)
        _v2_group(root / row / col,
                  {"well": {"version": "0.4", "images": [{"path": f} for f in fovs]}})
        for f in fovs:
            tr = (translations or {}).get((row + col, int(f)))
            _v2_group(root / row / col / f,
                      {"multiscales": _v04_multiscales(translation=tr, unit=unit),
                       "omero": _omero()})
            data = (np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
                    + 7 * int(f) + ord(col))
            _zarr_v2_array(root / row / col / f / "0", data)
            arrays[(row + col, int(f))] = data
    return arrays


# --- dispatch ---------------------------------------------------------------------------------

def test_open_reader_no_longer_raises_not_implemented_for_zarr(tmp_path):
    """The whole point of IMA-229: the seam that used to reject Zarr now serves it."""
    _write_v05_plate(tmp_path / "out")
    reader = open_reader(tmp_path / "out")
    assert isinstance(reader, SquidZarrReader)


def test_open_reader_accepts_the_plate_directory_itself(tmp_path):
    _write_v05_plate(tmp_path / "out")
    assert isinstance(open_reader(tmp_path / "out" / "plate.ome.zarr"), SquidZarrReader)


def test_open_reader_finds_non_hcs_zarr_folder(tmp_path):
    root = tmp_path / "acq"
    (root / "zarr").mkdir(parents=True)
    _v2_group(root / "zarr" / "manual0",
              {"multiscales": _v04_multiscales(), "omero": _omero()})
    _zarr_v2_array(root / "zarr" / "manual0" / "0",
                   np.zeros((1, 2, 3, 4, 4), np.uint16))
    r = open_reader(root)
    assert isinstance(r, SquidZarrReader)
    assert r.metadata["regions"] == ["manual0"]


# --- v0.5 round trip (our own writer) ---------------------------------------------------------

def test_v05_plate_round_trips_metadata(tmp_path):
    _write_v05_plate(tmp_path / "out")
    meta = open_reader(tmp_path / "out").metadata
    assert meta["regions"] == ["B2", "B3"]
    assert meta["fovs_per_region"] == {"B2": [0, 1], "B3": [0, 1]}
    assert meta["frame_shape"] == (8, 8)
    assert meta["dtype"] == np.uint16
    assert meta["n_t"] == 1
    assert meta["n_z"] == 1
    assert meta["z_levels"] == [0]
    assert meta["pixel_size_um"] == pytest.approx(0.3728571)
    assert meta["dz_um"] == pytest.approx(1.5)
    assert [c["name"] for c in meta["channels"]] == [c["name"] for c in _CH]


def test_v05_plate_round_trips_pixels_exactly(tmp_path):
    """Level 0 is pixel-exact, so a write/read round trip must be byte-identical."""
    arrays = _write_v05_plate(tmp_path / "out")
    reader = open_reader(tmp_path / "out")
    names = [c["name"] for c in reader.metadata["channels"]]
    for (region, fov), data in arrays.items():
        for c_i, ch in enumerate(names):
            got = reader.read(region, fov, ch, 0)
            assert got.dtype == np.uint16
            np.testing.assert_array_equal(got, data[0, c_i, 0])


def test_v05_fixture_is_a_valid_ngff_plate(tmp_path):
    """The layout we read is the one the official OME schema accepts — not an invented one."""
    pytest.importorskip("ome_zarr_models")
    from tests.ngff_check import assert_valid_ngff_plate

    _write_v05_plate(tmp_path / "out")
    assert_valid_ngff_plate(tmp_path / "out" / "plate.ome.zarr")


# --- v0.4 (zarr v2) ---------------------------------------------------------------------------

def test_v04_plate_metadata(tmp_path):
    """v0.4 puts the same keys in ``.zattrs`` instead of ``zarr.json/attributes/ome``."""
    root = tmp_path / "plate.ome.zarr"
    _write_v04_plate(root)
    meta = open_reader(root).metadata
    assert meta["regions"] == ["B2", "B3"]
    assert meta["fovs_per_region"] == {"B2": [0, 1], "B3": [0, 1]}
    assert meta["n_z"] == 3
    assert meta["z_levels"] == [0, 1, 2]
    assert meta["n_t"] == 1
    assert meta["frame_shape"] == (4, 4)
    assert meta["pixel_size_um"] == pytest.approx(0.5)
    assert meta["dz_um"] == pytest.approx(2.0)


def test_v04_read_selects_the_right_plane(tmp_path):
    root = tmp_path / "plate.ome.zarr"
    arrays = _write_v04_plate(root)
    reader = open_reader(root)
    names = [c["name"] for c in reader.metadata["channels"]]
    for (region, fov), data in arrays.items():
        for z in range(3):
            got = reader.read(region, fov, names[1], z)
            np.testing.assert_array_equal(got, data[0, 1, z])


def test_well_image_paths_drive_fov_ids_not_a_0_based_range(tmp_path):
    """The spec says field paths are arbitrary alphanumerics; use the listed paths."""
    root = tmp_path / "plate.ome.zarr"
    arrays = _write_v04_plate(root, wells=(("B", "2"),), fovs=("3", "17"))
    meta = open_reader(root).metadata
    assert meta["fovs_per_region"] == {"B2": [3, 17]}
    np.testing.assert_array_equal(
        open_reader(root).read("B2", 17, meta["channels"][0]["name"], 0),
        arrays[("B2", 17)][0, 0, 0],
    )


# --- positions: the units contract ------------------------------------------------------------

def test_positions_come_from_dataset_translation_in_um(tmp_path):
    """NGFF's only position mechanism is ``coordinateTransformations.translation`` (post-scale).

    The axes declare ``micrometer``, so the values pass through unscaled — and land under a key
    that says ``_um``.
    """
    root = tmp_path / "plate.ome.zarr"
    _write_v04_plate(root, wells=(("B", "2"),), fovs=("0", "1"),
                     translations={("B2", 0): [0.0, 0.0, 0.0, 2000.0, 1000.0],
                                   ("B2", 1): [0.0, 0.0, 0.0, 2000.0, 1500.0]})
    pos = open_reader(root).metadata["fov_positions_um"]
    assert pos == {("B2", 0): (1000.0, 2000.0), ("B2", 1): (1500.0, 2000.0)}


def test_translation_in_millimetres_is_converted_at_the_producer(tmp_path):
    """A store whose space axes say ``millimeter`` must still yield MICROMETRES here.

    This is the exact bug class fixed on main: a mm value under a ``_um`` key, right only because
    a consumer carried a compensating factor. The conversion belongs to this producer and there
    must be exactly one of it.
    """
    root = tmp_path / "plate.ome.zarr"
    _write_v04_plate(root, wells=(("B", "2"),), fovs=("0",), unit="millimeter",
                     translations={("B2", 0): [0.0, 0.0, 0.0, 2.0, 1.0]})
    meta = open_reader(root).metadata
    assert meta["fov_positions_um"][("B2", 0)] == pytest.approx((1000.0, 2000.0))
    # the same unit applies to the pixel size: 0.5 mm/px is 500 um/px
    assert meta["pixel_size_um"] == pytest.approx(500.0)


def test_positions_fall_back_to_a_sibling_coordinates_csv(tmp_path):
    """No translation in the store (our writer emits none) -> use coordinates.csv if it is there."""
    out = tmp_path / "out"
    _write_v05_plate(out, regions=("B2",), fovs=(0, 1))
    (out / "coordinates.csv").write_text(
        "region,x (mm),y (mm),z (mm)\nB2,1.0,2.0,\nB2,1.5,2.0,\n")
    pos = open_reader(out).metadata["fov_positions_um"]
    assert pos == {("B2", 0): (1000, 2000), ("B2", 1): (1500, 2000)}


def test_positions_empty_when_neither_source_exists(tmp_path):
    """Present-but-empty, never a missing key — same promise as the TIFF readers."""
    _write_v05_plate(tmp_path / "out", regions=("B2",), fovs=(0,))
    assert open_reader(tmp_path / "out").metadata["fov_positions_um"] == {}


# --- interface parity: the seam, not a fork ----------------------------------------------------

def test_metadata_keys_identical_to_the_tiff_reader(tmp_path, squid_dataset):
    """Same twelve keys, no more, no fewer — consumers must not have to ask which reader they hold."""
    tiff_root, _ = squid_dataset
    _write_v05_plate(tmp_path / "out", regions=("B2",), fovs=(0,))
    tiff_meta = open_reader(tiff_root).metadata
    zarr_meta = open_reader(tmp_path / "out").metadata
    assert set(zarr_meta) == set(tiff_meta)
    for key, value in zarr_meta.items():
        assert value is not None, f"metadata[{key!r}] is None — dead attribute"


def test_read_signature_matches_the_tiff_reader(tmp_path):
    import inspect

    from squidmip.reader import SquidOMEReader, SquidReader

    sig = inspect.signature(SquidZarrReader.read)
    assert sig.parameters.keys() == inspect.signature(SquidReader.read).parameters.keys()
    assert sig.parameters.keys() == inspect.signature(SquidOMEReader.read).parameters.keys()


def test_wellplate_format_is_resolved(tmp_path):
    _write_v05_plate(tmp_path / "out", regions=("B2",), fovs=(0,))
    assert open_reader(tmp_path / "out").metadata["wellplate_format"]


def test_declared_wellplate_format_beats_inference(tmp_path):
    out = tmp_path / "out"
    _write_v05_plate(out, regions=("B2",), fovs=(0,))
    (out / "acquisition.yaml").write_text("sample:\n  wellplate_format: 384 well plate\n")
    assert open_reader(out).metadata["wellplate_format"] == "384 well plate"


def test_read_is_lazy_one_plane_not_the_whole_field(tmp_path):
    """A plane read must not materialise the (T, C, Z, Y, X) field — that is the memory contract."""
    _write_v05_plate(tmp_path / "out", regions=("B2",), fovs=(0,), shape=(1, 2, 1, 8, 8))
    reader = open_reader(tmp_path / "out")
    ch = reader.metadata["channels"][0]["name"]
    got = reader.read("B2", 0, ch, 0)
    assert got.shape == (8, 8) and got.ndim == 2


def test_read_invalid_args_raise_keyerror(tmp_path):
    _write_v05_plate(tmp_path / "out", regions=("B2",), fovs=(0,))
    reader = open_reader(tmp_path / "out")
    ch = reader.metadata["channels"][0]["name"]
    for args in [("ZZ", 0, ch, 0), ("B2", 99, ch, 0), ("B2", 0, "Nope", 0)]:
        with pytest.raises(KeyError):
            reader.read(*args)


def test_read_out_of_range_z_and_t_raise_indexerror(tmp_path):
    _write_v05_plate(tmp_path / "out", regions=("B2",), fovs=(0,))
    reader = open_reader(tmp_path / "out")
    ch = reader.metadata["channels"][0]["name"]
    with pytest.raises(IndexError):
        reader.read("B2", 0, ch, 9)
    with pytest.raises(IndexError):
        reader.read("B2", 0, ch, 0, t=5)


def test_plane_ref_points_at_an_openable_ngff_image_group(tmp_path):
    """The viewer registers plane_ref() into ndviewer. For Zarr the honest referent is the field
    IMAGE GROUP (a valid NGFF node any ome-zarr reader opens), not a TIFF page."""
    _write_v05_plate(tmp_path / "out", regions=("B2",), fovs=(0,))
    reader = open_reader(tmp_path / "out")
    path, page = reader.plane_ref("B2", 0, reader.metadata["channels"][0]["name"], 0)
    assert Path(path).is_dir()
    assert (Path(path) / "zarr.json").exists()
    assert page == 0


def test_dtype_outside_uint8_uint16_is_refused(tmp_path):
    """Same dtype contract as the TIFF path: a float/uint32 store is not a raw Squid capture."""
    root = tmp_path / "plate.ome.zarr"
    _write_v04_plate(root, wells=(("B", "2"),), fovs=("0",))
    _zarr_v2_array(root / "B" / "2" / "0" / "0", np.zeros((1, 2, 3, 4, 4), np.uint32))
    with pytest.raises(ValueError, match="dtype"):
        open_reader(root).metadata


def test_empty_plate_raises_clearly(tmp_path):
    root = tmp_path / "plate.ome.zarr"
    _v2_group(root, {"plate": {"version": "0.4", "rows": [], "columns": [], "wells": []}})
    with pytest.raises(ValueError, match="no wells|No "):
        open_reader(root).metadata
