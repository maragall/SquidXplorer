"""contract.store: THE walk of an OME-NGFF store, one copy, v0.4 and v0.5 alike."""

from __future__ import annotations

import json

import pytest

from squidxplorer.contract.store import level_paths, ome_attrs, resolve_plate_dir


def test_v05_attributes_ome_and_v04_zattrs_read_the_same(tmp_path):
    """Three of the four deleted private copies could not read a v0.4 store at all."""
    payload = {"multiscales": [{"datasets": [{"path": "0"}, {"path": "1"}]}]}

    v5 = tmp_path / "v5"
    v5.mkdir()
    (v5 / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {"ome": payload}}))

    v4 = tmp_path / "v4"
    v4.mkdir()
    (v4 / ".zattrs").write_text(json.dumps(payload))

    assert ome_attrs(v5) == payload
    assert ome_attrs(v4) == payload
    assert [p.name for p in level_paths(v5)] == ["0", "1"]
    assert [p.name for p in level_paths(v4)] == ["0", "1"]
    assert ome_attrs(tmp_path / "neither") == {}


def test_resolve_plate_dir_takes_the_store_or_the_folder_holding_it(tmp_path):
    plate = tmp_path / "plate.ome.zarr"
    plate.mkdir()
    (plate / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group",
         "attributes": {"ome": {"plate": {"wells": []}}}}))

    assert resolve_plate_dir(plate) == plate
    assert resolve_plate_dir(tmp_path) == plate
    with pytest.raises(ValueError, match="not an OME-NGFF HCS plate"):
        resolve_plate_dir(tmp_path / "nowhere")


def test_the_contract_names_the_walk_the_DOC_promises():
    """docs/plate-contract.md names `_tilesource.plate_layout_from_store` as the canonical
    plate walk; the name must exist where the doc points."""
    from squidxplorer._tilesource import plate_layout_from_store

    assert callable(plate_layout_from_store)


def test_the_readers_identity_is_DECLARED_and_the_zarr_readers_is_the_acquisition_root(tmp_path):
    """`source_id` is the contract's identity member. The Zarr reader's is the acquisition
    ROOT (where the sidecars live) — its `_path` is the STORE, and keying the staleness token
    on the store statted acquisition.yaml/coordinates.csv files that never exist there."""
    from squidxplorer.contract.reader import SquidAcquisitionReader
    from squidxplorer.reader import SquidZarrReader

    assert hasattr(SquidAcquisitionReader, "source_id")

    store = tmp_path / "acq" / "plate.zarr"
    store.mkdir(parents=True)
    r = SquidZarrReader(store)
    assert r.source_id == str(tmp_path / "acq")
    assert r._path == store, "the store path is still the store path; only the IDENTITY moved"


def test_parse_coordinates_csv_is_pure_text_in_positions_out():
    from squidxplorer.reader import parse_coordinates_csv

    text = "region,x (mm),y (mm)\nA1,1.0,2.0\nA1,2.0,2.0\nA1,2.0,2.0\n"
    positions, mismatched = parse_coordinates_csv(text, {"A1": [0, 1]})
    assert positions == {("A1", 0): (1000.0, 2000.0), ("A1", 1): (2000.0, 2000.0)}
    assert mismatched == {}

    _positions, mismatched = parse_coordinates_csv(text, {"A1": [0, 1, 2]})
    assert mismatched == {"A1": (2, 3)}, "the per-region cross-check went missing"
