"""The plate contract: stamped, compared, and reconstructed from one place."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from squidxplorer import _output
from squidxplorer.contract import (
    PLATE_CONTRACT_VERSION,
    PlateContractError,
    compare_contract_version,
    contract_stamp,
    field_levels,
    field_path,
    read_contract_version,
)
from squidxplorer.contract.validate import validate_plate
from squidxplorer.reader import SquidZarrReader


def _write_plate(tmp_path, *, translation=True, omero=True, version=None, n_t=1) -> Path:
    """A two-well, one-field plate written by the real writers, then optionally degraded."""
    from squidxplorer._zarr_store import create_array, write_array, write_group

    plate_dir = tmp_path / "plate.ome.zarr"
    regions = ["A1", "B2"]
    attributes = contract_stamp()
    if version is not None:
        attributes = {"squidxplorer": {"plate_contract_version": version}}
    write_group(plate_dir, _output.plate_metadata(regions, field_count=1), attributes=attributes)

    channels = [{"name": "BF", "display_name": "BF", "display_color": "#FFFFFF"}]
    for region in regions:
        row, col = _output.parse_well_id(region)
        write_group(plate_dir / row)
        write_group(plate_dir / row / col,
                    {"version": "0.5", "well": {"images": [{"path": "0"}]}})
        field = plate_dir / row / col / "0"
        data = np.arange(16 * n_t, dtype=np.uint16).reshape(n_t, 1, 1, 4, 4)
        write_array(create_array(field / "0", data.shape, data.dtype), data)
        ome = {
            "version": "0.5",
            "multiscales": [_output._multiscales(
                [(4, 4)], pixel_size_um=0.5, dz_um=1.0,
                position_um=(10.0, 20.0) if translation else None)],
        }
        if omero:
            ome["omero"] = _output._omero(channels, np.uint16)
        write_group(field, ome)
    return plate_dir


def test_a_written_plate_carries_the_contract_version(tmp_path):
    plate_dir = _write_plate(tmp_path)
    assert read_contract_version(plate_dir) == PLATE_CONTRACT_VERSION


def test_the_stamp_lives_outside_the_ome_namespace(tmp_path):
    """attributes.ome belongs to OME; a private key goes beside it, not in it."""
    plate_dir = _write_plate(tmp_path)
    attrs = json.loads((plate_dir / "zarr.json").read_text())["attributes"]
    assert "squidxplorer" in attrs and "plate_contract_version" in attrs["squidxplorer"]
    assert "squidxplorer" not in attrs["ome"]
    assert "plate_contract_version" not in attrs["ome"]


def test_write_then_read_carries_the_version_end_to_end(tmp_path):
    """The real writer, the real reader, no fixture in between."""
    from tests.test_output import REGIONS, _image, _meta, _stream

    from squidxplorer._output import write_from_stream

    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1)
    plate_dir = Path(manifest["plate"])

    assert read_contract_version(plate_dir) == PLATE_CONTRACT_VERSION
    reader = SquidZarrReader(plate_dir)
    assert set(reader.metadata["regions"]) == set(REGIONS)   # plate order, not lexicographic
    assert reader._contract_version == PLATE_CONTRACT_VERSION
    assert validate_plate(plate_dir).ok, validate_plate(plate_dir).summary()


def test_a_major_mismatch_stops_the_reader_opening_the_store_naming_both_versions(tmp_path):
    """Enforced at the reader seam, not only in the pure function."""
    major = int(PLATE_CONTRACT_VERSION.split(".")[0])
    plate_dir = _write_plate(tmp_path, version=f"{major + 1}.0")
    with pytest.raises(PlateContractError) as excinfo:
        SquidZarrReader(plate_dir).metadata
    message = str(excinfo.value)
    assert f"{major + 1}.0" in message and PLATE_CONTRACT_VERSION in message, \
        "the refusal must name BOTH versions, or the user cannot act on it"


def test_a_newer_minor_warns_and_proceeds(tmp_path):
    """A minor bump may only ADD an optional guarantee, so the read is lossy, never wrong."""
    major, minor = (int(p) for p in PLATE_CONTRACT_VERSION.split("."))
    plate_dir = _write_plate(tmp_path, version=f"{major}.{minor + 9}")
    assert compare_contract_version(f"{major}.{minor + 9}") == "minor-ahead"
    with pytest.warns(UserWarning, match="newer than this build"):
        SquidZarrReader(plate_dir).metadata


def test_an_unstamped_store_is_read_without_complaint(tmp_path):
    """Every third-party NGFF store, and every plate written before this landed, is unstamped."""
    plate_dir = _write_plate(tmp_path)
    doc = json.loads((plate_dir / "zarr.json").read_text())
    doc["attributes"].pop("squidxplorer")
    (plate_dir / "zarr.json").write_text(json.dumps(doc))

    assert read_contract_version(plate_dir) is None
    assert compare_contract_version(None) == "absent"
    meta = SquidZarrReader(plate_dir).metadata          # must not raise
    assert meta["regions"] == ["A1", "B2"]


def test_an_unparseable_stamp_is_refused():
    """A promise was deliberately made and we cannot tell which one: worse than no stamp."""
    for bad in ("one.two", "1", "1.2.3", ""):
        with pytest.raises(PlateContractError):
            compare_contract_version(bad)


def test_the_spec_version_and_the_contract_version_are_different_things():
    assert _output._NGFF_VERSION == "0.5"
    assert PLATE_CONTRACT_VERSION != _output._NGFF_VERSION


def test_a_conforming_plate_validates_clean_and_names_its_single_level_zero(tmp_path):
    """Small fields are written single-level on purpose (_PYRAMID_MIN_YX). Legal, and lossy."""
    report = validate_plate(_write_plate(tmp_path))
    assert report.ok, report.summary()
    assert report.contract_version == PLATE_CONTRACT_VERSION
    assert any("level '0'" in w for w in report.warnings), report.summary()


def test_a_broken_stable_guarantee_is_an_ERROR(tmp_path):
    """Level 0 declared and missing: the store is not the thing it says it is."""
    plate_dir = _write_plate(tmp_path)
    import shutil

    shutil.rmtree(plate_dir / "A" / "1" / "0" / "0")
    report = validate_plate(plate_dir)
    assert not report.ok
    assert any("not a zarr array" in e for e in report.errors), report.summary()


def test_a_broken_axis_order_is_an_ERROR(tmp_path):
    """TCZYX is stable; a store that reorders it is not readable by this build at all."""
    plate_dir = _write_plate(tmp_path)
    field = plate_dir / "A" / "1" / "0"
    doc = json.loads((field / "zarr.json").read_text())
    axes = doc["attributes"]["ome"]["multiscales"][0]["axes"]
    axes[1], axes[2] = axes[2], axes[1]                 # TZCYX, Squid's other order
    (field / "zarr.json").write_text(json.dumps(doc))

    report = validate_plate(plate_dir)
    assert not report.ok
    assert any("stable TCZYX order" in e for e in report.errors), report.summary()


def test_a_missing_optional_sidecar_is_a_WARNING_not_an_error(tmp_path):
    """translation and omero both have named fallbacks, so their absence narrows a read, not breaks it."""
    plate_dir = _write_plate(tmp_path, translation=False, omero=False)
    report = validate_plate(plate_dir)
    assert report.ok, report.summary()
    assert any("translation" in w for w in report.warnings), report.summary()
    assert any("omero" in w for w in report.warnings), report.summary()
    assert any("coordinates.csv" in w for w in report.warnings), \
        "a warning must NAME the fallback, or it is just noise"


def test_an_incomplete_marker_is_a_WARNING(tmp_path):
    plate_dir = _write_plate(tmp_path)
    (plate_dir / ".squidxplorer-incomplete").write_text("{}")
    report = validate_plate(plate_dir)
    assert report.ok
    assert any("did not finish" in w for w in report.warnings), report.summary()


def test_a_major_mismatch_is_reported_by_validate_rather_than_raised(tmp_path):
    """A reader must stop; a validator must finish and list everything."""
    major = int(PLATE_CONTRACT_VERSION.split(".")[0])
    report = validate_plate(_write_plate(tmp_path, version=f"{major + 1}.0"))
    assert not report.ok
    assert any("MAJOR difference" in e for e in report.errors), report.summary()


def test_the_stamp_does_not_disturb_the_official_schema(tmp_path):
    """OME's own pydantic models still pass; the stamp sits beside their namespace, not in it."""
    pytest.importorskip("ome_zarr_models")
    from squidxplorer.contract.validate import assert_valid_ngff_plate

    assert_valid_ngff_plate(_write_plate(tmp_path))


def test_the_validator_still_runs_without_ome_zarr_models(tmp_path, monkeypatch):
    """Degrades to structural checks, and reports the skip: 'validated' != 'half-validated'."""
    import builtins

    real_import = builtins.__import__

    def _no_ome_zarr_models(name, *args, **kwargs):
        if name.startswith("ome_zarr_models"):
            raise ImportError("simulated: the [test] extra is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_ome_zarr_models)
    report = validate_plate(_write_plate(tmp_path))
    assert report.ok, report.summary()
    assert any("SKIPPED" in w for w in report.warnings), report.summary()


def test_the_validator_is_runnable_by_a_user_on_a_plate_they_were_handed(tmp_path, capsys):
    from squidxplorer.contract.validate import main

    assert main([str(_write_plate(tmp_path))]) == 0
    assert "OK" in capsys.readouterr().out


def test_field_path_builds_the_documented_layout_forward_slashed_from_the_well_path_verbatim():
    """TensorStore's file kvstore takes POSIX paths on every platform; wellpath is never re-derived."""
    assert field_path("/p/plate.ome.zarr", "B/2", 7, "1") == "/p/plate.ome.zarr/B/2/7/1"
    assert field_path("/p/plate.ome.zarr", "B/2", 7) == "/p/plate.ome.zarr/B/2/7"
    assert field_path("/p/plate.ome.zarr/", "/B/2/", "7", 0) == "/p/plate.ome.zarr/B/2/7/0"
    assert "\\" not in field_path("/p", "B/2", 7, 0)
    assert field_path("/p", "AA/12", 3, 0) == "/p/AA/12/3/0"


def test_field_levels_falls_back_to_level_zero_by_NAME_not_by_accident(tmp_path):
    missing = tmp_path / "nothing-here"
    assert field_levels(missing) == ["0"]
    plate_dir = _write_plate(tmp_path)
    assert field_levels(plate_dir / "A" / "1" / "0") == ["0"]


def test_field_path_is_the_only_place_that_knows_the_layout():
    """Greps for a base joined to 3+ slash-separated placeholders in one f-string; ``_montage`` and ``_tilesource`` are allowed the other legitimate route,"""
    root = Path(__file__).resolve().parent.parent / "squidxplorer"
    joined = re.compile(r'f"\{[^"{}]+\}/\{[^"{}]+\}/\{[^"{}]+\}')
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.parent.name == "contract":
            continue                                    # the seam is allowed to know the layout
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if joined.search(line):
                offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, (
        "a plate path is being reconstructed outside squidxplorer/contract again:\n"
        + "\n".join(offenders))


def test_the_viewer_read_paths_go_through_the_seam():
    """The four sites the review counted. A call, not a mention."""
    import inspect

    from squidxplorer import _viewer

    for func in (_viewer._ZarrLoupeSource._resolve_levels,
                 _viewer._ZarrLoupeSource._open,
                 _viewer._ComputedPlateWorker._read):
        src = inspect.getsource(func)
        assert "field_path(" in src or "field_levels(" in src, \
            f"{func.__qualname__} stopped using the contract seam"
    assert 'json.loads((Path(field) / "zarr.json")' not in inspect.getsource(_viewer), \
        "the loupe hand-parses multiscales again (it was a bare `except Exception`)"


def test_a_multi_timepoint_plate_is_WARNED_about_not_silently_flattened(tmp_path):
    """The store is valid, so this is a warning. Silence is what makes it a trap."""
    report = validate_plate(_write_plate(tmp_path, n_t=4))
    assert report.ok, report.summary()
    assert any("4 timepoints" in w for w in report.warnings), report.summary()
    assert any("t=0" in w for w in report.warnings), \
        "the warning must say WHAT collapses, or a user cannot tell what they are losing"
    single = validate_plate(_write_plate(tmp_path / "one", n_t=1))
    assert not any("timepoint" in w for w in single.warnings), single.summary()


def test_every_documented_read_site_takes_a_timepoint():
    """The doc's table is a claim about the code; pin it, or it rots the way the reader prose did."""
    import inspect

    from squidxplorer._plate_overview import _ZarrLoupeSource
    from squidxplorer._workers import _ComputedPlateWorker

    for func in (_ComputedPlateWorker._read, _ZarrLoupeSource.coarse, _ZarrLoupeSource.read_crop):
        src = inspect.getsource(func)
        assert "time_point" in src, (
            f"{func.__qualname__} stopped taking a timepoint. If that is deliberate, the Time "
            "section of docs/plate-contract.md is now wrong and must change in the same commit.")
        assert "[0, :, 0]" not in src, (
            f"{func.__qualname__} hardcodes timepoint 0 again. That is the bug the Time section of "
            "docs/plate-contract.md describes: a 40-timepoint plate looks like a 1-timepoint one, "
            "silently, and no fixture with Nt = 1 can catch it.")

    doc = (Path(__file__).resolve().parent.parent / "docs" / "plate-contract.md").read_text()
    assert "### Time: the format carries it" in doc
    assert "Nt = 1" in doc


def test_the_timepoint_control_is_one_class_for_plate_and_windows():
    """Two implementations would drift about which timepoint you are looking at."""
    from squidxplorer._region_viewer import RegionViewer
    from squidxplorer._time_point import TimePointBar
    from squidxplorer import _viewer

    for mod in (_viewer, RegionViewer.__module__ and __import__(RegionViewer.__module__,
                                                               fromlist=["_"])):
        assert getattr(mod, "TimePointBar", None) is TimePointBar, (
            f"{mod.__name__} does not use the shared TimePointBar")


def test_the_contract_is_written_down_and_split_in_two():
    doc = (Path(__file__).resolve().parent.parent / "docs" / "plate-contract.md").read_text()
    assert "## Stable" in doc and "## Optional, each with its fallback" in doc
    for fallback in ("coordinates.csv", "auto-contrast", 'level `"0"`'):
        assert fallback in doc, f"the optional section stopped naming the {fallback} fallback"
    assert "events.jsonl" in doc and "NOT in this contract" in doc
