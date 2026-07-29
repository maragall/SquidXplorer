"""The plate contract is written down, stamped, compared, and reconstructed in one place.

Gap 5 of the three-viewers review (Hongquan, 2026-07-28), pulled to full v1 scope by Julio on
2026-07-28. Verified against HEAD before writing anything, and the measurements are the reason
this file exists rather than a docstring:

* ``_output._NGFF_VERSION`` was written at FOUR sites and read at ZERO. Version discrimination in
  the reader was structural sniffing (``_group_attrs`` testing whether a ``.zattrs`` or a
  ``zarr.json`` exists), which is correct today and becomes a silent misparse the first time the
  layout moves, because nothing ever compared a declared version to a supported one.
* The plate path was reconstructed by f-string at FOUR sites in ``_viewer.py``, three of which
  handed the string straight to a store open and bypassed ``reader.py`` entirely. One of them
  hand-parsed ``zarr.json -> attributes -> ome -> multiscales[0] -> datasets[*].path`` behind a
  bare ``except Exception``, which is both a fifth copy of the layout and an unnamed fallback.
* The reader's own contract prose stated the OPPOSITE of what the writer does about
  ``translation``, in two places, from IMA-217 until 2026-07-29. That is the live defect an
  unversioned prose contract produces, and it is why this was not deferred.

WHAT WAS REJECTED, and why, so nobody re-proposes it:

* **Stamping the version per well or per field.** Rejected: a store has ONE layout, and the exact
  failure being fixed is a value written in many places and read in none. One stamp, on the plate
  group, one comparison in ``reader._discover``.
* **Putting the stamp inside ``attributes.ome``.** Rejected: that namespace is OME's and is what
  ``ome-zarr-models`` validates. A private key goes beside it, not in it. ``test_the_stamp_does
  _not_disturb_the_official_schema`` pins that.
* **Refusing an UNSTAMPED store.** Rejected: every plate written before 2026-07-29 and every
  third-party NGFF store carries no stamp, and this reader explicitly supports four zarr layouts
  from two spec versions. Refusing them would reject the installed base to enforce a rule invented
  after they were written. Absent means "no promise made", and the structural checks then earn it.
* **Warning on a major mismatch instead of raising.** Rejected: a major bump means a stable
  guarantee moved, so the store still opens, still finds its wells, and places them wrongly. That
  is the outcome ``reader._parse_fov_positions_um`` already refuses when it will not "place FOVs at
  positions that would look plausible but be wrong".
* **Making ``ome-zarr-models`` a runtime dependency** so the validator is always complete.
  Rejected: it is a ``[test]`` extra and the writer stays lean. The validator degrades to its
  structural checks and REPORTS the skip, because "validated" and "half-validated" must not look
  alike.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from squidmip import _output
from squidmip.contract import (
    PLATE_CONTRACT_VERSION,
    PlateContractError,
    compare_contract_version,
    contract_stamp,
    field_levels,
    field_path,
    read_contract_version,
)
from squidmip.contract.validate import validate_plate
from squidmip.reader import SquidZarrReader


# --- fixtures: the smallest real plate that satisfies the stable contract -----------------------

def _write_plate(tmp_path, *, translation=True, omero=True, version=None, n_t=1) -> Path:
    """A two-well, one-field plate written by the REAL writers, then optionally degraded.

    Built through ``_output.plate_metadata`` / ``_multiscales`` / ``_zarr_store`` rather than by
    hand: a fixture that spells the layout itself would pass while the writer drifted, which is
    the failure mode this whole file is about.
    """
    from squidmip._zarr_store import create_array, write_array, write_group

    plate_dir = tmp_path / "plate.ome.zarr"
    regions = ["A1", "B2"]
    attributes = contract_stamp()
    if version is not None:
        attributes = {"squidmip": {"plate_contract_version": version}}
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


# --- the version: round trip, and the mismatch policy -------------------------------------------

def test_a_written_plate_carries_the_contract_version(tmp_path):
    """The whole point: stamped by the writer, readable by anyone. Written at 4, read at 0 was the bug."""
    plate_dir = _write_plate(tmp_path)
    assert read_contract_version(plate_dir) == PLATE_CONTRACT_VERSION


def test_the_stamp_lives_outside_the_ome_namespace(tmp_path):
    """attributes.ome belongs to OME. A private key goes beside it, or a schema we did not write fails."""
    plate_dir = _write_plate(tmp_path)
    attrs = json.loads((plate_dir / "zarr.json").read_text())["attributes"]
    assert "squidmip" in attrs and "plate_contract_version" in attrs["squidmip"]
    assert "squidmip" not in attrs["ome"]
    assert "plate_contract_version" not in attrs["ome"]


def test_the_real_writer_stamps_the_plate(tmp_path, monkeypatch):
    """Not just the fixture: _output.write_plate itself must stamp, at its one plate-group write."""
    import inspect

    src = inspect.getsource(_output)
    assert "attributes=contract_stamp()" in src, "the writer stopped stamping the plate group"
    assert src.count("contract_stamp()") == 1, "the stamp is written at more than one site again"


def test_write_then_read_carries_the_version_end_to_end(tmp_path):
    """The real writer, the real reader, no fixture in between. This is the round trip that counts."""
    from tests.test_output import REGIONS, _image, _meta, _stream

    from squidmip._output import write_from_stream

    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1)
    plate_dir = Path(manifest["plate"])

    assert read_contract_version(plate_dir) == PLATE_CONTRACT_VERSION
    reader = SquidZarrReader(plate_dir)
    assert set(reader.metadata["regions"]) == set(REGIONS)   # plate order, not lexicographic
    assert reader._contract_version == PLATE_CONTRACT_VERSION
    assert validate_plate(plate_dir).ok, validate_plate(plate_dir).summary()


def test_a_reader_round_trip_carries_the_version(tmp_path):
    """The reader compares the stamp on open and remembers what it saw."""
    plate_dir = _write_plate(tmp_path)
    reader = SquidZarrReader(plate_dir)
    reader.metadata                                    # forces _discover
    assert reader._contract_version == PLATE_CONTRACT_VERSION


def test_a_major_mismatch_is_refused_not_warned():
    """A stable guarantee moved. The store would open, find its wells, and place them wrongly."""
    major = int(PLATE_CONTRACT_VERSION.split(".")[0])
    with pytest.raises(PlateContractError) as excinfo:
        compare_contract_version(f"{major + 1}.0")
    message = str(excinfo.value)
    assert f"{major + 1}.0" in message and PLATE_CONTRACT_VERSION in message, \
        "the refusal must name BOTH versions, or the user cannot act on it"


def test_a_major_mismatch_stops_the_reader_opening_the_store(tmp_path):
    """End to end: the policy is enforced at the reader seam, not only in a pure function."""
    major = int(PLATE_CONTRACT_VERSION.split(".")[0])
    plate_dir = _write_plate(tmp_path, version=f"{major + 1}.0")
    with pytest.raises(PlateContractError):
        SquidZarrReader(plate_dir).metadata


def test_a_newer_minor_warns_and_proceeds(tmp_path):
    """A minor bump may only ADD an optional guarantee, so the read is lossy, never wrong."""
    major, minor = (int(p) for p in PLATE_CONTRACT_VERSION.split("."))
    plate_dir = _write_plate(tmp_path, version=f"{major}.{minor + 9}")
    assert compare_contract_version(f"{major}.{minor + 9}") == "minor-ahead"
    with pytest.warns(UserWarning, match="newer than this build"):
        SquidZarrReader(plate_dir).metadata


def test_an_unstamped_store_is_read_without_complaint(tmp_path):
    """Every plate written before this landed, and every third-party NGFF store, is unstamped."""
    plate_dir = _write_plate(tmp_path)
    doc = json.loads((plate_dir / "zarr.json").read_text())
    doc["attributes"].pop("squidmip")
    (plate_dir / "zarr.json").write_text(json.dumps(doc))

    assert read_contract_version(plate_dir) is None
    assert compare_contract_version(None) == "absent"
    meta = SquidZarrReader(plate_dir).metadata          # must not raise
    assert meta["regions"] == ["A1", "B2"]


def test_an_unparseable_stamp_is_refused():
    """Something deliberately made a promise and we cannot tell which one. Worse than no stamp."""
    for bad in ("one.two", "1", "1.2.3", ""):
        with pytest.raises(PlateContractError):
            compare_contract_version(bad)


def test_the_spec_version_and_the_contract_version_are_different_things():
    """_NGFF_VERSION is OME's schema version. Conflating the two is how one stands in for the other."""
    assert _output._NGFF_VERSION == "0.5"
    assert PLATE_CONTRACT_VERSION != _output._NGFF_VERSION


# --- validate: errors are stable violations, warnings are missing optional sidecars -------------

def test_a_conforming_plate_validates_clean(tmp_path):
    report = validate_plate(_write_plate(tmp_path))
    assert report.ok, report.summary()
    assert report.contract_version == PLATE_CONTRACT_VERSION


def test_a_broken_stable_guarantee_is_an_ERROR(tmp_path):
    """Level 0 is declared and missing: the store is not the thing it says it is."""
    plate_dir = _write_plate(tmp_path)
    import shutil

    shutil.rmtree(plate_dir / "A" / "1" / "0" / "0")
    report = validate_plate(plate_dir)
    assert not report.ok
    assert any("not a zarr array" in e for e in report.errors), report.summary()


def test_a_broken_axis_order_is_an_ERROR(tmp_path):
    """TCZYX is stable. A store that reorders it is not readable by this build at all."""
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
    assert report.ok, report.summary()                  # <- the assertion that matters
    assert any("translation" in w for w in report.warnings), report.summary()
    assert any("omero" in w for w in report.warnings), report.summary()
    assert any("coordinates.csv" in w for w in report.warnings), \
        "a warning must NAME the fallback, or it is just noise"


def test_a_single_level_field_is_a_WARNING_and_names_level_zero(tmp_path):
    """Small fields are written single-level ON PURPOSE (_PYRAMID_MIN_YX). Legal, and lossy."""
    report = validate_plate(_write_plate(tmp_path))
    assert report.ok
    assert any("level '0'" in w for w in report.warnings), report.summary()


def test_an_incomplete_marker_is_a_WARNING(tmp_path):
    """A store mid-write is readable; what it promises may simply not all be there yet."""
    plate_dir = _write_plate(tmp_path)
    (plate_dir / ".squidmip-incomplete").write_text("{}")
    report = validate_plate(plate_dir)
    assert report.ok
    assert any("did not finish" in w for w in report.warnings), report.summary()


def test_a_major_mismatch_is_reported_by_validate_rather_than_raised(tmp_path):
    """Same policy, two deliveries: a reader must stop, a validator must finish and list everything."""
    major = int(PLATE_CONTRACT_VERSION.split(".")[0])
    report = validate_plate(_write_plate(tmp_path, version=f"{major + 1}.0"))
    assert not report.ok
    assert any("MAJOR difference" in e for e in report.errors), report.summary()


def test_the_stamp_does_not_disturb_the_official_schema(tmp_path):
    """OME's own pydantic models still pass. The stamp sits beside their namespace, not in it."""
    pytest.importorskip("ome_zarr_models")
    from tests.ngff_check import assert_valid_ngff_plate

    assert_valid_ngff_plate(_write_plate(tmp_path))


def test_the_validator_still_runs_without_ome_zarr_models(tmp_path, monkeypatch):
    """Degrade to structural checks, and SAY SO. 'validated' and 'half-validated' must differ."""
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
    """It was promoted out of tests/ for exactly this. An entry point with no CLI is still test-only."""
    from squidmip.contract.validate import main

    assert main([str(_write_plate(tmp_path))]) == 0
    assert "OK" in capsys.readouterr().out


# --- field_path: the one place that knows the layout --------------------------------------------

def test_field_path_builds_the_documented_layout():
    assert field_path("/p/plate.ome.zarr", "B/2", 7, "1") == "/p/plate.ome.zarr/B/2/7/1"
    assert field_path("/p/plate.ome.zarr", "B/2", 7) == "/p/plate.ome.zarr/B/2/7"


def test_field_path_is_forward_slashed_and_tolerant_of_stray_separators():
    """TensorStore's file kvstore takes POSIX paths on every platform, Windows included."""
    assert field_path("/p/plate.ome.zarr/", "/B/2/", "7", 0) == "/p/plate.ome.zarr/B/2/7/0"
    assert "\\" not in field_path("/p", "B/2", 7, 0)


def test_field_path_does_not_re_derive_the_well_path():
    """wellpath comes from plate.wells[].path verbatim. B2 is B/2, never B/02 (_output.parse_well_id)."""
    assert field_path("/p", "AA/12", 3, 0) == "/p/AA/12/3/0"


def test_field_levels_falls_back_to_level_zero_by_NAME_not_by_accident(tmp_path):
    """The fallback that used to be a bare `except Exception`. It is a documented guarantee now."""
    missing = tmp_path / "nothing-here"
    assert field_levels(missing) == ["0"]
    plate_dir = _write_plate(tmp_path)
    assert field_levels(plate_dir / "A" / "1" / "0") == ["0"]


def test_field_path_is_the_only_place_that_knows_the_layout():
    """Asserted by grepping the source, the way test_tsctx asserts nobody calls ts.open again.

    The pattern is a base joined to three or more slash-separated placeholders in one f-string,
    which is precisely the four reconstructions this seam replaced.

    Scope, stated so the guarantee is not read as wider than it is: this bans INVENTING a path.
    ``_montage`` and ``_tilesource`` reach a field by DESCENDING the plate metadata
    (``wells[].path`` -> ``well.images[].path``), which reads every name instead of assuming it,
    and docs/plate-contract.md names that as the other legitimate route.
    """
    root = Path(__file__).resolve().parent.parent / "squidmip"
    joined = re.compile(r'f"\{[^"{}]+\}/\{[^"{}]+\}/\{[^"{}]+\}')
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.parent.name == "contract":
            continue                                    # the seam is allowed to know the layout
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if joined.search(line):
                offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, (
        "a plate path is being reconstructed outside squidmip/contract again:\n"
        + "\n".join(offenders))


def test_the_viewer_read_paths_go_through_the_seam():
    """The four sites the review counted. A call, not a mention."""
    import inspect

    from squidmip import _viewer

    for func in (_viewer._ZarrLoupeSource._resolve_levels,
                 _viewer._ZarrLoupeSource._open,
                 _viewer._ComputedPlateWorker._read):
        src = inspect.getsource(func)
        assert "field_path(" in src or "field_levels(" in src, \
            f"{func.__qualname__} stopped using the contract seam"
    assert 'json.loads((Path(field) / "zarr.json")' not in inspect.getsource(_viewer), \
        "the loupe hand-parses multiscales again (it was a bare `except Exception`)"


# --- the t axis: what the format guarantees versus what this implementation reads ---------------
#
# Julio, 2026-07-29: users WILL drop multi-timepoint datasets on this tool. Today the store is
# written correctly (project_well runs with t=None, so every timepoint is on disk) and the plate
# overview and loupe read t=0 unconditionally, so such a plate renders as its first frame with no
# error anywhere. Nothing catches it because EVERY fixture in this suite is Nt=1. These three
# tests are the ones that would have.

def test_a_multi_timepoint_plate_is_WARNED_about_not_silently_flattened(tmp_path):
    """The store is valid, so this is a warning. Silence is what makes it a trap."""
    report = validate_plate(_write_plate(tmp_path, n_t=4))
    assert report.ok, report.summary()
    assert any("4 timepoints" in w for w in report.warnings), report.summary()
    assert any("t=0" in w for w in report.warnings), \
        "the warning must say WHAT collapses, or a user cannot tell what they are losing"


def test_a_single_timepoint_plate_says_nothing_about_time(tmp_path):
    """A warning that fires on every plate is a warning nobody reads."""
    report = validate_plate(_write_plate(tmp_path, n_t=1))
    assert not any("timepoint" in w for w in report.warnings), report.summary()


def test_the_documented_t_zero_read_sites_are_still_the_real_ones():
    """The doc's table is a claim about the code. Pin it, or it rots the way the reader prose did.

    This test is expected to FAIL the day someone adds a timepoint selector to the plate view.
    When it does, update the Time section of docs/plate-contract.md in the same commit: that is
    the point of the test.
    """
    import inspect

    from squidmip import _viewer

    for func in (_viewer._ComputedPlateWorker._read, _viewer._ZarrLoupeSource.coarse):
        assert "[0, :, 0]" in inspect.getsource(func), (
            f"{func.__qualname__} no longer hardcodes t=0. Good. Update the Time section of "
            "docs/plate-contract.md, which documents it as a known gap.")
    doc = (Path(__file__).resolve().parent.parent / "docs" / "plate-contract.md").read_text()
    assert "### Time: the format carries it" in doc
    assert "every fixture in the\nsuite is `Nt = 1`" in doc or "Nt = 1" in doc


# --- the document itself ------------------------------------------------------------------------

def test_the_contract_is_written_down_and_split_in_two():
    """A prose contract that drifts into a lie is the defect that pulled this into v1 scope."""
    doc = (Path(__file__).resolve().parent.parent / "docs" / "plate-contract.md").read_text()
    assert "## Stable" in doc and "## Optional, each with its fallback" in doc
    for fallback in ("coordinates.csv", "auto-contrast", 'level `"0"`'):
        assert fallback in doc, f"the optional section stopped naming the {fallback} fallback"
    # Not ported, deliberately: those describe a LIVE producer and v1 is post-acquisition only.
    assert "events.jsonl" in doc and "NOT in this contract" in doc


def test_the_reader_no_longer_says_the_writer_emits_no_translation():
    """The live defect. Two places said the opposite of what _output.py does, inside contract prose."""
    from squidmip import reader

    src = Path(reader.__file__).read_text()
    for i, line in enumerate(src.splitlines(), 1):
        if "emits no translation" in line:
            pytest.fail(f"reader.py:{i} says the writer emits no translation: {line.strip()}")
    assert '"type": "translation"' in Path(_output.__file__).read_text(), \
        "the writer stopped emitting a translation, so the contract prose is wrong the other way"
