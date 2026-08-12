"""coordinates.csv -> metadata["fov_positions_um"], on both reader classes.

The CSV records millimetres, the metadata key is micrometres; the conversion is the producer's job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from squidxplorer.reader import _fov_positions_um_or_empty, load_fov_positions_um, open_reader


def _csv(rows, header="region,x (mm),y (mm),z (mm)"):
    return header + "\n" + "\n".join(rows) + "\n"


# --- the row-order mapping ------------------------------------------------------------------

def test_positions_map_row_order_to_sorted_fovs(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "A1,1.5,2.0,", "A1,2.0,2.0,",
    ]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1, 2]})
    assert pos == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000), ("A1", 2): (2000, 2000)}


def test_mapping_follows_sorted_fov_ids_not_their_values(tmp_path):
    """Non-contiguous FOV ids (7, 9, 11) still map in sorted order to rows 1, 2, 3."""
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "A1,1.5,2.0,", "A1,2.0,2.0,",
    ]))
    pos = load_fov_positions_um(tmp_path, {"A1": [7, 9, 11]})
    assert pos[("A1", 7)] == (1000, 2000)
    assert pos[("A1", 9)] == (1500, 2000)
    assert pos[("A1", 11)] == (2000, 2000)


def test_multiple_regions_are_grouped_independently(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "B2,50.0,60.0,", "A1,1.5,2.0,", "B2,50.5,60.0,",
    ]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1], "B2": [0, 1]})
    assert pos[("A1", 1)] == (1500, 2000)
    assert pos[("B2", 1)] == (50500, 60000)


# --- multi-z de-duplication (the check that would otherwise break every real z-stack) --------

def test_repeated_positions_per_z_level_are_deduplicated(tmp_path):
    """A 3-z acquisition writes each position 3x. That must still resolve to 2 FOVs, not fail."""
    rows = []
    for _z in range(3):
        rows += ["A1,1.0,2.0,", "A1,1.5,2.0,"]
    (tmp_path / "coordinates.csv").write_text(_csv(rows))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert pos == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000)}


def test_dedup_preserves_first_seen_order(tmp_path):
    rows = ["A1,9.0,9.0,", "A1,1.0,1.0,", "A1,9.0,9.0,", "A1,1.0,1.0,"]
    (tmp_path / "coordinates.csv").write_text(_csv(rows))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert pos[("A1", 0)] == (9000, 9000)      # first seen wins, file order preserved
    assert pos[("A1", 1)] == (1000, 1000)


# --- the cross-check ------------------------------------------------------------------------

def test_too_few_positions_raises_named(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,"]))
    with pytest.raises(ValueError, match="distinct stage position"):
        load_fov_positions_um(tmp_path, {"A1": [0, 1, 2]})


def test_too_many_positions_raises_named(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "A1,1.5,2.0,", "A1,2.0,2.0,",
    ]))
    with pytest.raises(ValueError, match="distinct stage position"):
        load_fov_positions_um(tmp_path, {"A1": [0]})


# --- degradation + malformed input ----------------------------------------------------------

def test_absent_csv_returns_empty_not_missing(tmp_path):
    """Empty-but-present: consumers use .get()/[] freely and degrade to single-FOV rendering."""
    assert load_fov_positions_um(tmp_path, {"A1": [0]}) == {}


def test_unknown_regions_in_csv_are_ignored(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "ZZ9,5.0,5.0,"]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0]})
    assert set(pos) == {("A1", 0)}


def test_blank_coordinate_rows_are_skipped(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,,,", "A1,1.5,2.0,"]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert len(pos) == 2


def test_non_numeric_coordinate_raises_with_line_number(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,oops,2.0,"]))
    with pytest.raises(ValueError, match="line 3.*non-numeric"):
        load_fov_positions_um(tmp_path, {"A1": [0, 1]})


def test_missing_xy_columns_raises_named(tmp_path):
    (tmp_path / "coordinates.csv").write_text("region,foo,bar\nA1,1,2\n")
    with pytest.raises(ValueError, match="no recognisable x/y millimetre columns"):
        load_fov_positions_um(tmp_path, {"A1": [0]})


def test_header_whitespace_and_case_tolerated(tmp_path):
    (tmp_path / "coordinates.csv").write_text("region,X (MM),Y (mm),z\nA1,1.0,2.0,\n")
    assert load_fov_positions_um(tmp_path, {"A1": [0]}) == {("A1", 0): (1000, 2000)}


# --- reader integration (both classes expose the key) ---------------------------------------

def test_squid_reader_exposes_fov_positions_um(squid_dataset):
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert "fov_positions_um" in meta
    # conftest writes 2 regions x 2 fovs, each repeated per z-level
    assert len(meta["fov_positions_um"]) == 4
    assert meta["fov_positions_um"][("B2", 0)] == (10000, 20000)
    assert meta["fov_positions_um"][("B2", 1)] == (10500, 20000)


def test_fov_positions_um_present_even_without_csv(squid_dataset):
    """The key must exist on every acquisition — a missing key is a KeyError landmine."""
    root, _ = squid_dataset
    (root / "coordinates.csv").unlink()
    meta = open_reader(root).metadata
    assert meta["fov_positions_um"] == {}


def _ome_acquisition(root):
    """A minimal 2-channel OME-TIFF acquisition (mirrors tests/test_reader.py's fixture)."""
    import numpy as np
    import tifffile

    ome = root / "ome_tiff"
    ome.mkdir(parents=True)
    tifffile.imwrite(ome / "A1_0.ome.tiff", np.zeros((2, 2, 2, 16, 16), np.uint16),
                     metadata={"axes": "TZCYX"})
    tifffile.imwrite(ome / "A1_1.ome.tiff", np.zeros((2, 2, 2, 16, 16), np.uint16),
                     metadata={"axes": "TZCYX"})
    (root / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n- name: Fluorescence 405 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#20ADF8'\n      exposure_time_ms: 1.0\n"
        "- name: Fluorescence 488 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#00FF00'\n      exposure_time_ms: 1.0\n")
    (root / "acquisition.yaml").write_text(
        "sample:\n  wellplate_format: 384 well plate\nz_stack:\n  nz: 2\n  delta_z_mm: 0.0\n"
        "time_series:\n  nt: 2\n")
    return root


def test_ome_reader_exposes_fov_positions_um_empty_without_csv(tmp_path):
    """SquidOMEReader shares the interface, so it must carry the same key (empty is fine)."""
    meta = open_reader(_ome_acquisition(tmp_path / "acq")).metadata
    assert meta["fov_positions_um"] == {}


def test_ome_reader_reads_a_sibling_coordinates_csv(tmp_path):
    """An OME acquisition with a coordinates.csv beside it gets real placement for free."""
    root = _ome_acquisition(tmp_path / "acq")
    (root / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,1.5,2.0,"]))
    meta = open_reader(root).metadata
    assert meta["fov_positions_um"] == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000)}


# --- units invariant (world space is MICROMETRES, every key ends in _um) ---------------------

def test_metadata_key_is_um_suffixed_and_no_mm_key_survives(squid_dataset):
    """World space is µm and the key says so; the old un-suffixed mm key must be gone."""
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert "fov_positions_um" in meta
    assert "fov_positions" not in meta
    for key, value in meta.items():
        if key.startswith("fov_positions"):
            assert key.endswith("_um"), f"world-space key {key!r} must end in _um"
            assert all(isinstance(v, tuple) and len(v) == 2 for v in value.values())


def test_positions_are_micrometres_not_millimetres(squid_dataset):
    """conftest writes FOVs 0.5 mm apart. In µm that is 500, not 0.5 — the whole bug."""
    root, _ = squid_dataset
    pos = open_reader(root).metadata["fov_positions_um"]
    dx = pos[("B2", 1)][0] - pos[("B2", 0)][0]
    assert dx == pytest.approx(500.0), f"0.5 mm pitch must read as 500 um, got {dx}"


def test_placement_consumes_um_without_rescaling(squid_dataset):
    """End-to-end: reader µm -> _placement px, with no second mm->µm multiply anywhere."""
    from squidxplorer._placement import fov_offsets_px

    root, _ = squid_dataset
    meta = open_reader(root).metadata
    off = fov_offsets_px(meta["fov_positions_um"], "B2", [0, 1], 0.5)
    assert off == {0: (0, 0), 1: (0, 1000)}


# --- graceful degradation: a truncated CSV must not sink the whole acquisition ---------------

def test_truncated_coordinates_csv_still_yields_channels_and_dtype(squid_dataset):
    """Placement may degrade; identity (regions/channels/dtype) may not."""
    root, _ = squid_dataset
    lines = (root / "coordinates.csv").read_text().splitlines()
    (root / "coordinates.csv").write_text("\n".join(lines[:2]) + "\n")   # header + ONE row

    with pytest.warns(UserWarning, match="unusable"):
        meta = open_reader(root).metadata

    assert meta["channels"], "channels come from filenames + yaml; a short CSV cannot erase them"
    assert meta["dtype"] is not None
    assert meta["regions"] == ["B2", "B3"]
    assert meta["frame_shape"]
    assert meta["fov_positions_um"] == {}    # the only thing lost: placement


def test_malformed_coordinates_csv_header_still_yields_metadata(squid_dataset):
    """Same containment for the other CSV failure mode (no recognisable x/y columns)."""
    root, _ = squid_dataset
    (root / "coordinates.csv").write_text("region,foo,bar\nB2,1,2\n")
    with pytest.warns(UserWarning, match="unusable"):
        meta = open_reader(root).metadata
    assert [c["name"] for c in meta["channels"]]
    assert meta["dtype"] is not None
    assert meta["fov_positions_um"] == {}


# --- one truncated well must not cost the whole plate its mosaic ----------------------------

def _plate_csv(good_regions, short_region=None, planned=3, written=1):
    """A CSV where every *good_regions* entry cross-checks and *short_region* is truncated."""
    rows = [f"{r},{1.0 + i * 0.5},2.0," for r in good_regions for i in range(planned)]
    if short_region:
        rows += [f"{short_region},{10.0 + i * 0.5},20.0," for i in range(planned)]
    return _csv(rows)


def test_one_short_region_does_not_strip_positions_from_the_good_ones(tmp_path):
    """The regression. C3 is unknowable; A1 and B2 are not, and they keep their positions."""
    (tmp_path / "coordinates.csv").write_text(
        _plate_csv(["A1", "B2"], short_region="C3", planned=3)
    )
    fovs = {"A1": [0, 1, 2], "B2": [0, 1, 2], "C3": [0]}   # C3: 3 rows, 1 FOV written

    with pytest.warns(UserWarning, match="unusable"):
        pos = _fov_positions_um_or_empty(tmp_path, fovs)

    assert pos[("A1", 0)] == (1000, 2000), "A1 cross-checks; it must keep its positions"
    assert pos[("A1", 2)] == (2000, 2000)
    assert pos[("B2", 1)] == (1500, 2000)
    assert not any(region == "C3" for region, _ in pos), \
        "C3's mapping is unknowable — it must contribute nothing rather than guess"


def test_the_warning_names_both_what_was_dropped_and_what_survived(tmp_path):
    """A message that says only 'unusable' reads as a whole-plate failure. Name both halves."""
    (tmp_path / "coordinates.csv").write_text(
        _plate_csv(["A1", "B2"], short_region="C3", planned=3)
    )
    fovs = {"A1": [0, 1, 2], "B2": [0, 1, 2], "C3": [0]}

    with pytest.warns(UserWarning) as rec:
        _fov_positions_um_or_empty(tmp_path, fovs)

    msg = "\n".join(str(w.message) for w in rec)
    assert "C3" in msg, "the refusal must name the region at fault"
    assert "A1" in msg and "B2" in msg, "it must also say which regions kept their positions"


def test_strict_loader_still_refuses_the_whole_mapping(tmp_path):
    """load_fov_positions_um keeps its all-or-nothing contract; only the wrapper degrades."""
    (tmp_path / "coordinates.csv").write_text(
        _plate_csv(["A1", "B2"], short_region="C3", planned=3)
    )
    fovs = {"A1": [0, 1, 2], "B2": [0, 1, 2], "C3": [0]}
    with pytest.raises(ValueError, match="distinct stage position"):
        load_fov_positions_um(tmp_path, fovs)


def test_every_region_short_still_degrades_to_empty(tmp_path):
    """Nothing salvageable is still {} — the previous behaviour, when it is the right one."""
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "B2,5.0,6.0,"]))
    with pytest.warns(UserWarning, match="unusable"):
        pos = _fov_positions_um_or_empty(tmp_path, {"A1": [0, 1], "B2": [0, 1]})
    assert pos == {}


def test_a_malformed_file_is_still_all_or_nothing(tmp_path):
    """Per-REGION salvage, not per-row. A header with no x/y columns cannot judge any region."""
    (tmp_path / "coordinates.csv").write_text("region,foo,bar\nA1,1,2\nB2,3,4\n")
    with pytest.warns(UserWarning, match="unusable"):
        assert _fov_positions_um_or_empty(tmp_path, {"A1": [0], "B2": [0]}) == {}


def test_degradation_does_not_swallow_unexpected_errors(squid_dataset, monkeypatch):
    """Only the deliberate ValueErrors degrade; a genuine bug must still surface."""
    import squidxplorer.reader as reader_mod

    def boom(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(reader_mod, "_parse_fov_positions_um", boom)
    root, _ = squid_dataset
    with pytest.raises(RuntimeError, match="disk on fire"):
        open_reader(root).metadata


# ============================================================================================
# The second on-disk coordinates.csv format ("monkey style"), detected by header:
#   (a) monkey-style   region,fov,z_level,x (mm),y (mm),z (um),time
#   (b) 20x-style      region,x (mm),y (mm),z (mm)
# ============================================================================================

_MONKEY_HEADER = "region,fov,z_level,x (mm),y (mm),z (um),time"


def _monkey_csv(rows):
    return _csv(rows, header=_MONKEY_HEADER)


def test_monkey_header_uses_the_fov_column(tmp_path):
    """The explicit fov id wins: row order is irrelevant when the schema states the mapping."""
    (tmp_path / "coordinates.csv").write_text(_monkey_csv([
        "A1,2,0,3.0,4.0,100.0,t",
        "A1,0,0,1.0,2.0,100.0,t",
        "A1,1,0,1.5,2.0,100.0,t",
    ]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1, 2]})
    assert pos == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000), ("A1", 2): (3000, 4000)}


def test_monkey_positions_are_micrometres(tmp_path):
    """Same units contract as the 20x format: the file says mm, the key says _um."""
    (tmp_path / "coordinates.csv").write_text(_monkey_csv(["A1,0,0,98.2245316296875,10.1854,3930.75,t"]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0]})
    assert pos[("A1", 0)] == pytest.approx((98224.5316296875, 10185.4))


def test_monkey_z_um_column_is_not_multiplied_by_1000(tmp_path):
    """``z (um)`` is ALREADY µm. It is not stored, so no key can carry a doubled conversion."""
    (tmp_path / "coordinates.csv").write_text(_monkey_csv(["A1,0,0,1.0,2.0,3930.75,t"]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0]})
    assert all(len(v) == 2 for v in pos.values())     # x,y only — z is not smuggled in


def test_monkey_z_levels_are_deduplicated_per_fov(tmp_path):
    """A 10-z acquisition writes one row per z-level per FOV; that is 2 FOVs, not 20."""
    rows = [f"A1,{fov},{z},{1.0 + 0.5 * fov},2.0,{3930.0 + z},t"
            for z in range(10) for fov in (0, 1)]
    (tmp_path / "coordinates.csv").write_text(_monkey_csv(rows))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert pos == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000)}


def test_monkey_format_detected_by_header_not_row_count(tmp_path):
    """Row count == FOV count, yet the fov column still decides the mapping (rows are shuffled)."""
    (tmp_path / "coordinates.csv").write_text(_monkey_csv([
        "A1,1,0,9.0,9.0,0,t",
        "A1,0,0,1.0,1.0,0,t",
    ]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert pos[("A1", 0)] == (1000, 1000)
    assert pos[("A1", 1)] == (9000, 9000)


def test_twenty_x_format_still_positional_when_no_fov_column(tmp_path):
    """The (b) path is untouched: no fov column -> Nth distinct position is the Nth sorted FOV."""
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,9.0,9.0,", "A1,1.0,1.0,"]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert pos[("A1", 0)] == (9000, 9000)


def test_monkey_multiple_regions_grouped_independently(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_monkey_csv([
        "manual0,0,0,1.0,2.0,0,t", "manual1,0,0,50.0,60.0,0,t", "manual0,1,0,1.5,2.0,0,t",
    ]))
    pos = load_fov_positions_um(tmp_path, {"manual0": [0, 1], "manual1": [0]})
    assert pos[("manual0", 1)] == (1500, 2000)
    assert pos[("manual1", 0)] == (50000, 60000)


def test_monkey_cross_check_missing_fov_raises(tmp_path):
    """A truncated monkey CSV is still a mismatch — fail loud, same as the positional path."""
    (tmp_path / "coordinates.csv").write_text(_monkey_csv(["A1,0,0,1.0,2.0,0,t"]))
    with pytest.raises(ValueError, match="stage position"):
        load_fov_positions_um(tmp_path, {"A1": [0, 1, 2]})


def test_monkey_fov_id_not_in_filenames_raises(tmp_path):
    """A CSV fov id with no matching image is unknowable, not silently dropped."""
    (tmp_path / "coordinates.csv").write_text(_monkey_csv([
        "A1,0,0,1.0,2.0,0,t", "A1,7,0,1.5,2.0,0,t",
    ]))
    with pytest.raises(ValueError, match="stage position|fov"):
        load_fov_positions_um(tmp_path, {"A1": [0, 1]})


def test_monkey_non_numeric_fov_raises_with_line_number(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_monkey_csv(["A1,oops,0,1.0,2.0,0,t"]))
    with pytest.raises(ValueError, match="line 2"):
        load_fov_positions_um(tmp_path, {"A1": [0]})


def test_monkey_conflicting_positions_for_one_fov_raises(tmp_path):
    """Same fov id at two DIFFERENT stage positions is a corrupt file, not a dedup case."""
    (tmp_path / "coordinates.csv").write_text(_monkey_csv([
        "A1,0,0,1.0,2.0,0,t", "A1,0,1,5.0,6.0,0,t",
    ]))
    with pytest.raises(ValueError, match="conflicting|differing"):
        load_fov_positions_um(tmp_path, {"A1": [0]})


def test_monkey_malformed_csv_still_yields_metadata(squid_dataset):
    """The containment holds for the new format too: identity survives, placement degrades."""
    root, _ = squid_dataset
    (root / "coordinates.csv").write_text(_monkey_csv(["B2,0,0,1.0,2.0,0,t"]))   # too few
    with pytest.warns(UserWarning, match="unusable"):
        meta = open_reader(root).metadata
    assert meta["regions"] == ["B2", "B3"]
    assert meta["channels"] and meta["dtype"] is not None
    assert meta["fov_positions_um"] == {}


# --- real data: both formats, real numbers ---------------------------------------------------

_REAL_20X = Path.home() / "Downloads" / "synthetic_2x2_wellplate"
_REAL_MONKEY = (Path.home() / "Downloads"
                / "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")


@pytest.mark.integration
def test_real_20x_format_plate_span():
    """synthetic_2x2_wellplate: 4 wells x 36 FOVs, 20x-style header. Known-good numbers."""
    if not _REAL_20X.is_dir():
        pytest.skip("synthetic_2x2_wellplate not present")
    meta = open_reader(_REAL_20X).metadata
    pos = meta["fov_positions_um"]
    assert len(pos) == 144
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    assert max(xs) - min(xs) == pytest.approx(12526.1, abs=0.5)
    assert max(ys) - min(ys) == pytest.approx(12526.1, abs=0.5)
    a1 = [p for (r, _f), p in pos.items() if r == "A1"]
    assert max(p[0] for p in a1) - min(p[0] for p in a1) == pytest.approx(3526.1, abs=0.5)
    assert meta["pixel_size_um"] == pytest.approx(0.3728571, abs=1e-6)


@pytest.mark.integration
def test_real_monkey_format_parses_with_fov_column(tmp_path):
    """The monkey header on real data, read straight from Downloads (read-only)."""
    if not _REAL_MONKEY.is_dir():
        pytest.skip("10x laser-af z-stack dataset not present")
    src = _REAL_MONKEY / "original_coordinates" / "original_coordinates_0.csv"
    if not src.exists():
        pytest.skip("original_coordinates_0.csv not present")
    header = src.read_text().splitlines()[0]
    assert "fov" in [h.strip() for h in header.split(",")], "expected the monkey header"

    fovs_per_region = open_reader(_REAL_MONKEY).metadata["fovs_per_region"]
    # Parse the monkey file in isolation (link it into a scratch dir; the source is never written).
    (tmp_path / "coordinates.csv").write_text(src.read_text())
    pos = load_fov_positions_um(tmp_path, fovs_per_region)
    assert len(pos) == sum(len(v) for v in fovs_per_region.values())
    assert pos[("manual0", 0)] == pytest.approx((98224.18125, 10185.4))
