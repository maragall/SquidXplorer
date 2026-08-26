"""coordinates.csv -> metadata["fov_positions_um"], on both reader classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from squidxplorer.reader import _fov_positions_um_or_empty, load_fov_positions_um, open_reader


def _csv(rows, header="region,x (mm),y (mm),z (mm)"):
    return header + "\n" + "\n".join(rows) + "\n"


# --- the row-order mapping ------------------------------------------------------------------

def test_positions_map_row_order_to_sorted_fov_ids_per_region(tmp_path):
    """Non-contiguous FOV ids (7, 9, 11) still map in sorted order to rows 1, 2, 3; regions group independently."""
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "B2,50.0,60.0,", "A1,1.5,2.0,", "B2,50.5,60.0,", "A1,2.0,2.0,",
    ]))
    pos = load_fov_positions_um(tmp_path, {"A1": [7, 9, 11], "B2": [0, 1]})
    assert pos == {("A1", 7): (1000, 2000), ("A1", 9): (1500, 2000), ("A1", 11): (2000, 2000),
                   ("B2", 0): (50000, 60000), ("B2", 1): (50500, 60000)}


def test_repeated_positions_per_z_level_are_deduplicated_in_first_seen_order(tmp_path):
    """A 3-z acquisition writes each position 3x. That must still resolve to 2 FOVs, file order preserved."""
    rows = ["A1,9.0,9.0,", "A1,1.0,1.0,"] * 3
    (tmp_path / "coordinates.csv").write_text(_csv(rows))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert pos == {("A1", 0): (9000, 9000), ("A1", 1): (1000, 1000)}


def test_a_position_count_mismatch_raises_named(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,"]))
    with pytest.raises(ValueError, match="distinct stage position"):
        load_fov_positions_um(tmp_path, {"A1": [0, 1, 2]})
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "A1,1.5,2.0,", "A1,2.0,2.0,",
    ]))
    with pytest.raises(ValueError, match="distinct stage position"):
        load_fov_positions_um(tmp_path, {"A1": [0]})


# --- degradation + malformed input ----------------------------------------------------------

def test_unknown_regions_blank_rows_and_header_case_are_tolerated(tmp_path):
    (tmp_path / "coordinates.csv").write_text(
        "region,X (MM),Y (mm),z\nA1,1.0,2.0,\nA1,,,\nZZ9,5.0,5.0,\nA1,1.5,2.0,\n")
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert pos == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000)}


def test_a_malformed_csv_raises_naming_the_line_or_the_missing_columns(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,oops,2.0,"]))
    with pytest.raises(ValueError, match="line 3.*non-numeric"):
        load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    (tmp_path / "coordinates.csv").write_text("region,foo,bar\nA1,1,2\n")
    with pytest.raises(ValueError, match="no recognisable x/y millimetre columns"):
        load_fov_positions_um(tmp_path, {"A1": [0]})


# --- reader integration (both classes expose the key, in MICROMETRES) ------------------------

def test_squid_reader_exposes_fov_positions_in_micrometres(squid_dataset):
    """conftest writes FOVs 0.5 mm apart. In µm that is 500, not 0.5 — the whole bug."""
    root, _ = squid_dataset
    pos = open_reader(root).metadata["fov_positions_um"]
    assert len(pos) == 4
    assert pos[("B2", 0)] == (10000, 20000)
    assert pos[("B2", 1)] == (10500, 20000)
    assert all(isinstance(v, tuple) and len(v) == 2 for v in pos.values())


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


def test_ome_reader_carries_the_key_empty_without_csv_and_placed_with_one(tmp_path):
    """SquidOMEReader shares the interface: the same key, {} without a CSV, real placement with one."""
    root = _ome_acquisition(tmp_path / "acq")
    assert open_reader(root).metadata["fov_positions_um"] == {}
    (root / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,1.5,2.0,"]))
    meta = open_reader(root).metadata
    assert meta["fov_positions_um"] == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000)}


def test_placement_consumes_um_without_rescaling(squid_dataset):
    """End-to-end: reader µm -> _placement px, with no second mm->µm multiply anywhere."""
    from squidxplorer._placement import fov_offsets_px

    root, _ = squid_dataset
    meta = open_reader(root).metadata
    off = fov_offsets_px(meta["fov_positions_um"], "B2", [0, 1], 0.5)
    assert off == {0: (0, 0), 1: (0, 1000)}


# --- graceful degradation: a bad CSV must not sink the whole acquisition --------------------

_MONKEY_HEADER = "region,fov,z_level,x (mm),y (mm),z (um),time"


def _monkey_csv(rows):
    return _csv(rows, header=_MONKEY_HEADER)


@pytest.mark.parametrize("body", [
    None,                                              # truncated: header + ONE row
    "region,foo,bar\nB2,1,2\n",                        # no recognisable x/y columns
    _monkey_csv(["B2,0,0,1.0,2.0,0,t"]),               # the monkey format, too few
], ids=["truncated", "bad-header", "monkey-truncated"])
def test_an_unusable_csv_still_yields_channels_dtype_and_regions(squid_dataset, body):
    """Placement may degrade; identity (regions/channels/dtype) may not."""
    root, _ = squid_dataset
    if body is None:
        lines = (root / "coordinates.csv").read_text().splitlines()
        body = "\n".join(lines[:2]) + "\n"
    (root / "coordinates.csv").write_text(body)
    with pytest.warns(UserWarning, match="unusable"):
        meta = open_reader(root).metadata
    assert meta["channels"], "channels come from filenames + yaml; a short CSV cannot erase them"
    assert meta["dtype"] is not None
    assert meta["regions"] == ["B2", "B3"]
    assert meta["frame_shape"]
    assert meta["fov_positions_um"] == {}


# --- one truncated well must not cost the whole plate its mosaic ----------------------------

def _plate_csv(good_regions, short_region=None, planned=3):
    """A CSV where every *good_regions* entry cross-checks and *short_region* is truncated."""
    rows = [f"{r},{1.0 + i * 0.5},2.0," for r in good_regions for i in range(planned)]
    if short_region:
        rows += [f"{short_region},{10.0 + i * 0.5},20.0," for i in range(planned)]
    return _csv(rows)


def test_one_short_region_keeps_the_good_ones_and_the_warning_names_both(tmp_path):
    """The regression. C3 is unknowable; A1 and B2 are not, and they keep their positions."""
    (tmp_path / "coordinates.csv").write_text(
        _plate_csv(["A1", "B2"], short_region="C3", planned=3)
    )
    fovs = {"A1": [0, 1, 2], "B2": [0, 1, 2], "C3": [0]}   # C3: 3 rows, 1 FOV written

    with pytest.warns(UserWarning) as rec:
        pos = _fov_positions_um_or_empty(tmp_path, fovs)

    assert pos[("A1", 0)] == (1000, 2000), "A1 cross-checks; it must keep its positions"
    assert pos[("A1", 2)] == (2000, 2000)
    assert pos[("B2", 1)] == (1500, 2000)
    assert not any(region == "C3" for region, _ in pos), \
        "C3's mapping is unknowable — it must contribute nothing rather than guess"
    msg = "\n".join(str(w.message) for w in rec)
    assert "unusable" in msg and "C3" in msg, "the refusal must name the region at fault"
    assert "A1" in msg and "B2" in msg, "it must also say which regions kept their positions"
    with pytest.raises(ValueError, match="distinct stage position"):
        load_fov_positions_um(tmp_path, fovs)          # the strict loader stays all-or-nothing


def test_nothing_salvageable_still_degrades_to_empty(tmp_path):
    """Per-REGION salvage, not per-row: every region short, or a header no region can be judged by, is {}."""
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "B2,5.0,6.0,"]))
    with pytest.warns(UserWarning, match="unusable"):
        assert _fov_positions_um_or_empty(tmp_path, {"A1": [0, 1], "B2": [0, 1]}) == {}
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


def test_monkey_header_uses_the_fov_column_whatever_the_row_order(tmp_path):
    """The explicit fov id wins: rows shuffled, regions grouped independently, detected by header not row count."""
    (tmp_path / "coordinates.csv").write_text(_monkey_csv([
        "A1,2,0,3.0,4.0,100.0,t",
        "manual1,0,0,50.0,60.0,0,t",
        "A1,0,0,1.0,2.0,100.0,t",
        "A1,1,0,1.5,2.0,100.0,t",
    ]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1, 2], "manual1": [0]})
    assert pos == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000), ("A1", 2): (3000, 4000),
                   ("manual1", 0): (50000, 60000)}


def test_monkey_positions_are_micrometres_and_z_um_is_not_smuggled_in(tmp_path):
    """The file says mm, the key says _um; ``z (um)`` is not stored, so no key can carry a doubled conversion."""
    (tmp_path / "coordinates.csv").write_text(_monkey_csv(["A1,0,0,98.2245316296875,10.1854,3930.75,t"]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0]})
    assert pos[("A1", 0)] == pytest.approx((98224.5316296875, 10185.4))
    assert all(len(v) == 2 for v in pos.values())


def test_monkey_z_levels_are_deduplicated_per_fov(tmp_path):
    """A 10-z acquisition writes one row per z-level per FOV; that is 2 FOVs, not 20."""
    rows = [f"A1,{fov},{z},{1.0 + 0.5 * fov},2.0,{3930.0 + z},t"
            for z in range(10) for fov in (0, 1)]
    (tmp_path / "coordinates.csv").write_text(_monkey_csv(rows))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert pos == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000)}


@pytest.mark.parametrize("rows, fovs, match", [
    (["A1,0,0,1.0,2.0,0,t"], [0, 1, 2], "stage position"),
    (["A1,0,0,1.0,2.0,0,t", "A1,7,0,1.5,2.0,0,t"], [0, 1], "stage position|fov"),
    (["A1,oops,0,1.0,2.0,0,t"], [0], "line 2"),
    (["A1,0,0,1.0,2.0,0,t", "A1,0,1,5.0,6.0,0,t"], [0], "conflicting|differing"),
], ids=["missing-fov", "fov-not-in-filenames", "non-numeric-fov", "conflicting-positions"])
def test_a_bad_monkey_csv_fails_loud(tmp_path, rows, fovs, match):
    """A truncated, unknown, unparseable or self-contradicting fov column is a refusal, never a guess."""
    (tmp_path / "coordinates.csv").write_text(_monkey_csv(rows))
    with pytest.raises(ValueError, match=match):
        load_fov_positions_um(tmp_path, {"A1": fovs})


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
    (tmp_path / "coordinates.csv").write_text(src.read_text())
    pos = load_fov_positions_um(tmp_path, fovs_per_region)
    assert len(pos) == sum(len(v) for v in fovs_per_region.values())
    assert pos[("manual0", 0)] == pytest.approx((98224.18125, 10185.4))
