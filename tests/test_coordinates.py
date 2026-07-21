"""Tests for squidmip._coordinates (IMA-215) — per-FOV stage positions from coordinates.csv.

Covers the decisions locked in docs/ima-215-eng-review.md:
  D2    prefer {t}/coordinates.csv; dispatch on the header's column set
  D3-R  collapse one-row-per-(fov,z_level) to one entry per FOV; XY tolerance + warn
  D4-R  everything in micrometres
  D5-R  fabricate a fov index ONLY for single-row regions; omit multi-row regions
  D9-R  absent / malformed table -> {} and a warning, never an exception
  D10   parsed once, inside the memoized metadata property
"""

import warnings

import pytest

from squidmip import open_reader
from squidmip._coordinates import XY_TOLERANCE_UM, load_fov_positions_um
from tests.conftest import write_coordinates_labelled, write_coordinates_unlabelled


# --- D2: file location and precedence ---------------------------------------
def test_prefers_timepoint_over_root(tmp_path):
    # root says one thing, the timepoint copy says another — the timepoint copy must win,
    # mirroring the real disagreement (root 98.2245316296875 vs t0 98.22418125).
    write_coordinates_unlabelled(tmp_path, [("B2", 98.2245316296875, 10.0, 0.0)])
    write_coordinates_labelled(tmp_path / "0", {("B2", 0): (98.22418125, 10.0, 3930.75)})

    positions = load_fov_positions_um(tmp_path)
    x_um, _y, _z = positions[("B2", 0)]
    assert x_um == pytest.approx(98.22418125 * 1000)


def test_falls_back_to_root_when_no_timepoint_copy(tmp_path):
    write_coordinates_unlabelled(tmp_path, [("B2", 1.0, 2.0, 0.0)])
    assert load_fov_positions_um(tmp_path)[("B2", 0)][0] == pytest.approx(1000.0)


def test_explicit_time_folder_is_honoured(tmp_path):
    write_coordinates_labelled(tmp_path / "0", {("B2", 0): (5.0, 6.0, 100.0)})
    assert load_fov_positions_um(tmp_path, tmp_path / "0")[("B2", 0)][0] == pytest.approx(5000.0)


def test_absent_table_returns_empty(tmp_path):
    # D9-R: the regression guard. No coordinates.csv anywhere -> {}, no exception.
    assert load_fov_positions_um(tmp_path) == {}


# --- D2: header dispatch -----------------------------------------------------
def test_dispatch_is_on_column_set_not_spacing(tmp_path):
    # Both schemas spell it "x (mm)" with identical spacing, so only the presence of the
    # fov/z_level columns can distinguish them.
    write_coordinates_labelled(tmp_path / "0", {("B2", 3): (1.0, 2.0, 50.0)})
    labelled = load_fov_positions_um(tmp_path)
    assert set(labelled) == {("B2", 3)}          # fov came from the column, not row position

    write_coordinates_unlabelled(tmp_path, [("B2", 1.0, 2.0, 0.0)])
    root_only = load_fov_positions_um(tmp_path, tmp_path)
    assert set(root_only) == {("B2", 0)}         # fabricated, single row


def test_tolerates_bom_crlf_and_whitespace(tmp_path):
    (tmp_path / "coordinates.csv").write_bytes(
        b"\xef\xbb\xbf region , x (mm) , y (mm) , z (mm) \r\nB2,1.0,2.0,\r\n"
    )
    assert load_fov_positions_um(tmp_path)[("B2", 0)][:2] == (
        pytest.approx(1000.0),
        pytest.approx(2000.0),
    )


def test_unrecognised_header_returns_empty_with_warning(tmp_path):
    (tmp_path / "coordinates.csv").write_text("alpha,beta,gamma\n1,2,3\n")
    with pytest.warns(UserWarning, match="not a recognised Squid coordinates table"):
        assert load_fov_positions_um(tmp_path) == {}


def test_headerless_file_returns_empty(tmp_path):
    (tmp_path / "coordinates.csv").write_text("")
    with pytest.warns(UserWarning, match="no header row"):
        assert load_fov_positions_um(tmp_path) == {}


# --- D3-R: collapsing one-row-per-(fov, z_level) ------------------------------
def test_labelled_collapses_to_one_entry_per_fov(tmp_path):
    # Shape of the real 10x dataset: many z levels per FOV, one entry out.
    entries = {("manual0", fov): (98.0 + fov, 10.0, 3930.75) for fov in range(5)}
    write_coordinates_labelled(tmp_path / "0", entries, z_levels=range(10))

    positions = load_fov_positions_um(tmp_path)
    assert len(positions) == 5                    # 50 rows in, 5 entries out
    assert set(positions) == {("manual0", f) for f in range(5)}


def test_z_comes_from_the_lowest_z_level(tmp_path):
    write_coordinates_labelled(tmp_path / "0", {("B2", 0): (1.0, 2.0, 3930.75)}, z_levels=(0, 1, 2))
    assert load_fov_positions_um(tmp_path)[("B2", 0)][2] == pytest.approx(3930.75)


def test_lowest_z_level_used_when_level_zero_absent(tmp_path):
    write_coordinates_labelled(tmp_path / "0", {("B2", 0): (1.0, 2.0, 100.0)}, z_levels=(3, 4, 5))
    # z0_um + 1.5*3 is the value written for the lowest present level
    assert load_fov_positions_um(tmp_path)[("B2", 0)][2] == pytest.approx(100.0 + 1.5 * 3)


def test_float_repr_noise_does_not_warn(tmp_path):
    # The real failure this tolerance exists for: 3930.75 vs 3930.7499999999995.
    (tmp_path / "coordinates.csv").write_text(
        "region,fov,z_level,x (mm),y (mm),z (um),time\n"
        "B2,0,0,98.22418125,10.1854,3930.75,t0\n"
        "B2,0,1,98.224181250000001,10.185400000000001,3932.25,t1\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # any warning fails this test
        positions = load_fov_positions_um(tmp_path)
    assert positions[("B2", 0)][0] == pytest.approx(98.22418125 * 1000)


def test_real_xy_disagreement_warns_but_still_returns(tmp_path):
    shift_mm = (XY_TOLERANCE_UM * 10) / 1000.0    # comfortably beyond tolerance
    (tmp_path / "coordinates.csv").write_text(
        "region,fov,z_level,x (mm),y (mm),z (um),time\n"
        "B2,0,0,10.0,20.0,100.0,t0\n"
        f"B2,0,1,{10.0 + shift_mm},20.0,101.5,t1\n"
    )
    with pytest.warns(UserWarning, match="XY varying"):
        positions = load_fov_positions_um(tmp_path)
    assert positions[("B2", 0)][0] == pytest.approx(10_000.0)   # first row's XY kept


# --- D4-R: micrometres everywhere --------------------------------------------
def test_labelled_units(tmp_path):
    write_coordinates_labelled(tmp_path / "0", {("B2", 0): (1.5, 2.5, 3930.75)})
    x_um, y_um, z_um = load_fov_positions_um(tmp_path)[("B2", 0)]
    assert x_um == pytest.approx(1500.0)          # mm -> um
    assert y_um == pytest.approx(2500.0)
    assert z_um == pytest.approx(3930.75)         # z (um) passes through unscaled


def test_unlabelled_z_mm_is_scaled(tmp_path):
    write_coordinates_unlabelled(tmp_path, [("B2", 1.0, 2.0, 3.93075)], z_blank=False)
    assert load_fov_positions_um(tmp_path)[("B2", 0)][2] == pytest.approx(3930.75)


def test_empty_z_column_yields_none(tmp_path):
    # True of the root schema in 3 of 4 real datasets.
    write_coordinates_unlabelled(tmp_path, [("B2", 1.0, 2.0, 0.0)], z_blank=True)
    assert load_fov_positions_um(tmp_path)[("B2", 0)][2] is None


# --- D5-R: fabricate only when unambiguous ------------------------------------
def test_single_row_region_gets_fov_zero(tmp_path):
    # The sim_1536wp shape: one row per well, so row order carries no information to get wrong.
    rows = [(f"A{i}", 9.9 + i, 9.0, 0.0) for i in range(1, 6)]
    write_coordinates_unlabelled(tmp_path, rows)
    positions = load_fov_positions_um(tmp_path)
    assert set(positions) == {(f"A{i}", 0) for i in range(1, 6)}


def test_multi_row_region_indexed_by_row_order_when_count_agrees(tmp_path):
    # The IMA-187 case: multi-FOV wells in the unlabelled schema with no labelled copy.
    write_coordinates_unlabelled(
        tmp_path,
        [("B2", 1.0, 2.0, 0.0), ("B2", 1.5, 2.0, 0.0), ("B2", 2.0, 2.0, 0.0)],
    )
    positions = load_fov_positions_um(tmp_path, None, {"B2": [0, 1, 2]})
    assert set(positions) == {("B2", 0), ("B2", 1), ("B2", 2)}
    assert positions[("B2", 0)][0] == pytest.approx(1000.0)      # row 0 -> fov 0
    assert positions[("B2", 2)][0] == pytest.approx(2000.0)      # row 2 -> fov 2


def test_multi_row_region_omitted_when_count_disagrees(tmp_path):
    # An aborted run leaves the planned table longer than the images: row position would shift
    # every subsequent FOV, so the region is dropped rather than mis-assigned.
    write_coordinates_unlabelled(
        tmp_path,
        [("B2", 1.0, 2.0, 0.0), ("B2", 1.5, 2.0, 0.0), ("B2", 2.0, 2.0, 0.0)],
    )
    with pytest.warns(UserWarning, match="row count disagrees"):
        positions = load_fov_positions_um(tmp_path, None, {"B2": [0, 1]})
    assert positions == {}


def test_mismatched_region_dropped_while_agreeing_regions_survive(tmp_path):
    write_coordinates_unlabelled(
        tmp_path,
        [("B2", 1.0, 2.0, 0.0), ("B3", 3.0, 4.0, 0.0), ("B3", 3.5, 4.0, 0.0)],
    )
    with pytest.warns(UserWarning, match="row count disagrees"):
        positions = load_fov_positions_um(tmp_path, None, {"B2": [0], "B3": [0]})
    assert set(positions) == {("B2", 0)}


def test_multi_row_without_fovs_per_region_warns_but_indexes(tmp_path):
    write_coordinates_unlabelled(tmp_path, [("B2", 1.0, 2.0, 0.0), ("B2", 1.5, 2.0, 0.0)])
    with pytest.warns(UserWarning, match="without a cross-check"):
        positions = load_fov_positions_um(tmp_path)
    assert set(positions) == {("B2", 0), ("B2", 1)}


def test_labelled_multi_fov_is_not_omitted(tmp_path):
    # The labelled schema has a real fov column, so multi-FOV regions are fine there.
    write_coordinates_labelled(
        tmp_path / "0", {("B2", 0): (1.0, 2.0, 100.0), ("B2", 1): (1.7, 2.0, 100.0)}
    )
    assert set(load_fov_positions_um(tmp_path)) == {("B2", 0), ("B2", 1)}


# --- keying ------------------------------------------------------------------
def test_fov_numbering_restarts_per_region_without_collision(tmp_path):
    # Verified in the real 10x dataset: 28 distinct fov values across 2 regions, 55 FOVs total.
    write_coordinates_labelled(
        tmp_path / "0",
        {
            ("manual0", 0): (1.0, 1.0, 10.0),
            ("manual0", 1): (2.0, 1.0, 10.0),
            ("manual1", 0): (50.0, 1.0, 10.0),
            ("manual1", 1): (51.0, 1.0, 10.0),
        },
    )
    positions = load_fov_positions_um(tmp_path)
    assert len(positions) == 4
    assert positions[("manual0", 0)][0] != positions[("manual1", 0)][0]


# --- D9-R: never raises ------------------------------------------------------
def test_unparseable_rows_are_skipped_not_fatal(tmp_path):
    (tmp_path / "coordinates.csv").write_text(
        "region,x (mm),y (mm),z (mm)\n"
        "B2,1.0,2.0,\n"
        ",3.0,4.0,\n"                       # missing region
        "B4,not-a-number,4.0,\n"            # unparseable x
    )
    with pytest.warns(UserWarning, match="Skipped 2 unparseable row"):
        positions = load_fov_positions_um(tmp_path)
    assert set(positions) == {("B2", 0)}


def test_directory_named_coordinates_csv_does_not_raise(tmp_path):
    (tmp_path / "coordinates.csv").mkdir()
    assert load_fov_positions_um(tmp_path) == {}


# --- fovs_per_region cross-check is advisory, never gating --------------------
def test_positions_without_matching_files_warn_but_are_kept(tmp_path):
    write_coordinates_labelled(tmp_path / "0", {("B2", 0): (1.0, 2.0, 10.0), ("B2", 9): (2.0, 2.0, 10.0)})
    with pytest.warns(UserWarning, match="no matching image files"):
        positions = load_fov_positions_um(tmp_path, None, {"B2": [0]})
    assert set(positions) == {("B2", 0), ("B2", 9)}   # kept — filenames inform, never gate


def test_count_check_is_permutation_invariant_by_construction(tmp_path):
    # Documents the residual risk honestly: the count check cannot detect a reordered file.
    # Both orderings parse, and they disagree — which is why ordering rests on the writer
    # contract (verified below against real data), not on this check.
    write_coordinates_unlabelled(tmp_path, [("B2", 1.0, 0.0, 0.0), ("B2", 5.0, 0.0, 0.0)])
    forward = load_fov_positions_um(tmp_path, None, {"B2": [0, 1]})
    write_coordinates_unlabelled(tmp_path, [("B2", 5.0, 0.0, 0.0), ("B2", 1.0, 0.0, 0.0)])
    reversed_ = load_fov_positions_um(tmp_path, None, {"B2": [0, 1]})
    assert set(forward) == set(reversed_)
    assert forward[("B2", 0)] != reversed_[("B2", 0)]


# --- reader integration (D6, D10) --------------------------------------------
def test_squid_reader_exposes_empty_positions_without_csv(squid_dataset):
    root, _ = squid_dataset
    assert open_reader(root).metadata["fov_positions_um"] == {}


def test_squid_reader_exposes_positions_from_timepoint_csv(squid_dataset):
    root, _ = squid_dataset
    write_coordinates_labelled(
        root / "0",
        {("B2", 0): (1.0, 2.0, 100.0), ("B2", 1): (1.7, 2.0, 100.0),
         ("B3", 0): (5.0, 2.0, 100.0), ("B3", 1): (5.7, 2.0, 100.0)},
    )
    positions = open_reader(root).metadata["fov_positions_um"]
    assert set(positions) == {("B2", 0), ("B2", 1), ("B3", 0), ("B3", 1)}
    assert positions[("B2", 0)] == (pytest.approx(1000.0), pytest.approx(2000.0), pytest.approx(100.0))


def test_positions_parsed_once_across_repeated_metadata_access(squid_dataset, monkeypatch):
    # D10: the engine touches .metadata per well; the table must not be re-read each time.
    root, _ = squid_dataset
    write_coordinates_labelled(root / "0", {("B2", 0): (1.0, 2.0, 100.0)})

    import squidmip.reader as reader_mod

    calls = []
    real = reader_mod.load_fov_positions_um
    monkeypatch.setattr(
        reader_mod, "load_fov_positions_um",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    reader = open_reader(root)
    for _ in range(5):
        reader.metadata["fov_positions_um"]
    assert len(calls) == 1


def test_ome_reader_exposes_positions(tmp_path):
    # D6: both reader classes present the same field. Build a minimal OME acquisition.
    import numpy as np
    import tifffile

    ome = tmp_path / "ome_tiff"
    ome.mkdir()
    tifffile.imwrite(
        ome / "B2_0.ome.tiff", np.zeros((2, 2, 2, 16, 16), np.uint16),   # T,Z,C,Y,X
        metadata={"axes": "TZCYX"}, compression="lzw",
    )
    (tmp_path / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n- name: Fluorescence 405 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#20ADF8'\n      exposure_time_ms: 1.0\n"
        "- name: Fluorescence 488 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#00FF00'\n      exposure_time_ms: 1.0\n"
    )
    (tmp_path / "acquisition.yaml").write_text(
        "sample:\n  wellplate_format: 384 well plate\nz_stack:\n  nz: 2\n  delta_z_mm: 0.0\n"
        "time_series:\n  nt: 2\n"
    )

    write_coordinates_labelled(tmp_path / "0", {("B2", 0): (1.0, 2.0, 100.0)}, z_levels=(0,))

    reader = open_reader(tmp_path)
    assert type(reader).__name__ == "SquidOMEReader"
    assert reader.metadata["fov_positions_um"][("B2", 0)][0] == pytest.approx(1000.0)


def test_row_order_matches_labelled_truth_on_real_data():
    """The assumption row-order fabrication rests on, pinned to real Squid output.

    20x_scan_2025-09-05 is the one dataset carrying BOTH schemas for the same acquisition, so
    the unlabelled root table can be checked against the labelled timepoint table's real fov
    column. If Squid ever stops writing root rows in acquisition order, this fails loudly here
    rather than silently misplacing tiles in the IMA-187 mosaic.
    """
    import csv
    from pathlib import Path

    root_dir = Path("/Users/julioamaragall/Downloads/20x_scan_2025-09-05_17-57-50")
    if not (root_dir / "coordinates.csv").is_file() or not (root_dir / "0" / "coordinates.csv").is_file():
        pytest.skip("20x_scan dataset (with both schemas) not present")

    with (root_dir / "0" / "coordinates.csv").open(newline="", encoding="utf-8-sig") as fh:
        truth = {
            (r["region"], int(r["fov"])): (float(r["x (mm)"]) * 1000, float(r["y (mm)"]) * 1000)
            for r in csv.DictReader(fh)
        }
    fovs_per_region: dict = {}
    for region, fov in truth:
        fovs_per_region.setdefault(region, []).append(fov)

    # Parse the ROOT (unlabelled) table only — this is the fabricating path.
    fabricated = load_fov_positions_um(root_dir, root_dir, fovs_per_region)
    assert set(fabricated) == set(truth), "fabricated keys diverged from the labelled truth"

    worst = max(
        max(abs(fabricated[k][0] - truth[k][0]), abs(fabricated[k][1] - truth[k][1]))
        for k in truth
    )
    assert worst < 2.0, f"row-order fabrication disagrees with labelled truth by {worst:.3f} um"


# --- real data (skip when absent) --------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "/Users/julioamaragall/Downloads/20x_scan_2025-09-05_17-57-50",
        "/Users/julioamaragall/CEPHLA/Data/sim_1536wp",
    ],
)
def test_real_dataset_coordinates_parse(path):
    from pathlib import Path

    root = Path(path)
    if not (root / "coordinates.csv").is_file():
        pytest.skip(f"{root} not present")
    positions = load_fov_positions_um(root)
    assert positions, "expected at least one position from a real coordinates.csv"
    for (region, fov), (x_um, y_um, z_um) in positions.items():
        assert isinstance(region, str) and isinstance(fov, int)
        assert x_um > 0 and y_um > 0
        assert z_um is None or z_um > 0
