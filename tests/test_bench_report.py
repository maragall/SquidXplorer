"""Row schema, CSV append, and the comparison table.

Central rule under test: an unmeasurable cell renders "n/a" with a stated reason. It
must never render as blank (reads as "the tool failed") or 0 (reads as "perfect").
That distinction is the difference between an honest table and a misleading one.
"""

from __future__ import annotations

import csv

import pytest

from bench.report import (
    CSV_COLUMNS,
    STATUS_DISK_ABORT,
    STATUS_MISSING_TOOL,
    STATUS_OK,
    STATUS_QUALITY_NA,
    BenchmarkRow,
    append_csv,
    markdown_table,
    read_csv,
)


def test_row_autofills_provenance():
    row = BenchmarkRow(tool="tilefusion")
    assert row.timestamp.endswith("Z")
    assert row.host
    assert row.platform


def test_schema_has_no_git_sha():
    """git_sha is meaningless for external tools; tool_version replaces it."""
    assert "git_sha" not in CSV_COLUMNS
    assert "tool_version" in CSV_COLUMNS


def test_append_writes_header_once(tmp_path):
    path = tmp_path / "b.csv"
    append_csv(BenchmarkRow(tool="a"), path)
    append_csv(BenchmarkRow(tool="b"), path)
    with path.open() as fh:
        lines = list(csv.reader(fh))
    assert lines[0] == list(CSV_COLUMNS)
    assert len(lines) == 3
    assert [r["tool"] for r in read_csv(path)] == ["a", "b"]


def test_append_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "b.csv"
    append_csv(BenchmarkRow(tool="a"), path)
    assert path.is_file()


def test_append_refuses_a_mismatched_schema(tmp_path):
    """Appending under a different header silently shifts every column. Refuse."""
    path = tmp_path / "b.csv"
    path.write_text("only,three,cols\n1,2,3\n")
    with pytest.raises(ValueError, match="different schema"):
        append_csv(BenchmarkRow(tool="a"), path)


def test_round_trip_preserves_values(tmp_path):
    path = tmp_path / "b.csv"
    append_csv(
        BenchmarkRow(tool="ashlar", tool_version="1.18.0", t_wall_s=12.5, resid_median_px=0.42),
        path,
    )
    row = read_csv(path)[0]
    assert row["tool"] == "ashlar"
    assert row["tool_version"] == "1.18.0"
    assert float(row["t_wall_s"]) == pytest.approx(12.5)
    assert float(row["resid_median_px"]) == pytest.approx(0.42)


# ------------------------------------------------------------------------- table


def test_table_renders_a_measured_row():
    table = markdown_table(
        [
            BenchmarkRow(
                tool="tilefusion",
                tool_version="0.9",
                status=STATUS_OK,
                t_wall_s=10.0,
                peak_rss_mb=512.0,
                output_bytes=1024**3,
                resid_median_px=0.31,
                resid_p90_px=0.88,
                n_seams_measured=7,
            )
        ]
    )
    assert "tilefusion" in table
    assert "0.31" in table
    assert "1.0 GB" in table


def test_unmeasurable_rss_renders_na_not_zero():
    """A Docker-hosted tool's RSS is invisible; a small number would be a lie."""
    table = markdown_table(
        [BenchmarkRow(tool="mcmicro", peak_rss_mb=3.0, rss_measurable=False)]
    )
    row = [ln for ln in table.splitlines() if ln.startswith("| mcmicro")][0]
    assert "n/a" in row
    assert "3.00" not in row


def test_unmeasurable_output_bytes_renders_na():
    table = markdown_table(
        [BenchmarkRow(tool="mcmicro", output_bytes=10, output_bytes_measurable=False)]
    )
    row = [ln for ln in table.splitlines() if ln.startswith("| mcmicro")][0]
    assert "n/a" in row


def test_quality_na_renders_na_and_explains_itself():
    table = markdown_table(
        [
            BenchmarkRow(
                tool="bigstitcher",
                quality_status=STATUS_QUALITY_NA,
                quality_na_reason="non_rigid_model",
                resid_median_px=0.0,
            )
        ]
    )
    row = [ln for ln in table.splitlines() if ln.startswith("| bigstitcher")][0]
    assert "n/a" in row
    assert "0.00" not in row
    assert "non-rigid" in table


def test_missing_tool_row_is_still_reported():
    table = markdown_table([BenchmarkRow(tool="petakit5d", status=STATUS_MISSING_TOOL)])
    assert "petakit5d" in table
    assert STATUS_MISSING_TOOL in table


def test_disk_abort_row_is_visible_in_the_table():
    table = markdown_table([BenchmarkRow(tool="ashlar", status=STATUS_DISK_ABORT)])
    assert STATUS_DISK_ABORT in table


def test_nan_metrics_render_as_na():
    table = markdown_table([BenchmarkRow(tool="x")])
    assert "n/a" in table


def test_empty_table_says_so():
    assert "No benchmark rows" in markdown_table([])


def test_table_accepts_dicts_from_a_csv(tmp_path):
    path = tmp_path / "b.csv"
    append_csv(BenchmarkRow(tool="tilefusion", t_wall_s=1.0), path)
    table = markdown_table(read_csv(path))
    assert "tilefusion" in table


def test_table_carries_the_metric_caveats():
    table = markdown_table([BenchmarkRow(tool="x")])
    assert "solved positions" in table
    assert "sampled" in table
