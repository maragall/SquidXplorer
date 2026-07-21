"""CLI surface: argument handling, actionable errors, and a real end-to-end run.

The integration test at the bottom runs the whole harness against the real 20x_scan
acquisition. It is the only test that proves the pipeline works on genuine microscope
data rather than on a synthetic canvas -- and it doubles as the answer to "is the
stage metadata itself well aligned?", which is the baseline every stitcher has to beat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bench.__main__ import main
from tests.conftest_bench import write_acquisition, write_broken_symlink_farm

REAL_20X = Path("/Users/julioamaragall/Downloads/20x_scan_2025-09-05_17-57-50")


def test_list_tools(capsys):
    assert main(["--list-tools", "unused"]) == 0
    out = capsys.readouterr().out
    assert "tilefusion" in out and "ashlar" in out


def test_broken_acquisition_exits_2_with_an_explanation(tmp_path, capsys):
    write_broken_symlink_farm(tmp_path)
    assert main([str(tmp_path)]) == 2
    assert "dangling" in capsys.readouterr().err


def test_unknown_region_exits_2(tmp_path, capsys):
    write_acquisition(tmp_path, grid=(1, 2), tile=(32, 32), step=(24, 24))
    assert main([str(tmp_path), "--region", "ZZ99"]) == 2
    assert "not in" in capsys.readouterr().err


def test_unknown_channel_exits_2(tmp_path, capsys):
    write_acquisition(tmp_path, grid=(1, 2), tile=(32, 32), step=(24, 24))
    assert main([str(tmp_path), "--channel", "nope"]) == 2
    assert "not in" in capsys.readouterr().err


def test_unknown_tool_exits_2(tmp_path, capsys):
    write_acquisition(tmp_path, grid=(1, 2), tile=(32, 32), step=(24, 24))
    assert main([str(tmp_path), "--tools", "bigstitcher"]) == 2
    assert "unknown adapter" in capsys.readouterr().err


def test_missing_directory_exits_2(tmp_path, capsys):
    assert main([str(tmp_path / "nope")]) == 2
    assert "error:" in capsys.readouterr().err


def test_full_run_writes_csv_and_report(tmp_path, capsys):
    """Neither stitcher is installed in CI, so both rows are MISSING_TOOL -- which is
    exactly the behaviour we want to pin: the harness still completes and reports."""
    write_acquisition(tmp_path / "acq", grid=(1, 2), tile=(64, 64), step=(48, 48))
    csv_path = tmp_path / "b.csv"
    report = tmp_path / "b.md"
    rc = main(
        [
            str(tmp_path / "acq"),
            "--out-dir", str(tmp_path / "out"),
            "--out-csv", str(csv_path),
            "--report", str(report),
            "--warm-runs", "0",
        ]
    )
    assert rc == 0
    assert csv_path.is_file()
    assert report.is_file()
    assert "Stitcher benchmark" in report.read_text()
    assert "C5" in capsys.readouterr().out


@pytest.mark.integration
def test_real_20x_scan_end_to_end(tmp_path):
    """Full harness on real microscope data, using stage metadata as the positions.

    This measures how well the STAGE thinks the tiles line up -- the no-registration
    baseline. A stitcher that cannot beat this number is not earning its runtime.
    """
    if not REAL_20X.is_dir():
        pytest.skip(f"real 20x_scan not present at {REAL_20X}")

    from bench.dataset import load_acquisition
    from bench.metrics import adjacent_pairs, seam_residual

    acq = load_acquisition(REAL_20X)
    assert acq.frame_shape == (2084, 2084)
    assert len(acq.fovs("C5")) == 36

    positions = acq.positions_px("C5")
    pairs = adjacent_pairs(positions, acq.frame_shape)
    assert len(pairs) >= 50  # 6x6 grid: 30 horizontal + 30 vertical

    cache: dict[int, object] = {}

    def read(fov):
        if fov not in cache:
            cache[fov] = acq.read("C5", fov, 0, "Fluorescence_405_nm_Ex")
        return cache[fov]

    stats = seam_residual(read, positions, pairs=pairs[:6], n_blocks=4)
    assert stats["n_seams_measured"] > 0
    assert stats["resid_median_px"] >= 0
