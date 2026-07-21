"""End-to-end runner behaviour, driven by fake adapters that spawn real subprocesses.

Fakes rather than real stitchers, because ASHLAR and tilefusion are not installed in
CI -- but the subprocess, the sampler, the watchdog, the timeout, the reclaim and the
quality scoring are all the genuine code paths. Each fake is a one-line python -c, so
these tests exercise the harness rather than a mock of it.

Every branch of the status enum is covered here: OK, MISSING_TOOL, CRASH, TIMEOUT,
DISK_ABORT, and QUALITY_NA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bench.adapters.base import StitcherAdapter, StitchRequest
from bench.dataset import load_acquisition
from bench.report import (
    STATUS_CRASH,
    STATUS_DISK_ABORT,
    STATUS_MISSING_TOOL,
    STATUS_OK,
    STATUS_QUALITY_NA,
    STATUS_TIMEOUT,
    read_csv,
)
from bench.runner import RunConfig, run_benchmark, run_one
from tests.conftest_bench import write_acquisition

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def acq(tmp_path):
    write_acquisition(
        tmp_path / "acq", grid=(2, 3), tile=(128, 128), step=(96, 96), seed=21
    )
    return load_acquisition(tmp_path / "acq")


@pytest.fixture
def cfg(acq):
    return RunConfig(
        region="C5",
        channel=acq.channels[0],
        z=0,
        timeout_s=60,
        sampler_interval_s=0.05,
        warm_runs=0,
        min_free_bytes=1,
    )


class _Fake(StitcherAdapter):
    """Adapter whose subprocess is a python -c snippet we control."""

    name = "fake"

    def __init__(self, snippet: str = "pass", available: bool = True, version: str = "1.2.3"):
        self._snippet = snippet
        self._available = available
        self._version = version

    def is_available(self) -> bool:
        return self._available

    def version(self) -> str:
        return self._version

    def build_command(self, req: StitchRequest) -> list[str]:
        return [sys.executable, "-c", self._snippet, str(req.out_dir)]


def _emit_positions(positions: dict[int, tuple[float, float]]) -> str:
    """Snippet that writes the positions.json contract, like a real driver would."""
    payload = json.dumps({str(k): [v[0], v[1]] for k, v in positions.items()})
    return (
        "import sys,json,pathlib;"
        "d=pathlib.Path(sys.argv[1]);d.mkdir(parents=True,exist_ok=True);"
        f"(d/'positions.json').write_text(json.dumps({{'positions_px':{payload}}}))"
    )


# --------------------------------------------------------------------- happy path


def test_perfect_positions_score_near_zero(acq, cfg, tmp_path):
    truth = acq.positions_px("C5")
    row = run_one(
        _Fake(_emit_positions(truth)), acq, cfg, tmp_path / "out", repo_root=REPO_ROOT
    )
    assert row.status == STATUS_OK
    assert row.tool_version == "1.2.3"
    assert row.n_seams_measured >= 4
    assert row.resid_median_px < 0.5
    assert row.t_wall_cold_s > 0


def test_a_worse_stitcher_scores_worse(acq, cfg, tmp_path):
    """The table has to be able to RANK. Two tools, one deliberately misaligned."""
    truth = acq.positions_px("C5")
    bad = dict(truth)
    bad[1] = (bad[1][0] + 4.0, bad[1][1] - 3.0)  # 5 px off

    good_row = run_one(
        _Fake(_emit_positions(truth)), acq, cfg, tmp_path / "g", repo_root=REPO_ROOT
    )
    bad_row = run_one(
        _Fake(_emit_positions(bad)), acq, cfg, tmp_path / "b", repo_root=REPO_ROOT
    )
    assert good_row.resid_p90_px < bad_row.resid_p90_px
    assert bad_row.resid_p90_px == pytest.approx(5.0, abs=1.5)


def test_row_records_dataset_geometry(acq, cfg, tmp_path):
    row = run_one(
        _Fake(_emit_positions(acq.positions_px("C5"))), acq, cfg, tmp_path / "o",
        repo_root=REPO_ROOT,
    )
    assert row.n_tiles == 6
    assert (row.tile_y, row.tile_x) == (128, 128)
    assert row.dataset == "acq"
    assert row.region == "C5"


# ------------------------------------------------------------------ failure paths


def test_missing_tool(acq, cfg, tmp_path):
    row = run_one(_Fake(available=False), acq, cfg, tmp_path / "o", repo_root=REPO_ROOT)
    assert row.status == STATUS_MISSING_TOOL
    assert row.quality_status == STATUS_QUALITY_NA
    assert row.quality_na_reason == "not_run"


def test_crash_records_the_exit_code(acq, cfg, tmp_path):
    row = run_one(
        _Fake("import sys;sys.stderr.write('boom\\n');sys.exit(7)"),
        acq, cfg, tmp_path / "o", repo_root=REPO_ROOT,
    )
    assert row.status == STATUS_CRASH
    assert "exit 7" in row.detail
    assert row.quality_status == STATUS_QUALITY_NA


def test_timeout_kills_the_subprocess(acq, cfg, tmp_path):
    cfg.timeout_s = 0.5
    row = run_one(
        _Fake("import time;time.sleep(30)"), acq, cfg, tmp_path / "o", repo_root=REPO_ROOT
    )
    assert row.status == STATUS_TIMEOUT
    assert "0.5" in row.detail


def test_disk_abort_from_preflight(acq, cfg, tmp_path):
    cfg.min_free_bytes = 1 << 62  # nothing can satisfy this
    row = run_one(
        _Fake(_emit_positions(acq.positions_px("C5"))), acq, cfg, tmp_path / "o",
        repo_root=REPO_ROOT,
    )
    assert row.status == STATUS_DISK_ABORT
    assert "preflight" in row.detail


def test_disk_abort_mid_run_kills_the_subprocess(acq, cfg, tmp_path, monkeypatch):
    """Preflight passes, then free space collapses while the tool is running."""
    import bench.sampler as sampler_mod

    calls = {"n": 0}
    real_free = sampler_mod.free_bytes

    def collapsing(path):
        calls["n"] += 1
        return real_free(path) if calls["n"] <= 1 else 0

    monkeypatch.setattr(sampler_mod, "free_bytes", collapsing)
    row = run_one(
        _Fake("import time;time.sleep(30)"), acq, cfg, tmp_path / "o", repo_root=REPO_ROOT
    )
    assert row.status == STATUS_DISK_ABORT
    assert "low-water" in row.detail


def test_no_positions_is_quality_na_not_a_zero_score(acq, cfg, tmp_path):
    """A tool that ran fine but emitted no positions cannot be scored. Say so."""
    row = run_one(_Fake("pass"), acq, cfg, tmp_path / "o", repo_root=REPO_ROOT)
    assert row.status == STATUS_OK
    assert row.quality_status == STATUS_QUALITY_NA
    assert row.quality_na_reason == "no_positions"


def test_non_rigid_tool_is_quality_na(acq, cfg, tmp_path):
    """BigStitcher's model has no scalar shift: undefined, not unimplemented."""

    class NonRigid(_Fake):
        name = "bigstitcher-like"
        supports_quality = False
        quality_na_reason = "non_rigid_model"

    row = run_one(
        NonRigid(_emit_positions(acq.positions_px("C5"))), acq, cfg, tmp_path / "o",
        repo_root=REPO_ROOT,
    )
    assert row.quality_status == STATUS_QUALITY_NA
    assert row.quality_na_reason == "non_rigid_model"


def test_blank_tiles_are_quality_na_not_a_perfect_score(tmp_path, monkeypatch):
    """Blank overlaps fail the NCC gate. Reporting 0 px would flatter the tool."""
    write_acquisition(
        tmp_path / "blank", grid=(1, 2), tile=(128, 128), step=(96, 96), blank=True
    )
    acq = load_acquisition(tmp_path / "blank")
    cfg = RunConfig(
        region="C5", channel=acq.channels[0], timeout_s=60,
        sampler_interval_s=0.05, warm_runs=0, min_free_bytes=1,
    )
    row = run_one(
        _Fake(_emit_positions(acq.positions_px("C5"))), acq, cfg, tmp_path / "o",
        repo_root=REPO_ROOT,
    )
    assert row.quality_status == STATUS_QUALITY_NA
    assert row.quality_na_reason == "no_texture"


def test_non_overlapping_positions_are_quality_na(acq, cfg, tmp_path):
    apart = {f: (0.0, i * 100_000.0) for i, f in enumerate(acq.fovs("C5"))}
    row = run_one(
        _Fake(_emit_positions(apart)), acq, cfg, tmp_path / "o", repo_root=REPO_ROOT
    )
    assert row.quality_status == STATUS_QUALITY_NA
    assert row.quality_na_reason == "no_overlap"


# ----------------------------------------------------------------------- reclaim


def test_output_is_reclaimed_so_peak_disk_stays_at_one_mosaic(acq, cfg, tmp_path):
    out = tmp_path / "out"
    snippet = (
        "import sys,pathlib;d=pathlib.Path(sys.argv[1]);d.mkdir(parents=True,exist_ok=True);"
        "(d/'big.bin').write_bytes(b'0'*200000)"
    )
    row = run_one(_Fake(snippet), acq, cfg, out, repo_root=REPO_ROOT)
    assert row.output_bytes >= 200000  # measured before deletion
    assert not (out / "fake").exists()  # and then reclaimed


def test_keep_output_preserves_the_directory(acq, cfg, tmp_path):
    cfg.keep_output = True
    out = tmp_path / "out"
    run_one(
        _Fake(_emit_positions(acq.positions_px("C5"))), acq, cfg, out, repo_root=REPO_ROOT
    )
    assert (out / "fake" / "positions.json").is_file()


def test_output_reclaimed_even_after_a_crash(acq, cfg, tmp_path):
    out = tmp_path / "out"
    run_one(_Fake("import sys;sys.exit(3)"), acq, cfg, out, repo_root=REPO_ROOT)
    assert not (out / "fake").exists()


# --------------------------------------------------------------------- warm runs


def test_warm_run_replaces_the_cold_wall_time(acq, cfg, tmp_path):
    cfg.warm_runs = 1
    row = run_one(
        _Fake(_emit_positions(acq.positions_px("C5"))), acq, cfg, tmp_path / "o",
        repo_root=REPO_ROOT,
    )
    assert row.t_wall_cold_s > 0
    assert row.t_wall_s > 0


# ----------------------------------------------------------------- full benchmark


def test_run_benchmark_writes_one_row_per_tool(acq, cfg, tmp_path):
    truth = acq.positions_px("C5")
    good = _Fake(_emit_positions(truth))
    good.name = "good"
    missing = _Fake(available=False)
    missing.name = "missing"
    crash = _Fake("import sys;sys.exit(1)")
    crash.name = "crash"

    csv_path = tmp_path / "b.csv"
    rows = run_benchmark(
        [good, missing, crash], acq, cfg, tmp_path / "out",
        csv_path=csv_path, repo_root=REPO_ROOT,
    )
    assert [r.status for r in rows] == [STATUS_OK, STATUS_MISSING_TOOL, STATUS_CRASH]

    written = read_csv(csv_path)
    assert [r["tool"] for r in written] == ["good", "missing", "crash"]
    assert float(written[0]["resid_median_px"]) < 0.5


def test_rows_are_written_incrementally(acq, cfg, tmp_path):
    """A run killed at tool 3 must still leave tools 1 and 2 on disk."""
    a = _Fake(_emit_positions(acq.positions_px("C5")))
    a.name = "first"
    b = _Fake("import sys;sys.exit(2)")
    b.name = "second"
    csv_path = tmp_path / "b.csv"
    run_benchmark([a, b], acq, cfg, tmp_path / "out", csv_path=csv_path, repo_root=REPO_ROOT)
    assert len(read_csv(csv_path)) == 2
