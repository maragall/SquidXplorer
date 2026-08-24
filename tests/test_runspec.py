"""RunSpec: the provenance record beside every save, and ONLY beside saves.

The record's fields and JSON are pinned here; where it lands is pinned through
``run_operator_once`` for each of the three writers, because the dispatch is the one place that
sees operator + parameters + regions + reader for all of them. A preview leaves nothing, the
source acquisition is never written, and a failed write is a warning, never the run's failure.
"""

from __future__ import annotations

import json
import re

import pytest

import squidxplorer
from squidxplorer import open_reader
from squidxplorer import _runspec as runspec_mod
from squidxplorer._dispatch import run_operator_once
from squidxplorer._runspec import RUNSPEC_NAME, RunSpec, write_runspec

from .conftest import REGIONS


class _Reader:
    """A reader stub: the dispatch reads only source_id off it when the writers are faked."""

    def __init__(self, source_id=None):
        if source_id is not None:
            self.source_id = source_id


# ------------------------------------------------------------------ the record itself

def test_capture_carries_the_runs_identity_and_environment(blob_operator):
    spec = RunSpec.capture(_Reader("/data/acq"), operator="blob",
                           operator_kwargs={"min_area_px": 30}, regions=["B2"], n_fovs=None)
    assert spec.operator == "blob"
    assert spec.operator_kwargs == {"min_area_px": 30}
    assert spec.regions == ["B2"]
    assert spec.n_fovs is None
    assert spec.source_id == "/data/acq"
    assert spec.squidxplorer_version
    for dep in ("numpy", "tifffile", "zarr"):   # tilefusion rides only when installed
        assert dep in spec.dependencies, dep
    assert "T" in spec.timestamp                # ISO 8601


def test_to_json_is_deterministic_with_sorted_keys():
    spec = RunSpec.capture(_Reader(), operator="mip", n_fovs=1)
    assert spec.to_json() == spec.to_json()
    record = json.loads(spec.to_json())
    assert list(record) == sorted(record)


def test_the_loop_default_sentinel_serializes_by_its_name():
    from squidxplorer._engine import N_FOVS_LOOP_DEFAULT

    spec = RunSpec.capture(_Reader(), operator="mip", n_fovs=N_FOVS_LOOP_DEFAULT)
    assert json.loads(spec.to_json())["n_fovs"] == "N_FOVS_LOOP_DEFAULT"


def test_the_sha_is_none_on_an_installed_wheel(monkeypatch):
    # A wheel has no checkout: git is absent, or answers "not a repository". Both read as None,
    # and the written record SAYS so rather than omitting the field.
    def _no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(runspec_mod.subprocess, "run", _no_git)
    assert runspec_mod._git_sha() is None
    spec = RunSpec.capture(_Reader(), operator="mip", n_fovs=1)
    assert spec.git_sha is None
    assert json.loads(spec.to_json())["git_sha"] is None


def test_the_sha_is_a_full_hex_head_when_running_from_a_checkout():
    sha = runspec_mod._git_sha()
    assert sha is None or re.fullmatch(r"[0-9a-f]{40}", sha), sha


# ------------------------------------------- beside every writer's output, via the dispatch

def test_the_zarr_save_lands_a_runspec_in_the_plate_dir(monkeypatch, tmp_path):
    plate = tmp_path / "out.hcs"
    plate.mkdir()
    manifests = {"plate": str(plate), "n_fields_written": 4, "complete": True, "stopped": False}
    monkeypatch.setattr(squidxplorer, "write_plate", lambda reader, out_dir, **kw: dict(manifests))
    result = run_operator_once(_Reader(), operator="mip", save=True, owed=1,
                               out_dir=plate, n_fovs=1)
    f = plate / RUNSPEC_NAME
    assert f.is_file()
    record = json.loads(f.read_text())
    assert record["operator"] == "mip"
    assert record["result"] == {"n_fields_written": 4, "complete": True, "stopped": False}
    assert result.manifest["runspec"] == str(f)


def test_the_acquisition_format_save_lands_a_runspec_at_its_root(squid_dataset, tmp_path):
    # The real writer, end to end: the manifest's own "path" names where the record lands.
    root, _ = squid_dataset
    out_dir = tmp_path / f"mip_{root.name}"
    result = run_operator_once(open_reader(root), operator="mip", save=True,
                               owed=len(REGIONS), out_dir=out_dir, n_fovs=None)
    f = out_dir / RUNSPEC_NAME
    assert f.is_file() and result.manifest["runspec"] == str(f)
    record = json.loads(f.read_text())
    assert record["operator"] == "mip" and record["source_id"] == str(root)
    assert record["result"]["complete"] is True
    assert not (root / RUNSPEC_NAME).exists(), "the SOURCE acquisition is never written"


def test_the_fused_format_save_lands_a_runspec_at_its_root(monkeypatch, tmp_path):
    from squidxplorer import _fused_output

    fused = tmp_path / "stitch_acq"
    fused.mkdir()
    monkeypatch.setattr(_fused_output, "fused_format_dst", lambda reader, op: fused)
    monkeypatch.setattr(
        _fused_output, "write_fused_acquisition",
        lambda reader, op, dst, **kw: {"path": str(fused), "n_fields_written": 2,
                                       "complete": True, "stopped": False})
    result = run_operator_once(_Reader(), operator="stitch", save=True, owed=2)
    f = fused / RUNSPEC_NAME
    assert f.is_file() and result.manifest["runspec"] == str(f)
    assert json.loads(f.read_text())["operator"] == "stitch"


def test_a_partial_save_still_lands_its_runspec(monkeypatch, tmp_path):
    partial = tmp_path / "out.hcs.partial"
    partial.mkdir()
    monkeypatch.setattr(
        squidxplorer, "write_plate",
        lambda reader, out_dir, **kw: {"plate": str(partial), "n_fields_written": 1,
                                       "complete": False, "stopped": True})
    run_operator_once(_Reader(), operator="mip", save=True, owed=2, out_dir=partial, n_fovs=1)
    record = json.loads((partial / RUNSPEC_NAME).read_text())
    assert record["result"] == {"n_fields_written": 1, "complete": False, "stopped": True}


# --------------------------------------------------------------- where it must NOT land

def test_a_preview_leaves_no_runspec_anywhere(monkeypatch, tmp_path):
    src = tmp_path / "acq"
    src.mkdir()
    monkeypatch.setattr(squidxplorer, "run_plate", lambda reader, **kw: iter(()))
    result = run_operator_once(_Reader(str(src)), operator="mip", save=False, owed=1)
    assert result.manifest is None
    assert list(tmp_path.rglob(RUNSPEC_NAME)) == []


def test_the_source_acquisition_is_refused_even_when_a_manifest_names_it(monkeypatch, tmp_path):
    src = tmp_path / "acq"
    src.mkdir()
    monkeypatch.setattr(squidxplorer, "write_plate",
                        lambda reader, out_dir, **kw: {"plate": str(src)})
    result = run_operator_once(_Reader(str(src)), operator="mip", save=True, owed=1,
                               out_dir=src, n_fovs=1)
    assert not (src / RUNSPEC_NAME).exists()
    assert "runspec" not in result.manifest


def test_a_write_failure_is_a_warning_never_the_runs_failure(monkeypatch, tmp_path, caplog):
    gone = tmp_path / "never" / "made"          # no parent: the write must fail
    monkeypatch.setattr(squidxplorer, "write_plate",
                        lambda reader, out_dir, **kw: {"plate": str(gone),
                                                       "n_fields_written": 1})
    with caplog.at_level("WARNING", logger="squidxplorer._runspec"):
        result = run_operator_once(_Reader(), operator="mip", save=True, owed=1,
                                   out_dir=gone, n_fovs=1)
    assert result.outcome == "ok"               # the save's verdict is untouched
    assert "runspec" not in result.manifest
    assert any("could not write" in r.message for r in caplog.records)


def test_write_runspec_returns_the_path_it_wrote(tmp_path):
    spec = RunSpec.capture(_Reader(), operator="mip", n_fovs=1)
    path = write_runspec(spec, tmp_path, result={"n_fields_written": 3})
    assert path == tmp_path / RUNSPEC_NAME and path.is_file()
    assert json.loads(path.read_text())["result"]["n_fields_written"] == 3
