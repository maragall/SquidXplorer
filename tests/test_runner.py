"""The runner seam: a declared interface, with the in-process engine as its one implementation."""

from __future__ import annotations

import pytest

import squidxplorer
import squidxplorer._dispatch as dispatch_mod
from squidxplorer._dispatch import run_operator_once
from squidxplorer._engine import N_FOVS_LOOP_DEFAULT
from squidxplorer._runner import InProcessRunner, Runner
from squidxplorer._runspec import RunSpec


class _Reader:
    def __init__(self, source_id=None):
        if source_id is not None:
            self.source_id = source_id


# ------------------------------------------------------------------ the protocol's shape

def test_the_protocol_needs_both_arms_and_the_dispatch_holds_the_in_process_one():
    class SaveOnly:
        def run_save(self, reader, spec, **kw):
            return {}

    assert isinstance(InProcessRunner(), Runner)
    assert not isinstance(SaveOnly(), Runner)
    assert type(dispatch_mod._RUNNER) is InProcessRunner


# --------------------------------------------- parity: the old monkeypatch seams still work

def test_a_monkeypatched_write_plate_still_intercepts_the_save_arm(monkeypatch, tmp_path,
                                                                  blob_operator):
    seen = {}

    def fake_write_plate(reader, out_dir, *, n_fovs=1, workers=None, operator="mip",
                         tiff=True, on_well=None, stop=None, on_error=None,
                         regions=None, operator_kwargs=None):
        seen.update(out_dir=str(out_dir), operator=operator, tiff=tiff, n_fovs=n_fovs,
                    regions=regions, operator_kwargs=operator_kwargs)
        return {"plate": str(tmp_path), "n_fields_written": 1}

    monkeypatch.setattr(squidxplorer, "write_plate", fake_write_plate)
    run_operator_once(_Reader(), operator="blob", save=True, owed=1, out_dir=tmp_path,
                      n_fovs=2, regions=["B2"], parameters={"min_area_px": 30})
    assert seen == {"out_dir": str(tmp_path), "operator": "blob", "tiff": False, "n_fovs": 2,
                    "regions": ["B2"], "operator_kwargs": {"min_area_px": 30}}


def test_a_monkeypatched_run_plate_still_intercepts_the_preview_arm(monkeypatch,
                                                                    blob_operator):
    seen = {}

    def fake_run_plate(reader, **kw):
        seen.update(kw)
        return iter([("B2", 0, None), ("B3", 0, None)])

    monkeypatch.setattr(squidxplorer, "run_plate", fake_run_plate)
    landed = []
    result = run_operator_once(_Reader(), operator="blob", save=False, owed=2,
                               regions=["B2", "B3"], parameters={"min_area_px": 30},
                               on_well=lambda region, fov, image: landed.append(region))
    assert seen["operator"] == "blob" and seen["regions"] == ["B2", "B3"]
    assert seen["operator_kwargs"] == {"min_area_px": 30}
    assert seen["n_fovs"] is N_FOVS_LOOP_DEFAULT
    assert landed == ["B2", "B3"] and result.landed == 2 and result.outcome == "ok"


# ------------------------------------------------- a substituted runner receives the spec

class _RecordingRunner:
    def __init__(self, manifest=None):
        self.calls = []
        self.manifest = manifest or {"plate": "", "n_fields_written": 1}

    def run_save(self, reader, spec, *, out_dir=None, tiff=False, workers=None,
                 on_well=None, on_error=None, stop=None):
        self.calls.append(("save", reader, spec, out_dir))
        return dict(self.manifest)

    def run_preview(self, reader, spec, *, workers=None, on_well=None,
                    on_error=None, stop=None, z_level=None, windows=None):
        self.calls.append(("preview", reader, spec, None))
        return 1, False


def test_a_substituted_runner_receives_the_spec_on_the_save_arm(monkeypatch, tmp_path,
                                                                blob_operator):
    rec = _RecordingRunner(manifest={"plate": str(tmp_path), "n_fields_written": 1})
    monkeypatch.setattr(dispatch_mod, "_RUNNER", rec)
    reader = _Reader("/data/acq")
    result = run_operator_once(reader, operator="blob", save=True, owed=1, out_dir=tmp_path,
                               n_fovs=None, regions={"B2": [0, 1]},
                               parameters={"min_area_px": 30})
    (arm, got_reader, spec, out_dir), = rec.calls
    assert arm == "save" and got_reader is reader and out_dir == tmp_path
    assert isinstance(spec, RunSpec)
    assert spec.operator == "blob" and spec.operator_kwargs == {"min_area_px": 30}
    assert spec.regions == {"B2": [0, 1]} and spec.n_fovs is None
    assert spec.source_id == "/data/acq"
    assert result.outcome == "ok"
    assert (tmp_path / "runspec.json").is_file(), \
        "the provenance write is the dispatch's, not the runner's"


def test_a_substituted_runner_receives_the_spec_on_the_preview_arm(monkeypatch):
    rec = _RecordingRunner()
    monkeypatch.setattr(dispatch_mod, "_RUNNER", rec)
    result = run_operator_once(_Reader(), operator="mip", save=False, owed=1)
    (arm, _reader, spec, _), = rec.calls
    assert arm == "preview" and isinstance(spec, RunSpec)
    assert spec.operator == "mip" and spec.operator_kwargs is None
    assert spec.n_fovs is N_FOVS_LOOP_DEFAULT
    assert result.manifest is None and result.landed == 1


def test_the_copy_arm_reaches_a_runner_as_a_preview_with_copy_true(monkeypatch, tmp_path):
    rec = _RecordingRunner()
    monkeypatch.setattr(dispatch_mod, "_RUNNER", rec)
    src = tmp_path / "acq"
    src.mkdir()
    result = run_operator_once(_Reader(str(src)), operator="register", save=True, owed=1,
                               n_fovs=None)
    (arm, _reader, spec, _), = rec.calls
    assert arm == "preview"
    assert spec.operator_kwargs == {"copy": True}
    assert result.out_path == str(src.parent / f"stitched_{src.name}")
    assert list(tmp_path.rglob("runspec.json")) == [], \
        "the copy arm writes no manifest, so no runspec names a root to land at"
