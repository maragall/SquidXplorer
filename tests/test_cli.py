"""CLI (IMA-186) tests: the declarative params model + the run() that drives write_plate.

Headless, no Qt. Uses the shared tiny `squid_dataset` fixture (a real 2-well acquisition on disk).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from squidmip._cli import ProcessParameters, run


def test_input_folder_validator_rejects_missing(tmp_path):
    with pytest.raises(ValueError):
        ProcessParameters(input_folder=str(tmp_path / "nope"))


def test_run_writes_navigable_plate(squid_dataset, tmp_path):
    root, _ = squid_dataset                       # tiny real acquisition (B2, B3)
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), tiff=False)
    manifest = run(params)

    plate = Path(manifest["plate"])
    assert plate.name == "plate.ome.zarr"
    assert plate.parent.name.endswith(".hcs")     # <acq-name>.hcs sibling
    assert manifest["n_wells"] == 2
    assert manifest["tiff"] is None                # CLI default: no uncompressed TIFF duplicate
    # the plate group + both wells' fields are on disk (level 0 present)
    assert (plate / "zarr.json").exists()
    for row, col in (("B", "2"), ("B", "3")):
        assert (plate / row / col / "0" / "zarr.json").exists()


def test_run_skips_unreadable_well_instead_of_aborting(squid_dataset, tmp_path):
    # Resilience (IMA-186): one corrupt/missing plane must NOT abort a whole-plate run — the bad
    # well is SKIPPED (logged + reported), the good wells still write.
    root, _ = squid_dataset                       # B2, B3
    victim = sorted((Path(root) / "0").glob("B3_*"))[0]
    victim.unlink()                               # break B3 (a plane it needs is now gone)
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path))
    manifest = run(params)
    assert manifest["skipped"] == ["B3"]          # bad well skipped, not fatal
    assert manifest["n_fields_written"] == 1      # B2 still written
    plate = Path(manifest["plate"])
    assert (plate / "B" / "2" / "0" / "zarr.json").exists()
    assert not (plate / "B" / "3" / "0" / "0").exists()   # B3 field never written


def test_run_defaults_output_next_to_acquisition(squid_dataset):
    root, _ = squid_dataset
    params = ProcessParameters(input_folder=str(root))     # no output_folder -> sibling of the acq
    assert params.output_folder is None
    manifest = run(params)
    assert Path(manifest["plate"]).parent.parent == Path(root).parent


# --- IMA-230 storage guard -----------------------------------------------------------------------

def test_min_free_gb_rejects_negative():
    with pytest.raises(ValueError):
        ProcessParameters(input_folder=".", min_free_gb=-1)


def test_min_free_gb_defaults_to_five(squid_dataset):
    root, _ = squid_dataset
    assert ProcessParameters(input_folder=str(root)).min_free_gb == 5.0


def test_run_aborts_before_writing_when_disk_cannot_hold_one_field(squid_dataset, tmp_path,
                                                                   monkeypatch):
    """Acceptance #1: pre-flight rejection writes nothing at all."""
    import shutil as _sh
    from squidmip._storage import InsufficientDiskSpace

    monkeypatch.setattr(
        _sh, "disk_usage", lambda p: type("U", (), {"total": 100, "used": 99, "free": 1})()
    )
    root, _ = squid_dataset
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path))
    with pytest.raises(InsufficientDiskSpace):
        run(params)
    assert not list(tmp_path.glob("*.hcs")), "nothing may be written when pre-flight rejects"


def test_run_proceeds_when_bound_exceeds_free_but_compression_may_save_it(squid_dataset, tmp_path,
                                                                         monkeypatch, caplog):
    """The bound is uncompressed. Refusing here would be the cry-wolf failure — warn and run."""
    import logging as _lg
    import shutil as _sh

    real = _sh.disk_usage
    root, _ = squid_dataset
    # Plenty for the tiny real fields, but we shrink the *reported* total so the bound looks huge.
    monkeypatch.setattr(
        _sh, "disk_usage",
        lambda p: type("U", (), {"total": 1 << 40, "used": 0, "free": 40_000})(),
    )
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), min_free_gb=0)
    with caplog.at_level(_lg.WARNING, logger="squidmip"):
        run(params)
    assert any("disk check" in r.message for r in caplog.records)


def test_run_disk_check_logs_when_it_fits(squid_dataset, tmp_path, caplog):
    import logging as _lg

    root, _ = squid_dataset
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), min_free_gb=0)
    with caplog.at_level(_lg.INFO, logger="squidmip"):
        run(params)
    assert any("fits uncompressed" in r.message for r in caplog.records)


def test_main_exits_nonzero_with_a_clean_message_not_a_traceback(squid_dataset, tmp_path,
                                                                 monkeypatch, caplog):
    """A full disk is an operator problem: one actionable line and exit 2."""
    import logging as _lg
    import shutil as _sh
    from squidmip._cli import main

    monkeypatch.setattr(
        _sh, "disk_usage", lambda p: type("U", (), {"total": 100, "used": 99, "free": 1})()
    )
    root, _ = squid_dataset
    with caplog.at_level(_lg.ERROR, logger="squidmip"):
        with pytest.raises(SystemExit) as ei:
            main([str(root), "--output-folder", str(tmp_path)])
    assert ei.value.code == 2
    assert any("not enough disk space" in r.message for r in caplog.records)
