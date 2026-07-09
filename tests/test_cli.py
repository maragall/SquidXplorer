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
