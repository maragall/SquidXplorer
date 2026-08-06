"""CLI (IMA-186) tests: the declarative params model + the run() that drives the command layer.

Headless, no Qt. Uses the shared tiny `squid_dataset` fixture (a real 2-well acquisition on disk).

The thing most of these tests are actually about is the EXIT CODE. A batch surface whose failure
mode is `exit 0` is not a batch surface: `for d in */; do squidmip "$d"; done` cannot tell a
finished plate from one where every well was skipped, and that is exactly what this CLI did — the
`partial` verdict was computed in the command layer and then discarded by an unconditional
`_done(...)`. So `main()` is tested through its RETURN VALUE (the console script does
`sys.exit(main())`), not just `run()`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from squidmip._cli import (EXIT_INTERRUPTED, EXIT_NOTHING, EXIT_OK, EXIT_PARTIAL, EXIT_USAGE,
                           ProcessParameters, exit_code, main, run)


def _break_well(root, region: str) -> None:
    """Delete one plane of *region* so every read of that well raises (a per-well skip)."""
    for victim in sorted((Path(root) / "0").glob(f"{region}_*")):
        victim.unlink()
        return
    raise AssertionError(f"no planes to break for {region}")


# --- the model ----------------------------------------------------------------------------------

def test_input_folder_validator_rejects_missing(tmp_path):
    with pytest.raises(ValueError):
        ProcessParameters(input_folder=str(tmp_path / "nope"))


def test_projector_validator_accepts_region_operators(squid_dataset):
    # The CLI must not be NARROWER than the command layer it fronts: do_run_operator accepts
    # `projectors | region_operators` and write_plate dispatches on it, so `--projector stitch`
    # was a CLI-only refusal of a run the engine can do.
    root, _ = squid_dataset
    for name in ("mip", "stitch", "coordinate"):
        assert ProcessParameters(input_folder=str(root), projector=name).projector == name


def test_projector_validator_names_what_it_can_run(squid_dataset):
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="unknown operator 'nope'"):
        ProcessParameters(input_folder=str(root), projector="nope")


def test_param_is_refused_against_the_operators_own_declaration(squid_dataset):
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="does not take bogus"):
        ProcessParameters(input_folder=str(root), projector="spot", param=["bogus=1"])
    with pytest.raises(ValueError, match="'mip' declares no parameters"):
        ProcessParameters(input_folder=str(root), projector="mip", param=["min_area_px=80"])


def test_param_values_are_python_literals(squid_dataset):
    root, _ = squid_dataset
    p = ProcessParameters(input_folder=str(root), projector="spot",
                          param=["min_area_px=80", "split_touching=False", "sigma_px=1.5"])
    assert p.parameters() == {"min_area_px": 80, "split_touching": False, "sigma_px": 1.5}


def test_wells_parses_to_the_commands_regions_list(squid_dataset):
    root, _ = squid_dataset
    assert ProcessParameters(input_folder=str(root), wells="B2, B3").named_wells() == ["B2", "B3"]
    assert ProcessParameters(input_folder=str(root)).named_wells() is None


def test_help_lists_every_operators_declared_parameters():
    # Generated from the registry, not hand-written per operator: a plugin operator installed from
    # another package is documented here with no edit to _cli.py.
    described = ProcessParameters.model_fields["param"].description
    assert "cellpose(sigma_px=2.0" in described and "min_area_px=30" in described
    assert "stitch(" in described                  # region operators are listed too
    assert "mip()" in described                    # ...and an operator with no parameters says so


# --- the run ------------------------------------------------------------------------------------

def test_run_writes_navigable_plate(squid_dataset, tmp_path):
    root, _ = squid_dataset                       # tiny real acquisition (B2, B3)
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), tiff=False)
    manifest = run(params)

    plate = Path(manifest["plate"])
    assert plate.name == "plate.ome.zarr"
    assert plate.parent.name.endswith(".hcs")     # <acq-name>.hcs sibling
    assert manifest["n_wells"] == 2
    assert manifest["tiff"] is None                # CLI default: no uncompressed TIFF duplicate
    assert manifest["outcome"] == "ok"
    # the plate group + both wells' fields are on disk (level 0 present)
    assert (plate / "zarr.json").exists()
    for row, col in (("B", "2"), ("B", "3")):
        assert (plate / row / col / "0" / "zarr.json").exists()


def test_run_skips_unreadable_well_instead_of_aborting(squid_dataset, tmp_path):
    # Resilience (IMA-186): one corrupt/missing plane must NOT abort a whole-plate run — the bad
    # well is SKIPPED (logged + reported), the good wells still write.
    root, _ = squid_dataset                       # B2, B3
    _break_well(root, "B3")
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path))
    manifest = run(params)
    assert manifest["skipped"] == ["B3"]          # bad well skipped, not fatal
    assert manifest["n_fields_written"] == 1      # B2 still written
    assert manifest["outcome"] == "partial"       # ...and it SAYS so
    plate = Path(manifest["plate"])
    assert (plate / "B" / "2" / "0" / "zarr.json").exists()
    assert not (plate / "B" / "3" / "0" / "0").exists()   # B3 field never written


def test_run_defaults_output_next_to_acquisition(squid_dataset):
    root, _ = squid_dataset
    params = ProcessParameters(input_folder=str(root))     # no output_folder -> sibling of the acq
    assert params.output_folder is None
    manifest = run(params)
    assert Path(manifest["plate"]).parent.parent == Path(root).parent


def test_tiff_writes_the_second_copy(squid_dataset, tmp_path):
    root, _ = squid_dataset
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path),
                                     tiff=True))
    tiffs = sorted(p.name for p in Path(manifest["tiff"]).rglob("*.tif*"))
    assert tiffs, "--tiff wrote no TIFFs"
    assert any(t.startswith("B2_") for t in tiffs)


def test_limit_slices_the_plate(squid_dataset, tmp_path):
    root, _ = squid_dataset
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), limit=1))
    assert manifest["n_wells"] == 1
    assert manifest["n_fields_written"] == 1
    assert manifest["outcome"] == "ok"


def test_wells_selects_by_name_and_limit_truncates_that_list(squid_dataset, tmp_path):
    root, _ = squid_dataset
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path),
                                     wells="B3"))
    plate = Path(manifest["plate"])
    assert manifest["n_wells"] == 1
    assert (plate / "B" / "3" / "0" / "zarr.json").exists()
    assert not (plate / "B" / "2").exists()       # B2 was never a target

    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path / "b"),
                                     wells="B2,B3", limit=1))
    assert manifest["n_wells"] == 1               # --wells then --limit, in that order


def test_unknown_well_is_refused_by_name_before_anything_is_written(squid_dataset, tmp_path):
    root, _ = squid_dataset
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), wells="B2,ZZ99")
    with pytest.raises(SystemExit, match="ZZ99"):
        run(params)
    assert not list(tmp_path.glob("*.hcs"))        # no output tree was made


def test_operator_parameters_reach_the_operator(squid_dataset, tmp_path):
    # Not "the flag was accepted" — the PIXELS have to differ. `spot` labels objects, so an
    # absurd min_area_px must erase them. Accept-and-drop would give two identical stores.
    import zarr

    root, _ = squid_dataset
    loose = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path / "loose"),
                                  projector="spot", param=["min_area_px=1"]))
    strict = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path / "strict"),
                                   projector="spot", param=["min_area_px=100000"]))
    n_loose = int(zarr.open_array(str(Path(loose["plate"]) / "B" / "2" / "0" / "0"))[:].max())
    n_strict = int(zarr.open_array(str(Path(strict["plate"]) / "B" / "2" / "0" / "0"))[:].max())
    assert n_strict == 0 < n_loose


# --- the plate-format scope (IMA-219's shared helper, not a hand-rolled substring test) ----------

def test_a_96_well_plate_is_not_refused(squid_dataset, tmp_path):
    # The old guard was `any(s in fmt for s in ("384", "1536"))` — a comment that said 1536-only,
    # code that said 384-or-1536 and a message that said both. 96wp is a Squid format
    # (_plate_shape._STANDARD_FORMATS) and nothing about this pipeline cares how many wells the
    # plate has, so it was blocked for no reason.
    root, _ = squid_dataset
    acq = Path(root) / "acquisition.yaml"
    acq.write_text(acq.read_text().replace("1536 well plate", "96 well plate"))
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    assert manifest["n_fields_written"] == 2


def test_a_freeform_slide_acquisition_is_not_refused(real_dataset, tmp_path):
    # ...and the same guard refused a glass-slide / freeform acquisition (regions like "manual0",
    # format "glass slide") as 'unknown', which blocked the very datasets the stitch tests use.
    manifest = run(ProcessParameters(input_folder=str(real_dataset),
                                     output_folder=str(tmp_path), limit=1))
    assert manifest["n_fields_written"] == 1
    assert manifest["outcome"] == "ok"


def test_wellplate_format_override_is_honoured(squid_dataset, tmp_path, monkeypatch):
    # _plate_shape documents SQUIDMIP_WELLPLATE_FORMAT as the override "for headless / CLI runs".
    # The CLI ignored it entirely; now it resolves through the same helper, so it applies.
    root, _ = squid_dataset
    monkeypatch.setenv("SQUIDMIP_WELLPLATE_FORMAT", "96")
    assert run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))["n_wells"] == 2


# --- the overwrite guard ------------------------------------------------------------------------

def test_rerun_refuses_to_write_over_a_finished_plate(squid_dataset, tmp_path):
    root, _ = squid_dataset
    first = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    assert first["n_wells"] == 2
    with pytest.raises(SystemExit, match="--overwrite"):
        run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), limit=1))
    # ...and the good plate is untouched: it still DECLARES both wells.
    plate = Path(first["plate"])
    declared = json.loads((plate / "zarr.json").read_text())
    wells = [w["path"] for w in declared["attributes"]["ome"]["plate"]["wells"]]
    assert wells == ["B/2", "B/3"]


def test_overwrite_proceeds(squid_dataset, tmp_path):
    root, _ = squid_dataset
    run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path),
                                     limit=1, overwrite=True))
    assert manifest["n_wells"] == 1


# --- exit codes: the headline -------------------------------------------------------------------

def test_main_exits_zero_on_a_complete_run(squid_dataset, tmp_path):
    root, _ = squid_dataset
    assert main([str(root), "--output-folder", str(tmp_path)]) == EXIT_OK


def test_main_exits_nonzero_when_the_plate_is_empty(squid_dataset, tmp_path):
    # THE defect: every target skipped, "0/2 wells written", and the process exited 0, so no
    # batch loop could detect it.
    root, _ = squid_dataset
    for region in ("B2", "B3"):
        _break_well(root, region)
    assert main([str(root), "--output-folder", str(tmp_path)]) == EXIT_NOTHING


def test_main_exits_nonzero_when_the_plate_is_partial(squid_dataset, tmp_path):
    root, _ = squid_dataset
    _break_well(root, "B3")
    code = main([str(root), "--output-folder", str(tmp_path)])
    assert code == EXIT_PARTIAL
    assert code != EXIT_OK                         # a batch loop's `||` fires


def test_main_exits_two_on_a_bad_command_line(squid_dataset, tmp_path, capsys):
    root, _ = squid_dataset
    assert main([str(root), "--projector", "nope"]) == EXIT_USAGE
    assert "unknown operator" in capsys.readouterr().err
    # 2 is USAGE and must stay distinct from the data outcomes, or a script cannot tell "you
    # typed it wrong" from "the acquisition was bad".
    assert EXIT_USAGE not in (EXIT_OK, EXIT_NOTHING, EXIT_PARTIAL, EXIT_INTERRUPTED)


def test_exit_code_maps_every_outcome():
    assert exit_code({"outcome": "ok", "n_fields_written": 2}) == EXIT_OK
    assert exit_code({"outcome": "partial", "n_fields_written": 0}) == EXIT_NOTHING
    assert exit_code({"outcome": "partial", "n_fields_written": 1}) == EXIT_PARTIAL
    assert exit_code({"outcome": "stopped", "n_fields_written": 1}) == EXIT_INTERRUPTED


# --- progress + cancel --------------------------------------------------------------------------

def test_run_reports_every_well_as_it_lands(squid_dataset, tmp_path, caplog):
    import logging

    root, _ = squid_dataset
    with caplog.at_level(logging.INFO, logger="squid.xplorer"):
        run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    progress = [r.getMessage() for r in caplog.records if "wrote" in r.getMessage()]
    assert len(progress) == 2                      # one line per well, not silence for hours
    assert any("B2" in line for line in progress)


def test_stop_cuts_the_run_and_the_store_says_it_is_incomplete(squid_dataset, tmp_path):
    from squidmip._output import is_incomplete

    root, _ = squid_dataset
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)),
                   stop=lambda: True)              # "stop" before the first well
    assert manifest["outcome"] == "stopped"
    assert manifest["n_fields_written"] == 0
    assert exit_code(manifest) == EXIT_INTERRUPTED
    assert is_incomplete(Path(manifest["plate"]))  # never mistakable for a finished plate
