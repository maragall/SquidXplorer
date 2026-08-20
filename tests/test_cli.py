"""CLI tests: the declarative params model + the run() that drives the command layer.

Headless, no Qt. Most of these tests are about the EXIT CODE, since a batch surface whose failure
mode is `exit 0` cannot tell a finished plate from one where every well was skipped. `main()` is
tested through its return value, not just `run()`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from squidxplorer._cli import (EXIT_INTERRUPTED, EXIT_NOTHING, EXIT_OK, EXIT_PARTIAL, EXIT_USAGE,
                           ProcessParameters, exit_code, main, run)


def _break_well(root, region: str) -> None:
    """Delete one plane of *region* so every read of that well raises (a per-well skip)."""
    for victim in sorted((Path(root) / "0").glob(f"{region}_*")):
        victim.unlink()
        return
    raise AssertionError(f"no planes to break for {region}")


def _break_every_fov(root, region: str) -> None:
    """Delete one plane of every FOV of *region*, so the whole well produces nothing."""
    planes = sorted((Path(root) / "0").glob(f"{region}_*"))
    assert planes, f"no planes to break for {region}"
    for fov in sorted({p.name.split("_")[1] for p in planes}):
        next(iter(sorted((Path(root) / "0").glob(f"{region}_{fov}_*")))).unlink()


def test_input_folder_validator_rejects_missing(tmp_path):
    with pytest.raises(ValueError):
        ProcessParameters(input_folder=str(tmp_path / "nope"))


def test_operator_validator_accepts_region_operators(squid_dataset):
    # the CLI must not be narrower than the command layer: region operators are valid operators too.
    root, _ = squid_dataset
    for name in ("mip", "stitch", "coordinate"):
        assert ProcessParameters(input_folder=str(root), operator=name).operator == name


def test_operator_validator_names_what_it_can_run(squid_dataset):
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="unknown operator 'nope'"):
        ProcessParameters(input_folder=str(root), operator="nope")


def test_param_is_refused_against_the_operators_own_declaration(squid_dataset):
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="does not take bogus"):
        ProcessParameters(input_folder=str(root), operator="spot", param=["bogus=1"])
    with pytest.raises(ValueError, match="'mip' does not take min_area_px"):
        ProcessParameters(input_folder=str(root), operator="mip", param=["min_area_px=80"])


def test_param_values_are_python_literals(squid_dataset):
    root, _ = squid_dataset
    p = ProcessParameters(input_folder=str(root), operator="spot",
                          param=["min_area_px=80", "split_touching=False", "sigma_px=1.5"])
    assert p.parameters() == {"min_area_px": 80, "split_touching": False, "sigma_px": 1.5}


def test_wells_parses_to_the_commands_regions_list(squid_dataset):
    root, _ = squid_dataset
    assert ProcessParameters(input_folder=str(root), wells="B2, B3").named_wells() == ["B2", "B3"]
    assert ProcessParameters(input_folder=str(root)).named_wells() is None


def test_help_lists_every_operators_declared_parameters():
    """Generated from the registry, and checked against the registry itself, not a hand-copied
    list of today's declarations — a copy can't notice the declaration it copied is wrong."""
    from squidxplorer._engine import operator_params
    from squidxplorer._operations import runnable_operators

    described = ProcessParameters.model_fields["param"].description
    for name in runnable_operators():
        params = operator_params(name)
        expected = f"{name}({', '.join(f'{p.name}={p.default!r}' for p in params)})"
        if params:
            assert expected in described, (
                f"--param help must document {name} exactly as it is declared; expected "
                f"{expected!r} in the help text"
            )
    assert "stitch(" in described                  # region operators are listed too
    assert "mip()" in described                    # ...and an operator with no parameters says so


def test_a_fused_save_reports_off_its_own_manifest(squid_dataset, tmp_path):
    """The installer CI's exact path: a CLI stitch save lands the fused Squid OME-TIFF format,
    whose manifest carries no 'plate'/'levels'/'tiff' keys. The report must read the keys the
    manifest declares — this crashed with KeyError: 'plate' and turned every installer build
    red on 2026-08-19."""
    pytest.importorskip("tilefusion")
    root, _ = squid_dataset
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path),
                                     operator="stitch", param=["register=False"]))
    assert manifest["format"] == "fused-ome-tiff"
    assert manifest["outcome"] == "ok"
    assert manifest["n_fields_written"] == manifest["n_fields"] == 2   # one mosaic per region
    assert exit_code(manifest) == EXIT_OK


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
    assert (plate / "zarr.json").exists()
    for row, col in (("B", "2"), ("B", "3")):
        assert (plate / row / col / "0" / "zarr.json").exists()


def test_run_skips_unreadable_well_instead_of_aborting(squid_dataset, tmp_path):
    # one corrupt/missing plane must not abort a whole-plate run; the bad well is skipped, not fatal.
    root, _ = squid_dataset                       # B2, B3
    _break_well(root, "B3")
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path))
    manifest = run(params)
    assert manifest["skipped"] == ["B3"]
    assert manifest["n_fields_written"] == 1      # B2 still written
    assert manifest["outcome"] == "partial"
    plate = Path(manifest["plate"])
    assert (plate / "B" / "2" / "0" / "zarr.json").exists()
    assert not (plate / "B" / "3" / "0" / "0").exists()


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
    # the PIXELS have to differ, not just "the flag was accepted".
    import zarr

    root, _ = squid_dataset
    loose = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path / "loose"),
                                  operator="spot", param=["min_area_px=1"]))
    strict = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path / "strict"),
                                   operator="spot", param=["min_area_px=100000"]))
    n_loose = int(zarr.open_array(str(Path(loose["plate"]) / "B" / "2" / "0" / "0"))[:].max())
    n_strict = int(zarr.open_array(str(Path(strict["plate"]) / "B" / "2" / "0" / "0"))[:].max())
    assert n_strict == 0 < n_loose


def test_a_96_well_plate_is_not_refused(squid_dataset, tmp_path):
    # 96-well is a valid Squid format; nothing about this pipeline cares how many wells a plate has.
    root, _ = squid_dataset
    acq = Path(root) / "acquisition.yaml"
    acq.write_text(acq.read_text().replace("1536 well plate", "96 well plate"))
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    assert manifest["n_fields_written"] == 2


def test_a_freeform_slide_acquisition_is_not_refused(real_dataset, tmp_path):
    # a glass-slide / freeform acquisition ("manual0", format "glass slide") must not be refused.
    manifest = run(ProcessParameters(input_folder=str(real_dataset),
                                     output_folder=str(tmp_path), limit=1))
    assert manifest["n_fields_written"] == 1
    assert manifest["outcome"] == "ok"


def test_wellplate_format_override_is_honoured(squid_dataset, tmp_path, monkeypatch):
    # SQUIDXPLORER_WELLPLATE_FORMAT must apply to CLI runs too, resolved through the same helper.
    root, _ = squid_dataset
    monkeypatch.setenv("SQUIDXPLORER_WELLPLATE_FORMAT", "96")
    assert run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))["n_wells"] == 2


def test_rerun_refuses_to_write_over_a_finished_plate(squid_dataset, tmp_path):
    root, _ = squid_dataset
    first = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    assert first["n_wells"] == 2
    with pytest.raises(SystemExit, match="--overwrite"):
        run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), limit=1))
    # the good plate is untouched: it still declares both wells.
    plate = Path(first["plate"])
    declared = json.loads((plate / "zarr.json").read_text())
    wells = [w["path"] for w in declared["attributes"]["ome"]["plate"]["wells"]]
    assert wells == ["B/2", "B/3"]


def test_an_incomplete_plate_is_refused_as_INPUT(squid_dataset, tmp_path):
    """A written plate is a legal input; the opener must refuse one the store itself marks
    incomplete, rather than silently projecting whatever wells happened to land."""
    from squidxplorer._output import _mark_incomplete

    root, _ = squid_dataset
    done = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    hcs = Path(done["plate"]).parent
    assert run(ProcessParameters(input_folder=str(hcs), output_folder=str(tmp_path / "again"))), (
        "a FINISHED plate must still be processable as input")

    _mark_incomplete(Path(done["plate"]), {"wells": ["B2", "B3"], "fields": 2, "fields_written": 1,
                                           "stopped": True})
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        run(ProcessParameters(input_folder=str(hcs), output_folder=str(tmp_path / "third")))


def test_overwrite_proceeds(squid_dataset, tmp_path):
    root, _ = squid_dataset
    run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path),
                                     limit=1, overwrite=True))
    assert manifest["n_wells"] == 1


def test_main_exits_zero_on_a_complete_run(squid_dataset, tmp_path):
    root, _ = squid_dataset
    assert main([str(root), "--output-folder", str(tmp_path)]) == EXIT_OK


def test_main_exits_nonzero_when_the_plate_is_empty(squid_dataset, tmp_path):
    # every target skipped must not exit 0, or a batch loop can't detect it.
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
    assert main([str(root), "--operator", "nope"]) == EXIT_USAGE
    assert "unknown operator" in capsys.readouterr().err
    # USAGE must stay distinct from the data outcomes, or a script can't tell "typo" from "bad data".
    assert EXIT_USAGE not in (EXIT_OK, EXIT_NOTHING, EXIT_PARTIAL, EXIT_INTERRUPTED)


def test_exit_code_maps_every_outcome():
    assert exit_code({"outcome": "ok", "n_fields_written": 2}) == EXIT_OK
    assert exit_code({"outcome": "partial", "n_fields_written": 0}) == EXIT_NOTHING
    assert exit_code({"outcome": "partial", "n_fields_written": 1}) == EXIT_PARTIAL
    assert exit_code({"outcome": "stopped", "n_fields_written": 1}) == EXIT_INTERRUPTED


def test_run_reports_every_well_as_it_lands(squid_dataset, tmp_path, caplog):
    import logging

    root, _ = squid_dataset
    with caplog.at_level(logging.INFO, logger="squid.xplorer"):
        run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    progress = [r.getMessage() for r in caplog.records if "wrote" in r.getMessage()]
    assert len(progress) == 2                      # one line per well, not silence for hours
    assert any("B2" in line for line in progress)


def _summary_line(caplog) -> str:
    """The one line ``run()`` ends on — the ``done:`` / ``PARTIAL`` / ``STOPPED`` verdict."""
    # both formats' lines carry it: "... pyramid level(s)" and "... acquisition format"
    lines = [r.getMessage() for r in caplog.records if "fields written across" in r.getMessage()]
    assert len(lines) == 1, f"expected exactly one summary line, got {lines}"
    return lines[0]


def test_the_summary_line_counts_fields_and_wells_each_against_its_own_total(squid_dataset,
                                                                            tmp_path, caplog):
    """fields and wells are two different units; the summary must report each against its own
    total rather than dividing one by the other."""
    import logging

    root, _ = squid_dataset
    with caplog.at_level(logging.INFO, logger="squid.xplorer"):
        manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path),
                                         n_fovs=0))
    assert manifest["outcome"] == "ok"
    assert (manifest["n_fields_written"], manifest["n_fields"]) == (4, 4)
    assert (manifest["n_wells_written"], manifest["n_wells"]) == (2, 2)
    assert "4/4 fields written across 2/2 wells" in _summary_line(caplog), (
        f"each count must be against its OWN total; got {_summary_line(caplog)!r} "
        "(the old line said '4/2 wells written' — 4 FIELDS over 2 WELLS)")


def test_the_summary_line_counts_only_the_wells_that_actually_landed(squid_dataset, tmp_path,
                                                                     caplog):
    """A run that lost a well must not report a numerator larger than its denominator."""
    import logging

    root, _ = squid_dataset
    _break_every_fov(root, "B3")
    with caplog.at_level(logging.INFO, logger="squid.xplorer"):
        manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path),
                                         n_fovs=0))
    assert manifest["outcome"] != "ok"
    assert manifest["n_wells_written"] == 1, "only B2 produced a field"
    assert "2/4 fields written across 1/2 wells" in _summary_line(caplog), (
        f"a lost well must show in BOTH counts; got {_summary_line(caplog)!r} "
        "(the old line said '2/2 wells written' over a plate missing half its fields)")


def test_stop_cuts_the_run_and_the_store_says_it_is_incomplete(squid_dataset, tmp_path):
    from squidxplorer._output import is_incomplete

    root, _ = squid_dataset
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)),
                   stop=lambda: True)              # "stop" before the first well
    assert manifest["outcome"] == "stopped"
    assert manifest["n_fields_written"] == 0
    assert exit_code(manifest) == EXIT_INTERRUPTED
    assert is_incomplete(Path(manifest["plate"]))  # never mistakable for a finished plate
