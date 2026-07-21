"""IMA-212 odon bridge tests — samplesheet, binary discovery, headless probe, launch.

The samplesheet is built by WALKING a written plate, so most tests drive the real
``write_from_stream`` (tiny 8x8 frames) and then assert over its output — that way the
NGFF conformance assertions cover the artifact Odon will actually open, not a fabricated
one. Edge cases that a correct writer can't produce (a half-written field) are built by
hand.

No odon binary is required: discovery and launch are exercised with a fake executable and
a monkeypatched Popen. The one test that runs the real binary is skipped when it is absent.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from squidmip._odon import (
    SAMPLESHEET_COLUMNS,
    check_odon,
    find_odon,
    iter_fields,
    launch_odon,
    write_samplesheet,
)
from squidmip._output import write_from_stream

CH = [
    {"name": "Fluorescence_638_nm_-_Penta", "display_name": "638", "display_color": "#FF0000"},
    {"name": "Fluorescence_405_nm_-_Penta", "display_name": "405", "display_color": "#20ADF8"},
]
# B10 present so natural column sort is exercised (2, 3, 10 — not 10, 2, 3).
REGIONS = ["B2", "B3", "B10"]


def _meta(fovs=(0,)):
    return {
        "regions": REGIONS,
        "fovs_per_region": {r: list(fovs) for r in REGIONS},
        "channels": CH,
        "pixel_size_um": 0.325,
    }


def _image(y=8, x=8, dtype=np.uint16):
    return np.arange(2 * 1 * y * x, dtype=dtype).reshape(1, 2, 1, y, x)


@pytest.fixture
def hcs(tmp_path):
    """A real, complete .hcs output: 3 wells x 1 fov, written by the production writer."""
    out = tmp_path / "acq.hcs"
    stream = ((r, 0, _image()) for r in REGIONS)
    write_from_stream(_meta(), stream, out, n_fovs=1)
    return out


def _read_sheet(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fake_binary(tmp_path, name="odon", rc=0, stdout=""):
    """An executable stub standing in for the odon binary."""
    suffix = ".bat" if sys.platform == "win32" else ""
    path = tmp_path / f"{name}{suffix}"
    path.write_text(f"#!/bin/sh\n{'echo ' + repr(stdout) if stdout else ':'}\nexit {rc}\n")
    path.chmod(0o755)
    return path


# --- write_samplesheet ------------------------------------------------------------------

def test_samplesheet_has_one_row_per_field(hcs):
    rows = _read_sheet(write_samplesheet(hcs))
    assert len(rows) == 3
    assert [r["well"] for r in rows] == ["B2", "B3", "B10"]   # natural order, not B10 first
    assert [r["id"] for r in rows] == ["B2_0", "B3_0", "B10_0"]


def test_samplesheet_header_and_positional_columns(hcs):
    """Odon requires a header whose first TWO columns are positionally id, path."""
    csv_path = write_samplesheet(hcs)
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[:2] == ["id", "path"]
    assert header == ",".join(SAMPLESHEET_COLUMNS)


def test_samplesheet_paths_are_relative_and_resolve(hcs):
    """Odon resolves relative paths against the CSV's own directory."""
    csv_path = write_samplesheet(hcs)
    for row in _read_sheet(csv_path):
        assert not Path(row["path"]).is_absolute()
        assert row["path"].startswith("plate.ome.zarr/")
        assert (csv_path.parent / row["path"] / "zarr.json").is_file()


def test_samplesheet_survives_the_output_being_moved(hcs, tmp_path):
    """Relative paths are the whole point: a copied .hcs must still open."""
    write_samplesheet(hcs)
    moved = tmp_path / "elsewhere" / "acq.hcs"
    moved.parent.mkdir(parents=True)
    hcs.rename(moved)
    for row in _read_sheet(moved / "odon_samplesheet.csv"):
        assert (moved / row["path"] / "zarr.json").is_file()


def test_partial_write_is_excluded(hcs):
    """_write_field writes arrays FIRST and the group LAST, so a killed run leaves a field
    directory with no zarr.json. That is exactly Odon's hard error — it must not be listed."""
    victim = hcs / "plate.ome.zarr" / "B" / "3" / "0"
    (victim / "zarr.json").unlink()
    assert victim.is_dir()                       # the directory still exists...
    rows = _read_sheet(write_samplesheet(hcs))
    assert [r["well"] for r in rows] == ["B2", "B10"]   # ...but contributes no row


def test_glob_matches_fields_only_not_plate_row_well_or_arrays(hcs):
    """Plate/row/well groups and the pyramid arrays all carry their own zarr.json."""
    plate = hcs / "plate.ome.zarr"
    assert (plate / "zarr.json").is_file()                    # depth 0 — plate group
    assert (plate / "B" / "zarr.json").is_file()              # depth 1 — row group
    assert (plate / "B" / "2" / "zarr.json").is_file()        # depth 2 — well group
    assert (plate / "B" / "2" / "0" / "0" / "zarr.json").is_file()   # depth 4 — array
    fields = [d for *_, d in iter_fields(hcs)]
    assert fields == [plate / "B" / "2" / "0", plate / "B" / "3" / "0", plate / "B" / "10" / "0"]


def test_non_contiguous_fov_ids_stay_faithful(tmp_path):
    out = tmp_path / "acq.hcs"
    meta = _meta(fovs=(0, 7))
    stream = ((r, f, _image()) for r in REGIONS for f in (0, 7))
    write_from_stream(meta, stream, out, n_fovs=2)
    rows = _read_sheet(write_samplesheet(out))
    assert [r["fov"] for r in rows if r["well"] == "B2"] == ["0", "7"]
    assert {r["id"] for r in rows} >= {"B2_0", "B2_7", "B10_7"}


def test_zero_usable_rows_raises_rather_than_empty_csv(tmp_path):
    plate = tmp_path / "acq.hcs" / "plate.ome.zarr"
    plate.mkdir(parents=True)
    (plate / "zarr.json").write_text("{}")
    with pytest.raises(ValueError, match="no complete field groups"):
        write_samplesheet(tmp_path / "acq.hcs")


def test_missing_plate_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="plate.ome.zarr"):
        write_samplesheet(tmp_path)


def test_accepts_the_plate_dir_itself(hcs):
    assert _read_sheet(write_samplesheet(hcs / "plate.ome.zarr"))


def test_works_on_a_prior_output_with_no_reader(hcs):
    """The main real use: point --odon at an .hcs written days ago. No metadata, no reader."""
    rows = _read_sheet(write_samplesheet(hcs))
    assert len(rows) == 3


# --- NGFF conformance, asserted over the SAMPLESHEET'S OWN ROWS -------------------------
# Deliberately driven from the CSV, not from the plate at large: a test that walked the
# plate directly would pass even with _odon.py absent, and so would prove nothing.

def test_every_samplesheet_row_is_a_conformant_odon_group(hcs):
    csv_path = write_samplesheet(hcs)
    rows = _read_sheet(csv_path)
    assert rows
    for row in rows:
        group = csv_path.parent / row["path"]
        doc = json.loads((group / "zarr.json").read_text())
        ome = doc["attributes"]["ome"]                 # Odon unwraps the 0.5 {"ome": ...}
        multiscales = ome["multiscales"]
        assert multiscales, "Odon hard-errors on a group with no multiscales[0]"
        ms = multiscales[0]
        axis_names = [a["name"] for a in ms["axes"]]
        assert "y" in axis_names and "x" in axis_names, "Odon requires y and x axes by name"
        for dataset in ms["datasets"]:
            scale = dataset["coordinateTransformations"][0]["scale"]
            assert len(scale) == len(axis_names), "scale must be exactly ndim long"
            arr = json.loads((group / dataset["path"] / "zarr.json").read_text())
            assert len(arr["shape"]) == len(axis_names), "shape.len() must equal axes.len()"
            assert arr["data_type"] in ("uint8", "uint16"), "Odon supports uint8/uint16 only"


def test_omero_color_is_still_written_correctly(hcs):
    """REGRESSION GUARD. Odon ignores omero.color, but ndviewer_light depends on it —
    the bridge must never tempt anyone into dropping or rewriting it."""
    group = hcs / "plate.ome.zarr" / "B" / "2" / "0"
    omero = json.loads((group / "zarr.json").read_text())["attributes"]["ome"]["omero"]
    assert [c["color"] for c in omero["channels"]] == ["FF0000", "20ADF8"]


# --- find_odon --------------------------------------------------------------------------

def test_odon_bin_env_override_is_honoured(tmp_path, monkeypatch):
    binary = _fake_binary(tmp_path)
    monkeypatch.setenv("ODON_BIN", str(binary))
    assert find_odon() == binary


def test_odon_bin_set_but_invalid_raises(tmp_path, monkeypatch):
    """An override that is silently ignored is worse than no override."""
    monkeypatch.setenv("ODON_BIN", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="ODON_BIN"):
        find_odon()


def test_falls_back_to_path(tmp_path, monkeypatch):
    binary = _fake_binary(tmp_path)
    monkeypatch.delenv("ODON_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: str(binary) if name == "odon" else None)
    assert find_odon() == binary


def test_macos_app_bundle_fallback(tmp_path, monkeypatch):
    """The .dmg installs an .app and does NOT put odon on PATH — this is the common case."""
    bundle = tmp_path / "Applications" / "odon.app" / "Contents" / "MacOS"
    bundle.mkdir(parents=True)
    binary = _fake_binary(bundle)
    monkeypatch.delenv("ODON_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("squidmip._odon._platform_default", lambda: binary)
    assert find_odon() == binary


def test_linux_arm64_says_no_build_exists(monkeypatch):
    monkeypatch.delenv("ODON_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    with pytest.raises(FileNotFoundError, match="no Linux arm64 build"):
        find_odon()


def test_not_found_names_the_release_url(tmp_path, monkeypatch):
    monkeypatch.delenv("ODON_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("squidmip._odon._platform_default", lambda: tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="github.com/alexcoulton/odon/releases"):
        find_odon()


# --- check_odon -------------------------------------------------------------------------

def test_check_odon_parses_the_ok_line(tmp_path, hcs):
    binary = _fake_binary(tmp_path, stdout="OK: loaded tile level 0 path '0' shape [1,2,1,8,8]")
    assert check_odon(hcs / "plate.ome.zarr" / "B" / "2" / "0", odon_bin=binary) is True


def test_check_odon_reports_failure(tmp_path, hcs):
    binary = _fake_binary(tmp_path, rc=1, stdout="error: no multiscales found")
    assert check_odon(hcs / "plate.ome.zarr" / "B" / "2" / "0", odon_bin=binary) is False


@pytest.mark.skipif(not (os.environ.get("ODON_BIN") or __import__("shutil").which("odon")),
                    reason="odon binary not installed (optional third-party GUI tool)")
def test_real_odon_check_opens_a_written_field(hcs):
    """The one automated test that exercises the REAL binary. `odon --check` is its only
    headless path; it takes a single local dataset and cannot accept a samplesheet, which
    is why the samplesheet half of the oracle stays manual."""
    assert check_odon(hcs / "plate.ome.zarr" / "B" / "2" / "0") is True


# --- launch_odon ------------------------------------------------------------------------

def test_launch_argv_shape(tmp_path, monkeypatch):
    binary = _fake_binary(tmp_path)
    seen = {}

    class _Proc:
        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: seen.update(argv=argv, kw=kw) or _Proc())
    launch_odon(tmp_path / "sheet.csv", odon_bin=binary, crash_check_delay=0)
    assert seen["argv"] == [str(binary), "--mosaic-samplesheet", str(tmp_path / "sheet.csv")]
    if os.name == "posix":
        assert seen["kw"]["start_new_session"] is True    # detached: outlives the CLI


def test_launch_passes_mosaic_cols(tmp_path, monkeypatch):
    binary = _fake_binary(tmp_path)
    seen = {}

    class _Proc:
        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: seen.update(argv=argv) or _Proc())
    launch_odon(tmp_path / "s.csv", mosaic_cols=4, odon_bin=binary, crash_check_delay=0)
    assert seen["argv"][-2:] == ["--mosaic-cols", "4"]


def test_immediate_nonzero_exit_warns_instead_of_dying_silently(tmp_path, monkeypatch, caplog):
    """Detached launch means a crash (no GPU/display) is otherwise invisible."""
    binary = _fake_binary(tmp_path)

    class _Crashed:
        def poll(self):
            return 1

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: _Crashed())
    with caplog.at_level("WARNING"):
        launch_odon(tmp_path / "s.csv", odon_bin=binary, crash_check_delay=0)
    assert "exited immediately" in caplog.text


# --- CLI wiring -------------------------------------------------------------------------

def test_cli_default_writes_no_samplesheet(squid_dataset, tmp_path):
    """No behaviour change for existing users."""
    from squidmip._cli import ProcessParameters, run

    root, _ = squid_dataset
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    assert "odon_samplesheet" not in manifest
    assert not list(tmp_path.rglob("odon_samplesheet.csv"))


def test_cli_odon_flag_writes_sheet_and_launches(squid_dataset, tmp_path, monkeypatch):
    from squidmip import _odon
    from squidmip._cli import ProcessParameters, run

    launched = {}
    monkeypatch.setattr(_odon, "launch_odon", lambda sheet, **kw: launched.update(sheet=sheet))
    root, _ = squid_dataset
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), odon=True))

    sheet = Path(manifest["odon_samplesheet"])
    assert sheet.is_file() and sheet.name == "odon_samplesheet.csv"
    assert sheet.parent == Path(manifest["plate"]).parent
    assert launched["sheet"] == sheet
    assert {r["well"] for r in _read_sheet(sheet)} == {"B2", "B3"}


def test_cli_odon_without_binary_exits_clearly_but_still_wrote_the_plate(
    squid_dataset, tmp_path, monkeypatch
):
    """CRITICAL: a missing optional viewer must not cost the user their plate."""
    from squidmip import _odon
    from squidmip._cli import ProcessParameters, run

    def _absent(*a, **kw):
        raise FileNotFoundError("odon not found. Install it from https://github.com/alexcoulton/odon/releases")

    monkeypatch.setattr(_odon, "launch_odon", _absent)
    root, _ = squid_dataset
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), odon=True)
    with pytest.raises(SystemExit) as excinfo:
        run(params)

    message = str(excinfo.value)
    assert "releases" in message and "plate itself is written" in message
    plate = next(tmp_path.rglob("plate.ome.zarr"))
    assert (plate / "B" / "2" / "0" / "zarr.json").is_file()      # the plate survived
    assert (plate.parent / "odon_samplesheet.csv").is_file()
