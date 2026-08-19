"""The register operator: the stitcher's solve without fusion, and the registered copy.

The copy's contract: image files hardlinked (copy fallback), sidecars REAL copies, the
coordinates.csv rewritten with the solved positions, and the source acquisition never written.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from squidxplorer._register import (
    ensure_registered_copy,
    register_region,
    registered_copy_root,
    write_registered_rows,
)

_CSV = (
    "region,fov,z_level,x (mm),y (mm)\n"
    "A1,0,0,0.000000,0.000000\n"
    "A1,1,0,0.000040,0.000000\n"
    "A1,2,0,0.000000,0.000040\n"
    "A1,3,0,0.000040,0.000040\n"
    "B2,0,0,9.000000,9.000000\n"
)


def _acq(tmp_path, name="acq"):
    root = tmp_path / name
    (root / "0").mkdir(parents=True)
    (root / "acquisition.yaml").write_text("objective:\n  name: 4x\n")
    (root / "coordinates.csv").write_text(_CSV)
    (root / "0" / "coordinates.csv").write_text(_CSV)
    for f in range(4):
        (root / "0" / f"A1_{f}_0_405.tiff").write_bytes(b"tiff-bytes-%d" % f)
    (root / "0" / "B2_0_0_405.tiff").write_bytes(b"tiff-bytes-b2")
    return root


def _snapshot(root: Path) -> dict:
    return {p.relative_to(root): (p.stat().st_mtime_ns, p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_copy_links_images_and_really_copies_sidecars(tmp_path):
    src = _acq(tmp_path)
    dst, linked, copied = ensure_registered_copy(src)
    assert dst == registered_copy_root(src) and dst.name == "stitched_acq"
    img_s, img_d = src / "0" / "A1_0_0_405.tiff", dst / "0" / "A1_0_0_405.tiff"
    assert img_d.stat().st_ino == img_s.stat().st_ino          # same bytes, second name
    csv_s, csv_d = src / "coordinates.csv", dst / "coordinates.csv"
    assert csv_d.stat().st_ino != csv_s.stat().st_ino          # a rewrite must not reach src
    assert linked == 5 and copied == 3
    assert ensure_registered_copy(src) == (dst, 0, 0)          # idempotent: reused, not rebuilt


def test_the_source_acquisition_is_never_written(tmp_path):
    src = _acq(tmp_path)
    before = _snapshot(src)
    dst, _, _ = ensure_registered_copy(src)
    write_registered_rows(dst, "A1", {f: (100.0 + f, 200.0 + f) for f in range(4)})
    assert _snapshot(src) == before


def test_registered_rows_land_in_every_csv_and_only_for_the_region(tmp_path):
    src = _acq(tmp_path)
    dst, _, _ = ensure_registered_copy(src)
    n = write_registered_rows(dst, "A1", {f: (1000.0 * (f + 1), 2000.0 * (f + 1))
                                          for f in range(4)})
    assert n == 8                                              # 4 rows in each of the two csvs
    for path in (dst / "coordinates.csv", dst / "0" / "coordinates.csv"):
        rows = list(csv.reader(path.open()))
        assert rows[1][3:5] == ["1.000000", "2.000000"]        # µm -> mm at 6 decimals
        assert rows[4][3:5] == ["4.000000", "8.000000"]
        assert rows[5] == ["B2", "0", "0", "9.000000", "9.000000"]   # untouched


def test_an_unknown_region_is_refused_not_silently_kept(tmp_path):
    src = _acq(tmp_path)
    dst, _, _ = ensure_registered_copy(src)
    with pytest.raises(ValueError, match="C3"):
        write_registered_rows(dst, "C3", {0: (1.0, 2.0)})


def test_the_row_order_schema_still_maps_fovs(tmp_path):
    # No fov column: row order per DISTINCT position is the id, repeated z rows move together.
    root = tmp_path / "copy"
    root.mkdir()
    (root / "coordinates.csv").write_text(
        "region,x (mm),y (mm)\n"
        "A1,0.000000,0.000000\n"
        "A1,0.000000,0.000000\n"       # same position again: one row per z
        "A1,0.000040,0.000000\n")
    n = write_registered_rows(root, "A1", {0: (5.0, 6.0), 1: (7.0, 8.0)})
    assert n == 3
    rows = list(csv.reader((root / "coordinates.csv").open()))
    assert rows[1][1:] == ["0.005000", "0.006000"]
    assert rows[2][1:] == ["0.005000", "0.006000"]
    assert rows[3][1:] == ["0.007000", "0.008000"]


def _probe_reader():
    pytest.importorskip("tilefusion")
    from tests.test_operator_declaration import _StitchProbeReader

    return _StitchProbeReader()


def test_register_solves_and_the_copy_carries_the_solution(tmp_path):
    reader = _probe_reader()
    src = _acq(tmp_path)
    reader.source_id = str(src)

    result = register_region(reader, "A1", [0, 1, 2, 3], copy=True)
    p = result.placement
    assert result.shape == (1, 2, 1, p.shape[0], p.shape[1])
    assert p.reg_channel == "405" and p.reg_t == 0
    # the probe's content errors are real: the solve must move somebody
    assert any(abs(dy) > 0.5 or abs(dx) > 0.5 for dy, dx in p.offsets_px)

    dst = registered_copy_root(src)
    rows = list(csv.reader((dst / "0" / "coordinates.csv").open()))
    meta = reader.metadata
    for f in range(4):
        x_mm, y_mm = float(rows[1 + f][3]), float(rows[1 + f][4])
        x0, y0 = meta["fov_positions_um"][("A1", f)]
        dy, dx = p.offsets_px[f]
        assert x_mm * 1000 == pytest.approx(x0 + dx, abs=1e-3)
        assert y_mm * 1000 == pytest.approx(y0 + dy, abs=1e-3)
    # the paste sits at the registered origin
    assert p.bbox_um[0] == pytest.approx(min(
        meta["fov_positions_um"][("A1", f)][0] + p.offsets_px[f][1] for f in range(4)))


def test_copy_without_an_on_disk_source_is_refused_by_name():
    reader = _probe_reader()
    with pytest.raises(ValueError, match="source"):
        register_region(reader, "A1", [0, 1, 2, 3], copy=True)


def test_the_panel_carries_the_copy_switch_outside_the_params(qapp):
    # copy cannot be a Param (it cannot change the preview's pixels), so it rides `accepts`;
    # the OME-Zarr save is hidden because the registered copy IS this operator's disk artifact.
    from squidxplorer._param_panel import RegisterPanel
    from tests.test_op_panels import _Host

    host = _Host()
    panel = RegisterPanel(host)
    assert sorted(panel.widgets) == ["registration_channel", "registration_t"]
    assert "copy" not in panel.kwargs()
    assert panel.save_btn is not None and not panel.save_btn.isVisibleTo(panel)
    panel.copy_check.setChecked(True)
    panel.run_all_btn.click()
    key, kw = host.calls[-1]
    assert key == "register" and kw["save"] is False and kw["regions"] is None
    assert kw["operator_kwargs"] == {"registration_channel": 0, "registration_t": 0,
                                     "copy": True}


def test_a_save_of_register_is_the_registered_copy_never_a_plate(tmp_path):
    """The generic save toggle: a copy-saving operator (declared, operator_saves_copy) routes
    through the engine with copy=True — write_plate's HCS layout is never demanded, so a
    'manual' region saves fine."""
    from squidxplorer._dispatch import run_operator_once
    from squidxplorer._engine import operator_saves_copy

    assert operator_saves_copy("register")
    assert not operator_saves_copy("stitch") and not operator_saves_copy("mip")

    reader = _probe_reader()
    src = _acq(tmp_path)
    reader.source_id = str(src)
    result = run_operator_once(reader, operator="register", save=True, owed=1, n_fovs=None)
    dst = registered_copy_root(src)
    assert result.outcome == "ok" and result.out_path == str(dst)
    assert dst.is_dir() and not list(tmp_path.rglob("plate.ome.zarr"))


def test_register_runs_through_the_engine_without_a_flatfield_estimate(tmp_path, caplog):
    # The callable declares its keywords explicitly, so the region loop's flatfield probe says
    # no and a register run never buys a plate-wide BaSiC estimate.
    reader = _probe_reader()
    import squidxplorer

    results = list(squidxplorer.run_plate(reader, operator="register"))
    assert len(results) == 1 and results[0][0] == "A1"
    assert not any("Flatfield" in r.message for r in caplog.records)
