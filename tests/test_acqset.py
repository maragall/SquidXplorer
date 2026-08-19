"""Folder-of-acquisitions: discovery, set ingest + cycling, and the one-operator bulk save loop."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")   # is_acquisition reaches resolve_plate_root, whose module needs Qt
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded: Qt binding conflict", allow_module_level=True)

import tifffile  # noqa: E402

from squidxplorer import _acqset  # noqa: E402
from squidxplorer._dispatch import DispatchResult  # noqa: E402

from .conftest import _ACQ_YAML, _PARAMS, _YAML, _coordinates_csv, _write_timepoint  # noqa: E402


def _min_acq(root: Path) -> Path:
    """The smallest folder open_reader accepts: one individual-TIFF plane."""
    root.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(root / "A1_0_0_BF.tiff", np.zeros((4, 4), dtype=np.uint16))
    return root


def _full_acq(root: Path) -> Path:
    """A real tiny acquisition (the squid_dataset recipe), openable with full metadata."""
    arrays: dict = {}
    _write_timepoint(root / "0", arrays, tag=0)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    (root / "coordinates.csv").write_text(_coordinates_csv())
    return root


# -- discovery ------------------------------------------------------------------------------


def test_a_single_acquisition_discovers_as_itself(tmp_path):
    acq = _min_acq(tmp_path / "acq")
    assert _acqset.discover_acquisitions(acq) == [acq]


def test_a_folder_of_acquisitions_discovers_name_sorted(tmp_path):
    parent = tmp_path / "night_run"
    b = _min_acq(parent / "b_acq")
    a = _min_acq(parent / "a_acq")
    (parent / "notes").mkdir()                                   # junk dir: not a member
    (parent / "README.txt").write_text("x")                      # a file: never a member
    assert _acqset.discover_acquisitions(parent) == [a, b]


def test_a_folder_that_is_neither_is_refused_by_name(tmp_path):
    empty = tmp_path / "nothing_here"
    (empty / "notes").mkdir(parents=True)
    with pytest.raises(_acqset.AcqSetError) as e:
        _acqset.discover_acquisitions(empty)
    assert "nothing_here" in str(e.value) and "at least 2" in str(e.value)


def test_one_child_acquisition_is_not_a_set(tmp_path):
    parent = tmp_path / "wrapper"
    _min_acq(parent / "only_acq")
    with pytest.raises(_acqset.AcqSetError):
        _acqset.discover_acquisitions(parent)


def test_nesting_beyond_one_level_never_counts(tmp_path):
    parent = tmp_path / "deep"
    _min_acq(parent / "sub" / "acq1")                            # grandchild: one level too far
    _min_acq(parent / "sub" / "acq2")
    with pytest.raises(_acqset.AcqSetError):
        _acqset.discover_acquisitions(parent)


def test_a_written_plate_is_not_a_set_member(tmp_path):
    parent = tmp_path / "runs"
    a = _min_acq(parent / "a_acq")
    b = _min_acq(parent / "b_acq")
    (parent / "out.hcs" / "plate.ome.zarr").mkdir(parents=True)  # an OUTPUT, not a member
    assert _acqset.discover_acquisitions(parent) == [a, b]


# -- the bulk save loop ---------------------------------------------------------------------


def _ok_result() -> DispatchResult:
    return DispatchResult(outcome="ok", detail="", landed=4, stopped=False,
                          skipped_regions=frozenset(), manifest=None)


def test_the_bulk_loop_runs_the_operator_once_per_acquisition_with_the_same_parameters(
        tmp_path, monkeypatch):
    parent = tmp_path / "set"
    paths = [_full_acq(parent / n) for n in ("acq_a", "acq_b", "acq_c")]
    out = tmp_path / "out"
    out.mkdir()
    calls: list = []

    def fake_once(reader, **kw):
        calls.append((reader.source_id, kw))
        return _ok_result()

    monkeypatch.setattr("squidxplorer._dispatch.run_operator_once", fake_once)
    lines: list = []
    params = {"min_area_px": 12}
    summary = _acqset.run_over_set(paths, operator="mip", out_parent=str(out),
                                   parameters=params, log=lines.append)
    assert [c[0] for c in calls] == [str(p) for p in paths]
    assert all(c[1]["parameters"] == params for c in calls)      # the SAME parameters, each run
    assert all(c[1]["save"] is True and c[1]["regions"] is None for c in calls)
    assert summary == {"ok": 3, "partial": 0, "failed": 0, "total": 3, "stopped": False}
    assert any("acquisition 2 of 3: mip on acq_b" in ln and "ok" in ln for ln in lines)
    assert any("3 ok, 0 partial, 0 failed of 3" in ln for ln in lines)


def test_the_bulk_loop_continues_past_a_failing_acquisition(tmp_path, monkeypatch):
    parent = tmp_path / "set"
    paths = [_full_acq(parent / n) for n in ("acq_a", "acq_b", "acq_c")]
    n_calls = {"n": 0}

    def fake_once(reader, **kw):
        n_calls["n"] += 1
        if n_calls["n"] == 2:
            raise RuntimeError("this member is corrupt")
        return _ok_result()

    monkeypatch.setattr("squidxplorer._dispatch.run_operator_once", fake_once)
    lines: list = []
    summary = _acqset.run_over_set(paths, operator="mip", out_parent=str(tmp_path),
                                   log=lines.append)
    assert n_calls["n"] == 3                                     # the loop reached member 3
    assert summary["ok"] == 2 and summary["failed"] == 1 and not summary["stopped"]
    bad = [ln for ln in lines if "acq_b" in ln]
    assert bad and "failed" in bad[0] and "RuntimeError" in bad[0]


def test_a_stop_request_ends_the_set_run(tmp_path, monkeypatch):
    parent = tmp_path / "set"
    paths = [_full_acq(parent / n) for n in ("acq_a", "acq_b")]
    flag = {"stop": False}

    def fake_once(reader, **kw):
        flag["stop"] = True                                      # requested during member 1
        return _ok_result()

    monkeypatch.setattr("squidxplorer._dispatch.run_operator_once", fake_once)
    lines: list = []
    summary = _acqset.run_over_set(paths, operator="mip", out_parent=str(tmp_path),
                                   log=lines.append, stop=lambda: flag["stop"])
    assert summary["ok"] == 1 and summary["stopped"] is True
    assert any("stopped before acquisition 2 of 2" in ln for ln in lines)


# -- the window: set ingest, cycling, bulk routing ------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


def _set_window(qapp, tmp_path):
    from squidxplorer import _viewer as V

    parent = tmp_path / "night_run"
    _full_acq(parent / "acq_a")
    _full_acq(parent / "acq_b")
    win = V.PlateWindow(None)
    win.ingest(str(parent))
    return win, parent


def test_ingest_of_a_set_loads_the_first_and_records_the_set(qapp, tmp_path):
    win, parent = _set_window(qapp, tmp_path)
    try:
        assert win._acq_set == [parent / "acq_a", parent / "acq_b"]
        assert win._acq_set_index == 0
        assert win._acq_name == "acq_a"                          # the FIRST member is open
        assert not win._acq_cycle_label.isHidden()
        assert win._acq_cycle_label.text() == "acquisition 1 of 2"
        assert not win._bulk_all_box.isHidden()
    finally:
        win.close()


def test_cycling_re_ingests_the_neighbour(qapp, tmp_path):
    win, parent = _set_window(qapp, tmp_path)
    try:
        win._cycle_acq(+1)
        assert win._acq_name == "acq_b" and win._acq_set_index == 1
        assert win._acq_cycle_label.text() == "acquisition 2 of 2"
        assert win._acq_set == [parent / "acq_a", parent / "acq_b"]   # the set survives
        win._acq_next_btn.click()                                # the button drives the same path
        assert win._acq_name == "acq_a" and win._acq_set_index == 0  # and it wraps
        win._acq_prev_btn.click()
        assert win._acq_name == "acq_b" and win._acq_set_index == 1
    finally:
        win.close()


def test_a_single_acquisition_shows_no_cycling_ui(qapp, tmp_path):
    from squidxplorer import _viewer as V

    root = _full_acq(tmp_path / "solo_acq")
    win = V.PlateWindow(None)
    try:
        win.ingest(str(root))
        assert win._acq_set is None
        assert win._acq_cycle_label.isHidden() and win._bulk_all_box.isHidden()
        assert win._acq_prev_btn.isHidden() and win._acq_next_btn.isHidden()
        win._cycle_acq(+1)                                       # a no-op, never a crash
        assert win._acq_name == "solo_acq"
    finally:
        win.close()


def test_a_checked_bulk_box_routes_a_save_to_the_set_run(qapp, tmp_path, monkeypatch):
    win, _parent = _set_window(qapp, tmp_path)
    try:
        routed: list = []
        monkeypatch.setattr(win, "_run_bulk_over_set",
                            lambda key, out_parent, kwargs: routed.append((key, out_parent, kwargs)))
        win._bulk_all_box.setChecked(True)
        win.run_operator("mip", out_parent=str(tmp_path / "out"), save=True,
                         operator_kwargs={"a": 1})
        assert routed == [("mip", str(tmp_path / "out"), {"a": 1})]
        assert win._worker is None                               # no single-plate run started
    finally:
        win.close()


def test_the_bulk_done_slot_reports_the_tally(qapp, tmp_path):
    win, _parent = _set_window(qapp, tmp_path)
    try:
        win._on_bulk_done({"ok": 1, "partial": 1, "failed": 1, "total": 3, "stopped": False})
        assert "1 ok, 1 partial, 1 failed of 3" in win._readout.text()
    finally:
        win.close()
