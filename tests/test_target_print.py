"""The operator UI prints which windows a run is aimed at, and reconciles the
deduplicated region counts."""

from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede the Qt import

import pytest  # noqa: E402

from squidxplorer._run_scope import describe_view_target, distinct_view_regions  # noqa: E402


class _View:
    """A ``View`` as the printer reads one: duck-typed, no Qt, no napari, no reader."""

    def __init__(self, window_id, name, regions, roi_bbox=None):
        self.id = f"w{window_id}"
        self.window_id = window_id
        self.name = name
        self.regions = tuple(regions)
        self.roi_bbox = roi_bbox
        self.kind = "roi" if roi_bbox is not None else "window"


def _line(block: str, needle: str) -> str:
    hits = [ln for ln in block.splitlines() if needle in ln]
    assert len(hits) == 1, f"expected one line containing {needle!r}, got {hits!r}"
    return hits[0].strip()


# ------------------------------------------------------------------ the numbers must reconcile


def test_the_reconciliation_line_is_correct_when_dedup_removes_a_region():
    """13 region slots across 3 windows, B6 held twice, so 12 regions run."""
    views = [
        _View(2, "Deconvolution trial", ["A1", "A2", "A3", "A4"]),
        _View(5, "ROI · B6  ◂ view 2", ["B6"], roi_bbox=(120.0, 340.0, 636.0, 856.0)),
        _View(7, "C3, C4, C5, +5", ["C3", "C4", "C5", "C6", "C7", "C8", "C9", "B6"]),
    ]
    block = describe_view_target(views, action="Run decon")

    assert block.splitlines()[0] == "Run decon on 3 windows, 12 regions"
    assert _line(block, "region slots") == (
        "13 region slots across 3 windows, 12 distinct regions")
    assert _line(block, "processed once") == (
        "B6 is held by 2 windows and will be processed once")

    assert len(distinct_view_regions(views)) == 12
    assert sum(len(v.regions) for v in views) == 13, (
        "the fixture stopped exercising dedup, so this test stopped testing anything")


def test_no_overlap_prints_one_number_because_there_is_only_one():
    views = [
        _View(2, "Deconvolution trial", ["A1", "A2", "A3", "A4"]),
        _View(5, "B6", ["B6"]),
        _View(7, "C3, C4, C5, +4", ["C3", "C4", "C5", "C6", "C7", "C8", "C9"]),
    ]
    block = describe_view_target(views, action="Run stitch")

    assert block.splitlines()[0] == "Run stitch on 3 windows, 12 regions"
    assert _line(block, "across") == "12 regions across 3 windows"
    assert "region slots" not in block
    assert "processed once" not in block


def test_several_overlapping_regions_are_all_named_with_their_multiplicity():
    views = [_View(2, "one", ["A1", "B2"]), _View(11, "two", ["A1", "B2", "C3"]),
             _View(12, "three", ["A1"])]
    block = describe_view_target(views, action="Run decon")

    assert block.splitlines()[0] == "Run decon on 3 windows, 3 regions"
    assert _line(block, "region slots") == (
        "6 region slots across 3 windows, 3 distinct regions")
    assert _line(block, "processed once") == (
        "2 regions are held by more than one window and will each be processed once: "
        "A1 ×3, B2 ×2")


def test_a_window_that_lists_a_region_twice_still_counts_it_once():
    block = describe_view_target([_View(2, "dupe", ["A1", "A1", "B2"])], action="Run decon")
    assert block.splitlines()[0] == "Run decon on 1 window, 2 regions"
    assert _line(block, "A1") == "[2]  dupe  2 regions  A1, B2"
    assert _line(block, "across") == "2 regions across 1 window"


# ------------------------------------------------------------------------------- the layout


def test_the_bracket_is_the_same_token_the_log_prints():
    """The identity column comes from ``window_id``, never from the name field."""
    block = describe_view_target([_View(3, "whatever", ["A1"])])
    assert _line(block, "whatever").startswith("[3]  ")


def test_the_label_is_the_renamed_label_because_that_is_what_makes_a_rename_worth_having():
    block = describe_view_target([_View(2, "Deconvolution trial", ["A1"])])
    assert "Deconvolution trial" in block


def test_a_long_label_is_elided_rather_than_wrapped():
    block = describe_view_target([_View(2, "x" * 80, ["A1"]), _View(3, "short", ["A2"])])
    assert len(block.splitlines()) == 6, "a row wrapped onto a second line"
    assert "…" in block
    assert "x" * 80 not in block


def test_the_columns_line_up_across_rows():
    """Widths are computed from the rows; one id is wider than any plausible fixed column."""
    views = [_View(2, "aa", ["A1"]), _View(11, "bbbb", ["A2", "A3"]),
             _View(1207, "c", ["A4", "A5", "A6"])]
    rows = [ln for ln in describe_view_target(views).splitlines() if ln.startswith("  [")]
    assert len(rows) == 3
    assert len({r.index("region") for r in rows}) == 1, "the count column is ragged"
    assert len({r.index("]") for r in rows}) == 3, (
        "the fixture stopped exercising varying id widths")


def test_the_roi_subset_is_printed_in_the_existing_extent_spelling():
    """The printer derives the box from ``Extent.label()`` rather than copying the format."""
    from squidxplorer._address import Extent

    bbox = (120.0, 340.0, 636.0, 856.0)
    block = describe_view_target([_View(5, "ROI · B6", ["B6"], roi_bbox=bbox)])
    assert Extent(region_id="B6", bbox_um=bbox).label().split(" ", 1)[1] in block
    assert "roi [120.0,340.0 636.0,856.0] um" in block


def test_the_region_names_truncate_in_the_one_overflow_spelling_this_codebase_has():
    """Same ``, ... (+N more)`` tail ``describe_run_target`` prints."""
    block = describe_view_target([_View(2, "many", [f"A{i}" for i in range(1, 11)])])
    assert "A1, A2, A3, A4, A5, A6, ... (+4 more)" in block


# --------------------------------------------------------------------------------- refusals


def test_nothing_to_run_returns_None_rather_than_a_cheerful_zero():
    assert describe_view_target([]) is None
    assert describe_view_target(None) is None
    assert describe_view_target([_View(2, "empty", [])]) is None


# ------------------------------------------------------ the wiring: it prints where a user looks

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)

from qtpy.QtWidgets import QComboBox, QPushButton  # noqa: E402

from squidxplorer import _viewer as V  # noqa: E402

from .test_viewer import qapp  # noqa: E402,F401  (fixtures)


class _FakeWindow:
    """A ``RegionViewer`` as ``ViewerManager.views()`` reads one; a real one needs a GL canvas."""

    def __init__(self, window_id, regions, name=None, roi_bbox=None):
        self.window_id = int(window_id)
        self._regions = list(regions)
        self.display_name = name or ", ".join(regions)
        self._roi_bbox = roi_bbox
        self.parent_id = None


def _plate(squid_dataset, *wins):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    for w in wins:
        win._viewer_manager._windows[w.window_id] = w
    return win


def test_the_print_and_the_run_cannot_disagree(qapp, squid_dataset):
    """Both counts come out of one flattener, ``distinct_view_regions``."""
    win = _plate(squid_dataset, _FakeWindow(2, ["B2", "B3"]), _FakeWindow(5, ["B3"]))
    try:
        regions = win._open_views_regions()
        assert regions == ["B2", "B3"], "the flattener changed order or stopped deduplicating"

        block = V._run_scope.describe_view_target(win._open_view_targets(), action="Run decon")
        assert block.splitlines()[0] == "Run decon on 2 windows, 2 regions"
        assert _line(block, "region slots") == (
            "3 region slots across 2 windows, 2 distinct regions")
        assert str(len(regions)) in block.splitlines()[0]
    finally:
        win._stop_worker(); win.close()


def test_choosing_open_views_prints_the_block_before_anything_runs(qapp,
                                                                  squid_dataset, caplog):
    """Printed when the target is picked; driven through the combo so the wiring is exercised."""
    win = _plate(squid_dataset, _FakeWindow(2, ["B2", "B3"], name="Deconvolution trial"),
                 _FakeWindow(5, ["B3"]))
    try:
        tab = win._build_run_tab(V._OPERATIONS_BY_KEY["mip"])
        combo = [c for c in tab.findChildren(QComboBox)
                 if "Open views" in [c.itemText(i) for i in range(c.count())]][0]

        with caplog.at_level(logging.INFO):
            combo.setCurrentText("Open views")

        printed = [r.getMessage() for r in caplog.records if "region slots" in r.getMessage()]
        assert printed, "picking 'Open views' printed nothing to the log console"
        assert "Deconvolution trial" in printed[-1], "the block did not name the windows"
        assert "3 region slots across 2 windows, 2 distinct regions" in printed[-1]
        assert win._readout.text().startswith("Run "), (
            "the status line did not carry the headline")
    finally:
        win._stop_worker(); win.close()


def test_pressing_run_prints_it_again_as_the_record_of_what_ran(qapp, squid_dataset,
                                                               caplog, monkeypatch):
    win = _plate(squid_dataset, _FakeWindow(2, ["B2", "B3"]), _FakeWindow(5, ["B3"]))
    seen = {}
    monkeypatch.setattr(V.PlateWindow, "run_operator",
                        lambda self, key, out_parent=None, regions=None, **kw:
                        seen.update(regions=regions))
    try:
        tab = win._build_run_tab(V._OPERATIONS_BY_KEY["mip"])
        combo = [c for c in tab.findChildren(QComboBox)
                 if "Open views" in [c.itemText(i) for i in range(c.count())]][0]
        combo.setCurrentText("Open views")
        run = [b for b in tab.findChildren(QPushButton) if b.text() == "Run"][0]
        run.setEnabled(True)

        caplog.clear()
        with caplog.at_level(logging.INFO):
            run.click()

        assert seen["regions"] == ["B2", "B3"], "the run did not get the printed region set"
        assert any("region slots" in r.getMessage() for r in caplog.records), (
            "pressing Run left no record of what it was aimed at")
    finally:
        win._stop_worker(); win.close()


def test_open_views_with_no_windows_still_refuses_in_a_sentence(qapp, squid_dataset):
    win = _plate(squid_dataset)
    try:
        assert win._print_open_views_target("Run decon") is None
        assert win._readout.text() == (
            "Run on open views: no windows are open — open some first.")
    finally:
        win._stop_worker(); win.close()


def test_windows_that_hold_no_regions_say_so_rather_than_claiming_none_are_open(qapp,
                                                                                squid_dataset):
    win = _plate(squid_dataset, _FakeWindow(2, []), _FakeWindow(5, []))
    try:
        assert win._print_open_views_target("Run decon") is None
        assert "2 open window(s) hold no regions" in win._readout.text()
    finally:
        win._stop_worker(); win.close()
