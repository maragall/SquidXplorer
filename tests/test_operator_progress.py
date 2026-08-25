"""How far an operator run has got, said on screen instead of in the log."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any Qt import

import pytest

from squidxplorer._progress import (
    FOV_UNIT,
    REGION_UNIT,
    ProgressReport,
    RunProgress,
    format_eta,
    unit_plan,
)

# A two-region acquisition whose wells hold different FOV counts, so a per-region total and a
# per-FOV total cannot accidentally agree.
_META = {"regions": ["B2", "B3"], "fovs_per_region": {"B2": [0, 1, 2], "B3": [0, 1]}}


def test_a_per_fov_operator_counts_fovs_not_wells():
    """A decon over one region is 3 units of work, not 1, not a well count."""
    assert unit_plan(_META, ["B2"], region_op=False, n_fovs=None) == (3, FOV_UNIT)
    assert unit_plan(_META, None, region_op=False, n_fovs=None) == (5, FOV_UNIT)


def test_a_region_operator_counts_regions_because_that_is_its_unit():
    """the region loop yields ONE fused mosaic per region, so a region is the honest unit there."""
    assert unit_plan(_META, ["B2", "B3"], region_op=True, n_fovs=None) == (2, REGION_UNIT)


def test_an_explicit_fov_count_is_clamped_to_what_each_region_has():
    """A denominator larger than the work that will run would leave the bar short forever."""
    assert unit_plan(_META, None, region_op=False, n_fovs=2) == (4, FOV_UNIT)


def test_a_region_the_engine_would_drop_is_not_counted():
    """The engine drops unknown region names; a denominator that counts them never completes."""
    assert unit_plan(_META, ["B2", "NOPE"], region_op=False, n_fovs=None) == (3, FOV_UNIT)


def test_no_fov_table_means_no_total_rather_than_a_guess():
    """The one honest 'unknown'. The window draws an indeterminate bar off this None."""
    total, unit = unit_plan({"regions": ["B2"]}, None, region_op=False, n_fovs=None)
    assert total is None and unit == FOV_UNIT


def test_an_unknown_total_never_produces_a_percentage():
    """A progress bar that invents a denominator is a lie that gets believed."""
    r = ProgressReport("decon", done=7, total=None, unit=FOV_UNIT)
    assert r.percent is None and r.determinate is False
    assert "7 FOVs so far" in r.sentence()
    assert "of" not in r.sentence().split("·")[-1]


def test_the_first_unit_is_never_extrapolated():
    """The first arrival pays warm-up cost; an ETA from it would announce a number it can't meet."""
    p = RunProgress("decon", total=10)
    p.tick(100.0)
    assert p.eta() is None, "one completion is not a rate"


def test_time_remaining_is_a_rate_over_completions_after_the_first():
    """Two units 2 s apart, 8 left -> 16 s. The 100 s the FIRST unit took does not appear."""
    p = RunProgress("decon", total=10)
    p.tick(100.0)                     # warm-up, deliberately not measured
    p.tick(102.0)
    assert p.eta() == pytest.approx(16.0)
    p.tick(104.0)
    assert p.eta() == pytest.approx(14.0)


def test_a_finished_run_has_no_time_left_rather_than_a_negative_one():
    p = RunProgress("mip", total=2)
    p.tick(1.0)
    p.tick(2.0)
    assert p.eta() == 0.0
    assert p.report().percent == 100


def test_an_unknown_total_offers_no_time_remaining():
    p = RunProgress("decon", total=None)
    p.tick(1.0)
    p.tick(2.0)
    assert p.eta() is None and p.report().sentence().endswith("2 FOVs so far")


def test_the_estimate_is_stated_coarsely_because_the_sample_is_small():
    """Rounded UP, in buckets, so a run tends to beat its promise rather than miss it."""
    assert format_eta(None) == "" and format_eta(-1) == ""
    assert format_eta(3) == "a few seconds left"
    assert format_eta(41) == "~50 s left"
    assert format_eta(247) == "~5 min left"
    assert format_eta(3600 * 2) == "~2 h left"
    assert format_eta(3600 * 2 + 60) == "~2 h 1 min left"


def test_the_sentence_names_the_operator_the_count_and_the_wait():
    r = ProgressReport("decon", done=12, total=27, unit=FOV_UNIT, eta_seconds=200)
    assert r.sentence() == "decon · 12 of 27 FOVs · ~4 min left"
    assert ProgressReport("stitch", 0, 1, REGION_UNIT).sentence() == "stitch · 0 of 1 region"


pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:      # pragma: no cover - env gate
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

import squidxplorer._viewer as V  # noqa: E402
import squidxplorer._workers as W  # noqa: E402

from .conftest import FOVS, REGIONS  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixture)


class _Requester:
    """Duck-typed on the four ``operator_*`` names, which ARE the contract with the plate."""

    window_id = "test"

    def __init__(self):
        self.started, self.reports, self.done, self.failed = [], [], [], []

    def operator_started(self, action):
        self.started.append(action)

    def operator_progress(self, report):
        self.reports.append(report)

    def operator_done(self, action, seconds):
        self.done.append((action, seconds))

    def operator_failed(self, action, reason):
        self.failed.append((action, reason))


@pytest.fixture
def plate(qapp, monkeypatch, squid_dataset):
    win = V.PlateWindow(None)
    root, _arrays = squid_dataset
    win.ingest(str(root))
    yield win
    win._stop_worker()
    win._stop_preview()
    win.close()


def _run_to_completion(qapp, win, requester, **kw):
    win.run_operator("mip", regions=[REGIONS[0]], save=False, requester=requester, **kw)
    assert _drain_until(qapp, lambda: bool(requester.done or requester.failed), timeout=60), (
        "the run never closed its pair on the window that asked for it")


def test_the_window_that_asked_is_told_the_run_started(qapp, plate):
    r = _Requester()
    _run_to_completion(qapp, plate, r)
    assert r.started, "the requester was never told the run began"


def test_progress_climbs_monotonically_to_every_unit_of_the_run(qapp, plate):
    """N units in, N of N out, never a step backwards; the fixture's region holds 2 FOVs."""
    r = _Requester()
    _run_to_completion(qapp, plate, r)

    assert r.reports, "a run reported no progress at all"
    totals = {rep.total for rep in r.reports}
    assert totals == {len(FOVS)}, f"the denominator moved during the run: {totals}"
    dones = [rep.done for rep in r.reports]
    assert dones[0] == 0, "the first report must be 0 of N, before any unit lands"
    assert dones == sorted(dones), f"progress went backwards: {dones}"
    assert dones[-1] == r.reports[-1].total, f"the run ended at {dones[-1]} of {totals}"
    assert r.reports[-1].percent == 100


def test_a_failed_run_closes_the_pair_so_the_bar_cannot_be_left_running(qapp, plate):
    """A bar taken down only on success is left sweeping over a dead run."""
    import squidxplorer

    def _boom(*_a, **_kw):
        raise RuntimeError("no such plane")

    # the engine raising is the real shape of a failed run (`_run_body` catches it and emits `failed`)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(squidxplorer, "run_plate", _boom)
        r = _Requester()
        plate.run_operator("mip", regions=[REGIONS[0]], save=False, requester=r)
        assert _drain_until(qapp, lambda: bool(r.failed or r.done), timeout=60)
    assert r.failed and not r.done, f"a failed run reported {r.done!r} instead of a failure"
    assert "no such plane" in r.failed[0][1], "the failure did not name its cause"


def test_the_asking_window_is_left_knowing_the_run_reached_its_total(qapp, plate):
    """The last report and ``QThread.finished`` race; the drain reads the worker's own tally
    instead of the signal, pinned here with the signal path removed."""
    r = _Requester()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(V.PlateWindow, "_on_unit_progress", lambda self, report: None)
        plate.run_operator("mip", regions=[REGIONS[0]], save=False, requester=r)
        assert _drain_until(qapp, lambda: bool(r.done or r.failed), timeout=60)
    assert r.reports, "the run ended without ever telling the window how far it got"
    last = r.reports[-1]
    assert last.done == last.total == len(FOVS)


def test_the_plate_stops_talking_to_a_requester_once_its_run_has_drained(qapp, plate):
    """One run, one pair; a stale requester would leak the NEXT run's progress into it."""
    first = _Requester()
    _run_to_completion(qapp, plate, first)
    assert plate._run.requester is None
    n = len(first.reports)

    second = _Requester()
    _run_to_completion(qapp, plate, second)
    assert len(first.reports) == n, "a drained requester was still being fed"
    assert second.reports


def test_only_the_window_that_ASKED_gets_the_result_visible(qapp, plate):
    """Ruling r (Julio, 2026-08-25, "decon layer is != raw view"): a preview's result lands
    ONLY in the view that asked, visible; a view that did not ask receives nothing (the dark
    fan-out survives only for a run no view asked for)."""
    seen = {}

    class _Win:
        def __init__(self, name):
            self.window_id = name

        def deliver_result(self, op, result, *, visible):
            seen[self.window_id] = visible
            return 1

    asked, other = _Win("asked"), _Win("other")

    class _Mgr:
        windows = [asked, other]

    from squidxplorer._run import OperatorRun

    plate._run = OperatorRun(key="mip", layer_key="mip", label="mip", action=None, dest="",
                             address=None, requester=asked, is_partial=False, t0=0.0)
    plate._viewer_manager, saved = _Mgr(), plate._viewer_manager
    try:
        added = plate._deliver_to_views("mip", object())
    finally:
        plate._viewer_manager = saved
        plate._run = None

    assert added == 1, added
    assert seen == {"asked": True}, seen           # the view that did not ask gets nothing


@pytest.fixture
def region_window(qapp, napari_pane_stub, squid_dataset):
    from squidxplorer import open_reader
    from squidxplorer._region_viewer import ViewerManager

    root, _arrays = squid_dataset
    reader = open_reader(str(root))
    mgr = ViewerManager(reader, reader.metadata)
    win = mgr.open([REGIONS[0]])
    assert win is not None
    try:
        yield win
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()


def test_the_bar_is_absent_until_a_run_starts(region_window):
    """A bar at 0% over an idle window is indistinguishable from a wedged run."""
    assert region_window._op_progress is not None, "the window has no progress bar at all"
    assert region_window._op_progress.isHidden()


def test_the_bar_comes_up_indeterminate_before_the_first_report(region_window):
    region_window.operator_started("decon")
    bar = region_window._op_progress
    assert not bar.isHidden()
    assert (bar.minimum(), bar.maximum()) == (0, 0), "an unknown total must not draw a percentage"


def test_the_bar_goes_determinate_and_says_the_count_and_the_wait(region_window):
    region_window.operator_started("decon")
    region_window.operator_progress(
        ProgressReport("decon", done=12, total=27, unit=FOV_UNIT, eta_seconds=200))
    bar = region_window._op_progress
    assert (bar.minimum(), bar.maximum()) == (0, 100)
    assert bar.value() == 44
    assert bar.format() == "decon · 12 of 27 FOVs · ~4 min left"


def test_a_report_with_no_total_keeps_the_bar_indeterminate(region_window):
    """An operator that cannot know its total gets a sweep, not a fabricated percentage."""
    region_window.operator_started("decon")
    region_window.operator_progress(ProgressReport("decon", done=3, total=None, unit=FOV_UNIT))
    bar = region_window._op_progress
    assert (bar.minimum(), bar.maximum()) == (0, 0)
    assert "3 FOVs so far" in bar.format()


@pytest.mark.parametrize("close", [
    lambda w: w.operator_done("decon", 1.5),
    lambda w: w.operator_failed("decon", "no such plane"),
])
def test_every_outcome_takes_the_bar_down(region_window, close):
    """Done and failed alike; no outcome leaves the bar running."""
    region_window.operator_started("decon")
    region_window.operator_progress(ProgressReport("decon", 1, 27, FOV_UNIT))
    close(region_window)
    bar = region_window._op_progress
    assert bar.isHidden()
    assert (bar.minimum(), bar.maximum()) == (0, 100), "an indeterminate sweep was left behind"


@pytest.fixture
def navigator(qapp):
    """Deliberately dataset-free: the bar is a pure function of the report it is handed.
    `StatusRow` is what survived the Window navigator's deletion (2026-08-19) — the same bars,
    the same manager wiring, adopted by the log panel."""
    from squidxplorer._region_viewer import StatusRow, ViewerManager

    mgr = ViewerManager()
    mgr._mem_timer.stop()                       # no polling: this test is not about memory
    panel = StatusRow(mgr)
    try:
        yield mgr, panel
    finally:
        for w in panel.widgets():
            w.deleteLater()
        panel.deleteLater()
        qapp.processEvents()


def test_the_work_bar_is_ABSENT_while_nothing_is_running(navigator):
    """Absent, not parked at 0%, which would look like a run that started and produced nothing."""
    _mgr, panel = navigator
    assert panel._work_bar.isHidden()
    assert panel._work_label.isHidden()


def test_a_report_from_ANY_producer_raises_the_bar_beside_the_memory_bar(navigator):
    mgr, panel = navigator
    mgr.set_run_progress(
        ProgressReport("decon", done=12, total=27, unit=FOV_UNIT, eta_seconds=200))
    assert not panel._work_bar.isHidden()
    assert (panel._work_bar.minimum(), panel._work_bar.maximum()) == (0, 100)
    assert panel._work_bar.value() == 44
    assert panel._work_label.text() == "decon · 12 of 27 FOVs · ~4 min left"


def test_the_work_bar_stays_INDETERMINATE_when_the_total_is_not_known(navigator):
    """Same rule as the region window's bar: an invented denominator is a lie that gets believed."""
    mgr, panel = navigator
    mgr.set_run_progress(ProgressReport("preview", done=3, total=None, unit=FOV_UNIT))
    assert (panel._work_bar.minimum(), panel._work_bar.maximum()) == (0, 0)
    assert "3 FOVs so far" in panel._work_label.text()


def test_clearing_the_channel_takes_the_work_bar_down(navigator):
    mgr, panel = navigator
    mgr.set_run_progress(ProgressReport("decon", 1, 27, FOV_UNIT))
    mgr.set_run_progress(None)
    assert panel._work_bar.isHidden()
    assert panel._work_label.isHidden()


def test_a_status_row_built_MID_RUN_shows_the_bar_without_waiting_for_the_next_unit(qapp):
    """The manager holds the last report so a late subscriber does not wait for the next unit."""
    from squidxplorer._region_viewer import StatusRow, ViewerManager

    mgr = ViewerManager()
    mgr._mem_timer.stop()
    mgr.set_run_progress(ProgressReport("decon", 5, 27, FOV_UNIT))
    panel = StatusRow(mgr)                      # built AFTER the run was already reporting
    try:
        assert not panel._work_bar.isHidden()
        assert panel._work_bar.value() == 19
    finally:
        for w in panel.widgets():
            w.deleteLater()
        panel.deleteLater()
        qapp.processEvents()


def test_a_plate_wide_run_reaches_the_status_row_and_is_taken_down_when_it_drains(qapp, plate):
    """A real run on a real PlateWindow with NO requester window — the bulk case."""
    seen = []
    plate._viewer_manager.runProgressChanged.connect(seen.append)
    plate.run_operator("mip", regions=[REGIONS[0]], save=False)
    assert _drain_until(qapp, lambda: seen and seen[-1] is None, timeout=60), (
        f"the work bar was never taken down; last was {seen[-1] if seen else None!r}")
    reports = [r for r in seen if r is not None]
    assert reports, "a plate-wide run published no progress to the status row at all"
    assert reports[-1].done == reports[-1].total == len(FOVS)


class _PrintingReader:
    """Stands in for libraries (e.g. tilefusion) that report progress via bare ``print``."""

    def __init__(self, path, chatter: bool = False):
        self._path = str(path)               # the identity the plate cache's token asks for
        self.chatter = bool(chatter)

    def read(self, region, fov, channel, z_level, time_point=0):
        import numpy as np

        if self.chatter:
            print(f"Parallel registration: {region} fov {fov}")
        return np.zeros((8, 8), dtype=np.uint16)


def _preview_meta() -> dict:
    return {"channels": [{"name": "c0"}], "dtype": "uint16", "z_levels": [0, 1, 2],
            "regions": ["A1", "A2"], "fovs_per_region": {"A1": [0], "A2": [0]},
            "frame_shape": (8, 8), "pixel_size_um": 1.0, "fov_positions_um": {}}


def _preview_worker(tmp_path, cache=None, **kw):
    (tmp_path / "acq").mkdir(exist_ok=True)
    return W._PreviewWorker(_PrintingReader(tmp_path / "acq", **kw), _preview_meta(),
                            {"A1": {"rc": (0, 0)}, "A2": {"rc": (0, 1)}}, ["A1", "A2"],
                            cache=cache)


def test_the_PREVIEW_reports_on_the_same_channel_an_operator_run_does(qapp, tmp_path):
    """The preview emits the same ``ProgressReport`` type the navigator's bar already draws."""
    worker = _preview_worker(tmp_path)
    got = []
    worker.runProgress.connect(got.append)
    worker.run()                                 # in-thread: signal delivery is synchronous here

    assert got, "the raw preview reported no progress at all"
    assert all(isinstance(r, ProgressReport) for r in got), \
        "the preview invented a second progress type instead of sharing the channel"


def test_the_previews_bar_is_DETERMINATE_from_its_first_frame(qapp, tmp_path):
    """The total is known before the first read (``len(plan)``), so it never grows mid-pass."""
    worker = _preview_worker(tmp_path)
    got = []
    worker.runProgress.connect(got.append)
    worker.run()

    assert got[0].done == 0, "the preview's first report was not 0 of N"
    assert {r.total for r in got} == {2}, "the preview's denominator moved mid-pass"
    dones = [r.done for r in got]
    assert dones == sorted(dones), f"preview progress went backwards: {dones}"
    assert got[-1].done == got[-1].total, "the preview never reached its own total"
    assert got[-1].percent == 100


def test_the_preview_names_itself_so_the_one_bar_says_WHICH_work_is_running(qapp, tmp_path):
    """The label is the only field distinguishing a preview from an operator run on the wire."""
    from squidxplorer._progress import PREVIEW_LABEL

    worker = _preview_worker(tmp_path)
    got = []
    worker.runProgress.connect(got.append)
    worker.run()
    assert got[-1].label == PREVIEW_LABEL
    assert got[-1].sentence().startswith("preview · ")


def test_a_CACHED_well_is_not_counted_as_work_the_preview_still_has_to_do(qapp, tmp_path):
    """The denominator is the plan that SURVIVED the cache, not the replayed wells."""
    import numpy as np

    from squidxplorer._platecache import PlateCellCache

    exp = tmp_path / "acq"
    exp.mkdir(exist_ok=True)
    (exp / "acquisition.yaml").write_text("objective:\n  pixel_size_um: 1.0\n")

    def _cache():
        return PlateCellCache(exp, cell_px=88, channels=["c0"], dtype=np.uint16,
                              root=tmp_path / "cachedir")

    _preview_worker(tmp_path, cache=_cache()).run()      # fills the cache for A1 and A2

    second = _preview_worker(tmp_path, cache=_cache())
    got = []
    second.runProgress.connect(got.append)
    second.run()

    assert second.cache_hits == 2, "the fixture did not actually exercise the cache"
    assert got and got[0].total == 0, \
        f"a fully cached reopen still claimed {got[0].total} units of work to do"


# --- a copy-saving operator's preview run says how to get the artifact (2026-08-19) ---------------

def test_a_copy_saving_preview_run_names_the_artifact_it_did_not_write(region_window):
    """Julio: "Registering the wells doesn't do anything" — a register preview succeeded, wrote
    nothing, and nothing said how to get the stitched_ copy. The hint goes to the LOG now
    (the banner strip is retired, 2026-08-25); the done line's twin is log.done."""
    region_window._op_run_wrote = False
    said = []
    region_window._say = said.append
    region_window.operator_started("register")
    region_window.operator_done("register", 2.0)
    assert said, "the artifact hint never reached the log"
    line = said[-1]
    assert "preview only" in line and "stitched_" in line and "save" in line, line


def test_a_run_that_wrote_its_copy_gets_no_preview_hint(region_window):
    region_window._op_run_wrote = True
    said = []
    region_window._say = said.append
    region_window.operator_started("register")
    region_window.operator_done("register", 2.0)
    assert not any("preview only" in s for s in said), said


def test_an_ordinary_operator_preview_gets_no_copy_hint(region_window):
    """The hint is declaration-driven (operator_saves_copy), never name-matched."""
    region_window._op_run_wrote = False
    said = []
    region_window._say = said.append
    region_window.operator_started("mip")
    region_window.operator_done("mip", 1.0)
    assert not any("preview only" in s for s in said), said
