"""How far an operator run has got, said on screen instead of in the log.

Julio, 2026-08-03, on a decon over ONE region that took 433 seconds::

    "using decon as an example, I choose to run decon, and I know it's running because of the
     orange dot ... But there's nothing on the child window that tells me how much is left what's
     the progress, or that it is working. It only tells me that it worked after layers populated,
     but how long is that?"

WHAT IS ACTUALLY UNDER TEST
---------------------------
Two halves, tested at two different heights, and neither at the height of a screenshot.

* The arithmetic (:mod:`squidmip._progress`) is pure Python and is tested as such: which unit a
  run counts, whether the total is knowable, and when a time-remaining estimate is honest enough
  to show. This is the half that can be silently WRONG, so it is the half with the small tests.

* The wiring is tested at the HIGHEST seam that exists: a real ``PlateWindow`` running a real
  operator over a real acquisition, with a real ``RegionViewer`` as the ``requester``. That is the
  whole reported path — a window asks for a run and is told what is happening — and it is the path
  that had NO test at all, which is why ``requester=`` had been an accepted-and-dropped argument
  since 2026-07-29 without anything noticing.

NOTHING HERE WAS VERIFIED ON SCREEN. These assert the bar's RANGE, VALUE, FORMAT and VISIBILITY,
which is what the widget is asked for; they do not assert that a human can see it.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # headless Qt; must precede any Qt import

import pytest

from squidmip._progress import (
    FOV_UNIT,
    REGION_UNIT,
    ProgressReport,
    RunProgress,
    format_eta,
    unit_plan,
)

# --------------------------------------------------------------------------------------
# The arithmetic. No Qt, no window, no event loop.
# --------------------------------------------------------------------------------------

#: A two-region acquisition whose wells hold different FOV counts, so a per-region total and a
#: per-FOV total cannot accidentally agree.
_META = {"regions": ["B2", "B3"], "fovs_per_region": {"B2": [0, 1, 2], "B3": [0, 1]}}


def test_a_per_fov_operator_counts_fovs_not_wells():
    """THE BUG, as a unit. A decon over one region is 3 units of work, not 1.

    The well counter says 0 of 1 for the whole run and then 1 of 1, which is what left 433
    seconds with nothing on screen.
    """
    assert unit_plan(_META, ["B2"], region_op=False, n_fovs=None) == (3, FOV_UNIT)
    assert unit_plan(_META, None, region_op=False, n_fovs=None) == (5, FOV_UNIT)


def test_a_region_operator_counts_regions_because_that_is_its_unit():
    """stitch_plate yields ONE fused mosaic per region, so a region is the honest unit there."""
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
    """squidmip._activity's rule, enforced on the report the widget actually reads: a progress bar
    that invents a denominator is a lie that gets believed."""
    r = ProgressReport("decon", done=7, total=None, unit=FOV_UNIT)
    assert r.percent is None and r.determinate is False
    assert "7 FOVs so far" in r.sentence()
    assert "of" not in r.sentence().split("·")[-1]


def test_the_first_unit_is_never_extrapolated():
    """The interval from the click to the first arrival pays metadata warm-up, pool priming and a
    cache-cold read. A run whose ETA came from it would announce a number it cannot meet."""
    p = RunProgress("decon", total=10)
    p.tick(100.0)
    assert p.eta() is None, "one completion is not a rate"


def test_time_remaining_is_a_rate_over_completions_after_the_first():
    """Two units 2 s apart, 8 left -> 16 s. The 100 s the FIRST unit took does not appear."""
    p = RunProgress("decon", total=10)
    p.tick(100.0)                     # 100 s after the click; warm-up, deliberately not measured
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
    """Rounded UP, in buckets. '247 s left' claims a precision a handful of arrivals cannot give,
    and rounding up means a run tends to beat its promise rather than miss it."""
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


# --------------------------------------------------------------------------------------
# The wiring, through the real windows.
# --------------------------------------------------------------------------------------

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:      # pragma: no cover - env gate
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

import squidmip._viewer as V  # noqa: E402

from .conftest import FOVS, REGIONS  # noqa: E402
from .test_viewer import _drain_until, _StubDetail, qapp  # noqa: E402,F401  (fixture)


class _Requester:
    """A window standing in for a ``RegionViewer``, recording exactly what it was told.

    Deliberately duck-typed on the four ``operator_*`` names rather than subclassing: those four
    names ARE the contract between the plate and any window that asks it for a run, and a test that
    inherited the real class would pass even if the contract were renamed on both sides at once.
    """

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
    monkeypatch.setattr(V.PlateWindow, "_make_detail_viewer", lambda self: _StubDetail())
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
    """``requester=`` was accepted and dropped: nothing assigned ``_run_requester``, so the four
    callbacks the docstring promises were never called and the region window sat silent."""
    r = _Requester()
    _run_to_completion(qapp, plate, r)
    assert r.started, "the requester was never told the run began"


def test_progress_climbs_monotonically_to_every_unit_of_the_run(qapp, plate):
    """The whole point: N units in, N of N out, and never a step backwards on the way.

    The fixture's region holds 2 FOVs, so a per-FOV operator over it is 2 units — a count the
    WELL counter (1) cannot express at all.
    """
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
    """The safety property. A bar taken down only on success is a bar left sweeping over a dead
    run, which teaches the user the indicator lies."""
    import squidmip

    def _boom(*_a, **_kw):
        raise RuntimeError("no such plane")

    # The ENGINE raises, which is the real shape of a failed run: `_run_body` catches it, records
    # the outcome and emits `failed`. Breaking the worker's own run loop instead would test an
    # unhandled QThread exception, which is not a case this path claims to survive.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(squidmip, "project_plate", _boom)
        r = _Requester()
        plate.run_operator("mip", regions=[REGIONS[0]], save=False, requester=r)
        assert _drain_until(qapp, lambda: bool(r.failed or r.done), timeout=60)
    assert r.failed and not r.done, f"a failed run reported {r.done!r} instead of a failure"
    assert "no such plane" in r.failed[0][1], "the failure did not name its cause"


def test_the_asking_window_is_left_knowing_the_run_reached_its_total(qapp, plate):
    """The last unit's report and ``QThread.finished`` are two signals racing out of one thread,
    and the window was being torn out of the run by whichever won: on a fast run the bar's last
    frame was "1 of 2" and the final unit's report was dropped. Observed, not theorised.

    So the drain path reads the worker's own tally instead of waiting for the signal. This pins
    that with the signal path REMOVED, which is the only way to assert it without a race: no
    report can reach the requester here except the one the drain sends.
    """
    r = _Requester()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(V.PlateWindow, "_on_unit_progress", lambda self, report: None)
        plate.run_operator("mip", regions=[REGIONS[0]], save=False, requester=r)
        assert _drain_until(qapp, lambda: bool(r.done or r.failed), timeout=60)
    assert r.reports, "the run ended without ever telling the window how far it got"
    last = r.reports[-1]
    assert last.done == last.total == len(FOVS)


def test_the_plate_stops_talking_to_a_requester_once_its_run_has_drained(qapp, plate):
    """One run, one pair. A stale requester would take the NEXT run's progress into a window that
    did not ask for it — the same 'a result you did not ask for' rule ``deliver_result`` follows."""
    first = _Requester()
    _run_to_completion(qapp, plate, first)
    assert plate._run_requester is None
    n = len(first.reports)

    second = _Requester()
    _run_to_completion(qapp, plate, second)
    assert len(first.reports) == n, "a drained requester was still being fed"
    assert second.reports


# --------------------------------------------------------------------------------------
# The bar itself, on a real RegionViewer.
# --------------------------------------------------------------------------------------

@pytest.fixture
def region_window(qapp, napari_pane_stub, squid_dataset):
    from squidmip import open_reader
    from squidmip._region_viewer import ViewerManager

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
    """There is a real gap between the click and the first unit (metadata warm, pool priming), and
    a window that shows nothing across it is the state being complained about."""
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
    """The fallback, on the window rather than in the arithmetic: an operator that cannot know its
    total gets a sweep, not a fabricated percentage."""
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
    """Done and failed alike. The plate calls exactly one of these on success, failure and a
    STOPPED run, so there is no outcome that leaves the bar running."""
    region_window.operator_started("decon")
    region_window.operator_progress(ProgressReport("decon", 1, 27, FOV_UNIT))
    close(region_window)
    bar = region_window._op_progress
    assert bar.isHidden()
    assert (bar.minimum(), bar.maximum()) == (0, 100), "an indeterminate sweep was left behind"
