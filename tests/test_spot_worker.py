"""The GUI seam of spot detection: the QThread worker and the plane it counts on.

Kept out of ``test_viewer.py`` because none of this needs a PlateWindow — the worker is a
standalone QThread and ``_full_res_plane`` is a pure function. Qt is needed only for the signal
machinery, and no window is ever shown.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")

from qtpy.QtCore import QCoreApplication                       # noqa: E402
from qtpy.QtWidgets import QApplication                        # noqa: E402

import squidmip._viewer as V                                    # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _plane(n=4, shape=(128, 128)):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 120, shape, dtype=np.uint16)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for i, (cy, cx) in enumerate([(30, 30), (30, 90), (90, 30), (90, 90)][:n]):
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 36] = 3000
    return img


def _run(worker, timeout_ms=60000):
    """Start the worker and pump the event loop until it finishes. Returns the recorded signals."""
    rec = {"ready": [], "problem": [], "cancelled": 0, "progress": [], "stage": [], "done": []}
    worker.ready.connect(lambda *a: rec["ready"].append(a))
    worker.problem.connect(lambda m: rec["problem"].append(m))
    worker.cancelled.connect(lambda: rec.__setitem__("cancelled", rec["cancelled"] + 1))
    worker.progress.connect(lambda d, t: rec["progress"].append((d, t)))
    worker.stageChanged.connect(lambda s: rec["stage"].append(s))
    worker.finished_count.connect(lambda *a: rec["done"].append(a))

    worker.start()
    waited = 0
    while not worker.isFinished() and waited < timeout_ms:
        QCoreApplication.processEvents()
        worker.wait(10)
        waited += 10
    QCoreApplication.processEvents()
    assert worker.isFinished(), "the worker never finished"
    return rec


# ---------------------------------------------------------------- which plane gets counted


def test_a_plain_2d_plane_is_counted_as_is():
    img = _plane()
    assert V._full_res_plane(img, None) is not None
    assert np.array_equal(V._full_res_plane(img, None), img)


def test_a_z_stack_is_counted_at_the_z_napari_is_SHOWING():
    """napari owns the z slider. Counting a different plane than the one on screen puts a number
    in the readout that does not describe the picture."""
    stack = np.stack([np.full((8, 8), i, dtype=np.uint16) for i in range(5)])
    assert V._full_res_plane(stack, 3)[0, 0] == 3
    assert V._full_res_plane(stack, 0)[0, 0] == 0


def test_no_z_index_falls_back_to_the_middle_plane_not_plane_zero():
    stack = np.stack([np.full((8, 8), i, dtype=np.uint16) for i in range(5)])
    assert V._full_res_plane(stack, None)[0, 0] == 2


def test_an_out_of_range_z_clamps_instead_of_raising_IndexError():
    stack = np.stack([np.full((8, 8), i, dtype=np.uint16) for i in range(5)])
    assert V._full_res_plane(stack, 99)[0, 0] == 4
    assert V._full_res_plane(stack, -7)[0, 0] == 0


def test_a_MULTISCALE_pyramid_is_counted_at_LEVEL_ZERO_not_a_downsampled_level():
    """``_MosaicWorker.ready`` now hands the pane a LEVEL LIST. Counting level 1 would merge
    touching nuclei and silently under-report — a wrong number that looks entirely plausible."""
    full = _plane()
    levels = [full, full[::2, ::2], full[::4, ::4]]
    got = V._full_res_plane(levels, None)

    assert got.shape == full.shape, "a downsampled pyramid level was counted"
    assert np.array_equal(got, full)


def test_a_multiscale_z_stack_takes_level_zero_AND_the_shown_z():
    lv0 = np.stack([np.full((8, 8), i, dtype=np.uint16) for i in range(5)])
    got = V._full_res_plane([lv0, lv0[:, ::2, ::2]], 3)
    assert got.shape == (8, 8)
    assert got[0, 0] == 3


def test_an_empty_pyramid_says_so_instead_of_raising_IndexError():
    with pytest.raises(ValueError, match="EMPTY multiscale"):
        V._full_res_plane([], None)


def test_something_that_is_neither_names_what_it_got():
    with pytest.raises(ValueError, match=r"neither a pyramid level list"):
        V._full_res_plane(np.zeros((2, 3, 4, 5), dtype=np.uint16), None)


# ---------------------------------------------------------------- the worker's signals


def test_a_successful_run_emits_the_result_and_the_count(qapp):
    w = V._SpotWorker("B3", "405", _plane(), None, (0.0, 0.0, 128.0, 128.0))
    rec = _run(w)

    assert rec["problem"] == [], rec["problem"]
    assert rec["cancelled"] == 0
    assert len(rec["ready"]) == 1

    region, channel, labels, centroids, bbox, count = rec["ready"][0]
    assert (region, channel) == ("B3", "405")
    assert count == 4
    assert labels.shape == (128, 128)
    assert centroids.shape == (4, 2)
    assert bbox == (0.0, 0.0, 128.0, 128.0)
    assert rec["done"] == [("B3", "405", 4)]


def test_the_count_in_ready_and_in_finished_count_are_the_same_number(qapp):
    """Two signals carrying one number is exactly the shape that drifts. Pin it."""
    w = V._SpotWorker("B3", "405", _plane(n=3), None, None)
    rec = _run(w)
    assert rec["ready"][0][5] == rec["done"][0][2]


def test_progress_counts_stages_and_ends_at_the_total(qapp):
    """The busy indicator binds to progress(done, total). It must reach total, or a progress bar
    wired to it sits at 90% forever after a successful run.

    The denominator must also be ONE number for the whole run. It was not: the closing emit was
    hardcoded to ``len(_spot_stages())`` (the 7-stage otsu-watershed recipe) whatever algorithm had
    actually run, so under Cellpose (the preferred segmenter, which reports a total of 1) the bar
    filled at 1/1 and then jumped backwards to 7/7. Fixed in ``_SpotWorker.run``, which now closes
    on the total the running algorithm reported.

    MUTATION: emit ``len(_spot_stages())`` as the final total again -> two totals -> red.
    """
    w = V._SpotWorker("B3", "405", _plane(), None, None)
    rec = _run(w)

    assert rec["progress"], "no progress was ever emitted"
    totals = {t for _d, t in rec["progress"]}
    assert len(totals) == 1, f"the denominator changed mid-run: {totals}"
    total = totals.pop()
    assert rec["progress"][-1] == (total, total)


def test_the_stage_TEXT_goes_out_on_its_own_signal_because_progress_has_no_text_channel(qapp):
    """progress is Signal(int, int) — it cannot carry a label, and overloading an int with
    an enum would be a second representation of the stage list.

    This asserts the CHANNEL, not the labels. It used to assert the emitted text equalled
    ``_spots.STAGES`` verbatim, which pinned the otsu-watershed recipe rather than the mechanism
    the test is named for: with Cellpose preferred, ``['running cellpose', 'done']`` arrived on
    ``stageChanged`` exactly as designed and the test still went red. The labels belong to whichever
    segmenter ran; what must hold for every one of them is that the text has its own signal, that
    every progress tick is accompanied by one, and that the run closes with "done".

    MUTATION: smuggle the label into progress and drop stageChanged -> red.
    """
    w = V._SpotWorker("B3", "405", _plane(), None, None)
    rec = _run(w)

    assert "QString" not in w.progress.signal, (
        f"progress grew a text channel: {w.progress.signal}")
    assert "QString" in w.stageChanged.signal, w.stageChanged.signal
    assert rec["stage"], "no stage text was emitted at all"
    assert all(isinstance(s, str) and s for s in rec["stage"]), rec["stage"]
    assert len(rec["stage"]) == len(rec["progress"]), (
        f"a progress tick went out with no label: {rec['stage']} vs {rec['progress']}")
    assert rec["stage"][-1] == "done"


def test_a_cancelled_run_emits_cancelled_and_NO_result(qapp):
    """A half-finished mask presented as an answer is the silent failure this project bans."""
    w = V._SpotWorker("B3", "405", _plane(), None, None)
    w.stop()                                          # cancel before it starts
    rec = _run(w)

    assert rec["cancelled"] == 1
    assert rec["ready"] == []
    assert rec["done"] == []
    assert rec["problem"] == []


def test_a_failure_is_reported_BY_NAME_and_never_swallowed(qapp):
    """A region that cannot be segmented must SAY SO by name."""
    w = V._SpotWorker("B3", "405", "not an image at all", None, None)
    rec = _run(w)

    assert len(rec["problem"]) == 1
    msg = rec["problem"][0]
    assert msg.startswith("B3/405:"), msg
    assert "spot detection failed" in msg
    assert rec["ready"] == []
    assert rec["done"] == []


def test_a_blank_region_reports_zero_rather_than_failing(qapp):
    """Zero nuclei is an ANSWER. It must reach the readout, not the error path."""
    w = V._SpotWorker("B3", "405", np.zeros((64, 64), dtype=np.uint16), None, None)
    rec = _run(w)

    assert rec["problem"] == []
    assert rec["done"] == [("B3", "405", 0)]
    assert rec["ready"][0][5] == 0


def test_the_worker_declares_a_stop_so_teardown_can_retire_it(qapp):
    """``PlateWindow._retire`` calls ``stop()`` on a running worker; a worker without one aborts
    the app at teardown."""
    assert callable(V._SpotWorker.stop)


def test_the_stage_denominator_is_not_a_second_copy_of_the_stage_list():
    from squidmip._spots import STAGES

    assert V._spot_stages() is STAGES
