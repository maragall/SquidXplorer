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

import squidxplorer._viewer as V                                    # noqa: E402


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
    # `is not None` was here and is fully subsumed by the equality below, which fails on None
    # anyway. The claim worth adding is that a plain 2-D plane is handed back AS IS -- the same
    # object, not a copy or a re-derivation.
    got = V._full_res_plane(img, None)
    assert np.array_equal(got, img)
    assert got.shape == img.shape and got.dtype == img.dtype


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
    from squidxplorer._spots import STAGES

    assert V._spot_stages() is STAGES


# ------------------------------------------------- what napari actually hands back
#
# Every test above builds the pyramid as a plain Python list, which is what a CALLER passes IN.
# It is not what comes back out: napari wraps a multiscale layer's data in its own
# `MultiScaleData`, a Sequence that is neither a list nor a tuple and that reports level 0's
# `ndim`/`shape`/`dtype` as its own. `RegionViewer._detect_nuclei` hands the worker exactly
# `layer.data`, so that container -- not a list -- is what production has always passed here.
#
# Julio, from the running GUI on the real 10x acquisition: "multiscale data has no attribute max".
# Reproduced verbatim on `manual0` / 405 nm before the fix:
#
#     manual0/Fluorescence_405_nm_Ex: spot detection failed
#         — AttributeError: 'MultiScaleData' object has no attribute 'max'
#
# raised at `_workers._full_res_mip`'s `data.max(axis=0)`, because the pyramid check one line
# above it was `isinstance(data, (list, tuple))` and missed the container entirely.


def _pyramid_layer(levels):
    """A REAL napari layer over *levels*, so ``.data`` is whatever napari decides it is."""
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    mosaic = MosaicLayers(ViewerModel())
    mosaic.add_mosaic("raw", "405", levels, multiscale=True, bbox_um=(0.0, 0.0, 128.0, 128.0))
    return mosaic, mosaic.find("raw", "405")


def _stand_in_segmenter(monkeypatch, boxes=((2, 22), (60, 80))):
    """Replace the preferred segmenter with one that stamps KNOWN object ids.

    A stand-in, and deliberately so: the defect is in what this app does with a label image, not
    in how cellpose finds one, and real cellpose on this machine takes ~6.6 s for a 256x256 plane
    (measured) with a nondeterministic count.
    """
    import dataclasses

    from squidxplorer import _spots as SP

    def _fn(plane, params, *, on_stage=None, should_stop=None):
        if on_stage is not None:
            on_stage("stand-in", 1, 1)
        lab = np.zeros(np.asarray(plane).shape, dtype=np.int32)
        for i, (a, b) in enumerate(boxes, start=1):
            lab[a:b, a:b] = i
        return SP.result_from_labels(lab)

    name = SP.preferred_segmenter()
    monkeypatch.setitem(SP._SEGMENTERS, name,
                        dataclasses.replace(SP._SEGMENTERS[name], fn=_fn))
    return name


def test_a_napari_pyramids_own_data_reaches_the_viewer_as_a_labels_layer(qapp, monkeypatch):
    """THE cellpose crash, end to end: a real multiscale layer's ``data`` -> a Labels layer.

    Asserts the OUTCOME -- a napari Labels layer carrying the object ids the segmenter produced,
    at level-0 resolution -- rather than that any function was called. Before the fix the worker
    emitted ``problem`` instead of ``ready``, so no labels layer reached the viewer at all.

    MUTATION: put ``isinstance(data, (list, tuple))`` back in front of ``_full_res_mip``'s
    ``data.max(axis=0)`` -> AttributeError on ``MultiScaleData`` -> red.
    """
    _stand_in_segmenter(monkeypatch)
    lv0 = np.stack([_plane() for _ in range(5)])            # (z, y, x), 128x128 planes
    mosaic, layer = _pyramid_layer([lv0, lv0[:, ::2, ::2], lv0[:, ::4, ::4]])
    assert not isinstance(layer.data, (list, tuple)), (
        "napari handed the pyramid back as a list; this test no longer covers the container "
        f"production actually sees ({type(layer.data).__name__})")

    w = V._SpotWorker("manual0", "405", layer.data, None, (0.0, 0.0, 128.0, 128.0))
    rec = _run(w)

    assert rec["problem"] == [], rec["problem"]
    assert len(rec["ready"]) == 1, "no result reached the viewer"
    _region, channel, labels, _centroids, bbox, count = rec["ready"][0]
    assert labels.shape == lv0.shape[1:], "the mask is not at level-0 resolution"
    assert count == 2

    # ...and the layer the window would build out of it.
    mosaic.add_result("labels", "cellpose", channel, labels, bbox_um=bbox)
    lay = mosaic.find("cellpose", channel)
    assert type(lay).__name__ == "Labels", type(lay).__name__
    assert sorted(np.unique(np.asarray(lay.data)).tolist()) == [0, 1, 2]
    assert np.asarray(lay.data).shape == lv0.shape[1:]


def test_a_2d_napari_pyramid_is_counted_at_LEVEL_ZERO_not_the_coarsest_level(qapp, monkeypatch):
    """The silent half of the same defect, and the more dangerous one.

    A 2-D pyramid never reaches ``.max(axis=0)``, so it did not crash -- it fell through to
    ``np.asarray(data)``, and ``MultiScaleData.__array__`` returns the COARSEST level. Nuclei were
    counted on a 4x-downsampled picture, which merges touching nuclei and under-reports, with
    nothing on screen to say so.

    MUTATION: return ``np.asarray(data)`` from ``full_res_level`` for a pyramid -> the mask comes
    back 32x32 -> red.
    """
    _stand_in_segmenter(monkeypatch)
    full = _plane()                                         # (128, 128)
    mosaic, layer = _pyramid_layer([full, full[::2, ::2], full[::4, ::4]])

    w = V._SpotWorker("manual0", "405", layer.data, None, None)
    rec = _run(w)

    assert rec["problem"] == [], rec["problem"]
    labels = rec["ready"][0][2]
    assert labels.shape == full.shape, (
        f"segmented a downsampled pyramid level: {labels.shape} against level 0 {full.shape}")
