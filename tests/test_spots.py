"""Spot detection (nuclei counting) — the pure operator and its engine registration.

These tests are Qt-free and napari-free: the algorithm is a plain array -> array function, and
that is deliberate (the same property that lets ``_background``/``_decon`` be tested headless).
The napari layer seam is covered in ``test_napari_view.py``; the worker/thread seam in
``test_viewer.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip._spots import (
    DEFAULT_PARAMS,
    LAYER_KEY,
    SpotParams,
    SpotResult,
    centroid_layer_name,
    detect_spots,
    mask_layer_name,
    spots_op,
)


# ---------------------------------------------------------------- synthetic fixtures


def _blank(shape=(128, 128), dtype=np.uint16) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def _disk(img: np.ndarray, cy: int, cx: int, radius: int, value: int = 3000) -> np.ndarray:
    """Stamp a filled disk — a crude nucleus with hard edges, which Otsu separates cleanly."""
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = value
    return img


def _plane_with_disks(centres, radius=6, shape=(128, 128), noise=120, seed=0) -> np.ndarray:
    """A plane of well-separated disks on a dim noisy background."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, noise, shape, dtype=np.uint16)
    for cy, cx in centres:
        _disk(img, cy, cx, radius)
    return img


_FOUR = [(30, 30), (30, 90), (90, 30), (90, 90)]


# ---------------------------------------------------------------- counting


def test_it_counts_well_separated_nuclei():
    """The whole point of the operator: a number that matches what a human would count."""
    res = detect_spots(_plane_with_disks(_FOUR))
    assert isinstance(res, SpotResult)
    assert res.count == 4, f"expected 4 nuclei, got {res.count}"


def test_the_count_is_the_number_of_distinct_labels_not_a_second_tally():
    """count / labels / centroids are ONE truth in three shapes, never hand-synced.

    Two representations of one number that can disagree is this codebase's most-repeated defect
    shape; here the invariant is asserted on data.
    """
    res = detect_spots(_plane_with_disks(_FOUR))
    assert int(res.labels.max()) == res.count
    assert len(np.unique(res.labels)) == res.count + 1        # + background
    assert res.centroids.shape == (res.count, 2)


def test_a_blank_plane_counts_zero_instead_of_raising():
    """An empty well is a legitimate result, not an error. It must not blow up the run."""
    res = detect_spots(_blank())
    assert res.count == 0
    assert res.centroids.shape == (0, 2)
    assert res.labels.shape == (128, 128)
    assert int(res.labels.max()) == 0


def test_single_hot_pixels_are_not_counted_as_nuclei():
    """Sensor specks are noise, not cells."""
    img = _plane_with_disks(_FOUR)
    for cy, cx in [(10, 60), (60, 10), (118, 60)]:
        img[cy, cx] = 4000                                    # 1-px specks
    res = detect_spots(img)
    assert res.count == 4, f"specks leaked into the count: {res.count}"


def _squares(sizes, shape=(128, 128)):
    """Bright squares of EXACT pixel areas, laid out so none of them touch.

    ``sigma_px=0.5`` is small enough that Otsu recovers each square's area exactly (measured:
    a 4x4 thresholds to 16 px, 6x6 to 36 px), which is what makes an area-boundary test
    meaningful instead of approximate.
    """
    img = np.zeros(shape, dtype=np.uint16)
    for i, s in enumerate(sizes):
        y, x = 10 + (i // 4) * 30, 10 + (i % 4) * 30
        img[y: y + s, x: x + s] = 3000
    return img


_AREA_PARAMS = SpotParams(sigma_px=0.5, min_area_px=36, split_touching=False)


def test_objects_smaller_than_min_area_are_dropped_and_larger_ones_are_kept():
    """The actual job of ``remove_small_objects``, pinned on exact areas.

    Written this way after a mutation check: the 1-px-speck test above passes even with the
    ``remove_small_objects`` call DELETED, because the gaussian denoise erases a lone pixel
    before the threshold ever sees it. It was testing the smoothing, not the filter.
    """
    img = _squares([4, 5, 6, 7, 8])                            # areas 16, 25, 36, 49, 64
    res = detect_spots(img, _AREA_PARAMS)                      # min_area_px = 36
    assert res.count == 3, f"expected the 36/49/64 px objects only, got {res.count}"


def test_an_object_of_EXACTLY_min_area_is_kept_not_dropped():
    """``min_area_px`` means "smaller than this is noise", so exactly-this-size is a cell.

    skimage 0.26 renamed ``min_size`` -> ``max_size`` AND flipped the comparison to "smaller
    than or equal", so passing ``max_size=min_area_px`` instead of ``min_area_px - 1`` silently
    deletes every cell of exactly the minimum size. This is the test that catches that.
    """
    img = _squares([6])                                        # area 36 == min_area_px
    assert detect_spots(img, _AREA_PARAMS).count == 1

    img = _squares([5])                                        # area 25 < min_area_px
    assert detect_spots(img, _AREA_PARAMS).count == 0


def test_min_area_is_honoured_so_the_parameter_is_not_decorative():
    """Raise min_area above the disk area and the disks must stop counting."""
    img = _plane_with_disks(_FOUR, radius=6)                   # ~113 px each
    strict = detect_spots(img, SpotParams(min_area_px=5000))
    assert strict.count == 0


def test_touching_nuclei_are_split_rather_than_fused_into_one_blob():
    """Two overlapping disks are two cells. Without the watershed step they label as one."""
    img = _blank()
    _disk(img, 64, 58, 12)
    _disk(img, 64, 74, 12)                                     # overlapping -> one component
    assert detect_spots(img).count == 2

    fused = detect_spots(img, SpotParams(split_touching=False))
    assert fused.count == 1, (
        "the fixture no longer produces a single fused component, so the split test proves "
        "nothing — fix the fixture, not the assertion"
    )


def test_centroids_land_inside_the_nuclei_they_describe():
    """A Points layer whose points are not on the cells is a plausible-looking lie."""
    res = detect_spots(_plane_with_disks(_FOUR))
    found = {(round(r / 10) * 10, round(c / 10) * 10) for r, c in res.centroids}
    assert found == {(30, 30), (30, 90), (90, 30), (90, 90)}, found


def test_centroids_are_row_col_which_is_napari_world_order():
    """napari's 2D world axes are (row, col). A transposed Points layer looks plausible and is
    wrong — the same silent-transpose class ``scale_translate_from_bbox_um`` exists to prevent."""
    img = _blank()
    _disk(img, 20, 100, 6)                                     # far from the diagonal
    (row, col), = detect_spots(img).centroids
    assert 15 < row < 25 and 95 < col < 105, (row, col)


# ---------------------------------------------------------------- the read-only contract


def test_the_source_plane_is_never_modified():
    """DATASETS ARE READ-ONLY. The operator is a pure function of the plane it was handed."""
    img = _plane_with_disks(_FOUR)
    before = img.copy()
    detect_spots(img)
    assert np.array_equal(img, before)


# ---------------------------------------------------------------- loud failure


def test_a_non_2d_plane_fails_loud_and_names_the_shape():
    with pytest.raises(ValueError, match=r"2-D"):
        detect_spots(np.zeros((3, 16, 16), dtype=np.uint16))


def test_a_negative_min_area_is_refused_rather_than_silently_clamped():
    with pytest.raises(ValueError, match="min_area_px"):
        SpotParams(min_area_px=-1).validate()


def test_a_non_positive_sigma_is_refused():
    with pytest.raises(ValueError, match="sigma_px"):
        SpotParams(sigma_px=0.0).validate()


# ---------------------------------------------------------------- the engine registration


def test_spot_detection_is_a_peer_of_mip_in_the_ENGINE_registry():
    """Not a special case: it is in the same table mip/bgsub/decon are in (IMA-210's seam)."""
    from squidmip import available_projectors

    assert LAYER_KEY in available_projectors()


def test_it_declares_that_it_does_NOT_consume_z():
    """Segmentation is a per-plane MAP: every z is segmented, z survives at full depth.

    Collapsing z is ``mip``'s job, and 'mip then spots' is the chained-task model Fractal and
    CellProfiler both use. Declaring ``{"z"}`` here would silently throw away every plane but
    one's worth of cells.
    """
    from squidmip import projector_consumes

    assert projector_consumes(LAYER_KEY) == frozenset()


def test_the_registered_operator_returns_a_label_image_of_the_input_shape_and_dtype():
    """``project_well`` writes the result into a native-dtype ``(T,C,Z,Y,X)`` buffer, so the
    plane-op must hand back something that fits it without a lossy surprise."""
    img = _plane_with_disks(_FOUR)
    out = spots_op()([img])
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert int(out.max()) == 4


def test_more_nuclei_than_the_container_dtype_can_hold_fails_loud():
    """uint8 tops out at 255 labels. Truncating into it would report a WRONG cell count and look
    fine — exactly the silent failure this project bans. It must raise and say the number."""
    img = np.zeros((256, 256), dtype=np.uint8)
    ys, xs = np.mgrid[2:256:8, 2:256:8]                       # 1024 dots, well over 255
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            img[ys + dy, xs + dx] = 200                       # 3x3 blocks, so min_area passes
    params = SpotParams(sigma_px=0.5, min_area_px=2)

    assert detect_spots(img, params).count == 1024             # the pure function is honest…

    with pytest.raises(ValueError, match=r"uint8"):            # …and the engine adapter refuses
        spots_op(params)([img])


def test_registering_it_twice_is_refused_so_a_reimport_cannot_clobber_the_table():
    from squidmip import add_projector

    with pytest.raises(ValueError, match="already defined"):
        add_projector(LAYER_KEY, spots_op())


def test_the_plane_op_contract_refuses_a_whole_z_stack():
    """If it were ever registered as a z-reducer it would be handed the stack; that must raise
    rather than quietly segment plane 0 and call it the well's answer."""
    img = _plane_with_disks(_FOUR)
    with pytest.raises(ValueError, match="more than one plane"):
        spots_op()([img, img])


# ---------------------------------------------------------------- layer naming, one spelling


def test_the_layer_names_come_from_one_place_so_the_ui_and_the_result_cannot_drift():
    assert mask_layer_name("405") != centroid_layer_name("405")
    assert "405" in mask_layer_name("405")
    assert "405" in centroid_layer_name("405")


def test_the_defaults_are_valid():
    DEFAULT_PARAMS.validate()


# ---------------------------------------------------------------- progress + cancellation
# The Qt-free half of "responsiveness is important. And an indicator when its working."
# The worker turns these two callbacks into signals; the algorithm itself stays Qt-free.


def test_every_stage_announces_itself_so_the_indicator_has_something_to_say():
    from squidmip._spots import STAGES

    seen = []
    detect_spots(_plane_with_disks(_FOUR), on_stage=lambda n, d, t: seen.append((n, d, t)))

    assert [n for n, _d, _t in seen] == list(STAGES)
    assert [d for _n, d, _t in seen] == list(range(len(STAGES)))
    assert {t for _n, _d, t in seen} == {len(STAGES)}


def test_the_progress_total_is_derived_from_the_stage_list_not_hardcoded_twice():
    """Two representations of one number that can disagree is this codebase's defect shape."""
    from squidmip._spots import STAGES

    totals = []
    detect_spots(_plane_with_disks(_FOUR), on_stage=lambda n, d, t: totals.append(t))
    assert set(totals) == {len(STAGES)}, "the denominator drifted from STAGES"


def test_a_cancel_raises_instead_of_returning_a_half_finished_answer():
    """A partial segmentation that looks finished would report a WRONG count and look fine."""
    from squidmip._spots import SpotDetectionCancelled

    with pytest.raises(SpotDetectionCancelled):
        detect_spots(_plane_with_disks(_FOUR), should_stop=lambda: True)


def test_cancel_is_checked_at_every_stage_not_only_the_first():
    """Cancelling after the run has started must still take effect."""
    from squidmip._spots import SpotDetectionCancelled

    calls = {"n": 0}

    def stop_on_the_third_stage():
        calls["n"] += 1
        return calls["n"] >= 3

    with pytest.raises(SpotDetectionCancelled):
        detect_spots(_plane_with_disks(_FOUR), should_stop=stop_on_the_third_stage)
    assert calls["n"] == 3


def test_a_run_that_is_not_cancelled_is_unaffected_by_the_seam():
    plain = detect_spots(_plane_with_disks(_FOUR))
    watched = detect_spots(_plane_with_disks(_FOUR),
                           on_stage=lambda *a: None, should_stop=lambda: False)
    assert watched.count == plain.count
    assert np.array_equal(watched.labels, plain.labels)


# ---------------------------------------------------------------- the pluggable segmenter
# Julio: "we can import some nice cell detection algos. Like good ones, you know, shouldn't be
# complex if the GUI is well designed". These tests pin the seam that has to hold when Cellpose
# or StarDist lands: same signature, same result contract, same registry, no redesign.


def test_the_default_segmenter_is_registered_under_an_algorithm_neutral_operator_name():
    """The operator is called 'spot' — what it PRODUCES. 'otsu-watershed' is the algorithm, and
    it lives in a separate table so a sibling can replace it without renaming the operator."""
    from squidmip._spots import DEFAULT_SEGMENTER, available_segmenters

    assert LAYER_KEY == "spot"
    assert DEFAULT_SEGMENTER in available_segmenters()
    assert "skimage" not in LAYER_KEY and "otsu" not in LAYER_KEY


def test_a_new_segmenter_is_one_call_and_the_operator_does_not_change():
    """This is the Cellpose drop-in, rehearsed with a stub: register a function that returns a
    LABEL IMAGE and everything downstream — count, centroids, layers, readout — already works."""
    from squidmip._spots import add_segmenter, result_from_labels

    def fake_cellpose(plane, params, *, on_stage=None, should_stop=None):
        labels = np.zeros(plane.shape, dtype=np.int32)
        labels[2:6, 2:6] = 1
        labels[10:14, 10:14] = 2
        return result_from_labels(labels)

    name = "fake-cellpose-for-this-test"
    add_segmenter(name, fake_cellpose, blurb="stub")
    try:
        res = detect_spots(_blank(), algorithm=name)
        assert res.count == 2
        assert res.centroids.shape == (2, 2)
        assert int(res.labels.max()) == 2
    finally:
        from squidmip._spots import _SEGMENTERS

        del _SEGMENTERS[name]


def test_result_from_labels_gives_every_segmenter_the_same_counting_semantics():
    """Cellpose returns a label array with arbitrary, non-sequential ids. The shared helper is
    what stops two segmenters disagreeing about what 'how many' means."""
    from squidmip._spots import result_from_labels

    labels = np.zeros((32, 32), dtype=np.int32)
    labels[2:6, 2:6] = 7                                       # gappy, non-sequential ids…
    labels[10:14, 10:14] = 900
    res = result_from_labels(labels)

    assert res.count == 2
    assert sorted(np.unique(res.labels)) == [0, 1, 2]          # …relabelled 1..n
    assert len(res.centroids) == 2


def test_an_uninstalled_segmenter_is_a_NAMED_refusal_not_a_silent_absence():
    """A missing optional dep must not make the operator quietly vanish from the list — that is
    indistinguishable from nobody having written it."""
    from squidmip._spots import (
        MissingSegmenterDependency,
        _SEGMENTERS,
        add_segmenter,
        available_segmenters,
        segmenter_available,
    )

    name = "needs-a-package-that-is-not-here"
    add_segmenter(name, lambda *a, **k: None, requires=("definitely_not_installed_xyz",))
    try:
        assert name in available_segmenters(), "it vanished from the list instead of refusing"

        ok, why = segmenter_available(name)
        assert ok is False
        assert "definitely_not_installed_xyz" in why

        with pytest.raises(MissingSegmenterDependency, match="definitely_not_installed_xyz"):
            detect_spots(_blank(), algorithm=name)
    finally:
        del _SEGMENTERS[name]


def test_an_unknown_segmenter_names_the_ones_that_do_exist():
    with pytest.raises(KeyError, match="otsu-watershed"):
        detect_spots(_blank(), algorithm="no-such-algorithm")


def test_registering_a_segmenter_twice_is_refused():
    from squidmip._spots import DEFAULT_SEGMENTER, add_segmenter, skimage_watershed

    with pytest.raises(ValueError, match="already defined"):
        add_segmenter(DEFAULT_SEGMENTER, skimage_watershed)


def test_a_slow_segmenter_can_still_be_cancelled_and_report_progress():
    """Cellpose on a mosaic is seconds to minutes. The cancel/progress seam is the SEGMENTER's
    to honour, so it is part of the registered signature, not something the fast one gets away
    with ignoring."""
    from squidmip._spots import SpotDetectionCancelled, _SEGMENTERS, add_segmenter

    def slow(plane, params, *, on_stage=None, should_stop=None):
        if on_stage is not None:
            on_stage("running the model", 0, 1)
        if should_stop is not None and should_stop():
            raise SpotDetectionCancelled("cancelled during the model")
        raise AssertionError("should have been cancelled")

    name = "slow-stub"
    add_segmenter(name, slow)
    try:
        seen = []
        with pytest.raises(SpotDetectionCancelled):
            detect_spots(_blank(), algorithm=name,
                         on_stage=lambda *a: seen.append(a), should_stop=lambda: True)
        assert seen == [("running the model", 0, 1)]
    finally:
        del _SEGMENTERS[name]


def test_the_engine_operator_resolves_its_segmenter_lazily_not_at_import():
    """Registering a Cellpose operator must not import cellpose (or claim a GPU) at
    ``import squidmip`` time. Building the op is free; only running it resolves."""
    from squidmip._spots import MissingSegmenterDependency, _SEGMENTERS, add_segmenter

    name = "lazy-stub"
    add_segmenter(name, lambda *a, **k: None, requires=("definitely_not_installed_xyz",))
    try:
        op = spots_op(algorithm=name)                # must NOT raise
        with pytest.raises(MissingSegmenterDependency):
            op([_blank()])                           # …only running it does
    finally:
        del _SEGMENTERS[name]
