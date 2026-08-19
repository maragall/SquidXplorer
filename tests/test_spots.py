"""Spot detection (nuclei counting): the pure operator and its engine registration."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._spots import (
    DEFAULT_PARAMS,
    LAYER_KEY,
    SpotParams,
    SpotResult,
    detect_spots,
    spots_op,
)


def _blank(shape=(128, 128), dtype=np.uint16) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def _disk(img: np.ndarray, cy: int, cx: int, radius: int, value: int = 3000) -> np.ndarray:
    """Stamp a filled disk: a crude nucleus with hard edges, which Otsu separates cleanly."""
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


def test_it_counts_well_separated_nuclei():
    res = detect_spots(_plane_with_disks(_FOUR))
    assert isinstance(res, SpotResult)
    assert res.count == 4, f"expected 4 nuclei, got {res.count}"


def test_the_count_is_the_number_of_distinct_labels_not_a_second_tally():
    """count / labels / centroids are one truth in three shapes, never hand-synced."""
    res = detect_spots(_plane_with_disks(_FOUR))
    assert int(res.labels.max()) == res.count
    assert len(np.unique(res.labels)) == res.count + 1        # + background
    assert res.centroids.shape == (res.count, 2)


def test_a_blank_plane_counts_zero_instead_of_raising():
    """An empty well is a legitimate result, not an error."""
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
    """Bright squares of exact pixel areas, laid out so none of them touch."""
    img = np.zeros(shape, dtype=np.uint16)
    for i, s in enumerate(sizes):
        y, x = 10 + (i // 4) * 30, 10 + (i % 4) * 30
        img[y: y + s, x: x + s] = 3000
    return img


_AREA_PARAMS = SpotParams(sigma_px=0.5, min_area_px=36, split_touching=False)


def test_objects_smaller_than_min_area_are_dropped_and_larger_ones_are_kept():
    """Pins the actual job of ``remove_small_objects`` on exact areas (the speck test above
    passes even with that call deleted, since gaussian denoise erases lone pixels first)."""
    img = _squares([4, 5, 6, 7, 8])                            # areas 16, 25, 36, 49, 64
    res = detect_spots(img, _AREA_PARAMS)                      # min_area_px = 36
    assert res.count == 3, f"expected the 36/49/64 px objects only, got {res.count}"


def test_an_object_of_EXACTLY_min_area_is_kept_not_dropped():
    """skimage 0.26 renamed min_size -> max_size and flipped the comparison, so passing
    max_size=min_area_px instead of min_area_px - 1 would silently drop the boundary case."""
    img = _squares([6])                                        # area 36 == min_area_px
    assert detect_spots(img, _AREA_PARAMS).count == 1

    img = _squares([5])                                        # area 25 < min_area_px
    assert detect_spots(img, _AREA_PARAMS).count == 0


def test_min_area_is_honoured_so_the_parameter_is_not_decorative():
    img = _plane_with_disks(_FOUR, radius=6)                   # ~113 px each
    strict = detect_spots(img, SpotParams(min_area_px=5000))
    assert strict.count == 0


def test_touching_nuclei_are_split_rather_than_fused_into_one_blob():
    """Without the watershed step, two overlapping disks label as one."""
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
    res = detect_spots(_plane_with_disks(_FOUR))
    found = {(round(r / 10) * 10, round(c / 10) * 10) for r, c in res.centroids}
    assert found == {(30, 30), (30, 90), (90, 30), (90, 90)}, found


def test_centroids_are_row_col_which_is_napari_world_order():
    """napari's 2D world axes are (row, col); a transposed Points layer looks plausible and is wrong."""
    img = _blank()
    _disk(img, 20, 100, 6)                                     # far from the diagonal
    (row, col), = detect_spots(img).centroids
    assert 15 < row < 25 and 95 < col < 105, (row, col)


def test_the_source_plane_is_never_modified():
    img = _plane_with_disks(_FOUR)
    before = img.copy()
    detect_spots(img)
    assert np.array_equal(img, before)


def test_a_non_2d_plane_fails_loud_and_names_the_shape():
    with pytest.raises(ValueError, match=r"2-D"):
        detect_spots(np.zeros((3, 16, 16), dtype=np.uint16))


def test_a_negative_min_area_is_refused_rather_than_silently_clamped():
    with pytest.raises(ValueError, match="min_area_px"):
        SpotParams(min_area_px=-1).validate()


def test_a_non_positive_sigma_is_refused():
    with pytest.raises(ValueError, match="sigma_px"):
        SpotParams(sigma_px=0.0).validate()


def test_spot_detection_is_a_peer_of_mip_in_the_ENGINE_registry():
    """Not a special case: it is in the same table mip/bgsub/decon are in."""
    from squidxplorer import available_plane_operators

    assert LAYER_KEY in available_plane_operators()


def test_it_declares_that_it_does_NOT_consume_z():
    """Segmentation is a per-plane map: every z is segmented, z survives at full depth.
    Declaring {"z"} here would silently throw away every plane but one's worth of cells."""
    from squidxplorer import operator_consumes

    assert operator_consumes(LAYER_KEY) == frozenset()


def test_the_registered_operator_returns_a_label_image_of_the_input_shape_and_dtype():
    img = _plane_with_disks(_FOUR)
    out = spots_op()([img])
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert int(out.max()) == 4


def test_more_nuclei_than_the_container_dtype_can_hold_fails_loud():
    """uint8 tops out at 255 labels; truncating into it would report a wrong cell count and look fine."""
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
    from squidxplorer import add_operator

    with pytest.raises(ValueError, match="already defined"):
        add_operator(LAYER_KEY, spots_op())


def test_the_plane_op_contract_refuses_a_whole_z_stack():
    img = _plane_with_disks(_FOUR)
    with pytest.raises(ValueError, match="more than one plane"):
        spots_op()([img, img])


def test_the_defaults_are_valid():
    DEFAULT_PARAMS.validate()


def test_every_stage_announces_itself_so_the_indicator_has_something_to_say():
    from squidxplorer._spots import STAGES

    seen = []
    detect_spots(_plane_with_disks(_FOUR), on_stage=lambda n, d, t: seen.append((n, d, t)))

    assert [n for n, _d, _t in seen] == list(STAGES)
    assert [d for _n, d, _t in seen] == list(range(len(STAGES)))
    assert {t for _n, _d, t in seen} == {len(STAGES)}


def test_the_progress_total_is_derived_from_the_stage_list_not_hardcoded_twice():
    from squidxplorer._spots import STAGES

    totals = []
    detect_spots(_plane_with_disks(_FOUR), on_stage=lambda n, d, t: totals.append(t))
    assert set(totals) == {len(STAGES)}, "the denominator drifted from STAGES"


def test_a_cancel_raises_instead_of_returning_a_half_finished_answer():
    from squidxplorer._spots import SpotDetectionCancelled

    with pytest.raises(SpotDetectionCancelled):
        detect_spots(_plane_with_disks(_FOUR), should_stop=lambda: True)


def test_cancel_is_checked_at_every_stage_not_only_the_first():
    from squidxplorer._spots import SpotDetectionCancelled

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


def test_the_operator_is_named_for_what_it_produces_not_the_algorithm():
    """The operator is named 'spot' (what it produces), so a sibling algorithm registers
    beside it under its own name without renaming this one."""
    assert LAYER_KEY == "spot"
    assert "skimage" not in LAYER_KEY and "otsu" not in LAYER_KEY


def test_a_new_algorithm_is_one_function_and_the_plumbing_does_not_change():
    """The Cellpose drop-in, rehearsed with a stub: a function that returns a label image gets
    everything downstream — count, centroids, layers, readout — for free."""
    from squidxplorer._spots import result_from_labels

    def fake_cellpose(plane, params, *, on_stage=None, should_stop=None):
        labels = np.zeros(plane.shape, dtype=np.int32)
        labels[2:6, 2:6] = 1
        labels[10:14, 10:14] = 2
        return result_from_labels(labels)

    res = detect_spots(_blank(), segment=fake_cellpose)
    assert res.count == 2
    assert res.centroids.shape == (2, 2)
    assert int(res.labels.max()) == 2


def test_result_from_labels_gives_every_algorithm_the_same_counting_semantics():
    """Cellpose returns a label array with arbitrary, non-sequential ids; the shared helper is
    what stops two algorithms disagreeing about what 'how many' means."""
    from squidxplorer._spots import result_from_labels

    labels = np.zeros((32, 32), dtype=np.int32)
    labels[2:6, 2:6] = 7                                       # gappy, non-sequential ids…
    labels[10:14, 10:14] = 900
    res = result_from_labels(labels)

    assert res.count == 2
    assert sorted(np.unique(res.labels)) == [0, 1, 2]          # …relabelled 1..n
    assert len(res.centroids) == 2


def test_a_slow_algorithm_can_still_be_cancelled_and_report_progress():
    """The cancel/progress seam is the algorithm's to honour, part of the callable's signature,
    not something the fast one gets away with ignoring."""
    from squidxplorer._spots import SpotDetectionCancelled

    def slow(plane, params, *, on_stage=None, should_stop=None):
        if on_stage is not None:
            on_stage("running the model", 0, 1)
        if should_stop is not None and should_stop():
            raise SpotDetectionCancelled("cancelled during the model")
        raise AssertionError("should have been cancelled")

    seen = []
    with pytest.raises(SpotDetectionCancelled):
        detect_spots(_blank(), segment=slow,
                     on_stage=lambda *a: seen.append(a), should_stop=lambda: True)
    assert seen == [("running the model", 0, 1)]


def test_a_declared_parameter_no_SpotParams_field_backs_is_refused_at_registration():
    from squidxplorer._engine import Param
    from squidxplorer._spots import add_segmentation_operator

    with pytest.raises(ValueError, match="diameter"):
        add_segmentation_operator("bad-seg", lambda *a, **k: None,
                                  params=(Param("diameter", 30),))
