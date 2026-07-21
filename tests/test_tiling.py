"""IMA-216 viewport tiler — LOD pick, frustum cull, byte-budget LRU, parent pinning.

Pure numpy: every geometry here is fabricated, so these tests need no dataset and no Qt.
Wall-clock assertions live only under the ``integration`` marker (precedent:
``test_performance.py``) — CI asserts correctness, never speed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from squidmip._tiling import (
    Geometry,
    Level,
    TileCache,
    TileDescriptor,
    select_tiles,
    viewport,
)

# ---------------------------------------------------------------------------------------
# fabricated geometry: a 20x20 grid of 100 µm FOVs (2000 µm plate) with a coarse ladder
# above it — level 0 = per-FOV, level 1 = 2x2 groups, level 2 = the whole plate as one tile.
# ---------------------------------------------------------------------------------------

def _grid(n: int, size: float, level_tag: str) -> tuple[np.ndarray, list[str]]:
    x0 = np.repeat(np.arange(n, dtype=np.float64), n) * size
    y0 = np.tile(np.arange(n, dtype=np.float64), n) * size
    bboxes = np.stack([x0, y0, x0 + size, y0 + size], axis=1)
    keys = [f"{level_tag}:{int(a)},{int(b)}" for a, b in zip(x0 // size, y0 // size)]
    return bboxes, keys


def _ladder() -> Geometry:
    fine, fine_keys = _grid(20, 100.0, "L0")      # 400 FOVs @ 0.5 µm/px
    mid, mid_keys = _grid(10, 200.0, "L1")        # 100 tiles  @ 2.0 µm/px
    plate = np.array([[0.0, 0.0, 2000.0, 2000.0]])
    return Geometry([
        Level(0.5, fine, fine_keys),
        Level(2.0, mid, mid_keys),
        Level(8.0, plate, ["L2:plate"]),
    ])


def _arr(nbytes: int) -> np.ndarray:
    return np.zeros(nbytes, dtype=np.uint8)


def _desc(level, key, bbox, channel="0") -> TileDescriptor:
    return TileDescriptor(level, key, channel, bbox)


# ---------------------------------------------------------------------------------------
# contract: descriptors, Geometry / Level validation
# ---------------------------------------------------------------------------------------

def test_contract_descriptor_is_frozen_and_hashable():
    d = _desc(0, "L0:0,0", (0.0, 0.0, 100.0, 100.0))
    assert {d: 1}[d] == 1                                   # usable as a cache key
    assert d == _desc(0, "L0:0,0", (0.0, 0.0, 100.0, 100.0))
    with pytest.raises(Exception):
        d.level = 1                                         # frozen dataclass


def test_contract_geometry_rejects_bad_levels():
    bboxes, keys = _grid(2, 100.0, "L0")
    with pytest.raises(ValueError):
        Geometry([])                                        # no levels
    with pytest.raises(ValueError):                         # not finest-first
        Geometry([Level(2.0, bboxes, keys), Level(0.5, bboxes, keys)])
    with pytest.raises(ValueError):                         # duplicate scale
        Geometry([Level(2.0, bboxes, keys), Level(2.0, bboxes, keys)])


def test_contract_level_rejects_nonfinite_inverted_and_mismatched_keys():
    with pytest.raises(ValueError):
        Level(0.5, [[0.0, 0.0, np.nan, 10.0]], ["a"])       # NaN stage coord: loud, early
    with pytest.raises(ValueError):
        Level(0.5, [[10.0, 0.0, 0.0, 10.0]], ["a"])         # inverted box
    with pytest.raises(ValueError):
        Level(0.5, [[0.0, 0.0, 10.0, 10.0]], ["a", "b"])    # keys/bboxes length mismatch
    with pytest.raises(ValueError):
        Level(0.0, [[0.0, 0.0, 10.0, 10.0]], ["a"])         # scale must be > 0


def test_contract_empty_level_is_legal_and_culls_to_nothing():
    g = Geometry([Level(0.5, np.zeros((0, 4)), [])])
    assert select_tiles((0.0, 0.0, 100.0, 100.0), 0.5, g) == []


# ---------------------------------------------------------------------------------------
# LOD
# ---------------------------------------------------------------------------------------

def test_lod_picks_level_just_finer_than_screen_resolution():
    g = _ladder()
    assert g.pick_level(0.5) == 0                            # exact tie -> that level
    assert g.pick_level(1.9) == 0                            # finer than the mid rung
    assert g.pick_level(2.0) == 1                            # exact tie again
    assert g.pick_level(7.9) == 1
    assert g.pick_level(8.0) == 2


def test_lod_clamps_both_ends():
    g = _ladder()
    assert g.pick_level(0.01) == 0                           # finer than level 0 -> level 0
    assert g.pick_level(10_000.0) == 2                       # coarser than coarsest -> coarsest


def test_lod_single_level_ladder_always_picks_it():
    bboxes, keys = _grid(2, 100.0, "L0")
    g = Geometry([Level(1.0, bboxes, keys)])
    assert g.pick_level(0.001) == 0 and g.pick_level(1000.0) == 0


def test_lod_rejects_bad_zoom():
    g = _ladder()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            g.pick_level(bad)


def test_lod_hysteresis_holds_the_level_across_a_boundary():
    g = _ladder()
    # Boundary between level 0 and level 1 sits at 2.0 µm/px; the deadband is +/-25%.
    assert g.pick_level(2.1, current_level=0) == 0            # nudged over: stay on 0
    assert g.pick_level(2.4, current_level=0) == 0
    assert g.pick_level(3.0, current_level=0) == 1            # decisively past: switch
    assert g.pick_level(1.7, current_level=1) == 1            # zooming back in: stick on 1
    assert g.pick_level(1.2, current_level=1) == 0            # far enough in: switch


def test_lod_hysteresis_never_thrashes_on_jitter_at_the_boundary():
    g = _ladder()
    level = g.pick_level(2.0)
    switches = 0
    for jitter in np.linspace(-0.15, 0.15, 41):              # wheel jitter parked on the boundary
        new = g.pick_level(2.0 + jitter, current_level=level)
        switches += int(new != level)
        level = new
    assert switches == 0


def test_lod_without_hysteresis_does_thrash_the_same_jitter():
    """Control for the test above: the deadband, not the geometry, is what stops the churn."""
    g = _ladder()
    levels = {g.pick_level(2.0 + j, current_level=None) for j in (-0.05, 0.05)}
    assert levels == {0, 1}


# ---------------------------------------------------------------------------------------
# cull
# ---------------------------------------------------------------------------------------

def test_cull_viewport_inside_returns_only_overlapping_fovs():
    g = _ladder()
    # A 250 µm window at level-0 resolution touches a 3x3 block of 100 µm FOVs.
    tiles = select_tiles((50.0, 50.0, 300.0, 300.0), 0.5, g)
    assert {t.level for t in tiles} == {0}
    assert len(tiles) == 9
    assert {t.key for t in tiles} == {f"L0:{i},{j}" for i in range(3) for j in range(3)}


def test_cull_partial_overlap_at_the_plate_edge():
    g = _ladder()
    tiles = select_tiles((-500.0, -500.0, 50.0, 50.0), 0.5, g)
    assert [t.key for t in tiles] == ["L0:0,0"]              # only the corner FOV overlaps


def test_cull_touching_edges_do_not_count_as_overlap():
    g = _ladder()
    tiles = select_tiles((-500.0, -500.0, 0.0, 0.0), 0.5, g)
    assert tiles == []


def test_cull_fully_outside_returns_empty():
    g = _ladder()
    assert select_tiles((5000.0, 5000.0, 6000.0, 6000.0), 0.5, g) == []


def test_cull_rejects_degenerate_and_inverted_bbox():
    g = _ladder()
    for bad in ((0.0, 0.0, 0.0, 100.0),                      # zero area
                (100.0, 0.0, 0.0, 100.0),                    # inverted x
                (0.0, 100.0, 100.0, 0.0),                    # inverted y
                (0.0, 0.0, float("nan"), 100.0)):            # non-finite
        with pytest.raises(ValueError):
            select_tiles(bad, 0.5, g)


def test_cull_is_deterministic_and_channel_major():
    g = _ladder()
    a = select_tiles((50.0, 50.0, 300.0, 300.0), 0.5, g, channels=("488", "638"))
    b = select_tiles((50.0, 50.0, 300.0, 300.0), 0.5, g, channels=("488", "638"))
    assert a == b                                            # stable ordering, run to run
    assert [t.channel for t in a] == ["488"] * 9 + ["638"] * 9
    assert len({(t.key, t.channel) for t in a}) == 18        # per-channel descriptors


def test_cull_bboxes_match_the_geometry():
    g = _ladder()
    tiles = select_tiles((50.0, 50.0, 150.0, 150.0), 0.5, g)
    for t in tiles:
        x0, y0, x1, y1 = t.bbox_um
        assert (x1 - x0, y1 - y0) == (100.0, 100.0)
        assert x0 < 150.0 and x1 > 50.0


def test_cull_correctness_against_a_bruteforce_loop_on_55k_boxes():
    rng = np.random.default_rng(216)
    n = 55_000
    origins = rng.uniform(0.0, 50_000.0, size=(n, 2))
    sizes = rng.uniform(10.0, 200.0, size=(n, 2))
    bboxes = np.hstack([origins, origins + sizes])
    g = Geometry([Level(0.5, bboxes, [f"f{i}" for i in range(n)])])
    box = (12_000.0, 30_000.0, 13_000.0, 31_000.0)
    got = {t.key for t in select_tiles(box, 0.5, g)}
    want = {f"f{i}" for i in range(n)
            if bboxes[i, 0] < box[2] and bboxes[i, 2] > box[0]
            and bboxes[i, 1] < box[3] and bboxes[i, 3] > box[1]}
    assert got == want and got                               # non-empty, and exactly right


# ---------------------------------------------------------------------------------------
# the O(viewport) invariant — the whole point of the module
# ---------------------------------------------------------------------------------------

def test_tile_count_is_o_viewport_not_o_placements():
    g = _ladder()
    all_fovs = len(g.levels[0])
    zoomed_in = select_tiles((0.0, 0.0, 200.0, 200.0), 0.5, g)
    assert len(zoomed_in) == 4                               # 4 of 400 FOVs
    assert len(zoomed_in) < all_fovs / 50


def test_fit_to_plate_reads_a_handful_of_coarse_tiles_not_every_fov():
    g = _ladder()
    # Whole 2000 µm plate on an ~250 px widget: 8 µm/px -> the plate rung, one tile.
    tiles = select_tiles((0.0, 0.0, 2000.0, 2000.0), 8.0, g)
    assert len(tiles) == 1 and tiles[0].level == 2
    assert len(tiles) < len(g.levels[0])


def test_tile_count_grows_with_viewport_area_not_with_the_plate():
    g = _ladder()
    counts = [len(select_tiles((0.0, 0.0, s, s), 0.5, g)) for s in (100.0, 200.0, 400.0)]
    assert counts == [1, 4, 16]                              # ~area of the viewport


def test_viewport_wrapper_takes_pixels_per_world_unit():
    g = _ladder()
    assert viewport((0.0, 0.0, 200.0, 200.0), 2.0, g) == select_tiles((0.0, 0.0, 200.0, 200.0), 0.5, g)
    assert viewport((0.0, 0.0, 2000.0, 2000.0), 0.125, g)[0].level == 2
    with pytest.raises(ValueError):
        viewport((0.0, 0.0, 200.0, 200.0), 0.0, g)


# ---------------------------------------------------------------------------------------
# TileCache: LRU + byte budget
# ---------------------------------------------------------------------------------------

def test_cache_lru_evicts_oldest_first():
    c = TileCache(budget_bytes=300)
    ds = [_desc(0, f"k{i}", (i * 10.0, 0.0, i * 10.0 + 10.0, 10.0)) for i in range(4)]
    for d in ds[:3]:
        c.insert(d, _arr(100))
    assert c.nbytes == 300 and len(c) == 3
    c.insert(ds[3], _arr(100))
    assert ds[0] not in c and ds[3] in c                     # oldest went first
    assert c.nbytes == 300


def test_cache_reaccess_promotes_to_most_recently_used():
    c = TileCache(budget_bytes=300)
    ds = [_desc(0, f"k{i}", (i * 10.0, 0.0, i * 10.0 + 10.0, 10.0)) for i in range(4)]
    for d in ds[:3]:
        c.insert(d, _arr(100))
    assert c.get(ds[0]) is not None                          # touch the oldest
    c.insert(ds[3], _arr(100))
    assert ds[0] in c and ds[1] not in c                     # k1 is now the LRU victim


def test_cache_respects_the_byte_budget_under_mixed_sizes():
    c = TileCache(budget_bytes=1000)
    rng = np.random.default_rng(0)
    for i in range(50):
        size = int(rng.integers(50, 400))
        c.insert(_desc(0, f"k{i}", (i * 10.0, 0.0, i * 10.0 + 10.0, 10.0)), _arr(size))
        assert c.nbytes <= c.budget_bytes or len(c) == 1
    assert c.nbytes <= 1000


def test_cache_admits_an_oversized_single_tile_alone():
    c = TileCache(budget_bytes=100)
    small = _desc(0, "small", (0.0, 0.0, 10.0, 10.0))
    huge = _desc(0, "huge", (10.0, 0.0, 20.0, 10.0))
    c.insert(small, _arr(100))
    c.insert(huge, _arr(5000))                               # never refused: refusing = blank screen
    assert huge in c and small not in c and len(c) == 1


def test_cache_degenerate_budgets():
    zero = TileCache(budget_bytes=0)
    d = _desc(0, "k", (0.0, 0.0, 10.0, 10.0))
    zero.insert(d, _arr(10))
    assert d in zero and len(zero) == 1                      # one tile always survives
    one = TileCache(budget_bytes=10)
    e = _desc(0, "k2", (10.0, 0.0, 20.0, 10.0))
    one.insert(d, _arr(10)); one.insert(e, _arr(10))
    assert len(one) == 1 and e in one
    with pytest.raises(ValueError):
        TileCache(budget_bytes=-1)


def test_cache_reinsert_does_not_double_count_bytes():
    c = TileCache(budget_bytes=1000)
    d = _desc(0, "k", (0.0, 0.0, 10.0, 10.0))
    c.insert(d, _arr(100))
    c.insert(d, _arr(200))
    assert c.nbytes == 200 and len(c) == 1


# ---------------------------------------------------------------------------------------
# TileCache: keep-parent-until-child-ready (pins)
# ---------------------------------------------------------------------------------------

PARENT = _desc(1, "L1:0,0", (0.0, 0.0, 200.0, 200.0))
CHILD = _desc(0, "L0:0,0", (0.0, 0.0, 100.0, 100.0))


def test_pin_survives_filling_the_cache_to_capacity():
    c = TileCache(budget_bytes=300)
    c.insert(PARENT, _arr(100))
    c.mark_pending(CHILD)                                    # parent is now pinned
    assert c.pinned_descriptors() == [PARENT]
    for i in range(10):                                      # churn far past the budget
        c.insert(_desc(0, f"filler{i}", (1000.0 + i, 0.0, 1001.0 + i, 1.0)), _arr(100))
    assert PARENT in c                                       # never evicted: no blank hole
    assert c.nbytes <= c.budget_bytes


def test_pin_released_on_insert_of_the_child():
    c = TileCache(budget_bytes=300)
    c.insert(PARENT, _arr(100))
    c.mark_pending(CHILD)
    c.insert(CHILD, _arr(100))
    assert c.pinned_descriptors() == [] and c.pending_descriptors() == []
    for i in range(10):
        c.insert(_desc(0, f"filler{i}", (1000.0 + i, 0.0, 1001.0 + i, 1.0)), _arr(100))
    assert PARENT not in c                                   # unpinned -> evictable again


def test_pin_released_on_fetch_failed_so_it_never_leaks_an_immortal():
    c = TileCache(budget_bytes=300)
    c.insert(PARENT, _arr(100))
    c.mark_pending(CHILD)
    c.fetch_failed(CHILD)
    assert c.pinned_descriptors() == [] and c.pending_descriptors() == []
    for i in range(10):
        c.insert(_desc(0, f"filler{i}", (1000.0 + i, 0.0, 1001.0 + i, 1.0)), _arr(100))
    assert PARENT not in c


def test_pin_cap_drops_the_oldest_pending_request():
    c = TileCache(budget_bytes=400)                          # pins may hold at most 200 bytes
    parents = [_desc(1, f"p{i}", (i * 200.0, 0.0, i * 200.0 + 200.0, 200.0)) for i in range(4)]
    for p in parents:
        c.insert(p, _arr(100))
    for i, p in enumerate(parents):
        c.mark_pending(_desc(0, f"c{i}", (p.bbox_um[0], 0.0, p.bbox_um[0] + 100.0, 100.0)))
    assert len(c.pinned_descriptors()) <= 2                  # capped at budget/2
    assert "c0" not in {d.key for d in c.pending_descriptors()}   # oldest request dropped


def test_pin_only_matches_a_covering_ancestor_of_the_same_channel():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    c.mark_pending(_desc(0, "other-channel", CHILD.bbox_um, channel="638"))
    assert c.pinned_descriptors() == []                      # per-channel caching, per-channel pins
    c.mark_pending(_desc(0, "elsewhere", (900.0, 900.0, 1000.0, 1000.0)))
    assert c.pinned_descriptors() == []                      # not covered by the parent


def test_mark_pending_is_idempotent_and_skips_already_cached_tiles():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    c.mark_pending(CHILD); c.mark_pending(CHILD)
    assert c.pending_descriptors() == [CHILD]
    c.mark_pending(PARENT)                                   # already cached: nothing to fetch
    assert c.pending_descriptors() == [CHILD]


# ---------------------------------------------------------------------------------------
# TileCache: resolve() + invalidate()
# ---------------------------------------------------------------------------------------

def test_resolve_substitutes_the_nearest_cached_ancestor():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    c.mark_pending(CHILD)
    renderable = c.resolve([CHILD])
    assert [d for d, _ in renderable] == [PARENT]            # coarse stand-in, not a hole


def test_resolve_prefers_the_finest_ancestor_available():
    c = TileCache(budget_bytes=1000)
    grandparent = _desc(2, "L2:plate", (0.0, 0.0, 2000.0, 2000.0))
    c.insert(grandparent, _arr(100))
    c.insert(PARENT, _arr(100))
    assert [d for d, _ in c.resolve([CHILD])] == [PARENT]    # level 1 beats level 2


def test_resolve_drops_the_parent_once_the_child_lands():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    c.mark_pending(CHILD)
    c.insert(CHILD, _arr(100))
    assert [d for d, _ in c.resolve([CHILD])] == [CHILD]


def test_resolve_omits_slots_with_no_ancestor_at_all():
    c = TileCache(budget_bytes=1000)
    assert c.resolve([CHILD]) == []                          # cold cache: caller draws nothing


def test_resolve_dedupes_one_parent_covering_many_children():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    children = [_desc(0, f"L0:{i},{j}", (i * 100.0, j * 100.0, i * 100.0 + 100.0, j * 100.0 + 100.0))
                for i in range(2) for j in range(2)]
    assert [d for d, _ in c.resolve(children)] == [PARENT]   # 4 slots, 1 draw


def test_resolve_mixes_cached_children_with_stand_in_parents():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    c.insert(CHILD, _arr(100))
    sibling = _desc(0, "L0:1,0", (100.0, 0.0, 200.0, 100.0))
    got = [d for d, _ in c.resolve([CHILD, sibling])]
    assert got == [CHILD, PARENT]


def test_resolve_of_a_real_viewport_selection_returns_arrays():
    g = _ladder()
    c = TileCache(budget_bytes=10_000)
    ideal = select_tiles((0.0, 0.0, 200.0, 200.0), 0.5, g)
    for d in ideal:
        c.insert(d, _arr(100))
    renderable = c.resolve(ideal)
    assert len(renderable) == len(ideal)
    assert all(isinstance(a, np.ndarray) for _, a in renderable)


def test_invalidate_drops_matching_tiles_and_their_pins():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    keep = _desc(1, "L1:1,1", (200.0, 200.0, 400.0, 400.0))
    c.insert(keep, _arr(100))
    c.mark_pending(CHILD)
    dropped = c.invalidate(lambda d: d.key == "L1:0,0")      # a freshly written FOV kills its ancestor
    assert dropped == 1 and PARENT not in c and keep in c
    assert c.pinned_descriptors() == [] and c.nbytes == 100
    assert c.pending_descriptors() == [CHILD]                # the fetch itself is untouched


def test_invalidate_also_cancels_matching_pending_fetches():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    c.mark_pending(CHILD)
    c.invalidate(lambda d: d.level == 0)
    assert c.pending_descriptors() == [] and PARENT in c


def test_invalidate_of_nothing_is_a_noop():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    assert c.invalidate(lambda d: False) == 0 and len(c) == 1


# ---------------------------------------------------------------------------------------
# timing — integration-marked only, never a hard CI assert (test_performance.py precedent)
# ---------------------------------------------------------------------------------------

@pytest.mark.integration
def test_tiling_select_is_fast_on_a_55k_box_geometry():
    rng = np.random.default_rng(216)
    n = 55_000
    origins = rng.uniform(0.0, 50_000.0, size=(n, 2))
    bboxes = np.hstack([origins, origins + 100.0])
    g = Geometry([Level(0.5, bboxes, [f"f{i}" for i in range(n)])])
    select_tiles((0.0, 0.0, 1000.0, 1000.0), 0.5, g)         # warm numpy
    t0 = time.perf_counter()
    for _ in range(100):
        select_tiles((12_000.0, 30_000.0, 13_000.0, 31_000.0), 0.5, g)
    per_call_ms = (time.perf_counter() - t0) / 100 * 1e3
    print(f"\nselect_tiles over {n} boxes: {per_call_ms:.3f} ms/call")
    assert per_call_ms < 10.0                                # generous: a regression canary, not a target
