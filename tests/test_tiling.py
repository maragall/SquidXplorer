"""Viewport tiler: LOD pick, frustum cull, byte-budget LRU, parent pinning."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._tiling import (
    Geometry,
    Level,
    TileCache,
    TileDescriptor,
    select_tiles,
    viewport,
)


def _grid(n: int, size: float, level_tag: str) -> tuple[np.ndarray, list[str]]:
    x0 = np.repeat(np.arange(n, dtype=np.float64), n) * size
    y0 = np.tile(np.arange(n, dtype=np.float64), n) * size
    bboxes = np.stack([x0, y0, x0 + size, y0 + size], axis=1)
    keys = [f"{level_tag}:{int(a)},{int(b)}" for a, b in zip(x0 // size, y0 // size)]
    return bboxes, keys


def _ladder() -> Geometry:
    """A 20x20 FOV grid with a coarse ladder above it (per-FOV / 2x2 / plate)."""
    fine, fine_keys = _grid(20, 100.0, "L0")
    mid, mid_keys = _grid(10, 200.0, "L1")
    plate = np.array([[0.0, 0.0, 2000.0, 2000.0]])
    return Geometry([
        Level(0.5, fine, fine_keys),
        Level(2.0, mid, mid_keys),
        Level(8.0, plate, ["L2:plate"]),
    ])


def _arr(nbytes: int) -> np.ndarray:
    return np.zeros(nbytes, dtype=np.uint8)


def _desc(level, key, bbox, channel="0", t=0) -> TileDescriptor:
    return TileDescriptor(level, key, channel, bbox, t)


def _filler(i: int) -> TileDescriptor:
    return _desc(0, f"filler{i}", (1000.0 + i, 0.0, 1001.0 + i, 1.0))


PARENT = _desc(1, "L1:0,0", (0.0, 0.0, 200.0, 200.0))
CHILD = _desc(0, "L0:0,0", (0.0, 0.0, 100.0, 100.0))


# --- contract -----------------------------------------------------------------------------

def test_contract_descriptor_is_frozen_and_hashable():
    d = _desc(0, "L0:0,0", (0.0, 0.0, 100.0, 100.0))
    assert {d: 1}[d] == 1
    assert d == _desc(0, "L0:0,0", (0.0, 0.0, 100.0, 100.0))
    with pytest.raises(Exception):
        d.level = 1


def test_contract_geometry_and_level_refuse_bad_construction():
    bboxes, keys = _grid(2, 100.0, "L0")
    with pytest.raises(ValueError):
        Geometry([])
    with pytest.raises(ValueError):                         # not finest-first
        Geometry([Level(2.0, bboxes, keys), Level(0.5, bboxes, keys)])
    with pytest.raises(ValueError):                         # duplicate scale
        Geometry([Level(2.0, bboxes, keys), Level(2.0, bboxes, keys)])
    with pytest.raises(ValueError):
        Level(0.5, [[0.0, 0.0, np.nan, 10.0]], ["a"])       # NaN stage coord
    with pytest.raises(ValueError):
        Level(0.5, [[10.0, 0.0, 0.0, 10.0]], ["a"])         # inverted box
    with pytest.raises(ValueError):
        Level(0.5, [[0.0, 0.0, 10.0, 10.0]], ["a", "b"])    # keys/bboxes mismatch
    with pytest.raises(ValueError):
        Level(0.0, [[0.0, 0.0, 10.0, 10.0]], ["a"])         # scale must be > 0
    g = Geometry([Level(0.5, np.zeros((0, 4)), [])])         # an empty level is legal
    assert select_tiles((0.0, 0.0, 100.0, 100.0), 0.5, g) == []


# --- LOD ----------------------------------------------------------------------------------

def test_lod_picks_the_level_just_finer_than_screen_resolution_and_clamps():
    g = _ladder()
    assert g.pick_level(0.5) == 0
    assert g.pick_level(1.9) == 0
    assert g.pick_level(2.0) == 1
    assert g.pick_level(7.9) == 1
    assert g.pick_level(8.0) == 2
    assert g.pick_level(0.01) == 0 and g.pick_level(10_000.0) == 2
    bboxes, keys = _grid(2, 100.0, "L0")
    one = Geometry([Level(1.0, bboxes, keys)])
    assert one.pick_level(0.001) == 0 and one.pick_level(1000.0) == 0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            g.pick_level(bad)


def test_lod_hysteresis_holds_the_level_across_a_boundary_without_thrashing():
    g = _ladder()
    assert g.pick_level(2.1, current_level=0) == 0
    assert g.pick_level(2.4, current_level=0) == 0
    assert g.pick_level(3.0, current_level=0) == 1
    assert g.pick_level(1.7, current_level=1) == 1
    assert g.pick_level(1.2, current_level=1) == 0
    level, switches = g.pick_level(2.0), 0
    for jitter in np.linspace(-0.15, 0.15, 41):
        new = g.pick_level(2.0 + jitter, current_level=level)
        switches += int(new != level)
        level = new
    assert switches == 0
    assert {g.pick_level(2.0 + j, current_level=None) for j in (-0.05, 0.05)} == {0, 1}, (
        "the deadband, not the geometry, is what stops the churn")


# --- cull ---------------------------------------------------------------------------------

def test_cull_returns_exactly_the_overlapping_tiles():
    g = _ladder()
    tiles = select_tiles((50.0, 50.0, 300.0, 300.0), 0.5, g)
    assert {t.level for t in tiles} == {0} and len(tiles) == 9
    assert {t.key for t in tiles} == {f"L0:{i},{j}" for i in range(3) for j in range(3)}
    for t in select_tiles((50.0, 50.0, 150.0, 150.0), 0.5, g):
        x0, y0, x1, y1 = t.bbox_um
        assert (x1 - x0, y1 - y0) == (100.0, 100.0) and x0 < 150.0 and x1 > 50.0
    assert [t.key for t in select_tiles((-500.0, -500.0, 50.0, 50.0), 0.5, g)] == ["L0:0,0"]
    assert select_tiles((-500.0, -500.0, 0.0, 0.0), 0.5, g) == []       # touching edges
    assert select_tiles((5000.0, 5000.0, 6000.0, 6000.0), 0.5, g) == []
    for bad in ((0.0, 0.0, 0.0, 100.0), (100.0, 0.0, 0.0, 100.0),
                (0.0, 100.0, 100.0, 0.0), (0.0, 0.0, float("nan"), 100.0)):
        with pytest.raises(ValueError):
            select_tiles(bad, 0.5, g)


def test_cull_is_deterministic_and_channel_major():
    g = _ladder()
    a = select_tiles((50.0, 50.0, 300.0, 300.0), 0.5, g, channels=("488", "638"))
    b = select_tiles((50.0, 50.0, 300.0, 300.0), 0.5, g, channels=("488", "638"))
    assert a == b
    assert [t.channel for t in a] == ["488"] * 9 + ["638"] * 9
    assert len({(t.key, t.channel) for t in a}) == 18


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
    assert got == want and got


def test_tile_count_is_o_viewport_not_o_placements():
    g = _ladder()
    assert len(select_tiles((0.0, 0.0, 200.0, 200.0), 0.5, g)) == 4      # 4 of 400 FOVs
    counts = [len(select_tiles((0.0, 0.0, s, s), 0.5, g)) for s in (100.0, 200.0, 400.0)]
    assert counts == [1, 4, 16]
    fit = select_tiles((0.0, 0.0, 2000.0, 2000.0), 8.0, g)
    assert len(fit) == 1 and fit[0].level == 2
    assert viewport((0.0, 0.0, 200.0, 200.0), 2.0, g) == select_tiles((0.0, 0.0, 200.0, 200.0), 0.5, g)
    assert viewport((0.0, 0.0, 2000.0, 2000.0), 0.125, g)[0].level == 2
    with pytest.raises(ValueError):
        viewport((0.0, 0.0, 200.0, 200.0), 0.0, g)


# --- TileCache: LRU + byte budget ---------------------------------------------------------

def test_cache_is_lru_and_reaccess_promotes():
    c = TileCache(budget_bytes=300)
    ds = [_desc(0, f"k{i}", (i * 10.0, 0.0, i * 10.0 + 10.0, 10.0)) for i in range(5)]
    for d in ds[:3]:
        c.insert(d, _arr(100))
    assert c.nbytes == 300 and len(c) == 3
    c.insert(ds[3], _arr(100))
    assert ds[0] not in c and ds[3] in c and c.nbytes == 300
    assert c.get(ds[1]) is not None                          # touch the oldest survivor
    c.insert(ds[4], _arr(100))
    assert ds[1] in c and ds[2] not in c


def test_cache_respects_the_byte_budget_and_never_refuses_a_single_tile():
    c = TileCache(budget_bytes=1000)
    rng = np.random.default_rng(0)
    for i in range(50):
        c.insert(_desc(0, f"k{i}", (i * 10.0, 0.0, i * 10.0 + 10.0, 10.0)), _arr(int(rng.integers(50, 400))))
        assert c.nbytes <= c.budget_bytes or len(c) == 1
    d = _desc(0, "k", (0.0, 0.0, 10.0, 10.0))
    c = TileCache(budget_bytes=1000)
    c.insert(d, _arr(100))
    c.insert(d, _arr(200))
    assert c.nbytes == 200 and len(c) == 1                   # reinsert does not double count
    small = TileCache(budget_bytes=100)
    huge = _desc(0, "huge", (10.0, 0.0, 20.0, 10.0))
    small.insert(d, _arr(100))
    small.insert(huge, _arr(5000))                           # refusing = blank screen
    assert huge in small and d not in small and len(small) == 1
    zero = TileCache(budget_bytes=0)
    zero.insert(d, _arr(10))
    assert d in zero and len(zero) == 1
    with pytest.raises(ValueError):
        TileCache(budget_bytes=-1)


# --- TileCache: keep-parent-until-child-ready (pins) --------------------------------------

def test_pin_survives_filling_the_cache_and_is_released_by_the_child_or_a_failure():
    c = TileCache(budget_bytes=300)
    c.insert(PARENT, _arr(100))
    c.mark_pending(CHILD)
    assert c.pinned_descriptors() == [PARENT]
    for i in range(10):
        c.insert(_filler(i), _arr(100))
    assert PARENT in c and c.nbytes <= c.budget_bytes
    c.insert(CHILD, _arr(100))
    assert c.pinned_descriptors() == [] and c.pending_descriptors() == []
    for i in range(10):
        c.insert(_filler(i), _arr(100))
    assert PARENT not in c

    c = TileCache(budget_bytes=300)
    c.insert(PARENT, _arr(100))
    c.mark_pending(CHILD)
    c.fetch_failed(CHILD)
    assert c.pinned_descriptors() == [] and c.pending_descriptors() == []
    for i in range(10):
        c.insert(_filler(i), _arr(100))
    assert PARENT not in c


def test_pin_cap_drops_the_oldest_pending_request():
    c = TileCache(budget_bytes=400)
    parents = [_desc(1, f"p{i}", (i * 200.0, 0.0, i * 200.0 + 200.0, 200.0)) for i in range(4)]
    for p in parents:
        c.insert(p, _arr(100))
    for i, p in enumerate(parents):
        c.mark_pending(_desc(0, f"c{i}", (p.bbox_um[0], 0.0, p.bbox_um[0] + 100.0, 100.0)))
    assert len(c.pinned_descriptors()) <= 2
    assert "c0" not in {d.key for d in c.pending_descriptors()}


def test_pin_only_matches_a_covering_ancestor_of_the_same_channel_and_is_idempotent():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    c.mark_pending(_desc(0, "other-channel", CHILD.bbox_um, channel="638"))
    assert c.pinned_descriptors() == []
    c.mark_pending(_desc(0, "elsewhere", (900.0, 900.0, 1000.0, 1000.0)))
    assert c.pinned_descriptors() == []
    c.mark_pending(CHILD); c.mark_pending(CHILD)
    assert c.pending_descriptors().count(CHILD) == 1
    c.mark_pending(PARENT)                                   # already cached: nothing to fetch
    assert PARENT not in c.pending_descriptors()


# --- TileCache: resolve() + invalidate() --------------------------------------------------

def test_resolve_stands_in_the_finest_covering_ancestor_or_nothing():
    c = TileCache(budget_bytes=1000)
    assert c.resolve([CHILD]) == []                          # cold cache: draw nothing
    grandparent = _desc(2, "L2:plate", (0.0, 0.0, 2000.0, 2000.0))
    c.insert(grandparent, _arr(100))
    c.insert(PARENT, _arr(100))
    c.mark_pending(CHILD)
    assert [d for d, _ in c.resolve([CHILD])] == [PARENT]
    c.insert(CHILD, _arr(100))
    assert [d for d, _ in c.resolve([CHILD])] == [CHILD]


def test_resolve_dedupes_parents_and_mixes_them_with_cached_children():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    children = [_desc(0, f"L0:{i},{j}", (i * 100.0, j * 100.0, i * 100.0 + 100.0, j * 100.0 + 100.0))
                for i in range(2) for j in range(2)]
    assert [d for d, _ in c.resolve(children)] == [PARENT]   # 4 slots, 1 draw
    c.insert(CHILD, _arr(100))
    sibling = _desc(0, "L0:1,0", (100.0, 0.0, 200.0, 100.0))
    assert [d for d, _ in c.resolve([CHILD, sibling])] == [CHILD, PARENT]
    g = _ladder()
    ideal = select_tiles((0.0, 0.0, 200.0, 200.0), 0.5, g)
    for d in ideal:
        c.insert(d, _arr(100))
    renderable = c.resolve(ideal)
    assert len(renderable) == len(ideal) and all(isinstance(a, np.ndarray) for _, a in renderable)


def test_invalidate_drops_matching_tiles_pins_and_pending_fetches():
    c = TileCache(budget_bytes=1000)
    c.insert(PARENT, _arr(100))
    keep = _desc(1, "L1:1,1", (200.0, 200.0, 400.0, 400.0))
    c.insert(keep, _arr(100))
    c.mark_pending(CHILD)
    assert c.invalidate(lambda d: False) == 0 and len(c) == 2
    dropped = c.invalidate(lambda d: d.key == "L1:0,0")
    assert dropped == 1 and PARENT not in c and keep in c
    assert c.pinned_descriptors() == [] and c.nbytes == 100
    assert c.pending_descriptors() == [CHILD]                # the fetch itself is untouched
    c.invalidate(lambda d: d.level == 0)
    assert c.pending_descriptors() == [] and keep in c


# --- the fit-to-plate cost ----------------------------------------------------------------

def test_geometry_rejects_a_coarser_level_holding_more_tiles():
    fine, fk = _grid(4, 100.0, "L0")
    coarse, ck = _grid(8, 50.0, "L1")
    with pytest.raises(ValueError, match="cannot hold more tiles"):
        Geometry([Level(1.0, fine, fk), Level(8.0, coarse, ck)])


def test_worst_case_tiles_is_the_fit_to_plate_cost():
    geo = _ladder()
    assert geo.worst_case_tiles == 1
    assert len(select_tiles((0.0, 0.0, 2000.0, 2000.0), 1e6, geo, channels=("c",))) <= 1
    b, k = _grid(20, 100.0, "L0")
    per_fov = Geometry([Level(1.0, b, k), Level(8.0, b, list(k))])   # legal NGFF layout
    assert per_fov.worst_case_tiles == 400                            # loud, not a silent cliff
    assert len(select_tiles((0.0, 0.0, 2000.0, 2000.0), 1e6, per_fov, channels=("c",))) == 400
