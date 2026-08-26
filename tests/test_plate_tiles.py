"""ReaderTileSource — deep zoom over a RAW acquisition, with no written plate."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._tiling import TileCache, select_tiles, viewport
from squidxplorer._tilesource import ReaderTileSource, plate_ladder

FRAME = (64, 64)
PX_UM = 1.0
PITCH_UM = 64.0            # no overlap: each FOV owns a clean 64x64 µm square


class FakeReader:
    """Every FOV is a constant plane whose value is its fov index + 1."""

    def __init__(self, fail: set = frozenset()):
        self.reads = []
        self.fail = set(fail)

    def read(self, region, fov, channel, z_level, time_point=0):
        self.reads.append((region, int(fov), str(channel), int(z_level), int(time_point)))
        if (region, int(fov)) in self.fail:
            raise OSError("simulated unreadable field")
        return np.full(FRAME, int(fov) + 1, dtype=np.uint16)


def _meta(n=2):
    """An n x n grid of FOVs in one region, centres on a PITCH_UM lattice."""
    positions = {}
    for i in range(n * n):
        gy, gx = divmod(i, n)
        positions[("A1", i)] = (gx * PITCH_UM + PITCH_UM / 2, gy * PITCH_UM + PITCH_UM / 2)
    return {
        "fov_positions_um": positions,
        "pixel_size_um": PX_UM,
        "frame_shape": FRAME,
        "dtype": "uint16",
        "z_levels": [0, 1, 2],
        "channels": [{"name": "488"}],
        "fovs_per_region": {"A1": list(range(n * n))},
    }


def _src(meta, reader=None, **kw):
    ladder = plate_ladder(meta, tile_px=64)
    return ReaderTileSource(reader or FakeReader(), meta, ladder, **kw), ladder


def _desc(level, key, bbox):
    return type("D", (), {"time_point": 0, "level": level, "key": key, "channel": "488", "bbox_um": bbox})()


def test_a_tile_over_one_fov_reads_that_fov_and_empty_world_reads_as_zeros():
    meta = _meta()
    src, ladder = _src(meta)
    key = ("A1", 3)
    tile = src.read_tile(_desc(0, key, ladder.fov_bboxes[key]))
    assert tile.shape == (64, 64)
    assert set(np.unique(tile)) == {4}, "fov 3 must read as 3+1 everywhere in its own box"
    far = src.read_tile(_desc(0, ("A1", 0), (10_000.0, 10_000.0, 10_064.0, 10_064.0)))
    assert far.shape == (64, 64) and not far.any(), "world with no FOV is zeros, not an error"


def test_a_coarse_tile_composites_every_fov_it_covers_and_decodes_each_field_once():
    meta = _meta(n=2)
    reader = FakeReader()
    src, ladder = _src(meta, reader=reader)
    desc = _desc(len(ladder.geometry) - 1, (0, 0), ladder.world_bbox_um)
    tile = src.read_tile(desc)
    assert set(np.unique(tile)) == {1, 2, 3, 4}, "all four FOVs must land, none overwrite another"
    h = tile.shape[0] // 2
    assert tile[0, 0] == 1 and tile[0, -1] == 2, "row-major lattice must map to the same quadrants"
    assert tile[-1, 0] == 3 and tile[-1, -1] == 4
    assert tile[h - 1, h - 1] == 1
    first = len(reader.reads)
    src.read_tile(desc)
    assert len(reader.reads) == first, "second read of the same tile must come from the cache"


def test_an_unreadable_field_is_a_hole_not_a_dead_viewport():
    meta = _meta(n=2)
    src, ladder = _src(meta, reader=FakeReader(fail={("A1", 0)}))
    tile = src.read_tile(_desc(len(ladder.geometry) - 1, (0, 0), ladder.world_bbox_um))
    assert set(np.unique(tile)) == {0, 2, 3, 4}, "the failed field is zeros; the rest still render"


def test_tiles_are_maximum_intensity_projections_by_default_and_an_explicit_z_reads_one_plane():
    meta = _meta()
    reader = FakeReader()
    src, ladder = _src(meta, reader=reader)
    assert src.operator == "mip" and src.z_level is None
    key = ("A1", 0)
    src.read_tile(_desc(0, key, ladder.fov_bboxes[key]))
    assert sorted(r[3] for r in reader.reads) == sorted(meta["z_levels"]), "a MIP must read every z"
    one = FakeReader()
    src, ladder = _src(meta, reader=one, z_level=1)
    src.read_tile(_desc(0, key, ladder.fov_bboxes[key]))
    assert [r[3] for r in one.reads] == [1]


def test_an_operator_that_does_not_consume_z_is_refused():
    """A plane-op has no z to collapse; running it per z and keeping the last would be wrong."""
    from squidxplorer._engine import add_operator

    add_operator("_tiletest_planeop", lambda planes: next(iter(planes)), consumes=frozenset())
    with pytest.raises(ValueError, match="does not consume z"):
        _src(_meta(), operator="_tiletest_planeop")


def test_a_viewport_costs_its_screen_at_fine_zoom_and_clamps_to_the_coarsest_rung_zoomed_out():
    meta = _meta(n=6)
    _, ladder = _src(meta)
    g = ladder.geometry
    w = ladder.world_bbox_um
    everything = select_tiles(w, g.levels[0].scale_um_per_px, g, channels=("488",))
    corner = (w[0], w[1], w[0] + PITCH_UM, w[1] + PITCH_UM)     # one FOV's worth of world
    few = select_tiles(corner, g.levels[0].scale_um_per_px, g, channels=("488",))
    assert len(few) < len(everything) and len(few) <= 4, f"a one-FOV viewport asked for {len(few)} tiles"
    descs = select_tiles(w, g.levels[-1].scale_um_per_px * 10, g, channels=("488",))
    assert descs and descs[0].level == len(g) - 1 and len(descs) <= g.worst_case_tiles


def test_tiles_round_trip_through_the_cache_the_renderer_uses():
    meta = _meta(n=2)
    src, ladder = _src(meta)
    descs = viewport(ladder.world_bbox_um, 64.0 / PITCH_UM, ladder.geometry, channels=("488",))
    cache = TileCache(budget_bytes=8 << 20)
    assert cache.resolve(descs) == [], "nothing cached yet -> nothing drawable"
    for d in descs:
        cache.insert(d, src.read_tile(d))
    drawn = cache.resolve(descs)
    assert len(drawn) == len(descs) and all(arr.shape == (64, 64) for _d, arr in drawn)
    assert cache.nbytes <= cache.budget_bytes


@pytest.mark.parametrize("how", ["empty", "absent"])
def test_no_stage_positions_refuses_to_build_a_ladder(how):
    """Both shapes matter: `_fov_positions_um_or_empty` returns {} for a malformed file, and a metadata dict assembled elsewhere may omit the key entirely."""
    meta = _meta()
    if how == "empty":
        meta["fov_positions_um"] = {}
    else:
        meta.pop("fov_positions_um")
    with pytest.raises(ValueError, match="fov_positions_um|stage"):
        plate_ladder(meta)
