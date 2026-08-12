"""``ReaderTileSource`` — deep zoom over a RAW acquisition, with no written plate.

The plate overview smooth-scales one 88 px-per-well montage, so zooming in blurs instead of
resolving. IMA-216 (``_tiling``) and IMA-217 (``_tilesource``) already built the whole LOD stack,
but both existing sources need a plate that has been WRITTEN — which leaves the case the viewer is
in most of the time, an acquisition folder opened for a look, with no tile source at all.

These tests pin the source's contract. They use a fake reader rather than the ``squid_dataset``
fixture: what is under test is the world-µm-to-tile-pixel mapping, and a synthetic frame whose
value encodes its own identity makes a misplacement visible as a wrong NUMBER rather than as a
picture someone has to eyeball.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._tiling import TileCache, select_tiles, viewport
from squidxplorer._tilesource import ReaderTileSource, plate_ladder

FRAME = (64, 64)
PX_UM = 1.0
PITCH_UM = 64.0            # no overlap: each FOV owns a clean 64x64 µm square


class FakeReader:
    """Every FOV is a constant plane whose value IS its fov index + 1. Misplacement shows up as
    the wrong integer, which a test can assert on; a gradient could not."""

    def __init__(self, fail: set = frozenset()):
        self.reads = []
        self.fail = set(fail)

    def read(self, region, fov, channel, z, t=0):
        self.reads.append((region, int(fov), str(channel), int(z), int(t)))
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


# --- the world-to-pixel mapping ---------------------------------------------------------------

def test_a_tile_over_one_fov_reads_that_fovs_pixels():
    meta = _meta()
    src, ladder = _src(meta)
    key = ("A1", 3)
    desc = type("D", (), {"t": 0, "level": 0, "key": key, "channel": "488",
                          "bbox_um": ladder.fov_bboxes[key]})()
    tile = src.read_tile(desc)
    assert tile.shape == (64, 64)
    assert set(np.unique(tile)) == {4}, "fov 3 must read as 3+1 everywhere in its own box"


def test_world_with_no_fov_reads_as_zeros_not_an_error():
    """A viewport routinely covers stage area nothing was placed on. Black is the honest answer;
    raising would throw once per empty tile per frame on a part-acquired plate."""
    meta = _meta()
    src, _ = _src(meta)
    far = (10_000.0, 10_000.0, 10_064.0, 10_064.0)
    desc = type("D", (), {"t": 0, "level": 0, "key": ("A1", 0), "channel": "488", "bbox_um": far})()
    tile = src.read_tile(desc)
    assert tile.shape == (64, 64)
    assert not tile.any()


def test_a_coarse_tile_composites_every_fov_it_covers():
    """The plate-rung path: one tile, many FOVs, each in its own quadrant."""
    meta = _meta(n=2)
    src, ladder = _src(meta)
    whole = ladder.world_bbox_um
    desc = type("D", (), {"t": 0, "level": len(ladder.geometry) - 1, "key": (0, 0),
                          "channel": "488", "bbox_um": whole})()
    tile = src.read_tile(desc)
    assert set(np.unique(tile)) == {1, 2, 3, 4}, "all four FOVs must land, none overwrite another"
    h = tile.shape[0] // 2
    assert tile[0, 0] == 1 and tile[0, -1] == 2, "row-major lattice must map to the same quadrants"
    assert tile[-1, 0] == 3 and tile[-1, -1] == 4
    assert tile[h - 1, h - 1] == 1


def test_an_unreadable_field_is_a_hole_not_a_dead_viewport():
    meta = _meta(n=2)
    src, ladder = _src(meta, reader=FakeReader(fail={("A1", 0)}))
    desc = type("D", (), {"t": 0, "level": len(ladder.geometry) - 1, "key": (0, 0),
                          "channel": "488", "bbox_um": ladder.world_bbox_um})()
    tile = src.read_tile(desc)                       # must not raise
    assert set(np.unique(tile)) == {0, 2, 3, 4}, "the failed field is zeros; the rest still render"


# --- the plane cache --------------------------------------------------------------------------

def test_the_same_field_is_decoded_once_across_tiles():
    """Adjacent tiles touch the same FOVs; without the cache a pan re-decodes continuously."""
    meta = _meta(n=2)
    reader = FakeReader()
    src, ladder = _src(meta, reader=reader)
    desc = type("D", (), {"t": 0, "level": len(ladder.geometry) - 1, "key": (0, 0),
                          "channel": "488", "bbox_um": ladder.world_bbox_um})()
    src.read_tile(desc)
    first = len(reader.reads)
    src.read_tile(desc)
    assert len(reader.reads) == first, "second read of the same tile must come from the cache"


def test_tiles_are_maximum_intensity_projections_by_default():
    """Spencer: "I do want an MIP for this application." The default must project the stack, and
    it must go through the registered operator so `reference` and any add_projector op work too."""
    meta = _meta()
    reader = FakeReader()
    src, ladder = _src(meta, reader=reader)
    assert src.projector == "mip" and src.z is None

    key = ("A1", 0)
    desc = type("D", (), {"t": 0, "level": 0, "key": key, "channel": "488",
                          "bbox_um": ladder.fov_bboxes[key]})()
    src.read_tile(desc)
    zs_read = sorted(r[3] for r in reader.reads)
    assert zs_read == sorted(meta["z_levels"]), "a MIP must read every z, not one"


def test_an_explicit_z_reads_exactly_that_plane():
    """The escape hatch: one plane, no projection, for a fast path or a single-z acquisition."""
    meta = _meta()
    reader = FakeReader()
    src, ladder = _src(meta, reader=reader, z=1)
    key = ("A1", 0)
    desc = type("D", (), {"t": 0, "level": 0, "key": key, "channel": "488",
                          "bbox_um": ladder.fov_bboxes[key]})()
    src.read_tile(desc)
    assert [r[3] for r in reader.reads] == [1]


def test_a_projector_that_does_not_consume_z_is_refused():
    """This collapses a stack to one plane. A plane-op has no z to collapse, and running it per z
    and keeping the last would look plausible and be wrong."""
    meta = _meta()
    from squidxplorer._engine import add_projector

    add_projector("_tiletest_planeop", lambda planes: next(iter(planes)), consumes=frozenset())
    with pytest.raises(ValueError, match="does not consume z"):
        _src(meta, projector="_tiletest_planeop")


# --- the O(viewport) promise ------------------------------------------------------------------

def test_a_small_viewport_at_fine_zoom_does_not_fetch_the_plate():
    """The whole point of the ladder. A zoomed-in view must cost its SCREEN, not the sample."""
    meta = _meta(n=6)
    _, ladder = _src(meta)
    g = ladder.geometry
    w = ladder.world_bbox_um

    everything = select_tiles(w, g.levels[0].scale_um_per_px, g, channels=("488",))
    corner = (w[0], w[1], w[0] + PITCH_UM, w[1] + PITCH_UM)     # one FOV's worth of world
    few = select_tiles(corner, g.levels[0].scale_um_per_px, g, channels=("488",))

    assert len(few) < len(everything), "culling must drop the off-screen tiles"
    assert len(few) <= 4, f"a one-FOV viewport asked for {len(few)} tiles"


def test_zooming_out_past_the_ladder_clamps_to_the_coarsest_rung():
    """Fit-to-plate on a real plate is a coarse view; the pick must clamp rather than run off the
    end of the ladder. Driven by um_per_px directly — this fixture's plate is only a few hundred
    µm across, so "fit into a 512 px window" would be a zoom IN and would prove nothing."""
    meta = _meta(n=6)
    _, ladder = _src(meta)
    g = ladder.geometry
    coarser_than_the_ladder = g.levels[-1].scale_um_per_px * 10

    descs = select_tiles(ladder.world_bbox_um, coarser_than_the_ladder, g, channels=("488",))
    assert descs, "a zoomed-out view must still return something to draw"
    assert descs[0].level == len(g) - 1
    assert len(descs) <= g.worst_case_tiles


# --- integration with the cache the renderer will use -----------------------------------------

def test_tiles_round_trip_through_the_cache_the_renderer_uses():
    meta = _meta(n=2)
    src, ladder = _src(meta)
    g = ladder.geometry
    descs = viewport(ladder.world_bbox_um, 64.0 / PITCH_UM, g, channels=("488",))
    cache = TileCache(budget_bytes=8 << 20)

    assert cache.resolve(descs) == [], "nothing cached yet -> nothing drawable"
    for d in descs:
        cache.insert(d, src.read_tile(d))
    drawn = cache.resolve(descs)
    assert len(drawn) == len(descs)
    assert all(arr.shape == (64, 64) for _d, arr in drawn)
    assert cache.nbytes <= cache.budget_bytes


@pytest.mark.parametrize("how", ["empty", "absent"])
def test_no_stage_positions_refuses_to_build_a_ladder(how):
    """A region dropped for an unusable coordinates.csv must not silently pile every FOV at one
    spot — plate_ladder already refuses, and the viewer's montage fallback depends on that raise.

    Both shapes matter: `_fov_positions_um_or_empty` returns {} for a malformed file, and a
    metadata dict assembled elsewhere may omit the key entirely.
    """
    meta = _meta()
    if how == "empty":
        meta["fov_positions_um"] = {}
    else:
        meta.pop("fov_positions_um")

    with pytest.raises(ValueError, match="fov_positions_um|stage"):
        plate_ladder(meta)
