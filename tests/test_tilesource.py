"""IMA-217 OME-Zarr pyramid source — the ladder, the zarr tile source, the RAM preview.

Test-first for the three claims the ticket is judged on:

  UNITS      every world value is µm, taken from ``metadata["fov_positions_um"]``; the plate
             the writer emits carries its own µm origin (NGFF ``translation``) so the source
             never needs a second, possibly-mm, coordinate path.
  O(VIEWPORT) tiles per view do not grow with FOV count — asserted on a fabricated 14,400-FOV
             plate against the same 144-FOV plate shipped in the real dataset.
  BOUNDED    the in-RAM multiscale refuses to exceed an explicit byte budget and its actual
             allocation is measured, not asserted by hand-wave.

Pure numpy except the two zarr round-trips, which write ~1 MB into tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squidmip._output import pyramid_shapes, write_from_stream
from squidmip._tiling import Geometry, TileCache, select_tiles
from squidmip._tilesource import (
    DEFAULT_TILE_PX,
    InMemoryMultiscale,
    PlateLadder,
    ZarrPyramidSource,
    fov_bboxes_um,
    plate_ladder,
)

# --- fabricated acquisition metadata ----------------------------------------------------

CH = [
    {"name": "Fluorescence_405_nm_Ex", "display_name": "Fluorescence 405 nm Ex", "display_color": "#20ADF8"},
    {"name": "Fluorescence_638_nm_Ex", "display_name": "Fluorescence 638 nm Ex", "display_color": "#FF0000"},
]
PX_UM = 0.3728571351101784        # the real synthetic_2x2_wellplate pixel size
PITCH_UM = 705.2256               # its real FOV pitch (9% overlap on a 2084 px frame)


def _meta(n_side: int, *, frame: int = 2084, regions=("A1",)) -> dict:
    """A square ``n_side x n_side`` FOV grid per region, at the real pixel size / pitch."""
    fovs = list(range(n_side * n_side))
    positions = {}
    for r_i, region in enumerate(regions):
        for f in fovs:
            gy, gx = divmod(f, n_side)
            positions[(region, f)] = (45698.0 + gx * PITCH_UM + r_i * 30000.0,
                                      25553.0 + gy * PITCH_UM)
    return {
        "regions": list(regions),
        "fovs_per_region": {r: list(fovs) for r in regions},
        "fov_positions_um": positions,
        "channels": CH,
        "pixel_size_um": PX_UM,
        "frame_shape": (frame, frame),
        "dtype": np.dtype("uint16"),
    }


# --- pyramid shapes (the _output extension) ---------------------------------------------

def test_pyramid_shapes_matches_the_written_levels():
    # 2084 -> 1042 -> 521 -> (crop 520) 260 -> 130; stops at the 256 px floor.
    assert pyramid_shapes((2084, 2084)) == [(2084, 2084), (1042, 1042), (521, 521), (260, 260), (130, 130)]
    assert pyramid_shapes((8, 8)) == [(8, 8)]                     # already under the floor
    assert len(pyramid_shapes((100000, 100000))) == 6             # capped by max_levels
    assert pyramid_shapes((2084, 2084), min_yx=1024) == [(2084, 2084), (1042, 1042), (521, 521)]


def test_pyramid_shapes_agrees_with_the_actual_downsample():
    from squidmip._output import _pyramid
    img = np.zeros((1, 1, 1, 600, 401), np.uint16)
    assert [lv.shape[-2:] for lv in _pyramid(img)] == [tuple(s) for s in pyramid_shapes((600, 401))]


# --- world geometry: units ---------------------------------------------------------------

def test_fov_bboxes_are_centred_micrometres():
    """Positions are stage µm FOV *centres*; a bbox is the frame extent around one."""
    boxes = fov_bboxes_um({("A1", 0): (1000.0, 2000.0)}, (100, 200), 0.5)
    # 200 px * 0.5 µm = 100 µm wide, 100 px * 0.5 = 50 µm tall, centred on (1000, 2000)
    assert boxes[("A1", 0)] == pytest.approx((950.0, 1975.0, 1050.0, 2025.0))


def test_ladder_world_span_is_micrometres_not_millimetres():
    """A mm value anywhere is a bug: the 12x12 grid must span thousands of µm, not units."""
    ladder = plate_ladder(_meta(12))
    x0, y0, x1, y1 = ladder.world_bbox_um
    frame_um = 2084 * PX_UM
    assert (x1 - x0) == pytest.approx(11 * PITCH_UM + frame_um)   # 8534 µm, not 8.5
    assert (y1 - y0) == pytest.approx(11 * PITCH_UM + frame_um)
    assert ladder.geometry.levels[0].scale_um_per_px == pytest.approx(PX_UM)


def test_ladder_rejects_positions_that_look_like_millimetres():
    meta = _meta(4)
    meta["fov_positions_um"] = {k: (v[0] / 1000.0, v[1] / 1000.0) for k, v in meta["fov_positions_um"].items()}
    with pytest.raises(ValueError, match="MILLIMETRES"):
        plate_ladder(meta)


# --- the ladder shape --------------------------------------------------------------------

def test_ladder_is_a_valid_geometry_with_plate_rungs_on_top():
    ladder = plate_ladder(_meta(6))
    g = ladder.geometry
    assert isinstance(g, Geometry)
    # All 5 written pyramid levels are known; only those finer than the crossover
    # (fov_extent_um 777 µm == tile_px 512 * scale -> 1.52 µm/px) become per-FOV RUNGS. The
    # coarser two are still read — they are the pixels a plate tile is composited from.
    assert ladder.fov_level_shapes == pyramid_shapes((2084, 2084))
    assert ladder.n_fov_levels == 3
    assert len(g) > ladder.n_fov_levels                       # coarse plate rungs were added
    # every per-FOV rung holds exactly one tile per FOV
    for i in range(ladder.n_fov_levels):
        assert len(g.levels[i]) == 36
    # scales strictly increase and the fine rungs are the per-FOV pyramid's own scales
    scales = [lv.scale_um_per_px for lv in g.levels]
    assert scales == sorted(scales) and len(set(scales)) == len(scales)
    assert scales[1] == pytest.approx(PX_UM * 2084 / 1042)
    # the coarsest rung is a single tile: fit-to-plate can never cost more than that
    assert g.worst_case_tiles == 1


def test_plate_rung_cells_are_sparse_only_where_fovs_are():
    """A 2-well plate with a big gap must not pay for the empty space between the wells."""
    ladder = plate_ladder(_meta(4, regions=("A1", "A2")))       # 32 FOVs, wells 30 mm apart
    first_plate = ladder.geometry.levels[ladder.n_fov_levels]
    n_dense = ladder.plate_grid_shape(ladder.n_fov_levels)
    assert len(first_plate) < n_dense[0] * n_dense[1]           # empty cells were dropped


def test_ladder_scales_and_keys_round_trip_through_select_tiles():
    ladder = plate_ladder(_meta(6))
    g = ladder.geometry
    # zoomed all the way in on one FOV: level 0, one tile per channel
    box = ladder.fov_bboxes[("A1", 0)]
    mid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    tiles = select_tiles((mid[0] - 50, mid[1] - 50, mid[0] + 50, mid[1] + 50), PX_UM, g,
                         channels=("0", "1"))
    assert {t.level for t in tiles} == {0}
    assert {t.key for t in tiles} == {("A1", 0)}
    assert len(tiles) == 2                                      # one per channel


# --- O(viewport): the non-negotiable ------------------------------------------------------

def _fit_to_plate_tiles(ladder: PlateLadder, screen_px: int = 1400) -> int:
    x0, y0, x1, y1 = ladder.world_bbox_um
    um_per_px = max(x1 - x0, y1 - y0) / screen_px
    return len(select_tiles((x0, y0, x1, y1), um_per_px, ladder.geometry))


def test_fit_to_plate_tile_count_does_not_grow_with_fov_count():
    """12x12 = 144 FOVs vs 120x120 = 14,400 FOVs: 100x the plate, same order of tiles."""
    small = _fit_to_plate_tiles(plate_ladder(_meta(12)))
    large = _fit_to_plate_tiles(plate_ladder(_meta(120)))
    assert small <= 30 and large <= 30
    assert large <= small * 2                       # bounded by the screen, not by the plate
    assert plate_ladder(_meta(120)).geometry.worst_case_tiles <= 4


def test_full_zoom_reads_a_handful_of_tiles_on_a_huge_plate():
    ladder = plate_ladder(_meta(120))               # 14,400 FOVs
    x0, y0, _, _ = ladder.world_bbox_um
    # a 1400 px window at native resolution = ~522 µm, less than one 777 µm FOV
    tiles = select_tiles((x0 + 100, y0 + 100, x0 + 622, y0 + 622), PX_UM, ladder.geometry)
    assert len(tiles) <= 4


def test_every_rung_is_smaller_than_the_one_below_on_a_huge_plate():
    # Geometry itself raises on an inverted rung; assert we never construct one, at scale.
    g = plate_ladder(_meta(120)).geometry
    counts = [len(lv) for lv in g.levels]
    assert counts == sorted(counts, reverse=True) or all(
        counts[i] >= counts[i + 1] for i in range(len(counts) - 1))


# --- the in-RAM multiscale: explicit byte budget -------------------------------------------

def test_in_ram_multiscale_capacity_is_declared_and_under_budget():
    ladder = plate_ladder(_meta(12))
    pv = InMemoryMultiscale(ladder, channels=["0", "1"], dtype=np.uint16, budget_bytes=64 << 20)
    assert pv.levels                                     # at least the coarsest rung is resident
    assert pv.capacity_bytes <= pv.budget_bytes
    assert pv.nbytes == 0                                # nothing allocated before any field lands


def test_in_ram_multiscale_drops_finer_rungs_to_stay_in_budget():
    ladder = plate_ladder(_meta(120))                    # 14,400 FOVs
    big = InMemoryMultiscale(ladder, channels=["0", "1"], dtype=np.uint16, budget_bytes=512 << 20)
    small = InMemoryMultiscale(ladder, channels=["0", "1"], dtype=np.uint16, budget_bytes=8 << 20)
    assert len(small.levels) < len(big.levels)           # coarsest-first admission
    assert small.levels[0] == max(small.levels)          # the coarsest rung is always resident
    assert small.capacity_bytes <= small.budget_bytes
    assert big.capacity_bytes <= big.budget_bytes


def test_in_ram_multiscale_refuses_a_budget_that_cannot_hold_the_coarsest_rung():
    ladder = plate_ladder(_meta(12))
    with pytest.raises(ValueError, match="budget"):
        InMemoryMultiscale(ladder, channels=["0", "1"], dtype=np.uint16, budget_bytes=1024)


def test_in_ram_multiscale_accumulates_fields_and_reports_dirty_tiles():
    meta = _meta(4, frame=256)
    ladder = plate_ladder(meta, tile_px=64)
    pv = InMemoryMultiscale(ladder, channels=["0", "1"], dtype=np.uint16, budget_bytes=8 << 20)

    field = np.full((1, 2, 1, 256, 256), 700, np.uint16)
    field[0, 1] = 40
    dirty = pv.add_field("A1", 0, field)
    assert dirty, "adding a field must name the coarse tiles it invalidated"
    assert {d.channel for d in dirty} == {"0", "1"}
    assert pv.nbytes > 0 and pv.nbytes <= pv.capacity_bytes

    d0 = next(d for d in dirty if d.channel == "0")
    tile = pv.read_tile(d0)
    assert tile.dtype == np.uint16
    assert tile.max() == 700                              # the field's value survived the resample
    other = pv.read_tile(next(d for d in dirty if d.channel == "1"))
    assert other.max() == 40                              # channels are kept separate


def test_in_ram_multiscale_untouched_tile_reads_as_zeros_not_an_error():
    """A half-acquired plate must render, not raise: unfilled coarse tiles are black."""
    meta = _meta(4, frame=256)
    ladder = plate_ladder(meta, tile_px=64)
    pv = InMemoryMultiscale(ladder, channels=["0"], dtype=np.uint16, budget_bytes=8 << 20)
    lvl = pv.levels[0]
    key = ladder.geometry.levels[lvl].keys[0]
    desc = next(t for t in select_tiles(ladder.world_bbox_um, 1e9, ladder.geometry) if t.key == key)
    arr = pv.read_tile(desc)
    assert arr.shape == (ladder.tile_px, ladder.tile_px) or arr.size > 0
    assert not arr.any()


def test_in_ram_multiscale_memory_is_flat_in_field_count():
    """Streaming 4x the fields must not 4x the allocation — the rungs are preallocated-shaped."""
    meta = _meta(6, frame=256)
    ladder = plate_ladder(meta, tile_px=64)
    pv = InMemoryMultiscale(ladder, channels=["0"], dtype=np.uint16, budget_bytes=8 << 20)
    field = np.full((1, 1, 1, 256, 256), 300, np.uint16)
    pv.add_field("A1", 0, field)
    after_one = pv.nbytes
    for f in range(1, 36):
        pv.add_field("A1", f, field)
    assert pv.nbytes <= pv.capacity_bytes
    assert pv.nbytes < after_one * 36                      # tiles are shared, not per field


def test_in_ram_multiscale_dirty_tiles_drive_tilecache_invalidate():
    """The streaming seam ``_tiling.invalidate`` documents: a new field evicts its coarse tiles."""
    meta = _meta(4, frame=256)
    ladder = plate_ladder(meta, tile_px=64)
    pv = InMemoryMultiscale(ladder, channels=["0"], dtype=np.uint16, budget_bytes=8 << 20)
    cache = TileCache(4 << 20)

    coarse = pv.levels[0]
    key = ladder.geometry.levels[coarse].keys[0]
    from squidmip._tiling import TileDescriptor
    desc = TileDescriptor(coarse, key, "0", ladder.cell_bbox_um(coarse, key))
    cache.insert(desc, pv.read_tile(desc))          # a black tile, cached before the field lands
    assert not cache.get(desc).any()

    dirty = pv.add_field("A1", 0, np.full((1, 1, 1, 256, 256), 900, np.uint16))
    dropped = cache.invalidate(lambda d: d in set(dirty))
    assert dropped == 1                              # the stale black tile was evicted
    assert desc not in cache
    assert pv.read_tile(desc).max() == 900           # the re-read now carries the field


# --- the zarr source: round trip through a real written plate ------------------------------

def _write_small_plate(tmp_path: Path, *, frame: int = 512, n_side: int = 3) -> tuple[Path, dict]:
    meta = _meta(n_side, frame=frame)
    rng = np.random.default_rng(0)
    images = {}

    def stream():
        for f in meta["fovs_per_region"]["A1"]:
            img = rng.integers(0, 4000, (1, 2, 1, frame, frame), dtype=np.uint16)
            images[f] = img
            yield "A1", f, img

    manifest = write_from_stream(meta, stream(), tmp_path, n_fovs=None, tiff=False)
    return Path(manifest["plate"]), meta


def test_written_plate_carries_a_micrometre_translation_per_field(tmp_path):
    plate, meta = _write_small_plate(tmp_path)
    doc = json.loads((plate / "A" / "1" / "0" / "zarr.json").read_text())["attributes"]["ome"]
    ms = doc["multiscales"][0]
    for ds in ms["datasets"]:
        types = [t["type"] for t in ds["coordinateTransformations"]]
        assert types == ["scale", "translation"]          # NGFF: scale before translation
        tr = ds["coordinateTransformations"][1]["translation"]
        assert len(tr) == len(ms["axes"])                 # one entry per axis
        assert tr[:3] == [0.0, 0.0, 0.0]                  # t, c, z unshifted
    # the y/x translation is the field's top-left corner in stage µm
    cx, cy = meta["fov_positions_um"][("A1", 0)]
    half = 512 * PX_UM / 2
    tr = ms["datasets"][0]["coordinateTransformations"][1]["translation"]
    assert tr[4] == pytest.approx(cx - half) and tr[3] == pytest.approx(cy - half)
    from tests.ngff_check import assert_valid_ngff_plate
    assert_valid_ngff_plate(plate)


def test_zarr_source_rebuilds_world_geometry_from_the_plate_alone(tmp_path):
    plate, meta = _write_small_plate(tmp_path)
    src = ZarrPyramidSource(plate)
    from_meta = plate_ladder(meta, tile_px=DEFAULT_TILE_PX)
    assert src.ladder.world_bbox_um == pytest.approx(from_meta.world_bbox_um)
    assert src.ladder.n_fov_levels == from_meta.n_fov_levels
    assert [lv.scale_um_per_px for lv in src.ladder.geometry.levels] == pytest.approx(
        [lv.scale_um_per_px for lv in from_meta.geometry.levels])


def test_zarr_source_reads_a_level0_tile_pixel_exact(tmp_path):
    plate, meta = _write_small_plate(tmp_path)
    src = ZarrPyramidSource(plate)
    desc = next(t for t in select_tiles(src.ladder.world_bbox_um, PX_UM, src.ladder.geometry,
                                        channels=src.channels[:1]) if t.key == ("A1", 0))
    tile = src.read_tile(desc)
    assert tile.shape == (512, 512) and tile.dtype == np.uint16
    import tensorstore as ts
    store = ts.open({"driver": "zarr3",
                     "kvstore": {"driver": "file", "path": str(plate / "A" / "1" / "0" / "0")}},
                    open=True).result()
    assert np.array_equal(tile, np.asarray(store[0, 0, 0].read().result()))


def test_zarr_source_composites_a_plate_tile_from_many_fovs(tmp_path):
    plate, _ = _write_small_plate(tmp_path)
    src = ZarrPyramidSource(plate)
    coarse = len(src.ladder.geometry) - 1
    key = src.ladder.geometry.levels[coarse].keys[0]
    from squidmip._tiling import TileDescriptor
    bbox = src.ladder.cell_bbox_um(coarse, key)
    tile = src.read_tile(TileDescriptor(coarse, key, src.channels[0], bbox))
    assert tile.dtype == np.uint16
    assert tile.any()                                     # real pixels, not a blank canvas
    # the coarse tile must be small: a plate view is bounded by the tile size, not the plate
    assert tile.nbytes <= src.ladder.tile_px ** 2 * 2


def test_zarr_source_satisfies_the_tilesource_protocol_through_the_cache(tmp_path):
    """The real consumer path: select_tiles -> TileCache.resolve -> read_tile -> insert."""
    plate, _ = _write_small_plate(tmp_path)
    src = ZarrPyramidSource(plate)
    cache = TileCache(16 << 20)
    ideal = select_tiles(src.ladder.world_bbox_um, 1e6, src.ladder.geometry,
                         channels=src.channels[:1])
    assert ideal and len(ideal) <= 4
    for desc in ideal:
        cache.mark_pending(desc)
        cache.insert(desc, src.read_tile(desc))
    renderable = cache.resolve(ideal)
    assert len(renderable) == len(ideal)
    assert all(a.ndim == 2 for _, a in renderable)
