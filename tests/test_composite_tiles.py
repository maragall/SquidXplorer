"""The coarse rung is a lookup, not a 25 second full-plate decode."""
from __future__ import annotations

import time

import numpy as np
import pytest

from squidxplorer._platecache import CellTile
from squidxplorer._tilesource import (
    CompositePlateSource,
    ReaderTileSource,
    plate_ladder,
    region_bbox_um,
)
from squidxplorer._tiling import TileDescriptor

CH = "c0"
FRAME = (64, 64)
PIXEL_UM = 1.0
POSITIONS = {("A1", 0): (0.0, 0.0), ("A2", 0): (2000.0, 0.0)}
VALUES = {"A1": 4000, "A2": 0}


def _meta() -> dict:
    return {"channels": [{"name": CH}], "dtype": "uint16", "z_levels": [0],
            "regions": ["A1", "A2"], "fovs_per_region": {"A1": [0], "A2": [0]},
            "frame_shape": FRAME, "pixel_size_um": PIXEL_UM, "fov_positions_um": dict(POSITIONS)}


class _CountingReader:
    """A reader that says how many frames were decoded. That count IS the 25 s."""

    def __init__(self, delay_s: float = 0.0):
        self.reads = 0
        self.delay = float(delay_s)

    def read(self, region, fov, channel, z_level, time_point=0):
        self.reads += 1
        if self.delay:
            time.sleep(self.delay)
        return np.full(FRAME, VALUES[str(region)], dtype=np.uint16)


def _ladder():
    return plate_ladder(_meta())


def _plate_rung(ladder) -> int:
    plate = [i for i in range(len(ladder.geometry)) if not ladder.is_fov_level(i)]
    if not plate:
        pytest.skip("this ladder has no plate rung to test")
    return plate[-1]


def _desc(ladder, level, key, t=0) -> TileDescriptor:
    return TileDescriptor(level=level, key=key, channel=CH,
                          bbox_um=ladder.cell_bbox_um(level, key), time_point=t)


def _cells(value_per_region=None) -> dict:
    """What the preview pass produces and ``_platecache`` persists: one cell per well."""
    vals = VALUES if value_per_region is None else value_per_region
    return {r: CellTile(np.full((1, 88, 88), v, dtype=np.uint16), (0, 0, 88, 88))
            for r, v in vals.items()}


def test_a_coarse_tile_costs_no_frame_decodes_at_all():
    ladder = _ladder()
    reader = _CountingReader()
    src = CompositePlateSource(reader, _meta(), ladder, cells=_cells())
    lvl = _plate_rung(ladder)
    key = ladder.geometry.levels[lvl].keys[0]

    tile = src.read_tile(_desc(ladder, lvl, key))
    assert tile.shape == (ladder.tile_px, ladder.tile_px)
    assert reader.reads == 0, "a coarse tile still decoded frames; the composite is not composing"
    assert src.coarse_from_cells == 1


def test_the_same_coarse_tile_off_the_reader_does_decode_every_overlapping_field():
    ladder = _ladder()
    reader = _CountingReader()
    src = ReaderTileSource(reader, _meta(), ladder)
    lvl = _plate_rung(ladder)
    key = ladder.geometry.levels[lvl].keys[0]
    src.read_tile(_desc(ladder, lvl, key))
    assert reader.reads == len(ladder.fovs_overlapping(ladder.cell_bbox_um(lvl, key))) >= 2


def test_the_composite_and_the_reader_agree_about_where_the_sample_is():
    ladder = _ladder()
    lvl = _plate_rung(ladder)
    key = ladder.geometry.levels[lvl].keys[0]
    desc = _desc(ladder, lvl, key)

    from_cells = CompositePlateSource(_CountingReader(), _meta(), ladder,
                                      cells=_cells()).read_tile(desc)
    from_reader = ReaderTileSource(_CountingReader(), _meta(), ladder).read_tile(desc)
    assert from_cells.any() and from_reader.any()
    assert np.allclose(_centroid(from_cells), _centroid(from_reader), atol=2.0), (
        f"the cached plate rung puts the sample at {_centroid(from_cells)} and the reader puts it "
        f"at {_centroid(from_reader)}")


def _centroid(a: np.ndarray) -> tuple:
    w = a.astype(np.float64)
    total = w.sum()
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return (float((ys * w).sum() / total), float((xs * w).sum() / total))


def test_fov_rungs_are_the_reader_byte_for_byte():
    ladder = _ladder()
    desc = TileDescriptor(level=0, key=("A1", 0), channel=CH,
                          bbox_um=ladder.fov_bboxes[("A1", 0)], time_point=0)
    composite = CompositePlateSource(_CountingReader(), _meta(), ladder, cells=_cells())
    plain = ReaderTileSource(_CountingReader(), _meta(), ladder)
    assert np.array_equal(composite.read_tile(desc), plain.read_tile(desc))
    assert composite.coarse_from_cells == 0 and composite.coarse_from_reader == 0


def test_an_uncached_well_goes_to_the_reader_and_SAYS_so():
    ladder = _ladder()
    reader = _CountingReader()
    src = CompositePlateSource(reader, _meta(), ladder, cells={"A1": _cells()["A1"]})
    lvl = _plate_rung(ladder)
    key = ladder.geometry.levels[lvl].keys[0]
    tile = src.read_tile(_desc(ladder, lvl, key))
    assert tile.any(), "a partially cached plate rendered nothing"
    assert src.coarse_from_reader == 1 and src.coarse_from_cells == 0
    assert reader.reads > 0


def test_a_ladder_with_no_plate_rung_degrades_to_exactly_the_reader():
    meta = _meta()
    meta["regions"] = ["A1"]
    meta["fovs_per_region"] = {"A1": [0]}
    meta["fov_positions_um"] = {("A1", 0): (0.0, 0.0)}
    ladder = plate_ladder(meta)
    src = CompositePlateSource(_CountingReader(), meta, ladder)
    desc = TileDescriptor(level=0, key=("A1", 0), channel=CH,
                          bbox_um=ladder.fov_bboxes[("A1", 0)], time_point=0)
    assert src.read_tile(desc).any()


def test_a_region_bbox_is_the_union_of_its_frames():
    ladder = _ladder()
    box = region_bbox_um(ladder, "A1")
    assert box == ladder.fov_bboxes[("A1", 0)]
    assert region_bbox_um(ladder, "nope") is None


def test_the_composite_is_lazy_and_reads_the_cache_only_when_a_coarse_tile_is_asked_for():
    class _Cache:
        def __init__(self):
            self.loads = 0

        def load_all(self, regions):
            self.loads += 1
            return _cells()

    ladder = _ladder()
    cache = _Cache()
    src = CompositePlateSource(_CountingReader(), _meta(), ladder, cache=cache)
    src.read_tile(TileDescriptor(level=0, key=("A1", 0), channel=CH,
                                 bbox_um=ladder.fov_bboxes[("A1", 0)], time_point=0))
    assert cache.loads == 0, "the cells were read for a tile that did not need them"
    lvl = _plate_rung(ladder)
    src.read_tile(_desc(ladder, lvl, ladder.geometry.levels[lvl].keys[0]))
    assert cache.loads == 1
    src.read_tile(_desc(ladder, lvl, ladder.geometry.levels[lvl].keys[0]))
    assert cache.loads == 1, "the cells were re-read; seeding must happen once"


def test_the_deep_zoom_source_is_the_composite_one():
    import inspect

    from squidxplorer import _viewer

    src = inspect.getsource(_viewer.PlateOverview.set_tile_source)
    assert "CompositePlateSource(" in src, "the plate view stopped using the composite source"
