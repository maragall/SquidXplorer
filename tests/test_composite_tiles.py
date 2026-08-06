"""The coarse rung is a lookup, not a 25 second full-plate decode.

``NEXT_STEPS.md``, Spencer, 2026-07-28:

    Coarse rungs cannot be served by ``ReaderTileSource`` as it stands: a fit-to-plate tile
    overlaps all 72 FOVs and measured **25 s** to build. The fix that was scoped but not built is
    a composite source, ``InMemoryMultiscale`` fed from the existing ``_PreviewWorker`` pass for
    plate rungs, ``ReaderTileSource`` for FOV rungs.

The 25 s is arithmetic, not a slow loop: a fit-to-plate tile covers the whole sample, so every
FOV overlaps it, and a raw acquisition has no written pyramid to read a coarse version from -- so
each of those FOVs decodes a full frame to contribute a handful of pixels. On this repo's
1536-well fixture the same tile touches 1536 fields, not 72.

These tests pin what closes it: the plate rungs come from the cells the preview pass already
composited (and ``_platecache`` now keeps across restarts), the FOV rungs still come from the
reader byte for byte, a cell that is NOT cached goes to the reader rather than being drawn black,
and the two pictures agree about where the sample is.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from squidmip._platecache import CellTile
from squidmip._tilesource import (
    CompositePlateSource,
    ReaderTileSource,
    plate_ladder,
    region_bbox_um,
)
from squidmip._tiling import TileDescriptor

CH = "c0"
FRAME = (64, 64)
PIXEL_UM = 1.0
#: Two wells 2 mm apart: far enough that a plate rung exists above the crossover, close enough
#: that one coarse tile covers both. That is the tile Spencer measured.
POSITIONS = {("A1", 0): (0.0, 0.0), ("A2", 0): (2000.0, 0.0)}
VALUES = {"A1": 4000, "A2": 0}          # only A1 is bright: both sources must put it in one place


def _meta() -> dict:
    return {"channels": [{"name": CH}], "dtype": "uint16", "z_levels": [0],
            "regions": ["A1", "A2"], "fovs_per_region": {"A1": [0], "A2": [0]},
            "frame_shape": FRAME, "pixel_size_um": PIXEL_UM, "fov_positions_um": dict(POSITIONS)}


class _CountingReader:
    """A reader that says how many frames were decoded. That count IS the 25 s."""

    def __init__(self, delay_s: float = 0.0):
        self.reads = 0
        self.delay = float(delay_s)

    def read(self, region, fov, channel, z, t=0):
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
    return plate[-1]                     # the coarsest: the fit-to-plate view


def _desc(ladder, level, key, t=0) -> TileDescriptor:
    return TileDescriptor(level=level, key=key, channel=CH,
                          bbox_um=ladder.cell_bbox_um(level, key), t=t)


def _cells(value_per_region=None) -> dict:
    """What the preview pass produces and ``_platecache`` persists: one cell per well."""
    vals = VALUES if value_per_region is None else value_per_region
    return {r: CellTile(np.full((1, 88, 88), v, dtype=np.uint16), (0, 0, 88, 88))
            for r, v in vals.items()}


# --- the gap, closed --------------------------------------------------------------------------

def test_a_coarse_tile_costs_no_frame_decodes_at_all():
    """The whole point. Every FOV overlaps a fit-to-plate tile; none of them may be read."""
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
    """The before, so the after is a comparison and not an assertion of faith."""
    ladder = _ladder()
    reader = _CountingReader()
    src = ReaderTileSource(reader, _meta(), ladder)
    lvl = _plate_rung(ladder)
    key = ladder.geometry.levels[lvl].keys[0]
    src.read_tile(_desc(ladder, lvl, key))
    assert reader.reads == len(ladder.fovs_overlapping(ladder.cell_bbox_um(lvl, key))) >= 2


def test_the_composite_and_the_reader_agree_about_where_the_sample_is():
    """Cheap is worthless if it is cheap and wrong. The bright well must land in one place."""
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


# --- the FOV rungs are untouched -----------------------------------------------------------------

def test_fov_rungs_are_the_reader_byte_for_byte():
    """Delegation, not a second implementation. Real resolution still comes from the frames."""
    ladder = _ladder()
    desc = TileDescriptor(level=0, key=("A1", 0), channel=CH,
                          bbox_um=ladder.fov_bboxes[("A1", 0)], t=0)
    composite = CompositePlateSource(_CountingReader(), _meta(), ladder, cells=_cells())
    plain = ReaderTileSource(_CountingReader(), _meta(), ladder)
    assert np.array_equal(composite.read_tile(desc), plain.read_tile(desc))
    assert composite.coarse_from_cells == 0 and composite.coarse_from_reader == 0


# --- what happens when the cache cannot answer ---------------------------------------------------

def test_an_uncached_well_goes_to_the_reader_and_SAYS_so():
    """Not black, and not silent. Zeros would be a picture that looks acquired and is not."""
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
    """A single-FOV acquisition has nothing to compose. It must not become a special case."""
    meta = _meta()
    meta["regions"] = ["A1"]
    meta["fovs_per_region"] = {"A1": [0]}
    meta["fov_positions_um"] = {("A1", 0): (0.0, 0.0)}
    ladder = plate_ladder(meta)
    src = CompositePlateSource(_CountingReader(), meta, ladder)
    desc = TileDescriptor(level=0, key=("A1", 0), channel=CH,
                          bbox_um=ladder.fov_bboxes[("A1", 0)], t=0)
    assert src.read_tile(desc).any()


# --- geometry -------------------------------------------------------------------------------------

def test_a_region_bbox_is_the_union_of_its_frames():
    """The rectangle a cached cell covers, and the one ``cell_boxes`` scaled into the cell.

    Both are the bounding box of the region's placed frames -- one in micrometres, one in pixels.
    That equality is what lets a cell be pasted back into world space with no second geometry to
    keep in step.
    """
    ladder = _ladder()
    box = region_bbox_um(ladder, "A1")
    assert box == ladder.fov_bboxes[("A1", 0)]
    assert region_bbox_um(ladder, "nope") is None


def test_the_composite_is_lazy_and_reads_the_cache_only_when_a_coarse_tile_is_asked_for():
    """A session that never zooms out past the crossover must not pay for a rung it never sees."""
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
                                 bbox_um=ladder.fov_bboxes[("A1", 0)], t=0))
    assert cache.loads == 0, "the cells were read for a tile that did not need them"
    lvl = _plate_rung(ladder)
    src.read_tile(_desc(ladder, lvl, ladder.geometry.levels[lvl].keys[0]))
    assert cache.loads == 1
    src.read_tile(_desc(ladder, lvl, ladder.geometry.levels[lvl].keys[0]))
    assert cache.loads == 1, "the cells were re-read; seeding must happen once"


def test_the_deep_zoom_source_is_the_composite_one():
    """The regression that matters: nobody may quietly put the bare ReaderTileSource back."""
    import inspect

    from squidmip import _viewer

    src = inspect.getsource(_viewer.PlateOverview.set_tile_source)
    assert "CompositePlateSource(" in src, "the plate view stopped using the composite source"
