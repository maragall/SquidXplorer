"""Deep zoom reads the timepoint the plate is SHOWING, not the one its source was built with.

TileDescriptor used to carry (level, key, channel, bbox_um) with no timepoint, while both caches
on the read path keyed on t (_platecache: (token, t, region); ReaderTileSource._planes: t). So
every source answered from the t it was constructed with, and set_time_point touched neither the
source nor the tile cache: the plate could say "timepoint 2" over frame 0's tiles, byte-identical
before and after the change.

Fix puts t in the tile's identity, in the key (not the token), so a revisited timepoint stays a
cache hit — same rule docs/plate-contract.md gives for a plate cell's timepoint.

Uses multi_time_point_dataset, whose pixel value is t*100 + z*10 + c, so a stuck timepoint is a
wrong number, not a hash.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede any Qt import

import numpy as np
import pytest

from squidxplorer._tiling import TileCache, TileDescriptor
from squidxplorer._tilesource import CompositePlateSource, InMemoryMultiscale, ReaderTileSource, plate_ladder
from squidxplorer.reader import open_reader
from tests.conftest import TIME_SERIES_CHANNELS, time_series_pixel_value

N_T = 3


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("qtpy")
    if "PySide6" in sys.modules or "PySide2" in sys.modules:
        pytest.skip("PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 for the PyQt5 GUI tests.")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


def _open(root):
    reader = open_reader(str(root))
    meta = reader.metadata
    return reader, meta, plate_ladder(meta)


def _fov_desc(ladder, channel, t):
    key = next(iter(ladder.fov_bboxes))
    return TileDescriptor(level=0, key=key, channel=channel, bbox_um=ladder.fov_bboxes[key], t=t)


def _expected_mip(t, channel_index):
    """The MIP over z of this fixture's planes at *t*."""
    from tests.conftest import TIME_SERIES_NZ

    return max(time_series_pixel_value(t, z, channel_index) for z in range(TIME_SERIES_NZ))


def test_a_tile_descriptor_cannot_be_built_without_saying_which_timepoint():
    """No default: a defaulted 0 is precisely how the freeze happened one layer down."""
    with pytest.raises(TypeError):
        TileDescriptor(0, ("A1", 0), "c", (0.0, 0.0, 1.0, 1.0))     # type: ignore[call-arg]


def test_two_timepoints_of_one_tile_are_two_cache_entries():
    cache = TileCache(budget_bytes=1 << 20)
    box = (0.0, 0.0, 10.0, 10.0)
    a = TileDescriptor(0, ("A1", 0), "c", box, 0)
    b = TileDescriptor(0, ("A1", 0), "c", box, 2)
    cache.insert(a, np.zeros((4, 4), np.uint16))
    assert cache.get(b) is None, "a tile cached at t=0 answered a request for t=2"
    cache.insert(b, np.ones((4, 4), np.uint16))
    assert len(cache) == 2 and cache.get(a).sum() == 0


def test_a_coarse_ancestor_is_never_substituted_across_timepoints():
    """`resolve` is the one place a tile is drawn where a different one was asked for."""
    cache = TileCache(budget_bytes=1 << 20)
    parent_t0 = TileDescriptor(1, "coarse", "c", (0.0, 0.0, 100.0, 100.0), 0)
    cache.insert(parent_t0, np.zeros((4, 4), np.uint16))
    child_t2 = TileDescriptor(0, ("A1", 0), "c", (10.0, 10.0, 20.0, 20.0), 2)
    assert cache.resolve([child_t2]) == [], (
        "a t=0 tile was substituted for a missing t=2 one")
    assert [d for d, _ in cache.resolve([TileDescriptor(0, ("A1", 0), "c",
                                                        (10.0, 10.0, 20.0, 20.0), 0)])] \
        == [parent_t0], "the same-timepoint blur fallback must still work"


def test_one_reader_source_serves_every_timepoint(multi_time_point_dataset):
    """The source is built once and asked for three frames; before, ReaderTileSource took t
    at construction and every tile came from that frame."""
    root, _ = multi_time_point_dataset
    reader, meta, ladder = _open(root)
    channel = str(meta["channels"][0]["name"])
    src = ReaderTileSource(reader, meta, ladder)

    got = {}
    for t in range(N_T):
        tile = src.read_tile(_fov_desc(ladder, channel, t))
        got[t] = int(tile.max())
        assert got[t] == _expected_mip(t, 0), (
            f"tile at timepoint {t} carries pixel {got[t]}, which is timepoint "
            f"{got[t] // 100}'s value — the read is frozen at the frame the source was built with")
    assert len(set(got.values())) == N_T, f"three timepoints produced {got}"


def test_the_composite_source_serves_every_timepoint_too(multi_time_point_dataset):
    """The source the plate view actually builds; delegation must not reintroduce the freeze."""
    root, _ = multi_time_point_dataset
    reader, meta, ladder = _open(root)
    channel = str(meta["channels"][0]["name"])
    src = CompositePlateSource(reader, meta, ladder)
    seen = [int(src.read_tile(_fov_desc(ladder, channel, t)).max()) for t in range(N_T)]
    assert seen == [_expected_mip(t, 0) for t in range(N_T)], seen


def test_the_composite_refuses_a_cell_cache_for_another_timepoint(multi_time_point_dataset):
    """Reconciling silently is how a cell read at one frame gets published under another —
    same refusal _workers._PreviewWorker makes at the other end of the same cells."""
    root, _ = multi_time_point_dataset
    reader, meta, ladder = _open(root)

    class _Cache:
        time_point = 1

        def load_all(self, regions):
            return {}

    with pytest.raises(ValueError, match="timepoint 1"):
        CompositePlateSource(reader, meta, ladder, t=0, cache=_Cache())


def test_in_ram_plate_rungs_refuse_a_tile_from_another_timepoint():
    """These rungs hold one frame's cells; serving them under another is the defect itself."""
    meta = {"channels": [{"name": "c0"}], "dtype": "uint16", "z_levels": [0],
            "regions": ["A1", "A2"], "fovs_per_region": {"A1": [0], "A2": [0]},
            "frame_shape": (64, 64), "pixel_size_um": 1.0,
            "fov_positions_um": {("A1", 0): (0.0, 0.0), ("A2", 0): (2000.0, 0.0)}}
    ladder = plate_ladder(meta)
    pv = InMemoryMultiscale(ladder, channels=["c0"], dtype=np.uint16, budget_bytes=8 << 20, t=1)
    coarse = max(i for i in range(len(ladder.geometry)) if not ladder.is_fov_level(i))
    key = ladder.geometry.levels[coarse].keys[0]
    box = ladder.cell_bbox_um(coarse, key)
    assert pv.read_tile(TileDescriptor(coarse, key, "c0", box, 1)).shape[0] == ladder.tile_px
    with pytest.raises(KeyError, match="timepoint 1"):
        pv.read_tile(TileDescriptor(coarse, key, "c0", box, 0))


def test_the_plate_view_asks_for_the_timepoint_it_says_it_is_showing(
        qapp, monkeypatch, multi_time_point_dataset):
    """End to end: set_time_point used to touch neither _tile_src nor _tile_cache, so the plate
    said "timepoint 2" over frame 0's tiles."""
    monkeypatch.setenv("SQUIDXPLORER_DEEP_ZOOM", "1")
    from squidxplorer import _viewer as V

    root, _ = multi_time_point_dataset
    reader, meta, ladder = _open(root)
    region = meta["regions"][0]
    ov = V.PlateOverview(["A"], ["1"], {(0, 0): region})
    assert ov.set_tile_source(reader, meta), "deep zoom did not arm on the timepoint fixture"

    channel = str(meta["channels"][0]["name"])
    frames = {}
    for t in (0, 2, 0):
        ov.set_time_point(t)
        assert ov._time_point == t
        tile = ov._tile_src.read_tile(_fov_desc(ladder, channel, ov._time_point))
        frames[t] = int(tile.max())
        assert frames[t] == _expected_mip(t, 0), (
            f"the plate reports timepoint {ov._time_point} while its tiles carry pixel "
            f"{frames[t]}, i.e. timepoint {frames[t] // 100}")
    assert frames[0] != frames[2]
    ov.clear_tile_source()


def test_every_tile_the_plate_view_enumerates_carries_its_timepoint(
        qapp, monkeypatch, multi_time_point_dataset):
    """The producer's own descriptors, not a hand-built one: the stamp has to be in the widget."""
    monkeypatch.setenv("SQUIDXPLORER_DEEP_ZOOM", "1")
    from squidxplorer import _plate_overview as PO
    from squidxplorer import _viewer as V

    root, _ = multi_time_point_dataset
    reader, meta, _ = _open(root)
    ov = V.PlateOverview(["A"], ["1"], {(0, 0): meta["regions"][0]})
    assert ov.set_tile_source(reader, meta)
    ov.resize(800, 800)
    ov._cd = PO._CELL * 4                       # zoomed past the crossover: tiles engage
    ov.set_time_point(2)

    stamped = {d.t for d, _ in ov._visible_fov_tiles()}
    assert stamped == {2}, (
        f"the plate is showing timepoint 2 and enumerated tiles for timepoint(s) {stamped or 'none'}")
    ov.clear_tile_source()
