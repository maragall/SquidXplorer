"""read_pixels: ONE world-micrometre address for pixels, over the tile seam's two adapters."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._pixels import PixelRequest, read_pixels
from squidxplorer._tilesource import ReaderTileSource, ZarrPyramidSource, plate_ladder

from .test_plate_tiles import FRAME, PX_UM, FakeReader, _meta


def _reader_src(meta):
    ladder = plate_ladder(meta, tile_px=64)
    return ReaderTileSource(FakeReader(), meta, ladder), ladder


def test_the_full_world_composes_each_fov_into_its_own_quadrant():
    """A misplacement is a wrong NUMBER: every fake FOV is a constant plane of fov+1."""
    meta = _meta(2)
    src, ladder = _reader_src(meta)
    res = read_pixels(src, ladder, PixelRequest(
        bbox_um=ladder.world_bbox_um, out_px=(128, 128), channel="488"))

    assert res.pixels.shape == (128, 128)
    assert res.um_per_px == pytest.approx(PX_UM)
    # world (0,0) is FOV 0's corner; quadrants are row-major fov order
    assert res.pixels[32, 32] == 1      # top-left  -> fov 0
    assert res.pixels[32, 96] == 2      # top-right -> fov 1
    assert res.pixels[96, 32] == 3      # bottom-left -> fov 2
    assert res.pixels[96, 96] == 4      # bottom-right -> fov 3


def test_the_ladder_pick_is_INSIDE_a_small_ceiling_still_answers_correctly():
    """The caller spends fewer pixels; the LEVEL choice is not its problem."""
    meta = _meta(2)
    src, ladder = _reader_src(meta)
    res = read_pixels(src, ladder, PixelRequest(
        bbox_um=ladder.world_bbox_um, out_px=(32, 32), channel="488"))

    assert res.pixels.shape == (32, 32)
    assert res.um_per_px == pytest.approx(4 * PX_UM)
    assert res.pixels[8, 8] == 1 and res.pixels[8, 24] == 2
    assert res.pixels[24, 8] == 3 and res.pixels[24, 24] == 4


def test_the_raster_is_square_pixeled_to_the_bbox_and_SAYS_its_pitch():
    """out_px is a ceiling, never a promise: a wide bbox under a square ceiling comes back
    wide, and um_per_px is the truth a consumer scales by."""
    meta = _meta(2)
    src, ladder = _reader_src(meta)
    x0, y0, x1, y1 = ladder.world_bbox_um
    res = read_pixels(src, ladder, PixelRequest(
        bbox_um=(x0, y0, x1, y0 + (y1 - y0) / 2), out_px=(128, 128), channel="488"))

    assert res.pixels.shape == (64, 128), "the aspect followed the ceiling, not the bbox"
    assert res.um_per_px == pytest.approx(PX_UM)


def test_a_window_over_nothing_is_an_EMPTY_answer_not_a_crash():
    meta = _meta(2)
    src, ladder = _reader_src(meta)
    res = read_pixels(src, ladder, PixelRequest(
        bbox_um=(10_000.0, 10_000.0, 10_064.0, 10_064.0), out_px=(64, 64), channel="488"))
    assert res.pixels.shape == (64, 64)
    assert not res.pixels.any()


def test_a_nonsense_request_is_refused_by_name():
    meta = _meta(2)
    src, ladder = _reader_src(meta)
    with pytest.raises(ValueError, match="out_px"):
        read_pixels(src, ladder, PixelRequest(bbox_um=ladder.world_bbox_um,
                                              out_px=(0, 64), channel="488"))
    with pytest.raises(ValueError, match="bbox_um"):
        read_pixels(src, ladder, PixelRequest(bbox_um=(1.0, 1.0, 1.0, 1.0),
                                              out_px=(64, 64), channel="488"))


def test_the_TWO_adapters_answer_one_request_with_the_same_pixels(tmp_path):
    """The seam's proof: the reader-backed source and the written store answer ONE request
    identically — pixel for pixel — so a consumer cannot tell which adapter served it."""
    from squidxplorer._output import write_from_stream

    meta = _meta(2)
    writer_meta = dict(meta)
    writer_meta.update({
        "regions": ["A1"],
        "channels": [{"name": "488", "display_color": "#00FF00"}],
    })
    stream = iter([("A1", f, np.full((1, 1, 1, *FRAME), f + 1, dtype=np.uint16))
                   for f in range(4)])
    write_from_stream(writer_meta, stream, tmp_path, n_fovs=None, tiff=False)

    rsrc, ladder = _reader_src(meta)
    zsrc = ZarrPyramidSource(tmp_path)
    req = PixelRequest(bbox_um=ladder.world_bbox_um, out_px=(128, 128), channel="488")

    ours = read_pixels(rsrc, ladder, req)
    theirs = read_pixels(zsrc, zsrc.ladder, req)

    assert ours.um_per_px == pytest.approx(theirs.um_per_px)
    np.testing.assert_array_equal(ours.pixels, theirs.pixels)
