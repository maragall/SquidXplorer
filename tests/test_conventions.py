"""The named conventions: two FOV-box frames and two pitches, refused by type when mixed.

The 195.9 um half-frame gap between the box conventions and the fuse-decimation gap between the
pitches each rendered as a plausible image when crossed silently; these tests pin that crossing
either seam is a NAMED act or a TypeError with a sentence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from squidxplorer._conventions import (
    AcqPitchUm,
    CenterBoxUm,
    DisplayPitchUm,
    TopLeftBoxUm,
    acq_um,
    display_um,
)


class TestBoxConventionsCrossByNameOnly:
    def test_conversion_round_trips_to_the_pixel(self):
        """Corner -> centre -> corner lands on the same micrometres, well under one pixel."""
        tl = TopLeftBoxUm(45698.0, 25553.0, 46475.10531696118, 26330.10531696118)
        back = tl.to_center().to_top_left()
        assert back.bbox() == pytest.approx(tl.bbox(), abs=1e-9)
        c = CenterBoxUm(cx=1000.0, cy=2000.0, w=777.1, h=777.1)
        rt = c.to_top_left().to_center()
        assert (rt.cx, rt.cy, rt.w, rt.h) == pytest.approx((c.cx, c.cy, c.w, c.h), abs=1e-9)

    def test_both_encodings_answer_the_same_bbox_for_one_physical_box(self):
        tl = TopLeftBoxUm(10.0, 20.0, 110.0, 70.0)
        assert tl.to_center().bbox() == pytest.approx(tl.bbox(), abs=1e-12)

    def test_a_top_left_box_unpacks_as_its_own_bbox(self):
        """Iteration/indexing of a corner box IS its convention, so no crossing can happen."""
        box = TopLeftBoxUm(1.0, 2.0, 3.0, 4.0)
        x0, y0, x1, y1 = box
        assert (x0, y0, x1, y1) == box.bbox() == (box[0], box[1], box[2], box[3])
        assert len(box) == 4
        assert tuple(np.asarray(box)) == box.bbox()   # the drawing code's np.array path

    def test_a_centre_box_refuses_to_unpack_with_the_named_sentence(self):
        """(cx, cy, w, h) read as (x0, y0, x1, y1) is the 195.9 um mix-up; it must not unpack."""
        box = CenterBoxUm(cx=100.0, cy=200.0, w=50.0, h=50.0)
        with pytest.raises(TypeError, match=r"195\.9"):
            tuple(box)
        with pytest.raises(TypeError, match=r"\.bbox\(\) or \.to_top_left\(\)"):
            box[0]


class TestPitchesAreNotBareNumbers:
    @pytest.mark.parametrize("cls", [AcqPitchUm, DisplayPitchUm])
    def test_float_of_a_pitch_raises_with_the_named_sentence(self, cls):
        with pytest.raises(TypeError, match=r"read \.um"):
            float(cls(0.752))

    @pytest.mark.parametrize("cls", [AcqPitchUm, DisplayPitchUm])
    def test_arithmetic_on_a_pitch_raises_from_either_side(self, cls):
        p = cls(0.752)
        for op in (lambda: p * 2, lambda: 2 * p, lambda: p / 2, lambda: 2 / p,
                   lambda: p + 1, lambda: 1 + p, lambda: p - 1, lambda: 1 - p):
            with pytest.raises(TypeError, match=r"read \.um"):
                op()

    def test_um_is_the_one_door_and_the_px_helpers_use_it(self):
        p = AcqPitchUm(0.5)
        assert p.um == 0.5
        assert p.px_from_um(10.0) == 20.0
        assert p.um_from_px(4) == 2.0

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_a_pitch_that_is_not_a_pitch_is_refused_at_construction(self, bad):
        with pytest.raises(ValueError, match="finite positive"):
            AcqPitchUm(bad)


class TestMixingPitchesFailsByName:
    def test_a_display_pitch_where_the_acquisition_pitch_is_required(self):
        assert acq_um(AcqPitchUm(0.752)) == 0.752
        assert acq_um(0.752) == 0.752              # legacy bare-float callers keep working
        with pytest.raises(TypeError, match="ACQUISITION pitch is required"):
            acq_um(DisplayPitchUm(1.504))

    def test_an_acq_pitch_where_the_displayed_pitch_is_required(self):
        assert display_um(DisplayPitchUm(1.504)) == 1.504
        assert display_um(1.504) == 1.504
        with pytest.raises(TypeError, match="DISPLAYED pitch is required"):
            display_um(AcqPitchUm(0.752))

    def test_the_roi_clamp_counts_acquisition_pixels_by_type(self):
        """`clamp_bbox_um`'s contract is acquisition pixels; the wrong pitch now raises instead
        of passing a 2x box under a one-texture promise."""
        from squidxplorer import _bricks

        box = (0.0, 0.0, 1e6, 1e6)
        assert _bricks.clamp_bbox_um(box, AcqPitchUm(0.752), 2048) == \
            _bricks.clamp_bbox_um(box, 0.752, 2048)
        with pytest.raises(TypeError, match="ACQUISITION pitch is required"):
            _bricks.clamp_bbox_um(box, DisplayPitchUm(1.504), 2048)
        with pytest.raises(TypeError, match="ACQUISITION pitch is required"):
            _bricks.ceiling_line(2048, DisplayPitchUm(1.504), measured=True)

    def test_fov_placement_counts_acquisition_pixels_by_type(self):
        from squidxplorer._placement import fov_offsets_px

        positions = {("A1", 0): (0.0, 0.0), ("A1", 1): (500.0, 0.0)}
        typed = fov_offsets_px(positions, "A1", [0, 1], AcqPitchUm(1.0))
        assert typed == fov_offsets_px(positions, "A1", [0, 1], 1.0)
        with pytest.raises(TypeError, match="ACQUISITION pitch is required"):
            fov_offsets_px(positions, "A1", [0, 1], DisplayPitchUm(2.0))


class TestTheProducersReturnTheirOwnConvention:
    _META = {
        "regions": ["A1"],
        "fovs_per_region": {"A1": [0, 1]},
        "channels": [],
        "fov_positions_um": {("A1", 0): (0.0, 0.0), ("A1", 1): (500.0, 0.0)},
        "pixel_size_um": 1.0,
        "frame_shape": (100, 100),
    }

    def test_fov_bboxes_um_returns_centre_boxes(self):
        from squidxplorer._tilesource import fov_bboxes_um

        boxes = fov_bboxes_um(self._META["fov_positions_um"], (100, 100), 1.0)
        assert all(isinstance(b, CenterBoxUm) for b in boxes.values())
        assert boxes[("A1", 0)].bbox() == pytest.approx((-50.0, -50.0, 50.0, 50.0))

    def test_mosaic_fov_bboxes_um_returns_top_left_boxes(self):
        from squidxplorer._mosaic_source import mosaic_fov_bboxes_um

        boxes = mosaic_fov_bboxes_um(self._META, "A1")
        assert all(isinstance(b, TopLeftBoxUm) for b in boxes.values())
        assert boxes[0].bbox() == pytest.approx((0.0, 0.0, 100.0, 100.0))

    def test_the_half_frame_gap_is_visible_through_the_named_conversion(self):
        """The anchor FOV's two boxes sit exactly half a frame apart; crossing to compare them
        is spelled `.to_top_left()`, never an unpack."""
        from squidxplorer._mosaic_source import mosaic_fov_bboxes_um
        from squidxplorer._tilesource import fov_bboxes_um

        mosaic = mosaic_fov_bboxes_um(self._META, "A1")[0]
        plate = fov_bboxes_um(self._META["fov_positions_um"], (100, 100), 1.0)[("A1", 0)]
        gap = [a - b for a, b in zip(plate.to_top_left().bbox(), mosaic.bbox())]
        assert gap == pytest.approx([-50.0, -50.0, -50.0, -50.0])   # half a 100 um frame
