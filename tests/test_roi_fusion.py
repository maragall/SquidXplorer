"""Native cross-FOV ROI fusion geometry (``roi_window_px`` + ``read_brick``).

The 3D-of-an-ROI path fuses the FOVs the ROI overlaps at native resolution, all z, cropped to the
box. Placement must match the 2D mosaic exactly (each FOV pasted at its stage-pixel offset), or the
3D volume shows the wrong tissue. This pins that geometry with a synthetic two-FOV reader.

THERE IS ONE CONVERSION AND ONE FUSER, and this file exists to keep it that way. Until 2026-08-06
``_napari3d.native_roi_volume`` was a second copy of both -- and ``roi_window_px``'s own docstring
already claimed to BE that copy "lifted out, so the bricked path and the single-volume path cannot
drift apart". They had drifted, in three measurable ways, and the copy was the wrong one every
time (see the three regression tests at the bottom). It had no caller outside this file.
"""

import numpy as np
import pytest

from squidmip import _napari3d
from squidmip._napari3d import read_brick, roi_window_px


class _FakeReader:
    def read(self, region, fov, ch, z):
        base = 10 if int(fov) == 0 else 20        # FOV0 -> 10+z, FOV1 -> 20+z, so the seam is visible
        return np.full((4, 4), base + int(z), dtype=np.uint16)


def _meta(**over):
    m = {
        "fov_positions_um": {("A1", 0): (0.0, 0.0), ("A1", 1): (4.0, 0.0)},  # FOV1 is 4um right
        "fovs_per_region": {"A1": [0, 1]},
        "pixel_size_um": 1.0,
        "frame_shape": [4, 4],
        "z_levels": [0, 1],
        "channels": [{"name": "c0"}],
    }
    m.update(over)
    return m


def _roi_volume(meta, roi_bbox_um, channels=("c0",)):
    """THE path a drawn ROI's 3D takes: box -> window -> voxels. Both halves, no third spelling."""
    window = roi_window_px(meta, "A1", roi_bbox_um)
    if window is None:
        return {}
    out = {}
    for ch in channels:
        vol = read_brick(_FakeReader(), meta, "A1", window, ch)
        if vol is not None:
            out[ch] = vol
    return out


def test_roi_fusion_straddles_two_fovs():
    # ROI x[2,6] y[1,3] straddles the FOV0/FOV1 seam at x=4.
    v = _roi_volume(_meta(), (2.0, 1.0, 6.0, 3.0))["c0"]
    assert v.shape == (2, 2, 4)                    # (z, H, W)
    assert (v[0, :, 0:2] == 10).all()              # left half from FOV0
    assert (v[0, :, 2:4] == 20).all()              # right half from FOV1
    assert (v[1, :, 0:2] == 11).all() and (v[1, :, 2:4] == 21).all()   # z=1 layer


def test_roi_fusion_full_z_depth_preserved():
    vols = _roi_volume(_meta(), (0.0, 0.0, 4.0, 4.0))
    assert vols["c0"].shape[0] == 2                # both z levels survive (this was the "single z" bug)


def test_roi_fully_inside_one_fov():
    v = _roi_volume(_meta(), (0.0, 0.0, 3.0, 3.0))["c0"]
    assert v.shape == (2, 3, 3)
    assert (v[0] == 10).all()                      # entirely FOV0


# --- the three ways the deleted second copy disagreed with this one -----------------------------
#
# Each of these passes on `roi_window_px` + `read_brick` and FAILED on `native_roi_volume`, whose
# answer is quoted in the message. They are the reason the collapse is not merely tidier.

def test_a_box_past_the_region_edge_is_clipped_to_the_pixels_that_exist():
    """The mosaic is 8 px wide. A box 40 um across must not become a 40 px volume of zeros."""
    v = _roi_volume(_meta(), (0.0, 0.0, 40.0, 4.0))["c0"]
    assert v.shape == (2, 4, 8), (
        f"a box wider than the mosaic returned {v.shape}; the deleted copy returned (2, 4, 40) "
        "-- 32 of its 40 columns were zeros presented as acquired data"
    )


def test_a_box_drawn_up_and_to_the_left_is_the_same_box():
    """(x1, y1) before (x0, y0) is an ordinary drag direction, not an empty selection."""
    forward = _roi_volume(_meta(), (2.0, 1.0, 6.0, 3.0))["c0"]
    backward = _roi_volume(_meta(), (6.0, 3.0, 2.0, 1.0))
    assert "c0" in backward, (
        "a bottom-right-to-top-left drag returned NOTHING; the deleted copy compared the raw "
        "corners without min/max, so c1 <= c0 and it returned {} for a box it could see"
    )
    assert np.array_equal(backward["c0"], forward)


def test_an_acquisition_with_no_pixel_size_is_refused_not_guessed():
    meta = _meta(pixel_size_um=None)
    assert roi_window_px(meta, "A1", (2.0, 1.0, 6.0, 3.0)) is None, (
        "an ROI was converted without a pixel size; the deleted copy fell back to `or 1.0` and "
        "returned a plausible (2, 2, 4) crop of the wrong tissue"
    )


def test_there_is_no_second_roi_fuser_on_the_module():
    """A structural guard: adding the copy back would break this.

    `roi_window_px` is the ONE box->window conversion and `read_brick` the ONE window->voxels
    fuser. A sibling that does both again is what this file's header is about.
    """
    assert not hasattr(_napari3d, "native_roi_volume"), (
        "native_roi_volume is back. It is roi_window_px + read_brick open-coded, and every time "
        "it existed it disagreed with them -- see the three tests above."
    )


@pytest.mark.parametrize("box", [(2.0, 1.0, 6.0, 3.0), (0.0, 0.0, 4.0, 4.0), (0.0, 0.0, 3.0, 3.0)])
def test_the_window_is_the_shape_of_the_volume(box):
    """The two halves agree on size, which is the property a single copy cannot get wrong."""
    r0, r1, c0, c1 = roi_window_px(_meta(), "A1", box)
    assert _roi_volume(_meta(), box)["c0"].shape[1:] == (r1 - r0, c1 - c0)
