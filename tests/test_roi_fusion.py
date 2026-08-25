"""Native cross-FOV ROI fusion geometry (`roi_window_px` + `read_brick`)."""

import numpy as np

from squidxplorer._napari3d import read_brick, roi_window_px


class _FakeReader:
    """The reader protocol's own signature: `read(region, fov, channel, z_level, time_point=0)`."""

    def __init__(self):
        self.reads = []

    def read(self, region, fov, channel, z_level, time_point=0):
        self.reads.append((region, int(fov), channel, int(z_level), int(time_point)))
        base = 10 if int(fov) == 0 else 20        # FOV0 -> 10+z, FOV1 -> 20+z, so the seam is visible
        return np.full((4, 4), base + int(z_level), dtype=np.uint16)


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
    """The path a drawn ROI's 3D takes: box -> window -> voxels."""
    window = roi_window_px(meta, "A1", roi_bbox_um)
    if window is None:
        return {}
    out = {}
    for ch in channels:
        vol = read_brick(_FakeReader(), meta, "A1", window, ch)
        if vol is not None:
            out[ch] = vol
    return out


def test_roi_fusion_full_z_depth_preserved():
    vols = _roi_volume(_meta(), (0.0, 0.0, 4.0, 4.0))
    assert vols["c0"].shape[0] == 2                # both z levels survive (this was the "single z" bug)


# --- the three ways the deleted second copy disagreed with this one -----------------------------

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


def test_the_window_is_the_shape_of_the_volume():
    box = (2.0, 1.0, 6.0, 3.0)
    r0, r1, c0, c1 = roi_window_px(_meta(), "A1", box)
    assert _roi_volume(_meta(), box)["c0"].shape[1:] == (r1 - r0, c1 - c0)
