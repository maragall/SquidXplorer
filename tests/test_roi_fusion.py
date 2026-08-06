"""Native cross-FOV ROI fusion geometry (squidmip._napari3d.native_roi_volume).

The 3D-of-an-ROI path fuses the FOVs the ROI overlaps at native resolution, all z, cropped to the
box. Placement must match the 2D mosaic exactly (each FOV pasted at its stage-pixel offset), or the
3D volume shows the wrong tissue. This pins that geometry with a synthetic two-FOV reader.
"""

import numpy as np

from squidmip._napari3d import native_roi_volume


class _FakeReader:
    """The reader protocol's own signature: ``read(region, fov, channel, z, t=0)``.

    It said ``read(self, region, fov, ch, z)`` -- the third parameter RENAMED and no ``t`` at all
    -- so a keyword call would have raised and a timepoint could not be passed. Both real readers
    spell it ``channel`` (`tests/test_reader_protocol.py` pins the four signatures), and the
    3D path this file drives calls positionally, which is the only reason the rename never bit.
    """

    def __init__(self):
        self.reads = []

    def read(self, region, fov, channel, z, t=0):
        self.reads.append((region, int(fov), channel, int(z), int(t)))
        base = 10 if int(fov) == 0 else 20        # FOV0 -> 10+z, FOV1 -> 20+z, so the seam is visible
        return np.full((4, 4), base + int(z), dtype=np.uint16)


def _meta():
    return {
        "fov_positions_um": {("A1", 0): (0.0, 0.0), ("A1", 1): (4.0, 0.0)},  # FOV1 is 4um right
        "fovs_per_region": {"A1": [0, 1]},
        "pixel_size_um": 1.0,
        "frame_shape": [4, 4],
        "z_levels": [0, 1],
        "channels": [{"name": "c0"}],
    }


def test_roi_fusion_straddles_two_fovs():
    # ROI x[2,6] y[1,3] straddles the FOV0/FOV1 seam at x=4.
    vols = native_roi_volume(_FakeReader(), _meta(), "A1", (2.0, 1.0, 6.0, 3.0), ["c0"])
    v = vols["c0"]
    assert v.shape == (2, 2, 4)                    # (z, H, W)
    assert (v[0, :, 0:2] == 10).all()              # left half from FOV0
    assert (v[0, :, 2:4] == 20).all()              # right half from FOV1
    assert (v[1, :, 0:2] == 11).all() and (v[1, :, 2:4] == 21).all()   # z=1 layer


def test_roi_fusion_full_z_depth_preserved():
    vols = native_roi_volume(_FakeReader(), _meta(), "A1", (0.0, 0.0, 4.0, 4.0), ["c0"])
    assert vols["c0"].shape[0] == 2                # both z levels survive (this was the "single z" bug)


def test_roi_fully_inside_one_fov():
    vols = native_roi_volume(_FakeReader(), _meta(), "A1", (0.0, 0.0, 3.0, 3.0), ["c0"])
    v = vols["c0"]
    assert v.shape == (2, 3, 3)
    assert (v[0] == 10).all()                      # entirely FOV0
