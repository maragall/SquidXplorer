"""Bricking geometry and policy (squidmip._bricks) + the per-brick read (squidmip._napari3d).

These are the decisions that cannot be checked by looking at the screen: whether the tiles COVER
the volume exactly, whether a brick lands on the world micrometre it claims, and whether the stride
policy ever silently coarsens something the display could have resolved. The GL limit itself cannot
be probed headless (no GL context), which is precisely why the policy is a pure function here.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip import _bricks
from squidmip._napari3d import read_brick, region_origin_um, roi_window_px


# -- plan: the tiles must COVER, and never overlap --------------------------------------
@pytest.mark.parametrize("h,w,limit,edge", [
    (2084, 2084, 2048, 1024),
    (11538, 9645, 2048, 1024),      # a whole stitched region on the 10x set
    (1, 1, 2048, 1024),
    (2049, 100, 2048, 2048),        # one voxel over the limit: must become two bricks
    (5000, 5000, 512, 4096),        # edge is clamped DOWN to the live limit, never up
])
def test_plan_tiles_cover_exactly(h, w, limit, edge):
    bricks = _bricks.plan(h, w, limit=limit, edge=edge)
    assert bricks, "a positive-area plane must produce at least one brick"
    covered = np.zeros((h, w), dtype=np.uint8)
    for b in bricks:
        assert b.height <= limit and b.width <= limit, "a brick must fit ONE GL texture"
        covered[b.r0:b.r1, b.c0:b.c1] += 1
    assert covered.min() == 1 and covered.max() == 1, "bricks must partition, not overlap or gap"


def test_plan_empty_plane_is_no_bricks():
    assert _bricks.plan(0, 10, limit=2048) == ()
    assert _bricks.plan(10, -1, limit=2048) == ()


def test_a_volume_that_fits_one_texture_is_not_bricked():
    """The single-layer fast path is gallery-view's recipe and must survive: one brick, one layer."""
    assert _bricks.fits_single_texture(2000, 1500, 10, 2048)
    assert len(_bricks.plan(2000, 1500, limit=2048, edge=2048)) == 1
    assert not _bricks.fits_single_texture(2049, 1500, 10, 2048)


# -- placement: a brick must land on the micrometre it claims ----------------------------
def test_translate_is_the_bricks_own_world_corner():
    b = _bricks.Brick(iy=1, ix=2, r0=1024, r1=2048, c0=2048, c1=3072)
    z, y, x = b.translate_um((0.0, 100.0, 50.0), 0.752, 0.752)
    assert z == 0.0
    assert y == pytest.approx(100.0 + 1024 * 0.752)
    assert x == pytest.approx(50.0 + 2048 * 0.752)


def test_translate_does_not_move_with_the_stride():
    """A coarse brick must be replaceable by a fine one without anything shifting on screen: the
    stride rides on `scale`, never on `translate`."""
    b = _bricks.Brick(iy=0, ix=1, r0=0, r1=1024, c0=1024, c1=2048)
    assert b.translate_um((0, 0, 0), 1.0, 1.0) == b.translate_um((0, 0, 0), 1.0, 1.0)
    # sampled shape shrinks with the stride; the corner does not move
    assert b.sampled_shape(10, 1) == (10, 1024, 1024)
    assert b.sampled_shape(10, 4) == (10, 256, 256)


def test_sampled_shape_matches_numpy_striding_exactly():
    b = _bricks.Brick(iy=0, ix=0, r0=0, r1=1000, c0=0, c1=999)
    for step in (1, 2, 4, 8, 16):
        arr = np.zeros((3, b.height, b.width), np.uint16)
        assert b.sampled_shape(3, step) == arr[:, ::step, ::step].shape


# -- stride policy: never coarser than the screen, always a power of two ------------------
def test_stride_is_native_at_or_past_one_to_one():
    """The whole "full render" claim: if one screen pixel covers one voxel or less, stride is 1."""
    assert _bricks.uniform_step(0.752, 0.752) == 1
    assert _bricks.uniform_step(0.1, 0.752) == 1        # zoomed in past native
    assert _bricks.uniform_step(1.4, 0.752) == 1        # 1.86 voxels/px still rounds DOWN to 1


def test_stride_grows_with_zoom_out_and_is_a_power_of_two():
    # 8 voxels behind one screen pixel -> 8; 100 -> 64 (clamped by max_step)
    assert _bricks.uniform_step(0.752 * 8, 0.752) == 8
    assert _bricks.uniform_step(0.752 * 100, 0.752) == 64
    for ratio in (3, 5, 6, 7, 9, 15, 31):
        s = _bricks.uniform_step(0.752 * ratio, 0.752)
        assert s & (s - 1) == 0, f"stride {s} is not a power of two: bricks would misalign"
        assert s <= ratio, "the stride must never be coarser than the screen needs"


def test_stride_survives_a_degenerate_camera():
    assert _bricks.uniform_step(float("nan"), 0.752) == 1
    assert _bricks.uniform_step(1.0, 0.0) == 1
    assert _bricks.uniform_step(-5.0, 0.752) == 1


def test_power_of_two_stride_keeps_every_brick_on_one_global_lattice():
    """The alignment property the power-of-two rule exists for: with edge % step == 0 the samples
    of brick N continue the samples of brick N-1 without a jump."""
    edge = 1024
    for step in (1, 2, 4, 8, 16, 32, 64):
        assert edge % step == 0
        # brick k starts at k*edge; its first sample must sit on the global k*edge lattice point
        for k in range(4):
            assert (k * edge) % step == 0


# -- cull: only what the camera sees, centre first ---------------------------------------
def test_cull_keeps_only_intersecting_bricks_centre_first():
    bricks = _bricks.plan(4096, 4096, limit=2048, edge=1024)      # 4x4 grid
    origin, py = (0.0, 0.0), 1.0
    view = (1000.0, 1000.0, 1400.0, 1400.0)                       # a small window near brick (1,1)
    kept = _bricks.cull(bricks, origin_um=origin, py=py, px=py, view_um=view)
    assert kept, "a view inside the volume must keep something"
    assert len(kept) < len(bricks), "an offscreen brick must not be resident"
    assert kept[0].iy == 1 and kept[0].ix == 1, "the brick under the cursor must be loaded first"


def test_cull_with_no_camera_keeps_everything():
    bricks = _bricks.plan(2048, 2048, limit=2048, edge=1024)
    assert _bricks.cull(bricks, origin_um=(0, 0), py=1.0, px=1.0, view_um=None) == bricks


def test_margin_keeps_a_brick_just_outside_the_view():
    bricks = _bricks.plan(2048, 1024, limit=2048, edge=1024)      # two bricks stacked in y
    view = (0.0, 0.0, 900.0, 900.0)                               # only brick 0 truly intersects
    tight = _bricks.cull(bricks, origin_um=(0, 0), py=1.0, px=1.0, view_um=view)
    padded = _bricks.cull(bricks, origin_um=(0, 0), py=1.0, px=1.0, view_um=view, margin_um=300.0)
    assert len(tight) == 1 and len(padded) == 2


# -- budget: coarsen, do not punch holes -------------------------------------------------
def test_budget_coarsens_the_stride_rather_than_dropping_bricks():
    """A dropped brick is a black hole the user cannot distinguish from empty tissue; a coarser
    stride is uniform and visible. The budget must reach for the second one."""
    bricks = _bricks.plan(8192, 8192, limit=2048, edge=1024)      # 64 bricks
    b = _bricks.plan_budget(bricks, nz=10, itemsize=2, step=1,
                            bytes_limit=64 << 20, n_channels=1)
    assert b.dropped == 0, "no hole may be punched while a coarser stride is still available"
    assert b.step > 1 and b.step & (b.step - 1) == 0
    assert b.bytes_resident <= b.bytes_limit and b.within


def test_budget_leaves_a_fitting_set_at_native_stride():
    bricks = _bricks.plan(2048, 2048, limit=2048, edge=1024)      # 4 bricks, 10z uint16 = 84 MB
    b = _bricks.plan_budget(bricks, nz=10, itemsize=2, step=1,
                            bytes_limit=512 << 20, n_channels=1)
    assert b.step == 1, "a set that fits must NOT be coarsened"
    assert len(b.bricks) == len(bricks) and b.dropped == 0


def test_budget_accounts_for_every_channel():
    bricks = _bricks.plan(2048, 2048, limit=2048, edge=1024)
    one = _bricks.plan_budget(bricks, nz=10, itemsize=2, step=1,
                              bytes_limit=100 << 20, n_channels=1)
    four = _bricks.plan_budget(bricks, nz=10, itemsize=2, step=1,
                               bytes_limit=100 << 20, n_channels=4)
    assert four.step > one.step, "four channels cost four times as much and must be planned as such"


def test_budget_finally_drops_only_when_the_stride_is_exhausted():
    bricks = _bricks.plan(65536, 65536, limit=2048, edge=1024)
    b = _bricks.plan_budget(bricks, nz=10, itemsize=2, step=1,
                            bytes_limit=1 << 20, n_channels=1)
    assert b.step == 64, "the stride ceiling must be reached before anything is withheld"
    assert b.dropped > 0 and b.bytes_resident <= b.bytes_limit


# -- the read: a brick's voxels are the mosaic's voxels -----------------------------------
class _Reader:
    """Each FOV is filled with a value derived from its own index, so a mis-paste is visible."""

    def read(self, region, fov, channel, z):
        return np.full((4, 4), fov * 10 + z, dtype=np.uint16)


def _meta():
    return {
        "pixel_size_um": 1.0,
        "frame_shape": (4, 4),
        "z_levels": [0, 1],
        "channels": [{"name": "c0"}],
        "fovs_per_region": {"A1": [0, 1]},
        # two FOVs side by side in x, so the mosaic is 4 rows x 8 cols
        "fov_positions_um": {("A1", 0): (0.0, 0.0), ("A1", 1): (4.0, 0.0)},
    }


def test_region_origin_is_the_min_of_the_stage_positions():
    assert region_origin_um(_meta(), "A1") == (0.0, 0.0)
    assert region_origin_um({}, "A1") is None


def test_roi_window_is_clipped_to_the_region():
    w = roi_window_px(_meta(), "A1", (-5.0, -5.0, 100.0, 100.0))
    assert w == (0, 4, 0, 8), "a box dragged past the edge crops to what exists"
    assert roi_window_px(_meta(), "A1", (50.0, 50.0, 60.0, 60.0)) is None


def test_read_brick_fuses_across_the_fov_seam():
    """The brick straddling both FOVs must carry BOTH, at the right columns."""
    got = read_brick(_Reader(), _meta(), "A1", (0, 4, 0, 8), "c0")
    assert got.shape == (2, 4, 8)
    assert (got[0, :, :4] == 0).all(), "FOV 0 at z 0"
    assert (got[0, :, 4:] == 10).all(), "FOV 1 at z 0, pasted at its own column offset"
    assert (got[1, :, 4:] == 11).all(), "z is the leading axis"


def test_read_brick_reads_only_the_window_it_was_given():
    got = read_brick(_Reader(), _meta(), "A1", (0, 4, 4, 8), "c0")
    assert got.shape == (2, 4, 4) and (got[0] == 10).all()


def test_read_brick_strides_like_numpy():
    full = read_brick(_Reader(), _meta(), "A1", (0, 4, 0, 8), "c0")
    strided = read_brick(_Reader(), _meta(), "A1", (0, 4, 0, 8), "c0", step=2)
    assert strided.shape == (2, 2, 4)
    assert np.array_equal(strided, full[:, ::2, ::2])


def test_a_cancelled_read_returns_none_rather_than_a_partial_brick():
    """A half-read brick delivered as if it were the answer is a silent hole; None says so."""
    assert read_brick(_Reader(), _meta(), "A1", (0, 4, 0, 8), "c0",
                      should_stop=lambda: True) is None
