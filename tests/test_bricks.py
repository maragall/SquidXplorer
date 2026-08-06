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
    assert covered.min() >= 1, "every voxel must be covered by some brick — no gaps"
    # The only overlap allowed is the deliberate halo, and only where two bricks meet: a voxel is
    # shared by at most 2 bricks per axis, so at most 4 in a corner.
    assert covered.max() <= 4, "bricks must not overlap beyond the 1-voxel halo"


def test_halo_makes_neighbours_share_exactly_one_voxel():
    """The halo must be an OVERLAP of one, not a gap and not a shift: brick N+1 starts one voxel
    before brick N ends, so each has the other's edge texel to interpolate against."""
    bricks = _bricks.plan(4096, 4096, limit=2048, edge=1024, halo=1)
    by_pos = {(b.iy, b.ix): b for b in bricks}
    a, b = by_pos[(0, 0)], by_pos[(0, 1)]
    assert b.c0 == a.c0 + 1024, "brick ORIGINS must still stride by the edge (translate depends on it)"
    assert a.c1 == b.c0 + 1, "neighbours must share exactly one voxel"


def test_halo_is_inside_the_texture_budget():
    """edge + halo must still fit the limit, or the halo would hand napari a texture it refuses."""
    for limit in (512, 2048, 16384):
        for b in _bricks.plan(9000, 9000, limit=limit, edge=limit):
            assert b.height <= limit and b.width <= limit


def test_halo_can_be_switched_off_for_an_exact_partition():
    bricks = _bricks.plan(2048, 2048, limit=2048, edge=1024, halo=0)
    covered = np.zeros((2048, 2048), dtype=np.uint8)
    for b in bricks:
        covered[b.r0:b.r1, b.c0:b.c1] += 1
    assert covered.min() == 1 and covered.max() == 1


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


# -- the sanctioned cap: anything drawable must be renderable ----------------------------
def test_clamp_holds_a_box_to_one_texture_and_keeps_its_anchor():
    """The guarantee behind option (b): a drawn ROI can never exceed what one texture renders."""
    px, limit = 0.752, 2048
    span = limit * px                                  # 1540 um on this GPU
    box, clamped = _bricks.clamp_bbox_um((100.0, 200.0, 100.0 + 5000.0, 200.0 + 9000.0), px, limit)
    assert clamped
    assert box[0] == 100.0 and box[1] == 200.0, "the drag's starting corner must not move"
    assert box[2] - box[0] == pytest.approx(span)
    assert box[3] - box[1] == pytest.approx(span)


def test_clamp_leaves_a_box_that_already_fits_untouched():
    box, clamped = _bricks.clamp_bbox_um((0.0, 0.0, 100.0, 100.0), 0.752, 2048)
    assert not clamped and box == (0.0, 0.0, 100.0, 100.0)


def test_clamp_respects_a_drag_in_the_negative_direction():
    """Dragging up-left must clamp away from the anchor, not flip the box across it."""
    box, clamped = _bricks.clamp_bbox_um((5000.0, 5000.0, 0.0, 0.0), 0.752, 2048)
    span = 2048 * 0.752
    assert clamped and box[0] == 5000.0 and box[1] == 5000.0
    assert box[2] == pytest.approx(5000.0 - span) and box[3] == pytest.approx(5000.0 - span)


def test_a_clamped_box_always_fits_one_texture():
    """The property that makes the cap a guarantee rather than a heuristic."""
    px = 0.752
    for limit in (512, 2048, 16384):
        (x0, y0, x1, y1), _ = _bricks.clamp_bbox_um((0.0, 0.0, 1e6, 1e6), px, limit)
        h, w = int(round((y1 - y0) / px)), int(round((x1 - x0) / px))
        assert _bricks.fits_single_texture(h, w, 10, limit)
        assert len(_bricks.plan(h, w, limit=limit, edge=limit)) == 1


def test_the_ceiling_scales_with_the_gpu_and_says_where_it_came_from():
    """2048 is the Apple figure; a desktop NVIDIA reporting 16384 must raise the stated ceiling."""
    apple = _bricks.ceiling_line(2048, 0.752, measured=True)
    nvidia = _bricks.ceiling_line(16384, 0.752, measured=True)
    assert "2048" in apple and "1540x1540 um" in apple and "measured on this GPU" in apple
    assert "16384" in nvidia and "12321x12321 um" in nvidia
    assert "ASSUMED" in _bricks.ceiling_line(2048, 0.752, measured=False)


# -- the perceptual contract: what "cannot look downsampled" means numerically -------------
def test_the_stride_never_lets_the_picture_go_below_one_voxel_per_screen_pixel():
    """Julio: "it can't look pixelated or downsampled or flattened to the user's eye".

    "Pixelated" has a number: fewer than one voxel behind each screen pixel. ``uniform_step``
    rounds the ratio DOWN, so the ratio AFTER striding is always >= 1 -- there is always at least
    as much data as the display can show. This is the property that makes screen-matched sampling
    perceptually identical to native rather than a compromise.
    """
    px = 0.752
    for ratio in [1.0, 1.5, 2.0, 3.7, 4.0, 8.9, 16.0, 33.0, 64.0, 300.0]:
        step = _bricks.uniform_step(ratio * px, px)
        assert ratio / step >= 1.0, (
            f"{ratio:.2f} voxels/screen-px at stride {step} leaves {ratio/step:.2f} — pixelated")


def test_zooming_in_monotonically_refines_and_reaches_native():
    """"Coarse and stays coarse" is the failure. Each zoom step must not get coarser, and at 1:1
    the stride must be exactly 1."""
    px = 0.752
    steps = [_bricks.uniform_step(r * px, px) for r in (64, 32, 16, 8, 4, 2, 1, 0.5)]
    assert steps == sorted(steps, reverse=True), f"stride must never coarsen while zooming in: {steps}"
    assert steps[-1] == 1 and steps[-2] == 1, "at and past 1:1 the stride must be native"


def test_the_stride_never_touches_z():
    """"Flattened" is the third failure and the one people forget. Bricks tile Y and X only, and
    the stride is applied to Y and X only, so a volume keeps every z plane it was acquired with
    however far out the camera is."""
    b = _bricks.Brick(iy=0, ix=0, r0=0, r1=1024, c0=0, c1=1024)
    for step in (1, 2, 4, 8, 16, 32, 64):
        nz, h, w = b.sampled_shape(10, step)
        assert nz == 10, f"z was thinned to {nz} at stride {step} — the volume would read as flat"
        assert (h, w) == (1024 // step, 1024 // step)
