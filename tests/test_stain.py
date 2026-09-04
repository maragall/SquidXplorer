"""``_stain.stain_lut``: the density-binned Beer-Lambert fit."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from squidxplorer._stain import stain_lut

# The baked-in hue ramp: at low OD (light) R/G = 0.3, B/G = 0.7; at high OD the reverse-ish.
_K_R_LIGHT, _K_B_LIGHT = 0.3, 0.7
_K_R_DENSE, _K_B_DENSE = 0.8, 0.2
_OD_LO, _OD_HI = -np.log(0.9), -np.log(0.1)


def _k_at(od, k_light, k_dense):
    return k_light + (k_dense - k_light) * (od - _OD_LO) / (_OD_HI - _OD_LO)


def _density_varying_png(path):
    """Pixels whose (k_R, k_B) ramp linearly with green OD, plus white background pixels so the 99th-percentile white lands at 255 per channel."""
    t = np.linspace(0.1, 0.9, 40_000)
    od = -np.log(t)
    k_r = _k_at(od, _K_R_LIGHT, _K_R_DENSE)
    k_b = _k_at(od, _K_B_LIGHT, _K_B_DENSE)
    stained = np.stack([255 * t ** k_r, 255 * t, 255 * t ** k_b], axis=-1)
    white = np.full((10_000, 3), 255.0)
    px = np.concatenate([stained, white]).reshape(250, 200, 3)
    Image.fromarray(np.round(px).astype(np.uint8)).save(path)


def _constant_ratio_png(path, k_r=0.3, k_b=0.6):
    """The single-stain (H&E-like) case: one hue at every density."""
    t = np.tile(np.linspace(0.2, 1.0, 200), (120, 1))
    png = np.stack([255 * t ** k_r, 255 * t, 255 * t ** k_b], axis=-1).astype(np.uint8)
    Image.fromarray(png).save(path)


def _k_of_stop(row, t):
    """Recover the effective exponents from one LUT stop: k = ln(color) / ln(t)."""
    return np.log(row[0]) / np.log(t), np.log(row[2]) / np.log(t)


def test_the_lut_hue_follows_density(tmp_path):
    """Light-end and dark-end stops carry the DIFFERENT hue ratios baked into the PNG."""
    png = tmp_path / "trichrome.png"
    _density_varying_png(png)
    lut = stain_lut(png)
    assert lut is not None and len(lut) == 256
    arr = np.asarray(lut)
    assert arr.shape == (256, 3)
    assert tuple(arr[-1]) == (1.0, 1.0, 1.0)    # background stays exactly white
    assert tuple(arr[0]) == (0.0, 0.0, 0.0)     # black stays black
    assert (arr >= 0.0).all() and (arr <= 1.0).all()

    i_light = int(round(0.9 * 255))             # t = 0.9, the lightest fitted density
    k_r, k_b = _k_of_stop(lut[i_light], i_light / 255.0)
    assert abs(k_r - _K_R_LIGHT) < 0.1, k_r
    assert abs(k_b - _K_B_LIGHT) < 0.1, k_b

    i_dense = int(round(0.1 * 255))             # t = 0.1, the densest fitted density
    k_r_d, k_b_d = _k_of_stop(lut[i_dense], i_dense / 255.0)
    assert abs(k_r_d - _K_R_DENSE) < 0.1, k_r_d
    assert abs(k_b_d - _K_B_DENSE) < 0.1, k_b_d

    assert k_r_d - k_r > 0.3 and k_b - k_b_d > 0.3   # the ends measurably differ


def test_constant_ratio_data_matches_the_old_global_fit(tmp_path):
    """Single-stain data is unchanged in look: every stop is t^k for the one global k."""
    png = tmp_path / "he.png"
    _constant_ratio_png(png, k_r=0.3, k_b=0.6)
    lut = np.asarray(stain_lut(png))
    t = np.linspace(0.0, 1.0, 256)
    want = np.stack([t ** 0.3, t, t ** 0.6], axis=1)
    assert np.abs(lut - want).max() < 0.06


def test_too_few_stained_pixels_or_a_grayscale_png_is_a_refusal(tmp_path):
    blank = tmp_path / "blank.png"
    Image.fromarray(np.full((80, 80, 3), 255, np.uint8)).save(blank)
    assert stain_lut(blank) is None
    gray = tmp_path / "gray.png"
    Image.fromarray(np.full((80, 80), 128, np.uint8)).save(gray)
    assert stain_lut(gray) is None


# --- Chroma ratio luminance damping -----------------------------------------------------------
#
# The overview PNG's overlap zones hold NEIGHBOR frames' pixels (later-overwrites-earlier), and
# its 2 um blur smears tissue chroma over lumen edges. A tissue ratio (R/G ~ 3) landing on a
# BRIGHT file pixel clips to the dtype ceiling and glows hot magenta (measured: 27k px at the
# uint8 ceiling on FOV 72 of the 20x trichrome set, 96% in the overlap band). The ratio's
# confidence is the luminance agreement between the PNG and the file's own plane.


def _chroma_source(tmp_path, png_rgb):
    from squidxplorer._stain import ChromaSource

    path = tmp_path / "overview.png"
    Image.fromarray(png_rgb).save(path)
    return ChromaSource(path, top_left_mm_yx=(0.0, 0.0), resolution_um=2.0)


def _flat_png(shape, r, g, b):
    png = np.zeros(shape + (3,), np.uint8)
    png[..., 0], png[..., 1], png[..., 2] = r, g, b
    return png


def test_a_mismatched_overview_luminance_damps_the_chroma_ratio(tmp_path):
    """Where the file's plane is BRIGHTER than the PNG luminance that measured the ratio (the PNG holds another frame's tissue there), the ratio is pulled"""
    src = _chroma_source(tmp_path, _flat_png((30, 30), r=150, g=50, b=25))
    plane = np.full((16, 16), 40, np.uint8)
    plane[:, 12:] = 100
    out_r = src.component_plane(plane, 0, "manual", 0, 30.0, 30.0, 2.0)
    out_b = src.component_plane(plane, 2, "manual", 0, 30.0, 30.0, 2.0)
    assert np.array_equal(out_r[:, :11], np.full((16, 11), 120, np.uint8))   # 40 * 3.0
    assert np.array_equal(out_b[:, :11], np.full((16, 11), 20, np.uint8))    # 40 * 0.5
    assert np.array_equal(out_r[:, 13:], np.full((16, 3), 180, np.uint8))
    assert np.array_equal(out_b[:, 13:], np.full((16, 3), 80, np.uint8))     # 1 - 0.5*0.4


def test_a_uniformly_scaled_window_keeps_the_full_ratio(tmp_path):
    """PNG luminance that is one consistent scale of the plane is agreement, not mismatch: the gain absorbs the scale and the measured ratios apply in full."""
    src = _chroma_source(tmp_path, _flat_png((30, 30), r=150, g=100, b=50))
    plane = np.full((16, 16), 80, np.uint8)                      # gain 100/80, weight 1
    out_r = src.component_plane(plane, 0, "manual", 0, 30.0, 30.0, 2.0)
    out_b = src.component_plane(plane, 2, "manual", 0, 30.0, 30.0, 2.0)
    assert np.array_equal(out_r, np.full((16, 16), 120, np.uint8))           # 80 * 1.5
    assert np.array_equal(out_b, np.full((16, 16), 40, np.uint8))            # 80 * 0.5


def test_a_dense_stain_ratio_survives_while_a_near_black_denominator_stays_bounded(tmp_path):
    """A true R/G of 6 measured at adequate PNG luminance reaches the display (the old flat cap
    at 4 rendered dense pink tissue dark purple, measured 1.9% of the 20x trichrome set), while
    a near-black denominator's raw ratio (250/3 = 83) stays bounded by the denominator floor."""
    png = np.zeros((30, 30, 3), np.uint8)
    png[:, :12, 0], png[:, :12, 1] = 192, 32   # dense stain: R/G 6.0, bright denominator
    png[:, 12:, 0], png[:, 12:, 1] = 250, 3    # near-black denominator: raw R/G 83
    src = _chroma_source(tmp_path, png)
    plane = np.zeros((16, 16), np.uint8)       # plane == PNG G: gain 1, full trust everywhere
    plane[:, :5], plane[:, 5:] = 32, 3
    out_r = src.component_plane(plane, 0, "manual", 0, 30.0, 30.0, 2.0)
    assert np.array_equal(out_r[:, :5], np.full((16, 5), 192, np.uint8))     # 32 * 6.0, > old 4
    assert np.array_equal(out_r[:, 5:], np.full((16, 11), 47, np.uint8))     # 3 * 250/16, not 250


# --- Ownership-aware ratios --------------------------------------------------------------------
#
# The overview PNG is written tile over tile, later overwrites earlier: in an FOV's overlap
# band the PNG holds the NEIGHBOR frame's pixels (measured: FOV 72's right band correlates
# 0.894 with neighbor 73's frame vs 0.515 with its own). Where the neighbor's rendering is
# MISALIGNED, damping alone pulls those foreign ratios toward NEUTRAL, so the band renders
# dark on gray while the overview shows pink (the customer's second-round complaint).
# Ownership: a window pixel belongs to the LAST FOV in acquisition order whose footprint
# covers it; on unowned pixels the damping FALLBACK becomes the frame's own owned hue,
# extended across the band (chroma is low-frequency), instead of neutral. Where the PNG's
# luminance AGREES with the plane the measured ratio still stands in full: on the well
# aligned ground-truth set the band's measured hue is the same tissue seen by the neighbor
# and replacing it wholesale measured WORSE (window corr 0.977 -> 0.775), so ownership only
# decides what mistrust falls back to, never overrides trust.


def _acq_chroma_source(tmp_path, png_rgb, csv_rows):
    """A ChromaSource inside a real acquisition layout: root/0/mosaic_view/overview.png with the executed root/0/coordinates.csv naming the blit order."""
    from squidxplorer._stain import ChromaSource

    mosaic = tmp_path / "0" / "mosaic_view"
    mosaic.mkdir(parents=True)
    path = mosaic / "overview.png"
    Image.fromarray(png_rgb).save(path)
    (tmp_path / "0" / "coordinates.csv").write_text(
        "region,x (mm),y (mm)\n" + "\n".join(csv_rows) + "\n")
    return ChromaSource(path, top_left_mm_yx=(0.0, 0.0), resolution_um=2.0)


def test_ownership_mask_pins_the_blit_geometry(tmp_path):
    """A window pixel belongs to the LAST acquisition-order FOV covering it: FOV 0's right band under later FOV 1 is unowned; FOV 1, blitted last, owns its"""
    src = _acq_chroma_source(tmp_path, _flat_png((30, 30), r=50, g=50, b=50),
                             ["manual,0.016,0.016", "manual,0.040,0.016"])
    own0 = src._ownership_mask("manual", 16.0, 16.0, (16, 16), 2.0)
    assert own0.shape == (16, 16)
    assert own0[:, :12].all()
    assert not own0[:, 12:].any()
    own1 = src._ownership_mask("manual", 40.0, 16.0, (16, 16), 2.0)
    assert own1.all()


def test_no_coordinates_record_means_every_pixel_is_owned(tmp_path):
    """Without a blit-order record the mask cannot be computed: all-owned, damped-only."""
    src = _chroma_source(tmp_path, _flat_png((30, 30), r=50, g=50, b=50))
    assert src._ownership_mask("manual", 30.0, 30.0, (16, 16), 2.0).all()


def test_a_mistrusted_unowned_band_falls_back_to_the_frames_own_hue(tmp_path):
    """The PNG's band carries the neighbor's MISALIGNED (darker, different-hue) pixels: the damped ratio there must fall back to this frame's own owned hue,"""
    png = np.zeros((30, 30, 3), np.uint8)
    png[:, :12, 0], png[:, :12, 1], png[:, :12, 2] = 100, 50, 25   # own: R/G 2.0, B/G 0.5
    png[:, 12:, 0], png[:, 12:, 1], png[:, 12:, 2] = 10, 20, 40    # neighbor, dark: 0.5 / 2.0
    src = _acq_chroma_source(tmp_path, png, ["manual,0.016,0.016", "manual,0.040,0.016"])
    plane = np.full((16, 16), 40, np.uint8)
    out_r = src.component_plane(plane, 0, "manual", 0, 16.0, 16.0, 2.0)
    out_b = src.component_plane(plane, 2, "manual", 0, 16.0, 16.0, 2.0)
    assert np.array_equal(out_r[:, :11], np.full((16, 11), 80, np.uint8))    # 40 * 2.0
    assert np.array_equal(out_b[:, :11], np.full((16, 11), 20, np.uint8))    # 40 * 0.5
    assert np.array_equal(out_r[:, 13:], np.full((16, 3), 56, np.uint8))
    assert np.array_equal(out_b[:, 13:], np.full((16, 3), 44, np.uint8))


def test_an_agreeing_unowned_band_keeps_the_measured_hue(tmp_path):
    """Where the neighbor's band pixels AGREE in luminance the overview is describing this tissue correctly and the measured hue stands in full: pinned"""
    png = np.zeros((30, 30, 3), np.uint8)
    png[..., 1] = 50
    png[:, :12, 0], png[:, :12, 2] = 100, 25      # own hue: R/G 2.0, B/G 0.5
    png[:, 12:, 0], png[:, 12:, 2] = 25, 100      # neighbor's hue at the SAME luminance
    src = _acq_chroma_source(tmp_path, png, ["manual,0.016,0.016", "manual,0.040,0.016"])
    plane = np.full((16, 16), 40, np.uint8)       # gain 50/40, weight 1 everywhere
    out_r = src.component_plane(plane, 0, "manual", 0, 16.0, 16.0, 2.0)
    assert np.array_equal(out_r[:, :11], np.full((16, 11), 80, np.uint8))    # 40 * 2.0
    assert np.array_equal(out_r[:, 13:], np.full((16, 3), 20, np.uint8))     # 40 * 0.5 kept


def test_the_last_fov_keeps_its_measured_band(tmp_path):
    """FOV 1 was blitted last, so the overview's band there really is its own frame: its measured ratios apply unfilled."""
    png = np.zeros((30, 30, 3), np.uint8)
    png[..., 1] = 50
    png[:, :12, 0], png[:, :12, 2] = 100, 25
    png[:, 12:, 0], png[:, 12:, 2] = 25, 100
    src = _acq_chroma_source(tmp_path, png, ["manual,0.016,0.016", "manual,0.040,0.016"])
    plane = np.full((16, 16), 40, np.uint8)
    out_r = src.component_plane(plane, 0, "manual", 1, 40.0, 16.0, 2.0)
    assert np.array_equal(out_r, np.full((16, 16), 20, np.uint8))   # 40 * measured 0.5
