"""``_stain.stain_lut``: the density-binned Beer-Lambert fit.

The saved planes are single-channel, so one gray value maps to one color; but the hue may
vary WITH DENSITY (Masson's trichrome: light cytoplasm red-leaning, dense collagen blue).
The LUT must follow the PNG's own hue-vs-density curve and reduce to the old global fit on
constant-ratio (single-stain) data.
"""

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
    """Pixels whose (k_R, k_B) ramp linearly with green OD, plus white background pixels
    so the 99th-percentile white lands at 255 per channel."""
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


def test_lut_endpoints_and_range(tmp_path):
    png = tmp_path / "trichrome.png"
    _density_varying_png(png)
    lut = np.asarray(stain_lut(png))
    assert lut.shape == (256, 3)
    assert tuple(lut[-1]) == (1.0, 1.0, 1.0)    # background stays exactly white
    assert tuple(lut[0]) == (0.0, 0.0, 0.0)     # black stays black
    assert (lut >= 0.0).all() and (lut <= 1.0).all()


def test_too_few_stained_pixels_is_a_refusal(tmp_path):
    png = tmp_path / "blank.png"
    Image.fromarray(np.full((80, 80, 3), 255, np.uint8)).save(png)
    assert stain_lut(png) is None


def test_a_grayscale_png_is_a_refusal(tmp_path):
    png = tmp_path / "gray.png"
    Image.fromarray(np.full((80, 80), 128, np.uint8)).save(png)
    assert stain_lut(png) is None


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
    """Where the file's plane is BRIGHTER than the PNG luminance that measured the ratio (the
    PNG holds another frame's tissue there), the ratio is pulled toward neutral by the
    luminance agreement — the bright pixel must not glow at the dtype ceiling."""
    # PNG all "dark tissue": G=50, R=150 (ratio 3.0), B=25 (ratio 0.5).
    src = _chroma_source(tmp_path, _flat_png((30, 30), r=150, g=50, b=25))
    # Plane 16x16 at 2 um (1:1 with the PNG): cols 0..11 dark (40, agrees with the PNG's
    # luminance), cols 12..15 bright (100, the PNG lied about this area).
    plane = np.full((16, 16), 40, np.uint8)
    plane[:, 12:] = 100
    # gain = median(G/plane) = 50/40 = 1.25 (the dark majority).
    out_r = src.component_plane(plane, 0, "manual", 0, 30.0, 30.0, 2.0)
    out_b = src.component_plane(plane, 2, "manual", 0, 30.0, 30.0, 2.0)
    # Agreeing luminance keeps the full measured chroma: weight 50/(40*1.25) = 1.
    assert np.array_equal(out_r[:, :11], np.full((16, 11), 120, np.uint8))   # 40 * 3.0
    assert np.array_equal(out_b[:, :11], np.full((16, 11), 20, np.uint8))    # 40 * 0.5
    # Mismatched luminance damps: weight 50/(100*1.25) = 0.4, ratio 1 + 2.0*0.4 = 1.8,
    # NOT the undamped 100*3.0 = 255-clipped glow.
    assert np.array_equal(out_r[:, 13:], np.full((16, 3), 180, np.uint8))
    assert np.array_equal(out_b[:, 13:], np.full((16, 3), 80, np.uint8))     # 1 - 0.5*0.4


def test_a_uniformly_scaled_window_keeps_the_full_ratio(tmp_path):
    """PNG luminance that is one consistent scale of the plane is agreement, not mismatch:
    the gain absorbs the scale and the measured ratios apply in full."""
    src = _chroma_source(tmp_path, _flat_png((30, 30), r=150, g=100, b=50))
    plane = np.full((16, 16), 80, np.uint8)                      # gain 100/80, weight 1
    out_r = src.component_plane(plane, 0, "manual", 0, 30.0, 30.0, 2.0)
    out_b = src.component_plane(plane, 2, "manual", 0, 30.0, 30.0, 2.0)
    assert np.array_equal(out_r, np.full((16, 16), 120, np.uint8))           # 80 * 1.5
    assert np.array_equal(out_b, np.full((16, 16), 40, np.uint8))            # 80 * 0.5
