"""The semi-convergence QC tool's measurement, over synthetic volumes with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer import _decon_qc as decon_qc


DXY, DZ = 0.752, 1.5
CORE_UM = 0.61 * 0.525 / 0.3   # the NA-0.3 Airy radius, 1.0675 um
WINDOW_UM = 6.0


def _volume(halo_level, shape=(11, 64, 64), core_level=1000.0):
    """A bright core at the centre plus a uniform halo at *halo_level*, on a zero floor."""
    volume = np.zeros(shape, dtype=np.float32)
    zc, yc, xc = shape[0] // 2, shape[1] // 2, shape[2] // 2
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    r = np.sqrt(((zz - zc) * DZ) ** 2 + ((yy - yc) * DXY) ** 2 + ((xx - xc) * DXY) ** 2)
    volume[r <= WINDOW_UM] = halo_level
    volume[r <= CORE_UM] = core_level
    return volume, (zc, yc, xc)


def test_ratio_is_halo_brightness_over_core_brightness():
    volume, centre = _volume(halo_level=200.0, core_level=1000.0)
    got = decon_qc.halo_core_ratio(volume, centre, DXY, DZ, CORE_UM, WINDOW_UM)
    assert got == pytest.approx(0.2, abs=1e-6)


def test_a_brighter_halo_scores_higher():
    dim, centre = _volume(halo_level=100.0)
    bright, _ = _volume(halo_level=400.0)
    assert (decon_qc.halo_core_ratio(bright, centre, DXY, DZ, CORE_UM, WINDOW_UM)
            > decon_qc.halo_core_ratio(dim, centre, DXY, DZ, CORE_UM, WINDOW_UM))


def test_a_constant_camera_offset_does_not_change_the_answer():
    """The floor subtraction has to actually neutralise the sensor's offset."""
    volume, centre = _volume(halo_level=200.0)
    plain = decon_qc.halo_core_ratio(volume, centre, DXY, DZ, CORE_UM, WINDOW_UM)
    offset = decon_qc.halo_core_ratio(volume + 500.0, centre, DXY, DZ, CORE_UM, WINDOW_UM)
    assert offset == pytest.approx(plain, abs=1e-6)


def test_a_dark_core_is_refused_rather_than_divided_by():
    volume, centre = _volume(halo_level=0.0, core_level=0.0)
    with pytest.raises(ValueError, match="core"):
        decon_qc.halo_core_ratio(volume, centre, DXY, DZ, CORE_UM, WINDOW_UM)


def test_window_never_exceeds_what_the_stack_can_hold_axially():
    """A 10-plane stack at 1.5 um cannot hold the preferred 8-Airy-radius sphere."""
    assert decon_qc.qc_window_um(CORE_UM, nz=10, dz_um=1.5) == pytest.approx(6.0)
    # deep enough: the preferred size is used unchanged
    assert decon_qc.qc_window_um(CORE_UM, nz=40, dz_um=1.5) == pytest.approx(8 * CORE_UM)


def test_the_structure_is_picked_away_from_the_edges():
    stack = np.zeros((10, 64, 64), dtype=np.uint16)
    stack[0, 32, 32] = 60000          # brightest overall, but unusable: top plane
    stack[5, 30, 30] = 30000          # dimmer, but a window fits around it
    z, y, x = decon_qc.brightest_structure(stack, DXY, DZ, CORE_UM, z_margin=4, xy_margin=8)
    assert 4 <= z < 6
    assert (int(y), int(x)) == (30, 30)


def test_an_interior_minimum_is_reported_as_a_real_turn():
    best, kind, message = decon_qc.recommend([1, 2, 3, 4, 5], [0.9, 0.7, 0.5, 0.6, 0.8])
    assert (best, kind) == (3, "turn")
    assert "RECOMMENDATION: 3" in message


def test_a_still_falling_curve_is_not_dressed_up_as_a_turning_point():
    """Argmin of a monotone curve is just where the sweep ended."""
    best, kind, message = decon_qc.recommend([1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6])
    assert (best, kind) == (4, "still-falling")
    assert "NO TURN" in message and "RECOMMENDATION" not in message


def test_a_curve_that_only_rises_says_so():
    best, kind, message = decon_qc.recommend([1, 2, 3, 4], [0.6, 0.7, 0.8, 0.9])
    assert (best, kind) == (1, "rising")
    assert "NO TURN" in message


def test_orthogonal_slices_are_xz_and_yz_through_the_structure():
    volume = np.zeros((5, 20, 30), dtype=np.float32)
    volume[2, 7, 11] = 1.0
    xz, yz = decon_qc.orthogonal_slices(volume, (2, 7, 11))
    assert xz.shape == (5, 30) and yz.shape == (5, 20)
    assert xz[2, 11] == 1.0 and yz[2, 7] == 1.0


def test_the_montage_view_is_cropped_around_the_structure():
    volume = np.zeros((5, 60, 60), dtype=np.float32)
    volume[2, 30, 30] = 1.0
    xz, yz = decon_qc.orthogonal_slices(volume, (2, 30, 30), half=8)
    assert xz.shape == (5, 16) and yz.shape == (5, 16)


def test_display_puts_background_at_the_bottom_of_the_colormap():
    panel = np.full((10, 40), 500.0)
    panel[5, 20] = 5000.0
    shown = decon_qc._display(panel)
    assert shown.max() == pytest.approx(1.0)
    assert shown[0, 0] == pytest.approx(0.0)


def test_the_montage_has_one_row_per_iteration_and_two_columns(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    volume, centre = _volume(halo_level=200.0)
    rows = [("raw", volume), ("1", volume), ("2", volume)]
    out = tmp_path / "montage.png"
    decon_qc.write_montage(out, rows, centre, DXY, DZ, "test", view_half=8)
    assert out.exists() and out.stat().st_size > 0


def test_the_composite_is_the_xy_plane_with_the_two_strips_attached():
    """x-y (Y,X) with y-z (Y,Z) to its right and x-z (Z,X) below it."""
    volume = np.zeros((5, 40, 60), dtype=np.float32)
    volume[2, 20, 30] = 1.0
    m = decon_qc.qc_composite(volume, (2, 20, 30), gap=2)
    # (Y + gap + Z, X + gap + Z) = (40 + 2 + 5, 60 + 2 + 5)
    assert m.shape == (47, 67)


def test_the_composite_is_cropped_around_the_structure_like_the_montage():
    volume = np.zeros((5, 80, 80), dtype=np.float32)
    volume[2, 40, 40] = 1.0
    m = decon_qc.qc_composite(volume, (2, 40, 40), view_half=8, gap=2)
    assert m.shape == (16 + 2 + 5, 16 + 2 + 5)


def test_the_three_views_share_ONE_intensity_scale():
    """A dim strip must look dim; per-panel normalisation would hide the axial halo."""
    volume = np.zeros((5, 40, 40), dtype=np.float32)
    volume[2, 20, 20] = 1000.0        # the structure the composite is centred on
    # a 5x brighter structure in the x-y panel but in neither strip
    volume[2, 10, 10] = 5000.0
    # the volume's true peak on a different z, so it is in no panel at all
    volume[0, 30, 30] = 8000.0
    m = decon_qc.qc_composite(volume, (2, 20, 20), gap=2, gamma=1.0)
    # one scale, set by the volume's 8000 peak
    assert m[40 + 2 + 2, 20] == pytest.approx(0.125, abs=1e-6)  # x-z strip, z=2, at x=20
    assert m[20, 40 + 2 + 2] == pytest.approx(0.125, abs=1e-6)  # y-z strip, z=2, at y=20
    assert m[10, 10] == pytest.approx(0.625, abs=1e-6)          # x-y panel, its own peak
    assert m[:40, :40].max() == pytest.approx(0.625, abs=1e-6)  # nothing reaches 1.0


def test_the_gaps_are_nan_so_a_separator_is_never_mistaken_for_signal():
    """A gap filled with 0.0 would render as turbo's dark blue, indistinguishable from background."""
    volume = np.zeros((5, 40, 40), dtype=np.float32)
    volume[2, 20, 20] = 1.0
    m = decon_qc.qc_composite(volume, (2, 20, 20), gap=2)
    assert np.isnan(m[40:42, :]).all()          # the horizontal separator row band
    assert np.isnan(m[:, 40:42]).all()          # the vertical separator column band
    assert not np.isnan(m[:40, :40]).any()      # the x-y panel itself carries no NaN


def test_the_corner_that_no_section_covers_is_blank():
    """Bottom-right is (z vs z): no such section exists."""
    volume = np.zeros((5, 40, 40), dtype=np.float32)
    volume[2, 20, 20] = 1.0
    m = decon_qc.qc_composite(volume, (2, 20, 20), gap=2)
    assert np.isnan(m[42:, 42:]).all()


def test_a_click_in_the_xy_panel_picks_y_and_x_and_keeps_z():
    got = decon_qc.composite_centre_at((5, 40, 60), (2, 20, 30), row=7, col=11, gap=2)
    assert got == (2, 7, 11)


def test_a_click_in_the_xz_strip_picks_z_and_x_and_keeps_y():
    """The x-z strip is BELOW the x-y panel: rows are z, columns are x."""
    got = decon_qc.composite_centre_at((5, 40, 60), (2, 20, 30),
                                       row=40 + 2 + 3, col=11, gap=2)
    assert got == (3, 20, 11)


def test_a_click_in_the_yz_strip_picks_z_and_y_and_keeps_x():
    """The y-z strip is to the RIGHT: rows are y, columns are z."""
    got = decon_qc.composite_centre_at((5, 40, 60), (2, 20, 30),
                                       row=7, col=60 + 2 + 4, gap=2)
    assert got == (4, 7, 30)


def test_a_click_on_a_separator_or_the_dead_corner_moves_nothing():
    """None, not a nearby guess."""
    shape, centre = (5, 40, 60), (2, 20, 30)
    assert decon_qc.composite_centre_at(shape, centre, row=41, col=11) is None   # row gap
    assert decon_qc.composite_centre_at(shape, centre, row=7, col=61) is None    # col gap
    assert decon_qc.composite_centre_at(shape, centre, row=45, col=65) is None   # corner
    assert decon_qc.composite_centre_at(shape, centre, row=-1, col=11) is None   # outside
    assert decon_qc.composite_centre_at(shape, centre, row=999, col=11) is None


def test_the_click_map_is_the_composite_s_own_inverse_when_the_view_is_cropped():
    """With view_half a click at panel pixel (0, 0) is the crop's corner voxel, not (0, 0)."""
    volume = np.zeros((5, 80, 80), dtype=np.float32)
    volume[2, 40, 40] = 1000.0
    centre, half = (2, 40, 40), 8
    m = decon_qc.qc_composite(volume, centre, view_half=half, gap=2)
    assert m.shape == (16 + 2 + 5, 16 + 2 + 5)

    # clicking the middle of the cropped panel round-trips to the centre
    assert decon_qc.composite_centre_at(volume.shape, centre, row=8, col=8,
                                        view_half=half, gap=2) == centre
    # the panel's top-left pixel is the crop's corner, not the volume's
    assert decon_qc.composite_centre_at(volume.shape, centre, row=0, col=0,
                                        view_half=half, gap=2) == (2, 32, 32)
    # one pixel past the cropped panel is the separator, not the strip
    assert decon_qc.composite_centre_at(volume.shape, centre, row=0, col=16,
                                        view_half=half, gap=2) is None


def test_the_click_map_uses_the_cropped_width_so_the_strips_are_where_they_are_drawn():
    """The y-z strip starts at column w_xy + gap of the CROPPED panel, not the full width."""
    volume = np.zeros((5, 80, 80), dtype=np.float32)
    centre, half = (2, 40, 40), 8
    m = decon_qc.qc_composite(volume, centre, view_half=half, gap=2)
    first_strip_col = m.shape[1] - 5              # the y-z strip is nz columns wide
    got = decon_qc.composite_centre_at(volume.shape, centre, row=0, col=first_strip_col,
                                       view_half=half, gap=2)
    assert got == (0, 32, 40)                     # z=0, y=crop corner, x unchanged


def test_turbo_rgb_is_turbo_and_paints_the_gaps_neutral():
    """matplotlib's turbo, with NaN kept out of the ramp."""
    pytest.importorskip("matplotlib")
    panel = np.array([[0.0, 1.0], [np.nan, 0.5]], dtype=np.float64)
    rgb = decon_qc.turbo_rgb(panel)
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    import matplotlib.cm
    lo = (np.array(matplotlib.colormaps["turbo"](0.0)[:3]) * 255).round().astype(np.uint8)
    hi = (np.array(matplotlib.colormaps["turbo"](1.0)[:3]) * 255).round().astype(np.uint8)
    assert tuple(rgb[0, 0]) == tuple(lo)
    assert tuple(rgb[0, 1]) == tuple(hi)
    gap = tuple(int(v) for v in rgb[1, 0])
    assert gap == decon_qc.GAP_RGB
    assert gap != tuple(lo) and gap != tuple(hi)
