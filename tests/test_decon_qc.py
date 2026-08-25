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


def test_the_turbo_composite_machinery_is_gone():
    """Removed 2026-08-25 (Julio: "The turbo colormap preview makes no sense. remove it").
    The GUI preview is a normal data layer under the channel's own colormap; only the CLI
    montage (write_montage / write_curve, tools/decon_qc.py) still renders sections."""
    for name in ("qc_composite", "composite_centre_at", "turbo_rgb", "GAP_RGB"):
        assert not hasattr(decon_qc, name), f"_decon_qc still ships {name}"
