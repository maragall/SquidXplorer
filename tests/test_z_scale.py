"""The z step the 3D renderer draws with, and what happens when the acquisition does not state it.

`scale=(dz, px, px)` is the whole of 3D proportionality: a wrong `dz` produces no error, no
warning and a picture that looks like the microscope collected a flat sample — invisible by
construction, which is why it needs a test rather than an eyeball.
"""

from __future__ import annotations

import logging

import pytest

from squidxplorer._napari3d import z_step_um

PX = 0.752                      # 10x objective, 7.52 um sensor pixel
DZ = 1.5                        # the acquisition's real z step


def test_the_declared_z_step_is_the_one_used():
    assert z_step_um({"dz_um": DZ}, PX) == pytest.approx(DZ)


@pytest.mark.parametrize("meta", [{"dz_um": None}, {"dz_um": 0}, {"dz_um": 0.0}, {}],
                         ids=["none", "int-zero", "float-zero", "absent"])
def test_a_missing_z_step_falls_back_to_the_pixel_size_and_says_so(meta, caplog):
    """It still opens rather than refusing on a metadata gap, but the log names the substitution
    and the number, so "why is it flat" has an answer in the console.

    Zero is tested alongside None because `Acquisition` permits `dz_um=0.0`, which would collapse
    the volume to a plane — the most flattening value of all.
    """
    with caplog.at_level(logging.WARNING):
        assert z_step_um(meta, PX) == pytest.approx(PX)
    assert "NOT to scale" in caplog.text, "the fallback must be audible, not silent"
    assert f"{PX:.4f}" in caplog.text, "the log must name the number it substituted"


def test_the_warning_names_where_it_happened(caplog):
    """Three call sites share this helper (native FOV, ROI fusion, ROI volume)."""
    with caplog.at_level(logging.WARNING):
        z_step_um({}, PX, where="3D ROI B3")
    assert "3D ROI B3" in caplog.text


def test_a_real_stack_is_a_wafer_and_that_is_the_data_not_the_renderer():
    """10 planes at 1.5 um is 15 um of depth; 2084 pixels at 0.752 um is 1567 um of field — any
    correct renderer draws that ~104x wider than it is tall."""
    n_z, frame_px = 10, 2084
    z_extent = n_z * z_step_um({"dz_um": DZ}, PX)
    xy_extent = frame_px * PX
    assert z_extent == pytest.approx(15.0)
    assert xy_extent == pytest.approx(1567.2, abs=0.1)
    assert xy_extent / z_extent == pytest.approx(104.5, abs=0.5)
