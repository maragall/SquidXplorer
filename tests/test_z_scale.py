"""The z step the 3D renderer draws with, and what happens when the acquisition does not state it."""

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
    """It still opens rather than refusing on a metadata gap, but the log names the substitution and the number, so "why is it flat" has an answer in the console."""
    with caplog.at_level(logging.WARNING):
        assert z_step_um(meta, PX, where="3D ROI B3") == pytest.approx(PX)
    assert "NOT to scale" in caplog.text, "the fallback must be audible, not silent"
    assert f"{PX:.4f}" in caplog.text, "the log must name the number it substituted"
    assert "3D ROI B3" in caplog.text, "the log must name where it happened"


