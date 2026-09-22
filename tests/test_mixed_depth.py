"""A single-plane channel beside a z-stack (Nick, ValidTX: n = 1 phase contrast per z-stack,
overlaid with best-in-focus fluorescence): the z slider indexes the stacked channels, the
single-plane channel always shows its one plane, and the export's per-channel z clamp holds.

The reader/format half is NOT covered here on purpose: the acquisition metadata carries ONE
``z_levels`` list for every channel (``reader._assemble_metadata``), so a per-channel Nz is
not representable on disk today; that half is Squid-side schema work, reported, not forced.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("napari")

from squidxplorer._montage import composite  # noqa: E402
from squidxplorer._png import PngChannel, render_view_png  # noqa: E402
from squidxplorer._workers import _full_res_plane  # noqa: E402

Z = 3               # the z the stacked channels are on; beyond every 1-deep channel


def _arr(seed, shape):
    return np.random.default_rng(seed).integers(0, 4000, shape, dtype=np.uint16)


def test_the_export_takes_each_channel_at_its_own_depth_clamped():
    """One PNG of a mixed-depth scene: the stack at its on-screen z, a bare plane as
    itself (z ignored), a 1-deep stack clamped to its only plane."""
    stack, plane, one_deep = _arr(0, (5, 16, 16)), _arr(1, (16, 16)), _arr(2, (1, 16, 16))
    channels = [
        PngChannel("488", stack, (0.0, 4000.0), (0, 255, 0), z_index=Z),
        PngChannel("phase", plane, (0.0, 4000.0), (255, 255, 255), z_index=Z),
        PngChannel("561", one_deep, (0.0, 4000.0), (255, 0, 0), z_index=Z),
    ]
    got, step = render_view_png(channels)
    assert step == 1
    colors = np.stack([np.asarray(c.rgb, dtype=np.float32) / 255.0 for c in channels])
    expected = composite(np.stack([stack[Z], plane, one_deep[0]]), colors,
                         [(0.0, 4000.0)] * 3)
    np.testing.assert_array_equal(got, expected)
    # The clamp itself, pinned at the seam every export goes through.
    np.testing.assert_array_equal(_full_res_plane(one_deep, Z), one_deep[0])
    np.testing.assert_array_equal(_full_res_plane(plane, Z), plane)


def test_the_on_screen_overlay_tolerates_a_single_plane_channel_beside_a_stack():
    """One viewer: a raw z-stack channel and a single-plane channel. Stepping the z axis
    never unlights or breaks the single-plane channel; its layer keeps its one plane."""
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QApplication

    from squidxplorer._napari_view import MosaicLayers

    QApplication.instance() or QApplication([])
    ml = MosaicLayers(ViewerModel())
    ml.add_mosaic("raw", "488", _arr(0, (5, 16, 16)), bbox_um=(0.0, 0.0, 16.0, 16.0))
    ml.add_mosaic("raw", "phase", _arr(1, (16, 16)), bbox_um=(0.0, 0.0, 16.0, 16.0))

    assert ml.model.dims.nsteps[0] == 5           # the z slider indexes the stacked channel
    ml.model.dims.set_current_step(0, Z)          # napari broadcasts the 2-D layer across z

    phase = ml.find("raw", "phase")
    assert phase is not None and phase.visible
    assert tuple(phase.data.shape) == (16, 16)    # still its one plane, whatever z is on
    assert ml.top_visible_layer("phase") is phase  # the export would collect it
    assert ml.contrast("phase") is not None       # the numeric fields have a window to show
