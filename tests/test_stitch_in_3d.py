"""Stitching, then 3D in-window: z-depth delivery, second-open coverage, contrast harvest,
and the hidden-layer reslice crash. Only the Qt pane and the GL canvas are stood in for.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # headless Qt; must precede any Qt import

import sys                                              # noqa: E402

import numpy as np                                      # noqa: E402
import pytest                                           # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from napari.components import ViewerModel                # noqa: E402
from qtpy.QtCore import QObject                          # noqa: E402

from squidxplorer import _bricks                             # noqa: E402
from squidxplorer._napari3d import region_origin_um, roi_window_px   # noqa: E402
from squidxplorer._napari_view import META_KEY, MosaicKey, MosaicLayers, key_of  # noqa: E402
from squidxplorer._op_result import RegionResultAccumulator  # noqa: E402
from squidxplorer._region_viewer import _RAW_OP, RegionViewer  # noqa: E402
from squidxplorer._workers import _OperatorWorker            # noqa: E402

# a two-FOV region, small enough to hold in a test and shaped like the real one
N_Z, N_C = 4, 2
FRAME = 64                     # px per FOV, square
OVERLAP = 8                    # px
PX_UM, DZ_UM = 1.0, 2.5
CHANNELS = ["c0", "c1"]
REGION = "manual0"

#: The raw PREVIEW's decimation, so two pitches exist in one scene.
_PREVIEW_STEP = 2

# plane k carries its own number, so a dropped/duplicated/reordered plane is a wrong VALUE
def _plane_value(z: int, c: int) -> int:
    return 1000 + 100 * int(z) + int(c)


def _meta() -> dict:
    return {
        "regions": [REGION],
        "channels": [{"name": c} for c in CHANNELS],
        "z_levels": list(range(N_Z)),
        "n_z": N_Z,
        "n_t": 1,
        "dtype": "uint16",
        "frame_shape": (FRAME, FRAME),
        "pixel_size_um": PX_UM,
        "dz_um": DZ_UM,
        "fovs_per_region": {REGION: [0, 1]},
        "fov_positions_um": {(REGION, 0): (0.0, 0.0),
                             (REGION, 1): ((FRAME - OVERLAP) * PX_UM, 0.0)},
    }


def _stitched_5d(h: int, w: int) -> np.ndarray:
    """What the region loop yields for a plane-op: ``(T, C, Nz, Y, X)``, every plane fused."""
    out = np.empty((1, N_C, N_Z, h, w), np.uint16)
    for c in range(N_C):
        for z in range(N_Z):
            out[0, c, z] = _plane_value(z, c)
    return out


def _mosaic_hw(meta: dict) -> tuple:
    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    offsets = fov_offsets_px(meta["fov_positions_um"], REGION, [0, 1], PX_UM)
    return mosaic_extent_px(offsets, (FRAME, FRAME))


def _worker(meta: dict, operator: str) -> _OperatorWorker:
    """A real ``_OperatorWorker``, constructed but never started: only ``_on_well`` is exercised."""
    return _OperatorWorker(
        operator, reader=None, meta=meta,
        fov_index={REGION: {"rc": (0, 0), "well_id": "A1"}},
        out_dir="", regions=[REGION], save=False, n_fovs=None,
    )


def _deliver(worker, meta, image):
    """One region's run, from the engine's 5-D yield to the ``Result`` the display gets."""
    got: list = []
    worker.resultReady.connect(lambda region, fov, planes: got.append((region, fov, planes)))
    worker._on_well(REGION, 0, image)
    assert got, "the run produced no result for the display"
    _region, fov, planes = got[-1]
    acc = RegionResultAccumulator(worker._operator, REGION, meta, CHANNELS,
                                  region_operator=worker._region_op)
    acc.add(int(fov), np.asarray(planes))
    return acc.result()


# =============================================================================================
# 1. "It is not stitching the z-levels in the 3d view separately."
# =============================================================================================

def test_a_stitched_region_reaches_the_layer_with_EVERY_acquired_plane():
    """Each plane carries its own number, so a broadcast of plane 0 cannot pass."""
    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))

    assert result.z_depth == N_Z, (
        f"the stitched mosaic reached the display with z_depth {result.z_depth}; the engine "
        f"fused {N_Z} planes")
    for c, channel in enumerate(CHANNELS):
        vol = np.asarray(result.plane(channel))
        assert vol.shape == (N_Z, h, w), f"{channel}: {vol.shape}"
        assert [int(vol[z].flat[0]) for z in range(N_Z)] == [_plane_value(z, c) for z in range(N_Z)], (
            f"{channel}: the planes are not the ones the engine fused (a broadcast or a reorder)")


def test_the_layer_z_depth_drives_3D_and_a_brick_comes_back_with_every_plane():
    """The depth must survive into ``_volume_source``, at the acquisition's own pitch."""
    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))
    mos, _model = _pane_with(meta, op="stitch", result=result)

    win = _Window(meta, mos)
    read, source, pitch = RegionViewer._volume_source(win, (0, h, 0, w))
    assert source == "stitch" and read is not None, f"3D refused the stitched layer: {win.said}"
    # the raw preview in this same scene is at _PREVIEW_STEP * PX_UM, so this also says WHICH layer
    assert pitch is not None and tuple(round(float(v), 6) for v in pitch) == (PX_UM, PX_UM), (
        f"3D reads the stitched layer at {pitch} um/px, not the acquisition's {PX_UM}: a stitched "
        f"mosaic is native, and a volume of it must not be at the preview's "
        f"{_PREVIEW_STEP * PX_UM} um/px")

    brick = _bricks.Brick(iy=0, ix=0, r0=0, r1=min(16, h), c0=0, c1=min(16, w))
    voxels = read(brick, CHANNELS[0], 1, None)
    assert voxels is not None and voxels.shape[0] == N_Z, (
        f"a brick came back {None if voxels is None else voxels.shape}; a stitched volume is "
        f"{N_Z} planes deep")
    assert [int(voxels[z].flat[0]) for z in range(N_Z)] == [_plane_value(z, 0) for z in range(N_Z)]


def test_a_per_FOV_operator_still_delivers_one_plane_AND_SAYS_SO(caplog):
    """The per-FOV path still delivers plane 0, and names which plane of how many."""
    import logging

    meta = _meta()
    worker = _worker(meta, "decon")
    assert not worker._region_op, "'decon' is a per-FOV operator; this test asserts nothing"
    got: list = []
    worker.resultReady.connect(lambda region, fov, planes: got.append(planes))
    with caplog.at_level(logging.INFO):
        worker._on_well(REGION, 0, _stitched_5d(FRAME, FRAME))

    assert np.asarray(got[-1]).shape == (N_C, FRAME, FRAME), "the per-FOV path grew a z axis"
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "z plane 0 of 4" in said, f"the dropped planes were not named. log: {said!r}"


# =============================================================================================
# The seam: a pane's MosaicLayers over a bare ViewerModel, and a window borrowing real methods.
# =============================================================================================

def _pane_with(meta: dict, *, op: str, result=None, raw_clim=(5.0, 9.0),
               op_clim=(120.0, 900.0), raw_channels=CHANNELS):
    """A pane holding a RAW mosaic and, optionally, one operator layer — as a window builds them."""
    from squidxplorer._mosaic_source import mosaic_bbox_um

    model = ViewerModel()
    mos = MosaicLayers(model)
    h, w = _mosaic_hw(meta)
    bbox = mosaic_bbox_um(meta, REGION)
    for channel in raw_channels:
        # raw is DECIMATED, as fuse_region_pyramid really builds it, so the scene holds two pitches
        mos.add_mosaic(_RAW_OP, channel, np.zeros((N_Z, h // _PREVIEW_STEP, w // _PREVIEW_STEP),
                                                  np.uint16),
                       contrast_limits=raw_clim, bbox_um=bbox, z_scale_um=DZ_UM)
    if result is not None:
        for channel in CHANNELS:
            mos.add_result("intensity", op, channel, np.asarray(result.plane(channel)),
                           bbox_um=bbox, z_scale_um=DZ_UM, visible=True)
            mos.find(op, channel).contrast_limits = op_clim
    return mos, model


class _Pane:
    """The two things ``_open_roi_3d`` asks the pane for."""

    def __init__(self, mosaic):
        self.mosaic = mosaic
        self.ok = True
        self.settled = []

    def _live_max_3d_texture(self):
        # small on purpose, so the ROI is BRICKED rather than taking the single-texture path
        return 32

    def on_camera_settled(self, cb):
        self.settled.append(cb)


class _Cursor:
    region = REGION


class _Window(QObject):
    """A RegionViewer's 3D entry points, over a real pane model. No QWidget, no GL canvas."""

    _open_3d = RegionViewer._open_3d
    set_render_mode = RegionViewer.set_render_mode
    _refresh_controls_note = RegionViewer._refresh_controls_note
    _open_roi_3d = RegionViewer._open_roi_3d
    _volume_source = RegionViewer._volume_source
    _on_screen_luts = RegionViewer._on_screen_luts
    _replace_native3d = RegionViewer._replace_native3d
    _close_native3d = RegionViewer._close_native3d
    _refresh_bricks = RegionViewer._refresh_bricks

    def __init__(self, meta, mosaic, roi_bbox=None):
        super().__init__()
        self._meta = meta
        self._reader = object()
        self._pane = _Pane(mosaic)
        self._cursor = _Cursor()
        self._regions = [REGION]
        self._roi_bbox = roi_bbox
        self._native3d = None
        self.said: list = []

    def _napari_viewer(self):
        return self._pane.mosaic.model

    def _selected_roi(self):
        return None, None

    def _say(self, text):
        self.said.append(text)

    def current_region(self):
        return REGION


@pytest.fixture
def stub_bricked(monkeypatch):
    """``BrickedVolume`` with only the pieces that need a live Qt event loop stubbed out."""
    import squidxplorer._brick_view as BV

    class _Stubbed(BV.BrickedVolume):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._loader.start = lambda *x, **k: None
            self._loader.stop = lambda *x, **k: None
            self._loader.wait = lambda *x, **k: True
            self._frame_camera = lambda *x, **k: None
            self.refresh = lambda *x, **k: None

    monkeypatch.setattr(BV, "BrickedVolume", _Stubbed)
    return _Stubbed


#: A window a user drags to while the VOLUME is what they are looking at.
_DRAGGED_CLIM = (7.0, 8.0)


def _volume_up(win, roi, stub_bricked, op, drag_to=None):
    """Open a first 3D volume and let some of its bricks land, as a live view has them."""
    win._roi_bbox = roi
    win._open_3d()
    vol = win._native3d
    assert vol is not None, f"the first 3D view never opened: {win.said}"
    for iy, ix in ((0, 0), (0, 1), (1, 0)):
        vol._add_layer((CHANNELS[0], (iy, ix)), CHANNELS[0],
                       np.zeros((N_Z, 16, 16), np.uint16), (DZ_UM, PX_UM, PX_UM),
                       (0.0, iy * 16 * PX_UM, ix * 16 * PX_UM))
    assert [key_of(ly) for ly in win._napari_viewer().layers].count(
        MosaicKey(op, CHANNELS[0])) >= 3, "the bricks did not take the identity"
    if drag_to is not None:
        # a contrast drag in 3D addresses a brick; the write must not raise across the flip
        mos = win._pane.mosaic
        mos.find(op, CHANNELS[0]).contrast_limits = drag_to
    return vol


# =============================================================================================
# 2. "The stitched view is not exactly the same as that of raw (detected 2 FOVs instead of 4)."
# =============================================================================================

def test_a_SECOND_3d_open_reads_the_whole_ROI_and_not_one_brick_of_the_last_one(stub_bricked):
    """Every brick of the ROI must yield voxels, not just the one brick of the previous view."""
    from squidxplorer._mosaic_source import mosaic_bbox_um

    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))
    mos, _model = _pane_with(meta, op="stitch", result=result)
    roi = mosaic_bbox_um(meta, REGION)
    win = _Window(meta, mos)

    _volume_up(win, roi, stub_bricked, "stitch")
    win._open_3d()                                   # the second click
    vol = win._native3d
    assert vol is not None, f"the second 3D view never opened: {win.said}"

    read = vol._loader._read
    assert read is not None, "3D fell back to the raw reader instead of the stitched layer"
    covered = [b for b in vol._bricks
               if read(vol._offset_brick(b), CHANNELS[0], 1, None) is not None]
    assert len(covered) == len(vol._bricks), (
        f"{len(covered)} of {len(vol._bricks)} bricks of the ROI yielded voxels; the rest of the "
        "volume is empty because the source was one brick of the previous view")
    assert len(vol._bricks) > 1, "the ROI took the single-texture path; this asserts nothing"


def test_the_second_volumes_voxels_are_the_STITCHED_ones_at_full_extent(stub_bricked):
    """Coverage is not enough: the bricks must carry the operator's pixels, at their own places."""
    from squidxplorer._mosaic_source import mosaic_bbox_um

    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))
    mos, _model = _pane_with(meta, op="stitch", result=result)
    roi = mosaic_bbox_um(meta, REGION)
    win = _Window(meta, mos)
    _volume_up(win, roi, stub_bricked, "stitch")
    win._open_3d()

    vol = win._native3d
    far = vol._bricks[-1]                            # the corner furthest from brick 0,0
    voxels = vol._loader._read(vol._offset_brick(far), CHANNELS[1], 1, None)
    assert voxels is not None, "the far corner of the ROI has no voxels"
    assert [int(voxels[z].flat[0]) for z in range(N_Z)] == [_plane_value(z, 1) for z in range(N_Z)]


# =============================================================================================
# 3. "When I stitch in 3d, in window, the contrast changes."
# =============================================================================================

def test_the_volume_opens_on_the_window_the_RENDERED_layer_is_showing(stub_bricked):
    """The volume adopts the contrast of the layer it renders, for every channel that layer has."""
    from squidxplorer._mosaic_source import mosaic_bbox_um

    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))
    mos, _model = _pane_with(meta, op="stitch", result=result, raw_clim=(5.0, 9.0),
                             op_clim=(120.0, 900.0), raw_channels=[CHANNELS[0]])
    assert mos.find(_RAW_OP, CHANNELS[1]) is None, "this window has a raw layer for every channel"
    roi = mosaic_bbox_um(meta, REGION)
    win = _Window(meta, mos)

    win._roi_bbox = roi
    win._open_3d()
    vol = win._native3d
    assert vol is not None, f"3D never opened: {win.said}"
    for channel in CHANNELS:
        assert tuple(vol._contrast_by.get(channel) or ()) == (120.0, 900.0), (
            f"{channel} opened in 3D at {vol._contrast_by.get(channel)}; the stitched layer on "
            "screen is windowed (120.0, 900.0)")


def test_a_contrast_set_IN_3D_survives_the_flip_and_the_next_volume_opens_on_it(stub_bricked):
    """The window the user dragged to while the volume was up is what the next volume opens on."""
    from squidxplorer._mosaic_source import mosaic_bbox_um

    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))
    mos, _model = _pane_with(meta, op="stitch", result=result)
    roi = mosaic_bbox_um(meta, REGION)
    win = _Window(meta, mos)
    # held by OBJECT: while a volume is up this layer has surrendered its identity to the bricks
    flat = mos.find("stitch", CHANNELS[0])

    first = _volume_up(win, roi, stub_bricked, "stitch", drag_to=_DRAGGED_CLIM)
    # the drag reached the flat mosaic under the volume: ONE value per channel, everywhere
    assert tuple(flat.contrast_limits) == _DRAGGED_CLIM

    win._open_3d()

    vol = win._native3d
    # ...and it IS the second view, not the first one's window read back
    assert vol is not None and vol is not first, f"the second 3D view never opened: {win.said}"
    assert tuple(vol._contrast_by[CHANNELS[0]]) == _DRAGGED_CLIM, (
        f"the second 3D view opened at {vol._contrast_by.get(CHANNELS[0])}, not the "
        f"{_DRAGGED_CLIM} the user dragged to in 3D")


# =============================================================================================
# The crash under all of it: a HIDDEN layer's slice contradicts its own slice input across a flip
# =============================================================================================


def _flip_and_write(mos, ndisplay, layer, value):
    """Flip the pane and then write a contrast on *layer*, which is what a drag does."""
    mos.model.dims.ndisplay = ndisplay
    layer.contrast_limits = value


def test_a_contrast_write_survives_a_2D_3D_flip_that_left_a_layer_HIDDEN():
    """Both directions, because both were measured to raise."""
    meta = _meta()
    mos, model = _pane_with(meta, op="raw")
    hidden = mos.find(_RAW_OP, CHANNELS[1])
    shown = mos.find(_RAW_OP, CHANNELS[0])
    hidden.visible = False
    assert int(hidden.ndim) > 2, "a 2-D layer cannot take napari's np.max branch; premise gone"

    _flip_and_write(mos, 3, shown, (11.0, 22.0))
    assert int(hidden._slice_input.ndisplay) == 3
    assert np.ndim(hidden._slice.thumbnail.view) == 3, (
        "the hidden layer kept a 2-D thumbnail under a 3-D slice input")

    _flip_and_write(mos, 2, shown, (33.0, 44.0))
    assert np.ndim(hidden._slice.thumbnail.view) == 2, (
        "the hidden layer kept a 3-D thumbnail under a 2-D slice input")
    # ...and the value really arrived: contrast is one value per channel
    assert tuple(shown.contrast_limits) == (33.0, 44.0)


def test_a_contrast_drag_ON_THE_VOLUME_does_not_raise(stub_bricked):
    """The same defect as the gesture that hits it: the volume is up, the user drags contrast."""
    from squidxplorer._mosaic_source import mosaic_bbox_um

    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))
    mos, _model = _pane_with(meta, op="stitch", result=result)
    win = _Window(meta, mos)
    _volume_up(win, mosaic_bbox_um(meta, REGION), stub_bricked, "stitch")

    mos.set_contrast(CHANNELS[0], 250.0, 3000.0)          # the app's own API, as a drag arrives

    for ly in mos.layers_for("stitch", CHANNELS[0]):
        assert tuple(ly.contrast_limits) == (250.0, 3000.0)


def test_taking_the_volume_down_puts_the_MOSAIC_back_in_charge_of_the_key(stub_bricked):
    """``_close_native3d`` must hand the ``(op, channel)`` identity back to the 2D layers."""
    from squidxplorer._mosaic_source import mosaic_bbox_um

    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))
    mos, _model = _pane_with(meta, op="stitch", result=result)
    roi = mosaic_bbox_um(meta, REGION)
    win = _Window(meta, mos)
    flat = {ch: mos.find("stitch", ch) for ch in CHANNELS}

    _volume_up(win, roi, stub_bricked, "stitch")
    assert key_of(flat[CHANNELS[0]]) is None, "open() no longer surrenders the 2D identity"
    assert mos.find("stitch", CHANNELS[0]) is not flat[CHANNELS[0]], (
        "a brick is not holding the key, so this test's premise is gone")

    win._close_native3d()

    assert win._native3d is None
    for channel in CHANNELS:
        assert mos.find("stitch", channel) is flat[channel], (
            "the 2D layer never got its identity back, so the scene still answers about bricks")
        assert META_KEY in flat[channel].metadata
        assert flat[channel].visible is True, "the layer came back foreign or dark"
    assert int(mos.model.dims.ndisplay) == 2, "the pane was left in 3D with no volume in it"
