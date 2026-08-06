"""Stitching, then 3D in-window: the four things Julio reported driving the real build 2026-08-06.

Verbatim: *"When I stitch in 3d, in window, the contrast changes. Turning off one layer doesn't
turn the other like in the 2D view. The stitched view is not exactly the same as that of raw
(detected 2 FOVs instead of 4). It is not stitching the z-levels in the 3d view separately."*

Sentence 2 is the layer tree's and lives in ``tests/test_layer_tree.py`` beside the rest of the
tree. The other three are here, and two of them share ONE root cause:

* **the z-levels** -- ``_workers._OperatorWorker._on_well`` emitted ``image[0, :, 0]`` for every
  operator, so nine of a ten-plane stitched mosaic died on one unnamed index and the layer
  declared ``z_depth 1``. Measured on the real 10x set (manual0, ``projector="bgsub"``):
  ``stitch_plate`` yielded ``(1, 1, 10, 2084, 7711)`` and the display was handed
  ``(1, 2084, 7711)``.
* **the extent and the contrast** -- ``BrickedVolume.open()`` moves the ``(op, channel)`` identity
  off the pane's 2-D mosaic layers and onto its bricks (2026-08-05, and correct: it is what puts
  the volume in the tree). ``_open_3d`` then read the scene BEFORE closing the old volume, so
  ``MosaicLayers`` answered about the BRICKS: ``find`` returned one 512-px texture, the new
  volume's source was that texture, and its contrast was that texture's. Measured at the seam,
  second 3D click over a bgsub layer: **1 of 9 bricks** of the box yielded voxels, and the
  harvested window came back ``(0.0, 1.0)`` against the ``(120, 900)`` on screen.

The Qt pane and the GL canvas are the only things stood in for. ``_OperatorWorker._on_well``,
``RegionResultAccumulator``, ``MosaicLayers``, ``RegionViewer._open_3d`` / ``_open_roi_3d`` /
``_volume_source`` and ``BrickedVolume.open`` all run for real.
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

from squidmip import _bricks                             # noqa: E402
from squidmip._napari3d import region_origin_um, roi_window_px   # noqa: E402
from squidmip._napari_view import META_KEY, MosaicKey, MosaicLayers, key_of  # noqa: E402
from squidmip._op_result import RegionResultAccumulator  # noqa: E402
from squidmip._region_viewer import _RAW_OP, RegionViewer  # noqa: E402
from squidmip._workers import _OperatorWorker            # noqa: E402

# -- a two-FOV region, small enough to hold in a test and shaped like the real one --------------
N_Z, N_C = 4, 2
FRAME = 64                     # px per FOV, square
OVERLAP = 8                    # px
PX_UM, DZ_UM = 1.0, 2.5
CHANNELS = ["c0", "c1"]
REGION = "manual0"

#: plane k is filled with this, so a dropped, duplicated or reordered plane is a NUMBER that is
#: wrong rather than a shape that happens to match.
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
    """What ``stitch_plate`` yields for a plane-op: ``(T, C, Nz, Y, X)``, every plane fused."""
    out = np.empty((1, N_C, N_Z, h, w), np.uint16)
    for c in range(N_C):
        for z in range(N_Z):
            out[0, c, z] = _plane_value(z, c)
    return out


def _mosaic_hw(meta: dict) -> tuple:
    from squidmip._placement import fov_offsets_px, mosaic_extent_px

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
    """One region's run, from the engine's 5-D yield to the ``OperatorResult`` the display gets.

    ``_on_well`` is the real slot (it runs on write_plate's writer threads in production); the
    accumulator and the ``z_depth`` rule below are ``_viewer._on_result`` / ``_as_result``.
    """
    got: list = []
    worker.resultReady.connect(lambda region, fov, planes: got.append((region, fov, planes)))
    worker._on_well(REGION, 0, image)
    assert got, "the run produced no result for the display"
    _region, fov, planes = got[-1]
    acc = RegionResultAccumulator(worker._operator, REGION, meta, CHANNELS,
                                  region_operator=worker._region_op)
    acc.add(int(fov), np.asarray(planes))
    return acc.result()


def _z_depth(result) -> int:
    """``_viewer._as_result``'s rule, verbatim: the channel axis is already off, so a 3-D plane's
    leading axis can only be z."""
    first = result.planes[0]
    return int(first.shape[0]) if int(getattr(first, "ndim", 2)) >= 3 else 1


# =============================================================================================
# 1. "It is not stitching the z-levels in the 3d view separately."
# =============================================================================================

def test_a_stitched_region_reaches_the_layer_with_EVERY_acquired_plane():
    """The observable is the DEPTH the layer declares, and the VALUES in it.

    A shape check alone passes a stack that is plane 0 broadcast over z, which is why each plane
    carries its own number here.

    MUTATION: return ``well`` unconditionally from ``_OperatorWorker._result_pixels`` (i.e. put
    ``image[0, :, 0]`` back) -> z_depth 1 -> red.
    """
    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))

    assert _z_depth(result) == N_Z, (
        f"the stitched mosaic reached the display with z_depth {_z_depth(result)}; the engine "
        f"fused {N_Z} planes")
    for c, channel in enumerate(CHANNELS):
        vol = np.asarray(result.plane(channel))
        assert vol.shape == (N_Z, h, w), f"{channel}: {vol.shape}"
        assert [int(vol[z].flat[0]) for z in range(N_Z)] == [_plane_value(z, c) for z in range(N_Z)], (
            f"{channel}: the planes are not the ones the engine fused (a broadcast or a reorder)")


def test_the_layer_z_depth_drives_3D_and_a_brick_comes_back_with_every_plane():
    """End of the chain: the depth has to survive into what ``_volume_source`` hands the loader.

    ``_volume_source`` refuses a layer with no z ("carries no z depth here"), which is what a
    stitched layer used to be. This asserts the voxels, not the refusal message.

    MUTATION: as above -> `_volume_source` returns (None, None) and this fails on `source`.
    """
    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))
    mos, _model = _pane_with(meta, op="stitch", result=result)

    win = _Window(meta, mos)
    read, source, _pitch = RegionViewer._volume_source(win, (0, h, 0, w))
    assert source == "stitch" and read is not None, f"3D refused the stitched layer: {win.said}"

    brick = _bricks.Brick(iy=0, ix=0, r0=0, r1=min(16, h), c0=0, c1=min(16, w))
    voxels = read(brick, CHANNELS[0], 1, None)
    assert voxels is not None and voxels.shape[0] == N_Z, (
        f"a brick came back {None if voxels is None else voxels.shape}; a stitched volume is "
        f"{N_Z} planes deep")
    assert [int(voxels[z].flat[0]) for z in range(N_Z)] == [_plane_value(z, 0) for z in range(N_Z)]


def test_a_per_FOV_operator_still_delivers_one_plane_AND_SAYS_SO(caplog):
    """What is deliberately NOT fixed, pinned so it cannot pass for a result.

    A per-FOV operator's mosaic is re-fused for display from tiles the accumulator holds until the
    region is whole, so keeping its depth means ``Nz`` fusions over ~9.4 GB of tiles on one real
    27-FOV well. That path still delivers plane 0 — and now says which plane, and of how many,
    instead of letting a ten-plane operator look like a one-plane one.

    MUTATION: delete the `_z_dropped_note` call in `_result_pixels` -> no line -> red.
    """
    import logging

    meta = _meta()
    worker = _worker(meta, "bgsub")
    assert not worker._region_op, "'bgsub' is a per-FOV operator; this test asserts nothing"
    got: list = []
    worker.resultReady.connect(lambda region, fov, planes: got.append(planes))
    with caplog.at_level(logging.INFO):
        worker._on_well(REGION, 0, _stitched_5d(FRAME, FRAME))

    assert np.asarray(got[-1]).shape == (N_C, FRAME, FRAME), "the per-FOV path grew a z axis"
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "z plane 0 of 4" in said, f"the dropped planes were not named. log: {said!r}"


# =============================================================================================
# The seam: a pane's MosaicLayers over a bare ViewerModel, and a window that borrows the real
# methods. Everything Qt (the widget, the canvas, the loader thread) is what is stood in for.
# =============================================================================================

def _pane_with(meta: dict, *, op: str, result=None, raw_clim=(5.0, 9.0),
               op_clim=(120.0, 900.0), raw_channels=CHANNELS):
    """A pane holding a RAW mosaic and, optionally, one operator layer -- as a window builds them.

    Raw is added first and the operator second, so ``_darken_other_ops`` puts raw out exactly as it
    does in the app: at most one operator per channel is lit.

    *raw_channels* is which channels raw exists for. It defaults to all of them, and a test that
    narrows it is not being contrived: contrast is LINKED per channel across processing layers
    (``_register_channel``), so where raw and the operator both exist they share one window by
    construction and it does not matter which is read. Where raw does NOT exist for a channel the
    operator produced, reading raw yields nothing at all and the brick derives its own with
    ``_auto_clim`` -- which is the contrast changing under the user.
    """
    from squidmip._mosaic_source import mosaic_bbox_um

    model = ViewerModel()
    mos = MosaicLayers(model)
    h, w = _mosaic_hw(meta)
    bbox = mosaic_bbox_um(meta, REGION)
    for channel in raw_channels:
        mos.add_mosaic(_RAW_OP, channel, np.zeros((N_Z, h, w), np.uint16),
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
        # Deliberately small, so the ROI is BRICKED rather than taking the single-texture path:
        # the defect being pinned is one brick standing in for the whole box.
        return 32

    def on_camera_settled(self, cb):
        self.settled.append(cb)


class _Cursor:
    region = REGION


class _Window(QObject):
    """A RegionViewer's 3D entry points, over a real pane model. No QWidget, no GL canvas.

    A ``QObject`` because ``_open_roi_3d`` parents the brick loader (a ``QThread``) to the window,
    which is the join that stops "QThread: Destroyed while thread is still running".
    """

    _open_3d = RegionViewer._open_3d
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
    """``BrickedVolume`` with the three things that need a live Qt event loop stubbed out.

    Starting a real QThread with no QApplication aborts the interpreter, and the camera cannot be
    framed without a canvas. ``open()``'s own layer bookkeeping — which is the subject — runs for
    real, and so does ``_add_layer``.
    """
    import squidmip._brick_view as BV

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


#: The window a BRICK ends up carrying when it is not the mosaic's. Bricks derive their own with
#: ``_napari3d._auto_clim`` whenever the harvest had no entry for the channel, and a drag inside a
#: 3-D view propagates across the bricks and nowhere else -- so a brick's window is not the 2-D
#: layer's, and a harvest that lands on one silently changes what the next view shows.
_BRICK_CLIM = (7.0, 8.0)


def _volume_up(win, roi, stub_bricked, op):
    """Open a first 3D volume and let some of its bricks land, as a live view has them."""
    win._roi_bbox = roi
    win._open_3d()
    vol = win._native3d
    assert vol is not None, f"the first 3D view never opened: {win.said}"
    for iy, ix in ((0, 0), (0, 1), (1, 0)):
        vol._add_layer((CHANNELS[0], (iy, ix)), CHANNELS[0],
                       np.zeros((N_Z, 16, 16), np.uint16), (DZ_UM, PX_UM, PX_UM),
                       (0.0, iy * 16 * PX_UM, ix * 16 * PX_UM))
    for ly in win._napari_viewer().layers:
        if key_of(ly) is not None and ly is not None and "▪" in str(getattr(ly, "name", "")):
            ly.contrast_limits = _BRICK_CLIM
    assert [key_of(ly) for ly in win._napari_viewer().layers].count(
        MosaicKey(op, CHANNELS[0])) >= 3, "the bricks did not take the identity"
    return vol


# =============================================================================================
# 2. "The stitched view is not exactly the same as that of raw (detected 2 FOVs instead of 4)."
# =============================================================================================

def test_a_SECOND_3d_open_reads_the_whole_ROI_and_not_one_brick_of_the_last_one(stub_bricked):
    """THE REPORTED BUG, as coverage of the box rather than as a call order.

    Every brick of the ROI must yield voxels. With the old ordering the source was one 16-px brick
    of the volume still on screen, so exactly one brick of the box could be read and the rest of
    the volume was empty -- fewer fields than 2D shows, with nothing said.

    MUTATION: move `self._close_native3d()` in `_open_3d` to after `_open_roi_3d` returns (or
    delete it, leaving the close inside `_replace_native3d`) -> 1 of 9 bricks -> red.
    """
    from squidmip._mosaic_source import mosaic_bbox_um

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
    """Coverage is not enough: the bricks must carry the operator's pixels, at their own places.

    MUTATION: as above -> the far bricks read None and the plane values never appear -> red.
    """
    from squidmip._mosaic_source import mosaic_bbox_um

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
    """The volume adopts the contrast of the layer it RENDERS, for every channel that layer has.

    The harvest read ``find(_RAW_OP, ...)`` whatever was about to be rendered. Where raw and the
    operator both exist that is harmless -- contrast is linked per channel, so they hold one value
    -- and where raw does NOT exist for a channel the operator produced, it yields nothing at all
    and ``_brick_view._add_layer`` derives one with ``_auto_clim``. So the channel that raw has is
    the control, and the channel it does not is the measurement.

    MUTATION: pass `_RAW_OP` to `_on_screen_luts` in `_open_roi_3d` -> c1 has no entry -> KeyError
    -> red.
    """
    from squidmip._mosaic_source import mosaic_bbox_um

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


def test_a_SECOND_3d_open_keeps_the_window_instead_of_taking_a_bricks(stub_bricked):
    """The other half: while a volume is up the pane's own layers are FOREIGN, so the harvest
    landed on a brick and the second view opened at that texture's autoscale.

    MUTATION: move `self._close_native3d()` in `_open_3d` after the dispatch -> the harvest reads a
    brick, whose window is the (0, 0) `_auto_clim` of a zero array -> red.
    """
    from squidmip._mosaic_source import mosaic_bbox_um

    meta = _meta()
    h, w = _mosaic_hw(meta)
    result = _deliver(_worker(meta, "stitch"), meta, _stitched_5d(h, w))
    mos, _model = _pane_with(meta, op="stitch", result=result)
    roi = mosaic_bbox_um(meta, REGION)
    win = _Window(meta, mos)

    first = _volume_up(win, roi, stub_bricked, "stitch")
    win._open_3d()

    vol = win._native3d
    # ...and it IS the second view. Reading the first one's window back would pass this for the
    # wrong reason on a click that silently rendered nothing.
    assert vol is not None and vol is not first, f"the second 3D view never opened: {win.said}"
    assert tuple(vol._contrast_by[CHANNELS[0]]) == (120.0, 900.0), (
        f"the second 3D view opened at {vol._contrast_by.get(CHANNELS[0])}")


def test_taking_the_volume_down_puts_the_MOSAIC_back_in_charge_of_the_key(stub_bricked):
    """The mechanism under both defects above, stated as the property it rests on.

    ``BrickedVolume.open()`` takes ``(op, channel)`` off the 2-D layers on purpose -- it is what
    stops the tree laying a flat, coarser mosaic across the volume. The consequence is that while a
    volume is up, ``MosaicLayers`` is not describing the mosaic, so nothing may ask it what is on
    screen until the volume is DOWN. ``_close_native3d`` is that seam, and this is what it has to
    restore for the two tests above to mean anything.

    MUTATION: make `_close_native3d` a no-op -> `find` still answers with a brick -> red.
    """
    from squidmip._mosaic_source import mosaic_bbox_um

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
