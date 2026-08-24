"""napari's async slicing is ON for this app: a viewport reslice decodes off the Qt thread.

Julio, live on the 900-FOV 20x set: "when I zoom in rapidly, it's not responsive." The stall
is the Qt thread: sync slicing materialises the viewport's FOV decodes inside the draw.
``squidxplorer/__init__`` sets ``NAPARI_ASYNC=1`` (the one NON-persisting channel — see
``_async_slicing``) before napari's settings are born, so every viewer in the process slices
on the pool. These tests run under exactly that configuration, like production.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from napari.components import ViewerModel

from squidxplorer import _mosaic_source
from squidxplorer._async_slicing import configure
from squidxplorer._napari_view import MosaicLayers

CH = "Fluorescence_405_nm_Ex"
FRAME = (64, 64)
GRID = 4
STEP_UM = 64.0
PX = 1.0


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _meta() -> dict:
    fovs = list(range(GRID * GRID))
    return {
        "regions": ["A1"],
        "fovs_per_region": {"A1": fovs},
        "fov_positions_um": {("A1", f): ((f % GRID) * STEP_UM, (f // GRID) * STEP_UM)
                             for f in fovs},
        "channels": [{"name": CH}],
        "n_z": 1, "z_levels": [0], "n_t": 1,
        "pixel_size_um": PX, "frame_shape": FRAME, "dtype": "uint16",
    }


class _RecordingReader:
    """Planes keyed to the FOV id; every read records the thread that asked."""

    def __init__(self, meta):
        self.source_id = "mem://async"        # not a directory: the well-image path stays out
        self._meta = meta
        self.read_threads: list[int] = []

    def read(self, region, fov, channel, z_level, time_point=0):
        self.read_threads.append(threading.get_ident())
        return np.full(FRAME, 100 + int(fov) * 7, np.uint16)


def _scene(qapp):
    meta = _meta()
    reader = _RecordingReader(meta)
    # A cache too small to retain a plane: every reslice genuinely decodes, so the
    # thread-ident instrument sees the reads a big cache would absorb.
    levels, _step, _nz = _mosaic_source.fuse_region_pyramid(
        reader, meta, "A1", CH, cache_bytes=1)
    viewer = ViewerModel()
    # The Qt half: napari's only ready-event consumer is QtViewer, so a headless model
    # needs the same apply adapter ModelPane installs (production panes have QtViewer).
    from squidxplorer._napari_pane import attach_async_slice_apply

    apply_ref = attach_async_slice_apply(viewer)
    assert apply_ref is not None, "the apply adapter must attach on this napari"
    mosaic = MosaicLayers(viewer)
    layer = mosaic.add_mosaic("raw", CH, levels, bbox_um=(0.0, 0.0, 256.0, 256.0),
                              multiscale=len(levels) > 1)
    return viewer, mosaic, layer, reader


def _zoom_to_native(layer) -> None:
    """The real zoom entry: the canvas hands the layer its viewport on every camera move,
    and ``_update_draw`` picks the data level, sets the corners and triggers the reslice."""
    layer._update_draw(scale_factor=1.0,
                       corner_pixels_displayed=np.array([[0.0, 0.0], [255.0, 255.0]]),
                       shape_threshold=(256, 256))


def _drain(qapp, seconds=1.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        qapp.processEvents()
        time.sleep(0.01)


def test_the_decision_respects_both_opt_outs():
    env: dict = {}
    assert configure(env) is True and env["NAPARI_ASYNC"] == "1"
    env = {"SQUIDXPLORER_SYNC_SLICING": "1"}
    assert configure(env) is False and "NAPARI_ASYNC" not in env, \
        "the escape hatch must leave napari alone"
    env = {"NAPARI_ASYNC": "0"}
    assert configure(env) is False and env["NAPARI_ASYNC"] == "0", \
        "the user's own NAPARI_ASYNC wins either way"


def test_importing_squidxplorer_turns_async_slicing_on_for_every_viewer(qapp):
    from napari.settings import get_settings

    assert get_settings().experimental.async_ is True, \
        "importing squidxplorer must have enabled async slicing before settings were built"
    assert ViewerModel()._layer_slicer._force_sync is False, \
        "a fresh viewer's slicer must be asynchronous"


def test_a_zoom_reslice_never_decodes_on_the_thread_that_asked(qapp):
    viewer, _mosaic, layer, reader = _scene(qapp)

    reader.read_threads.clear()
    _zoom_to_native(layer)
    layer.refresh()                              # the gesture path: refresh routes async
    viewer._layer_slicer.wait_until_idle(timeout=10)
    _drain(qapp)

    here = threading.get_ident()
    assert reader.read_threads, "the reslice must have decoded something"
    assert all(t != here for t in reader.read_threads), \
        "a reslice must never decode on the thread that asked (the Qt thread in the app)"


def test_an_async_slice_lands_the_same_pixels_as_a_sync_one(qapp):
    viewer, _mosaic, layer, reader = _scene(qapp)
    slicer = viewer._layer_slicer

    _zoom_to_native(layer)
    layer.refresh()
    slicer.wait_until_idle(timeout=10)
    _drain(qapp)
    assert layer.loaded, "the async response must have been APPLIED, not only computed"
    got_async = np.array(np.asarray(layer._slice.image.raw), copy=True)

    with slicer.force_sync():
        layer._refresh_sync(data_displayed=True)
    got_sync = np.array(np.asarray(layer._slice.image.raw), copy=True)

    assert got_async.shape == got_sync.shape
    assert np.array_equal(got_async, got_sync), \
        "async and sync slicing must land byte-identical pixels"
