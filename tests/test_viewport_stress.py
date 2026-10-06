"""A burst of zooms and pans through napari's OWN async slicing leaves every channel loaded,
at one level, showing exactly the pixels of its window: the viewport guarantee, as a pin."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("napari")

from tests.test_mosaic_source import _StepReader, _pyr_meta  # noqa: E402

CANVAS = (600, 600)


@pytest.fixture
def scene():
    """Two channels of a 16-FOV, 4-z region in a real ViewerModel with async slices applied."""
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QApplication

    from squidxplorer._mosaic_source import fuse_region_pyramid, mosaic_bbox_um
    from squidxplorer._napari_pane import attach_async_slice_apply
    from squidxplorer._napari_view import MosaicLayers

    app = QApplication.instance() or QApplication([])
    model = ViewerModel()
    attach_async_slice_apply(model)
    ml = MosaicLayers(model)
    reader = _StepReader(frame=(256, 256))
    meta = _pyr_meta(nz=4, frame=(256, 256))
    meta["channels"] = [{"name": "405"}, {"name": "488"}]
    bbox = mosaic_bbox_um(meta, "A1")
    layers = {}
    for ch in ("405", "488"):
        levels, _step, _nz = fuse_region_pyramid(reader, meta, "A1", ch, max_px=1024)
        layers[ch] = ml.add_mosaic("raw", ch, levels, multiscale=True, bbox_um=bbox,
                                   z_scale_um=1.5)
    model.dims.set_point(0, 2 * 1.5)
    yield app, model, ml, layers, bbox
    ml.set_dataset(None) if hasattr(ml, "set_dataset") else None


def _settle(app, model):
    model._layer_slicer.wait_until_idle(timeout=30)
    for _ in range(50):
        app.processEvents()


def _draw(model, y0, x0, h, w):
    corners = np.array([[y0, x0], [y0 + h, x0 + w]], float)
    for ly in model.layers:
        ly._update_draw(scale_factor=h / CANVAS[0], corner_pixels_displayed=corners,
                        shape_threshold=np.array(CANVAS))


def test_a_burst_of_zooms_and_pans_leaves_every_channel_loaded_at_one_level_with_its_own_pixels(
        scene, monkeypatch):
    """Julio (2026-10-05): zooming in and out "can get stuck as well as show only one channel
    on the edges". Forty random viewports with no settling between them, through napari's
    own slicer, then one settle: every channel loaded (no stranded slice), both at the same
    level and window, no slicing task raised, and the slice on screen is bit-exact to the
    rung's own window."""
    from napari.components import _layer_slicer as LS

    app, model, ml, layers, bbox = scene
    raised = []
    orig = LS._LayerSlicer._slice_layers

    def spy(self, requests):
        try:
            return orig(self, requests)
        except BaseException as exc:            # noqa: BLE001 - recorded, then re-raised
            raised.append(repr(exc))
            raise

    monkeypatch.setattr(LS._LayerSlicer, "_slice_layers", spy)

    x0, y0, x1, y1 = bbox
    W, H = x1 - x0, y1 - y0
    rng = np.random.default_rng(7)
    last = None
    _settle(app, model)
    for _ in range(40):
        frac = float(rng.choice([1.0, 0.5, 0.25, 0.1, 0.05]))
        h = H * frac
        w = min(W, h * (W / H))
        yy = y0 + rng.uniform(0, max(0.0, H - h))
        xx = x0 + rng.uniform(0, max(0.0, W - w))
        _draw(model, yy, xx, h, w)
        last = (yy, xx, h, w)
        for _ in range(2):
            app.processEvents()
    _settle(app, model)
    _settle(app, model)                      # a re-requested slice's own round trip

    assert raised == [], f"slicing tasks raised: {raised}"
    states = {ch: (int(ly.data_level), ly.corner_pixels.tolist(), bool(ly.loaded))
              for ch, ly in layers.items()}
    assert all(s[2] for s in states.values()), f"a channel is stranded: {states}"
    assert len({(s[0], str(s[1])) for s in states.values()}) == 1, (
        f"channels disagree on level or window: {states}")

    # The pixels on screen ARE the rung's own window at the z on screen, for every channel.
    for ch, ly in layers.items():
        level = int(ly.data_level)
        rung = ly.data[level]
        (r0, c0), (r1, c1) = ly.corner_pixels[:, -2:].tolist()
        z = int(round(model.dims.point[0] / 1.5))
        want = np.asarray(rung[z, r0:r1 + 1, c0:c1 + 1])
        shown = np.asarray(ly._slice.image.view)
        assert shown.shape == want.shape, (ch, shown.shape, want.shape)
        assert np.array_equal(shown, want), f"{ch}: the slice on screen is not its own window"
