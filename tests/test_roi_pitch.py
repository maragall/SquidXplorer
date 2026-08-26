"""Which micrometres-per-pixel each 3D path is entitled to."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("qtpy")


@pytest.fixture(autouse=True)
def _sync_slicing_for_determinism(monkeypatch):
    """The volume-push pins here were written for SYNCHRONOUS slicing and flake under the async default (order-dependent, solo-green)."""
    from napari.components import ViewerModel

    orig = ViewerModel.__init__

    def patched(self, *a, **k):
        orig(self, *a, **k)
        self._layer_slicer._force_sync = True

    monkeypatch.setattr(ViewerModel, "__init__", patched)
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidxplorer import _bricks  # noqa: E402
from squidxplorer._address import Extent  # noqa: E402
from squidxplorer._result import Result  # noqa: E402
from squidxplorer import _viewer as V  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)

#: The decimation the real 10x acquisition gets out of `fuse_region_pyramid`.
MEASURED_FUSE_STEP = 2


def _window_with_layers(qapp, napari_pane_stub, root):  # noqa: F811
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = win._viewer_manager.open(list(win._order))
    assert w is not None, "no window was opened"
    pane = napari_pane_stub[-1]
    assert _drain_until(qapp, lambda: bool(len(pane._viewer.layers)), timeout=30), (
        "no mosaic ever reached the window's viewer")
    return win, w, pane


def _raw_layers(win, pane):
    got = [pane.mosaic.find("raw", c["name"]) for c in win._meta["channels"]]
    return [ly for ly in got if ly is not None]


class TestTheVolumeIsPushedAtThePitchItsPixelsHave:
    """``_render_roi_volume`` renders the 2D LAYER's pixels, so it must carry the LAYER's pitch."""

    def test_the_3d_scale_equals_the_2d_layers_own_scale_for_the_same_layer(
        self, qapp, napari_pane_stub, squid_dataset, monkeypatch  # noqa: F811
    ):
        """The 3D push of the layer's pixels must carry the layer's own scale."""
        import squidxplorer._napari3d as napari3d

        root, _ = squid_dataset
        win, w, pane = _window_with_layers(qapp, napari_pane_stub, root)
        px, dz = win._meta["pixel_size_um"], win._meta["dz_um"]
        for layer in _raw_layers(win, pane):
            layer.scale = (dz, MEASURED_FUSE_STEP * px, MEASURED_FUSE_STEP * px)
        assert _raw_layers(win, pane), "the fixture put no raw layer on screen"

        pushes = []
        monkeypatch.setattr(
            napari3d, "open_native_3d_volume",
            lambda volumes, **kw: pushes.append(kw) or object())

        w._render_roi_volume(pane.mosaic, {}, {})

        assert pushes, f"nothing was pushed in 3D: {pane.said}"
        scale = pushes[-1]["scale"]
        layer = _raw_layers(win, pane)[0]
        assert tuple(scale[-2:]) == tuple(layer.scale[-2:]), (
            f"3D pushed {tuple(scale[-2:])} um/px for pixels the 2D view places at "
            f"{tuple(layer.scale[-2:])} um/px. The acquisition's pixel_size_um is {px}, which is "
            f"the pitch of the READER's planes, not of this fused mosaic — using it here renders "
            f"the volume {MEASURED_FUSE_STEP}x too small in y and x with z at full size.")
        assert scale[0] == dz, "z must still be the acquisition's step"
        shutdown_plate_window(qapp, win)

    def test_the_dz_STANDIN_is_isotropic_with_these_voxels_not_with_the_acquisitions(
        self, qapp, napari_pane_stub, squid_dataset, monkeypatch  # noqa: F811
    ):
        """The isotropic dz stand-in must match the voxels being drawn, not the acquisition."""
        import squidxplorer._napari3d as napari3d

        root, _ = squid_dataset
        win, w, pane = _window_with_layers(qapp, napari_pane_stub, root)
        px = win._meta["pixel_size_um"]
        pitch = MEASURED_FUSE_STEP * px
        for layer in _raw_layers(win, pane):
            layer.scale = (1.0, pitch, pitch)
        w._meta = dict(win._meta, dz_um=None)  # an acquisition that declares no z step

        pushes = []
        monkeypatch.setattr(
            napari3d, "open_native_3d_volume",
            lambda volumes, **kw: pushes.append(kw) or object())

        w._render_roi_volume(pane.mosaic, {}, {})

        assert pushes, f"nothing was pushed in 3D: {pane.said}"
        scale = pushes[-1]["scale"]
        assert scale[0] == pytest.approx(scale[1]) == pytest.approx(pitch), (
            f"the isotropic stand-in pushed z={scale[0]} beside xy={scale[1:]}: that is not "
            f"isotropic, it is the acquisition's {px} um/px standing next to displayed voxels of "
            f"{pitch} um/px")
        shutdown_plate_window(qapp, win)

    def test_a_layer_with_no_placement_is_REFUSED_BY_NAME_not_redescribed_at_pixel_size_um(
        self, qapp, napari_pane_stub, squid_dataset, monkeypatch  # noqa: F811
    ):
        import squidxplorer._napari3d as napari3d
        from squidxplorer import _volume_view

        root, _ = squid_dataset
        win, w, pane = _window_with_layers(qapp, napari_pane_stub, root)
        channel = win._meta["channels"][0]["name"]

        class _Unplaced:
            scale = None

        assert _volume_view.displayed_pitch_um(w, _Unplaced(), what=channel) is None
        said = " ".join(pane.said)
        assert "refused" in said.lower(), f"the refusal was silent: {pane.said}"
        assert channel in said, f"the refusal did not name what is missing: {pane.said}"

        monkeypatch.setattr(_volume_view, "displayed_pitch_um",
                            lambda _win, _layer, *, what: None)
        pushes = []
        monkeypatch.setattr(
            napari3d, "open_native_3d_volume",
            lambda volumes, **kw: pushes.append(kw) or object())

        w._render_roi_volume(pane.mosaic, {}, {})

        assert not pushes, (
            f"a volume was pushed with scale {pushes[-1].get('scale')} for pixels whose pitch is "
            f"unknown — the only number available is pixel_size_um and it is the wrong one")
        shutdown_plate_window(qapp, win)


class TestTheVolumeSourceReportsWhichPitchItChose:
    """``_open_roi_3d`` prints what the voxels are; the pitch travels back with the source."""

    def test_the_raw_source_reports_no_pitch_of_its_own_because_it_is_the_acquisitions(
        self, qapp, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        root, _ = squid_dataset
        win, w, pane = _window_with_layers(qapp, napari_pane_stub, root)

        got = w._volume_source((0, 4, 0, 4))

        assert len(got) == 3, f"_volume_source answered {got!r}; _open_roi_3d unpacks three"
        read, source, pitch = got
        assert read is None and source == "raw", "no operator is displayed, so this is the reader"
        assert pitch is None, (
            f"the raw source reported a pitch of {pitch}: it has none of its own, because "
            f"read_brick reads the acquisition's own planes and the caller says so")
        shutdown_plate_window(qapp, win)

    def test_an_operator_layer_is_the_source_and_reports_ITS_OWN_pitch(
        self, qapp, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        """The source is the operator's layer, and the pitch that travels back is that layer's."""
        root, _ = squid_dataset
        win, w, pane = _window_with_layers(qapp, napari_pane_stub, root)
        channels = [c["name"] for c in win._meta["channels"]]
        px, dz = win._meta["pixel_size_um"], win._meta["dz_um"]

        planes = [np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8) + i * 100
                  for i, _ in enumerate(channels)]
        result = Result.of(Extent(region_id=w.current_region()), planes,
                           channels=tuple(channels), z_depth=2,
                           pixel_size_um=MEASURED_FUSE_STEP * px, dtype="uint16")
        assert w.deliver_result("demo", result, visible=True) == len(channels)
        for ch in channels:
            raw = pane.mosaic.find("raw", ch)
            if raw is not None:
                raw.visible = False
        for ch in channels:
            layer = pane.mosaic.find("demo", ch)
            assert layer is not None, f"the operator layer for {ch} never landed"
            layer.visible = True
            layer.scale = (dz, MEASURED_FUSE_STEP * px, MEASURED_FUSE_STEP * px)
            layer.translate = (0.0, 0.0, 0.0)

        read, source, pitch = w._volume_source((0, 4, 0, 4))

        assert source == "demo", (
            f"the visible operator layer is 'demo' and the source said {source!r} -- this is the "
            f"answer `visible_op()` decides, and a stub without it answered 'raw' forever")
        assert read is not None, "the operator branch produced no brick reader"
        assert tuple(pitch) == pytest.approx((MEASURED_FUSE_STEP * px, MEASURED_FUSE_STEP * px)), (
            f"the operator source reported {pitch} um/px where its own layer is placed at "
            f"{MEASURED_FUSE_STEP * px}: a fused preview is DECIMATED and does not carry the "
            f"acquisition's pitch")
        assert tuple(pitch) != pytest.approx((px, px)), (
            "the acquisition's pitch was reported for fused pixels")
        shutdown_plate_window(qapp, win)


class TestTheROIClampCountsACQUISITIONPixels:
    """A drawn ROI's 3D reads FOV planes off the reader, so the clamp counts acquisition pixels."""

    def test_the_clamped_box_is_limit_ACQUISITION_pixels_and_that_is_what_fits_one_texture(
        self, qapp, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        root, _ = squid_dataset
        win, w, pane = _window_with_layers(qapp, napari_pane_stub, root)
        px = win._meta["pixel_size_um"]
        displayed = MEASURED_FUSE_STEP * px
        for layer in _raw_layers(win, pane):
            layer.scale = (win._meta["dz_um"], displayed, displayed)

        class _Shapes:
            """A napari Shapes layer as ``_clamp_last_roi`` reads and writes it."""

            def __init__(self, rects):
                self.data = rects

        huge = np.array([[0.0, 0.0], [0.0, 1e6], [1e6, 1e6], [1e6, 0.0]])
        shapes = _Shapes([huge])
        w._clamp_last_roi(shapes)

        got = np.asarray(shapes.data[-1])
        span = float(got[:, -1].max() - got[:, -1].min())
        limit = w._live_texture_limit()
        nz = len(list(win._meta["z_levels"]))
        assert round(span / px) == limit, (
            f"the clamped box is {span:.1f} um = {span / px:.0f} acquisition pixels, not "
            f"{limit}. Level-0 mosaic pixels are the unit roi_window_px converts into and "
            f"_bricks.plan tiles; at the displayed {displayed} um/px this box is only "
            f"{span / displayed:.0f} voxels and the ceiling would be reported at half the ROI "
            f"it can really render.")
        assert _bricks.fits_single_texture(round(span / px), round(span / px), nz, limit), (
            "the clamped box does not fit one texture of the voxels 3D will read")
        at_displayed = round(limit * displayed / px)
        assert not _bricks.fits_single_texture(at_displayed, at_displayed, nz, limit), (
            f"this fixture cannot tell the two pitches apart ({at_displayed} still fits "
            f"{limit}), so the assertion above proves nothing about the unit")
        shutdown_plate_window(qapp, win)

