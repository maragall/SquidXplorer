"""WHICH MICROMETRES-PER-PIXEL each 3D path is entitled to, measured 2026-08-06.

There are two of them in this app and they are not interchangeable, which is why this file exists
rather than a single assertion somewhere:

* ``meta["pixel_size_um"]`` -- the ACQUISITION's pitch (0.752 um/px on the 10x set). It is the unit
  of LEVEL-0 MOSAIC PIXELS: what ``_placement.fov_offsets_px`` lays FOVs out in, what
  ``_napari3d.roi_window_px`` converts an ROI box into, what ``_bricks.plan`` tiles, and what
  ``read_brick`` hands back when it reads planes straight off the reader.
* the DISPLAYED pitch -- what the mosaic on screen actually has. ``fuse_region_pyramid`` decimates
  (``step = ceil(mosaic_px / _MAX_FUSED_PX)``, measured as 2 on
  ``test_10x_laser_af_z_stack_2025-10-28``, level 0 (10, 5731, 4794)), and the layer records the
  truth in its own ``scale`` because ``add_mosaic`` places it from ``bbox_um / shape``: 1.5040 y,
  1.5038 x, exactly 2x the acquisition's.

A path that renders the LAYER's pixels must be told the layer's pitch; a path that reads the
READER's must be told the acquisition's. Getting that backwards does not fail — it draws a volume
with the wrong aspect ratio, or clamps a box to a quarter of the area it promises, and both look
plausible on screen.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("qtpy")
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

#: The decimation the real 10x acquisition gets out of `fuse_region_pyramid`. The fixture's mosaic
#: is 1542 px wide, far under `_MAX_FUSED_PX`, so its step is 1 and the two pitches coincide there
#: — which is exactly why the bug survived: a fixture that cannot tell them apart cannot fail.
MEASURED_FUSE_STEP = 2


def _window_with_layers(qapp, napari_pane_stub, root):  # noqa: F811
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = win._viewer_manager.open(list(win._order))
    assert w is not None, "no window was opened"
    pane = napari_pane_stub[-1]
    assert _drain_until(qapp, lambda: bool(pane.mosaic.added), timeout=30), (
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
        """The whole fix, as one equality, checkable without a GPU.

        The 2D layer is placed from ``bbox_um / shape`` and therefore already carries the pitch of
        the pixels it holds. The 3D push of THOSE SAME PIXELS must carry the same number, or the
        volume is drawn at a physical size the 2D view disagrees with — 2x too small in y and x
        with z at full size, i.e. a stack that reads as tall rather than as wrong.

        MUTATION (the shipped code until 2026-08-06): ``px = meta["pixel_size_um"]`` and
        ``scale=(dz, px, px)`` -> pushes 0.325 against a layer at 0.650 -> red, with both numbers
        in the message.
        """
        import squidxplorer._napari3d as napari3d

        root, _ = squid_dataset
        win, w, pane = _window_with_layers(qapp, napari_pane_stub, root)
        px, dz = win._meta["pixel_size_um"], win._meta["dz_um"]
        # Make this fixture's mosaic DECIMATED, which is what every real acquisition's is: the
        # layer keeps the pitch of its own pixels whatever produced them.
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
        """``z_step_um`` substitutes the xy pitch when ``dz_um`` is missing, and "assume isotropic"
        has to mean isotropic with the voxels being drawn. Handed ``pixel_size_um`` next to a
        displayed pitch of 2x that, the stand-in itself renders the stack half as tall as it claims.

        MUTATION: ``z_step_um(meta, px, ...)`` -> z comes back 0.325 against xy 0.650 -> red.
        """
        import squidxplorer._napari3d as napari3d

        root, _ = squid_dataset
        win, w, pane = _window_with_layers(qapp, napari_pane_stub, root)
        px = win._meta["pixel_size_um"]
        pitch = MEASURED_FUSE_STEP * px
        for layer in _raw_layers(win, pane):
            layer.scale = (1.0, pitch, pitch)
        w._meta = dict(win._meta, dz_um=None)            # an acquisition that declares no z step

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
        """No plausible fallback. ``pixel_size_um`` is available at that moment and is exactly the
        wrong answer, so falling back to it would render a volume at a physical size nothing
        measured — silently, and only for the layers that lost their placement.

        MUTATION: ``pitch = pitch or (px, px)`` -> a volume is pushed -> red.
        """
        import squidxplorer._napari3d as napari3d

        root, _ = squid_dataset
        win, w, pane = _window_with_layers(qapp, napari_pane_stub, root)
        for layer in _raw_layers(win, pane):
            layer.scale = None                           # a layer napari never placed

        pushes = []
        monkeypatch.setattr(
            napari3d, "open_native_3d_volume",
            lambda volumes, **kw: pushes.append(kw) or object())

        w._render_roi_volume(pane.mosaic, {}, {})

        assert not pushes, (
            f"a volume was pushed with scale {pushes[-1].get('scale')} for pixels whose pitch is "
            f"unknown — the only number available is pixel_size_um and it is the wrong one")
        said = " ".join(pane.said)
        channel = win._meta["channels"][0]["name"]
        assert "refused" in said.lower(), f"the refusal was silent: {pane.said}"
        assert channel in said, f"the refusal did not name what is missing: {pane.said}"
        shutdown_plate_window(qapp, win)


class TestTheVolumeSourceReportsWhichPitchItChose:
    """``_open_roi_3d`` prints what the voxels are; ``_volume_source`` is the only thing that knows.

    The bricked path is the one a drawn ROI actually takes, and it has TWO sources: the reader's
    raw planes (acquisition pitch) and the window's own operator layer (that layer's pitch, which
    is the fused preview's). The log line said "at 0.752 um/px" for both. The pitch travels back
    with the source now, so the sentence cannot describe the wrong pixels.
    """

    def test_the_raw_source_reports_no_pitch_of_its_own_because_it_is_the_acquisitions(
        self, qapp, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        """MUTATION: return a 2-tuple again -> ``_open_roi_3d``'s unpack raises -> red here."""
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
        """The OTHER branch, and until now no window test could reach it.

        `_volume_source` asks `mosaic.visible_op()` inside ``except Exception: return raw``, and
        `tests/conftest.py`'s `StubMosaic` had no `visible_op` -- so every window in this suite
        answered RAW whatever was on screen, and the sibling above ("no operator is displayed, so
        this is the reader") passed for that reason rather than for its own. With the stub honest,
        this is what the operator branch is owed: the source is the OPERATOR's layer, and the
        pitch that travels back is THAT LAYER's, not the acquisition's.
        """
        root, _ = squid_dataset
        win, w, pane = _window_with_layers(qapp, napari_pane_stub, root)
        channels = [c["name"] for c in win._meta["channels"]]
        px, dz = win._meta["pixel_size_um"], win._meta["dz_um"]

        # A z-PRESERVING result: `_volume_source` refuses a single plane by declaration, and
        # rightly, so the volume has to have depth for the branch to be reached at all.
        planes = [np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8) + i * 100
                  for i, _ in enumerate(channels)]
        result = Result.of(Extent(region_id=w.current_region()), planes,
                           channels=tuple(channels), z_depth=2,
                           pixel_size_um=MEASURED_FUSE_STEP * px, dtype="uint16")
        assert w.deliver_result("bgsub", result, visible=True) == len(channels)
        # Raw is what a window shows underneath; the operator layer is the one on top and visible.
        for ch in channels:
            raw = pane.mosaic.find("raw", ch)
            if raw is not None:
                raw.visible = False
        for ch in channels:
            layer = pane.mosaic.find("bgsub", ch)
            assert layer is not None, f"the operator layer for {ch} never landed"
            layer.visible = True
            layer.scale = (dz, MEASURED_FUSE_STEP * px, MEASURED_FUSE_STEP * px)
            layer.translate = (0.0, 0.0, 0.0)

        read, source, pitch = w._volume_source((0, 4, 0, 4))

        assert source == "bgsub", (
            f"the visible operator layer is 'bgsub' and the source said {source!r} -- this is the "
            f"answer `visible_op()` decides, and a stub without it answered 'raw' forever")
        assert read is not None, "the operator branch produced no brick reader"
        # The pitch travels back as the layer's own (y, x) micrometres-per-pixel.
        assert tuple(pitch) == pytest.approx((MEASURED_FUSE_STEP * px, MEASURED_FUSE_STEP * px)), (
            f"the operator source reported {pitch} um/px where its own layer is placed at "
            f"{MEASURED_FUSE_STEP * px}: a fused preview is DECIMATED and does not carry the "
            f"acquisition's pitch")
        assert tuple(pitch) != pytest.approx((px, px)), (
            "the acquisition's pitch was reported for fused pixels")
        shutdown_plate_window(qapp, win)


class TestTheROIClampCountsACQUISITIONPixels:
    """The other half of the same question, and the answer is the OTHER pitch.

    Re-measured on ``test_10x_laser_af_z_stack_2025-10-28``, region manual0, because the displayed
    pitch is the tempting answer here too and it is wrong: a drawn ROI's 3D goes through
    ``_open_roi_3d`` -> ``BrickedVolume`` -> ``read_brick``, which reads FOV planes off the reader.
    A 1540 um clamped box became a 2048 x 2048 level-0 window, ``fits_single_texture`` True, and
    ``read_brick`` on a 256 px slice of it returned 256 voxels across 192.5 um = 0.7520 um/voxel.
    Clamped at the displayed 1.504 instead: 3080 um -> 4095 x 4095 level-0 px,
    ``fits_single_texture`` False, 16 bricks — 4x the pixels read, under a status line promising
    one texture at full resolution.
    """

    def test_the_clamped_box_is_limit_ACQUISITION_pixels_and_that_is_what_fits_one_texture(
        self, qapp, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        """MUTATION: clamp at the displayed pitch (``layer.scale[-1]``) -> the box doubles ->
        red on the second assertion, which is the 3D renderer refusing to fit it in one texture."""
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
        # ...and the consequence of counting the box in DISPLAYED pixels instead, stated as the
        # arithmetic rather than as a comment: the same rule at 1.504 um/px passes a box that the
        # renderer then has to brick, under a sentence promising one texture.
        at_displayed = round(limit * displayed / px)
        assert not _bricks.fits_single_texture(at_displayed, at_displayed, nz, limit), (
            f"this fixture cannot tell the two pitches apart ({at_displayed} still fits "
            f"{limit}), so the assertion above proves nothing about the unit")
        shutdown_plate_window(qapp, win)
