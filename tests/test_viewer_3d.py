"""3D volume rendering of a z-stack (IMA-255) — the SquidMIP half of the seam.

ndviewer_light owns the renderer; this repo owns exactly one thing: telling it the
PHYSICAL voxel size when a raw z-stack is pushed. Without that the volume renders
isotropic, which on the tissue set (dz 1.5um, pixel 0.752um) is 2x squashed in z.

Two guards live here:

* the raw push carries pixel_size_um and dz_um, in micrometres, from the acquisition
  metadata — asserted as NUMBERS off a real fixture, not as "the call happened";
* the INSTALLED ndviewer_light actually accepts those parameters. A stale installed copy
  that silently lacked ``register_array`` once cost this project a day of black-canvas
  debugging; the same failure mode here would quietly restore isotropic rendering, which
  looks plausible and is wrong. So it is checked against the live install, by signature.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidmip import _viewer as V  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)

# WHERE THE VOXEL SIZE NOW GOES (decentralization, 2026-07-23).
#
# `PlateWindow._detail` is permanently None and `_make_detail_viewer` has no production call sites,
# so `start_acquisition(..., pixel_size_um, dz_um)`, the seam the pushes below used to be asserted
# on, is not on any path the app takes. The declaration itself did not disappear: a region window
# hands napari `z_scale_um=meta["dz_um"]` with every mosaic (`_region_viewer.py:874`), and the 3D
# volume pushes hand it `scale=(dz, px, px)` (`_region_viewer.py:1038` and `:1078`).
#
# So the three tests below are re-pointed, not deleted: same question ("does the acquisition's real
# voxel size reach the renderer, as NUMBERS"), asked of the object that now answers it.


def _open_window(win, regions):
    w = win._viewer_manager.open(list(regions))
    assert w is not None, "no window was opened"
    return w


def _wait_for_layers(qapp, pane, timeout=30):
    assert _drain_until(qapp, lambda: bool(pane.mosaic.added), timeout=timeout), (
        "no mosaic ever reached the window's viewer")
    return pane.mosaic.added


# The ndviewer_light seam checks lived here: they asked the INSTALLED ndviewer_light whether
# start_acquisition still took pixel_size_um and dz_um, so a silent upstream signature change
# could not make 3D volumes render isotropic.
#
# Deleted with the fallback itself on 2026-07-30. They are worth a note rather than a silent
# removal, because they are what FOUND the Qt6 blocker: `pytest.importorskip("ndviewer_light.core")`
# pulled PyQt5 into a QT_API=pyqt6 process at module scope, vispy then refused to load PyQt6
# beside it, and the file aborted in teardown. A test that imports a second Qt binding is not a
# cheap test. The seam it guarded no longer exists: napari is the only renderer, and the voxel
# scale it is handed is asserted directly below.


class TestRawPushCarriesVoxelSize:
    """The raw z-stack push is the only one that declares a real n_z, so the only one
    where a volume means anything — and the only one that must carry the voxel size."""

    def test_a_window_declares_the_acquisitions_voxel_depth_to_napari(
        self, qapp, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        """Every raw mosaic a window adds carries the acquisition's dz, in micrometres.

        Without it napari scales z as 1, and on the tissue set (dz 1.5um, pixel 0.752um) the volume
        renders 2x squashed, a picture that looks entirely plausible and is wrong.

        MUTATION: drop `z_scale_um` from the `add_mosaic` call -> None -> red.
        """
        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, win._order)
        added = _wait_for_layers(qapp, napari_pane_stub[-1])

        meta = win._meta
        # The NUMBER, not just "something was passed": a None here is exactly the
        # silent degradation this test exists to catch.
        for op, channel, _levels, kw in added:
            assert kw.get("z_scale_um") == meta["dz_um"], (
                f"{op}/{channel} was added with z_scale_um={kw.get('z_scale_um')!r}")
        assert meta["dz_um"] is not None and meta["dz_um"] > 0
        assert meta["pixel_size_um"] is not None and meta["pixel_size_um"] > 0
        shutdown_plate_window(qapp, win)

    def test_the_3d_volume_push_carries_the_full_voxel_scale(
        self, qapp, napari_pane_stub, squid_dataset, monkeypatch  # noqa: F811
    ):
        """dz_um / pixel_size_um is the z stretch the renderer applies, and it must be
        computable from the pushed values alone, and finite and positive.

        The 3D push declares all three axes at once as `scale=(dz, px, px)`, so the aspect is
        recoverable as scale[0] / scale[1].

        MUTATION: pass `scale=(1.0, px, px)` (or omit it) -> aspect 1 -> red.
        """
        import squidmip._napari3d as napari3d

        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, win._order)
        pane = napari_pane_stub[-1]
        _wait_for_layers(qapp, pane)

        pushes = []
        monkeypatch.setattr(
            napari3d, "open_native_3d_volume",
            lambda volumes, **kw: pushes.append((volumes, kw)) or object())

        w._render_roi_volume(pane.mosaic, {}, {})

        assert pushes, f"nothing was pushed in 3D: {pane.said}"
        _volumes, kw = pushes[-1]
        scale = kw["scale"]
        assert len(scale) == 3, scale
        assert scale[0] == win._meta["dz_um"]
        assert scale[1] == scale[2] == win._meta["pixel_size_um"]
        aspect = scale[0] / scale[1]
        assert aspect > 0
        assert aspect == pytest.approx(win._meta["dz_um"] / win._meta["pixel_size_um"])
        shutdown_plate_window(qapp, win)

    def test_the_raw_mosaic_declares_the_full_z_stack(
        self, qapp, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        """A volume needs more than one plane; the raw mosaic must not collapse z.

        This is what makes the voxel size mean anything at all: a (1, y, x) layer has no z to
        stretch, so a correct `z_scale_um` on a flattened stack is still an isotropic picture.

        MUTATION: fuse a single z plane (or a MIP) into the window -> leading dim 1 -> red.
        """
        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, ["B3"])
        added = _wait_for_layers(qapp, napari_pane_stub[-1])

        n_z = win._meta["n_z"]
        assert n_z > 1, "fixture needs a real z-stack or this asserts nothing"
        for op, channel, levels, _kw in added:
            level0 = levels[0] if isinstance(levels, (list, tuple)) else levels
            assert level0.ndim == 3, f"{op}/{channel} is not a (z, y, x) volume: {level0.shape}"
            assert level0.shape[0] == n_z, (
                f"{op}/{channel} declared {level0.shape[0]} planes, not {n_z}")
        shutdown_plate_window(qapp, win)

    def test_the_3d_volume_is_LEVEL_ZERO_and_not_the_coarsest_pyramid_rung(
        self, qapp, napari_pane_stub, squid_dataset, monkeypatch  # noqa: F811
    ):
        """``_render_roi_volume``'s own docstring promises level 0; it took the pyramid.

        The rung was picked with ``data[0] if isinstance(data, (list, tuple)) else data``, and
        napari hands a multiscale layer's data back as ``MultiScaleData``, which is neither. So the
        whole pyramid went into the volume dict and ``np.asarray`` on it -- ``__array__`` returns
        ``_data[-1]`` -- silently substituted the COARSEST level. A blocky volume, no message.

        MUTATION: restore the isinstance branch in ``_render_roi_volume`` -> the pushed volume
        comes back at the smallest rung's shape -> red.
        """
        import squidmip._napari3d as napari3d

        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, ["B3"])
        pane = napari_pane_stub[-1]
        added = _wait_for_layers(qapp, pane)

        levels_by_channel = {ch: lv for _op, ch, lv, _kw in added}
        assert any(isinstance(lv, (list, tuple)) and len(lv) > 1
                   for lv in levels_by_channel.values()), (
            "the fixture produced a single-rung pyramid, so this asserts nothing")

        pushes = []
        monkeypatch.setattr(
            napari3d, "open_native_3d_volume",
            lambda volumes, **kw: pushes.append(volumes) or object())

        w._render_roi_volume(pane.mosaic, {}, {})

        assert pushes, f"nothing was pushed in 3D: {pane.said}"
        for channel, volume in pushes[-1].items():
            levels = levels_by_channel[channel]
            level0 = levels[0] if isinstance(levels, (list, tuple)) else levels
            assert volume.shape == tuple(level0.shape), (
                f"{channel} was rendered at {volume.shape}, not level 0's {tuple(level0.shape)}")
        shutdown_plate_window(qapp, win)


class TestThe3DPopoutDoesNotPileUp:
    """Julio: "consider changing the 3D interaction so clicking 3D reuses the current window
    instead of opening an extra one".

    Every 3D click used to construct a fresh ``napari.Viewer`` (``_napari3d.py:268``, ``:366``) and
    the window only remembered the LATEST, so the earlier ones stayed on screen: napari keeps every
    Viewer in a global set, so dropping our reference closes nothing. That is a window pile from an
    ordinary gesture, and it is the same family as the open window-lifetime ticket.
    """

    def test_a_second_3D_click_CLOSES_the_first_popout(
        self, qapp, napari_pane_stub, squid_dataset, monkeypatch  # noqa: F811
    ):
        """MUTATION: assign self._native3d directly instead of going through
        _replace_native3d -> the first popout is never closed -> red."""
        import squidmip._napari3d as napari3d

        class _FakeViewer:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        opened = []
        monkeypatch.setattr(
            napari3d, "open_native_3d_volume",
            lambda volumes, **kw: opened.append(_FakeViewer()) or opened[-1])

        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, win._order)
        pane = napari_pane_stub[-1]
        _wait_for_layers(qapp, pane)

        w._render_roi_volume(pane.mosaic, {}, {})
        w._render_roi_volume(pane.mosaic, {}, {})

        assert len(opened) == 2, f"the second 3D click opened nothing: {pane.said}"
        assert opened[0].closed, "the first 3D popout was left on screen"
        assert not opened[1].closed, "the popout the user just asked for was closed"
        assert w._native3d is opened[1]
        shutdown_plate_window(qapp, win)

    def test_a_popout_that_REFUSES_to_close_does_not_block_the_new_one(
        self, qapp, napari_pane_stub, squid_dataset, monkeypatch  # noqa: F811
    ):
        """A stale window whose close() raises (already destroyed, no Qt window) must not be the
        reason the user cannot open the view they asked for."""
        import squidmip._napari3d as napari3d

        class _StuckViewer:
            def close(self):
                raise RuntimeError("wrapped C/C++ object has been deleted")

        opened = []
        monkeypatch.setattr(
            napari3d, "open_native_3d_volume",
            lambda volumes, **kw: opened.append(_StuckViewer()) or opened[-1])

        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, win._order)
        pane = napari_pane_stub[-1]
        _wait_for_layers(qapp, pane)

        w._render_roi_volume(pane.mosaic, {}, {})
        w._render_roi_volume(pane.mosaic, {}, {})

        assert len(opened) == 2, f"a stuck popout blocked the new one: {pane.said}"
        assert w._native3d is opened[1]
        shutdown_plate_window(qapp, win)


# -- the 3D volume must live INSIDE the layer-visibility model ---------------------------------
#
# Julio, driving the real build 2026-08-05: "in 3d rendering, when all layers are off there is
# still a rendered image, unlike 2d that it's a black canvas since all layers are off ... there is
# still a layer that looks beautiful but that I can't control so then other controlled layers are
# overlayed".
#
# One root cause, both symptoms. `_brick_view._add_layer` created every brick with no
# `layer.metadata`, so `key_of` returned None and the layer tree classed the whole volume as a
# FOREIGN layer -- which it "deliberately tolerates and IGNORES". No group, no checkbox, nothing to
# switch off. Meanwhile `BrickedVolume.open()` force-hides the 2-D layers but leaves their
# checkboxes live, so switching one back on drew it over an uncontrollable volume.
#
# This asserts the OBSERVABLE the tree depends on -- `key_of` recovers (op, channel) -- rather than
# that a kwarg was passed. It also asserts the grouping property that makes one checkbox drive the
# whole volume: every brick of a channel must answer with the SAME key.

class _RecordingViewer:
    """The narrowest napari stand-in `_add_layer` needs: it records what was added."""

    class _Layer:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.visible = True
            self.events = None          # _link_contrast is try/except'd; None exercises that path

    def __init__(self):
        self.layers = []

    def add_image(self, arr, **kwargs):
        layer = self._Layer(data=arr, **kwargs)
        self.layers.append(layer)
        return layer


def _bricked(op, channels=("c0", "c1")):
    from squidmip._brick_view import BrickedVolume

    viewer = _RecordingViewer()
    vol = BrickedVolume(
        viewer, reader=None, meta={}, region="A1", window_px=(0, 8, 0, 8),
        channels=list(channels), scale=(1.5, 0.75, 0.75), origin_um=(0.0, 0.0, 0.0),
        limit=2048, budget_bytes=1 << 30, op=op,
    )
    return viewer, vol


@pytest.mark.parametrize("op", ["raw", "decon", "bgsub"])
def test_every_brick_declares_the_operator_whose_volume_it_is(op):
    """Without this the tree cannot see the volume at all, whatever the operator."""
    import numpy as np

    from squidmip._napari_view import key_of

    viewer, vol = _bricked(op)
    for ch in ("c0", "c1"):
        for tile in ((0, 0), (0, 1), (1, 0)):
            vol._add_layer((ch, tile), ch, np.zeros((2, 4, 4), np.uint16),
                           (1.5, 0.75, 0.75), (0.0, 0.0, 0.0))

    assert len(viewer.layers) == 6
    keys = [key_of(ly) for ly in viewer.layers]
    assert all(k is not None for k in keys), (
        "a brick with no metadata is a FOREIGN layer: no group, no checkbox, cannot be switched "
        f"off. keys={keys}")
    assert {k.op for k in keys} == {op}, f"bricks claim the wrong operator: {[k.op for k in keys]}"
    # ONE group per channel, so ONE checkbox drives the whole volume -- the bricks ARE one volume.
    assert len({(k.op, k.channel) for k in keys}) == 2, (
        "bricks of a channel must share one key, or the tree grows a row per brick")


def test_the_bricks_claim_the_same_identity_a_2d_mosaic_layer_would():
    """The tree must not need to know a brick from a mosaic: same op, same channel, same group.

    If these two disagreed, turning off 'decon / c0' would hide the 2-D layer and leave the volume
    lit -- which is the reported bug wearing a different hat.
    """
    import numpy as np

    from squidmip._napari_view import MosaicKey, key_of

    viewer, vol = _bricked("decon", channels=("c0",))
    vol._add_layer(("c0", (0, 0)), "c0", np.zeros((2, 4, 4), np.uint16),
                   (1.5, 0.75, 0.75), (0.0, 0.0, 0.0))
    assert key_of(viewer.layers[0]) == MosaicKey("decon", "c0")


# -- while 3D is up, the VOLUME owns the identity; the flat mosaic surrenders it ----------------
#
# Julio, 2026-08-05, after the first fix shipped: "When I turn on raw it overlays some probably
# downsampled copy of raw over an already full res version of raw that can't be controlled by the
# napari layer. Channel controls for 3D viewing not working well."
#
# The "downsampled copy" is the 2D mosaic layer -- a multiscale pyramid whose level 0 is capped to
# _MAX_FUSED_PX, so it really is coarser than the bricks. `open()` hid it but left it in the TREE
# with a live checkbox, so one click laid a flat coarse plane across the volume.
#
# Stamping the bricks ALONE would have made raw worse: brick and mosaic would then share one
# (op, channel) key, so the single group checkbox would light both at once. The identity has to be
# EXCLUSIVE while 3D is up.


class _Layer2D:
    """A pane mosaic layer as the tree sees it: identity in metadata, visible, multiscale."""

    def __init__(self, op="raw", channel="c0", visible=True):
        from squidmip._napari_view import MosaicKey
        self.metadata = dict(MosaicKey(op, channel).as_metadata())
        self.visible = visible


class _SceneViewer(_RecordingViewer):
    def __init__(self, existing):
        super().__init__()
        self.layers = list(existing)

    class _Dims:
        ndisplay = 2
    dims = _Dims()


def _open_over(existing, op="raw"):
    """A BrickedVolume over *existing* pane layers, with the three things that need a live Qt
    event loop stubbed out: the loader THREAD, the camera framing and the first refresh.

    Starting a real QThread with no QApplication aborts the interpreter, and none of the three is
    what these tests are about -- the subject is which layer owns the `(op, channel)` identity
    while 3D is up. `open()`'s own layer bookkeeping runs for real.
    """
    from squidmip._brick_view import BrickedVolume

    viewer = _SceneViewer(existing)
    vol = BrickedVolume(
        viewer, reader=None, meta={}, region="A1", window_px=(0, 8, 0, 8),
        channels=["c0"], scale=(1.5, 0.75, 0.75), origin_um=(0.0, 0.0, 0.0),
        limit=2048, budget_bytes=1 << 30, op=op,
    )
    vol._loader.start = lambda *a, **k: None
    vol._loader.stop = lambda *a, **k: None
    vol._loader.wait = lambda *a, **k: True
    vol._frame_camera = lambda *a, **k: None
    vol.refresh = lambda *a, **k: None
    return viewer, vol


def test_the_flat_mosaic_leaves_the_tree_while_3d_is_up():
    """It must be FOREIGN, not merely hidden: a hidden layer still has a checkbox."""
    from squidmip._napari_view import key_of

    shown, already_hidden = _Layer2D(visible=True), _Layer2D(channel="c1", visible=False)
    viewer, vol = _open_over([shown, already_hidden])
    vol.open()

    assert key_of(shown) is None, "the 2D mosaic can still be switched on over the volume"
    # The already-hidden one too: it is just as clickable in the tree as a shown one.
    assert key_of(already_hidden) is None, "a hidden mosaic layer keeps its checkbox"
    assert shown.visible is False


def test_the_mosaic_gets_its_identity_and_visibility_BACK_on_close():
    from squidmip._napari_view import MosaicKey, key_of

    shown = _Layer2D(op="decon", channel="c1", visible=True)
    viewer, vol = _open_over([shown], op="decon")
    vol.open()
    assert key_of(shown) is None
    vol.close()

    assert key_of(shown) == MosaicKey("decon", "c1"), "2D never got its tree row back"
    assert shown.visible is True, "2D never came back on"


def test_the_volume_is_the_only_thing_holding_that_key_while_3d_is_up():
    """THE REPORTED BUG. If both hold it, one group checkbox lights the volume AND a flat coarser
    plane across it -- which is what 'overlays a downsampled copy over the full res version' is."""
    import numpy as np

    from squidmip._napari_view import MosaicKey, key_of

    shown = _Layer2D(op="raw", channel="c0")
    viewer, vol = _open_over([shown], op="raw")
    vol.open()
    vol._add_layer(("c0", (0, 0)), "c0", np.zeros((2, 4, 4), np.uint16),
                   (1.5, 0.75, 0.75), (0.0, 0.0, 0.0))

    holders = [ly for ly in viewer.layers if key_of(ly) == MosaicKey("raw", "c0")]
    assert len(holders) == 1, f"{len(holders)} layers answer to raw/c0; exactly the brick should"
    assert holders[0] is not shown, "the flat mosaic still owns the key the volume needs"
