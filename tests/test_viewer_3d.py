"""3D volume rendering: voxel size reaching napari, and the one layer model shared by 2D and 3D."""

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

from squidxplorer import _viewer as V  # noqa: E402

from .conftest import (  # noqa: E402
    _scene_stack,
    build_flat_scene,
    build_volume_scene,
    shutdown_plate_window,
)
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)


def _open_window(win, regions):
    w = win._viewer_manager.open(list(regions))
    assert w is not None, "no window was opened"
    return w


def _wait_for_layers(qapp, pane, timeout=30):
    assert _drain_until(qapp, lambda: bool(pane.mosaic.added), timeout=timeout), (
        "no mosaic ever reached the window's viewer")
    return pane.mosaic.added


class TestRawPushCarriesVoxelSize:
    """The raw z-stack push must carry the acquisition's real voxel size, as numbers."""

    def test_a_window_declares_the_acquisitions_voxel_depth_to_napari(
        self, qapp, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, win._order)
        added = _wait_for_layers(qapp, napari_pane_stub[-1])

        meta = win._meta
        for op, channel, _levels, kw in added:
            assert kw.get("z_scale_um") == meta["dz_um"], (
                f"{op}/{channel} was added with z_scale_um={kw.get('z_scale_um')!r}")
        assert meta["dz_um"] is not None and meta["dz_um"] > 0
        assert meta["pixel_size_um"] is not None and meta["pixel_size_um"] > 0
        shutdown_plate_window(qapp, win)

    def test_the_3d_volume_push_carries_the_full_voxel_scale(
        self, qapp, napari_pane_stub, squid_dataset, monkeypatch  # noqa: F811
    ):
        """The pushed y/x pitch is the rendered LAYER's own scale, not the acquisition's."""
        import squidxplorer._napari3d as napari3d

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
        raw = pane.mosaic.find("raw", win._meta["channels"][0]["name"])
        pitch = tuple(float(v) for v in raw.scale[-2:])
        assert (scale[1], scale[2]) == pitch, (
            f"pushed {scale[1:]} um/px, but the layer being rendered is placed at {pitch} um/px "
            f"(meta pixel_size_um is {win._meta['pixel_size_um']}, which is NOT this layer's pitch "
            f"whenever the mosaic was fused decimated)")
        aspect = scale[0] / scale[1]
        assert aspect > 0
        assert aspect == pytest.approx(win._meta["dz_um"] / pitch[0])
        shutdown_plate_window(qapp, win)

    def test_the_raw_mosaic_declares_the_full_z_stack(
        self, qapp, napari_pane_stub, squid_dataset  # noqa: F811
    ):
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
        import squidxplorer._napari3d as napari3d

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
    """A 3D click replaces the previous popout instead of stacking a new window on it."""

    def test_a_second_3D_click_CLOSES_the_first_popout(
        self, qapp, napari_pane_stub, squid_dataset, monkeypatch  # noqa: F811
    ):
        import squidxplorer._napari3d as napari3d

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
        import squidxplorer._napari3d as napari3d

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


# -- one layer model, 2D and 3D: each rule is parametrized over both scenes ---------------------


@pytest.fixture
def mosaic():
    """The app's layer model over a bare, Qt-free ``ViewerModel``."""
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    return MosaicLayers(ViewerModel())


#: The two scenes one rule is asked of, built by the shared builders in ``tests/conftest.py``.
SCENES = [
    pytest.param(build_flat_scene, id="2D-flat-mosaic"),
    pytest.param(build_volume_scene, id="3D-bricked-volume"),
]

OP, CHANNELS = "raw", ("488", "561")


@pytest.mark.parametrize("build", SCENES)
def test_the_model_shows_one_group_with_one_row_per_channel(build, mosaic):
    build(mosaic, OP, CHANNELS)
    assert mosaic.ops() == [OP]
    assert mosaic.channels(OP) == list(CHANNELS)


@pytest.mark.parametrize("build", SCENES)
def test_switching_a_channel_off_darkens_EVERY_layer_rendering_it(build, mosaic):
    build(mosaic, OP, CHANNELS)
    rendering = mosaic.layers_for(OP, CHANNELS[0])
    assert rendering, "the scene rendered nothing"
    assert rendering, "the scene rendered nothing"

    mosaic.find(OP, CHANNELS[0]).visible = False

    assert [ly.visible for ly in rendering] == [False] * len(rendering)
    assert all(ly.visible for ly in mosaic.layers_for(OP, CHANNELS[1]))


@pytest.mark.parametrize("build", SCENES)
def test_the_group_toggle_reaches_every_layer_of_every_channel(build, mosaic):
    build(mosaic, OP, CHANNELS)
    mosaic.show_op(OP)
    assert all(ly.visible for ly in mosaic.ours())

    for ly in mosaic.ours():
        ly.visible = False
    assert mosaic.visible_op() is None
    assert not any(ly.visible for ly in mosaic.ours())


@pytest.mark.parametrize("build", SCENES)
def test_ONE_contrast_window_per_channel_whatever_renders_it(build, mosaic):
    build(mosaic, OP, CHANNELS)
    rendering = mosaic.layers_for(OP, CHANNELS[0])
    assert rendering, "the scene rendered nothing"

    mosaic.find(OP, CHANNELS[0]).contrast_limits = (11.0, 2222.0)

    for ly in rendering:
        assert tuple(ly.contrast_limits) == (11.0, 2222.0), (
            f"{ly.name} is windowed at {tuple(ly.contrast_limits)}")
    assert mosaic.contrast(CHANNELS[0]) == (11.0, 2222.0)


@pytest.mark.parametrize("build", SCENES)
def test_ONE_colormap_per_channel_whatever_renders_it(build, mosaic):
    build(mosaic, OP, CHANNELS)
    rendering = mosaic.layers_for(OP, CHANNELS[0])
    assert rendering, "the scene rendered nothing"

    mosaic.find(OP, CHANNELS[0]).colormap = "magenta"

    names = {getattr(ly.colormap, "name", None) for ly in rendering}
    assert names == {"magenta"}, f"the identity is tinted {names}"


@pytest.mark.parametrize("build", SCENES)
def test_a_result_delivered_over_this_scene_darkens_it_and_can_be_switched_back(build, mosaic):
    build(mosaic, OP, CHANNELS)
    ch = CHANNELS[0]
    scene_layers = mosaic.layers_for(OP, ch)

    mosaic.add_mosaic("decon", ch, _scene_stack(99, (4, 16, 16)), bbox_um=(0.0, 0.0, 16.0, 16.0))

    assert [ly.visible for ly in scene_layers] == [False] * len(scene_layers), (
        "the arriving result is summing with the scene it was supposed to replace")
    assert mosaic.find("decon", ch).visible is True

    mosaic.find(OP, ch).visible = True
    assert all(ly.visible for ly in scene_layers)
    assert mosaic.find("decon", ch).visible is False


@pytest.mark.parametrize("build", SCENES)
def test_removing_an_identity_removes_every_layer_that_rendered_it(build, mosaic):
    build(mosaic, OP, CHANNELS)
    assert len(list(mosaic.model.layers)) >= len(CHANNELS)

    mosaic.remove_op(OP)

    assert mosaic.layers_for(OP, CHANNELS[0]) == []
    assert mosaic.ours() == [], f"{len(mosaic.ours())} layers of {OP} outlived their identity"


@pytest.mark.parametrize("build", SCENES)
def test_the_units_on_every_layer_are_micrometres(build, mosaic):
    build(mosaic, OP, CHANNELS)
    assert mosaic.ours()
    for ly in mosaic.ours():
        # napari normalises "um" to a pint unit, so compare against what it made of a labelled
        # axis rather than against the string we handed it.
        assert {str(u) for u in ly.units} == {"micrometer"}, (
            f"{ly.name} is labelled {ly.units}, so the scale bar reports pixels")


# -- facts true of volumes, declared in the shared model ----------------------------------------


def test_a_volume_is_MANY_layers_under_ONE_identity(mosaic):
    """The premise the parametrized rules above rest on: the 3D scene really is several layers."""
    from squidxplorer._napari_view import MosaicKey, key_of

    build_volume_scene(mosaic, OP, CHANNELS, bricks=3)

    rendering = mosaic.layers_for(OP, CHANNELS[0])
    assert rendering, "the scene rendered nothing"
    assert len(rendering) == 3, f"the volume rendered as {len(rendering)} layer(s)"
    assert {key_of(ly) for ly in rendering} == {MosaicKey(OP, CHANNELS[0])}
    assert mosaic.find(OP, CHANNELS[0]) is rendering[0]


@pytest.mark.parametrize("op", ["raw", "decon", "bgsub"])
def test_every_brick_declares_the_operator_whose_volume_it_is(op, mosaic):
    from squidxplorer._napari_view import key_of

    build_volume_scene(mosaic, op, ("c0", "c1"), bricks=3)

    keys = [key_of(ly) for ly in mosaic.model.layers]
    assert len(keys) == 6
    assert all(k is not None for k in keys), (
        "a brick with no metadata is a FOREIGN layer: no group, no checkbox, cannot be switched "
        f"off. keys={keys}")
    assert {k.op for k in keys} == {op}, f"bricks claim the wrong operator: {[k.op for k in keys]}"
    assert len({(k.op, k.channel) for k in keys}) == 2, (
        "bricks of a channel must share one key, or the tree grows a row per brick")


def test_the_bricks_claim_the_same_identity_a_2d_mosaic_layer_would():
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicKey, MosaicLayers, key_of

    flat = MosaicLayers(ViewerModel())
    build_flat_scene(flat, "decon", ("c0",))
    volume = MosaicLayers(ViewerModel())
    build_volume_scene(volume, "decon", ("c0",), bricks=1)

    assert key_of(flat.model.layers[0]) == MosaicKey("decon", "c0")
    assert key_of(volume.model.layers[0]) == key_of(flat.model.layers[0])


def test_a_brick_that_arrives_LATE_takes_the_identity_it_is_joining(mosaic):
    vol = build_volume_scene(mosaic, OP, ("c0",), bricks=1)

    mosaic.find(OP, "c0").contrast_limits = (7.0, 900.0)
    mosaic.find(OP, "c0").colormap = "magenta"
    mosaic.find(OP, "c0").visible = False

    vol._add_layer(("c0", (0, 9)), "c0", _scene_stack(77, (4, 8, 8)),
                   (1.5, 0.75, 0.75), (0.0, 0.0, 60.0))

    late = mosaic.layers_for(OP, "c0")[-1]
    assert late.visible is False, "a brick of a switched-off volume arrived lit"
    assert tuple(late.contrast_limits) == (7.0, 900.0), (
        f"a late brick arrived at {tuple(late.contrast_limits)}, so the joins step in brightness")
    assert getattr(late.colormap, "name", None) == "magenta"


def test_evicting_one_brick_leaves_the_identity_and_its_other_bricks_alone(mosaic):
    vol = build_volume_scene(mosaic, OP, ("c0",), bricks=3)
    doomed = mosaic.layers_for(OP, "c0")[1]

    vol._drop(("c0", (0, 1)))

    survivors = mosaic.layers_for(OP, "c0")
    assert len(survivors) == 2
    assert doomed not in survivors
    assert doomed not in (mosaic._by_channel.get("c0") or []), (
        "an evicted brick is still a peer of its channel")
    assert mosaic.channels(OP) == ["c0"], "the group lost its row when one brick left"

    mosaic.find(OP, "c0").contrast_limits = (3.0, 33.0)
    assert all(tuple(ly.contrast_limits) == (3.0, 33.0) for ly in survivors)


# -- while 3D is up, the VOLUME owns the identity; the flat mosaic surrenders it ----------------


def _open_over_flat(mosaic, op="raw", channels=("c0",)):
    """A `BrickedVolume` opened over a scene that already holds flat mosaics of the same op."""
    from squidxplorer._brick_view import BrickedVolume

    build_flat_scene(mosaic, op, channels)
    vol = BrickedVolume(
        mosaic, reader=None, meta={}, region="A1", window_px=(0, 8, 0, 8),
        channels=list(channels), scale=(1.5, 0.75, 0.75), origin_um=(0.0, 0.0, 0.0),
        limit=2048, budget_bytes=1 << 30, op=op,
    )
    vol._loader.start = lambda *a, **k: None
    vol._loader.stop = lambda *a, **k: None
    vol._loader.wait = lambda *a, **k: True
    vol._frame_camera = lambda *a, **k: None
    vol.refresh = lambda *a, **k: None
    return vol


def test_the_flat_mosaic_leaves_the_tree_while_3d_is_up(mosaic):
    """It must be FOREIGN, not merely hidden: a hidden layer still has a checkbox."""
    from squidxplorer._napari_view import key_of

    vol = _open_over_flat(mosaic, "raw", ("c0", "c1"))
    shown, already_hidden = mosaic.find("raw", "c0"), mosaic.find("raw", "c1")
    already_hidden.visible = False
    vol.open()

    assert key_of(shown) is None, "the 2D mosaic can still be switched on over the volume"
    assert key_of(already_hidden) is None, "a hidden mosaic layer keeps its checkbox"
    assert shown.visible is False


def test_the_mosaic_gets_its_identity_and_visibility_BACK_on_close(mosaic):
    from squidxplorer._napari_view import MosaicKey, key_of

    vol = _open_over_flat(mosaic, "decon", ("c1",))
    shown = mosaic.find("decon", "c1")
    vol.open()
    assert key_of(shown) is None
    vol.close()

    assert key_of(shown) == MosaicKey("decon", "c1"), "2D never got its tree row back"
    assert shown.visible is True, "2D never came back on"


def test_the_volume_is_the_only_thing_holding_that_key_while_3d_is_up(mosaic):
    from squidxplorer._napari_view import MosaicKey, key_of

    vol = _open_over_flat(mosaic, "raw", ("c0",))
    shown = mosaic.find("raw", "c0")
    vol.open()
    vol._add_layer(("c0", (0, 0)), "c0", _scene_stack(5, (4, 8, 8)),
                   (1.5, 0.75, 0.75), (0.0, 0.0, 0.0))

    holders = [ly for ly in mosaic.model.layers if key_of(ly) == MosaicKey("raw", "c0")]
    assert len(holders) == 1, f"{len(holders)} layers answer to raw/c0; exactly the brick should"
    assert holders[0] is not shown, "the flat mosaic still owns the key the volume needs"
