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
    """``(op, channel, levels, layer)`` per identity, read back off the REAL model — the stub's recording list is gone; what the model holds is the"""
    from squidxplorer._napari_view import pyramid_levels

    assert _drain_until(qapp, lambda: bool(len(pane._viewer.layers)), timeout=timeout), (
        "no mosaic ever reached the window's viewer")
    mosaic = pane.mosaic
    added = []
    for op in mosaic.ops():
        for ch in mosaic.channels(op):
            layer = mosaic.find(op, ch)
            if layer is None:
                continue
            levels = pyramid_levels(layer.data)
            added.append((op, ch, levels if levels is not None else layer.data, layer))
    return added


class TestRawPushCarriesVoxelSize:
    """The raw z-stack push must carry the acquisition's real voxel size, as numbers."""

    def test_the_raw_mosaic_declares_the_full_z_stack_at_the_acquisitions_pitch(
        self, qapp, napari_pane_stub, squid_dataset  # noqa: F811
    ):
        root, _ = squid_dataset
        win = V.PlateWindow(None)
        win.ingest(str(root))
        w = _open_window(win, ["B3"])
        added = _wait_for_layers(qapp, napari_pane_stub[-1])

        meta = win._meta
        n_z = meta["n_z"]
        assert n_z > 1, "fixture needs a real z-stack or this asserts nothing"
        assert meta["dz_um"] is not None and meta["dz_um"] > 0
        for op, channel, levels, layer in added:
            level0 = levels[0] if isinstance(levels, (list, tuple)) else levels
            assert level0.ndim == 3, f"{op}/{channel} is not a (z, y, x) volume: {level0.shape}"
            assert level0.shape[0] == n_z, (
                f"{op}/{channel} declared {level0.shape[0]} planes, not {n_z}")
            assert float(layer.scale[0]) == meta["dz_um"], (
                f"{op}/{channel} was placed with z pitch {layer.scale[0]!r}")
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

        levels_by_channel = {ch: lv for _op, ch, lv, _layer in added}
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
        assert {str(u) for u in ly.units} == {"micrometer"}, (
            f"{ly.name} is labelled {ly.units}, so the scale bar reports pixels")


# -- facts true of volumes, declared in the shared model ----------------------------------------


def test_a_volume_is_MANY_bricks_under_the_ONE_key_a_2d_layer_would_carry(mosaic):
    """The premise the parametrized rules above rest on: the 3D scene really is several layers."""
    from squidxplorer._napari_view import MosaicKey, key_of

    build_volume_scene(mosaic, "decon", ("c0", "c1"), bricks=3)

    rendering = mosaic.layers_for("decon", "c0")
    assert len(rendering) == 3, f"the volume rendered as {len(rendering)} layer(s)"
    assert mosaic.find("decon", "c0") is rendering[0]
    keys = [key_of(ly) for ly in mosaic.model.layers]
    assert len(keys) == 6
    assert all(k is not None for k in keys), (
        "a brick with no metadata is a FOREIGN layer: no group, no checkbox, cannot be switched "
        f"off. keys={keys}")
    assert {k.op for k in keys} == {"decon"}, f"bricks claim the wrong operator: {keys}"
    assert {(k.op, k.channel) for k in keys} == {("decon", "c0"), ("decon", "c1")}, (
        "bricks of a channel must share one key, or the tree grows a row per brick")
    flat = build_flat_scene(_fresh_mosaic(), "decon", ("c0",))
    assert key_of(flat.model.layers[0]) == MosaicKey("decon", "c0") == key_of(rendering[0])


def _fresh_mosaic():
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    return MosaicLayers(ViewerModel())


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


def test_a_zoom_refined_brick_replaces_the_coarser_one_under_the_same_identity(mosaic):
    """Camera-settle refinement swaps a brick's PIXELS, never its identity: the layer object, the user's contrast window and the stride book survive the"""
    from squidxplorer import _bricks

    vol = build_volume_scene(mosaic, OP, ("c0",), bricks=0)
    b = _bricks.Brick(iy=0, ix=0, r0=0, r1=8, c0=0, c1=8)
    vol._epoch = 1
    vol._on_brick(b, "c0", _scene_stack(5, (4, 4, 4)), 2, 1)      # stride 2 of the 8x8 window
    layer = mosaic.find(OP, "c0")
    assert layer is not None
    layer.contrast_limits = (9.0, 750.0)

    vol._on_brick(b, "c0", _scene_stack(6, (4, 8, 8)), 1, 1)      # the refine at native stride

    assert mosaic.find(OP, "c0") is layer, "refinement rebuilt the layer instead of updating it"
    assert tuple(layer.data.shape) == (4, 8, 8)
    assert vol._steps[("c0", b.key)] == 1
    assert tuple(layer.contrast_limits) == (9.0, 750.0), "the user's window was lost on refine"
    assert tuple(layer.scale) == (1.5, 0.75, 0.75), "a stride-1 brick must carry the native pitch"


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


# -- a channel whose first brick is blank must still display (UI feedback 08.17 ticket #6) ------


def test_a_channel_whose_first_brick_is_blank_still_displays(mosaic):
    """A blank first brick seeded a DEGENERATE window (lo == hi), napari refuses that with ValueError, and the cached seed poisoned every later brick — so"""
    import numpy as np

    vol = build_volume_scene(mosaic, OP, ("488",), bricks=0)
    blank = np.zeros((4, 8, 8), dtype=np.uint16)
    signal = np.full((4, 8, 8), 3000, dtype=np.uint16)
    signal[:, 2:6, 2:6] = 12000

    vol._add_layer(("488", (0, 0)), "488", blank, (1.5, 0.75, 0.75), (0.0, 0.0, 0.0))
    vol._add_layer(("488", (0, 1)), "488", signal, (1.5, 0.75, 0.75), (0.0, 0.0, 6.0))

    rendering = mosaic.layers_for(OP, "488")
    assert len(rendering) == 2, (
        f"{len(rendering)} of 2 bricks made it to the canvas — a blank corner brick has taken "
        f"the whole channel down")
    lo, hi = (float(v) for v in mosaic.find(OP, "488").contrast_limits)
    assert hi > lo, "the channel is windowed on a degenerate (lo == hi) window"
    assert hi > 1.0, (
        f"the channel's window is ({lo}, {hi}) — the blank brick's autoscale, not the signal's "
        f"own window, so every voxel renders saturated")
    cached = vol._contrast_by.get("488")
    assert cached is None or float(cached[1]) > float(cached[0]), (
        f"a degenerate seed {cached} is cached: every future brick of this channel will be "
        f"refused by napari")


def test_a_depth_keeping_operators_volume_survives_its_own_toggle(mosaic):
    """Julio, live (2026-08-24): "I can't toggle on and off the 3D decon-ed layer in napari." decon declares ``consumes={"z"}`` (the engine hands it the"""
    build_volume_scene(mosaic, "decon", ("488",), bricks=3)

    for ly in mosaic.layers_for("decon", "488"):     # the tree checkbox: OFF ...
        ly.visible = False
    for ly in mosaic.layers_for("decon", "488"):     # ... and back ON
        ly.visible = True

    for ly in mosaic.layers_for("decon", "488"):
        assert ly.visible is True
        assert tuple(ly.data.shape) == (4, 8, 8), (
            f"the toggle flattened a decon brick to {tuple(ly.data.shape)} — the volume was "
            "treated as a z-reducer's single plane while decon keeps every plane")
    assert mosaic._reduces_z("decon") is False, (
        "the display side still reads decon's consumed axis as its output depth; volume_source "
        "refuses 3D on it and _present_z_axis flattens it")


def test_auto_clim_never_hands_napari_a_degenerate_window():
    """napari raises ValueError on contrast_limits with lo == hi; a blank stack must yield None (napari autoscales), never (0.0, 0.0)."""
    import numpy as np

    from squidxplorer._napari3d import _auto_clim

    assert _auto_clim(np.zeros((4, 8, 8), dtype=np.uint16)) is None
    assert _auto_clim(np.full((4, 8, 8), 7, dtype=np.uint16)) is None
    real = _auto_clim(np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8))
    assert real is not None and real[1] > real[0]


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


def test_the_flat_mosaic_surrenders_its_key_while_3d_is_up_and_gets_it_back_on_close(mosaic):
    """It must be FOREIGN, not merely hidden: a hidden layer still has a checkbox."""
    from squidxplorer._napari_view import MosaicKey, key_of

    vol = _open_over_flat(mosaic, "raw", ("c0", "c1"))
    shown, already_hidden = mosaic.find("raw", "c0"), mosaic.find("raw", "c1")
    already_hidden.visible = False
    vol.open()

    assert key_of(shown) is None, "the 2D mosaic can still be switched on over the volume"
    assert key_of(already_hidden) is None, "a hidden mosaic layer keeps its checkbox"
    assert shown.visible is False
    vol._add_layer(("c0", (0, 0)), "c0", _scene_stack(5, (4, 8, 8)),
                   (1.5, 0.75, 0.75), (0.0, 0.0, 0.0))
    holders = [ly for ly in mosaic.model.layers if key_of(ly) == MosaicKey("raw", "c0")]
    assert len(holders) == 1 and holders[0] is not shown, (
        f"{len(holders)} layers answer to raw/c0; exactly the brick should")

    vol.close()
    assert key_of(shown) == MosaicKey("raw", "c0"), "2D never got its tree row back"
    assert shown.visible is True, "2D never came back on"


def _opened_volume(mosaic, channels=("c0",)):
    """A `BrickedVolume` OPENED over an empty scene — the 3D child-tab flow, where the 2D mosaic has not landed yet."""
    from squidxplorer._brick_view import BrickedVolume

    vol = BrickedVolume(
        mosaic, reader=None, meta={}, region="A1", window_px=(0, 8, 0, 8),
        channels=list(channels), scale=(1.5, 0.75, 0.75), origin_um=(0.0, 0.0, 0.0),
        limit=2048, budget_bytes=1 << 30, op="raw",
    )
    vol._loader.start = lambda *a, **k: None
    vol._loader.stop = lambda *a, **k: None
    vol._loader.wait = lambda *a, **k: True
    vol._frame_camera = lambda *a, **k: None
    vol.open()
    return vol


def test_a_mosaic_landing_mid_volume_neither_shows_nor_strands_the_bricks(mosaic):
    """THE 2026-08-24 sequel to the brick-add failure: the 3D child tab's own 2D mosaic lands AFTER open (the fuse finishes late)."""
    import numpy as np

    from squidxplorer._napari_view import MosaicKey, key_of

    vol = _opened_volume(mosaic, ("c0",))
    brick = vol._bricks[0]
    vol._on_brick(vol._offset_brick(brick), "c0", _scene_stack(1, (4, 8, 8)),
                  step=1, epoch=vol._epoch)
    brick_layer = vol._layers[("c0", brick.key)]
    assert any(l is brick_layer for l in mosaic.model.layers)

    arriving = mosaic.add_mosaic(
        "raw", "c0", [np.zeros((4, 16, 16), np.uint16), np.zeros((4, 8, 8), np.uint16)],
        multiscale=True, bbox_um=(0.0, 0.0, 12.0, 12.0))

    assert not any(l is brick_layer for l in mosaic.model.layers), (
        "add_mosaic no longer removes a mismatched identity; this test's premise is stale")
    assert ("c0", brick.key) not in vol._layers, (
        "the volume still claims a brick the pane removed — no refresh will re-read it")
    requests = []
    vol._loader.request = lambda jobs, epoch: requests.append(list(jobs))
    vol._on_idle(vol._epoch)
    assert any(b.key == vol._offset_brick(brick).key and ch == "c0"
               for req in requests for (b, ch, _s) in req), (
        "the heal never re-queued the wiped brick")
    assert key_of(arriving) is None, "the arriving 2D mosaic kept the bricks' identity"
    assert arriving.visible is False, "the arriving 2D mosaic paints over the volume"
    vol._on_brick(vol._offset_brick(brick), "c0", _scene_stack(2, (4, 8, 8)),
                  step=1, epoch=vol._epoch)
    healed = vol._layers[("c0", brick.key)]
    assert any(l is healed for l in mosaic.model.layers)
    assert key_of(healed) == MosaicKey("raw", "c0")

    vol.close()
    assert key_of(arriving) == MosaicKey("raw", "c0"), "2D never got its identity back"
    assert arriving.visible is True, "2D never came back on"


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


# ══════════════════════════════════════════════════════════════════════════════════════════
# Framing the ROI: one camera rule, and it is napari's.
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_the_bricked_volume_frames_its_roi_through_the_one_camera_rule():
    """A tall ROI on a wide canvas must be fitted by its HEIGHT, not by its width."""
    from squidxplorer._brick_view import BrickedVolume
    from squidxplorer._napari_view import camera_for_bbox_um

    class _Camera:
        center = (0.0, 0.0, 0.0)
        zoom = 1.0

    class _Canvas:
        size = (720, 860)          # napari's own order: (height, width)

    class _Viewer:
        camera = _Camera()

        class window:
            class _qt_viewer:
                canvas = _Canvas()

    vol = object.__new__(BrickedVolume)
    vol._viewer = _Viewer()
    vol._window = (0, 1000, 0, 500)         # 1000 rows x 500 cols
    vol._scale = (1.0, 1.0, 1.0)            # 1 um per row / col -> a 1000 x 500 um ROI
    vol._origin_um = (0.0, 4000.0, 9000.0)
    vol._nz = 10
    vol._say = lambda _text: None

    vol._frame_camera()

    _centre, expected = camera_for_bbox_um((9000.0, 4000.0, 9500.0, 5000.0), (720, 860),
                                           margin=0.10)
    assert vol._viewer.camera.zoom == pytest.approx(expected)
    assert vol._viewer.camera.zoom == pytest.approx(0.9 * min(720 / 1000.0, 860 / 500.0))

    crossed = 0.9 * min(720 / 500.0, 860 / 1000.0)
    assert vol._viewer.camera.zoom != pytest.approx(crossed)

    assert vol._viewer.camera.center == pytest.approx((5.0, 4500.0, 9250.0))
