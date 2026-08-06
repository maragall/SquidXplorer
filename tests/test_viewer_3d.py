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

from .conftest import (  # noqa: E402
    _scene_stack,
    build_flat_scene,
    build_volume_scene,
    shutdown_plate_window,
)
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
        """dz_um over the DISPLAYED pitch is the z stretch the renderer applies, and it must be
        computable from the pushed values alone, and finite and positive.

        The 3D push declares all three axes at once as `scale=(dz, py, px)`, so the aspect is
        recoverable as scale[0] / scale[1].

        The y/x half was asserted as ``== meta["pixel_size_um"]`` until 2026-08-06, and that is
        the acquisition's pitch, not these pixels'. This renders the 2D LAYER's pixels, which come
        from ``fuse_region_pyramid`` and are decimated (step 2 on the 10x set: 1.504 um/px against
        0.752) -- so the assertion pinned a volume half as wide and half as deep as its own z,
        i.e. the wrong aspect ratio, which is the exact thing this test exists to catch. The layer
        knows the true pitch in its own ``scale`` (``placement_for`` = bbox / shape); that is the
        number, and on this fixture (step 1) it differs from ``pixel_size_um`` only in the 13th
        digit, which is why exact equality was green and honest equality is the assertion now.

        MUTATION: pass `scale=(1.0, py, px)` (or omit it) -> aspect 1 -> red. Pass
        `meta["pixel_size_um"]` for y/x -> red on the layer's own scale.
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


# ==============================================================================================
# ONE LAYER MODEL, 2D AND 3D
# ==============================================================================================
#
# Julio: "why is the layering of 2d and 3d different in the same place?"
#
# It was different because 2D assumed an identity IS a layer -- one `add_image` per
# (op, channel) -- and a volume is not: `_brick_view` tiles it into one Image layer per brick.
# Every rule the app enforces was written against the 2D assumption, so in 3D each one was either
# absent or re-implemented partially inside `BrickedVolume`. Four user-visible defects came out of
# that in one evening: a volume with no checkbox at all, a coarse 2D mosaic drawable over it, one
# checkbox reaching one brick, and a second contrast model.
#
# The model now declares the fact instead of branching on it: AN IDENTITY MAY BE RENDERED BY MORE
# THAN ONE LAYER, AND EVERY PROPERTY THAT BELONGS TO THE IDENTITY HOLDS ONE VALUE ACROSS ALL OF
# THEM (`MosaicLayers.IDENTITY_PROPS` / `layers_for` / `adopt` / `_mirror_identity`).
#
# So the tests below are PARAMETRIZED over the two scenes. Each one states a rule once and asks it
# of a flat mosaic and of a bricked volume, against a real `napari.components.ViewerModel` with
# real layers and real events -- so a future divergence cannot pass by being written on only one
# side. Where a scene is built out of several bricks, it is built out of SEVERAL: a rule that only
# holds for a single-brick volume is the bug wearing a disguise.

@pytest.fixture
def mosaic():
    """The app's layer model over a bare, Qt-free ``ViewerModel``.

    A `RegionViewer` cannot be built headless (the napari pane needs a GL context), but everything
    under test here is the layer model, and `ViewerModel` gives it real napari layers with real
    evented properties -- which is what makes the mirror, the contrast link and the identity
    bookkeeping assertable for real rather than through a stub that agrees by construction.
    """
    from napari.components import ViewerModel

    from squidmip._napari_view import MosaicLayers

    return MosaicLayers(ViewerModel())


#: The two scenes one rule is asked of, built by ONE definition shared with
#: ``tests/test_layer_tree.py`` (``tests/conftest.py``). Every test below reaches the layers
#: through `MosaicLayers`, never through the builder's return value, because "what the model can
#: see" is the property under test.
SCENES = [
    pytest.param(build_flat_scene, id="2D-flat-mosaic"),
    pytest.param(build_volume_scene, id="3D-bricked-volume"),
]

OP, CHANNELS = "raw", ("488", "561")


@pytest.mark.parametrize("build", SCENES)
def test_the_model_shows_one_group_with_one_row_per_channel(build, mosaic):
    """THE TREE'S SHAPE. `MosaicTreeModel.refresh` is built out of exactly these two calls, so a
    volume that answered per brick would grow one row per brick instead of one per channel.

    MUTATION: drop the `MosaicLayers.adopt` call in `BrickedVolume._add_layer` -> the bricks are
    foreign layers, `ops()` is empty and the tree shows nothing -> red.
    """
    build(mosaic, OP, CHANNELS)
    assert mosaic.ops() == [OP]
    assert mosaic.channels(OP) == list(CHANNELS)


@pytest.mark.parametrize("build", SCENES)
def test_switching_a_channel_off_darkens_EVERY_layer_rendering_it(build, mosaic):
    """Julio: "Turning off one layer doesn't turn the other like in the 2D view."

    The layer tree writes `mosaic.find(op, channel).visible`, and `find` is ONE layer. In 2D that
    is the whole identity; in 3D it was one brick out of many, so the checkbox switched off part
    of the volume and the rest stayed lit with no control left to reach it.

    MUTATION: delete the `_connect_identity_mirror` call in `_register_channel` -> the other
    bricks stay visible -> red.
    """
    build(mosaic, OP, CHANNELS)
    rendering = mosaic.layers_for(OP, CHANNELS[0])
    # `layers_for` returns [] when nothing matches, and a loop over [] asserts nothing:
    # a scene that rendered nothing would pass every claim below.
    assert rendering, "the scene rendered nothing"
    assert rendering, "the scene rendered nothing"

    mosaic.find(OP, CHANNELS[0]).visible = False

    assert [ly.visible for ly in rendering] == [False] * len(rendering)
    # ...and the OTHER channel is untouched: this is a channel toggle, not a group toggle.
    assert all(ly.visible for ly in mosaic.layers_for(OP, CHANNELS[1]))


@pytest.mark.parametrize("build", SCENES)
def test_the_group_toggle_reaches_every_layer_of_every_channel(build, mosaic):
    """What the processing-layer row does: `show_op` is the group toggle's rule.

    MUTATION, and it takes TWO: have `show_op` write `find(op, ch).visible` instead of walking
    `ours()` AND drop "visible" from `IDENTITY_PROPS`. Either alone stays green, because
    `show_op` writes every MEMBER of the group and the mirror keeps every SURFACE of a member
    equal -- two different rules that overlap on this gesture. The test asserts the outcome.
    """
    build(mosaic, OP, CHANNELS)
    mosaic.show_op(OP)
    assert all(ly.visible for ly in mosaic.ours())

    for ly in mosaic.ours():
        ly.visible = False
    assert mosaic.visible_op() is None
    assert not any(ly.visible for ly in mosaic.ours())


@pytest.mark.parametrize("build", SCENES)
def test_ONE_contrast_window_per_channel_whatever_renders_it(build, mosaic):
    """"there is exactly ONE contrast value per channel in the whole application" -- the module
    docstring of `_napari_view`. It was true of flat mosaics and false of volumes, where
    `BrickedVolume` kept its own `_contrast_by` dict and its own propagator.

    Asserted as the WINDOW APPLIED to each layer, not as a call: a slider that moves one brick
    makes the joins step in brightness, which is what the user sees.

    MUTATION: drop "contrast_limits" from `MosaicLayers.IDENTITY_PROPS` -> only the layer that was
    written moves -> red.
    """
    build(mosaic, OP, CHANNELS)
    rendering = mosaic.layers_for(OP, CHANNELS[0])
    # `layers_for` returns [] when nothing matches, and a loop over [] asserts nothing:
    # a scene that rendered nothing would pass every claim below.
    assert rendering, "the scene rendered nothing"

    mosaic.find(OP, CHANNELS[0]).contrast_limits = (11.0, 2222.0)

    for ly in rendering:
        assert tuple(ly.contrast_limits) == (11.0, 2222.0), (
            f"{ly.name} is windowed at {tuple(ly.contrast_limits)}")
    assert mosaic.contrast(CHANNELS[0]) == (11.0, 2222.0)


@pytest.mark.parametrize("build", SCENES)
def test_ONE_colormap_per_channel_whatever_renders_it(build, mosaic):
    """The same rule for the LUT. A volume tinted per brick is a volume with visible seams.

    MUTATION: drop "colormap" from `IDENTITY_PROPS` -> the other bricks keep the old tint -> red.
    """
    build(mosaic, OP, CHANNELS)
    rendering = mosaic.layers_for(OP, CHANNELS[0])
    # `layers_for` returns [] when nothing matches, and a loop over [] asserts nothing:
    # a scene that rendered nothing would pass every claim below.
    assert rendering, "the scene rendered nothing"

    mosaic.find(OP, CHANNELS[0]).colormap = "magenta"

    names = {getattr(ly.colormap, "name", None) for ly in rendering}
    assert names == {"magenta"}, f"the identity is tinted {names}"


@pytest.mark.parametrize("build", SCENES)
def test_a_result_delivered_over_this_scene_darkens_it_and_can_be_switched_back(build, mosaic):
    """ONE OPERATOR PER CHANNEL ON SCREEN, in both modes.

    `_connect_exclusive_op` exists because every mosaic is added `additive`, so the same channel on
    screen twice is that channel's signal summed with itself. A volume that was outside the rule
    kept adding to whatever the user switched on over it -- Julio: "there is still a layer that
    looks beautiful but that I can't control so then other controlled layers are overlayed".

    MUTATION: skip `_register_channel` in `adopt` -> the volume is not in `_by_channel`, the
    arriving result does not darken it -> red.
    """
    build(mosaic, OP, CHANNELS)
    ch = CHANNELS[0]
    scene_layers = mosaic.layers_for(OP, ch)

    mosaic.add_mosaic("decon", ch, _scene_stack(99, (4, 16, 16)), bbox_um=(0.0, 0.0, 16.0, 16.0))

    assert [ly.visible for ly in scene_layers] == [False] * len(scene_layers), (
        "the arriving result is summing with the scene it was supposed to replace")
    assert mosaic.find("decon", ch).visible is True

    # ...and switching the scene back on darkens the result, from either direction.
    mosaic.find(OP, ch).visible = True
    assert all(ly.visible for ly in scene_layers)
    assert mosaic.find("decon", ch).visible is False


@pytest.mark.parametrize("build", SCENES)
def test_removing_an_identity_removes_every_layer_that_rendered_it(build, mosaic):
    """`remove_op` took `find`, i.e. one layer. A volume left the rest of its bricks on screen as
    FOREIGN layers -- lit, unlabelled and with no tree row to switch them off, which is exactly
    the defect the bricks were given an identity to end.

    MUTATION: have `remove_op_channel` drop `find(op, channel)` alone -> layers survive -> red.
    """
    build(mosaic, OP, CHANNELS)
    assert len(list(mosaic.model.layers)) >= len(CHANNELS)

    mosaic.remove_op(OP)

    assert mosaic.layers_for(OP, CHANNELS[0]) == []
    assert mosaic.ours() == [], f"{len(mosaic.ours())} layers of {OP} outlived their identity"


@pytest.mark.parametrize("build", SCENES)
def test_the_units_on_every_layer_are_micrometres(build, mosaic):
    """The scale bar reads the LAYER's units, so a layer with none silently reports pixels.
    `_place` labels a flat mosaic; a brick was placed in micrometres and never labelled.

    MUTATION: delete the `_label_units` call in `adopt` -> the bricks report napari's default
    -> red.
    """
    build(mosaic, OP, CHANNELS)
    assert mosaic.ours()
    for ly in mosaic.ours():
        # napari normalises "um" to a pint unit, so compare against what it made of a labelled
        # axis rather than against the string we handed it.
        assert {str(u) for u in ly.units} == {"micrometer"}, (
            f"{ly.name} is labelled {ly.units}, so the scale bar reports pixels")


# -- the facts that are TRUE OF VOLUMES, declared in the shared model rather than branched on ----


def test_a_volume_is_MANY_layers_under_ONE_identity(mosaic):
    """The premise the parametrized rules above rest on: the 3D scene really is several layers.

    Without this the whole file could pass against a one-brick volume and prove nothing about the
    case that actually broke.
    """
    from squidmip._napari_view import MosaicKey, key_of

    build_volume_scene(mosaic, OP, CHANNELS, bricks=3)

    rendering = mosaic.layers_for(OP, CHANNELS[0])
    # `layers_for` returns [] when nothing matches, and a loop over [] asserts nothing:
    # a scene that rendered nothing would pass every claim below.
    assert rendering, "the scene rendered nothing"
    assert len(rendering) == 3, f"the volume rendered as {len(rendering)} layer(s)"
    assert {key_of(ly) for ly in rendering} == {MosaicKey(OP, CHANNELS[0])}
    # ...and `find` is one of them, the representative every control reads and writes.
    assert mosaic.find(OP, CHANNELS[0]) is rendering[0]


@pytest.mark.parametrize("op", ["raw", "decon", "bgsub"])
def test_every_brick_declares_the_operator_whose_volume_it_is(op, mosaic):
    """Without this the tree cannot see the volume at all, whatever the operator.

    The operator is never compared by name anywhere: it is carried from
    `RegionViewer._volume_source`, which picks it off the registry declaration.
    """
    from squidmip._napari_view import key_of

    build_volume_scene(mosaic, op, ("c0", "c1"), bricks=3)

    keys = [key_of(ly) for ly in mosaic.model.layers]
    assert len(keys) == 6
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
    from napari.components import ViewerModel

    from squidmip._napari_view import MosaicKey, MosaicLayers, key_of

    flat = MosaicLayers(ViewerModel())
    build_flat_scene(flat, "decon", ("c0",))
    volume = MosaicLayers(ViewerModel())
    build_volume_scene(volume, "decon", ("c0",), bricks=1)

    assert key_of(flat.model.layers[0]) == MosaicKey("decon", "c0")
    assert key_of(volume.model.layers[0]) == key_of(flat.model.layers[0])


def test_a_brick_that_arrives_LATE_takes_the_identity_it_is_joining(mosaic):
    """A volume is delivered brick by brick as the camera moves, so "join an identity" happens
    over and over while the user is looking at it. A brick that arrived with its own defaults
    repainted part of the volume back to a state the user had left -- lit after they switched it
    off, and at a different window than its neighbours.

    MUTATION: delete the sibling-copy block in `MosaicLayers.adopt` -> the late brick arrives
    visible and unwindowed -> red.
    """
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
    """The camera evicts bricks constantly. Dropping one must not take the group's row with it,
    and must not leave napari holding a contrast link onto a dead layer.

    MUTATION: have `_drop` call `viewer.layers.remove` again -> the model still lists the dead
    layer in `_by_channel` and re-links it -> red.
    """
    vol = build_volume_scene(mosaic, OP, ("c0",), bricks=3)
    doomed = mosaic.layers_for(OP, "c0")[1]

    vol._drop(("c0", (0, 1)))

    survivors = mosaic.layers_for(OP, "c0")
    assert len(survivors) == 2
    assert doomed not in survivors
    assert doomed not in (mosaic._by_channel.get("c0") or []), (
        "an evicted brick is still a peer of its channel")
    assert mosaic.channels(OP) == ["c0"], "the group lost its row when one brick left"

    # ...and the surviving bricks still move together.
    mosaic.find(OP, "c0").contrast_limits = (3.0, 33.0)
    assert all(tuple(ly.contrast_limits) == (3.0, 33.0) for ly in survivors)


# -- while 3D is up, the VOLUME owns the identity; the flat mosaic surrenders it ----------------
#
# Julio, 2026-08-05: "When I turn on raw it overlays some probably downsampled copy of raw over an
# already full res version of raw that can't be controlled by the napari layer."
#
# The "downsampled copy" is the 2D mosaic layer -- a multiscale pyramid whose level 0 is capped to
# _MAX_FUSED_PX, so it really is coarser than the bricks. `open()` hid it but left it in the TREE
# with a live checkbox, so one click laid a flat coarse plane across the volume.
#
# Stamping the bricks ALONE would have made raw worse: brick and mosaic would then share one
# (op, channel) key, so the single group checkbox would light both at once. The identity has to be
# EXCLUSIVE while 3D is up.


def _open_over_flat(mosaic, op="raw", channels=("c0",)):
    """A `BrickedVolume` opened over a scene that already holds flat mosaics of the same op."""
    from squidmip._brick_view import BrickedVolume

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
    from squidmip._napari_view import key_of

    vol = _open_over_flat(mosaic, "raw", ("c0", "c1"))
    shown, already_hidden = mosaic.find("raw", "c0"), mosaic.find("raw", "c1")
    already_hidden.visible = False
    vol.open()

    assert key_of(shown) is None, "the 2D mosaic can still be switched on over the volume"
    # The already-hidden one too: it is just as clickable in the tree as a shown one.
    assert key_of(already_hidden) is None, "a hidden mosaic layer keeps its checkbox"
    assert shown.visible is False


def test_the_mosaic_gets_its_identity_and_visibility_BACK_on_close(mosaic):
    from squidmip._napari_view import MosaicKey, key_of

    vol = _open_over_flat(mosaic, "decon", ("c1",))
    shown = mosaic.find("decon", "c1")
    vol.open()
    assert key_of(shown) is None
    vol.close()

    assert key_of(shown) == MosaicKey("decon", "c1"), "2D never got its tree row back"
    assert shown.visible is True, "2D never came back on"


def test_the_volume_is_the_only_thing_holding_that_key_while_3d_is_up(mosaic):
    """THE REPORTED BUG. If both hold it, one group checkbox lights the volume AND a flat coarser
    plane across it -- which is what 'overlays a downsampled copy over the full res version' is."""
    from squidmip._napari_view import MosaicKey, key_of

    vol = _open_over_flat(mosaic, "raw", ("c0",))
    shown = mosaic.find("raw", "c0")
    vol.open()
    vol._add_layer(("c0", (0, 0)), "c0", _scene_stack(5, (4, 8, 8)),
                   (1.5, 0.75, 0.75), (0.0, 0.0, 0.0))

    holders = [ly for ly in mosaic.model.layers if key_of(ly) == MosaicKey("raw", "c0")]
    assert len(holders) == 1, f"{len(holders)} layers answer to raw/c0; exactly the brick should"
    assert holders[0] is not shown, "the flat mosaic still owns the key the volume needs"
