"""napari mosaic view — the processing-layer/channel hierarchy and the binding guards.

These tests use ``napari.components.ViewerModel``, which is Qt-free, so the hierarchy is
exercised headless with no canvas, no display and no Qt binding conflict. Only the embedding
test needs Qt, and it skips itself when Qt is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip._napari_view import (
    META_KEY,
    MosaicKey,
    MosaicLayers,
    NapariBindingError,
    REQUIRED_NAPARI_BINDINGS,
    key_of,
    napari_enabled,
    resolve_viewer,
    scale_translate_from_bbox_um,
    verify_napari_bindings,
)

napari = pytest.importorskip("napari")


@pytest.fixture
def layers():
    from napari.components import ViewerModel

    return MosaicLayers(ViewerModel())


def _img(seed=0, shape=(32, 32)):
    return np.random.default_rng(seed).integers(0, 4000, shape, dtype=np.uint16)


# ---------------------------------------------------------------- the flag


def test_napari_is_the_default_viewer_now_that_the_gate_passed():
    assert resolve_viewer({}) == "napari"
    assert resolve_viewer({"SQUIDMIP_VIEWER": ""}) == "napari"
    assert napari_enabled({}) is True


def test_the_ndviewer_fallback_stays_reachable_by_name():
    """A bad napari path must never leave the window with no viewer during a feedback round."""
    for spelling in ("ndv", "ndviewer", "ndviewer_light", "  NDV  "):
        assert resolve_viewer({"SQUIDMIP_VIEWER": spelling}) == "ndv", spelling
    assert napari_enabled({"SQUIDMIP_VIEWER": "ndv"}) is False


def test_a_typo_does_not_silently_cost_you_the_viewer():
    assert resolve_viewer({"SQUIDMIP_VIEWER": "napri"}) == "napari"


def test_one_resolver_decides_so_the_pane_cannot_disagree_with_the_model():
    """Two readers of one environment variable is how controls end up disagreeing about what is
    on screen. _napari_pane.make_pane asks resolve_viewer rather than parsing it again."""
    import inspect

    from squidmip import _napari_pane

    src = inspect.getsource(_napari_pane.make_pane)
    assert "resolve_viewer" in src, "make_pane does not ask the one resolver"
    assert "os.environ" not in src, (
        "make_pane reads the environment itself instead of asking resolve_viewer — "
        "two readers of one variable is how controls end up disagreeing"
    )


# ------------------------------------------------- identity lives in metadata


def test_identity_is_read_from_metadata_not_parsed_out_of_the_name(layers):
    """The name is a label. Parsing identity back out of it is a known bug class here:
    petakit's reader emits channel names its own regex cannot parse, and 3f1bf3f fixed
    'Fluorescence_488_nm_Ex' failing a parser that wanted r'\\s*nm'."""
    lyr = layers.add_mosaic("stitched", "Fluorescence_488_nm_Ex", _img())

    # A name that would defeat a wavelength regex entirely...
    lyr.name = "something a parser would choke on"

    # ...but identity is unaffected, because it never came from the name.
    assert key_of(lyr) == MosaicKey("stitched", "Fluorescence_488_nm_Ex")
    assert layers.channels("stitched") == ["Fluorescence_488_nm_Ex"]


def test_foreign_layers_are_ignored_not_crashed_on(layers):
    layers.add_mosaic("raw", "488", _img())
    layers.model.add_points(np.zeros((3, 2)), name="user annotation")

    assert key_of(layers.model.layers["user annotation"]) is None
    assert layers.ops() == ["raw"]
    assert len(layers.ours()) == 1


def test_a_layer_with_partial_metadata_is_not_claimed(layers):
    lyr = layers.model.add_image(_img(), name="half", metadata={META_KEY: {"op": "raw"}})
    assert key_of(lyr) is None


# ------------------------------------------------------------ the hierarchy


def test_processing_layers_group_their_channels(layers):
    for op in ("raw", "stitched"):
        for ch in ("405", "488", "561"):
            layers.add_mosaic(op, ch, _img())

    assert layers.ops() == ["raw", "stitched"]
    assert layers.channels("raw") == ["405", "488", "561"]
    assert len(layers.group("stitched")) == 3


def test_show_op_is_the_before_after_toggle(layers):
    for op in ("raw", "stitched"):
        for ch in ("405", "488"):
            layers.add_mosaic(op, ch, _img())

    layers.show_op("raw")
    assert layers.visible_op() == "raw"
    assert all(ly.visible for ly in layers.group("raw"))
    assert not any(ly.visible for ly in layers.group("stitched"))

    layers.show_op("stitched")
    assert layers.visible_op() == "stitched"
    assert not any(ly.visible for ly in layers.group("raw"))


def test_show_op_rejects_an_unknown_processing_layer(layers):
    layers.add_mosaic("raw", "488", _img())
    with pytest.raises(KeyError):
        layers.show_op("deconvolved")


# --------------------------------- contrast: ONE value per channel, no duplication


def test_channel_contrast_survives_the_before_after_toggle(layers):
    """The whole point of linking per channel. Julio: 'I can still see the duplicated
    sliders' — a second control for the same channel must not be able to disagree."""
    for op in ("raw", "stitched"):
        for ch in ("488", "561"):
            layers.add_mosaic(op, ch, _img())

    layers.show_op("raw")
    layers.set_contrast("488", 123, 4321)

    layers.show_op("stitched")

    assert layers.contrast("488") == (123.0, 4321.0)
    assert layers.find("stitched", "488").contrast_limits == [123.0, 4321.0]


def test_contrast_is_per_channel_not_global(layers):
    for ch in ("488", "561"):
        layers.add_mosaic("raw", ch, _img())
        layers.add_mosaic("stitched", ch, _img())

    layers.set_contrast("488", 100, 200)
    assert layers.contrast("561") != (100.0, 200.0)


def test_setting_contrast_on_one_processing_layer_writes_the_other(layers):
    raw = layers.add_mosaic("raw", "488", _img())
    stitched = layers.add_mosaic("stitched", "488", _img())

    raw.contrast_limits = (7, 900)

    assert list(stitched.contrast_limits) == [7.0, 900.0]


def test_contrast_changes_arrive_on_the_public_event(layers):
    """Replaces the ndv contrast tap, which subclassed a private LutView and hooked
    `_lut_controllers`."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("stitched", "488", _img())

    seen = []
    layers.on_contrast_changed(lambda e: seen.append(True))
    layers.set_contrast("488", 50, 5000)

    assert seen, "layer.events.contrast_limits did not fire"


def test_a_degenerate_window_is_not_widened(layers):
    """_pct_window returns hi <= lo for a blank channel on purpose. Widening it to
    (lo, lo + 1) would render a blank channel as full white, i.e. as signal."""
    lyr = layers.add_mosaic("raw", "488", _img(), contrast_limits=(500.0, 500.0))
    assert list(lyr.contrast_limits) != [500.0, 501.0]


# ------------------------------------------------------- placement from stage µm


def test_bbox_um_maps_onto_napari_scale_and_translate_with_the_axis_flip():
    """_tiling speaks (x0, y0, x1, y1); napari speaks (row, col) = (y, x). The flip is the
    silent-transpose risk, so it is pinned."""
    scale, translate = scale_translate_from_bbox_um((100.0, 20.0, 300.0, 120.0), (50, 400))

    # height 100 µm over 50 rows; width 200 µm over 400 cols
    assert scale == pytest.approx((2.0, 0.5))
    # translate is (y0, x0), NOT (x0, y0)
    assert translate == (20.0, 100.0)


def test_bbox_um_rejects_a_degenerate_box():
    with pytest.raises(ValueError):
        scale_translate_from_bbox_um((10.0, 10.0, 10.0, 50.0), (8, 8))


def test_add_mosaic_places_the_layer_in_stage_micrometres(layers):
    lyr = layers.add_mosaic("raw", "488", _img(shape=(64, 64)),
                            bbox_um=(0.0, 0.0, 640.0, 640.0))
    assert tuple(lyr.scale) == pytest.approx((10.0, 10.0))
    assert tuple(lyr.translate) == pytest.approx((0.0, 0.0))


# ----------------------------------------------------------------- replacement


def test_re_adding_a_pair_replaces_it_rather_than_duplicating(layers):
    layers.add_mosaic("raw", "488", _img(seed=1))
    layers.add_mosaic("raw", "488", _img(seed=2))

    assert len(layers.group("raw")) == 1


def test_removing_a_processing_layer_drops_its_channels(layers):
    for ch in ("405", "488"):
        layers.add_mosaic("raw", ch, _img())
        layers.add_mosaic("stitched", ch, _img())

    assert sorted(layers.remove_op("stitched")) == ["405", "488"]
    assert layers.ops() == ["raw"]
    # the survivors still work
    layers.set_contrast("488", 10, 20)
    assert layers.contrast("488") == (10.0, 20.0)


# --------------------------------------------------- binding guards (mutation tested)


def test_bindings_are_present_on_the_installed_napari():
    verify_napari_bindings()


def test_binding_check_bites_when_a_symbol_is_renamed():
    """MUTATION TEST. An assertion nobody has watched fail is only a comment.

    This project lost a day to `_voxel_scale`, which bound cleanly, ran every time, and did
    nothing for its entire life because vispy's Visual.freeze() made the assignment raise into
    an `except AttributeError: pass`. So: rename the symbol, prove the guard fails.
    """
    import napari.qt

    class _Renamed:
        # QtViewer has been renamed away; everything else still looks fine.
        __all__ = ("NotQtViewer",)
        NotQtViewer = object

    with pytest.raises(NapariBindingError) as exc:
        verify_napari_bindings(modules={"napari.qt": _Renamed})

    assert "napari.qt.QtViewer" in str(exc.value)


def test_binding_check_bites_on_a_quiet_de_export():
    """A name that still exists but has left __all__ is a deprecation in progress — exactly
    what happened to Window.qt_viewer. Catch it while it is still only a warning."""

    class _DeExported:
        __all__ = ()          # no longer exported...
        QtViewer = object     # ...but still present

    with pytest.raises(NapariBindingError) as exc:
        verify_napari_bindings(modules={"napari.qt": _DeExported})

    assert "no longer in __all__" in str(exc.value)


def test_every_required_binding_is_individually_load_bearing():
    """Each entry must be able to fail the check on its own, so no entry is decorative."""
    for dotted, attr in REQUIRED_NAPARI_BINDINGS:
        stub = type("Stub", (), {"__all__": ()})
        with pytest.raises(NapariBindingError) as exc:
            verify_napari_bindings(modules={dotted: stub})
        assert f"{dotted}.{attr}" in str(exc.value)


# ------------------------------------------------------------------- embedding


# The embedding check builds a real vispy GL canvas. Doing that in-process under pytest
# aborts the interpreter: pytest/napari have already imported PySide6, and creating the GL
# canvas on top of that is the same Qt-binding conflict test_viewer.py documents ("segfaults
# offscreen under pytest's PySide6/napari-loaded environment — a Qt-binding conflict, not a
# code bug"). Skipping would delete the evidence for the central claim of this module, so the
# check runs in a clean SUBPROCESS instead, where it is a real assertion again and a crash is
# a test failure rather than a dead test session.

_EMBED_SCRIPT = r"""
import json, os, sys, traceback
# Deliberately NOT forcing QT_QPA_PLATFORM=offscreen: the offscreen plugin ships no GL
# ("QOpenGLWidget is not supported on this platform", "does not support
# createPlatformOpenGLContext"), so a vispy canvas segfaults under it. On a machine with a
# display this runs for real; on a headless box it fails cleanly and the test skips with the
# reason attached rather than pretending to have verified something.
import numpy as np
# PyQt5 explicitly, and QT_API pinned before any qtpy import. squidmip imports PyQt5 directly,
# while qtpy defaults to PySide6 here; loading both aborts the process with "Class QMacAutoRelease
# PoolTracker is implemented in both ... QtCore" long before any assertion runs. Test the binding
# production actually uses.
os.environ.setdefault("QT_API", "pyqt5")
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget
app = QApplication.instance() or QApplication([])

# Report OUR OWN errors as EMBEDFAIL, distinct from "this box has no GL". The previous version
# of this script destructured `widget, mosaic = build_pane()` after build_pane grew a third
# return value; it raised, printed no EMBED line, and the test SKIPPED -- so it read green for
# its whole life while asserting nothing. A skip and a bug must not look the same.
try:
    from squidmip._napari_pane import MosaicPane

    host = QWidget()
    lay = QHBoxLayout(host)
    pane = MosaicPane()
    lay.addWidget(pane)
    app.processEvents()

    pane.mosaic.add_mosaic("raw", "488", np.zeros((32, 32), dtype="uint16"))
    app.processEvents()

    win = pane._native_window
    central = win.centralWidget() if win is not None else None
    canvas = pane.canvas

    def descends_from(child, ancestor):
        node = child
        while node is not None:
            if node is ancestor:
                return True
            node = node.parent()
        return False

    out = {
        "native_window_embedded": win is not None,
        "window_is_in_our_pane": descends_from(win, pane) if win is not None else False,
        # THE INVARIANT THIS FILE EXISTS FOR: the canvas must still live inside napari's own
        # QMainWindow. Reparenting it out left the window gutted -- docks and layer controls
        # still showed, so the pane looked alive while the mosaic had nowhere to paint.
        "canvas_still_inside_napari_window": (
            descends_from(canvas, win) if (win is not None and canvas is not None) else False
        ),
        "central_is_not_empty": (
            len(central.findChildren(QWidget)) > 0 if central is not None else False
        ),
        # napari's real chrome SHOULD be present now -- that is the whole point of embedding the
        # real window instead of rebuilding its parts by hand.
        "layer_controls": len([c for c in win.findChildren(QWidget)
                               if "QtLayerControlsContainer" in type(c).__name__])
                          if win is not None else 0,
        "ops": pane.mosaic.ops(),
    }
    print("EMBED " + json.dumps(out))
except BaseException:
    print("EMBEDFAIL " + json.dumps(traceback.format_exc()))
sys.stdout.flush()
os._exit(0)
"""


def test_the_canvas_stays_inside_the_embedded_napari_window(tmp_path):
    """napari's canvas must remain its QMainWindow's central widget after we embed that window.

    ``MosaicPane`` used to call ``canvas.setParent(self)``, which RIPS the QtViewer out of
    napari's own window. ``_embed_native_window`` then embedded the gutted window: the docks and
    layer controls came along, so the pane looked alive and populated, while the canvas sat
    parented to the pane and added to no layout at all. Every mosaic layer was present and
    correct in the layer list and nothing painted -- reported as "canvas is still showing blank
    for the array, so I can't test the central viewer".

    This is the project's silent-failure shape again: the failure surfaced as absence (a black
    rectangle), and every readable signal -- layer list, contrast controls, blending -- said the
    viewer was fine. So assert the STRUCTURE, not the appearance.
    """
    import json
    import subprocess
    import sys

    pytest.importorskip("qtpy")

    script = tmp_path / "embed_check.py"
    script.write_text(_EMBED_SCRIPT)

    import os
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # The commit gate exports QT_QPA_PLATFORM=offscreen for the whole suite, and the offscreen
    # plugin has no GL, so inheriting it guarantees a segfault and a permanent skip. Drop it and
    # let Qt pick the real platform: on a machine with a display this actually verifies, and on
    # a headless one it fails cleanly into the skip below with the reason attached.
    env.pop("QT_QPA_PLATFORM", None)
    # Both PyQt5 and PySide6 are installed here. squidmip imports PyQt5, so qtpy (and napari
    # through it) must resolve to the same binding or the process aborts before asserting.
    env["QT_API"] = "pyqt5"

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=300, cwd=str(repo), env=env,
    )
    # An exception in OUR code is a FAILURE, not a skip. Only a genuinely GL-less box skips.
    failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("EMBEDFAIL ")]
    if failed:
        pytest.fail("embedding raised:\n" + json.loads(failed[0][len("EMBEDFAIL "):]))

    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("EMBED ")]
    if not line:
        pytest.skip(
            "napari Qt canvas could not be constructed in this environment "
            f"(rc={proc.returncode}); stderr tail: {proc.stderr[-400:]}"
        )

    got = json.loads(line[0][len("EMBED "):])
    assert got["native_window_embedded"] is True
    assert got["window_is_in_our_pane"] is True
    # THE regression: rip the canvas out of napari's window and the mosaic has nowhere to paint.
    assert got["canvas_still_inside_napari_window"] is True
    assert got["central_is_not_empty"] is True
    # napari's real controls are the reason we embed the real window; their absence means we are
    # back to hand-rebuilding them.
    assert got["layer_controls"] >= 1
    assert got["ops"] == ["raw"]


def test_channels_composite_additively_not_occluding_each_other():
    """Fluorescence channels must SUM, not stack opaquely.

    napari defaults every layer to blending='translucent', so the last-added layer occludes the
    rest. On the 10x tissue set the channel order ends at 638 nm, whose palette colour is
    #FF0000, so the whole mosaic rendered flat RED and read as a single-channel bug. Reported
    from the live GUI: "Mosaic showing red, so like single collor".

    _montage.py already states the intended model for the browser path: "the per-channel PNGs
    with screen blending, which is the same additive composite". The canvas must match it.
    """
    import numpy as np

    from napari.components import ViewerModel

    from squidmip._napari_view import MosaicLayers

    m = MosaicLayers(ViewerModel())
    for ch in ("Fluorescence_405_nm_Ex", "Fluorescence_638_nm_Ex"):
        m.add_mosaic("raw", ch, np.zeros((4, 4), dtype="uint16"))

    blendings = {str(layer.blending) for layer in m.ours()}
    assert blendings == {"additive"}, (
        f"channels must composite additively; got {blendings}. With 'translucent' the last "
        f"channel added hides every earlier one."
    )


def test_blending_is_overridable_without_reaching_into_the_layer(layers):
    lyr = layers.add_mosaic("raw", "488", _img(), blending="translucent")
    assert lyr.blending == "translucent"


# ------------------------------------ contrast ownership: the plate must never write back


def test_our_own_contrast_writes_do_not_look_like_the_user_moving_a_slider(layers):
    """The exact trap IMA-261 found: a SINK writing a viewer-originated autoscale back into its
    own policy state latched all four channels to MANUAL on open, killing per-region contrast
    while the plate still drew an amber 'wells NOT comparable' badge that was therefore lying."""
    seen = []
    layers.add_mosaic("raw", "488", _img(), contrast_limits=(10.0, 900.0))
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    # adding another layer for the same channel is OUR write, and it propagates via the link
    layers.add_mosaic("stitched", "488", _img(), contrast_limits=(20.0, 800.0))

    assert seen == [], f"programmatic contrast write leaked to the sink: {seen}"


def test_a_write_we_make_ourselves_after_arming_still_does_not_reach_the_sink(layers):
    """The guard above only covers a write made while the layer is being CONSTRUCTED, so it
    passes whether or not ``is_programmatic`` is checked — the tap is armed after the limits are
    set, and no event ever fires. This one writes through the tap, which is where the guard has
    to hold: the plate treating our own autoscale as a user gesture is what latched every
    channel MANUAL on open and killed per-region contrast from the first frame.

    MUTATION: drop the ``is_programmatic`` check in the tap and this goes red.
    """
    layers.add_mosaic("raw", "488", _img())
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    with layers.programmatic():
        layers.set_contrast("488", 11.0, 222.0)

    assert seen == [], f"our own write leaked to the sink: {seen}"
    # ...and the sink is not left deaf afterwards.
    layers.find("raw", "488").contrast_limits = (13.0, 444.0)
    assert seen == [("488", 13.0, 444.0)]


def test_a_user_drag_does_reach_the_sink(layers):
    layers.add_mosaic("raw", "488", _img())
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    layers.find("raw", "488").contrast_limits = (33.0, 777.0)   # the user moving napari's slider

    assert seen == [("488", 33.0, 777.0)]


def test_a_user_drag_on_a_LATER_layer_also_reaches_the_sink(layers):
    """The plate must keep following napari for layers added AFTER it subscribed.

    ``_bind_napari_contrast`` connects ONCE, on the first mosaic; every op run after that
    (a second region, a re-ingest, a plane-op) calls ``add_mosaic`` again. If the sink is
    only wired to the layers that happened to exist at subscribe time, those later layers
    drive napari and nothing else, and the plate's contrast silently diverges from what
    the user is looking at. Julio, repeatedly: "contrast of regions and napari are different."
    """
    seen: list = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    layers.add_mosaic("raw", "488", _img())            # added AFTER the subscribe
    layers.find("raw", "488").contrast_limits = (33.0, 777.0)     # a real user drag

    assert seen == [("488", 33.0, 777.0)], (
        "a layer added after on_user_contrast() never reached the sink"
    )


def test_a_second_op_layer_added_later_still_reaches_the_sink(layers):
    """Same defect, the shape it actually ships in: op 2 arrives after the bind."""
    layers.add_mosaic("raw", "488", _img())
    seen: list = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    layers.add_mosaic("bgsub", "561", _img())          # a NEW channel, after the bind
    layers.find("bgsub", "561").contrast_limits = (12.0, 345.0)

    assert seen == [("561", 12.0, 345.0)]


def test_the_sink_survives_the_layers_being_rebuilt(layers):
    """THE HALF-LIFE BUG: ``on_user_contrast`` connected to the layer objects that existed at
    SUBSCRIBE time, and ``_load_mosaic`` removes and re-adds every layer on each region change.
    So the plate followed napari's contrast until the user opened a second region, and then
    stopped — silently, with the slider still moving and nothing downstream listening.

    A subscription that dies the first time the thing it watches is rebuilt is worse than none:
    it works in the demo and is gone by the second click.

    MUTATION: connect only inside ``on_user_contrast`` (the old shape) and this goes red.
    """
    layers.add_mosaic("raw", "488", _img())
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    layers.remove_op("raw")                       # exactly what a region change does
    layers.add_mosaic("raw", "488", _img())       # ...and then rebuilds

    layers.find("raw", "488").contrast_limits = (12.0, 345.0)
    assert seen == [("488", 12.0, 345.0)], f"the sink went deaf after a rebuild: {seen}"


def test_a_channel_added_after_subscribing_is_also_heard(layers):
    """A subscriber must not have to know the channel order the mosaic worker happens to use."""
    ly = layers
    seen = []
    ly.on_user_contrast(lambda ch, lo, hi: seen.append(ch))
    ly.add_mosaic("raw", "561", _img())
    ly.find("raw", "561").contrast_limits = (1.0, 2.0)
    assert seen == ["561"]


def test_one_user_drag_is_reported_once_however_many_layers_share_the_channel(layers):
    """Contrast is LINKED per channel, so one drag moves every peer and each peer fires. The
    sink must still hear it once — a plate that recomposites per peer does N times the work for
    one gesture."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("stitched", "488", _img())
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    layers.find("raw", "488").contrast_limits = (5.0, 55.0)
    assert seen == [("488", 5.0, 55.0)]


def test_programmatic_is_reentrant_and_restores_state(layers):
    with layers.programmatic():
        assert layers.is_programmatic
        with layers.programmatic():
            assert layers.is_programmatic
        assert layers.is_programmatic
    assert not layers.is_programmatic


# --------------------------------------------------------- z axis and voxel geometry


def test_a_zstack_gets_a_navigable_axis_and_a_2d_plane_does_not(layers):
    """REPORT 2. napari puts a dimension slider on every axis it is not displaying, so a 3-D
    array is all that is needed. A 2-D array leaves no axis to put a slider on — which is why z
    was not controllable: the mosaic was fused at a fixed z before napari ever saw it."""
    layers.add_mosaic("raw", "488", _img(shape=(32, 32)))
    assert layers.model.dims.ndim == 2
    assert list(layers.model.dims.not_displayed) == []

    layers.remove_op("raw")
    layers.add_mosaic("raw", "488", np.zeros((10, 32, 32), dtype=np.uint16))
    assert layers.model.dims.ndim == 3
    assert list(layers.model.dims.not_displayed) == [0]


def test_the_z_axis_carries_the_step_in_micrometres_not_one(layers):
    """A unit z scale steps fine in 2-D and renders an isotropic block in 3-D out of
    anisotropic data. IMA-255 exists because dz/pixel has to reach the renderer."""
    lyr = layers.add_mosaic("raw", "488", np.zeros((10, 32, 32), dtype=np.uint16),
                            bbox_um=(0.0, 0.0, 320.0, 320.0), z_scale_um=1.5)
    assert tuple(lyr.scale) == pytest.approx((1.5, 10.0, 10.0))
    assert tuple(lyr.translate) == pytest.approx((0.0, 0.0, 0.0))


def test_xy_placement_is_unaffected_by_the_extra_z_axis(layers):
    """The trailing two axes are (y, x); a silent transpose here draws a plausible wrong mosaic."""
    flat = layers.add_mosaic("raw", "488", np.zeros((40, 80), dtype=np.uint16),
                             bbox_um=(0.0, 0.0, 800.0, 400.0))
    layers.remove_op("raw")
    stack = layers.add_mosaic("raw", "488", np.zeros((6, 40, 80), dtype=np.uint16),
                              bbox_um=(0.0, 0.0, 800.0, 400.0), z_scale_um=2.0)

    assert tuple(flat.scale) == pytest.approx((10.0, 10.0))
    assert tuple(stack.scale)[1:] == pytest.approx(tuple(flat.scale))


# ---------------------------- Defect 5: subscriptions must outlive the layers they watch
#
# on_user_contrast subscribed to layer OBJECTS that existed at the moment of subscription.
# _load_mosaic destroys and recreates every layer on each region change, so the sync had a
# half-life of exactly one region change and then went quiet -- silently, which is the worst
# way for a sync to stop. Subscriptions key on CHANNEL identity instead, so they survive layer
# recreation. The same shape would bite channel visibility and Z/T sync.


def test_a_channel_added_AFTER_subscribing_still_reaches_the_sink(layers):
    """THE BUG. The subscription is to a channel, not to whichever layers existed at the time."""
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    layers.add_mosaic("raw", "488", _img())            # channel arrives after the subscribe
    layers.find("raw", "488").contrast_limits = (12.0, 345.0)

    assert seen == [("488", 12.0, 345.0)], (
        f"a channel added after subscribing never reached the sink: {seen}"
    )


def test_the_sync_survives_a_region_change_that_recreates_every_layer(layers):
    """_load_mosaic's actual lifecycle: subscribe, then destroy and recreate the layers.

    Before the fix the recreated layer had no connection, so the second drag produced nothing
    and nothing said so.
    """
    layers.add_mosaic("raw", "488", _img())
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))
    layers.find("raw", "488").contrast_limits = (10.0, 100.0)
    assert len(seen) == 1, "baseline drag did not arrive"

    layers.remove_op_channel("raw", "488")             # region change: layers destroyed ...
    layers.add_mosaic("raw", "488", _img())            # ... and recreated
    layers.find("raw", "488").contrast_limits = (20.0, 200.0)

    assert seen[-1] == ("488", 20.0, 200.0), (
        f"contrast sync died when the layer was recreated: {seen}"
    )


def test_a_user_drag_fires_ONCE_even_though_the_channel_has_several_linked_layers(layers):
    """Linked layers propagate the write to their peers. If every peer fired, one drag would
    deliver N callbacks and a sink that counts or accumulates would be wrong."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("stitched", "488", _img())
    layers.add_mosaic("decon", "488", _img())
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    layers.find("raw", "488").contrast_limits = (44.0, 555.0)

    assert seen == [("488", 44.0, 555.0)], f"expected exactly one delivery, got {seen}"


def test_our_own_writes_still_do_not_look_like_the_user_after_the_rewiring(layers):
    """The programmatic guard is the safety property of this design; keying on channel must
    not have quietly cost it (add_mosaic writes contrast, and that is OUR write)."""
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))
    layers.add_mosaic("raw", "488", _img(), contrast_limits=(10.0, 900.0))
    layers.add_mosaic("stitched", "488", _img(), contrast_limits=(20.0, 800.0))
    assert seen == [], f"programmatic write leaked to the sink: {seen}"


def test_two_subscribers_both_receive_a_channel_added_later(layers):
    a, b = [], []
    layers.on_user_contrast(lambda ch, lo, hi: a.append(ch))
    layers.on_user_contrast(lambda ch, lo, hi: b.append(ch))
    layers.add_mosaic("raw", "561", _img())
    layers.find("raw", "561").contrast_limits = (5.0, 50.0)
    assert a == ["561"] and b == ["561"]


def test_dragging_BACK_to_a_previously_delivered_value_is_not_swallowed(layers):
    """The trap in deduping link echoes by value.

    Echoes are collapsed by comparing against the last value SEEN. If our own programmatic
    writes did not also update that record, this sequence would silently drop the final drag:
    the user returns the window to a value delivered earlier, and the sink never hears about
    it. Programmatic writes update _last_seen precisely so this stays live.
    """
    layers.add_mosaic("raw", "488", _img())
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((lo, hi)))

    layers.find("raw", "488").contrast_limits = (10.0, 100.0)     # user
    with layers.programmatic():                                    # us (a re-add / autoscale)
        layers.set_contrast("488", 30.0, 300.0)
    layers.find("raw", "488").contrast_limits = (10.0, 100.0)     # user, BACK to the first

    assert seen == [(10.0, 100.0), (10.0, 100.0)], f"a real drag was swallowed: {seen}"


# ---------------------------------------- the plate follows napari's COLORMAP, not just contrast
#
# Julio: "I change channel colormap in napari and plate view doesn't react." The plate composites
# with its own (C, 3) RGB table resolved once from the acquisition's display_color -- a second
# answer to "what colour is this channel", settled at open and never revised. Same defect shape
# as the contrast that would not follow, so it gets the same shape of fix: napari owns it, the
# plate subscribes.

def test_changing_a_colormap_reports_the_new_rgb(layers):
    layers.add_mosaic("raw", "488", _img())
    seen = []
    layers.on_user_colormap(lambda ch, rgb: seen.append((ch, rgb)))

    layers.find("raw", "488").colormap = "red"

    assert seen, "the plate was never told the channel changed colour"
    ch, rgb = seen[-1]
    assert ch == "488"
    assert rgb[0] > 0.9 and rgb[1] < 0.1 and rgb[2] < 0.1, f"expected red at full intensity, got {rgb}"


def test_the_colormap_sink_survives_the_layers_being_rebuilt(layers):
    """Same half-life bug the contrast tap had: subscribe once, then a region change destroys and
    recreates every layer. Armed in _register_channel, so a rebuilt layer is still wired.

    MUTATION: move the _connect_user_colormap call out of _register_channel and this goes red.
    """
    layers.add_mosaic("raw", "488", _img())
    seen = []
    layers.on_user_colormap(lambda ch, rgb: seen.append(ch))

    layers.remove_op("raw")                        # exactly what a region change does
    layers.add_mosaic("raw", "488", _img())        # ...and then rebuilds
    layers.find("raw", "488").colormap = "green"

    assert seen == ["488"], f"the colour sink went deaf after a rebuild: {seen}"


def test_our_own_colormap_writes_are_not_reported_as_user_gestures(layers):
    """add_mosaic sets the channel's colormap itself. If that echoed back as a gesture the plate
    would re-tint from its own defaults on every region change."""
    seen = []
    layers.on_user_colormap(lambda ch, rgb: seen.append(ch))
    with layers.programmatic():
        layers.add_mosaic("raw", "561", _img(), colormap="magenta")
    assert seen == []


def test_channel_rgb_reports_what_the_canvas_is_tinting_with(layers):
    layers.add_mosaic("raw", "638", _img(), colormap="blue")
    rgb = layers.channel_rgb("638")
    assert rgb is not None and rgb[2] > 0.9
    assert layers.channel_rgb("no-such-channel") is None
