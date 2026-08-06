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


def test_a_retired_ndviewer_name_still_builds_napari_and_says_so(caplog):
    """The fallback is gone, so asking for it by name must announce the substitution.

    This test asserted the opposite until 2026-07-30: that ``SQUIDMIP_VIEWER=ndv`` reached
    ndviewer_light, so "a bad napari path never leaves the window with no viewer". That stack was
    deleted, because napari won the written gate and because ndviewer_light imports PyQt5 at
    module scope and so cannot share a process with a Qt6 napari.

    What is pinned now is the NO FALLBACKS shape of it. A stale variable in someone's shell
    profile or launcher must not silently hand them a different viewer than the one they named:
    they get napari AND a sentence in the log saying why. Silence here would be the very defect
    the original fallback was written to avoid, just pointing the other way.
    """
    import logging

    for spelling in ("ndv", "ndviewer", "ndviewer_light", "  NDV  "):
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            assert resolve_viewer({"SQUIDMIP_VIEWER": spelling}) == "napari", spelling
        assert "deleted" in caplog.text, f"{spelling!r} was retired in silence"
    assert napari_enabled({"SQUIDMIP_VIEWER": "ndv"}) is True


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


# ---------------------- ONE operator per channel on screen (Julio, 2026-08-03)
#
# "Intensity grows with the amount of layers that are toggled on in my window."
#
# Every mosaic is added ``additive``. That is CORRECT across channels -- 405+488+561+638 summing
# is the composite, and it is why `add_mosaic` chose additive over napari's occluding default --
# and it is arithmetic nonsense across OPERATORS of one channel: raw · 488 and mip · 488 lit
# together is one channel's signal counted twice, three of them counted three times.
#
# A blending mode cannot draw that distinction: napari blends a layer against everything beneath
# it in one flat stack, so `translucent` would stop the operators summing only by making them
# OCCLUDE the other channels -- the exact defect additive was chosen to fix. Exclusivity is the
# fix, per CHANNEL, and it is enforced on `layer.visible` itself so that the layer tree AND
# napari's own eye icons both obey it.


def test_lighting_one_operator_darkens_the_other_operator_of_that_channel(layers):
    """The reported defect, driven the way napari's own eye icon drives it."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("mip", "488", _img(1), visible=False)
    assert layers.find("raw", "488").visible is True

    layers.find("mip", "488").visible = True

    assert layers.find("raw", "488").visible is False, (
        "raw · 488 and mip · 488 are both lit and both additive, so 488 is being summed twice")
    assert layers.find("mip", "488").visible is True


def test_channels_of_one_operator_still_all_sum(layers):
    """The other half, and the one that must NOT change. Four channels of one operator are a
    composite; darkening them would be the 'why is the mosaic only displaying a channel?' bug."""
    for ch in ("405", "488", "561", "638"):
        layers.add_mosaic("raw", ch, _img())

    assert [ly.visible for ly in layers.group("raw")] == [True] * 4
    assert {str(ly.blending) for ly in layers.group("raw")} == {"additive"}


def test_an_operator_result_arrives_instead_of_raw_rather_than_on_top_of_it(layers):
    """The delivery path. ``_add_result_layers`` / ``deliver_result`` add an operator layer with
    ``visible=True`` over a lit raw, so without this the sum is there before the user has touched
    anything -- same defect, no gesture."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("raw", "561", _img(1))

    layers.add_mosaic("mip", "488", _img(2))          # the result, as delivered

    assert layers.find("raw", "488").visible is False
    assert layers.find("raw", "561").visible is True, (
        "561 has no second copy on screen, so nothing about it had to change")


def test_a_result_delivered_dark_darkens_nothing(layers):
    """A window that did not ask gets ``visible=False`` (``_deliver_to_views``). A layer that is
    not on screen cannot be summing with anything, so it must not take raw down with it."""
    layers.add_mosaic("raw", "488", _img())

    layers.add_mosaic("mip", "488", _img(1), visible=False)

    assert layers.find("raw", "488").visible is True


def test_hiding_an_operator_lights_nothing(layers):
    """Only turning a layer ON can force another off. A checkbox going dark that turns something
    else on is a control moving a second control, which is this project's oldest defect shape."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("mip", "488", _img(1))         # raw · 488 is now dark

    layers.find("mip", "488").visible = False

    assert layers.find("raw", "488").visible is False
    assert layers.visible_op() is None


def test_an_analysis_overlay_is_never_darkened_by_the_mosaic_it_is_drawn_over(layers):
    """``add_labels``/``add_points`` skip ``_register_channel`` on purpose (they have no
    ``contrast_limits`` to link), so they are not in ``_by_channel`` and the exclusivity never
    sees them. That is what ``SPOTS_OP`` relies on: a mask is drawn OVER the mosaic, not instead
    of it, and it is ``translucent`` rather than additive so it does not sum either."""
    layers.add_mosaic("raw", "488", _img())
    mask = layers.add_labels("spots", "488 mask", np.zeros((32, 32), dtype="uint32"))

    layers.add_mosaic("mip", "488", _img(1))         # a switch under the overlay

    assert mask.visible is True, "the spot mask went dark when the mosaic under it was switched"
    assert layers.find("raw", "488").visible is False


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


def test_an_operator_layer_opens_on_its_own_window_not_raw_s(layers):
    """The half of the contrast story the two tests above cannot see, because both of them
    write a contrast value first and only check what happens AFTER.

    ``link_layers`` connects EVENTS. It does not equalise values at link time, so a freshly
    added operator layer keeps the window ``add_mosaic`` seeded from its own pixels until
    somebody writes one. That is deliberate -- a decon result has to be legible on its own
    terms or you cannot judge whether it used the right iteration count -- but it was
    documented as the opposite ("flipping between raw and this operator preserves the window")
    in two docstrings, so it is pinned here rather than assumed.
    """
    dim = np.full((32, 32), 300, dtype=np.uint16)
    dim[:4] = 900                        # a little signal over a dim background
    bright = np.full((32, 32), 6000, dtype=np.uint16)
    bright[:4] = 30000

    raw = layers.add_mosaic("raw", "488", dim)
    decon = layers.add_mosaic("decon", "488", bright)

    assert list(raw.contrast_limits) != list(decon.contrast_limits), (
        "the operator layer opened on raw's window; per-layer seeding is gone, and an operator "
        "result that is not individually legible cannot be judged on arrival"
    )


def test_match_raw_contrast_puts_every_operator_peer_on_raw_s_window(layers):
    """The explicit opt-in: one action turns the raw->operator flip into a real comparison.

    Asserted on the LAYERS' contrast values, across two channels and two operators, because the
    user's question is "do these two pictures now share a stretch", not "was a method called".
    """
    dim = np.full((32, 32), 300, dtype=np.uint16)
    dim[:4] = 900
    bright = np.full((32, 32), 6000, dtype=np.uint16)
    bright[:4] = 30000

    for ch in ("488", "561"):
        layers.add_mosaic("raw", ch, dim)
        layers.add_mosaic("decon", ch, bright)
        layers.add_mosaic("stitched", ch, bright)

    matched = layers.match_contrast_to("raw")

    assert matched == 4, "two operator layers on each of two channels should have been written"
    for ch in ("488", "561"):
        want = list(layers.find("raw", ch).contrast_limits)
        for op in ("decon", "stitched"):
            assert list(layers.find(op, ch).contrast_limits) == want, f"{op}/{ch}"


def test_match_raw_contrast_skips_a_channel_the_source_op_does_not_show(layers):
    """A channel with no raw layer is left alone rather than being matched to something else.

    The realistic case is an operator that emits a channel raw does not have; silently handing
    it another channel's window would be a wrong picture with no error.
    """
    layers.add_mosaic("raw", "488", np.full((32, 32), 300, dtype=np.uint16))
    layers.add_mosaic("decon", "488", np.full((32, 32), 6000, dtype=np.uint16))
    only_op = layers.add_mosaic("decon", "561", np.full((32, 32), 12000, dtype=np.uint16))
    before = list(only_op.contrast_limits)

    assert layers.match_contrast_to("raw") == 1
    assert list(only_op.contrast_limits) == before


def test_neither_the_link_nor_the_match_carries_the_COLORMAP(layers):
    """Contrast is shared in this window; COLOUR is not, by either mechanism.

    This is half of why the copy/paste-LUTs chips survive an audit that asked whether they were
    redundant (Julio: "if contrast are synched, the LUT copy paste should be removed"). The link
    is bound to ``("contrast_limits",)`` and nothing else, and ``match_contrast_to`` writes
    ``contrast_limits`` and nothing else, so a recolour in one layer stays in that layer. The
    window's ``_apply_luts`` -- the paste -- is the only thing in the app that writes
    ``layer.colormap``. Asserted against a real ``ViewerModel`` because the claim is about what
    napari's own linking does, not about our stub.
    """
    raw = layers.add_mosaic("raw", "488", _img(), colormap="green")
    decon = layers.add_mosaic("decon", "488", _img(1), colormap="green")

    raw.colormap = "magenta"
    raw.contrast_limits = (7, 900)

    assert list(decon.contrast_limits) == [7.0, 900.0], "the contrast link stopped working"
    assert decon.colormap.name == "green", "the link carried the colormap; the paste is now a dupe"

    layers.match_contrast_to("raw")
    assert decon.colormap.name == "green", (
        "match_contrast_to grew a second responsibility; it is the CONTRAST equaliser and the "
        "chip's label promises only that")


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


# ------------------------------------------- what a window open COSTS to put on screen
#
# The three tests below are one claim in three parts: opening a region window must pull each
# channel's mosaic ONCE, not four times. They are counting tests, not timing tests (ADR-0001),
# and they are here rather than in a benchmark because the cost is not incidental — it is decided
# by the ORDER of the calls in `add_mosaic`, which is exactly the sort of thing a later edit
# rearranges without noticing.


def _pyramid(nz=10, pulls=None):
    """A lazy ``(z, y, x)`` pyramid built EXACTLY as ``fuse_region_pyramid`` builds one — one dask
    block per z, per level — whose blocks record the z index when they are computed.

    Same construction as the real thing (``da.from_delayed`` per z, concatenated on axis 0) so the
    count this yields is the count of whole-region fuses a real open would pay. A plain ndarray
    would count nothing: napari would slice it for free and the test would pass on a viewer that
    fetches every plane.
    """
    import dask.array as da
    from dask import delayed

    pulls = [] if pulls is None else pulls
    levels = []
    for side in (64, 32):
        def _block(z, side=side):
            pulls.append(int(z))
            return np.full((side, side), 100 + int(z) * 7, dtype=np.uint16)

        levels.append(da.concatenate(
            [da.from_delayed(delayed(_block)(z), shape=(side, side), dtype=np.uint16)[None, ...]
             for z in range(nz)], axis=0))
    return levels, pulls


def test_the_contrast_seed_samples_the_plane_napari_shows(layers):
    """The seed and the display must be the SAME plane, or the window describes one the user is
    not looking at — and pays a whole extra fuse of the region to get it.

    They disagreed. ``sample_plane`` took ``nz // 2`` while napari centres its slider on
    ``(nz - 1) // 2`` of the world range, so on an even stack the seed came from plane 5 and the
    canvas showed plane 4. Invisible on screen (both are in-focus middle planes) and 27 extra
    whole-frame decodes per channel on the real 10x region.

    Driven through a REAL ``ViewerModel`` rather than by restating napari's centring arithmetic:
    the claim is about what napari does, so napari has to be the one asked. If a future napari
    centres differently, this fails instead of the cost quietly returning.
    """
    from squidmip._contrast import opening_z

    for nz in (10, 9, 2):
        mdl = MosaicLayers(type(layers._model)())
        data, _pulls = _pyramid(nz=nz)
        mdl.add_mosaic("raw", "488", data, multiscale=True,
                       bbox_um=(0.0, 0.0, 640.0, 640.0), z_scale_um=1.5)

        assert mdl._model.dims.current_step[0] == opening_z(nz), (
            f"nz={nz}: napari displays plane {mdl._model.dims.current_step[0]} but the contrast "
            f"seed samples plane {opening_z(nz)}")


def test_adding_a_placed_mosaic_pulls_two_z_not_four(layers):
    """The whole point of the change, stated as the number of region fuses an open costs.

    Measured on the real acquisition (manual0, 27 FOVs, 4 channels, nz=10) before this: FOUR
    whole-region decode passes per channel — the seed's z, napari's first slice at z=0, napari's
    centred slice, and a fourth because ``_place`` assigned ``layer.scale`` AFTER the layer
    existed, which moved the dims range and moved the slider again. 432 frame decodes for a window
    that needs 216.

    Two remain and only one is avoidable here: napari's ``Image`` slices itself at point 0 in its
    own constructor, before the viewer's dims can be consulted. That one is NOT fixed and is not
    fixable from outside napari.
    """
    from squidmip._contrast import opening_z

    data, pulls = _pyramid(nz=10)
    layers.add_mosaic("raw", "488", data, multiscale=True,
                      bbox_um=(0.0, 0.0, 640.0, 640.0), z_scale_um=1.5)

    assert set(pulls) == {0, opening_z(10)}, (
        f"a mosaic add materialised planes {sorted(set(pulls))}; every plane beyond "
        f"{{0, {opening_z(10)}}} is a whole region decoded and thrown away")


def test_placing_at_construction_puts_the_layer_exactly_where_placing_it_after_would(layers):
    """The saving must not move the picture. ``placement_for`` is now used twice — once as
    ``add_image`` kwargs and once by ``_place`` — and two placement rules that disagree is the
    defect ``_place``'s docstring exists to prevent.

    Units are asserted too: they used to be set inside ``_place``, and a layer that skips ``_place``
    would otherwise lose its µm scale-bar labels with nothing on screen to say so.
    """
    # (x0, y0, x1, y1) = 640 µm square over a 64 px level 0, so 10 µm/px on both displayed axes,
    # translate (y0, x0) after the flip, and 1.5 µm on the z axis in front of both.
    data, _pulls = _pyramid(nz=6)
    lyr = layers.add_mosaic("raw", "488", data, multiscale=True,
                            bbox_um=(100.0, 20.0, 740.0, 660.0), z_scale_um=1.5)

    assert tuple(lyr.scale) == pytest.approx((1.5, 10.0, 10.0))
    assert tuple(lyr.translate) == pytest.approx((0.0, 20.0, 100.0))
    assert [str(u) for u in lyr.units] == ["micrometer"] * 3, (
        "a layer placed at construction lost the micrometre axis labels the scale bar reads")


def test_a_bbox_the_placement_rule_refuses_is_still_refused(layers):
    """Placement moved into the ``add_image`` call, and moving WHERE a rule runs must not change
    WHETHER it runs. A degenerate box raised out of ``add_mosaic`` before this change (``_place``
    let ``scale_translate_from_bbox_um`` through) and must still: a mosaic that silently lands
    unplaced sits at scale 1 in a µm world, which looks like a rendering bug and reports as one.

    The ``except`` in ``add_mosaic`` only declines to place EARLY; ``_place`` is what then raises.
    """
    with pytest.raises(ValueError):
        layers.add_mosaic("raw", "488", _img(shape=(64, 64)),
                          bbox_um=(10.0, 10.0, 10.0, 50.0))


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
from qtpy.QtWidgets import QApplication, QHBoxLayout, QWidget
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


# ----------------------------------------------------------------- scale bar (IMA-265)


def test_enable_scale_bar_turns_it_on_in_micrometres():
    """The mosaic IS a zoomable micrometre view, so it must carry a scale bar. napari has one
    built in; this only configures it (visible, µm), it does not build a bar of our own.

    MUTATION: make enable_scale_bar a no-op -> visible stays False -> red.
    """
    from napari.components import ViewerModel

    from squidmip._napari_view import enable_scale_bar

    v = ViewerModel()
    enable_scale_bar(v)
    assert v.scale_bar.visible is True
    # napari >=0.8 DEPRECATED ScaleBar.unit: the setter is a no-op and the getter always returns
    # None, because units are now computed from the layers ("Use `Layer.units` to set units for
    # each layer"). That is not a regression here -- `enable_scale_bar`'s own docstring already
    # says `layer.units` is the source napari >=0.7 reads and `scale_bar.unit` is only the <0.7
    # fallback, and `add_mosaic` sets the former. So assert the label only where the model still
    # reports one, rather than pinning an API that now answers None on every napari the project
    # supports. The micrometre CORRECTNESS is pinned independently, and more strongly, by
    # test_the_bar_reads_micrometres_because_the_layer_scale_IS_micrometres below: the number the
    # bar draws is world = data * layer.scale, which no deprecation touches.
    unit = v.scale_bar.unit
    if unit is not None:
        # pint normalises "um" -> "micrometer"; either spelling is the micron, never pixels.
        assert str(unit) in ("um", "µm", "micrometer")


def test_the_bar_reads_micrometres_because_the_layer_scale_IS_micrometres():
    """The number the bar shows is world coordinates, and world = data * layer.scale. So the bar
    is correct IFF layer.scale is µm/px. add_mosaic sets scale from bbox_um, and this pins that a
    64 px layer given a 640 µm box spans 640 world units -- i.e. the bar's µm are real µm.

    This is the correctness check the plan demanded: a scale bar that lies is worse than none.

    MUTATION: divide bbox by 2 in scale_translate_from_bbox_um -> world extent halves -> red.
    """
    from napari.components import ViewerModel

    v = ViewerModel()
    layers = MosaicLayers(v)
    lyr = layers.add_mosaic("raw", "488", _img(shape=(64, 64)),
                            bbox_um=(0.0, 0.0, 640.0, 640.0))
    # world extent along x = cols * scale_x = 64 * 10 µm = 640 µm, exactly the box width.
    assert lyr.data.shape[-1] * float(lyr.scale[-1]) == pytest.approx(640.0)
    assert lyr.data.shape[-2] * float(lyr.scale[-2]) == pytest.approx(640.0)


def test_add_mosaic_labels_the_layer_units_micrometres():
    """napari >=0.7 reads the scale bar's unit from the LAYER, not the deprecated
    viewer.scale_bar.unit. Labelling each mosaic keeps the bar honest across that migration.

    MUTATION: stop setting layer.units -> stays pixel -> red.
    """
    from napari.components import ViewerModel

    v = ViewerModel()
    layers = MosaicLayers(v)
    lyr = layers.add_mosaic("raw", "488", _img(shape=(16, 16)),
                            bbox_um=(0.0, 0.0, 160.0, 160.0))
    assert all("meter" in str(u) or str(u) in ("um", "µm") for u in lyr.units)
# ---------------------------------------------------------------- analysis-result layers
# add_mosaic makes IMAGE layers. A segmentation operator's result is not an image: it is a
# Labels mask and a Points set (that is what every napari segmentation plugin returns, and what
# Cellpose will return when it lands). These are the siblings that carry those.


def _labels(n=3, shape=(32, 32)):
    lab = np.zeros(shape, dtype=np.int32)
    for i in range(1, n + 1):
        lab[i * 4: i * 4 + 3, i * 4: i * 4 + 3] = i
    return lab


def test_a_mask_lands_as_a_real_napari_Labels_layer_not_an_image(layers):
    """A Labels layer is what gives napari its label colormap, pick-by-click and 0-transparency.
    Adding a mask with add_image would render it as a near-black gradient and read as broken."""
    from napari.layers import Labels

    lyr = layers.add_labels("spots", "488 mask", _labels())
    assert isinstance(lyr, Labels)
    assert lyr in layers.model.layers


def test_centroids_land_as_a_real_napari_Points_layer(layers):
    from napari.layers import Points

    pts = np.array([[4.0, 4.0], [8.0, 8.0]])
    lyr = layers.add_points("spots", "488 centroids", pts)
    assert isinstance(lyr, Points)
    assert np.asarray(lyr.data).shape == (2, 2)


def test_analysis_layers_carry_our_metadata_so_the_layer_tree_groups_them(layers):
    """Identity lives in metadata, never parsed back out of the name (module docstring)."""
    layers.add_labels("spots", "488 mask", _labels())
    layers.add_points("spots", "488 centroids", np.array([[1.0, 2.0]]))

    assert key_of(layers.model.layers["spots · 488 mask"]) == MosaicKey("spots", "488 mask")
    assert set(layers.channels("spots")) == {"488 mask", "488 centroids"}
    assert "spots" in layers.ops()


def test_an_empty_points_layer_is_still_added_so_zero_nuclei_is_visible_as_a_result(layers):
    """Zero found is an ANSWER. Skipping the layer would make 'nothing ran' and 'nothing there'
    look identical — a silent failure."""
    from napari.layers import Points

    lyr = layers.add_points("spots", "488 centroids", np.zeros((0, 2)))
    assert isinstance(lyr, Points)
    assert len(np.asarray(lyr.data)) == 0


def test_analysis_layers_are_NOT_linked_into_the_per_channel_contrast_group(layers):
    """A Labels/Points layer has no contrast_limits. Registering it as a contrast peer would
    make link_layers raise on the next channel added — and napari OWNS contrast, not us."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_labels("spots", "488", _labels())          # same channel string, on purpose

    peers = layers._by_channel["488"]
    assert len(peers) == 1, "an analysis layer was registered as a contrast peer"
    layers.set_contrast("488", 10.0, 900.0)               # must not raise


def test_analysis_layers_are_placed_in_stage_micrometres_like_the_mosaic_they_describe(layers):
    """The mask must land ON the mosaic. Same bbox -> same scale/translate, or the overlay is
    a plausible-looking lie sitting somewhere else in world space."""
    img = layers.add_mosaic("raw", "488", np.zeros((40, 80), dtype=np.uint16),
                            bbox_um=(100.0, 200.0, 900.0, 600.0))
    mask = layers.add_labels("spots", "488 mask", np.zeros((40, 80), dtype=np.int32),
                             bbox_um=(100.0, 200.0, 900.0, 600.0))
    pts = layers.add_points("spots", "488 centroids", np.array([[20.0, 40.0]]),
                            bbox_um=(100.0, 200.0, 900.0, 600.0), shape=(40, 80))

    assert tuple(mask.scale) == pytest.approx(tuple(img.scale))
    assert tuple(mask.translate) == pytest.approx(tuple(img.translate))
    assert tuple(pts.scale) == pytest.approx(tuple(img.scale))
    assert tuple(pts.translate) == pytest.approx(tuple(img.translate))


def test_points_without_a_shape_cannot_be_placed_and_says_so(layers):
    """A Points layer carries no array shape, so bbox placement needs the mask's shape passed in.
    Silently leaving it unplaced would put every centroid at the world origin."""
    with pytest.raises(ValueError, match="shape"):
        layers.add_points("spots", "488 centroids", np.array([[1.0, 2.0]]),
                          bbox_um=(0.0, 0.0, 10.0, 10.0))


def test_re_running_replaces_the_previous_result_instead_of_stacking_duplicates(layers):
    layers.add_labels("spots", "488 mask", _labels(n=3))
    layers.add_labels("spots", "488 mask", _labels(n=5))
    assert len(layers.group("spots")) == 1
    assert int(np.asarray(layers.find("spots", "488 mask").data).max()) == 5


def test_remove_op_clears_masks_and_points_together(layers):
    layers.add_labels("spots", "488 mask", _labels())
    layers.add_points("spots", "488 centroids", np.array([[1.0, 2.0]]))
    gone = layers.remove_op("spots")

    assert set(gone) == {"488 mask", "488 centroids"}
    assert layers.group("spots") == []


def test_the_count_rides_on_the_points_layer_features_keyed_by_label(layers):
    """Fractal's feature-table contract — one row per object, indexed by label value — expressed
    in the shape napari already has. Neither cellpose-napari nor napari-sbatwm surfaces a count
    at all; Spencer asked for one, so it has to live somewhere addressable."""
    pts = np.array([[4.0, 4.0], [8.0, 8.0], [12.0, 12.0]])
    lyr = layers.add_points("spots", "488 centroids", pts,
                            features={"label": [1, 2, 3]})
    assert list(lyr.features["label"]) == [1, 2, 3]
    assert len(lyr.features) == 3


# --- the PROCESSING LAYER fan-out (Julio, 2026-08-03) ------------------------------------------
#
# "After I click an operator layer in our window, the thumbnails don't update." The state was
# already here -- `visible_op` could always answer -- and no signal was ever emitted for it, so
# the plate could not follow. These pin the producer half; tests/test_plate_follows_windows.py
# pins the sink.


def test_showing_a_processing_layer_is_reported_once_for_the_whole_group(layers):
    seen = []
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("raw", "561", _img(1))
    layers.add_mosaic("mip", "488", _img(2), visible=False)
    layers.add_mosaic("mip", "561", _img(3), visible=False)
    layers.on_user_op(lambda op, on: seen.append((op, on)))

    for ly in layers.group("mip"):        # the group checkbox, which writes every child
        ly.visible = True

    # ONE delivery per op, and the cause is reported before its consequence: the user lit mip,
    # and raw went dark BECAUSE of that (two additive operators of one channel would sum). The
    # group toggle writes four layers and must not arrive as four "mip" deliveries.
    assert seen == [("mip", True), ("raw", False)], f"one group toggle reported as {seen}"


def test_switching_operator_is_reported_by_the_op_tap_and_not_by_the_channel_tap(layers):
    """The exact swallow. ``on_user_visibility`` answers "is this CHANNEL on screen anywhere",
    which is unchanged across an operator switch -- 488 is up before and after -- so it returns
    early and the plate hears nothing. The op fan-out has to be a separate tap for that reason.

    The gesture is the SWITCH rather than a bare hide, because at most one operator per channel is
    lit at a time now (``_connect_exclusive_op``): "raw and mip both showing 488" is the summing
    state the exclusivity exists to prevent, so it is not a state a test may start from.
    """
    channel_seen, op_seen = [], []
    layers.add_mosaic("raw", "405", _img())
    layers.add_mosaic("raw", "488", _img(1))
    layers.add_mosaic("mip", "488", _img(2), visible=False)
    layers.on_user_visibility(lambda ch, on: channel_seen.append((ch, on)))
    layers.on_user_op(lambda op, on: op_seen.append((op, on)))

    layers.find("mip", "488").visible = True      # -> raw · 488 goes dark, 488 stays on screen

    assert channel_seen == [], (
        f"488 is on screen before and after the switch, so the channel tap has nothing to say and "
        f"the plate must not be told its channel changed: {channel_seen}")
    assert op_seen == [("mip", True)], (
        f"the processing layer change was swallowed: {op_seen}")
    assert layers.find("raw", "488").visible is False, "488 is being drawn twice, additively"
    assert layers.find("raw", "405").visible is True, (
        "darkening raw · 488 took raw · 405 with it -- the rule is per CHANNEL, and 405 has no "
        "second copy on screen to sum with")


def test_our_own_writes_are_never_reported_as_a_user_picking_a_layer(layers):
    """A result delivered to a window that did NOT ask arrives ``visible=False``. Reporting that
    as a gesture would turn the plate's layer off behind the user (``_deliver_to_views``)."""
    seen = []
    layers.add_mosaic("raw", "488", _img())
    layers.on_user_op(lambda op, on: seen.append((op, on)))

    layers.add_mosaic("mip", "488", _img(1), visible=False)
    layers.add_mosaic("mip", "488", _img(2), visible=True)      # a re-add, still ours

    assert seen == [], f"a programmatic write was reported as a user gesture: {seen}"


# --- one hex per channel, or an honest None --------------------------------------------------
#
# Minerva's story groups carry a single six-digit "color" per channel (_minerva.auto_groups) and
# have no second field for a gradient. colormap_hue_rgb decides which napari colormaps survive
# that and which cannot, and the answer must never be "approximate it".

def _hue_layer(viewer_layers, cmap):
    viewer_layers.add_mosaic("raw", "x", _img())
    layer = viewer_layers.find("raw", "x")
    layer.colormap = cmap
    return layer


@pytest.mark.parametrize("name,expected", [
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
    ("cyan", (0, 255, 255)),
    ("magenta", (255, 0, 255)),
    ("yellow", (255, 255, 0)),
    ("gray", (255, 255, 255)),
])
def test_a_single_hue_colormap_reduces_to_exactly_its_hue(layers, name, expected):
    """napari's own black-to-hue maps are the ones this app and its users actually pick."""
    from squidmip._napari_view import colormap_hue_rgb
    assert colormap_hue_rgb(_hue_layer(layers, name)) == expected


def test_the_colormap_this_app_builds_for_a_channel_reduces_to_its_palette_colour(layers):
    """_napari_pane._colormap_for builds [[0,0,0,1], [r,g,b,1]] from Squid's palette. If that did
    NOT reduce, the common case would silently fall back and the feature would do nothing."""
    from napari.utils import Colormap

    from squidmip._napari_view import colormap_hue_rgb

    cmap = Colormap([[0.0, 0.0, 0.0, 1.0], [0x1F / 255, 1.0, 0.0, 1.0]], name="squid-488")
    assert colormap_hue_rgb(_hue_layer(layers, cmap)) == (0x1F, 255, 0)


@pytest.mark.parametrize("name", ["viridis", "turbo", "inferno", "PiYG"])
def test_a_multi_stop_colormap_refuses_rather_than_approximating(layers, name):
    """THE case that must not be fudged. A perceptual map's last stop is the top of a ramp, not
    the map: viridis ends yellow and is mostly not yellow. Emitting that stop would put a colour
    into Minerva that is on no screen, so this returns None and the caller keeps what it had."""
    from squidmip._napari_view import colormap_hue_rgb
    try:
        layer = _hue_layer(layers, name)
    except (KeyError, ValueError):
        pytest.skip(f"this napari has no {name!r} colormap")
    assert colormap_hue_rgb(layer) is None


def test_a_colormap_ending_in_black_has_no_hue_to_name(layers):
    """The degenerate case: every row is a multiple of a zero vector, so the projection test would
    divide by zero and, unguarded, would call black a valid hue for anything."""
    from napari.utils import Colormap

    from squidmip._napari_view import colormap_hue_rgb

    cmap = Colormap([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]], name="inverted")
    assert colormap_hue_rgb(_hue_layer(layers, cmap)) is None


def test_a_layer_with_no_colormap_table_says_so_instead_of_guessing(layers):
    from squidmip._napari_view import colormap_hue_rgb
    assert colormap_hue_rgb(object()) is None


# ==============================================================================================
# A Z-REDUCED RESULT IS PRESENTED WITHOUT A Z AXIS
#
# Julio, from the running GUI: "the MIP layer doesn't collapse the z-level axis inside napari."
# docs/DESIGN.md:188 already states the rule ("z slider hidden when nz is 1 (a z reduced result)")
# and docs/rendering-contract.md:13 states the producer half of it.
#
# WHAT WAS ACTUALLY WRONG, measured before the fix rather than assumed. The MIP layer is ALREADY
# 2-D when it arrives: `project_well` returns (T, C, 1, Y, X) for a z-reducer, `_workers._on_well`
# takes `image[0, :, 0]`, and the fused plane handed to `add_result` is (Y, X). Nothing downstream
# keeps a singleton z. The slider belongs to the RAW pyramid sharing the pane -- `(nz, y, x)`
# levels -- and napari derives `dims.ndim` from the MAXIMUM over EVERY layer, VISIBLE OR NOT. So
# delivering the MIP darkened raw and left raw's slider standing over a picture it could not move.
#
# These drive a bare `ViewerModel`, so they pin the rule rather than a widget.

def _z_stack_pyramid(nz=10, shape=(64, 64)):
    """A raw mosaic the way `_mosaic_source.fuse_region_pyramid` hands it over: (z, y, x) levels."""
    h, w = shape
    return [np.zeros((nz, h, w), np.uint16), np.zeros((nz, h // 2, w // 2), np.uint16)]


_Z_BBOX = (0.0, 0.0, 100.0, 200.0)


def test_a_z_reducers_result_takes_the_z_axis_off_the_pane(layers):
    """THE reported defect, in one assertion."""
    layers.add_mosaic("raw", "405", _z_stack_pyramid(), multiscale=True,
                      bbox_um=_Z_BBOX, z_scale_um=2.0)
    assert layers.model.dims.ndim == 3, "the fixture must have a z axis, or this proves nothing"

    layers.add_result("intensity", "mip", "405", np.zeros((64, 64), np.uint16),
                      bbox_um=_Z_BBOX, visible=True)

    assert layers.model.dims.ndim == 2, (
        "the pane still carries a z axis while a z-reduced result is what is on screen: napari "
        "takes dims.ndim from every layer, visible or not, so raw's stack keeps the slider alive "
        "over a picture it cannot move")


def test_the_z_axis_comes_straight_back_when_raw_is_shown_again(layers):
    """The toggle is a comparison, so it has to be reversible -- pyramid, scale and translate."""
    raw = layers.add_mosaic("raw", "405", _z_stack_pyramid(), multiscale=True,
                            bbox_um=_Z_BBOX, z_scale_um=2.0)
    placed = tuple(raw.scale), tuple(raw.translate)
    layers.add_result("intensity", "mip", "405", np.zeros((64, 64), np.uint16),
                      bbox_um=_Z_BBOX, visible=True)

    layers.find("raw", "405").visible = True

    assert layers.model.dims.ndim == 3, "z browsing did not come back with raw"
    assert raw.multiscale is True, "raw came back as a single scale: the pyramid was lost"
    assert [tuple(np.asarray(lv).shape) for lv in raw.data] == [(10, 64, 64), (10, 32, 32)]
    assert (tuple(raw.scale), tuple(raw.translate)) == placed, (
        "the z step was rebuilt as 1.0 rather than restored, so an anisotropic stack would render "
        "isotropically -- the defect `_place`'s z scale exists to prevent")


def test_a_layer_a_z_reducer_never_touched_is_not_collapsed(layers):
    """The rule is scoped to the pane's own mosaics and to a REAL z axis: a 2-D layer has nothing
    to stash, so restoring it later must not be able to invent a stack."""
    flat = layers.add_mosaic("raw", "405", np.zeros((64, 64), np.uint16), bbox_um=_Z_BBOX)
    layers.add_result("intensity", "mip", "405", np.zeros((64, 64), np.uint16),
                      bbox_um=_Z_BBOX, visible=True)
    layers.find("raw", "405").visible = True

    assert flat.ndim == 2 and np.asarray(flat.data).shape == (64, 64)


def test_a_plane_op_result_KEEPS_the_z_axis(layers):
    """The other half, and the one an ``op == "mip"`` branch would get wrong.

    A plane-op declares ``consumes=frozenset()`` -- z survives at full depth -- so its result says
    nothing about whether the pane should show a z axis, and raw's stack must stay browsable under
    it. The distinction is read off the operator's DECLARATION; ``tests/test_operator_declaration``
    fails the build on a comparison against an operator's name for exactly this reason.
    """
    from squidmip import add_projector, available_projectors, plane_op

    if "zaxis_plane_op" not in available_projectors():
        add_projector("zaxis_plane_op", plane_op(lambda a: a))
    layers.add_mosaic("raw", "405", _z_stack_pyramid(), multiscale=True,
                      bbox_um=_Z_BBOX, z_scale_um=2.0)

    layers.add_result("intensity", "zaxis_plane_op", "405", np.zeros((64, 64), np.uint16),
                      bbox_um=_Z_BBOX, visible=True)

    assert layers.model.dims.ndim == 3, (
        "a plane-op's result took the z axis away; only an operator declaring consumes={'z'} may")


def test_the_rule_reads_the_declaration_through_a_namespaced_layer_key(layers):
    """A namespaced layer key (``"mip@tab2"``) is in no registry — the rule must strip it first."""
    layers.add_mosaic("raw", "405", _z_stack_pyramid(), multiscale=True,
                      bbox_um=_Z_BBOX, z_scale_um=2.0)

    layers.add_result("intensity", "mip@tab2", "405", np.zeros((64, 64), np.uint16),
                      bbox_um=_Z_BBOX, visible=True)

    assert layers.model.dims.ndim == 2, (
        "the scoped layer key was handed to the registry unsplit, so the declaration behind it "
        "could not be read and the z axis stayed")


def test_show_op_moves_the_z_axis_too(layers):
    """`show_op` is the before/after toggle's other entrance; one rule, every surface."""
    layers.add_mosaic("raw", "405", _z_stack_pyramid(), multiscale=True,
                      bbox_um=_Z_BBOX, z_scale_um=2.0)
    layers.add_result("intensity", "mip", "405", np.zeros((64, 64), np.uint16),
                      bbox_um=_Z_BBOX, visible=False)
    assert layers.model.dims.ndim == 3

    layers.show_op("mip")
    assert layers.model.dims.ndim == 2

    layers.show_op("raw")
    assert layers.model.dims.ndim == 3


# ------------------------------------------- a layer's data is NOT the list that went in
#
# napari wraps a multiscale layer's data in `napari.layers._multiscale_data.MultiScaleData`, a
# Sequence of levels that is neither a list nor a tuple. Every `isinstance(data, (list, tuple))`
# pyramid check in this codebase was therefore False for EVERY pyramid in the app, and each one
# failed differently: `_workers._full_res_mip` raised `AttributeError: 'MultiScaleData' object has
# no attribute 'max'` (the crash Julio hit running cellpose), while the two below failed silently.
# `_napari_view.pyramid_levels` is the one rule they all read now.


def test_the_3d_swap_actually_swaps_a_pyramid_napari_handed_back(layers):
    """``render_max_res_3d`` was a SILENT NO-OP on every real mosaic.

    Its first line was ``if not isinstance(ly.data, (list, tuple)): return`` -- "already
    single-scale, nothing to swap" -- and a layer built with ``multiscale=True`` never satisfies
    it, so the method returned before touching anything and napari went on dropping the layer to
    its COARSEST level in 3D. That blocky volume is the exact thing this method exists to prevent.

    MUTATION: put the isinstance check back -> the layer stays multiscale -> red.
    """
    raw = layers.add_mosaic("raw", "405", _z_stack_pyramid(), multiscale=True,
                            bbox_um=_Z_BBOX, z_scale_um=2.0)
    assert not isinstance(raw.data, (list, tuple)), (
        f"napari returned a plain {type(raw.data).__name__}; this test no longer covers the "
        "container production sees")

    layers.render_max_res_3d(True)

    assert raw.multiscale is False, "the 3D swap never ran: the layer is still multiscale"
    assert np.asarray(raw.data).shape == (10, 64, 64), (
        "3D got a coarser level than the texture budget allows")

    layers.render_max_res_3d(False)
    assert raw.multiscale is True, "the pyramid did not come back for 2D"
    assert [tuple(np.asarray(lv).shape) for lv in raw.data] == [(10, 64, 64), (10, 32, 32)]


def test_full_res_level_takes_level_zero_off_napari_s_own_container():
    """``np.asarray(MultiScaleData)`` is the COARSEST level -- ``__array__`` returns ``_data[-1]``.

    So "it is not a list, treat it as an array" is not a harmless miss: it is a silent
    substitution of the smallest picture for the largest one.

    MUTATION: make ``pyramid_levels`` return None for a non-list Sequence -> the coarsest level
    comes back -> red.
    """
    from napari.components import ViewerModel

    from squidmip._napari_view import full_res_level, pyramid_levels

    lv0 = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    data = ViewerModel().add_image([lv0, lv0[::2, ::2], lv0[::4, ::4]], multiscale=True).data

    assert np.asarray(data).shape == (16, 16), (
        "napari no longer coerces a pyramid to its coarsest level; this test's premise is stale")
    assert [tuple(lv.shape) for lv in pyramid_levels(data)] == [(64, 64), (32, 32), (16, 16)]
    assert np.array_equal(np.asarray(full_res_level(data)), lv0)


def test_a_plain_array_and_a_nested_list_are_not_pyramids():
    """The discriminator is the ELEMENT, not the container: ``[[1, 2], [3, 4]]`` encodes ONE array
    and must not be read as two levels."""
    from squidmip._napari_view import full_res_level, pyramid_levels

    arr = np.zeros((4, 4), np.uint16)
    assert pyramid_levels(arr) is None
    assert full_res_level(arr) is arr
    assert pyramid_levels([[1, 2], [3, 4]]) is None
    assert pyramid_levels("not an image at all") is None
    with pytest.raises(ValueError, match="EMPTY multiscale"):
        pyramid_levels([])
