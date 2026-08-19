"""napari mosaic view — the processing-layer/channel hierarchy and the binding guards.

These tests use ``napari.components.ViewerModel``, which is Qt-free, so the hierarchy is
exercised headless with no canvas, no display and no Qt binding conflict. Only the embedding
test needs Qt, and it skips itself when Qt is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import qtpy

from squidxplorer._napari_view import (
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
    assert resolve_viewer({"SQUIDXPLORER_VIEWER": ""}) == "napari"
    assert napari_enabled({}) is True


def test_a_retired_ndviewer_name_still_builds_napari_and_says_so(caplog):
    """No fallback exists anymore; asking for the retired ndviewer by name must build napari
    AND log why, rather than silently substituting a different viewer than the one named."""
    import logging

    for spelling in ("ndv", "ndviewer", "ndviewer_light", "  NDV  "):
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            assert resolve_viewer({"SQUIDXPLORER_VIEWER": spelling}) == "napari", spelling
        assert "deleted" in caplog.text, f"{spelling!r} was retired in silence"
    assert napari_enabled({"SQUIDXPLORER_VIEWER": "ndv"}) is True


def test_a_typo_does_not_silently_cost_you_the_viewer():
    assert resolve_viewer({"SQUIDXPLORER_VIEWER": "napri"}) == "napari"


def test_one_resolver_decides_so_the_pane_cannot_disagree_with_the_model():
    """Two readers of one environment variable is how controls end up disagreeing about what
    is on screen; ``make_pane`` must ask ``resolve_viewer`` rather than parse it again."""
    import inspect

    from squidxplorer import _napari_pane

    src = inspect.getsource(_napari_pane.make_pane)
    assert "resolve_viewer" in src, "make_pane does not ask the one resolver"
    assert "os.environ" not in src, (
        "make_pane reads the environment itself instead of asking resolve_viewer — "
        "two readers of one variable is how controls end up disagreeing"
    )


# ------------------------------------------------- identity lives in metadata


def test_identity_is_read_from_metadata_not_parsed_out_of_the_name(layers):
    """The name is a label; identity never comes from parsing it (petakit's channel names
    have defeated naive regex parsers before)."""
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
# Every mosaic is added ``additive`` — correct ACROSS channels (405+488+561+638 summing is the
# composite) but arithmetic nonsense across OPERATORS of one channel (raw·488 and mip·488 lit
# together double-counts that channel's signal). A blending mode cannot draw that distinction
# (napari blends flat against the whole stack beneath), so exclusivity is enforced on
# ``layer.visible`` itself, per channel, so both the layer tree and napari's own eye icons obey it.


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
    """The other half, which must NOT change: four channels of one operator are a composite."""
    for ch in ("405", "488", "561", "638"):
        layers.add_mosaic("raw", ch, _img())

    assert [ly.visible for ly in layers.group("raw")] == [True] * 4
    assert {str(ly.blending) for ly in layers.group("raw")} == {"additive"}


def test_an_operator_result_arrives_instead_of_raw_rather_than_on_top_of_it(layers):
    """The delivery path (``deliver_result``) adds a visible operator layer over a lit raw, so
    the sum exists before the user has touched anything — same defect, no gesture."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("raw", "561", _img(1))

    layers.add_mosaic("mip", "488", _img(2))          # the result, as delivered

    assert layers.find("raw", "488").visible is False
    assert layers.find("raw", "561").visible is True, (
        "561 has no second copy on screen, so nothing about it had to change")


def test_a_result_delivered_dark_darkens_nothing(layers):
    """A window that did not ask gets ``visible=False``; a layer off screen cannot be summing
    with anything and must not take raw down with it."""
    layers.add_mosaic("raw", "488", _img())

    layers.add_mosaic("mip", "488", _img(1), visible=False)

    assert layers.find("raw", "488").visible is True


def test_hiding_an_operator_lights_nothing(layers):
    """Only turning a layer ON may force another off; a checkbox going dark that turns
    something else on is a control moving a second control."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("mip", "488", _img(1))         # raw · 488 is now dark

    layers.find("mip", "488").visible = False

    assert layers.find("raw", "488").visible is False
    assert layers.visible_op() is None


def test_an_analysis_overlay_is_never_darkened_by_the_mosaic_it_is_drawn_over(layers):
    """Labels/Points layers skip ``_register_channel`` (no ``contrast_limits`` to link), so
    they never enter ``_by_channel`` and exclusivity never sees them."""
    layers.add_mosaic("raw", "488", _img())
    mask = layers.add_labels("spots", "488 mask", np.zeros((32, 32), dtype="uint32"))

    layers.add_mosaic("mip", "488", _img(1))         # a switch under the overlay

    assert mask.visible is True, "the spot mask went dark when the mosaic under it was switched"
    assert layers.find("raw", "488").visible is False


# --------------------------------- contrast: ONE value per channel, no duplication


def test_channel_contrast_survives_the_before_after_toggle(layers):
    """The whole point of linking per channel: a second control for the same channel must not
    be able to disagree."""
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
    """``link_layers`` connects events, not values: a freshly added operator layer keeps its
    own seeded window until something writes it. Deliberate — a decon result must be legible
    on its own terms — but had been documented as the opposite, so it is pinned here."""
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
    """The explicit opt-in that turns the raw->operator flip into a real comparison, asserted
    on the layers' actual contrast values across two channels and two operators."""
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
    """A channel with no raw layer is left alone, rather than matched to a wrong window."""
    layers.add_mosaic("raw", "488", np.full((32, 32), 300, dtype=np.uint16))
    layers.add_mosaic("decon", "488", np.full((32, 32), 6000, dtype=np.uint16))
    only_op = layers.add_mosaic("decon", "561", np.full((32, 32), 12000, dtype=np.uint16))
    before = list(only_op.contrast_limits)

    assert layers.match_contrast_to("raw") == 1
    assert list(only_op.contrast_limits) == before


def test_neither_the_link_nor_the_match_carries_the_COLORMAP(layers):
    """Contrast is shared; COLOUR is not, by either mechanism — the link is bound only to
    ``contrast_limits`` and ``match_contrast_to`` writes only that, so the LUT copy/paste chip
    (the app's only writer of ``layer.colormap``) is not made redundant by either."""
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
    """Replaces the ndv contrast tap, which subclassed a private LutView."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("stitched", "488", _img())

    seen = []
    layers.on_contrast_changed(lambda e: seen.append(True))
    layers.set_contrast("488", 50, 5000)

    assert seen, "layer.events.contrast_limits did not fire"


def test_a_degenerate_window_is_not_widened(layers):
    """``_pct_window`` returns hi <= lo for a blank channel on purpose; widening it to
    (lo, lo+1) would render a blank channel as full white, i.e. as signal."""
    lyr = layers.add_mosaic("raw", "488", _img(), contrast_limits=(500.0, 500.0))
    assert list(lyr.contrast_limits) != [500.0, 501.0]


# ------------------------------------------- the slider's TRAVEL (contrast_limits_range)


@pytest.fixture
def twelve_bit():
    """A dataset measured at 3437 -- the 14-bit set's region C3, which alone reads as 12-bit."""
    from squidxplorer import _bitdepth

    _bitdepth.new_dataset(np.uint16)
    _bitdepth.depth().observe(3437.0)
    return _bitdepth.depth()


def test_a_mosaic_layer_opens_on_the_DATASETS_range_not_the_seeded_window(layers, twelve_bit):
    """The seed says where to look; the range says how far the user may drag. Different jobs."""
    lyr = layers.add_mosaic("raw", "488", _img(), contrast_limits=(500.0, 900.0))

    assert list(lyr.contrast_limits) == [500.0, 900.0]
    assert list(lyr.contrast_limits_range) == [0.0, 4095.0]


def test_a_layer_with_NO_seed_still_gets_the_datasets_range(layers, twelve_bit):
    """The bug this change removes: the range used to be set only when a window was seeded, so a
    blank or degenerate channel was left pinned to whatever extent napari inferred by itself."""
    lyr = layers.add_mosaic("raw", "488", _img())

    assert list(lyr.contrast_limits_range) == [0.0, 4095.0]


def test_a_degenerate_seed_still_gets_the_datasets_range(layers, twelve_bit):
    """`hi <= lo` is dropped as a WINDOW on purpose; that must not also drop the RANGE."""
    lyr = layers.add_mosaic("raw", "488", _img(), contrast_limits=(500.0, 500.0))

    assert list(lyr.contrast_limits_range) == [0.0, 4095.0]


def test_a_window_wider_than_the_measured_depth_is_never_clamped_away(layers, twelve_bit):
    """A range narrower than the window on screen would let napari clip the window itself."""
    lyr = layers.add_mosaic("raw", "488", _img(), contrast_limits=(0.0, 50000.0))

    assert list(lyr.contrast_limits) == [0.0, 50000.0]
    assert lyr.contrast_limits_range[1] >= 50000.0


def test_widening_the_range_does_not_move_a_single_contrast_limit(layers, twelve_bit):
    """The anti-flash assertion: opening the slider's travel changes no pixel on the canvas."""
    lyr = layers.add_mosaic("raw", "488", _img(), contrast_limits=(100.0, 3000.0))

    assert layers.widen_contrast_range(0.0, 16383.0) == 1

    assert list(lyr.contrast_limits) == [100.0, 3000.0]
    assert list(lyr.contrast_limits_range) == [0.0, 16383.0]


def test_widening_NEVER_narrows(layers, twelve_bit):
    """A narrower range does not merely restyle the slider -- napari clips the window into it."""
    lyr = layers.add_mosaic("raw", "488", _img(), contrast_limits=(100.0, 3000.0))
    layers.widen_contrast_range(0.0, 16383.0)

    assert layers.widen_contrast_range(0.0, 4095.0) == 0
    assert list(lyr.contrast_limits_range) == [0.0, 16383.0]


def test_widening_the_range_is_not_reported_as_a_USER_gesture(layers, twelve_bit):
    """napari re-emits contrast_limits when the RANGE moves, even though the value does not.

    An echo that reaches the user-contrast tap latches the plate to manual and kills per-region
    contrast, which is what `programmatic()` exists to prevent.
    """
    layers.add_mosaic("raw", "488", _img(), contrast_limits=(100.0, 3000.0))
    seen = []
    layers.on_user_contrast(lambda *a: seen.append(a))

    layers.widen_contrast_range(0.0, 16383.0)

    assert seen == []


def test_widening_skips_the_layers_that_have_no_contrast_at_all(layers, twelve_bit):
    """Labels have no contrast_limits_range; a walk over the layer list must not care."""
    layers.add_mosaic("raw", "488", _img(), contrast_limits=(100.0, 3000.0))
    layers.add_labels("nuclei", "488", np.zeros((32, 32), dtype=np.uint32))

    assert layers.widen_contrast_range(0.0, 16383.0) == 1        # the image only


def test_a_float_result_layer_keeps_its_own_range(layers, twelve_bit):
    """The gate is the LAYER's dtype. A float operator result has no bit depth to apply one to."""
    lyr = layers.add_mosaic("flatfield", "488",
                            np.zeros((32, 32), dtype=np.float32), contrast_limits=(0.0, 1.0))

    assert list(lyr.contrast_limits_range) == [0.0, 1.0]


def test_a_region_change_re_widens_the_slider_without_moving_the_window(layers, twelve_bit):
    """The `_reuse_layer` path: C3 replaced by E7, which holds numbers C3's slider cannot reach."""
    from squidxplorer import _bitdepth

    lyr = layers.add_mosaic("raw", "488", _img(), contrast_limits=(100.0, 3000.0))
    assert list(lyr.contrast_limits_range) == [0.0, 4095.0]

    _bitdepth.depth().observe(16380.0)                       # E7 is read
    same = layers.add_mosaic("raw", "488", _img(seed=1))     # same identity -> reuse

    assert same is lyr
    assert list(lyr.contrast_limits) == [100.0, 3000.0]      # the user's window is untouched
    assert list(lyr.contrast_limits_range) == [0.0, 16383.0]


# ------------------------------------------------------- placement from stage µm


def test_bbox_um_maps_onto_napari_scale_and_translate_with_the_axis_flip():
    """``_tiling`` speaks (x0, y0, x1, y1); napari speaks (row, col) = (y, x)."""
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
# The three tests below are one claim: opening a region window must pull each channel's
# mosaic ONCE, not four times. Counting tests, not timing tests — the cost is decided by the
# ORDER of the calls in ``add_mosaic``, which a later edit could rearrange without noticing.


def _pyramid(nz=10, pulls=None):
    """A lazy ``(z, y, x)`` pyramid built the way ``fuse_region_pyramid`` builds one — one dask
    block per z per level, recording the z index when computed — so the pull count here is the
    count of whole-region fuses a real open would pay (a plain ndarray would count nothing)."""
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
    """The seed and the display must be the SAME plane. They disagreed: ``sample_plane`` took
    ``nz // 2`` while napari centres on ``(nz - 1) // 2``, costing 27 extra whole-frame decodes
    per channel on the real 10x region. Driven through a real ``ViewerModel`` since the claim
    is about what napari does."""
    from squidxplorer._contrast import opening_z

    for nz in (10, 9, 2):
        mdl = MosaicLayers(type(layers._model)())
        data, _pulls = _pyramid(nz=nz)
        mdl.add_mosaic("raw", "488", data, multiscale=True,
                       bbox_um=(0.0, 0.0, 640.0, 640.0), z_scale_um=1.5)

        assert mdl._model.dims.current_step[0] == opening_z(nz), (
            f"nz={nz}: napari displays plane {mdl._model.dims.current_step[0]} but the contrast "
            f"seed samples plane {opening_z(nz)}")


def test_adding_a_placed_mosaic_pulls_two_z_not_four(layers):
    """Measured on the real acquisition before this fix: FOUR whole-region decode passes per
    channel (seed's z, napari's z=0 slice, napari's centred slice, and a fourth because
    ``_place`` set ``layer.scale`` after the layer existed, moving the slider again) — 432
    frame decodes for a window that needs 216. Two remain; napari's own constructor slice at
    z=0 is not fixable from outside napari."""
    from squidxplorer._contrast import opening_z

    data, pulls = _pyramid(nz=10)
    layers.add_mosaic("raw", "488", data, multiscale=True,
                      bbox_um=(0.0, 0.0, 640.0, 640.0), z_scale_um=1.5)

    assert set(pulls) == {0, opening_z(10)}, (
        f"a mosaic add materialised planes {sorted(set(pulls))}; every plane beyond "
        f"{{0, {opening_z(10)}}} is a whole region decoded and thrown away")


def test_placing_at_construction_puts_the_layer_exactly_where_placing_it_after_would(layers):
    """The saving must not move the picture: ``placement_for`` is used both as ``add_image``
    kwargs and by ``_place``, and two placement rules disagreeing is exactly what ``_place``'s
    docstring exists to prevent. Units are checked too, since they used to be lost."""
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
    """Moving WHERE placement runs (into ``add_image``) must not change WHETHER it runs: a
    silently unplaced mosaic sits at scale 1 in a µm world and reads as a rendering bug."""
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
    """MUTATION TEST: this project lost a day to a symbol that bound cleanly but did nothing
    (vispy's ``Visual.freeze()`` made the assignment raise into a bare ``except: pass``), so
    the guard itself must be provably able to fail."""
    import napari.qt

    class _Renamed:
        # QtViewer has been renamed away; everything else still looks fine.
        __all__ = ("NotQtViewer",)
        NotQtViewer = object

    with pytest.raises(NapariBindingError) as exc:
        verify_napari_bindings(modules={"napari.qt": _Renamed})

    assert "napari.qt.QtViewer" in str(exc.value)


def test_binding_check_bites_on_a_quiet_de_export():
    """A name that still exists but has left ``__all__`` is a deprecation in progress — catch
    it while it is still only a warning."""

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


# Building a real vispy GL canvas in-process under pytest aborts the interpreter (a Qt-binding
# conflict with the PySide6 napari/pytest already loaded, not a code bug), so the check runs in
# a clean subprocess where a crash is a test failure rather than a dead session.

_EMBED_SCRIPT = r"""
import json, os, sys, traceback
# Deliberately NOT forcing QT_QPA_PLATFORM=offscreen: the offscreen plugin ships no GL, so a
# vispy canvas segfaults under it. Runs for real on a machine with a display; fails cleanly
# with the reason attached on a headless one.
import numpy as np
# PyQt5 explicitly, QT_API pinned before any qtpy import: squidxplorer imports PyQt5 directly
# while qtpy here defaults to PySide6, and loading both aborts the process.
os.environ.setdefault("QT_API", "pyqt5")
from qtpy.QtWidgets import QApplication, QHBoxLayout, QWidget
app = QApplication.instance() or QApplication([])

# Report OUR OWN errors as EMBEDFAIL, distinct from "this box has no GL" — an unhandled
# exception here used to print no EMBED line and read as a skip, hiding a real bug.
try:
    from squidxplorer._napari_pane import MosaicPane

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
        # QMainWindow. Reparenting it out left the window gutted — docks/controls still showed.
        "canvas_still_inside_napari_window": (
            descends_from(canvas, win) if (win is not None and canvas is not None) else False
        ),
        "central_is_not_empty": (
            len(central.findChildren(QWidget)) > 0 if central is not None else False
        ),
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
    """``MosaicPane`` used to call ``canvas.setParent(self)``, ripping the QtViewer out of
    napari's own window; the gutted window then embedded fine (docks/controls present) while
    the canvas painted nothing anywhere — a silent-failure shape (absence read as fine), so
    this asserts the STRUCTURE, not the appearance."""
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
    # The offscreen plugin has no GL, so inheriting the suite's QT_QPA_PLATFORM guarantees a
    # segfault; drop it and let Qt pick the real platform.
    env.pop("QT_QPA_PLATFORM", None)
    # Pin the child to the parent's binding, or qtpy defaults to PySide6 here and two bindings
    # in one process abort before any assertion runs.
    env["QT_API"] = os.environ.get("QT_API") or qtpy.API_NAME.lower()

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
    assert got["layer_controls"] >= 1
    assert got["ops"] == ["raw"]


def test_channels_composite_additively_not_occluding_each_other():
    """napari defaults every layer to ``translucent``, so the last-added layer occludes the
    rest; on the 10x set the last channel (638 nm, palette red) then rendered the whole mosaic
    flat red, read from the GUI as "single channel"."""
    import numpy as np

    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

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
    """A sink writing our own viewer-originated autoscale back into its own policy state would
    latch every channel MANUAL on open and kill per-region contrast."""
    seen = []
    layers.add_mosaic("raw", "488", _img(), contrast_limits=(10.0, 900.0))
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    # adding another layer for the same channel is OUR write, and it propagates via the link
    layers.add_mosaic("stitched", "488", _img(), contrast_limits=(20.0, 800.0))

    assert seen == [], f"programmatic contrast write leaked to the sink: {seen}"


def test_a_write_we_make_ourselves_after_arming_still_does_not_reach_the_sink(layers):
    """The guard above only covers construction-time writes; this one writes through the tap
    after it is armed, which is where the guard actually has to hold.

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
    """The plate must keep following napari for layers added AFTER it subscribed — every op
    run after the first mosaic calls ``add_mosaic`` again, and the sink must not be wired only
    to the layers that existed at subscribe time."""
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
    """``on_user_contrast`` connected to the layer OBJECTS at subscribe time, and
    ``_load_mosaic`` removes and re-adds every layer on each region change — so the sink
    followed napari until the second region, then went silently deaf.

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
    """Contrast is linked per channel, so one drag moves every peer and each peer fires; the
    sink must still hear it once, not N times for one gesture."""
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
    """napari puts a dimension slider on every axis it is not displaying, so z control comes
    for free from being a 3-D array — a 2-D array leaves nothing to put a slider on."""
    layers.add_mosaic("raw", "488", _img(shape=(32, 32)))
    assert layers.model.dims.ndim == 2
    assert list(layers.model.dims.not_displayed) == []

    layers.remove_op("raw")
    layers.add_mosaic("raw", "488", np.zeros((10, 32, 32), dtype=np.uint16))
    assert layers.model.dims.ndim == 3
    assert list(layers.model.dims.not_displayed) == [0]


def test_the_z_axis_carries_the_step_in_micrometres_not_one(layers):
    """A unit z scale steps fine in 2-D but renders an isotropic block in 3-D out of
    anisotropic data."""
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
# on_user_contrast subscribed to layer OBJECTS existing at subscribe time; _load_mosaic
# destroys and recreates every layer on each region change, giving the sync a half-life of one
# region change. Subscriptions key on CHANNEL identity instead, so they survive recreation.


def test_a_channel_added_AFTER_subscribing_still_reaches_the_sink(layers):
    """The subscription is to a channel, not to whichever layers existed at the time."""
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    layers.add_mosaic("raw", "488", _img())            # channel arrives after the subscribe
    layers.find("raw", "488").contrast_limits = (12.0, 345.0)

    assert seen == [("488", 12.0, 345.0)], (
        f"a channel added after subscribing never reached the sink: {seen}"
    )


def test_the_sync_survives_a_region_change_that_recreates_every_layer(layers):
    """``_load_mosaic``'s actual lifecycle: subscribe, then destroy and recreate the layers."""
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
    """Linked layers propagate the write to their peers; if every peer fired, a sink that
    counts or accumulates would be wrong."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("stitched", "488", _img())
    layers.add_mosaic("decon", "488", _img())
    seen = []
    layers.on_user_contrast(lambda ch, lo, hi: seen.append((ch, lo, hi)))

    layers.find("raw", "488").contrast_limits = (44.0, 555.0)

    assert seen == [("488", 44.0, 555.0)], f"expected exactly one delivery, got {seen}"


def test_our_own_writes_still_do_not_look_like_the_user_after_the_rewiring(layers):
    """Keying on channel must not have quietly cost the programmatic guard."""
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
    """Echoes are deduped by comparing against the last value SEEN; if a programmatic write
    did not also update that record, returning to an earlier value would be silently dropped."""
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
# The plate composites with its own (C, 3) RGB table resolved once at open from
# display_color — a second answer to "what colour is this channel" that never gets revised.
# Same defect shape as the contrast that would not follow, same shape of fix.

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
    """Same half-life bug the contrast tap had; armed in ``_register_channel`` so a rebuilt
    layer is still wired.

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
    """``add_mosaic`` sets the channel's colormap itself; an echo here would re-tint from
    defaults on every region change."""
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
    """napari has a built-in scale bar; this only configures it (visible, µm).

    MUTATION: make enable_scale_bar a no-op -> visible stays False -> red.
    """
    from napari.components import ViewerModel

    from squidxplorer._napari_view import enable_scale_bar

    v = ViewerModel()
    enable_scale_bar(v)
    assert v.scale_bar.visible is True
    # napari >=0.8 deprecated ScaleBar.unit (units now come from Layer.units, which add_mosaic
    # sets), so only assert the label where this napari still reports one.
    unit = v.scale_bar.unit
    if unit is not None:
        # pint normalises "um" -> "micrometer"; either spelling is the micron, never pixels.
        assert str(unit) in ("um", "µm", "micrometer")


def test_the_bar_reads_micrometres_because_the_layer_scale_IS_micrometres():
    """The bar shows world = data * layer.scale, so it is correct iff layer.scale is µm/px;
    this pins that a 64 px layer given a 640 µm box spans 640 world units.

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
    ``viewer.scale_bar.unit``.

    MUTATION: stop setting layer.units -> stays pixel -> red.
    """
    from napari.components import ViewerModel

    v = ViewerModel()
    layers = MosaicLayers(v)
    lyr = layers.add_mosaic("raw", "488", _img(shape=(16, 16)),
                            bbox_um=(0.0, 0.0, 160.0, 160.0))
    assert all("meter" in str(u) or str(u) in ("um", "µm") for u in lyr.units)
# ---------------------------------------------------------------- analysis-result layers
# add_mosaic makes IMAGE layers; a segmentation result is a Labels mask and a Points set
# instead. These are the siblings that carry those.


def _labels(n=3, shape=(32, 32)):
    lab = np.zeros(shape, dtype=np.int32)
    for i in range(1, n + 1):
        lab[i * 4: i * 4 + 3, i * 4: i * 4 + 3] = i
    return lab


def test_a_mask_lands_as_a_real_napari_Labels_layer_not_an_image(layers):
    """A Labels layer gives napari its label colormap, pick-by-click and 0-transparency; an
    Image would render it as a near-black gradient."""
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
    """Identity lives in metadata, never parsed back out of the name."""
    layers.add_labels("spots", "488 mask", _labels())
    layers.add_points("spots", "488 centroids", np.array([[1.0, 2.0]]))

    assert key_of(layers.model.layers["spots · 488 mask"]) == MosaicKey("spots", "488 mask")
    assert set(layers.channels("spots")) == {"488 mask", "488 centroids"}
    assert "spots" in layers.ops()


def test_an_empty_points_layer_is_still_added_so_zero_nuclei_is_visible_as_a_result(layers):
    """Zero found is an answer; skipping the layer would make "nothing ran" and "nothing
    there" look identical."""
    from napari.layers import Points

    lyr = layers.add_points("spots", "488 centroids", np.zeros((0, 2)))
    assert isinstance(lyr, Points)
    assert len(np.asarray(lyr.data)) == 0


def test_analysis_layers_are_NOT_linked_into_the_per_channel_contrast_group(layers):
    """A Labels/Points layer has no ``contrast_limits``; registering it as a contrast peer
    would make ``link_layers`` raise on the next channel added."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_labels("spots", "488", _labels())          # same channel string, on purpose

    peers = layers._by_channel["488"]
    assert len(peers) == 1, "an analysis layer was registered as a contrast peer"
    layers.set_contrast("488", 10.0, 900.0)               # must not raise


def test_analysis_layers_are_placed_in_stage_micrometres_like_the_mosaic_they_describe(layers):
    """The mask must land ON the mosaic: same bbox must give the same scale/translate."""
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
    """A Points layer carries no array shape, so placement needs it passed in explicitly, or
    every centroid would silently land at the world origin."""
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
    """A feature table keyed by label value, expressed in the shape napari already has: no
    other viewer plugin surfaces a count, so it has to live somewhere addressable."""
    pts = np.array([[4.0, 4.0], [8.0, 8.0], [12.0, 12.0]])
    lyr = layers.add_points("spots", "488 centroids", pts,
                            features={"label": [1, 2, 3]})
    assert list(lyr.features["label"]) == [1, 2, 3]
    assert len(lyr.features) == 3


# --- the PROCESSING LAYER fan-out (Julio, 2026-08-03) ------------------------------------------
#
# "After I click an operator layer, the thumbnails don't update." `visible_op` could already
# answer, but no signal was emitted for it. These pin the producer half;
# tests/test_plate_follows_windows.py pins the sink.


def test_showing_a_processing_layer_is_reported_once_for_the_whole_group(layers):
    seen = []
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("raw", "561", _img(1))
    layers.add_mosaic("mip", "488", _img(2), visible=False)
    layers.add_mosaic("mip", "561", _img(3), visible=False)
    layers.on_user_op(lambda op, on: seen.append((op, on)))

    for ly in layers.group("mip"):        # the group checkbox, which writes every child
        ly.visible = True

    # ONE delivery per op, cause before consequence: mip lit, raw went dark BECAUSE of that.
    assert seen == [("mip", True), ("raw", False)], f"one group toggle reported as {seen}"


def test_switching_operator_is_reported_by_the_op_tap_and_not_by_the_channel_tap(layers):
    """``on_user_visibility`` answers "is this CHANNEL on screen anywhere", unchanged across an
    operator switch, so it returns early and the plate hears nothing — the op fan-out needs
    its own tap for exactly that gesture."""
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
    """A result delivered to a window that did NOT ask arrives ``visible=False``; reporting
    that as a gesture would turn the plate's layer off behind the user."""
    seen = []
    layers.add_mosaic("raw", "488", _img())
    layers.on_user_op(lambda op, on: seen.append((op, on)))

    layers.add_mosaic("mip", "488", _img(1), visible=False)
    layers.add_mosaic("mip", "488", _img(2), visible=True)      # a re-add, still ours

    assert seen == [], f"a programmatic write was reported as a user gesture: {seen}"


# --- one hex per channel, or an honest None --------------------------------------------------
#
# A LUT snapshot carries a single rgb per channel with no field for a gradient.
# colormap_hue_rgb decides which napari colormaps survive that and which cannot; the
# answer must never be "approximate it".

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
    from squidxplorer._napari_view import colormap_hue_rgb
    assert colormap_hue_rgb(_hue_layer(layers, name)) == expected


def test_the_colormap_this_app_builds_for_a_channel_reduces_to_its_palette_colour(layers):
    """``_napari_pane._colormap_for`` builds ``[[0,0,0,1], [r,g,b,1]]`` from Squid's palette;
    if that did not reduce, the common case would silently do nothing."""
    from napari.utils import Colormap

    from squidxplorer._napari_view import colormap_hue_rgb

    cmap = Colormap([[0.0, 0.0, 0.0, 1.0], [0x1F / 255, 1.0, 0.0, 1.0]], name="squid-488")
    assert colormap_hue_rgb(_hue_layer(layers, cmap)) == (0x1F, 255, 0)


@pytest.mark.parametrize("name", ["viridis", "turbo", "inferno", "PiYG"])
def test_a_multi_stop_colormap_refuses_rather_than_approximating(layers, name):
    """A perceptual map's last stop is the top of a ramp, not the map (viridis ends yellow and
    is mostly not yellow); emitting it would put a colour into the snapshot that is on no screen."""
    from squidxplorer._napari_view import colormap_hue_rgb
    try:
        layer = _hue_layer(layers, name)
    except (KeyError, ValueError):
        pytest.skip(f"this napari has no {name!r} colormap")
    assert colormap_hue_rgb(layer) is None


def test_a_colormap_ending_in_black_has_no_hue_to_name(layers):
    """The degenerate case: every row is a multiple of the zero vector, so the projection
    would divide by zero and, unguarded, call black a valid hue for anything."""
    from napari.utils import Colormap

    from squidxplorer._napari_view import colormap_hue_rgb

    cmap = Colormap([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]], name="inverted")
    assert colormap_hue_rgb(_hue_layer(layers, cmap)) is None


def test_a_layer_with_no_colormap_table_says_so_instead_of_guessing(layers):
    from squidxplorer._napari_view import colormap_hue_rgb
    assert colormap_hue_rgb(object()) is None


# ==============================================================================================
# A Z-REDUCED RESULT IS PRESENTED WITHOUT A Z AXIS
#
# The MIP layer arrives already 2-D ((T, C, 1, Y, X) -> (Y, X)), and nothing downstream keeps a
# singleton z. The slider belongs to the RAW pyramid sharing the pane, and napari derives
# dims.ndim from the MAXIMUM over EVERY layer, visible or not — so delivering the MIP darkened
# raw and left raw's slider standing over a picture it could not move. These drive a bare
# ``ViewerModel``, so they pin the rule rather than a widget.

def _z_stack_pyramid(nz=10, shape=(64, 64)):
    """A raw mosaic the way ``fuse_region_pyramid`` hands it over: (z, y, x) levels."""
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
    """The toggle is a comparison, so it must be reversible — pyramid, scale and translate."""
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
    """Scoped to the pane's own mosaics and to a REAL z axis: a 2-D layer has nothing to stash,
    so restoring it later must not be able to invent a stack."""
    flat = layers.add_mosaic("raw", "405", np.zeros((64, 64), np.uint16), bbox_um=_Z_BBOX)
    layers.add_result("intensity", "mip", "405", np.zeros((64, 64), np.uint16),
                      bbox_um=_Z_BBOX, visible=True)
    layers.find("raw", "405").visible = True

    assert flat.ndim == 2 and np.asarray(flat.data).shape == (64, 64)


def test_a_plane_op_result_KEEPS_the_z_axis(layers):
    """A plane-op declares ``consumes=frozenset()`` — z survives at full depth — so its result
    says nothing about whether the pane should show a z axis. Read off the DECLARATION, never
    a name comparison."""
    from squidxplorer import add_operator, available_plane_operators, plane_op

    if "zaxis_plane_op" not in available_plane_operators():
        add_operator("zaxis_plane_op", plane_op(lambda a: a))
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
    """``show_op`` is the before/after toggle's other entrance; one rule, every surface."""
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
# napari wraps a multiscale layer's data in MultiScaleData, a Sequence that is neither a list
# nor a tuple, so every `isinstance(data, (list, tuple))` pyramid check in this codebase was
# False for every real pyramid — each site failed differently (one raised, two failed silently).
# `_napari_view.pyramid_levels` is the one rule they all read now.


def test_the_3d_swap_actually_swaps_a_pyramid_napari_handed_back(layers):
    """``render_max_res_3d`` was a silent no-op on every real mosaic: its first line was
    ``if not isinstance(ly.data, (list, tuple)): return``, which a ``multiscale=True`` layer
    never satisfies, so napari went on dropping it to its COARSEST level in 3D.

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
    """``np.asarray(MultiScaleData)`` returns the COARSEST level (``__array__`` is
    ``_data[-1]``), so treating a non-list Sequence as a plain array silently substitutes the
    smallest picture for the largest.

    MUTATION: make ``pyramid_levels`` return None for a non-list Sequence -> the coarsest level
    comes back -> red.
    """
    from napari.components import ViewerModel

    from squidxplorer._napari_view import full_res_level, pyramid_levels

    lv0 = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    data = ViewerModel().add_image([lv0, lv0[::2, ::2], lv0[::4, ::4]], multiscale=True).data

    assert np.asarray(data).shape == (16, 16), (
        "napari no longer coerces a pyramid to its coarsest level; this test's premise is stale")
    assert [tuple(lv.shape) for lv in pyramid_levels(data)] == [(64, 64), (32, 32), (16, 16)]
    assert np.array_equal(np.asarray(full_res_level(data)), lv0)


def test_a_plain_array_and_a_nested_list_are_not_pyramids():
    """The discriminator is the ELEMENT, not the container: ``[[1, 2], [3, 4]]`` encodes ONE
    array and must not be read as two levels."""
    from squidxplorer._napari_view import full_res_level, pyramid_levels

    arr = np.zeros((4, 4), np.uint16)
    assert pyramid_levels(arr) is None
    assert full_res_level(arr) is arr
    assert pyramid_levels([[1, 2], [3, 4]]) is None
    assert pyramid_levels("not an image at all") is None
    with pytest.raises(ValueError, match="EMPTY multiscale"):
        pyramid_levels([])


# ══════════════════════════════════════════════════════════════════════════════════════════
# Framing ONE box: the arithmetic behind stepping a camera across a region's FOVs.
#
# All of this is napari's own `fit_to_view` rule pointed at a box the scene does not describe.
# The test that matters most is the LAST one: it proves we are not a second, disagreeing copy.
# ══════════════════════════════════════════════════════════════════════════════════════════

def test_camera_for_bbox_um_centres_the_box():
    from squidxplorer._napari_view import camera_for_bbox_um

    (cy, cx), _zoom = camera_for_bbox_um((1000.0, 2000.0, 1400.0, 2300.0), (600, 800))
    assert (cy, cx) == (2150.0, 1200.0)


def test_camera_for_bbox_um_fits_the_TIGHTER_axis():
    """Height against height, width against width — and the tighter one decides.

    This is the assertion that fails on an (h, w) / (w, h) swap, which is why the box and the
    canvas are both deliberately non-square and of DIFFERENT aspect. A swap does not raise; it
    frames the box against the wrong edge and reads as somebody preferring a different zoom.
    `_brick_view._frame_camera` carried exactly that bug until this function existed.
    """
    from squidxplorer._napari_view import camera_for_bbox_um

    # canvas 600 tall x 800 wide; box 50 um tall x 100 um wide -> width is the tighter axis.
    _c, zoom = camera_for_bbox_um((0.0, 0.0, 100.0, 50.0), (600, 800))
    assert zoom == pytest.approx(0.95 * min(600 / 50, 800 / 100))
    assert zoom == pytest.approx(0.95 * 8.0), "800/100 is tighter than 600/50; the min must win"

    # Same canvas, box transposed -> now height is the tighter axis. A function that read the
    # canvas the other way round would return the SAME number for both of these.
    _c, zoom_t = camera_for_bbox_um((0.0, 0.0, 50.0, 100.0), (600, 800))
    assert zoom_t == pytest.approx(0.95 * min(600 / 100, 800 / 50))
    assert zoom_t != pytest.approx(zoom)


def test_camera_for_bbox_um_leaves_the_margin_napari_leaves():
    from squidxplorer._napari_view import FRAME_MARGIN, camera_for_bbox_um

    assert FRAME_MARGIN == 0.05, "napari's own reset_view default; a second convention drifts"
    _c, zoom = camera_for_bbox_um((0.0, 0.0, 100.0, 100.0), (400, 400), margin=0.0)
    assert zoom == pytest.approx(4.0), "no margin means the box touches the canvas edges"
    _c, zoomed = camera_for_bbox_um((0.0, 0.0, 100.0, 100.0), (400, 400), margin=0.5)
    assert zoomed == pytest.approx(2.0)


@pytest.mark.parametrize("bbox, canvas, margin, match", [
    ((0.0, 0.0, 0.0, 10.0), (600, 800), 0.05, "x1 > x0"),
    ((0.0, 0.0, 10.0, 0.0), (600, 800), 0.05, "y1 > y0"),
    ((0.0, 0.0, 10.0, 10.0), (0, 800), 0.05, "must be positive"),
    ((0.0, 0.0, 10.0, 10.0), (600, -1), 0.05, "must be positive"),
    ((0.0, 0.0, 10.0, 10.0), (600, 800), 1.0, r"\[0, 1\)"),
    ((0.0, 0.0, 10.0, 10.0), (600, 800), -0.1, r"\[0, 1\)"),
])
def test_camera_for_bbox_um_refuses_rather_than_guessing(bbox, canvas, margin, match):
    """A camera pointed at a number nobody measured shows somewhere else, with no error."""
    from squidxplorer._napari_view import camera_for_bbox_um

    with pytest.raises(ValueError, match=match):
        camera_for_bbox_um(bbox, canvas, margin=margin)


def test_framing_the_layer_s_own_box_gives_napari_s_own_camera():
    """THE test this function exists to pass: reset_view and frame_bbox_um must agree.

    If they disagree, we have written a second, subtly different answer to "what does fitting
    mean" — which is the whole defect this codebase's one-owner rule is about, wearing a camera.

    The ZOOM must be identical. The CENTRE differs by exactly half a pixel, and that is a real
    convention difference rather than an error: napari's layer extent runs between pixel CENTRES,
    while a ``bbox_um`` is the outer EDGE of the pixels. On the FOV boxes this is built for, that
    is 0.05 um on a 392 um field.
    """
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    model = ViewerModel()
    layers = MosaicLayers(model)
    bbox = (1000.0, 2000.0, 1400.0, 2300.0)          # 400 um wide, 300 um tall
    layers.add_mosaic("raw", "c0", np.zeros((300, 400), np.uint16), contrast_limits=(0, 1),
                      colormap="gray", multiscale=False, bbox_um=bbox)

    model.reset_view()
    ref_center, ref_zoom = tuple(model.camera.center), float(model.camera.zoom)

    model.camera.center, model.camera.zoom = (0.0, 0.0, 0.0), 1.0
    assert layers.frame_bbox_um(bbox) is None, "framing a box the layer covers must succeed"

    assert float(model.camera.zoom) == pytest.approx(ref_zoom)
    half_px_y, half_px_x = 300 / 300 / 2, 400 / 400 / 2
    assert model.camera.center[1] == pytest.approx(ref_center[1], abs=half_px_y)
    assert model.camera.center[2] == pytest.approx(ref_center[2], abs=half_px_x)


def test_frame_bbox_um_is_a_programmatic_write():
    """The plate is a SINK of a window's napari. Our camera move must not read as a user gesture."""
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    layers = MosaicLayers(ViewerModel())
    seen = []
    layers._model.camera.events.zoom.connect(lambda _e: seen.append(layers.is_programmatic))
    assert layers.frame_bbox_um((0.0, 0.0, 100.0, 100.0)) is None
    assert seen and all(seen), "every camera event this raised must be marked as ours"
    assert not layers.is_programmatic, "the marker must not leak past the write"


def test_frame_bbox_um_refuses_in_3d_by_name_instead_of_framing_something_wrong():
    """In 3D the visible extent depends on camera.angles, which this deliberately does not model."""
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    model = ViewerModel()
    model.dims.ndisplay = 3
    layers = MosaicLayers(model)
    said = layers.frame_bbox_um((0.0, 0.0, 100.0, 100.0))
    assert said is not None and "2D" in said


def test_frame_bbox_um_names_a_bad_box_rather_than_moving_the_camera():
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    model = ViewerModel()
    layers = MosaicLayers(model)
    before = tuple(model.camera.center), float(model.camera.zoom)
    said = layers.frame_bbox_um((0.0, 0.0, 0.0, 100.0))
    assert said is not None and "could not frame" in said
    assert (tuple(model.camera.center), float(model.camera.zoom)) == before


def test_a_napari_that_moved_its_canvas_size_is_named_not_guessed_around():
    """``_canvas_size`` is private, so its loss must be a sentence — never a silent (800, 600).

    Framing every camera against a canvas napari no longer reports would be wrong by the aspect
    ratio and would look like a zoom preference, which is the failure this whole binding-guard
    section exists to convert into a loud one.
    """
    from napari.components import ViewerModel

    from squidxplorer._napari_view import (
        REQUIRED_MODEL_PRIVATE_ATTRS,
        MosaicLayers,
        NapariBindingError,
    )

    assert "_canvas_size" in REQUIRED_MODEL_PRIVATE_ATTRS, "the guard must cover what we read"

    class _Moved(ViewerModel):
        _canvas_size = None

    layers = MosaicLayers(_Moved())
    with pytest.raises(NapariBindingError, match="_canvas_size"):
        layers.frame_bbox_um((0.0, 0.0, 100.0, 100.0))


# ------------------------------------------- selection follows visibility (ticket #8)
# napari's layer-controls panel shows the SELECTED layer, and napari never moves selection on a
# visibility flip — so unticking a layer left its controls up (405's colormap over an off 405).


def test_hiding_the_selected_layer_moves_selection_to_a_visible_layer(layers):
    l488 = layers.add_mosaic("raw", "488", _img())
    l561 = layers.add_mosaic("raw", "561", _img(1))
    layers.model.layers.selection.active = l488

    l488.visible = False

    assert layers.model.layers.selection.active is l561, (
        "the controls panel is still showing a layer that is OFF")


def test_selection_prefers_a_visible_layer_of_the_SAME_op(layers):
    raw488 = layers.add_mosaic("raw", "488", _img())
    raw561 = layers.add_mosaic("raw", "561", _img(1))
    decon561 = layers.add_mosaic("decon", "561", _img(2), visible=False)
    with layers.programmatic():          # craft the scene without tripping the exclusive rule
        raw561.visible = True
        decon561.visible = True
    layers.model.layers.selection.active = raw488

    raw488.visible = False

    # decon561 is the TOPMOST visible layer, but raw561 shares the hidden layer's op.
    assert layers.model.layers.selection.active is raw561


def test_hiding_the_last_visible_layer_keeps_the_selection(layers):
    l488 = layers.add_mosaic("raw", "488", _img())
    l561 = layers.add_mosaic("raw", "561", _img(1), visible=False)
    layers.model.layers.selection.active = l488

    l488.visible = False

    assert layers.model.layers.selection.active is l488, (
        "with nothing visible there is nowhere honest to move the selection")
    assert l561.visible is False


def test_hiding_an_UNSELECTED_layer_leaves_the_selection_alone(layers):
    l488 = layers.add_mosaic("raw", "488", _img())
    l561 = layers.add_mosaic("raw", "561", _img(1))
    layers.model.layers.selection.active = l488

    l561.visible = False

    assert layers.model.layers.selection.active is l488


def test_hiding_a_channel_of_a_VOLUME_moves_selection_to_a_visible_brick(layers):
    from .conftest import build_volume_scene

    from squidxplorer._napari_view import key_of

    build_volume_scene(layers, "raw", ("488", "561"), bricks=3)
    rep = layers.find("raw", "488")
    layers.model.layers.selection.active = rep

    rep.visible = False                  # the mirror darkens every 488 brick

    active = layers.model.layers.selection.active
    assert active is not None and bool(active.visible), (
        "selection is parked on a hidden brick")
    k = key_of(active)
    assert k is not None and k.channel == "561"
